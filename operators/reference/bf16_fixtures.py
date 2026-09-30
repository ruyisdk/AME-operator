"""Logical BF16 dyadics and FP32 GOLDEN, independent of packet packing."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import floating_exact as fp
TILES=(4,8,16,32,64,128)

def shapes(op,t):
    rows=[]
    for axis in (('m','n') if op=='gemm' else ('m',)):
        for label,value in (('minus',t-1),('exact',t),('plus',t+1),('double-plus',2*t+1)):
            shape=dict(m=3,n=5 if op=='gemm' else 1,k=5);shape[axis]=value
            rows.append((axis+'-'+label,shape))
    for packet in (1,2,4):
        for label,offset in (('minus',-1),('exact',0),('plus',1)):
            rows.append((f'k-{packet}t-{label}',dict(m=3,n=5 if op=='gemm' else 1,k=packet*t+offset)))
    rows.extend([('cross',dict(m=t+1,n=t-1 if op=='gemm' else 1,k=2*t+1)),
                 ('long-k',dict(m=5,n=3 if op=='gemm' else 1,k=6*t+1)),
                 ('fraction-hand',dict(m=2,n=2 if op=='gemm' else 1,k=2)),
                 ('exact-cancel',dict(m=1,n=1,k=3)),
                 ('limit-positive',dict(m=1,n=1,k=1)),
                 ('limit-negative',dict(m=1,n=1,k=1)),
                 ('zero-products',dict(m=2,n=3 if op=='gemm' else 1,k=3)),
                 ('zero-k',dict(m=2,n=3 if op=='gemm' else 1,k=0)),
                 ('zero-m',dict(m=0,n=3 if op=='gemm' else 1,k=3))])
    if op=='gemm':rows.append(('zero-n',dict(m=3,n=0,k=3)))
    return rows

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='bf16-f32-exact-v1':raise ValueError('unknown BF16 contract')
    rows=[]
    for t in TILES:
        for index,(purpose,shape) in enumerate(shapes(op,t)):
            m,n,k=(shape[x] for x in ('m','n','k'))
            for add in (0,1):
                lda,ldb,ldc=k+2,n+3 if op=='gemm' else 1,n+4 if op=='gemm' else 1
                def data(nr,nc,stride,salt):
                    return [(((i+1)*0x45d9f3b^(j+1)*0x119de1f3^salt)%15-7)
                            if j<nc else 123 for i in range(nr) for j in range(stride)]
                a=data(m,k,lda,t+index);b=data(k,n,ldb,t+index+19)
                c=[i*7-j*5+3 if j<n else 197 for i in range(m) for j in range(ldc)]
                if purpose=='fraction-hand':
                    a[:2]=[8,-4];a[lda:lda+2]=[2,6];b[0]=4;b[ldb]=-8
                    if op=='gemm':b[1]=8;b[ldb+1]=2
                elif purpose=='exact-cancel':
                    a[:3]=[4096,1,-4096];b[0]=1024;b[ldb]=1;b[2*ldb]=1024
                elif purpose.startswith('limit-'):
                    a[0]=-255 if purpose=='limit-negative' else 255;b[0]=255
                elif purpose=='zero-products':
                    for i in range(m):
                        for q in range(k):a[i*lda+q]=0
                g=fp.gemm(m,n,k,a,lda,b,ldb,c,ldc,add,input_format='bf16')
                rows.append(dict(name=f't{t}-{purpose}-{add}',tile=t,purpose=purpose,**shape,
                    lda=lda,ldb=ldb,ldc=ldc,accumulate=add,a_units=a,b_units=b,initial_units=c,golden_units=g,
                    a=[fp.bits(v,3,'bf16') for v in a],b=[fp.bits(v,3,'bf16') for v in b],
                    initial=[fp.bits(v,6) for v in c],golden=[fp.bits(v,6) for v in g]))
    return rows

def generate(manifest,out,config):
    if config['max_n']!=128:raise ValueError('unreviewed maximum tile')
    rows=create(json.loads(Path(manifest).read_text()))
    widths={key:max(len(r[key]) for r in rows) for key in ('a','b','initial')}
    text=['/* Generated BF16 bit images and independent exact FP32 GOLDEN. */',
          'typedef __UINT16_TYPE__ op_u16;',
          'struct op_fixture { ame_size tile,m,n,k,lda,ldb,ldc; int accumulate;',
          '  ame_size a_count,b_count,c_count; const op_u16 *a,*b; const u32 *initial,*golden; };']
    for i,r in enumerate(rows):
        for key in ('a','b','initial','golden'):
            typ='op_u16' if key in ('a','b') else 'u32'
            text.append(f'static const {typ} op_{i}_{key}[] = {{'+','.join(f'0x{v:x}u' for v in r[key] or [0])+'};')
    text.append('static const struct op_fixture op_fixtures[] = {')
    for i,r in enumerate(rows):
        values=[str(r[k]) for k in ('tile','m','n','k','lda','ldb','ldc','accumulate')]
        values += [str(len(r[k])) for k in ('a','b','initial')]
        values += [f'op_{i}_{k}' for k in ('a','b','initial','golden')]
        text.append('  {'+','.join(values)+'},')
    per=len(rows)//len(TILES)
    text += ['};',f'#define OP_CASES {len(rows)}',f'#define OP_SELECTED_CASES {per}',
             f'#define OP_MAX_A {widths["a"]}',f'#define OP_MAX_B {widths["b"]}',f'#define OP_MAX_C {widths["initial"]}','']
    (out/'op_fixtures.h').write_text('\n'.join(text))
    (out/'golden.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    checks={str(t):6+4*(128**2+32)+3+sum(11+sum(widths.values())+192+5*128**2
        -(5*t*t if r['m'] and r['n'] and r['k'] else 0) for r in rows if r['tile']==t) for t in TILES}
    return dict(logical_cases=len(rows),cases_by_n={str(t):per for t in TILES},checks_by_n=checks,
        widths=widths,reference='independent exact scaled-integer logical BF16/FP32 matrices',
        files=['op_fixtures.h','golden.json'])
