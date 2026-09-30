"""BF16 logical inputs and FP32 answers, independent of square-major packing."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bf16_fold as reference
TILES=(4,8,16,32,64,128)
ROOT=Path(__file__).resolve().parents[2]
CROSS=('fused-overflow-rescue','fused-subnormal-half','seed-before-fold',
       'order-sensitive','intermediate-flags','negative-zero-seed')

def witnesses():
    data=json.loads((ROOT/'operators/cases/bf16-rne-witnesses.json').read_text())
    if data['schema']!=1 or data['contract']!='bf16-f32-rne-fold-v1':
        raise ValueError('witness contract')
    values=data['witnesses']
    if len({r['name'] for r in values})!=len(values):raise ValueError('duplicate witness')
    return {r['name']:dict(a=[int(v,16) for v in r['a']],b=[int(v,16) for v in r['b']],
                         seed=int(r['seed'],16),add=r['add']) for r in values}

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='bf16-f32-rne-fold-v1':
        raise ValueError('unknown contract')
    literals=witnesses();rows=[]
    for t in TILES:
        vectors=[(name,data,0) for name,data in literals.items()]
        # T crosses the two input squares; 2*T and 4*T cross kernel K packets.
        for squares in (1,2,4):
            for name in CROSS:vectors.append((f'cross-{squares}-{name}',literals[name],squares*t-1))
        for name,data,offset in vectors:
            k=offset+len(data['a']);lda=k+2;ldb=4 if op=='gemm' else 1;ldc=5 if op=='gemm' else 1
            a=[0]*k+[0x7f81]*2;b=[0x7f81]*(k*ldb)
            for q in range(k):b[q*ldb]=0x8000
            a[offset:offset+len(data['a'])]=data['a']
            for q,v in enumerate(data['b']):b[(offset+q)*ldb]=v
            c=[data['seed']]+[0x7f800001]*(ldc-1)
            rows.append(dict(tile=t,purpose=name,m=1,n=1,k=k,lda=lda,ldb=ldb,ldc=ldc,
                accumulate=data['add'],a=a,b=b,initial=c))
        for label,m,n in (('rows',t+1,3 if op=='gemm' else 1),('columns',3,t+1)):
            if op=='gemv' and label=='columns':continue
            k=3;lda=5;ldb=n+3 if op=='gemm' else 1;ldc=n+4 if op=='gemm' else 1
            ap=(0x3f81,0x7f80,0x7f81,1,0x8000)
            bp=(0x3f7e,0,0x3700,0x7fc1,0xff80)
            cp=(0xbf800000,0,0x80000000,0x00800000,0x7fc12345)
            a=[ap[(i+q)%5] if q<k else 0x7f81 for i in range(m) for q in range(lda)]
            b=[bp[(q+j)%5] if j<n else 0x7f81 for q in range(k) for j in range(ldb)]
            c=[cp[(i+j)%5] if j<n else 0x7f800001 for i in range(m) for j in range(ldc)]
            for add in (0,1):rows.append(dict(tile=t,purpose='mixed-'+label,m=m,n=n,k=k,
                lda=lda,ldb=ldb,ldc=ldc,accumulate=add,a=a,b=b,initial=c))
    for index,r in enumerate(rows):
        r['name']=f't{r["tile"]}-{r["purpose"]}-{r["accumulate"]}'
        r['golden'],r['new_flags']=reference.gemm(*[r[x] for x in ('m','n','k','a','lda','b','ldb','initial','ldc','accumulate')])
        r.update(seed_flags=8 if index%2 else 0,seed_xsat=index%2,fcsr=0x6a if index%2 else 0x85)
    return rows

def generate(manifest,out,config):
    if config['max_n']!=128:raise ValueError('unreviewed maximum tile')
    rows=create(json.loads(Path(manifest).read_text()))
    widths={key:max(len(r[key]) for r in rows) for key in ('a','b','initial')}
    text=['/* Generated logical BF16/FP32 bit images and RNE FMA GOLDEN. */',
          'struct op_fixture { ame_size tile,m,n,k,lda,ldb,ldc; int accumulate;',
          '  ame_size a_count,b_count,c_count; u32 seed_flags,new_flags,seed_xsat,fcsr;',
          '  const u32 *a,*b,*initial,*golden; };']
    for i,r in enumerate(rows):
        for key in ('a','b','initial','golden'):
            text.append(f'static const u32 op_{i}_{key}[] = {{'+','.join(f'0x{v:08x}u' for v in r[key])+'};')
    text.append('static const struct op_fixture op_fixtures[] = {')
    for i,r in enumerate(rows):
        values=[str(r[k]) for k in ('tile','m','n','k','lda','ldb','ldc','accumulate')]
        values += [str(len(r[k])) for k in ('a','b','initial')]
        values += [str(r[k]) for k in ('seed_flags','new_flags','seed_xsat','fcsr')]
        values += [f'op_{i}_{k}' for k in ('a','b','initial','golden')]
        text.append('  {'+','.join(values)+'},')
    per=len(rows)//len(TILES)
    text += ['};',f'#define OP_CASES {len(rows)}',f'#define OP_SELECTED_CASES {per}',
             f'#define OP_MAX_A {widths["a"]}',f'#define OP_MAX_B {widths["b"]}',f'#define OP_MAX_C {widths["initial"]}','']
    (out/'op_fixtures.h').write_text('\n'.join(text))
    (out/'golden.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    return dict(logical_cases=len(rows),cases_by_n={str(t):per for t in TILES},
        checks_by_n={str(t):6+4*(128**2+32)+4+per*(12+sum(widths.values())+96+5*128**2+96-5*t*t) for t in TILES},
        widths=widths,reference='exact BF16 widening; increasing-K FP32 RNE FMA, Fraction arithmetic and R2 tininess',
        files=['op_fixtures.h','golden.json'])
