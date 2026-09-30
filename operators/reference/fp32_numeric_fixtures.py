"""Named-backend RNE fixtures, logical storage only; no AME packet layout."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fp32_fold as reference
TILES=(4,8,16,32,64,128)
ROOT=Path(__file__).resolve().parents[2]

def witnesses():
    data=json.loads((ROOT/'operators/cases/fp32-rne-witnesses.json').read_text())
    if data['schema']!=1 or data['contract']!='fp32-rne-fold-v1':raise ValueError('witness contract')
    values=data['witnesses']
    if len({r['name'] for r in values})!=len(values):raise ValueError('duplicate witness')
    return {r['name']:dict(a=[int(v,16) for v in r['a']],b=[int(v,16) for v in r['b']],
                         seed=int(r['seed'],16),add=r['add']) for r in values}

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='fp32-rne-fold-v1':raise ValueError('unknown contract')
    literals=witnesses();rows=[]
    for t in TILES:
        vectors=[(name,data,0) for name,data in literals.items()]
        for blocks in (1,2):
            for name in ('fused-cancel','seed-before-fold','order-sensitive','intermediate-flags','negative-zero-seed'):
                vectors.append((f'cross-{blocks}-{name}',literals[name],blocks*t-1))
        for name,data,offset in vectors:
            k=offset+len(data['a']);lda=k+2;ldb=4 if op=='gemm' else 1;ldc=5 if op=='gemm' else 1
            a=[0]*k+[0x7f800001]*2;b=[0x7f800001]*(k*ldb)
            for q in range(k):b[q*ldb]=0x80000000
            a[offset:offset+len(data['a'])]=data['a']
            for q,v in enumerate(data['b']):b[(offset+q)*ldb]=v
            c=[data['seed']]+[0x7f800001]*(ldc-1)
            rows.append(dict(tile=t,purpose=name,m=1,n=1,k=k,lda=lda,ldb=ldb,ldc=ldc,
                accumulate=data['add'],a=a,b=b,initial=c))
        # Cross-axis tails: every output lane is real data, including exceptions.
        for label,m,n in (('rows',t+1,3 if op=='gemm' else 1),
                          ('columns',3,t+1 if op=='gemm' else 1)):
            if op=='gemv' and label=='columns':continue
            k=3;lda=5;ldb=n+3 if op=='gemm' else 1;ldc=n+4 if op=='gemm' else 1
            ap=(0x3f800001,0x7f800000,0x7f800001,1,0x80000000)
            bp=(0x3f7ffffe,0,0x3f000000,0x7fc12345,0xff800000)
            cp=(0xbf800000,0,0x80000000,0x00800000,0x7fc12345)
            a=[ap[(i+q)%5] if q<k else 0x7f800001 for i in range(m) for q in range(lda)]
            b=[bp[(q+j)%5] if j<n else 0x7f800001 for q in range(k) for j in range(ldb)]
            c=[cp[(i+j)%5] if j<n else 0x7f800001 for i in range(m) for j in range(ldc)]
            for add in (0,1):rows.append(dict(tile=t,purpose='mixed-'+label,m=m,n=n,k=k,
                lda=lda,ldb=ldb,ldc=ldc,accumulate=add,a=a,b=b,initial=c))
    for index,r in enumerate(rows):
        r['name']=f't{r["tile"]}-{r["purpose"]}-{r["accumulate"]}'
        r['golden'],r['new_flags']=reference.gemm(*[r[x] for x in ('m','n','k','a','lda','b','ldb','initial','ldc','accumulate')])
        # DZ has no producer in this kernel and must remain sticky. Never seed NV/UF/NX here.
        r.update(seed_flags=8 if index%2 else 0,seed_xsat=index%2,fcsr=0x6a if index%2 else 0x85)
    return rows

def generate(manifest,out,config):
    if config['max_n']!=128:raise ValueError('unreviewed maximum tile')
    rows=create(json.loads(Path(manifest).read_text()))
    widths={key:max(len(r[key]) for r in rows) for key in ('a','b','initial')}
    text=['/* Generated logical bit images and independent RNE FMA GOLDEN. */',
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
        checks_by_n={str(t):6+4*(128**2+32)+3+per*(6+sum(widths.values())+96+3*128**2+32-3*t*t) for t in TILES},
        widths=widths,reference='logical increasing-K FP32 RNE FMA, Fraction arithmetic and R2 tininess',
        files=['op_fixtures.h','golden.json'])
