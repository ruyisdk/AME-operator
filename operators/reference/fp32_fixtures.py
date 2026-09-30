"""Logical exact-dyadic matrices, independent of target packing and Spike."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import floating_exact as fp
from geometry_fixtures import TILES, dimensions

def shapes(op,t):
    rows=dimensions(op,t)
    rows += [("fraction-hand",dict(m=2,n=2 if op=="gemm" else 1,k=2)),
             ("exact-cancel",dict(m=1,n=1,k=3)),
             ("limit-positive",dict(m=1,n=1,k=1)),
             ("limit-negative",dict(m=1,n=1,k=1)),
             ("zero-products",dict(m=2,n=3 if op=="gemm" else 1,k=3)),
             ("zero-k",dict(m=2,n=3 if op=="gemm" else 1,k=0)),
             ("zero-m",dict(m=0,n=3 if op=="gemm" else 1,k=3))]
    if op=="gemm":rows.append(("zero-n",dict(m=3,n=0,k=3)))
    return rows

def create(spec):
    op=spec["operator"]
    if op not in ("gemm","gemv") or spec["contract"]!="fp32-exact-v1":
        raise ValueError("unknown exact FP32 contract")
    rows=[]
    for t in TILES:
        for index,(purpose,shape) in enumerate(shapes(op,t)):
            m,n,k=(shape[x] for x in ("m","n","k"))
            for add in (0,1):
                lda,ldb,ldc=k+2,n+3 if op=="gemm" else 1,n+4 if op=="gemm" else 1
                def data(nr,nc,stride,salt):
                    return [(((i+1)*0x45d9f3b^(j+1)*0x119de1f3^salt)%15-7)
                            if j<nc else 123 for i in range(nr) for j in range(stride)]
                a=data(m,k,lda,t+index);b=data(k,n,ldb,t+index+19)
                c=[(i*7-j*5+3) if j<n else 197 for i in range(m) for j in range(ldc)]
                if purpose=="fraction-hand":
                    a[0:2]=[8,-4];a[lda:lda+2]=[2,6]
                    b[0]=4;b[ldb]=-8
                    if op=="gemm":b[1]=8;b[ldb+1]=2
                if purpose=="exact-cancel":
                    a[:3]=[4096,1,-4096]
                    b[0]=1024;b[ldb]=1;b[2*ldb]=1024
                if purpose.startswith("limit-"):
                    a[0]=-4095 if purpose=="limit-negative" else 4095;b[0]=4095
                if purpose=="zero-products":
                    for i in range(m):
                        for p in range(k):a[i*lda+p]=0
                g=(fp.gemm(m,n,k,a,lda,b,ldb,c,ldc,add) if op=="gemm"
                   else fp.gemv(m,k,a,lda,b,c,add))
                row=dict(name=f"t{t}-{purpose}-{add}",tile=t,purpose=purpose,**shape,
                    lda=lda,ldb=ldb,ldc=ldc,accumulate=add,
                    a_units=a,b_units=b,initial_units=c,golden_units=g)
                row.update(a=[fp.bits(v,3) for v in a],b=[fp.bits(v,3) for v in b],
                           initial=[fp.bits(v,6) for v in c],golden=[fp.bits(v,6) for v in g])
                rows.append(row)
    return rows

def generate(manifest,out,config):
    rows=create(json.loads(Path(manifest).read_text()))
    return emit(rows,out,config,'independent exact dyadic logical matrices; strict FP32 bits')

def emit(rows,out,config,reference):
    if config["max_n"]!=128:raise ValueError("unreviewed FP32 maximum tile")
    widths={key:max(len(r[key]) for r in rows) for key in ("a","b","initial")}
    text=['/* Generated logical FP32 bit images and independent exact GOLDEN. */',
          'struct op_fixture { ame_size tile,m,n,k,lda,ldb,ldc; int accumulate;',
          '  ame_size a_count,b_count,c_count; const u32 *a,*b,*initial,*golden; };']
    for i,r in enumerate(rows):
        for key in ("a","b","initial","golden"):
            values=r[key] or [0]
            text.append(f'static const u32 op_{i}_{key}[] = {{'+
                        ','.join(f'0x{v:08x}u' for v in values)+'};')
    text.append('static const struct op_fixture op_fixtures[] = {')
    for i,r in enumerate(rows):
        values=[str(r[k]) for k in ("tile","m","n","k","lda","ldb","ldc","accumulate")]
        values += [str(len(r[k])) for k in ("a","b","initial")]
        values += [f'op_{i}_{k}' for k in ("a","b","initial","golden")]
        text.append('  {'+','.join(values)+'},')
    per=len(rows)//6
    text += ['};',f'#define OP_CASES {len(rows)}',f'#define OP_SELECTED_CASES {per}',
             f'#define OP_MAX_A {widths["a"]}',f'#define OP_MAX_B {widths["b"]}',
             f'#define OP_MAX_C {widths["initial"]}','']
    (out/'op_fixtures.h').write_text('\n'.join(text))
    (out/'golden.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    checks={str(t):6+4*(128**2+32)+2+per*(4+sum(widths.values())+96+
                3*128**2+32-3*t*t) for t in TILES}
    return dict(logical_cases=len(rows),cases_by_n={str(t):per for t in TILES},
        checks_by_n=checks,widths=widths,reference=reference,
        files=['op_fixtures.h','golden.json'])
