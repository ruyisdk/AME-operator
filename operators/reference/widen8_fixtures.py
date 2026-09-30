"""Logical byte-input cases; GOLDEN never uses AME packing or model code."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import integer

TILES = (4,8,16,32,64,128)
GUARD = 17
def shapes(op,t):
    rows=[]
    for label,k in (("square-minus",t-1),("square",t),("square-plus",t+1),
                    ("packet-minus",4*t-1),("packet",4*t),("packet-plus",4*t+1),
                    ("three-packets",8*t+1)):
        rows.append((label,3,5 if op=="gemm" else 1,k))
    for axis in ("m","n") if op=="gemm" else ("m",):
        for label,value in (("minus",t-1),("exact",t),("plus",t+1)):
            rows.append((axis+"-"+label,value if axis=="m" else 3,
                         value if axis=="n" else (5 if op=="gemm" else 1),5))
    rows += [("cross",t+1,t+1 if op=="gemm" else 1,4*t+1),
             ("high-product",1,1,1),("high-mixed",1,1,1),
             ("zero-k",3,5 if op=="gemm" else 1,0),
             ("zero-m",0,5 if op=="gemm" else 1,3)]
    if op=="gemm": rows.append(("zero-n",3,0,3))
    return rows

def create(spec):
    op=spec["operator"]
    if op not in ("gemm","gemv") or spec["contract"]!="widen8-i32-v1":
        raise ValueError("unknown widening contract")
    rows=[]
    for t in TILES:
        for unsigned in (0,1):
            values=[0,1,127,128,254,255] if unsigned else [-128,-127,-1,0,1,126,127]
            for index,(purpose,m,n,k) in enumerate(shapes(op,t)):
                for add in (0,1):
                    lda,ldb,ldc=k+3,n+3 if op=="gemm" else 1,n+4 if op=="gemm" else 1
                    def data(nr,nc,stride,salt):
                        return [values[((i+1)*0x45d9f3b^(j+1)*0x119de1f3^salt)%len(values)]
                                if j<nc else (165 if unsigned else -91)
                                for i in range(nr) for j in range(stride)]
                    a=data(m,k,lda,t+index);b=data(k,n,ldb,t+index+19)
                    c=[((i+1)*31-(j+1)*19) if j<n else 0x13579b
                       for i in range(m) for j in range(ldc)]
                    if purpose=="high-product": a[0]=255 if unsigned else -128;b[0]=a[0]
                    if purpose=="high-mixed": a[0]=128 if unsigned else -128;b[0]=255 if unsigned else 127
                    golden=(integer.gemm(m,n,k,a,lda,b,ldb,c,ldc,add) if op=="gemm"
                            else integer.gemv(m,k,a,lda,b,c,add))
                    rows.append(dict(name=f"t{t}-{'u8' if unsigned else 'i8'}-{purpose}-{add}",
                        tile=t,unsigned=unsigned,purpose=purpose,m=m,n=n,k=k,lda=lda,ldb=ldb,ldc=ldc,
                        accumulate=add,a=a,b=b,initial=c,golden=golden))
    return rows

def generate(manifest,out,config):
    if config["max_n"]!=128: raise ValueError("unreviewed maximum tile")
    rows=create(json.loads(Path(manifest).read_text()))
    width=max(len(r["initial"]) for r in rows)
    text=['/* Logical fixtures; byte images include odd-offset guards. */',
          'struct op_fixture { ame_size tile,m,n,k,lda,ldb,ldc; int accumulate,is_unsigned;',
          '  ame_size a_count,b_count,c_count; ame_u8 *a,*b;',
          '  const ame_u8 *a_copy,*b_copy; const ame_i32 *initial,*golden; };']
    for i,r in enumerate(rows):
        for key in ("a","b","a_copy","b_copy","initial","golden"):
            byte=key[0] in ("a","b")
            vals=r[key.replace("_copy","")]
            if byte: vals=[0xa7]*GUARD+[v&255 for v in vals]+[0xd3]*GUARD
            if not vals: vals=[0]
            qualifier="" if key in ("a","b") else "const "
            typ="ame_u8" if byte else "ame_i32"
            text.append(f"static {qualifier}{typ} op_{i}_{key}[] __attribute__((aligned(16))) = {{"+
                        ",".join(map(str,vals))+"};")
    text.append("static const struct op_fixture op_fixtures[] = {")
    for i,r in enumerate(rows):
        vals=[str(r[k]) for k in ("tile","m","n","k","lda","ldb","ldc","accumulate","unsigned")]
        vals += [str(len(r[k])) for k in ("a","b","initial")]
        vals += [f"op_{i}_{k}" for k in ("a","b","a_copy","b_copy","initial","golden")]
        text.append("  {"+",".join(vals)+"},")
    per=len(rows)//6
    text += ["};",f"#define OP_CASES {len(rows)}",f"#define OP_MAX_C {width}",
             f"#define OP_SELECTED_CASES {per}","#define OP_INPUT_GUARD 17",""]
    (out/"op_fixtures.h").write_text("\n".join(text))
    (out/"golden.json").write_text(json.dumps(rows,separators=(",",":"))+"\n")
    checks={str(t):6+4*(128**2+32)+2+sum(
        1+width+32+3*128**2+32-3*t*t+len(r["a"])+len(r["b"])+4*GUARD
        for r in rows if r["tile"]==t) for t in TILES}
    return dict(logical_cases=len(rows),cases_by_n={str(t):per for t in TILES},
        checks_by_n=checks,max_c=width,reference="Python arbitrary integer, logical byte operands",
        files=["op_fixtures.h","golden.json"])
