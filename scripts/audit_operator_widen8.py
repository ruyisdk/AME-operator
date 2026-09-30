"""Independent widening coverage and logical integer result checks."""
import ame

def selected(build, rows, t):
    op=build["row"]["operator"]
    ame.require(op in ("gemm","gemv"),"unknown widening operator")
    shapes={}
    for label,k in (("square-minus",t-1),("square",t),("square-plus",t+1),
                    ("packet-minus",4*t-1),("packet",4*t),("packet-plus",4*t+1),
                    ("three-packets",8*t+1)):
        shapes[label]=(3,5 if op=="gemm" else 1,k)
    for axis in ("m","n") if op=="gemm" else ("m",):
        for label,value in (("minus",t-1),("exact",t),("plus",t+1)):
            shapes[axis+"-"+label]=(value if axis=="m" else 3,
                                    value if axis=="n" else (5 if op=="gemm" else 1),5)
    shapes.update({"cross":(t+1,t+1 if op=="gemm" else 1,4*t+1),
                   "high-product":(1,1,1),"high-mixed":(1,1,1),
                   "zero-k":(3,5 if op=="gemm" else 1,0),
                   "zero-m":(0,5 if op=="gemm" else 1,3)})
    if op=="gemm": shapes["zero-n"]=(3,0,3)
    cases=[r for r in rows if r["tile"]==t]
    wanted={(p,u,add,*shape) for p,shape in shapes.items() for u in (0,1) for add in (0,1)}
    actual={(r["purpose"],r["unsigned"],r["accumulate"],r["m"],r["n"],r["k"]) for r in cases}
    ame.require(actual==wanted and len(cases)==len(wanted),"widening coverage gap/duplicate")
    for r in cases:
        m,n,k=r["m"],r["n"],r["k"]
        ame.require((r["lda"],r["ldb"],r["ldc"])==
                    (k+3,n+3 if op=="gemm" else 1,n+4 if op=="gemm" else 1),
                    "widening stride changed")
        ame.require(len(r["a"])==m*r["lda"] and len(r["b"])==k*r["ldb"]
                    and len(r["initial"])==len(r["golden"])==m*r["ldc"],"short fixture")
        lo,hi=(0,255) if r["unsigned"] else (-128,127)
        ame.require(all(type(v) is int and lo<=v<=hi for v in r["a"]+r["b"]),"not byte operands")
        if r["purpose"] in ("high-product","high-mixed"):
            a=255 if r["unsigned"] and r["purpose"]=="high-product" else 128 if r["unsigned"] else -128
            b=a if r["purpose"]=="high-product" else 255 if r["unsigned"] else 127
            ame.require((r["a"][0],r["b"][0])==(a,b),"wrong widening signedness boundary")
        for i in range(m):
            for j in range(r["ldc"]):
                slot=i*r["ldc"]+j
                if j>=n:
                    ame.require(r["golden"][slot]==r["initial"][slot],"padding changed in GOLDEN")
                    continue
                seed=r["initial"][slot] if r["accumulate"] else 0
                terms=[r["a"][i*r["lda"]+p]*r["b"][p*r["ldb"]+j] for p in range(k)]
                ame.require(abs(seed)+sum(abs(v) for v in terms)<=2147483647,"outside i32 bound")
                ame.require(r["golden"][slot]==seed+sum(terms),"widening GOLDEN mismatch")
    return cases

def accounting(build,rows,t):
    cases=selected(build,rows,t)
    maximum=build["config"]["max_n"]
    width=max(len(r["initial"]) for r in rows)
    # Two driver counters; source buffers have 17-byte guards at either end.
    count=6+4*(maximum**2+32)+2
    count+=sum(1+width+32+3*maximum**2+32-3*t*t+len(r["a"])+len(r["b"])+68
               for r in cases)
    return cases,count
