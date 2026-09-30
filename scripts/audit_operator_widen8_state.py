"""Independent status, state inventory, hand results and count checks."""
import ame

def accounting(build,data,t):
    op=build["row"]["operator"]
    ame.require(data["operator"]==op and op in ("gemm","gemv"),"widening state operator")
    valid={"VALID_SET","VALID_ADD","REPEATED","INPUT_SHARED","LARGE_WS_SIZE"}
    empty={"ZERO_ROWS","ZERO_COLS","ZERO_K_SET","ZERO_K_ADD"}
    ws={"NULL_WS","SHORT_WS","UNALIGNED_WS"}
    bad={"BAD_MODE","BAD_MODE_EMPTY","NULL_A","NULL_B","NULL_C","UNALIGNED_C",
         "SHORT_LDA","SHORT_LDB","SHORT_LDC","C_ALIAS_A","C_ALIAS_B","C_OVERLAP_A","C_OVERLAP_B",
         "WS_ALIAS_A","WS_ALIAS_B","WS_ALIAS_C","INPUT_SPAN_OVERFLOW","B_SPAN_OVERFLOW",
         "OUTPUT_SPAN_OVERFLOW","INPUT_ADDRESS_WRAP","B_ADDRESS_WRAP","OUTPUT_ADDRESS_WRAP",
         "WS_ADDRESS_WRAP","OUTPUT_BYTES_OVERFLOW"}
    phases={0:valid|empty|ws|bad,
        1:{"VALID_SET","VALID_ADD","ZERO_ROWS","ZERO_COLS","ZERO_K_SET","ZERO_K_ADD",
           "BAD_MODE","BAD_MODE_EMPTY","NULL_WS","SHORT_WS"},
        2:{"VALID_SET","VALID_ADD","REPEATED","INPUT_SHARED"}}
    if op=="gemv":
        phases={p:names-{"ZERO_COLS","SHORT_LDB","SHORT_LDC","B_SPAN_OVERFLOW"} for p,names in phases.items()}
    wanted={(p,u,a) for p,names in phases.items() for u in (0,1) for a in names}
    rows=data["cases"]
    actual={(r["phase"],r["unsigned"],r["action"]) for r in rows}
    ame.require(actual==wanted and len(rows)==len(wanted),"widening state inventory gap/duplicate")
    ame.require(all(len(data[k])==64 for k in ("initial_a","initial_b","initial_c")),"state extent")
    a,b,c=[0xa7]*64,[0xa7]*64,[0x6b91d237]*64
    for i,v in {17:128,18:255,19:127,22:3,23:254,24:1}.items():a[i]=v
    for i,v in ({17:255,18:2,21:128,22:127,25:1,26:254} if op=="gemm"
                else {17:255,18:128,19:1}).items():b[i]=v
    for i,v in ({16:7,17:-5,20:11,21:13} if op=="gemm" else {16:7,17:11}).items():c[i]=v
    ame.require((data["initial_a"],data["initial_b"],data["initial_c"])==(a,b,c),
                "widening hand inputs/guards changed")
    # Literal expected matrices are independently hand-derived, not fixture helper calls.
    products={0:[383,-637,254,-250],1:[65407,64899,33278,32518]}
    seed=[7,-5,11,13]
    indices=[16,17,20,21] if op=="gemm" else [16,17]
    maximum=build["config"]["max_n"];count=6+4*(maximum**2+32)+5
    for r in rows:
        action,phase,u=r["action"],r["phase"],r["unsigned"]
        status=0 if action in valid|empty else 2 if action in ws else 1
        if phase==1 and action in valid|ws:status=3
        computed=status==0 and action in valid;calls=3 if action=="REPEATED" else 1
        ame.require((r["status"],r["computed"],r["calls"])==(status,int(computed),calls),
                    "wrong widening state expectation")
        want=list(data["initial_c"])
        if action=="ZERO_K_SET":
            for index in indices:want[index]=0
        elif computed:
            if action=="INPUT_SHARED":want[16]=16384
            else:
                values=products[u] if op=="gemm" else products[u][::2]
                old=seed if op=="gemm" else seed[::2]
                for index,v,s in zip(indices,values,old):
                    want[index]=v*calls+(s if action=="VALID_ADD" else 0)
        ame.require(r["golden"]==want,"wrong widening state GOLDEN")
        count+=calls+1+192+3*maximum**2+32-(3*t*t if computed else 0)
        if phase!=1:count+=3+int(not computed)
    return rows,count
