"""Widening API/state cases with independent logical integer GOLDEN."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import integer

COUNT, BYTE_GUARD, WORD_GUARD = 64,17,16
BYTE_CANARY, WORD_CANARY = 0xa7,0x6b91d237
ACTIONS = (
    "VALID_SET","VALID_ADD","REPEATED","INPUT_SHARED",
    "ZERO_ROWS","ZERO_COLS","ZERO_K_SET","ZERO_K_ADD","BAD_MODE","BAD_MODE_EMPTY",
    "NULL_A","NULL_B","NULL_C","UNALIGNED_C","SHORT_LDA","SHORT_LDB","SHORT_LDC",
    "C_ALIAS_A","C_ALIAS_B","C_OVERLAP_A","C_OVERLAP_B",
    "NULL_WS","SHORT_WS","UNALIGNED_WS","WS_ALIAS_A","WS_ALIAS_B","WS_ALIAS_C",
    "INPUT_SPAN_OVERFLOW","B_SPAN_OVERFLOW","OUTPUT_SPAN_OVERFLOW",
    "INPUT_ADDRESS_WRAP","B_ADDRESS_WRAP","OUTPUT_ADDRESS_WRAP","WS_ADDRESS_WRAP",
    "OUTPUT_BYTES_OVERFLOW","LARGE_WS_SIZE")
GEMM_ONLY={"ZERO_COLS","SHORT_LDB","SHORT_LDC","B_SPAN_OVERFLOW"}
COMPUTE={"VALID_SET","VALID_ADD","REPEATED","INPUT_SHARED","LARGE_WS_SIZE"}
EMPTY={"ZERO_ROWS","ZERO_COLS","ZERO_K_SET","ZERO_K_ADD"}
WS_ERROR={"NULL_WS","SHORT_WS","UNALIGNED_WS"}

def create(spec):
    op=spec["operator"]
    if op not in ("gemm","gemv") or spec["contract"]!="widen8-state-v2":
        raise ValueError("unknown widening state contract")
    n,ldb,ldc=(2,4,4) if op=="gemm" else (1,1,1)
    a,b=[BYTE_CANARY]*COUNT,[BYTE_CANARY]*COUNT
    c=[WORD_CANARY]*COUNT
    for i,row in enumerate(((128,255,127),(3,254,1))):
        for j,v in enumerate(row): a[BYTE_GUARD+i*5+j]=v
    for i,row in enumerate(((255,2),(128,127),(1,254))):
        for j in range(n): b[BYTE_GUARD+i*ldb+j]=row[j]
    for i,row in enumerate(((7,-5),(11,13))):
        for j in range(n): c[WORD_GUARD+i*ldc+j]=row[j]
    phases=(ACTIONS,
        ("VALID_SET","VALID_ADD","ZERO_ROWS","ZERO_COLS","ZERO_K_SET","ZERO_K_ADD",
         "BAD_MODE","BAD_MODE_EMPTY","NULL_WS","SHORT_WS"),
        ("VALID_SET","VALID_ADD","REPEATED","INPUT_SHARED"))
    cases=[]
    for phase,actions in enumerate(phases):
        for unsigned in (0,1):
            av=[v if unsigned or v<128 else v-256 for v in a[BYTE_GUARD:]]
            bv=[v if unsigned or v<128 else v-256 for v in b[BYTE_GUARD:]]
            for action in actions:
                if op=="gemv" and action in GEMM_ONLY: continue
                status=0 if action in COMPUTE|EMPTY else 2 if action in WS_ERROR else 1
                if phase==1 and action in COMPUTE|WS_ERROR: status=3
                computed=status==0 and action in COMPUTE
                calls=3 if action=="REPEATED" else 1
                want=list(c)
                if computed:
                    if action=="INPUT_SHARED":want[WORD_GUARD]=av[0]*av[0]
                    else:
                        for iteration in range(calls):
                            add=int(action=="VALID_ADD" or iteration!=0)
                            value=(integer.gemm(2,n,3,av,5,bv,ldb,want[WORD_GUARD:],ldc,add)
                                   if op=="gemm" else integer.gemv(2,3,av,5,bv,want[WORD_GUARD:],add))
                            want[WORD_GUARD:]=value
                elif action=="ZERO_K_SET":
                    for i in range(2):
                        for j in range(n):want[WORD_GUARD+i*ldc+j]=0
                cases.append(dict(name=f"{phase}-{'u8' if unsigned else 'i8'}-{action.lower()}",
                    phase=phase,unsigned=unsigned,action=action,action_id=ACTIONS.index(action),
                    status=status,computed=int(computed),calls=calls,golden=want))
    return dict(operator=op,initial_a=a,initial_b=b,initial_c=c,cases=cases)

def checks(data,t,maximum=128):
    return 6+4*(maximum**2+32)+5+sum(
        r["calls"]+1+3*COUNT+3*maximum**2+32-3*t*t*r["computed"]
        +(3+int(not r["computed"]) if r["phase"]!=1 else 0) for r in data["cases"])

def generate(manifest,out,config):
    if config["max_n"]!=128:raise ValueError("unreviewed maximum tile")
    data=create(json.loads(Path(manifest).read_text()))
    lines=["/* Generated widening state contract, with independent integer GOLDEN. */"]
    lines += [f"#define OP_{name} {i}" for i,name in enumerate(ACTIONS)]
    lines += [f"#define OP_WORDS {COUNT}",f"#define OP_BYTE_GUARD {BYTE_GUARD}",
              f"#define OP_WORD_GUARD {WORD_GUARD}",f"#define OP_CANARY {WORD_CANARY}",
              f"#define OP_CASES {len(data['cases'])}"]
    for name,typ in (("initial_a","ame_u8"),("initial_b","ame_u8"),("initial_c","ame_i32")):
        lines.append(f"static const {typ} op_{name}[] = {{"+",".join(map(str,data[name]))+"};")
    lines.append("struct op_state_case { unsigned phase,is_unsigned,action; int status,computed; unsigned calls; ame_i32 golden[OP_WORDS]; };")
    lines.append("static const struct op_state_case op_state_cases[] = {")
    for r in data["cases"]:
        fields=[r[k] for k in ("phase","unsigned","action_id","status","computed","calls")]
        lines.append("{"+",".join(map(str,fields))+",{"+",".join(map(str,r["golden"]))+"}},")
    lines += ["};",""]
    (out/"op_fixtures.h").write_text("\n".join(lines))
    (out/"golden.json").write_text(json.dumps(data,indent=2)+"\n")
    return dict(logical_cases=len(data["cases"]),
        checks_by_n={str(t):checks(data,t) for t in (4,8,16,32,64,128)},
        reference="Hand-sized byte operands, independent integer GOLDEN and API/state contract",
        files=["op_fixtures.h","golden.json"])
