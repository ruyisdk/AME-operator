"""FP32 ordinary-ABI state cases with exact dyadic and bit-preserving GOLDEN."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import floating_exact as fp

COUNT,GUARD,CANARY=64,16,0x7f812345
ACTIONS=(
    'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','LARGE_WS_SIZE',
    'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD',
    'BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_B','NULL_C',
    'UNALIGNED_A','UNALIGNED_B','UNALIGNED_C','SHORT_LDA','SHORT_LDB','SHORT_LDC',
    'C_ALIAS_A','C_ALIAS_B','C_OVERLAP_A','C_OVERLAP_B',
    'NULL_WS','SHORT_WS','UNALIGNED_WS','WS_ALIAS_A','WS_ALIAS_B','WS_ALIAS_C',
    'INPUT_SPAN_OVERFLOW','B_SPAN_OVERFLOW','OUTPUT_SPAN_OVERFLOW',
    'INPUT_BYTES_OVERFLOW','B_BYTES_OVERFLOW','OUTPUT_BYTES_OVERFLOW',
    'INPUT_ADDRESS_WRAP','B_ADDRESS_WRAP','OUTPUT_ADDRESS_WRAP','WS_ADDRESS_WRAP')
GEMM_ONLY={'ZERO_COLS','SHORT_LDB','SHORT_LDC','B_SPAN_OVERFLOW','B_BYTES_OVERFLOW','OUTPUT_SPAN_OVERFLOW'}
COMPUTE={'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','LARGE_WS_SIZE'}
EMPTY={'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'}
WS_ERROR={'NULL_WS','SHORT_WS','UNALIGNED_WS'}
SPECIAL=(0x80000000,0x7f800001,0x7fc12345,0x7f800000)

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='fp32-state-v3':
        raise ValueError('unknown FP32 state contract')
    n,ldb,ldc=(2,4,4) if op=='gemm' else (1,1,1)
    au,bu,cu=[0]*10,[0]*(3*ldb),[0]*(2*ldc)
    a,b,c=([CANARY]*COUNT for _ in range(3))
    for i,row in enumerate(((2,-3,4),(-1,5,2))):
        for j,v in enumerate(row):
            au[i*5+j]=v;a[GUARD+i*5+j]=fp.bits(v,3)
    for i,row in enumerate(((3,-2),(1,4),(-3,2))):
        for j in range(n):
            bu[i*ldb+j]=row[j];b[GUARD+i*ldb+j]=fp.bits(row[j],3)
    for i,row in enumerate(((7,-5),(11,13))):
        for j in range(n):
            cu[i*ldc+j]=row[j];c[GUARD+i*ldc+j]=fp.bits(row[j],6)
    phases=(ACTIONS,
        ('VALID_SET','VALID_ADD','ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD',
         'ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD','BAD_MODE','BAD_MODE_EMPTY',
         'NULL_A','NULL_C','NULL_WS','SHORT_WS'),
        ('VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'))
    cases=[]
    for phase,actions in enumerate(phases):
        for action in actions:
            if op=='gemv' and action in GEMM_ONLY:continue
            action_id=ACTIONS.index(action)
            status=0 if action in COMPUTE|EMPTY else 2 if action in WS_ERROR else 1
            if phase==1 and action in COMPUTE|WS_ERROR:status=3
            computed=int(status==0 and action in COMPUTE)
            calls=3 if action=='REPEATED' else 1
            initial=list(c)
            if action.startswith('ZERO_K_SPECIAL'):
                for i in range(2):
                    for j in range(n):initial[GUARD+i*ldc+j]=SPECIAL[i*n+j]
            want=list(initial)
            if computed:
                if action=='INPUT_SHARED':want[GUARD]=fp.bits(fp.dot([au[0]],[au[0]]),6)
                else:
                    value=list(cu)
                    for iteration in range(calls):
                        add=int(action=='VALID_ADD' or iteration!=0)
                        value=(fp.gemm(2,n,3,au,5,bu,ldb,value,ldc,add) if op=='gemm'
                               else fp.gemv(2,3,au,5,bu,value,add))
                    for i in range(2):
                        for j in range(n):want[GUARD+i*ldc+j]=fp.bits(value[i*ldc+j],6)
            elif action in ('ZERO_K_SET','ZERO_K_SPECIAL_SET'):
                for i in range(2):
                    for j in range(n):want[GUARD+i*ldc+j]=0
            flags=0 if action.startswith('ZERO_K_SPECIAL') else (0,1,16,31)[action_id%4]
            cases.append(dict(name=f'{phase}-{action.lower()}',phase=phase,action=action,action_id=action_id,
                status=status,computed=computed,calls=calls,fflags=flags,xsat=action_id%2,
                fcsr=0x6a if action_id%2 else 0x85,initial=initial,golden=want))
    return dict(operator=op,initial_a=a,initial_b=b,cases=cases)

def checks(data,t,maximum=128):
    return 6+4*(maximum**2+32)+5+sum(
        r['calls']+2+3*COUNT+3*maximum**2+32-3*t*t*r['computed']
        +(3+int(not r['computed']) if r['phase']!=1 else 0) for r in data['cases'])

def generate(manifest,out,config):
    if config['max_n']!=128:raise ValueError('unreviewed maximum tile')
    data=create(json.loads(Path(manifest).read_text()))
    words=lambda values:','.join(f'0x{v:08x}u' for v in values)
    lines=['/* FP32 logical state contract: integer-scaled GOLDEN and literal bits. */']
    lines += [f'#define OP_{name} {i}' for i,name in enumerate(ACTIONS)]
    lines += [f'#define OP_WORDS {COUNT}',f'#define OP_GUARD {GUARD}',
              f'#define OP_CANARY 0x{CANARY:08x}u',f'#define OP_CASES {len(data["cases"])}']
    for key in ('initial_a','initial_b'):
        lines.append(f'static const u32 op_{key}[] = {{'+words(data[key])+'};')
    lines.append('struct op_state_case { unsigned phase,action; int status,computed; unsigned calls,fflags,xsat,fcsr; u32 initial[OP_WORDS],golden[OP_WORDS]; };')
    lines.append('static const struct op_state_case op_state_cases[] = {')
    for r in data['cases']:
        fields=[r[k] for k in ('phase','action_id','status','computed','calls','fflags','xsat','fcsr')]
        lines.append('{'+','.join(map(str,fields))+',{'+words(r['initial'])+'},{'+words(r['golden'])+'}},')
    lines += ['};','']
    (out/'op_fixtures.h').write_text('\n'.join(lines))
    (out/'golden.json').write_text(json.dumps(data,indent=2)+'\n')
    return dict(logical_cases=len(data['cases']),
        checks_by_n={str(t):checks(data,t) for t in (4,8,16,32,64,128)},
        reference='Independent dyadic logical products and literal bit-preserving/error/state contract',
        files=['op_fixtures.h','golden.json'])
