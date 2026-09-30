"""Native BF16 ordinary-call state cases; logical exact and literal GOLDEN."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import floating_exact as fp
COUNT,GUARD,CANARY16,CANARY32=64,16,0x7f81,0x7f812345
ACTIONS=(
    'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','LARGE_CAPACITY',
    'INPUT_OFFSET_TWO','OUTPUT_OFFSET_FOUR',
    'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD',
    'BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_B','NULL_C',
    'UNALIGNED_A','UNALIGNED_B','UNALIGNED_C','SHORT_LDA','SHORT_LDB','SHORT_LDC',
    'C_ALIAS_A','C_ALIAS_B','C_OVERLAP_A','C_OVERLAP_B',
    'NULL_DESC','UNALIGNED_DESC','DESC_ADDRESS_WRAP','DESC_ALIAS_A','DESC_ALIAS_B','DESC_ALIAS_C',
    'NULL_WS_A','NULL_WS_B','NULL_WS_C','SHORT_WS_A','SHORT_WS_B','SHORT_WS_C',
    'UNALIGNED_WS_A','UNALIGNED_WS_B','UNALIGNED_WS_C',
    'WS_A_ALIAS_A','WS_A_ALIAS_B','WS_A_ALIAS_C',
    'WS_B_ALIAS_A','WS_B_ALIAS_B','WS_B_ALIAS_C',
    'WS_C_ALIAS_A','WS_C_ALIAS_B','WS_C_ALIAS_C',
    'WS_A_ALIAS_B_WORK','WS_A_ALIAS_C_WORK','WS_B_ALIAS_C_WORK',
    'WS_A_ALIAS_DESC','WS_B_ALIAS_DESC','WS_C_ALIAS_DESC',
    'WS_A_ADDRESS_WRAP','WS_B_ADDRESS_WRAP','WS_C_ADDRESS_WRAP',
    'INPUT_SPAN_OVERFLOW','B_SPAN_OVERFLOW','OUTPUT_SPAN_OVERFLOW',
    'INPUT_BYTES_OVERFLOW','B_BYTES_OVERFLOW','OUTPUT_BYTES_OVERFLOW',
    'INPUT_ADDRESS_WRAP','B_ADDRESS_WRAP','OUTPUT_ADDRESS_WRAP')
GEMM_ONLY={'ZERO_COLS','SHORT_LDB','SHORT_LDC','B_SPAN_OVERFLOW','B_BYTES_OVERFLOW','OUTPUT_SPAN_OVERFLOW'}
COMPUTE=set(ACTIONS[:7])
EMPTY={'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'}
WS_ERROR={'NULL_DESC'}|{prefix+suffix for prefix in ('NULL_WS_','SHORT_WS_','UNALIGNED_WS_') for suffix in 'ABC'}
POST_PLATFORM=set(ACTIONS[ACTIONS.index('NULL_DESC'):ACTIONS.index('INPUT_SPAN_OVERFLOW')])
SPECIAL=(0x80000000,0x7f800001,0x7fc12345,0x7f800000)

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='bf16-state-v1':raise ValueError('unknown BF16 state contract')
    n,ldb,ldc=(2,4,4) if op=='gemm' else (1,1,1)
    phases=(ACTIONS,
        ('VALID_SET','VALID_ADD','ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD',
         'ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD','BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_C',
         'NULL_DESC','UNALIGNED_DESC','DESC_ADDRESS_WRAP','DESC_ALIAS_C',
         'NULL_WS_A','SHORT_WS_B','UNALIGNED_WS_C','WS_C_ALIAS_A','WS_B_ADDRESS_WRAP'),
        ('VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','INPUT_OFFSET_TWO','OUTPUT_OFFSET_FOUR',
         'ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'))
    rows=[]
    for phase,actions in enumerate(phases):
        for action in actions:
            if op=='gemv' and action in GEMM_ONLY:continue
            index=ACTIONS.index(action);ia=GUARD+int(action=='INPUT_OFFSET_TWO')
            oc=GUARD+int(action=='OUTPUT_OFFSET_FOUR')
            a,b,c=[CANARY16]*COUNT,[CANARY16]*COUNT,[CANARY32]*COUNT
            au,bu,cu=[0]*10,[0]*(3*ldb),[0]*(2*ldc)
            for i,row in enumerate(((2,-3,4),(-1,5,2))):
                for j,v in enumerate(row):au[i*5+j]=v;a[ia+i*5+j]=fp.bits(v,3,'bf16')
            for i,row in enumerate(((3,-2),(1,4),(-3,2))):
                for j in range(n):bu[i*ldb+j]=row[j];b[ia+i*ldb+j]=fp.bits(row[j],3,'bf16')
            for i,row in enumerate(((7,-5),(11,13))):
                for j in range(n):cu[i*ldc+j]=row[j];c[oc+i*ldc+j]=fp.bits(row[j],6)
            status=0 if action in COMPUTE|EMPTY else 2 if action in WS_ERROR else 1
            if phase==1 and action in COMPUTE|POST_PLATFORM:status=3
            computed=int(status==0 and action in COMPUTE);calls=3 if action=='REPEATED' else 1
            if action.startswith('ZERO_K_SPECIAL'):
                for i in range(2):
                    for j in range(n):c[oc+i*ldc+j]=SPECIAL[i*n+j]
            want=list(c)
            if computed:
                if action=='INPUT_SHARED':want[oc]=fp.bits(fp.dot([au[0]],[au[0]],input_format='bf16'),6)
                else:
                    value=list(cu)
                    for iteration in range(calls):
                        add=int(action=='VALID_ADD' or iteration!=0)
                        value=fp.gemm(2,n,3,au,5,bu,ldb,value,ldc,add,input_format='bf16')
                    for i in range(2):
                        for j in range(n):want[oc+i*ldc+j]=fp.bits(value[i*ldc+j],6)
            elif action in ('ZERO_K_SET','ZERO_K_SPECIAL_SET'):
                for i in range(2):
                    for j in range(n):want[oc+i*ldc+j]=0
            rows.append(dict(name=f'{phase}-{action.lower()}',phase=phase,action=action,action_id=index,
                status=status,computed=computed,calls=calls,
                fflags=0 if action.startswith('ZERO_K_SPECIAL') else (0,1,16,31)[index%4],
                xsat=index%2,fcsr=0x6a if index%2 else 0x85,
                initial_a=a,initial_b=b,initial=c,golden=want))
    return dict(operator=op,cases=rows)

def checks(data,t,maximum=128):
    return 6+4*(maximum**2+32)+16+sum(
        7*r['calls']+2+3*COUNT+5*maximum**2+96-5*t*t*r['computed']
        +(3+int(not r['computed']) if r['phase']!=1 else 0) for r in data['cases'])

def generate(manifest,out,config):
    if config['max_n']!=128:raise ValueError('unreviewed maximum tile')
    data=create(json.loads(Path(manifest).read_text()))
    words=lambda values:','.join(f'0x{v:x}u' for v in values)
    lines=['/* BF16 state contract: logical scaled-integer products and literal bits. */',
           'typedef __UINT16_TYPE__ op_u16;']
    lines += [f'#define OP_{name} {i}' for i,name in enumerate(ACTIONS)]
    lines += [f'#define OP_WORDS {COUNT}',f'#define OP_GUARD {GUARD}',
              f'#define OP_CANARY16 0x{CANARY16:x}u',f'#define OP_CANARY32 0x{CANARY32:x}u',
              f'#define OP_CASES {len(data["cases"])}',
              'struct op_state_case { unsigned phase,action; int status,computed; unsigned calls,fflags,xsat,fcsr;',
              '  op_u16 a[OP_WORDS],b[OP_WORDS]; u32 initial[OP_WORDS],golden[OP_WORDS]; };',
              'static const struct op_state_case op_state_cases[] = {']
    for r in data['cases']:
        fields=[r[k] for k in ('phase','action_id','status','computed','calls','fflags','xsat','fcsr')]
        lines.append('{'+','.join(map(str,fields))+',')
        lines.append(','.join('{'+words(r[k])+'}' for k in ('initial_a','initial_b','initial','golden'))+'},')
    lines += ['};','']
    (out/'op_fixtures.h').write_text('\n'.join(lines))
    (out/'golden.json').write_text(json.dumps(data,indent=2)+'\n')
    return dict(logical_cases=len(data['cases']),checks_by_n={str(t):checks(data,t) for t in (4,8,16,32,64,128)},
        reference='Independent logical BF16 dyadics and literal state/bit-preserving GOLDEN',
        files=['op_fixtures.h','golden.json'])
