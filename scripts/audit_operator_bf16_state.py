"""Independent BF16 state inventory, literal images and target check counts."""
import ame
from audit_operator_fp32_state import verify_execution

ORDER=(
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

def accounting(build,data,t):
    op=build['row']['operator'];ame.require(op in ('gemm','gemv') and data['operator']==op,'BF16 state operator mismatch')
    valid={'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','LARGE_CAPACITY','INPUT_OFFSET_TWO','OUTPUT_OFFSET_FOUR'}
    empty={'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'}
    missing={'NULL_DESC','NULL_WS_A','NULL_WS_B','NULL_WS_C','SHORT_WS_A','SHORT_WS_B','SHORT_WS_C',
             'UNALIGNED_WS_A','UNALIGNED_WS_B','UNALIGNED_WS_C'}
    phases={0:set(ORDER),
        1:{'VALID_SET','VALID_ADD','ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD',
           'ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD','BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_C',
           'NULL_DESC','UNALIGNED_DESC','DESC_ADDRESS_WRAP','DESC_ALIAS_C',
           'NULL_WS_A','SHORT_WS_B','UNALIGNED_WS_C','WS_C_ALIAS_A','WS_B_ADDRESS_WRAP'},
        2:{'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','INPUT_OFFSET_TWO','OUTPUT_OFFSET_FOUR',
           'ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'}}
    if op=='gemv':
        phases={p:s-{'ZERO_COLS','SHORT_LDB','SHORT_LDC','B_SPAN_OVERFLOW','B_BYTES_OVERFLOW','OUTPUT_SPAN_OVERFLOW'}
                for p,s in phases.items()}
    rows=data['cases'];wanted={(p,a) for p,s in phases.items() for a in s}
    ame.require({(r['phase'],r['action']) for r in rows}==wanted and len(rows)==len(wanted),'BF16 state inventory gap/duplicate')
    n,ldb,ldc=(2,4,4) if op=='gemm' else (1,1,1)
    values={'set':((0xbe100000,0xbe000000),(0xbd800000,0x3ed00000)),
            'add':((0xbd000000,0xbe500000),(0x3de00000,0x3f1c0000)),
            'repeat':((0xbed80000,0xbec00000),(0xbe400000,0x3f9c0000))}
    maximum=build['config']['max_n'];count=6+4*(maximum**2+32)+16
    for r in rows:
        phase,action=r['phase'],r['action'];index=ORDER.index(action)
        status=0 if action in valid|empty else 2 if action in missing else 1
        if phase==1 and action not in empty|{'BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_C'}:status=3
        computed=status==0 and action in valid;calls=3 if action=='REPEATED' else 1
        flags=0 if action.startswith('ZERO_K_SPECIAL') else (0,1,16,31)[index%4]
        ame.require((r['status'],r['computed'],r['calls'],r['action_id'],r['fflags'],r['xsat'],r['fcsr'])==
            (status,int(computed),calls,index,flags,index%2,0x6a if index%2 else 0x85),'wrong BF16 state expectation')
        ia=16+int(action=='INPUT_OFFSET_TWO');oc=16+int(action=='OUTPUT_OFFSET_FOUR')
        a,b,c=[0x7f81]*64,[0x7f81]*64,[0x7f812345]*64
        av=((0x3e80,0xbec0,0x3f00),(0xbe00,0x3f20,0x3e80))
        bv=((0x3ec0,0xbe80),(0x3e00,0x3f00),(0xbec0,0x3e80))
        cv=((0x3de00000,0xbda00000),(0x3e300000,0x3e500000))
        for i in range(2):
            for j in range(3):a[ia+i*5+j]=av[i][j]
            for j in range(n):c[oc+i*ldc+j]=cv[i][j]
        for i in range(3):
            for j in range(n):b[ia+i*ldb+j]=bv[i][j]
        ame.require(r['initial_a']==a and r['initial_b']==b,'literal BF16 inputs changed')
        if action.startswith('ZERO_K_SPECIAL'):
            special=(0x80000000,0x7f800001,0x7fc12345,0x7f800000)
            for i in range(2):
                for j in range(n):c[oc+i*ldc+j]=special[i*n+j]
        ame.require(r['initial']==c,'literal BF16 state seed changed')
        want=list(c)
        if computed:
            if action=='INPUT_SHARED':want[oc]=0x3d800000
            else:
                matrix=values['add' if action=='VALID_ADD' else 'repeat' if action=='REPEATED' else 'set']
                for i in range(2):
                    for j in range(n):want[oc+i*ldc+j]=matrix[i][j]
        elif action in ('ZERO_K_SET','ZERO_K_SPECIAL_SET'):
            for i in range(2):
                for j in range(n):want[oc+i*ldc+j]=0
        ame.require(r['golden']==want,'BF16 state GOLDEN changed')
        # Each invoke checks all six descriptor fields before its status check.
        count+=7*calls+2+192+5*maximum**2+96-(5*t*t if computed else 0)
        if phase!=1:count+=3+int(not computed)
    return rows,count
