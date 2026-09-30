"""Independent inventory/count review for i32 target state cases."""
import ame

def accounting(build, data, n):
    op=build['row']['operator']
    ame.require(data['operator']==op and op in ('gemm','gemv'), 'state operator mismatch')
    valid={'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','LARGE_WS_SIZE'}
    empty={'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD'}
    ws_error={'NULL_WS','SHORT_WS','UNALIGNED_WS'}
    bad={'BAD_MODE','NULL_A','NULL_B','NULL_C','UNALIGNED_A','UNALIGNED_B','UNALIGNED_C',
         'SHORT_LDA','SHORT_LDB','SHORT_LDC','C_ALIAS_A','C_ALIAS_B','C_OVERLAP_A',
         'WS_ALIAS_A','WS_ALIAS_B','WS_ALIAS_C','INPUT_SPAN_OVERFLOW','OUTPUT_SPAN_OVERFLOW',
         'INPUT_ADDRESS_WRAP','OUTPUT_ADDRESS_WRAP','WS_ADDRESS_WRAP','INPUT_BYTES_OVERFLOW'}
    phases={0:valid|empty|ws_error|bad,
            1:{'VALID_SET','ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','BAD_MODE','NULL_WS'},
            2:{'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED'}}
    if op=='gemv':
        phases={p:names-{'ZERO_COLS','SHORT_LDB','SHORT_LDC'} for p,names in phases.items()}
    rows=data['cases']
    wanted={(p,name) for p,names in phases.items() for name in names}
    actual={(r['phase'],r['action']) for r in rows}
    ame.require(actual==wanted and len(rows)==len(wanted), 'state inventory gap/duplicate')
    ame.require(all(len(data[k])==64 for k in ('initial_a','initial_b','initial_c')),
                'state buffer size mismatch')
    maximum=build['config']['max_n']
    count=6+4*(maximum**2+32)+5
    for row in rows:
        phase, action = row['phase'], row['action']
        status=0 if action in valid|empty else 2 if action in ws_error else 1
        if phase==1 and action in valid|ws_error: status=3
        computed=status==0 and action in valid
        calls=3 if action=='REPEATED' else 1
        ame.require(row['status']==status and row['computed']==int(computed)
                    and row['calls']==calls, 'wrong state expectation')
        ame.require(len(row['golden'])==64, 'wrong output guard extent')
        if status or action in {'ZERO_ROWS','ZERO_COLS','ZERO_K_ADD'}:
            ame.require(row['golden']==data['initial_c'], 'error/no-op modifies expected output')
        count+=calls+1+192+3*maximum**2+32-(3*n*n if computed else 0)
        if phase!=1: count+=3+int(not computed)
    return rows,count
