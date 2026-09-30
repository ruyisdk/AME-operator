"""Independent literal/count audit for FP32 state and ordinary-call cases."""
import ame
import spike_fpstate_adapter

ORDER=(
    'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','LARGE_WS_SIZE',
    'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD',
    'BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_B','NULL_C',
    'UNALIGNED_A','UNALIGNED_B','UNALIGNED_C','SHORT_LDA','SHORT_LDB','SHORT_LDC',
    'C_ALIAS_A','C_ALIAS_B','C_OVERLAP_A','C_OVERLAP_B',
    'NULL_WS','SHORT_WS','UNALIGNED_WS','WS_ALIAS_A','WS_ALIAS_B','WS_ALIAS_C',
    'INPUT_SPAN_OVERFLOW','B_SPAN_OVERFLOW','OUTPUT_SPAN_OVERFLOW',
    'INPUT_BYTES_OVERFLOW','B_BYTES_OVERFLOW','OUTPUT_BYTES_OVERFLOW',
    'INPUT_ADDRESS_WRAP','B_ADDRESS_WRAP','OUTPUT_ADDRESS_WRAP','WS_ADDRESS_WRAP')

def accounting(build,data,t):
    op=build['row']['operator']
    ame.require(op in ('gemm','gemv') and data['operator']==op,'FP32 state operator mismatch')
    valid={'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','LARGE_WS_SIZE'}
    empty={'ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'}
    ws={'NULL_WS','SHORT_WS','UNALIGNED_WS'}
    phases={0:set(ORDER),
        1:{'VALID_SET','VALID_ADD','ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD',
           'ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD','BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_C','NULL_WS','SHORT_WS'},
        2:{'VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED','ZERO_K_SPECIAL_SET','ZERO_K_SPECIAL_ADD'}}
    if op=='gemv':
        phases={p:s-{'ZERO_COLS','SHORT_LDB','SHORT_LDC','B_SPAN_OVERFLOW','B_BYTES_OVERFLOW','OUTPUT_SPAN_OVERFLOW'}
                for p,s in phases.items()}
    rows=data['cases'];wanted={(p,a) for p,s in phases.items() for a in s}
    ame.require({(r['phase'],r['action']) for r in rows}==wanted and len(rows)==len(wanted),'FP32 state inventory gap/duplicate')
    n,ldb,ldc=(2,4,4) if op=='gemm' else (1,1,1)
    a,b,c=([0x7f812345]*64 for _ in range(3))
    av=((0x3e800000,0xbec00000,0x3f000000),(0xbe000000,0x3f200000,0x3e800000))
    bv=((0x3ec00000,0xbe800000),(0x3e000000,0x3f000000),(0xbec00000,0x3e800000))
    cv=((0x3de00000,0xbda00000),(0x3e300000,0x3e500000))
    for i in range(2):
        for j in range(3):a[16+i*5+j]=av[i][j]
        for j in range(n):c[16+i*ldc+j]=cv[i][j]
    for i in range(3):
        for j in range(n):b[16+i*ldb+j]=bv[i][j]
    ame.require(data['initial_a']==a and data['initial_b']==b,'literal FP32 inputs changed')
    values={'set':((0xbe100000,0xbe000000),(0xbd800000,0x3ed00000)),
            'add':((0xbd000000,0xbe500000),(0x3de00000,0x3f1c0000)),
            'repeat':((0xbed80000,0xbec00000),(0xbe400000,0x3f9c0000))}
    special=(0x80000000,0x7f800001,0x7fc12345,0x7f800000)
    maximum=build['config']['max_n'];count=6+4*(maximum**2+32)+5
    for r in rows:
        p,action=r['phase'],r['action'];index=ORDER.index(action)
        status=0 if action in valid|empty else 2 if action in ws else 1
        if p==1 and action in valid|ws:status=3
        computed=status==0 and action in valid;calls=3 if action=='REPEATED' else 1
        flags=0 if action.startswith('ZERO_K_SPECIAL') else (0,1,16,31)[index%4]
        ame.require((r['status'],r['computed'],r['calls'],r['action_id'],r['fflags'],r['xsat'],r['fcsr'])==
            (status,int(computed),calls,index,flags,index%2,0x6a if index%2 else 0x85),'wrong FP32 state expectation')
        initial=list(c)
        if action.startswith('ZERO_K_SPECIAL'):
            for i in range(2):
                for j in range(n):initial[16+i*ldc+j]=special[i*n+j]
        ame.require(r['initial']==initial,'literal FP32 seed changed')
        want=list(initial)
        if computed:
            if action=='INPUT_SHARED':want[16]=0x3d800000
            else:
                vals=values['add' if action=='VALID_ADD' else 'repeat' if action=='REPEATED' else 'set']
                for i in range(2):
                    for j in range(n):want[16+i*ldc+j]=vals[i][j]
        elif action in ('ZERO_K_SET','ZERO_K_SPECIAL_SET'):
            for i in range(2):
                for j in range(n):want[16+i*ldc+j]=0
        ame.require(r['golden']==want,'FP32 state GOLDEN changed')
        count+=calls+2+192+3*maximum**2+32-(3*t*t if computed else 0)
        if p!=1:count+=3+int(not computed)
    return rows,count

def verify_execution(build,directory,response):
    command=ame.load(directory/'spike-command.json')
    ame.require(f'--isa=rv{build["row"]["xlen"]}imf_zicsr_ztt' in command['argv'],
                'standard F not explicitly enabled for FCSR observation')
    observed,stop,mailbox=spike_fpstate_adapter.parse_debug(
        (directory/'spike.log').read_text(),build['symbols'],ame.CSR_MAP)
    ame.require((observed,stop,mailbox.hex())==
                (response['observed'],response['stop_pc'],response['mailbox_hex']),
                'raw FP state observations differ')
