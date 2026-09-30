"""Hand-sized operator contracts with independent logical integer GOLDEN."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import integer

CANARY = 0x6b91d237
COUNT = 64
GUARD = 16
ACTIONS = (
    'VALID_SET', 'VALID_ADD', 'REPEATED', 'INPUT_SHARED',
    'ZERO_ROWS', 'ZERO_COLS', 'ZERO_K_SET', 'ZERO_K_ADD',
    'BAD_MODE', 'NULL_A', 'NULL_B', 'NULL_C',
    'UNALIGNED_A', 'UNALIGNED_B', 'UNALIGNED_C',
    'SHORT_LDA', 'SHORT_LDB', 'SHORT_LDC',
    'C_ALIAS_A', 'C_ALIAS_B', 'C_OVERLAP_A',
    'NULL_WS', 'SHORT_WS', 'UNALIGNED_WS',
    'WS_ALIAS_A', 'WS_ALIAS_B', 'WS_ALIAS_C',
    'INPUT_SPAN_OVERFLOW', 'OUTPUT_SPAN_OVERFLOW',
    'INPUT_ADDRESS_WRAP', 'OUTPUT_ADDRESS_WRAP', 'WS_ADDRESS_WRAP',
    'INPUT_BYTES_OVERFLOW', 'LARGE_WS_SIZE')
GEMM_ONLY = {'ZERO_COLS', 'SHORT_LDB', 'SHORT_LDC'}
COMPUTE = {'VALID_SET', 'VALID_ADD', 'REPEATED', 'INPUT_SHARED', 'LARGE_WS_SIZE'}
OK = COMPUTE | {'ZERO_ROWS', 'ZERO_COLS', 'ZERO_K_SET', 'ZERO_K_ADD'}
WORKSPACE_ERROR = {'NULL_WS', 'SHORT_WS', 'UNALIGNED_WS'}


def create(spec):
    op = spec['operator']
    if op not in ('gemm', 'gemv') or spec['contract'] != 'i32-state-v3':
        raise ValueError('unsupported state contract')
    n, ldb, ldc = (2, 4, 4) if op == 'gemm' else (1, 1, 1)
    a, b, c = ([CANARY]*COUNT for _ in range(3))
    for i, row in enumerate(((2,-3,4), (-1,5,2))):
        for j, value in enumerate(row): a[GUARD+i*5+j] = value
    for i, row in enumerate(((3,-2),(1,4),(-3,2))):
        for j in range(n): b[GUARD+i*ldb+j] = row[j]
    for i, row in enumerate(((7,-5),(11,13))):
        for j in range(n): c[GUARD+i*ldc+j] = row[j]
    phases = (list(ACTIONS),
        ['VALID_SET','ZERO_ROWS','ZERO_COLS','ZERO_K_SET','ZERO_K_ADD','BAD_MODE','NULL_WS'],
        ['VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED'])
    cases = []
    for phase, actions in enumerate(phases):
        for action in actions:
            if op == 'gemv' and action in GEMM_ONLY: continue
            status = 0 if action in OK else 2 if action in WORKSPACE_ERROR else 1
            if phase == 1 and action in COMPUTE | WORKSPACE_ERROR:
                status = 3
            computed = status == 0 and action in COMPUTE
            calls = 3 if action == 'REPEATED' else 1
            want = list(c)
            if computed:
                if action == 'INPUT_SHARED':
                    want[GUARD] = 4
                else:
                    for iteration in range(calls):
                        add = int(action == 'VALID_ADD' or iteration != 0)
                        if op == 'gemm':
                            value = integer.gemm(2,n,3,a[GUARD:],5,b[GUARD:],ldb,
                                                 want[GUARD:],ldc,add)
                        else:
                            value = integer.gemv(2,3,a[GUARD:],5,b[GUARD:],want[GUARD:],add)
                        want[GUARD:] = value
            elif action == 'ZERO_K_SET':
                for i in range(2):
                    for j in range(n): want[GUARD+i*ldc+j] = 0
            cases.append(dict(name=f'{phase}-{action.lower()}', phase=phase,
                action=action, action_id=ACTIONS.index(action), status=status,
                computed=int(computed), calls=calls, golden=want))
    return dict(operator=op, initial_a=a, initial_b=b, initial_c=c, cases=cases)


def checks(data, n, maximum=128):
    return 6+4*(maximum*maximum+32)+5+sum(
        r['calls']+1+3*COUNT+3*maximum*maximum+32-3*n*n*r['computed']
        +(3+int(not r['computed']) if r['phase'] != 1 else 0)
        for r in data['cases'])


def generate(manifest, out, config):
    if config['max_n'] != 128:
        raise ValueError('state v3 is reviewed for max_n=128')
    data = create(json.loads(Path(manifest).read_text()))
    lines = ['/* Generated contract cases; expected numbers are host GOLDEN. */']
    lines += [f'#define OP_{name} {i}' for i, name in enumerate(ACTIONS)]
    lines += [f'#define OP_WORDS {COUNT}', f'#define OP_GUARD {GUARD}',
              f'#define OP_CANARY {CANARY}', f'#define OP_CASES {len(data["cases"])}']
    for name in ('initial_a', 'initial_b', 'initial_c'):
        lines.append(f'static const ame_i32 op_{name}[] = {{'+','.join(map(str,data[name]))+'};')
    lines.append('struct op_state_case { unsigned phase,action; int status,computed; unsigned calls; ame_i32 golden[OP_WORDS]; };')
    lines.append('static const struct op_state_case op_state_cases[] = {')
    for row in data['cases']:
        fields = [row[k] for k in ('phase','action_id','status','computed','calls')]
        lines.append('{'+','.join(map(str,fields))+',{'+','.join(map(str,row['golden']))+'}},')
    lines += ['};', '']
    (out/'op_fixtures.h').write_text('\n'.join(lines))
    (out/'golden.json').write_text(json.dumps(data,indent=2)+'\n')
    return dict(logical_cases=len(data['cases']),
        checks_by_n={str(n):checks(data,n) for n in (4,8,16,32,64,128)},
        reference='Hand-sized independent integer GOLDEN and explicit API/state contract',
        files=['op_fixtures.h','golden.json'])
