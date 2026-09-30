"""Cross-family ordinary-call state edges, logical bits and literal answers."""
import json
from pathlib import Path
ACTIONS=('SET','ADD','K0_SET','K0_ADD','EMPTY','BAD_MODE','NO_WORK','BAD_A')
FAMILIES=('i32','i8','u8','fp32','bf16')
WIDTH=(4,1,1,4,2)
LEFT=(0xfffffff9,0xfd,200,0xbf000000,0xbf00)
RIGHT=(11,7,3,0x3fc00000,0x3fc0)
SEED=(9,9,9,0x3e800000,0x3e800000)
PRODUCT=(0xffffffb3,0xffffffeb,600,0xbf400000,0xbf400000)
ADDED=(0xffffffbc,0xfffffff4,609,0xbf000000,0xbf000000)
SPECIAL=(0x80000000,0x7f800001,0x7fc12345,0x7f800000,
         0xff800000,1,0x7f7fffff,0)
WORDS,GUARD,CANARY=64,16,0x7f812345

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='state-edges-all-v1':
        raise ValueError('unknown state-edge contract')
    rows=[]
    phases=((0,(0,1),range(5),ACTIONS),
            (1,(0,),range(5),('SET','K0_SET','K0_ADD','EMPTY','BAD_MODE','NO_WORK','BAD_A')),
            (2,(1,),range(4,-1,-1),('SET','ADD')))
    for phase,old_values,families,actions in phases:
        for old in old_values:
            for family in families:
                for action in actions:
                    status=1 if action in ('BAD_MODE','BAD_A') else 2 if action=='NO_WORK' else 0
                    if phase==1 and action in ('SET','ADD','NO_WORK'):status=3
                    computed=int(status==0 and action in ('SET','ADD'))
                    a=[CANARY & ((1<<(8*WIDTH[family]))-1)]*WORDS;b=list(a)
                    a[GUARD]=LEFT[family];b[GUARD]=RIGHT[family]
                    c=[CANARY]*WORDS;c[GUARD]=SEED[family]
                    if action.startswith('K0_'):
                        for q,bits in enumerate(SPECIAL):
                            c[GUARD+(q//2)*3+q%2 if op=='gemm' else GUARD+q]=bits
                    want=list(c)
                    if computed:want[GUARD]=(ADDED if action=='ADD' else PRODUCT)[family]
                    if action=='K0_SET':
                        for q in range(8):want[GUARD+(q//2)*3+q%2 if op=='gemm' else GUARD+q]=0
                    ordinal=len(rows)
                    rows.append(dict(name=f'{phase}-{old}-{FAMILIES[family]}-{action.lower()}',
                        phase=phase,old_un=old,family=family,width=WIDTH[family],action=action,
                        action_id=ACTIONS.index(action),status=status,computed=computed,
                        flags=(0,1,16,31)[ordinal%4],xsat=ordinal%2,
                        fcsr=0x6a if ordinal%2 else 0x85,a=a,b=b,initial=c,golden=want))
    return dict(operator=op,cases=rows)

def checks(data,t,maximum=128):
    return 6+4*(maximum**2+32)+9+sum(
        7+2+3*WORDS+3*maximum**2+96-3*t*t*r['computed']
        +(4+int(not r['computed']) if r['phase']!=1 else 0) for r in data['cases'])

def generate(manifest,out,config):
    if config['max_n']!=128:raise ValueError('unreviewed maximum tile')
    data=create(json.loads(Path(manifest).read_text()))
    lines=['/* Logical cross-family state and bit-preservation witnesses. */']
    lines += [f'#define OP_{action} {i}' for i,action in enumerate(ACTIONS)]
    lines += [f'#define OP_WORDS {WORDS}',f'#define OP_GUARD {GUARD}',f'#define OP_CANARY 0x{CANARY:x}u',
              f'#define OP_CASES {len(data["cases"])}',
              'struct op_state_case { unsigned phase,old_un,family,width,action; int status,computed;',
              '  unsigned flags,xsat,fcsr; u32 a[OP_WORDS],b[OP_WORDS],initial[OP_WORDS],golden[OP_WORDS]; };',
              'static const struct op_state_case op_cases[] = {']
    for r in data['cases']:
        values=[r[k] for k in ('phase','old_un','family','width','action_id','status','computed','flags','xsat','fcsr')]
        lines.append('{'+','.join(map(str,values))+',')
        lines.append(','.join('{'+','.join(f'0x{v:x}u' for v in r[k])+'}' for k in ('a','b','initial','golden'))+'},')
    lines+=['};','']
    (out/'op_fixtures.h').write_text('\n'.join(lines))
    (out/'golden.json').write_text(json.dumps(data,indent=2)+'\n')
    return dict(logical_cases=len(data['cases']),
        checks_by_n={str(t):checks(data,t) for t in (4,8,16,32,64,128)},
        reference='Independent exact one-product literals and state/bit preservation',
        files=['op_fixtures.h','golden.json'])
