"""Hardware-guard call inventory and exact one-product/K=0 bit images."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import state_edges_fixtures as bits
ACTIONS=('SET','ADD','K0_SET','K0_ADD','EMPTY','BAD_MODE','BAD_A','NO_WORK','BAD_WORK')
WORDS,GUARD=32,8

def create(op):
    if op not in ('gemm','gemv'):raise ValueError('unknown operator')
    rows=[]
    for phase in range(3):
        for f in range(5):
            for action in (ACTIONS if phase<2 else ('SET','ADD')):
                mask=(1<<(8*bits.WIDTH[f]))-1
                a=[bits.CANARY&mask]*WORDS;b=list(a);c=[bits.CANARY]*WORDS
                a[GUARD]=bits.LEFT[f];b[GUARD]=bits.RIGHT[f];c[GUARD]=bits.SEED[f]
                spots=[GUARD+(i//2)*3+i%2 if op=='gemm' else GUARD+i for i in range(8)]
                if action.startswith('K0_'):
                    for at,value in zip(spots,bits.SPECIAL):c[at]=value
                want=list(c)
                if phase!=1 and action in ('SET','ADD'):
                    want[GUARD]=(bits.ADDED if action=='ADD' else bits.PRODUCT)[f]
                if action=='K0_SET':
                    for at in spots:want[at]=0
                rows.append(dict(phase=phase,family=f,width=bits.WIDTH[f],action=action,
                    action_id=ACTIONS.index(action),a=a,b=b,initial=c,success=want))
    return dict(operator=op,cases=rows)

def status(row,supported):
    action=row['action']
    if action in ('BAD_MODE','BAD_A'):return 1
    if action in ('K0_SET','K0_ADD','EMPTY'):return 0
    if row['phase']==1:return 3
    if not supported:return 4
    if action=='NO_WORK':return 2
    if action=='BAD_WORK':return 1 if row['family']==4 else 2
    return 0

def checks(data,supported):
    count=6+10
    for r in data['cases']:
        compute=supported and r['phase']!=1 and r['action'] in ('SET','ADD')
        count+=6+3+3*WORDS+(96 if r['family']==4 else 64)+(48 if r['family']==4 else 16)
        if r['phase']!=1:count+=3+int(not compute)
    return count

def generate(manifest,out,config):
    spec=json.loads(Path(manifest).read_text())
    if config['max_n']!=128 or spec['contract']!='platform-guards-v1':raise ValueError('guard contract')
    data=create(spec['operator'])
    lines=['/* Runtime hardware guard lane; numerical kernels are unchanged. */']
    lines += [f'#define OP_{v} {i}' for i,v in enumerate(ACTIONS)]
    lines += [f'#define OP_CASES {len(data["cases"])}',f'#define OP_WORDS {WORDS}',f'#define OP_GUARD {GUARD}',
              f'#define OP_CANARY 0x{bits.CANARY:x}u',
              'struct guard_case { unsigned phase,family,width,action; u32 a[OP_WORDS],b[OP_WORDS],initial[OP_WORDS],success[OP_WORDS]; };',
              'static const struct guard_case op_cases[] = {']
    for r in data['cases']:
        lines.append('{'+','.join(str(r[k]) for k in ('phase','family','width','action_id'))+',')
        lines.append(','.join('{'+','.join(f'0x{x:x}u' for x in r[k])+'}' for k in ('a','b','initial','success'))+'},')
    lines+=['};','']
    (out/'op_fixtures.h').write_text('\n'.join(lines))
    (out/'golden.json').write_text(json.dumps(data,indent=2)+'\n')
    return dict(logical_cases=len(data['cases']),checks_by_support={'yes':checks(data,True),'no':checks(data,False)},
                files=['op_fixtures.h','golden.json'],reference='Exact literals and explicit argument/platform precedence')
