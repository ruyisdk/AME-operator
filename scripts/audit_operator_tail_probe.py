#!/usr/bin/env python3
"""Exact before/after audit for two literal FP32 padding witnesses."""
import argparse
from pathlib import Path
import struct
import ame
from resource_capture_faults import validate_response

TILES=(4,8,16,32,64,128)

def expected(op,n,before):
    ame.require(op in ('gemm','gemv') and n in TILES,'unknown tail witness')
    v=TILES.index(n)
    if before:
        return (dict(status=3,checks=6,error=1,index=0x72010000+v,expected=0,actual=16)
                if op=='gemm' else dict(status=3,checks=24,error=1,
                    index=0x73000000+v*0x10000+16,expected=0x80000000,actual=0))
    widths=12 if op=='gemm' else 5
    count=6+4*(128**2+32)+2+4+widths+96+3*128**2+32-3*n*n
    return dict(status=2,checks=count,error=0,index=0,expected=0,actual=0)

def audit(root,before,out):
    rows=ame.load(root/'build/plan.json')['rows']
    ame.require({(r['operator'],r['xlen'],r['opt'],r['language']) for r in rows}==
                {(op,x,'O2','c') for op in ('gemm','gemv') for x in (32,64)} and
                len(rows)==4,'wrong diagnostic build matrix')
    builds={}
    for p in sorted((root/'build').glob('shard-*/build/*/build.json')):
        b=ame.load(p);key=b['row']['key'];builds[key]=b
        ame.require(b['row']['variant']=='fp32-tail-probe-v1','wrong variant')
        ame.require(ame.sha(b['elf'])==b['elf_sha256'],'ELF changed')
        ame.require(all(ame.sha(f)==h for f,h in b['generated'].items()),'fixture changed')
        for src,h in b['sources'].items():
            if Path(src).is_relative_to(ame.ROOT):
                ame.require(ame.sha(p.parent/'source'/Path(src).relative_to(ame.ROOT))==h,'source archive changed')
        golden=ame.load(p.parent/'golden.json');op=b['row']['operator']
        ame.require(len(golden)==6 and [r['tile'] for r in golden]==list(TILES),'witness inventory')
        for r in golden:
            gemv=op=='gemv'
            ame.require((r['m'],r['n'],r['k'],r['lda'],r['ldb'],r['ldc'],r['accumulate'])==
                        (1,1,1,3,1 if gemv else 4,1 if gemv else 5,int(gemv)),'shape changed')
            a=[0x80000000 if gemv else 0x3f800000,0x7f812345,0x7f812345]
            rhs=[0x3f800000] if gemv else [0x7f800000]+[0x7f812345]*3
            init=[0x80000000] if gemv else [0x7f812345]*5
            output=[0x80000000] if gemv else [0x7f800000]+[0x7f812345]*4
            ame.require((r['a'],r['b'],r['initial'],r['golden'])==(a,rhs,init,output),'literal witness changed')
        gate=ame.load(p.parent/'operator-safety.json')
        ame.require(gate['elf_sha256']==b['elf_sha256'],'stale frame gate')
    ame.require(set(builds)=={r['key'] for r in rows},'missing build')
    records=[];groups={}
    for x in (32,64):
        stage=root/f'probe-rv{x}';plan=ame.load(stage/'plan.json')
        ame.require(plan['platform']['n_values']==[4,128],'wrong probe tiles')
        wanted={(r['key'],n) for r in rows if r['xlen']==x for n in (4,128)}
        seen=set()
        for p in sorted(stage.glob('shard-*/execute/*/n*/response.json')):
            d=p.parent;key=d.parent.name;n=int(d.name[1:]);b=builds[key]
            ame.require((key,n) in wanted and (key,n) not in seen,'wrong/duplicate execution')
            seen.add((key,n));q=ame.load(d/'request.json');r=ame.load(p)
            ame.require(q['expected']==dict(xlen=x,n=n,**b['row']['profile'],
                        csr_map=ame.CSR_MAP,entry_contract=ame.ENTRY),'wrong platform request')
            ame.require(q['revision']==plan['platform']['revision'] and q['backend']=='spike','wrong backend')
            exp=expected(b['row']['operator'],n,before)
            actual=validate_response(b,dict(expected={str(n):exp}),q,r,n)
            ame.require(all(c['exit_status']==0 for c in ame.load(d/'commands.json')),'adapter failed')
            if not before:
                result=ame.load(d/'operator-result.json')
                ame.require(result['status']=='OPERATOR_PASS' and result['exact_checks']==exp['checks'],
                            'missing passing count check')
            words=list(struct.unpack('<32I',bytes.fromhex(r['mailbox_hex'])));words[4]=0
            groups.setdefault((b['row']['operator'],n),[]).append(words)
            records.append(dict(key=key,n=n,actual=actual,elf_sha256=b['elf_sha256'],
                                request_sha256=ame.sha(d/'request.json'),response_sha256=ame.sha(p)))
        ame.require(seen==wanted,'missing probe execution')
        for d in stage.glob('shard-*'):
            labels={f'execute {key} N={n}':{'FAIL' if before else 'PASS'}
                    for key in (d/'keys.txt').read_text().splitlines() for n in (4,128)}
            ame.require(ame.check_result_labels((d/'ameops.sum').read_text(),labels),'wrong DejaGNU inventory')
    ame.require(len(groups)==4 and all(len(v)==2 and v[0]==v[1] for v in groups.values()),'cross-XLEN mismatch')
    out.mkdir(exist_ok=False,parents=True)
    ame.save(out/'audit.json',dict(status='EXPECTED_FAILURES_REPRODUCED' if before else 'SELECTED_PROBES_PASS',
        records=records,executions=len(records),cross_xlen_groups=len(groups),
        before_fix=before,audit_sha256=ame.sha(Path(__file__))))
    print('FP32_TAIL_PROBE_AUDIT_PASS',len(records),'before' if before else 'after')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--before',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.root.resolve(),a.before,a.out.resolve())
