#!/usr/bin/env python3
"""Recheck exact guard configurations, source binding and real PC observations."""
import argparse
from collections import defaultdict
from pathlib import Path
import ame
import operator_platform_guards as guard

def audit(root,out):
    out.mkdir(parents=True,exist_ok=False);builds={};sources=None;tools=None
    for p in sorted((root/'build').glob('shard-*/build/*/build.json')):
        b=ame.load(p);key=b['row']['key'];ame.require(key not in builds,'duplicate build')
        ame.require(ame.sha(b['elf'])==b['elf_sha256'],'changed ELF')
        ame.require(b['row']['profile']==guard.COMPILED,'compiled profile changed')
        ame.require(all(ame.sha(q)==h for q,h in b['generated'].items()),'generated evidence changed')
        for q,h in b['sources'].items():
            if Path(q).is_relative_to(ame.ROOT):
                archive=p.parent/'source'/Path(q).relative_to(ame.ROOT)
                ame.require(ame.sha(archive)==h,'source archive changed')
        production={q:h for q,h in b['sources'].items() if any('/operators/'+x+'/' in q for x in ('include','common','kernels'))}
        if sources is None:sources=production;tools=b['tools']
        ame.require(production==sources and b['tools']==tools,'inconsistent production/tools')
        gate=ame.load(p.parent/'operator-safety.json')
        ame.require(gate['elf_sha256']==b['elf_sha256'],'frame evidence mismatch')
        builds[key]=(p.parent,b)
    ame.require(len(builds)==16 and set(builds)=={r['key'] for r in ame.load(root/'build/plan.json')['rows']},'build inventory')
    groups=defaultdict(list);reports={};trace_count=0
    for stage in ('probe-rv32','probe-rv64','main-rv32','main-rv64'):
        directory=root/stage;plan=ame.load(directory/'plan.json')
        hardware=['u32-t4','u8-t4'] if stage.startswith('probe') else list(guard.HARDWARE)
        ame.require(plan['hardware']==hardware,'incomplete hardware matrix')
        xlen=int(stage[-2:]);keys={k for k,(_,b) in builds.items() if b['row']['xlen']==xlen}
        ame.require({r['key'] for r in plan['rows']}==keys,'stage path inventory')
        wanted={(key,hw) for key in keys for hw in hardware};seen=set();records=[]
        for p in sorted(directory.glob('shard-*/execute/*/*/response.json')):
            key,hw=p.parent.parent.name,p.parent.name
            ame.require((key,hw) in wanted and (key,hw) not in seen,'unexpected/duplicate execution')
            seen.add((key,hw));artifact,b=builds[key];req=ame.load(p.parent/'request.json');resp=ame.load(p)
            uds,n=guard.HARDWARE[hw]
            ame.require(req['elf']==b['elf'] and req['elf_sha256']==b['elf_sha256'],'wrong execution ELF')
            ame.require(req['compiled_profile']==guard.COMPILED and req['guard_lane']=='platform-guards-v1','wrong guard contract')
            expected_hw=dict(xlen=xlen,n=n,uds=uds,mregs=16,accregs=4,csr_map=ame.CSR_MAP,entry_contract=ame.ENTRY)
            ame.require(req['expected']==expected_hw,'wrong hardware request')
            trace=b['row']['language']=='c' and b['row']['opt']=='O2' and n!=128
            ame.require(req['trace_driver']==trace,'missing trace selection')
            expected=guard.accounting(b,ame.load(artifact/'golden.json'),hw)
            words=guard.validate(resp,req,b,expected)
            observations=guard.audit_observations(p.parent,b,req,resp,expected)
            saved=ame.load(p.parent/'result.json')
            ame.require(saved==dict(status='GUARD_PASS',key=key,hardware=hw,elf_sha256=b['elf_sha256'],
                                   checks=expected,observations=observations,mailbox=words),'saved result differs')
            model=ame.load(p.parent/'spike-command.json')['model']
            approved=plan['model']
            ame.require(model==approved and req['revision']==approved['revision'] and req['backend']=='spike','unapproved model')
            normalized=list(words);normalized[4]=0
            if stage.startswith('main'):groups[(b['row']['operator'],hw)].append(normalized)
            trace_count+=trace
            records.append(dict(key=key,hardware=hw,checks=expected,response_sha256=ame.sha(p),observations=observations))
        ame.require(seen==wanted,'missing executions')
        for shard in directory.glob('shard-*'):
            labels={f'guard {key} {hw}':{'PASS'} for key in (shard/'keys.txt').read_text().splitlines() for hw in hardware}
            ame.require(ame.check_result_labels((shard/'ameopguard.sum').read_text(),labels),'DejaGNU label mismatch')
        reports[stage]=dict(executions=len(records),logical_cases=sum(r['checks']['logical_cases'] for r in records),records=records)
    ame.require(len(groups)==14 and all(len(v)==8 and all(w==v[0] for w in v) for v in groups.values()),'cross-path disagreement')
    ame.require(trace_count==32,'trace coverage changed')
    result=dict(status='GUARD_MATRIX_PASS',builds=16,stages=reports,
                cross_path_groups=14,traced_executions=trace_count,production_files=sources,
                source_sha256=ame.sha(Path(__file__)),guard_executor_sha256=ame.sha(Path(guard.__file__)))
    ame.save(out/'audit.json',result);print('GUARD_AUDIT_PASS 16 builds 112 main 32 probes 32 traces')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.root.resolve(),a.out.resolve())
