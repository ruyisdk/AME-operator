#!/usr/bin/env python3
"""Dedicated negative-platform executor; normal ame.execute stays strict."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import struct
import subprocess
import sys
import ame
import operators
import spike_operator_guard_adapter as adapter
sys.path.insert(0,str(ame.ROOT/'operators/reference'))
import floating_exact as exact

HARDWARE={'u32-t4':(32,4),'u32-t128':(32,128),'u8-t4':(8,4),
          'u16-t8':(16,8),'u64-t4':(64,4),'u32-t2':(32,2),'u32-t256':(32,256)}
COMPILED=dict(uds=32,mregs=16,accregs=4)
FAMILIES=('i32','i8','u8','f32','bf16')

def accounting(b,g,hardware):
    uds,n=HARDWARE[hardware];supported=uds==32 and n in (4,128)
    op=b['row']['operator'];ame.require(g['operator']==op and op in ('gemm','gemv'),'operator mismatch')
    actions=('SET','ADD','K0_SET','K0_ADD','EMPTY','BAD_MODE','BAD_A','NO_WORK','BAD_WORK')
    inventory=[(phase,f,a) for phase in range(3) for f in range(5)
               for a in (actions if phase<2 else ('SET','ADD'))]
    ame.require([(r['phase'],r['family'],r['action']) for r in g['cases']]==inventory,'guard case inventory')
    count=16;computed=0;statuses=[]
    for r,(phase,f,action) in zip(g['cases'],inventory):
        width=(4,1,1,4,2)[f];mask=(1<<(8*width))-1;canary=0x7f812345
        if f<3:
            left,right=((-7,11),(-3,7),(200,3))[f]
            av,bv=left&mask,right&mask;seed=9;result=(left*right+(9 if action=='ADD' else 0))&0xffffffff
        else:
            form='bf16' if f==4 else 'fp32'
            av,bv=exact.bits(-4,3,form),exact.bits(12,3,form);seed=exact.bits(16,6)
            result=exact.bits(exact.dot([-4],[12],16 if action=='ADD' else 0,input_format=form),6)
        a=[canary&mask]*32;rhs=list(a);c=[canary]*32
        a[8]=av;rhs[8]=bv;c[8]=seed
        spots=[8+(q//2)*3+q%2 if op=='gemm' else 8+q for q in range(8)]
        if action in ('K0_SET','K0_ADD'):
            for at,value in zip(spots,(0x80000000,0x7f800001,0x7fc12345,0x7f800000,0xff800000,1,0x7f7fffff,0)):c[at]=value
        want=list(c)
        if phase!=1 and action in ('SET','ADD'):want[8]=result
        if action=='K0_SET':
            for at in spots:want[at]=0
        ame.require((r['width'],r['action_id'])==(width,actions.index(action)),'guard call type')
        ame.require((r['a'],r['b'],r['initial'],r['success'])==(a,rhs,c,want),'guard bit images')
        status=0
        if action not in ('EMPTY','K0_SET','K0_ADD'):
            if phase==1:status=3
            elif not supported:status=4
            elif action=='NO_WORK':status=2
            elif action=='BAD_WORK':status=1 if f==4 else 2
        if action in ('BAD_MODE','BAD_A'):status=1
        statuses.append(status)
        compute=status==0 and action in ('SET','ADD');computed+=compute
        count+=105+(96 if f==4 else 64)+(48 if f==4 else 16)
        if phase!=1:count+=3+int(not compute)
    ame.require(len(inventory)==100 and computed==(20 if supported else 0),'computed case count')
    ame.require(b['fixtures']['checks_by_support']['yes' if supported else 'no']==count,'fixture accounting')
    return dict(checks=count,logical_cases=len(inventory),computing_calls=computed,
                supported=supported,status_histogram={str(v):statuses.count(v) for v in sorted(set(statuses))})

def validate(response,request,b,expected):
    ame.require(response['schema']==1 and response['nonce']==request['nonce'],'stale response')
    for field in ('elf_sha256','backend','revision'):
        ame.require(response[field]==request[field],'wrong '+field)
    ame.require(response['observed']==request['expected'],'wrong observed hardware')
    ame.require(response['stop_pc']==b['symbols']['ame_test_done'],'incomplete execution')
    raw=bytes.fromhex(response['mailbox_hex']);ame.require(len(raw)==128,'mailbox size')
    w=list(struct.unpack('<32I',raw));r=b['row'];h=request['expected']
    ame.require(w[2]==2,f'guard guest failed status={w[2]} error={w[11]} index={w[12]:#x} expected={w[13]:#x} actual={w[14]:#x} pc={(w[20]<<32)|w[17]:#x}')
    wanted=[0x414d4554,1,2,r['case']['id'],r['xlen'],h['n'],h['uds'],16,4,b['config']['seed']]
    ame.require(w[:10]==wanted,'mailbox identity/configuration')
    ame.require(w[10]==expected['checks'],'wrong exact check count')
    ame.require(not any(w[11:15]) and not any(w[16:]) and w[15]&1,'error/trap/reserved/acquire state')
    return w

def audit_observations(directory,b,request,response,expected):
    trace=request['trace_driver'];log=directory/('spike.log.gz' if trace else 'spike.log')
    if trace:
        meta=ame.load(directory/'trace-storage.json')
        ame.require(ame.sha(log)==meta['gzip_sha256'] and meta['lossless'],'trace storage')
        raw=gzip.decompress(log.read_bytes())
        ame.require(len(raw)==meta['raw_bytes'] and hashlib.sha256(raw).hexdigest()==meta['raw_sha256'],'trace content')
        text=raw.decode()
    else:text=log.read_text()
    observed,stop,mailbox=adapter.parse(text,b['symbols'],ame.CSR_MAP,trace)
    ame.require(observed==response['observed'] and stop==response['stop_pc'] and mailbox.hex()==response['mailbox_hex'],'raw response differs')
    result=dict(trace=trace)
    if trace:
        pcs=[int(x,16) for x in re.findall(r'^core\s+0: 0x([0-9a-f]+) \(0x[0-9a-f]+\)',text,re.M)]
        symbols=b['symbols'];starts=[symbols['ame_ops_gemm_'+f+'_kernel'] for f in FAMILIES]
        ends=starts[1:]+[symbols['ame_gemv_i32_owned']]
        ame.require(pcs and pcs[0]==symbols['guard_observation_begin'],'missing bounded instruction trace')
        ame.require(symbols['guard_observation_begin']!=symbols['guard_observation_end'],'aliased markers')
        entries=[pcs.count(p) for p in starts]
        visits=[sum(lo<=p<hi for p in pcs) for lo,hi in zip(starts,ends)]
        wanted=4 if expected['supported'] else 0
        ame.require(entries==[wanted]*5,'wrong typed-kernel entry inventory')
        ame.require(all(v>0 for v in visits) if wanted else not any(visits),'unexpected typed-kernel body')
        result.update(kernel_entries=dict(zip(FAMILIES,entries)),kernel_pc_rows=dict(zip(FAMILIES,visits)),
                      instruction_rows=len(pcs),scope='observed kernel-entry PC records')
    return result

def run(artifact,hardware,out,model_path):
    out.mkdir(parents=True,exist_ok=False);b=ame.load(artifact/'build.json');r=b['row']
    ame.require(r['variant']=='platform-guards-v1' and r['profile']==COMPILED,'not a compiled guard image')
    ame.require(b['row']['definition']['harness']=='operators/tests/platform_guard_harness.c','wrong harness')
    ame.require(ame.sha(b['elf'])==b['elf_sha256'],'changed ELF')
    ame.require(all(ame.sha(p)==h for p,h in b['generated'].items()),'changed fixtures')
    gate=ame.load(artifact/'operator-safety.json')
    ame.require(gate['elf_sha256']==b['elf_sha256'] and gate['allowed_n']==[4,8,16,32,64,128],'missing normal frame safety')
    g=ame.load(artifact/'golden.json');expected=accounting(b,g,hardware);uds,n=HARDWARE[hardware]
    model=ame.load(model_path)
    request=dict(schema=1,nonce=secrets.token_hex(16),elf=b['elf'],elf_sha256=b['elf_sha256'],
        symbols=b['symbols'],mailbox_bytes=128,backend='spike',revision=model['revision'],
        guard_lane=r['variant'],compiled_profile=COMPILED,
        trace_driver=r['language']=='c' and r['opt']=='O2' and n!=128,
        expected=dict(xlen=r['xlen'],n=n,uds=uds,mregs=16,accregs=4,csr_map=ame.CSR_MAP,entry_contract=ame.ENTRY))
    ame.save(out/'request.json',request);records=[]
    source=ame.ROOT/'scripts/spike_operator_guard_adapter.py'
    ame.require(ame.sha(source)==b['sources'][str(source)],'adapter changed since build')
    ame.command([sys.executable,source,'--model',model_path,'--request',out/'request.json',
                 '--response',out/'response.json'],out,'adapter',b['config'],records)
    response=ame.load(out/'response.json');words=validate(response,request,b,expected)
    observations=audit_observations(out,b,request,response,expected)
    ame.save(out/'result.json',dict(status='GUARD_PASS',key=r['key'],hardware=hardware,
        elf_sha256=b['elf_sha256'],checks=expected,observations=observations,mailbox=words))
    print('GUARD_PASS',r['key'],hardware,expected['checks'],flush=True)

def check(config,artifacts,out,xlen,probe,model_path):
    c=ame.config(config);rows=[r for r in operators.plan(c) if r['xlen']==xlen]
    ame.require(len(rows)==8,'guard path inventory')
    hardware=['u32-t4','u8-t4'] if probe else list(HARDWARE)
    out.mkdir(parents=True,exist_ok=False)
    ame.save(out/'plan.json',dict(rows=rows,hardware=hardware,probe=probe,model=ame.load(model_path)))
    jobs=[]
    for shard in range(2):
        d=out/f'shard-{shard}';d.mkdir();keys=[r['key'] for r in rows[shard::2]]
        (d/'keys.txt').write_text('\n'.join(keys)+'\n')
        (d/'hardware.txt').write_text('\n'.join(hardware)+'\n')
        links=d/'inputs';links.mkdir()
        for key in keys:
            matches=list(artifacts.glob('shard-*/build/'+key));ame.require(len(matches)==1,'missing build')
            (links/key).symlink_to(matches[0].resolve(),target_is_directory=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',AME_ROOT=str(ame.ROOT),AME_PYTHON=sys.executable,
                 AME_OUTPUT=str(d),AME_KEYS=str(d/'keys.txt'),AME_HARDWARE=str(d/'hardware.txt'),
                 AME_ARTIFACTS=str(links),AME_OPERATOR_MODEL=str(model_path))
        log=(d/'runtest.log').open('xb')
        argv=[str(Path(c['toolchain_root'])/c['dejagnu']),'--tool','ameopguard','--srcdir',str(ame.ROOT/'testsuite'),'ameopguard.exp']
        jobs.append((subprocess.Popen(argv,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT),log,d,keys))
    reports=[]
    for proc,log,d,keys in jobs:
        rc=proc.wait();log.close();summary=d/'ameopguard.sum'
        labels={f'guard {key} {hw}':{'PASS'} for key in keys for hw in hardware}
        ok=rc==0 and ame.check_result_labels(summary.read_text(),labels)
        reports.append(dict(shard=str(d),ok=ok,exit_status=rc,labels=len(labels),summary_sha256=ame.sha(summary)))
    ok=all(r['ok'] for r in reports);ame.save(out/'summary.json',dict(ok=ok,stages=reports))
    print(json.dumps(dict(ok=ok,stages=reports)),flush=True);return 0 if ok else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run','check'))
    p.add_argument('--artifact',type=Path);p.add_argument('--artifacts',type=Path)
    p.add_argument('--hardware',choices=HARDWARE);p.add_argument('--xlen',type=int,choices=(32,64))
    p.add_argument('--config',type=Path);p.add_argument('--probe',action='store_true')
    p.add_argument('--model',type=Path,default=os.environ.get('AME_OPERATOR_MODEL'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:
        ame.require(a.model is not None,'--model is required')
        if a.action=='run':run(a.artifact.resolve(),a.hardware,a.out.resolve(),a.model.resolve());return 0
        return check(a.config.resolve(),a.artifacts.resolve(),a.out.resolve(),a.xlen,a.probe,a.model.resolve())
    except Exception as exc:print(str(exc),file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
