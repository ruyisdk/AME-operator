#!/usr/bin/env python3
"""Guard-only hardware matrix, with optional driver-bounded PC observations."""
import argparse
import gzip
from pathlib import Path
import re
import resource
import subprocess
import sys
import ame
import spike_fpstate_adapter as base

def commands(symbols,trace):
    lines=[f'until pc 0 {symbols["_start"]:x}','pc 0','priv 0','reg 0 misa',
           'reg 0 mstatus','reg 0 mie','ame 0']
    if trace:
        lines += [f'until pc 0 {symbols["guard_observation_begin"]:x}','pc 0',
                  f'untiln pc 0 {symbols["guard_observation_end"]:x}','pc 0']
    lines += [f'until pc 0 {symbols["ame_test_done"]:x}','pc 0','ame 0']
    lines += [f'mem {symbols["ame_mailbox"]+4*i:x}' for i in range(32)]
    return '\n'.join(lines+['quit',''])

def parse(text,symbols,csr_map,trace):
    if trace:
        lines=text.splitlines();at=[i for i,l in enumerate(lines) if re.fullmatch(r'0x[0-9a-fA-F]+',l)]
        ame.require(len(at)==39,'incomplete guard PC observations')
        ame.require([int(lines[at[i]],16) for i in (4,5)]==
                    [symbols['guard_observation_begin'],symbols['guard_observation_end']],
                    'guard trace boundaries not reached')
        text='\n'.join(l for i,l in enumerate(lines) if i not in (at[4],at[5]))
    return base.parse_debug(text,symbols,csr_map)

def run(a):
    request,model=ame.load(a.request),ame.load(a.model)
    d=a.response.resolve().parent;elf=Path(request['elf']);expected=request['expected'];s=request['symbols']
    ame.require(request['schema']==1 and request['mailbox_bytes']==128,'request schema')
    ame.require(request['guard_lane']=='platform-guards-v1','not a guard request')
    ame.require(request['backend']=='spike' and request['revision']==model['revision'],'model revision')
    ame.require(request['compiled_profile']==dict(uds=32,mregs=16,accregs=4),'compiled profile')
    ame.require((expected['uds'],expected['n']) in ((32,4),(32,128),(8,4),(16,8),(64,4),(32,2),(32,256)),
                'unreviewed hardware configuration')
    ame.require(expected['mregs']==16 and expected['accregs']==4,'register configuration')
    ame.require(expected['csr_map']==model['csr_map']==ame.CSR_MAP and expected['entry_contract']==ame.ENTRY,'entry contract')
    ame.require(ame.sha(model['spike'])==model['spike_sha256'],'model binary changed')
    ame.require(ame.sha(elf)==request['elf_sha256'],'ELF changed')
    xlen,entry,spans=base.elf_layout(elf)
    ame.require(xlen==expected['xlen'] and entry==s['_start']==0x80000000,'entry/XLEN')
    ame.require(all(0x80000000<=lo<=hi<=base.BOOT for lo,hi in spans),'ELF/boot overlap')
    ame.require(s['ame_mailbox']%4==0 and any(lo<=s['ame_mailbox'] and s['ame_mailbox']+128<=hi for lo,hi in spans),'mailbox')
    boot=model['boot'][str(xlen)];ame.require(ame.sha(boot['path'])==boot['sha256'],'bootstrap changed')
    trace=request['trace_driver'];ame.require(isinstance(trace,bool),'trace selection')
    ame.require(not trace or expected['n']!=128,'large positive trace not selected')
    cmd=d/'spike-debug.cmd';cmd.write_text(commands(s,trace))
    argv=[model['spike'],f'--isa=rv{xlen}imf_zicsr_ztt','--priv=M','-p1',
          '-m0x80000000:0x1000000',f'--pc={base.BOOT:#x}',f'--ame-nelem={expected["n"]**2}',
          f'--ame-udsz={expected["uds"]}','--ame-mregs=16','--ame-accregs=4',
          '--instructions=500000000','-d',f'--debug-cmd={cmd}',f'+payload={boot["path"]}',str(elf)]
    ame.save(d/'spike-command.json',dict(argv=argv,model=model,trace_driver=trace,
        trace_contract='silent startup; noisy begin-to-end driver only; silent completion'))
    # Bound trace disk use as well as the inherited address-space/time limits.
    resource.setrlimit(resource.RLIMIT_FSIZE,(256*1024**2,)*2)
    log=d/'spike.log'
    with log.open('w') as stream:
        subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,check=True,timeout=90)
    text=log.read_text();observed,stop,mailbox=parse(text,s,model['csr_map'],trace)
    ame.require(ame.sha(elf)==request['elf_sha256'] and ame.sha(model['spike'])==model['spike_sha256'],'executable changed')
    ame.save(a.response,dict(schema=1,nonce=request['nonce'],elf_sha256=request['elf_sha256'],
        backend='spike',revision=model['revision'],observed=observed,stop_pc=stop,mailbox_hex=mailbox.hex()))
    if trace:
        raw_sha=ame.sha(log);raw_size=log.stat().st_size
        with log.open('rb') as src,gzip.open(d/'spike.log.gz','wb',compresslevel=1) as dst:
            import shutil
            shutil.copyfileobj(src,dst)
        ame.save(d/'trace-storage.json',dict(raw_sha256=raw_sha,raw_bytes=raw_size,
            gzip_sha256=ame.sha(d/'spike.log.gz'),lossless=True))
        log.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--response',type=Path,required=True)
    try:run(p.parse_args())
    except Exception as exc:print(str(exc),file=sys.stderr);sys.exit(1)
