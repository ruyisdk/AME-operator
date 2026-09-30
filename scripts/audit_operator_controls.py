#!/usr/bin/env python3
"""Audit derived operator ELF and exact positive/negative execution inventories."""
import argparse
from collections import defaultdict
from pathlib import Path
import struct
import ame
import audit_operator_bf16
from audit_bank_capture import insns
from operator_controls import checked_artifact, validate_manifest, NS
from resource_capture_faults import validate_response

def audit(root,out):
    pre=ame.load(root/'preflight.json')
    ame.require(all(ame.sha(p)==h for p,h in pre['inputs'].items()), 'control inputs changed')
    ame.require(ame.sha(pre['model']['spike'])==pre['model']['spike_sha256'], 'model changed')
    ame.require(ame.sha(pre['base_checkpoint'])==pre['base_checkpoint_sha256'], 'base checkpoint changed')
    manifest=ame.load(root/'manifest.json')
    control_kinds=validate_manifest(manifest)
    rows={r['key']:r for r in manifest}; groups=defaultdict(list); records=[]
    for xlen in (32,64):
        stage=root/f'rv{xlen}'
        ame.require(ame.load(stage/'summary.json')['expected_outcomes_matched'], 'stage failed')
        expected={(r['key'],n) for r in manifest if r['xlen']==xlen for n in NS}
        seen=set()
        for p in sorted(stage.glob('shard-*/execute/*/n*/response.json')):
            directory=p.parent; key=directory.parent.name; n=int(directory.name[1:])
            ame.require((key,n) in expected and (key,n) not in seen, 'unexpected/duplicate execution')
            seen.add((key,n)); row=rows[key]
            b,spec=checked_artifact(Path(row['artifact']))
            parent=ame.load(spec['parent'])
            code=insns((Path(spec['parent']).parent/'objdump.log').read_text())
            sites={pc for pc,(_,op,_) in code.items() if op=='mmulacc.2d'}
            ame.require(spec['kind']==row['kind'] and b['row']['opt']=='O2' and
                        b['row']['language']=='c' and b['row']['xlen']==xlen and
                        b['row']['operator']==row['operator'] and
                        b['row']['variant']==row['variant'], 'wrong parent path')
            if row['kind']=='positive-copy':
                ame.require(not spec['mutations'] and b['elf_sha256']==parent['elf_sha256'],
                            'positive copy is not identical')
            else:
                if row['variant']=='widen8-i32-v1':
                    s=parent['symbols']
                    start,end=((s['ame_ops_gemm_i8_kernel'],s['ame_ops_gemm_u8_kernel'])
                               if row['kind']=='skip-i8-mmulacc' else
                               (s['ame_ops_gemm_u8_kernel'],s['ame_gemv_i8_i32_owned']))
                    sites={p for p in sites if start<=p<end}
                ame.require(sites and
                            {m['pc'] for m in spec['mutations']}==sites and
                            all(m['new']==0x13 and m['old']==code[m['pc']][0] for m in spec['mutations']),
                            'wrong negative mutation')
            q,r=ame.load(directory/'request.json'),ame.load(p)
            if row['variant']=='bf16-f32-exact-v1':
                audit_operator_bf16.verify_execution(b,directory,r)
            ame.require(q['expected']==dict(xlen=xlen,n=n,**b['row']['profile'],
                csr_map=ame.CSR_MAP,entry_contract=ame.ENTRY), 'wrong configuration request')
            ame.require(q['backend']=='spike' and q['revision']==pre['model']['revision'], 'wrong model request')
            actual=validate_response(b,spec,q,r,n)
            verify=ame.load(directory/'verification.json')
            ame.require(verify['expected_outcome_matched'] and verify['actual']==actual and
                        actual['status']==row['expected_status'], 'verification differs')
            ame.require(all(c['exit_status']==0 for c in ame.load(directory/'commands.json')),
                        'adapter error')
            if actual['status']==3:
                ame.require((directory/'raw-failure.json').is_file(), 'lost raw failure')
            else:
                ame.require(ame.load(directory/'result.json')['status']=='EXEC_PASS', 'missing positive result')
            words=list(struct.unpack('<32I',bytes.fromhex(r['mailbox_hex']))); words[4]=0
            groups[(b['row']['operator'],row['kind'],n)].append(words)
            records.append(dict(key=key,n=n,actual=actual,response_sha256=ame.sha(p),
                request_sha256=ame.sha(directory/'request.json'),elf_sha256=b['elf_sha256']))
        ame.require(seen==expected, 'missing control execution')
        for d in stage.glob('shard-*'):
            labels={}
            for key in (d/'keys.txt').read_text().splitlines():
                for n in NS:
                    labels[f'execute {key} N={n}']={'PASS' if rows[key]['expected_status']==2 else 'FAIL'}
                    labels[f'verify {key} N={n}']={'PASS'}
            ame.require(ame.check_result_labels((d/'ameopcontrol.sum').read_text(),labels),
                        'control DejaGNU labels changed')
    ame.require(len(groups)==4*len(control_kinds) and all(len(v)==2 and v[0]==v[1] for v in groups.values()),
                'cross-XLEN control comparison')
    out.mkdir(parents=True,exist_ok=False)
    ame.save(out/'audit.json',dict(expected_outcomes_matched=True,records=records,
        executions=len(records),raw_failures=sum(r['actual']['status']==3 for r in records),
        positive_passes=sum(r['actual']['status']==2 for r in records),
        cross_xlen_groups=len(groups),audit_sha256=ame.sha(Path(__file__)),
        scope='O2/C variant-specific missing-multiply and identical-copy controls; no model modifications'))
    print('OPERATOR_CONTROL_AUDIT_PASS',len(records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.root.resolve(),a.out.resolve())
