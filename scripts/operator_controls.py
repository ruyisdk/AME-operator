#!/usr/bin/env python3
"""Operator oracle controls: preserve raw numerical FAIL and verify it separately."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import ame
from audit_bank_capture import insns
from audit_operator_frames import audit as frame_audit
from resource_capture_faults import patch_word, validate_response

NS=(4,128)
WIDEN_FAULTS={'skip-i8-mmulacc':0,'skip-u8-mmulacc':1}

def kinds(variant):
    if variant=='i32-geometry-v2': return ('positive-copy','skip-mmulacc')
    if variant=='fp32-exact-v1': return ('positive-copy','skip-f32-mmulacc')
    if variant=='bf16-f32-exact-v1': return ('positive-copy','skip-bf16-mmulacc')
    ame.require(variant=='widen8-i32-v1','unreviewed control variant')
    return ('positive-copy',*WIDEN_FAULTS)

def validate_manifest(rows):
    variants={r['variant'] for r in rows}
    ame.require(len(variants)==1,'mixed/empty control variants')
    selected=kinds(next(iter(variants)))
    wanted={(op,x,k) for op in ('gemm','gemv') for x in (32,64) for k in selected}
    actual={(r['operator'],r['xlen'],r['kind']) for r in rows}
    ame.require(actual==wanted and len(rows)==len(wanted) and
                len({r['key'] for r in rows})==len(rows),'control inventory gap/duplicate')
    ame.require(all(r['expected_status']==(2 if r['kind']=='positive-copy' else 3)
                    for r in rows),'wrong control status')
    return selected

def mutation_sites(report,code,kind):
    variant=report['row']['variant'];s=report['symbols']
    ame.require(kind in kinds(variant),'wrong control kind')
    if variant=='i32-geometry-v2':
        regions={'skip-mmulacc':(s['ame_ops_gemm_i32_kernel'],s['ame_gemv_i32_owned'])}
    elif variant=='fp32-exact-v1':
        regions={'skip-f32-mmulacc':(s['ame_ops_gemm_f32_kernel'],s['ame_gemv_f32_owned'])}
    elif variant=='bf16-f32-exact-v1':
        regions={'skip-bf16-mmulacc':(s['ame_ops_gemm_bf16_kernel'],s['ame_gemv_bf16_f32_owned'])}
    else:
        regions={'skip-i8-mmulacc':(s['ame_ops_gemm_i8_kernel'],s['ame_ops_gemm_u8_kernel']),
                 'skip-u8-mmulacc':(s['ame_ops_gemm_u8_kernel'],s['ame_gemv_i8_i32_owned'])}
    all_sites={pc for pc,(_,op,_) in code.items() if op=='mmulacc.2d'}
    by_kind={k:{pc for pc in all_sites if start<=pc<end}
             for k,(start,end) in regions.items()}
    ame.require(all(1<=len(v)<=4 for v in by_kind.values()) and
                set().union(*by_kind.values())==all_sites,'unexpected multiplication sites')
    return set() if kind=='positive-copy' else by_kind[kind]


def prediction(kind, report, golden, n):
    if kind=='positive-copy':
        return dict(status=2,checks=report['fixtures']['checks_by_n'][str(n)],
                    error=0,index=0,expected=0,actual=0)
    ame.require(kind in ('skip-mmulacc','skip-f32-mmulacc','skip-bf16-mmulacc') or kind in WIDEN_FAULTS,
                'unknown operator control')
    widen=kind in WIDEN_FAULTS
    fp32=kind=='skip-f32-mmulacc'
    bf16=kind=='skip-bf16-mmulacc'
    maximum=report['config']['max_n']
    width=max(len(r['initial']) for r in golden)
    count=4 if bf16 else 3  # Two harness checks; one/two workspace queries.
    for v,r in enumerate(golden):
        if r['tile']!=n: continue
        ame.require(r['accumulate'] in (0,1), 'unsupported output initialization')
        affected=not widen or r['unsigned']==WIDEN_FAULTS[kind]
        count+=11 if bf16 else 4 if fp32 else 1  # State and BF16 descriptor checks.
        for i in range(width+32):
            j=i-16
            wanted=r['golden'][j] if 0<=j<len(r['initial']) else (0x7f812345 if fp32 or bf16 else 0x6b91d237)
            actual=wanted
            if affected and 0<=j<len(r['initial']) and j//r['ldc']<r['m'] and j%r['ldc']<r['n']:
                actual=r['initial'][j] if r['accumulate'] else 0
            count+=1
            if wanted!=actual:
                index=(0x83000000 if bf16 else 0x73000000 if fp32 else 0x52000000)+v*0x10000+i
                return dict(status=3,checks=count,error=1,index=index,
                            expected=wanted&0xffffffff,actual=actual&0xffffffff)
        count+=(5*maximum**2+96-(5*n*n if r['m'] and r['n'] and r['k'] else 0) if bf16
                else 3*maximum**2+32-3*n*n)
        count+=(sum(max(len(v[k]) for v in golden)+32 for k in ('a','b')) if fp32 or bf16
                else len(r['a'])+len(r['b']))
        if widen: count+=68  # Two 17-byte guards on each byte-input buffer.
    raise ValueError('fixture does not detect missing multiplication')


def checked_artifact(artifact):
    b=ame.load(artifact/'build.json'); spec=ame.load(artifact/'control.json')
    parent=ame.load(spec['parent'])
    ame.require(ame.sha(parent['elf'])==spec['parent_elf_sha256'], 'parent ELF changed')
    raw=Path(parent['elf']).read_bytes()
    for item in spec['mutations']:
        raw,off=patch_word(raw,item['pc'],item['old'],item['new'])
        ame.require(off==item['offset'], 'patch offset changed')
    ame.require(Path(b['elf']).read_bytes()==raw and ame.sha(b['elf'])==b['elf_sha256'],
                'derived ELF changed')
    golden_path=Path(parent['elf']).parent/'golden.json'
    ame.require(ame.sha(golden_path)==spec['golden_sha256'], 'parent GOLDEN changed')
    golden=ame.load(golden_path)
    ame.require(spec['expected']=={str(n):prediction(spec['kind'],parent,golden,n) for n in NS},
                'control prediction changed')
    gate=ame.load(artifact/'operator-safety.json')
    ame.require(gate['elf_sha256']==b['elf_sha256'] and set(NS)<=set(gate['allowed_n']),
                'control frame review missing')
    return b,spec


def prepare(base,out):
    checkpoint=ame.load(base/'CHECKPOINT.json')
    ame.require(all(ame.sha(base/p)==h for p,h in checkpoint['artifacts'].items()),
                'frozen parent evidence changed')
    evidence=ame.load(base/'audit/audit.json')
    ame.require(evidence['builds']==16 and evidence['main_cross_path_groups']==12,
                'reviewed geometry matrix required')
    parents=[]
    for p in sorted((base/'build').glob('shard-*/build/*/build.json')):
        b=ame.load(p)
        if b['row']['opt']=='O2' and b['row']['language']=='c': parents.append((p,b))
    ame.require(len(parents)==4, 'need two operators and two XLEN parents')
    out.mkdir(parents=True,exist_ok=False)
    for xlen in (32,64):
        platform=ame.load(base/f'main-rv{xlen}/plan.json')['platform']
        platform['n_values']=list(NS)
        ame.save(out/f'platform-rv{xlen}.json',platform)
    rows=[]
    for path,b in parents:
        code=insns((path.parent/'objdump.log').read_text())
        original=Path(b['elf']).read_bytes()
        golden=ame.load(path.parent/'golden.json')
        for kind in kinds(b['row']['variant']):
            sites=mutation_sites(b,code,kind)
            key=b['row']['key']+'--'+kind
            d=out/'artifacts'/key; d.mkdir(parents=True)
            blob=original; changes=[]
            for pc in sorted(sites):
                old=code[pc][0]; blob,off=patch_word(blob,pc,old,0x13)
                changes.append(dict(pc=pc,offset=off,old=old,new=0x13))
            elf=d/'test.elf'; elf.write_bytes(blob)
            report=copy.deepcopy(b)
            report.update(status='DERIVED_CONTROL_ELF',elf=str(elf),elf_sha256=ame.sha(elf))
            spec=dict(key=key,kind=kind,parent=str(path),parent_elf_sha256=b['elf_sha256'],
                mutations=changes,golden_sha256=ame.sha(path.parent/'golden.json'),
                expected={str(n):prediction(kind,b,golden,n) for n in NS},paired_positive={})
            for n in NS:
                normal=list((base/f'main-rv{b["row"]["xlen"]}').glob(
                    f'shard-*/execute/{b["row"]["key"]}/n{n}/operator-result.json'))
                ame.require(len(normal)==1, 'missing original positive execution')
                ame.require(ame.load(normal[0])['status']=='OPERATOR_PASS', 'parent did not pass')
                spec['paired_positive'][str(n)]=dict(path=str(normal[0]),sha256=ame.sha(normal[0]))
            records=[]
            dump=ame.command([ame.tool_paths(b['config'])['objdump'],'-d',elf],d,'objdump',b['config'],records)
            decoded=insns(dump)
            ame.require(set(decoded)==set(code) and
                {p for p in code if code[p]!=decoded[p]}=={m['pc'] for m in changes},
                'unintended decoded instruction change')
            ame.save(d/'build.json',report)
            frame_audit(d)
            ame.save(d/'control.json',spec)
            checked_artifact(d)
            rows.append(dict(key=key,artifact=str(d),xlen=b['row']['xlen'],kind=kind,
                             operator=b['row']['operator'],variant=b['row']['variant'],
                             expected_status=2 if kind=='positive-copy' else 3))
    validate_manifest(rows)
    scripts=('scripts/operator_controls.py','scripts/resource_capture_faults.py',
             'scripts/audit_bank_capture.py','scripts/audit_operator_frames.py',
             'scripts/ame.py','testsuite/ameopcontrol/ameopcontrol.exp')
    inputs={str(ame.ROOT/p):ame.sha(ame.ROOT/p) for p in scripts}
    for p,h in list(inputs.items()):
        target=out/'source'/Path(p).relative_to(ame.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(Path(p).read_bytes())
    ame.save(out/'manifest.json',rows)
    ame.save(out/'preflight.json',dict(inputs=inputs,model=checkpoint['model'],
        base_checkpoint=str(base/'CHECKPOINT.json'),base_checkpoint_sha256=ame.sha(base/'CHECKPOINT.json'),
        maximum_workers=2,n_values=list(NS)))
    print('OPERATOR_CONTROLS_PREPARED',len(rows))


def run(artifact,platform,n,out):
    b,spec=checked_artifact(artifact)
    try:
        return ame.execute(b,ame.load(platform),n,out)
    except ValueError as exc:
        if out.is_dir(): ame.save(out/'raw-failure.json',dict(reason=str(exc)))
        print(str(exc),file=sys.stderr)
        return 1


def verify(artifact,n,out):
    b,spec=checked_artifact(artifact)
    result=validate_response(b,spec,ame.load(out/'request.json'),ame.load(out/'response.json'),n)
    ame.require(all(c['exit_status']==0 for c in ame.load(out/'commands.json')),
                'adapter command failed')
    normal=spec['paired_positive'][str(n)]
    ame.require(ame.sha(normal['path'])==normal['sha256'], 'original positive evidence changed')
    ame.save(out/'verification.json',dict(expected_outcome_matched=True,actual=result,
        raw_status='FAIL' if result['status']==3 else 'PASS',kind=spec['kind']))
    print('EXPECTED_FAULT_DETECTED' if result['status']==3 else 'POSITIVE_COPY_PASS')


def check(root,xlen):
    pre=ame.load(root/'preflight.json')
    ame.require(all(ame.sha(p)==h for p,h in pre['inputs'].items()), 'control source changed')
    ame.require(ame.sha(pre['model']['spike'])==pre['model']['spike_sha256'], 'model changed')
    manifest=ame.load(root/'manifest.json')
    selected_kinds=validate_manifest(manifest)
    rows=[r for r in manifest if r['xlen']==xlen]
    ame.require(len(rows)==2*len(selected_kinds), 'control XLEN inventory')
    d=root/f'rv{xlen}'; d.mkdir(exist_ok=False)
    platform=root/f'platform-rv{xlen}.json'
    config=ame.load(Path(rows[0]['artifact'])/'build.json')['config']
    ame.save(d/'plan.json',dict(rows=rows,n_values=list(NS),platform=ame.load(platform)))
    jobs=[]
    for shard in range(2):
        work=d/f'shard-{shard}'; work.mkdir()
        selected=rows[shard::2]
        (work/'keys.txt').write_text('\n'.join(r['key'] for r in selected)+'\n')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',AME_ROOT=str(ame.ROOT),
            AME_PYTHON=sys.executable,AME_OUTPUT=str(work),AME_KEYS=str(work/'keys.txt'),
            AME_ARTIFACTS=str(root/'artifacts'),AME_PLATFORM=str(platform))
        cmd=[str(Path(config['toolchain_root'])/config['dejagnu']),'--tool','ameopcontrol',
             '--srcdir',str(ame.ROOT/'testsuite'),'ameopcontrol.exp']
        log=(work/'runtest.log').open('xb')
        jobs.append((subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT),
                     log,work,selected))
    reports=[]
    for proc,log,work,selected in jobs:
        rc=proc.wait(); log.close()
        labels={}
        for row in selected:
            for n in NS:
                labels[f'execute {row["key"]} N={n}']={'PASS' if row['expected_status']==2 else 'FAIL'}
                labels[f'verify {row["key"]} N={n}']={'PASS'}
        expected_rc=int(any(r['expected_status']==3 for r in selected))
        good=rc==expected_rc and ame.check_result_labels((work/'ameopcontrol.sum').read_text(),labels)
        reports.append(dict(shard=str(work),raw_dejagnu_exit_status=rc,expected_outcomes_matched=good))
    ok=all(r['expected_outcomes_matched'] for r in reports)
    ame.require(all(ame.sha(p)==h for p,h in pre['inputs'].items()), 'control source changed during run')
    ame.save(d/'summary.json',dict(expected_outcomes_matched=ok,shards=reports,
        deliberate_failures=sum(r['expected_status']==3 for r in rows)*len(NS),
        positive_executions=sum(r['expected_status']==2 for r in rows)*len(NS),
        verification_checks=len(rows)*len(NS),failures_relabelled=False))
    print('OPERATOR_CONTROL_STAGE',xlen,ok)
    return 0 if ok else 1


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','run','verify','check'])
    p.add_argument('--base',type=Path); p.add_argument('--out',type=Path)
    p.add_argument('--root',type=Path); p.add_argument('--artifact',type=Path)
    p.add_argument('--platform',type=Path); p.add_argument('--n',type=int,choices=NS)
    p.add_argument('--xlen',type=int,choices=[32,64])
    a=p.parse_args()
    if a.action=='prepare': prepare(a.base.resolve(),a.out.resolve()); return 0
    if a.action=='check': return check(a.root.resolve(),a.xlen)
    if a.action=='run': return run(a.artifact.resolve(),a.platform,a.n,a.out.resolve())
    verify(a.artifact.resolve(),a.n,a.out.resolve()); return 0

if __name__=='__main__': sys.exit(main())
