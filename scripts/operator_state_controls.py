#!/usr/bin/env python3
"""Isolated FP32 operator faults; preserve target FAIL separately from detection."""
import argparse
import copy
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import ame
from audit_operator_frames import audit as frame_audit
from resource_capture_faults import validate_response
sys.path.insert(0, str(ame.ROOT / 'operators/reference'))
import fp32_fold

NS = (4, 128)
KINDS = ('positive-relink', 'clear-amefflags', 'clear-amexsat',
         'clear-fcsr', 'zero-acc-before-multiply')
FUNCTION = 'ame_ops_gemm_f32_kernel'


def transform(source, kind):
    ame.require(kind in KINDS, 'unknown state control')
    lines = source.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip() == FUNCTION + ':']
    ends = [i for i, line in enumerate(lines)
            if re.fullmatch(r'\s*\.size\s+' + FUNCTION + r',\s*\.-' + FUNCTION + r'\s*', line)]
    ame.require(len(starts) == len(ends) == 1 and starts[0] < ends[0], 'kernel boundary changed')
    sites = []
    output = []
    for i, line in enumerate(lines):
        if starts[0] < i < ends[0]:
            if kind.startswith('clear-') and line.strip() in ('jr\tra', 'ret'):
                csr = kind.removeprefix('clear-')
                output.append(f'\tcsrw\t{csr},zero\n')
                sites.append(dict(line=i + 1, insertion=output[-1].strip()))
            if kind == 'zero-acc-before-multiply':
                match = re.fullmatch(r'\s*mmulacc\.2d\s+(acc[0-9]+),m[0-9]+,m[0-9]+\s*', line)
                if match:
                    output.append(f'\tmzero.2d.acc\t{match[1]}\n')
                    sites.append(dict(line=i + 1, insertion=output[-1].strip()))
        output.append(line)
    ame.require(len(sites) == (0 if kind == 'positive-relink' else 1), 'unexpected fault site count')
    return ''.join(output), sites


def zero_acc_result(row, n):
    # Logical arithmetic only. Padding uses exact neutral operands in this lane.
    result = list(row['initial'])
    flags = 0
    for i in range(row['m']):
        for j in range(row['n']):
            value = row['initial'][i * row['ldc'] + j] if row['accumulate'] else 0
            for q in range(row['k']):
                if q % n == 0:
                    value = 0
                value, added = fp32_fold.fma(row['a'][i * row['lda'] + q],
                    row['b'][q * row['ldb'] + j], value)
                flags |= added
            result[i * row['ldc'] + j] = value
    return result, flags


def prediction(kind, report, rows, n):
    ame.require(kind in KINDS and n in NS, 'unreviewed prediction')
    if kind == 'positive-relink':
        return dict(status=2, checks=report['fixtures']['checks_by_n'][str(n)],
                    error=0, index=0, expected=0, actual=0)
    widths = {k: max(len(row[k]) for row in rows) for k in ('a', 'b', 'initial')}
    count = 4  # Harness acquire checks and two driver setup checks.
    for v, row in enumerate(rows):
        if row['tile'] != n:
            continue
        actual_data, flags = ((zero_acc_result(row, n)) if kind == 'zero-acc-before-multiply'
                              else (row['golden'], row['new_flags']))
        wanted = [0, 0, row['seed_flags'] | row['new_flags'], row['seed_xsat'], row['fcsr'], 1]
        actual = [0, 0, row['seed_flags'] | flags, row['seed_xsat'], row['fcsr'], 1]
        for name, pos in (('clear-amefflags', 2), ('clear-amexsat', 3), ('clear-fcsr', 4)):
            if kind == name:
                actual[pos] = 0
        for i, (expect, value) in enumerate(zip(wanted, actual)):
            count += 1
            if expect != value:
                return dict(status=3, checks=count, error=1, index=0x79000000 + i * 0x10000 + v,
                            expected=expect, actual=value)
        for i in range(widths['initial'] + 32):
            j = i - 16
            expect = row['golden'][j] if 0 <= j < len(row['initial']) else 0x7f812345
            value = actual_data[j] if 0 <= j < len(row['initial']) else 0x7f812345
            count += 1
            if expect != value:
                return dict(status=3, checks=count, error=1, index=0x7a000000 + v * 0x10000 + i,
                            expected=expect, actual=value)
        count += widths['a'] + widths['b'] + 64 + 3 * report['config']['max_n']**2 + 32 - 3 * n*n
    raise ValueError('control not detected by unchanged checker')


def manifest_rows(rows):
    expected = {(op, x, k) for op in ('gemm', 'gemv') for x in (32, 64) for k in KINDS}
    ame.require(len(rows) == len(expected) and
                {(r['operator'], r['xlen'], r['kind']) for r in rows} == expected and
                len({r['key'] for r in rows}) == len(rows), 'control inventory mismatch')
    ame.require(all(r['expected_status'] == (2 if r['kind'] == 'positive-relink' else 3)
                    for r in rows), 'wrong expected status')


def link(directory, parent, config, records):
    tools = ame.tool_paths(config)
    xlen = parent['row']['xlen']
    ame.command([tools['assembler'], f'-march=rv{xlen}im_zicsr_ztt0p6',
                 '-mabi=' + ('ilp32' if xlen == 32 else 'lp64'), '-o', directory/'4.o', directory/'4.s'],
                directory, 'assemble', config, records)
    ame.command([tools['linker'], '-m', f'elf{xlen}lriscv', '--no-relax', '-T', directory/'link.ld',
                 '-Map=' + str(directory/'test.map'), '-o', directory/'test.elf'] +
                 [directory/f'{i}.o' for i in range(7)], directory, 'link', config, records)


def checked_artifact(directory):
    b, spec = ame.load(directory/'build.json'), ame.load(directory/'control.json')
    parent = ame.load(spec['parent'])
    ame.require(all(ame.sha(p) == h for p, h in spec['parent_inputs'].items()), 'parent input changed')
    ame.require(all(ame.sha(directory/p) == h for p, h in spec['derived_inputs'].items()), 'derived input changed')
    text, sites = transform(Path(spec['parent']).with_name('4.s').read_text(), spec['kind'])
    ame.require(text == (directory/'4.s').read_text() and sites == spec['sites'], 'unapproved source mutation')
    ame.require(all((directory/f'{i}.o').read_bytes() == Path(spec['parent']).with_name(f'{i}.o').read_bytes()
                    for i in range(7) if i != 4), 'checker or non-kernel object modified')
    golden = ame.load(directory/'golden.json')
    ame.require(spec['expected'] == {str(n): prediction(spec['kind'], parent, golden, n) for n in NS},
                'prediction changed')
    ame.require(ame.sha(b['elf']) == b['elf_sha256'], 'ELF changed')
    if spec['kind'] == 'positive-relink':
        ame.require(Path(b['elf']).read_bytes() == Path(parent['elf']).read_bytes(), 'positive ELF not identical')
    gate = frame_audit(directory, write=False)
    ame.require(gate == ame.load(directory/'operator-safety.json'), 'frame review changed')
    for item in spec['paired_positive'].values():
        ame.require(ame.sha(item['path']) == item['sha256'] and
                    ame.load(item['path'])['status'] == 'OPERATOR_PASS', 'parent positive missing/changed')
    return b, spec


def prepare(base, out):
    checkpoint = ame.load(base/'CHECKPOINT.json')
    ame.require(all(ame.sha(base/p) == h for p, h in checkpoint['artifacts'].items()), 'parent checkpoint changed')
    parents = []
    for path in sorted((base/'build').glob('shard-*/build/*/build.json')):
        b = ame.load(path)
        if b['row']['opt'] == 'O2' and b['row']['language'] == 'c':
            ame.require(b['row']['variant'] == 'fp32-rne-fold-v1', 'wrong parent variant')
            parents.append((path, b))
    ame.require(len(parents) == 4, 'need four parent paths')
    tool_hashes = parents[0][1]['tools']
    ame.require(all(b['tools'] == tool_hashes for _, b in parents), 'mixed parent tools')
    ame.require(all(ame.sha(p) == h for p, h in tool_hashes.items()), 'parent toolchain changed')
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for path, b in parents:
        source = path.with_name('4.s').read_text()
        ame.require(b['row']['definition']['sources'][2] == 'operators/kernels/gemm/f32.c', 'object index changed')
        link_script = path.parent/'source/runtime/link.ld'
        inputs = [path, path.with_name('4.s'), path.with_name('golden.json'), Path(b['elf']), link_script]
        inputs += [path.with_name(f'{i}.o') for i in range(7)]
        for kind in KINDS:
            key = b['row']['key'] + '--' + kind
            d = out/'artifacts'/key
            d.mkdir(parents=True)
            text, sites = transform(source, kind)
            (d/'4.s').write_text(text)
            (d/'link.ld').write_bytes(link_script.read_bytes())
            (d/'golden.json').write_bytes(path.with_name('golden.json').read_bytes())
            for i in range(7):
                if i != 4:
                    (d/f'{i}.o').write_bytes(path.with_name(f'{i}.o').read_bytes())
            records = []
            link(d, b, b['config'], records)
            tools = ame.tool_paths(b['config'])
            ame.command([tools['objdump'], '-d', d/'test.elf'], d, 'objdump', b['config'], records)
            names = ame.command([tools['nm'], '-n', d/'test.elf'], d, 'symbols', b['config'], records)
            symbols = {line.split()[2]: int(line.split()[0], 16) for line in names.splitlines() if len(line.split()) == 3}
            report = copy.deepcopy(b)
            report.update(status='DERIVED_CONTROL_ELF', elf=str(d/'test.elf'),
                          elf_sha256=ame.sha(d/'test.elf'), symbols=symbols)
            ame.save(d/'build.json', report)
            frame_audit(d)
            positive = {}
            for n in NS:
                matches = list((base/f'main-rv{b["row"]["xlen"]}').glob(
                    f'shard-*/execute/{b["row"]["key"]}/n{n}/operator-result.json'))
                ame.require(len(matches) == 1, 'missing paired positive')
                positive[str(n)] = dict(path=str(matches[0]), sha256=ame.sha(matches[0]))
            derived = ['4.s', '4.o', 'link.ld', 'golden.json', 'test.elf', 'objdump.log', 'symbols.log']
            derived += [f'{i}.o' for i in range(7) if i != 4]
            ame.save(d/'control.json', dict(kind=kind, parent=str(path), sites=sites,
                parent_inputs={str(p): ame.sha(p) for p in inputs},
                derived_inputs={p: ame.sha(d/p) for p in derived}, paired_positive=positive,
                expected={str(n): prediction(kind, b, ame.load(d/'golden.json'), n) for n in NS}))
            checked_artifact(d)
            rows.append(dict(key=key, artifact=str(d), operator=b['row']['operator'],
                xlen=b['row']['xlen'], kind=kind, expected_status=2 if kind == 'positive-relink' else 3))
    manifest_rows(rows)
    sources = ('scripts/operator_state_controls.py', 'scripts/resource_capture_faults.py',
        'scripts/ame.py', 'scripts/audit_operator_frames.py', 'scripts/audit_bank_capture.py',
        'scripts/spike_fpstate_adapter.py', 'scripts/spike_adapter.py',
        'scripts/ieee_tininess.py', 'scripts/test_fused32.py', 'scripts/test_exponent32.py',
        'scripts/test_sqrt32.py',
        'operators/reference/fp32_fold.py', 'testsuite/ameopstatecontrol/ameopstatecontrol.exp')
    inputs = {str(ame.ROOT/p): ame.sha(ame.ROOT/p) for p in sources}
    for p in sources:
        dest = out/'source'/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ame.ROOT/p).read_bytes())
    ame.save(out/'manifest.json', rows)
    for x in (32, 64):
        platform = ame.load(base/f'main-rv{x}/plan.json')['platform']
        platform['n_values'] = list(NS)
        ame.save(out/f'platform-rv{x}.json', platform)
        inputs[str(out/f'platform-rv{x}.json')] = ame.sha(out/f'platform-rv{x}.json')
    ame.save(out/'preflight.json', dict(inputs=inputs, model=checkpoint['model'],
        tools=tool_hashes,
        base_checkpoint=str(base/'CHECKPOINT.json'), base_checkpoint_sha256=ame.sha(base/'CHECKPOINT.json'),
        maximum_workers=2, n_values=list(NS)))
    print('STATE_CONTROLS_PREPARED', len(rows), flush=True)


def run(artifact, platform, n, out):
    b, spec = checked_artifact(artifact)
    try:
        return ame.execute(b, ame.load(platform), n, out)
    except ValueError as exc:
        if out.is_dir():
            ame.save(out/'raw-failure.json', dict(reason=str(exc)))
        print(str(exc), file=sys.stderr)
        return 1


def verify(artifact, n, out):
    b, spec = checked_artifact(artifact)
    result = validate_response(b, spec, ame.load(out/'request.json'), ame.load(out/'response.json'), n)
    ame.require(all(c['exit_status'] == 0 for c in ame.load(out/'commands.json')), 'adapter failed')
    ame.save(out/'verification.json', dict(expected_outcome_matched=True, actual=result,
        raw_status='FAIL' if result['status'] == 3 else 'PASS', kind=spec['kind']))
    print('EXPECTED_FAULT_DETECTED' if result['status'] == 3 else 'POSITIVE_RELINK_PASS')


def preflight(root):
    pre = ame.load(root/'preflight.json')
    ame.require(all(ame.sha(p) == h for p, h in pre['inputs'].items()), 'control input changed')
    ame.require(ame.sha(pre['model']['spike']) == pre['model']['spike_sha256'], 'model changed')
    ame.require(all(ame.sha(p) == h for p, h in pre['tools'].items()), 'toolchain changed')
    ame.require(ame.sha(pre['base_checkpoint']) == pre['base_checkpoint_sha256'], 'base checkpoint changed')
    rows = ame.load(root/'manifest.json')
    manifest_rows(rows)
    return pre, rows


def check(root, xlen):
    pre, manifest = preflight(root)
    rows = [r for r in manifest if r['xlen'] == xlen]
    d = root/f'rv{xlen}'
    d.mkdir(exist_ok=False)
    platform = root/f'platform-rv{xlen}.json'
    config = ame.load(Path(rows[0]['artifact'])/'build.json')['config']
    jobs = []
    for shard in range(2):
        work = d/f'shard-{shard}'
        work.mkdir()
        selected = rows[shard::2]
        (work/'keys.txt').write_text('\n'.join(r['key'] for r in selected) + '\n')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', AME_ROOT=str(ame.ROOT),
            AME_PYTHON=sys.executable, AME_OUTPUT=str(work), AME_KEYS=str(work/'keys.txt'),
            AME_ARTIFACTS=str(root/'artifacts'), AME_PLATFORM=str(platform))
        cmd = [str(Path(config['toolchain_root'])/config['dejagnu']), '--tool', 'ameopstatecontrol',
               '--srcdir', str(ame.ROOT/'testsuite'), 'ameopstatecontrol.exp']
        log = (work/'runtest.log').open('xb')
        jobs.append((subprocess.Popen(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT), log, work, selected))
    reports = []
    for proc, log, work, selected in jobs:
        rc = proc.wait()
        log.close()
        labels = {}
        for r in selected:
            for n in NS:
                labels[f'execute {r["key"]} N={n}'] = {'PASS' if r['expected_status'] == 2 else 'FAIL'}
                labels[f'verify {r["key"]} N={n}'] = {'PASS'}
        good = rc == 1 and ame.check_result_labels((work/'ameopstatecontrol.sum').read_text(), labels)
        reports.append(dict(shard=str(work), raw_dejagnu_exit_status=rc, expected_outcomes_matched=good))
    preflight(root)
    ok = all(r['expected_outcomes_matched'] for r in reports)
    ame.save(d/'summary.json', dict(expected_outcomes_matched=ok, shards=reports))
    print('STATE_CONTROL_STAGE', xlen, ok, flush=True)
    return 0 if ok else 1


def audit(root, out):
    pre, rows = preflight(root)
    out.mkdir(parents=True, exist_ok=False)
    by_key = {r['key']: r for r in rows}
    records, groups = [], {}
    for row in rows:
        d = Path(row['artifact'])
        b, spec = checked_artifact(d)
        # Re-assemble and re-link from the reviewed transform and untouched objects.
        proof = out/'rebuild'/row['key']
        proof.mkdir(parents=True)
        for name in ['4.s', 'link.ld'] + [f'{i}.o' for i in range(7) if i != 4]:
            (proof/name).write_bytes((d/name).read_bytes())
        link(proof, b, b['config'], [])
        ame.require((proof/'test.elf').read_bytes() == Path(b['elf']).read_bytes(), 'derived ELF not reproducible')
    for xlen in (32, 64):
        stage = root/f'rv{xlen}'
        ame.require(ame.load(stage/'summary.json')['expected_outcomes_matched'], 'failed stage')
        expected = {(r['key'], n) for r in rows if r['xlen'] == xlen for n in NS}
        seen = set()
        for p in sorted(stage.glob('shard-*/execute/*/n*/response.json')):
            n = int(p.parent.name[1:]); key = p.parent.parent.name
            ame.require((key, n) in expected and (key, n) not in seen, 'extra/duplicate response')
            seen.add((key, n)); row = by_key[key]
            b, spec = checked_artifact(Path(row['artifact']))
            request, response = ame.load(p.with_name('request.json')), ame.load(p)
            ame.require(request['backend'] == 'spike' and request['revision'] == pre['model']['revision'], 'model identity changed')
            result = validate_response(b, spec, request, response, n)
            ame.require(result['status'] == row['expected_status'] and
                        ame.load(p.with_name('verification.json'))['actual'] == result, 'verification changed')
            ame.require(all(c['exit_status'] == 0 for c in ame.load(p.with_name('commands.json'))), 'adapter error')
            if result['status'] == 3:
                ame.require(p.with_name('raw-failure.json').is_file(), 'raw FAIL lost')
            else:
                ame.require(ame.load(p.with_name('result.json'))['status'] == 'EXEC_PASS', 'positive result lost')
            words = list(struct.unpack('<32I', bytes.fromhex(response['mailbox_hex'])))
            words[4] = 0
            groups.setdefault((row['operator'], row['kind'], n), []).append(words)
            records.append(dict(key=key, n=n, actual=result, response_sha256=ame.sha(p)))
        ame.require(seen == expected, 'control execution missing')
        seen_keys = []
        for d in stage.glob('shard-*'):
            keys = (d/'keys.txt').read_text().splitlines()
            seen_keys += keys
            labels = {}
            for key in keys:
                for n in NS:
                    labels[f'execute {key} N={n}'] = {'PASS' if by_key[key]['expected_status'] == 2 else 'FAIL'}
                    labels[f'verify {key} N={n}'] = {'PASS'}
            ame.require(ame.check_result_labels((d/'ameopstatecontrol.sum').read_text(), labels), 'label mismatch')
        ame.require(sorted(seen_keys) == sorted(r['key'] for r in rows if r['xlen'] == xlen), 'shard inventory')
    ame.require(len(groups) == 20 and all(len(v) == 2 and v[0] == v[1] for v in groups.values()), 'cross-XLEN mismatch')
    ame.save(out/'audit.json', dict(expected_outcomes_matched=True, executions=len(records),
        positive_passes=sum(r['actual']['status'] == 2 for r in records),
        raw_failures=sum(r['actual']['status'] == 3 for r in records), cross_xlen_groups=len(groups),
        records=records, scope='O2/C independent faults, N=4/128, no model or production change'))
    print('STATE_CONTROL_AUDIT_PASS', len(records), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'check', 'run', 'verify', 'audit'])
    for name in ('root', 'base', 'out', 'artifact', 'platform'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--xlen', type=int, choices=(32, 64)); p.add_argument('--n', type=int, choices=NS)
    a = p.parse_args()
    try:
        if a.action == 'prepare': prepare(a.base.resolve(), a.out.resolve())
        elif a.action == 'check': return check(a.root.resolve(), a.xlen)
        elif a.action == 'run': return run(a.artifact.resolve(), a.platform.resolve(), a.n, a.out.resolve())
        elif a.action == 'verify': verify(a.artifact.resolve(), a.n, a.out.resolve())
        else: audit(a.root.resolve(), a.out.resolve())
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
