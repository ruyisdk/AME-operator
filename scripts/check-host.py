#!/usr/bin/env python3
"""Check the actual host and all installed tools without changing the host."""
import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-target', action='store_true',
                        help='require x86_64 Ubuntu 26.04.1 LTS and glibc 2.43')
    parser.add_argument('--output', type=Path, help='save the report as JSON')
    args = parser.parse_args()
    os_release = {}
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os_release[key] = value.strip('"')
    libc = platform.libc_ver()
    host = dict(architecture=platform.machine(), os=os_release, libc=list(libc),
                python=platform.python_version())
    target = (host['architecture'] == 'x86_64'
              and os_release.get('PRETTY_NAME') == 'Ubuntu 26.04.1 LTS'
              and libc == ('glibc', '2.43'))
    errors = []
    if host['architecture'] != 'x86_64':
        errors.append('This binary bundle requires x86_64 Linux.')
    if args.require_target and not target:
        errors.append('Actual host differs from Ubuntu 26.04.1 LTS / glibc 2.43 target.')
    prefix = ROOT / 'toolchain/bin/riscv64-unknown-elf-'
    tools = [Path(str(prefix) + name) for name in
             ('gcc', 'g++', 'as', 'ld', 'objdump', 'nm', 'readelf')]
    tools += [ROOT / 'spike/bin/spike']
    if not shutil.which('dtc'):
        errors.append('Missing dtc (device-tree-compiler), required during Spike execution.')
    gcc = Path(str(prefix) + 'gcc')
    for name in ('cc1', 'cc1plus', 'as', 'ld'):
        try:
            query = subprocess.run([str(gcc), '-print-prog-name=' + name],
                                   capture_output=True, text=True, check=True, timeout=20)
            program = Path(query.stdout.strip())
            if (not program.is_file() or not os.access(program, os.X_OK)
                    or not program.resolve().is_relative_to(ROOT)):
                errors.append(f'Compiler helper {name} is missing or external: {program}')
            elif program not in tools:
                tools.append(program.resolve())
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f'Compiler helper lookup failed for {name}: {error}')
    observed = {}
    for tool in tools:
        rel = str(tool.relative_to(ROOT))
        if not tool.is_file() or not os.access(tool, os.X_OK):
            errors.append(f'Missing executable: {rel}')
            continue
        try:
            result = subprocess.run([str(tool), '--help' if tool.name == 'spike' else '--version'],
                                    capture_output=True, text=True, timeout=20)
            # Spike --help exits 0 or 1 depending on the upstream CLI.
            output = result.stdout + result.stderr
            valid = result.returncode == 0 or (tool.name == 'spike' and result.returncode == 1
                                               and 'RISC-V ISA Simulator' in output)
            if not valid:
                errors.append(f'{rel} cannot start: {output[:300]}')
            libraries = subprocess.run(['ldd', str(tool)], capture_output=True,
                                       text=True, timeout=20)
            needed = libraries.stdout + libraries.stderr
            if 'not found' in needed:
                errors.append(f'{rel} has missing shared libraries: {needed}')
            observed[rel] = dict(version=output.splitlines()[:3], dependencies=needed)
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f'{rel}: {error}')
    if gcc.is_file():
        try:
            query = subprocess.run([str(gcc), '-print-file-name=include'],
                                   capture_output=True, text=True, check=True, timeout=20)
            header = Path(query.stdout.strip()) / 'riscv_ztt.h'
            if not header.is_file() or not header.resolve().is_relative_to(ROOT):
                errors.append(f'AME header is missing or external: {header}')
            query = subprocess.run([str(gcc), '-print-libgcc-file-name'],
                                   capture_output=True, text=True, check=True, timeout=20)
            libgcc = Path(query.stdout.strip())
            if not libgcc.is_file() or not libgcc.resolve().is_relative_to(ROOT):
                errors.append(f'libgcc is missing or external: {libgcc}')
        except (OSError, subprocess.SubprocessError) as error:
            errors.append(f'Compiler header search failed: {error}')
    report = dict(host=host, requested_target_matched=target,
                  require_target=args.require_target, passed=not errors,
                  tools=observed, errors=errors)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Host: {host['architecture']}, {os_release.get('PRETTY_NAME')}, {' '.join(libc)}")
    print(f"Requested target matched: {target}")
    for error in errors:
        print('FAIL:', error)
    print('HOST_CHECK_PASS' if not errors else 'HOST_CHECK_FAIL')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
