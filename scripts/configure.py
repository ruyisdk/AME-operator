#!/usr/bin/env python3
"""Resolve operator test templates against a local AME toolchain."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import ame
import operators


def resolve(template, gcc, binutils, include, runtest):
    c = ame.config(template)
    c.update(toolchain_root="/", gcc_build=str(gcc.resolve()),
             binutils_build=str(binutils.resolve()), gcc_include=str(include.resolve()),
             dejagnu=str(runtest.resolve()))
    for name, tool in dict(ame.tool_paths(c), runtest=runtest).items():
        ame.require(tool.is_file() and os.access(tool, os.X_OK), f"missing executable {name}: {tool}")
    ame.require((include / "riscv_ztt.h").is_file(), "missing riscv_ztt.h")
    operators.plan(c)
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gcc-build", type=Path, required=True, help="directory containing xgcc and xg++")
    p.add_argument("--binutils-build", type=Path, required=True)
    p.add_argument("--gcc-include", type=Path, required=True, help="directory containing riscv_ztt.h")
    p.add_argument("--runtest", type=Path, default=shutil.which("runtest"))
    p.add_argument("--out", type=Path, default=ame.ROOT / "build/config")
    a = p.parse_args()
    ame.require(a.runtest is not None, "install DejaGNU or supply --runtest")
    configs = {}
    for template in sorted((ame.ROOT / "configs/operators").glob("*.json")):
        if "toolchain_root" in ame.load(template):
            configs[template.name] = resolve(template, a.gcc_build, a.binutils_build,
                                              a.gcc_include, a.runtest)
    a.out.mkdir(parents=True, exist_ok=False)
    for name, c in configs.items():
        ame.save(a.out / name, c)
    print(f"CONFIGURED {len(configs)} suites: {a.out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
