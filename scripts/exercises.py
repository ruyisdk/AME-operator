#!/usr/bin/env python3
"""Check the two intentional compiler failures used in the tutorial."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = (args.out or ROOT / "build" / f"exercises-{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = ROOT / "toolchain/bin/riscv64-unknown-elf-gcc"
    common = [str(compiler), "-mabi=lp64", "-mcmodel=medany", "-ffreestanding",
              "-O2", "-S", str(ROOT / "examples/gemm.c")]
    cases = [
        ("missing-isa", ["-march=rv64im_zicsr", "-mztt-profile=gcc-runtime-u32-m16-a4"],
         "requires the 'ztt0p6' ISA extension"),
        ("missing-profile", ["-march=rv64im_zicsr_ztt0p6"],
         "riscv_ztt.h requires a supported -mztt-profile configuration"),
    ]
    reports = []
    for name, flags, diagnostic in cases:
        argv = common + flags + ["-o", str(output / (name + ".s"))]
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=30)
        passed = result.returncode != 0 and diagnostic in result.stdout
        reports.append(dict(name=name, argv=argv, returncode=result.returncode,
                            expected_diagnostic=diagnostic, passed=passed))
        (output / "diagnostics.json").write_text(json.dumps(reports, indent=2) + "\n")
        if not passed:
            raise RuntimeError(f"{name}: expected rejection was not observed; see {output}")
        print(f"{name}: EXPECTED_REJECTION (exit {result.returncode})")
        print("  " + diagnostic)
    print(f"Evidence: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
