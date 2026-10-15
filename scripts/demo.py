#!/usr/bin/env python3
"""Build, disassemble, execute and independently verify the N=4 AME demo."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from reference import compare, gemm

ROOT = Path(__file__).resolve().parents[1]
ARCH = "rv64im_zicsr_ztt0p6"
PROFILE = "gcc-runtime-u32-m16-a4"
BOOT = 0x80f00000
SUMMARY = re.compile(
    r"^AME N=(\d+) NELEM=(\d+) UDSZ=(\d+) MREGS=(\d+) ACCREGS=(\d+) "
    r"owned=(\d+) fflags=0x([0-9a-f]+) xsat=(\d+) status=0x([0-9a-f]+)$", re.M)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_command(argv, directory, label, records, timeout=60):
    argv = [str(arg) for arg in argv]
    records.append(dict(argv=argv, step=label))
    save(directory / "commands.json", records)
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=timeout)
    require(result.returncode == 0,
            f"{label} failed ({result.returncode}):\n{result.stdout[-4000:]}")
    if label in ("disassembly", "attributes", "symbols"):
        (directory / (label + ".txt")).write_text(result.stdout)
    return result.stdout


def fixture_header(inputs, path):
    gemm(inputs["a"], inputs["b"])  # Validate exact arithmetic before building.
    lines = ["/* Generated from examples/inputs.json by scripts/demo.py. */"]
    for name in ("a", "b"):
        values = ", ".join(str(v) for row in inputs[name] for v in row)
        lines.append(f"#define INPUT_{name.upper()} {{{values}}}")
    path.write_text("\n".join(lines) + "\n")


def parse_symbols(text):
    return {parts[2]: int(parts[0], 16) for line in text.splitlines()
            if len(parts := line.split()) == 3}


def matrix(words):
    signed = [v - (1 << 32) if v & (1 << 31) else v for v in words]
    return [signed[i:i + 4] for i in range(0, 16, 4)]


def observe(log, symbols, inputs):
    values = [int(v, 16) for v in re.findall(r"^0x([0-9a-fA-F]+)$", log, re.M)]
    require(len(values) == 93, f"incomplete Spike observations: {len(values)}/93")
    start, misa, mstatus, mie, done = values[:5]
    require(start == symbols["_start"] == 0x80000000, "wrong entry PC")
    require(done == symbols["tutorial_stop"], "did not reach completion PC after fence")
    require(misa >> 62 == 2 and re.search(r"^M$", log, re.M), "expected RV64 M-mode")
    require(mstatus & 0x06000000 and not mstatus & 8 and mie == 0,
            "expected enabled AME MS and disabled interrupts")
    states = SUMMARY.findall(log)
    require(len(states) == 2, "missing AME entry/completion state")
    require(all(tuple(map(int, state[:6])) == (4, 16, 32, 16, 4, 0) for state in states),
            "wrong AME configuration or ownership at entry/completion")
    require(all(int(v, 16) == 0 for v in (states[1][6], states[1][8]))
            and int(states[1][7]) == 0, "nonzero final AME status/flags")
    # Spike's mem command may return 64 bits at aligned addresses. Keep low u32.
    words = [v & 0xffffffff for v in values[5:]]
    mailbox, a, b, output = words[:32], words[32:48], words[48:64], words[64:]
    require(mailbox[:4] == [0x414d4534, 1, 2, 0],
            f"target failed: mailbox={mailbox}; trap cause={mailbox[12]} "
            f"PC=0x{(mailbox[15] << 32) | mailbox[14]:x}")
    require(mailbox[4:6] == [4, 32] and mailbox[6] & 1
            and mailbox[7:12] == [0, 0, 0, 0, 0], "invalid target AME state")
    require(output[:4] + output[20:] == [0xa5c369f0] * 8, "output canary corrupted")
    require(matrix(a) == inputs["a"] and matrix(b) == inputs["b"],
            "target inputs differ from examples/inputs.json")
    return dict(a=matrix(a), b=matrix(b), c=matrix(output[4:20]),
                mailbox=mailbox, entry_pc=start, completion_pc=done,
                profile=dict(xlen=64, n=4, nelem=16, uds=32, mregs=16, accregs=4),
                canary_words_checked=8, ownership_released=True)


def build_and_run(language, opt, fault, output, inputs):
    label = f"{language}-{opt}" + ("-fault" if fault else "")
    directory = output / label
    directory.mkdir()
    records = []
    prefix = ROOT / "toolchain/bin/riscv64-unknown-elf-"
    tool = lambda name: Path(str(prefix) + name)
    spike = ROOT / "spike/bin/spike"
    needed = [tool(name) for name in ("gcc", "g++", "objdump", "nm", "readelf")] + [spike]
    for path in needed:
        require(path.is_file(), f"missing installed tool: {path}")
    fixture_header(inputs, directory / "fixture.h")
    common = ["-march=" + ARCH, "-mabi=lp64", "-mcmodel=medany",
              "-mztt-profile=" + PROFILE, "-ffreestanding", "-fno-builtin",
              "-fno-stack-protector", "-fno-pic", "-fno-tree-vectorize",
              "-fno-tree-slp-vectorize", "-g", "-Wall", "-Wextra", "-Werror",
              "-I" + str(directory), "-I" + str(ROOT / "examples")]
    objects = []
    sources = [("start", ROOT / "runtime/start.S", False),
               ("runtime", ROOT / "runtime/demo.c", False),
               ("gemm", ROOT / "examples" / ("gemm.cpp" if language == "cpp" else "gemm.c"),
                language == "cpp")]
    print(f"[{label}] compile + link + disassemble", flush=True)
    for name, source, cxx in sources:
        obj = directory / (name + ".o")
        compiler = tool("g++" if cxx else "gcc")
        flags = common + (["-DTUTORIAL_INJECT_FAULT"] if fault else [])
        if source.suffix != ".S":
            flags += ["-" + opt if name == "gemm" else "-O2",
                      "-std=gnu++17" if cxx else "-std=gnu11"]
            if cxx:
                flags += ["-fno-exceptions", "-fno-rtti"]
            asm = directory / (name + ".s")
            run_command([compiler] + flags + ["-S", source, "-o", asm], directory,
                        name + "-compile", records)
            source = asm
        run_command([compiler] + common + ["-c", source, "-o", obj], directory,
                    name + "-assemble", records)
        objects.append(obj)
    elf = directory / "gemm.elf"
    link = [tool("gcc")] + common + ["-nostdlib", "-static", "-Wl,--no-relax"]
    run_command(link + ["-Wl,-T," + str(ROOT / "runtime/link.ld"),
                       "-Wl,-Map," + str(directory / "gemm.map"), *objects, "-o", elf],
                directory, "link", records)
    boot = directory / "spike_boot.elf"
    run_command(link + [ROOT / "runtime/spike_boot.S",
                       "-Wl,-T," + str(ROOT / "runtime/spike_boot.ld"), "-o", boot],
                directory, "bootstrap", records)
    dump = run_command([tool("objdump"), "-d", elf], directory, "disassembly", records)
    for mnemonic in ("mls.rm", "mmov.a.m", "mmulacc.2d", "mmov.m.a", "mss.rm"):
        require(mnemonic in dump, "missing expected instruction " + mnemonic)
    attributes = run_command([tool("readelf"), "-hlA", elf], directory, "attributes", records)
    require("ztt0p6" in attributes, "missing Ztt v0.6 ELF attribute")
    symbols = parse_symbols(run_command([tool("nm"), "-n", elf], directory, "symbols", records))
    for name in ("_start", "tutorial_done", "tutorial_stop", "tutorial_mailbox", "tutorial_a",
                 "tutorial_b", "tutorial_output", "__stack_top"):
        require(name in symbols, "missing ELF symbol " + name)
    require(symbols["__stack_top"] <= BOOT, "target stack overlaps bootstrap")
    commands = [f'until pc 0 {symbols["_start"]:x}', "pc 0", "priv 0",
                "reg 0 misa", "reg 0 mstatus", "reg 0 mie", "ame 0",
                f'until pc 0 {symbols["tutorial_stop"]:x}', "pc 0", "ame 0"]
    for name, count in (("tutorial_mailbox", 32), ("tutorial_a", 16),
                        ("tutorial_b", 16), ("tutorial_output", 24)):
        commands += [f"mem {symbols[name] + 4 * i:x}" for i in range(count)]
    cmdfile = directory / "spike-debug.cmd"
    cmdfile.write_text("\n".join(commands + ["quit", ""]))
    print(f"[{label}] Spike: RV64 UDS32/M16/Acc4, N=4 (16 elements)", flush=True)
    log = run_command([spike, "--isa=rv64im_zicsr_ztt", "--priv=M", "-p1",
                       "-m0x80000000:0x1000000", f"--pc={BOOT:#x}",
                       "--ame-nelem=16", "--ame-udsz=32", "--ame-mregs=16", "--ame-accregs=4",
                       "--instructions=5000000", "-d", f"--debug-cmd={cmdfile}",
                       f"+payload={boot}", elf], directory, "spike", records, timeout=45)
    observation = observe(log, symbols, inputs)
    save(directory / "observation.json", observation)
    reference = compare(observation)
    save(directory / "reference.json", reference)
    if fault:
        require(not reference["passed"] and len(reference["mismatches"]) == 1
                and reference["mismatches"][0]["row"] == 0
                and reference["mismatches"][0]["column"] == 0
                and (reference["actual"][0][0] & 0xffffffff)
                    == ((reference["expected"][0][0] & 0xffffffff) ^ 1),
                "negative control did not produce exactly the intended numerical mismatch")
        result = "EXPECTED_REJECTION"
    else:
        require(reference["passed"], f"independent reference mismatch: {reference['mismatches']}")
        result = "PASS"
    report = dict(result=result, language=language, optimization=opt,
                  injected_fault=fault,
                  profile=observation["profile"], reference=reference)
    save(directory / "report.json", report)
    print(f"[{label}] {result}: 16 INT32 elements, 8 canaries, completion + AME state checked", flush=True)
    if not fault:
        for row in observation["c"]:
            print("  " + " ".join(f"{value:6d}" for value in row))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("c", "cpp", "all"), default="c")
    parser.add_argument("--opt", choices=("O0", "O2", "all"), default="O2")
    parser.add_argument("--negative-control", action="store_true",
                        help="also execute a deliberately corrupted C result and require rejection")
    parser.add_argument("--out", type=Path,
                        help="new output directory (default: build/demo-<time>)")
    args = parser.parse_args()
    output = (args.out or ROOT / "build" / f"demo-{time.time_ns()}").resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
        inputs = json.loads((ROOT / "examples/inputs.json").read_text())
        reports = []
        for language in (("c", "cpp") if args.language == "all" else (args.language,)):
            for opt in (("O0", "O2") if args.opt == "all" else (args.opt,)):
                reports.append(build_and_run(language, opt, False, output, inputs))
        if args.negative_control:
            reports.append(build_and_run("c", "O2", True, output, inputs))
        save(output / "summary.json", dict(passed=True, runs=reports))
        print(f"Evidence: {output}")
        return 0
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        print(f"Evidence: {output}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
