#!/usr/bin/env python3
"""Independent FCSR batch adapter: standard F enabled, same frozen AME model."""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

from ame import ENTRY, load, require, save, sha

BOOT = 0x80f00000
SUMMARY = re.compile(r"^AME N=(\d+) NELEM=(\d+) UDSZ=(\d+) MREGS=(\d+) "
                     r"ACCREGS=(\d+) owned=(\d+) fflags=0x([0-9a-f]+) "
                     r"xsat=(\d+) status=0x([0-9a-f]+)$", re.M)


def elf_layout(path):
    data = Path(path).read_bytes()
    require(data[:4] == b"\x7fELF" and data[5:7] == b"\x01\x01", "need LE ELF v1")
    xlen = {1: 32, 2: 64}.get(data[4])
    require(xlen is not None, "unsupported ELF class")
    fmt = "<HHIIIIIHHHHHH" if xlen == 32 else "<HHIQQQIHHHHHH"
    h = struct.unpack_from(fmt, data, 16)
    require(h[0] == 2 and h[1] == 243, "need executable RISC-V ELF")
    pf = "<IIIIIIII" if xlen == 32 else "<IIQQQQQQ"
    require(h[8] == struct.calcsize(pf), "unexpected program header size")
    spans = []
    for i in range(h[9]):
        ph = struct.unpack_from(pf, data, h[4] + i * h[8])
        if ph[0] == 1:
            addr, size = (ph[3], ph[5]) if xlen == 32 else (ph[4], ph[6])
            spans.append((addr, addr + size))
    require(spans, "ELF has no loadable segment")
    return xlen, h[3], spans


def debug_commands(symbols, trace=False):
    until = "untiln" if trace else "until"
    lines = [f'until pc 0 {symbols["_start"]:x}', "pc 0", "priv 0",
             "reg 0 misa", "reg 0 mstatus", "reg 0 mie", "ame 0",
             f'{until} pc 0 {symbols["ame_test_done"]:x}', "pc 0", "ame 0"]
    lines += [f'mem {symbols["ame_mailbox"] + i * 4:x}' for i in range(32)]
    return "\n".join(lines + ["quit", ""])


def parse_debug(text, symbols, csr_map):
    values = [int(v, 16) for v in re.findall(r"^0x([0-9a-fA-F]+)$", text, re.M)]
    require(len(values) == 37, f"incomplete Spike observations ({len(values)}/37)")
    pc, misa, status, mie, stop = values[:5]
    require(pc == symbols["_start"] and stop == symbols["ame_test_done"], "wrong stop PC")
    xlen = 64 if misa >> 62 == 2 else 32 if misa >> 30 == 1 else 0
    require(xlen in (32, 64), "invalid observed MXL")
    require(misa & (1 << 5), "standard F extension absent at entry")
    require(re.search(r"^M$", text, re.M), "entry is not M-mode")
    require(status & 0x06000000 and not status & 8 and mie == 0,
            "MS disabled or interrupts enabled at entry")
    states = SUMMARY.findall(text)
    require(len(states) == 2, "missing AME entry/completion observations")
    begin, end = states
    n, nelem, uds, m, a, owned = map(int, begin[:6])
    require(nelem == n * n and owned == 0, "bad initial AME shape/ownership")
    require(begin[:5] == end[:5], "AME configuration changed during execution")
    observed = dict(xlen=xlen, n=n, uds=uds, mregs=m, accregs=a,
                    csr_map=csr_map, entry_contract=ENTRY)
    # Debug mem loads either 32 or 64 bits depending on alignment. Every query
    # addresses the next u32 word; retain its low word in little-endian order.
    mailbox = struct.pack("<32I", *(v & 0xffffffff for v in values[5:]))
    return observed, stop, mailbox


def run(args):
    request, model = load(args.request), load(args.model)
    directory = args.response.resolve().parent
    require(request["schema"] == 1 and request["mailbox_bytes"] == 128, "unknown request")
    require(request["backend"] == "spike" and request["revision"] == model["revision"],
            "wrong backend/revision")
    require(sha(model["spike"]) == model["spike_sha256"], "Spike binary changed")
    elf = Path(request["elf"])
    require(sha(elf) == request["elf_sha256"], "ELF changed")
    xlen, entry, spans = elf_layout(elf)
    expected, symbols = request["expected"], request["symbols"]
    require(xlen == expected["xlen"] and entry == symbols["_start"] == 0x80000000,
            "ELF entry/XLEN mismatch")
    require(all(0x80000000 <= lo <= hi <= BOOT for lo, hi in spans),
            "ELF overlaps adapter bootstrap or lies outside test RAM")
    require(symbols["ame_mailbox"] % 4 == 0 and any(
        lo <= symbols["ame_mailbox"] and symbols["ame_mailbox"] + 128 <= hi
        for lo, hi in spans), "mailbox outside ELF")
    boot = model["boot"][str(xlen)]
    require(sha(boot["path"]) == boot["sha256"], "bootstrap changed")
    cmdfile = directory / "spike-debug.cmd"
    cmdfile.write_text(debug_commands(symbols, args.trace))
    argv = [model["spike"], f"--isa=rv{xlen}imf_zicsr_ztt", "--priv=M", "-p1",
            "-m0x80000000:0x1000000", f"--pc={BOOT:#x}",
            f'--ame-nelem={expected["n"] ** 2}', f'--ame-udsz={expected["uds"]}',
            f'--ame-mregs={expected["mregs"]}', f'--ame-accregs={expected["accregs"]}',
            "--instructions=500000000", "-d", f"--debug-cmd={cmdfile}"]
    if args.trace:
        argv += ["-l", "--log-commits", f'--log={directory / "spike-trace.log"}']
    argv += [f'+payload={boot["path"]}', str(elf)]
    save(directory / "spike-command.json", dict(argv=argv, model=model))
    with (directory / "spike.log").open("w") as output:
        subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                       check=True, timeout=90)
    text = (directory / "spike.log").read_text()
    observed, stop, mailbox = parse_debug(text, symbols, model["csr_map"])
    require(sha(elf) == request["elf_sha256"] and sha(model["spike"]) == model["spike_sha256"],
            "executable changed during run")
    save(args.response, dict(schema=1, nonce=request["nonce"], elf_sha256=sha(elf),
         backend="spike", revision=model["revision"], observed=observed,
         stop_pc=stop, mailbox_hex=mailbox.hex()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--response", type=Path, required=True)
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
