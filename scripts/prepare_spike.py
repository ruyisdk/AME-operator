#!/usr/bin/env python3
"""Record a locally built Spike and assemble its standalone MS bootstrap."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess

from ame import ROOT, CSR_MAP, ENTRY, command, config, save, sha, tool_paths, require
from spike_adapter import BOOT, elf_layout

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--spike", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--config", type=Path, required=True)
p.add_argument("--revision", help="source revision for an exported source tree without .git")
args = p.parse_args()
out = args.out.resolve()
source, spike = args.source.resolve(), args.spike.resolve()
if args.revision:
    head, diff = args.revision, ""
else:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
header = (source / "riscv/ame_encoding.h").read_text()
addresses = {name: int(value, 16) for name, value in re.findall(
    r"^#define CSR_(AME\w+)\s+(0x[0-9a-fA-F]+)", header, re.M)}
mapped = [addresses[k] for k in ("AMESTYPE", "AMEFFLAGS", "AMEXSAT")]
require(mapped == [0x810, 0x811, 0x812], "expected CSR addresses 0x810/0x811/0x812")
csr_map = CSR_MAP
c = config(args.config)
out.mkdir(parents=True, exist_ok=False)
shutil.copy2(spike, out / "spike")
spike = out / "spike"
(out / "model-source.patch").write_text(diff)
tools, records, boot = tool_paths(c), [], {}
for xlen in (32, 64):
    obj, elf = out / f"boot{xlen}.o", out / f"boot{xlen}.elf"
    command([tools["assembler"], f"-march=rv{xlen}im_zicsr",
             "-o", obj, ROOT / "runtime/spike_boot.S"], out, f"boot{xlen}-as", c, records)
    command([tools["linker"], "-m", f"elf{xlen}lriscv", "-T",
             ROOT / "runtime/spike_boot.ld", "-o", elf, obj], out, f"boot{xlen}-ld", c, records)
    bits, entry, spans = elf_layout(elf)
    require(bits == xlen and entry == BOOT and all(
        BOOT <= lo <= hi <= 0x81000000 for lo, hi in spans), "bad bootstrap layout")
    boot[str(xlen)] = dict(path=str(elf), sha256=sha(elf))
revision = head + ("+local-" + sha(out / "model-source.patch")[:12] if diff else "")
save(out / "model.json", dict(schema=1, revision=revision, source=str(source),
     source_patch_sha256=sha(out / "model-source.patch"), spike=str(spike),
     revision_source="user-supplied" if args.revision else "git",
     encoding_header_sha256=sha(source / "riscv/ame_encoding.h"),
     spike_sha256=sha(spike), csr_map=csr_map, csr_addresses=addresses, boot=boot,
     bootstrap_source_sha256=sha(ROOT / "runtime/spike_boot.S"),
     bootstrap_linker_sha256=sha(ROOT / "runtime/spike_boot.ld")))
print(out / "model.json")
