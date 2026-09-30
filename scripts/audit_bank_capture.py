"""RISC-V disassembly helpers used by operator checks."""
import re

LINE = re.compile(r'^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+([\w.]+)\s*(.*)$')

REGS = ('zero ra sp gp tp t0 t1 t2 s0 s1 a0 a1 a2 a3 a4 a5 a6 a7 '
        's2 s3 s4 s5 s6 s7 s8 s9 s10 s11 t3 t4 t5 t6').split()

def insns(text):
    return {int(m[1], 16): (int(m[2], 16), m[3], m[4].split('#')[0].strip())
            for line in text.splitlines() if (m := LINE.match(line))}

def signed(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v
