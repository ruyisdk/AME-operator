"""Independent rational intervals check the target integer square-root oracle."""
import ctypes
from fractions import Fraction
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
import ame

EXACT = (0, 0x80000000, 0x3f800000, 0x40800000, 0x41100000, 0x41800000,
         0x3e800000, 0x3d800000, 2, 0x00800000, 8, 0x7e800000,
         0x7f800000, 0x41c80000, 0x42800000, 0x3c800000)
ROUNDED = (0x40000000, 0x40400000, 0x40a00000, 0x40e00000, 0x41000000,
           0x41200000, 0x3f000000, 0x3f800001, 0x3f7fffff, 0x007fffff,
           1, 0x7f7fffff, 0x407fffff, 0x40800001, 3, 0x00800001)


def input_word(phase, index):
    slot = index % 16
    if phase == 0:
        return EXACT[slot]
    if phase in (1, 7):
        return ROUNDED[slot]
    if phase == 2:
        return 0xbf800000 if slot == 0 else 0xff800000 if slot == 15 else 0x80000000 | ROUNDED[slot]
    if phase == 3:
        return 0x7fc00000 | ((index & 1) << 31) | ((index * 37 + 1) & 0x3fffff)
    if phase == 4:
        return 0x7f800001 | ((index & 1) << 31) | ((index * 37) & 0x3fffff)
    if phase == 5:
        return 0x7f800001 if slot == 1 else 0x7fc00001 if slot == 2 else ROUNDED[slot]
    mixed = (index * 1664525 + 1013904223) & 0xffffffff
    return (mixed & 0x807fffff) | (((index * 73 + 17) % 255) << 23)


def finite(bits):
    exponent = (bits >> 23) & 255
    significand = (bits & 0x7fffff) + (2**23 if exponent else 0)
    power = (exponent - 127 if exponent else -126) - 23
    return Fraction(significand * (2**power if power >= 0 else 1),
                    1 if power >= 0 else 2**(-power))


def all_expected(bits):
    mag = bits & 0x7fffffff
    fraction = mag & 0x7fffff
    if mag >> 23 == 255 and fraction:
        return [(0x7fc00000, 0 if fraction & 0x400000 else 16)] * 5
    if mag == 0:
        return [(bits, 0)] * 5
    if bits >> 31:
        return [(0x7fc00000, 16)] * 5
    if mag == 0x7f800000:
        return [(bits, 0)] * 5
    value = finite(bits)
    # Search ordered FP32 encodings using rational squares, not an integer-sqrt algorithm.
    lower, upper = 0, 0x7f7fffff
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if finite(middle)**2 <= value:
            lower = middle
        else:
            upper = middle - 1
    low_value = finite(lower)
    if low_value**2 == value:
        return [(lower, 0)] * 5
    midpoint = (low_value + finite(lower + 1)) / 2
    halfway = midpoint**2
    nearest_even = lower + (value > halfway or (value == halfway and lower & 1))
    nearest_away = lower + (value >= halfway)
    return [(nearest_even, 1), (lower, 1), (lower, 1), (lower + 1, 1), (nearest_away, 1)]


class Sqrt32(unittest.TestCase):

    def test_actual_reference_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            so = Path(tmp) / "sqrt.so"
            subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined", "-I", str(ame.ROOT / "runtime"),
                str(ame.ROOT / "reference/sqrt32.c"), str(ame.ROOT / "reference/sqrt32-input.c"),
                "-o", str(so)], check=True)
            lib = ctypes.CDLL(str(so))
            ref, data = lib.i3m_sqrt32, lib.i3m_input
            ref.argtypes = [ctypes.c_uint, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
            ref.restype = ctypes.c_uint
            data.argtypes = [ctypes.c_uint, ctypes.c_uint]
            data.restype = ctypes.c_uint32
            patterns = {s << 31 | e << 23 | f for s in (0, 1) for e in range(256)
                        for f in (0, 1, 0x3fffff, 0x400000, 0x7fffff)}
            rng = random.Random(20260928)
            patterns.update(rng.getrandbits(32) for _ in range(3000))
            for phase in range(8):
                for i in range(128**2):
                    self.assertEqual(data(phase, i), input_word(phase, i))
                patterns.update(input_word(phase, i) for i in range(256))
            for bits in sorted(patterns):
                for rm, (answer, flags) in enumerate(all_expected(bits)):
                    output = ctypes.c_uint32()
                    actual = ref(rm, bits, ctypes.byref(output))
                    self.assertEqual((output.value, actual), (answer, flags), (rm, hex(bits)))
            output = ctypes.c_uint32(123)
            self.assertEqual(ref(5, 0, ctypes.byref(output)), 0xffffffff)
            self.assertEqual(output.value, 123)
            for rm in range(5):
                self.assertEqual({all_expected(x)[rm][1] for x in EXACT}, {0})
