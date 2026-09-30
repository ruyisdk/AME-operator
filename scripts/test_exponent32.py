"""Independent rational evaluation and quantization for FP32 exponent operations."""
import ctypes
from fractions import Fraction
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
import ame
from test_sqrt32 import finite

def quantize(value, rm, zero_sign=0):
    if not value:
        return zero_sign << 31, 0
    sign = int(value < 0)
    value = abs(value)
    def overflow():
        infinity = rm in (0, 4) or (rm == 2 and sign) or (rm == 3 and not sign)
        return (sign << 31) | (0x7f800000 if infinity else 0x7f7fffff), 5
    if value >= 2**128:
        return overflow()
    lower, upper = 0, 0x7f7fffff
    while lower < upper:
        mid = (lower + upper + 1) // 2
        if finite(mid) <= value:
            lower = mid
        else:
            upper = mid - 1
    low = finite(lower)
    if value == low:
        return sign << 31 | lower, 0
    high = finite(lower + 1) if lower < 0x7f7fffff else Fraction(2**128)
    if rm in (0, 4):
        increment = value - low > high - value or (
            value - low == high - value and (rm == 4 or lower & 1))
    else:
        increment = (rm == 2 and sign) or (rm == 3 and not sign)
    answer = lower + bool(increment)
    if answer == 0x7f800000:
        return overflow()
    return sign << 31 | answer, 3 if answer < 0x800000 else 1

def expected(rm, a, power, old, accumulate):
    operands = [a, old] if accumulate else [a]
    if any((x & 0x7fffffff) > 0x7f800000 for x in operands):
        invalid = any((x & 0x7fffffff) > 0x7f800000 and not x & 0x400000 for x in operands)
        return 0x7fc00000, 16 if invalid else 0
    infinities = [x for x in operands if (x & 0x7fffffff) == 0x7f800000]
    if infinities:
        if len(infinities) == 2 and (a ^ old) >> 31:
            return 0x7fc00000, 16
        return infinities[0], 0
    scale = Fraction(2**power) if power >= 0 else Fraction(1, 2**(-power))
    scaled = finite(a) * (-1 if a >> 31 else 1) * scale
    total = scaled + (finite(old) * (-1 if old >> 31 else 1) if accumulate else 0)
    zero_sign = a >> 31
    if accumulate and (a ^ old) >> 31:
        zero_sign = int(rm == 2)
    return quantize(total, rm, zero_sign)


def input_value(op, phase, buffer, index):
    j = index % 16
    if op >= 4:
        power = 0 if phase == 6 else 13 if phase == 7 else 1
    elif phase == 0:
        power = (1, 0, 2, -1)[j % 4]
    elif phase == 1:
        power = (-1, -2, -24, -128)[j % 4]
    elif phase == 2:
        power = 1 + (j % 3) * 127
    elif phase == 6:
        power = -1
    elif phase == 7:
        power = (j * 37 % 513) - 256
    else:
        power = 1
    if buffer == 1:
        return (-power if op in (2, 3) else power) & 0xffffffff
    if phase == 0:
        data = (0x3f800000,0xbf800000,0x3fc00000,0xbfc00000,0x40000000,0xc0000000,
                0x3f000000,0xbf000000,0,0x80000000,0x40800000,0xc0800000,
                0x3e800000,0xbe800000,0x41000000,0xc1000000)
        old = (0x40000000,0xc0000000,0x3f000000,0xbf000000)
        return data[j] if buffer == 0 else old[j % 4]
    if phase == 1:
        data = (1,3,0x007fffff,0x00800001,0x80000001,0x80000003,0x807fffff,0x80800001)
        return data[j % 8] if buffer == 0 else 0
    if phase in (2, 3):
        data = 0x7f7fffff | ((j & 1) << 31)
        return data if buffer == 0 else 0 if phase == 2 else data ^ 0x80000000
    if phase in (4, 5):
        special = (0x7fc12345,0x7f800001,0x7f800000,0xff800000,
                   0,0x80000000,0xffc12345,0xff800001)
        return special[j % 8] if buffer == 0 else 0 if phase == 4 else special[(j+3) % 8]
    if phase == 6:
        return ((j & 4) << 29) | (2 * (j % 4) + 1 if buffer == 0 else 1)
    mixed = (j * 1664525 + 1013904223 + (0x798523af if buffer == 2 else 0)) & 0xffffffff
    return (mixed & 0x807fffff) | (((j * 41 + buffer * 37) % 255) << 23)

class Exponent32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        so = Path(cls.tmp.name) / "exponent.so"
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined", "-I", str(ame.ROOT / "runtime"),
            str(ame.ROOT / "reference/exponent32-r3.c"),
            str(ame.ROOT / "reference/exponent32-input.c"), "-o", str(so)], check=True)
        cls.lib = ctypes.CDLL(str(so))
        cls.ref = cls.lib.i3n_scale32
        cls.ref.argtypes = [ctypes.c_uint, ctypes.c_uint32, ctypes.c_int,
                           ctypes.c_uint32, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint32)]
        cls.ref.restype = ctypes.c_uint

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def compare(self, a, power, old, accumulate):
        for rm in range(5):
            out = ctypes.c_uint32()
            flags = self.ref(rm, a, power, old, accumulate, ctypes.byref(out))
            self.assertEqual((out.value, flags), expected(rm, a, power, old, accumulate),
                             (rm, hex(a), power, hex(old), accumulate))

    def test_normative_flag_bits(self):
        # src/state.adoc: NV[4], DZ[3], OF[2], UF[1], NX[0].
        # Literal known answers also test each oracle independently.
        vectors = (
            (0, 0x7f7fffff, 1, 0, 0, 0x7f800000, 0b00101),
            (2, 0xff7fffff, 1, 0, 0, 0xff800000, 0b00101),
            (0, 1, -1, 0, 0, 0, 0b00011),
            (0, 0x7f800001, 1, 0, 0, 0x7fc00000, 0b10000),
            (0, 0x3f800000, -24, 0x3f800000, 1, 0x3f800000, 0b00001),
        )
        for rm, a, power, old, acc, bits, flags in vectors:
            self.assertEqual(expected(rm, a, power, old, acc), (bits, flags))
            out = ctypes.c_uint32()
            actual = self.ref(rm, a, power, old, acc, ctypes.byref(out))
            self.assertEqual((out.value, actual), (bits, flags))
            self.assertFalse(actual & 0b01000)  # No divide-by-zero here.

    def test_rational_oracle(self):
        patterns = (0, 0x80000000, 1, 2, 3, 0x80000001, 0x007fffff, 0x00800000,
                    0x00800001, 0x3f800000, 0xbf800000, 0x3f800001, 0x7f7fffff,
                    0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc12345,
                    0x7f800001, 0xff800001)
        for a in patterns:
            for power in (-256, -149, -24, -2, -1, 0, 1, 24, 128, 256):
                self.compare(a, power, 0, 0)
                for old in (0, 0x80000000, 1, a ^ 0x80000000):
                    self.compare(a, power, old, 1)
        rng = random.Random(20260928)
        for _ in range(1500):
            self.compare(rng.getrandbits(32), rng.randrange(-256,257),
                         rng.getrandbits(32), rng.randrange(2))

    def test_single_round_counterexamples(self):
        self.assertEqual(expected(0, 1, -1, 1, 1), (2, 3))
        self.assertEqual(expected(0, 0x7f7fffff, 1, 0xff7fffff, 1), (0x7f7fffff, 0))
        first, _ = expected(0, 1, -1, 0, 0)
        self.assertNotEqual(expected(0, first, 0, 1, 1), expected(0, 1, -1, 1, 1))

    def test_target_inputs(self):
        data, power = self.lib.i3n_input, self.lib.i3n_power
        data.argtypes = [ctypes.c_uint] * 4
        data.restype = ctypes.c_uint32
        power.argtypes = [ctypes.c_uint] * 3
        power.restype = ctypes.c_int
        for op in range(6):
            for phase in range(8):
                for i in range(64):
                    for buf in range(3):
                        self.assertEqual(data(op,phase,buf,i), input_value(op,phase,buf,i))
                    bits = input_value(op,phase,1,i)
                    exponent = bits - (2**32 if bits >> 31 else 0)
                    if op in (2,3):
                        exponent = -exponent
                    self.assertEqual(power(op,phase,i),exponent)
                    if op >= 4:
                        self.assertGreaterEqual(exponent,0)
                    if i < 16:
                        a, old = input_value(op,phase,0,i), input_value(op,phase,2,i)
                        self.compare(a,exponent,old,op & 1)
                        if phase == 0:
                            for rm in range(5):
                                self.assertEqual(expected(rm,a,exponent,old,op & 1)[1],0)


    def test_reject_invalid(self):
        for rm, power, accumulate in ((5,0,0),(0,-257,0),(0,257,0),(0,0,2)):
            out = ctypes.c_uint32(0xa5a5a5a5)
            self.assertEqual(self.ref(rm,0,power,0,accumulate,ctypes.byref(out)),0xffffffff)
            self.assertEqual(out.value,0xa5a5a5a5)
