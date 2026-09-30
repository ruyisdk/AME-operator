"""FP32 dyadic reference cross-checked with an independent rational oracle."""
import ctypes
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
import ame
from test_exponent32 import quantize
from test_sqrt32 import finite


def expected(op, rm, a, b):
    am, bm = a & 0x7fffffff, b & 0x7fffffff
    sa, sb = a >> 31, b >> 31
    if am > 0x7f800000 or bm > 0x7f800000:
        nv = (am > 0x7f800000 and not a & 0x400000) or (
            bm > 0x7f800000 and not b & 0x400000)
        return 0x7fc00000, 16 if nv else 0
    if op == 1:
        sa ^= 1
    if op == 2:
        if (am == 0x7f800000 and bm == 0) or (bm == 0x7f800000 and am == 0):
            return 0x7fc00000, 16
        if am == 0x7f800000 or bm == 0x7f800000:
            return ((sa ^ sb) << 31) | 0x7f800000, 0
        value = finite(a) * finite(b) * (-1 if sa ^ sb else 1)
        return quantize(value, rm, sa ^ sb)
    if am == 0x7f800000 or bm == 0x7f800000:
        if am == bm and sa != sb:
            return 0x7fc00000, 16
        return ((sa if am == 0x7f800000 else sb) << 31) | 0x7f800000, 0
    value = finite(a) * (-1 if sa else 1) + finite(b) * (-1 if sb else 1)
    return quantize(value, rm, sa if sa == sb else int(rm == 2))


def scalar(op, phase):
    k = op % 3
    return (0x3f800000,
            0x3f800003 if k == 2 else 0x3f800000,
            0x3f000000 if k == 2 else 1, 0x7f7fffff,
            0 if k == 2 else 0x7f800000, 0x7f800001,
            0x3f800003, 0xbf800003)[phase]


def input_value(op, p, buf, i):
    j, k = i % 16, op % 3
    if p == 0:
        a = (0x3f800000, 0xbf800000, 0, 0x80000000,
             0x3f000000, 0xbf000000, 0x40000000, 0xc0000000)
        b = (0x40400000, 0xc0000000, 0x3f800000, 0xbf800000,
             0x3e800000, 0xbe800000, 0, 0x80000000)
        return (b if buf else a)[j % 8]
    if p == 1:
        mul = (0x3fc00000,0xbfc00000,0x3f800001,0xbf800001,
               0x3fa00000,0xbfa00000,0x3f800000,0xbf800000)
        add = (0x33800000,0xb3800000,0x33c00000,0xb3c00000,
               1,0x80000001,0x34000000,0xb4000000)
        scalar_b = (0x4b800000,0xcb800000,0x4c000000,0xcc000000,
                    0x4c800000,0xcc800000,0x33800000,0xb3800000)
        if k == 2:
            return mul[j % 8] if buf else 0x3f800003
        if op >= 3 and buf:
            return scalar_b[(j + (2 if k == 1 else 0)) % 8]
        if k == 1:
            return 0x3f800000 if buf else add[j % 8] ^ 0x80000000
        return add[j % 8] if buf else 0x3f800000
    if p == 2:
        small = (1,3,0x007fffff,0x00800000,0x00ffffff,0x00800001,0x01000001,0)
        return (small[j % 8] | ((j & 8) << 28)) if buf else (0x3f000000 if k == 2 else 1)
    if p == 3:
        return ((0x40000000 if k == 2 else 0x7f7fffff) | ((j & 2) << 30)
                if buf else 0x7f7fffff | ((j & 1) << 31))
    if p in (4, 5):
        values = ((0x7f800000,0xff800000,0,0x80000000,
                   0x7fc12345,0xffc12345,0x3f800000,0xbf800000) if p == 4 else
                  (0x7f800001,0xff800001,0x7fa12345,0xffa12345,
                   0x7fc12345,0xffc12345,0,0x3f800000))
        return values[(j + ((2 if p == 4 else 3) if buf else 0)) % 8]
    v = (j * 1664525 + 1013904223 + buf * 0x798523af + p * 0x13579bd) & 0xffffffff
    return v ^ (v >> 13)


class Binary32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        so = Path(cls.tmp.name) / "binary.so"
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined", "-I", str(ame.ROOT / "runtime"),
            str(ame.ROOT / "reference/binary32.c"),
            str(ame.ROOT / "reference/binary32-input.c"), "-o", str(so)], check=True)
        cls.lib = ctypes.CDLL(str(so))
        cls.ref = cls.lib.i4a_binary32
        cls.ref.argtypes = [ctypes.c_uint] * 4 + [ctypes.POINTER(ctypes.c_uint32)]
        cls.ref.restype = ctypes.c_uint

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def actual(self, op, rm, a, b):
        out = ctypes.c_uint32()
        flags = self.ref(op, rm, a, b, ctypes.byref(out))
        return out.value, flags

    def compare(self, op, a, b):
        for rm in range(5):
            self.assertEqual(self.actual(op,rm,a,b), expected(op,rm,a,b),
                             (op,rm,hex(a),hex(b)))

    def test_known_answers(self):
        vectors = (
            (0,0,0x3f800000,0x33800000,0x3f800000,1),
            (0,4,0x3f800000,0x33800000,0x3f800001,1),
            (1,0,0x3f800000,0x40400000,0x40000000,0),
            (2,0,1,0x3f000000,0,3),
            (2,3,1,0x3f000000,1,3),
            (2,0,0x00ffffff,0x3f000000,0x00800000,1),
            (2,0,0x7f7fffff,0x40000000,0x7f800000,5),
            (0,0,0x7f800000,0xff800000,0x7fc00000,16),
            (2,0,0x7f800000,0,0x7fc00000,16),
            (0,2,0x3f800000,0xbf800000,0x80000000,0),
            (0,0,0x3f800000,0xbf800000,0,0),
            (0,0,0x80000000,0x80000000,0x80000000,0),
            (0,0,0x7f800001,0x3f800000,0x7fc00000,16),
            (2,0,0x7fc12345,0x3f800000,0x7fc00000,0),
        )
        for op,rm,a,b,bits,flags in vectors:
            self.assertEqual(expected(op,rm,a,b), (bits,flags))
            self.assertEqual(self.actual(op,rm,a,b), (bits,flags))

    def test_rational_oracle(self):
        patterns = (0,0x80000000,1,3,0x80000001,0x007fffff,0x00800000,
                    0x00800001,0x3f800000,0xbf800000,0x3f800001,0x3f800003,
                    0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
                    0x7fc12345,0xffc12345,0x7f800001,0xff800001)
        for op in range(3):
            for a in patterns:
                for b in patterns:
                    self.compare(op,a,b)
        rng = random.Random(20260928)
        for _ in range(1200):
            self.compare(rng.randrange(3), rng.getrandbits(32), rng.getrandbits(32))

    def test_rounding_across_exponents(self):
        # Odd significands times 1.5 exercise half-way rounding across limb
        # boundaries, including the subnormal range.  Both signs matter for
        # directed rounding; these are host oracle checks, not target inputs.
        for power in range(-149,128):
            for tail in (1,3):
                a = ((power+127)<<23) | tail if power >= -126 else (1<<(power+149)) | tail
                for sign in (0,0x80000000):
                    self.compare(2,a | sign,0x3fc00000)
                if power >= -125:
                    half = ((power-24+127)<<23) if power-24 >= -126 else 1<<(power-24+149)
                    self.compare(0,a,half)
                    self.compare(1,half,a)

    def test_inputs_and_rounding_distinguishability(self):
        data, sc = self.lib.i4a_input, self.lib.i4a_scalar
        data.argtypes, data.restype = [ctypes.c_uint] * 4, ctypes.c_uint32
        sc.argtypes, sc.restype = [ctypes.c_uint] * 2, ctypes.c_uint32
        for op in range(6):
            signatures = [[] for _ in range(5)]
            for phase in range(8):
                self.assertEqual(sc(op,phase), scalar(op,phase))
                for i in range(64):
                    for buf in range(2):
                        self.assertEqual(data(op,phase,buf,i), input_value(op,phase,buf,i))
                    if i >= 16:
                        continue
                    a = scalar(op,phase) if op >= 3 else input_value(op,phase,0,i)
                    b = input_value(op,phase,1,i)
                    self.compare(op % 3,a,b)
                    for rm in range(5):
                        answer = expected(op % 3,rm,a,b)
                        signatures[rm].append(answer)
                        if phase == 0:
                            self.assertEqual(answer[1],0)
            self.assertEqual(len({tuple(s) for s in signatures}),5,op)




    def test_invalid_contract(self):
        for op,rm in ((3,0),(0,5),(0xffffffff,0),(0,0xffffffff)):
            out = ctypes.c_uint32(0xa5a5a5a5)
            self.assertEqual(self.ref(op,rm,0,0,ctypes.byref(out)),0xffffffff)
            self.assertEqual(out.value,0xa5a5a5a5)
