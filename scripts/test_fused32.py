"""Compare actual integer C fused reference with independent rational arithmetic."""
import ctypes
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
import ame
from test_sqrt32 import finite
from test_exponent32 import quantize

def expected(op,rm,a,b,d):
    # The algebra below uses the API operands directly, not a rounded mul/add.
    p,q,c=(d,b,a) if op>=3 else (a,b,d if op else 0)
    pm,qm,cm=[x&0x7fffffff for x in (p,q,c)]
    sp=(p>>31)^(q>>31)^int(op in (0,2,4))
    invalid=(pm==0 and qm==0x7f800000) or (qm==0 and pm==0x7f800000)
    pinf=not invalid and pm<=0x7f800000 and qm<=0x7f800000 and 0x7f800000 in (pm,qm)
    invalid |= bool(op and pinf and cm==0x7f800000 and sp!=c>>31)
    snan=any((v&0x7fffffff)>0x7f800000 and not v&0x400000 for v in (p,q,c))
    if invalid or any(v>0x7f800000 for v in (pm,qm,cm)):
        return 0x7fc00000,16 if invalid or snan else 0
    if pinf or cm==0x7f800000:
        return ((sp if pinf else c>>31)<<31)|0x7f800000,0
    val=lambda x:finite(x)*(-1 if x>>31 else 1)
    av,bv=val(a),val(b)
    if op==0: return quantize(-av*bv,rm,sp)
    dv=val(d)
    value={1:dv+av*bv,2:dv-av*bv,3:av+bv*dv,4:av-bv*dv}[op]
    return quantize(value,rm,sp if sp==c>>31 else int(rm==2))

class Fused32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        so=Path(cls.tmp.name)/'fused.so'
        subprocess.run(['cc','-O2','-shared','-fPIC','-fsanitize=undefined','-fno-sanitize-recover=undefined',
            '-I',str(ame.ROOT/'runtime'),str(ame.ROOT/'reference/fused32.c'),
            str(ame.ROOT/'reference/fused32-input.c'),'-o',str(so)],check=True)
        cls.lib=ctypes.CDLL(str(so))
        cls.ref=cls.lib.fused32
        cls.ref.argtypes=[ctypes.c_uint]*5+[ctypes.POINTER(ctypes.c_uint32)]
        cls.ref.restype=ctypes.c_uint
        cls.data=cls.lib.fused32_input
        cls.data.argtypes=[ctypes.c_uint]*4
        cls.data.restype=ctypes.c_uint32
        cls.scalar=cls.lib.fused32_scalar
        cls.scalar.argtypes=[ctypes.c_uint]*2
        cls.scalar.restype=ctypes.c_uint32

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def actual(self,op,rm,a,b,d):
        out=ctypes.c_uint32()
        flags=self.ref(op,rm,a,b,d,ctypes.byref(out))
        return out.value,flags

    def test_literals(self):
        for op,rm,a,b,d,want,flags in (
            (1,0,0x3f800001,0x3f7ffffe,0xbf800000,0xa8800000,0),
            (1,0,0x3f800003,0x3fc00000,1,0x3fc00005,1),
            (1,0,0x3f800003,0x3fc00000,0,0x3fc00004,1),
            (1,0,0x7f7fffff,0x40000000,0xff7fffff,0x7f7fffff,0),
            (1,0,1,0x3f000000,1,2,3),
            (0,2,0x3f800003,0x3fc00000,0x7f800001,0xbfc00005,1),
            (0,3,0x3f800003,0x3fc00000,0,0xbfc00004,1),
            (1,0,0,0x7f800000,0x7fc00000,0x7fc00000,16),
            (3,0,0x7fc00000,0x7f800000,0,0x7fc00000,16),
            (1,0,0x3f800000,0x3f800000,0x7fc00000,0x7fc00000,0),
            (1,0,0x3f800000,0x3f800000,0x7f800001,0x7fc00000,16),
            (2,0,0x7f800000,0x3f800000,0x7f800000,0x7fc00000,16),
            (1,2,0,0x3f800000,0x80000000,0x80000000,0),
            (0,0,0,0x3f800000,0,0x80000000,0)):
            self.assertEqual(expected(op,rm,a,b,d),(want,flags))
            self.assertEqual(self.actual(op,rm,a,b,d),(want,flags))
        for op,want in enumerate((0xc0400000,0x40a00000,0xbf800000,0x40e00000,0xc0a00000)):
            self.assertEqual(self.actual(op,0,0x3f800000,0x40400000,0x40000000),(want,0))

    def test_rational_oracle(self):
        rng=random.Random(800120260928)
        patterns=(0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x3f800000,0xbf800000,
                  0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0x7f800001)
        triples=[(a,b,d) for a in patterns for b in patterns for d in (0,0x80000000,0x3f800000,0x7fc00001,0x7f800000)]
        triples += [tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(500)]
        for a,b,d in triples:
            for op in range(5):
                for rm in range(5):
                    self.assertEqual(self.actual(op,rm,a,b,d),expected(op,rm,a,b,d),(op,rm,hex(a),hex(b),hex(d)))

    def test_input_patterns(self):
        for op in range(10):
            signatures=[[] for _ in range(5)]
            for phase in range(10):
                self.assertEqual(self.scalar(op,phase),self.data(op,phase,0,0))
                for i in range(16):
                    values=[self.data(op,phase,f,i) for f in range(3)]
                    self.assertEqual(values,[self.data(op,phase,f,i+16) for f in range(3)])
                    if op>=5: values[0]=self.scalar(op,phase)
                    for rm in range(5):
                        want=expected(op%5,rm,*values)
                        self.assertEqual(self.actual(op%5,rm,*values),want)
                        signatures[rm].append(want)
                        if phase in (0,8): self.assertEqual(want[1],0)
            self.assertEqual(len({tuple(x) for x in signatures}),5)

    def test_invalid_contract(self):
        for op,rm in ((5,0),(0,5)):
            out=ctypes.c_uint32(0xabcd)
            self.assertEqual(self.ref(op,rm,0,0,0,ctypes.byref(out)),0xffffffff)
            self.assertEqual(out.value,0xabcd)





if __name__=='__main__': unittest.main(verbosity=2)
