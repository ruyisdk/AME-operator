import copy
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"operators/reference"), str(ROOT/"scripts")]
import integer
import matrix_fixtures
import operators

class Golden(unittest.TestCase):
    def test_rectangular(self):
        self.assertEqual(integer.gemm(2,2,3,[1,-2,3,4,0,-1],3,
                         [2,1,-1,3,5,-2],2,[7,-2,8,1],2,0),[19,-11,3,6])
    def test_add(self):
        self.assertEqual(integer.gemm(2,2,3,[1,-2,3,4,0,-1],3,
                         [2,1,-1,3,5,-2],2,[7,-2,8,1],2,1),[26,-13,11,7])
    def test_gemv(self):
        self.assertEqual(integer.gemv(2,3,[1,-2,3,4,0,-1],3,[2,-1,5],[7,8],1),[26,11])
    def test_padding(self):
        self.assertEqual(integer.gemm(1,1,2,[2,3,999],3,[4,888,5,777],2,
                         [99,123],2,0),[23,123])
    def test_zero_k(self):
        self.assertEqual(integer.gemm(2,2,0,[],0,[],2,[1,2,3,4],2,0),[0]*4)
        self.assertEqual(integer.gemv(2,0,[],0,[],[1,2],1),[1,2])
    def test_short(self):
        with self.assertRaises(ValueError): integer.gemv(2,3,[1],3,[1]*3,[0]*2,0)
    def test_bad_stride(self):
        with self.assertRaises(ValueError): integer.gemm(1,2,1,[1],1,[2,3],1,[0,0],2,0)
    def test_overflow(self):
        with self.assertRaises(ValueError): integer.gemv(1,1,[1<<30],1,[2],[0],0)
    def test_cancellation_not_sufficient(self):
        with self.assertRaises(ValueError):
            integer.gemv(1,2,[1<<30,1<<30],2,[2,-2],[0],0)
    def test_non_integer(self):
        with self.assertRaises(ValueError): integer.gemv(1,1,[1.0],1,[1],[0],0)
    def test_fixture_counts(self):
        for op,count in (("gemm",18),("gemv",16)):
            spec=json.loads((ROOT/f"operators/cases/{op}/i32.json").read_text())
            self.assertEqual(len(matrix_fixtures.create(spec)),count)
    def test_generated_reproducible(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            path=ROOT/"operators/cases/gemm/i32.json"
            m=matrix_fixtures.generate(path,Path(a),{"max_n":128})
            self.assertEqual(m,matrix_fixtures.generate(path,Path(b),{"max_n":128}))
            for f in m["files"]: self.assertEqual((Path(a)/f).read_bytes(),(Path(b)/f).read_bytes())
            self.assertGreater(m["checks_by_n"]["4"],m["checks_by_n"]["128"])
    def test_registry(self):
        c=operators.ame.config(ROOT/"configs/operators/integer-r1.json")
        self.assertEqual(len(operators.plan(c)),8)
    def test_registry_bad_selection(self):
        c=operators.ame.config(ROOT/"configs/operators/integer-r1.json")
        c["operators"]=["not-implemented"]
        with self.assertRaises(ValueError): operators.plan(c)
    def test_registry_bad_profile(self):
        c=operators.ame.config(ROOT/"configs/operators/integer-r1.json")
        c["profiles"][0]["uds"]=128
        with self.assertRaises(ValueError): operators.plan(c)

class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        so=Path(cls.tmp.name)/"contract.so"
        subprocess.run(["cc","-shared","-fPIC","-std=c11","-O2","-Wall","-Wextra","-Werror",
            str(ROOT/"operators/common/validate.c"),str(ROOT/"operators/kernels/gemv/i32.c"),
            str(ROOT/"operators/tests/host_support.c"),"-o",str(so)],check=True,timeout=60)
        cls.lib=C.CDLL(str(so))
        cls.lib.ame_gemm_i32_owned.argtypes=[C.c_size_t]*3+[C.c_void_p,C.c_size_t,
            C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p,C.c_size_t]
        cls.lib.ame_gemv_i32_owned.argtypes=[C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_size_t]
        cls.lib.ame_ops_i32_workspace_bytes.argtypes=[C.c_size_t]
        cls.lib.ame_ops_i32_workspace_bytes.restype=C.c_size_t
        cls.lib.test_pack.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_size_t]
        cls.lib.test_unpack.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_size_t]
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def setUp(self):
        C.c_int.in_dll(self.lib,"test_owned").value=1
        C.c_int.in_dll(self.lib,"test_supported").value=1
        C.c_int.in_dll(self.lib,"kernel_calls").value=0
        C.c_size_t.in_dll(self.lib,"test_tile").value=4
        self.a=(C.c_int32*4)(2,3,4,5); self.b=(C.c_int32*4)(6,7,8,9)
        self.c=(C.c_int32*4)(31,32,33,34)
        self.ws=C.create_string_buffer(256)
        self.wp=(C.addressof(self.ws)+15)&~15
    def invoke(self,**kw):
        v=dict(m=2,n=2,k=2,a=self.a,lda=2,b=self.b,ldb=2,c=self.c,ldc=2,
               accumulate=0,workspace=self.wp,workspace_bytes=192)
        v.update(kw)
        return self.lib.ame_gemm_i32_owned(*v.values())
    def test_valid(self):
        self.assertEqual(self.invoke(),0); self.assertEqual(list(self.c),[36,41,64,73])
    def test_accumulate(self):
        self.assertEqual(self.invoke(accumulate=1),0); self.assertEqual(list(self.c),[67,73,97,107])
    def test_zero_dimensions(self):
        self.assertEqual(self.invoke(m=0,a=None,b=None,c=None,workspace=None),0)
        self.assertEqual(C.c_int.in_dll(self.lib,"kernel_calls").value,0)
    def test_zero_k(self):
        C.c_int.in_dll(self.lib,"test_owned").value=0
        self.assertEqual(self.invoke(k=0,a=None,b=None,workspace=None),0)
        self.assertEqual(list(self.c),[0]*4)
    def test_short_workspace(self):
        self.assertEqual(self.invoke(workspace_bytes=191),2); self.assertEqual(list(self.c),[31,32,33,34])
    def test_misaligned_workspace(self): self.assertEqual(self.invoke(workspace=self.wp+4),2)
    def test_bad_mode(self): self.assertEqual(self.invoke(accumulate=2),1)
    def test_bad_stride(self): self.assertEqual(self.invoke(lda=1),1)
    def test_alias_output(self): self.assertEqual(self.invoke(c=self.a),1)
    def test_workspace_alias(self): self.assertEqual(self.invoke(workspace=C.addressof(self.c)&~15),1)
    def test_not_owned(self):
        C.c_int.in_dll(self.lib,"test_owned").value=0
        self.assertEqual(self.invoke(),3); self.assertEqual(list(self.c),[31,32,33,34])
    def test_unsupported(self):
        C.c_int.in_dll(self.lib,"test_supported").value=0
        self.assertEqual(self.invoke(),4)
    def test_span_overflow(self): self.assertEqual(self.invoke(m=C.c_size_t(-1).value),1)
    def test_address_overflow(self):
        self.assertEqual(self.invoke(a=C.c_size_t(-4).value),1)
    def test_workspace_size(self):
        self.assertEqual(self.lib.ame_ops_i32_workspace_bytes(128),196608)
        for n in (0,3,5,256): self.assertEqual(self.lib.ame_ops_i32_workspace_bytes(n),0)
    def test_pack_unpack(self):
        src=(C.c_int32*6)(1,2,99,3,4,88); packed=(C.c_int32*16)()
        self.lib.test_pack(packed,4,src,3,2,2)
        self.assertEqual(list(packed),[1,2,0,0,3,4,0,0]+[0]*8)
        out=(C.c_int32*6)(*[77]*6)
        self.lib.test_unpack(out,3,packed,4,2,2)
        self.assertEqual(list(out),[1,2,77,3,4,77])
    def test_gemv(self):
        self.assertEqual(self.lib.ame_gemv_i32_owned(2,2,self.a,2,self.b,self.c,0,self.wp,192),0)
        self.assertEqual(list(self.c),[33,59,33,34])

if __name__=="__main__": unittest.main()
