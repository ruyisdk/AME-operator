import ctypes as C
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"operators/reference"),str(ROOT/"scripts")]
import widen8_fixtures as fixtures
import operators
import audit_operator_widen8 as audit

class WidenFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={op:fixtures.create(dict(operator=op,contract="widen8-i32-v1"))
                  for op in ("gemm","gemv")}
    def test_shapes(self):
        for op,count in (("gemm",38),("gemv",30)):
            rows=self.rows[op]
            self.assertEqual(len(rows),6*2*count)
            for t in fixtures.TILES:
                for u in (0,1):
                    selected=[r for r in rows if r["tile"]==t and r["unsigned"]==u]
                    self.assertEqual(len(selected),count)
                    self.assertEqual({r["k"] for r in selected if r["purpose"].startswith("packet")},
                                     {4*t-1,4*t,4*t+1})
                    for r in selected:
                        self.assertTrue(all((0<=v<=255 if u else -128<=v<=127)
                                            for v in r["a"]+r["b"]))
            hands=[r for r in rows if r["purpose"] in ("high-product","high-mixed")]
            for r in hands:
                value=(65025 if r["purpose"]=="high-product" else 32640) if r["unsigned"] \
                      else (16384 if r["purpose"]=="high-product" else -16256)
                self.assertEqual(r["golden"][0],value+(12 if r["accumulate"] else 0))
    def test_independent_audit(self):
        for op in ("gemm","gemv"):
            for t in fixtures.TILES:
                cases=audit.selected(dict(row=dict(operator=op)),self.rows[op],t)
                self.assertEqual(len(cases),76 if op=="gemm" else 60)
    def test_missing_packet_boundary(self):
        rows=copy.deepcopy(self.rows["gemm"])
        rows=[r for r in rows if not (r["tile"]==4 and r["purpose"]=="packet-plus")]
        with self.assertRaisesRegex(ValueError,"coverage"):
            audit.selected(dict(row=dict(operator="gemm")),rows,4)
    def test_wrong_golden_and_byte_domain(self):
        rows=copy.deepcopy(self.rows["gemv"])
        rows[0]["golden"][0]+=1
        with self.assertRaisesRegex(ValueError,"GOLDEN"):
            audit.selected(dict(row=dict(operator="gemv")),rows,4)
        rows[0]["golden"][0]-=1;rows[0]["a"][0]=256
        with self.assertRaisesRegex(ValueError,"byte"):
            audit.selected(dict(row=dict(operator="gemv")),rows,4)
    def test_reproducible(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            manifest=ROOT/"operators/cases/gemm/widen8.json"
            meta=fixtures.generate(manifest,Path(a),dict(max_n=128))
            self.assertEqual(meta,fixtures.generate(manifest,Path(b),dict(max_n=128)))
            for file in meta["files"]:
                self.assertEqual((Path(a)/file).read_bytes(),(Path(b)/file).read_bytes())
    def test_registry(self):
        plan=operators.plan(operators.ame.config(ROOT/"configs/operators/integer-widen8-r5.json"))
        self.assertEqual(len(plan),16)
        for row in plan:
            self.assertIn("operators/kernels/gemm/i8.c",row["definition"]["typed_sources"])
            self.assertIn("operators/kernels/gemm/u8.c",row["definition"]["typed_sources"])

class WidenContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        so=Path(cls.tmp.name)/"widen.so"
        subprocess.run(["cc","-shared","-fPIC","-std=c11","-O2","-Wall","-Wextra","-Werror"]+
            [str(ROOT/p) for p in ("operators/common/validate_i8.c","operators/kernels/gemv/i8.c",
                "operators/tests/widen8_host.c","operators/tests/host_support.c")]+["-o",str(so)],
            check=True,timeout=60)
        cls.lib=C.CDLL(str(so))
        for prefix in ("i8","u8"):
            getattr(cls.lib,"ame_gemm_"+prefix+"_i32_owned").argtypes=[C.c_size_t]*3+[
                C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,
                C.c_int,C.c_void_p,C.c_size_t]
            getattr(cls.lib,"ame_gemv_"+prefix+"_i32_owned").argtypes=[
                C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,C.c_void_p,C.c_void_p,
                C.c_int,C.c_void_p,C.c_size_t]
        cls.lib.ame_ops_widen8_workspace_bytes.argtypes=[C.c_size_t]
        cls.lib.ame_ops_widen8_workspace_bytes.restype=C.c_size_t
        for name in ("test_pack_a8","test_pack_b8"):
            getattr(cls.lib,name).argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,
                                          C.c_size_t,C.c_size_t,C.c_size_t]
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def setUp(self):
        for name,value in (("test_owned",1),("test_supported",1),
                           ("widen_signed_calls",0),("widen_unsigned_calls",0)):
            C.c_int.in_dll(self.lib,name).value=value
        C.c_size_t.in_dll(self.lib,"test_tile").value=4
        self.a=(C.c_uint8*8)(0,128,127,99,255,2,88,0)
        self.b=(C.c_uint8*8)(0,255,128,77,1,127,66,0)
        self.c=(C.c_int32*6)(11,12,91,13,14,92)
        self.w=C.create_string_buffer(256);self.wp=(C.addressof(self.w)+15)&~15
    def invoke(self,u,**kw):
        p=dict(m=2,n=2,k=2,a=C.addressof(self.a)+1,lda=3,b=C.addressof(self.b)+1,ldb=3,
               c=self.c,ldc=3,add=0,w=self.wp,bytes=192)
        p.update(kw)
        return getattr(self.lib,"ame_gemm_"+("u8" if u else "i8")+"_i32_owned")(*p.values())
    def test_signed_and_unsigned_dispatch(self):
        self.assertEqual(self.invoke(0),0)
        self.assertEqual(list(self.c),[255,32513,91,3,382,92])
        self.assertEqual(self.invoke(1),0)
        self.assertEqual(list(self.c),[32767,32513,91,65027,32894,92])
        self.assertEqual(C.c_int.in_dll(self.lib,"widen_signed_calls").value,1)
        self.assertEqual(C.c_int.in_dll(self.lib,"widen_unsigned_calls").value,1)
    def test_accumulate(self):
        self.assertEqual(self.invoke(0,add=1),0)
        self.assertEqual(list(self.c),[266,32525,91,16,396,92])
    def test_empty_and_zero_k(self):
        C.c_int.in_dll(self.lib,"test_owned").value=0
        for u in (0,1):
            self.assertEqual(self.invoke(u,m=0,a=None,b=None,c=None,w=None),0)
            self.assertEqual(self.invoke(u,k=0,a=None,b=None,w=None),0)
        self.assertEqual(list(self.c),[0,0,91,0,0,92])
    def test_error_preservation(self):
        maximum=C.c_size_t(-1).value
        for u in (0,1):
            for changes,status in ((dict(bytes=191),2),(dict(w=self.wp+1),2),
                (dict(a=None),1),(dict(lda=1),1),(dict(ldb=1),1),(dict(ldc=1),1),
                (dict(c=C.addressof(self.c)+1),1),(dict(add=2),1),
                (dict(a=C.addressof(self.c)),1),(dict(b=C.addressof(self.c)),1),
                (dict(w=C.addressof(self.c)&~15),1),(dict(m=maximum),1),
                (dict(a=maximum),1),(dict(c=maximum-3),1)):
                self.assertEqual(self.invoke(u,**changes),status,changes)
                self.assertEqual(list(self.c),[11,12,91,13,14,92])
    def test_platform_checks(self):
        C.c_int.in_dll(self.lib,"test_owned").value=0
        for u in (0,1): self.assertEqual(self.invoke(u),3)
        C.c_int.in_dll(self.lib,"test_owned").value=1
        C.c_int.in_dll(self.lib,"test_supported").value=0
        for u in (0,1): self.assertEqual(self.invoke(u),4)
        C.c_int.in_dll(self.lib,"test_supported").value=1
        C.c_size_t.in_dll(self.lib,"test_tile").value=5
        for u in (0,1): self.assertEqual(self.invoke(u),4)
    def test_workspace_query(self):
        for t in fixtures.TILES:
            self.assertEqual(self.lib.ame_ops_widen8_workspace_bytes(t),12*t*t)
        for t in (0,3,5,256): self.assertEqual(self.lib.ame_ops_widen8_workspace_bytes(t),0)
    def test_gemv_dispatch(self):
        for prefix,want in (("i8",[-16128,-255]),("u8",[48896,65281])):
            f=getattr(self.lib,"ame_gemv_"+prefix+"_i32_owned")
            self.assertEqual(f(2,2,C.addressof(self.a)+1,3,C.addressof(self.b)+1,
                               self.c,0,self.wp,192),0)
            self.assertEqual(list(self.c)[:2],want)
    def test_square_major_pack(self):
        a=(C.c_uint8*18)(1,2,3,4,5,6,7,99,99,11,12,13,14,15,16,17,99,99)
        b=(C.c_uint8*28)(*[v for i in range(7) for v in (i+1,i+21,99,99)])
        dst=(C.c_uint8*64)()
        self.lib.test_pack_a8(dst,4,a,9,2,7)
        self.assertEqual(list(dst),[1,2,3,4,11,12,13,14]+[0]*8+
                         [5,6,7,0,15,16,17,0]+[0]*40)
        self.lib.test_pack_b8(dst,4,b,4,7,2)
        self.assertEqual(list(dst),[1,21,0,0,2,22,0,0,3,23,0,0,4,24,0,0,
                                    5,25,0,0,6,26,0,0,7,27,0,0]+[0]*36)

if __name__=="__main__": unittest.main()
