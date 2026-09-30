import copy
import ctypes as C
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'operators/reference')]
import fp32_fixtures as fixture
import audit_operator_fp32 as audit
import operators
from ieee_tininess import fp32_fused_expected as fma

class FP32Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={op:fixture.create(dict(operator=op,contract='fp32-exact-v1')) for op in ('gemm','gemv')}
    def test_inventory(self):
        for op in self.rows:
            for t in fixture.TILES:
                rows,count=audit.accounting(dict(row=dict(operator=op),config=dict(max_n=128)),self.rows[op],t)
                self.assertEqual(len(rows),44 if op=='gemm' else 34)
                self.assertGreater(count,0)
    def test_literal_numerical_witnesses(self):
        rows={r['purpose']:r for r in self.rows['gemm'] if r['tile']==4 and not r['accumulate']}
        self.assertEqual(rows['fraction-hand']['golden_units'][:2],[64,56])
        self.assertEqual(rows['exact-cancel']['golden'][0],0x3c800000)
        self.assertEqual(rows['limit-positive']['golden_units'][0],16769025)
        self.assertEqual(rows['limit-negative']['golden_units'][0],-16769025)
    def test_missing_shape_rejected(self):
        rows=[r for r in self.rows['gemm'] if not (r['tile']==4 and r['purpose']=='cross')]
        with self.assertRaisesRegex(ValueError,'coverage'):
            audit.accounting(dict(row=dict(operator='gemm'),config=dict(max_n=128)),rows,4)
    def test_corrupted_encoded_result_rejected(self):
        rows=copy.deepcopy(self.rows['gemv']);rows[0]['golden'][0]^=1
        with self.assertRaisesRegex(ValueError,'encoded'):
            audit.accounting(dict(row=dict(operator='gemv'),config=dict(max_n=128)),rows,4)
    def test_changed_special_inputs_rejected(self):
        rows=copy.deepcopy(self.rows['gemv'])
        next(r for r in rows if r['purpose']=='exact-cancel')['a_units'][0]=0
        with self.assertRaisesRegex(ValueError,'witness'):
            audit.accounting(dict(row=dict(operator='gemv'),config=dict(max_n=128)),rows,4)
    def test_reproducible_and_counts(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            m=ROOT/'operators/cases/gemv/fp32-exact.json'
            meta=fixture.generate(m,Path(a),dict(max_n=128))
            self.assertEqual(meta,fixture.generate(m,Path(b),dict(max_n=128)))
            for f in meta['files']:self.assertEqual((Path(a)/f).read_bytes(),(Path(b)/f).read_bytes())
            for t in fixture.TILES:
                _,count=audit.accounting(dict(row=dict(operator='gemv'),config=dict(max_n=128)),self.rows['gemv'],t)
                self.assertEqual(count,meta['checks_by_n'][str(t)])
    def test_registry(self):
        cfg=operators.ame.config(ROOT/'configs/operators/fp32-exact-r1.json')
        self.assertEqual(len(operators.plan(cfg)),16)

class FP32Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();so=Path(cls.tmp.name)/'fp32.so'
        subprocess.run(['cc','-shared','-fPIC','-O2','-std=c11','-Wall','-Wextra','-Werror']+
            [str(ROOT/p) for p in ('operators/common/validate_f32.c','operators/kernels/gemv/f32.c',
                'operators/tests/f32_host.c','operators/tests/host_support.c')]+['-o',str(so)],check=True,timeout=60)
        cls.lib=C.CDLL(str(so))
        cls.lib.ame_gemm_f32_owned.argtypes=[C.c_size_t]*3+[C.c_void_p,C.c_size_t,C.c_void_p,
            C.c_size_t,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p,C.c_size_t]
        cls.lib.ame_gemv_f32_owned.argtypes=[C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_size_t]
        cls.lib.ame_ops_f32_workspace_bytes.argtypes=[C.c_size_t]
        cls.lib.ame_ops_f32_workspace_bytes.restype=C.c_size_t
        for name in ('test_pack_f32','test_pack_f32_left','test_pack_f32_right','test_unpack_f32'):
            getattr(cls.lib,name).argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_size_t]
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        for name,value in (('test_owned',1),('test_supported',1),('f32_kernel_calls',0)):
            C.c_int.in_dll(self.lib,name).value=value
        C.c_size_t.in_dll(self.lib,'test_tile').value=4
        self.a=(C.c_float*6)(1,-.5,99,.25,.75,99)
        self.b=(C.c_float*6)(.5,1,99,-1,.25,99)
        self.c=(C.c_float*6)(5/64,-7/64,123,11/64,13/64,123)
        self.w=(C.c_float*64)();self.wp=(C.addressof(self.w)+15)&~15
    def call(self,**kw):
        p=dict(m=2,n=2,k=2,a=self.a,lda=3,b=self.b,ldb=3,c=self.c,ldc=3,add=0,w=self.wp,size=192)
        p.update(kw);return self.lib.ame_gemm_f32_owned(*p.values())
    def test_set_and_add(self):
        self.assertEqual(self.call(),0);self.assertEqual(list(self.c),[1,.875,123,-.625,.4375,123])
        self.assertEqual(self.call(add=1),0);self.assertEqual(list(self.c),[2,1.75,123,-1.25,.875,123])
    def test_gemv(self):
        x=(C.c_float*2)(.5,-1)
        self.assertEqual(self.lib.ame_gemv_f32_owned(2,2,self.a,3,x,self.c,0,self.wp,192),0)
        self.assertEqual(list(self.c)[:2],[1,-.625])
    def test_empty_and_k_zero_unowned(self):
        C.c_int.in_dll(self.lib,'test_owned').value=0
        self.assertEqual(self.call(m=0,a=None,b=None,c=None,w=None),0)
        self.assertEqual(self.call(k=0,a=None,b=None,w=None),0)
        self.assertEqual(list(self.c),[0,0,123,0,0,123])
        self.assertEqual(self.call(k=0,a=None,b=None,w=None,add=1),0)
        self.assertEqual(C.c_int.in_dll(self.lib,'f32_kernel_calls').value,0)
    def test_errors_leave_output_bytes(self):
        before=bytes(self.c);maximum=C.c_size_t(-1).value
        for changes,status in ((dict(a=None),1),(dict(b=None),1),(dict(c=None),1),
            (dict(a=C.addressof(self.a)+1),1),(dict(b=C.addressof(self.b)+1),1),
            (dict(c=C.addressof(self.c)+1),1),(dict(lda=1),1),(dict(ldb=1),1),(dict(ldc=1),1),
            (dict(add=2),1),(dict(m=0,add=2),1),(dict(a=self.c),1),(dict(b=self.c),1),
            (dict(w=None),2),(dict(w=self.wp+1),2),(dict(size=191),2),
            (dict(w=C.addressof(self.c)&~15),1),(dict(m=maximum),1),
            (dict(a=maximum-3),1),(dict(b=maximum-3),1),(dict(c=maximum-3),1),
            (dict(w=maximum-15),1)):
            self.assertEqual(self.call(**changes),status,changes);self.assertEqual(bytes(self.c),before)
        self.assertEqual(C.c_int.in_dll(self.lib,'f32_kernel_calls').value,0)
    def test_ownership_profile_and_size(self):
        C.c_int.in_dll(self.lib,'test_owned').value=0
        self.assertEqual(self.call(),3)
        C.c_int.in_dll(self.lib,'test_owned').value=1;C.c_int.in_dll(self.lib,'test_supported').value=0
        self.assertEqual(self.call(),4)
        C.c_int.in_dll(self.lib,'test_supported').value=1;C.c_size_t.in_dll(self.lib,'test_tile').value=5
        self.assertEqual(self.call(),4)
        for t in fixture.TILES:self.assertEqual(self.lib.ame_ops_f32_workspace_bytes(t),12*t*t)
        for t in (0,3,5,256):self.assertEqual(self.lib.ame_ops_f32_workspace_bytes(t),0)
    def test_pack_and_unpack(self):
        tile=(C.c_float*16)();out=(C.c_float*6)(11,12,99,13,14,99)
        self.lib.test_pack_f32(tile,4,self.a,3,2,2)
        self.assertEqual(list(tile),[1,-.5,-.5,-.5]+[.25,.75,.75,.75]*3)
        self.lib.test_unpack_f32(out,3,tile,4,2,2)
        self.assertEqual(list(out),[1,-.5,99,.25,.75,99])
    def test_pack_tail_roles_preserve_bits(self):
        patterns=(0x80000000,0,0x7f800000,0xff800000,0x7fc00001,0x7f800001)
        bits=[patterns[(i*3+j)%len(patterns)] for i in range(4) for j in range(6)]
        src=(C.c_float*24)();C.memmove(src,struct.pack('=24I',*bits),96)
        for rows in range(1,5):
            for cols in range(1,5):
                for role,fn in (('seed','test_pack_f32'),('left','test_pack_f32_left'),
                                ('right','test_pack_f32_right')):
                    tile=(C.c_float*16)()
                    getattr(self.lib,fn)(tile,4,src,6,rows,cols)
                    expected=[]
                    for i in range(4):
                        for j in range(4):
                            if role=='left' and j>=cols:expected.append(0)
                            elif role=='right' and i>=rows:expected.append(0x80000000)
                            else:expected.append(bits[min(i,rows-1)*6+min(j,cols-1)])
                    self.assertEqual(struct.unpack('=16I',bytes(tile)),tuple(expected),(rows,cols,role))

    def test_rne_packed_tiles_match_logical_fma(self):
        patterns=(0,0x80000000,1,0x007fffff,0x00800000,0x3f800000,0xbf800000,
                  0x7f7fffff,0x7f800000,0xff800000,0x7fc00001,0x7f800001)
        def packed(bits,rows,cols,fn):
            src=(C.c_float*len(bits))();C.memmove(src,struct.pack('='+str(len(bits))+'I',*bits),4*len(bits))
            dst=(C.c_float*16)();getattr(self.lib,fn)(dst,4,src,5,rows,cols)
            return struct.unpack('=16I',bytes(dst))
        def dot(a,b,seed):
            flags=0
            for av,bv in zip(a,b):
                seed,f=fma(1,0,av,bv,seed);flags|=f
            return seed,flags
        for m,n,k in ((1,1,1),(2,3,2),(3,2,3),(4,4,4)):
            for salt in range(len(patterns)):
                a=[patterns[(i*7+j+salt)%len(patterns)] for i in range(m) for j in range(5)]
                b=[patterns[(i*5+j+salt*3)%len(patterns)] for i in range(k) for j in range(5)]
                c=[patterns[(i*3+j+salt*5)%len(patterns)] for i in range(m) for j in range(5)]
                ap=packed(a,m,k,'test_pack_f32_left')
                bp=packed(b,k,n,'test_pack_f32_right')
                cp=packed(c,m,n,'test_pack_f32')
                logical={(i,j):dot(a[i*5:i*5+k],[b[p*5+j] for p in range(k)],c[i*5+j])
                         for i in range(m) for j in range(n)}
                for i in range(4):
                    for j in range(4):
                        value=dot(ap[i*4:i*4+4],[bp[p*4+j] for p in range(4)],cp[i*4+j])
                        self.assertEqual(value,logical[min(i,m-1),min(j,n-1)],(m,n,k,salt,i,j))

if __name__=='__main__':unittest.main()
