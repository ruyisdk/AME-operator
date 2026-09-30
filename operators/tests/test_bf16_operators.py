import copy
import ctypes as C
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'operators/reference')]
import bf16_fixtures as fixture
import audit_operator_bf16 as audit
import operators

class BF16Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={op:fixture.create(dict(operator=op,contract='bf16-f32-exact-v1')) for op in ('gemm','gemv')}
    def test_inventory_and_packet_boundaries(self):
        for op in self.rows:
            for t in fixture.TILES:
                rows,count=audit.accounting(dict(row=dict(operator=op),config=dict(max_n=128)),self.rows[op],t)
                self.assertEqual(len(rows),54 if op=='gemm' else 44)
                self.assertGreater(count,0)
    def test_literal_witnesses(self):
        rows={r['purpose']:r for r in self.rows['gemm'] if r['tile']==4 and not r['accumulate']}
        self.assertEqual(rows['fraction-hand']['golden_units'][:2],[64,56])
        self.assertEqual(rows['fraction-hand']['a'][:2],[0x3f80,0xbf00])
        self.assertEqual(rows['exact-cancel']['golden'][0],0x3c800000)
        self.assertEqual(rows['limit-positive']['golden_units'][0],65025)
        self.assertEqual(rows['limit-negative']['golden_units'][0],-65025)
    def test_auditor_negative_controls(self):
        base=self.rows['gemv'];build=dict(row=dict(operator='gemv'),config=dict(max_n=128))
        for key in ('a','golden','golden_units'):
            rows=copy.deepcopy(base);rows[0][key][0]^=1
            with self.assertRaises(ValueError):audit.accounting(build,rows,4)
        with self.assertRaisesRegex(ValueError,'coverage'):audit.accounting(build,base[1:],4)
        with self.assertRaisesRegex(ValueError,'coverage'):audit.accounting(build,base+[base[0]],4)
        rows=copy.deepcopy(base)
        next(r for r in rows if r['purpose']=='exact-cancel')['a_units'][0]=0
        with self.assertRaisesRegex(ValueError,'witness'):audit.accounting(build,rows,4)
    def test_reproducible_counts(self):
        for op in self.rows:
            with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
                manifest=ROOT/f'operators/cases/{op}/bf16-exact.json'
                meta=fixture.generate(manifest,Path(a),dict(max_n=128))
                self.assertEqual(meta,fixture.generate(manifest,Path(b),dict(max_n=128)))
                for f in meta['files']:self.assertEqual((Path(a)/f).read_bytes(),(Path(b)/f).read_bytes())
                for t in fixture.TILES:
                    _,count=audit.accounting(dict(row=dict(operator=op),config=dict(max_n=128)),self.rows[op],t)
                    self.assertEqual(count,meta['checks_by_n'][str(t)])
    def test_registry(self):
        cfg=operators.ame.config(ROOT/'configs/operators/bf16-exact-r5.json')
        self.assertEqual(len(operators.plan(cfg)),16)

class Workspace(C.Structure):
    _fields_=[('a',C.c_void_p),('a_elements',C.c_size_t),('b',C.c_void_p),
              ('b_elements',C.c_size_t),('c',C.c_void_p),('c_elements',C.c_size_t)]

class BF16RawObservations(unittest.TestCase):
    def check(self,change):
        build=dict(row=dict(xlen=32),symbols=dict(_start=0x80000000,ame_test_done=0x80000040))
        summary='AME N=4 NELEM=16 UDSZ=32 MREGS=16 ACCREGS=4 owned=0 fflags=0x0 xsat=0 status=0x0'
        log='\n'.join(['0x80000000','M','0x40001100','0x06000000','0x0',summary,
                       '0x80000040',summary]+['0x0']*32)+'\n'
        response=dict(observed=dict(xlen=32,n=4,uds=32,mregs=16,accregs=4,
            csr_map=operators.ame.CSR_MAP,entry_contract=operators.ame.ENTRY),
            stop_pc=0x80000040,mailbox_hex='00'*128)
        command=dict(argv=['--isa=rv32im_zicsr_ztt'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'spike.log').write_text(log)
            (p/'spike-command.json').write_text(json.dumps(command))
            audit.verify_execution(build,p,response)
            if change=='mailbox':response['mailbox_hex']='01'+'00'*127
            elif change=='isa':
                (p/'spike-command.json').write_text(json.dumps(dict(argv=['--isa=rv64im_zicsr_ztt'])))
            with self.assertRaises(ValueError):audit.verify_execution(build,p,response)
    def test_raw_mailbox_disagreement(self):self.check('mailbox')
    def test_wrong_execution_isa(self):self.check('isa')

class BF16Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();so=Path(cls.tmp.name)/'bf16.so'
        subprocess.run(['cc','-shared','-fPIC','-O2','-std=c11','-Wall','-Wextra','-Werror',
                       '-include',str(ROOT/'operators/tests/bf16_host_storage.h')]+
            [str(ROOT/p) for p in ('operators/common/validate_bf16.c','operators/kernels/gemv/bf16.c',
                'operators/tests/bf16_host.c','operators/tests/host_support.c')]+['-o',str(so)],check=True,timeout=60)
        cls.lib=C.CDLL(str(so))
        cls.lib.ame_gemm_bf16_f32_owned.argtypes=[C.c_size_t]*3+[C.c_void_p,C.c_size_t,C.c_void_p,
            C.c_size_t,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p]
        cls.lib.ame_gemv_bf16_f32_owned.argtypes=[C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,
            C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
        for name in ('ame_ops_bf16_input_elements','ame_ops_bf16_output_elements'):
            getattr(cls.lib,name).argtypes=[C.c_size_t];getattr(cls.lib,name).restype=C.c_size_t
        for name in ('test_pack_bf16_a','test_pack_bf16_b'):
            getattr(cls.lib,name).argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_size_t]
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        for name,value in (('test_owned',1),('test_supported',1),('bf16_kernel_calls',0)):
            C.c_int.in_dll(self.lib,name).value=value
        C.c_size_t.in_dll(self.lib,'test_tile').value=4
        self.a=(C.c_uint16*6)(0x3f80,0xbf00,0x7f81,0x3e80,0x3f40,0x7f81)
        self.b=(C.c_uint16*6)(0x3f00,0x3f80,0x7f81,0xbf80,0x3e80,0x7f81)
        self.c=(C.c_float*6)(5/64,-7/64,123,11/64,13/64,123)
        self.buffers=[C.create_string_buffer(80) for _ in range(3)]
        self.pointers=[(C.addressof(b)+15)&~15 for b in self.buffers]
        self.w=Workspace(self.pointers[0],32,self.pointers[1],32,self.pointers[2],16)
    def call(self,**kw):
        p=dict(m=2,n=2,k=2,a=self.a,lda=3,b=self.b,ldb=3,c=self.c,ldc=3,add=0,w=C.byref(self.w))
        p.update(kw);return self.lib.ame_gemm_bf16_f32_owned(*p.values())
    def test_set_and_add(self):
        descriptor=bytes(self.w);source=(bytes(self.a),bytes(self.b))
        self.assertEqual(self.call(),0);self.assertEqual(list(self.c),[1,.875,123,-.625,.4375,123])
        self.assertEqual(self.call(add=1),0);self.assertEqual(list(self.c),[2,1.75,123,-1.25,.875,123])
        self.assertEqual(bytes(self.w),descriptor);self.assertEqual((bytes(self.a),bytes(self.b)),source)
    def test_gemv(self):
        x=(C.c_uint16*2)(0x3f00,0xbf80)
        self.assertEqual(self.lib.ame_gemv_bf16_f32_owned(2,2,self.a,3,x,self.c,0,C.byref(self.w)),0)
        self.assertEqual(list(self.c)[:2],[1,-.625])
    def test_empty_and_k_zero_unowned(self):
        C.c_int.in_dll(self.lib,'test_owned').value=0
        self.assertEqual(self.call(m=0,a=None,b=None,c=None,w=None),0)
        self.assertEqual(self.call(n=0,a=None,b=None,c=None,w=None),0)
        self.assertEqual(self.call(k=0,a=None,b=None,w=None,add=1),0)
        self.assertEqual(list(self.c),[5/64,-7/64,123,11/64,13/64,123])
        self.assertEqual(self.call(k=0,a=None,b=None,w=None),0)
        self.assertEqual(list(self.c),[0,0,123,0,0,123])
        self.assertEqual(C.c_int.in_dll(self.lib,'bf16_kernel_calls').value,0)
    def test_invalid_matrix_arguments_have_no_effect(self):
        before=(bytes(self.c),bytes(self.w));maximum=C.c_size_t(-1).value
        for changes in (dict(a=None),dict(b=None),dict(c=None),dict(a=C.addressof(self.a)+1),
            dict(b=C.addressof(self.b)+1),dict(c=C.addressof(self.c)+1),dict(lda=1),dict(ldb=1),dict(ldc=1),
            dict(add=2),dict(m=0,add=2),dict(a=self.c),dict(b=self.c),dict(m=maximum),
            dict(a=maximum-1),dict(b=maximum-1),dict(c=maximum-3),dict(w=maximum-7),
            dict(w=C.addressof(self.w)+1),dict(w=self.c)):
            self.assertEqual(self.call(**changes),1,changes)
            self.assertEqual((bytes(self.c),bytes(self.w)),before)
        self.assertEqual(C.c_int.in_dll(self.lib,'bf16_kernel_calls').value,0)
    def test_workspace_requirements(self):
        self.assertEqual(self.call(w=None),2);original=bytes(self.w);before=bytes(self.c)
        for field in ('a','b','c'):
            for value in (None,getattr(self.w,field)+1):
                setattr(self.w,field,value);self.assertEqual(self.call(),2)
                C.memmove(C.byref(self.w),original,len(original))
            size=field+'_elements';setattr(self.w,size,getattr(self.w,size)-1)
            self.assertEqual(self.call(),2);C.memmove(C.byref(self.w),original,len(original))
        self.assertEqual(bytes(self.c),before)
        self.assertEqual(C.c_int.in_dll(self.lib,'bf16_kernel_calls').value,0)
    def test_workspace_overlap_and_wrap(self):
        original=bytes(self.w);before=bytes(self.c);maximum=C.c_size_t(-1).value
        for field in ('a','b','c'):
            targets=[C.addressof(self.c)&~15,C.addressof(self.a)&~15,C.addressof(self.b)&~15,
                     C.addressof(self.w)&~15,maximum-15]
            targets += [getattr(self.w,other) for other in ('a','b','c') if other!=field]
            for target in targets:
                setattr(self.w,field,target);self.assertEqual(self.call(),1,(field,target))
                C.memmove(C.byref(self.w),original,len(original))
        self.assertEqual(bytes(self.c),before)
        self.assertEqual(C.c_int.in_dll(self.lib,'bf16_kernel_calls').value,0)
    def test_ownership_profile_and_queries(self):
        C.c_int.in_dll(self.lib,'test_owned').value=0;self.assertEqual(self.call(),3)
        C.c_int.in_dll(self.lib,'test_owned').value=1;C.c_int.in_dll(self.lib,'test_supported').value=0
        self.assertEqual(self.call(),4)
        C.c_int.in_dll(self.lib,'test_supported').value=1;C.c_size_t.in_dll(self.lib,'test_tile').value=5
        self.assertEqual(self.call(),4)
        for t in fixture.TILES:
            self.assertEqual(self.lib.ame_ops_bf16_input_elements(t),2*t*t)
            self.assertEqual(self.lib.ame_ops_bf16_output_elements(t),t*t)
        for t in (0,3,5,256,C.c_size_t(-1).value):
            self.assertEqual(self.lib.ame_ops_bf16_input_elements(t),0)
            self.assertEqual(self.lib.ame_ops_bf16_output_elements(t),0)
    def test_pack_both_square_halves_and_tail_bits(self):
        patterns=(0x8000,0,0x7f80,0xff80,0x7fc1,0x7f81,0x0001,0x3f80)
        bits=[patterns[(i*3+j)%len(patterns)] for i in range(8) for j in range(10)]
        src=(C.c_uint16*80)(*bits);before=bytes(src)
        for role in ('a','b'):
            for rows in range(1,5 if role=='a' else 9):
                for cols in range(1,9 if role=='a' else 5):
                    tile=(C.c_uint16*32)()
                    getattr(self.lib,'test_pack_bf16_'+role)(tile,4,src,10,rows,cols)
                    want=[]
                    for q in range(2):
                        for i in range(4):
                            for j in range(4):
                                r,c=(i,4*q+j) if role=='a' else (4*q+i,j)
                                if role=='a' and c>=cols:want.append(0)
                                elif role=='b' and r>=rows:want.append(0x8000)
                                else:want.append(bits[min(r,rows-1)*10+min(c,cols-1)])
                    self.assertEqual(list(tile),want,(role,rows,cols))
        self.assertEqual(bytes(src),before)

if __name__=='__main__':unittest.main()
