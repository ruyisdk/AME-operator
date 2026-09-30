import copy
import ctypes as C
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'operators/reference'),str(ROOT/'scripts')]
import fp32_fold as ref
import fp32_numeric_fixtures as fixture
import audit_operator_fp32_numeric as audit
from ieee_tininess import fp32_fused_expected,fp32_binary_expected
import operators

class NumericOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();out=Path(cls.tmp.name)/'fma.so'
        subprocess.run(['cc','-O2','-shared','-fPIC','-fsanitize=undefined','-fno-sanitize-recover=undefined',
            '-I',str(ROOT/'runtime'),str(ROOT/'reference/fused32-r2.c'),'-o',str(out)],check=True,timeout=60)
        cls.lib=C.CDLL(str(out));cls.fn=cls.lib.fused32
        cls.fn.argtypes=[C.c_uint]*5+[C.POINTER(C.c_uint32)];cls.fn.restype=C.c_uint
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def integer_dot(self,a,b,seed):
        flags=0
        for x,y in zip(a,b):
            value=C.c_uint32();flags|=self.fn(1,0,x,y,seed,C.byref(value));seed=value.value
        return seed,flags
    def test_literal_bits_and_flags(self):
        want={'fused-cancel':(0xa8800000,0),'fused-overflow-rescue':(0x7f7fffff,0),
            'tie-even':(0x3f800000,1),'tie-odd':(0x3f800002,1),'negative-tie':(0xbf800002,1),
            'sticky-beyond-half':(0x3fc00005,1),'overflow-positive':(0x7f800000,5),
            'overflow-negative':(0xff800000,5),'underflow-tie':(0,3),'negative-underflow':(0x80000000,3),
            'exact-subnormal':(0x00400000,0),'tiny-normal-delivery':(0x00800000,3),
            'precision-carry-normal':(0x00800000,1),'nonzero-seed-tiny':(0x00800000,1),
            'subnormal-plus-half':(2,3),'infinity-right':(0x7f800000,0),'infinity-left':(0xff800000,0),
            'infinity-invalid-zero':(0x7fc00000,16),'opposed-infinities':(0x7fc00000,16),
            'quiet-nan-left':(0x7fc00000,0),'signaling-nan-left':(0x7fc00000,16),
            'signaling-nan-right':(0x7fc00000,16),'quiet-nan-seed':(0x7fc00000,0),
            'signaling-nan-seed':(0x7fc00000,16),'beta-zero-ignores-snan':(0x3f800000,0),
            'negative-zero-seed':(0x80000000,0),'opposed-zero-signs':(0,0),
            'seed-before-fold':(0,1),'order-sensitive':(0,1),'intermediate-flags':(0x3f800000,3),
            'overflow-then-cancel':(0x7f800000,5)}
        data=fixture.witnesses();self.assertEqual(set(data),set(want))
        for name,v in data.items():
            args=(v['a'],v['b'],v['seed'] if v['add'] else 0)
            self.assertEqual(ref.dot(*args),want[name],name)
            self.assertEqual(self.integer_dot(*args),want[name],name)
    def test_random_fold_independent_integer_oracle(self):
        rng=random.Random(2026093005)
        for _ in range(256):
            k=rng.randrange(1,9);a=[rng.getrandbits(32) for _ in range(k)];b=[rng.getrandbits(32) for _ in range(k)]
            seed=rng.getrandbits(32)
            self.assertEqual(ref.dot(a,b,seed),self.integer_dot(a,b,seed))
    def test_split_fma_control(self):
        a,b,c=0x3f800001,0x3f7ffffe,0xbf800000
        product,f1=fp32_binary_expected(2,0,a,b);value,f2=fp32_binary_expected(0,0,product,c)
        self.assertEqual((value,f1|f2),(0,1));self.assertEqual(ref.fma(a,b,c),(0xa8800000,0))
    def test_wrong_rm_control(self):
        self.assertEqual(ref.fma(0x3f800000,0x33800000,0x3f800001),(0x3f800002,1))
        self.assertEqual(fp32_fused_expected(1,1,0x3f800000,0x33800000,0x3f800001),(0x3f800001,1))
    def test_seed_after_control(self):
        a,b=[0x4b800000,0xcb800000],[0x3f800000]*2
        v,f=ref.dot(a,b,0);value,added=fp32_binary_expected(0,0,v,0x3f800000)
        self.assertEqual((value,f|added),(0x3f800000,0))
        self.assertEqual(ref.dot(a,b,0x3f800000),(0,1))
    def test_order_control(self):
        self.assertEqual(ref.dot([0x4b800000,0x3f800000,0xcb800000],[0x3f800000]*3,0),(0,1))
        self.assertEqual(ref.dot([0x4b800000,0xcb800000,0x3f800000],[0x3f800000]*3,0),(0x3f800000,0))
    def test_empty_preserves_noncomputing_seed(self):
        for seed in (0x80000000,0x7f800001,0x7fc12345):self.assertEqual(ref.dot([],[],seed),(seed,0))
    def test_invalid_contract(self):
        with self.assertRaises(ValueError):ref.dot([0],[],0)
        with self.assertRaises(ValueError):ref.fma(-1,0,0)
        with self.assertRaises(ValueError):ref.gemm(1,1,2,[0],2,[0,0],1,[0],1,0)
        with self.assertRaises(ValueError):ref.gemm(1,1,1,[0],1,[0],1,[0],1,2)

class NumericFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows={op:fixture.create(dict(operator=op,contract='fp32-rne-fold-v1')) for op in ('gemm','gemv')}
    def build(self,op):return dict(row=dict(operator=op),config=dict(max_n=128))
    def test_inventory(self):
        for op in self.rows:
            for t in fixture.TILES:
                rows,count=audit.accounting(self.build(op),self.rows[op],t)
                self.assertEqual(len(rows),45 if op=='gemm' else 43);self.assertGreater(count,0)
    def test_cross_block_witnesses(self):
        for rows in self.rows.values():
            for t in fixture.TILES:
                base={r['purpose']:r for r in rows if r['tile']==t}
                for blocks in (1,2):
                    for name in ('fused-cancel','seed-before-fold','order-sensitive','intermediate-flags','negative-zero-seed'):
                        r=base[f'cross-{blocks}-{name}']
                        self.assertEqual((r['golden'][0],r['new_flags']),(base[name]['golden'][0],base[name]['new_flags']))
                        self.assertEqual(r['k'],blocks*t-1+base[name]['k'])
    def test_missing_witness_rejected(self):
        rows=copy.deepcopy(self.rows['gemm']);rows.pop(0)
        with self.assertRaisesRegex(ValueError,'coverage'):audit.accounting(self.build('gemm'),rows,4)
    def test_changed_operand_rejected(self):
        rows=copy.deepcopy(self.rows['gemv']);rows[0]['a'][0]^=1
        with self.assertRaisesRegex(ValueError,'input'):audit.accounting(self.build('gemv'),rows,4)
    def test_changed_flags_and_bits_rejected(self):
        for key in ('new_flags','seed_flags','fcsr','golden'):
            rows=copy.deepcopy(self.rows['gemv'])
            if key=='golden':rows[0][key][0]^=1
            else:rows[0][key]^=1
            with self.assertRaises(ValueError):audit.accounting(self.build('gemv'),rows,4)
    def test_generation_reproducible_and_counted(self):
        for op in self.rows:
            with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
                spec=ROOT/f'operators/cases/{op}/fp32-numeric.json'
                meta=fixture.generate(spec,Path(a),dict(max_n=128))
                self.assertEqual(meta,fixture.generate(spec,Path(b),dict(max_n=128)))
                for f in meta['files']:self.assertEqual((Path(a)/f).read_bytes(),(Path(b)/f).read_bytes())
                for t in fixture.TILES:
                    _,count=audit.accounting(self.build(op),self.rows[op],t)
                    self.assertEqual(count,meta['checks_by_n'][str(t)])
    def test_registry_matrix(self):
        cfg=operators.ame.config(ROOT/'configs/operators/fp32-numeric-r4.json')
        self.assertEqual(len(operators.plan(cfg)),16)

if __name__=='__main__':unittest.main()
