import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'operators/reference')]
import fp32_state_fixtures as fixture
import audit_operator_fp32_state as audit
import operators

class FP32State(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data={op:fixture.create(dict(operator=op,contract='fp32-state-v3')) for op in ('gemm','gemv')}
    def report(self,op):return dict(row=dict(operator=op),config=dict(max_n=128))
    def case(self,op,phase,action):
        return next(r for r in self.data[op]['cases'] if (r['phase'],r['action'])==(phase,action))
    def test_inventory_counts(self):
        for op in self.data:
            for t in (4,8,16,32,64,128):
                rows,count=audit.accounting(self.report(op),self.data[op],t)
                self.assertEqual(len(rows),62 if op=='gemm' else 55)
                self.assertEqual(count,fixture.checks(self.data[op],t))
    def test_literal_products(self):
        want=[0xbe100000,0xbe000000,0xbd800000,0x3ed00000]
        self.assertEqual([self.case('gemm',0,'VALID_SET')['golden'][i] for i in (16,17,20,21)],want)
        self.assertEqual(self.case('gemv',0,'VALID_SET')['golden'][16:18],want[::2])
        self.assertEqual(self.case('gemm',0,'INPUT_SHARED')['golden'][16],0x3d800000)
    def test_add_and_repeated(self):
        for phase in (0,2):
            self.assertEqual(self.case('gemv',phase,'VALID_ADD')['golden'][16:18],[0xbd000000,0x3de00000])
            self.assertEqual(self.case('gemv',phase,'REPEATED')['golden'][16:18],[0xbed80000,0xbe400000])
    def test_zero_k_special_bits(self):
        for op in self.data:
            for phase in range(3):
                add=self.case(op,phase,'ZERO_K_SPECIAL_ADD');clear=self.case(op,phase,'ZERO_K_SPECIAL_SET')
                self.assertEqual(add['golden'],add['initial']);self.assertEqual(add['fflags'],0)
                self.assertEqual(add['initial'][16:18],[0x80000000,0x7f800001])
                self.assertEqual(clear['golden'][16:18],[0,0]);self.assertEqual(clear['fflags'],0)
    def test_released_precedence(self):
        for op in self.data:
            for action in ('VALID_SET','VALID_ADD','NULL_WS','SHORT_WS'):
                self.assertEqual(self.case(op,1,action)['status'],3)
            for action in ('BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_C'):
                self.assertEqual(self.case(op,1,action)['status'],1)
            self.assertEqual(self.case(op,1,'ZERO_K_SPECIAL_ADD')['status'],0)
    def test_missing_action_rejected(self):
        data=copy.deepcopy(self.data['gemm']);data['cases'].pop()
        with self.assertRaisesRegex(ValueError,'inventory'):audit.accounting(self.report('gemm'),data,4)
    def test_changed_input_rejected(self):
        data=copy.deepcopy(self.data['gemv']);data['initial_a'][16]^=1
        with self.assertRaisesRegex(ValueError,'inputs'):audit.accounting(self.report('gemv'),data,4)
    def test_changed_special_seed_rejected(self):
        data=copy.deepcopy(self.data['gemm'])
        next(r for r in data['cases'] if r['action']=='ZERO_K_SPECIAL_ADD')['initial'][16]=0
        with self.assertRaisesRegex(ValueError,'seed'):audit.accounting(self.report('gemm'),data,128)
    def test_changed_status_and_flags_rejected(self):
        for field in ('status','fcsr','fflags','xsat','calls','action_id'):
            data=copy.deepcopy(self.data['gemv']);data['cases'][0][field]^=1
            with self.assertRaisesRegex(ValueError,'expectation'):audit.accounting(self.report('gemv'),data,4)
    def test_changed_computed_output_rejected(self):
        data=copy.deepcopy(self.data['gemv']);data['cases'][0]['golden'][16]^=1
        with self.assertRaisesRegex(ValueError,'GOLDEN'):audit.accounting(self.report('gemv'),data,4)
    def test_reproducible(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            path=ROOT/'operators/cases/gemm/fp32-state.json'
            meta=fixture.generate(path,Path(a),dict(max_n=128))
            self.assertEqual(meta,fixture.generate(path,Path(b),dict(max_n=128)))
            for p in meta['files']:self.assertEqual((Path(a)/p).read_bytes(),(Path(b)/p).read_bytes())
    def test_registry_and_fp_observation(self):
        cfg=operators.ame.config(ROOT/'configs/operators/fp32-state-r3.json')
        self.assertEqual(len(operators.plan(cfg)),16)
        for x in (32,64):
            p=operators.ame.load(ROOT/f'configs/operators/spike-fp32-state-rv{x}.json')
            self.assertEqual(Path(p['command'][0]).name,'spike_fpstate_adapter.py')
            self.assertEqual(p['capabilities'],['csr-local-v1','fp32-state-fcsr'])

if __name__=='__main__':unittest.main()
