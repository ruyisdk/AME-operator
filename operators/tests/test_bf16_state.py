import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'operators/reference')]
import bf16_state_fixtures as fixture
import audit_operator_bf16_state as audit
import operators

class BF16State(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data={op:fixture.create(dict(operator=op,contract='bf16-state-v1')) for op in ('gemm','gemv')}
    def report(self,op):return dict(row=dict(operator=op),config=dict(max_n=128))
    def case(self,op,phase,action):
        return next(r for r in self.data[op]['cases'] if (r['phase'],r['action'])==(phase,action))
    def test_inventory_and_counts(self):
        self.assertEqual(len(fixture.ACTIONS),70)
        for op in self.data:
            for t in (4,8,16,32,64,128):
                rows,count=audit.accounting(self.report(op),self.data[op],t)
                self.assertEqual(len(rows),99 if op=='gemm' else 92)
                self.assertEqual(count,fixture.checks(self.data[op],t))
    def test_literal_products_and_repeated(self):
        self.assertEqual([self.case('gemm',0,'VALID_SET')['golden'][i] for i in (16,17,20,21)],
            [0xbe100000,0xbe000000,0xbd800000,0x3ed00000])
        for phase in (0,2):
            self.assertEqual(self.case('gemv',phase,'VALID_ADD')['golden'][16:18],[0xbd000000,0x3de00000])
            self.assertEqual(self.case('gemv',phase,'REPEATED')['golden'][16:18],[0xbed80000,0xbe400000])
            self.assertEqual(self.case('gemm',phase,'INPUT_SHARED')['golden'][16],0x3d800000)
    def test_legal_offset_storage(self):
        r=self.case('gemm',0,'INPUT_OFFSET_TWO')
        self.assertEqual(r['initial_a'][16:20],[0x7f81,0x3e80,0xbec0,0x3f00])
        self.assertEqual(r['initial_b'][16:19],[0x7f81,0x3ec0,0xbe80])
        r=self.case('gemm',0,'OUTPUT_OFFSET_FOUR')
        self.assertEqual(r['golden'][16:19],[0x7f812345,0xbe100000,0xbe000000])
    def test_all_workspace_members_and_pairs(self):
        for op in self.data:
            for field in 'ABC':
                for prefix in ('NULL_WS_','SHORT_WS_','UNALIGNED_WS_'):
                    r=self.case(op,0,prefix+field);self.assertEqual(r['status'],2);self.assertEqual(r['golden'],r['initial'])
                for other in ('A','B','C','DESC'):
                    self.assertEqual(self.case(op,0,f'WS_{field}_ALIAS_{other}')['status'],1)
                self.assertEqual(self.case(op,0,f'WS_{field}_ADDRESS_WRAP')['status'],1)
            for pair in ('A_ALIAS_B','A_ALIAS_C','B_ALIAS_C'):
                self.assertEqual(self.case(op,0,'WS_'+pair+'_WORK')['status'],1)
    def test_released_order(self):
        for op in self.data:
            for action in ('VALID_SET','VALID_ADD','NULL_DESC','UNALIGNED_DESC','DESC_ADDRESS_WRAP',
                           'DESC_ALIAS_C','NULL_WS_A','SHORT_WS_B','UNALIGNED_WS_C','WS_C_ALIAS_A','WS_B_ADDRESS_WRAP'):
                self.assertEqual(self.case(op,1,action)['status'],3)
            for action in ('BAD_MODE','BAD_MODE_EMPTY','NULL_A','NULL_C'):
                self.assertEqual(self.case(op,1,action)['status'],1)
    def test_zero_k_special_preservation(self):
        for op in self.data:
            for phase in range(3):
                add=self.case(op,phase,'ZERO_K_SPECIAL_ADD');clear=self.case(op,phase,'ZERO_K_SPECIAL_SET')
                self.assertEqual(add['golden'],add['initial']);self.assertEqual(add['fflags'],0)
                self.assertEqual(add['initial'][16:18],[0x80000000,0x7f800001])
                self.assertEqual(clear['golden'][16:18],[0,0])
    def test_missing_and_duplicate_rejected(self):
        for duplicate in (False,True):
            data=copy.deepcopy(self.data['gemm'])
            if duplicate:data['cases'].append(data['cases'][0])
            else:data['cases'].pop()
            with self.assertRaisesRegex(ValueError,'inventory'):audit.accounting(self.report('gemm'),data,4)
    def test_changed_inputs_and_offsets_rejected(self):
        for field in ('initial_a','initial_b'):
            data=copy.deepcopy(self.data['gemv'])
            next(r for r in data['cases'] if r['action']=='INPUT_OFFSET_TWO')[field][17]^=1
            with self.assertRaisesRegex(ValueError,'inputs'):audit.accounting(self.report('gemv'),data,4)
    def test_changed_seed_and_output_rejected(self):
        for field,message in (('initial','seed'),('golden','GOLDEN')):
            data=copy.deepcopy(self.data['gemm'])
            next(r for r in data['cases'] if r['action']=='ZERO_K_SPECIAL_ADD')[field][16]=0
            with self.assertRaisesRegex(ValueError,message):audit.accounting(self.report('gemm'),data,128)
    def test_changed_state_expectations_rejected(self):
        for field in ('status','computed','calls','action_id','fflags','xsat','fcsr'):
            data=copy.deepcopy(self.data['gemv']);data['cases'][0][field]^=1
            with self.assertRaisesRegex(ValueError,'expectation'):audit.accounting(self.report('gemv'),data,4)
    def test_reproducible(self):
        for op in self.data:
            with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
                path=ROOT/f'operators/cases/{op}/bf16-state.json'
                meta=fixture.generate(path,Path(a),dict(max_n=128))
                self.assertEqual(meta,fixture.generate(path,Path(b),dict(max_n=128)))
                for p in meta['files']:self.assertEqual((Path(a)/p).read_bytes(),(Path(b)/p).read_bytes())
    def test_registry_and_fcsr_platform(self):
        cfg=operators.ame.config(ROOT/'configs/operators/bf16-state-r6.json')
        self.assertEqual(len(operators.plan(cfg)),16)
        for x in (32,64):
            p=operators.ame.load(ROOT/f'configs/operators/spike-bf16-state-rv{x}.json')
            self.assertEqual(Path(p['command'][0]).name,'spike_fpstate_adapter.py')
            self.assertEqual(p['capabilities'],['csr-local-v1','bf16-state-fcsr'])

if __name__=='__main__':unittest.main()
