import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'operators/reference'),str(ROOT/'scripts')]
import state_edges_fixtures as fixture
import audit_operator_state_edges as audit
import operators

class StateEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data={op:fixture.create(dict(operator=op,contract='state-edges-all-v1')) for op in ('gemm','gemv')}
    def build(self,op):return dict(row=dict(operator=op),config=dict(max_n=128))
    def test_inventory_and_counts(self):
        for op,data in self.data.items():
            for t in (4,8,16,32,64,128):
                rows,count=audit.accounting(self.build(op),data,t)
                self.assertEqual(len(rows),125);self.assertEqual(count,fixture.checks(data,t))
                self.assertEqual(sum(r['computed'] for r in rows),30)
    def test_real_owned_un_states(self):
        rows=self.data['gemm']['cases']
        self.assertEqual(sum(r['phase']!=1 and r['old_un']==1 for r in rows),50)
        for family in range(5):
            for old in (0,1):
                found=[r for r in rows if r['phase']==0 and r['family']==family and r['old_un']==old]
                self.assertEqual({r['action'] for r in found},set(fixture.ACTIONS))
    def test_reacquired_reverse_type_order(self):
        rows=[r for r in self.data['gemv']['cases'] if r['phase']==2]
        self.assertEqual([r['family'] for r in rows],[4,4,3,3,2,2,1,1,0,0])
        self.assertTrue(all(r['old_un']==1 and r['computed'] for r in rows))
    def test_k_zero_every_special_seed_both_operators(self):
        for op,data in self.data.items():
            slots=[16+(q//2)*3+q%2 if op=='gemm' else 16+q for q in range(8)]
            for r in data['cases']:
                if r['action'] not in ('K0_SET','K0_ADD'):continue
                self.assertEqual([r['initial'][i] for i in slots],list(fixture.SPECIAL))
                self.assertEqual([r['golden'][i] for i in slots],[0]*8 if r['action']=='K0_SET' else list(fixture.SPECIAL))
                self.assertEqual((r['status'],r['computed']),(0,0))
    def test_literal_compute_answers(self):
        pairs=((0xffffffb3,0xffffffbc),(0xffffffeb,0xfffffff4),(600,609),
               (0xbf400000,0xbf000000),(0xbf400000,0xbf000000))
        for r in self.data['gemv']['cases']:
            if r['computed']:self.assertEqual(r['golden'][16],pairs[r['family']][r['action']=='ADD'])
    def test_release_precedence(self):
        want={'SET':3,'K0_SET':0,'K0_ADD':0,'EMPTY':0,'BAD_MODE':1,'NO_WORK':3,'BAD_A':1}
        for r in self.data['gemm']['cases']:
            if r['phase']==1:self.assertEqual((r['status'],r['computed']),(want[r['action']],0))
    def test_mutated_state_rejected(self):
        for key in ('old_un','status','computed','fcsr','flags'):
            data=copy.deepcopy(self.data['gemm']);data['cases'][0][key]^=1
            with self.assertRaises(ValueError):audit.accounting(self.build('gemm'),data,4)
    def test_mutated_bits_rejected(self):
        for key in ('a','b','initial','golden'):
            data=copy.deepcopy(self.data['gemv']);data['cases'][0][key][16]^=1
            with self.assertRaisesRegex(ValueError,'bits'):audit.accounting(self.build('gemv'),data,4)
    def test_missing_or_reordered_family_rejected(self):
        for missing in (False,True):
            data=copy.deepcopy(self.data['gemv'])
            if missing:data['cases'].pop(0)
            else:data['cases'][0],data['cases'][1]=data['cases'][1],data['cases'][0]
            with self.assertRaisesRegex(ValueError,'inventory'):audit.accounting(self.build('gemv'),data,4)
    def test_reproducible_generation(self):
        for op in self.data:
            with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
                spec=ROOT/f'operators/cases/{op}/state-edges.json'
                meta=fixture.generate(spec,Path(a),dict(max_n=128))
                self.assertEqual(meta,fixture.generate(spec,Path(b),dict(max_n=128)))
                for name in meta['files']:self.assertEqual((Path(a)/name).read_bytes(),(Path(b)/name).read_bytes())
                for t in (4,8,16,32,64,128):self.assertEqual(meta['checks_by_n'][str(t)],audit.accounting(self.build(op),self.data[op],t)[1])
    def test_registry_matrix(self):
        cfg=operators.ame.config(ROOT/'configs/operators/state-edges-r1.json')
        rows=operators.plan(cfg);self.assertEqual(len(rows),16)
        for row in rows:
            for family in ('i32','i8','u8','f32','bf16'):
                self.assertIn(f'operators/kernels/gemm/{family}.c',row['definition']['typed_sources'])
    def test_model_template(self):
        for suffix in ('rv32','rv64','probe-rv32','probe-rv64'):
            platform=operators.ame.load(ROOT/f'configs/operators/spike-state-edges-{suffix}.json')
            self.assertEqual(platform['revision'],'configured-model')

if __name__=='__main__':unittest.main()
