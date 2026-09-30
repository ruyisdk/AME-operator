import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'operators/reference'),str(ROOT/'scripts')]
import state_fixtures as fixture
import audit_operator_state as audit
import operators

class State(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data={op:fixture.create(dict(operator=op,contract='i32-state-v3'))
                  for op in ('gemm','gemv')}

    def report(self, op):
        return dict(row=dict(operator=op),config=dict(max_n=128))

    def case(self, op, phase, action):
        return next(r for r in self.data[op]['cases'] if r['phase']==phase and r['action']==action)

    def test_inventory(self):
        for op,count in (('gemm',45),('gemv',41)):
            for n in (4,8,16,32,64,128):
                rows,checks=audit.accounting(self.report(op),self.data[op],n)
                self.assertEqual(len(rows),count)
                self.assertEqual(checks,fixture.checks(self.data[op],n))

    def test_gemm_hand_results(self):
        for action,want in [('VALID_SET',[-9,-8,-4,26]),('VALID_ADD',[-2,-13,7,39]),
                            ('REPEATED',[-27,-24,-12,78])]:
            c=self.case('gemm',0,action)['golden']
            self.assertEqual([c[i] for i in (16,17,20,21)],want)

    def test_gemv_hand_results(self):
        for action,want in [('VALID_SET',[-9,-4]),('VALID_ADD',[-2,7]),('REPEATED',[-27,-12])]:
            self.assertEqual(self.case('gemv',0,action)['golden'][16:18],want)

    def test_input_alias_allowed(self):
        for op in self.data:
            self.assertEqual(self.case(op,0,'INPUT_SHARED')['golden'][16],4)

    def test_not_owned_and_precedence(self):
        for op in self.data:
            self.assertEqual(self.case(op,1,'VALID_SET')['status'],3)
            self.assertEqual(self.case(op,1,'NULL_WS')['status'],3)
            self.assertEqual(self.case(op,1,'BAD_MODE')['status'],1)
            self.assertEqual(self.case(op,1,'ZERO_K_SET')['status'],0)

    def test_reacquire_matches_owned(self):
        for op in self.data:
            for action in ('VALID_SET','VALID_ADD','REPEATED','INPUT_SHARED'):
                self.assertEqual(self.case(op,0,action)['golden'],self.case(op,2,action)['golden'])

    def test_missing_owned_rejected(self):
        data=copy.deepcopy(self.data['gemm']);data['cases'].pop(0)
        with self.assertRaisesRegex(ValueError,'inventory'):
            audit.accounting(self.report('gemm'),data,4)

    def test_missing_unowned_rejected(self):
        data=copy.deepcopy(self.data['gemv']);data['cases']=[r for r in data['cases'] if r['phase']!=1]
        with self.assertRaisesRegex(ValueError,'inventory'):
            audit.accounting(self.report('gemv'),data,128)

    def test_wrong_status_rejected(self):
        data=copy.deepcopy(self.data['gemm'])
        next(r for r in data['cases'] if r['phase']==1 and r['action']=='VALID_SET')['status']=0
        with self.assertRaisesRegex(ValueError,'expectation'):
            audit.accounting(self.report('gemm'),data,4)

    def test_erroneous_change_rejected(self):
        data=copy.deepcopy(self.data['gemm'])
        next(r for r in data['cases'] if r['action']=='BAD_MODE')['golden'][16]+=1
        with self.assertRaisesRegex(ValueError,'modifies'):
            audit.accounting(self.report('gemm'),data,4)

    def test_generation(self):
        with tempfile.TemporaryDirectory() as out:
            meta=fixture.generate(ROOT/'operators/cases/gemm/i32-state.json',Path(out),dict(max_n=128))
            self.assertEqual(meta['logical_cases'],45)
            data=json.loads((Path(out)/'golden.json').read_text())
            self.assertEqual(meta['checks_by_n']['128'],audit.accounting(self.report('gemm'),data,128)[1])

    def test_registry(self):
        c=operators.ame.config(ROOT/'configs/operators/integer-state-r3.json')
        self.assertEqual(len(operators.plan(c)),16)

if __name__=='__main__': unittest.main()
