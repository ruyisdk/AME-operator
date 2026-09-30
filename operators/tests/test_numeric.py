import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'operators/reference'),str(ROOT/'scripts')]
import numeric_fixtures
import audit_operator_numeric as audit
import integer
import operators

class Numeric(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={op:numeric_fixtures.create(dict(operator=op,contract='i32-numeric-v4'))
                  for op in ('gemm','gemv')}
    def test_coverage_and_values(self):
        for op,total in (('gemm',17),('gemv',15)):
            for n in (4,8,16,32,64,128):
                rows,extra=audit.selected(dict(row=dict(operator=op)),self.rows[op],n)
                self.assertEqual(len(rows),total);self.assertEqual(extra,9)
    def test_full_signed_input_zero(self):
        self.assertEqual(integer.gemv(1,1,[-2147483648],1,[0],[0],0),[0])
    def test_excluded_negative_result(self):
        with self.assertRaises(ValueError):
            integer.gemv(1,1,[-2147483648],1,[1],[0],0)
    def test_product_overflow_rejected(self):
        with self.assertRaises(ValueError):integer.gemv(1,1,[46341],1,[46341],[0],0)
    def test_seed_overflow_rejected(self):
        with self.assertRaises(ValueError):integer.gemv(1,1,[2],1,[3],[2147483642],1)
    def test_bad_boundary_rejected(self):
        rows=copy.deepcopy(self.rows['gemm']);rows[0]['a'][0]-=1
        with self.assertRaisesRegex(ValueError,'boundary'):
            audit.selected(dict(row=dict(operator='gemm')),rows,4)
    def test_bad_golden_rejected(self):
        rows=copy.deepcopy(self.rows['gemm'])
        next(r for r in rows if r['purpose']=='long-k-0')['golden'][0]+=1
        with self.assertRaisesRegex(ValueError,'GOLDEN'):
            audit.selected(dict(row=dict(operator='gemm')),rows,4)
    def test_registry(self):
        c=operators.ame.config(ROOT/'configs/operators/integer-numeric-r4.json')
        self.assertEqual(len(operators.plan(c)),16)

if __name__=='__main__':unittest.main()
