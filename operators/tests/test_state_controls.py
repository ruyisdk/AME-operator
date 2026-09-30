import copy
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'operators/reference')]
import operator_state_controls as controls
import fp32_numeric_fixtures as fixtures

SOURCE = '''\t.text
ame_ops_gemm_f32_kernel:
\taddi\tsp,sp,-16
\tmmulacc.2d\tacc0,m0,m1
\taddi\tsp,sp,16
\tjr\tra
\t.size\tame_ops_gemm_f32_kernel, .-ame_ops_gemm_f32_kernel
other:
\tjr\tra
'''

class TransformTests(unittest.TestCase):
    def test_positive_identical(self):
        self.assertEqual(controls.transform(SOURCE, 'positive-relink'), (SOURCE, []))
    def test_csr_only_before_kernel_return(self):
        for csr in ('amefflags', 'amexsat', 'fcsr'):
            result, sites = controls.transform(SOURCE, 'clear-' + csr)
            insertion = '\tcsrw\t' + csr + ',zero\n'
            self.assertEqual(result.replace(insertion, ''), SOURCE)
            self.assertEqual(len(sites), 1)
            self.assertIn(insertion + '\tjr\tra\n\t.size', result)
            self.assertTrue(result.endswith('other:\n\tjr\tra\n'))
    def test_zero_before_same_acc(self):
        source = SOURCE.replace('acc0', 'acc3')
        result, _ = controls.transform(source, 'zero-acc-before-multiply')
        self.assertEqual(result.replace('\tmzero.2d.acc\tacc3\n', ''), source)
        self.assertIn('mzero.2d.acc\tacc3\n\tmmulacc.2d\tacc3', result)
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError): controls.transform(SOURCE, 'bad')
    def test_missing_multiply_rejected(self):
        with self.assertRaises(ValueError): controls.transform(SOURCE.replace('mmulacc.2d', 'nop'), 'zero-acc-before-multiply')
    def test_duplicate_multiply_rejected(self):
        source = SOURCE.replace('\tmmulacc.2d\tacc0,m0,m1\n', '\tmmulacc.2d\tacc0,m0,m1\n' * 2)
        with self.assertRaises(ValueError): controls.transform(source, 'zero-acc-before-multiply')
    def test_missing_return_rejected(self):
        with self.assertRaises(ValueError): controls.transform(SOURCE.replace('jr\tra', 'ebreak'), 'clear-fcsr')
    def test_wrong_boundary_rejected(self):
        with self.assertRaises(ValueError): controls.transform(SOURCE.replace('.size', '.other'), 'clear-fcsr')
    def test_duplicate_boundary_rejected(self):
        with self.assertRaises(ValueError): controls.transform(SOURCE * 2, 'positive-relink')

class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {op: fixtures.create(dict(operator=op, contract='fp32-rne-fold-v1'))
                    for op in ('gemm', 'gemv')}
        cls.report = dict(config=dict(max_n=128), fixtures=dict(checks_by_n={'4':123, '128':456}))
    def test_positive_count(self):
        for n, count in ((4, 123), (128, 456)):
            self.assertEqual(controls.prediction('positive-relink', self.report, self.rows['gemm'], n),
                dict(status=2, checks=count, error=0, index=0, expected=0, actual=0))
    def test_fcsr_first_call(self):
        for op, rows in self.rows.items():
            for n in controls.NS:
                v = next(i for i, r in enumerate(rows) if r['tile'] == n)
                result = controls.prediction('clear-fcsr', self.report, rows, n)
                self.assertEqual(result, dict(status=3, checks=9, error=1,
                    index=0x79040000 + v, expected=rows[v]['fcsr'], actual=0), op)
    def test_zero_acc_exposes_fusion_seed(self):
        for rows in self.rows.values():
            for n in controls.NS:
                v = next(i for i, r in enumerate(rows) if r['tile'] == n)
                data, flags = controls.zero_acc_result(rows[v], n)
                self.assertEqual((data[0], flags), (0x3f800000, 1))
                result = controls.prediction('zero-acc-before-multiply', self.report, rows, n)
                self.assertEqual(result, dict(status=3, checks=7, error=1, index=0x79020000 + v,
                    expected=rows[v]['seed_flags'], actual=rows[v]['seed_flags'] | 1))
    def test_first_detectable_flag_loss(self):
        for rows in self.rows.values():
            for n in controls.NS:
                for kind, field, base in (('clear-amefflags', 'seed_flags', 0x79020000),
                                          ('clear-amexsat', 'seed_xsat', 0x79030000)):
                    result = controls.prediction(kind, self.report, rows, n)
                    selected = [i for i, r in enumerate(rows) if r['tile'] == n]
                    v = next(i for i in selected if rows[i][field] or (kind == 'clear-amefflags' and rows[i]['new_flags']))
                    self.assertEqual(result['index'], base + v)
                    self.assertEqual(result['actual'], 0)
                    self.assertGreater(result['expected'], 0)
    def test_reset_at_each_block(self):
        row = dict(m=1, n=1, k=5, lda=5, ldb=1, ldc=1, a=[0x3f800000]*5,
                   b=[0x3f800000]*5, initial=[0x40000000], accumulate=1)
        self.assertEqual(controls.zero_acc_result(row, 4), ([0x3f800000], 0))
        self.assertEqual(controls.zero_acc_result(row, 128), ([0x40a00000], 0))
    def test_wrong_n_rejected(self):
        with self.assertRaises(ValueError): controls.prediction('clear-fcsr', self.report, self.rows['gemm'], 8)
    def test_undetected_fault_rejected(self):
        rows = copy.deepcopy(self.rows['gemm'])
        for row in rows: row['seed_xsat'] = 0
        with self.assertRaises(ValueError): controls.prediction('clear-amexsat', self.report, rows, 4)

class InventoryTests(unittest.TestCase):
    def rows(self):
        return [dict(key=f'{op}-{x}-{k}', operator=op, xlen=x, kind=k,
            expected_status=2 if k == 'positive-relink' else 3)
            for op in ('gemm', 'gemv') for x in (32, 64) for k in controls.KINDS]
    def test_exact(self): controls.manifest_rows(self.rows())
    def test_missing(self):
        with self.assertRaises(ValueError): controls.manifest_rows(self.rows()[1:])
    def test_duplicate(self):
        rows = self.rows(); rows[-1] = rows[0]
        with self.assertRaises(ValueError): controls.manifest_rows(rows)
    def test_wrong_failure_classification(self):
        rows = self.rows(); rows[1]['expected_status'] = 2
        with self.assertRaises(ValueError): controls.manifest_rows(rows)

if __name__ == '__main__': unittest.main()
