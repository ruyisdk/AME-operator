import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'operators/reference'), str(ROOT/'scripts')]
import geometry_fixtures as fixture
import audit_operator_runs as audit
import operators


class Geometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {op: json.loads((ROOT/f'operators/cases/{op}/i32-geometry.json').read_text())
                     for op in ('gemm', 'gemv')}
        cls.data = {op: fixture.create(s) for op, s in cls.specs.items()}

    def report(self, op):
        return dict(row=dict(variant='i32-geometry-v2', operator=op),
                    fixtures=dict(cases_by_n={str(n):28 if op=='gemm' else 20 for n in fixture.TILES}))

    def test_covering_set(self):
        for op in self.specs:
            for n in fixture.TILES:
                rows, extra = audit.selected_cases(self.report(op), self.data[op], n)
                self.assertEqual(extra, 9)
                self.assertEqual(len(rows), 28 if op=='gemm' else 20)

    def test_no_duplicate_names(self):
        for rows in self.data.values():
            self.assertEqual(len(rows), len({r['name'] for r in rows}))

    def test_padding_retained(self):
        for r in self.data['gemm']:
            for i in range(r['m']):
                for j in range(r['n'], r['ldc']):
                    self.assertEqual(r['golden'][i*r['ldc']+j], 0x123456)

    def test_beta_pair(self):
        for rows in self.data.values():
            for zero, add in zip(rows[::2], rows[1::2]):
                self.assertEqual(zero['a'], add['a'])
                self.assertEqual(zero['b'], add['b'])
                for i in range(zero['m']):
                    for j in range(zero['n']):
                        p=i*zero['ldc']+j
                        self.assertEqual(add['golden'][p], zero['golden'][p]+add['initial'][p])

    def test_missing_case_rejected(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            audit.selected_cases(self.report('gemm'), self.data['gemm'][1:], 4)

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit.selected_cases(self.report('gemv'), self.data['gemv']+[self.data['gemv'][0]], 4)

    def test_wrong_large_boundary_rejected(self):
        rows = copy.deepcopy(self.data['gemm'])
        next(r for r in rows if r['name']=='t128-k-double-plus-set')['k']=256
        with self.assertRaisesRegex(ValueError, 'coverage'):
            audit.selected_cases(self.report('gemm'), rows, 128)

    def test_wrong_stride_rejected(self):
        rows=copy.deepcopy(self.data['gemv']); rows[0]['lda']-=1
        with self.assertRaisesRegex(ValueError, 'stride'):
            audit.selected_cases(self.report('gemv'), rows, 4)

    def test_tile_selection_rejected(self):
        spec=copy.deepcopy(self.specs['gemm']); spec['tiles']=[4]
        with self.assertRaises(ValueError): fixture.create(spec)

    def test_generated_selection(self):
        with tempfile.TemporaryDirectory() as out:
            p=Path(out)
            meta=fixture.generate(ROOT/'operators/cases/gemv/i32-geometry.json', p, dict(max_n=128))
            self.assertEqual(meta['logical_cases'], 120)
            self.assertEqual(set(meta['cases_by_n'].values()), {20})
            rows=json.loads((p/'golden.json').read_text())
            for n in fixture.TILES:
                expected=6+4*(128*128+32)+9
                for r in rows:
                    if r['tile']==n:
                        expected+=1+meta['max_c']+32+3*128*128+32-3*n*n+len(r['a'])+len(r['b'])
                self.assertEqual(meta['checks_by_n'][str(n)], expected)

    def test_registry_matrix(self):
        config=operators.ame.config(ROOT/'configs/operators/integer-geometry-r2.json')
        rows=operators.plan(config)
        self.assertEqual(len(rows), 16)
        self.assertEqual({r['language'] for r in rows}, {'c', 'c++'})
        self.assertTrue(all('operators/tests/geometry_driver.c' in r['definition']['typed_sources']
                            for r in rows))

if __name__=='__main__': unittest.main()
