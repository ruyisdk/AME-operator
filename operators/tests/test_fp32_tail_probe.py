import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'operators/reference')]
import fp32_tail_probe as probe
from ieee_tininess import fp32_fused_expected as fma
from audit_operator_tail_probe import expected

class TailWitnesses(unittest.TestCase):
    def test_infinity_unused_lane(self):
        self.assertEqual(fma(1,0,0x3f800000,0x7f800000,0),(0x7f800000,0))
        self.assertEqual(fma(1,0,0,0x7f800000,0),(0x7fc00000,16))
    def test_negative_zero_and_candidate_padding(self):
        self.assertEqual(fma(1,0,0x80000000,0x3f800000,0x80000000),(0x80000000,0))
        self.assertEqual(fma(1,0,0,0,0x80000000),(0,0))
        for acc in (0,0x80000000,0x3f800000,0xbf800000):
            self.assertEqual(fma(1,0,0,0x80000000,acc),(acc,0))
    def test_fixture_counts(self):
        for op in ('gemm','gemv'):
            with tempfile.TemporaryDirectory() as d:
                out=Path(d)
                meta=probe.generate(ROOT/f'operators/cases/{op}/fp32-tail-probe.json',out,dict(max_n=128))
                self.assertEqual(meta['logical_cases'],6)
                rows=json.loads((out/'golden.json').read_text())
                self.assertEqual([r['tile'] for r in rows],[4,8,16,32,64,128])
                for n in probe.TILES:
                    self.assertEqual(meta['checks_by_n'][str(n)],expected(op,n,False)['checks'])
    def test_predicted_failures_are_specific(self):
        for n in (4,128):
            a=expected('gemm',n,True);b=expected('gemv',n,True)
            self.assertEqual((a['checks'],a['expected'],a['actual']),(6,0,16))
            self.assertEqual((b['checks'],b['expected'],b['actual']),(24,0x80000000,0))
        with self.assertRaises(ValueError):expected('gemm',3,True)

if __name__=='__main__':unittest.main()
