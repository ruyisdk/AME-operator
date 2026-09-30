import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import operator_static_cost as cost

class StaticCensusTests(unittest.TestCase):
    def test_classes_disjoint(self):
        self.assertEqual(len(cost.CLASS),sum(map(len,cost.CATEGORIES.values())))
    def test_symbol_sizes(self):
        text='\n'.join(f'{0x100+16*i:x} 0000000c T {cost.PREFIX}{f}_kernel' for i,f in enumerate(cost.FAMILIES))
        self.assertEqual(cost.kernel_symbols(text)[cost.PREFIX+'bf16_kernel'],(0x140,12))
    def test_missing_symbol(self):
        with self.assertRaises(ValueError):cost.kernel_symbols('')
    def test_duplicate_symbol(self):
        text='100 00000004 T '+cost.PREFIX+'i32_kernel\n'
        with self.assertRaises(ValueError):cost.kernel_symbols(text+text)
    def test_no_size(self):
        with self.assertRaises(ValueError):cost.kernel_symbols('100 T '+cost.PREFIX+'i32_kernel')
    def test_counts(self):
        code=cost.instructions('100: 00000000 msettyp m0,a0\n104: 00000000 mls.1r m0,a0\n108: 00000000 ret')
        r=cost.census(code,0x100,12)
        self.assertEqual((r['instructions'],r['text_bytes']),(3,12))
        self.assertEqual(r['categories']['datatype'],1)
        self.assertEqual(r['categories']['ame-raw-load'],1)
        self.assertEqual(sum(r['categories'].values()),3)
    def test_missing_instruction(self):
        with self.assertRaises(ValueError):cost.census({0x100:('00000000','ret')},0x100,8)
    def test_duplicate_instruction(self):
        with self.assertRaises(ValueError):cost.instructions('100: 00000000 ret\n100: 00000000 ret')
    def test_short_instruction(self):
        with self.assertRaises(ValueError):cost.census({0x100:('0000','ret')},0x100,4)
    def test_unknown_instruction(self):
        with self.assertRaises(ValueError):cost.census({0x100:('00000000','munknown')},0x100,4)
    def test_scalar_mul_not_ame(self):
        r=cost.census({0x100:('00000000','mul')},0x100,4)
        self.assertEqual(r['categories']['scalar-other'],1)
        self.assertEqual(r['categories']['ame-compute'],0)

if __name__=='__main__':unittest.main()
