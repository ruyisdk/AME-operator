import itertools
from pathlib import Path
import struct
import sys
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "operators/reference"))
import floating_exact as fp

class FloatingExact(unittest.TestCase):
    def test_literal_encodings(self):
        for value, grid, word, half in ((0,3,0,0),(8,3,0x3f800000,0x3f80),
                (-12,3,0xbfc00000,0xbfc0),(1,3,0x3e000000,0x3e00),
                (255,0,0x437f0000,0x437f),(-256,0,0xc3800000,0xc380)):
            self.assertEqual(fp.bits(value, grid), word)
            self.assertEqual(fp.bits(value, grid, "bf16"), half)
    def test_exact_grid_against_host_encoding(self):
        # Small dyadics are exactly binary64 too; host conversion needs no rounding.
        for grid in (0,3,6):
            for value in range(-255,256):
                word=struct.unpack('<I',struct.pack('<f',value/2**grid))[0]
                self.assertEqual(fp.bits(value,grid),word)
                self.assertEqual(fp.bits(value,grid,"bf16")<<16,word)
    def test_precision_not_silent_rounding(self):
        with self.assertRaisesRegex(ValueError,"exactly"):fp.bits(257,0,"bf16")
        with self.assertRaisesRegex(ValueError,"exactly"):fp.bits(2**24+1,0)
        self.assertEqual(fp.bits(2**24+2,0),0x4b800001)
    def test_normal_boundary(self):
        self.assertEqual(fp.bits(1,126),0x00800000)
        self.assertEqual(fp.bits(1,126,"bf16"),0x0080)
        for value,grid in ((1,127),(1<<128,0)):
            with self.assertRaisesRegex(ValueError,"normal"):fp.bits(value,grid)
    def test_bad_encoding_arguments(self):
        for args in ((True,3),(1,-1),(1,257),(1,3,'float'),(1.0,3)):
            with self.assertRaises(ValueError):fp.bits(*args)
    def test_hand_dot_and_seed(self):
        # (-1 * .5) + (.375 * -1) + (1.5 * .875) = .4375.
        a,b=[-8,3,12],[4,-8,7]
        self.assertEqual(fp.dot(a,b),28)
        self.assertEqual(fp.bits(fp.dot(a,b),6),0x3ee00000)
        self.assertEqual(fp.dot(a,b,5,input_format='bf16'),33)
    def test_all_small_term_orders(self):
        a,b=[-8,3,12],[4,-8,7]
        for order in itertools.permutations(range(3)):
            total=5
            for index in order:
                total=fp.dot([a[index]],[b[index]],total)
                self.assertEqual(fp.bits(total,6),
                    struct.unpack('<I',struct.pack('<f',total/64))[0])
            self.assertEqual(total,33)
    def test_abs_bound_including_seed(self):
        self.assertEqual(fp.dot([1],[1],fp.EXACT_LIMIT-1),fp.EXACT_LIMIT)
        with self.assertRaisesRegex(ValueError,"order-independent"):
            fp.dot([1],[1],fp.EXACT_LIMIT)
    def test_small_final_sum_not_sufficient(self):
        # Cancellation is zero, but the possible partial sum exceeds the contract.
        with self.assertRaisesRegex(ValueError,"order-independent"):
            fp.dot([1<<23,-(1<<23)],[2,2])
    def test_dot_rejects_input_format_loss(self):
        with self.assertRaisesRegex(ValueError,"exactly"):
            fp.dot([257],[0],input_format='bf16')
        with self.assertRaisesRegex(ValueError,"extents"):fp.dot([1],[])
    def test_underflowing_products_cannot_hide_in_zero_sum(self):
        with self.assertRaisesRegex(ValueError,"grid"):
            fp.dot([1,-1],[1,1],fraction_bits=64)
        self.assertEqual(fp.dot([1],[1],fraction_bits=63),1)
        self.assertEqual(fp.bits(1,126),0x00800000)
    def test_empty_operation_still_validates_reference_options(self):
        with self.assertRaisesRegex(ValueError,"format"):fp.dot([],[],input_format='bad')
        with self.assertRaisesRegex(ValueError,"grid"):
            fp.gemm(0,0,0,[],0,[],0,[],0,0,fraction_bits=64)
    def test_hand_gemm_padding_set_add(self):
        a,b,c=[8,-4,99,2,6],[4,8,99,-8,2],[5,-7,123,11,13]
        self.assertEqual(fp.gemm(2,2,2,a,3,b,3,c,3,0),[64,56,123,-40,28])
        self.assertEqual(fp.gemm(2,2,2,a,3,b,3,c,3,1),[69,49,123,-29,41])
    def test_gemv_matches_first_column(self):
        self.assertEqual(fp.gemv(2,2,[8,-4,99,2,6],3,[4,-8],[5,11],1),[69,-29])
    def test_zero_k_and_empty_output(self):
        self.assertEqual(fp.gemm(2,2,0,[],0,[],2,[1,2,9,3,4],3,0),[0,0,9,0,0])
        self.assertEqual(fp.gemm(2,2,0,[],0,[],2,[1,2,9,3,4],3,1),[1,2,9,3,4])
        self.assertEqual(fp.gemm(0,0,0,[],0,[],0,[],0,0),[])
    def test_invalid_matrix_shape(self):
        with self.assertRaisesRegex(ValueError,"short"):
            fp.gemm(1,1,2,[1],2,[1,2],1,[0],1,0)
        with self.assertRaisesRegex(ValueError,"shape"):
            fp.gemm(1,1,2,[1,2],1,[1,2],1,[0],1,0)
        with self.assertRaisesRegex(ValueError,"mode"):
            fp.gemm(1,1,1,[1],1,[1],1,[0],1,2)

if __name__ == '__main__':unittest.main()
