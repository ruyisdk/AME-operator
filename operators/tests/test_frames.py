import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from audit_operator_frames import execute_prefix, review_kernel

class FrameReview(unittest.TestCase):
    def prefix(self):
        ops=[("addi","sp,sp,-208"),("csrr","s11,amenlen"),("mul","s11,s11,s11"),
             ("srli","s11,s11,2"),("li","s5,0"),("slli","s6,s11,4"),
             ("add","s7,s5,s6"),("sub","sp,sp,s7")]
        return [(0x80000000+4*i,(0,op,arg)) for i,(op,arg) in enumerate(ops)]
    def test_all_n_both_xlen(self):
        for xlen in (32,64):
            for n in (4,8,16,32,64,128):
                self.assertEqual(execute_prefix(self.prefix(),n,0x80000000,xlen),
                                 (208+4*n*n,n*n//4))
    def test_unknown_frame(self):
        seq=self.prefix(); seq[-1]=(seq[-1][0],(0,"sub","sp,sp,a7"))
        with self.assertRaises(ValueError): execute_prefix(seq,4,0x80000000,64)
    def test_unknown_instruction(self):
        seq=self.prefix(); seq[1]=(seq[1][0],(0,"jalr","0(ra)"))
        with self.assertRaises(ValueError): execute_prefix(seq,4,0x80000000,64)
    def test_neg_known_and_unknown(self):
        seq=self.prefix()
        # Known XLEN-wrapped negative value, negated back before allocation.
        seq[4]=(seq[4][0],(0,"li","s5,-256"))
        seq[5]=(seq[5][0],(0,"neg","s6,s5"))
        seq[6]=(seq[6][0],(0,"mv","s7,s6"))
        for xlen in (32,64):
            self.assertEqual(execute_prefix(seq,4,0x80000000,xlen),(464,4))
        seq[5]=(seq[5][0],(0,"neg","s6,a7"))
        with self.assertRaisesRegex(ValueError,"unknown frame"):
            execute_prefix(seq,4,0x80000000,64)
    def test_wrong_csr(self):
        seq=self.prefix(); seq[1]=(seq[1][0],(0,"csrr","s11,cycle"))
        with self.assertRaises(ValueError): execute_prefix(seq,4,0x80000000,64)
    def test_no_allocation(self):
        with self.assertRaises(ValueError): execute_prefix(self.prefix()[:-1],4,0x80000000,64)
    def full(self):
        seq=self.prefix()
        tail=[("li","t1,0"),("slli","t0,s11,4"),("add","t1,t1,t0"),
              ("add","sp,sp,t1"),("ld","s11,0(sp)"),("addi","sp,sp,208"),("ret","")]
        seq += [(0x80000020+4*i,(0,op,arg)) for i,(op,arg) in enumerate(tail)]
        return dict(seq)
    def test_review_leaf(self):
        frames=review_kernel(self.full(),0x80000000,0x8000003c,dict(xlen=64))
        self.assertEqual(frames["128"],208+4*128*128)
    def test_review_call_rejected(self):
        code=self.full();code[0x80000038]=(0,"jal","80001000")
        with self.assertRaisesRegex(ValueError,"leaf"):
            review_kernel(code,0x80000000,0x8000003c,dict(xlen=64))
    def test_review_outside_jump_rejected(self):
        code=self.full();code[0x80000038]=(0,"j","80001000")
        with self.assertRaisesRegex(ValueError,"outside"):
            review_kernel(code,0x80000000,0x8000003c,dict(xlen=64))
    def test_review_indirect_jump_rejected(self):
        code=self.full();code[0x80000038]=(0,"jr","a0")
        with self.assertRaisesRegex(ValueError,"leaf"):
            review_kernel(code,0x80000000,0x8000003c,dict(xlen=64))
    def test_review_sp_mutation_rejected(self):
        code=self.full();code[0x80000038]=(0,"mv","sp,a0")
        with self.assertRaisesRegex(ValueError,"SP"):
            review_kernel(code,0x80000000,0x8000003c,dict(xlen=64))
if __name__=="__main__": unittest.main()
