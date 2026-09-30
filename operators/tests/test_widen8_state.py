import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"operators/reference")]
import widen8_state_fixtures as fixture
import audit_operator_widen8_state as audit
import operators

class WidenState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data={op:fixture.create(dict(operator=op,contract="widen8-state-v2"))
                  for op in ("gemm","gemv")}
    def report(self,op):return dict(row=dict(operator=op),config=dict(max_n=128))
    def case(self,op,phase,u,action):
        return next(r for r in self.data[op]["cases"] if
                    (r["phase"],r["unsigned"],r["action"])==(phase,u,action))
    def test_inventory_and_count(self):
        for op in self.data:
            for t in (4,8,16,32,64,128):
                rows,count=audit.accounting(self.report(op),self.data[op],t)
                self.assertEqual(count,fixture.checks(self.data[op],t))
                self.assertEqual(len(rows),100 if op=="gemm" else 90)
    def test_signedness_hand_results(self):
        for u,want in ((0,[383,-637,254,-250]),(1,[65407,64899,33278,32518])):
            self.assertEqual([self.case("gemm",0,u,"VALID_SET")["golden"][i]
                              for i in (16,17,20,21)],want)
            self.assertEqual(self.case("gemv",0,u,"VALID_SET")["golden"][16:18],want[::2])
    def test_add_and_repeat(self):
        for op,slots,old in (("gemm",(16,17,20,21),(7,-5,11,13)),("gemv",(16,17),(7,11))):
            for u in (0,1):
                base=self.case(op,0,u,"VALID_SET")["golden"]
                for i,s in zip(slots,old):
                    self.assertEqual(self.case(op,0,u,"VALID_ADD")["golden"][i],base[i]+s)
                    self.assertEqual(self.case(op,2,u,"REPEATED")["golden"][i],base[i]*3)
    def test_owned_precedence(self):
        for op in self.data:
            for u in (0,1):
                for action in ("VALID_SET","VALID_ADD","NULL_WS","SHORT_WS"):
                    self.assertEqual(self.case(op,1,u,action)["status"],3)
                self.assertEqual(self.case(op,1,u,"BAD_MODE_EMPTY")["status"],1)
                self.assertEqual(self.case(op,1,u,"ZERO_K_SET")["status"],0)
    def test_input_alias_and_odd_alignment(self):
        for op in self.data:
            for u in (0,1):
                self.assertEqual(self.case(op,0,u,"INPUT_SHARED")["golden"][16],16384)
                self.assertEqual(self.case(op,0,u,"VALID_SET")["status"],0)
        self.assertNotIn("UNALIGNED_A",fixture.ACTIONS)
        self.assertNotIn("UNALIGNED_B",fixture.ACTIONS)
        self.assertEqual(fixture.BYTE_GUARD%16,1)
    def test_missing_unsigned_rejected(self):
        d=copy.deepcopy(self.data["gemm"]);d["cases"]=[r for r in d["cases"] if not r["unsigned"]]
        with self.assertRaisesRegex(ValueError,"inventory"):audit.accounting(self.report("gemm"),d,4)
    def test_wrong_input_rejected(self):
        d=copy.deepcopy(self.data["gemm"]);d["initial_a"][17]=127
        with self.assertRaisesRegex(ValueError,"inputs"):audit.accounting(self.report("gemm"),d,4)
    def test_wrong_status_rejected(self):
        d=copy.deepcopy(self.data["gemv"])
        next(r for r in d["cases"] if r["phase"]==1 and r["action"]=="VALID_SET")["status"]=0
        with self.assertRaisesRegex(ValueError,"expectation"):audit.accounting(self.report("gemv"),d,128)
    def test_changed_error_output_rejected(self):
        d=copy.deepcopy(self.data["gemm"])
        next(r for r in d["cases"] if r["action"]=="C_OVERLAP_B")["golden"][16]+=1
        with self.assertRaisesRegex(ValueError,"GOLDEN"):audit.accounting(self.report("gemm"),d,4)
    def test_reproducible(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            manifest=ROOT/"operators/cases/gemv/widen8-state.json"
            meta=fixture.generate(manifest,Path(a),dict(max_n=128))
            self.assertEqual(meta,fixture.generate(manifest,Path(b),dict(max_n=128)))
            for p in meta["files"]:self.assertEqual((Path(a)/p).read_bytes(),(Path(b)/p).read_bytes())
    def test_registry(self):
        cfg=operators.ame.config(ROOT/"configs/operators/integer-widen8-state-r6.json")
        self.assertEqual(len(operators.plan(cfg)),16)

if __name__=="__main__":unittest.main()
