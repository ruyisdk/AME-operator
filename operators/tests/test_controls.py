import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts')]
from operator_controls import prediction, mutation_sites, validate_manifest, kinds

class Controls(unittest.TestCase):
    def setUp(self):
        self.report=dict(config=dict(max_n=128),fixtures=dict(checks_by_n={'4':123,'128':456}))
        self.rows=[dict(tile=4,m=1,n=1,k=1,ldc=1,accumulate=0,
                        initial=[7],golden=[6],a=[2],b=[3])]
    def test_first_numeric_mismatch(self):
        self.assertEqual(prediction('skip-mmulacc',self.report,self.rows,4),
            dict(status=3,checks=21,error=1,index=0x52000010,expected=6,actual=0))
    def test_negative_word(self):
        self.rows[0]['golden']=[-6]
        self.assertEqual(prediction('skip-mmulacc',self.report,self.rows,4)['expected'],0xfffffffa)
    def test_add_keeps_initial(self):
        self.rows[0].update(accumulate=1,golden=[13])
        self.assertEqual(prediction('skip-mmulacc',self.report,self.rows,4)['actual'],7)
    def test_filtered_index(self):
        other=copy.deepcopy(self.rows[0]); other['tile']=128
        self.assertEqual(prediction('skip-mmulacc',self.report,self.rows+[other],128)['index'],0x52010010)
    def test_positive_copy(self):
        self.assertEqual(prediction('positive-copy',self.report,self.rows,128),
            dict(status=2,checks=456,error=0,index=0,expected=0,actual=0))
    def test_insensitive_rejected(self):
        self.rows[0]['golden']=[0]
        with self.assertRaisesRegex(ValueError,'does not detect'):
            prediction('skip-mmulacc',self.report,self.rows,4)
    def test_empty_selection_rejected(self):
        with self.assertRaises(ValueError): prediction('skip-mmulacc',self.report,self.rows,128)
    def test_unknown_mutation(self):
        with self.assertRaises(ValueError): prediction('other',self.report,self.rows,4)
    def wide_rows(self):
        a=copy.deepcopy(self.rows[0]);a['unsigned']=0
        b=copy.deepcopy(a);b.update(unsigned=1,golden=[65025],a=[255],b=[255])
        return [a,b]
    def test_signed_fault_first_case(self):
        self.assertEqual(prediction('skip-i8-mmulacc',self.report,self.wide_rows(),4),
            dict(status=3,checks=21,error=1,index=0x52000010,expected=6,actual=0))
    def test_unsigned_fault_preserves_signed_case(self):
        # The complete first fixture includes output/padding, workspace and byte guards.
        first=1+33+(3*128**2+32-3*4**2)+2+68
        self.assertEqual(prediction('skip-u8-mmulacc',self.report,self.wide_rows(),4),
            dict(status=3,checks=21+first,error=1,index=0x52010010,expected=65025,actual=0))
    def test_wide_unsigned_fault_does_not_accept_signed_witness(self):
        with self.assertRaisesRegex(ValueError,'does not detect'):
            prediction('skip-u8-mmulacc',self.report,self.wide_rows()[:1],4)
    def test_wide_signed_fault_does_not_accept_unsigned_witness(self):
        with self.assertRaisesRegex(ValueError,'does not detect'):
            prediction('skip-i8-mmulacc',self.report,self.wide_rows()[1:],4)
    def test_wide_seed_preserved_by_mutation(self):
        rows=self.wide_rows();rows[1].update(accumulate=1,golden=[65032])
        self.assertEqual(prediction('skip-u8-mmulacc',self.report,rows,4)['actual'],7)
    def sites(self):
        report=dict(row=dict(variant='widen8-i32-v1'),symbols=dict(
            ame_ops_gemm_i8_kernel=100,ame_ops_gemm_u8_kernel=200,ame_gemv_i8_i32_owned=300))
        code={104:('00','mmulacc.2d',''),208:('00','mmulacc.2d',''),212:('00','add','')}
        return report,code
    def test_mutation_sites_are_disjoint(self):
        r,c=self.sites()
        self.assertEqual(mutation_sites(r,c,'skip-i8-mmulacc'),{104})
        self.assertEqual(mutation_sites(r,c,'skip-u8-mmulacc'),{208})
        self.assertEqual(mutation_sites(r,c,'positive-copy'),set())
    def test_missing_unsigned_instruction_rejected(self):
        r,c=self.sites();del c[208]
        with self.assertRaisesRegex(ValueError,'sites'):mutation_sites(r,c,'skip-i8-mmulacc')
    def test_extra_instruction_outside_kernels_rejected(self):
        r,c=self.sites();c[304]=('00','mmulacc.2d','')
        with self.assertRaisesRegex(ValueError,'sites'):mutation_sites(r,c,'positive-copy')
    def manifest(self,variant='widen8-i32-v1'):
        return [dict(key=f'{o}-{x}-{k}',operator=o,xlen=x,kind=k,variant=variant,
                     expected_status=2 if k=='positive-copy' else 3)
                for o in ('gemm','gemv') for x in (32,64) for k in kinds(variant)]
    def test_complete_manifests(self):
        for variant in ('widen8-i32-v1','i32-geometry-v2','fp32-exact-v1','bf16-f32-exact-v1'):
            self.assertEqual(validate_manifest(self.manifest(variant)),kinds(variant))
    def test_missing_control_rejected(self):
        with self.assertRaisesRegex(ValueError,'inventory'):validate_manifest(self.manifest()[:-1])
    def test_duplicate_control_rejected(self):
        rows=self.manifest();rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'inventory'):validate_manifest(rows)
    def test_wrong_control_expected_status_rejected(self):
        rows=self.manifest();rows[0]['expected_status']=3
        with self.assertRaisesRegex(ValueError,'status'):validate_manifest(rows)
    def test_unreviewed_variant_rejected(self):
        with self.assertRaisesRegex(ValueError,'variant'):kinds('widen8-state-v2')
    def test_fp32_bits_and_flag_check_order(self):
        rows=copy.deepcopy(self.rows);rows[0].update(golden=[0xbf800000],initial=[0x3f800000])
        self.assertEqual(prediction('skip-f32-mmulacc',self.report,rows,4),
            dict(status=3,checks=24,error=1,index=0x73000010,expected=0xbf800000,actual=0))
    def test_fp32_retained_seed_bits(self):
        rows=copy.deepcopy(self.rows);rows[0].update(accumulate=1,initial=[0x3f800000],golden=[0x40000000])
        self.assertEqual(prediction('skip-f32-mmulacc',self.report,rows,4)['actual'],0x3f800000)
    def test_fp32_full_buffer_check_counts(self):
        rows=copy.deepcopy(self.rows);rows[0]['golden']=[0]
        other=copy.deepcopy(rows[0]);other['golden']=[0x3f800000];other['a']=[1]*7
        first=4+33+(3*128**2+32-3*4**2)+(7+32)+(1+32)
        self.assertEqual(prediction('skip-f32-mmulacc',self.report,rows+[other],4)['checks'],24+first)
    def test_fp32_no_insensitive_witness(self):
        rows=copy.deepcopy(self.rows);rows[0]['golden']=[0]
        with self.assertRaisesRegex(ValueError,'does not detect'):
            prediction('skip-f32-mmulacc',self.report,rows,4)
    def test_fp32_instruction_region(self):
        report=dict(row=dict(variant='fp32-exact-v1'),symbols=dict(ame_ops_gemm_f32_kernel=100,ame_gemv_f32_owned=200))
        code={104:('00','mmulacc.2d','')}
        self.assertEqual(mutation_sites(report,code,'skip-f32-mmulacc'),{104})
        code[204]=('00','mmulacc.2d','')
        with self.assertRaisesRegex(ValueError,'sites'):mutation_sites(report,code,'positive-copy')
    def test_bf16_descriptor_and_numeric_check_order(self):
        rows=copy.deepcopy(self.rows);rows[0].update(golden=[0xbf800000],initial=[0x3f800000])
        self.assertEqual(prediction('skip-bf16-mmulacc',self.report,rows,4),
            dict(status=3,checks=32,error=1,index=0x83000010,expected=0xbf800000,actual=0))
        rows[0]['accumulate']=1
        self.assertEqual(prediction('skip-bf16-mmulacc',self.report,rows,4)['actual'],0x3f800000)
    def test_bf16_distinct_workspace_counts(self):
        rows=copy.deepcopy(self.rows);rows[0]['golden']=[0]
        other=copy.deepcopy(rows[0]);other['golden']=[0x3f800000];other['a']=[1]*7
        first=11+33+(5*128**2+96-5*4**2)+(7+32)+(1+32)
        self.assertEqual(prediction('skip-bf16-mmulacc',self.report,rows+[other],4)['checks'],32+first)
        rows[0]['k']=0
        self.assertEqual(prediction('skip-bf16-mmulacc',self.report,rows+[other],4)['checks'],32+first+80)
    def test_bf16_no_insensitive_witness(self):
        rows=copy.deepcopy(self.rows);rows[0]['golden']=[0]
        with self.assertRaisesRegex(ValueError,'does not detect'):
            prediction('skip-bf16-mmulacc',self.report,rows,4)
    def test_bf16_instruction_region(self):
        report=dict(row=dict(variant='bf16-f32-exact-v1'),symbols=dict(
            ame_ops_gemm_bf16_kernel=100,ame_gemv_bf16_f32_owned=200))
        code={104:('00','mmulacc.2d','')}
        self.assertEqual(mutation_sites(report,code,'skip-bf16-mmulacc'),{104})
        code[204]=('00','mmulacc.2d','')
        with self.assertRaisesRegex(ValueError,'sites'):mutation_sites(report,code,'positive-copy')

if __name__=='__main__': unittest.main()
