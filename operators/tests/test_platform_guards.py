import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'operators/reference')]
import ame
import operators
import operator_platform_guards as runner
import platform_guard_fixtures as fixture
import spike_operator_guard_adapter as adapter

class PlatformGuards(unittest.TestCase):
    def build(self,op='gemm'):
        g=fixture.create(op)
        return dict(row=dict(operator=op,xlen=32,case=dict(id=9241)),config=dict(seed=1),
                    symbols=dict(ame_test_done=0x80000090),
                    fixtures=dict(checks_by_support={'yes':fixture.checks(g,True),'no':fixture.checks(g,False)})),g
    def test_inventory_and_counts(self):
        for op in ('gemm','gemv'):
            b,g=self.build(op)
            for hw in runner.HARDWARE:
                a=runner.accounting(b,g,hw)
                self.assertEqual(a['logical_cases'],100)
                self.assertEqual(a['checks'],19996 if a['supported'] else 20016)
                self.assertEqual(a['computing_calls'],20 if a['supported'] else 0)
    def test_hardware_axes(self):
        self.assertEqual(set(runner.HARDWARE.values()),{(32,4),(32,128),(8,4),(16,8),(64,4),(32,2),(32,256)})
    def test_ownership_precedence(self):
        _,g=self.build()
        for r in g['cases']:
            if r['action']=='NO_WORK':
                self.assertEqual(fixture.status(r,False),3 if r['phase']==1 else 4)
                self.assertEqual(fixture.status(r,True),3 if r['phase']==1 else 2)
    def test_bad_descriptor_precedence(self):
        _,g=self.build()
        r=next(r for r in g['cases'] if r['phase']==0 and r['family']==4 and r['action']=='BAD_WORK')
        self.assertEqual(fixture.status(r,True),1);self.assertEqual(fixture.status(r,False),4)
    def test_k0_both_operators(self):
        for op in ('gemm','gemv'):
            _,g=self.build(op)
            r=next(r for r in g['cases'] if r['phase']==1 and r['family']==4 and r['action']=='K0_ADD')
            self.assertEqual(r['success'],r['initial'])
            self.assertIn(0x7fc12345,r['success']);self.assertIn(0x7f800000,r['success'])
    def test_exact_literals(self):
        _,g=self.build()
        for f,answer in enumerate((0xffffffb3,0xffffffeb,600,0xbf400000,0xbf400000)):
            r=next(r for r in g['cases'] if r['phase']==0 and r['family']==f and r['action']=='SET')
            self.assertEqual(r['success'][8],answer)
    def test_missing_case(self):
        b,g=self.build();g['cases'].pop()
        with self.assertRaises(ValueError):runner.accounting(b,g,'u32-t4')
    def test_changed_golden(self):
        b,g=self.build();g['cases'][0]['success'][8]^=1
        with self.assertRaises(ValueError):runner.accounting(b,g,'u32-t4')
    def test_changed_type(self):
        b,g=self.build();g['cases'][0]['width']=2
        with self.assertRaises(ValueError):runner.accounting(b,g,'u8-t4')
    def test_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);m=d/'case.json';m.write_text(json.dumps(dict(operator='gemv',contract='platform-guards-v1')))
            meta=fixture.generate(m,d,dict(max_n=128));first=(d/'op_fixtures.h').read_bytes()
            fixture.generate(m,d,dict(max_n=128))
            self.assertEqual(first,(d/'op_fixtures.h').read_bytes())
            self.assertEqual(meta['logical_cases'],100)
    def test_registry_driver_not_typed(self):
        c=ame.config(ROOT/'configs/operators/platform-guards-r1.json');rows=operators.plan(c)
        self.assertEqual(len(rows),16)
        for r in rows:
            v=r['definition'];driver='operators/tests/platform_guard_driver.c'
            self.assertEqual(v['harness'],'operators/tests/platform_guard_harness.c')
            self.assertNotIn(driver,v['typed_sources']);self.assertIn(driver,v['cxx_sources'])
    def test_normal_harness_default(self):
        c=ame.config(ROOT/'configs/operators/state-edges-r1.json')
        for r in operators.plan(c):self.assertNotIn('harness',r['definition'])
    def normal_rejection(self,uds,n,reason):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);elf=d/'dummy';elf.write_bytes(b'not executed')
            b=dict(elf=str(elf),elf_sha256=ame.sha(elf),row=dict(xlen=32,profile=runner.COMPILED),config=dict(max_n=128))
            platform=dict(schema=1,csr_map=ame.CSR_MAP,entry_contract=ame.ENTRY,command=['/not/executed'],
                          xlen=32,uds=uds,mregs=16,accregs=4,n_values=[n])
            with self.assertRaisesRegex(ValueError,reason):ame.execute(b,platform,n,d/'out')
    def test_normal_runtime_minimum_not_relaxed(self):
        self.normal_rejection(32,2,'invalid runtime N')
    def test_normal_profile_match_not_relaxed(self):
        self.normal_rejection(8,4,'build/platform profile mismatch')
    def observations(self,trace):
        s=dict(_start=0x80000000,ame_test_done=0x80000090,guard_observation_begin=0x80004000,guard_observation_end=0x80004010)
        lines=['0x80000000','M','0x40000020','0x6000000','0x0',
               'AME N=2 NELEM=4 UDSZ=32 MREGS=16 ACCREGS=4 owned=0 fflags=0x0 xsat=0 status=0x0']
        if trace:lines+=['0x80004000','0x80004010']
        lines+=['0x80000090','AME N=2 NELEM=4 UDSZ=32 MREGS=16 ACCREGS=4 owned=0 fflags=0x0 xsat=0 status=0x0']+['0x0']*32
        return s,'\n'.join(lines)
    def test_raw_negative_hardware(self):
        for trace in (False,True):
            s,text=self.observations(trace);observed,_,_=adapter.parse(text,s,ame.CSR_MAP,trace)
            self.assertEqual(observed['n'],2)
    def test_missing_trace_boundary(self):
        s,text=self.observations(False)
        with self.assertRaises(ValueError):adapter.parse(text,s,ame.CSR_MAP,True)
    def test_wrong_trace_boundary(self):
        s,text=self.observations(True)
        with self.assertRaises(ValueError):adapter.parse(text.replace('0x80004010','0x80004014'),s,ame.CSR_MAP,True)
    def response(self):
        b,g=self.build();expect=runner.accounting(b,g,'u8-t4')
        req=dict(nonce='abc',elf_sha256='hash',backend='spike',revision='rev',
                 expected=dict(xlen=32,n=4,uds=8,mregs=16,accregs=4,csr_map=ame.CSR_MAP,entry_contract=ame.ENTRY))
        words=[0x414d4554,1,2,9241,32,4,8,16,4,1,expect['checks'],0,0,0,0,1]+[0]*16
        resp=dict(schema=1,nonce='abc',elf_sha256='hash',backend='spike',revision='rev',observed=req['expected'],
                  stop_pc=b['symbols']['ame_test_done'],mailbox_hex=struct.pack('<32I',*words).hex())
        return b,req,resp,expect
    def test_separate_runtime_identity(self):
        b,req,resp,e=self.response();self.assertEqual(runner.validate(resp,req,b,e)[6],8)
    def test_wrong_observed_hardware(self):
        b,req,resp,e=self.response();resp['observed']=dict(req['expected'],uds=32)
        with self.assertRaises(ValueError):runner.validate(resp,req,b,e)
    def test_missing_exact_checks(self):
        b,req,resp,e=self.response();e=dict(e,checks=e['checks']+1)
        with self.assertRaises(ValueError):runner.validate(resp,req,b,e)
    def test_nonzero_failure_words(self):
        b,req,resp,e=self.response();w=list(struct.unpack('<32I',bytes.fromhex(resp['mailbox_hex'])));w[12]=1
        resp['mailbox_hex']=struct.pack('<32I',*w).hex()
        with self.assertRaises(ValueError):runner.validate(resp,req,b,e)
    def test_stale_nonce(self):
        b,req,resp,e=self.response();resp['nonce']='wrong'
        with self.assertRaises(ValueError):runner.validate(resp,req,b,e)
    def trace_audit(self,supported,pcs,corrupt=False):
        s,text=self.observations(True)
        s.update({f'ame_ops_gemm_{f}_kernel':0x80010000+0x100*i for i,f in enumerate(runner.FAMILIES)})
        s['ame_gemv_i32_owned']=0x80010500
        if supported:text=text.replace('N=2 NELEM=4','N=4 NELEM=16')
        text+='\n'+'\n'.join(f'core   0: 0x{p:08x} (0x00000013) nop' for p in pcs)+'\n'
        observed,stop,mailbox=adapter.parse(text,s,ame.CSR_MAP,True)
        response=dict(observed=observed,stop_pc=stop,mailbox_hex=mailbox.hex())
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);raw=text.encode();packed=gzip.compress(raw)
            (d/'spike.log.gz').write_bytes(packed)
            ame.save(d/'trace-storage.json',dict(lossless=True,raw_bytes=len(raw),
                raw_sha256='wrong' if corrupt else hashlib.sha256(raw).hexdigest(),
                gzip_sha256=hashlib.sha256(packed).hexdigest()))
            return runner.audit_observations(d,dict(symbols=s),dict(trace_driver=True),response,dict(supported=supported))
    def positive_pcs(self):
        return [0x80004000]+[0x80010000+0x100*i for i in range(5) for _ in range(4)]
    def test_positive_trace_all_kernels(self):
        r=self.trace_audit(True,self.positive_pcs())
        self.assertEqual(r['kernel_entries'],dict.fromkeys(runner.FAMILIES,4))
    def test_negative_trace_no_kernels(self):
        r=self.trace_audit(False,[0x80004000,0x80008000])
        self.assertEqual(r['kernel_entries'],dict.fromkeys(runner.FAMILIES,0))
    def test_unexpected_kernel_entry(self):
        with self.assertRaises(ValueError):self.trace_audit(False,[0x80004000,0x80010000])
    def test_jump_into_kernel_body(self):
        with self.assertRaises(ValueError):self.trace_audit(False,[0x80004000,0x80010004])
    def test_missing_positive_kernel(self):
        with self.assertRaises(ValueError):self.trace_audit(True,self.positive_pcs()[:-1])
    def test_empty_trace_rejected(self):
        with self.assertRaises(ValueError):self.trace_audit(False,[])
    def test_wrong_trace_start(self):
        with self.assertRaises(ValueError):self.trace_audit(False,[0x80008000])
    def test_trace_hash_rejected(self):
        with self.assertRaises(ValueError):self.trace_audit(False,[0x80004000],True)

if __name__=='__main__':unittest.main()
