#!/usr/bin/env python3
"""Static instruction census of the five production kernels."""
import argparse
from collections import Counter
from pathlib import Path
import re
import ame

FAMILIES=('i32','i8','u8','f32','bf16')
PREFIX='ame_ops_gemm_'
CATEGORIES={
    'datatype': {'msettyp','mgettyp','asettyp','agettyp'},
    'ame-transfer': {'mmov.a.m','mmov.m.a'},
    'ame-raw-load': {'mls.1r'}, 'ame-raw-store': {'mss.1r'},
    'ame-tile-load': {'mls.rm'}, 'ame-tile-store': {'mss.rm'},
    'ame-compute': {'mmulacc.2d'},
    'csr': {'csrr','csrw','csrrs','csrrw'},
    'scalar-load': {'lb','lbu','lh','lhu','lw','lwu','ld'},
    'scalar-store': {'sb','sh','sw','sd'},
    'control': {'beq','beqz','bne','bnez','blt','bltu','bge','bgeu',
                'j','jal','jalr','ret','ebreak'},
    'scalar-other': {'add','addi','addw','addiw','sub','subw','mul','mulw',
                     'li','lui','auipc','mv','neg','and','andi','or','ori',
                     'xor','xori','slli','slliw','srli','srliw','srai','sraiw',
                     'seqz','snez','sltu','sltiu','sext.w','nop'},
}
CLASS={op:name for name,ops in CATEGORIES.items() for op in ops}
LINE=re.compile(r'^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+([\w.]+)\s*(.*)$')

def instructions(text):
    code={}
    for line in text.splitlines():
        m=LINE.match(line)
        if not m: continue
        address=int(m[1],16)
        ame.require(address not in code,'duplicate instruction address')
        code[address]=(m[2],m[3])
    return code

def kernel_symbols(text):
    found={}
    wanted={PREFIX+f+'_kernel' for f in FAMILIES}
    for line in text.splitlines():
        words=line.split()
        if words and words[-1] in wanted:
            ame.require(len(words)==4 and words[2] in ('T','t'), 'kernel symbol has no code size')
            name=words[-1]
            ame.require(name not in found,'duplicate kernel symbol')
            address,size=int(words[0],16),int(words[1],16)
            ame.require(size>0 and size%4==0 and address%4==0,'unreviewed kernel extent')
            found[name]=(address,size)
    ame.require(set(found)==wanted,'missing production kernel')
    return found

def census(code,address,size):
    expected=set(range(address,address+size,4))
    actual={p for p in code if address<=p<address+size}
    ame.require(actual==expected,'incomplete instruction coverage')
    hist=Counter();categories=Counter()
    for p in sorted(expected):
        encoding,op=code[p]
        ame.require(len(encoding)==8,'unreviewed instruction width')
        ame.require(op in CLASS,'unclassified instruction: '+op)
        hist[op]+=1;categories[CLASS[op]]+=1
    ame.require(sum(hist.values())*4==size,'instruction byte total')
    return dict(instructions=sum(hist.values()),text_bytes=size,
                categories={name:categories[name] for name in CATEGORIES},
                histogram=dict(sorted(hist.items())))

def report(stage,out):
    out.mkdir(parents=True,exist_ok=False)
    records=[];rows=[];seen=set()
    for path in sorted(stage.glob('shard-*/build/*/build.json')):
        b=ame.load(path);row=b['row'];key=row['key']
        ame.require(row['variant']=='state-edges-all-v1','unreviewed census variant')
        ame.require(key not in seen and ame.sha(b['elf'])==b['elf_sha256'],'duplicate/changed ELF')
        seen.add(key)
        gate=ame.load(path.parent/'operator-safety.json')
        ame.require(gate['elf_sha256']==b['elf_sha256'],'mismatched frame evidence')
        nm=ame.tool_paths(b['config'])['nm']
        ame.require(ame.sha(nm)==b['tools'][str(nm)],'nm identity changed')
        dest=out/key;dest.mkdir()
        text=ame.command([nm,'-S','--defined-only',b['elf']],dest,'symbols',b['config'],records)
        symbols=kernel_symbols(text)
        code=instructions((path.parent/'objdump.log').read_text())
        ame.require(ame.sha(path.parent/'objdump.log')==gate['objdump_sha256'],'disassembly changed')
        for family in FAMILIES:
            name=PREFIX+family+'_kernel';address,size=symbols[name]
            ame.require(address==b['symbols'][name],'symbol address changed')
            item=census(code,address,size)
            ame.require(item['categories']['ame-compute']==1,'unexpected matmul site count')
            rows.append(dict(key=key,operator=row['operator'],xlen=row['xlen'],
                language=row['language'],optimization=row['opt'],family=family,
                symbol=name,address=address,elf_sha256=b['elf_sha256'],
                objdump_sha256=gate['objdump_sha256'],
                frame_by_t=gate['kernels'][name],**item))
    ame.require(seen=={r['key'] for r in ame.load(stage/'plan.json')['rows']},'missing build')
    ame.save(out/'commands.json',records)
    result=dict(status='STATIC_KERNEL_CENSUS',builds=len(seen),kernels=len(rows),rows=rows,
        script_sha256=ame.sha(Path(__file__)),measurement='static',
        scope='Five leaf production kernels including inline packing; excludes public validators, wrappers and driver')
    ame.save(out/'census.json',result)
    lines=['# Production kernel static census','','Each row counts instructions in one compiled kernel.',
           'GEMV uses the same GEMM kernel. All GEMM/GEMV ELF paths remain in census.json.',
           'Raw moves include state/payload preservation and allocator spills.',
           'Only GEMM-container rows are shown below.','',
           '| XLEN | C/C++ | Opt | Family | Instructions | Text bytes | Datatype | Raw load/store | Tile load/store | Transfer | Frame T4/T128 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        if r['operator']!='gemm':continue
        c=r['categories'];f=r['frame_by_t']
        lines.append(f"| {r['xlen']} | {r['language']} | {r['optimization']} | {r['family']} | {r['instructions']} | {r['text_bytes']} | {c['datatype']} | {c['ame-raw-load']}/{c['ame-raw-store']} | {c['ame-tile-load']}/{c['ame-tile-store']} | {c['ame-transfer']} | {f['4']}/{f['128']} |")
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('STATIC_KERNEL_CENSUS',len(seen),len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.stage,a.out)
