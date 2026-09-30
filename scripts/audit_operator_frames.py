#!/usr/bin/env python3
"""Fail-closed frame review for registered leaf GEMM kernels."""
import argparse
from pathlib import Path
import ame
from audit_bank_capture import insns, REGS, signed

def execute_prefix(seq,n,initial,xlen):
    r=dict.fromkeys(REGS); r.update(zero=0,sp=initial)
    pc=0; mask=(1<<xlen)-1; steps=0
    addresses={v[0]:i for i,v in enumerate(seq)}
    while pc < len(seq):
        steps+=1
        ame.require(steps < len(seq)*2,"prologue loops")
        address,(_,op,arg)=seq[pc]; a=arg.split(","); dest=None; val=None
        if op in ("sw","sd"): pass
        elif op=="csrr":
            ame.require(a==["s11","amenlen"],"unreviewed CSR"); dest="s11"; val=n
        elif op in ("addi","add","sub","mul","slli","srli","and","or","sltiu"):
            dest=a[0]; lhs=r[a[1]]; rhs=r[a[2]] if a[2] in r else int(a[2],0)
            if lhs is not None and rhs is not None:
                if op in ("add","addi"): val=lhs+rhs
                elif op=="sub": val=lhs-rhs
                elif op=="mul": val=lhs*rhs
                elif op=="slli": val=lhs<<rhs
                elif op=="srli": val=lhs>>rhs
                elif op=="and": val=lhs&rhs
                elif op=="or": val=lhs|rhs
                else: val=int(lhs < rhs)
        elif op=="li": dest=a[0]; val=int(a[1],0)
        elif op=="mv": dest=a[0]; val=r[a[1]]
        elif op=="neg":
            dest=a[0]; val=None if r[a[1]] is None else -r[a[1]]
        elif op=="seqz":
            dest=a[0]; val=None if r[a[1]] is None else int(r[a[1]]==0)
        elif op in ("beq","beqz"):
            lhs=r[a[0]]; rhs=0 if op=="beqz" else r[a[1]]
            ame.require(lhs is not None and rhs is not None,"unknown guard")
            if lhs==rhs:
                target=int(a[-1].split()[0],16)
                ame.require(target>address and target in addresses,"unexpected prologue branch")
                pc=addresses[target]; continue
        elif op=="ebreak": raise ValueError("T rejected by compiler prologue")
        else: raise ValueError("unreviewed prologue instruction "+op)
        if dest and dest!="zero": r[dest]=None if val is None else val&mask
        ame.require(r["sp"] is not None,"unknown frame")
        if op=="sub" and a[0]=="sp":
            return initial-r["sp"], r["s11"]
        pc+=1
    raise ValueError("dynamic frame allocation missing")

def review_kernel(code, start, end, row):
    seq=[(p,code[p]) for p in sorted(code) if start<=p<end]
    writes=[(p,op,arg) for p,(_,op,arg) in seq if arg.startswith("sp,") and op not in ("sw","sd")]
    ame.require(len(writes)==4,"unexpected kernel SP mutations")
    ame.require([op for _,op,_ in writes]==["addi","sub","add","addi"],"unexpected frame sequence")
    fixed=-int(writes[0][2].rsplit(",",1)[1],0)
    ame.require(writes[0][2]==f"sp,sp,{-fixed}" and writes[-1][2]==f"sp,sp,{fixed}",
                "unmatched fixed frame")
    ame.require(writes[1][2].startswith("sp,sp,") and writes[2][2].startswith("sp,sp,"),
                "non-relative dynamic frame")
    ame.require(not any(op in ("jal","jalr","jr","call","tail") for _,(_,op,_) in seq),
                "kernel is no longer a reviewed leaf")
    for _,(_,op,arg) in seq:
        if op == "j" or op.startswith("b"):
            target=int(arg.split(",")[-1].split()[0],16)
            ame.require(start<=target<end,"kernel branches outside reviewed region")
    s11=[(p,op,arg) for p,(_,op,arg) in seq if arg.startswith("s11,") and op not in ("sw","sd")]
    ame.require([op for _,op,_ in s11]==["csrr","mul","srli",("lw" if row["xlen"]==32 else "ld")],
                "runtime size register changed in body")
    # Restore amount must be built solely from the stable size register plus zero.
    pos=next(i for i,v in enumerate(seq) if v[0]==writes[2][0])
    epilogue=seq[pos-3:pos+1]
    # O0/O2 both use li zero, size shift, add; allow their scheduling order.
    restore=[v[1][1:] for v in epilogue]
    ame.require(sorted(op for op,_ in restore)==["add","add","li","slli"],
                "unreviewed dynamic restore computation")
    frames={}
    for n in (4,8,16,32,64,128):
        frame,q=execute_prefix(seq,n,0x80000000,row["xlen"])
        r={"s11":q,"sp":0x80000000-frame}
        for op,arg in restore:
            a=arg.split(",")
            if op=="li": r[a[0]]=int(a[1],0)
            elif op=="slli": r[a[0]]=r[a[1]]<<int(a[2],0)
            else: r[a[0]]=r[a[1]]+r[a[2]]
        ame.require(r["sp"]==0x80000000-fixed,"dynamic stack not balanced")
        frames[str(n)]=frame
    return frames

def audit(directory, *, write=True):
    b=ame.load(directory/"build.json"); s=b["symbols"]; row=b["row"]
    ame.require(ame.sha(b["elf"])==b["elf_sha256"],"changed ELF")
    code=insns((directory/"objdump.log").read_text())
    if row["variant"] in ("state-edges-all-v1","platform-guards-v1"):
        names=["ame_ops_gemm_i32_kernel","ame_ops_gemm_i8_kernel",
               "ame_ops_gemm_u8_kernel","ame_ops_gemm_f32_kernel",
               "ame_ops_gemm_bf16_kernel","ame_gemv_i32_owned"]
        boundaries=list(zip(names,names[1:]))
    elif row["variant"] in ("widen8-i32-v1","widen8-state-v2"):
        boundaries=[("ame_ops_gemm_i8_kernel","ame_ops_gemm_u8_kernel"),
                    ("ame_ops_gemm_u8_kernel","ame_gemv_i8_i32_owned")]
    elif row["variant"] in ("bf16-f32-exact-v1", "bf16-state-v1", "bf16-f32-rne-fold-v1"):
        boundaries=[("ame_ops_gemm_bf16_kernel","ame_gemv_bf16_f32_owned")]
    elif row["variant"] in ("fp32-exact-v1", "fp32-tail-probe-v1", "fp32-state-v3", "fp32-rne-fold-v1"):
        boundaries=[("ame_ops_gemm_f32_kernel","ame_gemv_f32_owned")]
    else:
        ame.require(row["variant"] in ("i32-exact-v1","i32-geometry-v2",
                    "i32-state-v3","i32-numeric-v4"),"unreviewed frame variant")
        boundaries=[("ame_ops_gemm_i32_kernel","ame_gemv_i32_owned")]
    regions=[(s[first],s[last]) for first,last in boundaries]
    ame.require(all(start<end for start,end in regions),"unordered kernel symbols")
    kernels={name:review_kernel(code,start,end,row)
             for (name,_),(start,end) in zip(boundaries,regions)}
    # Both typed kernels are leaf and cannot branch into one another.
    frames={str(n):max(f[str(n)] for f in kernels.values()) for n in (4,8,16,32,64,128)}
    # Sum all external fixed decrements, overestimating mutually exclusive paths.
    init=[p for p,(_,op,arg) in code.items() if op=="auipc" and arg.startswith("sp,")]
    ame.require(len(init)==1,"startup SP")
    ip=init[0]; ame.require(code[ip+4][1]=="addi","startup SP sequence")
    top=ip+signed(int(code[ip][2].split(",")[1],0)<<12,32)+int(code[ip+4][2].rsplit(",",1)[1],0)
    ame.require(top==s["__stack_top"],"startup/top mismatch")
    allowance=0
    for p,(_,op,arg) in code.items():
        if any(start<=p<end for start,end in regions) or p in (ip,ip+4) or op in ("sw","sd") or not arg.startswith("sp,"): continue
        ame.require(op=="addi" and arg.startswith("sp,sp,"),"dynamic non-kernel frame")
        allowance+=max(0,-int(arg.rsplit(",",1)[1],0))
    ame.require(allowance<4096,"unexpected caller frame cost")
    reserve=s["__stack_top"]-s["__stack_bottom"]
    ame.require(max(frames.values())+4096<reserve,"frame exceeds reserved stack")
    ame.require(s["__bss_end"]<=s["__stack_bottom"] and s["__stack_top"]<=0x80f00000,
                "data/stack/boot overlap")
    result=dict(elf_sha256=b["elf_sha256"],allowed_n=[4,8,16,32,64,128],
        kernel_frame_bytes=frames,kernels=kernels,conservative_external_bytes=allowance,
        caller_allowance=4096,reserved_stack_bytes=reserve,kernel_leaf=True,
        reviewer_script_sha256=ame.sha(Path(__file__)),objdump_sha256=ame.sha(directory/"objdump.log"),
        scope="frame and linked region bounds")
    if write:
        ame.save(directory/"operator-safety.json",result)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",type=Path); a=p.parse_args()
    rows={str(b.parent):audit(b.parent) for b in sorted(a.stage.glob("shard-*/build/*/build.json"))}
    ame.require(len(rows)==len(ame.load(a.stage/"plan.json")["rows"]),"missing builds")
    ame.save(a.stage/"operator-safety-summary.json",rows)
    print("OPERATOR_FRAME_REVIEW_PASS",len(rows),max(max(v["kernel_frame_bytes"].values()) for v in rows.values()))
if __name__=="__main__": main()
