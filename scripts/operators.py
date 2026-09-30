#!/usr/bin/env python3
"""Operator registry, reproducible builds and separate DejaGNU execution."""
import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import ame

def module(path):
    spec = importlib.util.spec_from_file_location("ame_operator_fixture", path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def plan(c):
    registry = ame.load(ame.ROOT/c["registry"])
    ame.require(registry["schema"] == 1, "unknown operator registry")
    entries = registry["operators"]
    ame.require(len({op["id"] for op in entries}) == len(entries), "duplicate operators")
    ame.require(c["operators"] and len(set(c["operators"])) == len(c["operators"]), "empty/duplicate selection")
    ame.require(set(c["operators"]) <= {op["id"] for op in entries}, "unknown operator")
    rows, ids = [], set()
    for op in entries:
        ame.require(len({v["id"] for v in op["variants"]}) == len(op["variants"]), "duplicate variants")
        for v in op["variants"]:
            ame.require(v["case_id"] not in ids, "duplicate case ID")
            ids.add(v["case_id"])
        if op["id"] not in c["operators"]: continue
        candidates = [v for v in op["variants"] if v["id"] == c["variant"]]
        ame.require(len(candidates) == 1, "operator lacks selected variant")
        v = candidates[0]
        for name in [v["cases"],v["contract"],v["fixture_generator"],v.get("harness","runtime/harness.c")] + v["sources"] + v["inputs"]:
            path = (ame.ROOT/name).resolve()
            ame.require(path.is_relative_to(ame.ROOT) and path.is_file(), "missing/external registry input: "+name)
        ame.require(set(v["typed_sources"]) <= set(v["sources"]), "unlisted typed source")
        ame.require(set(v.get("cxx_sources",v["typed_sources"])) <= set(v["sources"]), "unlisted C++ source")
        for p,x,opt,lang in itertools.product(c["profiles"],c["xlens"],c["optimizations"],c["languages"]):
            ame.require(p in v["profiles"], "profile not implemented for "+op["id"])
            key = f'ops-{op["id"]}-{v["id"]}-rv{x}-u{p["uds"]}-m{p["mregs"]}-a{p["accregs"]}-{opt}-' + ("cxx" if lang=="c++" else "c")
            rows.append(dict(key=key,operator=op["id"],variant=v["id"],definition=v,
                case=dict(id=v["case_id"],capabilities=v["capabilities"]),
                profile=p,xlen=x,opt=opt,language=lang))
    ame.require(len({r["key"] for r in rows}) == len(rows), "duplicate build keys")
    return rows

def build(c,row,directory):
    directory.mkdir(parents=True,exist_ok=False)
    tools = ame.tool_paths(c)
    cache = os.environ.get("AME_TOOLS_SNAPSHOT")
    snapshot = ame.load(cache) if cache else ame.tool_snapshot(tools.values())
    ame.require(snapshot["identity"] == ame.tool_identity(tools.values()), "tools changed")
    v,p,x = row["definition"],row["profile"],row["xlen"]
    harness=v.get("harness","runtime/harness.c")
    fixed = ["runtime/start.S",harness,"runtime/test.h","runtime/link.ld",
             "reference/u32.h","scripts/ame.py","scripts/operators.py",
             "testsuite/ameops/ameops.exp",c["registry"]]
    inputs = sorted(set(fixed+v["sources"]+v["inputs"]+[v["cases"],v["contract"],v["fixture_generator"]]))
    hashes = {str(ame.ROOT/s):ame.sha(ame.ROOT/s) for s in inputs}
    inc = Path(c["toolchain_root"])/c["gcc_include"]
    hashes[str(inc/"riscv_ztt.h")] = ame.sha(inc/"riscv_ztt.h")
    archive = directory/"source"
    for name in inputs:
        dest=archive/name; dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ame.ROOT/name).read_bytes())
    meta = module(ame.ROOT/v["fixture_generator"]).generate(ame.ROOT/v["cases"],directory,c)
    ame.save(directory/"fixtures.json",meta)
    generated = {str(directory/s):ame.sha(directory/s) for s in meta["files"]+["fixtures.json"]}
    arch=f'rv{x}im_zicsr_ztt0p6'; abi="ilp32" if x==32 else "lp64"
    common=["-B"+str(tools["cc"].parent)+"/","-march="+arch,"-mabi="+abi,
        "-mcmodel=medany","-ffreestanding","-fno-builtin","-fno-stack-protector",
        "-fno-pic","-fno-tree-vectorize","-fno-tree-slp-vectorize","-g","-Wall","-Wextra"]
    common += ["-I"+str(d) for d in (ame.ROOT/"runtime",ame.ROOT/"reference",inc,directory)]
    defs=dict(CASE_ID=v["case_id"],MAX_N=c["max_n"],TEST_SEED=str(c["seed"])+"u",
              UDS=p["uds"],MREGS=p["mregs"],ACCREGS=p["accregs"],
              DATA_CASE=0,EXPECT_KIND=0,AUX_KIND=0,RAW_IMAGE=0,**v["defines"])
    common += [f"-D{k}={value}" for k,value in defs.items()]
    records=[]; objects=[]
    for i,name in enumerate(["runtime/start.S",harness]+v["sources"]):
        src=ame.ROOT/name; asm=directory/f"{i}.s"; obj=directory/f"{i}.o"
        typed=name in v["typed_sources"]
        cxx=name in v.get("cxx_sources",v["typed_sources"]) and row["language"]=="c++"
        argv=[tools["cxx" if cxx else "cc"]]+common
        if typed: argv += [f'-mztt-profile=gcc-runtime-u{p["uds"]}-m{p["mregs"]}-a{p["accregs"]}']
        if src.suffix==".S": argv += ["-E","-P","-x","assembler-with-cpp"]
        else:
            argv += ["-S","-"+(row["opt"] if name in v["sources"] else "O2"),
                     "-std=gnu++17" if cxx else "-std=gnu11"]
            if cxx: argv += ["-fno-exceptions","-fno-rtti"]
        ame.command(argv+[src,"-o",asm],directory,f"{i}-compile",c,records)
        ame.command([tools["assembler"],"-march="+arch,"-mabi="+abi,"-o",obj,asm],
                    directory,f"{i}-assemble",c,records)
        objects.append(obj)
    elf=directory/"test.elf"
    ame.command([tools["linker"],"-m",f"elf{x}lriscv","--no-relax","-T",
        ame.ROOT/"runtime/link.ld","-Map="+str(directory/"test.map"),"-o",elf]+objects,
        directory,"link",c,records)
    dump=ame.command([tools["objdump"],"-d",elf],directory,"objdump",c,records)
    attrs=ame.command([tools["readelf"],"-hlA",elf],directory,"attributes",c,records)
    names=ame.command([tools["nm"],"-n",elf],directory,"symbols",c,records)
    symbols={l.split()[2]:int(l.split()[0],16) for l in names.splitlines() if len(l.split())==3}
    ame.require(all(k in symbols for k in ("_start","ame_test_done","ame_mailbox","run_kernel")), "missing harness symbols")
    ame.require(symbols["__bss_end"] <= symbols["__stack_bottom"] < symbols["__stack_top"] <= 0x80f00000,
                "data/stack/adapter boot overlap")
    ame.require("ztt0p6" in attrs, "missing architecture attribute")
    for op in v["scan"]: ame.require(op in dump,"missing AME instruction "+op)
    ame.require(all(ame.sha(p)==h for p,h in hashes.items()),"source changed during build")
    ame.require(snapshot["identity"]==ame.tool_identity(tools.values()),"tools changed during build")
    ame.save(directory/"build.json",dict(status="BUILD_PASS",row=row,config=c,elf=str(elf),
        elf_sha256=ame.sha(elf),symbols=symbols,sources=hashes,generated=generated,
        tools=snapshot["hashes"],fixtures=meta,execution="NOT_RUN",compiler_profile_has_fixed_n=False))
    print("BUILD_PASS",row["key"],flush=True)

def run(artifact,platform,n,out):
    report=ame.load(artifact/"build.json")
    ame.require(all(ame.sha(p)==h for p,h in report["generated"].items()),"fixture evidence changed")
    # Safety reviews are separate from a successful build.
    gate=ame.load(artifact/"operator-safety.json")
    ame.require(gate["elf_sha256"]==report["elf_sha256"] and n in gate["allowed_n"],
                "missing matching operator frame/safety review")
    rc=ame.execute(report,platform,n,out)
    if rc: return rc
    result=ame.load(out/"result.json")
    expected=report["fixtures"]["checks_by_n"][str(n)]
    ame.require(result["checks"]==expected,"operator check-count mismatch")
    ame.save(out/"operator-result.json",dict(status="OPERATOR_PASS",operator=report["row"]["operator"],
        variant=report["row"]["variant"],
        logical_cases=report["fixtures"].get("cases_by_n", {}).get(str(n), report["fixtures"]["logical_cases"]),
        catalog_cases=report["fixtures"]["logical_cases"],
        exact_checks=expected,n=n,elf_sha256=report["elf_sha256"]))
    print("OPERATOR_PASS",report["row"]["key"],n,flush=True)
    return 0

def check(c,args):
    rows=[r for r in plan(c) if args.filter in r["key"]]
    ame.require(rows,"empty operator selection")
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    platform=ame.load(args.platform) if args.mode=="execute" else None
    if platform:
        ame.require(args.artifacts is not None,"execute requires preserved build artifacts")
        ame.require(all(r["xlen"]==platform["xlen"] and all(r["profile"][k]==platform[k]
            for k in ("uds","mregs","accregs")) for r in rows),"one profile/XLEN per stage")
    ame.save(out/"plan.json",dict(rows=rows,config=c,platform=platform,mode=args.mode))
    snap=ame.tool_snapshot(ame.tool_paths(c).values()); ame.save(out/"tools.json",snap)
    jobs=[]; count=min(2,len(rows))
    for shard in range(count):
        d=out/f"shard-{shard}"; d.mkdir()
        keys=[r["key"] for r in rows[shard::count]]
        (d/"keys.txt").write_text("\n".join(keys)+"\n")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",AME_ROOT=str(ame.ROOT),
            AME_PYTHON=sys.executable,AME_OUTPUT=str(d),AME_KEYS=str(d/"keys.txt"),
            AME_MODE=args.mode,AME_CONFIG=str(args.config.resolve()),
            AME_TOOLS_SNAPSHOT=str(out/"tools.json"))
        if platform:
            links=d/"inputs"/"build"; links.mkdir(parents=True)
            for key in keys:
                matches=list(args.artifacts.resolve().glob("shard-*/build/"+key))
                ame.require(len(matches)==1,"missing/duplicate operator artifact")
                (links/key).symlink_to(matches[0],target_is_directory=True)
            env.update(AME_ARTIFACTS=str(links.parent),AME_PLATFORM=str(args.platform.resolve()))
        log=(d/"runtest.log").open("xb")
        command=[str(Path(c["toolchain_root"])/c["dejagnu"]),"--tool","ameops",
                 "--srcdir",str(ame.ROOT/"testsuite"),"ameops.exp"]
        jobs.append((subprocess.Popen(command,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT),log,d,keys))
    reports=[]
    for proc,log,d,keys in jobs:
        rc=proc.wait(); log.close()
        text=(d/"ameops.sum").read_text()
        labels={("build "+key):{"PASS"} for key in keys} if not platform else {
            f"execute {key} N={n}":{"PASS"} for key in keys for n in platform["n_values"]}
        reports.append(dict(shard=str(d),exit_status=rc,ok=rc==0 and ame.check_result_labels(text,labels),
                            labels=len(labels),summary_sha256=ame.sha(d/"ameops.sum")))
    ok=all(r["ok"] for r in reports) and snap["identity"]==ame.tool_identity(ame.tool_paths(c).values())
    ame.save(out/"summary.json",dict(ok=ok,stages=reports,mode=args.mode))
    print(json.dumps(dict(ok=ok,stages=reports)),flush=True)
    return 0 if ok else 1

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["plan","build","run","check"])
    p.add_argument("--config",type=Path,default=ame.ROOT/"configs/operators/integer-r1.json")
    p.add_argument("--key"); p.add_argument("--out",type=Path)
    p.add_argument("--artifact",type=Path); p.add_argument("--artifacts",type=Path)
    p.add_argument("--platform",type=Path); p.add_argument("--n",type=int)
    p.add_argument("--mode",choices=["build","execute"],default="build")
    p.add_argument("--filter",default="")
    a=p.parse_args()
    try:
        c=ame.config(a.config)
        if a.action=="plan": print("\n".join(r["key"] for r in plan(c))); return 0
        ame.require(a.out is not None,"--out required")
        if a.action=="build":
            build(c,next(r for r in plan(c) if r["key"]==a.key),a.out.resolve()); return 0
        if a.action=="run":
            return run(a.artifact.resolve(),ame.load(a.platform),a.n,a.out.resolve())
        return check(c,a)
    except Exception as exc:
        print(str(exc),file=sys.stderr); return 1
if __name__=="__main__":
    sys.exit(main())
