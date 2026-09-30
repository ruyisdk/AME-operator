#!/usr/bin/env python3
"""Recheck operator ELF, fixture, mailbox and exact DejaGNU execution inventory."""
import argparse
from collections import defaultdict
from pathlib import Path
import struct
import ame
import audit_operator_state
import audit_operator_numeric
import audit_operator_widen8
import audit_operator_widen8_state
import audit_operator_fp32
import audit_operator_fp32_state
import audit_operator_fp32_numeric
import audit_operator_bf16
import audit_operator_bf16_state
import audit_operator_bf16_numeric
import audit_operator_state_edges

def selected_cases(build, golden, n):
    variant = build["row"]["variant"]
    if variant == "i32-numeric-v4":
        return audit_operator_numeric.selected(build,golden,n)
    if variant == "i32-exact-v1":
        return golden, 8
    ame.require(variant == "i32-geometry-v2", "unreviewed operator variant")
    op = build["row"]["operator"]
    ame.require(op in ("gemm", "gemv"), "unreviewed geometry operator")
    selected = [r for r in golden if r["tile"] == n]
    wanted = set()
    for axis in (("m", "n", "k") if op == "gemm" else ("m", "k")):
        for label, value in (("minus", n-1), ("exact", n),
                             ("plus", n+1), ("double-plus", 2*n+1)):
            shape = dict(m=3, n=5 if op == "gemm" else 1, k=5)
            shape[axis] = value
            for add in (0, 1):
                wanted.add((axis+"-"+label, add, shape["m"], shape["n"], shape["k"]))
    for add in (0, 1):
        wanted.add(("cross", add, n+1, n-1 if op == "gemm" else 1, n+1))
        wanted.add(("long-k", add, 5, 3 if op == "gemm" else 1, 3*n+1))
    actual = {(r["purpose"], r["accumulate"], r["m"], r["n"], r["k"]) for r in selected}
    ame.require(actual == wanted and len(actual) == len(selected), "geometry coverage gap/duplicate")
    for r in selected:
        ame.require(r["lda"] == r["k"]+2 and r["ldb"] == (r["n"]+3 if op == "gemm" else 1)
                    and r["ldc"] == (r["n"]+4 if op == "gemm" else 1), "stride coverage changed")
    ame.require(len(selected) == build["fixtures"]["cases_by_n"][str(n)], "wrong selected case count")
    return selected, 9

def verify_stage(stage, builds):
    plan=ame.load(stage/"plan.json"); platform=plan["platform"]
    expected={(row["key"],n) for row in plan["rows"] for n in platform["n_values"]}
    records=[]; seen=set(); groups=defaultdict(list)
    for response_file in sorted(stage.glob("shard-*/execute/*/n*/response.json")):
        directory=response_file.parent; key=directory.parent.name; n=int(directory.name[1:])
        ame.require((key,n) in expected and (key,n) not in seen,"unexpected/duplicate execution")
        seen.add((key,n)); b=builds[key]; request=ame.load(directory/"request.json")
        response=ame.load(response_file)
        ame.require(request["elf_sha256"]==b["elf_sha256"] and request["elf"]==b["elf"],"ELF request mismatch")
        required=dict(xlen=b["row"]["xlen"],n=n,**b["row"]["profile"],
                      csr_map=ame.CSR_MAP,entry_contract=ame.ENTRY)
        ame.require(request["expected"]==required,"wrong requested configuration")
        ame.require(request["revision"]==platform["revision"] and request["backend"]==platform["backend"],
                    "wrong requested backend")
        actual=ame.validate_response(response,request,b,n)
        golden=ame.load(Path(b["elf"]).parent/"golden.json")
        if b["row"]["variant"] == "state-edges-all-v1":
            selected,count = audit_operator_state_edges.accounting(b,golden,n)
            audit_operator_state_edges.verify_execution(b,directory,response)
            catalog=len(golden['cases'])
        elif b["row"]["variant"] == "i32-state-v3":
            selected,count = audit_operator_state.accounting(b,golden,n)
            catalog=len(golden["cases"])
        elif b["row"]["variant"] == "widen8-state-v2":
            selected,count = audit_operator_widen8_state.accounting(b,golden,n)
            catalog=len(golden["cases"])
        elif b["row"]["variant"] == "widen8-i32-v1":
            selected,count = audit_operator_widen8.accounting(b,golden,n)
            catalog=len(golden)
        elif b["row"]["variant"] == "fp32-state-v3":
            selected,count = audit_operator_fp32_state.accounting(b,golden,n)
            audit_operator_fp32_state.verify_execution(b,directory,response)
            catalog=len(golden['cases'])
        elif b["row"]["variant"] == "fp32-exact-v1":
            selected,count = audit_operator_fp32.accounting(b,golden,n)
            catalog=len(golden)
        elif b["row"]["variant"] == "bf16-f32-exact-v1":
            selected,count = audit_operator_bf16.accounting(b,golden,n)
            audit_operator_bf16.verify_execution(b,directory,response)
            catalog=len(golden)
        elif b["row"]["variant"] == "bf16-state-v1":
            selected,count = audit_operator_bf16_state.accounting(b,golden,n)
            audit_operator_bf16_state.verify_execution(b,directory,response)
            catalog=len(golden['cases'])
        elif b["row"]["variant"] == "bf16-f32-rne-fold-v1":
            selected,count = audit_operator_bf16_numeric.accounting(b,golden,n)
            audit_operator_bf16_numeric.verify_execution(b,directory,response)
            catalog=len(golden)
        elif b["row"]["variant"] == "fp32-rne-fold-v1":
            selected,count = audit_operator_fp32_numeric.accounting(b,golden,n)
            audit_operator_fp32_state.verify_execution(b,directory,response)
            catalog=len(golden)
        else:
            selected, driver_checks = selected_cases(b, golden, n)
            # Count independently from the public target loop, not its generated metadata.
            maximum=b["config"]["max_n"]; width=max(len(r["initial"]) for r in golden)
            count=6+4*(maximum**2+32)+driver_checks
            count+=sum(1+width+32+3*maximum**2+32-3*n*n+len(r["a"])+len(r["b"]) for r in selected)
            catalog=len(golden)
        ame.require(actual["checks"]==count==b["fixtures"]["checks_by_n"][str(n)],"wrong check count")
        done=ame.load(directory/"operator-result.json")
        ame.require(done["status"]=="OPERATOR_PASS" and done["exact_checks"]==count
                    and done["logical_cases"]==len(selected) and done["n"]==n
                    and done["elf_sha256"]==b["elf_sha256"],"incomplete operator result")
        words=list(struct.unpack("<32I",bytes.fromhex(response["mailbox_hex"])))
        normalized=list(words); normalized[4]=0
        groups[(b["row"]["operator"],n)].append(normalized)
        records.append(dict(key=key,n=n,logical_cases=len(selected),catalog_cases=catalog,checks=count,
            elf_sha256=b["elf_sha256"],response_sha256=ame.sha(response_file),
            request_sha256=ame.sha(directory/"request.json")))
    ame.require(seen==expected,"missing executions")
    for d in sorted(stage.glob("shard-*")):
        keys=(d/"keys.txt").read_text().splitlines()
        labels={f"execute {key} N={n}":{"PASS"} for key in keys for n in platform["n_values"]}
        ame.require(ame.check_result_labels((d/"ameops.sum").read_text(),labels),"DejaGNU inventory mismatch")
    for name,rows in groups.items():
        ame.require(all(r==rows[0] for r in rows),"cross-optimization mailbox difference")
    return dict(executions=len(records),logical_case_executions=sum(r["logical_cases"] for r in records),
                records=records),groups

def main():
    p=argparse.ArgumentParser(); p.add_argument("root",type=Path); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    builds={}
    for path in sorted((a.root/"build").glob("shard-*/build/*/build.json")):
        b=ame.load(path); key=b["row"]["key"]
        ame.require(key not in builds,"duplicate build")
        ame.require(ame.sha(b["elf"])==b["elf_sha256"],"ELF changed")
        ame.require(all(ame.sha(p)==h for p,h in b["generated"].items()),"generated input changed")
        for source,sha in b["sources"].items():
            if Path(source).is_relative_to(ame.ROOT):
                archived=path.parent/"source"/Path(source).relative_to(ame.ROOT)
                ame.require(ame.sha(archived)==sha,"source archive changed")
        gate=ame.load(path.parent/"operator-safety.json")
        ame.require(gate["elf_sha256"]==b["elf_sha256"],"wrong safety ELF")
        builds[key]=b
    ame.require(set(builds)=={r["key"] for r in ame.load(a.root/"build/plan.json")["rows"]},"missing build inventory")
    stages={}; combined=defaultdict(list)
    for name in ("probe-rv32","probe-rv64","main-rv32","main-rv64"):
        plan = ame.load(a.root/name/"plan.json")
        wanted_n = [4, 8, 16, 32, 64, 128] if name.startswith("main") else [4, 128]
        ame.require(plan["platform"]["n_values"] == wanted_n, "incomplete runtime N matrix")
        report,groups=verify_stage(a.root/name,builds); stages[name]=report
        if name.startswith("main"):
            for key,rows in groups.items(): combined[key]+=rows
    for key,rows in combined.items():
        expected = sum(b["row"]["operator"] == key[0] for b in builds.values())
        ame.require(len(rows)==expected and all(v==rows[0] for v in rows),
                    "missing/disagreeing XLEN/optimization/language comparison")
    ame.require(set(combined) == {(b["row"]["operator"], n) for b in builds.values()
                                 for n in (4, 8, 16, 32, 64, 128)}, "missing operator/N group")
    result=dict(status="SELECTED_MATRIX_PASS",builds=len(builds),stages=stages,
        main_cross_path_groups=len(combined),reference_scope="variant-defined numerical domain and cases",
        audit_sha256=ame.sha(Path(__file__)))
    ame.save(a.out/"audit.json",result)
    print("OPERATOR_AUDIT_PASS",len(builds),sum(v["executions"] for k,v in stages.items() if k.startswith("main")),
          "main",sum(v["executions"] for k,v in stages.items() if k.startswith("probe")),"probes")
if __name__=="__main__": main()
