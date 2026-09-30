#!/usr/bin/env python3
"""Build, execute and verify an operator suite through DejaGNU."""
import argparse
from pathlib import Path
import subprocess
import sys
import ame
import operators
from audit_operator_runs import verify_stage


def platform(rows, model_path, xlen, ns):
    model = ame.load(model_path)
    ame.require(model["csr_map"] == ame.CSR_MAP, "model CSR map mismatch")
    ame.require(ame.sha(model["spike"]) == model["spike_sha256"], "model binary changed")
    for bits in (32, 64):
        boot = model["boot"][str(bits)]
        ame.require(ame.sha(boot["path"]) == boot["sha256"], "bootstrap changed")
    profiles = {tuple(sorted(r["profile"].items())) for r in rows}
    ame.require(len(profiles) == 1, "one resource profile per suite")
    use_fcsr = any("scripts/spike_fpstate_adapter.py" in r["definition"]["inputs"] for r in rows)
    adapter = "spike_fpstate_adapter.py" if use_fcsr else "spike_adapter.py"
    # These tokens select the required test contract; every execution is checked.
    caps = sorted({cap for r in rows for cap in r["case"]["capabilities"]})
    return dict(schema=1, backend="spike", revision=model["revision"],
                command=[str(ame.ROOT / "scripts" / adapter), "--model", str(model_path)],
                timeout_seconds=120, csr_map=ame.CSR_MAP, xlen=xlen,
                **dict(next(iter(profiles))), n_values=ns, capabilities=caps,
                entry_contract=ame.ENTRY)


def invoke(script, *args):
    subprocess.run([sys.executable, str(ame.ROOT / "scripts" / script),
                    *map(str, args)], cwd=ame.ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--mode", choices=("build", "smoke", "full"), default="smoke")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    config = a.config.resolve()
    c = ame.config(config)
    rows = operators.plan(c)
    ame.require(set(c["xlens"]) == {32, 64}, "suite requires both XLENs")
    ame.require(c["variant"] not in ("platform-guards-v1", "fp32-tail-probe-v1"),
                "use the dedicated guard/tail commands for this variant")
    if a.mode != "build":
        ame.require(a.model is not None, "execution requires --model")
        a.model = a.model.resolve()
        platform(rows, a.model, 32, [4, 8])
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ame.save(out / "config.json", c)
    config = out / "config.json"
    invoke("operators.py", "check", "--config", config, "--out", out / "build")
    invoke("audit_operator_frames.py", out / "build")
    if a.mode == "build":
        ame.save(out / "SUMMARY.json", dict(status="BUILD_PASS", builds=len(rows), executions=0))
        return 0
    stages = {}
    builds = {b["row"]["key"]: b for b in (
        ame.load(path) for path in (out / "build").glob("shard-*/build/*/build.json"))}
    phases = [("smoke", [4, 8])] if a.mode == "smoke" else [
        ("probe", [4, 128]), ("main", [4, 8, 16, 32, 64, 128])]
    for phase, ns in phases:
        for xlen in (32, 64):
            name = f"{phase}-rv{xlen}"
            conf = out / f"{name}.json"
            ame.save(conf, platform(rows, a.model, xlen, ns))
            invoke("operators.py", "check", "--config", config, "--mode", "execute",
                   f"--filter=-rv{xlen}-", "--artifacts", out / "build",
                   "--platform", conf, "--out", out / name)
            report, _ = verify_stage(out / name, builds)
            stages[name] = report
    if a.mode == "full":
        invoke("audit_operator_runs.py", out, "--out", out / "audit")
    result = dict(status=a.mode.upper() + "_PASS", builds=len(builds),
                  executions=sum(s["executions"] for s in stages.values()),
                  variant=c["variant"], stages=stages)
    ame.save(out / "SUMMARY.json", result)
    evidence = [path for path in out.rglob("*.json") if path.name != "CHECKPOINT.json"]
    ame.save(out / "CHECKPOINT.json", dict(model=ame.load(a.model), artifacts={
        str(path.relative_to(out)): ame.sha(path) for path in evidence}))
    print(result["status"], result["builds"], "ELFs", result["executions"], "executions")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
