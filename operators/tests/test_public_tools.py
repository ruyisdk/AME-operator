import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import ame
import configure
import operators
import run_suite


class PublicTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "paths with spaces"
        self.root.mkdir()
        self.config = ROOT / "configs/operators/state-edges-r1.json"

    def tools(self):
        c = dict(toolchain_root=str(self.root), gcc_build="gcc", binutils_build="binutils")
        for p in list(ame.tool_paths(c).values()) + [self.root / "runtest"]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("#!/bin/sh\nexit 0\n")
            p.chmod(0o755)
        include = self.root / "include"
        include.mkdir()
        (include / "riscv_ztt.h").write_text("/* test header */\n")
        return self.root / "gcc", self.root / "binutils", include, self.root / "runtest"

    def model(self):
        binary = self.root / "spike"
        binary.write_bytes(b"model fixture")
        boot = {}
        for xlen in (32, 64):
            path = self.root / f"boot{xlen}.elf"
            path.write_bytes(f"boot {xlen}".encode())
            boot[str(xlen)] = dict(path=str(path), sha256=ame.sha(path))
        path = self.root / "model.json"
        ame.save(path, dict(csr_map=ame.CSR_MAP, revision="test-fixture",
                           spike=str(binary), spike_sha256=ame.sha(binary), boot=boot))
        return path

    def test_resolve_tools_and_preserve_template(self):
        before = self.config.read_bytes()
        c = configure.resolve(self.config, *self.tools())
        self.assertEqual(c["toolchain_root"], "/")
        self.assertTrue(all(p.is_absolute() for p in ame.tool_paths(c).values()))
        self.assertEqual(len(operators.plan(c)), 16)
        self.assertEqual(self.config.read_bytes(), before)

    def test_missing_compiler(self):
        args = self.tools()
        (args[0] / "cc1plus").unlink()
        with self.assertRaisesRegex(ValueError, "missing executable cc1plus"):
            configure.resolve(self.config, *args)

    def test_missing_header(self):
        args = self.tools()
        (args[2] / "riscv_ztt.h").unlink()
        with self.assertRaisesRegex(ValueError, "riscv_ztt.h"):
            configure.resolve(self.config, *args)

    def test_platform_and_fp_observation(self):
        rows = operators.plan(ame.config(self.config))
        p = run_suite.platform(rows, self.model(), 32, [4, 8])
        self.assertEqual(p["n_values"], [4, 8])
        self.assertEqual(p["xlen"], 32)
        self.assertEqual(Path(p["command"][0]).name, "spike_fpstate_adapter.py")
        self.assertTrue(Path(p["command"][0]).is_absolute())
        self.assertEqual(p["revision"], "test-fixture")

    def test_changed_model(self):
        model = self.model()
        (self.root / "spike").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model binary changed"):
            run_suite.platform(operators.plan(ame.config(self.config)), model, 32, [4])

    def test_wrong_csr_map(self):
        model = self.model()
        c = ame.load(model)
        c["csr_map"] = "old-map"
        ame.save(model, c)
        with self.assertRaisesRegex(ValueError, "CSR map mismatch"):
            run_suite.platform(operators.plan(ame.config(self.config)), model, 64, [4])

    def test_build_only_summary(self):
        out = self.root / "build-only"
        with patch.object(sys, "argv", ["run_suite", "--config", str(self.config),
                                        "--mode", "build", "--out", str(out)]), \
             patch.object(run_suite, "invoke") as invoke:
            self.assertEqual(run_suite.main(), 0)
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(ame.load(out / "SUMMARY.json"),
                         dict(status="BUILD_PASS", builds=16, executions=0))

    def test_missing_model_rejected_before_output(self):
        out = self.root / "no-model"
        with patch.object(sys, "argv", ["run_suite", "--config", str(self.config),
                                        "--out", str(out)]):
            with self.assertRaisesRegex(ValueError, "requires --model"):
                run_suite.main()
        self.assertFalse(out.exists())

    def test_filter_leading_dash(self):
        p = subprocess.run([sys.executable, str(ROOT / "scripts/operators.py"), "plan",
                            "--config", str(self.config), "--filter=-rv32-"],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len(p.stdout.splitlines()), 16)
