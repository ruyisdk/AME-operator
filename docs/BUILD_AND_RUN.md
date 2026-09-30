# Build and Run

## 1. Configure Tools

Follow the configuration command in the [README](../README.md). The script
validates the compiler, assembler, linker, inspection tools, header and
`runtest`, then writes resolved configs under `build/config/`. It leaves the
templates in `configs/operators/` unchanged. Python 3.11+ is required for
`hashlib.file_digest`; host reference tests also invoke `cc` and `c++`.

The GCC build must provide `xgcc`, `xg++`, `cc1`, `cc1plus` and accept
`-mztt-profile=gcc-runtime-u32-m16-a4`. Binutils must recognize `ztt0p6`.
Bare-metal tests link without libc or a target OS.

## 2. Prepare Spike

```sh
python3 scripts/prepare_spike.py \
  --source /path/to/spike-source --spike /path/to/spike-build/spike \
  --config build/config/state-edges-r1.json --out build/model
```

This copies the supplied binary, records its SHA256 and source revision,
checks the CSR map, and builds RV32/RV64 bootstraps. For an exported source
tree without `.git`, supply `--revision <source-revision-and-patch-id>`.
Keep the binary and its matching source together. The generated `model.json`
contains local paths and is not committed.

## 3. Run Host Tests

```sh
python3 -m unittest discover -s operators/tests -p 'test_*.py'
python3 -m unittest discover -s scripts -p 'test_*.py'
```

These tests check shape validation, packing, independent GOLDEN, state
accounting and error detection. Reference comparisons use host UBSan builds.

## 4. Run Numerical and State Suites

Start with the README smoke command, then run the full suites below. Each
invocation builds 16 ELFs, performs frame checks, executes 32 probes and 96 main
runs, then verifies all response and DejaGNU inventories.

```sh
python3 scripts/run_suite.py --config build/config/integer-numeric-r4.json \
  --model build/model/model.json --mode full --out results/integer-numeric
python3 scripts/run_suite.py --config build/config/integer-widen8-r5.json \
  --model build/model/model.json --mode full --out results/widen8
python3 scripts/run_suite.py --config build/config/fp32-numeric-r4.json \
  --model build/model/model.json --mode full --out results/fp32-numeric
python3 scripts/run_suite.py --config build/config/bf16-numeric-r7.json \
  --model build/model/model.json --mode full --out results/bf16-numeric
python3 scripts/run_suite.py --config build/config/state-edges-r1.json \
  --model build/model/model.json --mode full --out results/state-edges
```

Additional configs cover geometry, per-family errors/state and exactly
representable floating inputs. List their build cases with:

```sh
python3 scripts/operators.py plan --config build/config/integer-geometry-r2.json
```

`run_suite.py` uses two DejaGNU workers. Commands have a 120-second timeout and
4-GiB address-space limit per process. Run suites sequentially on constrained
machines. Compiler or simulator failure stops the suite with a nonzero exit.

## 5. Check Unsupported Platforms

This lane verifies rejection on mismatched UDS and tile sizes, separately
from normal numerical execution. It also checks that typed kernels are not
entered on selected rejected paths.

```sh
python3 scripts/operators.py check --config build/config/platform-guards-r1.json \
  --out results/platform-guards/build
python3 scripts/audit_operator_frames.py results/platform-guards/build
for xlen in 32 64; do
  python3 scripts/operator_platform_guards.py check \
    --config build/config/platform-guards-r1.json \
    --artifacts results/platform-guards/build --model build/model/model.json \
    --xlen "$xlen" --probe --out "results/platform-guards/probe-rv$xlen"
  python3 scripts/operator_platform_guards.py check \
    --config build/config/platform-guards-r1.json \
    --artifacts results/platform-guards/build --model build/model/model.json \
    --xlen "$xlen" --out "results/platform-guards/main-rv$xlen"
done
python3 scripts/audit_operator_platform_guards.py results/platform-guards \
  --out results/platform-guards/audit
```

## 6. Verify Deliberate Fault Detection

Run a full geometry parent first, then derive controlled ELF mutations:

```sh
python3 scripts/run_suite.py --config build/config/integer-geometry-r2.json \
  --model build/model/model.json --mode full --out results/geometry
python3 scripts/operator_controls.py prepare --base results/geometry \
  --out results/geometry-controls
python3 scripts/operator_controls.py check --root results/geometry-controls --xlen 32
python3 scripts/operator_controls.py check --root results/geometry-controls --xlen 64
python3 scripts/audit_operator_controls.py results/geometry-controls \
  --out results/geometry-controls/audit
```

Deliberately mutated executions retain their raw `FAIL` labels. The separate
verification must match the predicted error, while positive copies must pass.
The final command succeeds only when the expected inventory matches.

FP32 state controls use the full `results/fp32-numeric` parent from section 4:

```sh
python3 scripts/operator_state_controls.py prepare --base results/fp32-numeric \
  --out results/fp32-state-controls
python3 scripts/operator_state_controls.py check --root results/fp32-state-controls --xlen 32
python3 scripts/operator_state_controls.py check --root results/fp32-state-controls --xlen 64
python3 scripts/operator_state_controls.py audit --root results/fp32-state-controls \
  --out results/fp32-state-controls/audit
```

## 7. Count Generated Instructions

```sh
python3 scripts/operator_static_cost.py results/state-edges/build \
  --out results/static-cost
```

`census.json` and `SUMMARY.md` report code size, instruction categories and
runtime frame sizes for all five kernels across the 16 linked ELFs.

## Results and Troubleshooting

Start with `<out>/SUMMARY.json`. Full runs also write `audit/audit.json`.
Compiler commands, disassembly, generated fixtures and ELF hashes are stored
in `build/shard-*/build/<key>/`. Simulator logs, requests, responses and results
are stored in `<stage>/shard-*/execute/<key>/n<T>/`.

| Symptom | Check |
| --- | --- |
| Missing `xgcc`, `cc1plus` or `riscv_ztt.h` | Use matching GCC source/build paths |
| Unknown Ztt option or mnemonic | Use AME-enabled GCC and Binutils |
| CSR map mismatch | Align the model with 0x810/0x811/0x812 |
| Spike observations incomplete | Check AME CLI/debug support and `spike.log` |
| Expected/actual output or flag mismatch | Preserve response, raw log, GOLDEN and model identity |
| Frame review rejects new instruction sequence | Inspect generated prologue/epilogue before updating the checker |
| Output directory already exists | Choose a new output directory |

The bare-metal linker currently emits an RWX segment warning. Generated AME
C++ code can emit unused-operand warnings. Both remain in build logs.
