# Export Validation: 2026-09-30

Validation was run from this standalone operator repository using the rebased
AME toolchain and the BF16 NV r2 Spike model.

| Suite | Builds | Spike executions | Tile sizes | Result |
| --- | --- | --- | --- | --- |
| Five-family state integration | 16 | 128 | 4/8/16/32/64/128, plus T4/T128 probes | PASS |
| INT32 numerical smoke | 16 | 32 | 4/8 | PASS |
| INT8/UINT8 to INT32 smoke | 16 | 32 | 4/8 | PASS |
| FP32 numerical smoke | 16 | 32 | 4/8 | PASS |
| BF16 to FP32 numerical smoke | 16 | 32 | 4/8 | PASS |
| Host unit tests | 314 tests | n/a | n/a | PASS |
| Floating-reference unit tests | 15 tests | n/a | Integer/rational references | PASS |
| Clean-export host tests | Same 314 tests | n/a | Relocated source directory | PASS |

Every target suite includes GEMM/GEMV, RV32/RV64, C/C++, and O0/O2. Total:
80 ELF builds and 256 Spike executions. Frame checks passed for all 80 ELFs.
Host reruns are the same test inventory, not additional coverage counts.
Numerical smoke here covers T4/T8; six-T numerical commands, platform rejection
and deliberate-fault workflows are provided in [BUILD_AND_RUN.md](BUILD_AND_RUN.md).

## Revisions

- GCC: `5d04d3f40ce48c99f91eb0965aaf6a83ac22c252`.
- Binutils: `308ef3af1dcc309392997c20a4b3794679bfcce7`.
- Spike: `ead03459a3d9db96788d2d2fe626d0b9f3bcd367+local-cb7a408bb8b2`.
- Spike binary SHA256:
  `20b892a23129d3e6aa092ddb9cd5c5c042199cceace7076e546bd55b6bef94ff`.

The 27 production header/validator/kernel files are byte-identical to the
pre-export operator implementation. Export changes concern configuration,
script entry points, dependency extraction, tests and documentation.
The existing MIT LICENSE is unchanged. Source, tool and result hashes are in
[validation/20260930.json](validation/20260930.json).

## Commands

After configuring tools and preparing the model as in the README:

```sh
python3 -m unittest discover -s operators/tests -p 'test_*.py'
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/run_suite.py --config build/config/state-edges-r1.json \
  --model build/model/model.json --mode full --out results/state-edges
python3 scripts/run_suite.py --config build/config/integer-numeric-r4.json \
  --model build/model/model.json --mode smoke --out results/integer-numeric
python3 scripts/run_suite.py --config build/config/integer-widen8-r5.json \
  --model build/model/model.json --mode smoke --out results/widen8
python3 scripts/run_suite.py --config build/config/fp32-numeric-r4.json \
  --model build/model/model.json --mode smoke --out results/fp32-numeric
python3 scripts/run_suite.py --config build/config/bf16-numeric-r7.json \
  --model build/model/model.json --mode smoke --out results/bf16-numeric
```

The clean-export host run uses only Git-staged files extracted into a temporary
directory. Local tool configurations, model binaries and previous result
directories are absent from that export.
