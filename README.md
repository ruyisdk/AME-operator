# AME Operator

RISC-V AME/Ztt GEMM and GEMV kernels, independent numerical references, and
DejaGNU/Spike validation scripts.

## Supported Operators

| Operator | Input | Accumulator / output | Layout |
| --- | --- | --- | --- |
| GEMM, GEMV | INT32 | INT32 | Row-major NN |
| GEMM, GEMV | INT8 | INT32 | Row-major NN |
| GEMM, GEMV | UINT8 | INT32 | Row-major NN |
| GEMM, GEMV | FP32 | FP32 | Row-major NN, RNE |
| GEMM, GEMV | BF16 | FP32 | Row-major NN, RNE |

GEMM computes `C = A * B` or `C += A * B`. GEMV computes `y = A * x` or
`y += A * x`, using unit vector stride. Both use alpha=1 and beta=0/1.
The compiled resource profile is UDS32/M16/Acc4; the tile side `T` is queried
at runtime and supports 4, 8, 16, 32, 64 and 128. GEMV shares the single-column
GEMM kernel. Interfaces use ordinary pointers and scalars in C and C++.

## Quick Start

Requirements: Linux, Python 3.11+, host C/C++ compilers, DejaGNU/Expect,
AME-enabled GCC and Binutils build trees, and an AME-enabled Spike.
See [platform requirements](docs/PLATFORM.md) for the ISA, CSR and model contract.

Run from the repository root. Replace the four toolchain paths with your build
locations; `--gcc-build` points to the directory containing `xgcc` and `xg++`.

```sh
python3 scripts/configure.py \
  --gcc-build /path/to/gcc-build/gcc \
  --binutils-build /path/to/binutils-build \
  --gcc-include /path/to/gcc/gcc/config/riscv \
  --runtest /path/to/dejagnu/runtest

python3 -m unittest discover -s operators/tests -p 'test_*.py'

python3 scripts/prepare_spike.py \
  --source /path/to/spike-source --spike /path/to/spike-build/spike \
  --config build/config/state-edges-r1.json --out build/model

python3 scripts/run_suite.py \
  --config build/config/state-edges-r1.json \
  --model build/model/model.json --mode smoke --out results/state-smoke
```

The smoke command builds 16 ELFs and executes 32 runs: two operators, RV32/RV64,
C/C++, O0/O2, at T=4/8. This suite calls all five data paths. Successful output
ends with `SMOKE_PASS 16 ELFs 32 executions`.

Use `--mode full` for six tile sizes plus T4/T128 probes, or `--mode build`
without `--model` for compile/link and frame checks. Each output directory must
be new. Logs and generated artifacts stay under ignored `build/` and `results/`.

## Documentation

- [Build and run](docs/BUILD_AND_RUN.md): numerical suites, negative controls,
  unsupported-platform tests and troubleshooting.
- [Validation](docs/VALIDATION.md): GOLDEN construction, coverage axes and result formats.
- [API](docs/API.md): ownership, workspace, argument handling and numerical contracts.
- [Platform](docs/PLATFORM.md): toolchain/model requirements and execution adapter.
- [Release validation](docs/RELEASE_VALIDATION.md): validation performed for this export.
- [Adding an operator](docs/ADDING_OPERATOR.md): kernel, reference and registry integration.

## Layout

```text
operators/include/     Public C/C++ interfaces
operators/common/      Argument, workspace and platform checks
operators/kernels/     AME kernels and packing
operators/reference/   Independent logical GOLDEN generators
operators/cases/       Versioned input catalogs
operators/tests/       Target drivers and host unit tests
operators/docs/        Numerical and state contracts
configs/operators/     Portable suite and platform templates
runtime/               Bare-metal startup, linker script and mailbox
reference/             Integer-based floating reference helpers
scripts/               Configuration, build, execution and verification
testsuite/             DejaGNU drivers
```

MIT license. See [LICENSE](LICENSE).
