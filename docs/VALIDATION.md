# Validation

## GOLDEN

Expected results are generated from logical matrix coordinates, independently
of AME packing and Spike output.

| Domain | Reference | Comparison |
| --- | --- | --- |
| INT32 / INT8 / UINT8 | Python unbounded integer dot products, checked INT32 bounds | Exact words |
| Exact FP32 / BF16 | Scaled integer arithmetic in 1/8 and 1/64 units | Exact bits and flags |
| FP32 RNE | Rational FP32 FMA fold in increasing K order | Exact bits and flags |
| BF16 to FP32 RNE | BF16 bit decoding, FP32 FMA fold, signaling-NaN conversion flags | Exact bits and flags |

The floating references are cross-checked with independent C integer-limb
implementations under UBSan. Hand-computed witnesses cover cancellation,
rounding ties, subnormals, NaNs, infinity, signed zero and sticky exceptions.
The exact-domain tier is order independent under its magnitude bound.
The RNE tier selects increasing-K fused FP32 accumulation; the execution
platform must use that order and intermediate format.

## Coverage Axes

Each normal suite combines GEMM/GEMV, RV32/RV64, C/C++ and O0/O2 into 16 ELFs.
The same ELF executes with T=4,8,16,32,64,128. A full run has 32 T4/T128 probe
executions and 96 main executions. A smoke run has 32 T4/T8 executions.

Case catalogs include rectangular dimensions, tile boundaries, cross tails,
long K, set/add, padded leading dimensions, zero dimensions, invalid inputs,
workspace bounds, input/output guards and ownership transitions. Dedicated
state tests check UN, AME flags, saturation state and standard FCSR. The
platform guard suite checks actual mismatched UDS/T separately.

## PASS Conditions

1. Compiler and linker succeed; Ztt attributes and expected instructions appear.
2. Frame analysis validates runtime sizes and stack/data/bootstrap boundaries.
3. The model reports the requested XLEN, tile size, resource profile and CSR map.
4. The executable reaches the completion marker with a valid mailbox identity,
   fresh nonce, expected ELF hash and exact assertion count.
5. Numerical outputs, protected memory and required state match their references.
6. DejaGNU contains exactly the expected labels with no missing/duplicate results.
7. Full-run verification checks all six T values and cross-language/optimization/
   XLEN agreement for each operator.

`BUILD_PASS` reports build/frame checks; `SMOKE_PASS` reports T4/T8 execution;
`FULL_PASS` reports the selected full suite. Missing execution environments
remain build-only results. Deliberate fault results are counted separately
from normal passing executions.

## Artifacts

The runner preserves configs, compiler/tool hashes, copied test sources,
generated fixtures, GOLDEN, ELF/disassembly, model identity, raw debug output,
mailbox responses, DejaGNU summaries and a final `CHECKPOINT.json` digest list.
Keep the output directory when reporting a mismatch. Large build/model/log
artifacts are excluded from the source repository.

Validation is functional. Static instruction counts are available through
`scripts/operator_static_cost.py`; cycle and throughput measurements require
a platform-specific benchmark setup. SAIL integration uses the request/response
contract described in [PLATFORM.md](PLATFORM.md).
