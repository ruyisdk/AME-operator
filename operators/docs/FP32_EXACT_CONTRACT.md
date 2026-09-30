# FP32 Exact-Dyadic Contract

FP32 GEMM/GEMV uses row-major NN, alpha=1, beta=0/1, RNE and UDS32/M16/Acc4.
Tile side T is runtime 4,8,16,32,64,128.

## Numeric Domain

A/B are exactly representable FP32 values in integer units of 1/8; seed/output
use units of 1/64. For every output, the scaled integer bound is
`abs(seed_units)+sum(abs(A_units*B_units)) <= 2**24-1`. Every product and partial
sum is then exact FP32 normal or zero under any deterministic term order.
The reference enforces this bound; comparison uses exact bits. Computing
these inputs adds no AME flags or UN, and preserves prior flags and xsat.

The kernel relies on the caller's numeric domain. Exceptional and nonexact
data are tested under [FP32_RNE_FOLD_V1.md](FP32_RNE_FOLD_V1.md).

## Storage and Tests

A/B/C are real float objects, aligned to 4 bytes. Workspace holds three live
float tiles, 16-byte aligned, totaling 12*T*T bytes. Packing and raw bit checks
use memcpy. Output/workspace overlap rules and ordinary-call ownership follow
[API](../../docs/API.md).

Tests cover tile-relative dimensions, long K, fractions, cancellation, bounds,
set/add and strides. Inputs, outputs and workspaces have guards. K=0 assignment
writes +0; accumulation preserves exact seed bits. State and error coverage is
specified in [FP32_STATE_V3.md](FP32_STATE_V3.md).
