# BF16 Input / FP32 Output Contract

GEMM is row-major NN, alpha=1, beta=0/1, RNE. GEMV is its one-column form with
unit vector stride. Profile: UDS32/M16/Acc4; runtime T=4,8,16,32,64,128.

## Storage

Inputs are native GCC `__bf16` objects (ame_bf16); output is FP32 float.
Leading dimensions count elements. Inputs require 2-byte alignment, output
4-byte alignment. Raw initialization/copy uses memcpy.

The workspace descriptor holds two BF16 arrays of at least 2*T*T elements
and one float array of at least T*T elements, each 16-byte aligned. Capacities
are element counts. Total payload is 12*T*T bytes; arrays may be separate.
Descriptor storage must be aligned/readable and disjoint from matrices and
workspace. Workspace regions are mutually disjoint and disjoint from matrices;
A/B can share storage, while C cannot overlap either.

A/B shapes are 1x2 and 2x1; K advances by 2*T. Memory layout is square-major.
Inactive output lanes repeat valid boundary rows/columns. K padding is +0/-0.
Seed/output is one FP32 square with accx1.

## Numeric and API Rules

The exact test domain uses representable BF16 inputs in 1/8 units and FP32
seed/output in 1/64 units, with scaled `abs(seed)+sum(abs(products)) <= 2**24-1`.
The reference checks the bound; the kernel relies on it. Comparison is exact
bits, with existing flags/xsat preserved and no new exceptions.

Ownership, errors, overlap and zero dimensions follow [API](../../docs/API.md).
Tests cover tile/square/K boundaries, multiple packets, fractions, cancellation,
strides and complete input/output/workspace guards. Broader numerical cases use
[BF16_RNE_FOLD_V1.md](BF16_RNE_FOLD_V1.md); state/error cases use
[BF16_STATE_V1.md](BF16_STATE_V1.md).
