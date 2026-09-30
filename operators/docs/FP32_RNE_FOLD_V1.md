# FP32 RNE Numerical Contract

This variant uses the FP32 pointer API, row-major NN, alpha=1, beta=0/1,
UDS32/M16/Acc4 and runtime T=4,8,16,32,64,128.

The selected execution contract is an increasing-K fold of per-term fused,
once-rounded FP32 `a*b+old` under RNE. Assignment starts from +0 without reading
old output; accumulation starts from its exact bit image. ISA v0.6 permits a
deterministic implementation-defined term permutation, so a backend with a
different accumulation order needs a corresponding reference contract.

Results use canonical quiet NaN, signaling/invalid NV, overflow OF|NX,
after-precision-rounding tininess, sticky flags and RNE signed-zero rules.
Subnormals are preserved. Standard FCSR, ownership and xsat remain unchanged.
Padding duplicates valid boundary outputs; K padding uses left +0/right -0.

## GOLDEN and Coverage

`fp32_fold.py` walks logical matrices and calls the Fraction-based
`ieee_tininess.fp32_fused_expected`. It uses neither Spike nor host floating
arithmetic. Tests cross-check the C integer-limb `reference/fused32-r2.c`
implementation under UBSan, including precision-carry tininess witnesses.

Cases cover ties, fused cancellation, overflow rescue, subnormals, NaN/Inf,
signed zero, seed/order sensitivity and intermediate sticky flags. Cross-block
witnesses occur at T-1 and 2*T-1; mixed output matrices exercise both tails.
Target tests compare all output bits, new|old flags, memory guards and FCSR.
Standard F is enabled for observing FCSR; kernels compile for im_zicsr_ztt.
