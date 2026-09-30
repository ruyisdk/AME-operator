# BF16 to FP32 RNE Numerical Contract

This suite uses native BF16 GEMM/GEMV and square-major 1x2/2x1 packing under
UDS32/M16/Acc4. Logical products fold into old C (or +0 for assignment) in
increasing K, with one FP32 RNE FMA rounding per term. This order and
intermediate format are part of the selected backend contract.

BF16 encodings embed as the high 16 FP32 bits. The reference preserves original
sNaN identity, adds conversion NV, canonicalizes arithmetic NaNs and retains
subnormals. New flags combine conversion and FMA flags with prior sticky flags.
Tininess is detected after precision rounding.

## Independent Answers

Fraction arithmetic computes the logical fold without AME packets or host
floating arithmetic. Tests cross-check the C integer-limb reference under
UBSan, 33 literal witnesses and 256 seeded random BF16 dots with FP32 seeds.

Witnesses include precision/cancellation, midpoint parity, tiny/overflowing
products, fused overflow rescue and subnormal-half cases, normal-boundary
tininess, NaN/Inf/zero signs, seed-before-fold, ordering and sticky exceptions.
The overflow/subnormal witnesses distinguish fused computation from separate
multiply/add and from a single rounding of the whole exact sum.

At each T, sensitive dots are placed at T-1, 2*T-1 and 4*T-1 to cross squares
and packets. Mixed matrices cross output tails: 55 GEMM and 53 GEMV cases/T.

## Memory and State

Check native inputs/output, guards, unused work regions and descriptor fields.
Vary legal non-RNE FCSR patterns and require them unchanged. Check ownership,
UN=0, old|new AME flags and preserved xsat. The driver restores FS/FCSR.
Independent accounting verifies case inventory, logical results, raw logs and
exact counts. Host negative controls distinguish wrong RM, split/reordered/
late-seeded folds and BF16-truncated output.
