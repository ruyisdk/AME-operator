# BF16/FP32 Ordinary-Call State

This suite uses native BF16 inputs/work arrays and FP32 outputs/work arrays
under UDS32/M16/Acc4, all six T values, both XLENs, C/C++, O0/O2.

Owned/released/reacquired phases use ordinary pointer/scalar calls. Inputs can
share storage and be 2-mod-4 aligned; outputs can be 4-mod-16 aligned. Each
workspace array is 16-byte aligned. Only the required array prefixes are used
for access and bounds calculation, even with oversized declared capacities.

Mode/matrix argument errors precede ownership. Empty output touches nothing.
K=0 needs only valid output; set writes +0 and add retains all bits. Unowned
compute returns NOT_OWNED before descriptor access. Owned NULL descriptor or
missing/misaligned/short arrays return NO_WORKSPACE. Invalid descriptor
alignment/address/overlap, overflow and matrix/work overlap return BAD_ARGUMENT.
Callers provide readable descriptor storage when validation inspects fields.
All six valid descriptor fields remain unchanged.

Owned phases seed UN=0, amestype=32, flags and xsat. Exact computations preserve
flags/xsat/ownership; noncomputing calls also preserve amestype. M/ACC/Md/Ad
and datatype setup can be clobbered during computation.

Standard F enables FCSR observation; drivers save/restore FS/FCSR and check
it in every ownership phase. Released calls avoid protected AME state.
Whole source/output allocations include sNaN guards; tests also check unused
workspace and every workspace byte on errors/no-ops. GOLDEN is independent
exact-dyadic arithmetic, with literal special-bit preservation rules. Auditors
reconstruct scenarios, counts and raw observations across compiler paths.
