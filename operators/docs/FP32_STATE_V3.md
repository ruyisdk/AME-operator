# FP32 Ordinary-Call State and Errors

The state suite uses exact-dyadic compute inputs and raw negative-zero/NaN/Inf
seeds on K=0 paths. It covers UDS32/M16/Acc4, all six T values, both XLENs,
C/C++ and O0/O2.

Owned, released and reacquired phases exercise set/add, repeated calls and
allowed input sharing. Errors cover mode, null/misaligned matrices, short
strides, output/input overlap, missing/short/misaligned/aliased workspace,
element/byte overflow and address wrap. GEMV tests its implicit contiguous
output rather than GEMM-only n/ldb/ldc parameters.

Every scenario checks full source/output allocations with sNaN guards.
Errors/no-ops check the whole workspace; compute checks outside 12*T*T bytes.
Bit setup and inspection use memcpy with real float objects.

Argument checks precede ownership; workspace validation follows it. Empty and
K=0 calls avoid inputs/workspace and protected AME state. Owned calls seed
UN=0, amestype=32, sticky flags and xsat. Exact compute preserves flags/xsat;
noncomputing calls also preserve amestype. K=0 set writes +0 and add preserves
all seed bits without NV.

The driver independently seeds FCSR, saves/restores FS and FCSR, and checks it
after every call. The adapter enables standard F for observation. An auditor
reconstructs phase/action sets, literal matrices, status/flags, raw observations
and exact assertion counts; host rejection tests mutate these expectations.
