# Cross-Family Call and State Edges

Each GEMM/GEMV ELF links all five production paths: i32, i8/u8 to i32, FP32,
BF16 to FP32. Tests use scalar/pointer calls from C/C++, O0/O2, RV32/RV64,
runtime T=4,8,16,32,64,128, and UDS32/M16/Acc4.

There are 125 ordered scenarios per operator/T. Initial owned phases exercise
all eight actions for each family with UN=0 and UN=1. Actual unsupported
msettyp/madd produces UN=1, with no typed value live across the setup. Normal
owned returns preserve prior UN; subsequent test setup clears it explicitly.

Released phases check argument/ownership precedence and noncomputing calls
without reading protected state. Reacquisition reverses the family order and
repeats set/add with UN=1, testing stale datatype assumptions. M/ACC values
do not survive release.

Both operators test K=0 using eight raw seeds: -0, sNaN, qNaN, +Inf, -Inf,
minimum subnormal, maximum finite and +0. Set produces zero; add preserves
bits. Compute cases use distinct exact signed/unsigned/floating products.

Each call checks current-family source/output allocations, descriptor fields,
unused workspace (all workspace for noncomputing paths), ownership, FCSR and
owned flags/xsat/UN. Noncomputing calls retain amestype. The signed/unsigned
byte paths intentionally reuse workspace; other families have separate storage.

All kernels receive frame checks. The independent auditor reconstructs scenario
order, bits and assertion counts and compares raw observations across paths.
Standard F is enabled for FCSR observation; kernel compilation remains no-F.
