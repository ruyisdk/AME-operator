# Operator API

Headers live in `operators/include/`: `ame_ops.h`, `ame_ops_i8.h`,
`ame_ops_f32.h`, and `ame_ops_bf16.h`. They provide C linkage in C++.

| Entry point suffix | Inputs | Output |
| --- | --- | --- |
| `i32_owned` | `ame_i32` | `ame_i32` |
| `i8_i32_owned` | `ame_i8` | `ame_i32` |
| `u8_i32_owned` | `ame_u8` | `ame_i32` |
| `f32_owned` | `float` | `float` |
| `bf16_f32_owned` | `ame_bf16` (`__bf16`) | `float` |

Each suffix is available under `ame_gemm_` and `ame_gemv_`. GEMM takes
m/n/k, A/lda, B/ldb, C/ldc, `accumulate`, and workspace. GEMV takes m/k,
A/lda, x, y, `accumulate`, and workspace. Leading dimensions count elements.
`accumulate=0` selects assignment; `1` includes the old output.

## Ownership and Storage

The caller acquires AME resources before nonempty computation and retains
ownership until return. Kernels use M/Acc and Md/Ad state; acquisition/release
stays outside the operator API. See target drivers in `operators/tests/` for
ownership setup. The ABI boundary carries only ordinary pointers and scalars.

Workspace pointers are 16-byte aligned. i32/widen8/FP32 require 12*T*T bytes;
FP32 workspace contains live float objects. BF16 uses a descriptor for two
BF16 arrays of at least 2*T*T elements each and one float array of T*T elements.
Its capacities count elements. Use the workspace query functions in the headers.

A and B may share storage. Output cannot overlap inputs. Workspace regions
are mutually disjoint and disjoint from matrices and the BF16 descriptor.
Span checks conservatively include row padding. Callers supply valid readable/
writable objects for the required spans.

## Errors and Zero Dimensions

Status values: `AME_OPS_OK`, `AME_OPS_BAD_ARGUMENT`, `AME_OPS_NO_WORKSPACE`,
`AME_OPS_NOT_OWNED`, `AME_OPS_UNSUPPORTED`.

Invalid mode/matrix arguments precede ownership checks. Platform checks precede
workspace access. Rejected calls leave output unchanged. Empty outputs touch
no matrix/workspace data. K=0 requires valid output but no input, workspace or
ownership: assignment writes zero; accumulation preserves all output bits.

## Numerical Contracts

- Integer: `abs(seed) + sum(abs(A*B)) <= INT32_MAX` per output. References check
  this sufficient no-overflow bound; kernels rely on the caller's input domain.
- FP32/BF16: RNE, fused FP32 accumulation in increasing K order. Exact-domain
  tests also provide an order-independent tier within their scaled-integer bound.
- Only active output elements change. Packing handles tails and preserves RNE
  signed-zero/exception behavior. Flags accumulate as specified by the selected
  numerical contract; ownership, xsat and standard FCSR are preserved.

Detailed contracts are in [operators/docs](../operators/docs). Transposed
operands, general alpha/beta, nonunit GEMV strides and tuned GEMV kernels are
future API extensions.
