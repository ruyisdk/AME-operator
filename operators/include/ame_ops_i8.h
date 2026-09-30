#ifndef AME_OPS_I8_H
#define AME_OPS_I8_H
#include "ame_ops.h"
typedef __INT8_TYPE__ ame_i8;
typedef __UINT8_TYPE__ ame_u8;
#ifdef __cplusplus
extern "C" {
#endif
ame_size ame_ops_widen8_workspace_bytes (ame_size tile);
int ame_gemm_i8_i32_owned (ame_size m, ame_size n, ame_size k,
                          const ame_i8 *a, ame_size lda,
                          const ame_i8 *b, ame_size ldb,
                          ame_i32 *c, ame_size ldc, int accumulate,
                          void *workspace, ame_size workspace_bytes);
int ame_gemm_u8_i32_owned (ame_size m, ame_size n, ame_size k,
                          const ame_u8 *a, ame_size lda,
                          const ame_u8 *b, ame_size ldb,
                          ame_i32 *c, ame_size ldc, int accumulate,
                          void *workspace, ame_size workspace_bytes);
int ame_gemv_i8_i32_owned (ame_size m, ame_size k,
                          const ame_i8 *a, ame_size lda,
                          const ame_i8 *x, ame_i32 *y, int accumulate,
                          void *workspace, ame_size workspace_bytes);
int ame_gemv_u8_i32_owned (ame_size m, ame_size k,
                          const ame_u8 *a, ame_size lda,
                          const ame_u8 *x, ame_i32 *y, int accumulate,
                          void *workspace, ame_size workspace_bytes);
#ifdef __cplusplus
}
#endif
#endif
