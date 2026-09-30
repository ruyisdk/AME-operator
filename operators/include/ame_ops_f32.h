#ifndef AME_OPS_F32_H
#define AME_OPS_F32_H
#include "ame_ops.h"
#if __SIZEOF_FLOAT__ != 4 || __FLT_RADIX__ != 2 || __FLT_MANT_DIG__ != 24 || __FLT_MAX_EXP__ != 128
#error "AME FP32 operators require IEEE binary32 float storage"
#endif
#ifdef __cplusplus
extern "C" {
#endif
ame_size ame_ops_f32_workspace_bytes (ame_size tile);
int ame_gemm_f32_owned (ame_size m, ame_size n, ame_size k,
                       const float *a, ame_size lda, const float *b, ame_size ldb,
                       float *c, ame_size ldc, int accumulate,
                       void *workspace, ame_size workspace_bytes);
int ame_gemv_f32_owned (ame_size m, ame_size k, const float *a, ame_size lda,
                       const float *x, float *y, int accumulate,
                       void *workspace, ame_size workspace_bytes);
#ifdef __cplusplus
}
#endif
#endif
