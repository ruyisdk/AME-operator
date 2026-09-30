#ifndef AME_OPS_F32_INTERNAL_H
#define AME_OPS_F32_INTERNAL_H
#include "../include/ame_ops_f32.h"
#ifdef __cplusplus
extern "C" {
#endif
int ame_ops_platform (ame_size *tile);
void ame_ops_gemm_f32_kernel (ame_size m, ame_size n, ame_size k,
                             const float *a, ame_size lda, const float *b, ame_size ldb,
                             float *c, ame_size ldc, int accumulate,
                             float *workspace, ame_size tile);
#ifdef __cplusplus
}
#endif
#endif
