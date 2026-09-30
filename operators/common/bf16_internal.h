#ifndef AME_OPS_BF16_INTERNAL_H
#define AME_OPS_BF16_INTERNAL_H
#include "../include/ame_ops_bf16.h"
#ifdef __cplusplus
extern "C" {
#endif
int ame_ops_platform (ame_size *tile);
void ame_ops_gemm_bf16_kernel (ame_size m, ame_size n, ame_size k,
                               const ame_bf16 *a, ame_size lda,
                               const ame_bf16 *b, ame_size ldb,
                               float *c, ame_size ldc, int accumulate,
                               ame_bf16 *ap, ame_bf16 *bp, float *cp, ame_size t);
#ifdef __cplusplus
}
#endif
#endif
