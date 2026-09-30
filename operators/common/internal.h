#ifndef AME_OPS_INTERNAL_H
#define AME_OPS_INTERNAL_H
#include "../include/ame_ops.h"
#ifdef __cplusplus
extern "C" {
#endif
int ame_ops_platform (ame_size *tile);
void ame_ops_gemm_i32_kernel (ame_size m, ame_size n, ame_size k,
                             const ame_i32 *a, ame_size lda,
                             const ame_i32 *b, ame_size ldb,
                             ame_i32 *c, ame_size ldc, int accumulate,
                             ame_i32 *workspace, ame_size tile);
#ifdef __cplusplus
}
#endif
#endif
