#include "../../include/ame_ops.h"
/* Correctness baseline: one useful output column, not an optimized GEMV. */
int ame_gemv_i32_owned (ame_size m, ame_size k,
                       const ame_i32 *a, ame_size lda,
                       const ame_i32 *x, ame_i32 *y, int accumulate,
                       void *workspace, ame_size workspace_bytes)
{
  return ame_gemm_i32_owned (m,1,k,a,lda,x,1,y,1,accumulate,
                            workspace,workspace_bytes);
}
