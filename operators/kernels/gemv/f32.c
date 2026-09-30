#include "../../include/ame_ops_f32.h"
/* Single-column correctness baseline, not a bandwidth-tuned GEMV. */
int ame_gemv_f32_owned (ame_size m, ame_size k, const float *a, ame_size lda,
                       const float *x, float *y, int accumulate,
                       void *workspace, ame_size workspace_bytes)
{
  return ame_gemm_f32_owned (m,1,k,a,lda,x,1,y,1,accumulate,workspace,workspace_bytes);
}
