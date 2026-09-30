#include "../../include/ame_ops_bf16.h"
/* Unit-stride, single-column correctness baseline.  */
int ame_gemv_bf16_f32_owned (ame_size m, ame_size k,
                            const ame_bf16 *a, ame_size lda,
                            const ame_bf16 *x, float *y, int accumulate,
                            const struct ame_ops_bf16_workspace *workspace)
{
  return ame_gemm_bf16_f32_owned (m,1,k,a,lda,x,1,y,1,accumulate,workspace);
}
