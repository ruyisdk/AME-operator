#include "../../include/ame_ops_i8.h"
/* Single-column GEMM correctness baseline; not a tuned GEMV kernel. */
int ame_gemv_i8_i32_owned (ame_size m, ame_size k,
                          const ame_i8 *a, ame_size lda, const ame_i8 *x,
                          ame_i32 *y, int accumulate, void *workspace, ame_size bytes)
{
  return ame_gemm_i8_i32_owned (m,1,k,a,lda,x,1,y,1,accumulate,workspace,bytes);
}
int ame_gemv_u8_i32_owned (ame_size m, ame_size k,
                          const ame_u8 *a, ame_size lda, const ame_u8 *x,
                          ame_i32 *y, int accumulate, void *workspace, ame_size bytes)
{
  return ame_gemm_u8_i32_owned (m,1,k,a,lda,x,1,y,1,accumulate,workspace,bytes);
}
