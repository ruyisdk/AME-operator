#include "../common/pack_f32.h"
/* Host contract/packing stub only; never AME execution evidence. */
int f32_kernel_calls;
void ame_ops_gemm_f32_kernel (ame_size m, ame_size n, ame_size k,
                             const float *a, ame_size lda, const float *b, ame_size ldb,
                             float *c, ame_size ldc, int accumulate,
                             float *workspace, ame_size t)
{
  (void) workspace; (void) t;
  ++f32_kernel_calls;
  for (ame_size i=0; i<m; ++i)
    for (ame_size j=0; j<n; ++j)
      {
        float v=accumulate ? c[i*ldc+j] : 0.0f;
        for (ame_size q=0; q<k; ++q) v += a[i*lda+q]*b[q*ldb+j];
        c[i*ldc+j]=v;
      }
}
void test_pack_f32 (float *dst, ame_size t, const float *src,
                    ame_size stride, ame_size rows, ame_size cols)
{ ame_ops_pack_f32 (dst,t,src,stride,rows,cols,AME_OPS_F32_SEED); }
void test_pack_f32_left (float *dst, ame_size t, const float *src,
                         ame_size stride, ame_size rows, ame_size cols)
{ ame_ops_pack_f32 (dst,t,src,stride,rows,cols,AME_OPS_F32_LEFT); }
void test_pack_f32_right (float *dst, ame_size t, const float *src,
                          ame_size stride, ame_size rows, ame_size cols)
{ ame_ops_pack_f32 (dst,t,src,stride,rows,cols,AME_OPS_F32_RIGHT); }
void test_unpack_f32 (float *dst, ame_size stride, const float *src,
                      ame_size t, ame_size rows, ame_size cols)
{ ame_ops_unpack_f32 (dst,stride,src,t,rows,cols); }
