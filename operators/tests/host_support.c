#include "../common/pack_i32.h"
/* Host-only stub: validates API/packing, never counts as AME execution. */
int test_owned = 1, test_supported = 1, kernel_calls;
ame_size test_tile = 4;
int ame_ops_platform (ame_size *tile)
{
  if (!test_owned) return AME_OPS_NOT_OWNED;
  if (!test_supported) return AME_OPS_UNSUPPORTED;
  *tile = test_tile;
  return AME_OPS_OK;
}
void ame_ops_gemm_i32_kernel (ame_size m, ame_size n, ame_size k,
                             const ame_i32 *a, ame_size lda,
                             const ame_i32 *b, ame_size ldb,
                             ame_i32 *c, ame_size ldc, int accumulate,
                             ame_i32 *workspace, ame_size t)
{
  (void) workspace; (void) t;
  ++kernel_calls;
  for (ame_size i = 0; i < m; ++i)
    for (ame_size j = 0; j < n; ++j)
      {
        __INT64_TYPE__ v = accumulate ? c[i*ldc+j] : 0;
        for (ame_size p = 0; p < k; ++p) v += (__INT64_TYPE__) a[i*lda+p]*b[p*ldb+j];
        c[i*ldc+j] = (ame_i32) v;
      }
}
void test_pack (ame_i32 *dst, ame_size t, const ame_i32 *src,
                ame_size stride, ame_size rows, ame_size cols)
{
  ame_ops_pack_i32(dst,t,src,stride,rows,cols);
}
void test_unpack (ame_i32 *dst, ame_size stride, const ame_i32 *src,
                  ame_size t, ame_size rows, ame_size cols)
{
  ame_ops_unpack_i32(dst,stride,src,t,rows,cols);
}
