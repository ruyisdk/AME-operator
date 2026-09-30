#include "../common/pack_i8.h"
/* Host stubs test validation/packing only; they never count as AME execution. */
int widen_signed_calls, widen_unsigned_calls;
static void scalar (ame_size m, ame_size n, ame_size k,
                    const void *a, ame_size lda, const void *b, ame_size ldb,
                    ame_i32 *c, ame_size ldc, int add, int u)
{
  for (ame_size i = 0; i < m; ++i)
    for (ame_size j = 0; j < n; ++j)
      {
        __INT64_TYPE__ v = add ? c[i*ldc+j] : 0;
        for (ame_size p = 0; p < k; ++p)
          {
            int x = u ? ((const ame_u8 *) a)[i*lda+p] : ((const ame_i8 *) a)[i*lda+p];
            int y = u ? ((const ame_u8 *) b)[p*ldb+j] : ((const ame_i8 *) b)[p*ldb+j];
            v += x*y;
          }
        c[i*ldc+j] = (ame_i32) v;
      }
}
void ame_ops_gemm_i8_kernel (ame_size m, ame_size n, ame_size k,
                            const void *a, ame_size lda, const void *b, ame_size ldb,
                            ame_i32 *c, ame_size ldc, int add, void *w, ame_size t)
{
  (void) w; (void) t; ++widen_signed_calls;
  scalar(m,n,k,a,lda,b,ldb,c,ldc,add,0);
}
void ame_ops_gemm_u8_kernel (ame_size m, ame_size n, ame_size k,
                            const void *a, ame_size lda, const void *b, ame_size ldb,
                            ame_i32 *c, ame_size ldc, int add, void *w, ame_size t)
{
  (void) w; (void) t; ++widen_unsigned_calls;
  scalar(m,n,k,a,lda,b,ldb,c,ldc,add,1);
}
void test_pack_a8 (ame_u8 *d, ame_size t, const ame_u8 *s,
                   ame_size stride, ame_size rows, ame_size cols)
{ ame_ops_pack_i8_a(d,t,s,stride,rows,cols); }
void test_pack_b8 (ame_u8 *d, ame_size t, const ame_u8 *s,
                   ame_size stride, ame_size rows, ame_size cols)
{ ame_ops_pack_i8_b(d,t,s,stride,rows,cols); }
