#include "../common/pack_bf16.h"
int bf16_kernel_calls;
static float decode (ame_bf16 a)
{
  __UINT16_TYPE__ raw; __builtin_memcpy (&raw,&a,sizeof raw);
  __UINT32_TYPE__ bits=(__UINT32_TYPE__) raw<<16;
  float result; __builtin_memcpy (&result,&bits,sizeof result); return result;
}
/* Finite host contract stub, not the BF16 target or numerical GOLDEN.  */
void ame_ops_gemm_bf16_kernel (ame_size m,ame_size n,ame_size k,
                               const ame_bf16 *a,ame_size lda,const ame_bf16 *b,ame_size ldb,
                               float *c,ame_size ldc,int add,
                               ame_bf16 *ap,ame_bf16 *bp,float *cp,ame_size t)
{
  (void) ap; (void) bp; (void) cp; (void) t; ++bf16_kernel_calls;
  for (ame_size i=0; i<m; ++i)
    for (ame_size j=0; j<n; ++j)
      {
        float v=add ? c[i*ldc+j] : 0;
        for (ame_size q=0; q<k; ++q) v+=decode(a[i*lda+q])*decode(b[q*ldb+j]);
        c[i*ldc+j]=v;
      }
}
void test_pack_bf16_a (ame_bf16 *dst,ame_size t,const ame_bf16 *src,
                        ame_size stride,ame_size rows,ame_size cols)
{ ame_ops_pack_bf16_a(dst,t,src,stride,rows,cols); }
void test_pack_bf16_b (ame_bf16 *dst,ame_size t,const ame_bf16 *src,
                        ame_size stride,ame_size rows,ame_size cols)
{ ame_ops_pack_bf16_b(dst,t,src,stride,rows,cols); }
