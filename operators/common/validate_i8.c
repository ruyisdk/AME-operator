#include "i8_internal.h"
#define SIZE_MAX_VALUE ((ame_size) -1)

ame_size ame_ops_widen8_workspace_bytes (ame_size t)
{
  if (t < 4 || t > 128 || (t & (t-1))) return 0;
  return 12*t*t;
}
static int span (const void *p, ame_size rows, ame_size cols,
                 ame_size stride, ame_size element_bytes, ame_size *bytes)
{
  if (!p || (ame_uintptr) p % element_bytes || stride < cols) return 0;
  if (rows-1 && stride > (SIZE_MAX_VALUE-cols)/(rows-1)) return 0;
  ame_size elements = (rows-1)*stride+cols;
  if (elements > SIZE_MAX_VALUE/element_bytes) return 0;
  *bytes = elements*element_bytes;
  return (ame_uintptr) p <= (ame_uintptr) -1 - *bytes;
}
static int overlap (const void *a, ame_size na, const void *b, ame_size nb)
{
  ame_uintptr x = (ame_uintptr) a, y = (ame_uintptr) b;
  return x < y+nb && y < x+na;
}
static int gemm (ame_size m, ame_size n, ame_size k,
                 const void *a, ame_size lda, const void *b, ame_size ldb,
                 ame_i32 *c, ame_size ldc, int accumulate,
                 void *workspace, ame_size workspace_bytes, int is_unsigned)
{
  ame_size ca, aa, ba, t;
  if (accumulate != 0 && accumulate != 1) return AME_OPS_BAD_ARGUMENT;
  if (!m || !n) return AME_OPS_OK;
  if (!span (c,m,n,ldc,4,&ca)) return AME_OPS_BAD_ARGUMENT;
  if (!k)
    {
      if (!accumulate)
        for (ame_size i = 0; i < m; ++i)
          for (ame_size j = 0; j < n; ++j) c[i*ldc+j] = 0;
      return AME_OPS_OK;
    }
  if (!span (a,m,k,lda,1,&aa) || !span (b,k,n,ldb,1,&ba)
      || overlap (a,aa,c,ca) || overlap (b,ba,c,ca)) return AME_OPS_BAD_ARGUMENT;
  int status = ame_ops_platform (&t);
  if (status != AME_OPS_OK) return status;
  ame_size need = ame_ops_widen8_workspace_bytes (t);
  if (!need) return AME_OPS_UNSUPPORTED;
  if (!workspace || (ame_uintptr) workspace % 16 || workspace_bytes < need)
    return AME_OPS_NO_WORKSPACE;
  if ((ame_uintptr) workspace > (ame_uintptr) -1 - need
      || overlap (workspace,need,a,aa) || overlap (workspace,need,b,ba)
      || overlap (workspace,need,c,ca)) return AME_OPS_BAD_ARGUMENT;
  if (is_unsigned)
    ame_ops_gemm_u8_kernel (m,n,k,a,lda,b,ldb,c,ldc,accumulate,workspace,t);
  else
    ame_ops_gemm_i8_kernel (m,n,k,a,lda,b,ldb,c,ldc,accumulate,workspace,t);
  return AME_OPS_OK;
}
int ame_gemm_i8_i32_owned (ame_size m, ame_size n, ame_size k,
                          const ame_i8 *a, ame_size lda, const ame_i8 *b, ame_size ldb,
                          ame_i32 *c, ame_size ldc, int accumulate,
                          void *workspace, ame_size workspace_bytes)
{
  return gemm (m,n,k,a,lda,b,ldb,c,ldc,accumulate,workspace,workspace_bytes,0);
}
int ame_gemm_u8_i32_owned (ame_size m, ame_size n, ame_size k,
                          const ame_u8 *a, ame_size lda, const ame_u8 *b, ame_size ldb,
                          ame_i32 *c, ame_size ldc, int accumulate,
                          void *workspace, ame_size workspace_bytes)
{
  return gemm (m,n,k,a,lda,b,ldb,c,ldc,accumulate,workspace,workspace_bytes,1);
}
