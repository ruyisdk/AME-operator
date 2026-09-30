#include "f32_internal.h"
#define SIZE_MAX_VALUE ((ame_size) -1)

ame_size ame_ops_f32_workspace_bytes (ame_size t)
{
  if (t < 4 || t > 128 || (t & (t-1))) return 0;
  return 3*t*t*sizeof (float);
}
static int span (const void *p, ame_size rows, ame_size cols,
                 ame_size stride, ame_size *bytes)
{
  if (!p || (ame_uintptr) p % sizeof (float) || stride < cols) return 0;
  if (rows-1 && stride > (SIZE_MAX_VALUE-cols)/(rows-1)) return 0;
  ame_size elements = (rows-1)*stride+cols;
  if (elements > SIZE_MAX_VALUE/sizeof (float)) return 0;
  *bytes = elements*sizeof (float);
  return (ame_uintptr) p <= (ame_uintptr) -1 - *bytes;
}
static int overlap (const void *a, ame_size na, const void *b, ame_size nb)
{
  ame_uintptr x = (ame_uintptr) a, y = (ame_uintptr) b;
  return x < y+nb && y < x+na;
}
int ame_gemm_f32_owned (ame_size m, ame_size n, ame_size k,
                       const float *a, ame_size lda, const float *b, ame_size ldb,
                       float *c, ame_size ldc, int accumulate,
                       void *workspace, ame_size workspace_bytes)
{
  ame_size ca, aa, ba, t;
  if (accumulate != 0 && accumulate != 1) return AME_OPS_BAD_ARGUMENT;
  if (!m || !n) return AME_OPS_OK;
  if (!span (c,m,n,ldc,&ca)) return AME_OPS_BAD_ARGUMENT;
  if (!k)
    {
      if (!accumulate)
        for (ame_size i = 0; i < m; ++i)
          for (ame_size j = 0; j < n; ++j) c[i*ldc+j] = 0.0f;
      return AME_OPS_OK;
    }
  if (!span (a,m,k,lda,&aa) || !span (b,k,n,ldb,&ba)
      || overlap (a,aa,c,ca) || overlap (b,ba,c,ca)) return AME_OPS_BAD_ARGUMENT;
  int status = ame_ops_platform (&t);
  if (status != AME_OPS_OK) return status;
  ame_size need = ame_ops_f32_workspace_bytes (t);
  if (!need) return AME_OPS_UNSUPPORTED;
  if (!workspace || (ame_uintptr) workspace % 16 || workspace_bytes < need)
    return AME_OPS_NO_WORKSPACE;
  if ((ame_uintptr) workspace > (ame_uintptr) -1 - need
      || overlap (workspace,need,a,aa) || overlap (workspace,need,b,ba)
      || overlap (workspace,need,c,ca)) return AME_OPS_BAD_ARGUMENT;
  ame_ops_gemm_f32_kernel (m,n,k,a,lda,b,ldb,c,ldc,accumulate,(float *) workspace,t);
  return AME_OPS_OK;
}
