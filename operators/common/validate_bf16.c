#include "bf16_internal.h"
#define SIZE_MAX_VALUE ((ame_size) -1)
ame_size ame_ops_bf16_output_elements (ame_size t)
{
  if (t<4 || t>128 || (t & (t-1))) return 0;
  return t*t;
}
ame_size ame_ops_bf16_input_elements (ame_size t)
{ return 2*ame_ops_bf16_output_elements (t); }
static int span (const void *p, ame_size rows, ame_size cols,
                 ame_size stride, ame_size element_bytes, ame_size *bytes)
{
  if (!p || (ame_uintptr) p % element_bytes || stride<cols) return 0;
  if (rows-1 && stride>(SIZE_MAX_VALUE-cols)/(rows-1)) return 0;
  ame_size elements=(rows-1)*stride+cols;
  if (elements>SIZE_MAX_VALUE/element_bytes) return 0;
  *bytes=elements*element_bytes;
  return (ame_uintptr) p<=(ame_uintptr) -1-*bytes;
}
static int overlap (const void *a, ame_size na, const void *b, ame_size nb)
{
  ame_uintptr x=(ame_uintptr) a,y=(ame_uintptr) b;
  return x<y+nb && y<x+na;
}
int ame_gemm_bf16_f32_owned (ame_size m, ame_size n, ame_size k,
                            const ame_bf16 *a, ame_size lda,
                            const ame_bf16 *b, ame_size ldb,
                            float *c, ame_size ldc, int accumulate,
                            const struct ame_ops_bf16_workspace *w)
{
  ame_size ca,aa,ba,t;
  if (accumulate!=0 && accumulate!=1) return AME_OPS_BAD_ARGUMENT;
  if (!m || !n) return AME_OPS_OK;
  if (!span (c,m,n,ldc,sizeof (float),&ca)) return AME_OPS_BAD_ARGUMENT;
  if (!k)
    {
      if (!accumulate)
        for (ame_size i=0; i<m; ++i)
          for (ame_size j=0; j<n; ++j) c[i*ldc+j]=0.0f;
      return AME_OPS_OK;
    }
  if (!span (a,m,k,lda,sizeof (ame_bf16),&aa)
      || !span (b,k,n,ldb,sizeof (ame_bf16),&ba)
      || overlap (a,aa,c,ca) || overlap (b,ba,c,ca)) return AME_OPS_BAD_ARGUMENT;
  int status=ame_ops_platform (&t);
  if (status!=AME_OPS_OK) return status;
  ame_size elements=ame_ops_bf16_input_elements (t),outputs=ame_ops_bf16_output_elements (t);
  if (!elements) return AME_OPS_UNSUPPORTED;
  if (!w) return AME_OPS_NO_WORKSPACE;
  if ((ame_uintptr) w % __alignof__ (struct ame_ops_bf16_workspace)
      || (ame_uintptr) w>(ame_uintptr) -1-sizeof (*w)
      || overlap (w,sizeof (*w),a,aa) || overlap (w,sizeof (*w),b,ba)
      || overlap (w,sizeof (*w),c,ca)) return AME_OPS_BAD_ARGUMENT;
  if (!w->a || !w->b || !w->c || (ame_uintptr) w->a % 16
      || (ame_uintptr) w->b % 16 || (ame_uintptr) w->c % 16
      || w->a_elements<elements || w->b_elements<elements || w->c_elements<outputs)
    return AME_OPS_NO_WORKSPACE;
  const void *buffers[3]={w->a,w->b,w->c};
  ame_size bytes=outputs*sizeof (float);
  for (unsigned i=0; i<3; ++i)
    if ((ame_uintptr) buffers[i]>(ame_uintptr) -1-bytes) return AME_OPS_BAD_ARGUMENT;
  for (unsigned i=0; i<3; ++i)
    {
      if (overlap (buffers[i],bytes,a,aa) || overlap (buffers[i],bytes,b,ba)
          || overlap (buffers[i],bytes,c,ca) || overlap (buffers[i],bytes,w,sizeof (*w)))
        return AME_OPS_BAD_ARGUMENT;
      for (unsigned j=0; j<i; ++j)
        if (overlap (buffers[i],bytes,buffers[j],bytes)) return AME_OPS_BAD_ARGUMENT;
    }
  ame_ops_gemm_bf16_kernel (m,n,k,a,lda,b,ldb,c,ldc,accumulate,w->a,w->b,w->c,t);
  return AME_OPS_OK;
}
