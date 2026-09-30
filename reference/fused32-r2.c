/* R2 corrects UF at an inexact result delivered as minimum normal. */
#define fused32 fused32_r1_archival
#include "fused32.c"
#undef fused32
#include "fp32-tininess.h"
#ifdef __cplusplus
extern "C"
#endif
unsigned fused32(unsigned op,unsigned rm,u32 a,u32 b,u32 d,u32 *out)
{
  unsigned flags=fused32_r1_archival(op,rm,a,b,d,out);
  if (flags!=NX || (*out&0x7fffffffu)!=0x800000u) return flags;
  u32 p=op>=3?d:a,q=b,c=op>=3?a:d;
  if (!op) c=0;
  unsigned sp=(p>>31)^(q>>31)^(op==0 || op==2 || op==4),sc=c>>31;
  magnitude left,right;
  put(&left,(wide)mantissa(p)*mantissa(q),exponent(p)+exponent(q));
  if (op)
    {
      put(&right,mantissa(c),exponent(c));
      if (sp==sc) add(&left,&right);
      else if (compare(&left,&right)>=0) sub(&left,&right);
      else
        {
          sub(&right,&left);
          for (unsigned i=0;i<LIMBS;++i) left.w[i]=right.w[i];
        }
    }
  return flags|(fp32_tiny_after_precision_round(&left,*out>>31,rm)?UF:0);
}
