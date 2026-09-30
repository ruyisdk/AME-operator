/* Exact FP32 square root using integer arithmetic, independent of SoftFloat.  */
#include "test.h"
#pragma GCC optimize ("O2", "no-tree-loop-distribute-patterns")
#ifdef __cplusplus
extern "C" {
#endif
unsigned i3m_sqrt32(unsigned rm,u32 input,u32 *out)
{
  if (rm>=5) return 0xffffffffu;
  u32 magnitude=input&0x7fffffffu,fraction=magnitude&0x7fffffu;
  unsigned exponent=magnitude>>23;
  if (exponent==255 && fraction)
    { *out=0x7fc00000u; return (fraction&0x400000u)?0:16; }
  if (!magnitude)
    { *out=input; return 0; }
  if (input>>31)
    { *out=0x7fc00000u; return 16; }
  if (exponent==255)
    { *out=input; return 0; }
  int power=exponent?(int)exponent-127:-126;
  u32 mantissa=fraction|(exponent?0x800000u:0);
  while (mantissa<0x800000u) { mantissa<<=1; --power; }
  if (power%2) { mantissa<<=1; --power; }
  __UINT64_TYPE__ radicand=(__UINT64_TYPE__)mantissa<<23;
  __UINT64_TYPE__ remainder=radicand,root=0,bit=(__UINT64_TYPE__)1<<46;
  /* Restoring square root gives floor(sqrt(radicand)) and its exact remainder.  */
  while (bit)
    {
      if (remainder>=root+bit)
        { remainder-=root+bit; root=(root>>1)+bit; }
      else root>>=1;
      bit>>=2;
    }
  unsigned inexact=remainder!=0;
  if (inexact && (rm==3 || ((rm==0 || rm==4) && 4*remainder>4*root+1)))
    ++root;
  if (root==0x1000000u) { root>>=1; power+=2; }
  *out=((u32)(power/2+127)<<23)|((u32)root&0x7fffffu);
  return inexact;
}
#ifdef __cplusplus
}
#endif
