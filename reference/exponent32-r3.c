/* Exact dyadic FP32 scaling/addition; no host floating point or model helper.  */
#include "test.h"
#pragma GCC optimize ("O2", "no-tree-loop-distribute-patterns")
#define LIMBS 32
#define ORIGIN 512
/* AME state: flags use the RISC-V F bit layout, not enum ordering.  */
enum { NX=1, UF=2, OF=4, DZ=8, NV=16 };
typedef __UINT64_TYPE__ wide;
typedef struct { u32 w[LIMBS]; } magnitude;
static void decode(magnitude *a,u32 bits,int scale)
{
  for (unsigned i=0;i<LIMBS;++i) a->w[i]=0;
  unsigned exp=(bits>>23)&255;
  u32 m=(bits&0x7fffffu)|(exp?0x800000u:0);
  int shift=(exp?(int)exp-150:-149)+scale+ORIGIN;
  unsigned index=(unsigned)shift/32,offset=(unsigned)shift%32;
  wide v=(wide)m<<offset;
  a->w[index]=(u32)v;
  a->w[index+1]=(u32)(v>>32);
}
static int compare(const magnitude *a,const magnitude *b)
{
  for (int i=LIMBS-1;i>=0;--i)
    if (a->w[i]!=b->w[i]) return a->w[i]>b->w[i]?1:-1;
  return 0;
}
static void add(magnitude *a,const magnitude *b)
{
  wide carry=0;
  for (unsigned i=0;i<LIMBS;++i)
    { wide v=(wide)a->w[i]+b->w[i]+carry; a->w[i]=(u32)v; carry=v>>32; }
}
static void subtract(magnitude *a,const magnitude *b)
{
  wide borrow=0;
  for (unsigned i=0;i<LIMBS;++i)
    {
      wide v=a->w[i],part=(wide)b->w[i]+borrow;
      a->w[i]=(u32)(v-part); borrow=v<part;
    }
}
static unsigned below(const magnitude *a,unsigned end)
{
  for (unsigned i=0;i<end/32;++i) if (a->w[i]) return 1;
  return end%32 && (a->w[end/32]&((1u<<(end%32))-1));
}
static unsigned getbit(const magnitude *a,unsigned index)
{ return (a->w[index/32]>>(index%32))&1; }
static unsigned rounded(const magnitude *a,unsigned sign,unsigned rm,u32 *out)
{
  int high=LIMBS*32-1;
  while (high>=0 && !getbit(a,(unsigned)high)) --high;
  if (high<0) { *out=sign<<31; return 0; }
  int exponent=high-ORIGIN;
  unsigned shift=exponent< -126?ORIGIN-149:(unsigned)high-23;
  unsigned index=shift/32,offset=shift%32;
  u32 q=a->w[index]>>offset;
  if (offset) q|=a->w[index+1]<<(32-offset);
  unsigned inexact=below(a,shift),half=getbit(a,shift-1);
  unsigned up=(rm==0 && half && (below(a,shift-1)||(q&1)))
              || (rm==4 && half) || (rm==2 && sign && inexact)
              || (rm==3 && !sign && inexact);
  q+=up;
  if (exponent>=-126 && q==0x1000000u) { q>>=1; ++exponent; }
  if (exponent>127)
    {
      unsigned infinity=rm==0 || rm==4 || (rm==2 && sign) || (rm==3 && !sign);
      *out=(sign<<31)|(infinity?0x7f800000u:0x7f7fffffu);
      return OF|NX;
    }
  u32 bits=exponent< -126?q:((u32)(exponent+127)<<23)|(q&0x7fffffu);
  *out=(sign<<31)|bits;
  return inexact?(bits<0x800000u?UF|NX:NX):0;
}
#ifdef __cplusplus
extern "C" {
#endif
unsigned i3n_scale32(unsigned rm,u32 a,int power,u32 old,unsigned accumulate,u32 *out)
{
  if (rm>=5 || power< -256 || power>256 || accumulate>1) return 0xffffffffu;
  u32 am=a&0x7fffffffu,cm=old&0x7fffffffu;
  unsigned sa=a>>31,sc=old>>31;
  if (am>0x7f800000u || (accumulate && cm>0x7f800000u))
    {
      *out=0x7fc00000u;
      return ((am>0x7f800000u && !(am&0x400000u))
              || (accumulate && cm>0x7f800000u && !(cm&0x400000u)))?NV:0;
    }
  if (am==0x7f800000u)
    {
      if (accumulate && cm==0x7f800000u && sa!=sc)
        { *out=0x7fc00000u; return NV; }
      *out=a; return 0;
    }
  if (accumulate && cm==0x7f800000u) { *out=old; return 0; }
  magnitude left,right;
  decode(&left,a,power); decode(&right,accumulate?old:0,0);
  if (!accumulate) sc=sa;
  if (sa==sc) add(&left,&right);
  else
    {
      int order=compare(&left,&right);
      if (order>=0)
        { subtract(&left,&right); if (!order) sa=rm==2; }
      else
        {
          subtract(&right,&left);
          for (unsigned i=0;i<LIMBS;++i) left.w[i]=right.w[i];
          sa=sc;
        }
    }
  return rounded(&left,sa,rm,out);
}
#ifdef __cplusplus
}
#endif
