/* Exact FP32 product and sum; no intermediate floating-point format.  */
#include "fused32.h"
#pragma GCC optimize ("O2", "no-tree-loop-distribute-patterns")
#define LIMBS 20
#define ORIGIN 320
enum { NX=1, UF=2, OF=4, NV=16 };
typedef __UINT64_TYPE__ wide;
typedef struct { u32 w[LIMBS]; } magnitude;
static unsigned mantissa(u32 a)
{ return (a&0x7fffffu)|((a&0x7f800000u)?0x800000u:0); }
static int exponent(u32 a)
{ unsigned e=(a>>23)&255; return e?(int)e-150:-149; }
static void put(magnitude *a,wide m,int e)
{
  for (unsigned i=0;i<LIMBS;++i) a->w[i]=0;
  /* Finite products occupy bits 22..575; the sum has one spare carry bit.  */
  unsigned bit=(unsigned)(e+ORIGIN);
  while (m)
    {
      if (m&1) a->w[bit/32]|=1u<<(bit%32);
      m>>=1; ++bit;
    }
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
static void sub(magnitude *a,const magnitude *b)
{
  wide borrow=0;
  for (unsigned i=0;i<LIMBS;++i)
    { wide v=a->w[i],t=(wide)b->w[i]+borrow; a->w[i]=(u32)(v-t); borrow=v<t; }
}
static unsigned bit(const magnitude *a,unsigned i)
{ return (a->w[i/32]>>(i%32))&1; }
static unsigned below(const magnitude *a,unsigned end)
{
  for (unsigned i=0;i<end/32;++i) if (a->w[i]) return 1;
  return end%32 && (a->w[end/32]&((1u<<(end%32))-1));
}
static unsigned round_result(const magnitude *a,unsigned sign,unsigned rm,u32 *out)
{
  int high=LIMBS*32-1;
  while (high>=0 && !bit(a,(unsigned)high)) --high;
  if (high<0) { *out=sign<<31; return 0; }
  int exp=high-ORIGIN;
  unsigned shift=exp< -126?ORIGIN-149:(unsigned)high-23;
  unsigned at=shift/32,off=shift%32;
  u32 q=a->w[at]>>off;
  if (off) q|=a->w[at+1]<<(32-off);
  unsigned nx=below(a,shift),half=bit(a,shift-1);
  q+=(rm==0 && half && (below(a,shift-1)||(q&1)))
     || (rm==4 && half) || (rm==2 && sign && nx) || (rm==3 && !sign && nx);
  if (exp>=-126 && q==0x1000000u) { q>>=1; ++exp; }
  if (exp>127)
    {
      unsigned inf=rm==0 || rm==4 || (rm==2 && sign) || (rm==3 && !sign);
      *out=(sign<<31)|(inf?0x7f800000u:0x7f7fffffu);
      return OF|NX;
    }
  u32 result=exp< -126?q:((u32)(exp+127)<<23)|(q&0x7fffffu);
  *out=(sign<<31)|result;
  return nx?(result<0x800000u?UF|NX:NX):0;
}
#ifdef __cplusplus
extern "C" {
#endif
unsigned fused32(unsigned op,unsigned rm,u32 a,u32 b,u32 d,u32 *out)
{
  if (op>4 || rm>4) return 0xffffffffu;
  u32 p=op>=3?d:a,q=b,c=op>=3?a:d;
  if (!op) c=0; /* The negative product does not read the old destination.  */
  u32 pm=p&0x7fffffffu,qm=q&0x7fffffffu,cm=c&0x7fffffffu;
  unsigned sp=(p>>31)^(q>>31)^(op==0 || op==2 || op==4),sc=c>>31;
  unsigned invalid=(pm==0x7f800000u && !qm) || (qm==0x7f800000u && !pm);
  unsigned pinf=!invalid && pm<=0x7f800000u && qm<=0x7f800000u
    && (pm==0x7f800000u || qm==0x7f800000u);
  invalid|=op && pinf && cm==0x7f800000u && sp!=sc;
  unsigned nan=pm>0x7f800000u || qm>0x7f800000u || cm>0x7f800000u;
  unsigned snan=(pm>0x7f800000u && !(p&0x400000u))
    || (qm>0x7f800000u && !(q&0x400000u))
    || (cm>0x7f800000u && !(c&0x400000u));
  if (nan || invalid)
    {
      *out=0x7fc00000u;
      return invalid || snan?NV:0;
    }
  if (pinf || cm==0x7f800000u)
    {
      *out=((pinf?sp:sc)<<31)|0x7f800000u;
      return 0;
    }
  magnitude left,right;
  put(&left,(wide)mantissa(p)*mantissa(q),exponent(p)+exponent(q));
  if (!op) return round_result(&left,sp,rm,out);
  put(&right,mantissa(c),exponent(c));
  if (sp==sc) add(&left,&right);
  else
    {
      int order=compare(&left,&right);
      if (order>=0) { sub(&left,&right); if (!order) sp=rm==2; }
      else
        {
          sub(&right,&left);
          for (unsigned i=0;i<LIMBS;++i) left.w[i]=right.w[i];
          sp=sc;
        }
    }
  return round_result(&left,sp,rm,out);
}
#ifdef __cplusplus
}
#endif
