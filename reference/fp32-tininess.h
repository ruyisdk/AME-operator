/* Shared by the versioned exact-grid FP32 references.  The caller supplies
   magnitude.w, LIMBS and ORIGIN from its own frozen arithmetic grid. */
#ifndef AME_REF_FP32_TININESS_H
#define AME_REF_FP32_TININESS_H
static unsigned fp32_tiny_after_precision_round(const magnitude *a,unsigned sign,unsigned rm)
{
  int high=LIMBS*32-1;
  while (high>=0 && !((a->w[high/32]>>(high%32))&1)) --high;
  if (high<0 || high-ORIGIN>=-126) return 0;
  if (high-ORIGIN< -127) return 1;
  unsigned shift=(unsigned)high-23,index=shift/32,offset=shift%32;
  u32 q=a->w[index]>>offset;
  if (offset) q|=a->w[index+1]<<(32-offset);
  unsigned nx=below(a,shift),half=(a->w[(shift-1)/32]>>((shift-1)%32))&1;
  q+=(rm==0 && half && (below(a,shift-1)||(q&1))) || (rm==4 && half)
     || (rm==2 && sign && nx) || (rm==3 && !sign && nx);
  return q<0x1000000u;
}
#endif
