#include "test.h"
#ifdef __cplusplus
extern "C" {
#endif
u32 i3m_input(unsigned phase,unsigned index)
{
  static const u32 exact[16]={
    0,0x80000000u,0x3f800000u,0x40800000u,
    0x41100000u,0x41800000u,0x3e800000u,0x3d800000u,
    2,0x00800000u,8,0x7e800000u,
    0x7f800000u,0x41c80000u,0x42800000u,0x3c800000u};
  static const u32 rounded[16]={
    0x40000000u,0x40400000u,0x40a00000u,0x40e00000u,
    0x41000000u,0x41200000u,0x3f000000u,0x3f800001u,
    0x3f7fffffu,0x007fffffu,1,0x7f7fffffu,
    0x407fffffu,0x40800001u,3,0x00800001u};
  unsigned slot=index%16;
  if (phase==0) return exact[slot];
  if (phase==1 || phase==7) return rounded[slot];
  if (phase==2) return slot==0?0xbf800000u:slot==15?0xff800000u:0x80000000u|rounded[slot];
  if (phase==3) return 0x7fc00000u|((index&1)<<31)|((index*37+1)&0x3fffffu);
  if (phase==4) return 0x7f800001u|((index&1)<<31)|((index*37)&0x3fffffu);
  if (phase==5)
    return slot==1?0x7f800001u:slot==2?0x7fc00001u:rounded[slot];
  u32 mixed=index*1664525u+1013904223u;
  return (mixed&0x807fffffu)|(((index*73u+17u)%255u)<<23);
}
#ifdef __cplusplus
}
#endif
