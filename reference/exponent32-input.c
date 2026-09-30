#include "test.h"
#ifdef __cplusplus
extern "C" {
#endif
int i3n_power(unsigned op,unsigned phase,unsigned index)
{
  unsigned j=index%16;
  if (op>=4) return phase==6?0:phase==7?13:1;
  if (phase==0) { static const int e[4]={1,0,2,-1}; return e[j%4]; }
  if (phase==1) { static const int e[4]={-1,-2,-24,-128}; return e[j%4]; }
  if (phase==2) return 1+(int)(j%3)*127;
  if (phase==6) return -1;
  if (phase==7) return (int)(j*37%513)-256;
  return 1;
}
u32 i3n_input(unsigned op,unsigned phase,unsigned buffer,unsigned index)
{
  unsigned j=index%16;
  if (buffer==1)
    { int e=i3n_power(op,phase,j); return (u32)((op==2||op==3)?-e:e); }
  if (phase==0)
    {
      static const u32 a[16]={0x3f800000u,0xbf800000u,0x3fc00000u,0xbfc00000u,
        0x40000000u,0xc0000000u,0x3f000000u,0xbf000000u,0,0x80000000u,
        0x40800000u,0xc0800000u,0x3e800000u,0xbe800000u,0x41000000u,0xc1000000u};
      static const u32 old[4]={0x40000000u,0xc0000000u,0x3f000000u,0xbf000000u};
      return buffer==0?a[j]:old[j%4];
    }
  if (phase==1)
    {
      static const u32 small[8]={1,3,0x007fffffu,0x00800001u,
        0x80000001u,0x80000003u,0x807fffffu,0x80800001u};
      return buffer==0?small[j%8]:0;
    }
  if (phase==2 || phase==3)
    {
      u32 a=0x7f7fffffu|((j&1)<<31);
      return buffer==0?a:phase==2?0:a^0x80000000u;
    }
  if (phase==4 || phase==5)
    {
      static const u32 special[8]={0x7fc12345u,0x7f800001u,0x7f800000u,
        0xff800000u,0,0x80000000u,0xffc12345u,0xff800001u};
      if (buffer==0) return special[j%8];
      return phase==4?0:special[(j+3)%8];
    }
  if (phase==6)
    {
      u32 sign=(j&4)<<29;
      return sign|(buffer==0?2*(j%4)+1:1);
    }
  u32 mix=j*1664525u+1013904223u+(buffer==2?0x798523afu:0);
  return (mix&0x807fffffu)|(((j*41+buffer*37)%255)<<23);
}
#ifdef __cplusplus
}
#endif
