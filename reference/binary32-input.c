#include "test.h"
#ifdef __cplusplus
extern "C" {
#endif
u32 i4a_scalar(unsigned op,unsigned phase)
{
  unsigned k=op%3;
  if (phase==0) return 0x3f800000u;
  if (phase==1) return k==2?0x3f800003u:0x3f800000u;
  if (phase==2) return k==2?0x3f000000u:1;
  if (phase==3) return 0x7f7fffffu;
  if (phase==4) return k==2?0:0x7f800000u;
  if (phase==5) return 0x7f800001u;
  return phase==6?0x3f800003u:0xbf800003u;
}
u32 i4a_input(unsigned op,unsigned phase,unsigned buf,unsigned index)
{
  unsigned j=index%16,k=op%3;
  static const u32 exact_a[8]={0x3f800000,0xbf800000,0,0x80000000,
                             0x3f000000,0xbf000000,0x40000000,0xc0000000};
  static const u32 exact_b[8]={0x40400000,0xc0000000,0x3f800000,0xbf800000,
                             0x3e800000,0xbe800000,0,0x80000000};
  static const u32 small[8]={1,3,0x007fffff,0x00800000,
                           0x00ffffff,0x00800001,0x01000001,0};
  static const u32 special[8]={0x7f800000,0xff800000,0,0x80000000,
                             0x7fc12345,0xffc12345,0x3f800000,0xbf800000};
  static const u32 signal[8]={0x7f800001,0xff800001,0x7fa12345,0xffa12345,
                            0x7fc12345,0xffc12345,0,0x3f800000};
  if (phase==0) return buf?exact_b[j%8]:exact_a[j%8];
  if (phase==1)
    {
      static const u32 mul_b[8]={0x3fc00000,0xbfc00000,0x3f800001,0xbf800001,
                               0x3fa00000,0xbfa00000,0x3f800000,0xbf800000};
      static const u32 add_b[8]={0x33800000,0xb3800000,0x33c00000,0xb3c00000,
                               1,0x80000001,0x34000000,0xb4000000};
      static const u32 scalar_b[8]={0x4b800000,0xcb800000,0x4c000000,0xcc000000,
                                  0x4c800000,0xcc800000,0x33800000,0xb3800000};
      if (k==2) return buf?mul_b[j%8]:0x3f800003u;
      if (op>=3 && buf) return scalar_b[(j+(k==1?2:0))%8];
      if (k==1) return buf?0x3f800000u:(add_b[j%8]^0x80000000u);
      return buf?add_b[j%8]:0x3f800000u;
    }
  if (phase==2)
    {
      if (buf) return small[j%8]|((j&8)<<28);
      return k==2?0x3f000000u:1;
    }
  if (phase==3)
    {
      if (!buf) return 0x7f7fffffu|((j&1)<<31);
      return k==2?(0x40000000u|((j&2)<<30))
                 :(0x7f7fffffu|((j&2)<<30));
    }
  if (phase==4) return special[(j+(buf?2:0))%8];
  if (phase==5) return signal[(j+(buf?3:0))%8];
  u32 v=j*1664525u+1013904223u+buf*0x798523afu+phase*0x13579bdu;
  v^=v>>13;
  return v;
}
#ifdef __cplusplus
}
#endif
