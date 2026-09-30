/* Supplementary matrix zero signs, infinity cancellation and quiet NaNs.
   The manifests select phase 0, 4 or 5; phase 5 starts with sticky flags 7.  */
#include "test.h"
#ifdef __cplusplus
extern "C" {
#endif
u32 i4a_scalar(unsigned op,unsigned phase)
{ (void)op; (void)phase; return 0; }
u32 i4a_input(unsigned op,unsigned phase,unsigned buf,unsigned index)
{
  u32 sign=(index&1u)<<31;
  if (phase==0) return buf?((index&2u)<<30):sign;
  if (phase==4)
    return 0x7f800000u|(sign^((buf && op==0)?0x80000000u:0));
  if (!buf) return 0x7fc12345u|sign;
  return (index&2)?0x3f800000u:0xffc54321u;
}
#ifdef __cplusplus
}
#endif
