#ifndef AME_OPS_PACK_BF16_H
#define AME_OPS_PACK_BF16_H
#include "bf16_internal.h"
/* .rm uses square-major storage.  Repeat real output lanes, not zero lanes
   multiplying Inf; +0/-0 K padding is neutral for this RNE contract.  */
static inline __attribute__((always_inline)) void
ame_ops_pack_bf16_a (ame_bf16 *dst, ame_size t, const ame_bf16 *src,
                    ame_size stride, ame_size rows, ame_size cols)
{
  for (ame_size q=0; q<2; ++q)
    for (ame_size r=0; r<t; ++r)
      for (ame_size c=0; c<t; ++c)
        {
          ame_size at=(q*t+r)*t+c;
          if (q*t+c>=cols)
            {
              __UINT16_TYPE__ bits=0;
              __builtin_memcpy (&dst[at],&bits,sizeof (ame_bf16));
            }
          else
            __builtin_memcpy (&dst[at],&src[(r<rows ? r : rows-1)*stride+q*t+c],sizeof (ame_bf16));
        }
}
static inline __attribute__((always_inline)) void
ame_ops_pack_bf16_b (ame_bf16 *dst, ame_size t, const ame_bf16 *src,
                    ame_size stride, ame_size rows, ame_size cols)
{
  for (ame_size q=0; q<2; ++q)
    for (ame_size r=0; r<t; ++r)
      for (ame_size c=0; c<t; ++c)
        {
          ame_size at=(q*t+r)*t+c;
          if (q*t+r>=rows)
            {
              __UINT16_TYPE__ bits=0x8000u;
              __builtin_memcpy (&dst[at],&bits,sizeof (ame_bf16));
            }
          else
            __builtin_memcpy (&dst[at],&src[(q*t+r)*stride+(c<cols ? c : cols-1)],sizeof (ame_bf16));
        }
}
#endif
