#ifndef AME_OPS_PACK_F32_H
#define AME_OPS_PACK_F32_H
#include "f32_internal.h"
enum ame_ops_f32_tile { AME_OPS_F32_LEFT, AME_OPS_F32_RIGHT, AME_OPS_F32_SEED };
/* Nonempty tiles only.  Duplicate real output lanes; under RNE, a -0 K-tail
   product preserves either accumulator zero sign without new exceptions.  */
static inline __attribute__((always_inline)) void
ame_ops_pack_f32 (float *dst, ame_size t, const float *src,
                  ame_size stride, ame_size rows, ame_size cols,
                  enum ame_ops_f32_tile role)
{
  for (ame_size r = 0; r < t; ++r)
    for (ame_size c = 0; c < t; ++c)
      if ((role == AME_OPS_F32_LEFT && c >= cols)
          || (role == AME_OPS_F32_RIGHT && r >= rows))
        {
          __UINT32_TYPE__ bits = role == AME_OPS_F32_RIGHT ? 0x80000000u : 0;
          __builtin_memcpy (&dst[r*t+c], &bits, sizeof (float));
        }
      else
        {
          ame_size rr = r < rows ? r : rows-1;
          ame_size cc = c < cols ? c : cols-1;
          __builtin_memcpy (&dst[r*t+c], &src[rr*stride+cc], sizeof (float));
        }
}
static inline __attribute__((always_inline)) void
ame_ops_unpack_f32 (float *dst, ame_size stride, const float *src,
                    ame_size t, ame_size rows, ame_size cols)
{
  for (ame_size r = 0; r < rows; ++r)
    for (ame_size c = 0; c < cols; ++c)
      __builtin_memcpy (&dst[r*stride+c], &src[r*t+c], sizeof (float));
}
#endif
