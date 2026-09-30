#ifndef AME_OPS_PACK_I8_H
#define AME_OPS_PACK_I8_H
#include "i8_internal.h"
/* Packed .rm memory is square-major, not a flat T by 4T row-major tile. */
static inline __attribute__((always_inline)) void
ame_ops_pack_i8_a (ame_u8 *dst, ame_size t, const ame_u8 *src,
                   ame_size stride, ame_size rows, ame_size cols)
{
  for (ame_size q = 0; q < 4; ++q)
    for (ame_size r = 0; r < t; ++r)
      for (ame_size c = 0; c < t; ++c)
        dst[(q*t+r)*t+c] = r < rows && q*t+c < cols
                          ? src[r*stride+q*t+c] : 0;
}
static inline __attribute__((always_inline)) void
ame_ops_pack_i8_b (ame_u8 *dst, ame_size t, const ame_u8 *src,
                   ame_size stride, ame_size rows, ame_size cols)
{
  for (ame_size q = 0; q < 4; ++q)
    for (ame_size r = 0; r < t; ++r)
      for (ame_size c = 0; c < t; ++c)
        dst[(q*t+r)*t+c] = q*t+r < rows && c < cols
                          ? src[(q*t+r)*stride+c] : 0;
}
#endif
