#ifndef AME_OPS_PACK_I32_H
#define AME_OPS_PACK_I32_H
#include "internal.h"
static inline __attribute__((always_inline)) void
ame_ops_pack_i32 (ame_i32 *dst, ame_size t, const ame_i32 *src,
                  ame_size stride, ame_size rows, ame_size cols)
{
  for (ame_size r = 0; r < t; ++r)
    for (ame_size c = 0; c < t; ++c)
      dst[r*t+c] = r < rows && c < cols ? src[r*stride+c] : 0;
}
static inline __attribute__((always_inline)) void
ame_ops_unpack_i32 (ame_i32 *dst, ame_size stride, const ame_i32 *src,
                    ame_size t, ame_size rows, ame_size cols)
{
  for (ame_size r = 0; r < rows; ++r)
    for (ame_size c = 0; c < cols; ++c)
      dst[r*stride+c] = src[r*t+c];
}
#endif
