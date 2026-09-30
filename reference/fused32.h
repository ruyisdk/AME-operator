#ifndef AME_TEST_FUSED32_H
#define AME_TEST_FUSED32_H
#include "test.h"
typedef struct {
  const u32 *a, *b, *old;
  u32 *out, *copy_a, *copy_b, *copy_old;
  u32 scalar, echo;
} fused32_args;
typedef char fused32_scalar_offset[__builtin_offsetof(fused32_args,scalar)==7*sizeof(void *)?1:-1];
#ifdef __cplusplus
extern "C" {
#endif
unsigned fused32(unsigned,unsigned,u32,u32,u32,u32 *);
u32 fused32_input(unsigned,unsigned,unsigned,unsigned);
u32 fused32_scalar(unsigned,unsigned);
void fused32_kernel(fused32_args *);
void i3f_poison(const u32 *);
#ifdef __cplusplus
}
#endif
#endif
