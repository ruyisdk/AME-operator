#ifndef AME_OPS_I8_INTERNAL_H
#define AME_OPS_I8_INTERNAL_H
#include "internal.h"
#include "../include/ame_ops_i8.h"
#ifdef __cplusplus
extern "C" {
#endif
void ame_ops_gemm_i8_kernel (ame_size, ame_size, ame_size,
                            const void *, ame_size, const void *, ame_size,
                            ame_i32 *, ame_size, int, void *, ame_size);
void ame_ops_gemm_u8_kernel (ame_size, ame_size, ame_size,
                            const void *, ame_size, const void *, ame_size,
                            ame_i32 *, ame_size, int, void *, ame_size);
#ifdef __cplusplus
}
#endif
#endif
