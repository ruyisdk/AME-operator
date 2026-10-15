#ifndef TUTORIAL_GEMM_KERNEL_H
#define TUTORIAL_GEMM_KERNEL_H
#include <riscv_ztt.h>

/* One complete 4x4 tile: C = A * B, row-major, signed INT32.
   Preconditions: AME is acquired, UDS=32, N=4, M16/Acc4 profile.
   The inputs are contiguous; neither matrix is transposed. */
static inline void gemm_tile(const int *a, const int *b, int *c)
{
  const int zero[16] = {0};
  __riscv_ztt_i32_rnu_1x1_t av = __riscv_ztt_mls_rm_i32_rnu_1x1(a);
  __riscv_ztt_i32_rnu_1x1_t bv = __riscv_ztt_mls_rm_i32_rnu_1x1(b);
  __riscv_ztt_i32_rnu_accx1_t acc = __riscv_ztt_mcopy_m2a_i32_rnu_accx1(
      __riscv_ztt_mls_rm_i32_rnu_1x1(zero));
  acc = __riscv_ztt_mmulacc_2d_i32_rnu_accx1(acc, av, bv);
  __riscv_ztt_mss_rm(c, __riscv_ztt_mcopy_a2m_i32_rnu_1x1(acc));
}
#endif
