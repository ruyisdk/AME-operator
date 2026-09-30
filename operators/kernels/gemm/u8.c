#define WIDEN_KERNEL ame_ops_gemm_u8_kernel
#define WIDEN_ELEMENT ame_u8
#define WIDEN_ROW __riscv_ztt_u8_rnu_1x4_t
#define WIDEN_COL __riscv_ztt_u8_rnu_4x1_t
#define WIDEN_LOAD_ROW __riscv_ztt_mls_rm_u8_rnu_1x4
#define WIDEN_LOAD_COL __riscv_ztt_mls_rm_u8_rnu_4x1
#include "widen8.inc"
