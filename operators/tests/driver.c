#include "test.h"
#include "../include/ame_ops.h"
#include "op_fixtures.h"
static ame_i32 output[OP_MAX_C + 32] __attribute__((aligned(16)));
static ame_i32 workspace[3*MAX_N*MAX_N + 32] __attribute__((aligned(16)));
#define CANARY ((ame_i32) 0x6b91d237)
int run_kernel (const u32 *a0, const u32 *b0, u32 *c0, u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t = CSR_READ (amenlen);
  ame_size bytes = ame_ops_i32_workspace_bytes (t);
  check (0x50000000, 3*t*t*4, bytes);
  for (unsigned v = 0; v < OP_CASES; ++v)
    {
      const struct op_fixture *f = &op_fixtures[v];
      for (ame_size i = 0; i < OP_MAX_C+32; ++i) output[i] = CANARY;
      for (ame_size i = 0; i < 3*MAX_N*MAX_N+32; ++i) workspace[i] = CANARY;
      for (ame_size i = 0; i < f->c_count; ++i) output[16+i] = f->initial[i];
#if OP_GEMV
      int status = ame_gemv_i32_owned (f->m,f->k,f->a,f->lda,f->b,
                                      output+16,f->accumulate,workspace+16,bytes);
#else
      int status = ame_gemm_i32_owned (f->m,f->n,f->k,f->a,f->lda,f->b,f->ldb,
                                      output+16,f->ldc,f->accumulate,workspace+16,bytes);
#endif
      check (0x51000000 + v, AME_OPS_OK, status);
      for (ame_size i = 0; i < OP_MAX_C+32; ++i)
        {
          ame_size j = i-16;
          ame_i32 want = j < f->c_count ? f->golden[j] : CANARY;
          check (0x52000000 + v*0x10000 + i, (u32) want, (u32) output[i]);
        }
      for (ame_size i = 0; i < 3*MAX_N*MAX_N+32; ++i)
        if (i < 16 || i >= 16+3*t*t)
          check (0x53000000 + i, (u32) CANARY, (u32) workspace[i]);
      for (ame_size i = 0; i < f->a_count; ++i)
        check (0x54000000 + i, (u32) f->a_copy[i], (u32) f->a[i]);
      for (ame_size i = 0; i < f->b_count; ++i)
        check (0x55000000 + i, (u32) f->b_copy[i], (u32) f->b[i]);
    }
  /* Invalid calls must not alter output. Inputs use separate static objects. */
  ame_i32 av[1] = {2}, bv[1] = {3}, cv[1] = {17};
  check (0x56000000, AME_OPS_NO_WORKSPACE,
         ame_gemm_i32_owned (1,1,1,av,1,bv,1,cv,1,0,workspace+16,bytes-1));
  check (0x56000001, 17, cv[0]);
  check (0x56000002, AME_OPS_BAD_ARGUMENT,
         ame_gemm_i32_owned (1,1,1,av,0,bv,1,cv,1,0,workspace+16,bytes));
  check (0x56000003, 17, cv[0]);
  check (0x56000004, AME_OPS_BAD_ARGUMENT,
         ame_gemm_i32_owned (1,1,1,av,1,bv,1,av,1,0,workspace+16,bytes));
  check (0x56000005, 2, av[0]);
  check (0x56000006, AME_OPS_OK,
         ame_gemm_i32_owned (0,1,1,0,0,0,0,0,0,0,0,0));
  return 0;
}
