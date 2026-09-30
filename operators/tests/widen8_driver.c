#include "test.h"
#include "../include/ame_ops_i8.h"
#include "op_fixtures.h"
static ame_i32 output[OP_MAX_C+32] __attribute__((aligned(16)));
static ame_i32 workspace[3*MAX_N*MAX_N+32] __attribute__((aligned(16)));
#define CANARY ((ame_i32) 0x6b91d237)

int run_kernel (const u32 *a0, const u32 *b0, u32 *c0, u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t = CSR_READ (amenlen), bytes = ame_ops_widen8_workspace_bytes (t);
  unsigned selected = 0;
  check (0x50000000,12*t*t,bytes);
  for (unsigned v = 0; v < OP_CASES; ++v)
    {
      const struct op_fixture *f = &op_fixtures[v];
      if (f->tile != t) continue;
      ++selected;
      for (ame_size i = 0; i < OP_MAX_C+32; ++i) output[i] = CANARY;
      for (ame_size i = 0; i < 3*MAX_N*MAX_N+32; ++i) workspace[i] = CANARY;
      for (ame_size i = 0; i < f->c_count; ++i) output[16+i] = f->initial[i];
      const ame_u8 *a = f->a+OP_INPUT_GUARD, *b = f->b+OP_INPUT_GUARD;
      int status;
#if OP_GEMV
      if (f->is_unsigned)
        status = ame_gemv_u8_i32_owned (f->m,f->k,a,f->lda,b,output+16,
                                        f->accumulate,workspace+16,bytes);
      else
        status = ame_gemv_i8_i32_owned (f->m,f->k,(const ame_i8 *) a,f->lda,
                                        (const ame_i8 *) b,output+16,
                                        f->accumulate,workspace+16,bytes);
#else
      if (f->is_unsigned)
        status = ame_gemm_u8_i32_owned (f->m,f->n,f->k,a,f->lda,b,f->ldb,
                                        output+16,f->ldc,f->accumulate,workspace+16,bytes);
      else
        status = ame_gemm_i8_i32_owned (f->m,f->n,f->k,(const ame_i8 *) a,f->lda,
                                        (const ame_i8 *) b,f->ldb,output+16,f->ldc,
                                        f->accumulate,workspace+16,bytes);
#endif
      check (0x51000000+v,AME_OPS_OK,status);
      for (ame_size i = 0; i < OP_MAX_C+32; ++i)
        {
          ame_size j = i-16;
          ame_i32 want = j < f->c_count ? f->golden[j] : CANARY;
          check (0x52000000+v*0x10000+i,(u32) want,(u32) output[i]);
        }
      for (ame_size i = 0; i < 3*MAX_N*MAX_N+32; ++i)
        if (i < 16 || i >= 16+3*t*t)
          check (0x53000000+i,(u32) CANARY,(u32) workspace[i]);
      for (ame_size i = 0; i < f->a_count+2*OP_INPUT_GUARD; ++i)
        check (0x54000000+i,f->a_copy[i],f->a[i]);
      for (ame_size i = 0; i < f->b_count+2*OP_INPUT_GUARD; ++i)
        check (0x55000000+i,f->b_copy[i],f->b[i]);
    }
  check (0x50000001,OP_SELECTED_CASES,selected);
  return 0;
}
