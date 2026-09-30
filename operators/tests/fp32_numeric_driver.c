#include "test.h"
#include "../include/ame_ops_f32.h"
#include "op_fixtures.h"
static float input_a[OP_MAX_A+32] __attribute__((aligned(16)));
static float input_b[OP_MAX_B+32] __attribute__((aligned(16)));
static float output[OP_MAX_C+32] __attribute__((aligned(16)));
static float workspace[3*MAX_N*MAX_N+32] __attribute__((aligned(16)));
#define CANARY 0x7f812345u
static inline __attribute__((always_inline)) void put (float *p,u32 bits)
{ __builtin_memcpy (p,&bits,sizeof bits); }
static inline __attribute__((always_inline)) u32 get (const float *p)
{ u32 bits; __builtin_memcpy (&bits,p,sizeof bits); return bits; }

int run_kernel (const u32 *a0,const u32 *b0,u32 *c0,u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t=CSR_READ (amenlen),bytes=ame_ops_f32_workspace_bytes (t);
  unsigned selected=0;
  check (0x78000000,12*t*t,bytes);
  check (0x78000001,1,CSR_READ (ameown));
  usize oldfs=CSR_READ (mstatus) & 0x6000u;
  CSR_WRITE (mstatus,CSR_READ (mstatus) | 0x2000u);
  usize oldfcsr=CSR_READ (fcsr);
  for (unsigned v=0; v<OP_CASES; ++v)
    {
      const struct op_fixture *f=&op_fixtures[v];
      if (f->tile!=t) continue;
      ++selected;
      for (ame_size i=0; i<OP_MAX_A+32; ++i)
        put (&input_a[i],i-16<f->a_count ? f->a[i-16] : CANARY);
      for (ame_size i=0; i<OP_MAX_B+32; ++i)
        put (&input_b[i],i-16<f->b_count ? f->b[i-16] : CANARY);
      for (ame_size i=0; i<OP_MAX_C+32; ++i)
        put (&output[i],i-16<f->c_count ? f->initial[i-16] : CANARY);
      for (ame_size i=0; i<3*MAX_N*MAX_N+32; ++i) put (&workspace[i],CANARY);
      CSR_WRITE (fcsr,f->fcsr);
      CSR_WRITE (amestatus,0);
      CSR_WRITE (amefflags,f->seed_flags);
      CSR_WRITE (amexsat,f->seed_xsat);
      int status;
#if OP_GEMV
      status=ame_gemv_f32_owned (f->m,f->k,input_a+16,f->lda,input_b+16,output+16,
                                f->accumulate,workspace+16,bytes);
#else
      status=ame_gemm_f32_owned (f->m,f->n,f->k,input_a+16,f->lda,input_b+16,f->ldb,
                                output+16,f->ldc,f->accumulate,workspace+16,bytes);
#endif
      check (0x79000000+v,AME_OPS_OK,status);
      check (0x79010000+v,0,CSR_READ (amestatus));
      check (0x79020000+v,f->seed_flags | f->new_flags,CSR_READ (amefflags));
      check (0x79030000+v,f->seed_xsat,CSR_READ (amexsat));
      check (0x79040000+v,f->fcsr,CSR_READ (fcsr));
      check (0x79050000+v,1,CSR_READ (ameown));
      for (ame_size i=0; i<OP_MAX_C+32; ++i)
        check (0x7a000000+v*0x10000+i,i-16<f->c_count ? f->golden[i-16] : CANARY,get (&output[i]));
      for (ame_size i=0; i<OP_MAX_A+32; ++i)
        check (0x7b000000+i,i-16<f->a_count ? f->a[i-16] : CANARY,get (&input_a[i]));
      for (ame_size i=0; i<OP_MAX_B+32; ++i)
        check (0x7c000000+i,i-16<f->b_count ? f->b[i-16] : CANARY,get (&input_b[i]));
      for (ame_size i=0; i<3*MAX_N*MAX_N+32; ++i)
        if (i<16 || i>=16+3*t*t) check (0x7d000000+i,CANARY,get (&workspace[i]));
    }
  check (0x78000002,OP_SELECTED_CASES,selected);
  CSR_WRITE (fcsr,oldfcsr);
  CSR_WRITE (mstatus,(CSR_READ (mstatus) & ~(usize) 0x6000u) | oldfs);
  CSR_WRITE (amefflags,0);
  CSR_WRITE (amexsat,0);
  return 0;
}
