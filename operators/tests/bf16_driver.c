#include "test.h"
#include "../include/ame_ops_bf16.h"
#include "op_fixtures.h"
static ame_bf16 input_a[OP_MAX_A+32] __attribute__((aligned(16)));
static ame_bf16 input_b[OP_MAX_B+32] __attribute__((aligned(16)));
static float output[OP_MAX_C+32] __attribute__((aligned(16)));
static ame_bf16 workspace_a[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static ame_bf16 workspace_b[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static float workspace_c[MAX_N*MAX_N+32] __attribute__((aligned(16)));
#define CANARY16 0x7f81u
#define CANARY32 0x7f812345u
static inline __attribute__((always_inline)) void put16 (ame_bf16 *p,op_u16 bits)
{ __builtin_memcpy (p,&bits,sizeof bits); }
static inline __attribute__((always_inline)) u32 get16 (const ame_bf16 *p)
{ op_u16 bits; __builtin_memcpy (&bits,p,sizeof bits); return bits; }
static inline __attribute__((always_inline)) void put32 (float *p,u32 bits)
{ __builtin_memcpy (p,&bits,sizeof bits); }
static inline __attribute__((always_inline)) u32 get32 (const float *p)
{ u32 bits; __builtin_memcpy (&bits,p,sizeof bits); return bits; }

int run_kernel (const u32 *a0,const u32 *b0,u32 *c0,u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t=CSR_READ (amenlen),in=ame_ops_bf16_input_elements (t),out=ame_ops_bf16_output_elements (t);
  unsigned selected=0;
  check (0x80000000,2*t*t,in);
  check (0x80000001,t*t,out);
  for (unsigned v=0; v<OP_CASES; ++v)
    {
      const struct op_fixture *f=&op_fixtures[v];
      if (f->tile!=t) continue;
      ++selected;
      for (ame_size i=0; i<OP_MAX_A+32; ++i)
        put16 (&input_a[i],i-16<f->a_count ? f->a[i-16] : CANARY16);
      for (ame_size i=0; i<OP_MAX_B+32; ++i)
        put16 (&input_b[i],i-16<f->b_count ? f->b[i-16] : CANARY16);
      for (ame_size i=0; i<OP_MAX_C+32; ++i)
        put32 (&output[i],i-16<f->c_count ? f->initial[i-16] : CANARY32);
      for (ame_size i=0; i<2*MAX_N*MAX_N+32; ++i)
        { put16 (&workspace_a[i],CANARY16); put16 (&workspace_b[i],CANARY16); }
      for (ame_size i=0; i<MAX_N*MAX_N+32; ++i) put32 (&workspace_c[i],CANARY32);
      struct ame_ops_bf16_workspace w={workspace_a+16,in,workspace_b+16,in,workspace_c+16,out};
      u32 flags=f->accumulate ? 0x11 : 0;
      CSR_WRITE (amestatus,0);
      CSR_WRITE (amefflags,flags);
      CSR_WRITE (amexsat,f->accumulate);
      int status;
#if OP_GEMV
      status=ame_gemv_bf16_f32_owned (f->m,f->k,input_a+16,f->lda,input_b+16,output+16,f->accumulate,&w);
#else
      status=ame_gemm_bf16_f32_owned (f->m,f->n,f->k,input_a+16,f->lda,input_b+16,f->ldb,
                                     output+16,f->ldc,f->accumulate,&w);
#endif
      check (0x81000000+v,AME_OPS_OK,status);
      check (0x82000000+v,0,CSR_READ (amestatus));
      check (0x82010000+v,flags,CSR_READ (amefflags));
      check (0x82020000+v,f->accumulate,CSR_READ (amexsat));
      check (0x82030000+v,1,CSR_READ (ameown));
      check (0x82100000+v*8,1,w.a==workspace_a+16);
      check (0x82100001+v*8,in,w.a_elements);
      check (0x82100002+v*8,1,w.b==workspace_b+16);
      check (0x82100003+v*8,in,w.b_elements);
      check (0x82100004+v*8,1,w.c==workspace_c+16);
      check (0x82100005+v*8,out,w.c_elements);
      for (ame_size i=0; i<OP_MAX_C+32; ++i)
        check (0x83000000+v*0x10000+i,i-16<f->c_count ? f->golden[i-16] : CANARY32,get32 (&output[i]));
      for (ame_size i=0; i<OP_MAX_A+32; ++i)
        check (0x84000000+i,i-16<f->a_count ? f->a[i-16] : CANARY16,get16 (&input_a[i]));
      for (ame_size i=0; i<OP_MAX_B+32; ++i)
        check (0x85000000+i,i-16<f->b_count ? f->b[i-16] : CANARY16,get16 (&input_b[i]));
      int computed=f->m && f->n && f->k;
      for (ame_size i=0; i<2*MAX_N*MAX_N+32; ++i)
        if (!computed || i<16 || i>=16+in)
          {
            check (0x86000000+i,CANARY16,get16 (&workspace_a[i]));
            check (0x87000000+i,CANARY16,get16 (&workspace_b[i]));
          }
      for (ame_size i=0; i<MAX_N*MAX_N+32; ++i)
        if (!computed || i<16 || i>=16+out) check (0x88000000+i,CANARY32,get32 (&workspace_c[i]));
    }
  check (0x80000002,OP_SELECTED_CASES,selected);
  CSR_WRITE (amefflags,0);
  CSR_WRITE (amexsat,0);
  return 0;
}
