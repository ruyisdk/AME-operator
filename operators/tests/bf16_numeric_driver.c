#include "test.h"
#include "../include/ame_ops_bf16.h"
#include "op_fixtures.h"
static ame_bf16 input_a[OP_MAX_A+32] __attribute__((aligned(16)));
static ame_bf16 input_b[OP_MAX_B+32] __attribute__((aligned(16)));
static float output[OP_MAX_C+32] __attribute__((aligned(16)));
static ame_bf16 wa[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static ame_bf16 wb[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static float wc[MAX_N*MAX_N+32] __attribute__((aligned(16)));
#define CANARY16 0x7f81u
#define CANARY32 0x7f812345u
static inline __attribute__((always_inline)) void put16 (ame_bf16 *p,u32 v)
{ unsigned short bits=v; __builtin_memcpy (p,&bits,2); }
static inline __attribute__((always_inline)) u32 get16 (const ame_bf16 *p)
{ unsigned short bits; __builtin_memcpy (&bits,p,2); return bits; }
static inline __attribute__((always_inline)) void put32 (float *p,u32 bits)
{ __builtin_memcpy (p,&bits,4); }
static inline __attribute__((always_inline)) u32 get32 (const float *p)
{ u32 bits; __builtin_memcpy (&bits,p,4); return bits; }

int run_kernel (const u32 *a0,const u32 *b0,u32 *c0,u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t=CSR_READ (amenlen),ab=ame_ops_bf16_input_elements (t);
  ame_size cc=ame_ops_bf16_output_elements (t);
  check (0x78000000,2*t*t,ab);
  check (0x78000001,t*t,cc);
  check (0x78000002,1,CSR_READ (ameown));
  usize oldfs=CSR_READ (mstatus) & 0x6000u;
  CSR_WRITE (mstatus,CSR_READ (mstatus) | 0x2000u);
  usize oldfcsr=CSR_READ (fcsr);
  unsigned selected=0;
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
        { put16 (&wa[i],CANARY16); put16 (&wb[i],CANARY16); }
      for (ame_size i=0; i<MAX_N*MAX_N+32; ++i) put32 (&wc[i],CANARY32);
      struct ame_ops_bf16_workspace w={wa+16,ab,wb+16,ab,wc+16,cc};
      CSR_WRITE (fcsr,f->fcsr);
      CSR_WRITE (amestatus,0);
      CSR_WRITE (amefflags,f->seed_flags);
      CSR_WRITE (amexsat,f->seed_xsat);
      int status;
#if OP_GEMV
      status=ame_gemv_bf16_f32_owned (f->m,f->k,input_a+16,f->lda,input_b+16,
                                      output+16,f->accumulate,&w);
#else
      status=ame_gemm_bf16_f32_owned (f->m,f->n,f->k,input_a+16,f->lda,input_b+16,
                                      f->ldb,output+16,f->ldc,f->accumulate,&w);
#endif
      check (0x79000000+v,AME_OPS_OK,status);
      check (0x79010000+v,0,CSR_READ (amestatus));
      check (0x79020000+v,f->seed_flags | f->new_flags,CSR_READ (amefflags));
      check (0x79030000+v,f->seed_xsat,CSR_READ (amexsat));
      check (0x79040000+v,f->fcsr,CSR_READ (fcsr));
      check (0x79050000+v,1,CSR_READ (ameown));
      check (0x79060000+v,1,w.a==wa+16); check (0x79070000+v,1,w.a_elements==ab);
      check (0x79080000+v,1,w.b==wb+16); check (0x79090000+v,1,w.b_elements==ab);
      check (0x790a0000+v,1,w.c==wc+16); check (0x790b0000+v,1,w.c_elements==cc);
      for (ame_size i=0; i<OP_MAX_C+32; ++i)
        check (0x7a000000+v*0x10000+i,i-16<f->c_count ? f->golden[i-16] : CANARY32,get32 (&output[i]));
      for (ame_size i=0; i<OP_MAX_A+32; ++i)
        check (0x7b000000+i,i-16<f->a_count ? f->a[i-16] : CANARY16,get16 (&input_a[i]));
      for (ame_size i=0; i<OP_MAX_B+32; ++i)
        check (0x7c000000+i,i-16<f->b_count ? f->b[i-16] : CANARY16,get16 (&input_b[i]));
      for (ame_size i=0; i<2*MAX_N*MAX_N+32; ++i)
        if (i<16 || i>=16+ab)
          { check (0x7d000000+i,CANARY16,get16 (&wa[i])); check (0x7e000000+i,CANARY16,get16 (&wb[i])); }
      for (ame_size i=0; i<MAX_N*MAX_N+32; ++i)
        if (i<16 || i>=16+cc) check (0x7f000000+i,CANARY32,get32 (&wc[i]));
    }
  check (0x78000003,OP_SELECTED_CASES,selected);
  CSR_WRITE (fcsr,oldfcsr);
  CSR_WRITE (mstatus,(CSR_READ (mstatus) & ~(usize) 0x6000u) | oldfs);
  CSR_WRITE (amefflags,0);
  CSR_WRITE (amexsat,0);
  return 0;
}
