#include "test.h"
#include "../include/ame_ops_bf16.h"
#include "op_fixtures.h"
static ame_bf16 input_a[OP_WORDS] __attribute__((aligned(16)));
static ame_bf16 input_b[OP_WORDS] __attribute__((aligned(16)));
static float output[OP_WORDS] __attribute__((aligned(16)));
static ame_bf16 workspace_a[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static ame_bf16 workspace_b[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static float workspace_c[MAX_N*MAX_N+32] __attribute__((aligned(16)));
static inline __attribute__((always_inline)) void put16 (ame_bf16 *p,op_u16 bits)
{ __builtin_memcpy (p,&bits,sizeof bits); }
static inline __attribute__((always_inline)) u32 get16 (const ame_bf16 *p)
{ op_u16 bits; __builtin_memcpy (&bits,p,sizeof bits); return bits; }
static inline __attribute__((always_inline)) void put32 (float *p,u32 bits)
{ __builtin_memcpy (p,&bits,sizeof bits); }
static inline __attribute__((always_inline)) u32 get32 (const float *p)
{ u32 bits; __builtin_memcpy (&bits,p,sizeof bits); return bits; }

static int invoke (unsigned action,unsigned iteration,ame_size in,ame_size out)
{
  ame_size m=2,n=OP_GEMV ? 1 : 2,k=3,lda=5,ldb=OP_GEMV ? 1 : 4,ldc=OP_GEMV ? 1 : 4;
  const ame_bf16 *a=input_a+OP_GUARD,*b=input_b+OP_GUARD;
  float *c=output+OP_GUARD;
  struct ame_ops_bf16_workspace w __attribute__((aligned(16)))=
    {workspace_a+16,in,workspace_b+16,in,workspace_c+16,out};
  const struct ame_ops_bf16_workspace *descriptor=&w;
  int add=0;
  switch (action)
    {
    case OP_VALID_SET: break;
    case OP_VALID_ADD: add=1; break;
    case OP_REPEATED: add=iteration!=0; break;
    case OP_INPUT_SHARED: m=n=k=1; b=a; break;
    case OP_LARGE_CAPACITY: w.a_elements=w.b_elements=w.c_elements=(ame_size) -1; break;
    case OP_INPUT_OFFSET_TWO: ++a; ++b; break;
    case OP_OUTPUT_OFFSET_FOUR: ++c; break;
    case OP_ZERO_ROWS: m=0; a=b=0; c=0; descriptor=0; break;
    case OP_ZERO_COLS: n=0; a=b=0; c=0; descriptor=0; break;
    case OP_ZERO_K_SET: case OP_ZERO_K_SPECIAL_SET: k=0; a=b=0; descriptor=0; break;
    case OP_ZERO_K_ADD: case OP_ZERO_K_SPECIAL_ADD: k=0; a=b=0; descriptor=0; add=1; break;
    case OP_BAD_MODE: add=2; break;
    case OP_BAD_MODE_EMPTY: add=2; m=0; a=b=0; c=0; descriptor=0; break;
    case OP_NULL_A: a=0; break;
    case OP_NULL_B: b=0; break;
    case OP_NULL_C: c=0; break;
    case OP_UNALIGNED_A: a=(const ame_bf16 *) ((ame_uintptr) a+1); break;
    case OP_UNALIGNED_B: b=(const ame_bf16 *) ((ame_uintptr) b+1); break;
    case OP_UNALIGNED_C: c=(float *) ((ame_uintptr) c+1); break;
    case OP_SHORT_LDA: lda=k-1; break;
    case OP_SHORT_LDB: ldb=n-1; break;
    case OP_SHORT_LDC: ldc=n-1; break;
    case OP_C_ALIAS_A: c=(float *) a; break;
    case OP_C_ALIAS_B: c=(float *) b; break;
    case OP_C_OVERLAP_A: c=(float *) (a+2); break;
    case OP_C_OVERLAP_B: c=(float *) (b+2); break;
    case OP_NULL_DESC: descriptor=0; break;
    case OP_UNALIGNED_DESC:
      descriptor=(const struct ame_ops_bf16_workspace *) ((ame_uintptr) &w+1); break;
    case OP_DESC_ADDRESS_WRAP:
      descriptor=(const struct ame_ops_bf16_workspace *) ((ame_uintptr) -1 & ~(ame_uintptr) 15); break;
    case OP_DESC_ALIAS_A: descriptor=(const struct ame_ops_bf16_workspace *) a; break;
    case OP_DESC_ALIAS_B: descriptor=(const struct ame_ops_bf16_workspace *) b; break;
    case OP_DESC_ALIAS_C: descriptor=(const struct ame_ops_bf16_workspace *) c; break;
    case OP_NULL_WS_A: w.a=0; break;
    case OP_NULL_WS_B: w.b=0; break;
    case OP_NULL_WS_C: w.c=0; break;
    case OP_SHORT_WS_A: --w.a_elements; break;
    case OP_SHORT_WS_B: --w.b_elements; break;
    case OP_SHORT_WS_C: --w.c_elements; break;
    case OP_UNALIGNED_WS_A: ++w.a; break;
    case OP_UNALIGNED_WS_B: ++w.b; break;
    case OP_UNALIGNED_WS_C: ++w.c; break;
    case OP_WS_A_ALIAS_A: w.a=(ame_bf16 *) a; break;
    case OP_WS_A_ALIAS_B: w.a=(ame_bf16 *) b; break;
    case OP_WS_A_ALIAS_C: w.a=(ame_bf16 *) c; break;
    case OP_WS_B_ALIAS_A: w.b=(ame_bf16 *) a; break;
    case OP_WS_B_ALIAS_B: w.b=(ame_bf16 *) b; break;
    case OP_WS_B_ALIAS_C: w.b=(ame_bf16 *) c; break;
    case OP_WS_C_ALIAS_A: w.c=(float *) a; break;
    case OP_WS_C_ALIAS_B: w.c=(float *) b; break;
    case OP_WS_C_ALIAS_C: w.c=c; break;
    case OP_WS_A_ALIAS_B_WORK: w.a=w.b; break;
    case OP_WS_A_ALIAS_C_WORK: w.a=(ame_bf16 *) w.c; break;
    case OP_WS_B_ALIAS_C_WORK: w.b=(ame_bf16 *) w.c; break;
    case OP_WS_A_ALIAS_DESC: w.a=(ame_bf16 *) &w; break;
    case OP_WS_B_ALIAS_DESC: w.b=(ame_bf16 *) &w; break;
    case OP_WS_C_ALIAS_DESC: w.c=(float *) &w; break;
    case OP_WS_A_ADDRESS_WRAP: w.a=(ame_bf16 *) ((ame_uintptr) -1 & ~(ame_uintptr) 15); break;
    case OP_WS_B_ADDRESS_WRAP: w.b=(ame_bf16 *) ((ame_uintptr) -1 & ~(ame_uintptr) 15); break;
    case OP_WS_C_ADDRESS_WRAP: w.c=(float *) ((ame_uintptr) -1 & ~(ame_uintptr) 15); break;
    case OP_INPUT_SPAN_OVERFLOW: lda=(ame_size) -1; break;
    case OP_B_SPAN_OVERFLOW: ldb=(ame_size) -1; break;
    case OP_OUTPUT_SPAN_OVERFLOW: m=(ame_size) -1; break;
    case OP_INPUT_BYTES_OVERFLOW: m=1; k=(ame_size) -1/2+1; lda=k; break;
    case OP_B_BYTES_OVERFLOW: ldb=(ame_size) -1/4+1; break;
    case OP_OUTPUT_BYTES_OVERFLOW: m=(ame_size) -1/4+1; n=ldc=1; break;
    case OP_INPUT_ADDRESS_WRAP: a=(const ame_bf16 *) ((ame_uintptr) -1 & ~(ame_uintptr) 1); break;
    case OP_B_ADDRESS_WRAP: b=(const ame_bf16 *) ((ame_uintptr) -1 & ~(ame_uintptr) 1); break;
    case OP_OUTPUT_ADDRESS_WRAP: c=(float *) ((ame_uintptr) -1 & ~(ame_uintptr) 3); break;
    default: return -1;
    }
  struct ame_ops_bf16_workspace before=w;
  int result;
#if OP_GEMV
  (void) n; (void) ldb; (void) ldc;
  result=ame_gemv_bf16_f32_owned (m,k,a,lda,b,c,add,descriptor);
#else
  result=ame_gemm_bf16_f32_owned (m,n,k,a,lda,b,ldb,c,ldc,add,descriptor);
#endif
  check (0x71000000+action*8,1,w.a==before.a);
  check (0x71000001+action*8,1,w.a_elements==before.a_elements);
  check (0x71000002+action*8,1,w.b==before.b);
  check (0x71000003+action*8,1,w.b_elements==before.b_elements);
  check (0x71000004+action*8,1,w.c==before.c);
  check (0x71000005+action*8,1,w.c_elements==before.c_elements);
  return result;
}

int run_kernel (const u32 *a0,const u32 *b0,u32 *c0,u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t=CSR_READ (amenlen),in=ame_ops_bf16_input_elements (t),out=ame_ops_bf16_output_elements (t);
  check (0x70000000,2*t*t,in);
  check (0x70000001,t*t,out);
  check (0x70000002,1,CSR_READ (ameown));
  usize oldfs=CSR_READ (mstatus) & 0x6000u;
  CSR_WRITE (mstatus,CSR_READ (mstatus) | 0x2000u);
  usize oldfcsr=CSR_READ (fcsr);
  for (unsigned phase=0; phase<3; ++phase)
    {
      if (phase==1)
        {
          release (); check (0x70000003,0,CSR_READ (ameown));
          const ame_size invalid[]={0,3,5,256,(ame_size) -1};
          for (unsigned i=0; i<5; ++i)
            {
              check (0x70000010+i*2,0,ame_ops_bf16_input_elements (invalid[i]));
              check (0x70000011+i*2,0,ame_ops_bf16_output_elements (invalid[i]));
            }
        }
      if (phase==2)
        { check (0x70000004,1,acquire (0) & 1); check (0x70000005,1,CSR_READ (ameown)); }
      for (unsigned v=0; v<OP_CASES; ++v)
        {
          const struct op_state_case *f=&op_state_cases[v];
          if (f->phase!=phase) continue;
          for (unsigned i=0; i<OP_WORDS; ++i)
            { put16 (&input_a[i],f->a[i]); put16 (&input_b[i],f->b[i]); put32 (&output[i],f->initial[i]); }
          for (unsigned i=0; i<2*MAX_N*MAX_N+32; ++i)
            { put16 (&workspace_a[i],OP_CANARY16); put16 (&workspace_b[i],OP_CANARY16); }
          for (unsigned i=0; i<MAX_N*MAX_N+32; ++i) put32 (&workspace_c[i],OP_CANARY32);
          CSR_WRITE (fcsr,f->fcsr);
          if (phase!=1)
            {
              CSR_WRITE (amestatus,0); CSR_WRITE (amestype,32);
              CSR_WRITE (amefflags,f->fflags); CSR_WRITE (amexsat,f->xsat);
            }
          for (unsigned j=0; j<f->calls; ++j)
            check (0x72000000+v*16+j,f->status,invoke (f->action,j,in,out));
          check (0x73000000+v,phase!=1,CSR_READ (ameown));
          check (0x73010000+v,f->fcsr,CSR_READ (fcsr));
          if (phase!=1)
            {
              check (0x73020000+v*8,0,CSR_READ (amestatus));
              check (0x73020001+v*8,f->fflags,CSR_READ (amefflags));
              check (0x73020002+v*8,f->xsat,CSR_READ (amexsat));
              if (!f->computed) check (0x73020003+v*8,32,CSR_READ (amestype));
            }
          for (unsigned i=0; i<OP_WORDS; ++i)
            {
              check (0x74000000+v*256+i,f->a[i],get16 (&input_a[i]));
              check (0x74000040+v*256+i,f->b[i],get16 (&input_b[i]));
              check (0x74000080+v*256+i,f->golden[i],get32 (&output[i]));
            }
          for (unsigned i=0; i<2*MAX_N*MAX_N+32; ++i)
            if (!f->computed || i<16 || i>=16+in)
              { check (0x75000000+i,OP_CANARY16,get16 (&workspace_a[i]));
                check (0x76000000+i,OP_CANARY16,get16 (&workspace_b[i])); }
          for (unsigned i=0; i<MAX_N*MAX_N+32; ++i)
            if (!f->computed || i<16 || i>=16+out)
              check (0x77000000+i,OP_CANARY32,get32 (&workspace_c[i]));
        }
    }
  CSR_WRITE (fcsr,oldfcsr);
  CSR_WRITE (mstatus,(CSR_READ (mstatus) & ~(usize) 0x6000u) | oldfs);
  CSR_WRITE (amestatus,0); CSR_WRITE (amefflags,0); CSR_WRITE (amexsat,0);
  return 0;
}
