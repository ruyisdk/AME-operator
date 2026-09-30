#include "test.h"
#include "../include/ame_ops.h"
#include "op_fixtures.h"
static ame_i32 input_a[OP_WORDS] __attribute__((aligned(16)));
static ame_i32 input_b[OP_WORDS] __attribute__((aligned(16)));
static ame_i32 output[OP_WORDS] __attribute__((aligned(16)));
static ame_i32 workspace[3*MAX_N*MAX_N+32] __attribute__((aligned(16)));

static int invoke (unsigned action, unsigned iteration, ame_size bytes)
{
  ame_size m=2, n=OP_GEMV ? 1 : 2, k=3, lda=5;
  ame_size ldb=OP_GEMV ? 1 : 4, ldc=OP_GEMV ? 1 : 4, size=bytes;
  const ame_i32 *a=input_a+OP_GUARD, *b=input_b+OP_GUARD;
  ame_i32 *c=output+OP_GUARD;
  void *ws=workspace+16;
  int add=0;
  switch (action)
    {
    case OP_VALID_SET: break;
    case OP_VALID_ADD: add=1; break;
    case OP_REPEATED: add=iteration != 0; break;
    case OP_INPUT_SHARED: m=n=k=1; b=a; break;
    case OP_ZERO_ROWS: m=0; a=b=0; c=0; ws=0; break;
    case OP_ZERO_COLS: n=0; a=b=0; c=0; ws=0; break;
    case OP_ZERO_K_SET: k=0; a=b=0; ws=0; break;
    case OP_ZERO_K_ADD: k=0; a=b=0; ws=0; add=1; break;
    case OP_BAD_MODE: add=2; break;
    case OP_NULL_A: a=0; break;
    case OP_NULL_B: b=0; break;
    case OP_NULL_C: c=0; break;
    case OP_UNALIGNED_A: a=(const ame_i32 *) ((ame_uintptr) a+1); break;
    case OP_UNALIGNED_B: b=(const ame_i32 *) ((ame_uintptr) b+1); break;
    case OP_UNALIGNED_C: c=(ame_i32 *) ((ame_uintptr) c+1); break;
    case OP_SHORT_LDA: lda=k-1; break;
    case OP_SHORT_LDB: ldb=n-1; break;
    case OP_SHORT_LDC: ldc=n-1; break;
    case OP_C_ALIAS_A: c=input_a+OP_GUARD; break;
    case OP_C_ALIAS_B: c=input_b+OP_GUARD; break;
    case OP_C_OVERLAP_A: c=input_a+OP_GUARD+1; break;
    case OP_NULL_WS: ws=0; break;
    case OP_SHORT_WS: --size; break;
    case OP_UNALIGNED_WS: ws=(void *) ((ame_uintptr) ws+1); break;
    case OP_WS_ALIAS_A: ws=input_a+OP_GUARD; break;
    case OP_WS_ALIAS_B: ws=input_b+OP_GUARD; break;
    case OP_WS_ALIAS_C: ws=output+OP_GUARD; break;
    case OP_INPUT_SPAN_OVERFLOW: lda=(ame_size) -1; break;
    case OP_OUTPUT_SPAN_OVERFLOW: m=(ame_size) -1; break;
    case OP_INPUT_ADDRESS_WRAP: a=(const ame_i32 *) ((ame_uintptr) -1 & ~(ame_uintptr) 3); break;
    case OP_OUTPUT_ADDRESS_WRAP: c=(ame_i32 *) ((ame_uintptr) -1 & ~(ame_uintptr) 3); break;
    case OP_WS_ADDRESS_WRAP: ws=(void *) ((ame_uintptr) -1 & ~(ame_uintptr) 15); break;
    case OP_INPUT_BYTES_OVERFLOW: k=(ame_size) -1/4+1; lda=k; break;
    case OP_LARGE_WS_SIZE: size=(ame_size) -1; break;
    default: return -1;
    }
#if OP_GEMV
  (void) n; (void) ldb; (void) ldc;
  return ame_gemv_i32_owned (m,k,a,lda,b,c,add,ws,size);
#else
  return ame_gemm_i32_owned (m,n,k,a,lda,b,ldb,c,ldc,add,ws,size);
#endif
}

int run_kernel (const u32 *a0, const u32 *b0, u32 *c0, u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t=CSR_READ (amenlen);
  ame_size bytes=ame_ops_i32_workspace_bytes (t);
  check (0x60000000, 3*t*t*4, bytes);
  check (0x60000001, 1, CSR_READ (ameown));
  for (unsigned phase=0; phase<3; ++phase)
    {
      if (phase == 1)
        {
          release ();
          check (0x60000002, 0, CSR_READ (ameown));
        }
      if (phase == 2)
        {
          check (0x60000003, 1, acquire (0) & 1);
          check (0x60000004, 1, CSR_READ (ameown));
        }
      for (unsigned v=0; v<OP_CASES; ++v)
        {
          const struct op_state_case *f=&op_state_cases[v];
          if (f->phase != phase) continue;
          for (unsigned i=0; i<OP_WORDS; ++i)
            {
              input_a[i]=op_initial_a[i];
              input_b[i]=op_initial_b[i];
              output[i]=op_initial_c[i];
            }
          for (unsigned i=0; i<3*MAX_N*MAX_N+32; ++i)
            workspace[i]=OP_CANARY;
          if (phase != 1)
            {
              CSR_WRITE (amestatus, 0);
              CSR_WRITE (amestype, 32);
              CSR_WRITE (amefflags, 0x11);
              CSR_WRITE (amexsat, 1);
            }
          for (unsigned j=0; j<f->calls; ++j)
            check (0x61000000+v*16+j, f->status, invoke (f->action,j,bytes));
          check (0x62000000+v, phase != 1, CSR_READ (ameown));
          if (phase != 1)
            {
              check (0x63000000+v*16, 0, CSR_READ (amestatus));
              check (0x63000001+v*16, 0x11, CSR_READ (amefflags));
              check (0x63000002+v*16, 1, CSR_READ (amexsat));
              if (!f->computed)
                check (0x63000003+v*16, 32, CSR_READ (amestype));
            }
          for (unsigned i=0; i<OP_WORDS; ++i)
            {
              check (0x64000000+v*256+i, (u32) op_initial_a[i], (u32) input_a[i]);
              check (0x64000040+v*256+i, (u32) op_initial_b[i], (u32) input_b[i]);
              check (0x64000080+v*256+i, (u32) f->golden[i], (u32) output[i]);
            }
          for (unsigned i=0; i<3*MAX_N*MAX_N+32; ++i)
            if (!f->computed || i<16 || i>=16+3*t*t)
              check (0x65000000+i, OP_CANARY, (u32) workspace[i]);
        }
    }
  CSR_WRITE (amestatus, 0);
  CSR_WRITE (amefflags, 0);
  CSR_WRITE (amexsat, 0);
  return 0;
}
