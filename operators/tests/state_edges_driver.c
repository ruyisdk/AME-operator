#include "test.h"
#include "../include/ame_ops_i8.h"
#include "../include/ame_ops_bf16.h"
#include "op_fixtures.h"
static ame_i32 a32[OP_WORDS],b32[OP_WORDS],ci32[OP_WORDS];
static ame_i8 a8[OP_WORDS],b8[OP_WORDS];
static ame_u8 au8[OP_WORDS],bu8[OP_WORDS];
static ame_i32 ci8[OP_WORDS],cu8[OP_WORDS];
static float af[OP_WORDS],bf[OP_WORDS],cf[OP_WORDS],cbf[OP_WORDS];
static ame_bf16 abf[OP_WORDS],bbf[OP_WORDS];
#define WORK_WORDS (3*MAX_N*MAX_N+96)
#define BF_WORDS (MAX_N*MAX_N+32)
static ame_i32 wi[WORK_WORDS] __attribute__((aligned(16)));
static ame_i32 w8[WORK_WORDS] __attribute__((aligned(16)));
static float wf[WORK_WORDS] __attribute__((aligned(16)));
static ame_bf16 wba[2*BF_WORDS] __attribute__((aligned(16)));
static ame_bf16 wbb[2*BF_WORDS] __attribute__((aligned(16)));
static float wbc[BF_WORDS] __attribute__((aligned(16)));
struct buffers { void *a,*b,*c,*work; };

static inline __attribute__((always_inline)) void put (void *base,ame_size i,unsigned width,u32 bits)
{
  unsigned char *p=(unsigned char *) base+i*width;
  if (width==1) *p=(unsigned char) bits;
  else if (width==2) { __UINT16_TYPE__ v=bits; __builtin_memcpy (p,&v,2); }
  else __builtin_memcpy (p,&bits,4);
}
static inline __attribute__((always_inline)) u32 get (const void *base,ame_size i,unsigned width)
{
  const unsigned char *p=(const unsigned char *) base+i*width;
  if (width==1) return *p;
  if (width==2) { __UINT16_TYPE__ v; __builtin_memcpy (&v,p,2); return v; }
  u32 v; __builtin_memcpy (&v,p,4); return v;
}
static struct buffers select_buffers (unsigned family)
{
  switch (family)
    {
    case 0: return (struct buffers) {a32,b32,ci32,wi};
    case 1: return (struct buffers) {a8,b8,ci8,w8};
    case 2: return (struct buffers) {au8,bu8,cu8,w8};
    case 3: return (struct buffers) {af,bf,cf,wf};
    default: return (struct buffers) {abf,bbf,cbf,0};
    }
}
static int invoke (const struct op_state_case *f,struct buffers b,ame_size t)
{
  ame_size m=1,n=1,k=1,ldc=1;
  int add=f->action==OP_ADD || f->action==OP_K0_ADD;
  void *a=(unsigned char *) b.a+OP_GUARD*f->width;
  void *rhs=(unsigned char *) b.b+OP_GUARD*f->width;
  void *c=(unsigned char *) b.c+OP_GUARD*4;
  void *work=b.work ? (unsigned char *) b.work+16*4 : 0;
  struct ame_ops_bf16_workspace w={wba+32,2*t*t,wbb+32,2*t*t,wbc+16,t*t};
  const struct ame_ops_bf16_workspace *desc=&w;
  if (f->action==OP_K0_SET || f->action==OP_K0_ADD)
    { m=OP_GEMV ? 8 : 4; n=OP_GEMV ? 1 : 2; ldc=OP_GEMV ? 1 : 3; k=0; a=rhs=work=0; desc=0; }
  if (f->action==OP_EMPTY) { m=0; a=rhs=c=work=0; desc=0; }
  if (f->action==OP_BAD_MODE) add=2;
  if (f->action==OP_NO_WORK) { work=0; desc=0; }
  if (f->action==OP_BAD_A) a=0;
  int status;
#if OP_GEMV
  (void) n; (void) ldc;
#define INVOKE(NAME,TYPE,OUT) NAME (m,k,(const TYPE *) a,1,(const TYPE *) rhs,(OUT *) c,add,work,12*t*t)
  switch (f->family)
    {
    case 0: status=INVOKE (ame_gemv_i32_owned,ame_i32,ame_i32); break;
    case 1: status=INVOKE (ame_gemv_i8_i32_owned,ame_i8,ame_i32); break;
    case 2: status=INVOKE (ame_gemv_u8_i32_owned,ame_u8,ame_i32); break;
    case 3: status=INVOKE (ame_gemv_f32_owned,float,float); break;
    default: status=ame_gemv_bf16_f32_owned (m,k,(const ame_bf16 *) a,1,
              (const ame_bf16 *) rhs,(float *) c,add,desc); break;
    }
#else
#define INVOKE(NAME,TYPE,OUT) NAME (m,n,k,(const TYPE *) a,1,(const TYPE *) rhs,n,(OUT *) c,ldc,add,work,12*t*t)
  switch (f->family)
    {
    case 0: status=INVOKE (ame_gemm_i32_owned,ame_i32,ame_i32); break;
    case 1: status=INVOKE (ame_gemm_i8_i32_owned,ame_i8,ame_i32); break;
    case 2: status=INVOKE (ame_gemm_u8_i32_owned,ame_u8,ame_i32); break;
    case 3: status=INVOKE (ame_gemm_f32_owned,float,float); break;
    default: status=ame_gemm_bf16_f32_owned (m,n,k,(const ame_bf16 *) a,1,
              (const ame_bf16 *) rhs,n,(float *) c,ldc,add,desc); break;
    }
#endif
#undef INVOKE
  check (0x81000000,1,w.a==wba+32); check (0x81000001,1,w.a_elements==2*t*t);
  check (0x81000002,1,w.b==wbb+32); check (0x81000003,1,w.b_elements==2*t*t);
  check (0x81000004,1,w.c==wbc+16); check (0x81000005,1,w.c_elements==t*t);
  return status;
}

int run_kernel (const u32 *a0,const u32 *b0,u32 *c0,u32 *d0)
{
  (void) a0; (void) b0; (void) c0; (void) d0;
  ame_size t=CSR_READ (amenlen);
  check (0x80000000,1,CSR_READ (ameown));
  check (0x80000001,12*t*t,ame_ops_i32_workspace_bytes (t));
  check (0x80000002,12*t*t,ame_ops_widen8_workspace_bytes (t));
  check (0x80000003,12*t*t,ame_ops_f32_workspace_bytes (t));
  check (0x80000004,2*t*t,ame_ops_bf16_input_elements (t));
  check (0x80000005,t*t,ame_ops_bf16_output_elements (t));
  usize oldfs=CSR_READ (mstatus) & 0x6000u;
  CSR_WRITE (mstatus,CSR_READ (mstatus) | 0x2000u);
  usize oldfcsr=CSR_READ (fcsr);
  for (unsigned phase=0; phase<3; ++phase)
    {
      if (phase==1) { release (); check (0x80000006,0,CSR_READ (ameown)); }
      if (phase==2) { check (0x80000007,1,acquire (0)&1); check (0x80000008,1,CSR_READ (ameown)); }
      for (unsigned v=0; v<OP_CASES; ++v)
        {
          const struct op_state_case *f=&op_cases[v];
          if (f->phase!=phase) continue;
          struct buffers b=select_buffers (f->family);
          for (unsigned i=0; i<OP_WORDS; ++i)
            { put (b.a,i,f->width,f->a[i]); put (b.b,i,f->width,f->b[i]); put (b.c,i,4,f->initial[i]); }
          for (ame_size i=0; i<WORK_WORDS; ++i)
            if (f->family!=4) put (b.work,i,4,OP_CANARY);
            else
              { void *base=i<BF_WORDS ? (void *) wba : i<2*BF_WORDS ? (void *) wbb : (void *) wbc;
                put (base,i%BF_WORDS,4,OP_CANARY); }
          CSR_WRITE (fcsr,f->fcsr);
          if (phase!=1)
            {
              CSR_WRITE (amestatus,0);
              if (f->old_un)
                __asm__ volatile ("msettyp m0,zero\n\tmadd.ew m0,m0,m0" : : : "m0","memory");
              check (0x82000000+v,f->old_un,CSR_READ (amestatus));
              CSR_WRITE (amestype,32); CSR_WRITE (amefflags,f->flags); CSR_WRITE (amexsat,f->xsat);
            }
          check (0x83000000+v,f->status,invoke (f,b,t));
          check (0x84000000+v,phase!=1,CSR_READ (ameown));
          check (0x84010000+v,f->fcsr,CSR_READ (fcsr));
          if (phase!=1)
            {
              check (0x84020000+v,f->old_un,CSR_READ (amestatus));
              check (0x84030000+v,f->flags,CSR_READ (amefflags));
              check (0x84040000+v,f->xsat,CSR_READ (amexsat));
              if (!f->computed) check (0x84050000+v,32,CSR_READ (amestype));
            }
          for (unsigned i=0; i<OP_WORDS; ++i)
            {
              check (0x85000000+v*256+i,f->a[i],get (b.a,i,f->width));
              check (0x85000040+v*256+i,f->b[i],get (b.b,i,f->width));
              check (0x85000080+v*256+i,f->golden[i],get (b.c,i,4));
            }
          for (ame_size i=0; i<WORK_WORDS; ++i)
            {
              ame_size local=f->family==4 ? i%BF_WORDS : i;
              ame_size used=f->family==4 ? t*t : 3*t*t;
              if (!f->computed || local<16 || local>=16+used)
                {
                  const void *base=f->family!=4 ? b.work : i<BF_WORDS ? (void *) wba : i<2*BF_WORDS ? (void *) wbb : (void *) wbc;
                  check (0x86000000+i,OP_CANARY,get (base,local,4));
                }
            }
        }
    }
  CSR_WRITE (fcsr,oldfcsr);
  CSR_WRITE (mstatus,(CSR_READ (mstatus) & ~(usize) 0x6000u) | oldfs);
  CSR_WRITE (amestatus,0); CSR_WRITE (amefflags,0); CSR_WRITE (amexsat,0);
  return 0;
}
