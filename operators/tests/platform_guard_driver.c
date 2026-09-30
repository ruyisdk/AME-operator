#include "test.h"
#include "../common/internal.h"
#include "../include/ame_ops_i8.h"
#include "../include/ame_ops_bf16.h"
#include "op_fixtures.h"
static ame_i32 a32[OP_WORDS],b32[OP_WORDS],c32[OP_WORDS],c8[OP_WORDS],cu8[OP_WORDS];
static ame_i8 a8[OP_WORDS],b8[OP_WORDS];
static ame_u8 au8[OP_WORDS],bu8[OP_WORDS];
static float af[OP_WORDS],bf[OP_WORDS],cf[OP_WORDS],cbf[OP_WORDS];
static ame_bf16 abf[OP_WORDS],bbf[OP_WORDS];
static ame_i32 si[64] __attribute__((aligned(16))),s8[64] __attribute__((aligned(16)));
static float sf[64] __attribute__((aligned(16))),sbc[32] __attribute__((aligned(16)));
static ame_bf16 sba[64] __attribute__((aligned(16))),sbb[64] __attribute__((aligned(16)));
static ame_i32 wi[3*MAX_N*MAX_N+16] __attribute__((aligned(16)));
static ame_i32 w8[3*MAX_N*MAX_N+16] __attribute__((aligned(16)));
static float wf[3*MAX_N*MAX_N+16] __attribute__((aligned(16)));
static ame_bf16 wba[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static ame_bf16 wbb[2*MAX_N*MAX_N+32] __attribute__((aligned(16)));
static float wbc[MAX_N*MAX_N+16] __attribute__((aligned(16)));
struct buffers { void *a,*b,*c,*small[3],*large[3]; };

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((noinline,noclone)) void guard_observation_begin (void)
{ __asm__ volatile ("nop" : : : "memory"); }
__attribute__((noinline,noclone)) void guard_observation_end (void)
{ __asm__ volatile ("nop\n\tnop" : : : "memory"); }
#ifdef __cplusplus
}
#endif

static struct buffers select_buffers (unsigned f)
{
  switch (f)
    {
    case 0: return (struct buffers) {a32,b32,c32,{si,0,0},{wi,0,0}};
    case 1: return (struct buffers) {a8,b8,c8,{s8,0,0},{w8,0,0}};
    case 2: return (struct buffers) {au8,bu8,cu8,{s8,0,0},{w8,0,0}};
    case 3: return (struct buffers) {af,bf,cf,{sf,0,0},{wf,0,0}};
    default: return (struct buffers) {abf,bbf,cbf,{sba,sbb,sbc},{wba,wbb,wbc}};
    }
}
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
static int wanted_status (const struct guard_case *f,int supported)
{
  unsigned a=f->action;
  if (a==OP_BAD_MODE || a==OP_BAD_A) return AME_OPS_BAD_ARGUMENT;
  if (a==OP_K0_SET || a==OP_K0_ADD || a==OP_EMPTY) return AME_OPS_OK;
  if (f->phase==1) return AME_OPS_NOT_OWNED;
  if (!supported) return AME_OPS_UNSUPPORTED;
  if (a==OP_NO_WORK) return AME_OPS_NO_WORKSPACE;
  if (a==OP_BAD_WORK) return f->family==4 ? AME_OPS_BAD_ARGUMENT : AME_OPS_NO_WORKSPACE;
  return AME_OPS_OK;
}
static int invoke (const struct guard_case *f,struct buffers b,ame_size t,int supported)
{
  ame_size m=1,n=1,k=1,ldc=1,slots=supported ? t*t : 16;
  int add=f->action==OP_ADD || f->action==OP_K0_ADD;
  void *a=(unsigned char *) b.a+OP_GUARD*f->width;
  void *rhs=(unsigned char *) b.b+OP_GUARD*f->width;
  void *c=(unsigned char *) b.c+OP_GUARD*4;
  void **bank=supported ? b.large : b.small;
  void *work=(unsigned char *) bank[0]+32;
  struct ame_ops_bf16_workspace w={
    (ame_bf16 *) ((unsigned char *) bank[0]+32),2*slots,
    (ame_bf16 *) (bank[1] ? (unsigned char *) bank[1]+32 : 0),2*slots,
    (float *) (bank[2] ? (unsigned char *) bank[2]+32 : 0),slots};
  struct ame_ops_bf16_workspace before=w;
  const struct ame_ops_bf16_workspace *desc=&w;
  if (f->action==OP_K0_SET || f->action==OP_K0_ADD)
    { m=OP_GEMV ? 8 : 4; n=OP_GEMV ? 1 : 2; ldc=OP_GEMV ? 1 : 3; k=0; a=rhs=work=0; desc=0; }
  if (f->action==OP_EMPTY) { m=0; a=rhs=c=work=0; desc=0; }
  if (f->action==OP_BAD_MODE) add=2;
  if (f->action==OP_BAD_A) a=0;
  if (f->action==OP_NO_WORK) { work=0; desc=0; }
  if (f->action==OP_BAD_WORK)
    { work=(void *) (ame_uintptr) 1; desc=(const struct ame_ops_bf16_workspace *) (ame_uintptr) 1; }
  int status;
#if OP_GEMV
  (void) n; (void) ldc;
#define INVOKE(NAME,TYPE,OUT) NAME (m,k,(const TYPE *) a,1,(const TYPE *) rhs,(OUT *) c,add,work,12*slots)
  switch (f->family)
    {
    case 0: status=INVOKE (ame_gemv_i32_owned,ame_i32,ame_i32); break;
    case 1: status=INVOKE (ame_gemv_i8_i32_owned,ame_i8,ame_i32); break;
    case 2: status=INVOKE (ame_gemv_u8_i32_owned,ame_u8,ame_i32); break;
    case 3: status=INVOKE (ame_gemv_f32_owned,float,float); break;
    default: status=ame_gemv_bf16_f32_owned (m,k,(const ame_bf16 *) a,1,(const ame_bf16 *) rhs,(float *) c,add,desc); break;
    }
#else
#define INVOKE(NAME,TYPE,OUT) NAME (m,n,k,(const TYPE *) a,1,(const TYPE *) rhs,n,(OUT *) c,ldc,add,work,12*slots)
  switch (f->family)
    {
    case 0: status=INVOKE (ame_gemm_i32_owned,ame_i32,ame_i32); break;
    case 1: status=INVOKE (ame_gemm_i8_i32_owned,ame_i8,ame_i32); break;
    case 2: status=INVOKE (ame_gemm_u8_i32_owned,ame_u8,ame_i32); break;
    case 3: status=INVOKE (ame_gemm_f32_owned,float,float); break;
    default: status=ame_gemm_bf16_f32_owned (m,n,k,(const ame_bf16 *) a,1,(const ame_bf16 *) rhs,n,(float *) c,ldc,add,desc); break;
    }
#endif
#undef INVOKE
  check (0x91000000,1,w.a==before.a); check (0x91000001,1,w.a_elements==before.a_elements);
  check (0x91000002,1,w.b==before.b); check (0x91000003,1,w.b_elements==before.b_elements);
  check (0x91000004,1,w.c==before.c); check (0x91000005,1,w.c_elements==before.c_elements);
  return status;
}
int run_kernel (const u32 *unused_a,const u32 *unused_b,u32 *unused_c,u32 *unused_d)
{
  (void) unused_a; (void) unused_b; (void) unused_c; (void) unused_d;
  ame_size t=CSR_READ (amenlen),uds=CSR_READ (ameudsz);
  int supported=uds==32 && t>=4 && t<=128 && !(t & (t-1));
  ame_size safe_t=supported ? t : 4;
  check (0x90000000,1,CSR_READ (ameown));
  usize oldfs=CSR_READ (mstatus) & 0x6000u;
  CSR_WRITE (mstatus,CSR_READ (mstatus) | 0x2000u);
  usize oldfcsr=CSR_READ (fcsr);
  guard_observation_begin ();
  for (unsigned phase=0; phase<3; ++phase)
    {
      if (phase==1) { release (); check (0x90000001,0,CSR_READ (ameown)); }
      if (phase==2) { check (0x90000002,1,acquire (0)&1); check (0x90000003,1,CSR_READ (ameown)); }
      ame_size tile=12345;
      int expected=phase==1 ? AME_OPS_NOT_OWNED : supported ? AME_OPS_OK : AME_OPS_UNSUPPORTED;
      check (0x90000010+phase,expected,ame_ops_platform (&tile));
      check (0x90000020+phase,expected==AME_OPS_OK ? t : 12345,tile);
      for (unsigned v=0; v<OP_CASES; ++v)
        {
          const struct guard_case *f=&op_cases[v];
          if (f->phase!=phase) continue;
          struct buffers b=select_buffers (f->family);
          unsigned segments=f->family==4 ? 3 : 1,small_words=f->family==4 ? 32 : 64;
          ame_size used=(f->family==4 ? 1 : 3)*safe_t*safe_t;
          for (unsigned i=0; i<OP_WORDS; ++i)
            { put (b.a,i,f->width,f->a[i]); put (b.b,i,f->width,f->b[i]); put (b.c,i,4,f->initial[i]); }
          for (unsigned s=0; s<segments; ++s)
            {
              for (unsigned i=0; i<small_words; ++i) put (b.small[s],i,4,OP_CANARY);
              for (unsigned i=0; i<8; ++i)
                { put (b.large[s],i,4,OP_CANARY); put (b.large[s],8+used+i,4,OP_CANARY); }
            }
          CSR_WRITE (fcsr,0x6a);
          if (phase!=1)
            { CSR_WRITE (amestatus,0); CSR_WRITE (amestype,32); CSR_WRITE (amefflags,1); CSR_WRITE (amexsat,1); }
          int want=wanted_status (f,supported);
          int computed=want==AME_OPS_OK && (f->action==OP_SET || f->action==OP_ADD);
          check (0x92000000+v,want,invoke (f,b,t,supported));
          check (0x93000000+v,phase!=1,CSR_READ (ameown));
          check (0x93010000+v,0x6a,CSR_READ (fcsr));
          if (phase!=1)
            {
              check (0x93020000+v,0,CSR_READ (amestatus));
              check (0x93030000+v,1,CSR_READ (amefflags)); check (0x93040000+v,1,CSR_READ (amexsat));
              if (!computed) check (0x93050000+v,32,CSR_READ (amestype));
            }
          for (unsigned i=0; i<OP_WORDS; ++i)
            {
              check (0x94000000+v*128+i,f->a[i],get (b.a,i,f->width));
              check (0x94000020+v*128+i,f->b[i],get (b.b,i,f->width));
              check (0x94000040+v*128+i,want==AME_OPS_OK ? f->success[i] : f->initial[i],get (b.c,i,4));
            }
          for (unsigned s=0; s<segments; ++s)
            {
              for (unsigned i=0; i<small_words; ++i) check (0x95000000+s*256+i,OP_CANARY,get (b.small[s],i,4));
              for (unsigned i=0; i<8; ++i)
                { check (0x95001000+s*256+i,OP_CANARY,get (b.large[s],i,4));
                  check (0x95001080+s*256+i,OP_CANARY,get (b.large[s],8+used+i,4)); }
            }
        }
    }
  guard_observation_end ();
  CSR_WRITE (fcsr,oldfcsr); CSR_WRITE (mstatus,(CSR_READ (mstatus) & ~(usize) 0x6000u) | oldfs);
  CSR_WRITE (amestatus,0); CSR_WRITE (amefflags,0); CSR_WRITE (amexsat,0);
  return 0;
}
