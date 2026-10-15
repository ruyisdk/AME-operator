/* Freestanding tutorial driver. The host independently checks the result. */
#include "fixture.h"

typedef __UINT32_TYPE__ u32;
typedef __SIZE_TYPE__ usize;
_Static_assert(sizeof(int) == 4, "INT32 example needs a 32-bit int");
volatile u32 tutorial_mailbox[32] __attribute__((aligned(16)));
const int tutorial_a[16] __attribute__((aligned(16))) = INPUT_A;
const int tutorial_b[16] __attribute__((aligned(16))) = INPUT_B;
/* The host checks all eight canaries around the result, too. */
u32 tutorial_output[24] __attribute__((aligned(16)));

extern void tutorial_gemm(const int *, const int *, int *);
extern void tutorial_done(void) __attribute__((noreturn));
#define READ_CSR(name) ({ usize v; \
  __asm__ volatile("csrr %0," #name : "=r"(v) : : "memory"); v; })
#define WRITE_CSR(name, value) \
  __asm__ volatile("csrw " #name ",%0" : : "r"((usize)(value)) : "memory")

static void fail(u32 error)
{
  tutorial_mailbox[3] = error;
  tutorial_mailbox[2] = 3;
  tutorial_done();
}

void tutorial_main(void)
{
  tutorial_mailbox[0] = 0x414d4534; /* "AME4" */
  tutorial_mailbox[1] = 1;          /* mailbox ABI */
  tutorial_mailbox[2] = 1;          /* running, 2=completed, 3=failed */
  tutorial_mailbox[4] = (u32)READ_CSR(amenlen);
  tutorial_mailbox[5] = (u32)READ_CSR(ameudsz);
  tutorial_mailbox[7] = (u32)READ_CSR(ameown);
  if (tutorial_mailbox[4] != 4 || tutorial_mailbox[5] != 32
      || tutorial_mailbox[7] != 0)
    fail(1);
  usize acquired;
  __asm__ volatile("ame.acquire %0,%1" : "=r"(acquired)
                   : "r"((usize)0) : "memory");
  tutorial_mailbox[6] = (u32)acquired;
  if (!(acquired & 1) || READ_CSR(ameown) != 1)
    fail(2);
  WRITE_CSR(amestatus, 0);
  WRITE_CSR(amefflags, 0);
  WRITE_CSR(amexsat, 0);
  for (unsigned i = 0; i < 24; ++i)
    tutorial_output[i] = 0xa5c369f0u;

  tutorial_gemm(tutorial_a, tutorial_b, (int *)(tutorial_output + 4));
#ifdef TUTORIAL_INJECT_FAULT
  tutorial_output[4] ^= 1u; /* Deliberate fault: independent host must reject. */
#endif

  tutorial_mailbox[9] = (u32)READ_CSR(amestatus);
  tutorial_mailbox[10] = (u32)READ_CSR(amefflags);
  tutorial_mailbox[11] = (u32)READ_CSR(amexsat);
  __asm__ volatile("ame.release" : : : "memory");
  tutorial_mailbox[8] = (u32)READ_CSR(ameown);
  if (tutorial_mailbox[8] || tutorial_mailbox[9]
      || tutorial_mailbox[10] || tutorial_mailbox[11])
    fail(3);
  tutorial_mailbox[2] = 2; /* Completion alone is not numerical PASS. */
  tutorial_done();
}
