#ifndef AME_TEST_H
#define AME_TEST_H

typedef __UINT32_TYPE__ u32;
typedef __SIZE_TYPE__ usize;

#ifdef __cplusplus
extern "C" {
#endif
/* ABI v1: 32 little-endian u32 words, independent of XLEN. */
enum {
  MB_MAGIC, MB_VERSION, MB_STATUS, MB_CASE, MB_XLEN, MB_N, MB_UDS,
  MB_MREGS, MB_ACCREGS, MB_SEED, MB_CHECKS, MB_ERROR, MB_INDEX,
  MB_EXPECTED, MB_ACTUAL, MB_ACQUIRE, MB_CAUSE_LO, MB_PC_LO, MB_TVAL_LO,
  MB_CAUSE_HI, MB_PC_HI, MB_TVAL_HI, MB_WORDS = 32
};
enum { RUNNING = 1, PASSED = 2, FAILED = 3, SKIPPED = 4 };
enum {
  ERR_CHECK = 1, ERR_PLATFORM = 2, ERR_ACQUIRE = 3, ERR_UN = 4,
  ERR_KERNEL = 5, ERR_TRAP = 6
};
extern volatile u32 ame_mailbox[MB_WORDS];
void ame_test_done (void) __attribute__((noreturn));
void check (u32, u32, u32);
int run_kernel (const u32 *, const u32 *, u32 *, u32 *);
void clobber_bank (void);

#ifdef __cplusplus
}
#endif

#define CSR_READ(NAME) ({ usize value; \
  __asm__ volatile ("csrr %0," #NAME : "=r" (value) : : "memory"); value; })
#define CSR_WRITE(NAME, VALUE) \
  __asm__ volatile ("csrw " #NAME ",%0" : : "r" ((usize) (VALUE)) : "memory")

static inline usize acquire (usize descriptor)
{
  usize status;
  __asm__ volatile ("ame.acquire %0,%1" : "=r" (status)
                    : "r" (descriptor) : "memory");
  return status;
}

static inline void release (void)
{
  __asm__ volatile ("ame.release" : : : "memory");
}
#endif
