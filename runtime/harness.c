#include "test.h"
#include "u32.h"

#define GUARD 16u
#define WORDS (MAX_N * MAX_N)
#define TOTAL (WORDS + 2u * GUARD)
#define CANARY 0xa5c369f0u

volatile u32 ame_mailbox[MB_WORDS] __attribute__((aligned(16)));
static u32 source[TOTAL] __attribute__((aligned(16)));
static u32 rhs[TOTAL] __attribute__((aligned(16)));
static u32 output[TOTAL] __attribute__((aligned(16)));
static u32 auxiliary[TOTAL] __attribute__((aligned(16)));

static void fail (u32 code)
{
  ame_mailbox[MB_ERROR] = code;
  ame_mailbox[MB_STATUS] = FAILED;
  ame_test_done ();
}

void check (u32 index, u32 expected, u32 actual)
{
  ++ame_mailbox[MB_CHECKS];
  if (expected != actual)
    {
      ame_mailbox[MB_INDEX] = index;
      ame_mailbox[MB_EXPECTED] = expected;
      ame_mailbox[MB_ACTUAL] = actual;
      fail (ERR_CHECK);
    }
}

void test_main (void)
{
  ame_mailbox[MB_MAGIC] = 0x414d4554;
  ame_mailbox[MB_VERSION] = 1;
  ame_mailbox[MB_STATUS] = RUNNING;
  ame_mailbox[MB_CASE] = CASE_ID;
  ame_mailbox[MB_XLEN] = __riscv_xlen;
  ame_mailbox[MB_MREGS] = MREGS;
  ame_mailbox[MB_ACCREGS] = ACCREGS;
  ame_mailbox[MB_SEED] = TEST_SEED;
  usize n = CSR_READ(amenlen);
  usize uds = CSR_READ(ameudsz);
  ame_mailbox[MB_N] = (u32) n;
  ame_mailbox[MB_UDS] = (u32) uds;
  if (n < 4 || n > MAX_N || (n & (n - 1)) || uds != UDS)
    fail (ERR_PLATFORM);
  check (0, 0, (u32) CSR_READ(ameown));
  usize status = acquire (0);
  ame_mailbox[MB_ACQUIRE] = (u32) status;
  if (!(status & 1))
    fail (ERR_ACQUIRE);
  check (1, 1, (u32) CSR_READ(ameown));
  CSR_WRITE(amestatus, 0);
  CSR_WRITE(amefflags, 0);
  CSR_WRITE(amexsat, 0);

  for (u32 i = 0; i < TOTAL; ++i)
    {
      source[i] = input_word (i - GUARD, TEST_SEED);
      rhs[i] = input_word (i - GUARD, TEST_SEED ^ 0x6d2b79f5u);
      output[i] = auxiliary[i] = CANARY;
    }
  int rc = run_kernel (source + GUARD, rhs + GUARD,
                       output + GUARD, auxiliary + GUARD);
  if (rc)
    fail (ERR_KERNEL);
  check (2, 1, (u32) CSR_READ(ameown));
  if (CSR_READ(amestatus) != 0)
    fail (ERR_UN);
  check (3, 0, (u32) CSR_READ(amefflags));
  check (4, 0, (u32) CSR_READ(amexsat));
  u32 count = DATA_CASE ? (u32) (n * n) : 0;
  if (RAW_IMAGE)
    count = count * UDS / 32;
  for (u32 i = 0; i < TOTAL; ++i)
    {
      u32 expected = CANARY;
      u32 aux = CANARY;
      if (i >= GUARD && i - GUARD < count)
        {
          expected = expected_word (i - GUARD, (u32) n, TEST_SEED, EXPECT_KIND);
          if (AUX_KIND == 1)
            aux = input_word (i - GUARD, TEST_SEED);
          else if (AUX_KIND == 2)
            aux = 0;
        }
      check (0x10000000u + i, expected, output[i]);
      check (0x20000000u + i, aux, auxiliary[i]);
      check (0x30000000u + i, input_word (i - GUARD, TEST_SEED), source[i]);
      check (0x40000000u + i,
             input_word (i - GUARD, TEST_SEED ^ 0x6d2b79f5u), rhs[i]);
    }
  release ();
  check (5, 0, (u32) CSR_READ(ameown));
  ame_mailbox[MB_STATUS] = PASSED;
  ame_test_done ();
}
