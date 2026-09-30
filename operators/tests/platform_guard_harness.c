#include "test.h"
volatile u32 ame_mailbox[MB_WORDS] __attribute__((aligned(16)));
static void fail (u32 code)
{ ame_mailbox[MB_ERROR]=code; ame_mailbox[MB_STATUS]=FAILED; ame_test_done (); }
void check (u32 index,u32 expected,u32 actual)
{
  ++ame_mailbox[MB_CHECKS];
  if (expected!=actual)
    { ame_mailbox[MB_INDEX]=index; ame_mailbox[MB_EXPECTED]=expected;
      ame_mailbox[MB_ACTUAL]=actual; fail (ERR_CHECK); }
}
void test_main (void)
{
  ame_mailbox[MB_MAGIC]=0x414d4554; ame_mailbox[MB_VERSION]=1;
  ame_mailbox[MB_STATUS]=RUNNING; ame_mailbox[MB_CASE]=CASE_ID;
  ame_mailbox[MB_XLEN]=__riscv_xlen; ame_mailbox[MB_SEED]=TEST_SEED;
  ame_mailbox[MB_N]=CSR_READ (amenlen); ame_mailbox[MB_UDS]=CSR_READ (ameudsz);
  /* This lane varies UDS/T only. Register counts remain the entry contract. */
  ame_mailbox[MB_MREGS]=MREGS; ame_mailbox[MB_ACCREGS]=ACCREGS;
  check (0,0,CSR_READ (ameown));
  usize status=acquire (0); ame_mailbox[MB_ACQUIRE]=status;
  if (!(status & 1)) fail (ERR_ACQUIRE);
  check (1,1,CSR_READ (ameown));
  CSR_WRITE (amestatus,0); CSR_WRITE (amefflags,0); CSR_WRITE (amexsat,0);
  if (run_kernel (0,0,0,0)) fail (ERR_KERNEL);
  check (2,1,CSR_READ (ameown));
  if (CSR_READ (amestatus)) fail (ERR_UN);
  check (3,0,CSR_READ (amefflags)); check (4,0,CSR_READ (amexsat));
  release (); check (5,0,CSR_READ (ameown));
  ame_mailbox[MB_STATUS]=PASSED; ame_test_done ();
}
