#include "internal.h"
int ame_ops_platform (ame_size *tile)
{
  ame_size own, uds, n;
  __asm__ volatile ("csrr %0,ameown" : "=r" (own) : : "memory");
  if (!(own & 1)) return AME_OPS_NOT_OWNED;
  __asm__ volatile ("csrr %0,ameudsz" : "=r" (uds) : : "memory");
  __asm__ volatile ("csrr %0,amenlen" : "=r" (n) : : "memory");
  if (uds != 32 || n < 4 || n > 128 || (n & (n-1)))
    return AME_OPS_UNSUPPORTED;
  *tile = n;
  return AME_OPS_OK;
}
