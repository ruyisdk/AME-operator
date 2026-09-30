#ifndef AME_REFERENCE_U32_H
#define AME_REFERENCE_U32_H

/* Unsigned arithmetic is modulo 2^32.  No signed overflow or AME API. */
static inline u32 input_word (u32 index, u32 seed)
{
#if EDGE_DATA
  static const u32 edges[8] = {
    0, 1, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0xaaaaaaaau, 0x55555555u, 0xffffu
  };
  if ((index & 15) < 8)
    return edges[(index + seed) & 7];
#endif
  u32 x = index * 1664525u + seed;
  return x ^ (x >> 13) ^ (index << 24);
}

static inline u32 expected_word (u32 index, u32 n, u32 seed, unsigned kind)
{
  if (kind == 1)
    index = (index % n) * n + index / n;
  u32 a = input_word (index, seed);
  u32 b = input_word (index, seed ^ 0x6d2b79f5u);
  switch (kind)
    {
    case 2: return a + b;
    case 3: return b - a; /* ISA order: src2 minus src1. */
    case 4: return a * b;
    case 5: return a < b ? a : b;
    case 6: return a > b ? a : b;
    case 7: return a & b;
    case 8: return a & ~b;
    case 9: return a | b;
    case 10: return a | ~b;
    case 11: return a ^ b;
    case 12: return index / n;
    case 13: return index % n;
    default: return a;
    }
}
#endif
