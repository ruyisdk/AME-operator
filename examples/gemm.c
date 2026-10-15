#include "gemm_kernel.h"

/* Ordinary pointers cross the function boundary. AME tile types stay local. */
void tutorial_gemm(const int *a, const int *b, int *c)
{
  gemm_tile(a, b, c);
}
