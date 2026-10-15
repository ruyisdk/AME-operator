#include "gemm_kernel.h"

/* C++17 uses the same typed intrinsics. C linkage connects to our C runtime.
   No constructors, exceptions, RTTI, libc or libstdc++ are needed. */
extern "C" void tutorial_gemm(const int *a, const int *b, int *c)
{
  gemm_tile(a, b, c);
}
