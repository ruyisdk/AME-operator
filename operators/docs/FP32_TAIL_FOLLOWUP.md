# Floating Tail Packing

Two arithmetic properties determine the RNE packing strategy:

1. With A=1 and B=+Inf, inactive zero A rows would create 0*Inf and spuriously
   add NV. Repeat valid boundary rows/columns and seeds so inactive output lanes
   duplicate real calculations.
2. With old C=-0, A=-0 and B=1, the RNE result is -0. Appending +0*+0 K terms
   can change it to +0. Use left +0/right -0 for inactive K terms to preserve
   finite values and both zero signs under RNE.

Empty output and K=0 are separate paths, so packing never reads absent rows
or columns. Unpacking writes only active output elements.

Literal target probes exercise infinity/NV and negative zero. The broader
FP32/BF16 numerical suites check NaNs, subnormals, overflow/underflow and flags
across all tile sizes and both beta values. Host tests compare actual packing
bits with independent logical FMA. BF16 packing has two square-major input
squares and its own boundary tests.
