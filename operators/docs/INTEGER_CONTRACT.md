# Integer Operator Contract

GEMM computes C=A*B or C=C_old+A*B, row-major NN, alpha=1, beta=0/1.
GEMV uses unit vector stride and the single-column GEMM kernel. Inputs,
accumulator and output are INT32, with integer RM=RNU.

For each output, `abs(seed)+sum(abs(A*B)) <= INT32_MAX`. The caller supplies
data within this sufficient no-overflow bound; GOLDEN enforces it using
Python integers. The bound makes the result independent of term order.

UDS32/M16/Acc4 is the compile profile. Tile side T is read at runtime and is
a power of two in [4,128]. The same ELF is used across tile sizes.

## API and Memory

The caller holds ownership during nonempty computation and permits M/ACC,
Md/Ad and datatype setup to be clobbered. Public calls use scalar/pointer ABI.
See [API](../../docs/API.md) for status, error precedence and zero dimensions.

Inputs/output need 4-byte alignment. Workspace is 16-byte aligned with
12*T*T bytes. A/B may overlap each other; output cannot overlap either.
Workspace is disjoint from all matrices. Span checks include row padding and
reject offset/byte/address overflow. Only active output cells change.

## Implementation and Verification

Pack bounded A/B blocks, zero-pad tails, keep one typed accumulator across K
blocks, transfer to M, then unpack active cells. Workspace outside the required
extent and all input/output padding remain unchanged.

`operators/reference/integer.py` computes logical matrix products; GEMV uses
its own scalar loop. Fixtures preserve original inputs, full expected output,
guards and generator identity. Geometry, signed values, multiple K blocks,
set/add, errors and state have separate registered variants. Full runs check
dual XLEN, C/C++, O0/O2 and all six T values.
