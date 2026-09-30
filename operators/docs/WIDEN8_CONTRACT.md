# INT8 / UINT8 to INT32 Contract

Both GEMM and GEMV accept either two signed-byte inputs or two unsigned-byte
inputs and produce INT32 output. GEMM is row-major NN, alpha=1, beta=0/1;
GEMV uses unit stride and a single output column.

- Signed range is [-128,127], unsigned range [0,255].
- Per output, `abs(seed)+sum(abs(A*B)) <= INT32_MAX` is required.
- Input strides count bytes; output strides count INT32 elements. Inputs may
  be byte aligned, output requires 4-byte alignment.
- The compile profile is UDS32/M16/Acc4; runtime T is 4,8,16,32,64,128.
- Workspace is 16-byte aligned and uses 12*T*T bytes: two operands of 4*T*T
  bytes and one output tile of T*T INT32 elements.
- Inputs may share storage, but output and workspace are disjoint from inputs
  and each other. Bounds checks include row padding.
- Ownership, errors and zero-size behavior follow [API](../../docs/API.md).

A 1x4 byte-M operand and a 4x1 byte-M operand feed one INT32 accumulator.
K advances by 4*T. Memory packing is four square-major T-by-T images, with
zero-padded tails and active-cell unpacking.

The independent logical integer reference uses original matrices rather than
packed images. Tests check signed/unsigned interpretation, packing boundaries,
multiple blocks, output bits, source/padding guards and workspace bounds.
