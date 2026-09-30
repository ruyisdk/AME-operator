"""Exact BF16 widening followed by the named logical FP32 RNE FMA fold."""
import fp32_fold

def widen(bits):
    if type(bits) is not int or not 0 <= bits <= 0xffff:
        raise ValueError('invalid BF16 bit image')
    # This is an encoding embedding, not host floating conversion. It retains sNaN.
    return bits << 16

def dot(a, b, seed):
    return fp32_fold.dot([widen(v) for v in a], [widen(v) for v in b], seed)

def gemm(m,n,k,a,lda,b,ldb,c,ldc,accumulate):
    return fp32_fold.gemm(m,n,k,[widen(v) for v in a],lda,
                         [widen(v) for v in b],ldb,c,ldc,accumulate)
