"""Independent logical-matrix GOLDEN. No AME packing or model imports."""
I32_MIN, I32_MAX = -(1 << 31), (1 << 31)-1

def _check(rows, cols, data, stride):
    if rows < 0 or cols < 0 or stride < cols:
        raise ValueError("invalid logical shape/stride")
    need = 0 if not rows or not cols else (rows-1)*stride+cols
    if len(data) < need or any(type(v) is not int or not I32_MIN <= v <= I32_MAX for v in data):
        raise ValueError("short/non-i32 input")

def _bound(seed, terms):
    # A sufficient order-independent bound, stronger than checking only final C.
    if abs(seed)+sum(abs(a*b) for a,b in terms) > I32_MAX:
        raise ValueError("outside exact no-overflow operator contract")
    return seed+sum(a*b for a,b in terms)

def gemm(m,n,k,a,lda,b,ldb,c,ldc,accumulate):
    if accumulate not in (0,1):
        raise ValueError("invalid accumulation mode")
    _check(m,k,a,lda); _check(k,n,b,ldb); _check(m,n,c,ldc)
    result = list(c)
    for i in range(m):
        for j in range(n):
            terms = [(a[i*lda+p],b[p*ldb+j]) for p in range(k)]
            result[i*ldc+j] = _bound(c[i*ldc+j] if accumulate else 0, terms)
    return result

def gemv(m,k,a,lda,x,y,accumulate):
    if accumulate not in (0,1):
        raise ValueError("invalid accumulation mode")
    _check(m,k,a,lda); _check(k,1,x,1); _check(m,1,y,1)
    result = list(y)
    for i in range(m):
        terms = [(a[i*lda+p],x[p]) for p in range(k)]
        result[i] = _bound(y[i] if accumulate else 0, terms)
    return result
