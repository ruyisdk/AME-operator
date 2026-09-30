"""Logical FP32 RNE FMA fold; no target packing or simulator implementation."""
from functools import lru_cache
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from ieee_tininess import fp32_fused_expected

@lru_cache(maxsize=8192)
def fma(a,b,c):
    if any(type(v) is not int or not 0<=v<=0xffffffff for v in (a,b,c)):
        raise ValueError('invalid FP32 bit image')
    return fp32_fused_expected(1,0,a,b,c)

def dot(a,b,seed):
    if len(a)!=len(b):raise ValueError('dot extent mismatch')
    if type(seed) is not int or not 0<=seed<=0xffffffff:raise ValueError('invalid seed')
    flags=0
    for x,y in zip(a,b):
        seed,added=fma(x,y,seed);flags|=added
    return seed,flags

def gemm(m,n,k,a,lda,b,ldb,c,ldc,accumulate):
    if (any(type(v) is not int or v<0 for v in (m,n,k,lda,ldb,ldc))
            or lda<k or ldb<n or ldc<n or accumulate not in (0,1)):
        raise ValueError('invalid logical matrix parameters')
    for data,rows,cols,stride in ((a,m,k,lda),(b,k,n,ldb),(c,m,n,ldc)):
        if len(data)<((rows-1)*stride+cols if rows and cols else 0):
            raise ValueError('short logical storage')
        if any(type(v) is not int or not 0<=v<=0xffffffff for v in data):
            raise ValueError('invalid FP32 bit image')
    result=list(c);flags=0
    for i in range(m):
        for j in range(n):
            result[i*ldc+j],added=dot(a[i*lda:i*lda+k],
                [b[q*ldb+j] for q in range(k)],c[i*ldc+j] if accumulate else 0)
            flags|=added
    return result,flags
