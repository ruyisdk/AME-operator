"""Bounded signed i32 arithmetic edges, independent of AME/model internals."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import integer
from geometry_fixtures import emit, TILES

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='i32-numeric-v4':
        raise ValueError('unsupported numeric contract')
    out=[]
    for t in TILES:
        def case(name,m,n,k,av,bv,cv,add):
            lda,ldb,ldc=k+2,(n+3 if op=='gemm' else 1),(n+4 if op=='gemm' else 1)
            def data(rows,cols,stride,fn):
                return [fn(i,j) if j<cols else 0x123456 for i in range(rows) for j in range(stride)]
            a=data(m,k,lda,av);b=data(k,n,ldb,bv);c=data(m,n,ldc,cv)
            g=integer.gemm(m,n,k,a,lda,b,ldb,c,ldc,add) if op=='gemm' else integer.gemv(m,k,a,lda,b,c,add)
            out.append(dict(name=f't{t}-{name}',purpose=name,tile=t,m=m,n=n,k=k,
                lda=lda,ldb=ldb,ldc=ldc,accumulate=add,a=a,b=b,initial=c,golden=g))
        def scalar(name,a,b,c=0,add=0):
            case(name,1,1,1,lambda i,j:a,lambda i,j:b,lambda i,j:c,add)
        scalar('positive-limit',integer.I32_MAX,1)
        scalar('negative-limit',-integer.I32_MAX,1)
        scalar('high-signflip',1<<30,-1)
        scalar('add-positive-limit',2,3,integer.I32_MAX-6,1)
        scalar('add-negative-limit',-2,3,-integer.I32_MAX+6,1)
        scalar('product-near-limit',46340,46340)
        scalar('product-add-limit',46340,46340,integer.I32_MAX-46340**2,1)
        scalar('min-left-times-zero',integer.I32_MIN,0)
        scalar('min-right-times-zero',0,integer.I32_MIN)
        for add in (0,1):
            case('cancellation-'+str(add),1,1,2,
                 lambda i,j:(1<<30)-1,lambda i,j:1 if i==0 else -1,lambda i,j:1,add)
            k=2*t+1
            mag=integer.I32_MAX//(k+3)
            case('long-k-'+str(add),2,2 if op=='gemm' else 1,k,
                 lambda i,j:((-1)**((i+j)%3))*(mag-i*37-j%97),
                 lambda i,j:1 if (i+j)%3 else -1,lambda i,j:17-36*j,add)
            case('cross-m-'+str(add),t+1,2 if op=='gemm' else 1,3,
                 lambda i,j:(500000000-17*i-3*j)*(1 if (i+j)%2 else -1),
                 lambda i,j:1 if (i+j)%2 else -1,lambda i,j:13-i%23-j,add)
            if op=='gemm':
                case('cross-n-'+str(add),2,t+1,3,
                     lambda i,j:(500000000-17*i-3*j)*(1 if (i+j)%2 else -1),
                     lambda i,j:1 if (i+j)%2 else -1,lambda i,j:13-i-j%23,add)
    return out

def generate(manifest,out,config):
    result=emit(create(json.loads(Path(manifest).read_text())),out,config)
    result['reference']='Python exact signed integers, near-limit and high-bit fixtures'
    return result
