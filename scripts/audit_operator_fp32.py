"""Independently enumerate FP32 operator coverage and logical dot products."""
import sys
import ame
sys.path.insert(0,str(ame.ROOT/'operators/reference'))
import floating_exact as fp

def accounting(build,rows,t):
    op=build['row']['operator'];ame.require(op in ('gemm','gemv'),'unknown FP32 operator')
    selected=[r for r in rows if r['tile']==t]
    wanted=set()
    for axis in ('m','n','k') if op=='gemm' else ('m','k'):
        for label,value in (('minus',t-1),('exact',t),('plus',t+1),('double-plus',2*t+1)):
            shape=dict(m=3,n=5 if op=='gemm' else 1,k=5);shape[axis]=value
            for add in (0,1):wanted.add((axis+'-'+label,add,*[shape[x] for x in ('m','n','k')]))
    shapes=[('cross',t+1,t-1 if op=='gemm' else 1,t+1),
            ('long-k',5,3 if op=='gemm' else 1,3*t+1),
            ('fraction-hand',2,2 if op=='gemm' else 1,2),('exact-cancel',1,1,3),
            ('limit-positive',1,1,1),('limit-negative',1,1,1),
            ('zero-products',2,3 if op=='gemm' else 1,3),
            ('zero-k',2,3 if op=='gemm' else 1,0),('zero-m',0,3 if op=='gemm' else 1,3)]
    if op=='gemm':shapes.append(('zero-n',3,0,3))
    wanted.update((name,add,m,n,k) for name,m,n,k in shapes for add in (0,1))
    actual={(r['purpose'],r['accumulate'],r['m'],r['n'],r['k']) for r in selected}
    ame.require(actual==wanted and len(selected)==len(wanted),'FP32 coverage gap/duplicate')
    for r in selected:
        m,n,k=(r[x] for x in ('m','n','k'));lda,ldb,ldc=(r[x] for x in ('lda','ldb','ldc'))
        ame.require((lda,ldb,ldc)==(k+2,n+3 if op=='gemm' else 1,n+4 if op=='gemm' else 1),
                    'FP32 stride changed')
        a,b,c=(r[x+'_units'] for x in ('a','b','initial'))
        ame.require((len(a),len(b),len(c))==(m*lda,k*ldb,m*ldc),'FP32 storage extent')
        purpose=r['purpose']
        if purpose=='fraction-hand':
            ame.require((a[:2],a[lda:lda+2],b[0],b[ldb])==([8,-4],[2,6],4,-8),'hand inputs')
            if op=='gemm':ame.require((b[1],b[ldb+1])==(8,2),'hand second column')
        if purpose=='exact-cancel':
            ame.require(a[:3]==[4096,1,-4096] and [b[q*ldb] for q in range(3)]==[1024,1,1024],
                        'cancellation witness')
        if purpose.startswith('limit-'):
            ame.require((a[0],b[0])==(-4095 if purpose=='limit-negative' else 4095,4095),
                        'precision boundary witness')
        if purpose=='zero-products':
            ame.require(all(a[i*lda+q]==0 for i in range(m) for q in range(k)),'zero witness')
        want=list(c)
        for i in range(m):
            for j in range(n):
                seed=c[i*ldc+j] if r['accumulate'] else 0
                terms=[a[i*lda+q]*b[q*ldb+j] for q in range(k)]
                ame.require(abs(seed)+sum(map(abs,terms)) <= (1<<24)-1,'not order-independent exact')
                want[i*ldc+j]=seed+sum(terms)
        ame.require(want==r['golden_units'],'wrong logical FP32 GOLDEN')
        for key,values,grid in (('a',a,3),('b',b,3),('initial',c,6),('golden',want,6)):
            ame.require(r[key]==[fp.bits(v,grid) for v in values],'wrong FP32 encoded image')
    widths=[max(len(r[k]) for r in rows) for k in ('a','b','initial')]
    maximum=build['config']['max_n']
    count=6+4*(maximum**2+32)+2+len(selected)*(4+sum(widths)+96+3*maximum**2+32-3*t*t)
    return selected,count
