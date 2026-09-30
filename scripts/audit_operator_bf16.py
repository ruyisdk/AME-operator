"""BF16 coverage and scaled-integer GOLDEN audit, independent of packing."""
import sys
import ame
import spike_adapter
sys.path.insert(0,str(ame.ROOT/'operators/reference'))
import floating_exact as fp

def accounting(build,rows,t):
    op=build['row']['operator'];ame.require(op in ('gemm','gemv'),'unknown BF16 operator')
    selected=[r for r in rows if r['tile']==t]
    wanted=set()
    for axis in ('m','n') if op=='gemm' else ('m',):
        for label,value in (('minus',t-1),('exact',t),('plus',t+1),('double-plus',2*t+1)):
            shape=dict(m=3,n=5 if op=='gemm' else 1,k=5);shape[axis]=value
            for add in (0,1):wanted.add((axis+'-'+label,add,shape['m'],shape['n'],shape['k']))
    for multiple in (1,2,4):
        for label,offset in (('minus',-1),('exact',0),('plus',1)):
            for add in (0,1):wanted.add((f'k-{multiple}t-{label}',add,3,5 if op=='gemm' else 1,multiple*t+offset))
    shapes=[('cross',t+1,t-1 if op=='gemm' else 1,2*t+1),
            ('long-k',5,3 if op=='gemm' else 1,6*t+1),
            ('fraction-hand',2,2 if op=='gemm' else 1,2),('exact-cancel',1,1,3),
            ('limit-positive',1,1,1),('limit-negative',1,1,1),
            ('zero-products',2,3 if op=='gemm' else 1,3),
            ('zero-k',2,3 if op=='gemm' else 1,0),('zero-m',0,3 if op=='gemm' else 1,3)]
    if op=='gemm':shapes.append(('zero-n',3,0,3))
    wanted.update((name,add,m,n,k) for name,m,n,k in shapes for add in (0,1))
    actual={(r['purpose'],r['accumulate'],r['m'],r['n'],r['k']) for r in selected}
    ame.require(actual==wanted and len(selected)==len(wanted),'BF16 coverage gap/duplicate')
    for r in selected:
        m,n,k=(r[x] for x in ('m','n','k'));lda,ldb,ldc=(r[x] for x in ('lda','ldb','ldc'))
        ame.require((lda,ldb,ldc)==(k+2,n+3 if op=='gemm' else 1,n+4 if op=='gemm' else 1),
                    'BF16 stride changed')
        a,b,c=(r[x+'_units'] for x in ('a','b','initial'))
        ame.require((len(a),len(b),len(c))==(m*lda,k*ldb,m*ldc),'BF16 storage extent')
        purpose=r['purpose']
        if purpose=='fraction-hand':
            ame.require((a[:2],a[lda:lda+2],b[0],b[ldb])==([8,-4],[2,6],4,-8),'hand inputs')
            if op=='gemm':ame.require((b[1],b[ldb+1])==(8,2),'hand second column')
        if purpose=='exact-cancel':
            ame.require(a[:3]==[4096,1,-4096] and [b[q*ldb] for q in range(3)]==[1024,1,1024],
                        'cancellation witness')
        if purpose.startswith('limit-'):
            ame.require((a[0],b[0])==(-255 if purpose=='limit-negative' else 255,255),
                        'BF16 precision boundary witness')
        if purpose=='zero-products':
            ame.require(all(a[i*lda+q]==0 for i in range(m) for q in range(k)),'zero witness')
        want=list(c)
        for i in range(m):
            for j in range(n):
                seed=c[i*ldc+j] if r['accumulate'] else 0
                terms=[a[i*lda+q]*b[q*ldb+j] for q in range(k)]
                ame.require(abs(seed)+sum(map(abs,terms)) <= (1<<24)-1,'not order-independent exact')
                want[i*ldc+j]=seed+sum(terms)
        ame.require(want==r['golden_units'],'wrong logical BF16 GOLDEN')
        for key,values,grid,fmt in (('a',a,3,'bf16'),('b',b,3,'bf16'),
                                    ('initial',c,6,'fp32'),('golden',want,6,'fp32')):
            ame.require(r[key]==[fp.bits(v,grid,fmt) for v in values],'wrong BF16 encoded image')
    widths=[max(len(r[k]) for r in rows) for k in ('a','b','initial')]
    maximum=build['config']['max_n']
    count=6+4*(maximum**2+32)+3
    for r in selected:
        # Five state/results, six descriptor fields, three guarded matrices,
        # and the unused portion of three independently typed work arrays.
        count+=11+sum(widths)+3*32+5*maximum**2+3*32
        if r['m'] and r['n'] and r['k']:count-=5*t*t
    return selected,count

def verify_execution(build,directory,response):
    command=ame.load(directory/'spike-command.json')
    ame.require(f'--isa=rv{build["row"]["xlen"]}im_zicsr_ztt' in command['argv'],
                'unreviewed BF16 execution ISA')
    observed,stop,mailbox=spike_adapter.parse_debug(
        (directory/'spike.log').read_text(),build['symbols'],ame.CSR_MAP)
    ame.require((observed,stop,mailbox.hex())==
                (response['observed'],response['stop_pc'],response['mailbox_hex']),
                'raw BF16 observations differ')
