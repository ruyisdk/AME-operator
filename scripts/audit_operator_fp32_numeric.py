"""Independently enumerate named-RNE operator witnesses and exact check counts."""
import json
import sys
import ame
sys.path.insert(0,str(ame.ROOT/'operators/reference'))
from fp32_fold import fma

def accounting(build,rows,t):
    op=build['row']['operator'];ame.require(op in ('gemm','gemv'),'unknown operator')
    catalog=json.loads((ame.ROOT/'operators/cases/fp32-rne-witnesses.json').read_text())['witnesses']
    literals={r['name']:r for r in catalog}
    ame.require(len(literals)==31,'literal witness inventory')
    specs={name:(r,0) for name,r in literals.items()}
    for blocks in (1,2):
        for name in ('fused-cancel','seed-before-fold','order-sensitive','intermediate-flags','negative-zero-seed'):
            specs[f'cross-{blocks}-{name}']=(literals[name],blocks*t-1)
    wanted={(name,r['add']) for name,(r,_) in specs.items()}
    wanted|={(name,add) for name in (('mixed-rows','mixed-columns') if op=='gemm' else ('mixed-rows',)) for add in (0,1)}
    selected=[r for r in rows if r['tile']==t]
    ame.require({(r['purpose'],r['accumulate']) for r in selected}==wanted and len(selected)==len(wanted),
                'FP32 numeric coverage gap/duplicate')
    for ordinal,r in enumerate(rows):
        if r['tile']!=t:continue
        ame.require((r['seed_flags'],r['seed_xsat'],r['fcsr'])==
                    (8 if ordinal%2 else 0,ordinal%2,0x6a if ordinal%2 else 0x85),'state seed changed')
        name=r['purpose'];m,n,k,lda,ldb,ldc=(r[v] for v in ('m','n','k','lda','ldb','ldc'))
        if name in specs:
            literal,offset=specs[name]
            ak=[int(v,16) for v in literal['a']];bk=[int(v,16) for v in literal['b']]
            ame.require(len(ak)==len(bk),'literal lengths')
            ame.require((m,n,k,lda,ldb,ldc)==(1,1,offset+len(ak),offset+len(ak)+2,
                4 if op=='gemm' else 1,5 if op=='gemm' else 1),'numeric witness shape')
            a=[0]*k+[0x7f800001]*2;b=[0x7f800001]*(k*ldb)
            for q in range(k):
                if offset<=q<offset+len(ak):a[q]=ak[q-offset];b[q*ldb]=bk[q-offset]
                else:b[q*ldb]=0x80000000
            c=[int(literal['seed'],16)]+[0x7f800001]*(ldc-1)
        else:
            want_m,want_n=(t+1,3 if op=='gemm' else 1) if name=='mixed-rows' else (3,t+1)
            ame.require((m,n,k,lda,ldb,ldc)==(want_m,want_n,3,5,
                want_n+3 if op=='gemm' else 1,want_n+4 if op=='gemm' else 1),'mixed witness shape')
            av=(0x3f800001,0x7f800000,0x7f800001,1,0x80000000)
            bv=(0x3f7ffffe,0,0x3f000000,0x7fc12345,0xff800000)
            cv=(0xbf800000,0,0x80000000,0x00800000,0x7fc12345)
            a=[av[(p//lda+p%lda)%5] if p%lda<k else 0x7f800001 for p in range(m*lda)]
            b=[bv[(p//ldb+p%ldb)%5] if p%ldb<n else 0x7f800001 for p in range(k*ldb)]
            c=[cv[(p//ldc+p%ldc)%5] if p%ldc<n else 0x7f800001 for p in range(m*ldc)]
        ame.require((r['a'],r['b'],r['initial'])==(a,b,c),'numeric input/seed changed')
        want=list(c);flags=0
        for i in range(m):
            for j in range(n):
                value=c[i*ldc+j] if r['accumulate'] else 0
                for q in range(k):
                    value,added=fma(a[i*lda+q],b[q*ldb+j],value);flags|=added
                want[i*ldc+j]=value
        ame.require((r['golden'],r['new_flags'])==(want,flags),'numeric bits/flags changed')
    maximum=build['config']['max_n'];width=sum(max(len(r[k]) for r in rows) for k in ('a','b','initial'))
    count=6+4*(maximum**2+32)+3
    count+=len(selected)*(6+width+96+3*maximum**2+32-3*t*t)
    return selected,count
