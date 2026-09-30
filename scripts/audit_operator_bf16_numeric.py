"""Reconstruct BF16 numerical coverage and inputs without using its generator."""
import json
import sys
import ame
sys.path.insert(0,str(ame.ROOT/'operators/reference'))
from fp32_fold import fma
from audit_operator_fp32_state import verify_execution

NAMES=set('''exact-cancel fused-overflow-rescue tie-even tie-odd negative-tie
sticky-beyond-half overflow-positive overflow-negative underflow-tie negative-underflow
exact-subnormal minimum-subnormal tiny-normal-delivery nonzero-seed-tiny
fused-subnormal-half above-underflow-half below-underflow-half infinity-right infinity-left
infinity-invalid-zero opposed-infinities quiet-nan-left signaling-nan-left
signaling-nan-right quiet-nan-seed signaling-nan-seed beta-zero-ignores-snan
negative-zero-seed opposed-zero-signs seed-before-fold order-sensitive
intermediate-flags overflow-then-cancel'''.split())

def accounting(build,rows,t):
    op=build['row']['operator'];ame.require(op in ('gemm','gemv'),'unknown operator')
    catalog=json.loads((ame.ROOT/'operators/cases/bf16-rne-witnesses.json').read_text())['witnesses']
    literals={r['name']:r for r in catalog}
    ame.require(set(literals)==NAMES and len(catalog)==33,'literal witness inventory')
    specs={name:(r,0) for name,r in literals.items()}
    for squares in (1,2,4):
        for name in ('fused-overflow-rescue','fused-subnormal-half','seed-before-fold',
                     'order-sensitive','intermediate-flags','negative-zero-seed'):
            specs[f'cross-{squares}-{name}']=(literals[name],squares*t-1)
    wanted={(name,r['add']) for name,(r,_) in specs.items()}
    wanted|={(name,add) for name in (('mixed-rows','mixed-columns') if op=='gemm' else ('mixed-rows',)) for add in (0,1)}
    selected=[r for r in rows if r['tile']==t]
    ame.require({(r['purpose'],r['accumulate']) for r in selected}==wanted and len(selected)==len(wanted),
                'BF16 numeric coverage gap/duplicate')
    for ordinal,r in enumerate(rows):
        if r['tile']!=t:continue
        ame.require((r['seed_flags'],r['seed_xsat'],r['fcsr'])==
                    (8 if ordinal%2 else 0,ordinal%2,0x6a if ordinal%2 else 0x85),'state seed changed')
        name=r['purpose'];m,n,k,lda,ldb,ldc=(r[v] for v in ('m','n','k','lda','ldb','ldc'))
        if name in specs:
            literal,offset=specs[name]
            ak=[int(v,16) for v in literal['a']];bk=[int(v,16) for v in literal['b']]
            ame.require(len(ak)==len(bk) and all(0<=v<=0xffff for v in ak+bk),'literal BF16 inputs')
            ame.require((m,n,k,lda,ldb,ldc)==(1,1,offset+len(ak),offset+len(ak)+2,
                4 if op=='gemm' else 1,5 if op=='gemm' else 1),'numeric witness shape')
            a=[0]*k+[0x7f81]*2;b=[0x7f81]*(k*ldb)
            for q in range(k):
                if offset<=q<offset+len(ak):a[q]=ak[q-offset];b[q*ldb]=bk[q-offset]
                else:b[q*ldb]=0x8000
            c=[int(literal['seed'],16)]+[0x7f800001]*(ldc-1)
        else:
            want_m,want_n=(t+1,3 if op=='gemm' else 1) if name=='mixed-rows' else (3,t+1)
            ame.require((m,n,k,lda,ldb,ldc)==(want_m,want_n,3,5,
                want_n+3 if op=='gemm' else 1,want_n+4 if op=='gemm' else 1),'mixed witness shape')
            av=(0x3f81,0x7f80,0x7f81,1,0x8000)
            bv=(0x3f7e,0,0x3700,0x7fc1,0xff80)
            cv=(0xbf800000,0,0x80000000,0x00800000,0x7fc12345)
            a=[av[(p//lda+p%lda)%5] if p%lda<k else 0x7f81 for p in range(m*lda)]
            b=[bv[(p//ldb+p%ldb)%5] if p%ldb<n else 0x7f81 for p in range(k*ldb)]
            c=[cv[(p//ldc+p%ldc)%5] if p%ldc<n else 0x7f800001 for p in range(m*ldc)]
        ame.require((r['a'],r['b'],r['initial'])==(a,b,c),'numeric input/seed changed')
        want=list(c);flags=0
        for i in range(m):
            for j in range(n):
                value=c[i*ldc+j] if r['accumulate'] else 0
                for q in range(k):
                    value,added=fma(a[i*lda+q]*65536,b[q*ldb+j]*65536,value);flags|=added
                want[i*ldc+j]=value
        ame.require((r['golden'],r['new_flags'])==(want,flags),'numeric bits/flags changed')
    maximum=build['config']['max_n'];width=sum(max(len(r[k]) for r in rows) for k in ('a','b','initial'))
    count=6+4*(maximum**2+32)+4
    count+=len(selected)*(12+width+96+5*maximum**2+96-5*t*t)
    return selected,count
