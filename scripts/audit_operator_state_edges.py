"""Reconstruct cross-family call order, bit images and state expectations."""
import sys
import ame
sys.path.insert(0,str(ame.ROOT/'operators/reference'))
import floating_exact as exact
from audit_operator_fp32_state import verify_execution

def accounting(build,data,t):
    op=build['row']['operator'];ame.require(op in ('gemm','gemv') and data['operator']==op,'operator')
    names=('SET','ADD','K0_SET','K0_ADD','EMPTY','BAD_MODE','NO_WORK','BAD_A')
    expected=[]
    for old in (0,1):
        expected += [(0,old,f,a) for f in range(5) for a in names]
    expected += [(1,0,f,a) for f in range(5) for a in ('SET','K0_SET','K0_ADD','EMPTY','BAD_MODE','NO_WORK','BAD_A')]
    expected += [(2,1,f,a) for f in range(4,-1,-1) for a in ('SET','ADD')]
    rows=data['cases']
    ame.require([(r['phase'],r['old_un'],r['family'],r['action']) for r in rows]==expected,'state edge order/inventory')
    count=6+4*(build['config']['max_n']**2+32)+9
    for ordinal,(r,(phase,old,family,action)) in enumerate(zip(rows,expected)):
        width=(4,1,1,4,2)[family];canary=0x7f812345;mask=(1<<(8*width))-1
        if family<3:
            left,right=((-7,11),(-3,7),(200,3))[family]
            av,bv=left&mask,right&mask;seed=9;product=left*right
            result=(product+(9 if action=='ADD' else 0))&0xffffffff
        else:
            form='bf16' if family==4 else 'fp32'
            av,bv=exact.bits(-4,3,form),exact.bits(12,3,form);seed=exact.bits(16,6)
            result=exact.bits(exact.dot([-4],[12],16 if action=='ADD' else 0,input_format=form),6)
        a=[canary&mask]*64;b=list(a);c=[canary]*64
        a[16]=av;b[16]=bv;c[16]=seed
        spots=[16+(q//2)*3+q%2 if op=='gemm' else 16+q for q in range(8)]
        if action in ('K0_SET','K0_ADD'):
            for at,bits in zip(spots,(0x80000000,0x7f800001,0x7fc12345,0x7f800000,0xff800000,1,0x7f7fffff,0)):
                c[at]=bits
        status=0
        if action in ('BAD_MODE','BAD_A'):status=1
        elif action=='NO_WORK':status=2
        if phase==1 and action in ('SET','ADD','NO_WORK'):status=3
        computed=int(status==0 and action in ('SET','ADD'))
        want=list(c)
        if computed:want[16]=result
        if action=='K0_SET':
            for at in spots:want[at]=0
        ame.require((r['width'],r['action_id'],r['status'],r['computed'])==
            (width,names.index(action),status,computed),'state edge call contract')
        ame.require((r['flags'],r['xsat'],r['fcsr'])==((0,1,16,31)[ordinal%4],ordinal%2,0x6a if ordinal%2 else 0x85),'state seed changed')
        ame.require((r['a'],r['b'],r['initial'],r['golden'])==(a,b,c,want),'state edge bits changed')
        count+=7+2+192+3*build['config']['max_n']**2+96-3*t*t*computed
        if phase!=1:count+=4+int(not computed)
    return rows,count
