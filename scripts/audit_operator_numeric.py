"""Check actual boundary values and safe integer domains, not just case labels."""
import ame

def selected(build,rows,n):
    op=build['row']['operator']
    scalars={'positive-limit':(2147483647,1,0,0,2147483647),
        'negative-limit':(-2147483647,1,0,0,-2147483647),
        'high-signflip':(1073741824,-1,0,0,-1073741824),
        'add-positive-limit':(2,3,2147483641,1,2147483647),
        'add-negative-limit':(-2,3,-2147483641,1,-2147483647),
        'product-near-limit':(46340,46340,0,0,2147395600),
        'product-add-limit':(46340,46340,88047,1,2147483647),
        'min-left-times-zero':(-2147483648,0,0,0,0),
        'min-right-times-zero':(0,-2147483648,0,0,0)}
    names=set(scalars)|{f'{kind}-{add}' for add in (0,1) for kind in
        (('cancellation','long-k','cross-m','cross-n') if op=='gemm' else
         ('cancellation','long-k','cross-m'))}
    cases=[r for r in rows if r['tile']==n]
    ame.require(len(cases)==len(names) and {r['purpose'] for r in cases}==names,
                'numeric coverage gap/duplicate')
    for r in cases:
        if r['purpose'] in scalars:
            ame.require((r['m'],r['n'],r['k'])==(1,1,1), 'wrong scalar shape')
            ame.require((r['a'][0],r['b'][0],r['initial'][0],r['accumulate'],r['golden'][0])
                        ==scalars[r['purpose']], 'wrong numeric boundary values')
        elif r['purpose'].startswith('long-k'):
            ame.require(r['k']==2*n+1, 'missing multi-block high-bit case')
        elif r['purpose'].startswith('cross-m'):
            ame.require(r['m']==n+1, 'missing output-row tile transition')
        elif r['purpose'].startswith('cross-n'):
            ame.require(r['n']==n+1, 'missing output-column tile transition')
        for i in range(r['m']):
            for j in range(r['n']):
                seed=r['initial'][i*r['ldc']+j] if r['accumulate'] else 0
                terms=[r['a'][i*r['lda']+p]*r['b'][p*r['ldb']+j] for p in range(r['k'])]
                ame.require(abs(seed)+sum(abs(v) for v in terms)<=2147483647,
                            'outside bounded integer domain')
                ame.require(r['golden'][i*r['ldc']+j]==seed+sum(terms), 'numeric GOLDEN mismatch')
    return cases,9
