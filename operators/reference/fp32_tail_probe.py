"""Literal RNE padding witnesses, not a general floating GEMM oracle."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fp32_fixtures import TILES, emit

def create(spec):
    op=spec['operator']
    if op not in ('gemm','gemv') or spec['contract']!='fp32-tail-probe-v1':
        raise ValueError('unknown tail probe')
    rows=[]
    for t in TILES:
        gemv=op=='gemv'
        # NaN padding belongs to the external logical storage, never active data.
        a=[0x80000000 if gemv else 0x3f800000,0x7f812345,0x7f812345]
        b=[0x3f800000] if gemv else [0x7f800000]+[0x7f812345]*3
        initial=[0x80000000] if gemv else [0x7f812345]*5
        golden=[0x80000000] if gemv else [0x7f800000]+initial[1:]
        rows.append(dict(name=f't{t}-'+('negative-zero' if gemv else 'inf-unused-row'),
            purpose='negative-zero' if gemv else 'inf-unused-row',tile=t,
            m=1,n=1,k=1,lda=3,ldb=1 if gemv else 4,ldc=1 if gemv else 5,
            accumulate=int(gemv),a=a,b=b,initial=initial,golden=golden))
    return rows

def generate(manifest,out,config):
    rows=create(json.loads(Path(manifest).read_text()))
    return emit(rows,out,config,'literal 1*Inf+0=Inf without NV; (-0)*1+(-0)=-0 under RNE')
