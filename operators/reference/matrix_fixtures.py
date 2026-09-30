"""Host fixture generator for logical i32 GEMM/GEMV (not AME tile layout)."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import integer

def create(spec):
    op = spec["operator"]
    if op not in ("gemm", "gemv") or spec["contract"] != "i32-exact-v1":
        raise ValueError("unsupported matrix fixture contract")
    if not spec["cases"] or len({c["name"] for c in spec["cases"]}) != len(spec["cases"]):
        raise ValueError("empty/duplicate logical case names")
    result = []
    for case in spec["cases"]:
        m,n,k = (case[x] for x in ("m","n","k"))
        if any(type(x) is not int or x < 0 or x > 256 for x in (m,n,k)):
            raise ValueError("invalid dimensions")
        if op == "gemv" and (n != 1 or case["ldb"] != 1 or case["ldc"] != 1):
            raise ValueError("gemv requires a single unit-stride vector")
        seed = case["seed"]
        def data(rows,cols,stride,s):
            if stride < cols or stride > 512:
                raise ValueError("invalid leading dimension")
            return [(s+17*i+13*j)%7-3 if j < cols else 0x123456
                    for i in range(rows) for j in range(stride)] or [0x123456]
        a = data(m,k,case["lda"],seed)
        b = data(k,n,case["ldb"],seed+3)
        c = data(m,n,case["ldc"],seed+5)
        if op == "gemm":
            golden = integer.gemm(m,n,k,a,case["lda"],b,case["ldb"],c,case["ldc"],case["accumulate"])
        else:
            golden = integer.gemv(m,k,a,case["lda"],b,c,case["accumulate"])
        result.append(dict(case, a=a,b=b,initial=c,golden=golden))
    return result

def generate(manifest, out, config):
    rows = create(json.loads(Path(manifest).read_text()))
    max_c = max(len(r["initial"]) for r in rows)
    text = ['/* Generated from logical cases and independent host integer GOLDEN. */',
        'struct op_fixture { ame_size m,n,k,lda,ldb,ldc; int accumulate;',
        '  ame_size a_count,b_count,c_count; ame_i32 *a,*b;',
        '  const ame_i32 *a_copy,*b_copy,*initial,*golden; };']
    for i,row in enumerate(rows):
        for name in ("a","b","a_copy","b_copy","initial","golden"):
            key = name.replace("_copy","")
            values = row[key]
            qualifier = "" if name in ("a","b") else "const "
            text.append(f'static {qualifier}ame_i32 op_{i}_{name}[] = {{' +
                        ",".join(str(v) for v in values) + '};')
    text.append('static const struct op_fixture op_fixtures[] = {')
    for i,row in enumerate(rows):
        values = [str(row[k]) for k in ("m","n","k","lda","ldb","ldc","accumulate")]
        values += [str(len(row[k])) for k in ("a","b","initial")]
        values += [f'op_{i}_{k}' for k in ("a","b","a_copy","b_copy","initial","golden")]
        text.append('  {' + ','.join(values) + '},')
    text += ['};',f'#define OP_CASES {len(rows)}',f'#define OP_MAX_C {max_c}', '']
    (out/'op_fixtures.h').write_text("\n".join(text))
    (out/'golden.json').write_text(json.dumps(rows,indent=2)+"\n")
    maximum = config["max_n"]
    checks = {str(n): 6+4*(maximum**2+32)+8+sum(
        1+max_c+32+3*maximum**2+32-3*n*n+len(r["a"])+len(r["b"]) for r in rows)
        for n in (4,8,16,32,64,128) if n <= maximum}
    return dict(logical_cases=len(rows),max_c=max_c,checks_by_n=checks,
                reference="Python exact integer, logical indices, no AME packing",
                files=["op_fixtures.h","golden.json"])
