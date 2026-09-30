"""T-relative logical fixtures; no target packing or instruction semantics."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integer

TILES = (4, 8, 16, 32, 64, 128)


def dimensions(op, tile):
    rows = []
    for axis in (("m", "n", "k") if op == "gemm" else ("m", "k")):
        for label, value in (("minus", tile-1), ("exact", tile),
                             ("plus", tile+1), ("double-plus", 2*tile+1)):
            shape = dict(m=3, n=5 if op == "gemm" else 1, k=5)
            shape[axis] = value
            rows.append((axis+"-"+label, shape))
    rows.append(("cross", dict(m=tile+1, n=tile-1 if op == "gemm" else 1,
                               k=tile+1)))
    rows.append(("long-k", dict(m=5, n=3 if op == "gemm" else 1, k=3*tile+1)))
    return rows


def create(spec):
    op = spec["operator"]
    if op not in ("gemm", "gemv") or spec["contract"] != "i32-geometry-v2":
        raise ValueError("unsupported geometry contract")
    if spec["tiles"] != list(TILES) or type(spec["seed"]) is not int:
        raise ValueError("geometry requires the six reviewed runtime tiles")
    result = []
    for t in TILES:
        for index, (purpose, shape) in enumerate(dimensions(op, t)):
            m, n, k = (shape[x] for x in ("m", "n", "k"))
            for accumulate in (0, 1):
                lda, ldb, ldc = k+2, n+3 if op == "gemm" else 1, n+4 if op == "gemm" else 1
                def data(rows, cols, stride, salt):
                    # Coordinate hashing avoids short periodic rows hiding stride faults.
                    return [(((i+1)*0x45d9f3b ^ (j+1)*0x119de1f3 ^ salt)
                             % 7 - 3) if j < cols else 0x123456
                            for i in range(rows) for j in range(stride)]
                seed = spec["seed"] + index*97 + t
                a = data(m, k, lda, seed)
                b = data(k, n, ldb, seed+17)
                c = data(m, n, ldc, seed+31)
                if op == "gemm":
                    golden = integer.gemm(m,n,k,a,lda,b,ldb,c,ldc,accumulate)
                else:
                    golden = integer.gemv(m,k,a,lda,b,c,accumulate)
                result.append(dict(name=f"t{t}-{purpose}-{'add' if accumulate else 'set'}",
                    tile=t, purpose=purpose, **shape, lda=lda, ldb=ldb, ldc=ldc,
                    accumulate=accumulate, a=a, b=b, initial=c, golden=golden))
    return result


def emit(rows, out, config):
    if config["max_n"] != 128:
        raise ValueError("geometry v2 storage is reviewed for max_n=128")
    max_c = max(len(r["initial"]) for r in rows)
    text = ['/* Generated independent logical geometry and integer GOLDEN. */',
        'struct op_fixture { ame_size tile,m,n,k,lda,ldb,ldc; int accumulate;',
        '  ame_size a_count,b_count,c_count; ame_i32 *a,*b;',
        '  const ame_i32 *a_copy,*b_copy,*initial,*golden; };']
    for i, row in enumerate(rows):
        for name in ("a", "b", "a_copy", "b_copy", "initial", "golden"):
            values = row[name.replace("_copy", "")]
            qualifier = "" if name in ("a", "b") else "const "
            text.append(f'static {qualifier}ame_i32 op_{i}_{name}[] = {{' +
                        ','.join(str(v) for v in values) + '};')
    text.append('static const struct op_fixture op_fixtures[] = {')
    for i, row in enumerate(rows):
        values = [str(row[k]) for k in ("tile","m","n","k","lda","ldb","ldc","accumulate")]
        values += [str(len(row[k])) for k in ("a","b","initial")]
        values += [f'op_{i}_{k}' for k in ("a","b","a_copy","b_copy","initial","golden")]
        text.append('  {' + ','.join(values) + '},')
    per_tile = len(rows)//len(TILES)
    text += ['};', f'#define OP_CASES {len(rows)}', f'#define OP_MAX_C {max_c}',
             f'#define OP_SELECTED_CASES {per_tile}', '']
    (out/'op_fixtures.h').write_text('\n'.join(text))
    (out/'golden.json').write_text(json.dumps(rows, separators=(',', ':'))+'\n')
    checks = {str(t): 6+4*(128**2+32)+9+sum(
        1+max_c+32+3*128**2+32-3*t*t+len(r['a'])+len(r['b'])
        for r in rows if r['tile']==t) for t in TILES}
    return dict(logical_cases=len(rows), cases_by_n={str(t):per_tile for t in TILES},
        max_c=max_c, checks_by_n=checks,
        reference='Python exact integer, logical indices, T-relative selected cases',
        files=['op_fixtures.h', 'golden.json'])


def generate(manifest, out, config):
    return emit(create(json.loads(Path(manifest).read_text())), out, config)
