"""Order-independent exact dyadic GOLDEN; no AME packing or model imports.

Inputs are integer units of 2**-fraction_bits, and C is in squared units.
This finite-normal-or-positive-zero subset deliberately does not round.
"""
FORMATS = {"fp32": 23, "bf16": 7}
EXACT_LIMIT = (1 << 24) - 1

def bits(units, fraction_bits, fmt="fp32"):
    if (type(units) is not int or type(fraction_bits) is not int
            or not 0 <= fraction_bits <= 256 or fmt not in FORMATS):
        raise ValueError("invalid dyadic encoding input")
    if not units:
        return 0
    mantissa = FORMATS[fmt]
    magnitude = abs(units)
    exponent = magnitude.bit_length() - 1 - fraction_bits
    if not -126 <= exponent <= 127:
        raise ValueError("outside finite-normal dyadic subset")
    shift = magnitude.bit_length() - 1 - mantissa
    if shift > 0:
        significant, remainder = divmod(magnitude, 1 << shift)
        if remainder:
            raise ValueError("dyadic value is not exactly representable")
    else:
        significant = magnitude << -shift
    return ((int(units < 0) << (mantissa + 8))
            | ((exponent + 127) << mantissa)
            | (significant - (1 << mantissa)))

def _shape(rows, cols, values, stride):
    if (any(type(v) is not int for v in (rows, cols, stride))
            or rows < 0 or cols < 0 or stride < cols):
        raise ValueError("invalid logical shape/stride")
    need = (rows - 1) * stride + cols if rows and cols else 0
    if len(values) < need:
        raise ValueError("short logical input")

def _options(fraction_bits, input_format):
    # Even a one-unit product must be normal FP32; cancellation to zero must
    # not conceal an underflowing intermediate. RNE and positive-zero input.
    if (type(fraction_bits) is not int or not 0 <= fraction_bits <= 63
            or input_format not in FORMATS):
        raise ValueError("invalid exact floating grid/format")

def dot(a, b, seed=0, *, fraction_bits=3, input_format="fp32"):
    _options(fraction_bits, input_format)
    if len(a) != len(b):
        raise ValueError("unequal dot extents")
    for value in (*a, *b):
        bits(value, fraction_bits, input_format)
    bits(seed, 2*fraction_bits)
    # Every possible partial sum lies on the same grid with <=24 significant
    # bits. This sufficient bound is stronger than checking only the final sum.
    products = [x*y for x, y in zip(a, b)]
    if abs(seed) + sum(map(abs, products)) > EXACT_LIMIT:
        raise ValueError("outside order-independent exact FP32 contract")
    result = seed + sum(products)
    bits(result, 2*fraction_bits)
    return result

def gemm(m, n, k, a, lda, b, ldb, c, ldc, accumulate,
         *, fraction_bits=3, input_format="fp32"):
    _options(fraction_bits, input_format)
    if accumulate not in (0, 1):
        raise ValueError("invalid accumulation mode")
    _shape(m, k, a, lda); _shape(k, n, b, ldb); _shape(m, n, c, ldc)
    for value in (*a, *b):
        bits(value, fraction_bits, input_format)
    for value in c:
        bits(value, 2*fraction_bits)
    result = list(c)
    for i in range(m):
        for j in range(n):
            result[i*ldc+j] = dot(a[i*lda:i*lda+k],
                [b[p*ldb+j] for p in range(k)], c[i*ldc+j] if accumulate else 0,
                fraction_bits=fraction_bits, input_format=input_format)
    return result

def gemv(m, k, a, lda, x, y, accumulate, **options):
    return gemm(m, 1, k, a, lda, x, 1, y, 1, accumulate, **options)
