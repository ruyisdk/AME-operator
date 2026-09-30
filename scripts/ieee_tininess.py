"""FP32 precision and after-rounding tininess reference."""
from fractions import Fraction

def pow2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)

def tiny_after_rounding(value, precision, emin, rm):
    if precision < 2 or rm not in range(5):
        raise ValueError('invalid precision or rounding mode')
    value = Fraction(value)
    if not value:
        return False
    sign, value = int(value < 0), abs(value)
    e = value.numerator.bit_length() - value.denominator.bit_length()
    if value < pow2(e):
        e -= 1
    scaled = value / pow2(e - precision + 1)
    q, rem = divmod(scaled.numerator, scaled.denominator)
    half = 2 * rem - scaled.denominator
    inc = ((rm in (0, 4) and (half > 0 or (half == 0 and (rm == 4 or q & 1))))
           or (bool(rem) and ((rm == 2 and sign) or (rm == 3 and not sign))))
    if q + inc == 1 << precision:
        e += 1
    return e < emin

def fp32_binary_expected(op, rm, a, b):
    from test_binary32 import expected as archived
    from test_sqrt32 import finite
    bits, flags = archived(op, rm, a, b)
    if not flags & 1:
        return bits, flags
    av = finite(a) * (-1 if a >> 31 else 1)
    bv = finite(b) * (-1 if b >> 31 else 1)
    value = (av+bv, bv-av, av*bv)[op]
    return bits, (flags & ~2) | (2 if tiny_after_rounding(value, 24, -126, rm) else 0)

def fp32_fused_expected(op, rm, a, b, d):
    from test_fused32 import expected as archived
    from test_sqrt32 import finite
    bits, flags = archived(op, rm, a, b, d)
    if not flags & 1:
        return bits, flags
    def signed(v): return finite(v) * (-1 if v >> 31 else 1)
    av,bv = signed(a),signed(b)
    if not op:
        value = -av*bv
    else:
        dv = signed(d)
        value = {1:dv+av*bv,2:dv-av*bv,3:av+bv*dv,4:av-bv*dv}[op]
    return bits, (flags & ~2) | (2 if tiny_after_rounding(value, 24, -126, rm) else 0)
