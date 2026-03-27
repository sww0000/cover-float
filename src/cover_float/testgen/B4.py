
# B4 Model: Overflow and Near Overflow
#
# This model creates a test-case for each of the following constraints on the
# intermediate results:
#
#   i.   All the numbers in the range [+MaxNorm – 3 ulp, +MaxNorm + 3 ulp]
#   ii.  All the numbers in the range [-MaxNorm – 3 ulp, -MaxNorm + 3 ulp]
#   iii. A random number that is larger than +MaxNorm + 3 ulp
#   iv.  A random number that is smaller than -MaxNorm – 3 ulp
#   v.   One number for every exponent in the range
#        [MaxNorm.exp - 3, MaxNorm.exp + 3] for positive and negative numbers
#
# Operation:     All
# Rounding Mode: All
# Enable Bits:   XE, OE (Both On and both Off)
#
# ULP definition used here:
#   ±1 ULP = last bit of mantissa (LSB=1), guard bit = 0, sticky = 0

import random
from pathlib import Path
from typing import TextIO
from random import seed 

import cover_float.common.constants as const
from cover_float.common.util import reproducible_hash
from cover_float.reference import run_and_store_test_vector

MAXNORM_MANTISSA = { #maxnrom mantissa in dec form -> only check for Half Precision for now
    const.FMT_HALF: [1020,  # Positive Max Norm - 3 ULP ~ 65408
                     1021,  # Positive Max Norm - 2 ULP ~ 65440
                     1022,  # Positive Max Norm - 1 ULP ~ 65504
                     1023,  # Positive Max Norm ~ 65504
                     1024,  # Positive Max Norm + 1 ULP~ 65600 (Inf)
                     1025,  # Positive Max Norm + 2 ULP~ 65792 (Inf)
                     1026], # Positive Max Norm + 3 ULP~ 66048 (Inf)
}


ROUNDING_MODES = [
    const.ROUND_NEAR_EVEN,
    const.ROUND_MINMAG,
    const.ROUND_MIN,
    const.ROUND_MAX,
    const.ROUND_NEAR_MAXMAG,
    const.ROUND_ODD
]

SRC1_OPS = [const.OP_SQRT, const.OP_CLASS]

SRC2_OPS = [
    const.OP_ADD,
    const.OP_SUB,
    const.OP_MUL,
    const.OP_DIV,
    const.OP_FEQ,
    const.OP_FLT,
    const.OP_FLE,
    const.OP_MIN,
    const.OP_MAX,
    const.OP_FSGNJ,
    const.OP_FSGNJN,
    const.OP_FSGNJX,
]

SRC3_OPS = [const.OP_FMADD, const.OP_FMSUB, const.OP_FNMADD, const.OP_FNMSUB]

import struct
from decimal import Decimal


def decimalToComponents(value:int, fmt: str) -> tuple[int, int]:
    """
    Break a decimal number into its IEEE 754 components for the given format.
    Sign is excluded — pass it separately to decimalComponentsToHex().

    Args:
        value: exact numeric value (int, float, or Decimal), must be positive
        fmt:   one of FMT_HALF, FMT_BF16, FMT_SINGLE, FMT_DOUBLE, FMT_QUAD

    Returns:
        (biased_exp, mantissa) as integers, ready for decimalComponentsToHex()

    Example:
        exp, mant = decimalToComponents(65504, const.FMT_HALF)
        hex_val = decimalComponentsToHex(const.FMT_HALF, sign, exp, mant)
    """
    exp_bits = const.EXPONENT_BITS[fmt]
    man_bits = const.MANTISSA_BITS[fmt]
    bias     = (1 << (exp_bits - 1)) - 1

    if fmt == const.FMT_SINGLE:
        raw        = struct.unpack(">I", struct.pack(">f", float(value)))[0]
        biased_exp = (raw >> 23) & 0xFF
        mantissa   = raw & 0x7FFFFF

    elif fmt == const.FMT_DOUBLE:
        raw        = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
        biased_exp = (raw >> 52) & 0x7FF
        mantissa   = raw & 0x000FFFFFFFFFFFFF

    elif fmt == const.FMT_HALF:
        # F16 MaxNorm-range values are exactly representable in F32, float() is safe
        raw32      = struct.unpack(">I", struct.pack(">f", float(value)))[0]
        f32_exp    = (raw32 >> 23) & 0xFF
        f32_mant   = raw32 & 0x7FFFFF
        unbiased   = f32_exp - 127
        biased_exp = unbiased + 15
        mantissa   = f32_mant >> (23 - 10)

    elif fmt == const.FMT_BF16:
        # BF16 values are exactly representable in F32, float() is safe
        raw32      = struct.unpack(">I", struct.pack(">f", float(value)))[0]
        biased_exp = (raw32 >> 23) & 0xFF
        mantissa   = (raw32 >> 16) & 0x7F

    elif fmt == const.FMT_QUAD:
        # Never call float() here — F128 values exceed float64 precision
        value      = value if not isinstance(value, Decimal) else value
        value      = abs(value)
        if value == 0:
            return (0, 0)
        # Compute unbiased exponent via integer bit_length (exact, no float)
        unbiased   = int(value).bit_length() - 1
        # Refine in case int truncation was off by one
        scale      = Decimal(2) ** unbiased
        normalized = value / scale
        if normalized >= 2:
            unbiased  += 1
            scale     *= 2
            normalized = value / scale
        elif normalized < 1:
            unbiased  -= 1
            scale     /= 2
            normalized = value / scale

        biased_exp = unbiased + bias

        # Extract mantissa bits via repeated doubling (arbitrary precision)
        fraction   = normalized - 1  # strip implicit leading 1
        mantissa   = 0
        for _ in range(man_bits):
            fraction  *= 2
            bit        = int(fraction)
            mantissa   = (mantissa << 1) | bit
            fraction  -= bit

    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return (biased_exp, mantissa)






def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    b_sign = f"{sign:01b}"
    b_exponent = f"{biased_exp:0{const.EXPONENT_BITS[fmt]}b}"
    b_mantissa = f"{mantissa:0{const.MANTISSA_BITS[fmt]}b}"
    b_complete = b_sign + b_exponent + b_mantissa
    h_complete = f"{int(b_complete, 2):032X}"
    return h_complete



from decimal import Decimal

MAXNORM_DECIMAL = {
    # FMT_HALF  |  MaxNorm = 65504  |  ulp = 32
    const.FMT_HALF: [
        65408,   # MaxNorm -3 ulp
        65440,   # MaxNorm -2 ulp
        65472,   # MaxNorm -1 ulp
        65504,   # MaxNorm
        65536,   # MaxNorm +1 ulp  (intermediate only - overflows)
        65568,   # MaxNorm +2 ulp  (intermediate only - overflows)
        65600,   # MaxNorm +3 ulp  (intermediate only - overflows)
    ],
}





def write_maxnorm_tests(fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    for target in MAXNORM_DECIMAL[fmt]:
        hashval = reproducible_hash(const.OP_ADD + fmt + "b4")
        seed(hashval)

        # Pass target directly as operand A — guarantees intermediate IS target
        exp_a, mant_a = decimalToComponents(target, fmt)

        for sign in (0, 1):
            hex_a = decimalComponentsToHex(fmt, sign, exp_a, mant_a)
            # B is ±0: sign bit matches, exp=0, mant=0
            hex_b = decimalComponentsToHex(fmt, sign, 0, 0)

            for rm in ROUNDING_MODES:
                run_and_store_test_vector(
                    f"{const.OP_ADD}_{rm}_{hex_a}_{hex_b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                    test_f, cover_f
                )

    
            











def write_exp_range_tests(fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    target_exp = const.BIASED_EXP[fmt][1]  # MaxNorm exponent
    for exp_diff in range(-3, 4):  # Sweep from -3 to +3 ULPs
        biased_exp = target_exp + exp_diff
        for sign in (0, 1):
            mant = random.getrandbits(const.MANTISSA_BITS[fmt])          # integer, like B14
            hex_a = decimalComponentsToHex(fmt, sign, biased_exp, mant)

            for rm in ROUNDING_MODES:
                for op in SRC1_OPS: 
                    hashval = reproducible_hash(op + fmt + "b4")
                    seed(hashval) 
                    run_and_store_test_vector(
                        f"{op}_{rm}_{hex_a}_{32 * '0'}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f, cover_f
                    )
                for op in SRC2_OPS:
                    hashval = reproducible_hash(op + fmt + "b4")
                    seed(hashval) 
                    mant_b = random.getrandbits(const.MANTISSA_BITS[fmt])
                    hex_b = decimalComponentsToHex(fmt, sign, biased_exp, mant_b)
                    run_and_store_test_vector(
                        f"{op}_{rm}_{hex_a}_{hex_b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f, cover_f
                    )
                for op in SRC3_OPS:
                    hashval = reproducible_hash(op + fmt + "b4")
                    seed(hashval) 
                    mant_b = random.getrandbits(const.MANTISSA_BITS[fmt])
                    mant_c = random.getrandbits(const.MANTISSA_BITS[fmt])
                    hex_b = decimalComponentsToHex(fmt, sign, biased_exp, mant_b)
                    hex_c = decimalComponentsToHex(fmt, sign, biased_exp, mant_c)
                    run_and_store_test_vector(
                        f"{op}_{rm}_{hex_a}_{hex_b}_{hex_c}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f, cover_f
                    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def main() -> None:
    # This will generate the hex for the internal intermediate result of MaxNorm + 1 ulp
    hex_result = decimalComponentsToHex(const.FMT_HALF, 0, 31, 0)
    print(hex_result)


    with (
        Path("tests/testvectors/B4_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B4_cv.txt").open("w") as cover_f,
    ):
       # write_maxnorm_tests(const.FMT_HALF, test_f, cover_f)
        for fmt in const.FLOAT_FMTS:
            write_exp_range_tests(fmt, test_f, cover_f)
    print("B4 generation complete.")


if __name__ == "__main__":
    main()

