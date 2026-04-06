"""
Angela Zheng (angela20061015@gmail.com)

Created:         April 6, 2026
Last Edited:     April 6, 2026
"""

import random
from decimal import Context, Decimal, setcontext
from pathlib import Path
from random import seed
from typing import TextIO

from cover_float.common.constants import (
    BIAS,
    BIASED_EXP,
    EXPONENT_BITS,
    FLOAT_FMTS,
    MANTISSA_BITS,
    OP_ADD,
    OP_DIV,
    OP_FMADD,
    OP_MUL,
    OP_SQRT,
    OP_SUB,
    ROUND_NEAR_EVEN,
)
from cover_float.common.util import reproducible_hash
from cover_float.reference import run_and_store_test_vector, run_test_vector

# binary128 has a huge exponent range. Let's set a very safe context.
# Emax for binary128 is 16383. Python's Decimal default is usually 999999,
# but let's be explicit to avoid any "Subnormal" flags tripping in Python.
custom_context = Context(prec=110, Emax=999999, Emin=-999999)
setcontext(custom_context)


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    """Converts binary fp components into a 32-character padded hex string."""
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"


def hexToDecimal(hex_str: str, fmt: str) -> Decimal:
    """Converts a hex string representing a float into a high-precision Decimal object."""
    bits_total = 1 + EXPONENT_BITS[fmt] + MANTISSA_BITS[fmt]
    val = int(hex_str, 16)

    sign = (val >> (bits_total - 1)) & 1
    exp_raw = (val >> MANTISSA_BITS[fmt]) & ((1 << EXPONENT_BITS[fmt]) - 1)
    mantissa_raw = val & ((1 << MANTISSA_BITS[fmt]) - 1)

    bias = BIAS[fmt]

    if exp_raw == 0:  # Subnormal
        exponent = 1 - bias
        fraction = Decimal(mantissa_raw) / Decimal(2 ** MANTISSA_BITS[fmt])
    else:  # Normal
        exponent = exp_raw - bias
        fraction = Decimal(1) + (Decimal(mantissa_raw) / Decimal(2 ** MANTISSA_BITS[fmt]))

    res = fraction * (Decimal(2) ** exponent)
    return -res if sign else res


def decimalToHex(dec_val: Decimal, fmt: str) -> str:
    """Robust conversion of a Decimal back to the closest hex representation."""
    sign = 1 if dec_val < 0 else 0
    abs_val = abs(dec_val)

    if abs_val == 0:
        return decimalComponentsToHex(fmt, sign, 0, 0)

    # Calculate the base-2 exponent safely using Decimal math
    # log2(x) = ln(x) / ln(2)
    ln_2 = Decimal(2).ln()
    exp_val = int((abs_val.ln() / ln_2).to_integral_value(rounding="ROUND_FLOOR"))

    bias = BIAS[fmt]
    biased_exp = exp_val + bias
    m_bits = MANTISSA_BITS[fmt]

    # Handle Subnormals
    if biased_exp <= 0:
        biased_exp = 0
        # For subnormals, the value is: 0.mantissa * 2^(1-bias)
        # So: mantissa = abs_val / 2^(1-bias) * 2^m_bits
        divisor = Decimal(2) ** (1 - bias)
        mantissa = int((abs_val / divisor) * Decimal(2) ** m_bits)
    else:
        # Handle Normals
        # The value is: (1 + mantissa/2^m_bits) * 2^exp_val
        # So: mantissa = (abs_val / 2^exp_val - 1) * 2^m_bits
        divisor = Decimal(2) ** exp_val  # Fix: Power happens inside Decimal
        mantissa = int(((abs_val / divisor) - 1) * Decimal(2) ** m_bits)

    # Final clamping to prevent overflow of mantissa bits due to precision noise
    mantissa = max(0, min(mantissa, (1 << m_bits) - 1))

    return decimalComponentsToHex(fmt, sign, biased_exp, mantissa)


def get_result_from_ref(op: str, a: str, b: str, c: str, fmt: str) -> str:
    """Calls reference model and extracts the result hex."""
    vector = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00"
    res_str = run_test_vector(vector)
    return res_str.split("_")[6]


# --- Updated Test Generation Functions ---


def test_add(
    fmt: str, desired_result: str, base_e: int, maxnorm: bool, sign: int, test_f: TextIO, cover_f: TextIO
) -> None:
    m_bits = MANTISSA_BITS[fmt]
    a_exp = base_e

    # Create operand A
    a_sign = sign if maxnorm else random.randint(0, 1)
    a_hex = decimalComponentsToHex(fmt, a_sign, a_exp, random.getrandbits(m_bits))

    # Math: B = Desired - A
    d_dec = hexToDecimal(desired_result, fmt)
    a_dec = hexToDecimal(a_hex, fmt)
    b_dec = d_dec - a_dec
    b_hex = decimalToHex(b_dec, fmt)

    run_and_store_test_vector(
        f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{'0' * 32}_{fmt}_00", test_f, cover_f
    )


def test_sub(
    fmt: str, desired_result: str, base_e: int, maxnorm: bool, sign: int, test_f: TextIO, cover_f: TextIO
) -> None:
    m_bits = MANTISSA_BITS[fmt]
    a_exp = base_e

    a_sign = sign if maxnorm else random.randint(0, 1)
    a_hex = decimalComponentsToHex(fmt, a_sign, a_exp, random.getrandbits(m_bits))

    # Math: A - B = Desired -> B = A - Desired
    d_dec = hexToDecimal(desired_result, fmt)
    a_dec = hexToDecimal(a_hex, fmt)
    b_dec = a_dec - d_dec
    b_hex = decimalToHex(b_dec, fmt)

    run_and_store_test_vector(
        f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{'0' * 32}_{fmt}_00", test_f, cover_f
    )


def test_mul(fmt: str, desired_result: str, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    max_exp = BIASED_EXP[fmt][1]
    bias = BIAS[fmt]

    min_safe_exp, max_safe_exp = (bias, max_exp) if maxnorm else (1, bias)
    a_exp = random.randint(min_safe_exp, max_safe_exp)
    a_hex = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))

    # Math: B = Desired / A
    d_dec = hexToDecimal(desired_result, fmt)
    a_dec = hexToDecimal(a_hex, fmt)
    if a_dec != 0:
        b_dec = d_dec / a_dec
        b_hex = decimalToHex(b_dec, fmt)
    else:
        b_hex = "0" * 32

    run_and_store_test_vector(
        f"{OP_MUL}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{'0' * 32}_{fmt}_00", test_f, cover_f
    )


def test_div(fmt: str, desired_result: str, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    bias = BIAS[fmt]
    m_bits = MANTISSA_BITS[fmt]

    min_safe_exp, max_safe_exp = (bias, BIASED_EXP[fmt][1]) if maxnorm else (1, bias - m_bits)
    a_exp = random.randint(min_safe_exp, max_safe_exp)
    a_hex = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))

    # Math: A / B = Desired -> B = A / Desired
    d_dec = hexToDecimal(desired_result, fmt)
    a_dec = hexToDecimal(a_hex, fmt)
    if d_dec != 0:
        b_dec = a_dec / d_dec
        b_hex = decimalToHex(b_dec, fmt)
    else:
        b_hex = "0" * 32

    run_and_store_test_vector(
        f"{OP_DIV}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{'0' * 32}_{fmt}_00", test_f, cover_f
    )


def test_sqrt(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    a = get_result_from_ref(OP_MUL, desired_result, desired_result, "0" * 32, fmt)

    run_and_store_test_vector(
        f"{OP_SQRT}_{ROUND_NEAR_EVEN}_{a}_{32 * '0'}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_fmadd(fmt: str, desired_result: str, base_e: int, maxnorm: bool, test_f: TextIO, cover_f: TextIO) -> None:
    bias, m_bits = BIAS[fmt], MANTISSA_BITS[fmt]

    if maxnorm:
        a_exp = random.randint(bias, BIASED_EXP[fmt][1])
        b_exp = random.randint(max(0, BIASED_EXP[fmt][1] - a_exp - m_bits), BIASED_EXP[fmt][1] - a_exp)
    else:
        a_exp = random.randint(1, bias)
        b_exp = base_e - a_exp + bias

    a_hex = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(m_bits))
    b_hex = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(m_bits))

    # Math: C = Desired - (A * B)
    d_dec = hexToDecimal(desired_result, fmt)
    a_dec = hexToDecimal(a_hex, fmt)
    b_dec = hexToDecimal(b_hex, fmt)
    c_dec = d_dec - (a_dec * b_dec)
    c_hex = decimalToHex(c_dec, fmt)

    run_and_store_test_vector(
        f"{OP_FMADD}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{c_hex}_{fmt}_{'0' * 32}_{fmt}_00", test_f, cover_f
    )


def main() -> None:
    with (
        Path("./tests/testvectors/B2_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B2_cv.txt").open("w") as cover_f,
    ):
        for fmt in FLOAT_FMTS:
            m_bits = MANTISSA_BITS[fmt]
            bases = {
                "Zero": (0, 0),
                "One": (0, (1 << (EXPONENT_BITS[fmt] - 1)) - 1),
                "MinSub": (1, 0),
                "MaxSub": ((1 << m_bits) - 1, 0),
                "MinNorm": (0, 1),
                "MaxNorm": ((1 << m_bits) - 1, (1 << EXPONENT_BITS[fmt]) - 2),
            }

            for base, (base_m, base_e) in bases.items():
                for i in range(m_bits):
                    desired_m = base_m ^ (1 << i)
                    for sign in [0, 1]:
                        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
                        maxnorm = base == "MaxNorm"

                        seed(reproducible_hash(f"{fmt}_b2_add_{base_e}_{i}_{sign}"))
                        test_add(fmt, desired_result, base_e, maxnorm, sign, test_f, cover_f)

                        seed(reproducible_hash(f"{fmt}_b2_sub_{base_e}_{i}_{sign}"))
                        test_sub(fmt, desired_result, base_e, maxnorm, sign, test_f, cover_f)

                        seed(reproducible_hash(f"{fmt}_b2_mul_{base_e}_{i}_{sign}"))
                        test_mul(fmt, desired_result, maxnorm, test_f, cover_f)

                        seed(reproducible_hash(f"{fmt}_b2_div_{base_e}_{i}_{sign}"))
                        test_div(fmt, desired_result, maxnorm, test_f, cover_f)

                        if sign == 0 and base == "One":
                            seed(reproducible_hash(f"{fmt}_b2_sqrt_{base_e}_{i}_{sign}"))
                            test_sqrt(fmt, desired_result, test_f, cover_f)

                        seed(reproducible_hash(f"{fmt}_b2_fmadd_{base_e}_{i}_{sign}"))
                        test_fmadd(fmt, desired_result, base_e, maxnorm, test_f, cover_f)

                        # seed(reproducible_hash(f"{fmt}_b2_fmsub_{base_e}_{i}_{sign}"))
                        # test_fmsub(fmt, desired_result, base_e, maxnorm, test_f, cover_f)

                        # seed(reproducible_hash(f"{fmt}_b2_fnmadd_{base_e}_{i}_{sign}"))
                        # test_fnmadd(fmt, desired_result, base_e, maxnorm, test_f, cover_f)

                        # seed(reproducible_hash(f"{fmt}_b2_fnmsub_{base_e}_{i}_{sign}"))
                        # test_fnmsub(fmt, desired_result, base_e, maxnorm, test_f, cover_f)


if __name__ == "__main__":
    main()
