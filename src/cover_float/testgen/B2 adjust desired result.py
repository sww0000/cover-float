"""
Angela Zheng (angela20061015@gmail.com)

Created:        April 6, 2026
Last Edited:    April 6, 2026
"""
# coverage 92.59%

import random
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
    OP_FMSUB,
    OP_FNMADD,
    OP_FNMSUB,
    OP_MUL,
    OP_SQRT,
    OP_SUB,
    ROUND_NEAR_EVEN,
)
from cover_float.common.util import reproducible_hash
from cover_float.reference import run_and_store_test_vector, run_test_vector


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    """Converts binary fp components into a 32-character padded hex string."""
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"


def get_result_from_ref(op: str, a: str, b: str, c: str, fmt: str) -> str:
    """Calls reference model and extracts the result hex."""
    vector = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00"
    res_str = run_test_vector(vector)
    return res_str.split("_")[6]


def testAddition(fmt: str, desired_result: str, base_e: int, maxnorm: bool, sign: int) -> tuple[str, str, str]:
    m_bits = MANTISSA_BITS[fmt]
    a_exp = base_e

    if maxnorm:
        if sign == 0:
            a = decimalComponentsToHex(fmt, 0, a_exp, random.getrandbits(m_bits))
        else:
            a = decimalComponentsToHex(fmt, 1, a_exp, random.getrandbits(m_bits))
    else:
        a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(m_bits))

    b = get_result_from_ref(OP_SUB, desired_result, a, "0" * 32, fmt)
    return a, b, get_result_from_ref(OP_ADD, a, b, "0" * 32, fmt)


def test_add(
    fmt: str,
    desired_result: str,
    base_e: int,
    maxnorm: bool,
    sign: int,
    test_f: TextIO,
    cover_f: TextIO,
    desired_m: int,
) -> None:
    a, b, result = testAddition(fmt, desired_result, base_e, maxnorm, sign)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, result = testAddition(fmt, desired_result, base_e, maxnorm, sign)

    run_and_store_test_vector(
        f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def testSubtraction(fmt: str, desired_result: str, base_e: int, maxnorm: bool, sign: int) -> tuple[str, str, str]:
    m_bits = MANTISSA_BITS[fmt]
    a_exp = base_e

    if maxnorm:
        if sign == 0:
            a = decimalComponentsToHex(fmt, 0, a_exp, random.getrandbits(m_bits))
        else:
            a = decimalComponentsToHex(fmt, 1, a_exp, random.getrandbits(m_bits))
    else:
        a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(m_bits))

    b = get_result_from_ref(OP_SUB, a, desired_result, "0" * 32, fmt)
    return a, b, get_result_from_ref(OP_SUB, a, b, "0" * 32, fmt)


def test_sub(
    fmt: str,
    desired_result: str,
    base_e: int,
    maxnorm: bool,
    sign: int,
    test_f: TextIO,
    cover_f: TextIO,
    desired_m: int,
) -> None:
    a, b, result = testSubtraction(fmt, desired_result, base_e, maxnorm, sign)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, result = testSubtraction(fmt, desired_result, base_e, maxnorm, sign)

    run_and_store_test_vector(
        f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def writeVectors(fmt: str, op: str, a: str, b: str, c: str, test_f: TextIO, cover_f: TextIO) -> None:
    run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def testMultiplication(
    fmt: str, desired_result: str, maxnorm: bool, test_f: TextIO, cover_f: TextIO
) -> tuple[str, str, str]:
    max_exp = BIASED_EXP[fmt][1]
    bias = BIAS[fmt]

    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
    else:
        min_safe_exp = 1
        max_safe_exp = bias

    a_exp = random.randint(min_safe_exp, max_safe_exp)
    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = get_result_from_ref(OP_DIV, desired_result, a, "0" * 32, fmt)

    return a, b, get_result_from_ref(OP_MUL, a, b, "0" * 32, fmt)


def test_mul(
    fmt: str,
    desired_result: str,
    maxnorm: bool,
    test_f: TextIO,
    cover_f: TextIO,
    sign: int,
    base_e: int,
    desired_m: int,
) -> None:
    a, b, result = testMultiplication(fmt, desired_result, maxnorm, test_f, cover_f)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, result = testMultiplication(fmt, desired_result, maxnorm, test_f, cover_f)

    run_and_store_test_vector(
        f"{OP_MUL}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def testDivision(fmt: str, desired_result: str, maxnorm: bool) -> tuple[str, str, str]:
    max_exp = BIASED_EXP[fmt][1]
    bias = BIAS[fmt]
    m_bits = MANTISSA_BITS[fmt]

    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
    else:
        min_safe_exp = 1
        max_safe_exp = bias - m_bits

    a_exp = random.randint(min_safe_exp, max_safe_exp)
    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = get_result_from_ref(OP_DIV, a, desired_result, "0" * 32, fmt)

    return a, b, get_result_from_ref(OP_DIV, a, b, "0" * 32, fmt)


def test_div(
    fmt: str,
    desired_result: str,
    maxnorm: bool,
    test_f: TextIO,
    cover_f: TextIO,
    sign: int,
    base_e: int,
    desired_m: int,
) -> None:
    a, b, result = testDivision(fmt, desired_result, maxnorm)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, result = testDivision(fmt, desired_result, maxnorm)

    run_and_store_test_vector(
        f"{OP_DIV}_{ROUND_NEAR_EVEN}_{a}_{b}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def test_sqrt(fmt: str, desired_result: str, test_f: TextIO, cover_f: TextIO) -> None:
    a = get_result_from_ref(OP_MUL, desired_result, desired_result, "0" * 32, fmt)

    run_and_store_test_vector(
        f"{OP_SQRT}_{ROUND_NEAR_EVEN}_{a}_{32 * '0'}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f
    )


def testFusedMultiplyAdd(fmt: str, desired_result: str, base_e: int, maxnorm: bool) -> tuple[str, str, str, str]:
    max_exp = BIASED_EXP[fmt][1]
    bias = BIAS[fmt]
    m_bits = MANTISSA_BITS[fmt]

    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)
    else:
        min_safe_exp = 1
        max_safe_exp = bias
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    c = get_result_from_ref(OP_FNMSUB, a, b, desired_result, fmt)

    return a, b, c, get_result_from_ref(OP_FMADD, a, b, c, fmt)


def test_fmadd(
    fmt: str,
    desired_result: str,
    base_e: int,
    maxnorm: bool,
    test_f: TextIO,
    cover_f: TextIO,
    sign: int,
    desired_m: int,
) -> None:
    a, b, c, result = testFusedMultiplyAdd(fmt, desired_result, base_e, maxnorm)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, c, result = testFusedMultiplyAdd(fmt, desired_result, base_e, maxnorm)

    run_and_store_test_vector(f"{OP_FMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def testFusedMultiplySubtract(fmt: str, desired_result: str, base_e: int, maxnorm: bool) -> tuple[str, str, str, str]:
    max_exp = BIASED_EXP[fmt][1]
    bias = BIAS[fmt]
    m_bits = MANTISSA_BITS[fmt]

    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)
    else:
        min_safe_exp = 1
        max_safe_exp = bias
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    c = get_result_from_ref(OP_FMSUB, a, b, desired_result, fmt)

    return a, b, c, get_result_from_ref(OP_FMSUB, a, b, c, fmt)


def test_fmsub(
    fmt: str,
    desired_result: str,
    base_e: int,
    maxnorm: bool,
    test_f: TextIO,
    cover_f: TextIO,
    sign: int,
    desired_m: int,
) -> None:
    a, b, c, result = testFusedMultiplySubtract(fmt, desired_result, base_e, maxnorm)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, c, result = testFusedMultiplySubtract(fmt, desired_result, base_e, maxnorm)

    run_and_store_test_vector(f"{OP_FMSUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def testNegatedFusedMultiplyAdd(fmt: str, desired_result: str, base_e: int, maxnorm: bool) -> tuple[str, str, str, str]:
    max_exp = BIASED_EXP[fmt][1]
    bias = BIAS[fmt]
    m_bits = MANTISSA_BITS[fmt]

    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)
    else:
        min_safe_exp = 1
        max_safe_exp = bias
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    c = get_result_from_ref(OP_FNMADD, a, b, desired_result, fmt)

    return a, b, c, get_result_from_ref(OP_FNMADD, a, b, c, fmt)


def test_fnmadd(
    fmt: str,
    desired_result: str,
    base_e: int,
    maxnorm: bool,
    test_f: TextIO,
    cover_f: TextIO,
    sign: int,
    desired_m: int,
) -> None:
    a, b, c, result = testNegatedFusedMultiplyAdd(fmt, desired_result, base_e, maxnorm)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, c, result = testNegatedFusedMultiplyAdd(fmt, desired_result, base_e, maxnorm)

    run_and_store_test_vector(f"{OP_FNMADD}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


def testNegatedFusedMultiplySubtract(
    fmt: str, desired_result: str, base_e: int, maxnorm: bool
) -> tuple[str, str, str, str]:
    max_exp = BIASED_EXP[fmt][1]
    bias = BIAS[fmt]
    m_bits = MANTISSA_BITS[fmt]

    if maxnorm:
        min_safe_exp = bias
        max_safe_exp = max_exp
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = random.randint(max(0, max_exp - a_exp - m_bits), max_exp - a_exp)
    else:
        min_safe_exp = 1
        max_safe_exp = bias
        a_exp = random.randint(min_safe_exp, max_safe_exp)
        b_exp = base_e - a_exp + bias

    a = decimalComponentsToHex(fmt, random.randint(0, 1), a_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    b = decimalComponentsToHex(fmt, random.randint(0, 1), b_exp, random.getrandbits(MANTISSA_BITS[fmt]))
    c = get_result_from_ref(OP_FMADD, a, b, desired_result, fmt)

    return a, b, c, get_result_from_ref(OP_FNMSUB, a, b, c, fmt)


def test_fnmsub(
    fmt: str,
    desired_result: str,
    base_e: int,
    maxnorm: bool,
    test_f: TextIO,
    cover_f: TextIO,
    sign: int,
    desired_m: int,
) -> None:
    a, b, c, result = testNegatedFusedMultiplySubtract(fmt, desired_result, base_e, maxnorm)
    desired_result = desired_result.lower()

    if result != desired_result:
        m_max = (1 << MANTISSA_BITS[fmt]) - 1
        if result > desired_result and desired_m > 0:
            desired_m -= 1
        elif result < desired_result and desired_m < m_max:
            desired_m += 1

        desired_result = decimalComponentsToHex(fmt, sign, base_e, desired_m)
        a, b, c, result = testNegatedFusedMultiplySubtract(fmt, desired_result, base_e, maxnorm)

    run_and_store_test_vector(f"{OP_FNMSUB}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{32 * '0'}_{fmt}_00", test_f, cover_f)


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
                        test_add(fmt, desired_result, base_e, maxnorm, sign, test_f, cover_f, desired_m)

                        seed(reproducible_hash(f"{fmt}_b2_sub_{base_e}_{i}_{sign}"))
                        test_sub(fmt, desired_result, base_e, maxnorm, sign, test_f, cover_f, desired_m)

                        seed(reproducible_hash(f"{fmt}_b2_mul_{base_e}_{i}_{sign}"))
                        test_mul(fmt, desired_result, maxnorm, test_f, cover_f, sign, base_e, desired_m)

                        seed(reproducible_hash(f"{fmt}_b2_div_{base_e}_{i}_{sign}"))
                        test_div(fmt, desired_result, maxnorm, test_f, cover_f, sign, base_e, desired_m)

                        if sign == 0 and base == "One":
                            seed(reproducible_hash(f"{fmt}_b2_sqrt_{base_e}_{i}_{sign}"))
                            test_sqrt(fmt, desired_result, test_f, cover_f)

                        seed(reproducible_hash(f"{fmt}_b2_fmadd_{base_e}_{i}_{sign}"))
                        test_fmadd(fmt, desired_result, base_e, maxnorm, test_f, cover_f, sign, desired_m)

                        seed(reproducible_hash(f"{fmt}_b2_fmsub_{base_e}_{i}_{sign}"))
                        test_fmsub(fmt, desired_result, base_e, maxnorm, test_f, cover_f, sign, desired_m)

                        seed(reproducible_hash(f"{fmt}_b2_fnmadd_{base_e}_{i}_{sign}"))
                        test_fnmadd(fmt, desired_result, base_e, maxnorm, test_f, cover_f, sign, desired_m)

                        seed(reproducible_hash(f"{fmt}_b2_fnmsub_{base_e}_{i}_{sign}"))
                        test_fnmsub(fmt, desired_result, base_e, maxnorm, test_f, cover_f, sign, desired_m)


if __name__ == "__main__":
    main()
