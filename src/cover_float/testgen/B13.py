"""
Angela Zheng (angela20061015@gmail.com)

Basically just loop through -p to +1 that results in -126
"""

import random
from pathlib import Path
from random import seed
from typing import TextIO

from cover_float.common.constants import (
    BIASED_EXP,
    EXPONENT_BITS,
    FLOAT_FMTS,
    MANTISSA_BITS,
    OP_ADD,
    OP_SUB,
    ROUND_NEAR_EVEN,
)
from cover_float.common.util import reproducible_hash
from cover_float.reference import run_and_store_test_vector


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"


def writeAdd(fmt: str, a_hex: str, b_hex: str, test_f: TextIO, cover_f: TextIO) -> None:
    run_and_store_test_vector(
        f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
        test_f,
        cover_f,
    )


def writeSub(fmt: str, a_hex: str, b_hex: str, test_f: TextIO, cover_f: TextIO) -> None:
    run_and_store_test_vector(
        f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
        test_f,
        cover_f,
    )


def makeNegPMantissas(fmt: str) -> tuple[int, int]:
    m = MANTISSA_BITS[fmt]
    a_m = 0
    b_m = (1 << m) - 1
    return a_m, b_m


def makeCancellationMantissas(fmt: str, d: int) -> tuple[int, int]:
    m = MANTISSA_BITS[fmt]
    k = -d

    prefix = random.getrandbits(k - 1) << m - k + 1 if k > 1 else 0
    diff_bit = 1 << (m - k)

    if k < (m - 1):
        a_tail = 1 << (m - k - 2) | random.getrandbits(m - k - 2)
        b_tail = random.getrandbits(m - k - 2)
    else:
        a_tail = 0
        b_tail = 0

    a_m = prefix | diff_bit | a_tail
    b_m = prefix | b_tail

    return a_m, b_m


def makeNoCancelMantissas(fmt: str) -> tuple[int, int]:
    m = MANTISSA_BITS[fmt]
    a_m = (1 << m) - 1
    b_m = ((1 << (m - 1)) - 1) << 1
    return a_m, b_m


def makeTestVectors_B13(
    fmt: str,
    d: int,
    operation: str,
    test_f: TextIO,
    cover_f: TextIO,
) -> None:
    m = MANTISSA_BITS[fmt]
    p = m + 1
    min_exp = BIASED_EXP[fmt][0]

    is_add = operation == "add"
    write_fn = writeAdd if is_add else writeSub

    # B13 sweep exponent near minimum
    base_exp = min_exp - d

    a_exp = base_exp
    b_exp = base_exp

    # Mantissas (same logic as B12)
    if d == 0:
        a_m, b_m = makeNoCancelMantissas(fmt)
        b_exp -= 1
    elif d == -p:
        a_m, b_m = makeNegPMantissas(fmt)
        b_exp -= 1
    else:
        a_m, b_m = makeCancellationMantissas(fmt, d)

    # Sign logic
    if is_add:
        a_sign = 0
        b_sign = 1
    else:
        a_sign = 0
        b_sign = 0

    a_hex = decimalComponentsToHex(fmt, a_sign, a_exp, a_m)
    b_hex = decimalComponentsToHex(fmt, b_sign, b_exp, b_m)

    write_fn(fmt, a_hex, b_hex, test_f, cover_f)


def SubnormCancellationTests(test_f: TextIO, cover_f: TextIO, fmt: str) -> None:
    p = MANTISSA_BITS[fmt] + 1

    for d in range(-p, 1):  # d = 1 is impossible
        seed(reproducible_hash(OP_ADD + fmt + "b13"))
        makeTestVectors_B13(fmt, d, "add", test_f, cover_f)

        seed(reproducible_hash(OP_SUB + fmt + "b13"))
        makeTestVectors_B13(fmt, d, "sub", test_f, cover_f)


def main() -> None:
    with (
        Path("./tests/testvectors/B13_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B13_cv.txt").open("w") as cover_f,
    ):
        test_f.write("// B13 Cancellation + Subnormal Region Tests\n")
        test_f.write("// Operations: ADD, SUB\n")

        for fmt in FLOAT_FMTS:
            SubnormCancellationTests(test_f, cover_f, fmt)


if __name__ == "__main__":
    main()
