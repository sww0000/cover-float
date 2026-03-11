"""
Angela Zheng

March 3, 2026

SUMMARY
This script generates test vectors for the B2 model: Near FP Base Values - Hamming Distance.
It takes specific boundary values (Zero, One, MinSubNorm, etc.) and enumerates over
small deviations by flipping one bit of the significand at a time.

DEFINITION
Base Values: Zero, One, MinSubNorm, MaxSubNorm, MinNorm, MaxNorm
Operations: add, sub, multiply, fmadd, fmsub, fnmadd, fnmsub, sqrt
Total test vectors generated: TBD

For 32 b fp,
Zero        00000000
One         3F800000
Minsubnorm  00000001
Maxsubnorm  007FFFFF
MinNorm     00800000
MaxNrom     7F7FFFFF
"""

import random
from pathlib import Path
from typing import TextIO

from cover_float.common.constants import (
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
from cover_float.reference import run_and_store_test_vector, run_test_vector

THREE_OP_OPS = [OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]
ALL_OPS = [OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_SQRT, OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]


def decimalComponentsToHex(fmt: str, sign: int, biased_exp: int, mantissa: int) -> str:
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{MANTISSA_BITS[fmt]}b}"
    bits = b_sign + b_exp + b_man
    return f"{int(bits, 2):032X}"


def get_result_from_ref(op: str, a_hex: str, b_hex: str, c_hex: str, fmt: str) -> str:
    """Helper to call reference model and extract the hex result string."""
    vector = f"{op}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{c_hex}_{fmt}_{fmt}_{fmt}_00"
    res_str = run_test_vector(vector)
    # Extracts result hex: typically at index 7 in the underscore-separated string
    return res_str.split("_")[7]


def adjust_ulp(hex_val: str, delta: int) -> str:
    """Increments or decrements a hex value by 1 ULP (integer math)."""
    val = int(hex_val, 16)
    return f"{(val + delta):032X}"


def get_bases(fmt: str) -> dict[str, tuple[int, int]]:
    """Definition of base boundary values."""
    m = MANTISSA_BITS[fmt]
    e = EXPONENT_BITS[fmt]
    bias = (2 ** (e - 1)) - 1
    return {
        "Zero": (0, 0),
        "One": (0, bias),
        "MinSubNorm": (1, 0),
        "MaxSubNorm": ((1 << m) - 1, 0),
        "MinNorm": (0, 1),
        "MaxNorm": ((1 << m) - 1, (1 << e) - 2),
    }


# --- Test Generation Methods ---


def addTest(fmt: str, target: str, test_f: TextIO, cover_f: TextIO) -> None:
    a_hex = f"{random.getrandbits(128):032X}"[: (EXPONENT_BITS[fmt] + MANTISSA_BITS[fmt] + 1) // 4]  # Trim to fmt
    # Solve: b = target - a (using reference model subtraction)
    b_hex = get_result_from_ref(OP_SUB, target, a_hex, "0" * 32, fmt)
    run_and_store_test_vector(
        f"{OP_ADD}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{fmt}_{fmt}_00", test_f, cover_f
    )


def subTest(fmt: str, target: str, test_f: TextIO, cover_f: TextIO) -> None:
    a_hex = f"{random.getrandbits(128):032X}"[:8]
    # Solve: b = a - target
    b_hex = get_result_from_ref(OP_SUB, a_hex, target, "0" * 32, fmt)
    run_and_store_test_vector(
        f"{OP_SUB}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{fmt}_{fmt}_00", test_f, cover_f
    )


def mulTest(fmt: str, target: str, test_f: TextIO, cover_f: TextIO) -> None:
    a_hex = f"{random.getrandbits(128):032X}"[:8]
    b_hex = get_result_from_ref(OP_DIV, target, a_hex, "0" * 32, fmt)

    current_res = get_result_from_ref(OP_MUL, a_hex, b_hex, "0" * 32, fmt)
    if int(current_res, 16) < int(target, 16):
        b_hex = adjust_ulp(b_hex, 1)
    elif int(current_res, 16) > int(target, 16):
        b_hex = adjust_ulp(b_hex, -1)

    run_and_store_test_vector(
        f"{OP_MUL}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{fmt}_{fmt}_00", test_f, cover_f
    )


def divTest(fmt: str, target: str, test_f: TextIO, cover_f: TextIO) -> None:
    b_hex = f"{random.getrandbits(128):032X}"[:8]
    a_hex = get_result_from_ref(OP_MUL, target, b_hex, "0" * 32, fmt)

    current_res = get_result_from_ref(OP_DIV, a_hex, b_hex, "0" * 32, fmt)
    if int(current_res, 16) < int(target, 16):
        a_hex = adjust_ulp(a_hex, 1)
    elif int(current_res, 16) > int(target, 16):
        a_hex = adjust_ulp(a_hex, -1)

    run_and_store_test_vector(
        f"{OP_DIV}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{'0' * 32}_{fmt}_{fmt}_{fmt}_00", test_f, cover_f
    )


def sqrtTest(fmt: str, target: str, test_f: TextIO, cover_f: TextIO) -> None:
    a_hex = get_result_from_ref(OP_MUL, target, target, "0" * 32, fmt)

    current_res = get_result_from_ref(OP_SQRT, a_hex, "0" * 32, "0" * 32, fmt)
    if int(current_res, 16) < int(target, 16):
        a_hex = adjust_ulp(a_hex, 1)
    elif int(current_res, 16) > int(target, 16):
        a_hex = adjust_ulp(a_hex, -1)

    run_and_store_test_vector(
        f"{OP_SQRT}_{ROUND_NEAR_EVEN}_{a_hex}_{'0' * 32}_{'0' * 32}_{fmt}_{fmt}_{fmt}_00", test_f, cover_f
    )


def fmaSuite(op: str, inv_op: str, fmt: str, target: str, test_f: TextIO, cover_f: TextIO) -> None:
    """Generic solver for FMADD/FMSUB/FNMADD/FNMSUB using inverse logic."""
    a_hex = f"{random.getrandbits(128):032X}"[:8]
    b_hex = f"{random.getrandbits(128):032X}"[:8]

    # Logic for solving C: roughly C = target - (A*B)
    # We use the inverse op provided to find an initial C_hex
    c_hex = get_result_from_ref(inv_op, a_hex, b_hex, target, fmt)

    current_res = get_result_from_ref(op, a_hex, b_hex, c_hex, fmt)
    if int(current_res, 16) < int(target, 16):
        c_hex = adjust_ulp(c_hex, 1)
    elif int(current_res, 16) > int(target, 16):
        c_hex = adjust_ulp(c_hex, -1)

    run_and_store_test_vector(f"{op}_{ROUND_NEAR_EVEN}_{a_hex}_{b_hex}_{c_hex}_{fmt}_{fmt}_{fmt}_00", test_f, cover_f)


# --- Main Logic ---


def main() -> None:
    test_path = Path("./tests/testvectors/B2_tv.txt")
    cover_path = Path("./tests/covervectors/B2_cv.txt")
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with test_path.open("w") as test_f, cover_path.open("w") as cover_f:
        for fmt in FLOAT_FMTS:
            bases = get_bases(fmt)
            for _, (base_m, base_e) in bases.items():
                for i in range(MANTISSA_BITS[fmt]):
                    # Toggle bit i of mantissa
                    desired_m = base_m ^ (1 << i)
                    for sign in [0, 1]:
                        target_hex = decimalComponentsToHex(fmt, sign, base_e, desired_m)

                        addTest(fmt, target_hex, test_f, cover_f)
                        subTest(fmt, target_hex, test_f, cover_f)
                        mulTest(fmt, target_hex, test_f, cover_f)
                        divTest(fmt, target_hex, test_f, cover_f)

                        if sign == 0:
                            sqrtTest(fmt, target_hex, test_f, cover_f)

                        # Solve FMAs by calling reference with the "inverse" logic
                        fmaSuite(OP_FMADD, OP_FNMADD, fmt, target_hex, test_f, cover_f)
                        fmaSuite(OP_FMSUB, OP_FMSUB, fmt, target_hex, test_f, cover_f)
                        fmaSuite(OP_FNMADD, OP_FMADD, fmt, target_hex, test_f, cover_f)
                        fmaSuite(OP_FNMSUB, OP_FNMSUB, fmt, target_hex, test_f, cover_f)


if __name__ == "__main__":
    main()
