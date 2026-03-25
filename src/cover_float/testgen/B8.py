# B8 (rwolk@hmc.edu)

import itertools
import random
from pathlib import Path
from typing import Optional, TextIO

import cover_float.common.constants as constants
from cover_float.common.util import generate_test_vector, reproducible_hash
from cover_float.reference import run_test_vector  # , store_cover_vector

# Test Plan: Add/Sub (effective ops), Mul, FMA (effective ops), DIV, SQRT, then Converts are easy


def extract_rounding_info(cover_vector: str) -> dict[str, int]:
    fields = cover_vector.split("_")
    sgn = fields[-3]
    result_fmt = fields[-5].upper()

    # Place in a leading one so that we get all the significant figures possible
    interm_significand = int("1" + fields[-1], 16)
    interm_significand = bin(interm_significand)[2:][1:]
    if result_fmt in constants.FLOAT_FMTS:
        mantissa_length = constants.MANTISSA_BITS[result_fmt]
    # elif result_fmt in constants.INT_FMTS:
    #     mantissa_length = constants.INT_MAX_EXPS[result_fmt]
    else:
        raise ValueError(f"Unknown Result Format: {result_fmt}")

    lsb = interm_significand[mantissa_length - 1]
    guard = interm_significand[mantissa_length]
    sticky = interm_significand[mantissa_length + 1 :]
    return {
        "Sign": int(sgn),
        "LSB": int(lsb),
        "Guard": int(guard),
        "Sticky": 1 if any(x == "1" for x in sticky) else 0,
    }


def generate_float(sign: int, exponent: int, mantissa: int, fmt: str) -> int:
    exponent += constants.BIAS[fmt]
    return (
        (sign << (constants.MANTISSA_BITS[fmt] + constants.EXPONENT_BITS[fmt]))
        | (exponent << constants.MANTISSA_BITS[fmt])
        | mantissa
    )


def bezout_inverse(x: int, base: int) -> int:
    # Find the inverse of an element using the Euclidean algorithm and applying Bezout's identity
    # The euclidean algorithm says: gcd(x, y) = gcd(y, x % y) for x > y
    # Bezout's identity says that there exists A, B in Z such that Ax + By = gcd(x, y)
    # With proper book keeping, we can find these X and Y, and noticing that
    # Ax + By = 1 when x, y are relatively prime (as x and base are assumed to be),
    # Ax = 1 - By implies Ax = 1 (mod y) and thus A inverts X in base y

    # Algorithm taken from https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    r = [base, x]
    s = [1, 0]
    t = [0, 1]

    while r[-1] != 0:
        q = r[-2] // r[-1]
        r.append(r[-2] - q * r[-1])
        s.append(s[-2] - q * s[-1])
        t.append(t[-2] - q * t[-1])

    gcd = r[-2]
    _bezout_A = s[-2]
    bezout_B = t[-2]

    if gcd != 1:
        return -1

    # We have bezout_A(base) + bezout_B(x) = gcd
    # So, as shown above, bezout_B inverts x
    return bezout_B % base


def divideSetRounding(
    lsb: int, guard: int, target: int, extra_bits: int, fmt: str, should_shift: Optional[bool] = None
) -> Optional[tuple[int, int]]:
    nf = constants.MANTISSA_BITS[fmt]

    # There is a potential shift when sigA < sigB, so make it random
    if not should_shift:
        should_shift = bool(random.randint(0, 1))

    for _ in range(100):
        sigB = random.getrandbits(nf) | 1 << nf | 1  # B must be odd in this approach ...

        bottom = sigB * target // (2**extra_bits)
        if bottom * (2**extra_bits) != sigB * target:
            bottom += 1  # Ceil division (through floating point) breaks for quads

        top = sigB * (target + 1) // (2**extra_bits)

        for c in range(bottom, top):
            sigA = (c * pow(2 ** (nf + 1 + should_shift), -1, sigB)) % sigB

            if sigA.bit_length() != nf + 1:
                sigA += sigB
                if sigA.bit_length() != nf + 1:
                    continue

            if (should_shift and sigA > sigB) or (not should_shift and sigA < sigB):
                continue

            # The shift is + 1 for the guard, and then another if we should shift

            divide_result = (sigA << (nf + extra_bits + 1 + should_shift)) // sigB
            binary_result = bin(divide_result)[2:]

            generated_lsb = int(binary_result[nf])
            generated_guard = int(binary_result[nf + 1])
            generated_sticky = int(binary_result[nf + 2 :], 2)

            if generated_lsb == lsb and generated_guard == guard and generated_sticky == target:
                return (sigA, sigB)

    return None


def check_div_result(result: str, target: int, sticky_length: int) -> bool:
    interm_mantissa = bin(int("1" + result.split("_")[-1], 16))[3:]
    fmt = result.split("_")[-5].upper()
    nf = constants.MANTISSA_BITS[fmt]

    sticky = interm_mantissa[nf + 1 :]
    relevant_bits = sticky[:sticky_length]

    return int(relevant_bits, 2) == target


def generate_div_tests(fmt: str, test_f: TextIO, cover_f: TextIO, target_bits: Optional[int] = None) -> None:
    seed = reproducible_hash(f"B8 DIV {fmt}")
    random.seed(seed)

    nf = constants.MANTISSA_BITS[fmt]
    if not target_bits:
        target_bits = nf - 2

    for target in range(1, 4):
        for lsb, guard in itertools.product(range(2), range(2)):
            maybe_result = divideSetRounding(lsb, guard, target, target_bits, fmt)
            if not maybe_result:
                print(f"Failure for lsb={lsb}, guard={guard}, sticky={target:b}")
                continue

            s1, s2 = maybe_result
            # print(bin((s1 << (nf + nf + 1)) // s2))

            f1 = generate_float(0, 0, s1 & ((1 << nf) - 1), fmt)
            f2 = generate_float(0, 0, s2 & ((1 << nf) - 1), fmt)

            tv = generate_test_vector(constants.OP_DIV, f1, f2, 0, fmt, fmt)
            result = run_test_vector(tv)
            info = extract_rounding_info(result)
            if (
                check_div_result(result, target, target_bits) and info["Guard"] == guard and info["LSB"] == lsb
            ) or fmt == "03":  # FIXME
                # store_cover_vector(result, test_f, cover_f)
                pass
            else:
                print("Div Result Failure")
                breakpoint()

    for target_offset in range(4, 0, -1):
        target = (1 << target_bits) - target_offset
        for lsb, guard in itertools.product(range(2), range(2)):
            maybe_result = divideSetRounding(lsb, guard, target, target_bits, fmt)
            if not maybe_result:
                print(f"Failure for lsb={lsb}, guard={guard}, sticky={target:b}")
                continue

            s1, s2 = maybe_result
            # print(bin((s1 << (nf + nf + 1)) // s2))

            f1 = generate_float(0, 0, s1 & ((1 << nf) - 1), fmt)
            f2 = generate_float(0, 0, s2 & ((1 << nf) - 1), fmt)

            tv = generate_test_vector(constants.OP_DIV, f1, f2, 0, fmt, fmt)
            result = run_test_vector(tv)
            info = extract_rounding_info(result)
            if (
                check_div_result(result, target, target_bits) and info["Guard"] == guard and info["LSB"] == lsb
            ) or fmt == "03":  # FIXME
                # store_cover_vector(result, test_f, cover_f)
                pass
            else:
                print("Div Result Failure")
                breakpoint()


def main() -> None:
    with (
        Path("tests/testvectors/B8_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B8_cv.txt").open("w") as cover_f,
    ):
        for fmt in constants.FLOAT_FMTS:
            generate_div_tests(fmt, test_f, cover_f)


if __name__ == "__main__":
    main()
