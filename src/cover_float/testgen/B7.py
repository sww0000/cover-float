# Ryan Wolk (rwolk@hmc.edu)
# B7: Sticky Bit Calculation
# This model checks that the sticky bit is calculated correctly in each of the
# following cases (for every possible combination in the table). The Guard bit
# should be always 0, and the sign positive, so that miscalculation of the sticky bit
# will alter the final result.

# Operations: FMA, ADD, SUB, MUL
#   This excludes DIV and SQRT as targeting specific sticky values is impossible

import random
from pathlib import Path
from typing import TextIO

import cover_float.common.constants as constants
from cover_float.common.util import generate_float, generate_test_vector, reproducible_hash
from cover_float.reference import run_test_vector, store_cover_vector


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


def add_sub_tests(fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    hashval = reproducible_hash(fmt + "add sub" + "b9")
    random.seed(hashval)

    for op in [constants.OP_ADD, constants.OP_SUB]:
        nf = constants.MANTISSA_BITS[fmt]
        exp_min, exp_max = constants.BIASED_EXP[fmt]
        exp_min -= constants.BIAS[fmt]
        exp_min += nf
        exp_max -= constants.BIAS[fmt]

        # There are nf possible extra_bits
        for extra_bit in range(1, nf):
            sigA = random.getrandbits(nf) | (1 << nf)
            sigB = random.getrandbits(nf) | (1 << nf)

            exp_a = random.randint(exp_min, exp_max)
            shift_amount = random.randint(extra_bit + 1, nf)

            exp_b = exp_a - shift_amount
            rounding_bits_mask = (1 << shift_amount) - 1

            sigB &= ~rounding_bits_mask
            if op == constants.OP_ADD:
                sigB |= 1 << (shift_amount - extra_bit - 1)
            else:
                # We want to do something less trivial than invoking effective addition in both
                subtraction_target = rounding_bits_mask ^ ((1 << (shift_amount - extra_bit - 1)) - 1)
                sigB |= subtraction_target

            if ((sigA << shift_amount) + sigB).bit_length() != (sigA << shift_amount).bit_length():
                sigA ^= 1 << nf - 1

            f1 = generate_float(0, exp_a, sigA ^ (1 << nf), fmt)
            f2 = generate_float(0, exp_b, sigB ^ (1 << nf), fmt)

            if random.random() < 0.5:
                f1, f2 = f2, f1

            tv = generate_test_vector(op, f1, f2, 0, fmt, fmt, constants.ROUND_MAX)
            # run_and_store_test_vector(tv, cover_f, test_f)
            result = run_test_vector(tv)

            pre_rounding_mantissa = int("1" + result.split("_")[-1], 16)
            # Mask off 0b, leading one, and significant bits
            rounding_bits = bin(pre_rounding_mantissa)[2 + 1 + nf :]
            # Only get the remaining rounding bits
            rounding_bits = rounding_bits[:nf]

            if int(rounding_bits, 2) != 1 << (nf - extra_bit - 1):
                breakpoint()


def mul_tests(fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    hashval = reproducible_hash(fmt + "mul" + "b9")
    random.seed(hashval)

    # Mul generates U2.2Nf, this means that rounding bits are either going to have
    # length Nf or Nf + 1 depending on whether or not the first bit is zero.

    # The model is about the correct calculation of each sticky bit, so I believe that the
    # correct test plan is hitting each of these cases for all Nf + 1 possible sticky bits

    # A number theory approach works well here: We are searching for a pair of significands
    # a and b such that a, b = (target_bit) (mod 2^(nf + 2)). Finding significands that fit our
    # criteria should be very easy with a random search (quick math says 16 iterations on average
    # to find working significands for both mul behaviors)

    nf = constants.MANTISSA_BITS[fmt]
    for extra_bit in range(nf):  # The first bit of sticky being active in a shifted case is impossible
        target = 1 << extra_bit

        hit_with_shift = False
        hit_without_shift = False

        for _ in range(100):
            sig_a = (
                1 << nf | random.getrandbits(nf) | 1
            )  # A must be odd, this is a place for randomization in the future
            sig_a_inv = bezout_inverse(sig_a, 2 ** (nf + 2))

            sig_b = (sig_a_inv * target) % (2 ** (nf + 2))

            if sig_b.bit_length() != nf + 1:
                continue

            sign = random.randint(0, 1)
            expA = random.randint(-5, 5)  # Make this more robust
            expB = random.randint(-5, 5)  # Make this more robust
            f1 = generate_float(sign, expA, sig_a ^ (1 << nf), fmt)  # Sign is the same so that the output is positive
            f2 = generate_float(sign, expB, sig_b ^ (1 << nf), fmt)

            tv = generate_test_vector(constants.OP_MUL, f1, f2, 0, fmt, fmt, constants.ROUND_MAX)
            result = run_test_vector(tv)

            pre_rounding_mantissa = int("1" + result.split("_")[-1], 16)
            expected_shift_left = len(bin(sig_a * sig_b)[2:]) == (2 * nf + 2)
            # Mask off 0b, leading one, and significant bits
            rounding_bits = bin(pre_rounding_mantissa)[2 + 1 + nf :]
            # Only get the remaining rounding bits
            rounding_bits = rounding_bits[: nf + expected_shift_left]

            if int(rounding_bits, 2) != target:
                continue

            if expected_shift_left and not hit_with_shift:
                store_cover_vector(result, test_f, cover_f)
                hit_with_shift = True
            elif not expected_shift_left and not hit_without_shift:
                store_cover_vector(result, test_f, cover_f)
                hit_without_shift = True

            if hit_with_shift and hit_without_shift:
                break
        else:
            print(
                f"Failure to generate mul tests :(, fmt={fmt}, extra_bit={extra_bit}, hit_with_shift={hit_with_shift}, "
                f"hit_without_shift={hit_without_shift}"
            )


def two_ones_multiplicands(fmt: str) -> list[int]:
    # I'm not doing all this work every time, trust that it is the best. Work is taken from
    # B15 sparse ones
    if fmt == constants.FMT_QUAD:
        return [
            0b10000010111011001011000100011001000100100111101111101001100110011000011110000010000100000001111010101111010101001,
            0b11111010010010000001000010010110011001000000000000000000000000000000000000011111010010010000001000010010110011001,
        ]
    elif fmt == constants.FMT_DOUBLE:
        return [
            0b111001100100111001100000000000000000110011001001110011,
            0b10100000001001011001001011010000101011100001010111011,
        ]
    elif fmt == constants.FMT_SINGLE:
        return [0b111111111110000000000010, 0b100000000001000000000001]
    elif fmt == constants.FMT_HALF:
        return [0b10110001001, 0b10111001000]
    elif fmt == constants.FMT_BF16:
        return [0b11100010, 0b10010001]
    else:
        raise ValueError("Invalid Format")


def fma_tests(fmt: str, test_f: TextIO, cover_f: TextIO) -> None:
    hashval = reproducible_hash(fmt + "fma" + "b9")
    random.seed(hashval)

    # For FMA, the output is U(Nf + 4).(2Nf + 2)
    # So, (Nf + 4) refers to adding a full significand (Nf + 1) bits then a U2.2nf where the top bit hits the sticky
    # (thus, it is 1 Nf (0 -- guard) 2.2Nf) for a total of Nf + 4 . 2Nf bits
    # Then we get 2Nf + 2 in the other end because the Z addend is squashed into the last sticky bit

    # TODO: The (2nf + 1)th sticky bit!

    # For the purposes of our tests, we want to set all of the possible Nf + 1 extra_bits with the Z shifted
    # into the sticky, Then we want to test sticky calculation with the multiplication result shifted into the
    # right place

    min_exp, max_exp = constants.BIASED_EXP[fmt]
    min_exp -= constants.BIAS[fmt]
    max_exp -= constants.BIAS[fmt]

    for op in [constants.OP_FMADD, constants.OP_FMSUB, constants.OP_FNMADD, constants.OP_FNMSUB]:
        for extra_bit in range(1, constants.MANTISSA_BITS[fmt] + 1):
            for _ in range(100):
                # Logic taken from B3.py
                signA = random.randint(0, 1)
                signB = random.randint(0, 1)

                sigA_initial = random.randint(0, (1 << constants.MANTISSA_BITS[fmt]) - 1)
                sigB_initial = random.randint(0, (1 << constants.MANTISSA_BITS[fmt]) - 1)
                expA = random.randint(-10, 10) + constants.BIAS[fmt]
                expB = random.randint(-10, 10) + constants.BIAS[fmt]

                if fmt == constants.FMT_HALF:
                    # Just be careful that we don't generate things that need
                    # to add a number that we don't have the exponents to add
                    expA = random.randint(-1, 6) + constants.BIAS[fmt]
                    expB = random.randint(-1, 6) + constants.BIAS[fmt]

                # Put in the leading one
                sigA = sigA_initial | (1 << constants.MANTISSA_BITS[fmt])
                sigB = sigB_initial | (1 << constants.MANTISSA_BITS[fmt])

                # Actually Multiply
                sigProd = sigA * sigB
                signProd = signA ^ signB  # zero iff both are the same
                expProd = expA + expB - constants.BIAS[fmt] + 1

                # Correct for the actual operation
                if op == constants.OP_FNMADD or op == constants.OP_FNMSUB:
                    signProd ^= 1  # These ops induce a sign flip

                # Now we ensure that our leading one is in the correct bit, and the
                # product exponent is correct
                if sigProd < (1 << (constants.MANTISSA_BITS[fmt] * 2 + 1)):
                    sigProd <<= 1
                    expProd -= 1

                # The leading one should be in bit MANTISSA_BIT * 2 + 2, so
                # bits MANTISSA_BIT * 2 + 1 --> MANTISSA_BIT + 2 (inclusive) are mantissa
                # Thus, G = MANTISSA_BIT + 1, STICKY = MANTISSA_BIT --> 1
                mask = 2 ** (constants.MANTISSA_BITS[fmt] + 1) - 1
                rounding_bits = sigProd & mask
                sticky_bits = rounding_bits & (mask >> 1)
                _not_sticky = sigProd & (~mask)

                # Sticky bits should be aligned to already, so
                signC = signProd
                target = 1 << (constants.MANTISSA_BITS[fmt] - extra_bit)
                if op == constants.OP_FMADD or op == constants.OP_FNMSUB:
                    # These are effective addition
                    if sticky_bits < target:
                        sigC_initial = target - sticky_bits
                    else:
                        sigC_initial = (target | (1 << constants.MANTISSA_BITS[fmt])) - sticky_bits

                    sigC = sigC_initial | (1 << constants.MANTISSA_BITS[fmt])
                else:
                    # Then its effective subtraction
                    if sticky_bits > target:
                        sigC_initial = sticky_bits - target
                    else:
                        sigC_initial = sticky_bits - target + (1 << constants.MANTISSA_BITS[fmt])

                    sigC = sigC_initial | (1 << constants.MANTISSA_BITS[fmt])

                # Figure out alignment
                expC = expProd - constants.MANTISSA_BITS[fmt] - 1
                expDiff = expProd - expC

                # Align sigC to correct bits of sigProd, the shifts are a no-op but
                # they are there for correctness
                if op == constants.OP_FMADD or op == constants.OP_FNMSUB:
                    sigZ64 = sigProd + ((sigC << (constants.MANTISSA_BITS[fmt] + 1)) >> expDiff)
                else:
                    sigZ64 = sigProd - ((sigC << constants.MANTISSA_BITS[fmt] + 1) >> expDiff)

                # In some cases, especially in lower precision formats (i.e. bf16 and half),
                # we get an "overflow" here (i.e. we move up or down an exponent and have to shift)
                # This means we can accidentally cause a shift of guard into the sticky bit
                # which we do not guarantee to be zero, so we check that here
                if len(bin(sigZ64)) != len(bin(sigProd)):
                    continue

                # Get new rounding info, if we want to log it
                new_rounding = sigZ64 & mask
                _new_sticky = new_rounding & (mask >> 1)

                # Generate the FMA result
                in1 = generate_float(signA, expA - constants.BIAS[fmt], sigA_initial, fmt)
                in2 = generate_float(signB, expB - constants.BIAS[fmt], sigB_initial, fmt)
                in3 = generate_float(signC, expC - constants.BIAS[fmt], sigC_initial, fmt)

                tv = generate_test_vector(op, in1, in2, in3, fmt, fmt, constants.ROUND_MAX)
                result = run_test_vector(tv)

                interm_mantissa = bin(int("1" + result.split("_")[-1], 16))[3:]
                actual_extra_bits = interm_mantissa[constants.MANTISSA_BITS[fmt] :]
                expected_extra_bits = bin(target)[2:].zfill(constants.MANTISSA_BITS[fmt] + 1)

                actual_extra_bits = actual_extra_bits.strip("0")
                expected_extra_bits = expected_extra_bits.strip("0")

                if actual_extra_bits == expected_extra_bits:
                    store_cover_vector(result, test_f, cover_f)
                    break
                if actual_extra_bits[1:] == expected_extra_bits[1:]:
                    print("Failure in FMA Test Generation, failed to create the expected bits")
            else:
                print("Failure to generate a Guard=0 Case in FMA Testgen")
                breakpoint()

        # Now consider the cases where the multiplicand is responsible for sticky bit generation,
        # That is, cases where the addend has an exponent that does not let it cancel the remaining
        # Sticky Bits.
        sigA, sigB = two_ones_multiplicands(fmt)
        last_one = bin(sigA * sigB)[2:].rfind("1")

        # Idea is that the last one starts just barely hanging off of the addend, and then we go until
        # the leading one from the multiplication mantissa is in the lsb
        for exp_diff in range(-last_one + constants.MANTISSA_BITS[fmt] + 1, constants.MANTISSA_BITS[fmt]):
            # Randomized Exponents so that we get the desired exponent difference
            prod_exp = random.randint(max(min_exp, min_exp - exp_diff), min(max_exp, max_exp - exp_diff))
            add_exp = prod_exp + exp_diff

            # Find two exponents that add to prod_exp
            mul_exp1 = random.randint(max(min_exp, prod_exp - max_exp), min(max_exp, prod_exp - min_exp))
            mul_exp2 = prod_exp - mul_exp1

            # Order doesn't matter so randomly swap them
            if random.random() < 0.5:
                mul_exp1, mul_exp2 = mul_exp2, mul_exp1

            if random.random() < 0.5:
                sigA, sigB = sigB, sigA

            sigC = random.getrandbits(constants.MANTISSA_BITS[fmt])
            if exp_diff < 0:
                # Then we have overlap
                mask = (1 << -exp_diff) - 1
                sigC &= ~mask

            signA = random.randint(0, 1)
            signB = signA
            if op == constants.OP_FNMADD or op == constants.OP_FNMSUB:
                signB ^= 1

            signC = 0
            if op == constants.OP_FMSUB or op == constants.OP_FNMADD:
                if exp_diff >= 0:
                    continue  # TODO: Support Effective Subtraction in these cases
                else:
                    pass

            floatA = generate_float(signA, mul_exp1, sigA ^ (1 << constants.MANTISSA_BITS[fmt]), fmt)
            floatB = generate_float(signB, mul_exp2, sigB ^ (1 << constants.MANTISSA_BITS[fmt]), fmt)
            floatC = generate_float(signC, add_exp, sigC, fmt)

            tv = generate_test_vector(op, floatA, floatB, floatC, fmt, fmt, constants.ROUND_MAX)
            result = run_test_vector(tv)

            interm_mantissa = bin(int("1" + result.split("_")[-1], 16))[3:]
            actual_extra_bits = interm_mantissa[constants.MANTISSA_BITS[fmt] :]

            if fmt == "04":
                print(actual_extra_bits)
                breakpoint()


def main() -> None:
    with (
        Path("tests/testvectors/B7_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B7_cv.txt").open("w") as cover_f,
    ):
        for fmt in constants.FLOAT_FMTS:
            add_sub_tests(fmt, test_f, cover_f)
            mul_tests(fmt, test_f, cover_f)
            fma_tests(fmt, test_f, cover_f)


if __name__ == "__main__":
    main()
