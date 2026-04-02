# B19 - Compare: Different Input Fields Relations
# Aharoni et al.
#
# This model checks various possible differences between two inputs for
# comparison and min/max operations. Test cases are generated for each combination
# of exponent and significand relationship constraints.
#
# Total vectors: 8200 (328 per format×op, across 5 formats and 5 operations)

from pathlib import Path
from typing import NamedTuple, TextIO

import cover_float.common.constants as const
from cover_float.reference import run_and_store_test_vector


# Operations to test
OPS = [const.OP_FEQ, const.OP_FLT, const.OP_FLE, const.OP_MIN, const.OP_MAX]

# Exponent anchor values per format (4 evenly-spaced representatives per format)
EXP_ANCHORS = {
    const.FMT_BF16: [1, 85, 169, 254],
    const.FMT_HALF: [1, 10, 19, 30],
    const.FMT_SINGLE: [1, 85, 169, 254],
    const.FMT_DOUBLE: [1, 682, 1363, 2046],
    const.FMT_QUAD: [1, 10922, 21843, 32766],
}

# Exponent anchor values for exp> cases (restricted to be < max_norm_bexp)
EXP_ANCHORS_LT_MAX = {
    const.FMT_BF16: [1, 85, 169, 253],
    const.FMT_HALF: [1, 10, 19, 29],
    const.FMT_SINGLE: [1, 85, 169, 253],
    const.FMT_DOUBLE: [1, 682, 1363, 2045],
    const.FMT_QUAD: [1, 10922, 21843, 32765],
}

# Exponent anchor values for exp< cases (restricted to be > min_norm_bexp)
EXP_ANCHORS_GT_MIN = {
    const.FMT_BF16: [2, 86, 170, 254],
    const.FMT_HALF: [2, 11, 20, 30],
    const.FMT_SINGLE: [2, 86, 170, 254],
    const.FMT_DOUBLE: [2, 683, 1364, 2046],
    const.FMT_QUAD: [2, 10923, 21844, 32766],
}


def _decimalComponentsToHex(fmt, sign, biased_exp, mantissa):
    """Convert sign, biased exponent, and mantissa to 32-bit hex string."""
    b_sign = f"{sign:01b}"
    b_exp = f"{biased_exp:0{const.EXPONENT_BITS[fmt]}b}"
    b_man = f"{mantissa:0{const.MANTISSA_BITS[fmt]}b}"
    b_complete = b_sign + b_exp + b_man
    h_complete = f"{int(b_complete, 2):032X}"
    return h_complete


def _tile_0x5a_pattern(bit_width):
    """Create a tiled 0x5A pattern to fill the given bit width."""
    pattern = 0
    for i in range(0, bit_width, 8):
        pattern |= 0x5A << i
    return pattern & ((1 << bit_width) - 1)


def _get_frac_values(fmt):
    """Get frac field constants for the format."""
    frac_bits = const.MANTISSA_BITS[fmt]
    return {
        "0s": 0,
        "1s": (1 << frac_bits) - 1,
        "00001": 1,
        "10000": 1 << (frac_bits - 1),
        "any": _tile_0x5a_pattern(frac_bits),
    }


def _case_1_normal_x_normal(fmt, test_f, cover_f, op):
    """Case 1: Normal x Normal (144 vectors per format)."""
    count = 0
    fracs = _get_frac_values(fmt)
    max_norm_bexp = (1 << const.EXPONENT_BITS[fmt]) - 1 - 1

    # Exp > cells (3 sig_rel variants)
    for sig_rel, frac1, frac2 in [
        ("gt", fracs["any"], fracs["0s"]),
        ("<", fracs["0s"], fracs["any"]),
        ("=", fracs["0s"], fracs["0s"]),
    ]:
        for bexp2 in EXP_ANCHORS_LT_MAX[fmt]:
            for sign1 in [0, 1]:
                for sign2 in [0, 1]:
                    hex1 = _decimalComponentsToHex(fmt, sign1, max_norm_bexp, frac1)
                    hex2 = _decimalComponentsToHex(fmt, sign2, bexp2, frac2)
                    run_and_store_test_vector(
                        f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f,
                        cover_f,
                    )
                    count += 1

    # Exp < cells (3 sig_rel variants)
    for sig_rel, frac1, frac2 in [
        ("gt", fracs["1s"], fracs["any"]),
        ("<", fracs["any"], fracs["1s"]),
        ("=", fracs["1s"], fracs["1s"]),
    ]:
        for bexp2 in EXP_ANCHORS_GT_MIN[fmt]:
            for sign1 in [0, 1]:
                for sign2 in [0, 1]:
                    hex1 = _decimalComponentsToHex(fmt, sign1, 1, frac1)
                    hex2 = _decimalComponentsToHex(fmt, sign2, bexp2, frac2)
                    run_and_store_test_vector(
                        f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f,
                        cover_f,
                    )
                    count += 1

    # Exp = cells (3 sig_rel variants, same bexp for both operands)
    for sig_rel, frac1, frac2 in [
        ("gt", fracs["1s"], fracs["0s"]),
        ("<", fracs["0s"], fracs["1s"]),
        ("=", fracs["0s"], fracs["0s"]),
    ]:
        for common_bexp in EXP_ANCHORS[fmt]:
            for sign1 in [0, 1]:
                for sign2 in [0, 1]:
                    hex1 = _decimalComponentsToHex(fmt, sign1, common_bexp, frac1)
                    hex2 = _decimalComponentsToHex(fmt, sign2, common_bexp, frac2)
                    run_and_store_test_vector(
                        f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f,
                        cover_f,
                    )
                    count += 1

    return count


def _case_2_normal_x_subnormal(fmt, test_f, cover_f, op):
    """Case 2: Normal x SubNormal (48 vectors per format)."""
    count = 0
    fracs = _get_frac_values(fmt)

    for sig_rel, frac1, frac2 in [
        ("gt", fracs["any"], fracs["00001"]),
        ("<", fracs["0s"], fracs["00001"]),
        ("=", fracs["any"], fracs["any"]),
    ]:
        for bexp1 in EXP_ANCHORS[fmt]:
            for sign1 in [0, 1]:
                for sign2 in [0, 1]:
                    hex1 = _decimalComponentsToHex(fmt, sign1, bexp1, frac1)
                    hex2 = _decimalComponentsToHex(fmt, sign2, 0, frac2)
                    run_and_store_test_vector(
                        f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f,
                        cover_f,
                    )
                    count += 1

    return count


def _case_3_subnormal_x_normal(fmt, test_f, cover_f, op):
    """Case 3: SubNormal x Normal (48 vectors per format)."""
    count = 0
    fracs = _get_frac_values(fmt)

    for sig_rel, frac1, frac2 in [
        ("gt", fracs["1s"], fracs["0s"]),
        ("<", fracs["any"], fracs["1s"]),
        ("=", fracs["any"], fracs["any"]),
    ]:
        for bexp2 in EXP_ANCHORS[fmt]:
            for sign1 in [0, 1]:
                for sign2 in [0, 1]:
                    hex1 = _decimalComponentsToHex(fmt, sign1, 0, frac1)
                    hex2 = _decimalComponentsToHex(fmt, sign2, bexp2, frac2)
                    run_and_store_test_vector(
                        f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f,
                        cover_f,
                    )
                    count += 1

    return count


def _case_4_subnormal_x_subnormal(fmt, test_f, cover_f, op):
    """Case 4: SubNormal x SubNormal (20 vectors per format)."""
    count = 0
    fracs = _get_frac_values(fmt)

    for frac1, frac2 in [
        (fracs["any"], fracs["00001"]),
        (fracs["00001"], fracs["any"]),
        (fracs["1s"], fracs["10000"]),
        (fracs["10000"], fracs["1s"]),
        (fracs["any"], fracs["any"]),
    ]:
        for sign1 in [0, 1]:
            for sign2 in [0, 1]:
                hex1 = _decimalComponentsToHex(fmt, sign1, 0, frac1)
                hex2 = _decimalComponentsToHex(fmt, sign2, 0, frac2)
                run_and_store_test_vector(
                    f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                    test_f,
                    cover_f,
                )
                count += 1

    return count


def _case_5_zero_x_normal(fmt, test_f, cover_f, op):
    """Case 5: Zero x Normal (32 vectors per format)."""
    count = 0
    fracs = _get_frac_values(fmt)

    for frac2 in [fracs["any"], fracs["0s"]]:
        for bexp2 in EXP_ANCHORS[fmt]:
            for sign1 in [0, 1]:
                for sign2 in [0, 1]:
                    hex1 = _decimalComponentsToHex(fmt, sign1, 0, fracs["0s"])
                    hex2 = _decimalComponentsToHex(fmt, sign2, bexp2, frac2)
                    run_and_store_test_vector(
                        f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f,
                        cover_f,
                    )
                    count += 1

    return count


def _case_7_normal_x_zero(fmt, test_f, cover_f, op):
    """Case 7: Normal x Zero (32 vectors per format)."""
    count = 0
    fracs = _get_frac_values(fmt)

    for frac1 in [fracs["any"], fracs["0s"]]:
        for bexp1 in EXP_ANCHORS[fmt]:
            for sign1 in [0, 1]:
                for sign2 in [0, 1]:
                    hex1 = _decimalComponentsToHex(fmt, sign1, bexp1, frac1)
                    hex2 = _decimalComponentsToHex(fmt, sign2, 0, fracs["0s"])
                    run_and_store_test_vector(
                        f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                        test_f,
                        cover_f,
                    )
                    count += 1

    return count


def _case_9_zero_x_zero(fmt, test_f, cover_f, op):
    """Case 9: Zero x Zero (4 vectors per format)."""
    count = 0
    fracs = _get_frac_values(fmt)

    for sign1 in [0, 1]:
        for sign2 in [0, 1]:
            hex1 = _decimalComponentsToHex(fmt, sign1, 0, fracs["0s"])
            hex2 = _decimalComponentsToHex(fmt, sign2, 0, fracs["0s"])
            run_and_store_test_vector(
                f"{op}_{const.ROUND_NEAR_EVEN}_{hex1}_{hex2}_{32 * '0'}_{fmt}_{32 * '0'}_{fmt}_00",
                test_f,
                cover_f,
            )
            count += 1

    return count


def generate_b19_tests(test_f, cover_f, fmt):
    """Generate all B19 test vectors for a given format."""
    for op in OPS:
        case1 = _case_1_normal_x_normal(fmt, test_f, cover_f, op)
        case2 = _case_2_normal_x_subnormal(fmt, test_f, cover_f, op)
        case3 = _case_3_subnormal_x_normal(fmt, test_f, cover_f, op)
        case4 = _case_4_subnormal_x_subnormal(fmt, test_f, cover_f, op)
        case5 = _case_5_zero_x_normal(fmt, test_f, cover_f, op)
        case7 = _case_7_normal_x_zero(fmt, test_f, cover_f, op)
        case9 = _case_9_zero_x_zero(fmt, test_f, cover_f, op)

        total = case1 + case2 + case3 + case4 + case5 + case7 + case9

        print(f"  {op}: case1={case1}, case2={case2}, case3={case3}, case4={case4}, "
              f"case5={case5}, case7={case7}, case9={case9}, total={total}")


def main():
    """Main entry point for B19 test generation."""
    print("Generating B19 test vectors...")
    with (
        Path("./tests/testvectors/B19_tv.txt").open("w") as test_f,
        Path("./tests/covervectors/B19_cv.txt").open("w") as cover_f,
    ):
        for fmt in const.FLOAT_FMTS:
            print(f"Format: {fmt}")
            generate_b19_tests(test_f, cover_f, fmt)
    print("Done!")


if __name__ == "__main__":
    main()
