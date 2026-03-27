# Created By: cover-float team
#
# B4. Overflow and Near Overflow
#
# From Aharoni et al. 2008:
#
#   This model creates a test-case for each of the following constraints on the
#   intermediate results:
#
#   i.   All the numbers in the range [+MaxNorm - 3ulp, +MaxNorm + 3ulp]
#   ii.  All the numbers in the range [-MaxNorm - 3ulp, -MaxNorm + 3ulp]
#   iii. A random number that is larger than +MaxNorm + 3ulp
#   iv.  A random number that is smaller than -MaxNorm - 3ulp
#   v.   One number for every exponent in the range [MaxNorm.exp - 3, MaxNorm.exp + 3]
#        for positive and negative numbers
#
# ---------------------------------------------------------------------------
# Coverpoint analysis (B4.svh)
# ---------------------------------------------------------------------------
#
# Three coverpoints per format, all crossed with
#   FP_result_ops × rounding_mode_all(5 bins) × sign × result_fmt_gate:
#
# ── 1. F*_maxNorm_pm_3ulp ──────────────────────────────────────────────────
# Samples: intermM[(INTERM_M_BITS - M) -: 3]  — the 3-bit {L, G, S} window
# Guard:   intermX == F*_MAXNORM_EXP   (biased MaxNorm exponent)
#       AND intermM[(INTERM_M_BITS-1) -: (M-1)] == '1   (top M-1 fractional bits)
# Bins:    LGS ∈ {001 .. 111}  (7 bins, 000 excluded)
#
# intermM layout: after stripping the implicit leading-1 and left-aligning,
# bit 191 is the MSB of the fractional part. For a format with M bits:
#   intermM[191 : 192-M]  = the M fractional mantissa bits
#   intermM[192-M]        = L  (LSB of the M-bit mantissa)
#   intermM[191-M]        = G  (first sub-ulp bit — the guard bit)
#   intermM[190-M]        = S  (second sub-ulp bit — the sticky bit)
#
# The coverpoint samples {L, G, S} = intermM[192-M : 190-M].
# The guard requires all of intermM[191 : 193-M] (the top M-1 bits) to be 1.
#
# Construction — what sets {L, G, S}:
#   For FADD(A, B) the infinite-precision intermediate is A + B exactly.
#   B = lgs * 2^(ulp_exp - 3)  (sub-ulp, so the intermediate stays at the same
#   biased exponent as A and doesn't shift the mantissa window).
#
#   The lgs integer's bits map to intermM as follows:
#     lgs bit 2  →  G  (position 191-M)
#     lgs bit 1  →  S  (position 190-M)
#     lgs bit 0  →  one bit BELOW the {L,G,S} window  (not sampled)
#
#   So the sample is {L-from-A, lgs[2], lgs[1]}.
#
#   A = MaxNorm    (mantissa all-ones)  → L = 1  → sample = {1, lgs[2], lgs[1]}
#     lgs=0 → 100,  lgs=2 → 101,  lgs=4 → 110,  lgs=6 → 111
#
#   A = MaxNorm-1ulp (mantissa = all-ones with LSB cleared) → L = 0
#     lgs=2 → 001,  lgs=4 → 010,  lgs=6 → 011
#
#   (lgs=1,3,5,7 give the same sample as lgs=0,2,4,6 respectively because
#    lgs bit 0 falls outside the sample window.)
#
#   Negative intermediates: negate both A and B (same magnitude, sign flipped).
#
# Ops that can precisely control the intermediate:
#   ADD:    FADD(±A,  ±B)              → intermediate = ±(A + B)
#   SUB:    FSUB(±A,  ∓B)             → intermediate = ±(A + B)
#   FMADD:  FMADD(±A, 1.0, ±B)        → intermediate = ±(A*1 + B) = ±(A + B)
#   FMSUB:  FMSUB(±A, 1.0, ∓B)        → intermediate = ±(A*1 - (-B)) = ±(A + B)
#   FNMADD: FNMADD(∓A, 1.0, ∓B)       → intermediate = -(∓A + ∓B) = ±(A + B)
#   FNMSUB: FNMSUB(∓A, 1.0, ±B)       → intermediate = -(∓A - ±B) = ±(A + B)
#
#   MUL/DIV: MUL(A, 1.0) = A exactly → G=0, S=0 → only LGS=100 reachable.
#     Used to cover the op_mul and op_div cross bins for LGS=100.
#   MIN/MAX/FSGNJ*: intermediate = verbatim input → same as MUL; LGS=100 only.
#   CSN: listed in FP_result_op_bins.svh but not implemented by the reference model.
#   SQRT: sqrt(MaxNorm) has exponent ≈ MaxNorm.unbiased/2, never at MaxNorm exponent.
#         op_sqrt × pm_3ulp / gt_maxNorm / exp_range[d≥0] bins are structurally
#         unreachable. The covergroup has no ignore_bins for them.
#   NOTE: op_fma bin ({[OP_FMA:OP_FMA|0xF]}) is satisfied by FMADD (0x51 ∈ [0x50:0x5F])
#         since SV coverpoint bins count each sample independently.
#
# ── 2. F*_gt_maxNorm_p_3ulp ───────────────────────────────────────────────
# Guard:  intermX == F*_MAXNORM_EXP
# Bin:    intermM >= 2^(INTERM_M_BITS - M - 2)   (a single "large significand" bin)
#
# The threshold 2^(192-M-2) is far below MaxNorm's significand (which has all M
# mantissa bits set, giving intermM ≈ 2^191). Every vector that satisfies the
# _pm_3ulp guard also satisfies this bin. No extra vectors needed for this
# coverpoint — it is fully covered by the _pm_3ulp vectors.
#
# ── 3. F*_maxNorm_pm3_exp_range ───────────────────────────────────────────
# Samples: intermX  (biased exponent of the intermediate)
# Bins:    one per exponent in [MAXNORM_EXP-3 : MAXNORM_EXP+3]   (7 individual bins)
#
# For d ∈ {-3,-2,-1,0}  (exponents ≤ MAXNORM_EXP):
#   A = normal FP at biased_exp = MAXNORM_EXP + d, mantissa all-ones.
#   Any op applied to A with a neutral second operand passes A's exponent through.
#   Use ADD(A, 0), MUL(A, 1.0), DIV(A, 1.0), FMADD(A, 1.0, 0), and pass-through
#   ops (MIN, MAX, FSGNJ*) to cover all FP_result_ops bins.
#
# For d ∈ {+1,+2,+3}  (exponents > MAXNORM_EXP, in the overflow range):
#   Use FMUL(±MaxNorm, 2^d).  The infinite-precision product has unbiased exponent
#   MaxNorm.unbiased + d → biased = MAXNORM_EXP + d.  The rounded result overflows
#   to ±Inf, which is the intended overflow-boundary behavior.
#   Also use ADD(MaxNorm, MaxNorm_at_exp+d-1) and FMA variants.

from pathlib import Path
from typing import TextIO

import cover_float.common.constants as const
from cover_float.reference import run_and_store_test_vector

# rounding_mode_all in the covergroup has exactly 5 bins (ROD is absent)
ROUND_MODES = [
    const.ROUND_NEAR_EVEN,
    const.ROUND_MINMAG,
    const.ROUND_MIN,
    const.ROUND_MAX,
    const.ROUND_NEAR_MAXMAG,
]

ZERO_PAD = "0" * 32



# ---------------------------------------------------------------------------
# FP construction helpers
# ---------------------------------------------------------------------------

def _fp_hex(sign: int, biased_exp: int, mantissa: int, E: int, M: int) -> str:
    """Pack (sign, biased_exp, mantissa) into a 32-char left-zero-padded hex string."""
    raw = (sign << (E + M)) | (biased_exp << M) | mantissa
    hex_chars = (1 + E + M + 3) // 4
    return f"{raw:0{hex_chars}x}".rjust(32, "0")


def _maxnorm(sign: int, E: int, M: int) -> str:
    """±MaxNorm: largest finite normal number."""
    return _fp_hex(sign, (1 << E) - 2, (1 << M) - 1, E, M)


def _maxnorm_m1ulp(sign: int, E: int, M: int) -> str:
    """±(MaxNorm - 1ulp): same exponent as MaxNorm, mantissa LSB cleared."""
    return _fp_hex(sign, (1 << E) - 2, ((1 << M) - 1) ^ 1, E, M)


def _one(E: int, M: int, bias: int) -> str:
    """+1.0"""
    return _fp_hex(0, bias, 0, E, M)


def _lgs_b(lgs: int, sign: int, E: int, M: int, bias: int) -> str:
    """
    Return ±lgs * 2^(ulp_exp - 3), the sub-ulp operand that sets the G and S
    bits of the intermediate when added to ±MaxNorm or ±(MaxNorm-1ulp).

    lgs must be even (0, 2, 4, 6) to produce distinct {L,G,S} samples, since
    lgs bit 0 falls outside the 3-bit sample window and is ignored by the
    coverpoint.  Passing an odd lgs is safe but produces the same sample as
    lgs-1.

    lgs == 0 → returns +0.0 (zero regardless of sign).
    """
    if lgs == 0:
        return ZERO_PAD
    max_biased = (1 << E) - 2
    unbiased   = max_biased - bias
    ulp_exp    = unbiased - M
    sub_exp    = ulp_exp - 3           # exponent of 1/8 ulp

    k_bits   = lgs.bit_length() - 1
    b_biased = (sub_exp + k_bits) + bias
    frac     = lgs ^ (1 << k_bits)
    mantissa = frac << (M - k_bits)

    return _fp_hex(sign, b_biased, mantissa, E, M)


def _at_exp(sign: int, biased_exp: int, E: int, M: int) -> str:
    """A normal FP number at the given biased exponent with all-ones mantissa."""
    biased_exp = max(1, min(biased_exp, (1 << E) - 2))
    return _fp_hex(sign, biased_exp, (1 << M) - 1, E, M)


def _scale(d: int, E: int, M: int, bias: int) -> str:
    """+2^d as a FP number (mantissa = 0, biased_exp = d + bias)."""
    biased = max(1, min(d + bias, (1 << E) - 2))
    return _fp_hex(0, biased, 0, E, M)


# ---------------------------------------------------------------------------
# Emit -> writes the test vectors
# ---------------------------------------------------------------------------

def _emit(op: str, rm: str, a: str, b: str, c: str,
          fmt: str, res_fmt: str,
          test_f: TextIO, cover_f: TextIO) -> None:
    tv = f"{op}_{rm}_{a}_{b}_{c}_{fmt}_{ZERO_PAD}_{res_fmt}_00"
    run_and_store_test_vector(tv, test_f, cover_f)


# ---------------------------------------------------------------------------
# Coverpoints 1 & 2: _pm_3ulp and _gt_maxNorm_p_3ulp
# ---------------------------------------------------------------------------

# Map: target LGS sample → (A constructor, lgs_value_for_B)
# {L, G, S} = {A_mantissa_LSB, lgs[2], lgs[1]}
_LGS_CONFIGS = [
    # (use_maxnorm_m1ulp, lgs_for_b)  — target sample shown for clarity
    (True,  2),   # L=0, G=0, S=1  → sample 001
    (True,  4),   # L=0, G=1, S=0  → sample 010
    (True,  6),   # L=0, G=1, S=1  → sample 011
    (False, 0),   # L=1, G=0, S=0  → sample 100  (B=0)
    (False, 2),   # L=1, G=0, S=1  → sample 101
    (False, 4),   # L=1, G=1, S=0  → sample 110
    (False, 6),   # L=1, G=1, S=1  → sample 111
]


def _write_pm_3ulp(fmt: str, E: int, M: int, bias: int,
                   test_f: TextIO, cover_f: TextIO) -> None:
    """
    Covers F*_maxNorm_pm_3ulp (all 7 LGS bins) and implicitly
    F*_gt_maxNorm_p_3ulp (large-significand bin at intermX==MAXNORM_EXP).

    For each LGS target × 5 rounding modes × 2 signs:
      ADD, SUB, FMADD, FMSUB, FNMADD, FNMSUB  — can precisely set intermediate
      MUL, DIV, MIN, MAX, FSGNJ*              — limited to LGS=100 (MUL(A,1.0) etc.)
    """
    one = _one(E, M, bias)

    for use_m1ulp, lgs in _LGS_CONFIGS:
        for sign in (0, 1):
            # Magnitude of A operand
            a_mag  = _maxnorm_m1ulp(0, E, M) if use_m1ulp else _maxnorm(0, E, M)
            a_mag_neg = _maxnorm_m1ulp(1, E, M) if use_m1ulp else _maxnorm(1, E, M)

            # The signed A and B for each polarity
            # Positive intermediate: A positive, B positive
            a_pos = a_mag
            b_pos = _lgs_b(lgs, 0, E, M, bias)

            # Negative intermediate: A negative, B negative
            a_neg = a_mag_neg
            b_neg = _lgs_b(lgs, 1, E, M, bias)

            for rm in ROUND_MODES:
                # ── Positive intermediate ──────────────────────────────────
                # ADD(+A, +B)
                _emit(const.OP_ADD, rm, a_pos, b_pos, ZERO_PAD, fmt, fmt, test_f, cover_f)
                # SUB(+A, -B) = A - (-B) = A + B
                b_neg_for_sub = _lgs_b(lgs, 1, E, M, bias)
                _emit(const.OP_SUB, rm, a_pos, b_neg_for_sub, ZERO_PAD, fmt, fmt, test_f, cover_f)
                # FMADD(+A, 1, +B) = A*1 + B
                _emit(const.OP_FMADD, rm, a_pos, one, b_pos, fmt, fmt, test_f, cover_f)
                # FMSUB(+A, 1, -B) = A*1 - (-B) = A + B
                _emit(const.OP_FMSUB, rm, a_pos, one, b_neg_for_sub, fmt, fmt, test_f, cover_f)
                # FNMADD(-A, 1, -B) = -((-A)*1 + (-B)) = A + B
                _emit(const.OP_FNMADD, rm, a_neg, one, b_neg_for_sub, fmt, fmt, test_f, cover_f)
                # FNMSUB(-A, 1, +B) = -((-A)*1 - B) = A + B
                _emit(const.OP_FNMSUB, rm, a_neg, one, b_pos, fmt, fmt, test_f, cover_f)

                # ── Negative intermediate ──────────────────────────────────
                # ADD(-A, -B)
                _emit(const.OP_ADD, rm, a_neg, b_neg, ZERO_PAD, fmt, fmt, test_f, cover_f)
                # SUB(-A, +B) = -A - B = -(A + B)
                _emit(const.OP_SUB, rm, a_neg, b_pos, ZERO_PAD, fmt, fmt, test_f, cover_f)
                # FMADD(-A, 1, -B) = -A - B
                _emit(const.OP_FMADD, rm, a_neg, one, b_neg, fmt, fmt, test_f, cover_f)
                # FMSUB(-A, 1, +B) = -A - B
                _emit(const.OP_FMSUB, rm, a_neg, one, b_pos, fmt, fmt, test_f, cover_f)
                # FNMADD(+A, 1, +B) = -(A + B)
                _emit(const.OP_FNMADD, rm, a_pos, one, b_pos, fmt, fmt, test_f, cover_f)
                # FNMSUB(+A, 1, -B) = -(A - (-B)) = -(A + B)
                _emit(const.OP_FNMSUB, rm, a_pos, one, b_neg_for_sub, fmt, fmt, test_f, cover_f)

    # ── LGS=100 coverage for remaining FP_result_ops ──────────────────────
    # MUL(A, 1.0), DIV(A, 1.0), MIN(A, A), MAX(A, A), FSGNJ*(A, A)
    # These can only produce LGS=100 (intermediate = A exactly, G=0, S=0)
    # but they DO cover their respective op bins for LGS=100 × all rm × both signs.
    # NOTE: OP_CSN (op_csn bin) is listed in FP_result_op_bins.svh but is not
    # implemented by the reference model and cannot be emitted.
    for sign in (0, 1):
        a = _maxnorm(sign, E, M)
        for rm in ROUND_MODES:
            _emit(const.OP_MUL,    rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_DIV,    rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_MIN,    rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_MAX,    rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_FSGNJ,  rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_FSGNJN, rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_FSGNJX, rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)


# ---------------------------------------------------------------------------
# Coverpoint 3: _maxNorm_pm3_exp_range
# ---------------------------------------------------------------------------

def _write_exp_range(fmt: str, E: int, M: int, bias: int,
                     test_f: TextIO, cover_f: TextIO) -> None:
    """
    Covers F*_maxNorm_pm3_exp_range: one bin per exponent in
    [MAXNORM_EXP-3 : MAXNORM_EXP+3], crossed with FP_result_ops × 5 rm × sign.

    d ≤ 0: A = normal number at biased_exp = MAXNORM_EXP+d (all-ones mantissa).
           Neutral second operand leaves A's exponent as the intermediate exponent.
    d > 0: FMUL(±MaxNorm, 2^d) → infinite-precision product at biased = MAXNORM_EXP+d.
           Result overflows to ±Inf (expected).
    """
    max_biased = (1 << E) - 2
    one = _one(E, M, bias)

    for d in range(-3, 4):
        for sign in (0, 1):
            if d <= 0:
                a = _at_exp(sign, max_biased + d, E, M)
                for rm in ROUND_MODES:
                    # ADD / SUB with zero
                    _emit(const.OP_ADD,    rm, a, ZERO_PAD, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_SUB,    rm, a, ZERO_PAD, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # MUL / DIV by 1
                    _emit(const.OP_MUL,    rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_DIV,    rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # FMA family: A*1 + 0
                    _emit(const.OP_FMADD,  rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FMSUB,  rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FNMADD, rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FNMSUB, rm, a, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # Pass-through ops (intermediate = A, exponent unchanged)
                    _emit(const.OP_MIN,    rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_MAX,    rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FSGNJ,  rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FSGNJN, rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FSGNJX, rm, a, a,         ZERO_PAD, fmt, fmt, test_f, cover_f)
            else:
                # d > 0: intermediate exponent above MaxNorm
                mx  = _maxnorm(sign, E, M)
                sc  = _scale(d, E, M, bias)           # +2^d
                # Large operand at MAXNORM_EXP + d - 1 (same sign) for ADD/SUB
                b_add = _at_exp(sign, max_biased + d - 1, E, M)
                # Opposite-sign version for SUB: SUB(MaxNorm, -B) = MaxNorm + B
                b_sub = _at_exp(sign ^ 1, max_biased + d - 1, E, M)
                for rm in ROUND_MODES:
                    # MUL(MaxNorm, 2^d): product exponent = MAXNORM_EXP + d
                    _emit(const.OP_MUL,    rm, mx, sc,    ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # ADD(MaxNorm, large_at_d-1): sum exponent ≈ MAXNORM_EXP + d
                    _emit(const.OP_ADD,    rm, mx, b_add, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # SUB(MaxNorm, -large_at_d-1) = MaxNorm + large: same exponent as ADD
                    _emit(const.OP_SUB,    rm, mx, b_sub, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # FMADD(MaxNorm, 2^d, 0)
                    _emit(const.OP_FMADD,  rm, mx, sc,    ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # FMSUB(MaxNorm, 2^d, 0)
                    _emit(const.OP_FMSUB,  rm, mx, sc,    ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # FNMADD(-MaxNorm, 2^d, 0) = -(MaxNorm * 2^d + 0) = MaxNorm * 2^d
                    mx_neg = _maxnorm(sign ^ 1, E, M)
                    _emit(const.OP_FNMADD, rm, mx_neg, sc, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FNMSUB, rm, mx_neg, sc, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # DIV(MaxNorm, 2^-d): product exponent = MAXNORM_EXP + d
                    sc_neg = _scale(-d, E, M, bias)    # 2^-d (tiny divisor)
                    _emit(const.OP_DIV,    rm, mx, sc_neg, ZERO_PAD, fmt, fmt, test_f, cover_f)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def _write_all(fmt: str, E: int, M: int, bias: int,
               test_f: TextIO, cover_f: TextIO) -> None:
    _write_pm_3ulp(fmt, E, M, bias, test_f, cover_f)
    _write_exp_range(fmt, E, M, bias, test_f, cover_f)


def main() -> None:
    fmt_params = {
        const.FMT_HALF:   (const.EXPONENT_BITS[const.FMT_HALF],
                           const.MANTISSA_BITS[const.FMT_HALF],
                           const.BIAS[const.FMT_HALF]),
        const.FMT_BF16:   (const.EXPONENT_BITS[const.FMT_BF16],
                           const.MANTISSA_BITS[const.FMT_BF16],
                           const.BIAS[const.FMT_BF16]),
        const.FMT_SINGLE: (const.EXPONENT_BITS[const.FMT_SINGLE],
                           const.MANTISSA_BITS[const.FMT_SINGLE],
                           const.BIAS[const.FMT_SINGLE]),
        const.FMT_DOUBLE: (const.EXPONENT_BITS[const.FMT_DOUBLE],
                           const.MANTISSA_BITS[const.FMT_DOUBLE],
                           const.BIAS[const.FMT_DOUBLE]),
        const.FMT_QUAD:   (const.EXPONENT_BITS[const.FMT_QUAD],
                           const.MANTISSA_BITS[const.FMT_QUAD],
                           const.BIAS[const.FMT_QUAD]),
    }

    Path("tests/testvectors").mkdir(parents=True, exist_ok=True)
    Path("tests/covervectors").mkdir(parents=True, exist_ok=True)

    with (
        Path("tests/testvectors/B4_tv.txt").open("w") as test_f,
        Path("tests/covervectors/B4_cv.txt").open("w") as cover_f
    ):
        test_f.write("// B4: Overflow and Near Overflow\n")
        test_f.write("// Aharoni et al. 2008, Section B4\n")

        for fmt in const.FLOAT_FMTS:
            E, M, bias = fmt_params[fmt]
            _write_all(fmt, E, M, bias, test_f, cover_f)

    total = sum(1 for ln in open("tests/testvectors/B4_tv.txt")
                if not ln.startswith("//"))
    print(f"Generated {total} total B4 test vectors.")


if __name__ == "__main__":
    main()