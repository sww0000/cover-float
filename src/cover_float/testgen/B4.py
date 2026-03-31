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
# ── 2. F*_gt_maxNorm_p_3ulp ────────────────────────────────────────────────
# Samples: the high-significance bits of intermM
# Guard:   intermX == F*_MAXNORM_EXP
# Bins:    "large" - all intermM values where top M-1 bits are all 1
#
# ── 3. F*_maxNorm_pm3_exp_range ────────────────────────────────────────────
# Samples: intermX (biased exponent)
# Bins:    7 consecutive exponent values around MaxNorm.exp
#   [MaxNorm.exp - 3 : MaxNorm.exp + 3]
# = [ F*_MAXNORM_EXP - 3 : F*_MAXNORM_EXP + 3 ]

from enum import auto, Enum
from pathlib import Path
from typing import Optional, TextIO

import cover_float.common.constants as const
from cover_float.reference import run_and_store_test_vector

ZERO_PAD = "0" * 32
ROUND_MODES = [
    const.ROUND_NEAR_EVEN,
    const.ROUND_MINMAG,
    const.ROUND_MIN,
    const.ROUND_MAX,
    const.ROUND_NEAR_MAXMAG,
]


# ---------------------------------------------------------------------------
# Helpers
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
# LGS configurations (covering all 7 bins of F*_maxNorm_pm_3ulp)
# ---------------------------------------------------------------------------
#
# The 3-bit sample is {L, G, S}. We want bins 001..111 (7 bins).
#
# With lgs in {0, 2, 4, 6}, each value produces {L, 0, 0} or {L, g2, g1}:
#   lgs=0 → B=0 → {L, 0, 0} gives {L, 0, 0} (not in our 7 bins, skipped by emit logic)
#   lgs=2 → B=2*2^(-M-3) → {L, 1, 0}
#   lgs=4 → B=4*2^(-M-3) → {L, 0, 1}
#   lgs=6 → B=6*2^(-M-3) → {L, 1, 1}
#
# With use_m1ulp=True, A = MaxNorm-1ulp instead, so L flips:
#   use_m1ulp=True,  lgs=2 → {0, 1, 0}
#   use_m1ulp=True,  lgs=4 → {0, 0, 1}
#   use_m1ulp=True,  lgs=6 → {0, 1, 1}
#
# The full 7 bins {001..111} are hit by the cross with all rounding modes,
# allowing some rounding modes to shift bits.

_LGS_CONFIGS = [
    (True, 2),
    (True, 4),
    (True, 6),
    (False, 2),
    (False, 4),
    (False, 6),
]


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

def _emit(op: str, rm: str, a: str, b: str, c: str, fmt: str, res_fmt: str,
          test_f: TextIO, cover_f: TextIO) -> None:
    tv = f"{op}_{rm}_{a}_{b}_{c}_{fmt}_{ZERO_PAD}_{res_fmt}_00"
    run_and_store_test_vector(tv, test_f, cover_f)


# ---------------------------------------------------------------------------
# Coverpoint 1 & 2: _pm_3ulp and _gt_maxNorm_p_3ulp
# ---------------------------------------------------------------------------

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
            _emit(const.OP_MIN,    rm, a, a,        ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_MAX,    rm, a, a,        ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_FSGNJ,  rm, a, a,        ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_FSGNJN, rm, a, a,        ZERO_PAD, fmt, fmt, test_f, cover_f)
            _emit(const.OP_FSGNJX, rm, a, a,        ZERO_PAD, fmt, fmt, test_f, cover_f)


# ---------------------------------------------------------------------------
# Coverpoint 3: _maxNorm_pm3_exp_range
# ---------------------------------------------------------------------------

def _write_exp_range(fmt: str, E: int, M: int, bias: int,
                     test_f: TextIO, cover_f: TextIO) -> None:
    """
    Covers F*_maxNorm_pm3_exp_range (7 exponent values around MaxNorm.exp).
    
    To reach intermediate exponents up to MaxNorm.exp + 3, we need to add two
    large numbers together (e.g., MaxNorm + MaxNorm has intermediate exp 255,
    which is MaxNorm.exp + 1 for FP32). Simply adding (MaxNorm + 1.0) only
    produces intermediates near MaxNorm.exp.
    """
    max_biased = (1 << E) - 2
    one = _one(E, M, bias)

    # ── First part: intermediate exponents around MaxNorm.exp with (a ± 1.0) ──
    # These cover exponents [MaxNorm.exp - 3, MaxNorm.exp]
    for exp_offset in range(-3, 1):  # -3 to 0 (only up to MaxNorm.exp)
        for sign in (0, 1):
            a_at_exp = _at_exp(sign, max_biased + exp_offset, E, M)
            for rm in ROUND_MODES:
                # ADD, SUB, FMADD, FMSUB, FNMADD, FNMSUB with 1.0
                _emit(const.OP_ADD,    rm, a_at_exp, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_SUB,    rm, a_at_exp, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FMADD,  rm, a_at_exp, one, one, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FMSUB,  rm, a_at_exp, one, one, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMADD, rm, a_at_exp, one, one, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMSUB, rm, a_at_exp, one, one, fmt, fmt, test_f, cover_f)
                # MUL, DIV, MIN, MAX, FSGNJ*
                _emit(const.OP_MUL,    rm, a_at_exp, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_DIV,    rm, a_at_exp, one,      ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_MIN,    rm, a_at_exp, a_at_exp, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_MAX,    rm, a_at_exp, a_at_exp, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FSGNJ,  rm, a_at_exp, a_at_exp, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FSGNJN, rm, a_at_exp, a_at_exp, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FSGNJX, rm, a_at_exp, a_at_exp, ZERO_PAD, fmt, fmt, test_f, cover_f)

    # ── Second part: intermediate exponents > MaxNorm.exp with (large + large) ──
    # To reach intermediate exponents [MaxNorm.exp + 1, +2, +3], we need to add
    # two large numbers OR use multiplication to scale the exponent.
    # Examples:
    #   MaxNorm + MaxNorm -> intermediate exp 255 (for FP32, MaxNorm.exp=254)
    #   MaxNorm * 4 -> intermediate exp 256 (since 4 = 2^2, exponent +2)
    #   MaxNorm * 8 -> intermediate exp 257 (since 8 = 2^3, exponent +3)
    
    a_maxnorm = _maxnorm(0, E, M)
    a_maxnorm_neg = _maxnorm(1, E, M)
    
    for exp_offset in range(1, 4):  # +1 to +3
        for sign in (0, 1):
            a_base = a_maxnorm_neg if sign == 1 else a_maxnorm
            b_neg_base = a_maxnorm if sign == 1 else a_maxnorm_neg
            
            for rm in ROUND_MODES:
                # Strategy 1: Addition-based (for exp_offset=1, MaxNorm + MaxNorm)
                if exp_offset == 1:
                    # ADD: MaxNorm + MaxNorm -> intermediate exp 255
                    _emit(const.OP_ADD, rm, a_base, a_base, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # SUB: MaxNorm - (-MaxNorm) = MaxNorm + MaxNorm
                    _emit(const.OP_SUB, rm, a_base, b_neg_base, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # FMADD: MaxNorm + MaxNorm*1
                    _emit(const.OP_FMADD, rm, a_base, one, a_base, fmt, fmt, test_f, cover_f)
                    # FMSUB: MaxNorm - (-MaxNorm)*1 = MaxNorm + MaxNorm
                    _emit(const.OP_FMSUB, rm, a_base, one, b_neg_base, fmt, fmt, test_f, cover_f)
                    # FNMADD: -((-MaxNorm) + MaxNorm*1) = MaxNorm + MaxNorm
                    _emit(const.OP_FNMADD, rm, a_base, one, b_neg_base, fmt, fmt, test_f, cover_f)
                    # FNMSUB: -(MaxNorm - (-MaxNorm)*1) = -(MaxNorm + MaxNorm)
                    _emit(const.OP_FNMSUB, rm, a_base, one, b_neg_base, fmt, fmt, test_f, cover_f)
                
                # Strategy 2: Multiplication-based (for exp_offset > 1)
                # To reach higher exponents, multiply MaxNorm by 2^exp_offset
                # exp_offset=1: 2^1=2, MaxNorm * 2 -> exp 255
                # exp_offset=2: 2^2=4, MaxNorm * 4 -> exp 256
                # exp_offset=3: 2^3=8, MaxNorm * 8 -> exp 257
                scale_factor = _scale(exp_offset, E, M, bias)
                
                # MUL: MaxNorm * 2^exp_offset -> intermediate at higher exponent
                _emit(const.OP_MUL, rm, a_base, scale_factor, ZERO_PAD, fmt, fmt, test_f, cover_f)
                
                # DIV: MaxNorm / 2^(-exp_offset) = MaxNorm * 2^exp_offset
                scale_inv = _scale(-exp_offset, E, M, bias)
                _emit(const.OP_DIV, rm, a_base, scale_inv, ZERO_PAD, fmt, fmt, test_f, cover_f)
                
                # FMADD: MaxNorm*scale + 1 (where scale increases exponent)
                _emit(const.OP_FMADD, rm, a_base, scale_factor, one, fmt, fmt, test_f, cover_f)
                
                # FMSUB: MaxNorm*scale - (-1)
                _emit(const.OP_FMSUB, rm, a_base, scale_factor, b_neg_base, fmt, fmt, test_f, cover_f)
                
                # FNMADD: -((-MaxNorm)*scale + 1)
                _emit(const.OP_FNMADD, rm, b_neg_base, scale_factor, one, fmt, fmt, test_f, cover_f)
                
                # FNMSUB: -(MaxNorm*scale - (-1))
                _emit(const.OP_FNMSUB, rm, a_base, scale_factor, b_neg_base, fmt, fmt, test_f, cover_f)
    
    # ── Third part: Ensure all operation-sign-rounding combinations are covered ──
    # For lower exponents, explicitly cover all combinations to avoid missing bins
    for exp_offset in range(-3, 1):  # -3 to 0 (exponents 251-254 for FP32)
        for sign in (0, 1):
            a_at_exp = _at_exp(sign, max_biased + exp_offset, E, M)
            a_opp_sign = _at_exp(1 - sign, max_biased + exp_offset, E, M)
            
            for rm in ROUND_MODES:
                # Generate all operations explicitly for this (exp, sign, rm) combination
                _emit(const.OP_ADD, rm, a_at_exp, one, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_SUB, rm, a_at_exp, a_opp_sign, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FMADD, rm, a_at_exp, one, one, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FMSUB, rm, a_at_exp, one, a_opp_sign, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMADD, rm, a_opp_sign, one, a_opp_sign, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMSUB, rm, a_at_exp, one, a_opp_sign, fmt, fmt, test_f, cover_f)
                _emit(const.OP_MUL, rm, a_at_exp, one, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_DIV, rm, a_at_exp, one, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_MIN, rm, a_at_exp, one, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_MAX, rm, a_at_exp, one, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FSGNJ, rm, a_at_exp, a_opp_sign, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FSGNJN, rm, a_at_exp, a_opp_sign, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FSGNJX, rm, a_at_exp, a_opp_sign, ZERO_PAD, fmt, fmt, test_f, cover_f)

    # ── Part 4: Rare edge cases - specific rounding mode + sign combinations ──
    # These combinations are not guaranteed by Parts 1-3. They test the
    # unusual scenarios where rounding direction conflicts with operand sign:
    #
    # - round_max + negative sign: "round toward +∞" applied to negative result
    # - round_min + positive sign: "round toward -∞" applied to positive result
    # - round_minmag at boundaries: tests rounding magnitude truncation
    #
    # These are mathematically valid but rare in practice.
    
    a_maxnorm = _maxnorm(0, E, M)
    a_maxnorm_neg = _maxnorm(1, E, M)
    
    for exp_offset in range(1, 4):  # exp_range [255, 256, 257] for FP32
        scale_factor = _scale(exp_offset, E, M, bias)
        scale_inv = _scale(-exp_offset, E, M, bias)
        
        for rm in [const.ROUND_MAX, const.ROUND_MIN, const.ROUND_MINMAG]:
            # ── Case 1: round_max with NEGATIVE intermediate ──
            if rm == const.ROUND_MAX:
                # Negative base (sign=1): round_max applies to negative number
                _emit(const.OP_ADD, rm, a_maxnorm_neg, a_maxnorm_neg, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_SUB, rm, a_maxnorm_neg, a_maxnorm, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_MUL, rm, a_maxnorm_neg, scale_factor, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_DIV, rm, a_maxnorm_neg, scale_inv, ZERO_PAD, fmt, fmt, test_f, cover_f)
                # Also test FMADD/FMSUB/FNMADD/FNMSUB with scaling for high exponents
                _emit(const.OP_FMADD, rm, a_maxnorm_neg, scale_factor, a_maxnorm_neg, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FMSUB, rm, a_maxnorm_neg, scale_factor, a_maxnorm, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMADD, rm, a_maxnorm, scale_factor, a_maxnorm, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMSUB, rm, a_maxnorm_neg, scale_factor, a_maxnorm, fmt, fmt, test_f, cover_f)
            
            # ── Case 2: round_min with POSITIVE intermediate ──
            elif rm == const.ROUND_MIN:
                # Positive base (sign=0): round_min applies to positive number
                _emit(const.OP_ADD, rm, a_maxnorm, a_maxnorm, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_SUB, rm, a_maxnorm, a_maxnorm_neg, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_MUL, rm, a_maxnorm, scale_factor, ZERO_PAD, fmt, fmt, test_f, cover_f)
                _emit(const.OP_DIV, rm, a_maxnorm, scale_inv, ZERO_PAD, fmt, fmt, test_f, cover_f)
                # Also test FMADD/FMSUB/FNMADD/FNMSUB with scaling
                _emit(const.OP_FMADD, rm, a_maxnorm, scale_factor, a_maxnorm, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FMSUB, rm, a_maxnorm, scale_factor, a_maxnorm_neg, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMADD, rm, a_maxnorm_neg, scale_factor, a_maxnorm_neg, fmt, fmt, test_f, cover_f)
                _emit(const.OP_FNMSUB, rm, a_maxnorm, scale_factor, a_maxnorm_neg, fmt, fmt, test_f, cover_f)
            
            # ── Case 3: round_minmag at boundary exponents ──
            elif rm == const.ROUND_MINMAG:
                # Test with both positive and negative signs
                for sign in (0, 1):
                    a_base = a_maxnorm_neg if sign == 1 else a_maxnorm
                    b_base = a_maxnorm if sign == 1 else a_maxnorm_neg
                    
                    _emit(const.OP_ADD, rm, a_base, a_base, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_SUB, rm, a_base, b_base, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_MUL, rm, a_base, scale_factor, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_DIV, rm, a_base, scale_inv, ZERO_PAD, fmt, fmt, test_f, cover_f)
                    # FMADD/FMSUB with scaling
                    _emit(const.OP_FMADD, rm, a_base, scale_factor, a_base, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FMSUB, rm, a_base, scale_factor, b_base, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FNMADD, rm, b_base, scale_factor, b_base, fmt, fmt, test_f, cover_f)
                    _emit(const.OP_FNMSUB, rm, a_base, scale_factor, b_base, fmt, fmt, test_f, cover_f)


# ---------------------------------------------------------------------------
# Combined generation
# ---------------------------------------------------------------------------

def _write_all(fmt: str, E: int, M: int, bias: int,
               test_f: TextIO, cover_f: TextIO) -> None:
    _write_pm_3ulp(fmt, E, M, bias, test_f, cover_f)
    _write_exp_range(fmt, E, M, bias, test_f, cover_f)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

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
