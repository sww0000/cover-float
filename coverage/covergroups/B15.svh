
covergroup B15_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    /************************************************************************
     *`
     * Operation helper coverpoint (f{n}m{add|sub})
     *
     ************************************************************************/

    FP_madd_ops: coverpoint CFI.op {
        type_option.weight = 0;
        bins op_fma    = {[OP_FMA : OP_FMA | 32'hF]};
        bins op_fmadd  = {OP_FMADD};
        bins op_fmsub  = {OP_FMSUB};
        bins op_fnmadd = {OP_FNMADD};
        bins op_fnmsub = {OP_FNMSUB};
    }

    /************************************************************************
     *
     * Operand format helper coverpoints
     *
     ************************************************************************/

    F16_src_fmt: coverpoint (CFI.operandFmt == FMT_HALF) {
        type_option.weight = 0;
        bins f16 = {1};
    }

    BF16_src_fmt: coverpoint (CFI.operandFmt == FMT_BF16) {
        type_option.weight = 0;
        bins bf16 = {1};
    }

    F32_src_fmt: coverpoint (CFI.operandFmt == FMT_SINGLE) {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F64_src_fmt: coverpoint (CFI.operandFmt == FMT_DOUBLE) {
        type_option.weight = 0;
        bins f64 = {1};
    }

    F128_src_fmt: coverpoint (CFI.operandFmt == FMT_QUAD) {
        type_option.weight = 0;
        bins f128 = {1};
    }


    /************************************************************************
     *
     * Special significand helper coverpoints
     *
     ************************************************************************/

    F16_c_special_sigs:  coverpoint (CFI.c[F16_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F16_special_sigs.svh"
    }
    BF16_c_special_sigs: coverpoint (CFI.c[BF16_M_UPPER : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_BF16_special_sigs.svh"
    }
    F32_c_special_sigs:  coverpoint (CFI.c[F32_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F32_special_sigs.svh"
    }
    F64_c_special_sigs:  coverpoint (CFI.c[F64_M_UPPER  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F64_special_sigs.svh"
    }
    F128_c_special_sigs: coverpoint (CFI.c[F128_M_UPPER : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F128_special_sigs.svh"
    }

    // 2.2*NF format for prod significand
    F16_prod_special_sigs:  coverpoint (CFI.fmaPreAddition[2*F16_M_BITS + 1 : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F16_prod_special_sigs.svh"
    }
    BF16_prod_special_sigs: coverpoint (CFI.fmaPreAddition[2*BF16_M_BITS + 1 : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_BF16_prod_special_sigs.svh"
    }
    F32_prod_special_sigs:  coverpoint (CFI.fmaPreAddition[2*F32_M_BITS + 1  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F32_prod_special_sigs.svh"
    }
    F64_prod_special_sigs:  coverpoint (CFI.fmaPreAddition[2*F64_M_BITS + 1  : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F64_prod_special_sigs.svh"
    }
    F128_prod_special_sigs: coverpoint (CFI.fmaPreAddition[2*F128_M_BITS + 1 : 0]) {
        type_option.weight = 0;

        `include "bins_templates/generated/B15_F128_prod_special_sigs.svh"
    }


    /************************************************************************
     *
     * Shift classification helper coverpoints
     *
     * Modeled as:
     *   unbiased_exp(result) - unbiased_exp(addend)
     *
     * This reflects the effective alignment shift applied to the addend
     * during the multiply-add.
     *
     ************************************************************************/

    F16_madd_shift: coverpoint $signed(get_product_exponent(CFI.a, CFI.b, FMT_HALF) - int'(CFI.c[14:10])) {
        type_option.weight = 0;

        bins small_diff = {[ $             : -(2*F16_P + 2)]};
        bins mid_diff[] = {[-(2*F16_P + 1) :  (F16_P + 1)  ]};
        bins large_diff = {[ (F16_P   + 2) :  $            ]};

    }

    BF16_madd_shift: coverpoint $signed(get_product_exponent(CFI.a, CFI.b, FMT_BF16) - int'(CFI.c[14:7])) {
        type_option.weight = 0;

        bins small_diff = {[ $              : -(2*BF16_P + 2)]};
        bins mid_diff[] = {[-(2*BF16_P + 1) :  (BF16_P + 1)  ]};
        bins large_diff = {[ (BF16_P   + 2) :  $             ]};
    }

    F32_madd_shift: coverpoint $signed(get_product_exponent(CFI.a, CFI.b, FMT_SINGLE) - int'(CFI.c[30:23])) {
        type_option.weight = 0;

        bins small_diff = {[ $             : -(2*F32_P + 2)]};
        bins mid_diff[] = {[-(2*F32_P + 1) :  (F32_P + 1)  ]};
        bins large_diff = {[ (F32_P   + 2) :  $            ]};
    }

    F64_madd_shift: coverpoint $signed(get_product_exponent(CFI.a, CFI.b, FMT_DOUBLE) - int'(CFI.c[62:52])) {
        type_option.weight = 0;

        bins small_diff = {[ $             : -(2*F64_P + 2)]};
        bins mid_diff[] = {[-(2*F64_P + 1) :  (F64_P + 1)  ]};
        bins large_diff = {[ (F64_P   + 2) :  $            ]};
    }

    F128_madd_shift: coverpoint $signed(get_product_exponent(CFI.a, CFI.b, FMT_QUAD) - int'(CFI.c[126:112])) {
        type_option.weight = 0;

        bins small_diff = {[ $              : -(2*F128_P + 2)]};
        bins mid_diff[] = {[-(2*F128_P + 1) :  (F128_P + 1)  ]};
        bins large_diff = {[ (F128_P   + 2) :  $             ]};
    }


    /************************************************************************
     *
     * Main crosses (precision-sorted)
     *
     ************************************************************************/

    `ifdef COVER_F16
        B15_F16:  cross FP_addsub_ops, F16_exp_diff,  F16_prod_special_sigs,  F16_c_special_sigs,  F16_result_fmt;
    `endif // COVER_F16

    `ifdef COVER_BF16
        B15_BF16: cross FP_addsub_ops, BF16_exp_diff, BF16_prod_special_sigs, BF16_c_special_sigs, BF16_result_fmt;
    `endif // COVER_BF16

    `ifdef COVER_F32
        B15_F32:  cross FP_addsub_ops, F32_exp_diff,  F32_prod_special_sigs,  F32_c_special_sigs,  F32_result_fmt;
    `endif // COVER_F32

    `ifdef COVER_F64
        B15_F64:  cross FP_addsub_ops, F64_exp_diff,  F64_prod_special_sigs,  F64_c_special_sigs,  F64_result_fmt;
    `endif // COVER_F64

    `ifdef COVER_F128
        B15_F128: cross FP_addsub_ops, F128_exp_diff, F128_prod_special_sigs, F128_c_special_sigs, F128_result_fmt;
    `endif // COVER_F128


endgroup
