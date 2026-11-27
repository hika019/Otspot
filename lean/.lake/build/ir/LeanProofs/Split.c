// Lean compiler output
// Module: LeanProofs.Split
// Imports: public import Init public meta import Init public import LeanProofs.OrderedField
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_vzeros___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_vzeros(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_dot___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_dot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_objOf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_objOf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_vzeros___redArg(lean_object* v_inst_1_, lean_object* v_n_2_){
_start:
{
lean_object* v_toZero_3_; lean_object* v___x_4_; 
v_toZero_3_ = lean_ctor_get(v_inst_1_, 0);
lean_inc(v_toZero_3_);
lean_dec_ref(v_inst_1_);
v___x_4_ = l_List_replicateTR___redArg(v_n_2_, v_toZero_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_vzeros(lean_object* v_F_5_, lean_object* v_inst_6_, lean_object* v_n_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_otspot__reduction__proofs_LeanProofs_vzeros___redArg(v_inst_6_, v_n_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_dot___redArg(lean_object* v_inst_9_, lean_object* v_x_10_, lean_object* v_x_11_){
_start:
{
if (lean_obj_tag(v_x_10_) == 0)
{
lean_object* v_toZero_12_; 
lean_dec(v_x_11_);
v_toZero_12_ = lean_ctor_get(v_inst_9_, 0);
lean_inc(v_toZero_12_);
lean_dec_ref(v_inst_9_);
return v_toZero_12_;
}
else
{
if (lean_obj_tag(v_x_11_) == 0)
{
lean_object* v_toZero_13_; 
lean_dec_ref_known(v_x_10_, 2);
v_toZero_13_ = lean_ctor_get(v_inst_9_, 0);
lean_inc(v_toZero_13_);
lean_dec_ref(v_inst_9_);
return v_toZero_13_;
}
else
{
lean_object* v_toAdd_14_; lean_object* v_toMul_15_; lean_object* v_head_16_; lean_object* v_tail_17_; lean_object* v_head_18_; lean_object* v_tail_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v_toAdd_14_ = lean_ctor_get(v_inst_9_, 2);
lean_inc(v_toAdd_14_);
v_toMul_15_ = lean_ctor_get(v_inst_9_, 3);
v_head_16_ = lean_ctor_get(v_x_10_, 0);
lean_inc(v_head_16_);
v_tail_17_ = lean_ctor_get(v_x_10_, 1);
lean_inc(v_tail_17_);
lean_dec_ref_known(v_x_10_, 2);
v_head_18_ = lean_ctor_get(v_x_11_, 0);
lean_inc(v_head_18_);
v_tail_19_ = lean_ctor_get(v_x_11_, 1);
lean_inc(v_tail_19_);
lean_dec_ref_known(v_x_11_, 2);
lean_inc(v_toMul_15_);
v___x_20_ = lean_apply_2(v_toMul_15_, v_head_16_, v_head_18_);
v___x_21_ = lp_otspot__reduction__proofs_LeanProofs_dot___redArg(v_inst_9_, v_tail_17_, v_tail_19_);
v___x_22_ = lean_apply_2(v_toAdd_14_, v___x_20_, v___x_21_);
return v___x_22_;
}
}
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_dot(lean_object* v_F_23_, lean_object* v_inst_24_, lean_object* v_x_25_, lean_object* v_x_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_otspot__reduction__proofs_LeanProofs_dot___redArg(v_inst_24_, v_x_25_, v_x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_objOf___redArg(lean_object* v_inst_28_, lean_object* v_P_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_c_31_; lean_object* v___x_32_; 
v_c_31_ = lean_ctor_get(v_P_29_, 2);
lean_inc(v_c_31_);
lean_dec_ref(v_P_29_);
v___x_32_ = lp_otspot__reduction__proofs_LeanProofs_dot___redArg(v_inst_28_, v_c_31_, v_x_30_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_objOf(lean_object* v_F_33_, lean_object* v_inst_34_, lean_object* v_P_35_, lean_object* v_x_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_otspot__reduction__proofs_LeanProofs_objOf___redArg(v_inst_34_, v_P_35_, v_x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__0(lean_object* v_c_38_, lean_object* v_inst_39_, lean_object* v_r_40_){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = l_List_lengthTR___redArg(v_c_38_);
v___x_42_ = lp_otspot__reduction__proofs_LeanProofs_vzeros___redArg(v_inst_39_, v___x_41_);
v___x_43_ = l_List_appendTR___redArg(v_r_40_, v___x_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__0___boxed(lean_object* v_c_44_, lean_object* v_inst_45_, lean_object* v_r_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__0(v_c_44_, v_inst_45_, v_r_46_);
lean_dec(v_c_44_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__1(lean_object* v_c_48_, lean_object* v_inst_49_, lean_object* v_r_50_){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_51_ = l_List_lengthTR___redArg(v_c_48_);
v___x_52_ = lp_otspot__reduction__proofs_LeanProofs_vzeros___redArg(v_inst_49_, v___x_51_);
v___x_53_ = l_List_appendTR___redArg(v___x_52_, v_r_50_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__1___boxed(lean_object* v_c_54_, lean_object* v_inst_55_, lean_object* v_r_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__1(v_c_54_, v_inst_55_, v_r_56_);
lean_dec(v_c_54_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine___redArg(lean_object* v_inst_58_, lean_object* v_P_u2081_59_, lean_object* v_P_u2082_60_){
_start:
{
lean_object* v_A_61_; lean_object* v_b_62_; lean_object* v_c_63_; lean_object* v_lb_64_; lean_object* v_ub_65_; lean_object* v_A_66_; lean_object* v_b_67_; lean_object* v_c_68_; lean_object* v_lb_69_; lean_object* v_ub_70_; lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_87_; 
v_A_61_ = lean_ctor_get(v_P_u2081_59_, 0);
lean_inc(v_A_61_);
v_b_62_ = lean_ctor_get(v_P_u2081_59_, 1);
lean_inc(v_b_62_);
v_c_63_ = lean_ctor_get(v_P_u2081_59_, 2);
lean_inc(v_c_63_);
v_lb_64_ = lean_ctor_get(v_P_u2081_59_, 3);
lean_inc(v_lb_64_);
v_ub_65_ = lean_ctor_get(v_P_u2081_59_, 4);
lean_inc(v_ub_65_);
lean_dec_ref(v_P_u2081_59_);
v_A_66_ = lean_ctor_get(v_P_u2082_60_, 0);
v_b_67_ = lean_ctor_get(v_P_u2082_60_, 1);
v_c_68_ = lean_ctor_get(v_P_u2082_60_, 2);
v_lb_69_ = lean_ctor_get(v_P_u2082_60_, 3);
v_ub_70_ = lean_ctor_get(v_P_u2082_60_, 4);
v_isSharedCheck_87_ = !lean_is_exclusive(v_P_u2082_60_);
if (v_isSharedCheck_87_ == 0)
{
v___x_72_ = v_P_u2082_60_;
v_isShared_73_ = v_isSharedCheck_87_;
goto v_resetjp_71_;
}
else
{
lean_inc(v_ub_70_);
lean_inc(v_lb_69_);
lean_inc(v_c_68_);
lean_inc(v_b_67_);
lean_inc(v_A_66_);
lean_dec(v_P_u2082_60_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_87_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___f_74_; lean_object* v___f_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_85_; 
lean_inc_ref(v_inst_58_);
lean_inc(v_c_68_);
v___f_74_ = lean_alloc_closure((void*)(lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_74_, 0, v_c_68_);
lean_closure_set(v___f_74_, 1, v_inst_58_);
lean_inc(v_c_63_);
v___f_75_ = lean_alloc_closure((void*)(lp_otspot__reduction__proofs_LeanProofs_combine___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_75_, 0, v_c_63_);
lean_closure_set(v___f_75_, 1, v_inst_58_);
v___x_76_ = lean_box(0);
v___x_77_ = l_List_mapTR_loop___redArg(v___f_74_, v_A_61_, v___x_76_);
v___x_78_ = l_List_mapTR_loop___redArg(v___f_75_, v_A_66_, v___x_76_);
v___x_79_ = l_List_appendTR___redArg(v___x_77_, v___x_78_);
v___x_80_ = l_List_appendTR___redArg(v_b_62_, v_b_67_);
v___x_81_ = l_List_appendTR___redArg(v_c_63_, v_c_68_);
v___x_82_ = l_List_appendTR___redArg(v_lb_64_, v_lb_69_);
v___x_83_ = l_List_appendTR___redArg(v_ub_65_, v_ub_70_);
if (v_isShared_73_ == 0)
{
lean_ctor_set(v___x_72_, 4, v___x_83_);
lean_ctor_set(v___x_72_, 3, v___x_82_);
lean_ctor_set(v___x_72_, 2, v___x_81_);
lean_ctor_set(v___x_72_, 1, v___x_80_);
lean_ctor_set(v___x_72_, 0, v___x_79_);
v___x_85_ = v___x_72_;
goto v_reusejp_84_;
}
else
{
lean_object* v_reuseFailAlloc_86_; 
v_reuseFailAlloc_86_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_86_, 0, v___x_79_);
lean_ctor_set(v_reuseFailAlloc_86_, 1, v___x_80_);
lean_ctor_set(v_reuseFailAlloc_86_, 2, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_86_, 3, v___x_82_);
lean_ctor_set(v_reuseFailAlloc_86_, 4, v___x_83_);
v___x_85_ = v_reuseFailAlloc_86_;
goto v_reusejp_84_;
}
v_reusejp_84_:
{
return v___x_85_;
}
}
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_combine(lean_object* v_F_88_, lean_object* v_inst_89_, lean_object* v_P_u2081_90_, lean_object* v_P_u2082_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_otspot__reduction__proofs_LeanProofs_combine___redArg(v_inst_89_, v_P_u2081_90_, v_P_u2082_91_);
return v___x_92_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_otspot__reduction__proofs_LeanProofs_OrderedField(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_otspot__reduction__proofs_LeanProofs_Split(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_otspot__reduction__proofs_LeanProofs_OrderedField(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
