// Lean compiler output
// Module: LeanProofs.OrderedField
// Imports: public import Init public meta import Init
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instSub___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instSub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instDiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instDiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instDiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt___redArg();
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instSub___redArg___lam__0(lean_object* v_toNeg_1_, lean_object* v_toAdd_2_, lean_object* v_a_3_, lean_object* v_b_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_apply_1(v_toNeg_1_, v_b_4_);
v___x_6_ = lean_apply_2(v_toAdd_2_, v_a_3_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instSub___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v_toAdd_8_; lean_object* v_toNeg_9_; lean_object* v___f_10_; 
v_toAdd_8_ = lean_ctor_get(v_inst_7_, 2);
lean_inc(v_toAdd_8_);
v_toNeg_9_ = lean_ctor_get(v_inst_7_, 4);
lean_inc(v_toNeg_9_);
lean_dec_ref(v_inst_7_);
v___f_10_ = lean_alloc_closure((void*)(lp_otspot__reduction__proofs_LeanProofs_instSub___redArg___lam__0), 4, 2);
lean_closure_set(v___f_10_, 0, v_toNeg_9_);
lean_closure_set(v___f_10_, 1, v_toAdd_8_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instSub(lean_object* v_F_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_otspot__reduction__proofs_LeanProofs_instSub___redArg(v_inst_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instDiv___redArg___lam__0(lean_object* v_toInv_14_, lean_object* v_toMul_15_, lean_object* v_a_16_, lean_object* v_b_17_){
_start:
{
lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_18_ = lean_apply_1(v_toInv_14_, v_b_17_);
v___x_19_ = lean_apply_2(v_toMul_15_, v_a_16_, v___x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instDiv___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v_toMul_21_; lean_object* v_toInv_22_; lean_object* v___f_23_; 
v_toMul_21_ = lean_ctor_get(v_inst_20_, 3);
lean_inc(v_toMul_21_);
v_toInv_22_ = lean_ctor_get(v_inst_20_, 5);
lean_inc(v_toInv_22_);
lean_dec_ref(v_inst_20_);
v___f_23_ = lean_alloc_closure((void*)(lp_otspot__reduction__proofs_LeanProofs_instDiv___redArg___lam__0), 4, 2);
lean_closure_set(v___f_23_, 0, v_toInv_22_);
lean_closure_set(v___f_23_, 1, v_toMul_21_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instDiv(lean_object* v_F_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_otspot__reduction__proofs_LeanProofs_instDiv___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt___redArg(){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt___redArg___boxed(lean_object* v___dummy_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_otspot__reduction__proofs_LeanProofs_instLt___redArg();
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt(lean_object* v_F_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_box(0);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instLt___boxed(lean_object* v_F_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_otspot__reduction__proofs_LeanProofs_instLt(v_F_34_, v_inst_35_);
lean_dec_ref(v_inst_35_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec___redArg(lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_x_40_){
_start:
{
lean_object* v_zero_41_; uint8_t v_isZero_42_; 
v_zero_41_ = lean_unsigned_to_nat(0u);
v_isZero_42_ = lean_nat_dec_eq(v_x_40_, v_zero_41_);
if (v_isZero_42_ == 1)
{
lean_dec(v_inst_39_);
lean_dec(v_inst_38_);
lean_inc(v_inst_37_);
return v_inst_37_;
}
else
{
lean_object* v_one_43_; lean_object* v_n_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v_one_43_ = lean_unsigned_to_nat(1u);
v_n_44_ = lean_nat_sub(v_x_40_, v_one_43_);
lean_inc(v_inst_39_);
lean_inc(v_inst_38_);
v___x_45_ = lp_otspot__reduction__proofs_LeanProofs_natRec___redArg(v_inst_37_, v_inst_38_, v_inst_39_, v_n_44_);
lean_dec(v_n_44_);
v___x_46_ = lean_apply_2(v_inst_39_, v___x_45_, v_inst_38_);
return v___x_46_;
}
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec___redArg___boxed(lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_x_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_otspot__reduction__proofs_LeanProofs_natRec___redArg(v_inst_47_, v_inst_48_, v_inst_49_, v_x_50_);
lean_dec(v_x_50_);
lean_dec(v_inst_47_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec(lean_object* v_F_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_x_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_otspot__reduction__proofs_LeanProofs_natRec___redArg(v_inst_53_, v_inst_54_, v_inst_55_, v_x_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_natRec___boxed(lean_object* v_F_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_x_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_otspot__reduction__proofs_LeanProofs_natRec(v_F_58_, v_inst_59_, v_inst_60_, v_inst_61_, v_x_62_);
lean_dec(v_x_62_);
lean_dec(v_inst_59_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF___redArg(lean_object* v_z_64_, lean_object* v_o_65_, lean_object* v_a_66_, lean_object* v_n_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_otspot__reduction__proofs_LeanProofs_natRec___redArg(v_z_64_, v_o_65_, v_a_66_, v_n_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF___redArg___boxed(lean_object* v_z_69_, lean_object* v_o_70_, lean_object* v_a_71_, lean_object* v_n_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_otspot__reduction__proofs_LeanProofs_instOfNatOF___redArg(v_z_69_, v_o_70_, v_a_71_, v_n_72_);
lean_dec(v_n_72_);
lean_dec(v_z_69_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF(lean_object* v_F_74_, lean_object* v_z_75_, lean_object* v_o_76_, lean_object* v_a_77_, lean_object* v_n_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_otspot__reduction__proofs_LeanProofs_natRec___redArg(v_z_75_, v_o_76_, v_a_77_, v_n_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOfNatOF___boxed(lean_object* v_F_80_, lean_object* v_z_81_, lean_object* v_o_82_, lean_object* v_a_83_, lean_object* v_n_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_otspot__reduction__proofs_LeanProofs_instOfNatOF(v_F_80_, v_z_81_, v_o_82_, v_a_83_, v_n_84_);
lean_dec(v_n_84_);
lean_dec(v_z_81_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC___redArg(lean_object* v_self_86_){
_start:
{
lean_inc_ref(v_self_86_);
return v_self_86_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC___redArg___boxed(lean_object* v_self_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_otspot__reduction__proofs_LeanProofs_instOFC___redArg(v_self_87_);
lean_dec_ref(v_self_87_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC(lean_object* v_F_89_, lean_object* v_self_90_){
_start:
{
lean_inc_ref(v_self_90_);
return v_self_90_;
}
}
LEAN_EXPORT lean_object* lp_otspot__reduction__proofs_LeanProofs_instOFC___boxed(lean_object* v_F_91_, lean_object* v_self_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_otspot__reduction__proofs_LeanProofs_instOFC(v_F_91_, v_self_92_);
lean_dec_ref(v_self_92_);
return v_res_93_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_otspot__reduction__proofs_LeanProofs_OrderedField(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
