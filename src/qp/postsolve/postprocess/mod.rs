//! 全体 postprocess loop と LSQ y / dual-only IR 計算。
//!
//! - `compute_lsq::compute_lsq_dual_y`: 元空間で A^T y = -(Qx+c+bound_contrib) を LSQ + DD-IR
//! - `dual_ir::try_dual_only_ir`: 自由列に対し r_d_free を厳密に 0 にする dual-only IR
//! - `loop_::run_dual_recovery_postprocess`: refine 系を組み合わせた KKT 改善ループ

mod compute_lsq;
mod dual_ir;
mod loop_;

pub(crate) use compute_lsq::compute_lsq_dual_y;
pub(crate) use dual_ir::try_dual_only_ir;
pub(crate) use loop_::run_dual_recovery_postprocess;
