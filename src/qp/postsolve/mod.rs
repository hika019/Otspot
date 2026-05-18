//! QP postsolve: 元問題空間での dual recovery / refinement 集約。
//!
//! - `bound_dual`: bound dual の remap / 射影 / inactive zero
//! - `dual_recovery`: 共通 helper (active row 判定 / cluster / local bound 抽出)
//! - `refine/*`: dual / primal の反復精密化
//! - `postprocess`: LSQ y 計算 / dual-only IR / 全体 postprocess loop

pub(crate) mod bound_dual;
pub(crate) mod dual_recovery;
pub(crate) mod postprocess;
pub(crate) mod refine;
