//! KKT iterative refinement (Wilkinson) と bound dual の KKT 再計算。

mod bound_refit;
mod iterative;

pub(crate) use bound_refit::refit_bound_duals_kkt;
pub(crate) use iterative::refine_kkt_iterative;
