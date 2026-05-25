//! dual / primal の反復精密化サブモジュール。

pub(crate) mod kkt_iterative;
pub(crate) mod lsq;
pub(crate) mod primal_lsq;
pub(crate) mod projected_gradient;
pub(crate) mod worst_active;
