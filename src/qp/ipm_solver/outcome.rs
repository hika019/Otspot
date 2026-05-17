//! 内部 outcome: status mutation を API 境界 1 箇所に集約するための struct。

use crate::sparse::CscMatrix;

/// 残差ベース収束判定を `satisfies_eps` に集約。Infeasible / Unbounded のみ
/// `infeasibility_status` で保持し、finalize で Timeout に丸めない。
#[derive(Clone, Debug)]
pub struct IpmOutcome {
    pub solution: Vec<f64>,
    pub dual_solution: Vec<f64>,
    /// lb 有限の y_lb + ub 有限の y_ub。
    pub bound_duals: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    /// 成分相対化 stationarity 残差 max_j |Qx+c+Aᵀy+z|_j / scale_j。
    pub kkt_residual_rel: f64,
    /// 成分正規化 primal violation max_i violation/(1+|a|+|b|)。
    pub primal_residual_rel: f64,
    /// max_j max(lb−x, x−ub)。
    pub bound_violation: f64,
    /// 成分相対化 complementarity 残差。stationarity のみでは「feasible だが optimal でない点」を見逃すため別立て。
    pub complementarity_residual_rel: f64,
    /// |p − d| / max(|p|,|d|,1)。rank-deficient Q の偽 Optimal を弾く。
    pub duality_gap_rel: f64,
    pub numerical_failure: bool,
    /// 確定判定された Infeasible / Unbounded のみ保持 (他 status は残差から外部判定)。
    pub infeasibility_status: Option<crate::problem::SolveStatus>,
    /// 慣性修正付き IPM が走った場合、収束時に Optimal でなく LocallyOptimal を返す。
    pub is_locally_optimal: bool,
}

impl IpmOutcome {
    pub fn empty() -> Self {
        Self {
            solution: Vec::new(),
            dual_solution: Vec::new(),
            bound_duals: Vec::new(),
            objective: f64::INFINITY,
            iterations: 0,
            kkt_residual_rel: f64::INFINITY,
            primal_residual_rel: f64::INFINITY,
            bound_violation: f64::INFINITY,
            complementarity_residual_rel: f64::INFINITY,
            duality_gap_rel: f64::INFINITY,
            numerical_failure: false,
            infeasibility_status: None,
            is_locally_optimal: false,
        }
    }

    /// 構造的判定 (Infeasible / Unbounded / NonConvex) を保持する outcome。
    pub fn infeasibility(status: crate::problem::SolveStatus) -> Self {
        debug_assert!(
            matches!(
                status,
                crate::problem::SolveStatus::Infeasible
                    | crate::problem::SolveStatus::Unbounded
                    | crate::problem::SolveStatus::NonConvex(_)
            ),
            "infeasibility outcome must be Infeasible / Unbounded / NonConvex, got {:?}",
            status
        );
        Self {
            infeasibility_status: Some(status),
            ..Self::empty()
        }
    }

    /// Suboptimal→Optimal 昇格時の rel gap 上限 (scaling.rs::PROMOTION_GAP_TOL と整合)。
    pub const PROMOTION_GAP_TOL: f64 = 1e-1;

    pub fn satisfies_eps(&self, eps: f64) -> bool {
        !self.solution.is_empty()
            && !self.numerical_failure
            && self.kkt_residual_rel <= eps
            && self.primal_residual_rel <= eps
            && self.bound_violation <= eps
            && self.complementarity_residual_rel <= eps
            && self.duality_gap_rel < Self::PROMOTION_GAP_TOL
    }

    /// satisfies_eps と整合する max-componentwise 残差 (小さいほど良い)。
    pub fn quality_score(&self) -> f64 {
        if self.solution.is_empty() || self.numerical_failure {
            return f64::INFINITY;
        }
        self.kkt_residual_rel
            .max(self.primal_residual_rel)
            .max(self.bound_violation)
            .max(self.complementarity_residual_rel)
    }
}

/// KKT 計算に必要な要素だけを参照する軽量 view。
pub struct ProblemView<'a> {
    pub q: &'a CscMatrix,
    pub a: &'a CscMatrix,
    pub c: &'a [f64],
    pub b: &'a [f64],
    pub bounds: &'a [(f64, f64)],
    pub constraint_types: &'a [crate::problem::ConstraintType],
}
