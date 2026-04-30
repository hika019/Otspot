//! 内部 outcome struct: status mutation を 1 箇所に集約する。
//!
//! 既存 ippmm.rs では `SolveStatus` (Optimal/Suboptimal/MaxIterations 等) を
//! 9 箇所以上で mutation していた。v2 では「内部は KKT 残差・iter 数・解ベクトルを持つ struct」
//! のみで操作し、status の決定は API 境界 1 箇所のみで行う。

use crate::sparse::CscMatrix;

/// 1 回の IPM attempt の結果。残差ベースの収束判定はここに集約 (`satisfies_eps`)。
///
/// 通常は status mutation を持たない設計だが、内部 solver が **確定的な Infeasible /
/// Unbounded** を検出した場合のみ `infeasibility_status` でその事実を保持する。
/// これがないと真の Infeasible が finalize_outcome で Timeout に丸められてしまい
/// status 隠蔽が起きる (cmd_session8 で発見)。
#[derive(Clone, Debug)]
pub struct IpmOutcome {
    /// primal 解 x (n 長, 元空間)
    pub solution: Vec<f64>,
    /// 線形等式・不等式 dual y (m 長, 元空間)
    pub dual_solution: Vec<f64>,
    /// bound dual z (lb 有限変数の y_lb + ub 有限変数の y_ub, 元空間)
    pub bound_duals: Vec<f64>,
    /// 目的関数値 (元 Q, c で計算済)
    pub objective: f64,
    /// 反復回数
    pub iterations: usize,
    /// 元空間 KKT 残差 (成分相対化 max_j |Qx+c+A^Ty+z|_j / scale_j)
    pub kkt_residual_rel: f64,
    /// 元空間 primal 残差 (成分正規化 max_i violation/(1+|a|+|b|))
    pub primal_residual_rel: f64,
    /// 元空間 bounds 違反 (max_j max(lb-x, x-ub))
    pub bound_violation: f64,
    /// 元空間 双対ギャップ相対値 |primal_obj - dual_obj| / max(|p|, |d|, 1)。
    /// rank-deficient Q (UBH1 等) で KKT 残差は小さいが obj が大きく外れる
    /// 偽 Optimal を検出するためのゲート。
    pub duality_gap_rel: f64,
    /// 内部数値エラー (NaN / Inf 等で解が無効) フラグ
    pub numerical_failure: bool,
    /// 内部 solver が確定的に判定した Infeasible / Unbounded を保持する。
    /// `None` = 通常 (収束/未収束)、`Some(Infeasible|Unbounded)` = 確定判定。
    /// 他の status (Optimal/Timeout/...) はここに入れない (残差から外部で判定するため)。
    pub infeasibility_status: Option<crate::problem::SolveStatus>,
}

impl IpmOutcome {
    /// 解が空の状態を表す empty outcome (timeout 等で何も得られなかった場合)。
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
            duality_gap_rel: f64::INFINITY,
            numerical_failure: false,
            infeasibility_status: None,
        }
    }

    /// 確定的 Infeasible / Unbounded / NonConvex を保持する outcome を構築する。
    /// これらは数値解の eps 判定では復元できない構造的判定なので、
    /// IpmOutcome から SolverResult への変換時に最優先で外部 status に伝搬する。
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

    /// 双対ギャップ閾値: rank-deficient Q (UBH1) の偽 Optimal を弾く promotion gate。
    /// IPM 内部 (ippmm.rs) の Optimal_main 判定でも DUALITY_GAP_TOL=1e-3 を使うが、
    /// post-promotion (Suboptimal→Optimal の昇格) は IPM_PROMOTION_GAP_TOL=1e-1 で
    /// より緩い (内部判定漏れの最終防壁、scaling.rs::PROMOTION_GAP_TOL と整合)。
    pub const PROMOTION_GAP_TOL: f64 = 1e-1;

    /// ユーザー指定 eps を満たすか判定する (元空間, 成分相対化)。
    /// rank-deficient Q (UBH1 等) では KKT 残差が小さくても obj が大きく外れるため、
    /// duality_gap_rel < PROMOTION_GAP_TOL を最終防壁として加える。
    pub fn satisfies_eps(&self, eps: f64) -> bool {
        !self.solution.is_empty()
            && !self.numerical_failure
            && self.kkt_residual_rel <= eps
            && self.primal_residual_rel <= eps
            && self.bound_violation <= eps
            && self.duality_gap_rel < Self::PROMOTION_GAP_TOL
    }

    /// 残差の合算 score (小さいほど良い)。retry での best-so-far 比較用。
    pub fn quality_score(&self) -> f64 {
        if self.solution.is_empty() || self.numerical_failure {
            return f64::INFINITY;
        }
        self.kkt_residual_rel + self.primal_residual_rel + self.bound_violation
    }
}

/// QP 問題の参照を保持する軽量 view (KKT 計算に必要な要素のみ)
pub struct ProblemView<'a> {
    pub q: &'a CscMatrix,
    pub a: &'a CscMatrix,
    pub c: &'a [f64],
    pub b: &'a [f64],
    pub bounds: &'a [(f64, f64)],
    pub constraint_types: &'a [crate::problem::ConstraintType],
}
