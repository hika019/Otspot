//! 内部 outcome: status mutation を API 境界 1 箇所に集約するための struct。

use crate::sparse::CscMatrix;

/// 内部 IPM がどの終端条件で停止したか。
///
/// `finalize_outcome` が「eps 未達かつ非 timeout」の outcome へ正直な status を
/// 割り当てるための事実情報。解品質は常に `satisfies_eps` + `prove_optimal` で
/// 独立判定するため、この enum は品質を主張しない。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpmTermination {
    /// 内部 (scaled 空間) の収束判定を満たして停止した。元空間の user_eps を
    /// 満たすかは別判定 (満たさない場合は精度床 = Stalled として報告する)。
    Converged,
    /// α-stall / 残差停滞 / 方向発散からの best-so-far 復帰で停止した。
    Stalled,
    /// 反復予算 (`ipm.max_iter`) を使い切って停止した。
    IterationLimit,
    /// deadline / cancel を検知して停止した。
    Deadline,
}

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
    /// postsolve の saddle-point Krylov IR が `kkt_already_pass` ゲートで省略されたか。
    /// 既に user_eps を満たす収束解で重い拡大 KKT 因子化を回避した場合に true。
    /// ゲートを外す (常時 refine) と false になる sentinel 用観測点。
    pub postsolve_krylov_ir_skipped: bool,
    /// IPM + postsolve stage 別計測。常時収集 (instrumentation only)。
    pub timing: Option<crate::problem::TimingBreakdown>,
    /// 内部 IPM の終端条件。`solution` が非空の通常経路でのみ意味を持つ
    /// (empty / numerical_failure / infeasibility は finalize が先に分岐する)。
    pub termination: IpmTermination,
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
            postsolve_krylov_ir_skipped: false,
            timing: None,
            termination: IpmTermination::Deadline,
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
///
/// `eliminated_cols[j] == true` の col は presolve が物理削除した EmptyCol で、
/// postsolve が `x[j]=val` を埋め戻したあとの original space stationarity 評価から除外する
/// (bd=0 慣例で r=0 になる前提)。reduced space (IPM 内部) では `&[]` を渡す:
/// 削除済み col は構造的に存在しないため。長さ != bounds.len() の slice は無視する。
pub struct ProblemView<'a> {
    pub q: &'a CscMatrix,
    pub a: &'a CscMatrix,
    pub c: &'a [f64],
    pub b: &'a [f64],
    pub bounds: &'a [(f64, f64)],
    pub constraint_types: &'a [crate::problem::ConstraintType],
    pub eliminated_cols: &'a [bool],
}

impl<'a> ProblemView<'a> {
    /// presolve 情報なしで構築する (IPM internal / tests)。eliminated_cols = `&[]`。
    pub fn from_problem(problem: &'a crate::qp::problem::QpProblem) -> Self {
        Self {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &[],
        }
    }
}
