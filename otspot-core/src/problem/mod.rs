//! LP問題定義モジュール
//!
//! 線形計画問題（LP）の構造定義・制約種別・ソルバー結果の表現を提供する。
//! 問題は標準形 `min c^T x  s.t.  Ax {<=,>=,=} b,  x in [lb, ub]` で定義される。

pub mod certificate;
use certificate::{BoundGapCertificate, OptimalCertificate};

use crate::error::SolverError;
use crate::options::WarmStartBasis;
use crate::sparse::CscMatrix;
use std::fmt;

/// LP問題における制約条件の種別
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintType {
    /// 以下（<=）
    Le,
    /// 以上（>=）
    Ge,
    /// 等式（==）
    Eq,
}

/// Route taken by a solve call (populated per-result, race-free).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SolveRoute {
    /// Route not yet set (default for uninitialized results).
    #[default]
    Unknown,
    /// Called directly via `crate::lp::solve_lp_with`.
    LpDirect,
    /// LP forwarded from `solve_qp_with(Q=0)`.
    LpForwardedFromQp,
    /// QP solved via IPM (Q≠0).
    QpIpm,
}

/// Per-solve routing and warm-start statistics (race-free, per-result).
///
/// Replaces process-global `AtomicU64` counters so parallel tests observe
/// independent stats without reset/race issues.
#[derive(Debug, Clone, Default)]
pub struct SolveStats {
    /// Route taken for this solve.
    pub route: SolveRoute,
    /// Whether the solver stopped because the deadline (timeout_secs / deadline) was reached.
    ///
    /// `true` iff `result.status == SolveStatus::Timeout`. Deterministic sentinel for
    /// deadline-enforcement tests: assert this field instead of measuring wall time.
    pub deadline_triggered: bool,
    /// Whether the postsolve saddle-point Krylov IR was skipped because the solution
    /// already met the user tolerance (`kkt_already_pass`). Deterministic sentinel for
    /// the gate: removing the gate (always refine) flips this to `false`.
    pub postsolve_krylov_ir_skipped: bool,
    /// For LP solves via the QP dispatch (`LpForwardedFromQp`): `true` if the
    /// returned result came from the IPM, `false` if from simplex. Lets callers
    /// (e.g. benchmarks) label the actual route instead of a static size guess.
    pub lp_ipm_path: bool,
}

/// ソルバーの求解結果ステータス
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SolveStatus {
    /// 最適解が求まった
    Optimal,
    /// 局所的最適解（非凸QP: 慣性修正付きIPMが収束したKKT点）
    ///
    /// Q行列が不定（indefinite）の場合、大域的最適性は保証されないが、
    /// KKT条件を満たす局所最適解またはサドル点が返される。
    /// 慣性修正（Gershgorin 評価から導出した δI 加算）により IPM を収束させた。
    LocallyOptimal,
    /// 問題が実行不可能（infeasible）
    Infeasible,
    /// 問題が非有界（unbounded）
    Unbounded,
    /// 反復回数上限に到達した（最適性未確認）
    MaxIterations,
    /// 解は見つかったが精度基準未達（偽Optimal検出: スケール解除後の残差超過）
    SuboptimalSolution,
    /// タイムアウト（timeout_secs を超過した）
    Timeout,
    /// 数値エラー（LDL分解失敗等、問題が数値的に解けない）
    NumericalError,
    /// Q行列が不定（非凸QP）。IPMはQ正半定値を前提とする。
    NonConvex(String),
    /// 非凸 QP の局所最適解 (= `solve_qp_global` 経由で incumbent あり、ε-global 証明なし)。
    ///
    /// BB driver が deadline / max_nodes / max_depth で打ち切られ、incumbent ある状態。
    /// `LocallyOptimal` (= IPM inertia 補正後の単発解) と区別して、caller が「探索打切」
    /// vs「単発 KKT 収束」を識別できる。`Optimal` には**含めない** (= global proof なし)。
    NonconvexLocal,
    /// 非凸 QP の大域 ε-最適解 (= `solve_qp_global` で gap_tol まで証明済み + Q が indefinite)。
    ///
    /// `Optimal` は「Q が PSD で IPM/BB が global 達成」専用に維持し、indefinite Q の場合は
    /// 本 variant で明示分離する (caller が「global 証明済」かを fact で判別)。
    NonconvexGlobal,
    /// Problem type not supported by this solver.
    ///
    /// Returned when the caller passes a problem that cannot be handled, e.g.
    /// a QCQP (quadratic constraints present) submitted to the QP/LP entry.
    NotSupported(String),
}

impl fmt::Display for SolveStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolveStatus::Optimal => write!(f, "Optimal"),
            SolveStatus::LocallyOptimal => write!(f, "LocallyOptimal"),
            SolveStatus::Infeasible => write!(f, "Infeasible"),
            SolveStatus::Unbounded => write!(f, "Unbounded"),
            SolveStatus::MaxIterations => write!(f, "MaxIterations"),
            SolveStatus::SuboptimalSolution => write!(f, "SuboptimalSolution"),
            SolveStatus::Timeout => write!(f, "Timeout"),
            SolveStatus::NumericalError => write!(f, "NumericalError"),
            SolveStatus::NonConvex(msg) => write!(f, "NonConvex({})", msg),
            SolveStatus::NonconvexLocal => write!(f, "NonconvexLocal"),
            SolveStatus::NonconvexGlobal => write!(f, "NonconvexGlobal"),
            SolveStatus::NotSupported(msg) => write!(f, "NotSupported({})", msg),
        }
    }
}

/// LP/QP共通求解結果型
///
/// LP求解（Simplex等）と QP求解（AS/IPM/Concurrent）の両方で使用できる統一結果型。
/// LP固有フィールド（`reduced_costs`, `slack`, `warm_start_basis`）は QP求解時は空/None。
/// QP固有フィールド（`bound_duals`, `iterations`）は LP求解時は空/0。
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// 求解ステータス
    pub status: SolveStatus,
    /// 最適目的関数値（最適解が存在する場合）
    pub objective: f64,
    /// 解ベクトル（最適解が存在する場合）
    pub solution: Vec<f64>,
    /// 双対変数ベクトル（各制約の影価格、最適解が存在する場合）
    pub dual_solution: Vec<f64>,
    // --- LP固有フィールド ---
    /// 被縮小費用ベクトル（各決定変数に対して、最適解が存在する場合）
    pub reduced_costs: Vec<f64>,
    /// スラック変数ベクトル（各制約のスラック b_i - a_i^T x、最適解が存在する場合）
    pub slack: Vec<f64>,
    /// warm-start用の基底情報（Optimal時のみ Some）
    pub warm_start_basis: Option<WarmStartBasis>,
    // --- QP固有フィールド ---
    /// Bound dual values (shadow prices for variable bounds).
    ///
    /// Maps to original variable indices via col_map.
    /// Empty if no bound constraints are active.
    ///
    /// - 除去変数 (presolveで固定された変数) の bound_dual = 0.0 (近似)
    /// - presolve tightening で追加された境界の dual は報告しない（元問題基準）
    /// - 配列順: `[lb_dual(j0), ..., lb_dual(j_{n_lb-1}), ub_dual(j0), ..., ub_dual(j_{n_ub-1})]`
    pub bound_duals: Vec<f64>,
    /// 反復回数（WSR実績回数）
    pub iterations: usize,
    /// 最終反復の残差実値 (pfeas, dfeas, duality_gap)。Optimal/MaxIterations時のみ Some。
    pub final_residuals: Option<(f64, f64, f64)>,
    /// 相対双対ギャップ (|p_obj - d_obj| / max(|p|,|d|,1))。
    /// IPPMM 内部の best-so-far に紐づく値。unscale_ipm_result の Suboptimal→Optimal 昇格ゲート用。
    /// None = 未計測（LP simplex 等 gap を持たない経路）。
    pub duality_gap_rel: Option<f64>,
    /// 各 phase の所要時間 (LP simplex 経路のみ、None なら未計測)。
    /// 「どこに時間が掛かっているか」事実観測用 (CLAUDE.md「順調に収束に向けて探索」)。
    pub timing_breakdown: Option<TimingBreakdown>,
    /// Postsolve が最終的に採用した y_orig の dfeas violation (bound-aware sup ノルム).
    /// LP simplex + presolve 経路のみ Some。caller (solve_with) が値が `PIVOT_TOL` を
    /// 超えるとき presolve=off で再解する fallback gate に使う (greenbea-class 問題対策)。
    pub postsolve_dfeas: Option<f64>,
    /// Per-solve routing and warm-start statistics (race-free).
    pub stats: SolveStats,
    /// Branch-and-bound gap certificate.
    ///
    /// Present iff the solver completed a B&B search with a fully authenticated gap
    /// (no `proof_uncertain` region, `within_gap` satisfied). `None` for direct LP/QP solves.
    pub bound_gap_cert: Option<BoundGapCertificate>,
    /// KKT optimality certificate — minted by `prove_optimal` on the B&B incumbent.
    ///
    /// Set when `finalize_proven` verifies all KKT conditions on the returned point.
    /// `None` for demoted (LocallyOptimal/NonconvexLocal) or non-B&B results.
    pub opt_cert: Option<OptimalCertificate>,
}

/// 各 phase 所要時間 (μs精度)。LP simplex と QP IPM の両経路で共用。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TimingBreakdown {
    // ── LP simplex 経路 ───────────────────────────────────────────────────────
    /// Presolve 全体 (run_presolve)
    pub presolve_us: u64,
    /// 縮約後 simplex 本体 (solve_without_presolve)
    pub solve_us: u64,
    /// Postsolve 総計 (run_postsolve / QP 経路では下記 4 field の合計)
    pub postsolve_us: u64,

    // ── QP IPM 経路: IPM 反復内部 ────────────────────────────────────────────
    /// KKT 行列 LDL 数値因子化の累計 (全 iteration 合計)
    pub ipm_factorize_us: u64,
    /// KKT solve (predictor/corrector/Gondzio) の累計
    pub ipm_solve_us: u64,
    /// LDL regularization retry の累計回数 (健全性プローブ失敗含む)
    pub ipm_reg_retries: u32,
    /// MINRES (iterative) backend が 1 回以上使われたか
    pub ipm_used_iterative: bool,

    // ── QP IPM 経路: postsolve 段階別 ────────────────────────────────────────
    /// postsolve_qp_with_dual_recovery (reduced → orig 空間写像)
    pub postsolve_map_us: u64,
    /// refine_postsolve_dual_lsq (元空間 y LSQ refine)
    pub postsolve_lsq_us: u64,
    /// refine_postsolve_recovery (Stage 0: SingletonRow 後退代入)
    pub postsolve_recovery_us: u64,
    /// refine_post_processing (Stage 1+2: primal projection / y-z refit)
    pub postsolve_refine_us: u64,
    /// refine_krylov_and_projection (saddle-point Krylov IR)
    pub postsolve_krylov_ir_us: u64,
}

impl Default for SolverResult {
    fn default() -> Self {
        SolverResult {
            status: SolveStatus::NumericalError,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            final_residuals: None,
            duality_gap_rel: None,
            timing_breakdown: None,
            postsolve_dfeas: None,
            stats: SolveStats::default(),
            bound_gap_cert: None,
            opt_cert: None,
        }
    }
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Status: {}, Objective: {}", self.status, self.objective)
    }
}

/// 線形計画問題: min c^T x  s.t.  Ax {op} b,  x in [lb, ub]
///
/// 目的関数・制約行列・右辺ベクトル・変数上下限をまとめて保持する。
/// 制約種別（`<=`, `>=`, `=`）と変数ごとの上下限を個別に指定できる。
#[derive(Debug, Clone)]
pub struct LpProblem {
    /// 目的関数係数ベクトル（長さ: `num_vars`）
    pub c: Vec<f64>,
    /// 制約行列（CSC形式、サイズ: `num_constraints` x `num_vars`）
    pub a: CscMatrix,
    /// 制約右辺ベクトル（長さ: `num_constraints`）
    pub b: Vec<f64>,
    /// 決定変数の数
    pub num_vars: usize,
    /// 制約式の数
    pub num_constraints: usize,
    /// 各制約の種別（長さ: `num_constraints`）
    pub constraint_types: Vec<ConstraintType>,
    /// 各変数の上下限 `(lower, upper)`（長さ: `num_vars`）
    pub bounds: Vec<(f64, f64)>,
    /// 問題名（オプション）
    pub name: Option<String>,
}

impl LpProblem {
    /// 新しいLP問題を検証付きで生成する（後方互換版）
    ///
    /// 標準形 `min c^T x  s.t.  Ax <= b,  x >= 0` を作成する。
    /// 全制約を `<=`、全変数の下限を 0・上限を `+∞` とする。
    ///
    /// # 引数
    /// * `c` - 目的関数係数ベクトル
    /// * `a` - 制約行列（CSC形式）
    /// * `b` - 制約右辺ベクトル
    ///
    /// # 戻り値
    /// * `Ok(LpProblem)` - 次元が有効な場合
    /// * `Err(String)` - 次元不一致などの検証エラー時
    pub fn new(c: Vec<f64>, a: CscMatrix, b: Vec<f64>) -> Result<Self, SolverError> {
        let num_vars = c.len();
        let num_constraints = b.len();

        // Set defaults for backward compatibility
        let constraint_types = vec![ConstraintType::Le; num_constraints];
        let bounds = vec![(0.0, f64::INFINITY); num_vars];
        let name = None;

        Self::new_general(c, a, b, constraint_types, bounds, name)
    }

    /// 制約種別と変数上下限を完全指定して新しいLP問題を生成する
    ///
    /// # 引数
    /// * `c` - 目的関数係数ベクトル
    /// * `a` - 制約行列（CSC形式）
    /// * `b` - 制約右辺ベクトル
    /// * `constraint_types` - 各制約の種別（`Le` / `Ge` / `Eq`）
    /// * `bounds` - 各変数の上下限 `(lower, upper)`
    /// * `name` - 問題名（オプション）
    ///
    /// # 戻り値
    /// * `Ok(LpProblem)` - 次元が有効な場合
    /// * `Err(String)` - 次元不一致などの検証エラー時
    pub fn new_general(
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        constraint_types: Vec<ConstraintType>,
        bounds: Vec<(f64, f64)>,
        name: Option<String>,
    ) -> Result<Self, SolverError> {
        // Validate dimensions
        if c.len() != a.ncols {
            return Err(SolverError::DimensionMismatch {
                field: "c",
                expected: a.ncols,
                got: c.len(),
            });
        }
        if b.len() != a.nrows {
            return Err(SolverError::DimensionMismatch {
                field: "b",
                expected: a.nrows,
                got: b.len(),
            });
        }
        if constraint_types.len() != b.len() {
            return Err(SolverError::DimensionMismatch {
                field: "constraint_types",
                expected: b.len(),
                got: constraint_types.len(),
            });
        }
        if bounds.len() != c.len() {
            return Err(SolverError::DimensionMismatch {
                field: "bounds",
                expected: c.len(),
                got: bounds.len(),
            });
        }
        for (i, &v) in c.iter().enumerate() {
            if !v.is_finite() {
                return Err(SolverError::NonFiniteCoefficient { field: "c", index: i });
            }
        }
        for (i, &v) in b.iter().enumerate() {
            if !v.is_finite() {
                return Err(SolverError::NonFiniteCoefficient { field: "b", index: i });
            }
        }
        for (i, &v) in a.values.iter().enumerate() {
            if !v.is_finite() {
                return Err(SolverError::NonFiniteCoefficient { field: "A", index: i });
            }
        }
        for (i, &(lb, ub)) in bounds.iter().enumerate() {
            if lb.is_nan() || ub.is_nan() || lb > ub {
                return Err(SolverError::InvalidBounds { index: i, lb, ub });
            }
        }

        Ok(LpProblem {
            num_vars: c.len(),
            num_constraints: b.len(),
            c,
            a,
            b,
            constraint_types,
            bounds,
            name,
        })
    }
}

impl fmt::Display for LpProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LP: min c^T x, {} vars, {} constraints",
            self.num_vars, self.num_constraints
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SolverError;

    #[test]
    fn test_lp_problem_new_valid() {
        // 2 variables, 2 constraints
        let c = vec![1.0, 2.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0];

        let lp = LpProblem::new(c, a, b).unwrap();
        assert_eq!(lp.num_vars, 2);
        assert_eq!(lp.num_constraints, 2);
    }

    #[test]
    fn test_lp_problem_new_invalid_c_dimension() {
        // c.len() = 3, but a.ncols = 2
        let c = vec![1.0, 2.0, 3.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0];

        let result = LpProblem::new(c, a, b);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SolverError::DimensionMismatch { field: "c", .. }
        ));
    }

    #[test]
    fn test_lp_problem_new_invalid_b_dimension() {
        // b.len() = 3, but a.nrows = 2
        let c = vec![1.0, 2.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0, 7.0];

        let result = LpProblem::new(c, a, b);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SolverError::DimensionMismatch { field: "b", .. }
        ));
    }

    #[test]
    fn test_lp_problem_display() {
        let c = vec![1.0, 2.0];
        let a = CscMatrix::new(2, 2);
        let b = vec![5.0, 6.0];
        let lp = LpProblem::new(c, a, b).unwrap();

        let display = format!("{}", lp);
        assert_eq!(display, "LP: min c^T x, 2 vars, 2 constraints");
    }

    #[test]
    fn test_solve_status_display() {
        assert_eq!(format!("{}", SolveStatus::Optimal), "Optimal");
        assert_eq!(format!("{}", SolveStatus::Infeasible), "Infeasible");
        assert_eq!(format!("{}", SolveStatus::Unbounded), "Unbounded");
    }

    #[test]
    fn test_solver_result_display() {
        let result = SolverResult {
            status: SolveStatus::Optimal,
            objective: 42.5,
            solution: vec![1.0, 2.0],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            ..Default::default()
        };
        let display = format!("{}", result);
        assert_eq!(display, "Status: Optimal, Objective: 42.5");
    }

    #[test]
    fn solver_result_default_is_not_success() {
        let result = SolverResult::default();
        assert_eq!(result.status, SolveStatus::NumericalError);
        assert!(result.solution.is_empty());
    }

    fn make_lp(c: Vec<f64>, b: Vec<f64>, a_vals: Vec<f64>, bounds: Vec<(f64, f64)>)
        -> Result<LpProblem, SolverError>
    {
        let n = c.len();
        let m = b.len();
        let a = if a_vals.is_empty() {
            CscMatrix::new(m, n)
        } else {
            let rows = vec![0usize; n];
            let cols: Vec<usize> = (0..n).collect();
            CscMatrix::from_triplets(&rows, &cols, &a_vals, m, n).unwrap()
        };
        let ct = vec![ConstraintType::Le; m];
        LpProblem::new_general(c, a, b, ct, bounds, None)
    }

    #[test]
    fn lp_valid_accepted() {
        let res = make_lp(
            vec![1.0, 2.0], vec![5.0], vec![1.0, 1.0],
            vec![(0.0, f64::INFINITY), (0.0, 10.0)],
        );
        assert!(res.is_ok());
    }

    #[test]
    fn lp_nan_in_c_rejected() {
        let bad_vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in bad_vals {
            let res = make_lp(vec![bad, 1.0], vec![5.0], vec![1.0, 1.0],
                              vec![(0.0, f64::INFINITY); 2]);
            assert!(
                matches!(res, Err(SolverError::NonFiniteCoefficient { field: "c", .. })),
                "expected NonFiniteCoefficient for c={bad}"
            );
        }
    }

    #[test]
    fn lp_nan_in_b_rejected() {
        let bad_vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in bad_vals {
            let res = make_lp(vec![1.0, 2.0], vec![bad], vec![1.0, 1.0],
                              vec![(0.0, f64::INFINITY); 2]);
            assert!(
                matches!(res, Err(SolverError::NonFiniteCoefficient { field: "b", .. })),
                "expected NonFiniteCoefficient for b={bad}"
            );
        }
    }

    #[test]
    fn lp_nan_in_a_rejected() {
        let n = 2;
        let bad_vals = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        for bad in bad_vals {
            // from_triplets drops NaN via DROP_TOL; inject bad value directly.
            let mut a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
            a.values[0] = bad;
            let res = LpProblem::new_general(
                vec![1.0, 2.0], a, vec![5.0],
                vec![ConstraintType::Le], vec![(0.0, f64::INFINITY); n], None,
            );
            assert!(
                matches!(res, Err(SolverError::NonFiniteCoefficient { field: "A", .. })),
                "expected NonFiniteCoefficient for A val={bad}"
            );
        }
    }

    #[test]
    fn lp_nan_in_bounds_rejected() {
        let cases: Vec<(f64, f64)> = vec![
            (f64::NAN, 1.0),
            (0.0, f64::NAN),
            (f64::NAN, f64::NAN),
        ];
        for (lb, ub) in cases {
            let res = make_lp(
                vec![1.0, 2.0], vec![5.0], vec![1.0, 1.0],
                vec![(lb, ub), (0.0, f64::INFINITY)],
            );
            assert!(
                matches!(res, Err(SolverError::InvalidBounds { index: 0, .. })),
                "expected InvalidBounds for ({lb},{ub})"
            );
        }
    }

    #[test]
    fn lp_lb_gt_ub_rejected() {
        let cases: Vec<(f64, f64)> = vec![
            (5.0, 1.0),
            (1.0, 0.0),
            (f64::INFINITY, f64::NEG_INFINITY),
            (0.1, 0.0),
        ];
        for (lb, ub) in cases {
            let res = make_lp(
                vec![1.0, 2.0], vec![5.0], vec![1.0, 1.0],
                vec![(lb, ub), (0.0, f64::INFINITY)],
            );
            assert!(
                matches!(res, Err(SolverError::InvalidBounds { .. })),
                "expected InvalidBounds for lb={lb} ub={ub}"
            );
        }
    }

    #[test]
    fn lp_inf_bounds_accepted() {
        let res = make_lp(
            vec![1.0, 2.0], vec![5.0], vec![1.0, 1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
        );
        assert!(res.is_ok(), "±inf bounds should be valid");
    }
}
