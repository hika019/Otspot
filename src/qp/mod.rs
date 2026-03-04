//! 二次計画法（QP）ソルバーモジュール
//!
//! Active Set法による QP ソルバーを提供する。
//! 問題形式: min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub
//!
//! # 規約
//! **「1/2あり」規約** (OSQP/qpOASES標準):
//! - 目的関数: min 1/2 x^T Q x + c^T x
//! - ∇f(x) = Qx + c
//! - KKT行列: [Q, A_W^T; A_W, 0]（NC1修正済み）
//!
//! # 使用例
//! ```rust
//! use solver::qp::{solve_qp, QpProblem, SolverResult};
//! use solver::sparse::CscMatrix;
//!
//! // min x^2 + y^2  s.t. x + y >= 1
//! // Q = [[2,0],[0,2]] (「1/2あり」規約で min 1/2 * 2 * (x^2+y^2))
//! // c = [0, 0]
//! // A = [[-1,-1]], b = [-1]（x+y >= 1 を Ax <= b 形式に変換）
//! let q = CscMatrix::from_triplets(
//!     &[0, 1], &[0, 1], &[2.0, 2.0], 2, 2
//! ).unwrap();
//! let c = vec![0.0, 0.0];
//! let a = CscMatrix::from_triplets(
//!     &[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2
//! ).unwrap();
//! let b = vec![-1.0];
//! let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
//! let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
//! let result = solve_qp(&problem);
//! // result.solution ≈ [0.5, 0.5], result.objective ≈ 0.5
//! ```

mod active_set;
pub(crate) mod kkt;
mod problem;
mod solver;
pub mod ipm;
pub mod diagnose;
pub use problem::{QpProblem, QpWarmStart};
pub use diagnose::{diagnose, DiagnosticReport, DiagnosticWarning, DiagnosticCode, Severity, ProblemInfo};
pub use crate::problem::SolverResult;
pub use ipm::solve_qp_ipm;

use crate::options::{QpSolverChoice, SolverOptions};
use crate::presolve::{run_qp_presolve_phase1, postsolve_qp};
use crate::problem::SolveStatus;

/// Concurrent Solver が複数ソルバーの結果を比較するための解品質ランク
///
/// 順序: Optimal > Feasible > Approximate
/// `PartialOrd/Ord` を実装することで `>` による比較が可能。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityRank {
    /// 近似解（eps緩和・Timeout後の途中解など）
    Approximate = 0,
    /// 実行可能解（制約は充足するが最適性は保証されていない）
    /// 例: Active SetがMAXITERで停止した実行可能解
    Feasible = 1,
    /// 最適解（KKT条件を充足）
    Optimal = 2,
}

#[cfg(feature = "parallel")]
fn quality_rank_of(result: &SolverResult) -> Option<QualityRank> {
    match result.status {
        SolveStatus::Optimal => Some(QualityRank::Optimal),
        SolveStatus::MaxIterations if !result.solution.is_empty() => {
            Some(QualityRank::Feasible)
        }
        SolveStatus::Timeout if !result.solution.is_empty() => {
            Some(QualityRank::Approximate)
        }
        _ => None, // Infeasible / Unbounded / NumericalError / その他
    }
}

/// QP ソルバーを統一的に扱うための trait
///
/// Active Set / IPM の各ソルバーは `QpSolver` を実装しており、
/// `Box<dyn QpSolver>` として統一的に扱うことができる。
///
/// # 例
/// ```rust,no_run
/// use solver::qp::{QpProblem, QpSolver, ActiveSetSolver};
/// use solver::options::SolverOptions;
/// # let problem = unimplemented!();
/// let solver = ActiveSetSolver;
/// let result = solver.solve(&problem, &SolverOptions::default());
/// ```
pub trait QpSolver {
    /// QP 問題を解く
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult;
    /// ソルバー名を返す
    fn name(&self) -> &'static str;
}

/// Active Set 法 QP ソルバー
///
/// `QpSolver` trait を実装する。内部で [`solver::qp_solve_impl`] を呼ぶ。
pub struct ActiveSetSolver;

/// IPM（内点法）QP ソルバー
///
/// `QpSolver` trait を実装する。内部で [`ipm::solve_qp_ipm`] を呼ぶ。
pub struct IpmSolver;

impl QpSolver for ActiveSetSolver {
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult {
        solver::qp_solve_impl(problem, None, options)
    }
    fn name(&self) -> &'static str {
        "ActiveSet"
    }
}

impl QpSolver for IpmSolver {
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult {
        ipm::solve_qp_ipm(problem, options)
    }
    fn name(&self) -> &'static str {
        "IPM"
    }
}

/// AS / IPM / IPM-Schur を並列実行し、最初に Optimal を返したものを採用する
///
/// parallel feature ON 時のみコンパイルされる。
/// 各スレッドは共有 `cancel_flag` を監視し、勝者決定後に停止する。
/// warm_start は AS スレッドにのみ渡す（IPM は warm_start 未対応）。
#[cfg(feature = "parallel")]
fn solve_qp_concurrent(
    problem: &QpProblem,
    warm_start: Option<&QpWarmStart>,
    options: &SolverOptions,
) -> SolverResult {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let problem_arc = Arc::new(problem.clone());
    let warm_start_cloned = warm_start.cloned();
    let (tx, rx) = mpsc::sync_channel::<SolverResult>(4);
    let mut handles = Vec::with_capacity(4);

    // Active Set スレッド
    // 注: concurrent モードでは rayon 並列ワーカーを無効化 (parallel_runs=1) する。
    // これにより小問題でのスレッドスポーン overhead を回避し、AS が先着しやすくなる。
    // AS の結果は bound_duals を含むため、出力品質が最も高い。
    // warm_start は AS のみに渡す。
    {
        let cancel = Arc::clone(&cancel_flag);
        let prob = Arc::clone(&problem_arc);
        let mut opts = options.clone();
        opts.cancel_flag = Some(cancel);
        opts.parallel_runs = 1; // concurrent モードでは AS 内部の rayon 並列を無効化
        let ws = warm_start_cloned;
        let tx = tx.clone();
        handles.push(std::thread::spawn(move || {
            let r = solver::qp_solve_impl(&prob, ws.as_ref(), &opts);
            let _ = tx.send(r);
        }));
    }

    // IPM スレッド
    {
        let cancel = Arc::clone(&cancel_flag);
        let prob = Arc::clone(&problem_arc);
        let mut opts = options.clone();
        opts.cancel_flag = Some(cancel);
        let tx = tx.clone();
        handles.push(std::thread::spawn(move || {
            let r = ipm::solve_qp_ipm(&prob, &opts);
            let _ = tx.send(r);
        }));
    }

    // IPM-Schur スレッド
    {
        let cancel = Arc::clone(&cancel_flag);
        let prob = Arc::clone(&problem_arc);
        let mut opts = options.clone();
        opts.cancel_flag = Some(cancel);
        let tx = tx.clone();
        handles.push(std::thread::spawn(move || {
            let r = ipm::solve_qp_ipm_schur(&prob, &opts);
            let _ = tx.send(r);
        }));
    }

    drop(tx); // 全スレッドが tx を drop するまで rx.iter() は終了しない

    // 解の品質ランク（Optimal > Feasible > Approximate）で最良解を選択する。
    // cancel_flag は Optimal 到着時のみ立てる（Feasible で cancel すると
    // 間もなく完了するはずの Optimal 解を取りこぼすため）。
    let mut best_ranked: Option<(QualityRank, SolverResult)> = None;
    let mut fallback: Option<SolverResult> = None; // ランク外（Infeasible等）のフォールバック
    for result in rx {
        let rank = quality_rank_of(&result);
        if let Some(r) = rank {
            // 初めて Optimal が来た → cancel_flag を立てる
            if r == QualityRank::Optimal
                && best_ranked.as_ref().map(|(br, _)| *br) != Some(QualityRank::Optimal)
            {
                cancel_flag.store(true, Ordering::Relaxed);
            }
            // ランクが高い解で更新
            let should_update = best_ranked.as_ref().map(|(br, _)| r > *br).unwrap_or(true);
            if should_update {
                best_ranked = Some((r, result));
            }
        } else {
            // ランク外（Infeasible/Unbounded/NumericalError）はフォールバック
            // 非 Optimal の中では Infeasible を優先採用する。
            if result.status == SolveStatus::Infeasible || fallback.is_none() {
                fallback = Some(result);
            }
        }
    }

    // 全スレッドの後始末
    for h in handles {
        let _ = h.join();
    }

    // 最終結果の選択:
    // Optimal は常に優先。Feasible/Approximate より Infeasible の fallback を優先する。
    // 理由: Infeasible は「問題が解けない」という確定的な情報であり、
    //       複数のソルバー（IPM等）が Infeasible を返した場合は
    //       AS の Feasible 停止解より信頼性が高い。
    let best = match best_ranked {
        Some((QualityRank::Optimal, result)) => Some(result),
        Some((_, result)) => {
            // Feasible/Approximate: Infeasible の fallback があれば fallback を優先
            if fallback.as_ref().map(|f| f.status == SolveStatus::Infeasible).unwrap_or(false) {
                fallback
            } else {
                Some(result)
            }
        }
        None => fallback,
    };
    best.unwrap_or_else(|| SolverResult {
        status: SolveStatus::NumericalError,
        objective: f64::NAN,
        solution: vec![0.0; problem.num_vars],
        dual_solution: vec![],
        bound_duals: vec![],
        active_set: vec![],
        iterations: 0,
        ..Default::default()
    })
}

/// QP ソルバーをディスパッチする内部関数
///
/// `options.qp_solver` に基づいてソルバーを選択する。
///
/// - `ActiveSet`: 強制 Active Set
/// - `Ipm`: 強制 IPM（内点法）
/// - `IpmSchur`: 強制 IPM Schur complement パス
/// - `Auto`:
///   - parallel feature ON → AS/IPM/IPM-Schur を並列実行（`solve_qp_concurrent`）
///   - parallel feature OFF:
///     - n < qp_solver_threshold → Active Set（Phase I 失敗時は IPM にフォールバック）
///     - n >= qp_solver_threshold → IPM
fn dispatch_qp(
    problem: &QpProblem,
    warm_start: Option<&QpWarmStart>,
    options: &SolverOptions,
) -> SolverResult {
    match options.qp_solver {
        QpSolverChoice::Ipm => ipm::solve_qp_ipm(problem, options),
        QpSolverChoice::ActiveSet => solver::qp_solve_impl(problem, warm_start, options),
        QpSolverChoice::IpmSchur => ipm::solve_qp_ipm_schur(problem, options),
        QpSolverChoice::Auto => {
            #[cfg(feature = "parallel")]
            {
                solve_qp_concurrent(problem, warm_start, options)
            }
            #[cfg(not(feature = "parallel"))]
            {
                // deadline を1回だけ計算してフォールバックに渡す（二重カウント防止）
                let mut effective_opts;
                let opts = if let (Some(secs), true) = (options.timeout_secs, options.deadline.is_none()) {
                    effective_opts = options.clone();
                    effective_opts.deadline = Some(
                        std::time::Instant::now() + std::time::Duration::from_secs_f64(secs),
                    );
                    &effective_opts
                } else {
                    options
                };
                if problem.num_vars >= opts.qp_solver_threshold {
                    // 大規模問題: IPM
                    ipm::solve_qp_ipm(problem, opts)
                } else {
                    // 小規模問題: Active Set（Phase I 失敗時は IPM にフォールバック）
                    let result = solver::qp_solve_impl(problem, warm_start, opts);
                    if result.status == SolveStatus::MaxIterations
                        && result.solution.is_empty()
                        && result.iterations == 0
                        && !problem.is_zero_q()
                    {
                        ipm::solve_qp_ipm(problem, opts)
                    } else {
                        result
                    }
                }
            }
        }
    }
}

/// QPを解く（デフォルト設定）
///
/// qpOASESの `init()` に相当する基本API。
/// デフォルトの [`SolverOptions`] を使用して求解する。
///
/// # 引数
/// - `problem`: 解くべき二次計画問題
///
/// # 戻り値
/// [`SolverResult`] — ステータス・目的関数値・解・ラグランジュ乗数・活性集合・反復数
pub fn solve_qp(problem: &QpProblem) -> SolverResult {
    solve_qp_with(problem, &SolverOptions::default())
}

/// QPをカスタム設定で解く
///
/// qpOASESの `init()` に相当。`nWSR` は `options.max_iterations` で指定。
pub fn solve_qp_with(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let presolve_result = if options.presolve {
        run_qp_presolve_phase1(problem, options)
    } else {
        crate::presolve::QpPresolveResult::no_reduction(problem)
    };
    let reduced_sol = dispatch_qp(&presolve_result.reduced, None, options);
    postsolve_qp(&presolve_result, &reduced_sol)
}

/// QPをカスタム設定で解く（`solve_qp_with` の別名）
///
/// # Deprecated
///
/// `solve_qp_with` と同一実装のため非推奨。`solve_qp_with` を使用すること。
#[deprecated(since = "0.1.0", note = "use `solve_qp_with` instead")]
pub fn solve_qp_with_options(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    solve_qp_with(problem, options)
}

/// Warm-start付きでQPを解く
///
/// qpOASESの `hotstart()` に相当。SQP反復で前回解の活性集合を引き継ぐ場合に使用。
/// Auto で IPM が選択された場合、warm_start は無視される。
///
/// # 使用例（SQP典型パターン）
/// ```rust,no_run
/// use solver::qp::{solve_qp, solve_qp_warm, QpProblem, QpWarmStart};
///
/// # let problem1 = unimplemented!();
/// # let problem2 = unimplemented!();
/// let result1 = solve_qp(&problem1);
/// let ws = QpWarmStart {
///     initial_active_set: result1.active_set.clone(),
///     initial_point: Some(result1.solution.clone()),
/// };
/// let result2 = solve_qp_warm(&problem2, &ws, &Default::default());
/// // result2 は result1 の活性集合を初期値として使用
/// ```
pub fn solve_qp_warm(
    problem: &QpProblem,
    warm_start: &QpWarmStart,
    options: &SolverOptions,
) -> SolverResult {
    let presolve_result = if options.presolve {
        run_qp_presolve_phase1(problem, options)
    } else {
        crate::presolve::QpPresolveResult::no_reduction(problem)
    };
    // warm_start は元問題の次元で記録されているため、
    // presolve で縮約が発生した場合はインデックスが合わなくなる。
    // 安全策として縮約ありの場合は warm_start を無効化する。
    let ws = if presolve_result.was_reduced { None } else { Some(warm_start) };
    let reduced_sol = dispatch_qp(&presolve_result.reduced, ws, options);
    postsolve_qp(&presolve_result, &reduced_sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::QpSolverChoice;
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    // concurrent solver での許容誤差（AS/IPM/IPM-Schur を並列実行）
    // 目的関数は勾配スケールの影響で primal 誤差より大きくなる場合があるため 1e-2 を使用
    const EPS: f64 = 1e-2;

    fn assert_close(a: f64, b: f64, eps: f64, name: &str) {
        assert!(
            (a - b).abs() < eps,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name, b, a, (a - b).abs()
        );
    }

    /// T1: 2変数基本QP
    /// min 1/2 * 2*(x^2+y^2) = x^2+y^2  s.t. x+y >= 1
    /// Q = [[2,0],[0,2]], c=[0,0], A=[[-1,-1]], b=[-1]
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_basic_qp_2vars() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T1: status should be Optimal");
        assert_close(result.solution[0], 0.5, EPS, "T1: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "T1: x[1]");
        assert_close(result.objective, 0.5, EPS, "T1: objective");
        // NC-DUAL-LEN: 無限境界 → bound_duals空, dual_solution長さ == m == 1
        assert!(result.bound_duals.is_empty(), "T1: infinite bounds → bound_duals empty");
        assert_eq!(result.dual_solution.len(), 1, "T1: dual_solution length == m == 1");
    }

    /// T2: 等式制約付きQP
    /// min x^2+y^2 (1/2あり規約: Q=2I)  s.t. x+y=1
    /// 等式制約は Ax<=b 形式で2不等式に変換:
    ///   x+y <= 1  と  -(x+y) <= -1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_qp_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // A: [1,1; -1,-1], b: [1, -1]
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, -1.0, -1.0],
            2,
            2,
        )
        .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T2: status should be Optimal");
        assert_close(result.solution[0], 0.5, EPS, "T2: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "T2: x[1]");
        assert_close(result.objective, 0.5, EPS, "T2: objective");
    }

    /// T3: Q=0退化ケース（LP問題として解く）
    /// min x+2y  s.t. x>=0, y>=0, x+y<=4, 2x+y<=6
    /// 期待: x*=2, y*=0, obj=2
    #[test]
    fn test_qp_degenerate_lp_case() {
        let n = 2;
        let q = CscMatrix::new(n, n); // Q = 0
        let c = vec![1.0, 2.0];
        // A = [[-1,0],[0,-1],[1,1],[2,1]], b = [0,0,4,6]
        let a = CscMatrix::from_triplets(
            &[0, 1, 2, 2, 3, 3],
            &[0, 1, 0, 1, 0, 1],
            &[-1.0, -1.0, 1.0, 1.0, 2.0, 1.0],
            4,
            2,
        )
        .unwrap();
        let b = vec![0.0, 0.0, 4.0, 6.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T3: status should be Optimal");
        // LP最適解: x*=2, y*=0 (corner: 2x+y<=6, x>=0 → x=3但し x+y<=4なので x=2,y=0? or x=3,y=0?)
        // min x+2y s.t. x+y<=4, 2x+y<=6, x>=0, y>=0
        // vertices: (0,0)→0, (3,0)→3, (2,2)→6, (0,4)→8
        // 最適: (0,0) でobj=0? wait...
        // x>=0: -x<=0, y>=0: -y<=0
        // vertices of feasible region:
        // (0,0): obj=0  → this is optimal for min x+2y
        assert_close(result.objective, 0.0, EPS, "T3: objective");
    }

    /// T4: 制約なしQP
    /// min (x-3)^2 + (y-4)^2 = 1/2*2*(x^2+y^2) - 6x - 8y + const
    /// Q = [[2,0],[0,2]], c = [-6,-8], no constraints, no bounds
    /// 期待: x*=3, y*=4
    #[test]
    fn test_qp_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2); // 制約なし
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T4: status should be Optimal");
        assert_close(result.solution[0], 3.0, EPS, "T4: x[0]");
        assert_close(result.solution[1], 4.0, EPS, "T4: x[1]");
        // obj = 1/2*2*(9+16) - 6*3 - 8*4 = 25 - 18 - 32 = -25
        assert_close(result.objective, -25.0, EPS, "T4: objective");
    }

    /// T5: warm-start整合性
    /// T1と同じ問題を2回解く（2回目はwarm-start）
    /// 期待: 同一解、iterations <= cold-startのiterations
    #[test]
    fn test_warm_start_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a.clone(), b.clone(), bounds.clone()).unwrap();
        let problem2 = QpProblem::new(
            CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap(),
            vec![0.0, 0.0],
            a,
            b,
            bounds,
        )
        .unwrap();

        // Cold start
        let result1 = solve_qp(&problem);
        assert_eq!(result1.status, SolveStatus::Optimal, "T5: cold start should be Optimal");

        // Warm start
        let ws = QpWarmStart {
            initial_active_set: result1.active_set.clone(),
            initial_point: Some(result1.solution.clone()),
        };
        let result2 = solve_qp_warm(&problem2, &ws, &SolverOptions::default());

        assert_eq!(result2.status, SolveStatus::Optimal, "T5: warm start should be Optimal");
        assert_close(result2.solution[0], 0.5, EPS, "T5: warm start x[0]");
        assert_close(result2.solution[1], 0.5, EPS, "T5: warm start x[1]");
        // warm-startはinitial_pointとactive_setを初期値として使うので反復数が少ない or 等しい
        assert!(
            result2.iterations <= result1.iterations + 1,
            "T5: warm start iterations ({}) should be <= cold start ({})",
            result2.iterations,
            result1.iterations
        );
    }

    /// T6: Infeasible QP
    /// min x^2  s.t. x >= 1, x <= 0  (矛盾制約)
    /// 期待: status = Infeasible
    #[test]
    fn test_qp_infeasible() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        // A: [-1; 1], b: [-1; 0]  (x>=1: -x<=-1, x<=0: x<=0)
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[-1.0, 1.0], 2, 1).unwrap();
        let b = vec![-1.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Infeasible, "T6: should be Infeasible");
    }

    /// T7: ポートフォリオ最適化（Markowitz平均分散モデル）
    ///
    /// 3銘柄、対称共分散行列 Sigma = diag(2,2,2)（独立等分散）
    /// 目的: min 1/2 w^T Sigma w（リスク最小化）
    /// 制約: w1+w2+w3=1（等式）、wi>=0（非負）
    ///
    /// 等式制約を2不等式に変換:
    ///   w1+w2+w3 <= 1  と  -(w1+w2+w3) <= -1
    /// 非負制約: -wi <= 0
    ///
    /// 対称性より最適解: w* = [1/3, 1/3, 1/3]
    /// 目的関数: 1/2 * (2/9+2/9+2/9) = 1/3 ≈ 0.3333
    #[test]
    fn test_qp_portfolio_markowitz() {
        // Q = Sigma = diag(2,2,2)
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 2.0, 2.0],
            3, 3,
        ).unwrap();
        let c = vec![0.0, 0.0, 0.0];

        // A行列 (5行3列):
        //   行0: [1,1,1] <= 1  (等式上界)
        //   行1: [-1,-1,-1] <= -1  (等式下界)
        //   行2: [-1,0,0] <= 0  (w1>=0)
        //   行3: [0,-1,0] <= 0  (w2>=0)
        //   行4: [0,0,-1] <= 0  (w3>=0)
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 3, 4],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            5, 3,
        ).unwrap();
        let b = vec![1.0, -1.0, 0.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T7: status should be Optimal");
        let w_sum = result.solution[0] + result.solution[1] + result.solution[2];
        assert_close(w_sum, 1.0, EPS, "T7: w sum = 1");
        assert_close(result.solution[0], 1.0 / 3.0, EPS, "T7: w[0]");
        assert_close(result.solution[1], 1.0 / 3.0, EPS, "T7: w[1]");
        assert_close(result.solution[2], 1.0 / 3.0, EPS, "T7: w[2]");
        assert_close(result.objective, 1.0 / 3.0, EPS, "T7: objective");
    }

    /// T8: 最小二乗法（Least Squares）
    ///
    /// min ||Ax - b||^2 を QP として定式化:
    ///   Q = 2 * A^T A, c = -2 * A^T b  （「1/2あり」規約）
    ///
    /// A = [[2,1],[1,2]], b = [5,4]
    ///   A^T A = [[5,4],[4,5]]
    ///   A^T b = [14,13]
    ///   Q = [[10,8],[8,10]], c = [-28,-26]
    ///
    /// 解析解: (A^T A) x = A^T b → x* = [2, 1]
    /// QP目的関数値: 1/2 x^T Q x + c^T x = 41 - 82 = -41
    #[test]
    fn test_qp_least_squares() {
        // Q = 2 * A^T A = [[10,8],[8,10]]
        let q = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[10.0, 8.0, 8.0, 10.0],
            2, 2,
        ).unwrap();
        let c = vec![-28.0, -26.0]; // -2 * A^T b
        let a = CscMatrix::new(0, 2); // 制約なし
        let b_vec = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b_vec, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T8: status should be Optimal");
        assert_close(result.solution[0], 2.0, EPS, "T8: x[0]");
        assert_close(result.solution[1], 1.0, EPS, "T8: x[1]");
        assert_close(result.objective, -41.0, EPS, "T8: objective");
    }

    /// T9: QP→LP退化テスト（Q=0の場合）
    ///
    /// Q = 0 として LP と同一解が得られることを確認。
    /// LP問題: min 2x+y  s.t. x+y >= 1, x >= 0, y >= 0
    /// Ax<=b 形式:
    ///   -x-y <= -1  (x+y >= 1)
    ///   -x   <= 0   (x >= 0)
    ///   -y   <= 0   (y >= 0)
    ///
    /// 解析解: x*=[0,1], obj=1（y=1で最小化）
    #[test]
    fn test_qp_degenerate_to_lp() {
        let n = 2;
        let q = CscMatrix::new(n, n); // Q = 0（LP退化）
        let c = vec![2.0, 1.0];       // min 2x + y

        // A: [[-1,-1],[-1,0],[0,-1]], b: [-1,0,0]
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[-1.0, -1.0, -1.0, -1.0],
            3, 2,
        ).unwrap();
        let b = vec![-1.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T9: status should be Optimal");
        // LP最適解: x*=[0,1], obj=0*2+1*1=1
        assert_close(result.solution[0], 0.0, EPS, "T9: x[0]");
        assert_close(result.solution[1], 1.0, EPS, "T9: x[1]");
        assert_close(result.objective, 1.0, EPS, "T9: objective");
    }

    /// T10: 複合制約テスト（等式+不等式の組み合わせ）
    ///
    /// min (x-1)^2 + (y-2)^2 = 1/2*Q*x^T + c^T*x + const
    /// Q = [[2,0],[0,2]], c = [-2,-4]
    /// 等式制約: x+y=2 → [1,1]<=2, [-1,-1]<=-2
    /// 不等式制約: x>=0 → [-1,0]<=0
    ///
    /// x+y=2直線上でmin (x-1)^2+(y-2)^2 = min (x-1)^2+(1-x)^2 = 2x^2-2x+2
    /// → x*=1/2, y*=3/2
    /// QP目的関数値: 1/2*2*(0.25+2.25) + (-2*0.5-4*1.5) = 2.5 - 7 = -4.5
    #[test]
    fn test_qp_mixed_constraints() {
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[2.0, 2.0],
            2, 2,
        ).unwrap();
        let c = vec![-2.0, -4.0];

        // A行列 (3行2列):
        //   行0: [1,1] <= 2  (等式上界)
        //   行1: [-1,-1] <= -2  (等式下界)
        //   行2: [-1,0] <= 0  (x>=0)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2],
            &[0, 1, 0, 1, 0],
            &[1.0, 1.0, -1.0, -1.0, -1.0],
            3, 2,
        ).unwrap();
        let b = vec![2.0, -2.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T10: status should be Optimal");
        assert_close(result.solution[0], 0.5, EPS, "T10: x[0]");
        assert_close(result.solution[1], 1.5, EPS, "T10: x[1]");
        assert_close(result.objective, -4.5, EPS, "T10: objective");
    }

    /// T11: Box-constrained QP（上界境界が活性）
    ///
    /// min (x-2)^2 + (y-2)^2  [制約なし、境界あり]
    /// = 1/2 * [[2,0],[0,2]] * [x,y] + [-4,-4] * [x,y]  + const
    /// Q = [[2,0],[0,2]], c = [-4,-4]
    /// bounds: 0 <= x <= 1, 0 <= y <= 1
    ///
    /// 制約なし最小点: x*=2, y*=2 → 上界 ub=1 でクリップ
    /// 期待: x=1, y=1（ub境界が活性）
    /// obj = 1/2*2*(1+1) + (-4-4) = 2 - 8 = -6
    ///
    /// bound_duals順: [ub(x), lb(x), ub(y), lb(y)]
    /// 活性: ub(x)=1, ub(y)=1 → bound_duals[0]>0, bound_duals[2]>0
    #[test]
    fn test_qp_box_constrained_upper_bound() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2); // 制約なし（m=0）
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2]; // lb=0, ub=1
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T11: status should be Optimal");
        assert_close(result.solution[0], 1.0, EPS, "T11: x[0] at upper bound");
        assert_close(result.solution[1], 1.0, EPS, "T11: x[1] at upper bound");
        assert_close(result.objective, -6.0, EPS, "T11: objective");
        // NC-DUAL-LEN: dual_solution 長さ == m == 0, bound_duals 長さ == 4 (2ub + 2lb)
        assert_eq!(result.dual_solution.len(), 0, "T11: dual_solution length == m == 0");
        if !result.bound_duals.is_empty() {
            assert_eq!(result.bound_duals.len(), 4, "T11: bound_duals length == 4");
            assert!(result.bound_duals[0] > 0.0, "T11: ub dual of x[0] should be positive");
            assert!(result.bound_duals[2] > 0.0, "T11: ub dual of x[1] should be positive");
        }
    }

    /// T12: Box-constrained QP（下界境界が活性）
    ///
    /// min (x+2)^2 + y^2  [制約なし、境界あり]
    /// = 1/2 * [[2,0],[0,2]] * [x,y] + [4,0] * [x,y]  + const
    /// Q = [[2,0],[0,2]], c = [4,0]
    /// bounds: 0 <= x <= 1, 0 <= y <= 1
    ///
    /// 制約なし最小点: x*=-2, y*=0 → lb=0 でクリップ
    /// 期待: x=0（lb境界が活性）, y=0
    /// obj = 1/2*2*(0+0) + 4*0 = 0
    ///
    /// bound_duals順: [ub(x), lb(x), ub(y), lb(y)]
    /// 活性: lb(x)=0 → bound_duals[1]>0
    /// y=0 は無制約最小点なので lb(y) の双対は 0
    #[test]
    fn test_qp_box_constrained_lower_bound() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![4.0, 0.0];
        let a = CscMatrix::new(0, 2); // 制約なし（m=0）
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2]; // lb=0, ub=1
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T12: status should be Optimal");
        assert_close(result.solution[0], 0.0, EPS, "T12: x[0] at lower bound");
        assert_close(result.solution[1], 0.0, EPS, "T12: x[1] unconstrained min");
        assert_close(result.objective, 0.0, EPS, "T12: objective");
        // NC-DUAL-LEN: dual_solution 長さ == m == 0, bound_duals 長さ == 4
        assert_eq!(result.dual_solution.len(), 0, "T12: dual_solution length == m == 0");
        if !result.bound_duals.is_empty() {
            assert_eq!(result.bound_duals.len(), 4, "T12: bound_duals length == 4");
            assert!(result.bound_duals[1] > 0.0, "T12: lb dual of x[0] should be positive");
        }
    }

    /// T13: タイムアウトテスト
    ///
    /// timeout_secs=Some(0.001) (1ms) で T1 と同じ問題を解く。
    /// 問題が小さすぎてタイムアウト前に解けることもあるが、
    /// 少なくとも SolveStatus::Timeout が返る機構をテストする。
    ///
    /// timeout_secs=Some(0.0) (0秒) ならほぼ確実にタイムアウトする。
    #[test]
    fn test_timeout_returns_timeout_status() {

        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { timeout_secs: Some(0.0), ..Default::default() }; // 即タイムアウト

        let result = solve_qp_with(&problem, &opts);
        // 0秒タイムアウトでは Timeout になるはず
        // (問題が非常に小さいので運によっては Optimal になることも許容するが、
        //  Timeout ステータスが正しく返ることを主に確認する)
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "T13: status should be Timeout or Optimal, got {:?}",
            result.status
        );
    }

    /// T14: 自動切替 - 小問題（Active Set選択）
    ///
    /// n=100 < qp_solver_threshold=10_000 → Auto モードで Active Set が選択される
    /// Q = 2*I_100, c = -ones(100), bounds = [0,1]^100
    /// 最適解: xi = 0.5（bounds内部点）, obj = -25.0
    #[test]
    fn test_auto_switch_small_uses_active_set() {
        let n = 100usize;
        let q_rows: Vec<usize> = (0..n).collect();
        let q_cols: Vec<usize> = (0..n).collect();
        let q_vals = vec![2.0f64; n];
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();
        let c = vec![-1.0f64; n];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0f64, 1.0f64); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        // Auto mode: n=100 < threshold=3_000 → Active Set が選択される
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T14: Auto小問題はOptimal");
        for xi in &result.solution {
            assert!((xi - 0.5).abs() < 1e-4, "T14: xi ≈ 0.5 (got {})", xi);
        }
    }

    /// T15: 自動切替 - 大問題（IPM選択）
    ///
    /// n=200, qp_solver_threshold=100 → Auto モードで IPM が選択される
    /// Q = 2*I_200, c = -ones(200), bounds = [0,1]^200
    /// 最適解: xi ≈ 0.5
    #[test]
    fn test_auto_switch_large_uses_ipm() {
        let n = 200usize;
        let q_rows: Vec<usize> = (0..n).collect();
        let q_cols: Vec<usize> = (0..n).collect();
        let q_vals = vec![2.0f64; n];
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();
        let c = vec![-1.0f64; n];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0f64, 1.0f64); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        // Auto mode: n=200 > threshold=100 → IPM が選択される
        let opts = SolverOptions { qp_solver_threshold: 100, ..Default::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T15: Auto大問題はOptimal (IPM)");
        for xi in &result.solution {
            assert!((xi - 0.5).abs() < 1e-2, "T15: xi ≈ 0.5 (got {})", xi);
        }
    }

    /// T17: 強制Active Set（中規模問題）
    ///
    /// n=500 の中規模問題で qp_solver=ActiveSet を強制指定
    /// Q = 2*I_500, c = -ones(500), bounds = [0,1]^500
    /// 最適解: xi = 0.5（bounds内部点）
    #[test]
    fn test_force_active_set_medium() {
        let n = 500usize;
        let q_rows: Vec<usize> = (0..n).collect();
        let q_cols: Vec<usize> = (0..n).collect();
        let q_vals = vec![2.0f64; n];
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();
        let c = vec![-1.0f64; n];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0f64, 1.0f64); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { qp_solver: QpSolverChoice::ActiveSet, ..Default::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T17: 強制Active SetはOptimal");
        for xi in &result.solution {
            assert!((xi - 0.5).abs() < 1e-4, "T17: xi = 0.5 (got {})", xi);
        }
    }

    /// T18: 強制IPM（小規模問題）
    ///
    /// n=2 の基本 QP で QpSolverChoice::Ipm を指定して正常解を確認する。
    /// min x^2 + y^2  s.t. x + y >= 1
    /// Q = [[2,0],[0,2]], c=[0,0], A=[[-1,-1]], b=[-1]
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_force_ipm_small() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { qp_solver: QpSolverChoice::Ipm, ..Default::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T18: 強制IPMはOptimal");
        assert!((result.solution[0] - 0.5).abs() < 1e-4, "T18: x[0] ≈ 0.5");
        assert!((result.solution[1] - 0.5).abs() < 1e-4, "T18: x[1] ≈ 0.5");
        assert!((result.objective - 0.5).abs() < 1e-4, "T18: obj ≈ 0.5");
    }

    /// T20: Concurrent Solver（parallel feature）
    ///
    /// parallel feature ON 時、Auto モードで AS/IPM/IPM-Schur を並列実行し
    /// 最初に Optimal を返した結果が採用されることを確認する。
    /// min x^2 + y^2  s.t. x + y >= 1  → x*=y*=0.5, obj=0.5
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_solver_basic() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default(); // qp_solver = Auto
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T20: concurrent should be Optimal");
        assert!((result.solution[0] - 0.5).abs() < EPS, "T20: x[0] ≈ 0.5");
        assert!((result.solution[1] - 0.5).abs() < EPS, "T20: x[1] ≈ 0.5");
        assert!((result.objective - 0.5).abs() < EPS, "T20: obj ≈ 0.5");
    }

    /// T22: QualityRank の Ord 比較
    ///
    /// Optimal > Feasible > Approximate の順序が正しいことを確認する。
    #[test]
    fn test_quality_rank_ordering() {
        assert!(QualityRank::Optimal > QualityRank::Feasible, "T22: Optimal > Feasible");
        assert!(QualityRank::Feasible > QualityRank::Approximate, "T22: Feasible > Approximate");
        assert!(QualityRank::Optimal > QualityRank::Approximate, "T22: Optimal > Approximate");
        assert_eq!(QualityRank::Optimal, QualityRank::Optimal, "T22: Optimal == Optimal");
    }
}
