//! 二次計画法（QP）ソルバーモジュール
//!
//! IPM（内点法）および IPM-Schur complement による QP ソルバーを提供する。
//! 問題形式: min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub
//!
//! # 規約
//! **「1/2あり」規約** (OSQP/qpOASES標準):
//! - 目的関数: min 1/2 x^T Q x + c^T x
//! - ∇f(x) = Qx + c
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

mod problem;
pub mod ipm;
pub mod diagnose;
pub use problem::{QpProblem, QpWarmStart};
pub use diagnose::{diagnose, DiagnosticReport, DiagnosticWarning, DiagnosticCode, Severity, ProblemInfo};
pub use crate::problem::SolverResult;
pub use ipm::solve_qp_ipm;

use crate::options::{QpSolverChoice, SolverOptions};
use crate::presolve::{run_qp_presolve_phase1, run_qp_presolve_phase2, postsolve_qp};
use crate::presolve::qp_transforms::QpPresolveStatus;
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::backend::{LpBackend, SimplexBackend};
use crate::sparse::CscMatrix;
use crate::tolerances::PIVOT_TOL;

/// Concurrent Solver が複数ソルバーの結果を比較するための解品質ランク
///
/// 順序: Optimal > Feasible > Approximate
/// `PartialOrd/Ord` を実装することで `>` による比較が可能。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityRank {
    /// 近似解（eps緩和・Timeout後の途中解など）
    Approximate = 0,
    /// 実行可能解（制約は充足するが最適性は保証されていない）
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
/// IPM の各ソルバーは `QpSolver` を実装しており、
/// `Box<dyn QpSolver>` として統一的に扱うことができる。
///
/// # 例
/// ```rust,no_run
/// use solver::qp::{QpProblem, QpSolver, IpmSolver};
/// use solver::options::SolverOptions;
/// # let problem = unimplemented!();
/// let solver = IpmSolver;
/// let result = solver.solve(&problem, &SolverOptions::default());
/// ```
pub trait QpSolver {
    /// QP 問題を解く
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult;
    /// ソルバー名を返す
    fn name(&self) -> &'static str;
}

/// IPM（内点法）QP ソルバー
///
/// `QpSolver` trait を実装する。内部で [`ipm::solve_qp_ipm`] を呼ぶ。
pub struct IpmSolver;

impl QpSolver for IpmSolver {
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult {
        ipm::solve_qp_ipm(problem, options)
    }
    fn name(&self) -> &'static str {
        "IPM"
    }
}

/// IPM + IPM-Schur を並列実行し、最初に Optimal を返したものを採用する
///
/// parallel feature ON 時のみコンパイルされる。
/// 各スレッドは共有 `cancel_flag` を監視し、勝者決定後に停止する。
///
/// # Timeout accuracy
/// The actual elapsed time may exceed `timeout_secs` by at most one LDL
/// factorization step. For typical QP problems this overhead is negligible,
/// but for very large problems (n > 100_000) the overhead may reach tens of
/// seconds. This is consistent with other solvers (Gurobi, Clarabel, OSQP)
/// which also check timeout at iteration boundaries.
#[cfg(feature = "parallel")]
fn solve_qp_concurrent(
    problem: &QpProblem,
    options: &SolverOptions,
) -> SolverResult {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let problem_arc = Arc::new(problem.clone());
    let (tx, rx) = mpsc::sync_channel::<SolverResult>(4);
    let mut handles = Vec::with_capacity(3);

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

    // IPM-Nyström スレッド（cmd_295: ランダム化前処理付き CG）
    {
        let cancel = Arc::clone(&cancel_flag);
        let prob = Arc::clone(&problem_arc);
        let mut opts = options.clone();
        opts.cancel_flag = Some(cancel);
        let tx = tx.clone();
        handles.push(std::thread::spawn(move || {
            let r = ipm::solve_qp_ipm_nystrom(&prob, &opts);
            let _ = tx.send(r);
        }));
    }

    drop(tx); // 全スレッドが tx を drop するまで rx.iter() は終了しない

    // 残り時間を計算（deadline が設定されている場合はその時点まで、なければ十分大きい値）
    let deadline = options.deadline.or_else(|| {
        options.timeout_secs.map(|s| std::time::Instant::now() + std::time::Duration::from_secs_f64(s))
    });

    // 解の品質ランク（Optimal > Feasible > Approximate）で最良解を選択する。
    // cancel_flag は Optimal 到着時のみ立てる。
    // recv_timeout で deadline を超えたら受信ループを打ち切る。
    let mut best_ranked: Option<(QualityRank, SolverResult)> = None;
    let mut fallback: Option<SolverResult> = None;
    let mut timed_out = false;
    loop {
        let remaining = deadline.map(|d| {
            let now = std::time::Instant::now();
            if d > now { d - now } else { std::time::Duration::ZERO }
        }).unwrap_or(std::time::Duration::from_secs(3600));

        match rx.recv_timeout(remaining) {
            Ok(result) => {
                let rank = quality_rank_of(&result);
                if let Some(r) = rank {
                    if r == QualityRank::Optimal
                        && best_ranked.as_ref().map(|(br, _)| *br) != Some(QualityRank::Optimal)
                    {
                        cancel_flag.store(true, Ordering::Relaxed);
                    }
                    let should_update = best_ranked.as_ref().map(|(br, _)| r > *br).unwrap_or(true);
                    if should_update {
                        best_ranked = Some((r, result));
                    }
                } else if result.status == SolveStatus::Infeasible || fallback.is_none() {
                    fallback = Some(result);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // deadline 到達: cancel して残りスレッドを止める
                cancel_flag.store(true, Ordering::Relaxed);
                timed_out = true;
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // 全スレッド完了
                break;
            }
        }
    }

    if !timed_out {
        for h in handles {
            let _ = h.join();
        }
    }
    // On timeout: handles are dropped here, detaching threads.
    // Threads will self-terminate when checking cancel_flag at the
    // next iteration boundary. This is safe because cancel_flag is
    // already set to true before we reach here.
    // NOTE: timeout accuracy = at most 1 LDL factorization extra
    //       (typically < 1s for small/medium problems, up to tens of
    //        seconds for very large problems n>100k).

    let best = match best_ranked {
        Some((_, result)) => Some(result),
        None => fallback,
    };
    best.unwrap_or_else(|| {
        if timed_out {
            SolverResult {
                status: SolveStatus::Timeout,
                objective: f64::INFINITY,
                solution: vec![],
                dual_solution: vec![],
                bound_duals: vec![],
                active_set: vec![],
                iterations: 0,
                ..Default::default()
            }
        } else {
            SolverResult {
                status: SolveStatus::NumericalError,
                objective: f64::NAN,
                solution: vec![0.0; problem.num_vars],
                dual_solution: vec![],
                bound_duals: vec![],
                active_set: vec![],
                iterations: 0,
                ..Default::default()
            }
        }
    })
}

/// Q=0 退化ケース（LP 問題）を LP ソルバーに委譲して QP 結果に変換する
fn solve_as_lp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    let ct = vec![ConstraintType::Le; m];
    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        ct,
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return SolverResult::infeasible(),
    };

    let result = SimplexBackend.solve(&lp, options);
    match result.status {
        SolveStatus::Optimal => {
            let x = result.solution.clone();
            let obj = problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
            let active: Vec<usize> = (0..m)
                .filter(|&i| {
                    let ax_i: f64 = (0..n)
                        .map(|j| get_a_element(&problem.a, i, j) * x[j])
                        .sum();
                    (ax_i - problem.b[i]).abs() < PIVOT_TOL
                })
                .collect();
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution: x,
                dual_solution: result.dual_solution,
                bound_duals: vec![],
                active_set: active,
                iterations: 0,
                ..Default::default()
            }
        }
        SolveStatus::Infeasible => SolverResult::infeasible(),
        SolveStatus::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
            ..Default::default()
        },
        SolveStatus::MaxIterations => SolverResult::numerical_error(),
        SolveStatus::Timeout => SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
            ..Default::default()
        },
        SolveStatus::NumericalError => SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
            ..Default::default()
        },
    }
}

/// 行列 A の (row, col) 要素を返す
fn get_a_element(a: &CscMatrix, row: usize, col: usize) -> f64 {
    let start = a.col_ptr[col];
    let end = a.col_ptr[col + 1];
    for k in start..end {
        if a.row_ind[k] == row {
            return a.values[k];
        }
    }
    0.0
}

/// QP ソルバーをディスパッチする内部関数
///
/// `options.qp_solver` に基づいてソルバーを選択する。
///
/// - `Ipm`: 強制 IPM（内点法）
/// - `IpmSchur`: 強制 IPM Schur complement パス
/// - `Concurrent`:
///   - parallel feature ON → IPM/IPM-Schur を並列実行（`solve_qp_concurrent`）
///   - parallel feature OFF → IPM
///
/// Q=0 の場合は LP ソルバーに委譲する。
fn dispatch_qp(
    problem: &QpProblem,
    options: &SolverOptions,
) -> SolverResult {
    // Q=0 退化ケース（LP 問題）: LP ソルバーに委譲
    if problem.is_zero_q() {
        return solve_as_lp(problem, options);
    }

    match options.qp_solver {
        QpSolverChoice::Ipm => ipm::solve_qp_ipm(problem, options),
        QpSolverChoice::IpmSchur => ipm::solve_qp_ipm_schur(problem, options),
        QpSolverChoice::IpmNystrom => ipm::solve_qp_ipm_nystrom(problem, options),
        QpSolverChoice::Concurrent => {
            #[cfg(feature = "parallel")]
            {
                solve_qp_concurrent(problem, options)
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
                ipm::solve_qp_ipm(problem, opts)
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
/// [`SolverResult`] — ステータス・目的関数値・解・ラグランジュ乗数・反復数
pub fn solve_qp(problem: &QpProblem) -> SolverResult {
    solve_qp_with(problem, &SolverOptions::default())
}

/// QPをカスタム設定で解く
///
/// qpOASESの `init()` に相当。timeout が反復制御の主ガード。
pub fn solve_qp_with(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let presolve_result = if options.presolve {
        let phase1 = run_qp_presolve_phase1(problem, options);
        run_qp_presolve_phase2(phase1, options)
    } else {
        crate::presolve::QpPresolveResult::no_reduction(problem)
    };
    if presolve_result.presolve_status == QpPresolveStatus::Infeasible {
        return crate::problem::SolverResult::infeasible();
    }
    let opts_no_ruiz;
    let dispatch_opts: &SolverOptions = if presolve_result.ruiz_scaler.is_some() {
        opts_no_ruiz = SolverOptions { use_ruiz_scaling: false, ..options.clone() };
        &opts_no_ruiz
    } else {
        options
    };
    let mut reduced_sol = dispatch_qp(&presolve_result.reduced, dispatch_opts);
    if let Some(ref scaler) = presolve_result.ruiz_scaler {
        let (x, y) = scaler.unscale_solution(&reduced_sol.solution, &reduced_sol.dual_solution);
        reduced_sol.solution = x;
        reduced_sol.dual_solution = y;
        if scaler.c.abs() > 1e-300 { reduced_sol.objective /= scaler.c; }
    }
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
/// qpOASESの `hotstart()` に相当。SQP反復で前回解の情報を引き継ぐ場合に使用。
/// IPM は warm_start の `initial_point` を初期値のヒントとして利用できるが、
/// `initial_active_set` は無視される。
pub fn solve_qp_warm(
    problem: &QpProblem,
    _warm_start: &QpWarmStart,
    options: &SolverOptions,
) -> SolverResult {
    // IPM は warm_start 未対応のため通常の solve_qp_with に委譲する
    solve_qp_with(problem, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::QpSolverChoice;
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    // concurrent solver での許容誤差（IPM/IPM-Schur を並列実行）
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
        assert!(result.bound_duals.is_empty(), "T1: infinite bounds → bound_duals empty");
        assert_eq!(result.dual_solution.len(), 1, "T1: dual_solution length == m == 1");
    }

    /// T2: 等式制約付きQP
    /// min x^2+y^2 (1/2あり規約: Q=2I)  s.t. x+y=1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_qp_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
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
    /// 期待: obj=0（(0,0)が最小）
    #[test]
    fn test_qp_degenerate_lp_case() {
        let n = 2;
        let q = CscMatrix::new(n, n); // Q = 0
        let c = vec![1.0, 2.0];
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
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T4: status should be Optimal");
        assert_close(result.solution[0], 3.0, EPS, "T4: x[0]");
        assert_close(result.solution[1], 4.0, EPS, "T4: x[1]");
        assert_close(result.objective, -25.0, EPS, "T4: objective");
    }

    /// T5: warm-start整合性
    /// T1と同じ問題を2回解く（2回目はwarm-start）
    /// IPMはwarm-startを無視するため同一解が返ることを確認する
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

        let result1 = solve_qp(&problem);
        assert_eq!(result1.status, SolveStatus::Optimal, "T5: cold start should be Optimal");

        let ws = crate::qp::QpWarmStart {
            initial_active_set: result1.active_set.clone(),
            initial_point: Some(result1.solution.clone()),
        };
        let result2 = solve_qp_warm(&problem2, &ws, &SolverOptions::default());

        assert_eq!(result2.status, SolveStatus::Optimal, "T5: warm start should be Optimal");
        assert_close(result2.solution[0], 0.5, EPS, "T5: warm start x[0]");
        assert_close(result2.solution[1], 0.5, EPS, "T5: warm start x[1]");
    }

    /// T6: Infeasible QP
    /// min x^2  s.t. x >= 1, x <= 0  (矛盾制約)
    /// 期待: status = Infeasible
    #[test]
    fn test_qp_infeasible() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[-1.0, 1.0], 2, 1).unwrap();
        let b = vec![-1.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Infeasible, "T6: should be Infeasible");
    }

    /// T7: ポートフォリオ最適化（Markowitz平均分散モデル）
    #[test]
    fn test_qp_portfolio_markowitz() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 2.0, 2.0],
            3, 3,
        ).unwrap();
        let c = vec![0.0, 0.0, 0.0];
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
    #[test]
    fn test_qp_least_squares() {
        let q = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[10.0, 8.0, 8.0, 10.0],
            2, 2,
        ).unwrap();
        let c = vec![-28.0, -26.0];
        let a = CscMatrix::new(0, 2);
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
    #[test]
    fn test_qp_degenerate_to_lp() {
        let n = 2;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 1.0];
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
        assert_close(result.solution[0], 0.0, EPS, "T9: x[0]");
        assert_close(result.solution[1], 1.0, EPS, "T9: x[1]");
        assert_close(result.objective, 1.0, EPS, "T9: objective");
    }

    /// T10: 複合制約テスト（等式+不等式の組み合わせ）
    #[test]
    fn test_qp_mixed_constraints() {
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[2.0, 2.0],
            2, 2,
        ).unwrap();
        let c = vec![-2.0, -4.0];
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
    #[test]
    fn test_qp_box_constrained_upper_bound() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T11: status should be Optimal");
        assert_close(result.solution[0], 1.0, EPS, "T11: x[0] at upper bound");
        assert_close(result.solution[1], 1.0, EPS, "T11: x[1] at upper bound");
        assert_close(result.objective, -6.0, EPS, "T11: objective");
    }

    /// T12: Box-constrained QP（下界境界が活性）
    #[test]
    fn test_qp_box_constrained_lower_bound() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![4.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T12: status should be Optimal");
        assert_close(result.solution[0], 0.0, EPS, "T12: x[0] at lower bound");
        assert_close(result.solution[1], 0.0, EPS, "T12: x[1] unconstrained min");
        assert_close(result.objective, 0.0, EPS, "T12: objective");
    }

    /// T13: タイムアウトテスト
    #[test]
    fn test_timeout_returns_timeout_status() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { timeout_secs: Some(0.0), ..Default::default() };

        let result = solve_qp_with(&problem, &opts);
        assert!(
            result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
            "T13: status should be Timeout or Optimal, got {:?}",
            result.status
        );
    }

    /// T18: 強制IPM（小規模問題）
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
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_solver_basic() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T20: concurrent should be Optimal");
        assert!((result.solution[0] - 0.5).abs() < EPS, "T20: x[0] ≈ 0.5");
        assert!((result.solution[1] - 0.5).abs() < EPS, "T20: x[1] ≈ 0.5");
        assert!((result.objective - 0.5).abs() < EPS, "T20: obj ≈ 0.5");
    }

    /// T22: QualityRank の Ord 比較
    #[test]
    fn test_quality_rank_ordering() {
        assert!(QualityRank::Optimal > QualityRank::Feasible, "T22: Optimal > Feasible");
        assert!(QualityRank::Feasible > QualityRank::Approximate, "T22: Feasible > Approximate");
        assert!(QualityRank::Optimal > QualityRank::Approximate, "T22: Optimal > Approximate");
        assert_eq!(QualityRank::Optimal, QualityRank::Optimal, "T22: Optimal == Optimal");
    }
}
