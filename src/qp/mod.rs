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
//! let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
//! let result = solve_qp(&problem);
//! // result.solution ≈ [0.5, 0.5], result.objective ≈ 0.5
//! ```

mod problem;
pub mod ipm;
pub mod diagnose;
mod refine;
pub use problem::{QpProblem, QpWarmStart};
pub use diagnose::{diagnose, DiagnosticReport, DiagnosticWarning, DiagnosticCode, Severity, ProblemInfo};
pub use crate::problem::SolverResult;
pub use ipm::solve_qp_ipm;

use crate::options::{QpSolverChoice, SolverOptions};
use crate::presolve::{run_qp_presolve_phase1, run_qp_presolve_phase2, postsolve_qp};
use crate::presolve::qp_transforms::QpPresolveStatus;
use crate::problem::{LpProblem, SolveStatus};
use crate::backend::{LpBackend, SimplexBackend};
use crate::sparse::CscMatrix;


/// Concurrent Solver が複数ソルバーの結果を比較するための解品質ランク
///
/// 順序: Optimal > Feasible > Approximate
/// `PartialOrd/Ord` を実装することで `>` による比較が可能。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityRank {
    /// 近似解（eps緩和・Timeout後の途中解など）
    Approximate = 0,
    /// 実行可能解（制約は充足するが最適性は保証されていない）
    Feasible = 1,
    /// 最適解（KKT条件を充足）
    Optimal = 2,
}

/// 両Optimal時の残差比較スコアを計算する（小さいほど良い解）。
///
/// score = max(pfeas/(1+norm_b), dfeas/(1+norm_c))
/// final_residualsがNoneの場合はf64::INFINITYを返す（最低優先）。
#[cfg(feature = "parallel")]
fn residual_score(result: &SolverResult, problem: &QpProblem) -> f64 {
    match result.final_residuals {
        Some((pfeas, dfeas, _gap)) => {
            let norm_b = problem.b.iter().fold(0.0_f64, |a, &bi| a.max(bi.abs())).max(1.0);
            let norm_c = problem.c.iter().fold(0.0_f64, |a, &ci| a.max(ci.abs())).max(1.0);
            let pfeas_norm = pfeas / (1.0 + norm_b);
            let dfeas_norm = dfeas / (1.0 + norm_c);
            pfeas_norm.max(dfeas_norm)
        }
        None => f64::INFINITY,
    }
}

#[cfg(feature = "parallel")]
fn quality_rank_of(result: &SolverResult) -> Option<QualityRank> {
    match result.status {
        SolveStatus::Optimal => Some(QualityRank::Optimal),
        SolveStatus::Timeout if !result.solution.is_empty() => {
            Some(QualityRank::Approximate)
        }
        _ => None, // Infeasible / Unbounded / NumericalError / SuboptimalSolution / その他
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
        let mut forced_opts = options.clone();
        forced_opts.qp_solver = QpSolverChoice::Ipm; // IpmSolverの意味論を維持
        solve_qp_with(problem, &forced_opts)
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
/// # cancel_flag の契約
/// `options.cancel_flag` が指定されている場合、Optimal 検出時・deadline 到達時に
/// `cancel_flag.store(true, Relaxed)` が呼ばれる。同一 flag を複数問題に使い回す場合は、
/// 各呼び出し前に `cancel_flag.store(false, Relaxed)` でリセットするか、
/// 問題ごとに新しい `Arc<AtomicBool>` を生成すること。
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

    // 外部 cancel_flag がある場合は Arc::clone で共有（BUG-CONC-001修正）。
    // 外部 cancel_flag がない場合は従来通り内部で新規作成。
    // Note: Optimal検出時・deadline到達時に store(true) されるため、同一 flag を
    // 複数回の solve_qp_with 呼び出しに使い回す場合は各呼び出し前にリセットすること。
    let cancel_flag = options.cancel_flag
        .as_ref()
        .map(Arc::clone)
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    // Arc::new(problem.clone()) は不要: thread::scope により参照渡しが可能
    let (tx, rx) = mpsc::sync_channel::<SolverResult>(4);

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

    // thread::scope を使用: スコープ終了時に全スレッドが確実にjoinされる。
    // タイムアウト時も detach せずjoin → LDL因子化終了後にメモリが解放される。
    // これにより 138問バッチ処理でのメモリ蓄積（旧実装: ~500-630MB/問）を解消する。
    std::thread::scope(|s| {
        // IPM スレッド
        {
            let cancel = Arc::clone(&cancel_flag);
            let mut opts = options.clone();
            opts.cancel_flag = Some(cancel);
            let tx = tx.clone();
            s.spawn(move || {
                let r = ipm::solve_qp_ipm(problem, &opts);
                let _ = tx.send(r);
            });
        }

        // IP-PMM 独立実装スレッド
        {
            let cancel = Arc::clone(&cancel_flag);
            let mut opts = options.clone();
            opts.cancel_flag = Some(cancel);
            let tx = tx.clone();
            s.spawn(move || {
                let r = ipm::solve_qp_ippmm(problem, &opts);
                let _ = tx.send(r);
            });
        }

        drop(tx); // 全スレッドが tx を drop するまで rx.recv_timeout が Disconnected を返さない

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
                        let should_update = match &best_ranked {
                            None => true,
                            Some((br, prev_result)) => {
                                if r > *br {
                                    true
                                } else if r == *br {
                                    // 同ランク: 正規化最大残差で比較（小さい方が良い）
                                    // score差 < 1e-12 なら先着を維持（ケース5: 決定論性保証）
                                    let score_new = residual_score(&result, problem);
                                    let score_prev = residual_score(prev_result, problem);
                                    score_new < score_prev - 1e-12
                                } else {
                                    false
                                }
                            }
                        };
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
        // スコープ終了: 全スレッドが自動joinされる（timeout時も同様）
        // NOTE: timeout accuracy = at most 1 LDL factorization extra
        //       (typically < 1s for small/medium problems, up to tens of
        //        seconds for very large problems n>100k).
    });

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

                iterations: 0,
                ..Default::default()
            }
        } else {
            SolverResult {
                status: SolveStatus::NumericalError,
                objective: f64::NAN,
                solution: vec![],
                dual_solution: vec![],
                bound_duals: vec![],

                iterations: 0,
                ..Default::default()
            }
        }
    })
}

/// Q=0 退化ケース（LP 問題）を LP ソルバーに委譲して QP 結果に変換する
fn solve_as_lp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    // Eq/Ge/Le制約型をそのままSimplexに渡す（設計書§2.4）。
    // Simplexは ConstraintType::Eq を Phase I 人工変数で正しく処理する。
    // 旧実装の to_all_le() + 全Le方式は Eq→2Le展開で同一係数行が生まれ、
    // 基底の数値的不安定を引き起こすため廃止。
    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        problem.constraint_types.clone(),
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
            // 双対解はSimplex出力をそのまま使用（展開なし）
            let dual = result.dual_solution.clone();
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution: x,
                dual_solution: dual,
                reduced_costs: result.reduced_costs.clone(),
                slack: result.slack.clone(),
                warm_start_basis: result.warm_start_basis.clone(),
                bound_duals: vec![],
                iterations: result.iterations,
                solver_used: None,
                final_residuals: None,
                pfeas: None,
                dfeas: None,
                gap: None,
            }
        }
        SolveStatus::Infeasible => SolverResult::infeasible(),
        SolveStatus::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
        },
        SolveStatus::MaxIterations => {
            // DEAD PATH: SimplexOutcome::MaxIterations廃止（cmd_595）により到達不能。
            // SolveStatus enum variant自体は未削除（別cmd対応）。
            unreachable!("MaxIterations is dead code - not reachable via simplex path")
        }
        SolveStatus::SuboptimalSolution => {
            // DEAD PATH: SuboptimalSolution is not reachable via current simplex implementation
            SolverResult::numerical_error()
        }
        SolveStatus::Timeout => SolverResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
        },
        SolveStatus::NumericalError => SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
        },
        SolveStatus::NonConvex(_) => SolverResult {
            status: result.status,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            bound_duals: vec![],
            iterations: 0,
            solver_used: None,
            final_residuals: None,
            pfeas: None,
            dfeas: None,
            gap: None,
        },
    }
}

/// Q行列が正半定値かどうかを確認する。
///
/// Q + epsilon*I を密行列コレスキー分解で確認する。
/// 全ピボットが正なら PSD、負のピボットが出た時点で false を返す（非凸QP）。
/// n > CHECK_SIZE_LIMIT の場合は高コストを避けてスキップ（true を返す）。
/// Q は上三角CSC形式で渡す（IPMの内部表現と同じ）。
///
/// # Limitations
/// - n > 1000 の問題はスキップし true（PSD扱い）を返す。
///   大規模非凸QPを検出できない可能性がある。
///   現在のQPLIBターゲット問題は n ≤ 500 のためこのケースは存在しない。
///
/// # 既知制限
/// 対角チェックは対角負値（Q[i,i] < -1e-10）のみ検出する。
/// 対角全正の不定行列はn≤1000ではCholeskyで検出、n>1000では未検出（既知制限）。
pub(crate) fn check_q_positive_semidefinite(q: &CscMatrix) -> bool {
    let n = q.nrows;
    if n == 0 {
        return true;
    }

    // ★ 追加: 対角チェック (O(nnz), サイズ非依存)
    // 対角に負値があれば → 非PSD確定（十分条件）
    // 上三角CSCなのでrow == colが対角要素
    // 閾値: < -1e-10（数値ノイズ -1e-15程度を除外）
    // ★ SAFETY: この閾値は慎重に設定。変更要注意。
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col && q.values[k] < -1e-10 {
                return false;  // 対角負値 → 非PSD確定
            }
        }
    }

    // n > 1000: Cholesky分解はO(n³)コスト過大のため省略を維持
    // （対角チェック済みのため、対角負値は検出完了）
    const CHECK_SIZE_LIMIT: usize = 1000;
    if n > CHECK_SIZE_LIMIT {
        return true;
    }

    // eps: double precision 機械イプシロン (~2e-16) の約10^8倍。
    // PSD境界判定の数値的余裕として設定。半正定値Q（最小固有値=0）は
    // Q+eps*I の最小ピボット = eps > 0 → PSD判定される設計。
    let eps = 1e-8_f64;

    // 上三角CSC から密対称行列を構築（Q + eps*I）
    let mut a = vec![0.0f64; n * n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                let v = q.values[k];
                a[row * n + col] = v;
                if row != col {
                    a[col * n + row] = v;
                }
            }
        }
    }
    for i in 0..n {
        a[i * n + i] += eps;
    }

    // 密コレスキー L L^T 分解。負のピボットが出たら non-PSD。
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let sqrt_d = d.sqrt();
        a[j * n + j] = sqrt_d;
        for i in (j + 1)..n {
            let mut l_ij = a[i * n + j];
            for k in 0..j {
                l_ij -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = l_ij / sqrt_d;
        }
    }
    true
}

/// QP ソルバーをディスパッチする内部関数
///
/// `options.qp_solver` に基づいてソルバーを選択する。
///
/// - `Ipm`: 強制 IPM（内点法）
/// - `Concurrent`:
///   - parallel feature ON → IPM/IPM-PMM を並列実行（`solve_qp_concurrent`）
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

    // Q不定値チェック（非凸QP検出）: IPMはQ正半定値を前提とするため早期終了
    if !check_q_positive_semidefinite(&problem.q) {
        return SolverResult {
            status: SolveStatus::NonConvex(
                "Q matrix is indefinite (non-convex QP). IPM requires Q to be positive semidefinite.".to_string()
            ),
            ..Default::default()
        };
    }

    match options.qp_solver {
        QpSolverChoice::Ipm => ipm::solve_qp_ipm(problem, options),
        QpSolverChoice::IpPmmNew => ipm::solve_qp_ippmm(problem, options),
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

/// Eq/Ge制約のdual逆変換ヘルパー
///
/// IPM内部で `to_all_le()` 展開された dual を元制約空間に折り畳む。
/// - Le:  expanded[rows[0]] そのまま
/// - Ge:  -expanded[rows[0]] （to_all_le で符号反転されているため）
/// - Eq:  expanded[rows[0]] - expanded[rows[1]]
///
/// 展開後のサイズと dual_expanded のサイズが一致しない場合はそのまま返す。
pub(crate) fn collapse_le_expansion_dual(
    dual_expanded: &[f64],
    le_map: &crate::qp::problem::LeExpansionMap,
    orig_types: &[crate::problem::ConstraintType],
) -> Vec<f64> {
    use crate::problem::ConstraintType;
    let m_orig = orig_types.len();
    let total_expanded: usize = le_map.original_to_expanded.iter().map(|rows| rows.len()).sum();
    if dual_expanded.len() < total_expanded {
        // サイズ不一致: フォールバックとして直接使用
        return dual_expanded.to_vec();
    }
    let mut collapsed = vec![0.0f64; m_orig];
    for (i, (ct, rows)) in orig_types.iter().zip(le_map.original_to_expanded.iter()).enumerate() {
        collapsed[i] = match ct {
            ConstraintType::Le => dual_expanded[rows[0]],
            ConstraintType::Ge => -dual_expanded[rows[0]],
            ConstraintType::Eq => {
                let mu1 = dual_expanded[rows[0]];
                let mu2 = if rows.len() > 1 { dual_expanded[rows[1]] } else { 0.0 };
                mu1 - mu2
            }
        };
    }
    collapsed
}

/// API境界でSuboptimalSolutionをOptimalまたはTimeoutに変換する
///
/// solve_qp_withの全returnパスから呼び出す。内部ではSubを保持し、ここで最終変換を行う。
/// - Sub（有効解あり）→ Optimal
/// - Sub（精度未達）→ Timeout（解なし）
/// - その他のステータス → 変換なし（パススルー）
fn apply_api_boundary_conversion(
    result: SolverResult,
    problem: &QpProblem,
    opts: &SolverOptions,
) -> SolverResult {
    if result.status != SolveStatus::SuboptimalSolution {
        return result;
    }
    let eps = opts.ipm_eps();
    let verified = ipm::post_verify_solution(result, problem, eps);
    if verified.status == SolveStatus::Optimal {
        verified
    } else {
        // Sub（精度未達）→ Timeout
        SolverResult {
            status: SolveStatus::Timeout,
            solution: vec![],
            ..verified
        }
    }
}

/// QPをカスタム設定で解く
///
/// qpOASESの `init()` に相当。timeout が反復制御の主ガード。
///
/// # cancel_flag の注意事項
/// `Concurrent` ソルバー（`parallel` feature 有効時のみ）では、`options.cancel_flag`
/// を指定した場合、内部で `store(true)` される可能性がある（Optimal検出時・deadline到達時）。
/// 同一 flag を複数の `solve_qp_with` 呼び出しに使い回す場合は、各呼び出し前に
/// `cancel_flag.store(false, Relaxed)` でリセットするか、問題ごとに新しい
/// `Arc<AtomicBool>` を生成すること。他のソルバーモードでは flag の書き込みは行われない。
pub fn solve_qp_with(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let start_time = std::time::Instant::now();
    let mut current_opts = options.clone();
    let mut first_result: Option<SolverResult> = None; // C-1: Ruiz無し再ソルブ失敗時フォールバック用
    loop {
        // 大規模問題（n+m > 50_000）のみpresolveをスレッド化してdeadlineで打ち切る。
        // 小規模問題はpresolveが速いためスレッドオーバーヘッドを避ける。
        let presolve_result = if current_opts.presolve {
            const PRESOLVE_THREAD_THRESHOLD: usize = 50_000;
            let use_thread = problem.num_vars + problem.num_constraints > PRESOLVE_THREAD_THRESHOLD;
            let presolve_deadline = if use_thread {
                current_opts.deadline.or_else(|| {
                    current_opts.timeout_secs.map(|s| {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let remaining = (s - elapsed).max(0.0);
                        std::time::Instant::now() + std::time::Duration::from_secs_f64(remaining)
                    })
                })
            } else {
                None
            };
            if let Some(d) = presolve_deadline {
                let now = std::time::Instant::now();
                if now >= d {
                    crate::presolve::QpPresolveResult::no_reduction(problem)
                } else {
                    let remaining = d - now;
                    let problem_owned = problem.clone();
                    let opts_owned = current_opts.clone();
                    let (tx, rx) = std::sync::mpsc::channel::<crate::presolve::QpPresolveResult>();
                    // スタックサイズを64MBに設定（大規模問題でのpresolve内スタック溢れ防止）
                    std::thread::Builder::new()
                        .stack_size(64 * 1024 * 1024)
                        .spawn(move || {
                            // 短期対処。presolve内部へのdeadline伝播はcmd_403以降の課題。
                            let phase1 = run_qp_presolve_phase1(&problem_owned, &opts_owned);
                            let _ = tx.send(run_qp_presolve_phase2(phase1, &opts_owned));
                        })
                        .expect("presolveスレッド起動失敗");
                    match rx.recv_timeout(remaining) {
                        Ok(r) => r,
                        Err(_) => crate::presolve::QpPresolveResult::no_reduction(problem),
                    }
                }
            } else {
                let phase1 = run_qp_presolve_phase1(problem, &current_opts);
                run_qp_presolve_phase2(phase1, &current_opts)
            }
        } else {
            crate::presolve::QpPresolveResult::no_reduction(problem)
        };
        if presolve_result.presolve_status == QpPresolveStatus::Infeasible {
            return crate::problem::SolverResult::infeasible();
        }
        // Bug-T2修正 (cmd_575): presolve後の残余時間を常にdeadlineとして設定する。
        // 大小規模問わず全ケースで残り時間を計算してdeadlineに変換することで、
        // presolve時間がtimeout予算に確実に算入される（UBH1 17.6s超過の原因を修正）。
        if current_opts.deadline.is_none() {
            if let Some(secs) = current_opts.timeout_secs {
                let elapsed = start_time.elapsed().as_secs_f64();
                let remaining = (secs - elapsed).max(0.0);
                current_opts.deadline = Some(
                    std::time::Instant::now() + std::time::Duration::from_secs_f64(remaining),
                );
                current_opts.timeout_secs = None;
            }
        }
        // post-verification retry ループ（cmd_400）
        // Ruiz unscale 増幅によるpfeas/bfeas失敗を、tighter eps の再dispatch で解消する。
        // Ruizパスなし（presolve_result.ruiz_scaler.is_none()）のときは pv_try=0 のみ実行。
        const PV_RETRY_MAX: usize = 3;
        // 行ノルム正規化pfeasチェック用: 行列Aはリトライ中に変わらないためループ外で1回だけ計算 [cmd_680]
        let row_norms = problem.a.row_infinity_norms();
        let mut result = {
            let mut pv_last: Option<SolverResult> = None;
            for pv_try in 0..PV_RETRY_MAX {
                let pv_opts_owned;
                let pv_opts: &SolverOptions = if presolve_result.ruiz_scaler.is_some() {
                    let tighten = 10f64.powi(pv_try as i32);
                    // PARAM: 1e-15 — post-verification の eps 調整下限（実装的根拠）。
                    // EPS_FLOOR=1e-12 より 1000 倍厳しい。double 精度限界（~2.2e-16）付近。
                    // tighten=10^pv_try の各段階で eps を 10 倍ずつ厳格化する際の安全弁。
                    // 承認=家老承認済み（cmd_576）
                    let adjusted_eps = (current_opts.ipm_eps() / tighten).max(1e-15);
                    let mut adj = current_opts.clone();
                    adj.ipm.eps = adjusted_eps;
                    adj.use_ruiz_scaling = false;
                    pv_opts_owned = adj;
                    &pv_opts_owned
                } else {
                    &current_opts
                };
                let mut reduced_sol = dispatch_qp(&presolve_result.reduced, pv_opts);
                // IR補正（Ruiz-scaled空間）[cmd_337]
                if !reduced_sol.solution.is_empty() && presolve_result.ruiz_scaler.is_some() {
                    let reduced_problem = &presolve_result.reduced;
                    let eps = current_opts.ipm_eps();
                    if reduced_problem.num_constraints > 0 {
                        if let Ok(ax) = reduced_problem.a.mat_vec_mul(&reduced_sol.solution) {
                            let pfeas_scaled = ax.iter().zip(reduced_problem.b.iter())
                                .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
                                .fold(0.0_f64, f64::max);
                            let norm_b_scaled = reduced_problem.b.iter()
                                .fold(0.0_f64, |a, &bi| a.max(bi.abs()))
                                .max(1.0);
                            if pfeas_scaled >= eps * (1.0 + norm_b_scaled) {
                                let mut y_tmp = reduced_sol.dual_solution.clone();
                                let mut z_tmp = reduced_sol.bound_duals.clone();
                                refine::iterative_refine(
                                    reduced_problem,
                                    &mut reduced_sol.solution,
                                    &mut y_tmp,
                                    &mut z_tmp,
                                    3,
                                    eps,
                                );
                            }
                        }
                    }
                }
                // Ruiz unscale
                if let Some(ref scaler) = presolve_result.ruiz_scaler {
                    let (x, y) = scaler.unscale_solution(&reduced_sol.solution, &reduced_sol.dual_solution);
                    reduced_sol.solution = x;
                    reduced_sol.dual_solution = y;
                    // ★追加: bound_duals もアンスケール
                    reduced_sol.bound_duals = scaler.unscale_bound_duals(
                        &reduced_sol.bound_duals,
                        &presolve_result.reduced.bounds,
                    );
                    if scaler.c.abs() > 1e-300 { reduced_sol.objective /= scaler.c; }
                    // reduced_costs を逆変換: rc_orig[j] = rc_scaled[j] / (d[j] * c)
                    // LP経路(Q=0)でのみ reduced_costs が非空になる
                    if scaler.c.abs() > 1e-300 && !reduced_sol.reduced_costs.is_empty() {
                        for (j, rc) in reduced_sol.reduced_costs.iter_mut().enumerate() {
                            let d_j = scaler.d.get(j).copied().unwrap_or(1.0);
                            if d_j.abs() > 1e-300 {
                                *rc /= d_j * scaler.c;
                            }
                        }
                    }
                }
                // Eq/Ge dual逆変換: IPM内部の to_all_le() 展開で増えたdualを元制約空間に折り畳む。
                // postsolve_qp の row_map は 1:1 マッピング前提のため、展開前に折り畳む必要がある。
                if presolve_result.reduced.constraint_types.iter().any(|ct| !matches!(ct, crate::problem::ConstraintType::Le)) {
                    let (_, le_map) = presolve_result.reduced.to_all_le();
                    reduced_sol.dual_solution = collapse_le_expansion_dual(
                        &reduced_sol.dual_solution, &le_map, &presolve_result.reduced.constraint_types,
                    );
                }
                let mut r = postsolve_qp(&presolve_result, &reduced_sol);
                // bound_duals リマップ: 縮約後空間 → 元問題空間 [cmd_689]
                // IPM/IPPMMが返すbound_dualsは縮約後の有限境界変数に対して格納されているため、
                // presolveで除去された変数を含む元問題空間に展開する。
                // 処理順序: Ruiz unscale(完了済み) → postsolve_qp → 本リマップ → bounds clip
                if presolve_result.was_reduced {
                    let orig_bounds = &problem.bounds;
                    let n_lb_orig = orig_bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
                    let n_ub_orig = orig_bounds.iter().filter(|(_, ub)| ub.is_finite()).count();

                    if n_lb_orig + n_ub_orig > 0 {
                        let reduced_bounds = &presolve_result.reduced.bounds;
                        let n_lb_reduced = reduced_bounds.iter()
                            .filter(|(lb, _)| lb.is_finite()).count();

                        // 縮約後変数jj → bound_duals配列内のlb/ubインデックスを構築
                        let n_reduced = reduced_bounds.len();
                        let mut lb_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
                        let mut ub_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
                        {
                            let mut li = 0;
                            for (jj, &(lb, _)) in reduced_bounds.iter().enumerate() {
                                if lb.is_finite() {
                                    lb_bd_idx[jj] = Some(li);
                                    li += 1;
                                }
                            }
                            let mut ui = 0;
                            for (jj, &(_, ub)) in reduced_bounds.iter().enumerate() {
                                if ub.is_finite() {
                                    ub_bd_idx[jj] = Some(n_lb_reduced + ui);
                                    ui += 1;
                                }
                            }
                        }

                        // 元問題空間のbound_dualsを構築（初期値0.0 = 除去変数のデフォルト）
                        let mut new_bd = vec![0.0f64; n_lb_orig + n_ub_orig];

                        // bound_dualsが非空の場合のみ値をコピー
                        // （空の場合: EmptyCol等で全有限境界変数が除去 → new_bd全要素0.0のまま）
                        if !r.bound_duals.is_empty() {
                            // lb部分: 元問題の各lb有限変数についてcol_map経由でマップ
                            let mut orig_li = 0;
                            for (j, &(lb, _)) in orig_bounds.iter().enumerate() {
                                if lb.is_finite() {
                                    if let Some(jj) = presolve_result.col_map[j] {
                                        if let Some(bd_idx) = lb_bd_idx[jj] {
                                            debug_assert!(bd_idx < r.bound_duals.len(),
                                                "bound_duals lb index out of range: {} >= {}",
                                                bd_idx, r.bound_duals.len());
                                            new_bd[orig_li] = r.bound_duals[bd_idx];
                                        }
                                    }
                                    // col_map[j] == None → 除去変数 → 0.0（初期値）
                                    orig_li += 1;
                                }
                            }

                            // ub部分: 元問題の各ub有限変数についてcol_map経由でマップ
                            let mut orig_ui = 0;
                            for (j, &(_, ub)) in orig_bounds.iter().enumerate() {
                                if ub.is_finite() {
                                    if let Some(jj) = presolve_result.col_map[j] {
                                        if let Some(bd_idx) = ub_bd_idx[jj] {
                                            debug_assert!(bd_idx < r.bound_duals.len(),
                                                "bound_duals ub index out of range: {} >= {}",
                                                bd_idx, r.bound_duals.len());
                                            new_bd[n_lb_orig + orig_ui] = r.bound_duals[bd_idx];
                                        }
                                    }
                                    orig_ui += 1;
                                }
                            }
                        }

                        r.bound_duals = new_bd;
                    }
                }
                // Ruiz unscale増幅由来のbounds微小違反を補正（clip）[cmd_400 (B)]
                // scaled空間では境界内だがunscale後に微小違反が生じるケース（例: QPCBOEI2）に対応
                if presolve_result.ruiz_scaler.is_some() && !r.solution.is_empty() {
                    for (xi, &(lb, ub)) in r.solution.iter_mut().zip(problem.bounds.iter()) {
                        if lb.is_finite() { *xi = xi.max(lb); }
                        if ub.is_finite() { *xi = xi.min(ub); }
                    }
                }
                // slack再計算: postsolve_qp はorig_problemを持たないため、呼び出し元で b-Ax を直接計算。
                // LP postsolve (postsolve.rs:71-79) と同方式。Ruiz/LCS/row_map の問題を全て回避。
                // ax_opt は下記 pfeas 計算（M5: 2重計算排除）と共有する。
                // 将来 postsolve_qp に orig_problem 引数を追加する際に移動可能。
                //
                // guard条件: LP経路の場合にのみ再計算する。
                //   (a) reduced_sol.slack が非空 = LP経路（Simplexがslackを返した）
                //   (b) reduced.num_vars == 0 = LP経路で全変数がpresolve除去済み
                //       → Simplexはn=0でslack=[]を返すが、solutionは正しい（postsolve後）
                //   QP/IPM経路: reduced.num_vars>0かつslack=[] → Noneのまま（slack=[]を維持）
                let ax_opt = if problem.num_constraints > 0 &&
                    (!reduced_sol.slack.is_empty() || presolve_result.reduced.num_vars == 0)
                {
                    problem.a.mat_vec_mul(&r.solution).ok()
                } else {
                    None
                };
                if let Some(ref ax) = ax_opt {
                    r.slack = problem.b.iter().zip(ax.iter()).map(|(&b, &a)| b - a).collect();
                }
                // post-postsolve検証: 元問題(A,b,bounds)で直接pfeas+bfeasを確認（偽Optimal防止）
                // scaled行列経由の逆変換は数学的複雑さによるバグを誘発するため、元問題で直接計算する
                if r.status == SolveStatus::Optimal {
                    let eps = current_opts.ipm_eps();
                    if problem.num_constraints > 0 {
                        // ax_opt（slack計算済み）があれば再利用（M5: 2重計算排除）、なければ独立計算
                        // 行ノルム正規化pfeasチェック [cmd_680]
                        // 判定式: max_k [ violation_k / (1 + ||a_k||_∞ + |b_k|) ] ≥ eps → SubOptimal
                        // row_norms はpv_retryループ外で1回だけ計算済み
                        let ax_for_pfeas = if let Some(ref ax) = ax_opt {
                            Some(ax.clone())
                        } else {
                            problem.a.mat_vec_mul(&r.solution).ok()
                        };
                        if let Some(ax) = ax_for_pfeas {
                            let pfeas_normalized = ax.iter()
                                .zip(problem.b.iter())
                                .zip(problem.constraint_types.iter())
                                .zip(row_norms.iter())
                                .map(|(((&ax_i, &b_i), ct), &rn)| {
                                    let violation = if matches!(ct, crate::problem::ConstraintType::Eq) {
                                        (ax_i - b_i).abs()
                                    } else {
                                        (ax_i - b_i).max(0.0)
                                    };
                                    violation / (1.0 + rn + b_i.abs())
                                })
                                .fold(0.0_f64, f64::max);
                            if pfeas_normalized >= eps {
                                r.status = SolveStatus::SuboptimalSolution;
                            }
                        }
                    }
                    if r.status == SolveStatus::Optimal {
                        let bfeas = r.solution.iter()
                            .zip(problem.bounds.iter())
                            .map(|(&xi, &(lb, ub))| {
                                let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
                                let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
                                lb_viol.max(ub_viol)
                            })
                            .fold(0.0_f64, f64::max);
                        if bfeas >= eps {  // 絶対閾値 eps（bnd_norm 不使用）[cmd_400 (B)]
                            r.status = SolveStatus::SuboptimalSolution;
                        }
                    }
                }
                // Ruizパスかつ SuboptimalSolution → 常にretry（tighter epsで精度改善の機会を与える）
                // 旧: dfeas/gap > eps_nextならretryスキップしていたが、DTOC3の有効retryまでブロックしていた
                // QFORPLAN保護は退行防止ガード(下記)が担当 [cmd_441]
                if r.status == SolveStatus::SuboptimalSolution
                    && presolve_result.ruiz_scaler.is_some()
                    && pv_try + 1 < PV_RETRY_MAX
                {
                    pv_last = Some(r);
                    continue;
                }
                // 退行防止ガード: retry後にOptimal/SubOptimal以外(NE/Timeout等)が返った場合、
                // pv_lastにSubOptimalが残っていればそちらを採用。
                // QFORPLAN事例: retry→NE(0iters) → 前回SubOptimalにフォールバック →
                // 後段のIR/Ruiz再ソルブで最終PASS化 [cmd_441]
                if pv_try > 0 {
                    if let Some(ref prev) = pv_last {
                        if prev.status == SolveStatus::SuboptimalSolution
                            && !matches!(r.status, SolveStatus::Optimal | SolveStatus::SuboptimalSolution)
                        {
                            break; // pv_lastのSubOptimal結果を保持
                        }
                    }
                }
                pv_last = Some(r);
                break;
            }
            pv_last.expect("PV_RETRY_MAX >= 1")
        };
        // iterative refinement: SuboptimalSolutionのとき、原問題空間でpfeasを改善（cmd_330）
        // n <= 300 の問題のみ対象（refine::iterative_refine 内でチェック）
        // Concurrent Solver経由でも solve_qp_with を通るため自動的に適用される（§6.2参照）
        if result.status == SolveStatus::SuboptimalSolution && !result.solution.is_empty() {
            let eps = current_opts.ipm_eps();
            let mut y = result.dual_solution.clone();
            let mut z = result.bound_duals.clone();
            if refine::iterative_refine(problem, &mut result.solution, &mut y, &mut z, 3, eps) {
                // 対策A: bfeas再チェック — IRはpfeasのみ修正。bfeas起因のSuboptimalSolutionは維持 [cmd_337]
                let bfeas_after = result.solution.iter()
                    .zip(problem.bounds.iter())
                    .map(|(&xi, &(lb, ub))| {
                        let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
                        let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
                        lb_viol.max(ub_viol)
                    })
                    .fold(0.0_f64, f64::max);
                if bfeas_after < eps {  // 絶対閾値 eps（bnd_norm 不使用）[cmd_400 (B)]
                    // IR後pfeas_normalized再検証 [cmd_680 QC-C1修正]
                    // IR内部は旧方式(norm_b)で収束判定するため、新方式(行ノルム正規化)で
                    // 再検証しないと偽Optimalがバイパスされる。
                    // row_normsはpv_retryループ外(L719)で計算済みを再利用。
                    let mut ir_pfeas_ok = true;
                    if problem.num_constraints > 0 {
                        if let Ok(ax) = problem.a.mat_vec_mul(&result.solution) {
                            let pfeas_normalized_post_ir = ax.iter()
                                .zip(problem.b.iter())
                                .zip(problem.constraint_types.iter())
                                .zip(row_norms.iter())
                                .map(|(((&ax_i, &b_i), ct), &rn)| {
                                    let violation = if matches!(ct, crate::problem::ConstraintType::Eq) {
                                        (ax_i - b_i).abs()
                                    } else {
                                        (ax_i - b_i).max(0.0)
                                    };
                                    violation / (1.0 + rn + b_i.abs())
                                })
                                .fold(0.0_f64, f64::max);
                            if pfeas_normalized_post_ir >= eps {
                                ir_pfeas_ok = false;
                            }
                        }
                    }
                    if ir_pfeas_ok {
                        result.status = SolveStatus::Optimal;
                        result.dual_solution = y;
                        result.bound_duals = z;
                    }
                    // else: IR後もpfeas_normalized不足 → SuboptimalSolution維持
                }
                // else: bfeas起因のSuboptimalSolution維持
            }
        }
        // Phase2: SuboptimalSolution時のRuiz無し再ソルブ（cmd_340→cmd_372非再帰化）
        // Ruiz unscale増幅起因のSuboptimalSolutionに対して、use_ruiz_scaling=falseで再ソルブを試みる
        // loop+フラグ方式で再帰呼び出しを排除（スタックフレーム削減）
        if result.status == SolveStatus::SuboptimalSolution && current_opts.use_ruiz_scaling {
            let has_time = if let Some(d) = current_opts.deadline {
                d > std::time::Instant::now() + std::time::Duration::from_secs_f64(0.5)
            } else if let Some(secs) = current_opts.timeout_secs {
                let elapsed = start_time.elapsed().as_secs_f64();
                secs - elapsed > 0.5
            } else {
                true // タイムアウト設定なし → 制限なく再ソルブ
            };
            if has_time {
                // 残り時間をdeadlineとして設定（二重カウント防止）
                if current_opts.deadline.is_some() {
                    // C-2: deadline自己代入（no-op）を削除。timeout_secs=Noneのみ設定
                    current_opts.timeout_secs = None;
                } else if let Some(secs) = current_opts.timeout_secs {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let remaining = (secs - elapsed).max(0.0);
                    current_opts.deadline = Some(
                        std::time::Instant::now() + std::time::Duration::from_secs_f64(remaining),
                    );
                    current_opts.timeout_secs = None;
                }
                first_result = Some(result); // C-1: 1st solve結果を保持（2nd solve失敗時フォールバック用）
                current_opts.use_ruiz_scaling = false;
                continue;
            }
        }
        // C-1: Ruiz無し2nd solve結果の条件付き採用
        // Optimal/SuboptimalSolutionのみ採用。Timeout/NumericalError/FAILは1st結果にフォールバック
        if let Some(saved) = first_result.take() {
            match result.status {
                SolveStatus::Optimal | SolveStatus::SuboptimalSolution => {} // 2nd solve成功: resultを使用
                _ => return apply_api_boundary_conversion(saved, problem, &current_opts), // 2nd solve失敗: 1st solve結果（SuboptimalSolution）を保持
            }
        }
        return apply_api_boundary_conversion(result, problem, &current_opts);
    }
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
///
/// 注意: 現在の実装では `warm_start` は使用されない（`solve_qp_with` に委譲するため、
/// `initial_point` および `initial_active_set` は共に無視される）。
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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a.clone(), b.clone(), bounds.clone()).unwrap();
        let problem2 = QpProblem::new_all_le(
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
            initial_active_set: vec![],
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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b_vec, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

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
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T20: concurrent should be Optimal");
        assert!((result.solution[0] - 0.5).abs() < EPS, "T20: x[0] ≈ 0.5");
        assert!((result.solution[1] - 0.5).abs() < EPS, "T20: x[1] ≈ 0.5");
        assert!((result.objective - 0.5).abs() < EPS, "T20: obj ≈ 0.5");
    }

    /// T-Concurrent-1: residual_score 単体テスト — score計算の正確性確認
    #[cfg(feature = "parallel")]
    #[test]
    fn test_residual_score_calculation() {
        use crate::problem::SolveStatus;
        // min x^2 + y^2  s.t. x+y >= 1（T1問題）
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // pfeas=0.01, dfeas=0.02 → score = max(0.01/2, 0.02/2) = 0.01
        let result_good = SolverResult {
            status: SolveStatus::Optimal,
            final_residuals: Some((0.01, 0.02, 0.001)),
            ..Default::default()
        };
        // pfeas=0.05, dfeas=0.02 → score = max(0.05/2, 0.02/2) = 0.025
        let result_bad = SolverResult {
            status: SolveStatus::Optimal,
            final_residuals: Some((0.05, 0.02, 0.001)),
            ..Default::default()
        };
        let score_good = residual_score(&result_good, &problem);
        let score_bad = residual_score(&result_bad, &problem);
        assert!(
            score_good < score_bad,
            "T-Concurrent-1: pfeas小さい方がscore小さいこと: good={score_good:.4e} < bad={score_bad:.4e}"
        );

        // final_residuals=None → INFINITY
        let result_none = SolverResult {
            status: SolveStatus::Optimal,
            final_residuals: None,
            ..Default::default()
        };
        assert_eq!(
            residual_score(&result_none, &problem),
            f64::INFINITY,
            "T-Concurrent-1: final_residuals=NoneならINFINITYを返すこと"
        );
    }

    /// T-Concurrent-2: 両Optimal・scoreが同値（eps以内）→ 先着維持
    #[cfg(feature = "parallel")]
    #[test]
    fn test_residual_score_same_value_keeps_first() {
        use crate::problem::SolveStatus;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 全く同じスコアの2つの結果
        let result_first = SolverResult {
            status: SolveStatus::Optimal,
            final_residuals: Some((0.01, 0.01, 0.001)),
            ..Default::default()
        };
        let result_second = SolverResult {
            status: SolveStatus::Optimal,
            final_residuals: Some((0.01, 0.01, 0.001)),
            ..Default::default()
        };
        let score_first = residual_score(&result_first, &problem);
        let score_second = residual_score(&result_second, &problem);
        // score差 < 1e-12 なら先着維持（should_update = false）
        let should_update = score_second < score_first - 1e-12;
        assert!(
            !should_update,
            "T-Concurrent-2: score同値の場合は先着を維持すること（should_update=false）"
        );
    }

    /// T-Concurrent-3: Optimal + Feasible → score不問でOptimalが採用される
    #[cfg(feature = "parallel")]
    #[test]
    fn test_quality_rank_optimal_beats_feasible() {
        use crate::problem::SolveStatus;
        // QualityRankはOptimal > Feasibleなのでrankで比較すればOptimalが勝つ
        let result_optimal = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.5, 0.5],
            final_residuals: Some((0.9, 0.9, 0.001)), // score高いがOptimal
            ..Default::default()
        };
        let result_feasible = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            solution: vec![0.5, 0.5],
            final_residuals: Some((0.0, 0.0, 0.0)),
            ..Default::default()
        };
        let rank_opt = quality_rank_of(&result_optimal);
        let rank_feas = quality_rank_of(&result_feasible);
        assert_eq!(rank_opt, Some(QualityRank::Optimal), "T-Concurrent-3: Optimal rankはOptimal");
        assert_eq!(rank_feas, None, "T-Concurrent-3: SuboptimalSolution はランク外（None）");
        assert!(
            rank_opt > rank_feas,
            "T-Concurrent-3: Optimal(Some) > SuboptimalSolution(None) でOptimalが勝つこと"
        );
    }

    /// T-Concurrent-4: 片方Optimal・片方NumericalError → Optimal側採用
    #[cfg(feature = "parallel")]
    #[test]
    fn test_quality_rank_optimal_beats_numerical_error() {
        use crate::problem::SolveStatus;
        let result_optimal = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.5, 0.5],
            ..Default::default()
        };
        let result_err = SolverResult {
            status: SolveStatus::NumericalError,
            solution: vec![],
            ..Default::default()
        };
        let rank_opt = quality_rank_of(&result_optimal);
        let rank_err = quality_rank_of(&result_err);
        assert_eq!(rank_opt, Some(QualityRank::Optimal), "T-Concurrent-4: Optimal rankはOptimal");
        assert_eq!(rank_err, None, "T-Concurrent-4: NumericalError rankはNone");
        // None は Some より小さい（should_update=trueになる）
        assert!(
            rank_opt > rank_err,
            "T-Concurrent-4: Optimal(Some) > NumericalError(None)"
        );
    }

    /// T-Concurrent-5: 片方Optimal・片方Infeasible → Optimal側採用
    #[cfg(feature = "parallel")]
    #[test]
    fn test_quality_rank_optimal_beats_infeasible() {
        use crate::problem::SolveStatus;
        let result_optimal = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.5, 0.5],
            ..Default::default()
        };
        let result_infeas = SolverResult {
            status: SolveStatus::Infeasible,
            solution: vec![],
            ..Default::default()
        };
        let rank_opt = quality_rank_of(&result_optimal);
        let rank_inf = quality_rank_of(&result_infeas);
        assert_eq!(rank_opt, Some(QualityRank::Optimal), "T-Concurrent-5: Optimal rankはOptimal");
        assert_eq!(rank_inf, None, "T-Concurrent-5: Infeasible rankはNone（fallbackに回る）");
        assert!(
            rank_opt > rank_inf,
            "T-Concurrent-5: Optimal(Some) > Infeasible(None)"
        );
    }

    /// T22: QualityRank の Ord 比較
    #[test]
    fn test_quality_rank_ordering() {
        assert!(QualityRank::Optimal > QualityRank::Feasible, "T22: Optimal > Feasible");
        assert!(QualityRank::Feasible > QualityRank::Approximate, "T22: Feasible > Approximate");
        assert!(QualityRank::Optimal > QualityRank::Approximate, "T22: Optimal > Approximate");
        assert_eq!(QualityRank::Optimal, QualityRank::Optimal, "T22: Optimal == Optimal");
    }

    /// T23: presolveパス pfeas検証 — 大行ノルム制約でのRuiz scaling耐性確認
    ///
    /// 旧T23はverify_post_ruiz_unscaleを恒等スケーラー(e=[1.0])で直接テストしていたため、
    /// `* e_i` vs `/ e_i` のバグを検出できなかった。
    /// 新T23は行ノルムが大きい制約（Ruiz scaling後にe[i]<<1になる）を含む問題を
    /// solve_qp_withで解き、元問題でpfeasが正しく計算されることを確認する。
    ///
    /// ★旧コードのバグ: e[i]=0.01のとき `* 0.01` で pfeas を100倍小さく評価→偽Optimalを見逃す
    /// 新コード: 元問題(A,b)で直接A*x-bを計算するためe[i]に依存しない
    #[test]
    fn test_presolve_pfeas_large_row_norm() {
        // min x^2  s.t. 1000*x <= -500  (解: x=0 は不可、問題は実行不可能)
        // → 実行可能な問題として: min x^2  s.t. 1000*x <= 500 (解: x=0)
        // 行ノルム=1000 → Ruiz scaling後のe[i] ≈ 1/sqrt(1000) << 1
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0], 1, 1).unwrap();
        let b = vec![500.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default(); // presolve=true
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "T23: Optimal解が得られること");
        // 元問題でpfeasを直接検証: A*x - b <= 0 のはず
        let ax = problem.a.mat_vec_mul(&result.solution).unwrap();
        let pfeas = ax.iter().zip(problem.b.iter())
            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
            .fold(0.0_f64, f64::max);
        let norm_b = problem.b.iter().fold(0.0_f64, |a, &bi| a.max(bi.abs())).max(1.0);
        let eps = opts.ipm_eps();
        assert!(
            pfeas < eps * (1.0 + norm_b),
            "T23: 元問題でpfeas={pfeas:.2e} < eps*(1+norm_b)={:.2e}（e[i]<<1でも正しく検証）",
            eps * (1.0 + norm_b)
        );
    }

    /// T24: presolveパス bfeas検証 — bounds付き問題でOptimal解が境界を満たすことを確認
    ///
    /// 旧T24はverify_post_ruiz_unscaleに人工的な違反解を注入して直接テストしていた。
    /// 新T24はsolve_qp_with経由で、boundsを持つ問題が正しくOptimalを返し、
    /// post-postsolve bfeasチェックが正常解を誤降格しないことを確認する。
    #[test]
    fn test_presolve_bfeas_bounded_problem() {
        // min x^2  s.t. なし  0 <= x <= 1  (最適解: x=0)
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default(); // presolve=true
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "T24: bounds付き問題でOptimal解が得られること");
        let x = result.solution[0];
        assert!(x >= -1e-4, "T24: x >= lb=0, got x={x}");
        assert!(x <= 1.0 + 1e-4, "T24: x <= ub=1, got x={x}");
    }

    /// T25: post-postsolve pfeas+bfeas — 正常解でOptimalを維持することを確認
    ///
    /// 旧T25はverify_post_ruiz_unscaleにx=[0]を注入してOKを確認。
    /// 新T25はsolve_qp_with経由で制約+bounds付き問題を解き、
    /// post-postsolveチェックが正常解を誤降格しないことを確認する。
    #[test]
    fn test_presolve_pfeas_bfeas_ok() {
        // min x^2  s.t. x <= 1.0  0 <= x <= 0.5  (最適解: x=0)
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![1.0];
        let bounds = vec![(0.0_f64, 0.5_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default(); // presolve=true
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T25: pfeas・bfeas ともにOKの場合はOptimalを維持すること"
        );
    }

    /// T26: presolve有効ケース — solve_qp_with経由でpresolveパスの検証コードが動くことを確認
    ///
    /// presolve=true（デフォルト）で問題を解かせ、正常なOptimal解が得られることを確認する。
    /// これはpresolveパスのpost-unscaling検証コードが実行されても正常問題には影響しないことを示す。
    #[test]
    fn test_solve_qp_with_presolve_path_verified() {
        // min x^2 + y^2  s.t. x + y >= 1  (bounds: -∞ <= x,y <= ∞)
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // presolve=true (デフォルト) で解く → presolveパスのコードが動く
        let opts = SolverOptions::default();
        assert!(opts.presolve, "T26: デフォルトはpresolve=true");
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "T26: presolve有効時もOptimalを返すこと");
        let eps = 1e-3_f64;
        assert!(
            (result.solution[0] - 0.5).abs() < eps,
            "T26: x[0] ≈ 0.5, got {}",
            result.solution[0]
        );
        assert!(
            (result.solution[1] - 0.5).abs() < eps,
            "T26: x[1] ≈ 0.5, got {}",
            result.solution[1]
        );
    }

    /// T27: 不定Q行列（対角に負値）→ NonConvex返却
    /// Q = diag(-1.0, 1.0, 1.0) → 最小固有値 = -1.0 → 非凸QP
    /// 期待: SolveStatus::NonConvex(...)
    #[test]
    fn test_qp_nonconvex_indefinite_q() {
        // Q = diag(-1.0, 1.0, 1.0)（不定行列: 対角に負値）
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[-1.0, 1.0, 1.0],
            3,
            3,
        ).unwrap();
        let c = vec![0.0, 0.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert!(
            matches!(result.status, SolveStatus::NonConvex(_)),
            "T27: 不定Q行列はNonConvexを返すこと。got: {:?}", result.status
        );
    }

    /// T28: 半正定値Q行列（最小固有値=0）→ PSD判定（NonConvexでないこと）
    /// Q = diag(0.0, 1.0, 1.0) → Q+eps*I の全ピボット > 0 → PSD判定
    /// 期待: check_q_positive_semidefinite が true を返す
    #[test]
    fn test_qp_psd_semidefinite_q() {
        // Q = diag(0.0, 1.0, 1.0)（半正定値行列: 最小固有値=0）
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[0.0, 1.0, 1.0],
            3,
            3,
        ).unwrap();
        assert!(
            check_q_positive_semidefinite(&q),
            "T28: 半正定値Q（最小固有値=0）はPSD判定されること"
        );
    }

    /// T29: SolveStatus::NonConvex の Display 確認
    /// 期待: format!("{}", NonConvex(msg)) == "NonConvex(msg)"
    #[test]
    fn test_solve_status_display_nonconvex() {
        let msg = "Q matrix is indefinite".to_string();
        let status = SolveStatus::NonConvex(msg.clone());
        assert_eq!(format!("{}", status), format!("NonConvex({})", msg));
    }

    /// T_NEW1: n=1001(>1000) の対角負値行列 → NonConvex検出（案A）
    /// Q = diag(-1.0, 1.0, ..., 1.0), n=1001 → Q[0,0]=-1.0 < -1e-10 → 非PSD確定
    #[test]
    fn test_qp_nonconvex_large_diagonal_negative() {
        let n = 1001_usize;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = std::iter::once(-1.0_f64)
            .chain(std::iter::repeat(1.0_f64).take(n - 1))
            .collect();
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert!(
            !check_q_positive_semidefinite(&q),
            "n>1000の対角負値行列はNonPSD（NonConvex）を返すこと"
        );
    }

    /// T_NEW2: n=1001 の対角全正値行列 → PSD（偽陽性防止）
    /// Q = diag(1.0, 1.0, ..., 1.0), n=1001 → 対角に負値なし → true（PSD）
    #[test]
    fn test_qp_psd_large_diagonal_positive() {
        let n = 1001_usize;
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = vec![1.0_f64; n];
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert!(
            check_q_positive_semidefinite(&q),
            "n>1000の対角全正値行列はPSD判定されること（偽陽性なし）"
        );
    }

    /// T_NEW3: 境界値 Q[0,0]=-1e-11（閾値 -1e-10 より大きい） → PSD（数値ノイズ無視）
    /// -1e-11 > -1e-10 のため閾値未満と判定され、非凸検出しない
    #[test]
    fn test_qp_diagonal_boundary_below_threshold() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[-1e-11_f64, 1.0, 1.0],
            3,
            3,
        ).unwrap();
        assert!(
            check_q_positive_semidefinite(&q),
            "Q[0,0]=-1e-11 は閾値 -1e-10 より大きいため非凸検出しないこと"
        );
    }

    /// T_NEW3b: 境界値 Q[0,0]=-1e-10 exact（閾値ちょうど） → PSD（非凸検出しない）
    /// チェック条件は q < -1e-10。-1e-10 == -1e-10 のため条件を満たさず PSD を返す
    #[test]
    fn test_qp_diagonal_boundary_exact_threshold() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[-1e-10_f64, 1.0, 1.0],
            3,
            3,
        ).unwrap();
        assert!(
            check_q_positive_semidefinite(&q),
            "Q[0,0]=-1e-10 は閾値ちょうどのため非凸検出しないこと（条件は < -1e-10）"
        );
    }

    /// T_NEW4: 境界値 Q[0,0]=-1e-9（閾値 -1e-10 を超える） → NonConvex
    /// -1e-9 < -1e-10 のため対角負値として検出される
    #[test]
    fn test_qp_diagonal_boundary_above_threshold() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[-1e-9_f64, 1.0, 1.0],
            3,
            3,
        ).unwrap();
        assert!(
            !check_q_positive_semidefinite(&q),
            "Q[0,0]=-1e-9 は閾値 -1e-10 を超えるため非凸検出されること"
        );
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // cmd_589 TDD赤フェーズ: バグ再現テスト（cmd_607で修正済み）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// BUG-QP-001: solve_as_lp が MaxIterations→NumericalError 変換（MEDIUM）
    /// 修正（cmd_607）: qp/mod.rs の MaxIterations branch を unreachable!() に置換。
    /// MaxIterations は SimplexOutcome::MaxIterations廃止（cmd_595）により到達不能なdead path。
    #[test]
    fn test_qp001_solve_as_lp_no_numerical_error() {
        // SPEC: BUG-QP-001 — regression test
        // MaxIterationsはsimplexパスから到達不能（cmd_595/cmd_607確認）のためPASS。
        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap(); // Q=0 → LP
        let c = vec![-1.0, -1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![4.0];
        let bounds = vec![(0.0f64, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: false,
            qp_solver: QpSolverChoice::Ipm,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        // NumericalError はバグステータス。返ってはならない。
        assert_ne!(
            result.status,
            SolveStatus::NumericalError,
            "BUG-QP-001: solve_as_lp は NumericalError を返してはならない"
        );
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // cmd_589 TDD赤フェーズ: テスト不足 (△) 項目
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// A2-T03: QP timeout_secs=None で有限ステップ収束
    #[test]
    fn test_a2t03_qp_no_deadline_converges() {
        // SPEC: A2-T03 (QP版)
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { timeout_secs: None, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "A2-T03: QP タイムアウトなしで収束すること");
    }

    /// A3-C02: cancel_flag 事前設定で即停止（QP版）
    #[test]
    fn test_a3c02_cancel_flag_preset_qp_returns_timeout() {
        // SPEC: A3-C02 (QP版)
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let cancel = Arc::new(AtomicBool::new(true)); // 事前に true
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            qp_solver: QpSolverChoice::Ipm,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "A3-C02: cancel_flag 事前設定で Timeout が返ること"
        );
    }

    /// A4-P01: presolve の透過性（presolve 有無で解が一致）
    #[test]
    fn test_a4p01_presolve_transparency_qp() {
        // SPEC: A4-P01
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts_with = SolverOptions {
            presolve: true,
            qp_solver: QpSolverChoice::Ipm,
            ..SolverOptions::default()
        };
        let opts_without = SolverOptions {
            presolve: false,
            qp_solver: QpSolverChoice::Ipm,
            ..SolverOptions::default()
        };
        let result_with = solve_qp_with(&problem, &opts_with);
        let result_without = solve_qp_with(&problem, &opts_without);
        assert_eq!(result_with.status, SolveStatus::Optimal, "A4-P01: presolve=true → Optimal");
        assert_eq!(result_without.status, SolveStatus::Optimal, "A4-P01: presolve=false → Optimal");
        assert!(
            (result_with.solution[0] - result_without.solution[0]).abs() < 1e-3,
            "A4-P01: presolve 有無で x[0] が一致すること"
        );
        assert!(
            (result_with.solution[1] - result_without.solution[1]).abs() < 1e-3,
            "A4-P01: presolve 有無で x[1] が一致すること"
        );
    }

    /// A6-I03: n>1000 で NonConvex 検出がスキップされること（既知の制限）
    #[test]
    fn test_a6i03_nonconvex_skip_for_large_n() {
        // SPEC: A6-I03
        // n=1001 > CHECK_SIZE_LIMIT=1000: Cholesky 省略 + 対角チェックのみ
        // 対角負値は n に関係なく対角チェックで検出される
        let n = 1001usize;
        // case1: 対角負値 → 対角チェックで検出（n>1000 でも有効）
        let mut rows = vec![0usize];
        let mut cols = vec![0usize];
        let mut vals = vec![-1e-9_f64]; // -1e-9 < -1e-10 → 検出
        for i in 1..n {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
        }
        let q1 = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert!(!check_q_positive_semidefinite(&q1), "A6-I03: n=1001 対角負値は NonConvex を検出");

        // case2: 非対角の非 PSD（対角チェックには引っかからない）→ n>1000 でスキップ
        let mut rows2: Vec<usize> = (0..n).collect();
        let mut cols2: Vec<usize> = (0..n).collect();
        let mut vals2: Vec<f64> = vec![1.0; n]; // 全て正の対角
        // 非対角に負値追加（非 PSD だが対角チェックには引っかからない）
        rows2.push(0); cols2.push(1); vals2.push(-2.0);
        let q2 = CscMatrix::from_triplets(&rows2, &cols2, &vals2, n, n).unwrap();
        // n>1000 では Cholesky 省略 → 対角チェックのみ → true を返す（スキップ）
        assert!(
            check_q_positive_semidefinite(&q2),
            "A6-I03: n>1000 の非対角非 PSD は NonConvex 検出をスキップする（既知の制限）"
        );
    }

    /// A7-CS04: バグステータスのフィルタリング（SuboptimalSolution → Feasible バグ）
    /// quality_rank_of で SuboptimalSolution が Feasible として残ることを確認
    #[cfg(feature = "parallel")]
    #[test]
    fn test_a7cs04_suboptimal_not_filtered_bug() {
        // SPEC: A7-CS04 / BUG
        // quality_rank_of で SuboptimalSolution は None を返すべき（バグステータスとして除外）
        // 現状: Some(QualityRank::Feasible) が返る → assert FAIL
        let result = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            solution: vec![0.5, 0.5], // 非空（現状フィルタリングされない）
            ..Default::default()
        };
        let rank = quality_rank_of(&result);
        // 修正後: SuboptimalSolution は None（除外）になるべき
        assert!(
            rank.is_none(),
            "A7-CS04: SuboptimalSolution は quality_rank_of から除外されるべき。現状 {:?} が返る",
            rank
        );
    }

    /// A7-CS02: concurrent solver スレッド安全性（cancel_flag 経由の停止）
    #[cfg(feature = "parallel")]
    #[test]
    fn test_a7cs02_concurrent_cancel_flag_thread_safety() {
        // SPEC: A7-CS02
        // concurrent solver で Optimal を発見したとき cancel_flag でリソースリーク・
        // データ競合なしに停止することを確認（10回繰り返してクラッシュなし）
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        for _ in 0..10 {
            let opts = SolverOptions {
                qp_solver: QpSolverChoice::Concurrent,
                ..SolverOptions::default()
            };
            let result = solve_qp_with(&problem, &opts);
            assert_eq!(
                result.status,
                SolveStatus::Optimal,
                "A7-CS02: concurrent solver はスレッド安全に Optimal を返すこと"
            );
        }
    }

    /// A7-CS03: 全スレッド Timeout の場合 Timeout が返ること
    #[cfg(feature = "parallel")]
    #[test]
    fn test_a7cs03_concurrent_all_timeout_returns_timeout() {
        // SPEC: A7-CS03
        // timeout_secs=0 で concurrent solver → 全スレッドが Timeout → Timeout が返る
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            qp_solver: QpSolverChoice::Concurrent,
            timeout_secs: Some(0.0), // 即座にタイムアウト
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "A7-CS03: 全スレッド Timeout のとき Timeout が返ること"
        );
    }

    /// A3-C01: cancel_flag 即停止 / A3-C03: cancel_flag スレッド間共有（concurrent）
    /// BUG-CONC-001修正済み: 外部 cancel_flag が concurrent solver に正しく伝搬される。
    #[cfg(feature = "parallel")]
    #[test]
    fn test_a3c01_cancel_flag_concurrent_returns_timeout() {
        // SPEC: A3-C01 / A3-C03
        // concurrent solver で cancel_flag=true（事前設定）→ Timeout が返ること
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let cancel = Arc::new(AtomicBool::new(true)); // 事前 true
        let opts = SolverOptions {
            qp_solver: QpSolverChoice::Concurrent,
            cancel_flag: Some(Arc::clone(&cancel)),
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "A3-C01/C03: cancel_flag 事前設定で concurrent solver は Timeout を返すこと"
        );
    }

    // ========== postsolve F1/F2修正検証テスト (T1-T7 + E1-E4) ==========
    // 設計書: /Users/hika019/Develop/solver/reports/cmd667_postsolve_fix_design.md §5
    // T1: presolve OFF 基準線
    // T2: FixedVar + col_mapリマップ（核心テスト）
    // T3: SingletonRow + row_map
    // T4: FixedVar + 小Ruiz
    // T5: FixedVar+LCS + 大Ruiz（C1指摘の核心ケース）
    // T6: EmptyCol
    // T7: QP IPM（slack=[], rc=[]）
    // E1-E4: エッジケース

    /// T1: presolve OFF 基準線
    /// min 2x+3y  s.t. x+y<=4, x<=3, x,y>=0
    /// presolve=false → postsolve経路（identity mapping）
    /// 期待: solution=[0,0], obj=0, slack=[4,3], reduced_costs=[2,3]
    #[test]
    fn test_postsolve_t1_presolve_off_baseline() {
        let n = 2usize;
        let q = CscMatrix::new(n, n); // Q=0 → LP path
        let c = vec![2.0, 3.0];
        // x+y<=4, x<=3
        let a = CscMatrix::from_triplets(
            &[0, 0, 1],
            &[0, 1, 0],
            &[1.0, 1.0, 1.0],
            2, n,
        ).unwrap();
        let b = vec![4.0, 3.0];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: false, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T1: status");
        let tol = 1e-8_f64;
        // solution
        assert!((result.solution[0]).abs() < tol, "T1: x≈0");
        assert!((result.solution[1]).abs() < tol, "T1: y≈0");
        // obj
        assert!((result.objective).abs() < tol, "T1: obj=0");
        // slack = b - Ax
        assert_eq!(result.slack.len(), 2, "T1: slack.len");
        assert!((result.slack[0] - 4.0).abs() < tol, "T1: slack[0]=4");
        assert!((result.slack[1] - 3.0).abs() < tol, "T1: slack[1]=3");
        // reduced_costs
        assert_eq!(result.reduced_costs.len(), n, "T1: rc.len");
        assert!((result.reduced_costs[0] - 2.0).abs() < tol, "T1: rc[0]=2");
        assert!((result.reduced_costs[1] - 3.0).abs() < tol, "T1: rc[1]=3");
    }

    /// T2: FixedVar + col_mapリマップ（核心テスト）
    /// min 2x+3y+z  s.t. x+y<=4, x+2y<=6, x,y>=0, z=5 (lb=ub=5)
    /// presolve: z除去(FixedVar)。x,yは2制約に登場するためsingletonCol除去されない。
    /// → 縮約後n=2でSimplexがrc=[2,3]を返し、postsolveでrc=[2,3,0]に展開されるF2テスト。
    #[test]
    fn test_postsolve_t2_fixed_var_col_map() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0, 1.0];
        // x+y<=4, x+2y<=6 (z not in constraints)
        // x,yが2制約に登場 → singletonCol最適化の対象外 → 縮約後問題に残る
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 2.0],
            2, n,
        ).unwrap();
        let b = vec![4.0, 6.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)]; // z fixed
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T2: status");
        let tol = 1e-6_f64;
        // solution: x=0, y=0, z=5 (全て非負コスト、lb=0が最適)
        assert_eq!(result.solution.len(), 3, "T2: solution.len");
        assert!((result.solution[0]).abs() < tol, "T2: x≈0");
        assert!((result.solution[1]).abs() < tol, "T2: y≈0");
        assert!((result.solution[2] - 5.0).abs() < tol, "T2: z=5");
        // obj = 2*0+3*0+1*5 = 5
        assert!((result.objective - 5.0).abs() < tol, "T2: obj=5");
        // reduced_costs: len=3 (F2: col_mapリマップ検証), rc[2]=0 (FixedVar除去)
        assert_eq!(result.reduced_costs.len(), 3, "T2: rc.len=3 (F2 col_map)");
        assert!((result.reduced_costs[0] - 2.0).abs() < tol, "T2: rc[0]=2 (x non-basic at lb)");
        assert!((result.reduced_costs[1] - 3.0).abs() < tol, "T2: rc[1]=3 (y non-basic at lb)");
        assert!((result.reduced_costs[2]).abs() < tol, "T2: rc[2]=0 (z fixed by FixedVar)");
        // slack = b - Ax
        assert_eq!(result.slack.len(), 2, "T2: slack.len=2");
        assert!((result.slack[0] - 4.0).abs() < tol, "T2: slack[0]=4");
        assert!((result.slack[1] - 6.0).abs() < tol, "T2: slack[1]=6");
        // 相補性: x[j]*rc[j] ≈ 0
        for j in 0..3 {
            assert!((result.solution[j] * result.reduced_costs[j]).abs() < 1e-7,
                "T2: complementarity x[{}]*rc[{}]", j, j);
        }
    }

    /// T3: SingletonRow + row_map
    /// min x+y  s.t. x=2 (Eq singleton), y<=3, x,y>=0
    /// presolve: x除去(SingletonRow), 行0除去
    /// 期待: solution=[2,0], slack.len()=2, slack[0]=0 (Eq制約)
    #[test]
    fn test_postsolve_t3_singleton_row() {
        use crate::problem::ConstraintType;
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        // x=2 (Eq), y<=3 (Le)
        let rows = &[0usize, 1usize];
        let cols = &[0usize, 1usize];
        let vals = &[1.0, 1.0];
        let a = CscMatrix::from_triplets(rows, cols, vals, 2, n).unwrap();
        let b = vec![2.0, 3.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new(
            q, c, a, b, bounds,
            vec![ConstraintType::Eq, ConstraintType::Le],
        ).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T3: status");
        let tol = 1e-6_f64;
        // solution: x=2, y=0
        assert_eq!(result.solution.len(), 2, "T3: solution.len");
        assert!((result.solution[0] - 2.0).abs() < tol, "T3: x=2");
        assert!((result.solution[1]).abs() < tol, "T3: y=0");
        // slack sizes
        assert_eq!(result.slack.len(), 2, "T3: slack.len=2");
        assert!((result.slack[0]).abs() < tol, "T3: slack[0]=0 (Eq)");
        // reduced_costs size
        assert_eq!(result.reduced_costs.len(), 2, "T3: rc.len=2");
    }

    /// T4: Ruiz + FixedVar複合
    /// min 2x+3y+z  s.t. 10x+y<=10, x<=3, x,y>=0, z=5
    /// Ruizスケール + FixedVar
    /// 期待: solution=[0,0,5], obj=5, slack=[10,3], rc=[2,3,0]
    #[test]
    fn test_postsolve_t4_ruiz_fixed_var() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0, 1.0];
        // 10x+y<=10, x<=3 (z not in constraints)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1],
            &[0, 1, 0],
            &[10.0, 1.0, 1.0],
            2, n,
        ).unwrap();
        let b = vec![10.0, 3.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T4: status");
        let tol = 1e-6_f64;
        assert_eq!(result.solution.len(), 3, "T4: solution.len");
        assert!((result.solution[0]).abs() < tol, "T4: x≈0");
        assert!((result.solution[1]).abs() < tol, "T4: y≈0");
        assert!((result.solution[2] - 5.0).abs() < tol, "T4: z=5");
        assert!((result.objective - 5.0).abs() < tol, "T4: obj=5");
        // slack = [10-0, 3-0]
        assert_eq!(result.slack.len(), 2, "T4: slack.len=2");
        assert!((result.slack[0] - 10.0).abs() < tol, "T4: slack[0]=10");
        assert!((result.slack[1] - 3.0).abs() < tol, "T4: slack[1]=3");
        assert_eq!(result.reduced_costs.len(), 3, "T4: rc.len=3");
        assert!((result.reduced_costs[2]).abs() < tol, "T4: rc[2]=0 (fixed)");
    }

    /// T5: FixedVar+LCS + 大Ruiz（C1指摘の核心ケース）
    /// min x+y+z  s.t. 1e7*x+y<=1e7, x+y<=2, z=0.5 (fixed), x,y,z>=0
    /// LCS閾値 >1e6 → 1e7係数でLCS発動。b-Ax再計算でLCS逆変換問題を回避。
    /// 期待: slack[0] = 1e7 - 1e7*x - y（元問題空間で正確）
    #[test]
    fn test_postsolve_t5_lcs_ruiz_fixed_var() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0, 1.0];
        // 1e7*x+y<=1e7, x+y<=2 (z not in constraints)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1e7, 1.0, 1.0, 1.0],
            2, n,
        ).unwrap();
        let b = vec![1e7, 2.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.5, 0.5)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T5: status");
        let x = result.solution[0];
        let y = result.solution[1];
        // slack精度: b[i] - (Ax)[i] で元問題空間で直接計算して照合
        assert_eq!(result.slack.len(), 2, "T5: slack.len=2");
        let slack0_expected = 1e7 - 1e7*x - y;
        let slack1_expected = 2.0 - x - y;
        // 相対誤差で確認（LCS+Ruizで1e7スケールのため絶対誤差は大きくなりうる）
        let tol_rel = 1e-5_f64;
        assert!(
            (result.slack[0] - slack0_expected).abs() <= tol_rel * slack0_expected.abs().max(1.0),
            "T5: slack[0]={} expected={} (LCS b-Ax精度)", result.slack[0], slack0_expected
        );
        assert!(
            (result.slack[1] - slack1_expected).abs() <= tol_rel * slack1_expected.abs().max(1.0),
            "T5: slack[1]={} expected={}", result.slack[1], slack1_expected
        );
        // reduced_costs.len = 3
        assert_eq!(result.reduced_costs.len(), 3, "T5: rc.len=3");
        assert!((result.reduced_costs[2]).abs() < 1e-6, "T5: rc[2]=0 (fixed z)");
    }

    /// T6: EmptyCol（空列除去）
    /// min 2x+3y+z  s.t. x+y<=4, x<=3, x,y>=0, 0<=z<=3 (z制約行列ゼロ)
    /// presolve: zはEmptyCol→z=lb=0 に固定
    /// 期待: solution=[0,0,0], obj=0, slack=[4,3], rc=[2,3,0]
    #[test]
    fn test_postsolve_t6_empty_col() {
        let n = 3usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0, 1.0];
        // x+y<=4, x<=3 (z absent from constraints)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1],
            &[0, 1, 0],
            &[1.0, 1.0, 1.0],
            2, n,
        ).unwrap();
        let b = vec![4.0, 3.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, 3.0)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T6: status");
        let tol = 1e-8_f64;
        assert_eq!(result.solution.len(), 3, "T6: solution.len=3");
        assert!((result.solution[0]).abs() < tol, "T6: x≈0");
        assert!((result.solution[1]).abs() < tol, "T6: y≈0");
        assert!((result.solution[2]).abs() < tol, "T6: z≈0 (EmptyCol→lb)");
        assert!((result.objective).abs() < tol, "T6: obj=0");
        assert_eq!(result.slack.len(), 2, "T6: slack.len=2");
        assert!((result.slack[0] - 4.0).abs() < tol, "T6: slack[0]=4");
        assert!((result.slack[1] - 3.0).abs() < tol, "T6: slack[1]=3");
        assert_eq!(result.reduced_costs.len(), 3, "T6: rc.len=3");
        assert!((result.reduced_costs[2]).abs() < tol, "T6: rc[2]=0 (empty col fixed)");
    }

    /// T7: QP IPM経路（slack=[], rc=[]）
    /// min 1/2*(x^2+y^2)  s.t. x+y<=2, x,y>=0
    /// IPM経路 → slack空、reduced_costs空を確認
    #[test]
    fn test_postsolve_t7_qp_ipm_empty_slack_rc() {
        let n = 2usize;
        // Q = [[2,0],[0,2]] (1/2規約で min 1/2*2*x^2 = min x^2)
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![2.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "T7: status");
        // IPM経路ではslack空・reduced_costs空
        assert!(result.slack.is_empty(), "T7: slack=[] for IPM path");
        assert!(result.reduced_costs.is_empty(), "T7: rc=[] for IPM path");
    }

    /// E1: 全変数固定（全てFixedVar）
    /// min x+y  s.t. x,y free, x=1 (lb=ub=1), y=2 (lb=ub=2)
    /// 制約なし（A=0x2空行列）
    /// 期待: solution=[1,2], rc.len()=2, slack.len()=0
    #[test]
    fn test_postsolve_e1_all_vars_fixed() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(1.0_f64, 1.0_f64), (2.0_f64, 2.0_f64)]; // both fixed
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "E1: status");
        assert_eq!(result.solution.len(), 2, "E1: solution.len=2");
        assert_eq!(result.reduced_costs.len(), 2, "E1: rc.len=2 (元変数空間)");
        assert_eq!(result.slack.len(), 0, "E1: slack.len=0 (no constraints)");
    }

    /// E2: 全制約除去（A=0 all-redundant）
    /// 制約なし問題: min 2x+3y, x,y >= 0
    /// slack.len()=0, rc.len()=2
    #[test]
    fn test_postsolve_e2_no_constraints() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0, 3.0];
        let a = CscMatrix::new(0, n);
        let b: Vec<f64> = vec![];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        // 制約なし、目的関数最小化 → x=y=0
        assert_eq!(result.status, SolveStatus::Optimal, "E2: status");
        let tol = 1e-8_f64;
        assert_eq!(result.slack.len(), 0, "E2: slack.len=0 (no constraints)");
        assert_eq!(result.reduced_costs.len(), n, "E2: rc.len=2");
        assert!((result.solution[0]).abs() < tol, "E2: x=0");
        assert!((result.solution[1]).abs() < tol, "E2: y=0");
    }

    /// E3: presolve発動なし（presolve=true but no reduction）
    /// min x+y  s.t. x+y<=2, x>=0, y>=0
    /// 変数除去なし → col_map = [Some(0), Some(1)] (identity)
    /// rc.len()=2, slack.len()=1
    #[test]
    fn test_postsolve_e3_presolve_no_reduction() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![2.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default(); // presolve=true but no reduction expected
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "E3: status");
        assert_eq!(result.reduced_costs.len(), n, "E3: rc.len=2");
        assert_eq!(result.slack.len(), 1, "E3: slack.len=1");
        let tol = 1e-8_f64;
        // b-Ax: slack[0] = 2 - x - y = 2 - 0 = 2 (optimal at x=y=0)
        assert!((result.slack[0] - 2.0).abs() < tol, "E3: slack[0]=2");
    }

    /// E4: LCS発動 + presolve変数除去なし
    /// min x+y  s.t. 1e7*x+y<=1e7, x>=0, y>=0
    /// LCS発動でslack精度問題発生しうる → b-Ax再計算で回避
    #[test]
    fn test_postsolve_e4_lcs_no_presolve_elimination() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1e7, 1.0], 1, n).unwrap();
        let b = vec![1e7];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "E4: status");
        let x = result.solution[0];
        let y = result.solution[1];
        assert_eq!(result.slack.len(), 1, "E4: slack.len=1");
        // slack = 1e7 - 1e7*x - y（b-Ax, 元問題空間で正確）
        let slack_expected = 1e7 - 1e7*x - y;
        let tol_rel = 1e-5_f64;
        assert!(
            (result.slack[0] - slack_expected).abs() <= tol_rel * slack_expected.abs().max(1.0),
            "E4: slack[0]={} expected={} (LCS b-Ax精度)", result.slack[0], slack_expected
        );
        assert_eq!(result.reduced_costs.len(), n, "E4: rc.len=2");
    }
    // ========== ここまで postsolve F1/F2修正検証テスト ==========

    /// Q=0のQP（実質LP）をsolve_qp_withで解き、reduced_costsが非空かつ理論値に一致することを確認
    #[test]
    fn test_solve_as_lp_preserves_reduced_costs() {
        // min x + 2y  s.t. x + y >= 1, x >= 0, y >= 0
        // → LP: 最適 x=1, y=0, obj=1
        //
        // reduced_costs 理論値（手計算）:
        //   制約行列 A = [-1, -1]（1×2）, b = [-1]
        //   最適基底: {x}（x=1が基底変数, y=0が非基底）
        //   B = A[:,{x}] = [-1], B^{-1} = -1
        //   双対変数: λ = c_B * B^{-1} = 1.0 * (-1) = -1.0
        //   rc_x = c_x - λ * A[0,x] = 1.0 - (-1.0)(-1.0) = 1.0 - 1.0 = 0.0  (基底変数)
        //   rc_y = c_y - λ * A[0,y] = 2.0 - (-1.0)(-1.0) = 2.0 - 1.0 = 1.0  (非基底変数)
        //   → reduced_costs = [0.0, 1.0]
        let n = 2usize;
        let q = CscMatrix::new(n, n); // ゼロ行列 → LP経路
        let c = vec![1.0, 2.0];
        // x + y >= 1  →  -x - y <= -1
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert_eq!(result.reduced_costs.len(), n,
            "LP path must preserve reduced_costs from Simplex");

        // 値一致アサーション（許容誤差 1e-8）
        let expected = [0.0_f64, 1.0_f64];
        let tol = 1e-8_f64;
        for (j, (&got, &exp)) in result.reduced_costs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tol,
                "reduced_costs[{}]: expected {}, got {} (diff={})",
                j, exp, got, (got - exp).abs()
            );
        }
    }

    // ===== bound_duals col_mapリマップ テスト (BD-T1〜BD-T6) [cmd_689] =====

    /// BD-T1: baseline（presolve OFF, 全変数有限境界あり）
    /// min 1/2*(0.001*x^2 + 0.001*y^2) + x + y
    /// s.t. x + y <= 10, 0 <= x <= 5, 0 <= y <= 5
    /// → 最適解: x=0, y=0（下界活性、上界非活性）
    /// → bound_duals.len() == 4 (lb_x, lb_y, ub_x, ub_y)
    #[test]
    fn test_bd_t1_baseline_presolve_off() {
        let n = 2usize;
        // Q = diag(0.001, 0.001)
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
        let c = vec![1.0, 1.0];
        // x + y <= 10
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        let bounds = vec![(0.0_f64, 5.0_f64); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: false, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T1: status");
        let sol_tol = 1e-3_f64;
        let tol = 1e-4_f64;
        // 解: x=0, y=0
        assert!((result.solution[0]).abs() < sol_tol, "BD-T1: x≈0 (got {})", result.solution[0]);
        assert!((result.solution[1]).abs() < sol_tol, "BD-T1: y≈0 (got {})", result.solution[1]);
        // bound_duals長: n_lb_orig=2 + n_ub_orig=2 = 4
        assert_eq!(result.bound_duals.len(), 4, "BD-T1: bound_duals.len()==4");
        // x=0=lb活性 → lb_dual > 0
        assert!(result.bound_duals[0] > tol, "BD-T1: lb_x>0 (active lower)");
        // y=0=lb活性 → lb_dual > 0
        assert!(result.bound_duals[1] > tol, "BD-T1: lb_y>0 (active lower)");
        // x上界非活性 → ub_dual ≈ 0
        assert!(result.bound_duals[2].abs() < tol, "BD-T1: ub_x≈0 (inactive)");
        // y上界非活性 → ub_dual ≈ 0
        assert!(result.bound_duals[3].abs() < tol, "BD-T1: ub_y≈0 (inactive)");
    }

    /// BD-T2: FixedVar + bound_dualsリマップ（核心テスト、非対称化済み）
    /// min 1/2*(0.001*x^2 + 0.001*y^2 + 0.001*z^2) + 2*x + y + z
    /// s.t. x + y <= 10, 0 <= x <= 5, 0 <= y <= 5, z=3 (fixed)
    /// presolve: z除去（FixedVar）
    /// → リマップ後bound_duals長: 6 (lb_x, lb_y, lb_z=0, ub_x, ub_y, ub_z=0)
    #[test]
    fn test_bd_t2_fixed_var_remap_core() {
        let n = 3usize;
        // Q = diag(0.001, 0.001, 0.001)
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![2.0, 1.0, 1.0];
        // x + y <= 10
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        // z=3 → lb=ub=3（FixedVar）
        let bounds = vec![(0.0_f64, 5.0_f64), (0.0_f64, 5.0_f64), (3.0_f64, 3.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: true, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T2: status");
        let sol_tol = 5e-3_f64; // IPM解の精度（primal解は双対精度より粗め）
        let tol = 1e-4_f64;     // bound_duals精度（符号・大小比較用）
        // 解: x≈0, y≈0, z≈3
        assert!((result.solution[0]).abs() < sol_tol, "BD-T2: x≈0 (got {})", result.solution[0]);
        assert!((result.solution[1]).abs() < sol_tol, "BD-T2: y≈0 (got {})", result.solution[1]);
        assert!((result.solution[2] - 3.0).abs() < sol_tol, "BD-T2: z≈3 (got {})", result.solution[2]);
        // bound_duals長: n_lb_orig=3 + n_ub_orig=3 = 6
        assert_eq!(result.bound_duals.len(), 6, "BD-T2: bound_duals.len()==6");
        // x=0=lb活性 → lb_dual ≈ 2 (目的関数x係数=2)
        assert!(result.bound_duals[0] > tol, "BD-T2: lb_x>0");
        // y=0=lb活性 → lb_dual ≈ 1 (目的関数y係数=1)
        assert!(result.bound_duals[1] > tol, "BD-T2: lb_y>0");
        // 非対称検証: lb_x ≠ lb_y（変数順序バグ検出）
        assert!((result.bound_duals[0] - result.bound_duals[1]).abs() > tol,
            "BD-T2: lb_x({}) != lb_y({}) — 変数順序バグ検出",
            result.bound_duals[0], result.bound_duals[1]);
        // z除去変数 → lb_dual = 0.0
        assert!((result.bound_duals[2]).abs() < tol, "BD-T2: lb_z==0 (removed)");
        // x上界非活性 → ub_dual ≈ 0（IPM精度のため5e-3まで許容）
        assert!(result.bound_duals[3].abs() < 5e-3, "BD-T2: ub_x≈0 (got {})", result.bound_duals[3]);
        // y上界非活性 → ub_dual ≈ 0
        assert!(result.bound_duals[4].abs() < 5e-3, "BD-T2: ub_y≈0 (got {})", result.bound_duals[4]);
        // z除去変数 → ub_dual = 0.0
        assert!((result.bound_duals[5]).abs() < tol, "BD-T2: ub_z==0 (removed)");
        // S2: KKT停止性検証（全変数で ∇f[j] - (A^T y)[j] - lb_dual[j] + ub_dual[j] ≈ 0）
        // ∇f(x*)_x = 0.001*0 + 2 = 2, ∇f(x*)_y = 0.001*0 + 1 = 1
        // dual_solution: x+y<=10 の双対変数（最適解x=y=0なので制約非活性→ dual≈0）
        let dual = if result.dual_solution.is_empty() { 0.0 } else { result.dual_solution[0] };
        // KKT for x: 2 - dual - lb_x + ub_x ≈ 0 → lb_x ≈ 2
        let kkt_x = 2.0 - dual - result.bound_duals[0] + result.bound_duals[3];
        assert!(kkt_x.abs() < 1e-3, "BD-T2: KKT_x≈0, got {}", kkt_x);
        // KKT for y: 1 - dual - lb_y + ub_y ≈ 0 → lb_y ≈ 1
        let kkt_y = 1.0 - dual - result.bound_duals[1] + result.bound_duals[4];
        assert!(kkt_y.abs() < 1e-3, "BD-T2: KKT_y≈0, got {}", kkt_y);
    }

    /// BD-T3: FixedVar + lb_only変数
    /// min 1/2*(0.001*x^2 + 0.001*y^2) + x + y
    /// s.t. x + y <= 10, x >= 0 (ub=∞), y=2 (fixed)
    /// presolve: y除去 → n_lb_orig=2, n_ub_orig=1 → bound_duals.len()==3
    #[test]
    fn test_bd_t3_fixed_var_lb_only() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
        let c = vec![1.0, 1.0];
        // x + y <= 10
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        // x: [0, ∞), y: [2, 2] (fixed)
        let bounds = vec![(0.0_f64, f64::INFINITY), (2.0_f64, 2.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: true, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T3: status");
        // 元問題: lb有限=2(x:0, y:2), ub有限=1(y:2) → bound_duals.len()==3
        assert_eq!(result.bound_duals.len(), 3, "BD-T3: bound_duals.len()==3");
    }

    /// BD-T4: EmptyCol + lb+ub変数（bound_duals空 → 0埋め展開）
    /// min 1/2*(0.001*x^2 + 0.001*y^2) - x - y + z
    /// s.t. x + y <= 4, x,y∈(-∞,∞), 0 <= z <= 3
    /// z は制約に登場しない → EmptyCol → z=lb=0
    /// → IPMが返すbound_duals空 → リマップ後: bound_duals.len()==2, 全0.0
    #[test]
    fn test_bd_t4_empty_col_zero_fill() {
        let n = 3usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 1.0];
        // x + y <= 4
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![4.0];
        // x: (-∞, ∞), y: (-∞, ∞), z: [0, 3]
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY), (0.0_f64, 3.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: true, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T4: status");
        // n_lb_orig=1(z:0), n_ub_orig=1(z:3) → bound_duals.len()==2
        assert_eq!(result.bound_duals.len(), 2, "BD-T4: bound_duals.len()==2");
        let tol = 1e-8_f64;
        assert!((result.bound_duals[0]).abs() < tol, "BD-T4: z_lb==0.0");
        assert!((result.bound_duals[1]).abs() < tol, "BD-T4: z_ub==0.0");
    }

    /// BD-T5: 無境界（全変数±∞ → bound_duals空）
    /// min 1/2*(x^2 + y^2)  s.t. x + y <= 10
    /// x, y ∈ (-∞, +∞)
    #[test]
    fn test_bd_t5_unbounded_vars_empty() {
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T5: status");
        assert!(result.bound_duals.is_empty(), "BD-T5: bound_duals empty for unbounded vars");
    }

    /// BD-T6: FixedVar + ub活性変数（ub_dual非ゼロ × presolve残存変数）
    /// min 1/2*(0.001*x^2 + 0.001*y^2 + 0.001*z^2) - x - y + z
    /// s.t. x + y <= 10, 0 <= x <= 3, 0 <= y <= 5, z=2 (fixed)
    /// → 最適解: x=3(ub活性), y=5(ub活性), z=2
    /// → bound_duals[3]>0 (ub_x活性), bound_duals[4]>0 (ub_y活性)
    #[test]
    fn test_bd_t6_ub_active_with_presolve() {
        let n = 3usize;
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 1.0];
        // x + y <= 10
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        // z=2 → fixed
        let bounds = vec![(0.0_f64, 3.0_f64), (0.0_f64, 5.0_f64), (2.0_f64, 2.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: true, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T6: status");
        let sol_tol = 1e-3_f64; // IPM primal解精度
        let tol = 1e-4_f64;     // bound_duals符号・大小比較精度
        // 最適解: x=3, y=5, z=2
        assert!((result.solution[0] - 3.0).abs() < sol_tol, "BD-T6: x≈3 (got {})", result.solution[0]);
        assert!((result.solution[1] - 5.0).abs() < sol_tol, "BD-T6: y≈5 (got {})", result.solution[1]);
        assert!((result.solution[2] - 2.0).abs() < sol_tol, "BD-T6: z≈2 (got {})", result.solution[2]);
        // bound_duals長: n_lb_orig=3 + n_ub_orig=3 = 6
        assert_eq!(result.bound_duals.len(), 6, "BD-T6: bound_duals.len()==6");
        // x=3=ub活性 → lb_dual≈0, ub_dual>0
        assert!(result.bound_duals[0].abs() < tol, "BD-T6: lb_x≈0 (inactive)");
        assert!(result.bound_duals[1].abs() < tol, "BD-T6: lb_y≈0 (inactive)");
        assert!((result.bound_duals[2]).abs() < tol, "BD-T6: lb_z==0 (removed)");
        assert!(result.bound_duals[3] > tol, "BD-T6: ub_x>0 (active upper)");
        assert!(result.bound_duals[4] > tol, "BD-T6: ub_y>0 (active upper)");
        assert!((result.bound_duals[5]).abs() < tol, "BD-T6: ub_z==0 (removed)");
    }

    /// BD-T7: constraint active × lb_dual nonzero × KKT照合 [cmd_689]
    ///
    /// min 1/2*(x^2 + y^2)
    /// s.t. -x - y <= -3  (等価: x + y >= 3, 常にactive at optimal)
    ///      x >= 2, y >= 0 (x の下界が活性)
    ///
    /// 最適解: x=2, y=1
    ///   - 制約 -x-y=-3 (active) → dual_solution[0] ≈ 1.0 ≠ 0
    ///   - x=2=lb (active)       → bound_duals[0] (lb_x) ≈ 1.0 ≠ 0
    ///   - y=1 > lb=0 (inactive) → bound_duals[1] (lb_y) ≈ 0.0
    ///
    /// KKT停止性 (r_d = -(Qx + c + A_ext^T y_ext) = 0):
    ///   A[0,x]=-1, A[0,y]=-1 なので:
    ///   x: x* + (-1)*dual - lb_x = 0 → 2 - dual - lb_x ≈ 0
    ///   y: y* + (-1)*dual - lb_y = 0 → 1 - dual - lb_y ≈ 0
    #[test]
    fn test_bd_t7_constraint_active_lb_dual_nonzero_kkt() {
        let n = 2usize;
        // Q = I (係数1: 双対値が明確に非ゼロになるよう tiny Q は避ける)
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        // -x - y <= -3  (x + y >= 3)
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
        let b = vec![-3.0];
        // x ∈ [2, ∞), y ∈ [0, ∞) → n_lb=2, n_ub=0 → bound_duals.len()==2
        let bounds = vec![(2.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: false, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T7: status");
        let sol_tol = 1e-3_f64;
        let tol = 1e-4_f64;
        // 最適解: x=2, y=1
        assert!((result.solution[0] - 2.0).abs() < sol_tol,
            "BD-T7: x≈2 (got {})", result.solution[0]);
        assert!((result.solution[1] - 1.0).abs() < sol_tol,
            "BD-T7: y≈1 (got {})", result.solution[1]);
        // bound_duals長: n_lb_orig=2(x,y), n_ub_orig=0 → len==2
        assert_eq!(result.bound_duals.len(), 2, "BD-T7: bound_duals.len()==2");
        // 制約dual ≠ 0 (constraint active)
        let dual = if result.dual_solution.is_empty() { 0.0 } else { result.dual_solution[0] };
        assert!(dual > tol, "BD-T7: constraint dual>0 (active), got {}", dual);
        // lb_x ≠ 0 (x=2=lb, active)
        assert!(result.bound_duals[0] > tol,
            "BD-T7: lb_x>0 (active lower bound), got {}", result.bound_duals[0]);
        // lb_y ≈ 0 (y=1 > lb=0, inactive)
        assert!(result.bound_duals[1].abs() < tol,
            "BD-T7: lb_y≈0 (inactive lower bound), got {}", result.bound_duals[1]);
        // KKT停止性: A[0,j]=-1 なので A^T y の寄与は -dual
        // x: x* - dual - lb_x ≈ 0 → 2 - dual - lb_x
        let kkt_x = result.solution[0] - dual - result.bound_duals[0];
        assert!(kkt_x.abs() < 1e-3, "BD-T7: KKT_x≈0, got {}", kkt_x);
        // y: y* - dual - lb_y ≈ 0 → 1 - dual - lb_y
        let kkt_y = result.solution[1] - dual - result.bound_duals[1];
        assert!(kkt_y.abs() < 1e-3, "BD-T7: KKT_y≈0, got {}", kkt_y);
    }

    /// CscMatrix::row_infinity_norms の基本テスト
    #[test]
    fn test_row_infinity_norms_basic() {
        // 2x3行列:
        // [ 1.0  0.0  -3.0 ]
        // [ 0.0  2.5   0.0 ]
        let a = CscMatrix::from_triplets(
            &[0, 1, 0],    // rows
            &[0, 1, 2],    // cols
            &[1.0, 2.5, -3.0], // vals
            2, 3,
        ).unwrap();
        let norms = a.row_infinity_norms();
        assert_eq!(norms.len(), 2);
        assert!((norms[0] - 3.0).abs() < 1e-15, "row0 norm: expected 3.0, got {}", norms[0]);
        assert!((norms[1] - 2.5).abs() < 1e-15, "row1 norm: expected 2.5, got {}", norms[1]);
    }

    /// 大係数行と小係数行が混在するケースで行ノルム正規化pfeasが正しく機能するテスト [cmd_680]
    ///
    /// 行ノルム正規化なしでは大係数行の残差が全体のpfeasを支配して偽SubOptimalになるが、
    /// 行ノルム正規化ありでは正規化残差が小さく正しくOptimal判定されることを確認
    #[test]
    fn test_pfeas_row_norm_mixed_scale() {
        // 問題: min x^2  s.t. x <= 1 (小係数), 1000*x <= 1000 (大係数)
        // 最適解: x = 0（制約内）
        // 大係数行で微小な数値誤差 x=1e-7 → violation = 1000*1e-7 - 0 = 1e-4
        // 旧方式: pfeas = 1e-4, threshold = eps*(1+1000) ≈ 1e-3 → PASS（この例では旧方式でもPASS）
        // 別の例: x = 1+1e-7 → 大係数行 violation = 1000*(1+1e-7)-1000 = 1e-4
        //         小係数行 violation = (1+1e-7)-1 = 1e-7
        //
        // 行ノルム正規化: 1e-4 / (1+1000+1000) = 5e-8 < eps → PASS

        // 直接row_infinity_normsの正しさを検証
        let a = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 0],
            &[1.0, 1000.0],
            2, 1,
        ).unwrap();
        let norms = a.row_infinity_norms();
        assert!((norms[0] - 1.0).abs() < 1e-15);
        assert!((norms[1] - 1000.0).abs() < 1e-15);

        // 正規化判定のロジック検証
        let b: Vec<f64> = vec![1.0, 1000.0];
        let x_val: f64 = 1.0 + 1e-7; // 微小な制約違反
        let ax: Vec<f64> = vec![x_val, 1000.0 * x_val]; // [1.0000001, 1000.0001]
        let eps: f64 = 1e-6;

        // 旧方式: max violation
        let pfeas_old = ax.iter().zip(b.iter())
            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
            .fold(0.0_f64, f64::max);
        // pfeas_old = max(1e-7, 1e-4) = 1e-4
        assert!(pfeas_old > 1e-5, "旧方式pfeasは大係数行に引きずられるべき: {}", pfeas_old);

        // 新方式: 行ノルム正規化
        let pfeas_normalized = ax.iter().zip(b.iter()).zip(norms.iter())
            .map(|((&ax_i, &b_i), &rn)| {
                let violation = (ax_i - b_i).max(0.0);
                violation / (1.0 + rn + b_i.abs())
            })
            .fold(0.0_f64, f64::max);
        // 大係数行: 1e-4 / (1+1000+1000) = 5e-8
        // 小係数行: 1e-7 / (1+1+1) = 3.3e-8
        assert!(pfeas_normalized < eps, "正規化pfeasはeps未満であるべき: {}", pfeas_normalized);
    }

    /// 正規化なしでは判定が歪むが正規化ありで正しく判定できるケース [cmd_680]
    #[test]
    fn test_pfeas_row_norm_false_suboptimal_prevention() {
        // b=0の大係数行: 1e6*x = 0 (等号制約として)
        // x = 1e-9 → violation = |1e6 * 1e-9 - 0| = 1e-3
        // 旧方式: pfeas = 1e-3, threshold = eps*(1+0).max(1.0) = eps*1 = 1e-6 → FAIL (偽SubOptimal)
        // 新方式: 1e-3 / (1 + 1e6 + 0) ≈ 1e-9 < eps → PASS (正しくOptimal)

        let a = CscMatrix::from_triplets(
            &[0],
            &[0],
            &[1e6],
            1, 1,
        ).unwrap();
        let norms = a.row_infinity_norms();
        assert!((norms[0] - 1e6).abs() < 1e-9);

        let b_val: f64 = 0.0;
        let ax_val: f64 = 1e6 * 1e-9; // = 1e-3
        let eps: f64 = 1e-6;

        // 旧方式: 偽SubOptimal
        let norm_b = b_val.abs().max(1.0); // max(|b|, 1.0) = 1.0
        let pfeas_old = (ax_val - b_val).abs();
        assert!(pfeas_old >= eps * (1.0 + norm_b), "旧方式では偽SubOptimalになるべき");

        // 新方式: 正しくOptimal
        let pfeas_norm = (ax_val - b_val).abs() / (1.0 + norms[0] + b_val.abs());
        assert!(pfeas_norm < eps, "正規化方式ではOptimalであるべき: {}", pfeas_norm);
    }
    }

}
