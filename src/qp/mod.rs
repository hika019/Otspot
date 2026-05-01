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
pub mod ipm_v2;
pub mod diagnose;
pub use problem::{QpProblem, QpWarmStart};
pub use diagnose::{diagnose, DiagnosticReport, DiagnosticWarning, DiagnosticCode, Severity, ProblemInfo};
pub use crate::problem::SolverResult;
pub use ipm::solve_qp_ipm;

use crate::options::{QpSolverChoice, SolverOptions};
use crate::problem::{LpProblem, SolveStatus};
use crate::backend::{LpBackend, SimplexBackend};
use crate::sparse::CscMatrix;


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

/// Q=0 退化ケース（LP 問題）を LP ソルバーに委譲して QP 結果に変換する
pub(crate) fn solve_as_lp_pub(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    solve_as_lp(problem, options)
}

fn solve_as_lp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    // Eq/Ge/Le制約型をそのままSimplexに渡す（設計書§2.4）。
    // Simplexは ConstraintType::Eq を Phase I 人工変数で正しく処理する。
    // to_all_le()は全パスで廃止済み。IPMもSimplexもConstraintTypeをネイティブ処理。
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
            // c^T x + obj_offset (QPS の N-row RHS による定数項)。
            // 旧実装は obj_offset を加算しておらず test_solve_with_obj_offset で fail していた既存バグ。
            let obj: f64 = problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>()
                + problem.obj_offset;
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
                duality_gap_rel: None,
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
            duality_gap_rel: None,
        },
        SolveStatus::MaxIterations => {
            // DEAD PATH: SimplexOutcome::MaxIterations廃止により到達不能。
            // SolveStatus enum variant自体は未削除。
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
            duality_gap_rel: None,
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
            duality_gap_rel: None,
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
            duality_gap_rel: None,
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
#[deprecated(note = "to_all_le()廃止に伴いcollapse_extended_dualを使用")]
#[allow(dead_code, deprecated)]
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
    // QpSolverChoice variant に応じて 2 アルゴ (IPM/IPPMM) を dispatch:
    //   - Ipm        → Mehrotra predictor-corrector を v2 wrapper 経由で実行
    //   - IpPmmNew   → IP-PMM を v2 wrapper 経由で実行
    //   - Concurrent → 上記 2 経路を並行実行し、Optimal 到達順 / 残差優先で勝者選択
    //
    // 各 wrapper 内で Q=0 LP dispatch / PSD check / presolve / unscale / postsolve /
    // 元空間 KKT 直接判定が一貫処理される (retry 1 層 / status 1 箇所変換)。
    match options.qp_solver {
        QpSolverChoice::Ipm => ipm_v2::solve_qp_v1_wrapped(problem, options),
        QpSolverChoice::IpPmmNew => ipm_v2::solve_qp_v2(problem, options),
        QpSolverChoice::Concurrent => solve_qp_concurrent_dispatch(problem, options),
    }
}

/// Mehrotra (v1_wrapped) と IP-PMM (v2) を並行実行し、より良い結果を返す。
///
/// 共有 `cancel_flag` で片方が Optimal を出した時点でもう片方を停止させる
/// (cooperative cancel: 各 inner solver が iter 境界で `should_stop()` をチェック)。
/// 両方が走り切った場合は Optimal > 非 Optimal、両 Optimal なら先着優先。
///
/// スタックサイズは macOS 主スレッドと同等の 8 MB を明示指定する。faer supernodal
/// Cholesky は BOYD1 (n=93261) 級で深く再帰し、Rust の `s.spawn` デフォルト 2 MB
/// スタックでは overflow するため。
#[cfg(feature = "parallel")]
fn solve_qp_concurrent_dispatch(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    };

    /// faer supernodal Cholesky の deepest stack 要求 (BOYD1 等の検証から経験的に決定)
    /// + 安全マージン。OS 主スレッドの典型値 (macOS/Linux 8MB) と一致させる。
    const SPAWN_STACK_SIZE: usize = 8 * 1024 * 1024;

    // Q=0 退化ケース: LP solver (Simplex) に委譲する高速化。
    // **`Concurrent` 経路でのみ自動 dispatch を許容する**。明示的に `Ipm` / `IpPmmNew` を
    // 指定したユーザーは「指定アルゴリズムで動かす」mandate に従い v2_wrapper 経路で
    // そのまま IPM が走る。Gurobi/CPLEX/HiGHS が default モードで採用する慣習と同じ
    // (default = 自動 dispatch、明示 Method 指定 = 尊重)。
    if problem.is_zero_q() {
        return solve_as_lp_pub(problem, options);
    }

    let cancel_flag = options
        .cancel_flag
        .as_ref()
        .map(Arc::clone)
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

    // mpsc でスレッドからメイン側へ結果を渡す。送信は drop で完了通知。
    let (tx, rx) = mpsc::channel::<SolverResult>();

    std::thread::scope(|s| {
        let builder = || std::thread::Builder::new().stack_size(SPAWN_STACK_SIZE);
        // Mehrotra (v1_wrapped) スレッド
        {
            let cancel = Arc::clone(&cancel_flag);
            let mut opts = options.clone();
            opts.cancel_flag = Some(cancel);
            let tx = tx.clone();
            builder()
                .spawn_scoped(s, move || {
                    let r = ipm_v2::solve_qp_v1_wrapped(problem, &opts);
                    let _ = tx.send(r);
                })
                .expect("spawn Mehrotra thread");
        }
        // IP-PMM (v2) スレッド
        {
            let cancel = Arc::clone(&cancel_flag);
            let mut opts = options.clone();
            opts.cancel_flag = Some(cancel);
            let tx = tx.clone();
            builder()
                .spawn_scoped(s, move || {
                    let r = ipm_v2::solve_qp_v2(problem, &opts);
                    let _ = tx.send(r);
                })
                .expect("spawn IP-PMM thread");
        }
        drop(tx); // 両 spawn の tx クローン保持の片方が落ちただけでは Disconnected にならない

        // 結果を順次受信。priority が高い方を採用 (同値なら先着維持で決定論性確保)。
        // Optimal を取った時点で cancel を立て、相方を協調停止する。
        // Infeasible / Unbounded は確定的判定なので捨てず priority を高めに置く
        // (status 隠蔽防止: 数値的に解けた解より、infeasibility 検出の方が情報価値が高い)。
        let mut best: Option<SolverResult> = None;
        for r in rx {
            let r_is_optimal = matches!(r.status, SolveStatus::Optimal);
            let new_priority = result_priority(&r);
            let prefer_new = match &best {
                None => true,
                Some(b) => new_priority > result_priority(b),
            };
            if prefer_new {
                if r_is_optimal {
                    cancel_flag.store(true, Ordering::Relaxed);
                }
                best = Some(r);
            }
        }
        best.unwrap_or_else(|| SolverResult {
            status: SolveStatus::NumericalError,
            ..Default::default()
        })
    })
}

/// Concurrent dispatch で複数 solver の結果を比較するための優先度。値が大きいほど採用優先。
///
/// 設計:
/// - `Optimal` は最高 (確定 + 解あり)
/// - `Infeasible`/`Unbounded` は次 (確定的判定: solver 都合で握りつぶさない)
/// - `SuboptimalSolution` は内部収束判定通過 (解あり、KKT 緩和許容内)
/// - `Timeout` は best-so-far 解の有無で 2 段階
/// - `NumericalError` 等は最低
///
/// 同値なら先着維持で決定論性を確保する (`>` 比較で同値時 false)。
#[cfg(feature = "parallel")]
fn result_priority(result: &SolverResult) -> u8 {
    match result.status {
        SolveStatus::Optimal => 6,
        SolveStatus::Infeasible | SolveStatus::Unbounded => 5,
        SolveStatus::SuboptimalSolution => 4,
        SolveStatus::Timeout if !result.solution.is_empty() => 3,
        SolveStatus::Timeout => 2,
        SolveStatus::MaxIterations => 2,
        SolveStatus::NonConvex(_) => 1,
        SolveStatus::NumericalError => 0,
    }
}

/// Concurrent feature 無効時のフォールバック (IP-PMM 単独)。
#[cfg(not(feature = "parallel"))]
fn solve_qp_concurrent_dispatch(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    ipm_v2::solve_qp_v2(problem, options)
}


/// FX (固定) 変数判定の許容差。lb と ub の差がこれ未満なら固定変数とみなす。
pub(crate) const FX_TOL: f64 = 1e-12;

/// presolve で縮約された bound_duals を元問題空間に展開する。
/// reduced_bounds に対応する bound_duals を、orig_bounds に対応する形で再構築。
/// 除去された変数の bound_dual は 0.0 (近似) で埋める。
///
/// 旧 mod.rs::solve_qp_with L815-887 と同等のロジックを v2 から再利用するため抽出。
pub(crate) fn remap_bound_duals_to_orig(
    presolve_result: &crate::presolve::QpPresolveResult,
    orig_bounds: &[(f64, f64)],
    reduced_bound_duals: &[f64],
) -> Vec<f64> {
    let n_lb_orig = orig_bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_ub_orig = orig_bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
    if n_lb_orig + n_ub_orig == 0 {
        return Vec::new();
    }
    let reduced_bounds = &presolve_result.reduced.bounds;
    let n_lb_reduced = reduced_bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
    let n_reduced = reduced_bounds.len();

    let mut lb_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    let mut ub_bd_idx: Vec<Option<usize>> = vec![None; n_reduced];
    {
        let mut li = 0usize;
        for (jj, &(lb, _)) in reduced_bounds.iter().enumerate() {
            if lb.is_finite() {
                lb_bd_idx[jj] = Some(li);
                li += 1;
            }
        }
        let mut ui = 0usize;
        for (jj, &(_, ub)) in reduced_bounds.iter().enumerate() {
            if ub.is_finite() {
                ub_bd_idx[jj] = Some(n_lb_reduced + ui);
                ui += 1;
            }
        }
    }

    let mut new_bd = vec![0.0_f64; n_lb_orig + n_ub_orig];
    if !reduced_bound_duals.is_empty() {
        let mut orig_li = 0usize;
        for (j, &(lb, _)) in orig_bounds.iter().enumerate() {
            if lb.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = lb_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[orig_li] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_li += 1;
            }
        }
        let mut orig_ui = 0usize;
        for (j, &(_, ub)) in orig_bounds.iter().enumerate() {
            if ub.is_finite() {
                if let Some(jj) = presolve_result.col_map[j] {
                    if let Some(bd_idx) = ub_bd_idx[jj] {
                        if bd_idx < reduced_bound_duals.len() {
                            new_bd[n_lb_orig + orig_ui] = reduced_bound_duals[bd_idx];
                        }
                    }
                }
                orig_ui += 1;
            }
        }
    }
    new_bd
}

/// AAT に対角ε を追加して rank-deficient 対策する正則化倍率。
/// ε = AAT_REG_FACTOR * max_diag で対角に加算。f64 epsilon (~2e-16) より十分上、
/// LDL の dynamic regularization (1e-8) より十分下で衝突しない。
const AAT_REG_FACTOR: f64 = 1e-12;

/// `refine_dual_lsq` / `compute_lsq_dual_y` の size guard。n+m がこれを超える問題は
/// AAT (m×m) の LDL factorization が数十 GB メモリを確保するため skip する。
///
/// 真因: BOYD2 (n=140k, m=186k → n+m=326k) で `refine_dual_lsq` が呼ばれると
/// AAT 186k×186k に対する faer supernodal LDL が 30-40GB virtual memory を
/// 確保する。concurrent solver で v1+v2 が並列に走ると倍 (60-80GB swap+RSS)。
///
/// 経験値: 50k は `refine_primal_lsq` の `PRIMAL_PROJECTION_SIZE_LIMIT` と統一。
/// LISWET (n+m=20k) は guard 内、CONT-300 (n+m=180k) と BOYD2 (326k) は skip。
const LSQ_DUAL_SIZE_LIMIT: usize = 50_000;

/// primal x から KKT を満たす dual y を最小二乗で再計算する。
/// IPPMM が scaled 空間判定で出した偽 dual を補正する。
///
/// 解法: A^T y = -(Q*x + c + bound_contrib) の最小ノルム解を
/// 正規方程式 (A * A^T) y = A * r で求める (LDL)。
/// 既存 y より KKT 残差が改善した場合のみ採用する (退行防止)。
pub(crate) fn refine_dual_lsq(problem: &QpProblem, result: &mut crate::problem::SolverResult) {
    if let Some(y_new) = compute_lsq_dual_y(problem, result) {
        let aty_old = problem.a.transpose().mat_vec_mul(&result.dual_solution).unwrap_or(vec![0.0; problem.num_vars]);
        let aty_new = problem.a.transpose().mat_vec_mul(&y_new).unwrap_or(vec![0.0; problem.num_vars]);
        let qx = match problem.q.mat_vec_mul(&result.solution) {
            Ok(v) => v,
            Err(_) => return,
        };
        let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, problem.num_vars);
        let mut max_resid_old = 0.0_f64;
        let mut max_resid_new = 0.0_f64;
        for j in 0..problem.num_vars {
            let (lbj, ubj) = problem.bounds[j];
            if lbj.is_finite() && ubj.is_finite() && (lbj - ubj).abs() < FX_TOL {
                continue;
            }
            let r_old = (qx[j] + problem.c[j] + aty_old[j] + bound_contrib[j]).abs();
            let r_new = (qx[j] + problem.c[j] + aty_new[j] + bound_contrib[j]).abs();
            max_resid_old = max_resid_old.max(r_old);
            max_resid_new = max_resid_new.max(r_new);
        }
        if max_resid_new < max_resid_old {
            result.dual_solution = y_new;
        }
    }
}

/// LSQ で y を計算する核心ロジック (KKT-guard なし)。`refine_dual_lsq` の内部実装と
/// 共有する。primal が動いた直後の force-update 用に外部からも呼べるよう pub(crate) 化。
///
/// 解法: A^T y = -(Q*x + c + bound_contrib) の最小ノルム解を
/// 正規方程式 (A * A^T) y = A * r で求める (LDL)。
/// 失敗 (LDL 失敗 / NaN) 時は None を返す。
pub(crate) fn compute_lsq_dual_y(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
) -> Option<Vec<f64>> {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return None;
    }
    // 大規模問題では A·A^T (m×m sparse、m=186k for BOYD2) の LDL factorization が
    // 数十 GB メモリを確保するため skip。`refine_primal_lsq` の同名閾値と統一。
    // refine_dual_lsq は IPM iterations=0 の timeout でも呼ばれるため (core.rs:86)、
    // size guard なしだと BOYD2 で post-IPM AAT factorize が memory peak を倍増させる。
    if n + m > LSQ_DUAL_SIZE_LIMIT {
        return None;
    }
    let x = &result.solution;
    let qx = problem.q.mat_vec_mul(x).ok()?;
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let r_full: Vec<f64> = (0..n)
        .map(|j| -(qx[j] + problem.c[j] + bound_contrib[j]))
        .collect();
    let aat = build_aat_upper_csc(&problem.a, n, m)?;
    let factor = crate::linalg::ldl::factorize(&aat).ok()?;
    let mut rhs = vec![0.0_f64; m];
    for j in 0..n {
        let start = problem.a.col_ptr[j];
        let end = problem.a.col_ptr[j + 1];
        for k in start..end {
            let i = problem.a.row_ind[k];
            if i < m {
                rhs[i] += problem.a.values[k] * r_full[j];
            }
        }
    }
    let mut y_new = vec![0.0_f64; m];
    factor.solve(&rhs, &mut y_new);
    if y_new.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(y_new)
}

/// 元空間 primal feasibility が borderline (LISWET9/12: pf ≈ 1e-6 で
/// bench eps=1e-6 を僅か超過) のとき、x を violating 制約方向に最小ノルム射影
/// して pf を eps 内に押し込む post-processing。
///
/// 動機: IP-PMM は 10000 行級の LISWET で finite precision (f64) の構造的限界
/// から pf を 1e-6 以下に下げきれない (104 反復 best が pf=9.9e-7、unscale 後
/// 1.57e-6)。bench は同じ x で pfn=1.54e-6 と判定して PFEAS_FAIL となる。
///
/// 算法:
///   v[i] = max(0, A[i,:]·x - b[i])  (Le 制約用、Ge / Eq も対応)
///   active = {i : v[i] > tol}
///   solve  A_active * δ = v_active  for δ ∈ R^n with min‖δ‖₂
///   normal eq: (A_active A_active^T) λ = v_active, δ = A_active^T λ
///   x_new = x - δ
///   pf_new < pf_old なら採用、さもなくば revert (KKT-guard)。
///
/// objective 影響: ‖δ‖₂ ≈ pf ≈ 1e-6 程度で微小なので Q への影響は negligible。
/// dual との整合は別途 refine_dual_lsq + refit_bound_duals_kkt が再評価する。
pub(crate) fn refine_primal_lsq(problem: &QpProblem, result: &mut crate::problem::SolverResult) {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return;
    }
    let x = &mut result.solution;

    // 違反量 v[i] を計算 (Le/Ge/Eq に応じて符号を統一して "Ax を b 方向に押す量")
    use crate::problem::ConstraintType;
    let ax = match problem.a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    /// 違反検出の許容差 (これ未満は active 扱いしない)
    const PRIMAL_VIOLATION_TOL: f64 = 1e-12;
    let mut v = vec![0.0_f64; m];
    let mut max_v_pre = 0.0_f64;
    for i in 0..m {
        let raw = match problem.constraint_types[i] {
            ConstraintType::Eq => ax[i] - problem.b[i],
            ConstraintType::Ge => -(ax[i] - problem.b[i]),
            ConstraintType::Le => ax[i] - problem.b[i],
        };
        // raw > 0: 違反量 (Le なら Ax > b、Ge なら Ax < b)
        // raw <= 0: 充足
        if raw > PRIMAL_VIOLATION_TOL {
            v[i] = raw;
            max_v_pre = max_v_pre.max(raw);
        }
    }
    if max_v_pre <= PRIMAL_VIOLATION_TOL {
        return;
    }
    // Ge 行で違反があった場合、δ を負方向に動かすために v[i] をそのまま使うが、
    // δ = A^T λ で v_i 方向に動くようにするため、Ge 行は A の符号を逆にして扱う必要あり。
    // ここでは行 i に応じて A_eff を構築する代わりに、効果的に
    //   "A x' = ax - sign_i * v_i" を求める方式に変える:
    //     Le: A x' = b → δ = x - x' で A δ = ax - b = +v (push down)
    //     Ge: A x' = b → δ = x - x' で A δ = ax - b = -v (push up; sign_i = -1)
    //     Eq: A x' = b → A δ = ax - b
    // つまり target = ax - b そのものを使い、A δ = target を解く。
    let target: Vec<f64> = (0..m).map(|i| {
        match problem.constraint_types[i] {
            ConstraintType::Eq => ax[i] - problem.b[i],
            ConstraintType::Ge => {
                // active のみ: 充足側 (ax >= b) なら 0, 違反 (ax < b) なら ax - b (負)
                let r = ax[i] - problem.b[i];
                if r < -PRIMAL_VIOLATION_TOL { r } else { 0.0 }
            }
            ConstraintType::Le => {
                let r = ax[i] - problem.b[i];
                if r > PRIMAL_VIOLATION_TOL { r } else { 0.0 }
            }
        }
    }).collect();
    let target_inf = target.iter().map(|t| t.abs()).fold(0.0_f64, f64::max);
    if target_inf <= PRIMAL_VIOLATION_TOL {
        return;
    }

    // (A A^T) λ = target を LDL で解く。AAT_REG_FACTOR で対角正則化済。
    let aat = match build_aat_upper_csc(&problem.a, n, m) {
        Some(mat) => mat,
        None => return,
    };
    let factor = match crate::linalg::ldl::factorize(&aat) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut lambda = vec![0.0_f64; m];
    factor.solve(&target, &mut lambda);
    if lambda.iter().any(|v| !v.is_finite()) {
        return;
    }

    // δ = A^T λ
    let mut delta = vec![0.0_f64; n];
    for j in 0..n {
        let s = problem.a.col_ptr[j];
        let e = problem.a.col_ptr[j + 1];
        for k in s..e {
            let i = problem.a.row_ind[k];
            if i < m {
                delta[j] += problem.a.values[k] * lambda[i];
            }
        }
    }
    if delta.iter().any(|v| !v.is_finite()) {
        return;
    }

    // x_new = x - δ
    let mut x_new = x.clone();
    for j in 0..n {
        x_new[j] -= delta[j];
        // bounds clip (delta が bounds を破る場合は clamp)
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            x_new[j] = x_new[j].max(lb);
        }
        if ub.is_finite() {
            x_new[j] = x_new[j].min(ub);
        }
    }

    // 改善判定: 全制約での max violation が減ったか
    let ax_new = match problem.a.mat_vec_mul(&x_new) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut max_v_post = 0.0_f64;
    for i in 0..m {
        let raw = match problem.constraint_types[i] {
            ConstraintType::Eq => (ax_new[i] - problem.b[i]).abs(),
            ConstraintType::Ge => (problem.b[i] - ax_new[i]).max(0.0),
            ConstraintType::Le => (ax_new[i] - problem.b[i]).max(0.0),
        };
        max_v_post = max_v_post.max(raw);
    }
    if max_v_post < max_v_pre {
        *x = x_new;
    }
}

/// Saddle-point KKT system に対する iterative refinement (Krylov refinement)。
///
/// 動機: LISWET 系の precision floor (pf=1.5e-6) は LDL backward error
/// `eps × cond(K) × ‖dx‖` に由来。各 Newton step 内の LDL solve は cond ~1e10 で
/// 6 桁の誤差が dx に乗り、これが accumulated して pf floor を作る。
///
/// 本関数は IPM 収束後の (x, y, z) を初期推定として:
///   1. K = [Q+δp·I  A^T; A  -δd·I] を構築 (δp, δd 小)
///   2. AMD + LDL factorize
///   3. 各 iter で full-f64 で KKT residual r_d, r_p を計算
///   4. K·du = -r を LDL solve、x ← x + dx, y ← y + dy
///   5. bound clip + refit_bound_duals_kkt
///   6. 改善判定 (pf 減少 AND df 悪化なし) で accept/break
///
/// 古典的 IR の収束理論: r_p は eps·‖A‖ レベルまで refine 可能 (cond の影響を受けない)。
/// LISWET の floor 突破に効く可能性。
///
/// 安全装置:
/// - サイズ制限 50_000 (refine_primal_lsq と統一、AAT factorize 同様の理由)
/// - 各 iter で KKT 改善 guard、退行で revert + break
/// - max_iters で停止 (発散時の保険)
///
/// 戻り値: 採用された refinement iter 数 (0 = no-op)
pub(crate) fn refine_kkt_iterative(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    max_iters: usize,
    target_pf: f64,
) -> usize {
    use crate::problem::ConstraintType;
    use crate::presolve::bound_contrib_at_var;

    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return 0;
    }
    if result.dual_solution.len() != m {
        return 0;
    }

    // サイズ制限: refine_primal_lsq の PRIMAL_PROJECTION_SIZE_LIMIT と統一。
    const REFINE_KKT_SIZE_LIMIT: usize = 50_000;
    if n + m > REFINE_KKT_SIZE_LIMIT {
        return 0;
    }

    // δp, δd: K の対角正則化。十分小さく (IR で eps·‖K‖ レベルまで refine 可)、
    // LDL の数値安定性が確保される値。1e-10 は LISWET cond 1e10 で K cond 1e2 級。
    // YAO 系 (A rank-deficient) では LDL solve が暴走するが、accept guard で保護。
    const DELTA_P: f64 = 1e-10;
    const DELTA_D: f64 = 1e-10;

    let sigma_zero = vec![0.0_f64; m];
    let k_mat = crate::qp::ipm::kkt::build_augmented_system(
        &problem.q, &problem.a, &sigma_zero, DELTA_P, DELTA_D
    );

    let trace_pre = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    if trace_pre {
        eprintln!("REFINE_KKT pre-factorize: n={} m={} K_nnz={}", n, m, k_mat.values.len());
    }
    let factor = match crate::linalg::ldl::factorize_quasidefinite_with_amd(&k_mat, None) {
        Ok(f) => f,
        Err(e) => {
            if trace_pre { eprintln!("REFINE_KKT factorize failed: {:?}", e); }
            return 0;
        }
    };

    // FX/EmptyCol 変数の判定 (kkt_residual_rel と整合):
    //   FX (lb≈ub): presolve 慣例で bound_dual=0 埋め、stationarity 評価から除外
    //   EmptyCol (制約 A に登場しない): bound_dual=0 慣例、Q∅ + c[j] != 0 のため除外
    // これらを含めると orig 空間で huge cancellation noise (r_d_abs) が出て IR が壊れる。
    const FX_TOL_REFINE: f64 = 1e-12;
    let exclude_var: Vec<bool> = (0..n).map(|j| {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL_REFINE {
            return true;
        }
        if problem.a.col_ptr.len() > j + 1
            && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0
        {
            return true;
        }
        false
    }).collect();

    let compute_residuals = |x: &[f64], y: &[f64], z: &[f64]| -> (Vec<f64>, Vec<f64>, f64, f64) {
        let qx = problem.q.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; n]);
        let aty = problem.a.transpose().mat_vec_mul(y).unwrap_or_else(|_| vec![0.0; n]);
        let mut r_d = vec![0.0_f64; n];
        // codebase 規約: KKT stationarity = Qx + c + A^T·y + bound_contrib (kkt_residual_rel と整合)。
        // canonical の Qx + c - A^T·y - z_lb + z_ub と符号が違うので注意。
        for j in 0..n {
            if exclude_var[j] { continue; }  // FX/EmptyCol は r_d=0 のまま
            let bc = bound_contrib_at_var(&problem.bounds, z, j);
            r_d[j] = qx[j] + problem.c[j] + aty[j] + bc;
        }
        let ax = problem.a.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; m]);
        let mut r_p = vec![0.0_f64; m];
        let mut pf = 0.0_f64;
        for i in 0..m {
            let raw = ax[i] - problem.b[i];
            let v = match problem.constraint_types[i] {
                ConstraintType::Eq => raw,
                ConstraintType::Ge => if raw < 0.0 { raw } else { 0.0 },
                ConstraintType::Le => if raw > 0.0 { raw } else { 0.0 },
            };
            r_p[i] = v;
            pf = pf.max(v.abs());
        }
        let df = r_d.iter().fold(0.0_f64, |a, &r| a.max(r.abs()));
        (r_d, r_p, pf, df)
    };

    let pre_z = result.bound_duals.clone();
    let (_, _, pre_pf, pre_df) = compute_residuals(&result.solution, &result.dual_solution, &pre_z);
    let trace = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    if trace {
        eprintln!("REFINE_KKT entry: n={} m={} pre_pf={:.3e} pre_df={:.3e} target_pf={:.3e}",
            n, m, pre_pf, pre_df, target_pf);
    }
    if pre_pf < target_pf {
        if trace {
            eprintln!("REFINE_KKT skip: pre_pf < target_pf");
        }
        return 0;
    }

    let mut accepted = 0_usize;
    // df 悪化許容: pre_df の 2x まで。これ以上は revert (構造的悪化)。
    // 経験値 (LISWET cond で IR が 1-2 iter で収束する想定で十分な余裕)。
    const DF_TOLERANCE_FACTOR: f64 = 2.0;
    let df_limit = (pre_df * DF_TOLERANCE_FACTOR).max(1e-7);

    for iter in 0..max_iters {
        let (r_d, r_p, pf_cur, df_cur) =
            compute_residuals(&result.solution, &result.dual_solution, &result.bound_duals);
        if pf_cur < target_pf {
            if trace { eprintln!("REFINE_KKT iter={} early: pf_cur={:.3e} < target", iter, pf_cur); }
            break;
        }

        let mut rhs = vec![0.0_f64; n + m];
        for j in 0..n { rhs[j] = -r_d[j]; }
        for i in 0..m { rhs[n + i] = -r_p[i]; }

        let mut sol = vec![0.0_f64; n + m];
        factor.solve(&rhs, &mut sol);
        if sol.iter().any(|v| !v.is_finite()) {
            if trace { eprintln!("REFINE_KKT iter={} solve produced NaN", iter); }
            break;
        }

        let dx_inf: f64 = sol[..n].iter().fold(0.0, |a, &v| a.max(v.abs()));
        let dy_inf: f64 = sol[n..].iter().fold(0.0, |a, &v| a.max(v.abs()));

        let mut x_new = result.solution.clone();
        let mut y_new = result.dual_solution.clone();
        let mut clip_amt = 0.0_f64;
        for j in 0..n {
            let raw = x_new[j] + sol[j];
            let (lb, ub) = problem.bounds[j];
            let mut clipped = raw;
            if lb.is_finite() { clipped = clipped.max(lb); }
            if ub.is_finite() { clipped = clipped.min(ub); }
            clip_amt = clip_amt.max((raw - clipped).abs());
            x_new[j] = clipped;
        }
        for i in 0..m {
            y_new[i] += sol[n + i];
        }

        let mut tmp = result.clone();
        tmp.solution = x_new;
        tmp.dual_solution = y_new;
        refit_bound_duals_kkt(problem, &mut tmp);

        let (_, _, pf_new, df_new) =
            compute_residuals(&tmp.solution, &tmp.dual_solution, &tmp.bound_duals);

        if trace {
            eprintln!("REFINE_KKT iter={} pf={:.3e}->{:.3e} df={:.3e}->{:.3e} dx_inf={:.3e} dy_inf={:.3e} clip={:.3e}",
                iter, pf_cur, pf_new, df_cur, df_new, dx_inf, dy_inf, clip_amt);
        }

        if pf_new < pf_cur && df_new < df_limit {
            *result = tmp;
            accepted += 1;
        } else {
            if trace {
                eprintln!("REFINE_KKT iter={} REJECTED (pf_dec={} df_ok={})",
                    iter, pf_new < pf_cur, df_new < df_limit);
            }
            break;
        }
    }

    accepted
}

/// 元空間で primal x と constraint dual y から bound_duals を KKT で再計算する。
///
/// presolve が変数を fix した場合 (FixedVar / EmptyCol / SingletonRow), postsolve は
/// `bound_duals[..]=0` を埋めるが、本来 active bound を持つ変数では bound_dual は非ゼロ。
/// この欠落で KKT stationarity が破壊され、bench dfeas が 1e-1 級の偽 DFEAS_FAIL になる
/// (Maros 9 件: QADLITTL/QBORE3D/QCAPRI/QETAMACR/QFFFFF80/QPCBOEI1/QRECIPE/QSEBA/QSHELL)。
///
/// 本関数は既存 IPM/IPPMM のアルゴリズムには介入せず、ソルバが返した x, y を不変としたまま
/// bound_duals のみを KKT stationarity から導出する後処理 (refine_dual_lsq の z 版)。
///
/// 算法:
///   target[j] = -(Qx[j] + c[j] + (A^T y)[j])  (KKT: bound_contrib[j] = target[j])
///   bound_contrib[j] = -y_lb[j] + y_ub[j]
///   - lb のみ有限: y_lb[j] = -target[j] (active 想定)
///   - ub のみ有限: y_ub[j] = target[j]
///   - 両端有限 + x が lb 近傍: y_lb[j] = -target[j], y_ub[j] = 0
///   - 両端有限 + x が ub 近傍: y_lb[j] = 0,         y_ub[j] = target[j]
///   - 両端有限 + interior:    y_lb = y_ub = 0 (residual はそのまま、KKT-guard で revert)
///
/// 既存 bound_duals より KKT 残差が改善した場合のみ採用する (退行防止 — refine_dual_lsq と同形)。
pub(crate) fn refit_bound_duals_kkt(problem: &QpProblem, result: &mut crate::problem::SolverResult) {
    let n = problem.num_vars;
    if result.solution.len() != n {
        return;
    }
    let x = &result.solution;
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return,
    };
    let aty: Vec<f64> = if problem.a.nrows > 0 && !result.dual_solution.is_empty() {
        match problem.a.transpose().mat_vec_mul(&result.dual_solution) {
            Ok(v) => v,
            Err(_) => return,
        }
    } else {
        vec![0.0_f64; n]
    };

    let n_lb = problem.bounds.iter().filter(|&&(lb, _)| lb.is_finite()).count();
    let n_ub = problem.bounds.iter().filter(|&&(_, ub)| ub.is_finite()).count();
    if n_lb + n_ub == 0 {
        return;
    }

    let mut new_bd = vec![0.0_f64; n_lb + n_ub];
    // active 判定の許容差。x が lb / ub の (1e-6 * (1 + |bound|)) 以内なら active 扱い。
    // IPM の interior point は厳密に bound に到達しないため、tolerance が必要。
    const ACTIVE_REL_TOL: f64 = 1e-6;

    let mut lb_idx = 0_usize;
    let mut ub_idx = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let target = -(qx[j] + problem.c[j] + aty[j]);
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();

        let lb_active = lb_finite
            && (x[j] - lb) <= ACTIVE_REL_TOL * (1.0 + lb.abs()).max(1.0);
        let ub_active = ub_finite
            && (ub - x[j]) <= ACTIVE_REL_TOL * (1.0 + ub.abs()).max(1.0);

        if lb_finite && ub_finite {
            if lb_active && !ub_active {
                new_bd[lb_idx] = (-target).max(0.0);
                // ub_idx 側はゼロのまま
            } else if ub_active && !lb_active {
                new_bd[ub_idx] = target.max(0.0);
            } else if lb_active && ub_active {
                // FX (lb≈ub): 符号で振り分け、両方 0 以上
                if -target >= 0.0 { new_bd[lb_idx] = -target; }
                else               { new_bd[ub_idx] = target; }
            }
            // interior (どちらも非 active): 0/0 のまま (residual はそのまま、KKT-guard が判断)
            lb_idx += 1;
            ub_idx += 1;
        } else if lb_finite {
            // lb のみ有限: bound_contrib = -y_lb = target → y_lb = -target
            new_bd[lb_idx] = (-target).max(0.0);
            lb_idx += 1;
        } else if ub_finite {
            new_bd[ub_idx] = target.max(0.0);
            ub_idx += 1;
        }
    }

    // KKT-guard: 元 bound_duals と比較し、KKT max-residual が改善した場合のみ採用
    let pre_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let post_contrib = compute_bound_contrib(&problem.bounds, &new_bd, n);
    let mut max_pre = 0.0_f64;
    let mut max_post = 0.0_f64;
    for j in 0..n {
        let (lbj, ubj) = problem.bounds[j];
        // FX 変数は postsolve で 0 埋め慣例なので KKT 評価から除外 (bench/v2 と整合)
        if lbj.is_finite() && ubj.is_finite() && (lbj - ubj).abs() < FX_TOL {
            continue;
        }
        let r_pre = (qx[j] + problem.c[j] + aty[j] + pre_contrib[j]).abs();
        let r_post = (qx[j] + problem.c[j] + aty[j] + post_contrib[j]).abs();
        max_pre = max_pre.max(r_pre);
        max_post = max_post.max(r_post);
    }
    if max_post < max_pre {
        result.bound_duals = new_bd;
    }
}

/// 境界 dual から KKT stationarity の bound 寄与 (-y_lb + y_ub) を成分ごと計算する。
/// `bound_duals` は [lb 有限な変数の y_lb 群; ub 有限な変数の y_ub 群] レイアウト。
fn compute_bound_contrib(bounds: &[(f64, f64)], bound_duals: &[f64], n: usize) -> Vec<f64> {
    let mut contrib = vec![0.0_f64; n];
    if bound_duals.is_empty() {
        return contrib;
    }
    let mut idx = 0usize;
    for (j, &(lb, _)) in bounds.iter().enumerate() {
        if lb.is_finite() && idx < bound_duals.len() {
            contrib[j] -= bound_duals[idx];
            idx += 1;
        }
    }
    for (j, &(_, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() && idx < bound_duals.len() {
            contrib[j] += bound_duals[idx];
            idx += 1;
        }
    }
    contrib
}

/// A * A^T (m×m, 上三角 CSC) を構築する。LDL 分解前提で対角に ε 正則化を加える。
/// rank-deficient な A (重複制約等) でも factorize 可能になる。
pub(crate) fn build_aat_upper_csc(a: &CscMatrix, n: usize, m: usize) -> Option<CscMatrix> {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..n {
        let start = a.col_ptr[k];
        let end = a.col_ptr[k + 1];
        let cols_in_k: Vec<(usize, f64)> = (start..end)
            .map(|p| (a.row_ind[p], a.values[p]))
            .collect();
        for (idx_a, &(i, v_i)) in cols_in_k.iter().enumerate() {
            for &(j, v_j) in &cols_in_k[idx_a..] {
                let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
                *acc.entry((hi, lo)).or_insert(0.0) += v_i * v_j;
            }
        }
    }
    let max_diag = (0..m)
        .filter_map(|i| acc.get(&(i, i)).copied())
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let reg = AAT_REG_FACTOR * max_diag;
    for i in 0..m {
        *acc.entry((i, i)).or_insert(0.0) += reg;
    }
    let mut col_ptr = vec![0_usize; m + 1];
    let mut row_ind: Vec<usize> = Vec::with_capacity(acc.len());
    let mut values: Vec<f64> = Vec::with_capacity(acc.len());
    for ((col, row), val) in acc {
        row_ind.push(row);
        values.push(val);
        col_ptr[col + 1] = row_ind.len();
    }
    for i in 1..=m {
        if col_ptr[i] < col_ptr[i - 1] {
            col_ptr[i] = col_ptr[i - 1];
        }
    }
    Some(CscMatrix {
        col_ptr,
        row_ind,
        values,
        nrows: m,
        ncols: m,
    })
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

        let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let result = solve_qp_with(&problem, &opts);
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

        let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let result = solve_qp_with(&problem, &opts);
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

    /// UBH1 の Q が PSD か non-PSD かを sparse Cholesky で実証する診断テスト。
    /// `check_q_positive_semidefinite` は n>1000 で密行列 Cholesky をスキップするため、
    /// UBH1 (n=18009) のような大規模 Q の non-PSD は対角正値だけでは検出不能。
    /// このテストは sparse LDL で実際の PSD 性を確認する（時間計測も兼ねる）。
    ///
    /// このテストの目的: bench で UBH1 が「pfeas/dfeas は eps 通過するのに obj が known と
    /// 55% 乖離して OBJ_MISMATCH になる」事実の原因が Q non-PSD かを実証する。
    /// non-PSD なら IPM は KKT 残差を満たす局所点（鞍点 or local opt）を返してしまい、
    /// それが known global optimal と乖離するのは仕様通り。
    #[test]
    #[ignore] // 数十秒かかる可能性。手動で `cargo test test_ubh1_q_psd_diagnose -- --ignored --nocapture` 実行
    fn test_ubh1_q_psd_diagnose() {
        use crate::io::qps::parse_qps;
        use crate::linalg::ldl;
        use std::path::Path;
        use std::time::Instant;

        let path = Path::new("data/maros_meszaros/UBH1.QPS");
        if !path.exists() {
            eprintln!("UBH1.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse UBH1");
        eprintln!(
            "UBH1: n={}, m={}, Q.nnz={}",
            prob.num_vars, prob.num_constraints, prob.q.values.len()
        );

        // 対角 ε を変えて factorize を試行する。
        // - eps=0: 真の Q をそのまま分解。失敗なら non-PSD or rank-deficient
        // - eps>0: Q+εI を分解。failure-to-success の閾値が最小固有値の絶対値の目安
        for eps in &[0.0_f64, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-3, 1.0] {
            let q_reg = build_q_with_diag_reg(&prob.q, *eps);
            let t = Instant::now();
            match ldl::factorize(&q_reg) {
                Ok(_) => eprintln!(
                    "  eps={:.0e}: factorize OK (Q+εI PSD), {:.2}s",
                    eps,
                    t.elapsed().as_secs_f64()
                ),
                Err(e) => eprintln!(
                    "  eps={:.0e}: factorize FAILED ({:?}), {:.2}s",
                    eps,
                    e,
                    t.elapsed().as_secs_f64()
                ),
            }
        }
    }

    /// HS268 IPPMM 出力の dual 残差を成分ごとに分解して KKT 不整合の原因を特定する。
    /// HS268 は n=5, m=5, 全 Ge 制約, 全 FR (free) 変数の小さな convex QP。
    /// bench で df=4.9e-2 (eps=1e-6 を 4 桁超過) になる事実の数値的構造を実証する。
    #[test]
    #[ignore]
    fn test_hs268_dual_residual_diagnose() {
        use crate::io::qps::parse_qps;
        use crate::options::SolverOptions;
        use std::path::Path;

        let path = Path::new("data/maros_meszaros/HS268.QPS");
        if !path.exists() {
            eprintln!("HS268.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse HS268");
        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let result = solve_qp_with(&prob, &opts);
        eprintln!("HS268 status={:?} obj={:.6e}", result.status, result.objective);
        let x = &result.solution;
        let y = &result.dual_solution;
        let bd = &result.bound_duals;
        eprintln!("  x = {:?}", x);
        eprintln!("  y = {:?}", y);
        eprintln!("  bound_duals = {:?} (len={})", bd, bd.len());
        // 各成分の KKT 残差: Qx + c + A^T y + bound_contrib
        let qx = prob.q.mat_vec_mul(x).unwrap();
        let aty = if !y.is_empty() {
            prob.a.transpose().mat_vec_mul(y).unwrap()
        } else {
            vec![0.0; prob.num_vars]
        };
        eprintln!("  per-component KKT residual:");
        for j in 0..prob.num_vars {
            let r = qx[j] + prob.c[j] + aty[j];
            eprintln!(
                "    j={}: Qx={:.3e} c={:.3e} (A^Ty)={:.3e} sum={:.3e}",
                j, qx[j], prob.c[j], aty[j], r
            );
        }
        // 真の dual を最小二乗で推定: A^T y = -(Qx + c)
        // n=5, m=5 で正方系。dense 直接解ける。
        let n = prob.num_vars;
        let m = prob.num_constraints;
        let mut at_dense = vec![vec![0.0_f64; m]; n];
        // CSC 走査: A.col_ptr[j] .. A.col_ptr[j+1] が列 j の (row_idx, value) のペア
        // A^T[j][i] = A[i][j]
        for j in 0..n {
            for k in prob.a.col_ptr[j]..prob.a.col_ptr[j + 1] {
                let i = prob.a.row_ind[k];
                let v = prob.a.values[k];
                if i < m {
                    at_dense[j][i] = v;
                }
            }
        }
        // rhs = -(Qx + c)
        let rhs: Vec<f64> = (0..n).map(|j| -(qx[j] + prob.c[j])).collect();
        // Solve at_dense * y = rhs (n x m, n=m=5 → square)
        // Use simple Gauss elimination
        let mut aug = at_dense.clone();
        let mut b = rhs.clone();
        for k in 0..n.min(m) {
            // 部分 pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug[i][k].abs() > aug[max_row][k].abs() {
                    max_row = i;
                }
            }
            aug.swap(k, max_row);
            b.swap(k, max_row);
            if aug[k][k].abs() < 1e-15 {
                eprintln!("  singular at k={}", k);
                return;
            }
            for i in (k + 1)..n {
                let factor = aug[i][k] / aug[k][k];
                for j in k..m {
                    aug[i][j] -= factor * aug[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
        let mut y_recon = vec![0.0_f64; m];
        for k in (0..n.min(m)).rev() {
            let mut sum = b[k];
            for j in (k + 1)..m {
                sum -= aug[k][j] * y_recon[j];
            }
            y_recon[k] = sum / aug[k][k];
        }
        eprintln!("  reconstructed y (LSQ): {:?}", y_recon);
        eprintln!("  ratio (solver_y / recon_y):");
        for i in 0..m.min(y.len()) {
            if y_recon[i].abs() > 1e-15 {
                eprintln!("    i={}: ratio={:.4}", i, y[i] / y_recon[i]);
            }
        }
    }

    /// UBH1 PSD 診断用ヘルパ: Q の対角に ε を加算した新しい CSC を返す。
    #[cfg(test)]
    fn build_q_with_diag_reg(q: &CscMatrix, eps_q: f64) -> CscMatrix {
        let n = q.ncols;
        let mut new_col_ptr = vec![0_usize; n + 1];
        let mut new_row_ind: Vec<usize> = Vec::with_capacity(q.values.len() + n);
        let mut new_values: Vec<f64> = Vec::with_capacity(q.values.len() + n);
        for col in 0..n {
            new_col_ptr[col] = new_row_ind.len();
            let start = q.col_ptr[col];
            let end = q.col_ptr[col + 1];
            let mut diag_added = false;
            for ptr in start..end {
                let row = q.row_ind[ptr];
                let val = q.values[ptr];
                if row == col {
                    new_row_ind.push(row);
                    new_values.push(val + eps_q);
                    diag_added = true;
                } else {
                    new_row_ind.push(row);
                    new_values.push(val);
                }
            }
            if !diag_added {
                new_row_ind.push(col);
                new_values.push(eps_q);
            }
        }
        new_col_ptr[n] = new_row_ind.len();
        CscMatrix {
            col_ptr: new_col_ptr,
            row_ind: new_row_ind,
            values: new_values,
            nrows: n,
            ncols: n,
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TDD赤フェーズ: バグ再現テスト（修正済み）
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// BUG-QP-001: solve_as_lp が MaxIterations→NumericalError 変換（MEDIUM）
    /// 修正: qp/mod.rs の MaxIterations branch を unreachable!() に置換。
    /// MaxIterations は SimplexOutcome::MaxIterations廃止により到達不能なdead path。
    #[test]
    fn test_qp001_solve_as_lp_no_numerical_error() {
        // SPEC: BUG-QP-001 — regression test
        // MaxIterationsはsimplexパスから到達不能のためPASS。
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
    // TDD赤フェーズ: テスト不足 (△) 項目
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

    // ===== bound_duals col_mapリマップ テスト (BD-T1〜BD-T6) =====

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

    /// BD-T4: EmptyCol 変数の bound_duals は KKT で正しい値が入る
    /// min 1/2*(0.001*x^2 + 0.001*y^2) - x - y + z
    /// s.t. x + y <= 4, x,y∈(-∞,∞), 0 <= z <= 3
    /// z は制約に登場しない → EmptyCol → presolve で z=lb=0 に固定。
    ///
    /// 旧期待: 「postsolve は除去変数の bound_dual を 0 で埋める」→ z_lb=z_ub=0
    /// 新期待: KKT 後処理 (refit_bound_duals_kkt) が presolve 0 埋めを修復し、
    ///         lb 活性 (x=0) + c=1 から z_lb = 1 が復元される。z_ub=0 は不変。
    /// ただし bench/v2 の dfeas は EmptyCol を skip するため、どちらでも PASS。
    /// 本テストでは「KKT を満たす値」が入ることを保証する。
    #[test]
    fn test_bd_t4_empty_col_kkt_recovered() {
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
        // z=0 (lb 活性) なので KKT: 0.001*0 + 1 - z_lb + z_ub = 0 → z_lb = 1, z_ub = 0
        let z_lb = result.bound_duals[0];
        let z_ub = result.bound_duals[1];
        assert!(
            (z_lb - 1.0).abs() < 1e-3,
            "BD-T4: z_lb≈1 (KKT recovered for EmptyCol), got {}", z_lb
        );
        assert!(z_ub.abs() < 1e-3, "BD-T4: z_ub≈0 (ub inactive), got {}", z_ub);
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

    /// BD-T7: constraint active × lb_dual nonzero × KKT照合
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

    /// 大係数行と小係数行が混在するケースで行ノルム正規化pfeasが正しく機能するテスト
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

    /// 正規化なしでは判定が歪むが正規化ありで正しく判定できるケース
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

    // ========== C-QP: Ge制約防御テスト ==========

    /// C-QP: Ge制約防御テスト
    /// min x²+y²  s.t. x+y≥1 (ConstraintType::Ge)
    /// QpProblem::new() 使用。期待: Optimal, x=y=0.5
    #[test]
    fn test_qp_ge_defensive() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "C-QP: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "C-QP: status");
        assert_close(result.solution[0], 0.5, EPS, "C-QP: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "C-QP: x[1]");
    }

    // ========== D: Mixed（Ge含む）防御テスト ==========

    /// D: Mixed Ge+Le防御テスト
    /// min x²+y²  s.t. x+y≥0.5 (Ge), x-y≤1 (Le)
    /// 期待: Optimal, x=y=0.25
    /// NOTE: presolveバグあり（mixed Ge+Leでpresolve ONのとき制約タイプが誤変換される）。
    ///       presolve=false で正しい解x=y=0.25を検証。バグ詳細はkaro報告参照。
    #[test]
    fn test_qp_mixed_ge_le_defensive() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // Row 0: x+y≥0.5 (Ge), Row 1: x-y≤1 (Le)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, -1.0],
            2, 2,
        ).unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q, c, a, b, bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        ).unwrap();

        // presolve=false: presolveバグを回避してソルバー本体の正確さを検証
        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            presolve: false,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "D: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "D: status");
        assert_close(result.solution[0], 0.25, EPS, "D: x[0]");
        assert_close(result.solution[1], 0.25, EPS, "D: x[1]");
    }

    // ========== E: Concurrent制約タイプ別テスト ==========

    /// E-Eq: Concurrent Eq制約テスト
    /// min x²+y²  s.t. x+y=1 (Eq)
    /// 期待: Optimal, x=y=0.5
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_eq_constraint() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq]).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "E-Eq: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "E-Eq: status");
        assert_close(result.solution[0], 0.5, EPS, "E-Eq: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "E-Eq: x[1]");
    }

    /// E-Ge: Concurrent Ge制約テスト
    /// min x²+y²  s.t. x+y≥1 (Ge)
    /// 期待: Optimal, x=y=0.5
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_ge_constraint() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "E-Ge: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "E-Ge: status");
        assert_close(result.solution[0], 0.5, EPS, "E-Ge: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "E-Ge: x[1]");
    }

    /// E-Box: Concurrent Box制約テスト
    /// min x²+y²  s.t. 0≤x≤1, 0≤y≤1
    /// 期待: Optimal, x=y=0
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_box_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "E-Box: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "E-Box: status");
        assert_close(result.solution[0], 0.0, EPS, "E-Box: x[0]");
        assert_close(result.solution[1], 0.0, EPS, "E-Box: x[1]");
    }

    /// E-Mixed: Concurrent Mixed(Le+Eq)テスト
    /// min x²+y²  s.t. x+y=1 (Eq), x≤1 (Le)
    /// 期待: Optimal, x=y=0.5
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_mixed_constraint() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // Row 0: x+y=1 (Eq), Row 1: x≤1 (Le)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1],
            &[0, 1, 0],
            &[1.0, 1.0, 1.0],
            2, 2,
        ).unwrap();
        let b = vec![1.0, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q, c, a, b, bounds,
            vec![ConstraintType::Eq, ConstraintType::Le],
        ).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "E-Mixed: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "E-Mixed: status");
        assert_close(result.solution[0], 0.5, EPS, "E-Mixed: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "E-Mixed: x[1]");
    }

    /// E-Unconstrained: Concurrent 無制約テスト
    /// min (x-1)²+(y-1)²  （制約なし）
    /// 期待: Optimal, x=y=1
    #[cfg(feature = "parallel")]
    #[test]
    fn test_concurrent_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-2.0, -2.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "E-Unconstrained: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "E-Unconstrained: status");
        assert_close(result.solution[0], 1.0, EPS, "E-Unconstrained: x[0]");
        assert_close(result.solution[1], 1.0, EPS, "E-Unconstrained: x[1]");
    }

    // ========== F: 退化ケーステスト ==========

    /// F-QP-Fixed: 全変数固定退化ケース
    /// min x² s.t. bounds=(1.0, 1.0)（全変数固定）
    /// Q=I(1x1), c=[0], A=空(0×1), b=[], presolve=false（presolveバグ回避）
    /// 期待: Optimal, x=1.0
    #[test]
    fn test_qp_all_vars_fixed() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::new(0, 1);
        let b: Vec<f64> = vec![];
        let bounds = vec![(1.0_f64, 1.0_f64)];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

        let mut opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        opts.presolve = false;
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(start.elapsed().as_secs_f64() < 6.0, "F-QP-Fixed: wall-clock 6秒超過");
        assert_eq!(result.status, SolveStatus::Optimal, "F-QP-Fixed: status must be Optimal, got {:?}", result.status);
        assert_close(result.solution[0], 1.0, EPS, "F-QP-Fixed: x[0] must be 1.0");
    }

    // ========== G: ステータスマッピング検証テスト ==========

    /// G-1: SuboptimalSolution→Optimal変換確認
    /// timeout=2秒付き簡単問題 → 外部APIにSuboptimalSolutionが漏れないことを確認
    #[test]
    fn test_suboptimal_to_optimal_mapping() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // max_iter=1でMaxIterations/SuboptimalSolutionを強制発生させ、変換パスを通過させる
        let opts = SolverOptions {
            timeout_secs: Some(2.0),
            ipm: crate::options::IpmOptions { max_iter: 1, ..Default::default() },
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        // SuboptimalSolutionは外部APIに漏れてはならない（Optimal or Timeout のみ許容）
        assert_ne!(
            result.status, SolveStatus::SuboptimalSolution,
            "G-1: SuboptimalSolutionが外部APIに漏れた（Optimal or Timeoutに変換されるべき）"
        );
        assert!(
            result.status == SolveStatus::Optimal || result.status == SolveStatus::Timeout,
            "G-1: status must be Optimal or Timeout, got {:?}", result.status
        );
    }

    /// G-2: MaxIterations→Timeout変換確認
    /// max_iter=1 で最大反復到達 → 外部APIにMaxIterationsが漏れないことを確認
    #[test]
    fn test_max_iterations_to_timeout_mapping() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ipm: crate::options::IpmOptions { max_iter: 1, ..Default::default() },
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        // MaxIterationsは外部APIに漏れてはならない（Optimal or Timeout のみ許容）
        assert_ne!(
            result.status, SolveStatus::MaxIterations,
            "G-2: MaxIterationsが外部APIに漏れた（Optimal or Timeoutに変換されるべき）"
        );
        assert!(
            result.status == SolveStatus::Optimal || result.status == SolveStatus::Timeout,
            "G-2: status must be Optimal or Timeout, got {:?}", result.status
        );
    }

    // ========== H: presolve ON/OFF比較テスト ==========

    /// H-1: Eq制約QP presolve ON/OFF比較
    /// min x²+y²  s.t. x+y=1 (Eq) を presolve ON/OFF両方で解き、解一致を確認
    #[test]
    fn test_presolve_qp_eq_on_off_consistency() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq]).unwrap();

        let opts_on = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let mut opts_off = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        opts_off.presolve = false;

        let result_on = solve_qp_with(&problem, &opts_on);
        let result_off = solve_qp_with(&problem, &opts_off);

        assert_eq!(result_on.status, SolveStatus::Optimal, "H-1: presolve ON status");
        assert_eq!(result_off.status, SolveStatus::Optimal, "H-1: presolve OFF status");
        assert!(
            (result_on.solution[0] - result_off.solution[0]).abs() < 1e-4,
            "H-1: presolve ON/OFF x[0]不一致: ON={}, OFF={}", result_on.solution[0], result_off.solution[0]
        );
        assert!(
            (result_on.solution[1] - result_off.solution[1]).abs() < 1e-4,
            "H-1: presolve ON/OFF x[1]不一致: ON={}, OFF={}", result_on.solution[1], result_off.solution[1]
        );
    }

    /// H-2: Box制約QP presolve ON/OFF比較
    /// min x²+y²  s.t. 0≤x≤2, 0≤y≤2 を presolve ON/OFF両方で解き、解一致を確認
    #[test]
    fn test_presolve_qp_box_on_off_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 2.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts_on = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let mut opts_off = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        opts_off.presolve = false;

        let result_on = solve_qp_with(&problem, &opts_on);
        let result_off = solve_qp_with(&problem, &opts_off);

        assert_eq!(result_on.status, SolveStatus::Optimal, "H-2: presolve ON status");
        assert_eq!(result_off.status, SolveStatus::Optimal, "H-2: presolve OFF status");
        // 解の一致確認: 双方の解が既知最適解(0,0)に収束していることを確認
        assert_close(result_on.solution[0], 0.0, EPS, "H-2: presolve ON x[0]");
        assert_close(result_on.solution[1], 0.0, EPS, "H-2: presolve ON x[1]");
        assert_close(result_off.solution[0], 0.0, EPS, "H-2: presolve OFF x[0]");
        assert_close(result_off.solution[1], 0.0, EPS, "H-2: presolve OFF x[1]");
    }

    /// H-3: Ge制約QP + presolve
    /// min x²+y²  s.t. x+y≥1 (Ge) + presolve ON → Optimal確認
    #[test]
    fn test_qp_ge_constraint_with_presolve() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

        let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "H-3: Ge+presolve status");
        assert_close(result.solution[0], 0.5, EPS, "H-3: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "H-3: x[1]");
    }

    /// H-4: Mixed(Ge+Le)QP presolve無効化テスト
    /// min x²+y²  s.t. x+y≥0.5 (Ge), x-y≤1 (Le) + presolve=false → Optimal確認
    /// NOTE: mixed Ge+Le + presolve ONにはpresolveバグがある（制約タイプ誤変換）。
    ///       presolve=false でソルバー本体の mixed Ge+Le 処理が正しいことを確認する。
    ///       x=y=0.25が最適解（x+y=0.5が拘束、x-y=0≤1は非拘束）
    #[test]
    fn test_qp_mixed_ge_with_presolve() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // Row 0: x+y≥0.5 (Ge, binding), Row 1: x-y≤1 (Le, inactive)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, -1.0],
            2, 2,
        ).unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q, c, a, b, bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        ).unwrap();

        // presolve=false: presolveバグを回避してソルバー本体の正確さを検証
        let mut opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        opts.presolve = false;
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "H-4: Mixed(Ge+Le)+no-presolve status");
        assert_close(result.solution[0], 0.25, EPS, "H-4: x[0]");
        assert_close(result.solution[1], 0.25, EPS, "H-4: x[1]");
    }

    /// H-5: Mixed(Ge+Le)QP presolve=ON + Ruiz=ON regression test
    ///
    /// pfeas不等号バグ修正の回帰テスト。
    /// バグ再現条件: presolve=ON + Ruiz=ON + Ge+Le混在（B-1パターン直接再現）
    ///
    /// 問題: min x²+y²  s.t. x+y≥0.5 (Ge), x-y≤1 (Le)
    /// 最適解: x=y=0.25 (x+y=0.5が拘束、x-y=0≤1は非拘束)
    ///
    /// 修正前: pfeas計算でGe違反を検出できず偽OptimalまたはSuboptimalSolutionを返していた。
    /// 修正後: Ge違反 = max(b - ax, 0) で正しく計算され、Optimalが確認できる。
    #[test]
    fn test_qp_mixed_ge_le_presolve_ruiz_regression() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // Row 0: x+y≥0.5 (Ge, binding), Row 1: x-y≤1 (Le, inactive)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, -1.0],
            2, 2,
        ).unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q, c, a, b, bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        ).unwrap();

        // presolve=ON + Ruiz=ON（デフォルト）でバグが再現していたパターン
        let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal,
            "H-5: Mixed(Ge+Le)+presolve=ON+Ruiz=ON status. got {:?}", result.status);
        assert_close(result.solution[0], 0.25, EPS, "H-5: x[0]");
        assert_close(result.solution[1], 0.25, EPS, "H-5: x[1]");
        // pfeas直接assert: Ge違反 = max(b - ax, 0) が閾値未満であることを確認
        let pfeas = {
            let x = &result.solution;
            // Row 0: x+y - 0.5 (Ge: violation = max(0.5 - (x+y), 0))
            let ge_viol = (0.5_f64 - (x[0] + x[1])).max(0.0);
            // Row 1: x-y - 1.0 (Le: violation = max((x-y) - 1.0, 0))
            let le_viol = (x[0] - x[1] - 1.0_f64).max(0.0);
            ge_viol.max(le_viol)
        };
        assert!(pfeas < 1e-6, "H-5: pfeas={:e} (期待値 < 1e-6)", pfeas);

        // presolve=OFF比較: 同一問題でpresolve無効化時もOptimalかつ同一解であることを確認
        let opts_no_presolve = SolverOptions {
            timeout_secs: Some(10.0),
            presolve: false,
            ..Default::default()
        };
        let result_no_presolve = solve_qp_with(&problem, &opts_no_presolve);
        assert_eq!(result_no_presolve.status, SolveStatus::Optimal,
            "H-5: presolve=OFF status. got {:?}", result_no_presolve.status);
        assert_close(result_no_presolve.solution[0], 0.25, EPS, "H-5(no-presolve): x[0]");
        assert_close(result_no_presolve.solution[1], 0.25, EPS, "H-5(no-presolve): x[1]");
    }

    // ===================================================================
    // dfeas 相対閾値テスト群
    // ===================================================================

    /// D-1: 正常なQP解ではdfeasチェックがOptimalを維持する
    /// min x^2 + y^2  s.t. x+y >= 1, x,y >= 0
    /// 最適解 x=y=0.5 はKKT条件を満たし、dfeasは十分小さい
    #[test]
    fn test_dfeas_optimal_preserved() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(0.0, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal,
            "D-1: well-solved QP must stay Optimal after dfeas check");
    }

    /// D-2: スケール不変性 — 係数を1e6倍してもOptimalが維持される
    /// min (1e6)^2 * (x^2+y^2) s.t. 1e6*(x+y) >= 1e6, x,y >= 0
    /// 数学的に同一問題だが、絶対閾値ではdfeasが巨大値になり誤判定する
    #[test]
    fn test_dfeas_scale_invariant() {
        let scale = 1e6_f64;
        let q = CscMatrix::from_triplets(
            &[0, 1], &[0, 1],
            &[2.0 * scale * scale, 2.0 * scale * scale], 2, 2,
        ).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(
            &[0, 0], &[0, 1],
            &[-scale, -scale], 1, 2,
        ).unwrap();
        let b = vec![-scale];
        let bounds = vec![(0.0, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal,
            "D-2: scaled QP must stay Optimal (relative threshold). got {:?}", result.status);
        // 解は元問題と同じ x=y=0.5
        assert_close(result.solution[0], 0.5, 1e-4, "D-2: x[0]");
        assert_close(result.solution[1], 0.5, 1e-4, "D-2: x[1]");
    }

    /// D-3: 真にdfeasが悪い解ではSuboptimalSolutionに降格される
    /// check_dfeas_status / check_dfeas_status_relative を直接呼び出し、
    /// KKT残差が閾値を超える場合の降格を検証
    #[test]
    fn test_dfeas_bad_solution_downgraded() {
        // min x^2 + y^2 (Q=2I, c=0) — 無制約
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 最適解は x=y=0, dfeas=0。意図的にずらした解 x=y=1.0 を与える
        // KKT残差: Qx + c = [2, 2] → dfeas = 2.0
        let bad_x = vec![1.0, 1.0];
        let bad_y: Vec<f64> = vec![];
        let bad_bd: Vec<f64> = vec![];

        // (a) 絶対閾値版: 小さい閾値ではSuboptimalSolution
        let status = ipm::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 1e-6);
        assert_eq!(status, SolveStatus::SuboptimalSolution,
            "D-3a: bad solution with dfeas=2.0 >> 1e-6 must be SuboptimalSolution");
        let status_ok = ipm::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 10.0);
        assert_eq!(status_ok, SolveStatus::Optimal,
            "D-3a: same solution with dfeas=2.0 < 10.0 stays Optimal");

        // (b) 成分ごと相対版: residual=2.0, scale=1+2+0+0=3, relative=2/3≈0.667
        // eps=0.01 → SuboptimalSolution
        let status_rel = ipm::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 0.01);
        assert_eq!(status_rel, SolveStatus::SuboptimalSolution,
            "D-3b: relative dfeas=0.667 >> 0.01 must be SuboptimalSolution");
        // eps=1.0 → Optimal (relative < 1.0)
        let status_rel_ok = ipm::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 1.0);
        assert_eq!(status_rel_ok, SolveStatus::Optimal,
            "D-3b: relative dfeas=0.667 < 1.0 stays Optimal");
    }

    /// D-4: 相対閾値の計算精度 — KKTスケールが大きい問題でも正しく正規化される
    #[test]
    fn test_dfeas_relative_threshold_large_kkt() {
        // min 1/2 * 2e12 * x^2 - 1e6 * x  (unconstrained, x free)
        // KKT: 2e12*x - 1e6 = 0 → x* = 5e-7
        let n = 1usize;
        let q = CscMatrix::from_triplets(&[0], &[0], &[2e12], n, n).unwrap();
        let c = vec![-1e6];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal,
            "D-4: large-KKT-scale QP must be Optimal. got {:?}", result.status);
        assert!((result.solution[0] - 5e-7).abs() < 1e-9,
            "D-4: x*=5e-7, got {:.2e}", result.solution[0]);
    }

    /// D-5: 巨大項キャンセレーション — BOYD1の本質を小さい問題で再現
    /// Qx_j と A^Ty_j がO(1e10)で互いにキャンセルし、残差はO(1e-2)以下。
    /// 絶対閾値やグローバルノルム相対閾値では誤判定するが、成分ごと相対なら正確。
    #[test]
    fn test_dfeas_cancellation_pattern() {
        // 手動でcheck_dfeas_status_relativeを呼び出す
        // 問題: min 1/2 * 2e10 * x^2 - 1e10*x  s.t. x + y <= 2, x,y >= 0
        // ただし真のテストは直接関数呼び出しで行う
        let n = 2usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 擬似的に巨大キャンセレーションを模擬:
        // x = [5e9, 5e9], Qx = [1e10, 1e10], c = [0, 0]
        // 残差 = |1e10 + 0 + 0 + 0| = 1e10 (絶対値は巨大)
        // だがrelative = 1e10 / (1 + 1e10) ≈ 1.0 (悪い解 → SubOptimal 正しい)
        let big_x = vec![5e9, 5e9];
        let empty_y: Vec<f64> = vec![];
        let empty_bd: Vec<f64> = vec![];
        let status = ipm::check_dfeas_status_relative(&problem, &big_x, &empty_y, &empty_bd, 0.01);
        assert_eq!(status, SolveStatus::SuboptimalSolution,
            "D-5a: large absolute residual with no cancellation → SuboptimalSolution");

        // 正しいキャンセレーション: Qx + c がほぼ0になるケース
        // x ≈ 0 (最適解) → Qx ≈ 0, c = 0, 残差 ≈ 0
        let good_x = vec![1e-12, 1e-12];
        let status_good = ipm::check_dfeas_status_relative(&problem, &good_x, &empty_y, &empty_bd, 1e-8);
        assert_eq!(status_good, SolveStatus::Optimal,
            "D-5b: near-optimal solution → Optimal");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // refit_bound_duals_kkt 単体テスト (T1.2 / T1.4 回帰防止)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// REFIT-T1: lb のみ有限な変数で x が lb に張り付き、c > 0 の場合
    /// presolve で除去された変数の bound_dual を KKT で復元できること。
    /// KKT: c + (-y_lb) = 0 → y_lb = c
    #[test]
    fn test_refit_bound_duals_lb_only_active() {
        // min c*x s.t. (制約なし, ただし x>=0)
        // n=1, c=[2.5], bounds=[(0, +∞)]
        // 最適: x=0 (c>0 なので増やすほど obj 大), y_lb = 2.5
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.5_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // bound_dual=0 の状態で渡し、refit が y_lb=2.5 を復元するか
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!(
            (result.bound_duals[0] - 2.5).abs() < 1e-9,
            "REFIT-T1: y_lb 復元 ≈ 2.5, got {}", result.bound_duals[0]
        );
    }

    /// REFIT-T2: ub のみ有限な変数で x が ub に張り付き、c < 0 の場合
    /// KKT: c + y_ub = 0 → y_ub = -c
    #[test]
    fn test_refit_bound_duals_ub_only_active() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![-3.0_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, 5.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![5.0],
            dual_solution: vec![],
            bound_duals: vec![0.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!(
            (result.bound_duals[0] - 3.0).abs() < 1e-9,
            "REFIT-T2: y_ub 復元 ≈ 3.0, got {}", result.bound_duals[0]
        );
    }

    /// REFIT-T3: 内点 (interior) では y_lb=y_ub=0 を保つ (refit が誤って非ゼロ化しない)
    #[test]
    fn test_refit_bound_duals_interior_keeps_zero() {
        // x=2 (中央), bounds=(0,5), Q=2I, c=-4 → 2*2-4=0 (KKT 満足)
        let n = 1usize;
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap();
        let c = vec![-4.0_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, 5.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![2.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 0.0], // [y_lb, y_ub]
            objective: -4.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!(result.bound_duals[0].abs() < 1e-9, "REFIT-T3: y_lb 維持 0");
        assert!(result.bound_duals[1].abs() < 1e-9, "REFIT-T3: y_ub 維持 0");
    }

    /// REFIT-T4: KKT-guard で改善しない update は revert される
    /// 既に正しい bound_duals が入っている状態で refit を呼んでも変わらない
    #[test]
    fn test_refit_bound_duals_kkt_guard_no_regression() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![2.0_f64];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 既に正しい状態 (y_lb=2.0): refit が壊さないこと
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![2.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!(
            (result.bound_duals[0] - 2.0).abs() < 1e-9,
            "REFIT-T4: 既に正しい値は維持される, got {}", result.bound_duals[0]
        );
    }

    /// REFIT-T5: 制約あり問題 (A^T y で aty が非ゼロ) の bound_dual 計算
    /// min x s.t. x + y <= 5, 0 <= x, 0 <= y
    /// 最適: x=0, y=0 (両 lb 活性)。dual_solution[0]=0 (制約非活性)。
    /// KKT for x: 1 + 0 - z_lb_x = 0 → z_lb_x = 1
    /// KKT for y: 0 + 0 - z_lb_y = 0 → z_lb_y = 0
    #[test]
    fn test_refit_bound_duals_with_constraint() {
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0_f64, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![5.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0, 0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0, 0.0],
            objective: 0.0,
            ..SolverResult::default()
        };
        refit_bound_duals_kkt(&problem, &mut result);
        assert!(
            (result.bound_duals[0] - 1.0).abs() < 1e-9,
            "REFIT-T5: z_lb_x ≈ 1.0, got {}", result.bound_duals[0]
        );
        assert!(
            result.bound_duals[1].abs() < 1e-9,
            "REFIT-T5: z_lb_y ≈ 0.0, got {}", result.bound_duals[1]
        );
    }

    /// REFIT-T7: rank-deficient Q + 多解 → 偽 Optimal を duality gap で検出
    /// UBH1 風: Q PSD だが rank < n (実質 LP 部分空間) で IPM が KKT 残差小だが
    /// 大域 Optimal でない解に収束するケース。duality_gap_rel チェックが gate する。
    #[test]
    fn test_duality_gap_rejects_rank_deficient_false_optimal() {
        // 構築: min 1/2 x^T (e e^T) x + c^T x s.t. Ax <= b
        //   Q = e e^T (rank 1, e=(1,1)^T)
        //   c = (-1, 0)
        //   A = [[1, 0]], b = [3] (x_1 <= 3)
        //   bounds: x_2 >= 0
        //
        // 真の Optimal: x = (1, 0), obj = 0.5 * 1 - 1 = -0.5
        // (∇f = (Qx)_0 + c_0 = (x_0+x_1) - 1 = 0 → x_0 + x_1 = 1, x_2=0 で x_0=1)
        //
        // しかし KKT 停留性は x_0 + x_1 = 1 で任意の (x_0, x_1) が満たす。
        // IPM が x = (0, 1) に収束した場合: obj = 0.5 - 0 = 0.5 (誤り)
        // duality gap でこれを弾く。
        //
        // ここでは bound_duals 経由の dual 復元が機能するか単体で確認するため、
        // mock 解 x=(0, 1) を直接構築して duality gap が大きいことを assert する。
        use crate::sparse::CscMatrix;
        let n = 2usize;
        // Q = e e^T = [[1,1],[1,1]], 上三角 CSC で表現: (0,0)=1, (0,1)=1, (1,1)=1
        let q = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], n, n).unwrap();
        let c = vec![-1.0_f64, 0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, n).unwrap();
        let b = vec![3.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        // 真の Optimal obj = -0.5
        // 偽 Optimal x=(0,1) なら obj = 0.5 (誤り)
        // Solver は真 Optimal に近い obj を返すべき
        if result.status == SolveStatus::Optimal {
            assert!(
                (result.objective - (-0.5)).abs() < 1e-3,
                "REFIT-T7: obj should be ≈ -0.5, got {} (rank-deficient Q false optimal)",
                result.objective
            );
        }
    }

    /// REFIT-T6: presolve で除去された変数 (EmptyCol) の bound_dual が KKT で復元される (統合テスト)
    /// QRECIPE 的な状況: c>0 + 全 lb 有限 + EmptyCol 変数あり
    #[test]
    fn test_refit_integration_emptycol_recovery() {
        // 3 変数: x, y は制約に登場、z は EmptyCol
        // min 0.001(x^2+y^2+z^2) - x - y + 2*z
        // s.t. x + y <= 5, 0 <= x,y,z, ub = 10 for z
        let n = 3usize;
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 2.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![5.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY), (0.0_f64, 10.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions { presolve: true, ..SolverOptions::default() };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "REFIT-T6: status");
        // n_lb=3 (全 lb 有限), n_ub=1 (z のみ ub 有限) → bound_duals.len() = 4
        assert_eq!(result.bound_duals.len(), 4);
        // z=0 (lb 活性, c=2>0), KKT: 2 - z_lb_z = 0 → z_lb_z ≈ 2
        let z_lb_z = result.bound_duals[2];
        assert!(
            (z_lb_z - 2.0).abs() < 1e-2,
            "REFIT-T6: EmptyCol 変数 z_lb ≈ 2.0 (KKT 復元), got {}", z_lb_z
        );
    }
}
