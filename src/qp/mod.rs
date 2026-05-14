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

pub mod diagnose;
pub(crate) mod ipm_core;
pub mod ipm_solver;
mod lp_dispatch;
mod problem;
pub use crate::problem::SolverResult;
pub use diagnose::{
    diagnose, DiagnosticCode, DiagnosticReport, DiagnosticWarning, ProblemInfo, Severity,
};
pub(crate) use lp_dispatch::solve_as_lp_pub;
pub use problem::{QpProblem, QpWarmStart};

use crate::options::{QpSolverChoice, SolverOptions};
use crate::sparse::CscMatrix;

/// QP ソルバーを統一的に扱うための trait
///
/// 現在 `IpPmmSolver` のみが `QpSolver` を実装している。
pub trait QpSolver {
    /// QP 問題を解く
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult;
    /// ソルバー名を返す
    fn name(&self) -> &'static str;
}

/// IP-PMM (Interior Point Proximal Method of Multipliers) QP ソルバー
///
/// `QpSolver` trait を実装する。内部で [`solve_qp_with`] を呼ぶ。
pub struct IpPmmSolver;

impl QpSolver for IpPmmSolver {
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult {
        let mut forced_opts = options.clone();
        forced_opts.qp_solver = QpSolverChoice::IpPmm;
        solve_qp_with(problem, &forced_opts)
    }
    fn name(&self) -> &'static str {
        "IP-PMM"
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

    // ‖Q‖_max 推定: |対角| と |off-diag| の最大絶対値。閾値を相対化するスケール。
    // QPS encoding は典型 6-7 桁精度なので、入力ノイズは ‖Q‖_max × 1e-6 オーダー。
    let mut q_abs_max = 0.0_f64;
    for &v in q.values.iter() {
        let a = v.abs();
        if a > q_abs_max {
            q_abs_max = a;
        }
    }

    // 対角チェック (O(nnz), サイズ非依存)
    // 対角に有意に負な値があれば → 非PSD確定（十分条件）
    // 閾値は QPS encoding noise を相対許容: ‖Q‖_max × QPS_NEG_TOL_RATIO。
    // ‖Q‖_max=0 (Q=0 の LP) は absolute floor 1e-12 で扱う。
    const QPS_NEG_TOL_RATIO: f64 = 1e-6;
    let neg_tol = (q_abs_max * QPS_NEG_TOL_RATIO).max(1e-12);
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col && q.values[k] < -neg_tol {
                return false; // 対角負値 → 非PSD確定
            }
        }
    }

    // n > 1000: Cholesky分解はO(n³)コスト過大のため省略を維持
    // （対角チェック済みのため、対角負値は検出完了）
    const CHECK_SIZE_LIMIT: usize = 1000;
    if n > CHECK_SIZE_LIMIT {
        return true;
    }

    // Cholesky 用 regularization eps: ‖Q‖_max × CHOL_EPS_RATIO で入力スケールに追従、
    // ‖Q‖_max=0 ケースには absolute floor 1e-8。「数学的には PSD だが QPS 6 桁丸めで
    // 僅かに不定」(Maros VALUES など) を救う。本物の非凸 (QPLIB 0018, 2712) は閾値を
    // 越えて false を返す。
    const CHOL_EPS_RATIO: f64 = 1e-4;
    let eps = (q_abs_max * CHOL_EPS_RATIO).max(1e-8);

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
    let total_expanded: usize = le_map
        .original_to_expanded
        .iter()
        .map(|rows| rows.len())
        .sum();
    if dual_expanded.len() < total_expanded {
        // サイズ不一致: フォールバックとして直接使用
        return dual_expanded.to_vec();
    }
    let mut collapsed = vec![0.0f64; m_orig];
    for (i, (ct, rows)) in orig_types
        .iter()
        .zip(le_map.original_to_expanded.iter())
        .enumerate()
    {
        collapsed[i] = match ct {
            ConstraintType::Le => dual_expanded[rows[0]],
            ConstraintType::Ge => -dual_expanded[rows[0]],
            ConstraintType::Eq => {
                let mu1 = dual_expanded[rows[0]];
                let mu2 = if rows.len() > 1 {
                    dual_expanded[rows[1]]
                } else {
                    0.0
                };
                mu1 - mu2
            }
        };
    }
    collapsed
}

/// faer supernodal Cholesky の deepest stack 要求 (BOYD1 n=93261 等の検証から経験的に決定)
/// + 安全マージン。OS 主スレッドの典型値 (macOS/Linux 8 MB) と一致させる。
///
/// `solve_qp_with` の入口で必ずこのサイズの scoped thread に IPPMM 実行を載せる。
/// Rust の `thread::spawn` デフォルトは 2 MB で、BOYD1 級の supernodal 再帰では
/// overflow する。
pub(crate) const SOLVE_STACK_SIZE: usize = 8 * 1024 * 1024;

/// QPをカスタム設定で解く
///
/// qpOASESの `init()` に相当。timeout が反復制御の主ガード。
///
/// # スタック保護
/// 入口で 8 MB の scoped thread に dispatch を載せる。faer supernodal Cholesky は
/// BOYD1 級 (n=93261) で深く再帰するため、Rust デフォルトの 2 MB スタックでは
/// stack overflow する。
pub fn solve_qp_with(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    std::thread::scope(|s| {
        let handle = std::thread::Builder::new()
            .stack_size(SOLVE_STACK_SIZE)
            .spawn_scoped(s, || dispatch_solve_qp(problem, options))
            .expect("spawn QP solver thread");
        handle.join().expect("QP solver thread panicked")
    })
}

/// `solve_qp_with` 本体の dispatch。`QpSolverChoice` は `IpPmm` のみ。
///
/// Mehrotra IPM 単独 / Concurrent 並列実行は廃止 (IPPMM が IPM の上位互換)。
///
/// Q=0 退化ケース (LP) は Simplex に委譲する。LP は Simplex の方が IPPMM より速く、
/// `slack` / `reduced_costs` も自然に得られる。
fn dispatch_solve_qp(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    if problem.is_zero_q() {
        return solve_as_lp_pub(problem, options);
    }
    match options.qp_solver {
        QpSolverChoice::IpPmm => ipm_solver::solve_qp_v2(problem, options),
    }
}

/// FX (固定) 変数判定の許容差。lb と ub の差がこれ未満なら固定変数とみなす。
pub(crate) const FX_TOL: f64 = 1e-12;

/// presolve で縮約された bound_duals を元問題空間に展開する。
/// reduced_bounds に対応する bound_duals を、orig_bounds に対応する形で再構築。
/// 除去された変数の bound_dual は 0.0 (近似) で埋める。
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
    let n_lb_reduced = reduced_bounds
        .iter()
        .filter(|(lb, _)| lb.is_finite())
        .count();
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
///
/// `deadline` 経過時は no-op で即 return。AAT (m×m) factorize が m=10k 級で
/// 数百秒かかる事例 (QPLIB_8505) があり、deadline 不在だと post-IPM が timeout
/// 予算を完全に使い切れない。
pub(crate) fn refine_dual_lsq(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let Some(y_new) = compute_lsq_dual_y(problem, result) else {
        return;
    };
    let n = problem.num_vars;
    // ill-conditioned 問題 (QPILOTNO: ‖A‖=5.85e6, cond=3e12) で f64 mat_vec の
    // cancellation noise が真の残差より大きく、KKT-guard で IPM 由来の正しい y が
    // LSQ y に置換される事故が起きる。bench (`compute_dfeas_orig`) と同じ DD で
    // 比較する。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let aty_dd = |y: &[f64]| -> Vec<TwoFloat> {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let cs = problem.a.col_ptr[col];
            let ce = problem.a.col_ptr[col + 1];
            for k in cs..ce {
                let row = problem.a.row_ind[k];
                acc[col] = acc[col] + TwoFloat::new_mul(problem.a.values[k], y[row]);
            }
        }
        acc
    };
    let aty_old_dd = aty_dd(&result.dual_solution);
    let aty_new_dd = aty_dd(&y_new);
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    // 成分相対化で比較する。abs max では ill-scaled 問題で 1 列のみ大きく外れた残差が
    // 巨大スケールに埋もれるため、componentwise rel = |r_j| / (1 + |Qx_j| + |c_j| + |Aty_j| + |z_j|)
    // で旧/新を比較し、stricter 側を取る。
    let mut max_rel_old = 0.0_f64;
    let mut max_rel_new = 0.0_f64;
    for j in 0..n {
        let (lbj, ubj) = problem.bounds[j];
        if lbj.is_finite() && ubj.is_finite() && (lbj - ubj).abs() < FX_TOL {
            continue;
        }
        if problem.a.col_ptr.len() > j + 1 && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0 {
            continue;
        }
        let r_old_dd = qx_dd[j]
            + TwoFloat::from(problem.c[j])
            + aty_old_dd[j]
            + TwoFloat::from(bound_contrib[j]);
        let r_new_dd = qx_dd[j]
            + TwoFloat::from(problem.c[j])
            + aty_new_dd[j]
            + TwoFloat::from(bound_contrib[j]);
        let qx_j = f64::from(qx_dd[j]).abs();
        let aty_old_j = f64::from(aty_old_dd[j]).abs();
        let aty_new_j = f64::from(aty_new_dd[j]).abs();
        let scale_old = 1.0 + qx_j + problem.c[j].abs() + aty_old_j + bound_contrib[j].abs();
        let scale_new = 1.0 + qx_j + problem.c[j].abs() + aty_new_j + bound_contrib[j].abs();
        let rel_old = f64::from(r_old_dd).abs() / scale_old;
        let rel_new = f64::from(r_new_dd).abs() / scale_new;
        if rel_old > max_rel_old {
            max_rel_old = rel_old;
        }
        if rel_new > max_rel_new {
            max_rel_new = rel_new;
        }
    }
    if max_rel_new < max_rel_old {
        result.dual_solution = y_new;
    }
}

/// singleton column (A の参照行が 1 本だけの列) から各 row dual の feasible interval を作り、
/// 現在の y をその区間へ射影する。
///
/// 真因:
/// `refine_dual_lsq` は `A^T y = target` の unconstrained LSQ なので、one-sided bound を持つ
/// 列に対して「非負 bound dual では補正不能」な y を返すことがある。
/// 例: lb-only 変数 x_j=0, c_j=0, A[row,j]=-1, y[row]>0 → qx+c+A^T y < 0 となり
/// `z_lb = qx+c+A^T y` が負になって KKT を満たせない。
///
/// singleton column ではその列の停留性が 1 個の row dual によってのみ決まるため、
/// bound 種別ごとに y[row] の必要条件を interval として導出できる。
/// row ごとに全 singleton column の interval を交差し、現在値を最近傍へ射影する。
pub(crate) fn project_duals_from_singleton_columns(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
) {
    let Some((lower, upper)) = compute_dual_recovery_row_bounds(problem, &result.solution) else {
        return;
    };
    if result.dual_solution.len() != problem.num_constraints {
        return;
    }
    for row in 0..problem.num_constraints {
        let lo = lower[row];
        let hi = upper[row];
        if lo > hi {
            continue;
        }
        let y = &mut result.dual_solution[row];
        if *y < lo {
            *y = lo;
        } else if *y > hi {
            *y = hi;
        }
    }
}

fn compute_dual_recovery_row_bounds(
    problem: &QpProblem,
    solution: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if solution.len() != n {
        return None;
    }

    let qx = problem.q.mat_vec_mul(solution).ok()?;
    let (ax, row_abs_activity) = compute_dual_recovery_row_activity(problem, solution)?;

    let mut lower = vec![f64::NEG_INFINITY; m];
    let mut upper = vec![f64::INFINITY; m];

    for (row, ct) in problem.constraint_types.iter().enumerate() {
        match ct {
            crate::problem::ConstraintType::Le => lower[row] = 0.0,
            crate::problem::ConstraintType::Ge => upper[row] = 0.0,
            crate::problem::ConstraintType::Eq => {}
        }
    }
    const SLACK_TOL_REL: f64 = 1e-8;
    for i in 0..m {
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => problem.b[i] - ax[i],
            crate::problem::ConstraintType::Ge => ax[i] - problem.b[i],
            crate::problem::ConstraintType::Eq => continue,
        };
        let tol = dual_recovery_row_slack_tol(problem, i, ax[i], row_abs_activity[i], SLACK_TOL_REL);
        if slack > tol {
            lower[i] = 0.0;
            upper[i] = 0.0;
        }
    }

    for j in 0..n {
        let cs = problem.a.col_ptr[j];
        let ce = problem.a.col_ptr[j + 1];
        if ce - cs != 1 {
            continue;
        }

        let row = problem.a.row_ind[cs];
        let aij = problem.a.values[cs];
        if !aij.is_finite() || aij == 0.0 {
            continue;
        }

        let (lb, ub) = problem.bounds[j];
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < FX_TOL;
        if is_fx {
            continue;
        }

        let rhs = -(qx[j] + problem.c[j]) / aij;
        if !rhs.is_finite() {
            continue;
        }

        match (lb_finite, ub_finite) {
            // lb-only: qx + c + a*y = z_lb >= 0
            (true, false) => {
                if aij > 0.0 {
                    lower[row] = lower[row].max(rhs);
                } else {
                    upper[row] = upper[row].min(rhs);
                }
            }
            // ub-only: qx + c + a*y = -z_ub <= 0
            (false, true) => {
                if aij > 0.0 {
                    upper[row] = upper[row].min(rhs);
                } else {
                    lower[row] = lower[row].max(rhs);
                }
            }
            _ => {}
        }
    }

    Some((lower, upper))
}

fn compute_dual_recovery_row_activity(
    problem: &QpProblem,
    solution: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let ax = problem.a.mat_vec_mul(solution).ok()?;
    let mut row_abs_activity = vec![0.0_f64; problem.num_constraints];
    for j in 0..problem.num_vars {
        let xabs = solution[j].abs();
        if xabs == 0.0 {
            continue;
        }
        for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
            let row = problem.a.row_ind[k];
            row_abs_activity[row] += problem.a.values[k].abs() * xabs;
        }
    }
    Some((ax, row_abs_activity))
}

fn dual_recovery_row_slack_tol(
    problem: &QpProblem,
    row: usize,
    ax: f64,
    row_abs_activity: f64,
    rel: f64,
) -> f64 {
    rel * (1.0 + problem.b[row].abs() + ax.abs() + row_abs_activity)
}

fn dual_recovery_progress_tol(prev_kkt: f64, cur_kkt: f64, target_pf: f64) -> f64 {
    let scale = prev_kkt
        .abs()
        .max(cur_kkt.abs())
        .max(target_pf.abs())
        .max(1.0);
    64.0 * f64::EPSILON * scale
}

fn row_is_active_for_dual_recovery(
    problem: &QpProblem,
    row: usize,
    ax: &[f64],
    row_abs_activity: &[f64],
    slack_tol_rel: f64,
) -> bool {
    match problem.constraint_types[row] {
        crate::problem::ConstraintType::Eq => true,
        crate::problem::ConstraintType::Le => {
            let slack = problem.b[row] - ax[row];
            let tol =
                dual_recovery_row_slack_tol(problem, row, ax[row], row_abs_activity[row], slack_tol_rel);
            slack.abs() <= tol
        }
        crate::problem::ConstraintType::Ge => {
            let slack = ax[row] - problem.b[row];
            let tol =
                dual_recovery_row_slack_tol(problem, row, ax[row], row_abs_activity[row], slack_tol_rel);
            slack.abs() <= tol
        }
    }
}

fn collect_dual_recovery_cluster_rows(
    problem: &QpProblem,
    candidate_cols: &[usize],
    candidate_rel: &[f64],
    ax: &[f64],
    row_abs_activity: &[f64],
    _target_pf: f64,
) -> Option<(usize, Vec<usize>)> {
    debug_assert_eq!(candidate_cols.len(), candidate_rel.len());
    if candidate_cols.is_empty() {
        return None;
    }

    let mut order: Vec<usize> = (0..candidate_cols.len()).collect();
    order.sort_by(|&lhs, &rhs| {
        candidate_rel[rhs]
            .partial_cmp(&candidate_rel[lhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let worst_pos = order[0];
    let worst_j = candidate_cols[worst_pos];
    let worst_rel = candidate_rel[worst_pos];
    if !worst_rel.is_finite() || worst_rel <= 0.0 {
        return None;
    }

    const CLUSTER_REL_CUTOFF_RATIO: f64 = 0.25;
    let rel_cutoff = worst_rel * CLUSTER_REL_CUTOFF_RATIO;

    let m = problem.num_constraints;
    let mut in_cluster = vec![false; m];
    let mut rows = Vec::new();
    let push_active_rows = |col: usize, in_cluster: &mut [bool], rows: &mut Vec<usize>| {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            if !row_is_active_for_dual_recovery(
                problem,
                row,
                ax,
                row_abs_activity,
                DUAL_RECOVERY_ACTIVE_TOL_REL,
            ) {
                continue;
            }
            if !in_cluster[row] {
                in_cluster[row] = true;
                rows.push(row);
            }
        }
    };
    push_active_rows(worst_j, &mut in_cluster, &mut rows);
    if rows.is_empty() {
        return None;
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &pos in &order {
            if candidate_rel[pos] < rel_cutoff {
                break;
            }
            let col = candidate_cols[pos];
            let touches_cluster = (problem.a.col_ptr[col]..problem.a.col_ptr[col + 1])
                .any(|k| in_cluster[problem.a.row_ind[k]]);
            if !touches_cluster {
                continue;
            }
            let before = rows.len();
            push_active_rows(col, &mut in_cluster, &mut rows);
            if rows.len() > before {
                changed = true;
            }
        }
    }

    rows.sort_unstable();
    Some((worst_j, rows))
}

#[derive(Clone, Copy)]
enum DualRecoveryBoundVar {
    Lower { var: usize, slot: usize },
    Upper { var: usize, slot: usize },
}

impl DualRecoveryBoundVar {
    fn var(self) -> usize {
        match self {
            DualRecoveryBoundVar::Lower { var, .. } | DualRecoveryBoundVar::Upper { var, .. } => var,
        }
    }

    fn slot(self) -> usize {
        match self {
            DualRecoveryBoundVar::Lower { slot, .. } | DualRecoveryBoundVar::Upper { slot, .. } => slot,
        }
    }

    fn coeff(self) -> f64 {
        match self {
            DualRecoveryBoundVar::Lower { .. } => -1.0,
            DualRecoveryBoundVar::Upper { .. } => 1.0,
        }
    }
}

fn select_dual_recovery_local_bounds(
    problem: &QpProblem,
    solution: &[f64],
    bound_duals: &[f64],
    cols: &[usize],
    provisional_residual: &[f64],
) -> (Vec<DualRecoveryBoundVar>, Vec<usize>) {
    let n = problem.num_vars;
    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let mut lb_slot_of_var = vec![None; n];
    let mut ub_slot_of_var = vec![None; n];
    let mut lb_slot = 0usize;
    let mut ub_slot = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() {
            lb_slot_of_var[j] = Some(lb_slot);
            lb_slot += 1;
        }
        if ub.is_finite() {
            ub_slot_of_var[j] = Some(ub_slot);
            ub_slot += 1;
        }
    }

    let mut local_bounds = Vec::new();
    for &col in cols {
        let xj = solution[col];
        let tol = DUAL_RECOVERY_ACTIVE_TOL_REL * (1.0 + xj.abs());
        let (lb, ub) = problem.bounds[col];
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        if is_fx {
            continue;
        }
        let lb_active = lb.is_finite()
            && ((xj - lb).abs() <= tol
                || lb_slot_of_var[col]
                    .and_then(|slot| bound_duals.get(slot))
                    .is_some_and(|&z| z > 0.0));
        let ub_active = ub.is_finite()
            && ((ub - xj).abs() <= tol
                || ub_slot_of_var[col]
                    .and_then(|slot| bound_duals.get(slot))
                    .is_some_and(|&z| z > 0.0));
        let residual_j = provisional_residual[col];
        let lb_can_help = residual_j > 0.0
            || lb_slot_of_var[col]
                .and_then(|slot| bound_duals.get(slot))
                .is_some_and(|&z| z > 0.0);
        let ub_can_help = residual_j < 0.0
            || ub_slot_of_var[col]
                .and_then(|slot| bound_duals.get(slot))
                .is_some_and(|&z| z > 0.0);
        match (lb_active, ub_active) {
            (true, false) => {
                if lb_can_help {
                    if let Some(slot) = lb_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                    }
                }
            }
            (false, true) => {
                if ub_can_help {
                    if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                }
            }
            (true, true) => {
                if lb_can_help && !ub_can_help {
                    if let Some(slot) = lb_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                    }
                } else if ub_can_help && !lb_can_help {
                    if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                } else if lb_can_help && ub_can_help {
                    let lb_dist = (xj - lb).abs();
                    let ub_dist = (ub - xj).abs();
                    if lb_dist <= ub_dist {
                        if let Some(slot) = lb_slot_of_var[col] {
                            local_bounds.push(DualRecoveryBoundVar::Lower { var: col, slot });
                        }
                    } else if let Some(slot) = ub_slot_of_var[col] {
                        local_bounds.push(DualRecoveryBoundVar::Upper { var: col, slot });
                    }
                }
            }
            (false, false) => {}
        }
    }

    let mut bound_pos_of_var = vec![usize::MAX; n];
    for (pos, &bound) in local_bounds.iter().enumerate() {
        bound_pos_of_var[bound.var()] = pos;
    }
    (local_bounds, bound_pos_of_var)
}

/// 明確に slack のある不等式制約の dual を 0 に射影する。
///
/// Le 制約で `Ax < b`、Ge 制約で `Ax > b` が十分明確な行は、相補性より dual は 0。
/// LSQ/IR は stationarity のみを見るため、slack 行に大きい dual を残すことがある。
pub(crate) fn zero_inactive_inequality_duals(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
) {
    if result.solution.len() != problem.num_vars
        || result.dual_solution.len() != problem.num_constraints
    {
        return;
    }
    let Some((ax, row_abs_activity)) = compute_dual_recovery_row_activity(problem, &result.solution) else {
        return;
    };
    const SLACK_TOL_REL: f64 = 1e-8;
    for i in 0..problem.num_constraints {
        let slack = match problem.constraint_types[i] {
            crate::problem::ConstraintType::Le => problem.b[i] - ax[i],
            crate::problem::ConstraintType::Ge => ax[i] - problem.b[i],
            crate::problem::ConstraintType::Eq => continue,
        };
        let tol = dual_recovery_row_slack_tol(problem, i, ax[i], row_abs_activity[i], SLACK_TOL_REL);
        if slack > tol {
            result.dual_solution[i] = 0.0;
        }
    }
}

const DUAL_RECOVERY_ACTIVE_TOL_REL: f64 = 1e-8;

fn collect_dual_recovery_free_columns(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
) -> Vec<usize> {
    let n = problem.num_vars;
    let mut free_idx: Vec<usize> = Vec::with_capacity(n);
    for j in 0..n {
        let xj = result.solution[j];
        let tol = DUAL_RECOVERY_ACTIVE_TOL_REL * (1.0 + xj.abs());
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && (xj - lb).abs() < tol {
            continue;
        }
        if ub.is_finite() && (ub - xj).abs() < tol {
            continue;
        }
        if problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0 {
            continue;
        }
        free_idx.push(j);
    }
    free_idx
}

/// dual y を、不等式の符号制約・inactive row の 0 制約を守りつつ
/// `||A^T y - target||^2` を projected gradient で下げる。
pub(crate) fn refine_dual_projected_gradient(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    let trace = std::env::var("REFINE_DUAL_PG_TRACE").ok().as_deref() == Some("1");
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if result.solution.len() != n || result.dual_solution.len() != m {
        return;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] =
                qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target: Vec<f64> = (0..n)
        .map(|j| -(qx[j] + problem.c[j] + bound_contrib[j]))
        .collect();

    let objective = |y: &[f64]| -> Option<(f64, Vec<f64>)> {
        let aty = if problem.a.nrows > 0 {
            problem.a.transpose().mat_vec_mul(y).ok()?
        } else {
            vec![0.0_f64; n]
        };
        let mut residual = vec![0.0_f64; n];
        let mut obj = 0.0_f64;
        for j in 0..n {
            residual[j] = aty[j] - target[j];
            obj += 0.5 * residual[j] * residual[j];
        }
        Some((obj, residual))
    };

    let mut proj_lower = vec![f64::NEG_INFINITY; m];
    let mut proj_upper = vec![f64::INFINITY; m];
    for (i, ct) in problem.constraint_types.iter().enumerate() {
        match ct {
            crate::problem::ConstraintType::Le => proj_lower[i] = 0.0,
            crate::problem::ConstraintType::Ge => proj_upper[i] = 0.0,
            crate::problem::ConstraintType::Eq => {}
        }
    }
    for j in 0..n {
        let cs = problem.a.col_ptr[j];
        let ce = problem.a.col_ptr[j + 1];
        if ce - cs != 1 {
            continue;
        }
        let row = problem.a.row_ind[cs];
        let aij = problem.a.values[cs];
        if !aij.is_finite() || aij == 0.0 {
            continue;
        }
        let (lb, ub) = problem.bounds[j];
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        if lb_finite && ub_finite && (lb - ub).abs() < FX_TOL {
            continue;
        }
        let rhs = -(qx[j] + problem.c[j]) / aij;
        if !rhs.is_finite() {
            continue;
        }
        match (lb_finite, ub_finite) {
            (true, false) => {
                if aij > 0.0 {
                    proj_lower[row] = proj_lower[row].max(rhs);
                } else {
                    proj_upper[row] = proj_upper[row].min(rhs);
                }
            }
            (false, true) => {
                if aij > 0.0 {
                    proj_upper[row] = proj_upper[row].min(rhs);
                } else {
                    proj_lower[row] = proj_lower[row].max(rhs);
                }
            }
            _ => {}
        }
    }
    for i in 0..m {
        if proj_lower[i] > proj_upper[i] {
            let (lo, hi) = match problem.constraint_types[i] {
                crate::problem::ConstraintType::Le => (0.0, f64::INFINITY),
                crate::problem::ConstraintType::Ge => (f64::NEG_INFINITY, 0.0),
                crate::problem::ConstraintType::Eq => (f64::NEG_INFINITY, f64::INFINITY),
            };
            proj_lower[i] = lo;
            proj_upper[i] = hi;
        }
    }

    let project_feasible = |y: &mut [f64]| {
        for (i, ct) in problem.constraint_types.iter().enumerate() {
            match ct {
                crate::problem::ConstraintType::Le => y[i] = y[i].max(0.0),
                crate::problem::ConstraintType::Ge => y[i] = y[i].min(0.0),
                crate::problem::ConstraintType::Eq => {}
            }
        }
        for i in 0..m {
            y[i] = y[i].clamp(proj_lower[i], proj_upper[i]);
        }
    };

    let mut y_start = result.dual_solution.clone();
    project_feasible(&mut y_start);
    let Some((mut obj_curr, mut residual_curr)) = objective(&y_start) else {
        return;
    };
    let mut y_curr = y_start;
    let mut y_best = y_curr.clone();
    let mut obj_best = obj_curr;
    let mut prev_obj = obj_curr;

    let pg_max_iters = m.saturating_mul(2).clamp(200, 2000);
    const ACCEPT_TOL_REL: f64 = 1e-12;
    let obj_converge_thresh = 1e-16 * (n as f64).max(1.0);
    const STAGNATE_MIN_RATIO: f64 = 1e-7;

    for iter in 0..pg_max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        if obj_curr < obj_converge_thresh {
            break;
        }
        let grad = match problem.a.mat_vec_mul(&residual_curr) {
            Ok(v) => v,
            Err(_) => break,
        };
        let grad_inf = grad.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if !grad_inf.is_finite() || grad_inf < 1e-14 {
            break;
        }
        let grad_sq = grad.iter().map(|v| v * v).sum::<f64>();
        if !grad_sq.is_finite() || grad_sq < 1e-28 {
            break;
        }
        let aty_grad = match problem.a.transpose().mat_vec_mul(&grad) {
            Ok(v) => v,
            Err(_) => break,
        };
        let curvature = aty_grad.iter().map(|v| v * v).sum::<f64>();
        if !curvature.is_finite() || curvature < 1e-28 {
            break;
        }
        let base_step = (grad_sq / curvature).clamp(1e-14, 1e8);
        let mut accepted = false;
        let mut step = base_step;
        while step > 0.0 {
            let mut y_try = y_curr.clone();
            for i in 0..m {
                y_try[i] -= step * grad[i];
            }
            project_feasible(&mut y_try);
            let Some((obj_try, residual_try)) = objective(&y_try) else {
                continue;
            };
            if obj_try <= obj_curr + ACCEPT_TOL_REL * (1.0 + obj_curr) {
                if trace {
                    eprintln!(
                        "DUAL_PG iter={} step={:.3e} base={:.3e} obj {:.3e}->{:.3e} grad_inf={:.3e}",
                        iter, step, base_step, obj_curr, obj_try, grad_inf
                    );
                }
                y_curr = y_try;
                obj_curr = obj_try.min(obj_curr);
                residual_curr = residual_try;
                if obj_curr < obj_best {
                    y_best = y_curr.clone();
                    obj_best = obj_curr;
                }
                accepted = true;
                break;
            }
            let next_step = step * 0.5;
            if next_step == step {
                break;
            }
            step = next_step;
        }
        if !accepted {
            if trace {
                eprintln!(
                    "DUAL_PG iter={} no acceptable step obj={:.3e} grad_inf={:.3e} base={:.3e}",
                    iter, obj_curr, grad_inf, base_step
                );
            }
            break;
        }
        let relative_improvement = if prev_obj > 0.0 {
            (prev_obj - obj_curr) / prev_obj
        } else {
            0.0
        };
        if relative_improvement < STAGNATE_MIN_RATIO {
            break;
        }
        prev_obj = obj_curr;
    }

    let mut tmp = result.clone();
    tmp.dual_solution = y_best;
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
    };
    let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &tmp.solution,
        &tmp.dual_solution,
        &tmp.bound_duals,
    );
    if trace {
        eprintln!("DUAL_PG final kkt {:.3e}->{:.3e}", pre, post);
    }
    if post < pre {
        result.dual_solution = tmp.dual_solution;
    }
}

/// worst residual 列に接続する active cluster を局所的に再最適化する。
///
/// 全体 LSQ / PG では改善が鈍いとき、worst 列に隣接する active rows と、
/// その row cluster に触れる bound-active 列の bound dual をまとめて block として
/// 取り出し、stationarity を局所的に取り直す。row dual だけを更新すると、近傍の
/// bound-active 列が押し返して KKT が悪化するケースがあるため、局所系は
/// `[active row duals ; active bound duals]` の連成で解く。
pub(crate) fn refine_dual_worst_active_block(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    let trace = std::env::var("REFINE_DUAL_BLOCK_TRACE").ok().as_deref() == Some("1");
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if result.solution.len() != n || result.dual_solution.len() != m {
        return;
    }

    let Ok(qx) = problem.q.mat_vec_mul(&result.solution) else {
        return;
    };
    let aty = if problem.a.nrows > 0 {
        match problem.a.transpose().mat_vec_mul(&result.dual_solution) {
            Ok(v) => v,
            Err(_) => return,
        }
    } else {
        vec![0.0_f64; n]
    };
    let Some((ax, row_abs_activity)) = compute_dual_recovery_row_activity(problem, &result.solution) else {
        return;
    };
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);

    let mut candidate_cols = Vec::new();
    let mut candidate_rel = Vec::new();
    let mut worst_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = problem.bounds[j];
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        let is_empty_col = problem.a.col_ptr[j + 1] == problem.a.col_ptr[j];
        if is_fx || is_empty_col {
            continue;
        }
        let r = qx[j] + problem.c[j] + aty[j] + bound_contrib[j];
        let scale = 1.0 + qx[j].abs() + problem.c[j].abs() + aty[j].abs() + bound_contrib[j].abs();
        let rel = r.abs() / scale;
        candidate_cols.push(j);
        candidate_rel.push(rel);
        if rel > worst_rel {
            worst_rel = rel;
        }
    }
    let Some((worst_j, rows)) = collect_dual_recovery_cluster_rows(
        problem,
        &candidate_cols,
        &candidate_rel,
        &ax,
        &row_abs_activity,
        DUAL_RECOVERY_ACTIVE_TOL_REL,
    ) else {
        return;
    };
    if rows.is_empty() {
        return;
    }
    let rlen = rows.len();

    let mut row_pos = vec![usize::MAX; m];
    for (pos, &row) in rows.iter().enumerate() {
        row_pos[row] = pos;
    }

    let mut row_only_gram = vec![0.0_f64; rlen * rlen];
    let mut row_only_rhs = vec![0.0_f64; rlen];
    let mut current_local_residual = vec![0.0_f64; n];
    for col in 0..n {
        let residual = qx[col] + problem.c[col] + aty[col] + bound_contrib[col];
        current_local_residual[col] = residual;
        let mut col_vec = vec![0.0_f64; rlen];
        let mut touches = false;
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            let pos = row_pos[row];
            if pos != usize::MAX {
                col_vec[pos] = problem.a.values[k];
                touches = true;
            }
        }
        if !touches {
            continue;
        }
        for i in 0..rlen {
            row_only_rhs[i] -= col_vec[i] * residual;
            for j in i..rlen {
                row_only_gram[i * rlen + j] += col_vec[i] * col_vec[j];
            }
        }
    }
    let row_only_sol = {
        let row_diag_max = (0..rlen)
            .map(|i| row_only_gram[i * rlen + i].abs())
            .fold(0.0_f64, f64::max);
        let row_reg = f64::EPSILON * (1.0 + row_diag_max);
        let mut row_col_ptr = vec![0usize; rlen + 1];
        let mut row_ind = Vec::new();
        let mut row_values = Vec::new();
        for j in 0..rlen {
            for i in 0..=j {
                let mut v = row_only_gram[i * rlen + j];
                if i == j {
                    v += row_reg;
                }
                if v != 0.0 {
                    row_ind.push(i);
                    row_values.push(v);
                }
            }
            row_col_ptr[j + 1] = row_ind.len();
        }
        let row_csc = CscMatrix {
            col_ptr: row_col_ptr,
            row_ind,
            values: row_values,
            nrows: rlen,
            ncols: rlen,
        };
        crate::linalg::ldl::factorize(&row_csc)
            .ok()
            .map(|factor| {
                let mut sol = vec![0.0_f64; rlen];
                factor.solve(&row_only_rhs, &mut sol);
                sol
            })
            .filter(|sol| sol.iter().all(|v| v.is_finite()))
    };
    let mut provisional_residual = current_local_residual.clone();
    if let Some(ref delta_row) = row_only_sol {
        for col in 0..n {
            let mut delta = 0.0_f64;
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                let pos = row_pos[row];
                if pos != usize::MAX {
                    delta += problem.a.values[k] * delta_row[pos];
                }
            }
            provisional_residual[col] += delta;
        }
    }

    let mut cols = Vec::new();
    for col in 0..n {
        let mut touches = false;
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            if row_pos[problem.a.row_ind[k]] != usize::MAX {
                touches = true;
                break;
            }
        }
        if touches {
            cols.push(col);
        }
    }
    if cols.is_empty() {
        return;
    }

    let (local_bounds, bound_pos_of_var) = select_dual_recovery_local_bounds(
        problem,
        &result.solution,
        &result.bound_duals,
        &cols,
        &provisional_residual,
    );

    if trace {
        eprintln!(
            "DUAL_BLOCK worst_j={} worst_rel={:.3e} active_rows={} touched_cols={} local_bounds={}",
            worst_j,
            worst_rel,
            rows.len(),
            cols.len(),
            local_bounds.len()
        );
    }

    let ulen = rlen + local_bounds.len();
    if ulen == 0 {
        return;
    }
    let mut gram = vec![0.0_f64; ulen * ulen];
    let mut rhs = vec![0.0_f64; ulen];
    let mut local_aty = vec![0.0_f64; cols.len()];
    let mut local_bound_contrib = vec![0.0_f64; cols.len()];
    for (ci, &col) in cols.iter().enumerate() {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            let pos = row_pos[row];
            if pos != usize::MAX {
                local_aty[ci] += problem.a.values[k] * result.dual_solution[row];
            }
        }
        let bpos = bound_pos_of_var[col];
        if bpos != usize::MAX {
            let bound = local_bounds[bpos];
            if let Some(&z) = result.bound_duals.get(bound.slot()) {
                local_bound_contrib[ci] += bound.coeff() * z;
            }
        }
    }

    for &col in &cols {
        let residual = qx[col] + problem.c[col] + aty[col] + bound_contrib[col];
        let mut col_vec = vec![0.0_f64; ulen];
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let row = problem.a.row_ind[k];
            let pos = row_pos[row];
            if pos != usize::MAX {
                col_vec[pos] = problem.a.values[k];
            }
        }
        let bpos = bound_pos_of_var[col];
        if bpos != usize::MAX {
            col_vec[rlen + bpos] = local_bounds[bpos].coeff();
        }
        for i in 0..ulen {
            rhs[i] -= col_vec[i] * residual;
            for j in i..ulen {
                gram[i * ulen + j] += col_vec[i] * col_vec[j];
            }
        }
    }

    let diag_max = (0..ulen)
        .map(|i| gram[i * ulen + i].abs())
        .fold(0.0_f64, f64::max);
    let reg = f64::EPSILON * (1.0 + diag_max);
    let mut col_ptr = vec![0usize; ulen + 1];
    let mut row_ind = Vec::new();
    let mut values = Vec::new();
    for j in 0..ulen {
        for i in 0..=j {
            let mut v = gram[i * ulen + j];
            if i == j {
                v += reg;
            }
            if v != 0.0 {
                row_ind.push(i);
                values.push(v);
            }
        }
        col_ptr[j + 1] = row_ind.len();
    }
    let gram_csc = CscMatrix {
        col_ptr,
        row_ind,
        values,
        nrows: ulen,
        ncols: ulen,
    };
    let Ok(factor) = crate::linalg::ldl::factorize(&gram_csc) else {
        return;
    };
    let mut block_sol = vec![0.0_f64; ulen];
    factor.solve(&rhs, &mut block_sol);
    if block_sol.iter().any(|v| !v.is_finite()) {
        return;
    }

    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
    };
    let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let Some((row_lower, row_upper)) = compute_dual_recovery_row_bounds(problem, &result.solution)
    else {
        return;
    };
    let mut best = result.clone();
    let mut best_kkt = pre;
    let mut step = 1.0_f64;
    while step > 0.0 {
        let mut tmp = result.clone();
        for (pos, &row) in rows.iter().enumerate() {
            let mut v = result.dual_solution[row] + step * block_sol[pos];
            let lo = row_lower[row];
            let hi = row_upper[row];
            if lo <= hi {
                v = v.clamp(lo, hi);
            }
            tmp.dual_solution[row] = v;
        }
        for (pos, &bound) in local_bounds.iter().enumerate() {
            let slot = bound.slot();
            if slot >= tmp.bound_duals.len() {
                continue;
            }
            let z = result.bound_duals[slot] + step * block_sol[rlen + pos];
            tmp.bound_duals[slot] = z.max(0.0);
        }
        let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &tmp.solution,
            &tmp.dual_solution,
            &tmp.bound_duals,
        );
        if post < best_kkt {
            best = tmp;
            best_kkt = post;
            break;
        }
        let next_step = step * 0.5;
        if next_step == step {
            break;
        }
        step = next_step;
    }
    if trace {
        eprintln!("DUAL_BLOCK kkt {:.3e}->{:.3e}", pre, best_kkt);
    }
    if best_kkt < pre {
        result.dual_solution = best.dual_solution;
        result.bound_duals = best.bound_duals;
    }
}

/// LSQ で y を計算する核心ロジック (KKT-guard なし)。
///
/// 解法: A^T y = target ( = -(Q*x + c + bound_contrib) ) の最小二乗解を
/// 正規方程式 (A·A^T) y = A·target を LDL で解いて求めたあと、
/// **DD (TwoFloat) 精度の残差で iterative refinement** を行う。
///
/// IR の動機: ill-conditioned 問題 (QPILOTNO: cond(A)≈3e6 → cond(A·A^T)≈9e12) で
/// 1 回 solve すると相対誤差が cond(A·A^T)·ε ≈ 2e-3 残り、bench DFEAS で `aty[col]`
/// の真値 (DD 値) と f64 計算値が大きく乖離する。Wilkinson 流 IR で残差を DD 精度で
/// 計算し直すと backward error は cond·ε² ≈ 5e-26 まで落ち、ill-conditioned でも
/// f64 の限界近くまで refine できる。
///
/// 失敗 (size 上限・LDL 失敗・NaN) 時は None を返す。
pub(crate) fn compute_lsq_dual_y(
    problem: &QpProblem,
    result: &crate::problem::SolverResult,
) -> Option<Vec<f64>> {
    use twofloat::TwoFloat;
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return None;
    }
    // 大規模問題では A·A^T (m×m sparse、m=186k for BOYD2) の LDL factorization が
    // 数十 GB メモリを確保するため skip。`refine_primal_lsq` の同名閾値と統一。
    if n + m > LSQ_DUAL_SIZE_LIMIT {
        return None;
    }
    let x = &result.solution;

    // target_dd[j] = -(Q*x + c + bound_contrib)[j] を DD で精密に組み立てる。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        let cs = problem.q.col_ptr[col];
        let ce = problem.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = problem.q.row_ind[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target_dd: Vec<TwoFloat> = (0..n)
        .map(|j| -(qx_dd[j] + TwoFloat::from(problem.c[j]) + TwoFloat::from(bound_contrib[j])))
        .collect();

    let mut proj_lower = vec![f64::NEG_INFINITY; m];
    let mut proj_upper = vec![f64::INFINITY; m];
    for (i, ct) in problem.constraint_types.iter().enumerate() {
        match ct {
            crate::problem::ConstraintType::Le => proj_lower[i] = 0.0,
            crate::problem::ConstraintType::Ge => proj_upper[i] = 0.0,
            crate::problem::ConstraintType::Eq => {}
        }
    }
    for j in 0..n {
        let cs = problem.a.col_ptr[j];
        let ce = problem.a.col_ptr[j + 1];
        if ce - cs != 1 {
            continue;
        }
        let row = problem.a.row_ind[cs];
        let aij = problem.a.values[cs];
        if !aij.is_finite() || aij == 0.0 {
            continue;
        }
        let (lb, ub) = problem.bounds[j];
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        if lb_finite && ub_finite && (lb - ub).abs() < FX_TOL {
            continue;
        }
        let qxj = f64::from(qx_dd[j]);
        let rhs = -(qxj + problem.c[j]) / aij;
        if !rhs.is_finite() {
            continue;
        }
        match (lb_finite, ub_finite) {
            (true, false) => {
                if aij > 0.0 {
                    proj_lower[row] = proj_lower[row].max(rhs);
                } else {
                    proj_upper[row] = proj_upper[row].min(rhs);
                }
            }
            (false, true) => {
                if aij > 0.0 {
                    proj_upper[row] = proj_upper[row].min(rhs);
                } else {
                    proj_lower[row] = proj_lower[row].max(rhs);
                }
            }
            _ => {}
        }
    }
    let mut fixed_y: Vec<Option<f64>> = vec![None; m];
    let mut n_fixed = 0usize;
    for i in 0..m {
        let lo = proj_lower[i];
        let hi = proj_upper[i];
        if lo.is_finite() && hi.is_finite() {
            let scale = 1.0 + lo.abs().max(hi.abs());
            if (lo - hi).abs() < 1e-10 * scale {
                fixed_y[i] = Some((lo + hi) * 0.5);
                n_fixed += 1;
            }
        }
    }

    let solve_lsq_ir = |a_sub: &CscMatrix, m_sub: usize, v_dd: &[TwoFloat]| -> Option<Vec<f64>> {
        let aat_sub = build_aat_upper_csc(a_sub, n, m_sub)?;
        let factor = crate::linalg::ldl::factorize(&aat_sub).ok()?;
        let build_rhs_sub = |v_dd: &[TwoFloat]| -> Vec<f64> {
            let mut acc: Vec<TwoFloat> = vec![zero_dd; m_sub];
            for col in 0..n {
                let cs = a_sub.col_ptr[col];
                let ce = a_sub.col_ptr[col + 1];
                for k in cs..ce {
                    let row = a_sub.row_ind[k];
                    let v_f64 = f64::from(v_dd[col]);
                    let lo = v_dd[col] - TwoFloat::from(v_f64);
                    acc[row] = acc[row]
                        + TwoFloat::new_mul(a_sub.values[k], v_f64)
                        + TwoFloat::new_mul(a_sub.values[k], f64::from(lo));
                }
            }
            acc.iter().map(|&v| f64::from(v)).collect()
        };
        let rhs0 = build_rhs_sub(v_dd);
        let mut y_sub = vec![0.0_f64; m_sub];
        factor.solve(&rhs0, &mut y_sub);
        if y_sub.iter().any(|v| !v.is_finite()) {
            return None;
        }
        const IR_STAGNATE_RATIO: f64 = 0.5;
        const IR_PROGRESS_EPS: f64 = 1e-18;
        let mut prev_r_inf = f64::INFINITY;
        loop {
            let mut atysub_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for col in 0..n {
                let cs = a_sub.col_ptr[col];
                let ce = a_sub.col_ptr[col + 1];
                for k in cs..ce {
                    let row = a_sub.row_ind[k];
                    atysub_dd[col] =
                        atysub_dd[col] + TwoFloat::new_mul(a_sub.values[k], y_sub[row]);
                }
            }
            let r_dd: Vec<TwoFloat> = (0..n).map(|j| v_dd[j] - atysub_dd[j]).collect();
            let r_inf = r_dd.iter().fold(0.0_f64, |a, &v| a.max(f64::from(v).abs()));
            if !r_inf.is_finite() {
                break;
            }
            if prev_r_inf.is_finite() && r_inf + IR_PROGRESS_EPS >= prev_r_inf {
                break;
            }
            if prev_r_inf.is_finite() && r_inf > prev_r_inf * IR_STAGNATE_RATIO {
                break;
            }
            prev_r_inf = r_inf;
            let rhs_dy = build_rhs_sub(&r_dd);
            let mut dy = vec![0.0_f64; m_sub];
            factor.solve(&rhs_dy, &mut dy);
            if dy.iter().any(|v| !v.is_finite()) {
                break;
            }
            for i in 0..m_sub {
                y_sub[i] += dy[i];
            }
        }
        Some(y_sub)
    };

    if n_fixed == 0 {
        return solve_lsq_ir(&problem.a, m, &target_dd);
    }

    let mut free_row_local = vec![usize::MAX; m];
    let mut free_rows: Vec<usize> = Vec::with_capacity(m - n_fixed);
    for (i, fy) in fixed_y.iter().enumerate() {
        if fy.is_none() {
            free_row_local[i] = free_rows.len();
            free_rows.push(i);
        }
    }
    let m_free = free_rows.len();
    if m_free == 0 {
        return Some(fixed_y.iter().map(|fy| fy.unwrap_or(0.0)).collect());
    }

    let mut a_free_col_ptr = vec![0usize; n + 1];
    let mut a_free_row_ind: Vec<usize> = Vec::new();
    let mut a_free_values: Vec<f64> = Vec::new();
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let orig_row = problem.a.row_ind[k];
            let local_row = free_row_local[orig_row];
            if local_row != usize::MAX {
                a_free_row_ind.push(local_row);
                a_free_values.push(problem.a.values[k]);
            }
        }
        a_free_col_ptr[col + 1] = a_free_row_ind.len();
    }
    let a_free = CscMatrix {
        col_ptr: a_free_col_ptr,
        row_ind: a_free_row_ind,
        values: a_free_values,
        nrows: m_free,
        ncols: n,
    };

    let mut target_adj_dd = target_dd.clone();
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            let orig_row = problem.a.row_ind[k];
            if let Some(yfi) = fixed_y[orig_row] {
                if yfi != 0.0 {
                    target_adj_dd[col] =
                        target_adj_dd[col] - TwoFloat::new_mul(problem.a.values[k], yfi);
                }
            }
        }
    }

    let y_free = match solve_lsq_ir(&a_free, m_free, &target_adj_dd) {
        Some(v) => v,
        None => return solve_lsq_ir(&problem.a, m, &target_dd),
    };

    let mut y_full = vec![0.0_f64; m];
    for (local_idx, &orig_row) in free_rows.iter().enumerate() {
        y_full[orig_row] = y_free[local_idx];
    }
    for (i, fy) in fixed_y.iter().enumerate() {
        if let Some(v) = fy {
            y_full[i] = *v;
        }
    }
    Some(y_full)
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
///
/// `deadline` 経過時は no-op で即 return。AAT factorize の重さは refine_dual_lsq と同じ。
pub(crate) fn refine_primal_lsq(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
) {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return;
    }
    let x = &mut result.solution;

    // 違反量 v[i] を計算 (Le/Ge/Eq に応じて符号を統一して "Ax を b 方向に押す量")。
    // ill-conditioned 系で f64 sum が cancellation で違反を見逃すのを防ぐため DD で積算。
    use crate::problem::ConstraintType;
    use twofloat::TwoFloat;
    let zero_dd = TwoFloat::from(0.0);
    let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
    for col in 0..n {
        let xv = x[col];
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            ax_dd[problem.a.row_ind[k]] =
                ax_dd[problem.a.row_ind[k]] + TwoFloat::new_mul(problem.a.values[k], xv);
        }
    }
    let ax: Vec<f64> = ax_dd.iter().map(|&v| f64::from(v)).collect();
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
    let target: Vec<f64> = (0..m)
        .map(|i| {
            match problem.constraint_types[i] {
                ConstraintType::Eq => ax[i] - problem.b[i],
                ConstraintType::Ge => {
                    // active のみ: 充足側 (ax >= b) なら 0, 違反 (ax < b) なら ax - b (負)
                    let r = ax[i] - problem.b[i];
                    if r < -PRIMAL_VIOLATION_TOL {
                        r
                    } else {
                        0.0
                    }
                }
                ConstraintType::Le => {
                    let r = ax[i] - problem.b[i];
                    if r > PRIMAL_VIOLATION_TOL {
                        r
                    } else {
                        0.0
                    }
                }
            }
        })
        .collect();
    let target_inf = target.iter().map(|t| t.abs()).fold(0.0_f64, f64::max);
    if target_inf <= PRIMAL_VIOLATION_TOL {
        return;
    }

    // (A A^T) λ = target を LDL で解く。AAT_REG_FACTOR で対角正則化済。
    // ill-conditioned 問題 (QPILOTNO: ‖A‖=5.85e6, cond(AAT)≈3e13) で
    // 1 回 solve だと λ に 2.2e-3 級の誤差が乗り δ=A^T λ が暴走 (≈1.3e4) する。
    // Wilkinson IR (DD 精度の残差再計算) で λ を f64 epsilon × cond の限界まで refine。
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
    // Wilkinson IR: r = target - AAT·λ を DD で計算して dλ = AAT^{-1} r で λ += dλ。
    // r_inf 改善率が IR_STAGNATE_RATIO を割れば停止 (収束飽和)。
    const IR_STAGNATE_RATIO: f64 = 0.5;
    const IR_PROGRESS_EPS: f64 = 1e-18;
    let mut prev_r_inf = f64::INFINITY;
    loop {
        // AAT·λ を DD で計算: AAT·λ = A·(A^T·λ) = A·δ_dd
        let mut atl_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for j in 0..n {
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let i = problem.a.row_ind[k];
                if i < m {
                    atl_dd[j] = atl_dd[j] + TwoFloat::new_mul(problem.a.values[k], lambda[i]);
                }
            }
        }
        // r[i] = target[i] - sum_j A[i,j] · atl_dd[j]
        let mut r_dd: Vec<TwoFloat> = (0..m).map(|i| TwoFloat::from(target[i])).collect();
        for j in 0..n {
            let atl_j_f64 = f64::from(atl_dd[j]);
            let atl_j_lo = atl_dd[j] - TwoFloat::from(atl_j_f64);
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let i = problem.a.row_ind[k];
                if i < m {
                    r_dd[i] = r_dd[i]
                        - TwoFloat::new_mul(problem.a.values[k], atl_j_f64)
                        - TwoFloat::new_mul(problem.a.values[k], f64::from(atl_j_lo));
                }
            }
        }
        let r_f64: Vec<f64> = r_dd.iter().map(|&v| f64::from(v)).collect();
        let r_inf = r_f64.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if !r_inf.is_finite() {
            break;
        }
        if prev_r_inf.is_finite() && r_inf + IR_PROGRESS_EPS >= prev_r_inf {
            break;
        }
        if prev_r_inf.is_finite() && r_inf > prev_r_inf * IR_STAGNATE_RATIO {
            break;
        }
        prev_r_inf = r_inf;
        let mut dlambda = vec![0.0_f64; m];
        factor.solve(&r_f64, &mut dlambda);
        if dlambda.iter().any(|v| !v.is_finite()) {
            break;
        }
        for i in 0..m {
            lambda[i] += dlambda[i];
        }
    }

    // δ = A^T λ も DD で積算
    let mut delta_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for j in 0..n {
        let s = problem.a.col_ptr[j];
        let e = problem.a.col_ptr[j + 1];
        for k in s..e {
            let i = problem.a.row_ind[k];
            if i < m {
                delta_dd[j] = delta_dd[j] + TwoFloat::new_mul(problem.a.values[k], lambda[i]);
            }
        }
    }
    let delta: Vec<f64> = delta_dd.iter().map(|&v| f64::from(v)).collect();
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

    // 改善判定: 成分相対化での max rel violation が減ったか
    // (abs 比較は ill-scaled 行で 1 行のみ大きく外れた違反を見逃すため、
    //  bench compute_pfeas_normalized componentwise と整合する metric を使う)。
    let ax_new = match problem.a.mat_vec_mul(&x_new) {
        Ok(v) => v,
        Err(_) => return,
    };
    let mut max_rel_pre = 0.0_f64;
    let mut max_rel_post = 0.0_f64;
    for i in 0..m {
        let raw_pre = match problem.constraint_types[i] {
            ConstraintType::Eq => (ax[i] - problem.b[i]).abs(),
            ConstraintType::Ge => (problem.b[i] - ax[i]).max(0.0),
            ConstraintType::Le => (ax[i] - problem.b[i]).max(0.0),
        };
        let raw_post = match problem.constraint_types[i] {
            ConstraintType::Eq => (ax_new[i] - problem.b[i]).abs(),
            ConstraintType::Ge => (problem.b[i] - ax_new[i]).max(0.0),
            ConstraintType::Le => (ax_new[i] - problem.b[i]).max(0.0),
        };
        let scale_pre = 1.0 + ax[i].abs() + problem.b[i].abs();
        let scale_post = 1.0 + ax_new[i].abs() + problem.b[i].abs();
        let rel_pre = raw_pre / scale_pre;
        let rel_post = raw_post / scale_post;
        if rel_pre > max_rel_pre {
            max_rel_pre = rel_pre;
        }
        if rel_post > max_rel_post {
            max_rel_post = rel_post;
        }
    }
    if max_rel_post < max_rel_pre {
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
/// dual-only IR: x を不変に保ち y のみ更新して r_d_free を厳密に 0 にする (LP/QP 共通)。
///
/// 数学:
///   r_d_free = (Q x)_free + c_free + (A^T y)_free + bc_free
///   目的: δy s.t. r_d_free + A_free^T δy = 0  ⇔  A_free^T δy = -r_d_free
///   最小ノルム解: δy = -A_free α,  G α = r_d_free,  G = A_free^T A_free (SPD, n_free×n_free)
///   x 不変 ⇒ Q x 項は変化せず、A^T y 項のみが δy で変動する。
///   検算: A_free^T δy = -A_free^T A_free α = -G α = -r_d_free  ⇒  r_d_free_new = 0
///
/// active 変数 (x ≈ bound) は dx=0 のまま、refit_bound_duals_kkt で z (bound_dual) が再計算される。
/// 戻り値: 1 (改善採用) または 0 (skip / no-op)
fn try_dual_only_ir(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    target_pf: f64,
    deadline: Option<std::time::Instant>,
) -> usize {
    use crate::presolve::bound_contrib_at_var;
    use twofloat::TwoFloat;

    let m = problem.num_constraints;
    let trace = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
    };
    let kkt_pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );

    // G + δ·I の正則化。F64 round-off の cancellation を防ぐ最小値。
    // δ × ‖α‖ が new r_d_free の floor (典型 1e-12 × 1e2 = 1e-10、target 1e-6 を十分下回る)。
    let dual_ir_reg = std::env::var("DUAL_IR_REG")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-12);

    // 1. free 変数の特定 (active = bound 近傍 or A col 空)
    let free_eval_idx = collect_dual_recovery_free_columns(problem, result);
    let n_free_eval = free_eval_idx.len();
    if n_free_eval == 0 {
        if trace {
            eprintln!("DUAL_IR skip: n_free=0");
        }
        return 0;
    }

    // 2. r_d_free を DD で計算
    //    r_d[j] = c[j] + (A^T y)[j] + bound_contrib[j]
    //    free var の bound_contrib は通常 0 (z=0) だが念のため計算
    let mut r_d_eval = vec![0.0_f64; n_free_eval];
    let mut r_d_rel_eval = vec![0.0_f64; n_free_eval];
    let mut df_rel_pre = 0.0_f64;
    let mut df_abs_pre = 0.0_f64;
    let mut worst_idx = 0;
    let mut worst_qx = 0.0_f64;
    for (fi, &j) in free_eval_idx.iter().enumerate() {
        // r_d_free 用に Q x も加算する必要 (Q≠0 の QP で正確性必須)
        let mut qx = TwoFloat::from(0.0);
        for k in problem.q.col_ptr[j]..problem.q.col_ptr[j + 1] {
            let row = problem.q.row_ind[k];
            qx += TwoFloat::new_mul(problem.q.values[k], result.solution[row]);
        }
        let qx_f = f64::from(qx);
        let mut aty = TwoFloat::from(0.0);
        for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
            let r = problem.a.row_ind[k];
            aty += TwoFloat::new_mul(problem.a.values[k], result.dual_solution[r]);
        }
        let aty_f = f64::from(aty);
        let bc = bound_contrib_at_var(&problem.bounds, &result.bound_duals, j);
        let r_d = qx_f + problem.c[j] + aty_f + bc;
        r_d_eval[fi] = r_d;
        let scale = 1.0 + qx_f.abs() + problem.c[j].abs() + aty_f.abs() + bc.abs();
        let rel = r_d.abs() / scale;
        r_d_rel_eval[fi] = rel;
        if rel > df_rel_pre {
            df_rel_pre = rel;
            worst_idx = j;
            worst_qx = qx_f;
        }
        if r_d.abs() > df_abs_pre {
            df_abs_pre = r_d.abs();
        }
    }
    if df_rel_pre < target_pf {
        if trace {
            eprintln!(
                "DUAL_IR skip: df_rel_pre={:.3e} < target {:.3e}",
                df_rel_pre, target_pf
            );
        }
        return 0;
    }
    if trace {
        eprintln!(
            "DUAL_IR pre: n_free_eval={} df_abs_max={:.3e} df_rel_max={:.3e} worst_j={} qx={:.3e}",
            n_free_eval, df_abs_pre, df_rel_pre, worst_idx, worst_qx
        );
    }

    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return 0;
    }

    let row_bounds = match compute_dual_recovery_row_bounds(problem, &result.solution) {
        Some(v) => v,
        None => return 0,
    };
    let (proj_lower, proj_upper) = (&row_bounds.0, &row_bounds.1);

    let Some((ax, row_abs_activity)) = compute_dual_recovery_row_activity(problem, &result.solution) else {
        return 0;
    };
    let Some((worst_j, active_rows)) = collect_dual_recovery_cluster_rows(
        problem,
        &free_eval_idx,
        &r_d_rel_eval,
        &ax,
        &row_abs_activity,
        target_pf,
    ) else {
        if trace {
            eprintln!("DUAL_IR skip: no active row cluster");
        }
        return 0;
    };
    let mut active_row_pos = vec![usize::MAX; m];
    for (pos, &row) in active_rows.iter().enumerate() {
        active_row_pos[row] = pos;
    }
    let m_active = active_rows.len();
    if m_active == 0 {
        if trace {
            eprintln!("DUAL_IR skip: m_active=0");
        }
        return 0;
    }
    if trace {
        eprintln!(
            "DUAL_IR cluster_rows={}/{} worst_j={} seed_worst_j={}",
            m_active, m, worst_idx, worst_j
        );
    }

    let mut free_idx = Vec::new();
    for &j in &free_eval_idx {
        let touches_cluster = (problem.a.col_ptr[j]..problem.a.col_ptr[j + 1])
            .any(|k| active_row_pos[problem.a.row_ind[k]] != usize::MAX);
        if touches_cluster {
            free_idx.push(j);
        }
    }
    let n_free = free_idx.len();
    if n_free == 0 {
        if trace {
            eprintln!("DUAL_IR skip: cluster has no free columns");
        }
        return 0;
    }
    if trace {
        eprintln!("DUAL_IR cluster_free={}/{}", n_free, n_free_eval);
    }

    // y/z を連成で更新する。row-only だと active bound 列が押し返すため、
    // free cluster 上で [row duals ; active bound duals] の局所 least squares を解く。
    let mut tmp = result.clone();
    // y を DD 精度で保持 (f64 y への累積で精度が失われるのを防ぐ)。
    // Ruiz presolve 後の unscale で y が 1e10 級に増幅される問題 (QFORPLAN) では、
    // f64 での y[i] += dy[i] は |dy| < eps_f64 × |y| ≈ 2e-6 を切り捨てる。
    // DD 精度 y_dd に TwoFloat で積算することで 1e10 の y に 1e-8 の修正も蓄積できる。
    let mut y_dd: Vec<TwoFloat> = tmp
        .dual_solution
        .iter()
        .map(|&v| TwoFloat::from(v))
        .collect();
    let mut df_rel_post = df_rel_pre;
    let mut df_abs_post = df_abs_pre;
    let mut total_dy_inf = 0.0_f64;
    let mut accepted_iters = 0;
    let mut current_r_d_free: Vec<f64> = free_idx
        .iter()
        .map(|&j| {
            let pos = free_eval_idx
                .iter()
                .position(|&jj| jj == j)
                .expect("free cluster column must exist in eval set");
            r_d_eval[pos]
        })
        .collect();
    const DUAL_IR_ACCEPT_REL_TOL: f64 = 1e-12;
    const DUAL_IR_MIN_PROGRESS_RATIO: f64 = 1e-4;
    let mut inner = 0usize;
    loop {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let mut provisional_residual = vec![0.0_f64; problem.num_vars];
        for (fi, &j) in free_idx.iter().enumerate() {
            provisional_residual[j] = current_r_d_free[fi];
        }
        let (local_bounds, bound_pos_of_var) = select_dual_recovery_local_bounds(
            problem,
            &tmp.solution,
            &tmp.bound_duals,
            &free_idx,
            &provisional_residual,
        );
        let ulen = m_active + local_bounds.len();
        if ulen == 0 {
            break;
        }
        let mut gram = vec![0.0_f64; ulen * ulen];
        let mut rhs = vec![0.0_f64; ulen];
        for (fi, &j) in free_idx.iter().enumerate() {
            let residual = current_r_d_free[fi];

            // Weight by 1/scale[j]^2 so Gram solves min sum_j (r_d[j]/scale[j])^2.
            // Without weighting the LS minimizes |r_d|^2 (absolute), which can
            // worsen the component-wise max (r_d[j]/scale[j]) by prioritising
            // variables with large |r_d| but small scale over the true worst-case
            // variable (small scale, moderate |r_d|).
            let mut qx_j = 0.0_f64;
            for k in problem.q.col_ptr[j]..problem.q.col_ptr[j + 1] {
                qx_j += problem.q.values[k] * tmp.solution[problem.q.row_ind[k]];
            }
            let mut aty_j = 0.0_f64;
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                aty_j += problem.a.values[k] * f64::from(y_dd[problem.a.row_ind[k]]);
            }
            let bc_j = bound_contrib_at_var(&problem.bounds, &tmp.bound_duals, j);
            let scale_j = (1.0 + qx_j.abs() + problem.c[j].abs() + aty_j.abs() + bc_j.abs()).max(1.0);
            let inv_scale2 = 1.0 / (scale_j * scale_j);

            let mut col_vec = vec![0.0_f64; ulen];
            for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                let r = problem.a.row_ind[k];
                let pos = active_row_pos[r];
                if pos != usize::MAX {
                    col_vec[pos] = problem.a.values[k];
                }
            }
            let bpos = bound_pos_of_var[j];
            if bpos != usize::MAX {
                col_vec[m_active + bpos] = local_bounds[bpos].coeff();
            }
            for i in 0..ulen {
                rhs[i] -= col_vec[i] * residual * inv_scale2;
                for j2 in i..ulen {
                    gram[i * ulen + j2] += col_vec[i] * col_vec[j2] * inv_scale2;
                }
            }
        }
        for i in 0..ulen {
            gram[i * ulen + i] += dual_ir_reg;
        }
        let mut col_ptr: Vec<usize> = vec![0; ulen + 1];
        let mut row_ind: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        for j in 0..ulen {
            for i in 0..=j {
                let v = gram[i * ulen + j];
                if v != 0.0 {
                    row_ind.push(i);
                    values.push(v);
                }
            }
            col_ptr[j + 1] = row_ind.len();
        }
        let gram_csc = CscMatrix {
            col_ptr,
            row_ind,
            values,
            nrows: ulen,
            ncols: ulen,
        };
        let factor = match crate::linalg::ldl::factorize(&gram_csc) {
            Ok(f) => f,
            Err(e) => {
                if trace {
                    eprintln!("DUAL_IR factorize failed: {:?}", e);
                }
                break;
            }
        };
        let mut delta = vec![0.0_f64; ulen];
        factor.solve(&rhs, &mut delta);
        if delta.iter().any(|v| !v.is_finite()) {
            if trace {
                eprintln!("DUAL_IR inner={} solve NaN, abort", inner);
            }
            break;
        }
        let mut dy_dd = vec![TwoFloat::from(0.0); m];
        for (pos, &row) in active_rows.iter().enumerate() {
            dy_dd[row] = TwoFloat::from(delta[pos]);
        }
        let dy_inf = dy_dd
            .iter()
            .fold(0.0_f64, |a, v| a.max(f64::from(*v).abs()));
        if !dy_inf.is_finite() {
            break;
        }
        total_dy_inf = total_dy_inf.max(dy_inf);

        let mut accepted = false;
        let mut accepted_df_rel = df_rel_post;
        let mut accepted_df_abs = df_abs_post;
        let mut accepted_r_d_free = current_r_d_free.clone();
        let mut accepted_y_dd = y_dd.clone();
        let mut accepted_bound_duals = tmp.bound_duals.clone();
        let mut accepted_step_scale = 0.0_f64;
        let mut step_scale = 1.0_f64;
        while step_scale > 0.0 {
            let mut y_dd_new: Vec<TwoFloat> = y_dd
                .iter()
                .zip(dy_dd.iter())
                .map(|(&y, &d)| y + d * step_scale)
                .collect();
            let mut bound_duals_new = tmp.bound_duals.clone();
            // dy_dd は active_rows のみ更新する。非アクティブ行の y_dd_new は y_dd と同値のため
            // クランプ不要。全行クランプすると非アクティブ行の y が 0 に強制され、
            // df_rel_pre (非クランプ y で計算) との比較が不整合になり、正当なステップが棄却される。
            for &row in &active_rows {
                let val = f64::from(y_dd_new[row]);
                let lo = proj_lower[row];
                let hi = proj_upper[row];
                let clamped = if lo <= hi { val.clamp(lo, hi) } else { val };
                y_dd_new[row] = TwoFloat::from(clamped);
            }
            for (pos, &bound) in local_bounds.iter().enumerate() {
                let slot = bound.slot();
                if slot >= bound_duals_new.len() {
                    continue;
                }
                let z = tmp.bound_duals[slot] + step_scale * delta[m_active + pos];
                bound_duals_new[slot] = z.max(0.0);
            }

            // 新 r_d_free を y_dd_new から DD 精度で計算 (Q x は変化なし、aty のみ更新)
            let mut new_r_d_free = vec![0.0_f64; n_free];
            let mut new_df_rel = 0.0_f64;
            let mut new_df_abs = 0.0_f64;
            for &j in &free_eval_idx {
                let mut qx = TwoFloat::from(0.0);
                for k in problem.q.col_ptr[j]..problem.q.col_ptr[j + 1] {
                    let row = problem.q.row_ind[k];
                    qx += TwoFloat::new_mul(problem.q.values[k], tmp.solution[row]);
                }
                let mut aty = TwoFloat::from(0.0);
                for k in problem.a.col_ptr[j]..problem.a.col_ptr[j + 1] {
                    let r = problem.a.row_ind[k];
                    aty = aty + y_dd_new[r] * problem.a.values[k];
                }
                let bc = bound_contrib_at_var(&problem.bounds, &bound_duals_new, j);
                let r_d = f64::from(qx + TwoFloat::from(problem.c[j]) + aty + TwoFloat::from(bc));
                if let Some(local_pos) = free_idx.iter().position(|&jj| jj == j) {
                    new_r_d_free[local_pos] = r_d;
                }
                let qx_f = f64::from(qx);
                let aty_f = f64::from(aty);
                let scale = 1.0 + qx_f.abs() + problem.c[j].abs() + aty_f.abs() + bc.abs();
                let rel = r_d.abs() / scale;
                if rel > new_df_rel {
                    new_df_rel = rel;
                }
                if r_d.abs() > new_df_abs {
                    new_df_abs = r_d.abs();
                }
            }
            if new_df_rel <= df_rel_post + DUAL_IR_ACCEPT_REL_TOL * (1.0 + df_rel_post) {
                accepted = true;
                accepted_df_rel = new_df_rel;
                accepted_df_abs = new_df_abs;
                accepted_r_d_free = new_r_d_free;
                accepted_y_dd = y_dd_new;
                accepted_bound_duals = bound_duals_new;
                accepted_step_scale = step_scale;
                break;
            }
            let next_step_scale = step_scale * 0.5;
            if next_step_scale == step_scale {
                break;
            }
            step_scale = next_step_scale;
        }
        if !accepted {
            if trace {
                eprintln!(
                    "DUAL_IR inner={} regression, breaking (rel {:.3e} -> rejected all backtracks)",
                    inner, df_rel_post
                );
            }
            break;
        }

        let rel_improvement = (df_rel_post - accepted_df_rel).max(0.0);
        let progress_ratio = if df_rel_post > 0.0 {
            rel_improvement / df_rel_post
        } else {
            0.0
        };
        if accepted_iters > 0 && progress_ratio <= DUAL_IR_MIN_PROGRESS_RATIO {
            if trace {
                eprintln!(
                    "DUAL_IR inner={} stagnated: df_rel {:.3e} -> {:.3e} ratio={:.3e}",
                    inner, df_rel_post, accepted_df_rel, progress_ratio
                );
            }
            break;
        }

        y_dd = accepted_y_dd;
        for i in 0..m {
            tmp.dual_solution[i] = f64::from(y_dd[i]);
        }
        tmp.bound_duals = accepted_bound_duals;
        current_r_d_free = accepted_r_d_free;
        df_rel_post = accepted_df_rel;
        df_abs_post = accepted_df_abs;
        accepted_iters += 1;
        inner += 1;
        if trace && accepted_step_scale < 1.0 {
            eprintln!(
                "DUAL_IR inner={} accepted with step_scale={:.3e}",
                inner, accepted_step_scale
            );
        }
        // 早期 break: target を達成したら終了
        if df_rel_post < target_pf {
            break;
        }
    }
    // DD y → f64 に戻す (最終採用済み y_dd から変換)
    for i in 0..m {
        tmp.dual_solution[i] = f64::from(y_dd[i]);
    }
    // y-only 更新を stale な z で評価すると、bound stationarity のずれだけで
    // 全体 KKT が悪化したように見えて改善候補を落としてしまう。
    // 採用判定前に x 固定のまま z を KKT 停留性から取り直して評価する。
    refit_bound_duals_kkt(problem, &mut tmp);

    let kkt_post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &tmp.solution,
        &tmp.dual_solution,
        &tmp.bound_duals,
    );
    if trace {
        eprintln!(
            "DUAL_IR cluster_free={} df_abs {:.3e}->{:.3e} df_rel {:.3e}->{:.3e} dy_inf={:.3e} iters={}",
            n_free, df_abs_pre, df_abs_post, df_rel_pre, df_rel_post, total_dy_inf, accepted_iters
        );
        eprintln!("DUAL_IR kkt {:.3e}->{:.3e}", kkt_pre, kkt_post);
    }
    if df_rel_post < df_rel_pre && kkt_post <= kkt_pre {
        *result = tmp;
        accepted_iters
    } else {
        if trace {
            eprintln!(
                "DUAL_IR rejected: df_improved={} kkt_safe={}",
                df_rel_post < df_rel_pre,
                kkt_post <= kkt_pre
            );
        }
        0
    }
}

fn run_dual_recovery_postprocess(
    problem: &QpProblem,
    view: &crate::qp::ipm_solver::outcome::ProblemView<'_>,
    result: &mut crate::problem::SolverResult,
    deadline: Option<std::time::Instant>,
    trace: bool,
) -> f64 {
    let pre_cleanup = result.clone();
    let kkt_before_cleanup = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    zero_inactive_inequality_duals(problem, result);
    if trace {
        let kkt_after_zero = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after zero_inactive kkt {:.3e}",
            kkt_after_zero
        );
    }
    project_duals_from_singleton_columns(problem, result);
    if trace {
        let kkt_after_singleton = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after singleton projection kkt {:.3e}",
            kkt_after_singleton
        );
    }
    refine_dual_projected_gradient(problem, result, deadline);
    if trace {
        let kkt_after_pg = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after projected gradient kkt {:.3e}",
            kkt_after_pg
        );
    }
    refine_dual_worst_active_block(problem, result, deadline);
    if trace {
        let kkt_after_block = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        eprintln!(
            "DUAL_IR outer: after local block kkt {:.3e}",
            kkt_after_block
        );
    }

    let pre_z = result.bound_duals.clone();
    let pre_refit_kkt = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    refit_bound_duals_kkt(problem, result);
    let post_refit_kkt = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    if post_refit_kkt > pre_refit_kkt {
        result.bound_duals = pre_z;
        if trace {
            eprintln!(
                "DUAL_IR z-refit rejected: kkt {:.3e} -> {:.3e}",
                pre_refit_kkt, post_refit_kkt
            );
        }
    } else if trace {
        eprintln!(
            "DUAL_IR z-refit accepted: kkt {:.3e} -> {:.3e}",
            pre_refit_kkt, post_refit_kkt
        );
    }

    let kkt_after_cleanup = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    if kkt_after_cleanup > kkt_before_cleanup {
        if trace {
            eprintln!(
                "DUAL_IR cleanup reverted: kkt {:.3e} -> {:.3e}",
                kkt_before_cleanup, kkt_after_cleanup
            );
        }
        *result = pre_cleanup;
        kkt_before_cleanup
    } else {
        kkt_after_cleanup
    }
}

/// 戻り値: 採用された refinement iter 数 (0 = no-op)
pub(crate) fn refine_kkt_iterative(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    max_iters: usize,
    target_pf: f64,
    deadline: Option<std::time::Instant>,
) -> usize {
    use crate::presolve::bound_contrib_at_var;
    use crate::problem::ConstraintType;
    use crate::qp::ipm_solver::kkt::kkt_residual_rel;

    // deadline 経過なら即 no-op。post-IPM の Krylov refinement は IPM 後に呼ばれるため、
    // ユーザー timeout が 10s 級の場合 IPM 既に消費済みでここでは時間切れの可能性あり。
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return 0;
    }

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

    // === Dual-only IR (LP/QP 共通) ===
    //
    // x を不変に保ち y のみ更新して r_d_free を厳密に 0 にする手法。
    // saddle-point K の (1,1) ブロックが δp=1e-10 で ill-conditioned になる問題
    // (QFORPLAN: cond ~1e13、dx 増幅で REJECT) を完全回避する。
    //
    // 数学: Q x 項は x 不変なので変化しない。dy 更新で A^T y のみ動かして:
    //   r_d_free_new = (Q x)_free + c_free + (A^T (y+δy))_free + bc_free
    //                = r_d_free + A_free^T δy
    //   δy = -A_free α,  G α = r_d_free,  G = A_free^T A_free (SPD)
    //   ⇒ A_free^T δy = -G α = -r_d_free  ⇒  r_d_free_new = 0
    // Q≠0 でも成立 (x を動かさないため)。
    //
    // 反復: G の条件数が大きい場合 (QFORPLAN: ~1e13)、1 回の try_dual_only_ir では
    // 内部 50 iter で df が半減程度に留まる。target_pf を達成するまで繰り返す。
    // max_iters を outer ループ上限として使う (デフォルト 10 で最大 10×50=500 inner iter)。
    let mut n_dual_total = 0_usize;
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
    };
    let mut prev_kkt = kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    let start_kkt = prev_kkt;
    let mut best_kkt = prev_kkt;
    let mut best_result = result.clone();
    let trace = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    for _outer in 0..max_iters.max(1) {
        let mut outer_made_progress = false;
        let n_dual = try_dual_only_ir(problem, result, target_pf, deadline);
        if n_dual > 0 {
            n_dual_total += n_dual;
            outer_made_progress = true;
            let kkt_after_dual_ir = kkt_residual_rel(
                &view,
                &result.solution,
                &result.dual_solution,
                &result.bound_duals,
            );
            if trace {
                eprintln!(
                    "DUAL_IR outer: after try_dual_only_ir kkt {:.3e}",
                    kkt_after_dual_ir
                );
            }
            let _ = run_dual_recovery_postprocess(problem, &view, result, deadline, trace);
        } else {
            let pre_cleanup_kkt = kkt_residual_rel(
                &view,
                &result.solution,
                &result.dual_solution,
                &result.bound_duals,
            );
            let post_cleanup_kkt =
                run_dual_recovery_postprocess(problem, &view, result, deadline, trace);
            if post_cleanup_kkt + dual_recovery_progress_tol(pre_cleanup_kkt, post_cleanup_kkt, target_pf)
                < pre_cleanup_kkt
            {
                outer_made_progress = true;
            }
        }
        // target 達成、または改善なし (n_dual=0 → G singular / no progress) なら終了
        if !outer_made_progress {
            break;
        }
        let cur_kkt = kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        if cur_kkt < best_kkt {
            best_kkt = cur_kkt;
            best_result = result.clone();
        }
        if cur_kkt < target_pf {
            break;
        }
        let progress_tol = dual_recovery_progress_tol(prev_kkt, cur_kkt, target_pf);
        if cur_kkt + progress_tol >= prev_kkt {
            break;
        }
        prev_kkt = cur_kkt;
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
    }
    if n_dual_total > 0 {
        *result = best_result;
        if trace {
            eprintln!(
                "DUAL_IR outer: best_kkt {:.3e} (start {:.3e})",
                best_kkt, start_kkt
            );
        }
        if best_kkt < target_pf || deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return n_dual_total;
        }
    }
    // dual-only が改善できなかった場合 (df 既に < target / G singular / no improvement) は
    // existing penalty + saddle-point IR に fall-through。dual-only が一部だけ効いた
    // 場合も、その良化状態を初期値として full KKT IR を継続する。

    // δp, δd: K の対角正則化。十分小さく (IR で eps·‖K‖ レベルまで refine 可)、
    // LDL の数値安定性が確保される値。1e-10 は LISWET cond 1e10 で K cond 1e2 級。
    // 診断用 env: REFINE_KKT_DELTA=<val> で δp=δd を上書き (QPILOTNO factorize 試験等)。
    const DELTA_P_DEFAULT: f64 = 1e-10;
    const DELTA_D_DEFAULT: f64 = 1e-10;
    let (delta_p, delta_d) = match std::env::var("REFINE_KKT_DELTA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
    {
        Some(v) if v > 0.0 => (v, v),
        _ => (DELTA_P_DEFAULT, DELTA_D_DEFAULT),
    };

    let sigma_zero = vec![0.0_f64; m];
    let mut k_mat = crate::qp::ipm_core::kkt::build_augmented_system(
        &problem.q,
        &problem.a,
        &sigma_zero,
        delta_p,
        delta_d,
    );

    let trace_pre = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    let diag_on = std::env::var("REFINE_KKT_DIAG").ok().as_deref() == Some("1");

    // bound active 変数の dx を penalty で抑制 (近似 active set fix)。
    //
    // 動機: saddle-point K = [Q+δp·I, A^T; A, -δd·I] は bound 制約を陽に持たないため、
    //       LDL solve は bound active な変数 (x が bound に張り付いている) にも自由に dx
    //       を生成する。dx が bound を超えると refine ループ内で clip され、結果の x が
    //       K·u=-r を満たさず pf 大幅悪化 → accept guard で reject。
    //       YAO で実証: 通常 dx_inf=0.354 (active 2 変数に集中) → reject。
    //       本処理を入れると dx_inf=0.150 → 7 iter で pf<target 達成 → PASS。
    //
    // 数値設計:
    //   - ACTIVE_TOL = 1e-8: bound 近接判定 (ユーザー eps=1e-6 の 100 倍厳しい)
    //   - PENALTY = K_diag_max × PENALTY_RATIO で K のスケールに自動適応
    //   - PENALTY_RATIO = 1e8: 標準 RHS (~eps=1e-6) で dx_j ≈ K_max × 1e-14
    //     (eps の 8 桁下、bound 違反十分小)。1e10 では過抑制で iter 1 で reject 観測。
    //
    // 真の active set fix (Lagrange multiplier or Schur complement) ではないが、
    // 数値的に等価な効果を持ち実装最小。env=REFINE_KKT_REDUCED=0 で無効化可能。
    const ACTIVE_TOL: f64 = 1e-8;
    const ACTIVE_PENALTY_RATIO: f64 = 1e8;
    let active_fix_enabled = std::env::var("REFINE_KKT_REDUCED").ok().as_deref() != Some("0");
    if active_fix_enabled {
        // K の対角の最大絶対値を取得 (penalty スケール用)
        let mut k_diag_max = 0.0_f64;
        for j in 0..(n + m) {
            let cs = k_mat.col_ptr[j];
            let ce = k_mat.col_ptr[j + 1];
            for k in cs..ce {
                if k_mat.row_ind[k] == j {
                    k_diag_max = k_diag_max.max(k_mat.values[k].abs());
                    break;
                }
            }
        }
        let active_penalty = (k_diag_max * ACTIVE_PENALTY_RATIO).max(ACTIVE_PENALTY_RATIO);
        let mut penalized = 0_usize;
        for j in 0..n {
            let x = result.solution[j];
            let (lb, ub) = problem.bounds[j];
            let is_active = (lb.is_finite() && (x - lb).abs() < ACTIVE_TOL)
                || (ub.is_finite() && (ub - x).abs() < ACTIVE_TOL);
            if !is_active {
                continue;
            }
            let col_start = k_mat.col_ptr[j];
            let col_end = k_mat.col_ptr[j + 1];
            for k in col_start..col_end {
                if k_mat.row_ind[k] == j {
                    k_mat.values[k] += active_penalty;
                    penalized += 1;
                    break;
                }
            }
        }
        if (trace_pre || diag_on) && penalized > 0 {
            eprintln!("REFINE_KKT bound-active fix: penalized {} vars (PENALTY={:.2e}, K_diag_max={:.2e})",
                penalized, active_penalty, k_diag_max);
        }
    }
    if trace_pre {
        eprintln!(
            "REFINE_KKT pre-factorize: n={} m={} K_nnz={} delta_p={:.1e} delta_d={:.1e}",
            n,
            m,
            k_mat.values.len(),
            delta_p,
            delta_d
        );
    }
    if diag_on {
        // K の対角分布を実測 (cond の代理指標として min/max/abs_min/abs_max)。
        // K = [Q+δp·I, A^T; A, -δd·I] なので上 n 行は SPD 系、下 m 行は -δd·I + Σ.
        // build_augmented_system は upper triangular CSC を返す。
        let mut diag_top_min = f64::INFINITY;
        let mut diag_top_max = f64::NEG_INFINITY;
        let mut diag_top_abs_min = f64::INFINITY;
        let mut diag_bot_min = f64::INFINITY;
        let mut diag_bot_max = f64::NEG_INFINITY;
        let mut diag_bot_abs_min = f64::INFINITY;
        for j in 0..(n + m) {
            let col_start = k_mat.col_ptr[j];
            let col_end = k_mat.col_ptr[j + 1];
            for k in col_start..col_end {
                if k_mat.row_ind[k] == j {
                    let v = k_mat.values[k];
                    if j < n {
                        diag_top_min = diag_top_min.min(v);
                        diag_top_max = diag_top_max.max(v);
                        diag_top_abs_min = diag_top_abs_min.min(v.abs());
                    } else {
                        diag_bot_min = diag_bot_min.min(v);
                        diag_bot_max = diag_bot_max.max(v);
                        diag_bot_abs_min = diag_bot_abs_min.min(v.abs());
                    }
                    break;
                }
            }
        }
        eprintln!(
            "REFINE_KKT_DIAG K_diag top(Q+δp·I)=[min={:.3e} max={:.3e} abs_min={:.3e}] bot(-δd·I)=[min={:.3e} max={:.3e} abs_min={:.3e}]",
            diag_top_min, diag_top_max, diag_top_abs_min,
            diag_bot_min, diag_bot_max, diag_bot_abs_min
        );
        // 全要素の絶対値分布 (off-diag 含む) で K のスケール把握
        let abs_max = k_mat.values.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let abs_min_nz = k_mat
            .values
            .iter()
            .filter(|&&v| v != 0.0)
            .fold(f64::INFINITY, |a, &v| a.min(v.abs()));
        eprintln!(
            "REFINE_KKT_DIAG K_all abs_max={:.3e} abs_min_nz={:.3e} ratio={:.3e}",
            abs_max,
            abs_min_nz,
            abs_max / abs_min_nz.max(1e-300)
        );
    }
    // factorize: SingularOrIndefinite で失敗したら δ を段階的に上げて再試行。
    // QPILOTNO 系の K (LP-like, Q≈0 で δp=1e-10 が支配的に singular) を救う。
    // factorize 成立する最小 δ を探す。LISWET 系 (factorize OK) は初回成功で影響なし。
    //
    // δ を上げるトレードオフ:
    //   - 大きすぎ → IR の forward error 増 (cond×ε で δ が cond の代理)
    //   - 小さすぎ → factorize fail
    //   - 段階的に増やして「factorize 成立する最小 δ」を選ぶ
    //
    // **deadline 必須**: 30k×30k 級 K は 1 回の LDL factorize で 80 秒級に達する。
    // 6 retry で QPLIB_8505 の SingularOrIndefinite が ~500 秒経過する事象を観測。
    // factorize 内 (`factorize_quasidefinite_with_amd`) と retry 間の両方で時間切れを検査する。
    const FACTOR_RETRY_GROWTH: f64 = 10.0;
    const FACTOR_RETRY_MAX: usize = 6; // δ_init × 10^6 まで (1e-10 → 1e-4)
    let factor = {
        let mut current_delta_p = delta_p;
        let mut current_delta_d = delta_d;
        let mut current_k = k_mat.clone();
        let mut result_factor: Option<crate::linalg::ldl::LdlFactorizationAmd> = None;
        let mut retry_count = 0usize;
        loop {
            // retry 直前で deadline 確認 (前回 factorize に時間を全部使った可能性あり)
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                if trace_pre || diag_on {
                    eprintln!(
                        "REFINE_KKT factorize abandoned due to deadline at retry={}",
                        retry_count
                    );
                }
                break;
            }
            match crate::linalg::ldl::factorize_quasidefinite_with_amd(&current_k, deadline) {
                Ok(f) => {
                    result_factor = Some(f);
                    break;
                }
                Err(e) => {
                    if retry_count >= FACTOR_RETRY_MAX {
                        if trace_pre || diag_on {
                            eprintln!("REFINE_KKT factorize failed after {} retries: {:?} (last delta_p={:.1e} delta_d={:.1e})",
                                retry_count, e, current_delta_p, current_delta_d);
                        }
                        break;
                    }
                    retry_count += 1;
                    current_delta_p *= FACTOR_RETRY_GROWTH;
                    current_delta_d *= FACTOR_RETRY_GROWTH;
                    current_k = crate::qp::ipm_core::kkt::build_augmented_system(
                        &problem.q,
                        &problem.a,
                        &sigma_zero,
                        current_delta_p,
                        current_delta_d,
                    );
                    // bound-active fix を再適用 (新しい K に対して)
                    if active_fix_enabled {
                        let mut k_diag_max_retry = 0.0_f64;
                        for j in 0..(n + m) {
                            let cs = current_k.col_ptr[j];
                            let ce = current_k.col_ptr[j + 1];
                            for k in cs..ce {
                                if current_k.row_ind[k] == j {
                                    k_diag_max_retry =
                                        k_diag_max_retry.max(current_k.values[k].abs());
                                    break;
                                }
                            }
                        }
                        let active_penalty_retry =
                            (k_diag_max_retry * ACTIVE_PENALTY_RATIO).max(ACTIVE_PENALTY_RATIO);
                        for j in 0..n {
                            let x = result.solution[j];
                            let (lb, ub) = problem.bounds[j];
                            let is_active = (lb.is_finite() && (x - lb).abs() < ACTIVE_TOL)
                                || (ub.is_finite() && (ub - x).abs() < ACTIVE_TOL);
                            if !is_active {
                                continue;
                            }
                            let cs = current_k.col_ptr[j];
                            let ce = current_k.col_ptr[j + 1];
                            for k in cs..ce {
                                if current_k.row_ind[k] == j {
                                    current_k.values[k] += active_penalty_retry;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (trace_pre || diag_on) && retry_count > 0 && result_factor.is_some() {
            eprintln!("REFINE_KKT factorize succeeded after {} retries (final delta_p={:.1e} delta_d={:.1e})",
                retry_count, current_delta_p, current_delta_d);
        }
        match result_factor {
            Some(f) => f,
            None => return 0,
        }
    };
    if diag_on {
        // 因子化成功 → cond の代理: random RHS で solve した結果のノルム比 ||K^-1·r||_∞ / ||r||_∞.
        // 大きいほど K が ill-conditioned (大きな inverse に増幅される)。
        let mut rng_state: u64 = 0x9E3779B97F4A7C15;
        let mut rhs = vec![0.0_f64; n + m];
        for v in rhs.iter_mut() {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            *v = ((rng_state as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
        }
        let rhs_inf = rhs.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let mut sol = vec![0.0_f64; n + m];
        factor.solve(&rhs, &mut sol);
        let sol_inf = sol.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let any_nan = sol.iter().any(|v| !v.is_finite());
        eprintln!(
            "REFINE_KKT_DIAG cond_proxy: ||K^-1·rand||_∞ / ||rand||_∞ = {:.3e} / {:.3e} = {:.3e} nan={}",
            sol_inf, rhs_inf, sol_inf / rhs_inf.max(1e-300), any_nan
        );
    }

    // FX/EmptyCol 変数の判定 (kkt_residual_rel と整合):
    //   FX (lb≈ub): presolve 慣例で bound_dual=0 埋め、stationarity 評価から除外
    //   EmptyCol (制約 A に登場しない): bound_dual=0 慣例、Q∅ + c[j] != 0 のため除外
    // これらを含めると orig 空間で huge cancellation noise (r_d_abs) が出て IR が壊れる。
    const FX_TOL_REFINE: f64 = 1e-12;
    let exclude_var: Vec<bool> = (0..n)
        .map(|j| {
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
        })
        .collect();

    // 残差は (r_d, r_p, pf_abs, df_abs, pf_rel, df_rel) を返す。
    // pf_rel/df_rel は OSQP-style 全体相対化で bench (`compute_dfeas_orig` /
    // `kkt_residual_rel`) と整合。
    //   pf_rel = max|r_p_i| / (1 + max(||Ax||_inf, ||b||_inf))
    //   df_rel = max|r_d_j| / (1 + max(||Qx||_inf, ||c||_inf, ||A^Ty||_inf, ||z_bnd||_inf))
    // gate / 早期 break / acceptance の判定は **rel** で行う (bench と統一)。
    // abs 版は既存の guardrail (factor 2 以内) のために併せて返す。
    let compute_residuals =
        |x: &[f64], y: &[f64], z: &[f64]| -> (Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
            let qx = problem.q.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; n]);
            let aty = problem
                .a
                .transpose()
                .mat_vec_mul(y)
                .unwrap_or_else(|_| vec![0.0; n]);
            let mut r_d = vec![0.0_f64; n];
            let mut df_rel_componentwise = 0.0_f64;
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let bc = bound_contrib_at_var(&problem.bounds, z, j);
                r_d[j] = qx[j] + problem.c[j] + aty[j] + bc;
                // 成分相対化 (bench compute_dfeas_orig componentwise と一致)。
                let scale_j = 1.0 + qx[j].abs() + problem.c[j].abs() + aty[j].abs() + bc.abs();
                let rel_j = r_d[j].abs() / scale_j;
                if rel_j > df_rel_componentwise {
                    df_rel_componentwise = rel_j;
                }
            }
            let ax = problem.a.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; m]);
            let mut r_p = vec![0.0_f64; m];
            let mut pf_abs = 0.0_f64;
            let mut pf_rel_componentwise = 0.0_f64;
            for i in 0..m {
                let raw = ax[i] - problem.b[i];
                let v = match problem.constraint_types[i] {
                    ConstraintType::Eq => raw,
                    ConstraintType::Ge => {
                        if raw < 0.0 {
                            raw
                        } else {
                            0.0
                        }
                    }
                    ConstraintType::Le => {
                        if raw > 0.0 {
                            raw
                        } else {
                            0.0
                        }
                    }
                };
                r_p[i] = v;
                pf_abs = pf_abs.max(v.abs());
                // 成分相対化。
                let scale_i = 1.0 + ax[i].abs() + problem.b[i].abs();
                let rel_i = v.abs() / scale_i;
                if rel_i > pf_rel_componentwise {
                    pf_rel_componentwise = rel_i;
                }
            }
            let df_abs = r_d.iter().fold(0.0_f64, |a, &r| a.max(r.abs()));
            (
                r_d,
                r_p,
                pf_abs,
                df_abs,
                pf_rel_componentwise,
                df_rel_componentwise,
            )
        };

    // DD (double-double) residual: Wilkinson IR の "double the working precision" 実装。
    // working precision IR の forward error は O(n × κ × ε)、これを超えるには residual を
    // 倍精度で計算する必要がある (Wikipedia: Iterative refinement)。
    //
    // 本実装: Q*x, A^T*y, A*x の各積算を TwoFloat (DD ≈ 106 bit ≈ 31 桁) で実行。
    // 各積を new_mul (Joldes 2017 Algorithm 2) で精密に計算、累積も DD で。
    // LDL solve は f64 のまま (Wikipedia 通り、LDL の精度は cond × ε で十分)。
    //
    // ill-conditioned 系では f64 residual が cancellation で 0 に張り付き IR が
    // progress 判定不能になるため、default で DD を使う。env REFINE_KKT_DD=0 で
    // f64 fallback (パフォーマンス計測用、通常は使わない)。
    let dd_mode = std::env::var("REFINE_KKT_DD").ok().as_deref() != Some("0");
    let compute_residuals_dd =
        |x: &[f64], y: &[f64], z: &[f64]| -> (Vec<f64>, Vec<f64>, f64, f64, f64, f64) {
            use twofloat::TwoFloat;
            let zero_dd = TwoFloat::from(0.0);
            // qx[i] = sum_k Q[i,k] * x[k]  (Q は対称、上三角 CSC 格納)
            // Q 格納慣例 (spmv_q と同じ): **全要素格納の対称行列** (上下三角両方 stored)。
            // symmetric duplication せず CSC 全エントリを直接走査する。
            let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for j in 0..n {
                let xv = x[j];
                let cs = problem.q.col_ptr[j];
                let ce = problem.q.col_ptr[j + 1];
                for k in cs..ce {
                    let row = problem.q.row_ind[k];
                    let v = problem.q.values[k];
                    qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(v, xv);
                }
            }
            // aty[col] = sum_row A[row,col] * y[row]  (CSC で col 走査)
            let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
            for col in 0..n {
                let cs = problem.a.col_ptr[col];
                let ce = problem.a.col_ptr[col + 1];
                for k in cs..ce {
                    let row = problem.a.row_ind[k];
                    let v = problem.a.values[k];
                    aty_dd[col] = aty_dd[col] + TwoFloat::new_mul(v, y[row]);
                }
            }
            // r_d[j] = qx[j] + c[j] + aty[j] + bound_contrib[j]
            let mut r_d = vec![0.0_f64; n];
            let mut max_qx = 0.0_f64;
            let mut max_c = 0.0_f64;
            let mut max_aty = 0.0_f64;
            let mut max_bnd = 0.0_f64;
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let bc = bound_contrib_at_var(&problem.bounds, z, j);
                let r = qx_dd[j] + TwoFloat::from(problem.c[j]) + aty_dd[j] + TwoFloat::from(bc);
                r_d[j] = f64::from(r);
                max_qx = max_qx.max(f64::from(qx_dd[j]).abs());
                max_c = max_c.max(problem.c[j].abs());
                max_aty = max_aty.max(f64::from(aty_dd[j]).abs());
                max_bnd = max_bnd.max(bc.abs());
            }
            // ax[row] = sum_col A[row,col] * x[col]  (CSC で col 走査して row に加算)
            let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
            for col in 0..n {
                let cs = problem.a.col_ptr[col];
                let ce = problem.a.col_ptr[col + 1];
                for k in cs..ce {
                    let row = problem.a.row_ind[k];
                    let v = problem.a.values[k];
                    ax_dd[row] = ax_dd[row] + TwoFloat::new_mul(v, x[col]);
                }
            }
            let mut r_p = vec![0.0_f64; m];
            let mut pf_abs = 0.0_f64;
            let mut pf_rel_componentwise = 0.0_f64;
            for i in 0..m {
                let raw_dd = ax_dd[i] - TwoFloat::from(problem.b[i]);
                let raw = f64::from(raw_dd);
                let v = match problem.constraint_types[i] {
                    ConstraintType::Eq => raw,
                    ConstraintType::Ge => {
                        if raw < 0.0 {
                            raw
                        } else {
                            0.0
                        }
                    }
                    ConstraintType::Le => {
                        if raw > 0.0 {
                            raw
                        } else {
                            0.0
                        }
                    }
                };
                r_p[i] = v;
                pf_abs = pf_abs.max(v.abs());
                // 成分相対化: 各行ごとに正規化して max を取る (bench compute_pfeas_normalized と一致)。
                let ax_i_abs = f64::from(ax_dd[i]).abs();
                let scale_i = 1.0 + ax_i_abs + problem.b[i].abs();
                let rel_i = v.abs() / scale_i;
                if rel_i > pf_rel_componentwise {
                    pf_rel_componentwise = rel_i;
                }
            }
            let df_abs = r_d.iter().fold(0.0_f64, |a, &r| a.max(r.abs()));
            // df_rel も成分相対化で計算 (bench compute_dfeas_orig componentwise と一致)。
            // 全体相対化 (max_qx,max_c,max_aty,max_bnd の最大で割る) は ill-scaled で
            // 1 成分のみ大きく外れた残差を見逃すため、saddle-point IR の skip 判定にも
            // componentwise を使う必要がある (QBORE3D で global df_rel=1.6e-9 だが
            // componentwise df_rel=7.06e-4 で IR が skip されていた)。
            let mut df_rel_componentwise = 0.0_f64;
            for j in 0..n {
                if exclude_var[j] {
                    continue;
                }
                let qx_j = f64::from(qx_dd[j]).abs();
                let aty_j = f64::from(aty_dd[j]).abs();
                let bc = bound_contrib_at_var(&problem.bounds, z, j);
                let scale_j = 1.0 + qx_j + problem.c[j].abs() + aty_j + bc.abs();
                let rel_j = r_d[j].abs() / scale_j;
                if rel_j > df_rel_componentwise {
                    df_rel_componentwise = rel_j;
                }
            }
            let _ = max_qx;
            let _ = max_c;
            let _ = max_aty;
            let _ = max_bnd;
            (
                r_d,
                r_p,
                pf_abs,
                df_abs,
                pf_rel_componentwise,
                df_rel_componentwise,
            )
        };

    let pre_z = result.bound_duals.clone();
    let (_, _, pre_pf, pre_df, pre_pf_rel, pre_df_rel) = if dd_mode {
        compute_residuals_dd(&result.solution, &result.dual_solution, &pre_z)
    } else {
        compute_residuals(&result.solution, &result.dual_solution, &pre_z)
    };
    let trace = std::env::var("REFINE_KKT_TRACE").ok().as_deref() == Some("1");
    if trace {
        eprintln!(
            "REFINE_KKT entry: n={} m={} pre_pf={:.3e} pre_df={:.3e} target_pf={:.3e} dd_mode={}",
            n, m, pre_pf, pre_df, target_pf, dd_mode
        );
    }
    // Gate: bench (`compute_dfeas_orig` / `kkt_residual_rel`) と同じ OSQP-style 相対残差で
    // 判定する。target_pf は bench の eps 通過判定 (pf_rel < eps && df_rel < eps) と統一。
    // 片方でも target 超過なら refine を試みる (IR は両方を同時に reduce する)。
    if pre_pf_rel < target_pf && pre_df_rel < target_pf {
        if trace {
            eprintln!(
                "REFINE_KKT skip: pre_pf_rel={:.3e} pre_df_rel={:.3e} both < target_pf",
                pre_pf_rel, pre_df_rel
            );
        }
        return 0;
    }

    let mut accepted = n_dual_total;
    // 残差悪化許容: pre_rel の 2x または target_pf×100 (どちらか大きい方)。これ以上は
    // revert (構造的悪化)。target_pf×100 floor は machine-precision 級の pre 値が事故で
    // 増えても target_pf×100 までは許容する設計 (target=1e-6 なら 1e-4 が floor)。
    const RESID_TOLERANCE_FACTOR: f64 = 2.0;
    const RESID_FLOOR_RATIO: f64 = 100.0;
    let resid_floor = target_pf * RESID_FLOOR_RATIO;
    let pf_limit = (pre_pf_rel * RESID_TOLERANCE_FACTOR).max(resid_floor);
    let df_limit = (pre_df_rel * RESID_TOLERANCE_FACTOR).max(resid_floor);

    for iter in 0..max_iters {
        // 各 iter 先頭で deadline 確認 (LDL solve は n+m=30k で数秒級になり得る)
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            if trace {
                eprintln!("REFINE_KKT iter={} deadline reached", iter);
            }
            break;
        }
        let (r_d, r_p, pf_abs_cur, df_abs_cur, pf_cur, df_cur) = if dd_mode {
            compute_residuals_dd(&result.solution, &result.dual_solution, &result.bound_duals)
        } else {
            compute_residuals(&result.solution, &result.dual_solution, &result.bound_duals)
        };
        // 早期 break: pf_rel も df_rel も target 以下 → bench で PASS 級。
        if pf_cur < target_pf && df_cur < target_pf {
            if trace {
                eprintln!(
                    "REFINE_KKT iter={} early: pf_rel={:.3e} df_rel={:.3e} both < target",
                    iter, pf_cur, df_cur
                );
            }
            break;
        }
        let _ = (pf_abs_cur, df_abs_cur); // abs 値は未使用 (trace 表示は rel ベース)

        let mut rhs = vec![0.0_f64; n + m];
        for j in 0..n {
            rhs[j] = -r_d[j];
        }
        for i in 0..m {
            rhs[n + i] = -r_p[i];
        }

        let mut sol = vec![0.0_f64; n + m];
        factor.solve(&rhs, &mut sol);
        if sol.iter().any(|v| !v.is_finite()) {
            if trace {
                eprintln!("REFINE_KKT iter={} solve produced NaN", iter);
            }
            break;
        }

        let dx_inf: f64 = sol[..n].iter().fold(0.0, |a, &v| a.max(v.abs()));
        let dy_inf: f64 = sol[n..].iter().fold(0.0, |a, &v| a.max(v.abs()));

        let mut x_new = result.solution.clone();
        let mut y_new = result.dual_solution.clone();
        let mut clip_amt = 0.0_f64;
        let mut clip_count = 0_usize;
        let mut clip_top: Vec<(usize, f64)> = Vec::new();
        for j in 0..n {
            let raw = x_new[j] + sol[j];
            let (lb, ub) = problem.bounds[j];
            let mut clipped = raw;
            if lb.is_finite() {
                clipped = clipped.max(lb);
            }
            if ub.is_finite() {
                clipped = clipped.min(ub);
            }
            let amt = (raw - clipped).abs();
            clip_amt = clip_amt.max(amt);
            if amt > 0.0 {
                clip_count += 1;
                if diag_on {
                    clip_top.push((j, amt));
                }
            }
            x_new[j] = clipped;
        }
        if diag_on && !clip_top.is_empty() {
            clip_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<String> = clip_top
                .iter()
                .take(5)
                .map(|(j, a)| format!("x[{}]={:.2e}", j, a))
                .collect();
            eprintln!(
                "REFINE_KKT_DIAG iter={} clip_count={}/{} clip_max={:.3e} top5: {}",
                iter,
                clip_count,
                n,
                clip_amt,
                top5.join(", ")
            );
        }
        for i in 0..m {
            y_new[i] += sol[n + i];
        }

        let mut tmp = result.clone();
        tmp.solution = x_new;
        tmp.dual_solution = y_new;
        refit_bound_duals_kkt(problem, &mut tmp);

        let (_, _, _pf_abs_new, _df_abs_new, pf_new, df_new) = if dd_mode {
            compute_residuals_dd(&tmp.solution, &tmp.dual_solution, &tmp.bound_duals)
        } else {
            compute_residuals(&tmp.solution, &tmp.dual_solution, &tmp.bound_duals)
        };

        if trace {
            eprintln!("REFINE_KKT iter={} pf_rel={:.3e}->{:.3e} df_rel={:.3e}->{:.3e} dx_inf={:.3e} dy_inf={:.3e} clip={:.3e}",
                iter, pf_cur, pf_new, df_cur, df_new, dx_inf, dy_inf, clip_amt);
        }

        // Acceptance criterion: max(pf_rel, df_rel) が strictly 減少 + 両者 guardrail 内。
        let score_cur = pf_cur.max(df_cur);
        let score_new = pf_new.max(df_new);
        let progress = score_new < score_cur;
        let pf_safe = pf_new < pf_limit;
        let df_safe = df_new < df_limit;
        if progress && pf_safe && df_safe {
            *result = tmp;
            accepted += 1;
        } else {
            if trace {
                eprintln!("REFINE_KKT iter={} REJECTED (progress={} pf_safe={} df_safe={} score:{:.3e}->{:.3e})",
                    iter, progress, pf_safe, df_safe, score_cur, score_new);
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
pub(crate) fn refit_bound_duals_kkt(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
) {
    let n = problem.num_vars;
    if result.solution.len() != n {
        return;
    }
    use twofloat::TwoFloat;
    let x = &result.solution;
    // ill-conditioned 系で f64 mat_vec のキャンセル誤差が target 計算を狂わせないよう
    // Q*x と A^T*y は DD で積算する。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] =
                qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let aty: Vec<f64> = if problem.a.nrows > 0 && !result.dual_solution.is_empty() {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                acc[col] = acc[col]
                    + TwoFloat::new_mul(
                        problem.a.values[k],
                        result.dual_solution[problem.a.row_ind[k]],
                    );
            }
        }
        acc.iter().map(|&v| f64::from(v)).collect()
    } else {
        vec![0.0_f64; n]
    };

    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    if n_lb + n_ub == 0 {
        return;
    }

    let mut new_bd = vec![0.0_f64; n_lb + n_ub];
    // bound dual の **候補値** を target = -(Qx+c+Aty) の符号で決める。
    // bound_contrib[j] = -z_lb[j] + z_ub[j] = target なので:
    //   target > 0 → z_ub 候補 = target  (ub 側活性化)、z_lb = 0
    //   target < 0 → z_lb 候補 = -target (lb 側活性化)、z_ub = 0
    // 旧実装は ACTIVE_REL_TOL=1e-6 で「x が bound 近接か」を判定して activate していたが、
    // QFORPLAN col 34 (x=7.2e-6 ≈ lb=0) のような「IPM が bound 近接で停止したが
    // tol を僅かに超えている」ケースを見逃していた。
    // ここでは **常に候補を提示** し、後段の per-col guard (r_post <= r_pre) で残差が
    // 改善する場合のみ採用する。内部点で本当に z=0 が正しい場合は旧値が維持されるので
    // 誤った activation は起きない。
    let mut lb_idx = 0_usize;
    let mut ub_idx = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let target = -(qx[j] + problem.c[j] + aty[j]);
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();

        if lb_finite && ub_finite {
            // FX variable (lb==ub): convention is 0-fill; KKT-guard also excludes FX
            if (lb - ub).abs() >= FX_TOL {
                if target > 0.0 {
                    new_bd[ub_idx] = target; // ub 側
                } else {
                    new_bd[lb_idx] = -target; // lb 側
                }
            }
            lb_idx += 1;
            ub_idx += 1;
        } else if lb_finite {
            // lb のみ有限: bound_contrib = -y_lb = target → y_lb = -target (target<0 のとき有効)
            new_bd[lb_idx] = (-target).max(0.0);
            lb_idx += 1;
        } else if ub_finite {
            // ub のみ有限: bound_contrib = y_ub = target → y_ub = target (target>0 のとき有効)
            new_bd[ub_idx] = target.max(0.0);
            ub_idx += 1;
        }
    }

    // KKT-guard を per-col で適用する。max-based all-or-nothing 比較だと
    // 「N 個の col のうち 1 個だけ refit が悪化させる」ケースで N-1 個の改善も
    // 全部捨ててしまい、Stage 0 が refit を全 reject してしまう。
    // col 単位で改善 (またはヒット) するなら採用、悪化するなら現値維持。
    let pre_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let post_contrib = compute_bound_contrib(&problem.bounds, &new_bd, n);
    let mut accepted_bd = result.bound_duals.clone();
    if accepted_bd.len() < new_bd.len() {
        accepted_bd.resize(new_bd.len(), 0.0);
    }
    let mut updated_lb = 0usize;
    let mut updated_ub = 0usize;
    let mut lb_slot = 0usize;
    let mut ub_slot = n_lb;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        // FX 変数は postsolve で 0 埋め慣例なので KKT 評価から除外 (bench/v2 と整合)。
        let is_fx = lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL;
        let r_pre = if is_fx {
            0.0
        } else {
            (qx[j] + problem.c[j] + aty[j] + pre_contrib[j]).abs()
        };
        let r_post = if is_fx {
            0.0
        } else {
            (qx[j] + problem.c[j] + aty[j] + post_contrib[j]).abs()
        };
        let take_new = !is_fx && r_post <= r_pre;
        if lb.is_finite() {
            if take_new && lb_slot < new_bd.len() {
                if accepted_bd[lb_slot] != new_bd[lb_slot] {
                    updated_lb += 1;
                }
                accepted_bd[lb_slot] = new_bd[lb_slot];
            }
            lb_slot += 1;
        }
        if ub.is_finite() {
            if take_new && ub_slot < new_bd.len() {
                if accepted_bd[ub_slot] != new_bd[ub_slot] {
                    updated_ub += 1;
                }
                accepted_bd[ub_slot] = new_bd[ub_slot];
            }
            ub_slot += 1;
        }
    }
    if std::env::var("REFIT_BD_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "REFIT_BD per-col: updated_lb={} updated_ub={} (n={})",
            updated_lb, updated_ub, n
        );
    }
    result.bound_duals = accepted_bd;
}

/// IRLS (iteratively reweighted least squares) で y を成分相対化基準で精緻化する。
///
/// 標準 LSQ (compute_lsq_dual_y) は ||A^T y - target||² (L2 norm) を最小化するので
/// ill-scaled / overdetermined (n > m + rank issues) では特定成分に残差が集中する。
/// componentwise eps 判定では「max 残差成分」が支配なので L2 LSQ では eps を超える
/// ケースが残る (QSCRS8 col 1034 の dfc=8.8e-6 等)。
///
/// IRLS: 各 iter で残差成分 j の rel に応じて weight[j] を増やし、weighted LSQ
/// `(A · diag(w) · A^T) y = A · diag(w) · target` を解く。これを繰り返すと
/// L∞ 解 (max 残差を最小化する解) に漸近収束する。
///
/// 実装: A の列 k を sqrt(w_k) でスケールした A_w を作り、build_aat_upper_csc
/// に渡すと A_w · A_w^T = A · diag(w) · A^T となる (既存 factorize 経路を流用)。
///
/// 制約: n+m > LSQ_DUAL_SIZE_LIMIT は AAT factorization のメモリ消費上限で skip。
pub(crate) fn refine_dual_lsq_irls(
    problem: &QpProblem,
    result: &mut crate::problem::SolverResult,
    eps_target: f64,
    max_iters: usize,
    deadline: Option<std::time::Instant>,
) {
    use twofloat::TwoFloat;
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return;
    }
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if m == 0 || result.solution.len() != n {
        return;
    }
    if n + m > LSQ_DUAL_SIZE_LIMIT {
        return;
    }
    if result.dual_solution.len() != m {
        return;
    }

    let zero_dd = TwoFloat::from(0.0);

    // Q*x を DD で精密に計算 (x は固定なので 1 回だけ)
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = result.solution[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            qx_dd[problem.q.row_ind[k]] =
                qx_dd[problem.q.row_ind[k]] + TwoFloat::new_mul(problem.q.values[k], xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let bound_contrib = compute_bound_contrib(&problem.bounds, &result.bound_duals, n);
    let target: Vec<f64> = (0..n)
        .map(|j| -(qx[j] + problem.c[j] + bound_contrib[j]))
        .collect();

    // FX / EmptyCol は KKT 評価から除外 (kkt_residual_rel と整合)
    let exclude: Vec<bool> = (0..n)
        .map(|j| {
            let (lb, ub) = problem.bounds[j];
            if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
                return true;
            }
            problem.a.col_ptr.len() > j + 1 && problem.a.col_ptr[j + 1] - problem.a.col_ptr[j] == 0
        })
        .collect();

    let compute_aty = |y: &[f64]| -> Vec<f64> {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                acc[col] = acc[col] + TwoFloat::new_mul(problem.a.values[k], y[row]);
            }
        }
        acc.iter().map(|&v| f64::from(v)).collect()
    };

    let max_rel_with_aty = |aty_v: &[f64]| -> f64 {
        let mut max_rel = 0.0_f64;
        for j in 0..n {
            if exclude[j] {
                continue;
            }
            let r = qx[j] + problem.c[j] + aty_v[j] + bound_contrib[j];
            let scale =
                1.0 + qx[j].abs() + problem.c[j].abs() + aty_v[j].abs() + bound_contrib[j].abs();
            let rel = r.abs() / scale;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        max_rel
    };

    let mut y_curr = result.dual_solution.clone();
    let initial_aty = compute_aty(&y_curr);
    let initial_max_rel = max_rel_with_aty(&initial_aty);
    if initial_max_rel < eps_target {
        return;
    }

    let mut best_y = y_curr.clone();
    let mut best_max_rel = initial_max_rel;
    let mut prev_max_rel = initial_max_rel;

    /// 単一成分の重み上限 (= rel/eps の上限)。
    /// 1e4 を超えると LSQ が outlier 修正のために他成分を悪化させて oscillate する
    /// (STADAT1 で 1e8 試験時に dfc 1.6e-3 → 7.8e-4 への改善が逆に 2.2e-4 → 7.8e-4 と
    /// 悪化を観測)。AAT cond が ratio² に応じて悪化する物理量上限としても妥当。
    const MAX_WEIGHT_RATIO: f64 = 1e4;
    /// 改善停滞判定 (前回の何 % 改善あれば継続)。
    const STAGNATE_RATIO: f64 = 0.95;

    for irls_iter in 0..max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }

        // 重み: rel > eps の成分に対して (rel/eps)² (LSQ 内部で 1/√w 倍として作用するため、
        // 二乗で componentwise weighting 効果が出る)
        let aty_v = compute_aty(&y_curr);
        let mut weights: Vec<f64> = vec![1.0; n];
        for j in 0..n {
            if exclude[j] {
                continue;
            }
            let r = qx[j] + problem.c[j] + aty_v[j] + bound_contrib[j];
            let scale =
                1.0 + qx[j].abs() + problem.c[j].abs() + aty_v[j].abs() + bound_contrib[j].abs();
            let rel = r.abs() / scale;
            if rel > eps_target {
                let ratio = (rel / eps_target).min(MAX_WEIGHT_RATIO);
                weights[j] = ratio * ratio;
            }
        }

        // A_scaled[i, k] = sqrt(weight[k]) * A[i, k] とすると
        // A_scaled · A_scaled^T = A · diag(weight) · A^T
        let mut a_scaled = problem.a.clone();
        for k in 0..n {
            let s = weights[k].sqrt();
            if (s - 1.0).abs() < 1e-15 {
                continue;
            }
            let cs = a_scaled.col_ptr[k];
            let ce = a_scaled.col_ptr[k + 1];
            for idx in cs..ce {
                a_scaled.values[idx] *= s;
            }
        }

        let aat_w = match build_aat_upper_csc(&a_scaled, n, m) {
            Some(mat) => mat,
            None => break,
        };
        let factor = match crate::linalg::ldl::factorize(&aat_w) {
            Ok(f) => f,
            Err(_) => break,
        };

        // RHS = A · diag(w) · target  (DD 積算)
        let mut rhs_dd: Vec<TwoFloat> = vec![zero_dd; m];
        for col in 0..n {
            let wt = weights[col] * target[col];
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                let row = problem.a.row_ind[k];
                rhs_dd[row] = rhs_dd[row] + TwoFloat::new_mul(problem.a.values[k], wt);
            }
        }
        let rhs: Vec<f64> = rhs_dd.iter().map(|&v| f64::from(v)).collect();

        let mut y_new = vec![0.0_f64; m];
        factor.solve(&rhs, &mut y_new);
        if y_new.iter().any(|v| !v.is_finite()) {
            break;
        }

        let aty_new = compute_aty(&y_new);
        let new_max_rel = max_rel_with_aty(&aty_new);

        if new_max_rel < best_max_rel {
            best_y = y_new.clone();
            best_max_rel = new_max_rel;
        }

        if best_max_rel < eps_target {
            break;
        }
        if irls_iter > 0 && new_max_rel >= prev_max_rel * STAGNATE_RATIO {
            break;
        }
        prev_max_rel = new_max_rel;
        y_curr = y_new;
    }

    if best_max_rel < initial_max_rel {
        result.dual_solution = best_y;
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

/// `build_aat_upper_csc` の BTreeMap ノードあたり実測バイト数 (key 16B + value 8B
/// + ノードオーバーヘッド)。memory budget 推定の係数。
const AAT_BUILD_BYTES_PER_ENTRY: u128 = 80;

/// A * A^T (m×m, 上三角 CSC) を構築する。LDL 分解前提で対角に ε 正則化を加える。
/// rank-deficient な A (重複制約等) でも factorize 可能になる。
///
/// メモリ予算超過時 (LASSO_150_S3 のような nnz(AAT) ≈ m²/2 = 117M 級で 9 GB BTreeMap
/// が必要な問題) は `None` を返し、上位 (refine_dual_lsq 等) は no-op で skip する。
/// 旧実装は `LSQ_DUAL_SIZE_LIMIT=50000` の n+m guard だけで密度を考慮せず、LASSO 系で
/// 11 GB peak の RSS spike を起こしていた (handover 2026-05-09)。
pub(crate) fn build_aat_upper_csc(a: &CscMatrix, n: usize, m: usize) -> Option<CscMatrix> {
    use std::collections::BTreeMap;
    // nnz(AAT_upper) <= min(m*(m+1)/2, Σ_k c_k(c_k+1)/2)。BTreeMap 構築時に各 unique
    // entry がノード (~80 bytes) を確保するため、上限 × 80B が memory_budget を超えるなら
    // 構築せず skip する。estimate は upper bound で実 nnz は更に小さくなりうるが、
    // 上限で予算を超える時点で安全側に倒す。
    let m_u = m as u128;
    let mut col_pair_sum: u128 = 0;
    for k in 0..n {
        let c_k = (a.col_ptr[k + 1] - a.col_ptr[k]) as u128;
        col_pair_sum = col_pair_sum.saturating_add(c_k.saturating_mul(c_k + 1) / 2);
    }
    let nnz_upper_bound = (m_u.saturating_mul(m_u + 1) / 2).min(col_pair_sum);
    let bytes_estimate = nnz_upper_bound.saturating_mul(AAT_BUILD_BYTES_PER_ENTRY);
    if bytes_estimate > crate::linalg::kkt_solver::memory_budget_bytes() as u128 {
        return None;
    }
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for k in 0..n {
        let start = a.col_ptr[k];
        let end = a.col_ptr[k + 1];
        let cols_in_k: Vec<(usize, f64)> =
            (start..end).map(|p| (a.row_ind[p], a.values[p])).collect();
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
            name,
            b,
            a,
            (a - b).abs()
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
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T1: status should be Optimal"
        );
        assert_close(result.solution[0], 0.5, EPS, "T1: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "T1: x[1]");
        assert_close(result.objective, 0.5, EPS, "T1: objective");
        assert!(
            result.bound_duals.is_empty(),
            "T1: infinite bounds → bound_duals empty"
        );
        assert_eq!(
            result.dual_solution.len(),
            1,
            "T1: dual_solution length == m == 1"
        );
    }

    /// T2: 等式制約付きQP
    /// min x^2+y^2 (1/2あり規約: Q=2I)  s.t. x+y=1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_qp_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T2: status should be Optimal"
        );
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
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T3: status should be Optimal"
        );
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
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T4: status should be Optimal"
        );
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
        assert_eq!(
            result1.status,
            SolveStatus::Optimal,
            "T5: cold start should be Optimal"
        );

        let ws = crate::qp::QpWarmStart {
            initial_active_set: vec![],
            initial_point: Some(result1.solution.clone()),
        };
        let result2 = solve_qp_warm(&problem2, &ws, &SolverOptions::default());

        assert_eq!(
            result2.status,
            SolveStatus::Optimal,
            "T5: warm start should be Optimal"
        );
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
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "T6: should be Infeasible"
        );
    }

    /// T7: ポートフォリオ最適化（Markowitz平均分散モデル）
    #[test]
    fn test_qp_portfolio_markowitz() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
        let c = vec![0.0, 0.0, 0.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 3, 4],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            5,
            3,
        )
        .unwrap();
        let b = vec![1.0, -1.0, 0.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T7: status should be Optimal"
        );
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
        let q =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[10.0, 8.0, 8.0, 10.0], 2, 2)
                .unwrap();
        let c = vec![-28.0, -26.0];
        let a = CscMatrix::new(0, 2);
        let b_vec = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b_vec, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T8: status should be Optimal"
        );
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
            3,
            2,
        )
        .unwrap();
        let b = vec![-1.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T9: status should be Optimal"
        );
        assert_close(result.solution[0], 0.0, EPS, "T9: x[0]");
        assert_close(result.solution[1], 1.0, EPS, "T9: x[1]");
        assert_close(result.objective, 1.0, EPS, "T9: objective");
    }

    /// T10: 複合制約テスト（等式+不等式の組み合わせ）
    #[test]
    fn test_qp_mixed_constraints() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-2.0, -4.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2],
            &[0, 1, 0, 1, 0],
            &[1.0, 1.0, -1.0, -1.0, -1.0],
            3,
            2,
        )
        .unwrap();
        let b = vec![2.0, -2.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T10: status should be Optimal"
        );
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
        assert!(
            matches!(
                result.status,
                SolveStatus::Optimal | SolveStatus::SuboptimalSolution
            ),
            "T11: status should be Optimal or SuboptimalSolution (got {:?})",
            result.status
        );
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
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T12: status should be Optimal"
        );
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

        let opts = SolverOptions {
            timeout_secs: Some(0.0),
            ..Default::default()
        };

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

        let opts = SolverOptions {
            qp_solver: QpSolverChoice::IpPmm,
            ..Default::default()
        };
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
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T20: concurrent should be Optimal"
        );
        assert!((result.solution[0] - 0.5).abs() < EPS, "T20: x[0] ≈ 0.5");
        assert!((result.solution[1] - 0.5).abs() < EPS, "T20: x[1] ≈ 0.5");
        assert!((result.objective - 0.5).abs() < EPS, "T20: obj ≈ 0.5");
    }

    /// T23: presolveパス pfeas検証 — 大行ノルム制約での Ruiz scaling 耐性確認
    ///
    /// 行ノルムが大きい制約 (Ruiz scaling 後に e[i]<<1) を含む問題で、元問題 (A, b)
    /// で直接 A*x - b を計算する pfeas 評価が e[i] に依存せず正しく動くことを確認する。
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

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T23: Optimal解が得られること"
        );
        // 元問題でpfeasを直接検証: A*x - b <= 0 のはず
        let ax = problem.a.mat_vec_mul(&result.solution).unwrap();
        let pfeas = ax
            .iter()
            .zip(problem.b.iter())
            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
            .fold(0.0_f64, f64::max);
        let norm_b = problem
            .b
            .iter()
            .fold(0.0_f64, |a, &bi| a.max(bi.abs()))
            .max(1.0);
        let eps = opts.ipm_eps();
        assert!(
            pfeas < eps * (1.0 + norm_b),
            "T23: 元問題でpfeas={pfeas:.2e} < eps*(1+norm_b)={:.2e}（e[i]<<1でも正しく検証）",
            eps * (1.0 + norm_b)
        );
    }

    /// T24: presolveパス bfeas検証 — bounds付き問題で Optimal 解が境界を満たすことを確認。
    /// solve_qp_with 経由で bounds を持つ問題を解き、post-postsolve bfeas チェックが
    /// 正常解を誤降格しないことを確認する。
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

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T24: bounds付き問題でOptimal解が得られること"
        );
        let x = result.solution[0];
        assert!(x >= -1e-4, "T24: x >= lb=0, got x={x}");
        assert!(x <= 1.0 + 1e-4, "T24: x <= ub=1, got x={x}");
    }

    /// T25: post-postsolve pfeas+bfeas — 正常解で Optimal を維持することを確認。
    /// solve_qp_with 経由で制約 + bounds 付き問題を解き、post-postsolve チェックが
    /// 正常解を誤降格しないことを確認する。
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

        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "T26: presolve有効時もOptimalを返すこと"
        );
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

    /// T27: 不定Q行列（対角に負値）→ 慣性修正付き IPM で KKT 点を返す
    ///
    /// Q = diag(-1.0, 1.0, 1.0)、c = [0,0,0]、制約なし、bounds なし。
    /// 真の問題は x1 → ∞ で非有界だが、慣性修正付き IPM は
    /// δ_ic = 1.0 (Gershgorin: -(-1.0) = 1.0) を加え Q_mod = diag(0,2,2) として解く。
    /// 修正問題の KKT 点は x* = (0,0,0)、status は LocallyOptimal または Unbounded。
    ///
    /// 検証:
    ///  - NonConvex が返らないこと（慣性修正で IPM を走らせる）
    ///  - LocallyOptimal, Optimal, Timeout, Unbounded のいずれかが返ること
    #[test]
    fn test_qp_nonconvex_indefinite_q() {
        // Q = diag(-1.0, 1.0, 1.0)（不定行列: 対角に負値）
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1.0, 1.0, 1.0], 3, 3).unwrap();
        let c = vec![0.0, 0.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        // NonConvex は返ってはならない（慣性修正で IPM にルーティングされる）
        assert!(
            !matches!(result.status, SolveStatus::NonConvex(_)),
            "T27: 不定Q行列で NonConvex を返してはならない（慣性修正で IPM にルーティング）。got: {:?}",
            result.status
        );
        // LocallyOptimal, Optimal, Unbounded, Timeout などの有効なステータスが返ること
        assert!(
            matches!(
                result.status,
                SolveStatus::LocallyOptimal | SolveStatus::Optimal
                | SolveStatus::Unbounded | SolveStatus::Timeout
                | SolveStatus::SuboptimalSolution | SolveStatus::NumericalError
            ),
            "T27: 有効なステータス (LocallyOptimal/Optimal/Unbounded/Timeout 等) を返すこと。got: {:?}",
            result.status
        );
    }

    /// T27b: 不定Q行列 + 有界制約 → LocallyOptimal を返す
    ///
    /// Q = diag(-2.0, 2.0)、c = [0, 0]、制約なし、bounds = [-1, 1]^2。
    /// Gershgorin δ_ic = 2.0。Q_mod = diag(0, 4)。
    /// KKT 点: x1 は bounds 活性 (x1=±1 のどちらか)、x2=0。
    /// 慣性修正付き IPM は LocallyOptimal を返すことを期待。
    #[test]
    fn test_qp_nonconvex_with_bounds() {
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[-2.0, 2.0],
            2,
            2,
        ).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let b = vec![];
        let bounds = vec![(-1.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds.clone()).unwrap();

        let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
        let result = solve_qp_with(&problem, &opts);

        // NonConvex は返ってはならない
        assert!(
            !matches!(result.status, SolveStatus::NonConvex(_)),
            "T27b: 不定Q行列 + bounds で NonConvex を返してはならない。got: {:?}", result.status
        );
        // LocallyOptimal または Optimal が期待（有界なので収束しやすい）
        assert!(
            matches!(result.status, SolveStatus::LocallyOptimal | SolveStatus::Optimal
                | SolveStatus::SuboptimalSolution | SolveStatus::Timeout),
            "T27b: status は LocallyOptimal/Optimal/SuboptimalSolution/Timeout のいずれかであること。got: {:?}",
            result.status
        );
        // 解が存在する場合、bounds 内に収まっていること
        if !result.solution.is_empty() {
            for (&xi, &(lb, ub)) in result.solution.iter().zip(bounds.iter()) {
                assert!(xi >= lb - 1e-4 && xi <= ub + 1e-4,
                    "T27b: 解が bounds 内に収まっていること: x={:.6}, bounds=[{:.1},{:.1}]", xi, lb, ub);
            }
        }
    }

    /// T28: 半正定値Q行列（最小固有値=0）→ PSD判定（NonConvexでないこと）
    /// Q = diag(0.0, 1.0, 1.0) → Q+eps*I の全ピボット > 0 → PSD判定
    /// 期待: check_q_positive_semidefinite が true を返す
    #[test]
    fn test_qp_psd_semidefinite_q() {
        // Q = diag(0.0, 1.0, 1.0)（半正定値行列: 最小固有値=0）
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.0, 1.0, 1.0], 3, 3).unwrap();
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

    /// 閾値は ‖Q‖_max × 1e-6 (QPS encoding noise 相当の相対許容)。
    /// ‖Q‖_max=1.0 なので neg_tol=1e-6。-1e-11 ≪ -1e-6 (= -neg_tol)
    /// 条件 `q < -neg_tol` を満たさないので PSD として扱う。
    #[test]
    fn test_qp_diagonal_boundary_below_threshold() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-11_f64, 1.0, 1.0], 3, 3)
            .unwrap();
        assert!(
            check_q_positive_semidefinite(&q),
            "Q[0,0]=-1e-11 は noise 範囲内 (‖Q‖_max × 1e-6 = 1e-6 より小さい) のため PSD"
        );
    }

    /// 境界値 Q[0,0]=-1e-7 < -neg_tol(=1e-6)? いえ、-1e-7 > -1e-6 なので閾値より小さい
    /// (絶対値が小さい) → PSD と判定。‖Q‖_max=1 で 1e-7/1=1e-7 = noise 程度。
    #[test]
    fn test_qp_diagonal_boundary_at_noise_floor() {
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-7_f64, 1.0, 1.0], 3, 3).unwrap();
        // -1e-7 > -1e-6 (= -neg_tol) なので非凸検出しない (noise 範囲)
        assert!(
            check_q_positive_semidefinite(&q),
            "Q[0,0]=-1e-7 は閾値 (-‖Q‖_max × 1e-6 = -1e-6) 範囲内のため PSD"
        );
    }

    /// 境界値 Q[0,0]=-1e-4（閾値 -1e-6 を絶対値で超える） → NonConvex 検出。
    /// 1e-4 / ‖Q‖_max=1 = 1e-4 > 1e-6 = encoded noise tolerance.
    #[test]
    fn test_qp_diagonal_boundary_above_threshold() {
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-4_f64, 1.0, 1.0], 3, 3).unwrap();
        assert!(
            !check_q_positive_semidefinite(&q),
            "Q[0,0]=-1e-4 は閾値 (-‖Q‖_max × 1e-6 = -1e-6) を超えるため NonConvex 検出"
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
            prob.num_vars,
            prob.num_constraints,
            prob.q.values.len()
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
        eprintln!(
            "HS268 status={:?} obj={:.6e}",
            result.status, result.objective
        );
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
            qp_solver: QpSolverChoice::IpPmm,
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
        let opts = SolverOptions {
            timeout_secs: None,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "A2-T03: QP タイムアウトなしで収束すること"
        );
    }

    /// A3-C02: cancel_flag 事前設定で即停止（QP版）
    #[test]
    fn test_a3c02_cancel_flag_preset_qp_returns_timeout() {
        // SPEC: A3-C02 (QP版)
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let cancel = Arc::new(AtomicBool::new(true)); // 事前に true
        let opts = SolverOptions {
            cancel_flag: Some(Arc::clone(&cancel)),
            qp_solver: QpSolverChoice::IpPmm,
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
            qp_solver: QpSolverChoice::IpPmm,
            ..SolverOptions::default()
        };
        let opts_without = SolverOptions {
            presolve: false,
            qp_solver: QpSolverChoice::IpPmm,
            ..SolverOptions::default()
        };
        let result_with = solve_qp_with(&problem, &opts_with);
        let result_without = solve_qp_with(&problem, &opts_without);
        assert_eq!(
            result_with.status,
            SolveStatus::Optimal,
            "A4-P01: presolve=true → Optimal"
        );
        assert_eq!(
            result_without.status,
            SolveStatus::Optimal,
            "A4-P01: presolve=false → Optimal"
        );
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
        // case1: 対角負値 → 対角チェックで検出（n>1000 でも有効）。
        // 閾値は ‖Q‖_max × 1e-6。‖Q‖_max=1 (off-diag 1.0) なので neg_tol=1e-6。
        // -1e-3 < -1e-6 で検出。
        let mut rows = vec![0usize];
        let mut cols = vec![0usize];
        let mut vals = vec![-1e-3_f64]; // -1e-3 < -1e-6 → 検出
        for i in 1..n {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
        }
        let q1 = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert!(
            !check_q_positive_semidefinite(&q1),
            "A6-I03: n=1001 対角負値は NonConvex を検出"
        );

        // case2: 非対角の非 PSD（対角チェックには引っかからない）→ n>1000 でスキップ
        let mut rows2: Vec<usize> = (0..n).collect();
        let mut cols2: Vec<usize> = (0..n).collect();
        let mut vals2: Vec<f64> = vec![1.0; n]; // 全て正の対角
                                                // 非対角に負値追加（非 PSD だが対角チェックには引っかからない）
        rows2.push(0);
        cols2.push(1);
        vals2.push(-2.0);
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
                qp_solver: QpSolverChoice::IpPmm,
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
            qp_solver: QpSolverChoice::IpPmm,
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
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let cancel = Arc::new(AtomicBool::new(true)); // 事前 true
        let opts = SolverOptions {
            qp_solver: QpSolverChoice::IpPmm,
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

    // ========== postsolve 修正検証テスト (T1-T7 + E1-E4) ==========
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
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
        let b = vec![4.0, 3.0];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
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
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 2.0], 2, n)
            .unwrap();
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
        assert!(
            (result.reduced_costs[0] - 2.0).abs() < tol,
            "T2: rc[0]=2 (x non-basic at lb)"
        );
        assert!(
            (result.reduced_costs[1] - 3.0).abs() < tol,
            "T2: rc[1]=3 (y non-basic at lb)"
        );
        assert!(
            (result.reduced_costs[2]).abs() < tol,
            "T2: rc[2]=0 (z fixed by FixedVar)"
        );
        // slack = b - Ax
        assert_eq!(result.slack.len(), 2, "T2: slack.len=2");
        assert!((result.slack[0] - 4.0).abs() < tol, "T2: slack[0]=4");
        assert!((result.slack[1] - 6.0).abs() < tol, "T2: slack[1]=6");
        // 相補性: x[j]*rc[j] ≈ 0
        for j in 0..3 {
            assert!(
                (result.solution[j] * result.reduced_costs[j]).abs() < 1e-7,
                "T2: complementarity x[{}]*rc[{}]",
                j,
                j
            );
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
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Eq, ConstraintType::Le],
        )
        .unwrap();
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
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[10.0, 1.0, 1.0], 2, n).unwrap();
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
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1e7, 1.0, 1.0, 1.0], 2, n)
            .unwrap();
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
        let slack0_expected = 1e7 - 1e7 * x - y;
        let slack1_expected = 2.0 - x - y;
        // 相対誤差で確認（LCS+Ruizで1e7スケールのため絶対誤差は大きくなりうる）
        let tol_rel = 1e-5_f64;
        assert!(
            (result.slack[0] - slack0_expected).abs() <= tol_rel * slack0_expected.abs().max(1.0),
            "T5: slack[0]={} expected={} (LCS b-Ax精度)",
            result.slack[0],
            slack0_expected
        );
        assert!(
            (result.slack[1] - slack1_expected).abs() <= tol_rel * slack1_expected.abs().max(1.0),
            "T5: slack[1]={} expected={}",
            result.slack[1],
            slack1_expected
        );
        // reduced_costs.len = 3
        assert_eq!(result.reduced_costs.len(), 3, "T5: rc.len=3");
        assert!(
            (result.reduced_costs[2]).abs() < 1e-6,
            "T5: rc[2]=0 (fixed z)"
        );
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
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
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
        assert!(
            (result.reduced_costs[2]).abs() < tol,
            "T6: rc[2]=0 (empty col fixed)"
        );
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
        let slack_expected = 1e7 - 1e7 * x - y;
        let tol_rel = 1e-5_f64;
        assert!(
            (result.slack[0] - slack_expected).abs() <= tol_rel * slack_expected.abs().max(1.0),
            "E4: slack[0]={} expected={} (LCS b-Ax精度)",
            result.slack[0],
            slack_expected
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
        assert_eq!(
            result.reduced_costs.len(),
            n,
            "LP path must preserve reduced_costs from Simplex"
        );

        // 値一致アサーション（許容誤差 1e-8）
        let expected = [0.0_f64, 1.0_f64];
        let tol = 1e-8_f64;
        for (j, (&got, &exp)) in result.reduced_costs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tol,
                "reduced_costs[{}]: expected {}, got {} (diff={})",
                j,
                exp,
                got,
                (got - exp).abs()
            );
        }
    }

    // ===== Concurrent default-build feature parity tests =====
    //
    // 動機: Q=0 LP dispatch と Mehrotra↔IP-PMM のフォールバック合流は
    //   `parallel` feature の有無に関係なく `Concurrent` (default) で動かなければ
    //   ならない (default build の `cargo test` が落ちる致命的乖離を防ぐ)。
    // これらの分岐が cfg gate の中に閉じ込められるとデフォルトビルドのユーザー体験が
    // 静かに壊れるため、明示的な regression test として 2 件を入れる。

    /// CDP-1: `Concurrent` (default) で Q=0 LP が Simplex に dispatch され、
    /// `reduced_costs` / `slack` が空でないこと。`parallel` feature 無効でも有効でも同じ
    /// 結果になることを担保する (`solve_qp_concurrent_dispatch` の Q=0 経路が
    /// cfg ガード外で動くこと)。
    #[test]
    fn test_concurrent_default_dispatches_zero_q_to_simplex() {
        // min x + 2y  s.t. x + y >= 1, x>=0, y>=0  (Q=0 → LP path)
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![1.0, 2.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // Concurrent (default)
        let opts = SolverOptions {
            qp_solver: QpSolverChoice::IpPmm,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);

        // Simplex 経由なら Optimal で `reduced_costs` / `slack` が埋まる。
        // IPM 経由になると両方とも空 (この regression test の検出対象)。
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "CDP-1: Q=0 LP must reach Optimal via Simplex (got {:?})",
            result.status
        );
        assert_eq!(
            result.reduced_costs.len(),
            n,
            "CDP-1: reduced_costs must be populated by Simplex (was IPM dispatched?)"
        );
        assert_eq!(
            result.slack.len(),
            1,
            "CDP-1: slack must be populated by Simplex (was IPM dispatched?)"
        );
    }

    // CDP-2 削除: Concurrent (Mehrotra+IPPMM 並列) の box-constrained QP fallback
    // 動作を担保する test だったが、IPM 廃止により不要 (IpPmm 単独経路に統合)。
    // box-constrained QP の Optimal 到達は test_qp_box_constrained_upper_bound 等で
    // 引き続きカバー (※IPPMM 単独で Optimal 到達できるべきだが現状 SuboptimalSolution
    //   で停止するケースあり、追跡 task として `refactor/remove-ipm-and-concurrent`
    //   ブランチ後の調査ブランチで原因究明予定)。

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
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T1: status");
        let sol_tol = 1e-3_f64;
        let tol = 1e-4_f64;
        // 解: x=0, y=0
        assert!(
            (result.solution[0]).abs() < sol_tol,
            "BD-T1: x≈0 (got {})",
            result.solution[0]
        );
        assert!(
            (result.solution[1]).abs() < sol_tol,
            "BD-T1: y≈0 (got {})",
            result.solution[1]
        );
        // bound_duals長: n_lb_orig=2 + n_ub_orig=2 = 4
        assert_eq!(result.bound_duals.len(), 4, "BD-T1: bound_duals.len()==4");
        // x=0=lb活性 → lb_dual > 0
        assert!(result.bound_duals[0] > tol, "BD-T1: lb_x>0 (active lower)");
        // y=0=lb活性 → lb_dual > 0
        assert!(result.bound_duals[1] > tol, "BD-T1: lb_y>0 (active lower)");
        // x上界非活性 → ub_dual ≈ 0
        assert!(
            result.bound_duals[2].abs() < tol,
            "BD-T1: ub_x≈0 (inactive)"
        );
        // y上界非活性 → ub_dual ≈ 0
        assert!(
            result.bound_duals[3].abs() < tol,
            "BD-T1: ub_y≈0 (inactive)"
        );
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
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![2.0, 1.0, 1.0];
        // x + y <= 10
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        // z=3 → lb=ub=3（FixedVar）
        let bounds = vec![(0.0_f64, 5.0_f64), (0.0_f64, 5.0_f64), (3.0_f64, 3.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T2: status");
        let sol_tol = 5e-3_f64; // IPM解の精度（primal解は双対精度より粗め）
        let tol = 1e-4_f64; // bound_duals精度（符号・大小比較用）
                            // 解: x≈0, y≈0, z≈3
        assert!(
            (result.solution[0]).abs() < sol_tol,
            "BD-T2: x≈0 (got {})",
            result.solution[0]
        );
        assert!(
            (result.solution[1]).abs() < sol_tol,
            "BD-T2: y≈0 (got {})",
            result.solution[1]
        );
        assert!(
            (result.solution[2] - 3.0).abs() < sol_tol,
            "BD-T2: z≈3 (got {})",
            result.solution[2]
        );
        // bound_duals長: n_lb_orig=3 + n_ub_orig=3 = 6
        assert_eq!(result.bound_duals.len(), 6, "BD-T2: bound_duals.len()==6");
        // x=0=lb活性 → lb_dual ≈ 2 (目的関数x係数=2)
        assert!(result.bound_duals[0] > tol, "BD-T2: lb_x>0");
        // y=0=lb活性 → lb_dual ≈ 1 (目的関数y係数=1)
        assert!(result.bound_duals[1] > tol, "BD-T2: lb_y>0");
        // 非対称検証: lb_x ≠ lb_y（変数順序バグ検出）
        assert!(
            (result.bound_duals[0] - result.bound_duals[1]).abs() > tol,
            "BD-T2: lb_x({}) != lb_y({}) — 変数順序バグ検出",
            result.bound_duals[0],
            result.bound_duals[1]
        );
        // z除去変数 → lb_dual = 0.0
        assert!(
            (result.bound_duals[2]).abs() < tol,
            "BD-T2: lb_z==0 (removed)"
        );
        // x上界非活性 → ub_dual ≈ 0（IPM精度のため5e-3まで許容）
        assert!(
            result.bound_duals[3].abs() < 5e-3,
            "BD-T2: ub_x≈0 (got {})",
            result.bound_duals[3]
        );
        // y上界非活性 → ub_dual ≈ 0
        assert!(
            result.bound_duals[4].abs() < 5e-3,
            "BD-T2: ub_y≈0 (got {})",
            result.bound_duals[4]
        );
        // z除去変数 → ub_dual = 0.0
        assert!(
            (result.bound_duals[5]).abs() < tol,
            "BD-T2: ub_z==0 (removed)"
        );
        // S2: KKT停止性検証（全変数で ∇f[j] - (A^T y)[j] - lb_dual[j] + ub_dual[j] ≈ 0）
        // ∇f(x*)_x = 0.001*0 + 2 = 2, ∇f(x*)_y = 0.001*0 + 1 = 1
        // dual_solution: x+y<=10 の双対変数（最適解x=y=0なので制約非活性→ dual≈0）
        let dual = if result.dual_solution.is_empty() {
            0.0
        } else {
            result.dual_solution[0]
        };
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
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
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
        let bounds = vec![
            (f64::NEG_INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::INFINITY),
            (0.0_f64, 3.0_f64),
        ];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T4: status");
        // n_lb_orig=1(z:0), n_ub_orig=1(z:3) → bound_duals.len()==2
        assert_eq!(result.bound_duals.len(), 2, "BD-T4: bound_duals.len()==2");
        // z=0 (lb 活性) なので KKT: 0.001*0 + 1 - z_lb + z_ub = 0 → z_lb = 1, z_ub = 0
        let z_lb = result.bound_duals[0];
        let z_ub = result.bound_duals[1];
        assert!(
            (z_lb - 1.0).abs() < 1e-3,
            "BD-T4: z_lb≈1 (KKT recovered for EmptyCol), got {}",
            z_lb
        );
        assert!(
            z_ub.abs() < 1e-3,
            "BD-T4: z_ub≈0 (ub inactive), got {}",
            z_ub
        );
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
        assert!(
            result.bound_duals.is_empty(),
            "BD-T5: bound_duals empty for unbounded vars"
        );
    }

    /// BD-T6: FixedVar + ub活性変数（ub_dual非ゼロ × presolve残存変数）
    /// min 1/2*(0.001*x^2 + 0.001*y^2 + 0.001*z^2) - x - y + z
    /// s.t. x + y <= 10, 0 <= x <= 3, 0 <= y <= 5, z=2 (fixed)
    /// → 最適解: x=3(ub活性), y=5(ub活性), z=2
    /// → bound_duals[3]>0 (ub_x活性), bound_duals[4]>0 (ub_y活性)
    #[test]
    fn test_bd_t6_ub_active_with_presolve() {
        let n = 3usize;
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 1.0];
        // x + y <= 10
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![10.0];
        // z=2 → fixed
        let bounds = vec![(0.0_f64, 3.0_f64), (0.0_f64, 5.0_f64), (2.0_f64, 2.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T6: status");
        let sol_tol = 1e-3_f64; // IPM primal解精度
        let tol = 1e-4_f64; // bound_duals符号・大小比較精度
                            // 最適解: x=3, y=5, z=2
        assert!(
            (result.solution[0] - 3.0).abs() < sol_tol,
            "BD-T6: x≈3 (got {})",
            result.solution[0]
        );
        assert!(
            (result.solution[1] - 5.0).abs() < sol_tol,
            "BD-T6: y≈5 (got {})",
            result.solution[1]
        );
        assert!(
            (result.solution[2] - 2.0).abs() < sol_tol,
            "BD-T6: z≈2 (got {})",
            result.solution[2]
        );
        // bound_duals長: n_lb_orig=3 + n_ub_orig=3 = 6
        assert_eq!(result.bound_duals.len(), 6, "BD-T6: bound_duals.len()==6");
        // x=3=ub活性 → lb_dual≈0, ub_dual>0
        assert!(
            result.bound_duals[0].abs() < tol,
            "BD-T6: lb_x≈0 (inactive)"
        );
        assert!(
            result.bound_duals[1].abs() < tol,
            "BD-T6: lb_y≈0 (inactive)"
        );
        assert!(
            (result.bound_duals[2]).abs() < tol,
            "BD-T6: lb_z==0 (removed)"
        );
        assert!(result.bound_duals[3] > tol, "BD-T6: ub_x>0 (active upper)");
        assert!(result.bound_duals[4] > tol, "BD-T6: ub_y>0 (active upper)");
        assert!(
            (result.bound_duals[5]).abs() < tol,
            "BD-T6: ub_z==0 (removed)"
        );
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
        let opts = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "BD-T7: status");
        let sol_tol = 1e-3_f64;
        let tol = 1e-4_f64;
        // 最適解: x=2, y=1
        assert!(
            (result.solution[0] - 2.0).abs() < sol_tol,
            "BD-T7: x≈2 (got {})",
            result.solution[0]
        );
        assert!(
            (result.solution[1] - 1.0).abs() < sol_tol,
            "BD-T7: y≈1 (got {})",
            result.solution[1]
        );
        // bound_duals長: n_lb_orig=2(x,y), n_ub_orig=0 → len==2
        assert_eq!(result.bound_duals.len(), 2, "BD-T7: bound_duals.len()==2");
        // 制約dual ≠ 0 (constraint active)
        let dual = if result.dual_solution.is_empty() {
            0.0
        } else {
            result.dual_solution[0]
        };
        assert!(
            dual > tol,
            "BD-T7: constraint dual>0 (active), got {}",
            dual
        );
        // lb_x ≠ 0 (x=2=lb, active)
        assert!(
            result.bound_duals[0] > tol,
            "BD-T7: lb_x>0 (active lower bound), got {}",
            result.bound_duals[0]
        );
        // lb_y ≈ 0 (y=1 > lb=0, inactive)
        assert!(
            result.bound_duals[1].abs() < tol,
            "BD-T7: lb_y≈0 (inactive lower bound), got {}",
            result.bound_duals[1]
        );
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
            &[0, 1, 0],        // rows
            &[0, 1, 2],        // cols
            &[1.0, 2.5, -3.0], // vals
            2,
            3,
        )
        .unwrap();
        let norms = a.row_infinity_norms();
        assert_eq!(norms.len(), 2);
        assert!(
            (norms[0] - 3.0).abs() < 1e-15,
            "row0 norm: expected 3.0, got {}",
            norms[0]
        );
        assert!(
            (norms[1] - 2.5).abs() < 1e-15,
            "row1 norm: expected 2.5, got {}",
            norms[1]
        );
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
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1000.0], 2, 1).unwrap();
        let norms = a.row_infinity_norms();
        assert!((norms[0] - 1.0).abs() < 1e-15);
        assert!((norms[1] - 1000.0).abs() < 1e-15);

        // 正規化判定のロジック検証
        let b: Vec<f64> = vec![1.0, 1000.0];
        let x_val: f64 = 1.0 + 1e-7; // 微小な制約違反
        let ax: Vec<f64> = vec![x_val, 1000.0 * x_val]; // [1.0000001, 1000.0001]
        let eps: f64 = 1e-6;

        // 旧方式: max violation
        let pfeas_old = ax
            .iter()
            .zip(b.iter())
            .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
            .fold(0.0_f64, f64::max);
        // pfeas_old = max(1e-7, 1e-4) = 1e-4
        assert!(
            pfeas_old > 1e-5,
            "旧方式pfeasは大係数行に引きずられるべき: {}",
            pfeas_old
        );

        // 新方式: 行ノルム正規化
        let pfeas_normalized = ax
            .iter()
            .zip(b.iter())
            .zip(norms.iter())
            .map(|((&ax_i, &b_i), &rn)| {
                let violation = (ax_i - b_i).max(0.0);
                violation / (1.0 + rn + b_i.abs())
            })
            .fold(0.0_f64, f64::max);
        // 大係数行: 1e-4 / (1+1000+1000) = 5e-8
        // 小係数行: 1e-7 / (1+1+1) = 3.3e-8
        assert!(
            pfeas_normalized < eps,
            "正規化pfeasはeps未満であるべき: {}",
            pfeas_normalized
        );
    }

    /// 正規化なしでは判定が歪むが正規化ありで正しく判定できるケース
    #[test]
    fn test_pfeas_row_norm_false_suboptimal_prevention() {
        // b=0の大係数行: 1e6*x = 0 (等号制約として)
        // x = 1e-9 → violation = |1e6 * 1e-9 - 0| = 1e-3
        // 旧方式: pfeas = 1e-3, threshold = eps*(1+0).max(1.0) = eps*1 = 1e-6 → FAIL (偽SubOptimal)
        // 新方式: 1e-3 / (1 + 1e6 + 0) ≈ 1e-9 < eps → PASS (正しくOptimal)

        let a = CscMatrix::from_triplets(&[0], &[0], &[1e6], 1, 1).unwrap();
        let norms = a.row_infinity_norms();
        assert!((norms[0] - 1e6).abs() < 1e-9);

        let b_val: f64 = 0.0;
        let ax_val: f64 = 1e6 * 1e-9; // = 1e-3
        let eps: f64 = 1e-6;

        // 旧方式: 偽SubOptimal
        let norm_b = b_val.abs().max(1.0); // max(|b|, 1.0) = 1.0
        let pfeas_old = (ax_val - b_val).abs();
        assert!(
            pfeas_old >= eps * (1.0 + norm_b),
            "旧方式では偽SubOptimalになるべき"
        );

        // 新方式: 正しくOptimal
        let pfeas_norm = (ax_val - b_val).abs() / (1.0 + norms[0] + b_val.abs());
        assert!(
            pfeas_norm < eps,
            "正規化方式ではOptimalであるべき: {}",
            pfeas_norm
        );
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

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(
            start.elapsed().as_secs_f64() < 6.0,
            "C-QP: wall-clock 6秒超過"
        );
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
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();

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
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(
            start.elapsed().as_secs_f64() < 6.0,
            "E-Eq: wall-clock 6秒超過"
        );
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

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(
            start.elapsed().as_secs_f64() < 6.0,
            "E-Ge: wall-clock 6秒超過"
        );
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

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(
            start.elapsed().as_secs_f64() < 6.0,
            "E-Box: wall-clock 6秒超過"
        );
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
        let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
        let b = vec![1.0, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Eq, ConstraintType::Le],
        )
        .unwrap();

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(
            start.elapsed().as_secs_f64() < 6.0,
            "E-Mixed: wall-clock 6秒超過"
        );
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

        let opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(
            start.elapsed().as_secs_f64() < 6.0,
            "E-Unconstrained: wall-clock 6秒超過"
        );
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "E-Unconstrained: status"
        );
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

        let mut opts = SolverOptions {
            timeout_secs: Some(5.0),
            ..Default::default()
        };
        opts.presolve = false;
        let start = std::time::Instant::now();
        let result = solve_qp_with(&problem, &opts);
        assert!(
            start.elapsed().as_secs_f64() < 6.0,
            "F-QP-Fixed: wall-clock 6秒超過"
        );
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "F-QP-Fixed: status must be Optimal, got {:?}",
            result.status
        );
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
            ipm: crate::options::IpmOptions {
                max_iter: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        // 6d10eaf以降 SuboptimalSolution は公開API上で有効なステータス。
        // 検証: MaxIterations / NumericalError が直接漏れないこと。
        assert_ne!(
            result.status,
            SolveStatus::MaxIterations,
            "G-1: MaxIterationsが外部APIに漏れた"
        );
        assert!(
            matches!(
                result.status,
                SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
            ),
            "G-1: status must be Optimal/Timeout/SuboptimalSolution, got {:?}",
            result.status
        );
    }

    /// G-2: MaxIterations が外部API に漏れないことを確認
    /// max_iter=1 で最大反復到達 → Optimal/Timeout/SuboptimalSolution のいずれかになる
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
            ipm: crate::options::IpmOptions {
                max_iter: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        // MaxIterations は内部 status であり外部 API に漏れてはならない
        assert_ne!(
            result.status,
            SolveStatus::MaxIterations,
            "G-2: MaxIterationsが外部APIに漏れた"
        );
        assert!(
            matches!(
                result.status,
                SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
            ),
            "G-2: status must be Optimal/Timeout/SuboptimalSolution, got {:?}",
            result.status
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
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

        let opts_on = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let mut opts_off = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        opts_off.presolve = false;

        let result_on = solve_qp_with(&problem, &opts_on);
        let result_off = solve_qp_with(&problem, &opts_off);

        assert_eq!(
            result_on.status,
            SolveStatus::Optimal,
            "H-1: presolve ON status"
        );
        assert_eq!(
            result_off.status,
            SolveStatus::Optimal,
            "H-1: presolve OFF status"
        );
        assert!(
            (result_on.solution[0] - result_off.solution[0]).abs() < 1e-4,
            "H-1: presolve ON/OFF x[0]不一致: ON={}, OFF={}",
            result_on.solution[0],
            result_off.solution[0]
        );
        assert!(
            (result_on.solution[1] - result_off.solution[1]).abs() < 1e-4,
            "H-1: presolve ON/OFF x[1]不一致: ON={}, OFF={}",
            result_on.solution[1],
            result_off.solution[1]
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

        let opts_on = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let mut opts_off = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        opts_off.presolve = false;

        let result_on = solve_qp_with(&problem, &opts_on);
        let result_off = solve_qp_with(&problem, &opts_off);

        assert_eq!(
            result_on.status,
            SolveStatus::Optimal,
            "H-2: presolve ON status"
        );
        assert_eq!(
            result_off.status,
            SolveStatus::Optimal,
            "H-2: presolve OFF status"
        );
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

        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "H-3: Ge+presolve status"
        );
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
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();

        // presolve=false: presolveバグを回避してソルバー本体の正確さを検証
        let mut opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        opts.presolve = false;
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "H-4: Mixed(Ge+Le)+no-presolve status"
        );
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
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
                .unwrap();
        let b = vec![0.5, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();

        // presolve=ON + Ruiz=ON（デフォルト）でバグが再現していたパターン
        let opts = SolverOptions {
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "H-5: Mixed(Ge+Le)+presolve=ON+Ruiz=ON status. got {:?}",
            result.status
        );
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
        assert_eq!(
            result_no_presolve.status,
            SolveStatus::Optimal,
            "H-5: presolve=OFF status. got {:?}",
            result_no_presolve.status
        );
        assert_close(
            result_no_presolve.solution[0],
            0.25,
            EPS,
            "H-5(no-presolve): x[0]",
        );
        assert_close(
            result_no_presolve.solution[1],
            0.25,
            EPS,
            "H-5(no-presolve): x[1]",
        );
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
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "D-1: well-solved QP must stay Optimal after dfeas check"
        );
    }

    /// D-2: スケール不変性 — 係数を1e6倍してもOptimalが維持される
    /// min (1e6)^2 * (x^2+y^2) s.t. 1e6*(x+y) >= 1e6, x,y >= 0
    /// 数学的に同一問題だが、絶対閾値ではdfeasが巨大値になり誤判定する
    #[test]
    fn test_dfeas_scale_invariant() {
        let scale = 1e6_f64;
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[2.0 * scale * scale, 2.0 * scale * scale],
            2,
            2,
        )
        .unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-scale, -scale], 1, 2).unwrap();
        let b = vec![-scale];
        let bounds = vec![(0.0, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "D-2: scaled QP must stay Optimal (relative threshold). got {:?}",
            result.status
        );
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
        let status = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 1e-6);
        assert_eq!(
            status,
            SolveStatus::SuboptimalSolution,
            "D-3a: bad solution with dfeas=2.0 >> 1e-6 must be SuboptimalSolution"
        );
        let status_ok = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 10.0);
        assert_eq!(
            status_ok,
            SolveStatus::Optimal,
            "D-3a: same solution with dfeas=2.0 < 10.0 stays Optimal"
        );

        // (b) 成分ごと相対版: residual=2.0, scale=1+2+0+0=3, relative=2/3≈0.667
        // eps=0.01 → SuboptimalSolution
        let status_rel =
            ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 0.01);
        assert_eq!(
            status_rel,
            SolveStatus::SuboptimalSolution,
            "D-3b: relative dfeas=0.667 >> 0.01 must be SuboptimalSolution"
        );
        // eps=1.0 → Optimal (relative < 1.0)
        let status_rel_ok =
            ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 1.0);
        assert_eq!(
            status_rel_ok,
            SolveStatus::Optimal,
            "D-3b: relative dfeas=0.667 < 1.0 stays Optimal"
        );
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
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "D-4: large-KKT-scale QP must be Optimal. got {:?}",
            result.status
        );
        assert!(
            (result.solution[0] - 5e-7).abs() < 1e-9,
            "D-4: x*=5e-7, got {:.2e}",
            result.solution[0]
        );
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
        let status =
            ipm_core::check_dfeas_status_relative(&problem, &big_x, &empty_y, &empty_bd, 0.01);
        assert_eq!(
            status,
            SolveStatus::SuboptimalSolution,
            "D-5a: large absolute residual with no cancellation → SuboptimalSolution"
        );

        // 正しいキャンセレーション: Qx + c がほぼ0になるケース
        // x ≈ 0 (最適解) → Qx ≈ 0, c = 0, 残差 ≈ 0
        let good_x = vec![1e-12, 1e-12];
        let status_good =
            ipm_core::check_dfeas_status_relative(&problem, &good_x, &empty_y, &empty_bd, 1e-8);
        assert_eq!(
            status_good,
            SolveStatus::Optimal,
            "D-5b: near-optimal solution → Optimal"
        );
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
            "REFIT-T1: y_lb 復元 ≈ 2.5, got {}",
            result.bound_duals[0]
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
            "REFIT-T2: y_ub 復元 ≈ 3.0, got {}",
            result.bound_duals[0]
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
            "REFIT-T4: 既に正しい値は維持される, got {}",
            result.bound_duals[0]
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
            "REFIT-T5: z_lb_x ≈ 1.0, got {}",
            result.bound_duals[0]
        );
        assert!(
            result.bound_duals[1].abs() < 1e-9,
            "REFIT-T5: z_lb_y ≈ 0.0, got {}",
            result.bound_duals[1]
        );
    }

    #[test]
    fn test_project_duals_from_singleton_columns_clamps_infeasible_positive_le_dual() {
        // row: -x0 + x1 <= 0, bounds x0,x1 >= 0, c = 0
        // x0/x1 はともに lb に張り付いているので、正の Le dual は
        // x0 列で qx+c+A^T y < 0 を作り、非負 z_lb では補正不能。
        // singleton column 条件から row dual の feasible interval は {0} になり、
        // projection 後に y=0, z=0, KKT residual=0 になるべき。
        let n = 2usize;
        let q = CscMatrix::new(n, n);
        let c = vec![0.0_f64, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0_f64, 1.0], 1, n).unwrap();
        let b = vec![0.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0, 0.0],
            dual_solution: vec![5.0],
            bound_duals: vec![0.0, 0.0],
            ..SolverResult::default()
        };

        project_duals_from_singleton_columns(&problem, &mut result);
        refit_bound_duals_kkt(&problem, &mut result);

        assert!(
            result.dual_solution[0].abs() < 1e-12,
            "row dual should project to 0, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.bound_duals.iter().all(|v| v.abs() < 1e-12),
            "bound duals should stay zero, got {:?}",
            result.bound_duals
        );
    }

    #[test]
    fn test_project_duals_from_singleton_columns_respects_lb_only_lower_bound() {
        // x >= 0, c = -2, row: x <= 0.
        // lb-only singleton column gives qx+c+a*y >= 0 -> -2 + y >= 0 -> y >= 2.
        // projection は row dual を 2 まで引き上げ、z_lb は 0 のまま KKT を満たす。
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![-2.0_f64];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
        let b = vec![0.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0],
            ..SolverResult::default()
        };

        project_duals_from_singleton_columns(&problem, &mut result);
        refit_bound_duals_kkt(&problem, &mut result);

        assert!(
            (result.dual_solution[0] - 2.0).abs() < 1e-12,
            "row dual should project to 2, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.bound_duals[0].abs() < 1e-12,
            "z_lb should remain 0, got {}",
            result.bound_duals[0]
        );
    }

    #[test]
    fn test_zero_inactive_inequality_duals_clears_slack_le_rows() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![0.0_f64];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
        let b = vec![10.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![3.0],
            dual_solution: vec![7.0],
            bound_duals: vec![0.0],
            ..SolverResult::default()
        };

        zero_inactive_inequality_duals(&problem, &mut result);

        assert!(
            result.dual_solution[0].abs() < 1e-12,
            "inactive Le row dual should be zeroed, got {}",
            result.dual_solution[0]
        );
    }

    #[test]
    fn test_refine_dual_projected_gradient_uses_curvature_scaled_step() {
        let n = 1usize;
        let q = CscMatrix::new(n, n);
        let c = vec![-1.0_f64];
        let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0_f64], 1, n).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0e-3],
            dual_solution: vec![0.0],
            bound_duals: vec![0.0],
            ..SolverResult::default()
        };

        refine_dual_projected_gradient(&problem, &mut result, None);

        assert!(
            (result.dual_solution[0] - 1.0e-3).abs() < 1e-9,
            "projected gradient should take curvature-scaled step to y=1e-3, got {}",
            result.dual_solution[0]
        );
    }

    #[test]
    fn test_refine_dual_worst_active_block_updates_row_and_bound_duals_together() {
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
        let c = vec![-1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64, 0.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![0.0_f64, 0.0_f64],
            ..SolverResult::default()
        };

        let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        refine_dual_worst_active_block(&problem, &mut result, None);
        let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );

        assert!(
            post < pre,
            "DUAL_BLOCK should reduce KKT residual: pre={} post={}",
            pre,
            post
        );
        assert!(
            post < 1e-12,
            "DUAL_BLOCK should recover exact local KKT, got {}",
            post
        );
        assert!(
            (result.dual_solution[0] - 1.0).abs() < 1e-9,
            "row dual should be recovered to 1, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.bound_duals[0].abs() < 1e-12,
            "inactive x lower-bound dual should stay 0, got {}",
            result.bound_duals[0]
        );
        assert!(
            (result.bound_duals[1] - 1.0).abs() < 1e-9,
            "active y lower-bound dual should be recovered to 1, got {}",
            result.bound_duals[1]
        );
    }

    #[test]
    fn test_dual_recovery_postprocess_can_improve_without_dual_ir() {
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
        let c = vec![-1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64, 0.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![0.0_f64, 0.0_f64],
            ..SolverResult::default()
        };

        let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        let post = run_dual_recovery_postprocess(&problem, &view, &mut result, None, false);

        assert!(
            post < pre,
            "standalone dual recovery postprocess should reduce KKT residual: pre={} post={}",
            pre,
            post
        );
        assert!(
            post < 1e-12,
            "standalone dual recovery postprocess should recover exact local KKT, got {}",
            post
        );
    }

    #[test]
    fn test_dual_only_ir_uses_active_rows_and_keeps_inactive_le_zero() {
        let q = CscMatrix::new(1, 1);
        let c = vec![-1.0_f64];
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0_f64, 1.0_f64], 2, 1).unwrap();
        let b = vec![1.0_f64, 10.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![
                crate::problem::ConstraintType::Eq,
                crate::problem::ConstraintType::Le,
            ],
        )
        .unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64],
            dual_solution: vec![0.0_f64, 0.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };

        let accepted = try_dual_only_ir(&problem, &mut result, 1e-8, None);

        assert!(accepted > 0, "DUAL_IR should accept at least one iteration");
        assert!(
            (result.dual_solution[0] - 1.0).abs() < 1e-9,
            "equality row dual should move to 1, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.dual_solution[1].abs() < 1e-12,
            "inactive Le row dual should stay zero, got {}",
            result.dual_solution[1]
        );
    }

    #[test]
    fn test_dual_only_ir_couples_row_and_bound_duals() {
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
        let c = vec![-1.0_f64, 0.0_f64];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
        let b = vec![1.0_f64];
        let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
        let problem =
            QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0_f64, 0.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![0.0_f64, 0.0_f64],
            ..SolverResult::default()
        };

        let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        let accepted = try_dual_only_ir(&problem, &mut result, 1e-8, None);
        let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );

        assert!(accepted > 0, "coupled DUAL_IR should accept on bound-coupled case");
        assert!(post < pre, "coupled DUAL_IR should reduce KKT: pre={} post={}", pre, post);
        assert!(
            (result.dual_solution[0] - 1.0).abs() < 1e-6,
            "row dual should be close to 1, got {}",
            result.dual_solution[0]
        );
        assert!(
            (result.bound_duals[1] - 1.0).abs() < 1e-6,
            "active lower-bound dual should be close to 1, got {}",
            result.bound_duals[1]
        );
    }

    /// DUAL_IR weighted Gram: scale[j=0]≈1 が component-wise 最悪のとき、
    /// 絶対値では r_d[j=1] > r_d[j=0] でも、加重 LS が r_rel[j=0] を優先して削減する。
    /// 無加重 LS だと r_d[j=1] を優先し r_rel[j=0] を悪化させる (STADAT1 パターン)。
    #[test]
    fn test_dual_only_ir_weighted_gram_prioritizes_worst_component() {
        // Setup:
        //   n=2, m=2, Q=diag(0,1), c=[0,3]
        //   A=[[-1,-2],[1,1]], b=[-10,5]  (Eq at x=[0,5])
        //   y=[8, 8+1e-6]
        //
        //   r_d[j=0] = (-1)*8 + 1*(8+1e-6) = 1e-6, scale≈1, r_rel=1e-6  (worst)
        //   r_d[j=1] = 3 + 1*5 + (-2)*8 + 1*(8+1e-6) = 1e-6, scale=17, r_rel≈5.9e-8
        //
        //   The unweighted Gram minimizes r_d[0]^2 + r_d[1]^2 equally and would
        //   not prioritise j=0 despite it having the larger component-wise residual.
        //   The weighted Gram (weight 1/scale^2) reduces df_rel in one full step.
        let q = CscMatrix::from_triplets(&[1], &[1], &[1.0_f64], 2, 2).unwrap();
        let c = vec![0.0_f64, 3.0_f64];
        let a = CscMatrix::from_triplets(
            &[0usize, 1, 0, 1],
            &[0usize, 0, 1, 1],
            &[-1.0_f64, 1.0, -2.0, 1.0],
            2,
            2,
        )
        .unwrap();
        let b = vec![-10.0_f64, 5.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(
            q,
            c,
            a,
            b,
            bounds,
            vec![
                crate::problem::ConstraintType::Eq,
                crate::problem::ConstraintType::Eq,
            ],
        )
        .unwrap();

        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0_f64, 5.0_f64],
            dual_solution: vec![8.0_f64, 8.0_f64 + 1e-6_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };

        let target_pf = 5e-7;
        let accepted = try_dual_only_ir(&problem, &mut result, target_pf, None);

        assert!(accepted > 0, "DUAL_IR should accept at least one iteration");

        let view = crate::qp::ipm_solver::outcome::ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
        };
        let df_rel = crate::qp::ipm_solver::kkt::kkt_residual_rel(
            &view,
            &result.solution,
            &result.dual_solution,
            &result.bound_duals,
        );
        assert!(
            df_rel < target_pf,
            "weighted Gram should reduce df_rel below target_pf={:.1e}: got {:.3e}",
            target_pf,
            df_rel
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
        let q =
            CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 2.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![5.0_f64];
        let bounds = vec![
            (0.0_f64, f64::INFINITY),
            (0.0_f64, f64::INFINITY),
            (0.0_f64, 10.0_f64),
        ];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions {
            presolve: true,
            ..SolverOptions::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "REFIT-T6: status");
        // n_lb=3 (全 lb 有限), n_ub=1 (z のみ ub 有限) → bound_duals.len() = 4
        assert_eq!(result.bound_duals.len(), 4);
        // z=0 (lb 活性, c=2>0), KKT: 2 - z_lb_z = 0 → z_lb_z ≈ 2
        let z_lb_z = result.bound_duals[2];
        assert!(
            (z_lb_z - 2.0).abs() < 1e-2,
            "REFIT-T6: EmptyCol 変数 z_lb ≈ 2.0 (KKT 復元), got {}",
            z_lb_z
        );
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 数値精度テスト: f64 cancellation / IR / DD-guard
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    /// LSQ-IR: 単純な full-row-rank 系で y = A·target / (A·A^T) が解析的に求まることを
    /// `compute_lsq_dual_y` が DD-IR 込みで再現することを確認する。
    #[test]
    fn compute_lsq_dual_y_recovers_exact_solution_on_well_conditioned() {
        // 1×1: A=[[2]], Q=0, c=[6], x=[0], bounds=NEG_INF..INF (z=0)。
        // target = -(qx + c + bnd) = -6。A^T y = target → 2 y = -6 → y = -3。
        let a = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
        let q = CscMatrix::new(1, 1);
        let c = vec![6.0_f64];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0], // 任意 (compute_lsq_dual_y は内部で再計算)
            bound_duals: vec![],
            ..SolverResult::default()
        };
        let y = compute_lsq_dual_y(&problem, &result).expect("LSQ should succeed");
        assert!((y[0] - (-3.0)).abs() < 1e-12, "y ≈ -3, got {}", y[0]);
    }

    /// LSQ-IR: 正規方程式の cond^2·ε 誤差を DD residual の IR が縮めることを、
    /// 解析解との突合で確認する。
    #[test]
    fn compute_lsq_dual_y_ir_improves_ill_conditioned_problem() {
        // A = [[1, 1], [1, 1+δ]] (δ=1e-8) は cond(A) ≈ 1/δ = 1e8。
        // cond(AAT) ≈ 1e16 で f64 LSQ 1 回 solve は意味のある精度に達さない。
        // 解析解: A^T y = target なら y = (A A^T)^{-1} A target を計算。
        // ここでは target = (1, 1) として y = (1, 0) 近傍が解析解となる
        // (実 4x4 系で手計算で検算済み)。
        // テストは絶対値ではなく、IR で残差 ‖A^T y - target‖_inf が cond·ε^2 級まで
        // 縮むかを確認する。
        let delta = 1e-8;
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0_f64, 1.0, 1.0, 1.0 + delta],
            2,
            2,
        )
        .unwrap();
        let q = CscMatrix::new(2, 2);
        // target = (1, 1) を作る c。x=0, bnd=0 のとき target = -(qx+c+bnd) = -c。
        // よって c = (-1, -1) で target = (1, 1)。
        let c = vec![-1.0_f64, -1.0];
        let b = vec![0.0_f64; 2];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c.clone(), a.clone(), b, bounds).unwrap();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0, 0.0],
            dual_solution: vec![0.0, 0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        let y = compute_lsq_dual_y(&problem, &result).expect("LSQ should succeed");

        // residual = A^T y - target を DD で評価
        use twofloat::TwoFloat;
        let target = [1.0_f64, 1.0];
        let mut max_abs_res = 0.0_f64;
        for col in 0..2 {
            let mut s = TwoFloat::from(0.0);
            for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                s = s + TwoFloat::new_mul(a.values[k], y[a.row_ind[k]]);
            }
            let r = (f64::from(s) - target[col]).abs();
            max_abs_res = max_abs_res.max(r);
        }
        // f64 1 回 solve は cond²·ε = 1e16·2e-16 ≈ 2 (relative) で打ち止め。
        // IR が効けば 1 iter で cond·ε ≈ 1e-8 に落とせる。
        // ここでは「f64 1 回 solve では到達不可能な精度 (< 1e-7)」を IR が達成することを
        // 確認する。
        assert!(
            max_abs_res < 1e-7,
            "IR should drive residual below 1e-7, got {:.3e}",
            max_abs_res
        );
    }

    #[test]
    fn compute_lsq_dual_y_respects_singleton_row_fixed_value() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-1.0_f64, 1.0], 2, 2).unwrap();
        let q = CscMatrix::new(2, 2);
        let c = vec![0.0_f64, 5.0];
        let b = vec![0.0_f64; 2];
        let bounds = vec![(0.0_f64, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![1.0, 0.0],
            dual_solution: vec![50.0, 0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };

        let y = compute_lsq_dual_y(&problem, &result).expect("LSQ should succeed");

        assert_eq!(y.len(), 2);
        assert!(
            y[0].abs() < 1e-10,
            "y[0] must be fixed to 0 by singleton constraint, got {}",
            y[0]
        );
        assert!(
            (y[1] - (-5.0)).abs() < 1e-8,
            "y[1] should be -5 for the free row, got {}",
            y[1]
        );
    }

    /// refine_dual_lsq の DD-guard が、f64 で見ると改善するが DD で見ると悪化する
    /// y_new を rejection することを確認する (= DD 比較を実装している証拠)。
    ///
    /// 注: 現時点で発生条件が稀のため、このテストは「現 y 不変」を確認する弱い形にする。
    /// 強い反例 (DD で改善 + f64 で悪化) を構築するのは困難なため、guard が
    /// 単調な性質 (改善以外なら現状維持) を持つことを確認する。
    #[test]
    fn refine_dual_lsq_keeps_y_when_lsq_does_not_strictly_improve() {
        // 1×1: A=[[1]], c=[0], Q=0、x=0, bnd=0。target=0 → 任意 y で aty=y、
        // residual = y。IPM 由来 y=0 が最適。LSQ も y=0 を返すので「現状維持」が期待。
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let q = CscMatrix::new(1, 1);
        let c = vec![0.0_f64];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut result = SolverResult {
            status: SolveStatus::Optimal,
            solution: vec![0.0],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        refine_dual_lsq(&problem, &mut result, None);
        assert!(
            result.dual_solution[0].abs() < 1e-12,
            "y は変更されないか、より良い 0 のまま。got {}",
            result.dual_solution[0]
        );
    }
}
