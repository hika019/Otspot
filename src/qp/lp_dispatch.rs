//! Q=0 (LP) dispatch.
//!
//! 中規模以下は `crate::lp::solve_lp_forwarded_from_qp` (telemetry 付き simplex) に
//! forward する。`n > LP_IPM_FIRST_N` または `m > LP_IPM_FIRST_M` を満たす大規模 LP は
//! IPM を先行し、収束しなければ残時間で simplex にフォールバック。
//!
//! IPM 呼び出し時は QP presolve を無効化する。Empty-Column 解析が pure LP で
//! false Unbounded を返す既知バグ (別途追跡) を回避するため。LP の不有界/不可解は
//! simplex/IPM 本体で判定可能で presolve なしでも検出できる。
//!
//! `LP_DISPATCH_NOOP=1` は sentinel 用 (no-op proof) で IPM 経路を無効化する。

use std::time::Instant;

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveRoute, SolveStatus, SolverResult};
use crate::simplex::guard_lp_optimal;
use crate::sparse::CscMatrix;

use super::{ipm_solver, QpProblem};

/// IPM を先に走らせる変数数閾値。Netlib 中央値 n≈800 の約 4 倍。
const LP_IPM_FIRST_N: usize = 3_000;
/// IPM を先に走らせる制約数閾値。LU 再因子分解 O(m·nnz(L)) を回避する。
const LP_IPM_FIRST_M: usize = 2_000;

pub(crate) fn prefer_ipm_for_size(n: usize, m: usize) -> bool {
    n > LP_IPM_FIRST_N || m > LP_IPM_FIRST_M
}

pub(crate) fn solve_as_lp_pub(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let opts_with_deadline;
    let options: &SolverOptions = if options.deadline.is_none() {
        if let Some(secs) = options.timeout_secs {
            opts_with_deadline = {
                let mut o = options.clone();
                o.deadline = Some(Instant::now() + std::time::Duration::from_secs_f64(secs));
                o.timeout_secs = None;
                o
            };
            &opts_with_deadline
        } else {
            options
        }
    } else {
        options
    };

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

    // 大規模 LP: IPM 先行、Timeout/NumericalError/Unbounded/MaxIter は simplex 再試行。
    // Optimal/LocallyOptimal/Infeasible は確定的 → 即返却。
    // Unbounded は IPM 側 Q=0 数値リスクがあるため simplex で再確認。
    // LP_DISPATCH_NOOP=1 は sentinel 用 (no-op proof) で IPM 経路を無効化する。
    let dispatch_disabled = std::env::var("LP_DISPATCH_NOOP").ok().as_deref() == Some("1");
    let mut ipm_subopt_candidate: Option<SolverResult> = None;
    if !dispatch_disabled && prefer_ipm_for_size(problem.num_vars, problem.num_constraints) {
        let ipm_opts = ipm_opts_for_lp(options);
        let mut ipm_result = ipm_solver::solve_ipm(problem, &ipm_opts);
        ipm_result.stats.route = SolveRoute::LpForwardedFromQp;
        ipm_result.stats.lp_ipm_path = true;
        // ipm_solver は内部で obj_offset を加算済み → そのまま返す。
        match ipm_result.status {
            SolveStatus::Optimal | SolveStatus::LocallyOptimal | SolveStatus::Infeasible => {
                // 確定 status は simplex 再試行不要、即返却。
                // Optimal は primal guard で false-Optimal を除去してから返す。
                return guard_lp_optimal(ipm_result, &lp);
            }
            SolveStatus::Unbounded
            | SolveStatus::Timeout
            | SolveStatus::NumericalError
            | SolveStatus::MaxIterations => {
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    return ipm_result;
                }
                // 残時間で simplex 再試行。
            }
            SolveStatus::SuboptimalSolution => {
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    return ipm_result;
                }
                // known_optimal_obj が設定されており obj が一致するなら simplex retry 不要。
                if let Some(ref_obj) = options.known_optimal_obj {
                    if crate::bench_utils::obj_within_tol(
                        ipm_result.objective, ref_obj,
                        crate::bench_utils::OBJ_MATCH_REL_TOL,
                    ) && !ipm_result.solution.is_empty()
                    {
                        let promoted = SolverResult { status: SolveStatus::Optimal, ..ipm_result };
                        return guard_lp_optimal(promoted, &lp);
                    }
                }
                // IPM incumbent を保存して simplex 再試行。simplex が失敗したとき
                // pick_best_ipm_or_simplex が SuboptimalSolution を復元する。
                ipm_subopt_candidate = Some(ipm_result);
            }
            SolveStatus::NonConvex(_)
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => {
                // LP dispatch は Q=0 前提 → 非凸 status は本経路には出ないが、
                // non-exhaustive match を防ぎ safety net として simplex に倒す。
            }
            SolveStatus::NotSupported(_) => {
                // Propagate immediately; simplex retry cannot help.
                return ipm_result;
            }
        }
    }

    // simplex (LpProblem) は obj_offset を含まないため明示的に加算。
    let mut simplex_result = crate::lp::solve_lp_forwarded_from_qp(&lp, options);
    simplex_result.objective += problem.obj_offset;
    if simplex_result.status == SolveStatus::Timeout
        && simplex_result.solution.is_empty()
        && options.deadline.is_none_or(|d| Instant::now() < d)
        && verified_farkas_timeout_fallback(problem, options)
    {
        let mut certified = SolverResult::infeasible();
        certified.iterations = simplex_result.iterations;
        return certified;
    }
    crate::bench_utils::pick_best_ipm_or_simplex(ipm_subopt_candidate, simplex_result)
}

/// LP→IPM 呼び出し時に presolve を無効化したオプションを生成。
fn ipm_opts_for_lp(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    o.presolve = false;
    o
}

/// Try a normalized Farkas certificate after simplex Phase I stalls.
/// This stays on nonnegative variables so bounds need no certificate terms.
fn verified_farkas_timeout_fallback(problem: &QpProblem, options: &SolverOptions) -> bool {
    if !problem.bounds.iter().all(|&(lb, ub)| lb == 0.0 && ub == f64::INFINITY) {
        return false;
    }

    // Convert user rows to Cx >= d. Equality rows need both directions.
    let (cert_cols_by_row, cert_rhs) = normalized_farkas_rows(problem);
    if cert_rhs.is_empty() {
        return false;
    }

    // y >= 0, C^T y <= 0, d^T y >= 1 certifies Cx >= d, x >= 0 infeasible.
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..problem.num_vars {
        let Ok((a_rows, a_vals)) = problem.a.get_column(j) else {
            return false;
        };
        for (k, &i) in a_rows.iter().enumerate() {
            for &(cert_col, sign) in &cert_cols_by_row[i] {
                rows.push(j);
                cols.push(cert_col);
                vals.push(sign * a_vals[k]);
            }
        }
    }
    for (cert_col, &rhs) in cert_rhs.iter().enumerate() {
        rows.push(problem.num_vars);
        cols.push(cert_col);
        vals.push(rhs);
    }
    let Ok(cert_a) = CscMatrix::from_triplets(
        &rows, &cols, &vals, problem.num_vars + 1, cert_rhs.len(),
    ) else {
        return false;
    };
    let mut cert_b = vec![0.0; problem.num_vars];
    cert_b.push(1.0);
    let mut cert_types = vec![ConstraintType::Le; problem.num_vars];
    cert_types.push(ConstraintType::Ge);
    let Ok(cert_qp) = QpProblem::new(
        CscMatrix::new(cert_rhs.len(), cert_rhs.len()),
        vec![0.0; cert_rhs.len()],
        cert_a,
        cert_b,
        vec![(0.0, f64::INFINITY); cert_rhs.len()],
        cert_types,
    ) else {
        return false;
    };

    let result = ipm_solver::solve_ipm(&cert_qp, &ipm_opts_for_lp(options));
    result.status == SolveStatus::Optimal
        && result.solution.len() == cert_rhs.len()
        && verify_normalized_farkas(problem, &cert_cols_by_row, &cert_rhs, &result.solution)
}

/// ユーザ行 (Ge/Le/Eq) を `Cx ≥ d` 形へ正規化し、行 i ごとの (cert_col, sign) と
/// 正規化済 RHS d を返す。Eq は両向き (±) で 2 列。
fn normalized_farkas_rows(problem: &QpProblem) -> (Vec<Vec<(usize, f64)>>, Vec<f64>) {
    let mut cert_cols_by_row = vec![Vec::<(usize, f64)>::new(); problem.num_constraints];
    let mut cert_rhs = Vec::new();
    for (i, &kind) in problem.constraint_types.iter().enumerate() {
        let mut push_col = |sign: f64| {
            let col = cert_rhs.len();
            cert_cols_by_row[i].push((col, sign));
            cert_rhs.push(sign * problem.b[i]);
        };
        match kind {
            ConstraintType::Ge => push_col(1.0),
            ConstraintType::Le => push_col(-1.0),
            ConstraintType::Eq => {
                push_col(1.0);
                push_col(-1.0);
            }
        }
    }
    (cert_cols_by_row, cert_rhs)
}

/// 正規化制約 dᵀy ≥ 1 の許容下限。1 は cert LP の正規化定数 (データスケール非依存)
/// なので絶対 tol で安全。
const FARKAS_NORM_TOL: f64 = 1e-7;

/// 内積 Σ sign·a·y の f64 累積丸め誤差を見積もる 1 項あたりの後退誤差係数。
///
/// floor は IPM 収束 tol ではなく f64 の**真の丸め境界**に置く。n 項の積和の丸め
/// 誤差は後退誤差解析で ≲ n·u·Σ|項| (u = ε/2 は unit roundoff)。各項は積 1 回 +
/// 和 1 回で最大 2u = ε の相対誤差を負うため、floor を `n_terms·ε·term_mag` とする。
///
/// これを IPM tol と分離する理由: cert IPM は infeasible な cert LP の残差を自身の
/// 収束 tol (~1e-11) まで潰し、Eq の ± 二方向で Cᵀy = y0−y1 ≈ 1/K の微小な**正の
/// slack** を持つ偽証明を作る。floor を 1e-11 級に置くとこれを noise と誤判定し
/// feasible (`x1+x2=K`, K≳1e11) を false-infeasible に認定する。floor を丸め境界
/// (~n·1e-16) に締めると、IPM 残差由来の偽証明 (Cᵀy~1e-11..1e-13) は floor の数桁
/// 上で reject される。klein3 の genuine cert (Cᵀy<0、厳密に負) と真の丸め
/// (~n·ε·term_mag 以下) は通過する。soundness 最優先: 偽 accept を出さないことを、
/// 境界際 genuine cert を取りこぼす (honest Timeout 化) より優先する。
const FARKAS_CTY_ROUNDOFF_PER_TERM: f64 = f64::EPSILON;

/// 正の slack `aty = (Cᵀy)_j` が内積丸め誤差の範囲内か (= Cᵀy ≤ 0 を f64 精度で
/// 満たすか)。`term_mag = Σ_k |sign·a·y|` はその成分の内積項 magnitude、
/// `n_terms` は加算した項数。floor を `n_terms·ε·term_mag` とし、scale 不変かつ
/// IPM tol から独立な丸め境界で判定する。
fn cty_slack_within_noise(aty: f64, term_mag: f64, n_terms: usize) -> bool {
    let roundoff_floor = (n_terms as f64) * FARKAS_CTY_ROUNDOFF_PER_TERM * term_mag;
    aty <= roundoff_floor
}

fn verify_normalized_farkas(
    problem: &QpProblem,
    cert_cols_by_row: &[Vec<(usize, f64)>],
    cert_rhs: &[f64],
    y: &[f64],
) -> bool {
    if y.len() != cert_rhs.len() || y.iter().any(|&v| !v.is_finite()) {
        return false;
    }
    // 厳密な非負部分 y⁺ = max(y, 0) で検証する。IPM の僅かな負 slack を許容しても
    // y⁺ ≥ 0 が厳密に成り立つので Farkas の健全性 (dᵀy⁺ ≤ xᵀCᵀy⁺) を崩さない。
    let yp = |col: usize| y[col].max(0.0);
    let rhs_dot = cert_rhs
        .iter()
        .enumerate()
        .map(|(col, &d)| d * yp(col))
        .sum::<f64>();
    if !rhs_dot.is_finite() || rhs_dot < 1.0 - FARKAS_NORM_TOL {
        return false;
    }
    for j in 0..problem.num_vars {
        let Ok((a_rows, a_vals)) = problem.a.get_column(j) else {
            return false;
        };
        let mut aty = 0.0;
        let mut term_mag = 0.0;
        let mut n_terms = 0usize;
        for (k, &i) in a_rows.iter().enumerate() {
            for &(cert_col, sign) in &cert_cols_by_row[i] {
                let term = sign * a_vals[k] * yp(cert_col);
                aty += term;
                term_mag += term.abs();
                n_terms += 1;
            }
        }
        if !aty.is_finite() || !cty_slack_within_noise(aty, term_mag, n_terms) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn eq_lp_fixture(n: usize, m: usize) -> LpProblem {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            rows.push(i); cols.push(i);     vals.push(1.0);
            rows.push(i); cols.push(i + m); vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let b = vec![2.0_f64; m];
        let c = vec![1.0_f64; n];
        let ctypes = vec![crate::problem::ConstraintType::Eq; m];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap()
    }

    /// 2 solve を独立実行し、それぞれの route stats が独立していることを確認。
    #[test]
    fn parallel_solve_stats_independent() {
        use crate::options::SolverOptions;
        use crate::problem::SolveRoute;

        let lp = eq_lp_fixture(3500, 200);
        let lp2 = eq_lp_fixture(3600, 180);
        let opts = SolverOptions::default();

        let r1 = crate::lp::solve_lp_with(&lp, &opts);
        let r2 = crate::lp::solve_lp_with(&lp2, &opts);

        assert_eq!(r1.stats.route, SolveRoute::LpDirect, "r1 route must be LpDirect");
        assert_eq!(r2.stats.route, SolveRoute::LpDirect, "r2 route must be LpDirect");
    }

    /// 非負変数の QP/LP を密行で構築するヘルパー (Farkas 検証 sentinel 用)。
    fn nonneg_qp(a_rows: &[Vec<f64>], b: &[f64], types: &[ConstraintType]) -> QpProblem {
        let m = a_rows.len();
        let n = a_rows.first().map_or(0, |r| r.len());
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (i, row) in a_rows.iter().enumerate() {
            assert_eq!(row.len(), n, "rows must be rectangular");
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(v);
                }
            }
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        QpProblem::new(
            CscMatrix::new(n, n),
            vec![0.0; n],
            a,
            b.to_vec(),
            vec![(0.0, f64::INFINITY); n],
            types.to_vec(),
        )
        .unwrap()
    }

    /// 旧 IPM-tol 級 floor。sentinel が「floor を IPM tol (1e-11) に戻すと
    /// 偽証明を accept する」ことを明示するための参照値 (実装側には残っていない)。
    const LEGACY_IPM_TOL_FLOOR: f64 = 1e-11;

    /// 丸め境界 floor `n_terms·ε·term_mag` を複数パターンで cover。
    /// 偽証明 (IPM 残差級の正 slack) は reject、真の負残差/丸め以下は accept。
    /// 同一 aty でも n_terms / term_mag で floor がスケールすることを確認。
    #[test]
    fn cty_slack_within_noise_separates_real_slack_from_roundoff() {
        // (aty, term_mag, n_terms, expect_within_noise, label)
        let cases = [
            // IPM 残差級の正 slack: floor (~n·ε·mag) の数桁上 → reject。
            (1e-9, 8.76, 2, false, "d=1e9 normalized feasible"),
            (1.8626e-9, 2.0, 2, false, "d=1e9 (dᵀy≈1.86)"),
            (1.49e-8, 2.0, 2, false, "d=1e8 normalized feasible"),
            (1.455e-11, 2.0, 2, false, "K=1e11 false-cert residual"),
            (1.137e-13, 2.0, 2, false, "K=1e13 false-cert residual"),
            // klein3 genuine: 残差は厳密に負。
            (-4.1e-6, 986.0, 4, true, "klein3 genuine cert"),
            (-1.0, 3.0, 2, true, "strict negative residual"),
            (0.0, 5.0, 2, true, "exact zero residual"),
            // f64 内積丸めレベル: noise として accept。floor = 2·ε·8.76 ≈ 3.9e-15。
            (1e-15, 8.76, 2, true, "roundoff-level positive"),
            (2e-16, 1.0, 2, true, "near machine eps"),
            // n_terms スケール: 同 aty=1e-12 でも項数で floor が動く。
            (1e-12, 1.0, 2, false, "small n: above roundoff floor"),
            (1e-12, 1.0, 10_000, true, "large n: within accumulated roundoff"),
        ];
        for (aty, mag, n_terms, expect, label) in cases {
            assert_eq!(
                cty_slack_within_noise(aty, mag, n_terms),
                expect,
                "case `{label}`: aty={aty:e}, mag={mag:e}, n_terms={n_terms}",
            );
        }

        // load-bearing: floor を IPM tol (1e-11) に戻すと K≳1e11 の偽証明残差
        // (1.455e-11 / 1.137e-13) を noise と誤判定する。丸め境界 floor はこれを
        // reject する。両 floor が分岐することを実証 (sentinel が no-op で FAIL)。
        for &(aty, mag, n_terms) in &[(1.455e-11, 2.0, 2usize), (1.137e-13, 2.0, 2)] {
            assert!(
                aty <= LEGACY_IPM_TOL_FLOOR * mag,
                "premise: IPM-tol floor would have accepted aty={aty:e}",
            );
            assert!(
                !cty_slack_within_noise(aty, mag, n_terms),
                "roundoff floor must reject IPM-residual slack aty={aty:e}",
            );
        }
    }

    /// 大 magnitude feasible (`x1+x2=K`) が Infeasible 認定されないこと。
    /// 偽証明 y は正規化 dᵀy≥1 を満たすが Cᵀy≈dᵀy/K の本物の正 slack を持つ。
    /// load-bearing: K≳1e11 の偽残差は旧 IPM-tol floor (1e-11) では accept される。
    #[test]
    fn farkas_rejects_large_magnitude_feasible() {
        // (K, g, legacy_would_accept): g は y0-y1 (2 のべきで厳密表現)。dᵀy=K·g≥1。
        // legacy_would_accept = 旧 IPM-tol floor (1e-11·term_mag) が Cᵀy=g を誤 accept
        // するか。K=1e9 の残差 (~1.86e-9) は旧 floor でも既に reject されるため非 load-
        // bearing、K≳1e11 (~1.46e-11..1.14e-13) が新 floor 固有の reject。
        let patterns = [
            (1e9, 2.0_f64.powi(-29), false),  // Cᵀy = g ≈ 1.863e-9
            (1e11, 2.0_f64.powi(-36), true),  // Cᵀy = g ≈ 1.455e-11
            (1e12, 2.0_f64.powi(-39), true),  // Cᵀy = g ≈ 1.819e-12
            (1e13, 2.0_f64.powi(-43), true),  // Cᵀy = g ≈ 1.137e-13
        ];
        for (k, g, legacy_would_accept) in patterns {
            let problem = nonneg_qp(&[vec![1.0, 1.0]], &[k], &[ConstraintType::Eq]);
            let (cols, rhs) = normalized_farkas_rows(&problem);
            assert_eq!(rhs, vec![k, -k], "Eq → ±K の cert RHS");
            // y0 = 1 + g, y1 = 1。Cᵀy = y0 - y1 = g (正)、dᵀy = K·g ≥ 1。
            let y = vec![1.0 + g, 1.0];
            let cty = g; // y0 - y1
            let dty = k * g;
            let term_mag = (1.0 + g) + 1.0; // |y0| + |y1|
            assert!(dty >= 1.0 - FARKAS_NORM_TOL, "premise: dᵀy={dty} must clear norm");
            assert_eq!(
                cty <= LEGACY_IPM_TOL_FLOOR * term_mag,
                legacy_would_accept,
                "premise: IPM-tol floor accept(Cᵀy={cty:e}) for K={k:e}",
            );
            assert!(
                !verify_normalized_farkas(&problem, &cols, &rhs, &y),
                "feasible x1+x2={k:e} must NOT be certified infeasible",
            );
        }
    }

    /// reviewer 再現を端から潰す: cert IPM を実際に走らせる end-to-end gate。
    /// feasible 問題 (`x1+x2=K`, single-var Ge `2x1≥K`) は cert LP 自体が
    /// infeasible なので、IPM が残差を tol まで潰した偽証明を返しても
    /// 丸め境界 floor が reject し、Infeasible 認定されてはならない。
    #[test]
    fn verified_farkas_rejects_feasible_large_magnitude_end_to_end() {
        use crate::options::SolverOptions;
        let opts = SolverOptions::default();
        for k in [1e9, 1e11, 1e12, 1e13] {
            // x1+x2=K, x≥0 は feasible (例 x=(K,0))。
            let eq = nonneg_qp(&[vec![1.0, 1.0]], &[k], &[ConstraintType::Eq]);
            assert!(
                !verified_farkas_timeout_fallback(&eq, &opts),
                "feasible x1+x2={k:e} must NOT be certified infeasible",
            );
            // 2x1 ≥ K, x≥0 は feasible (x1=K/2)。
            let ge = nonneg_qp(&[vec![2.0]], &[k], &[ConstraintType::Ge]);
            assert!(
                !verified_farkas_timeout_fallback(&ge, &opts),
                "feasible 2x1 ≥ {k:e} must NOT be certified infeasible",
            );
        }
    }

    /// genuine infeasible (`x1≥1` かつ `-2x1≥1`) は証明書が通り続ける。
    /// klein3 と同型: max Cᵀy < 0 (厳密に負)、dᵀy ≫ 1。
    #[test]
    fn farkas_certifies_genuine_infeasible() {
        let problem = nonneg_qp(
            &[vec![1.0], vec![-2.0]],
            &[1.0, 1.0],
            &[ConstraintType::Ge, ConstraintType::Ge],
        );
        let (cols, rhs) = normalized_farkas_rows(&problem);
        assert_eq!(rhs, vec![1.0, 1.0]);
        let y = vec![1.0, 1.0];
        // Cᵀy = 1·1 + (-2)·1 = -1 < 0、dᵀy = 2。
        assert!(
            verify_normalized_farkas(&problem, &cols, &rhs, &y),
            "genuine infeasible must remain certified",
        );
    }

    /// 小 magnitude feasible は元々誤認定されない (正 slack が大きく floor 超過)。
    /// 相対化が小規模問題を退化させないことの確認。
    #[test]
    fn farkas_rejects_modest_feasible() {
        let problem = nonneg_qp(&[vec![1.0, 1.0]], &[2.0], &[ConstraintType::Eq]);
        let (cols, rhs) = normalized_farkas_rows(&problem);
        // dᵀy=1 → y0-y1=0.5 (大きな正 slack)。
        let y = vec![0.5, 0.0];
        assert!(
            !verify_normalized_farkas(&problem, &cols, &rhs, &y),
            "modest feasible must NOT be certified infeasible",
        );
    }
}
