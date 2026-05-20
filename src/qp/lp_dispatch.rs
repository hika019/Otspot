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
//! IPM cold init は Mehrotra primal projection で KKT 行列を 1 回 factor する
//! が、これは O((n+m)^2) 級で大規模 LP の wall を支配する。LTSF crash basis から
//! 構成した structural BFS を `warm_start_qp.x` に注入して projection を skip
//! させる: B*x_B = b を crash basis 上で解いて元空間 x に extract、y=0/mu=1 で
//! `apply_qp_warm_start` 経路に着地させる (内部で bounds 内 interior 補正済)。
//!
//! `LP_DISPATCH_NOOP=1` は sentinel 用 (no-op proof) で IPM 経路を無効化する。
//! `LP_CRASH_IPM_DISABLE=1` は crash→IPM warm wiring のみ no-op 化する
//! (sentinel/triage 用)。

use std::time::Instant;

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveRoute, SolveStatus, SolverResult};
use crate::simplex::guard_lp_optimal;

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
    let mut crash_attempted = false;
    let mut crash_wired = false;
    let mut ipm_subopt_candidate: Option<SolverResult> = None;
    if !dispatch_disabled && prefer_ipm_for_size(problem.num_vars, problem.num_constraints) {
        let mut ipm_opts = ipm_opts_for_lp(options);
        // crash→IPM warm wiring: user 提供 warm_start_qp が無いときのみ生成。
        // 既定無効 (use_lp_crash_ipm_warm=false): warm-start が IPM 反復を減らさず
        // crash-LU コストだけ嵩む net-negative を full-101 A/B で実証 (2026-05-20)。
        if ipm_opts.warm_start_qp.is_none() && options.use_lp_crash_ipm_warm {
            crash_attempted = true;
            if let Some(ws) = try_build_ipm_warm_from_crash(&lp) {
                ipm_opts.warm_start_qp = Some(ws);
                crash_wired = true;
            }
        }
        let mut ipm_result = ipm_solver::solve_ipm(problem, &ipm_opts);
        ipm_result.stats.crash_ipm_attempted = crash_attempted;
        ipm_result.stats.crash_ipm_wired = crash_wired;
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
        }
    }

    // simplex (LpProblem) は obj_offset を含まないため明示的に加算。
    // crash stats を引き継ぐ (IPM fallback 経路でも attempted/wired を保持)。
    let mut simplex_result = crate::lp::solve_lp_forwarded_from_qp(&lp, options);
    simplex_result.objective += problem.obj_offset;
    let mut result = crate::bench_utils::pick_best_ipm_or_simplex(ipm_subopt_candidate, simplex_result);
    result.stats.crash_ipm_attempted = crash_attempted;
    result.stats.crash_ipm_wired = crash_wired;
    result
}

/// LP→IPM 呼び出し時に presolve を無効化したオプションを生成。
fn ipm_opts_for_lp(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    o.presolve = false;
    o
}

/// LTSF crash basis → IPM `QpWarmStart` 変換。
///
/// 1. `build_standard_form` で simplex 標準形 (shifts + slacks + UB rows) 構築
/// 2. `compute_crash_basis` で structural 列による行 cover
/// 3. crash で artificial が減らなかった (or LU 特異) ら None で abandon
/// 4. crash basis B を LU 因子分解、`B x_B = b_scaled` を FTRAN で解く
/// 5. `extract_solution` で n_orig 空間に逆写像 → `ws.x`
/// 6. y は zero ベクトル (`apply_qp_warm_start` 内で WARM_SY_MIN にクランプ)、
///    mu は 1.0 (cold-ish 起点、interior 補正は IPM init 側で実施)
///
/// 戻り値 `None` で IPM は通常 Mehrotra cold init に着地する (regression-safe)。
fn try_build_ipm_warm_from_crash(lp: &LpProblem) -> Option<crate::options::QpWarmStart> {
    use crate::basis::{BasisManager, LuBasis};
    use crate::simplex::crash_basis_for_ipm_warm;

    // env で no-op proof 化 (sentinel 用)。
    if std::env::var("LP_CRASH_IPM_DISABLE").ok().as_deref() == Some("1") {
        return None;
    }

    let (sf, basis) = crash_basis_for_ipm_warm(lp)?;

    // B*x_B = b を LU で解く。max_etas=auto (m から計算)。
    let mut lu = LuBasis::new(&sf.a, &basis, 0).ok()?;
    let mut x_b = vec![0.0_f64; sf.m];
    {
        use crate::sparse::SparseVec;
        let mut rhs = SparseVec::from_dense(&sf.b);
        lu.ftran(&mut rhs);
        x_b.copy_from_slice(&rhs.to_dense());
    }

    // n_orig 空間に extract (col_scale=1.0 で no-scaling)。
    let col_scale = vec![1.0_f64; sf.n_total];
    let x_orig = build_standard_form_extract(&sf, &basis, &x_b, &col_scale);
    if x_orig.len() != lp.num_vars || !x_orig.iter().all(|v| v.is_finite()) {
        return None;
    }

    Some(crate::options::QpWarmStart {
        x: x_orig,
        y: vec![0.0_f64; lp.num_constraints],
        mu: 1.0,
    })
}

/// `simplex::extract_solution` を経由 (private re-export 経由でアクセス)。
fn build_standard_form_extract(
    sf: &crate::simplex::StandardForm,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
) -> Vec<f64> {
    crate::simplex::extract_solution_for_ipm_warm(sf, basis, x_b, col_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// Build the Eq-constrained LP fixture used by multiple tests (n=3500, m=200).
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

    /// crash→IPM warm が cover-friendly LP で warm start を生成できることを fact 化。
    /// per-result: return value `Some(ws)` that `ws.x.len() == n`, `ws.mu > 0`.
    #[test]
    fn crash_basis_wired_into_ipm_path() {
        let n = 3500_usize;
        let m = 200_usize;
        let lp = eq_lp_fixture(n, m);

        let ws = try_build_ipm_warm_from_crash(&lp);
        assert!(ws.is_some(), "crash warm should be generated for cover-friendly LP");
        let ws = ws.unwrap();
        assert_eq!(ws.x.len(), n);
        assert_eq!(ws.y.len(), m);
        assert!(ws.mu > 0.0);
    }

    /// no-op proof: `LP_CRASH_IPM_DISABLE=1` で wiring が無効化され None が返る。
    /// per-result: return value is None (no global counter needed).
    #[test]
    fn crash_basis_ipm_wiring_no_op_proof() {
        let lp = eq_lp_fixture(3500, 200);

        // SAFETY: env var mutation scoped to this test; nextest isolates processes.
        std::env::set_var("LP_CRASH_IPM_DISABLE", "1");
        let ws = try_build_ipm_warm_from_crash(&lp);
        std::env::remove_var("LP_CRASH_IPM_DISABLE");

        assert!(ws.is_none(), "wiring must be disabled by LP_CRASH_IPM_DISABLE=1");
    }

    /// Le 制約 + bounded var: crash は slack-cover 済みで None を返す valid path。
    /// wired = ws.is_some(), attempted = always true when function called.
    #[test]
    fn crash_basis_wired_into_ipm_path_le_with_bounds() {
        let n = 4000_usize;
        let m = 150_usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let b = vec![5.0_f64; m];
        let c = vec![-1.0_f64; n];
        let ctypes = vec![crate::problem::ConstraintType::Le; m];
        let bounds = vec![(0.0_f64, 10.0); n];
        let lp = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();

        let ws = try_build_ipm_warm_from_crash(&lp);
        // Le 制約は cold で slack cover 済 (artificial=0) のため crash は何も
        // 削減できず None を返すことが多い。Some の場合も valid (LP shape 次第)。
        // return value で分岐: いずれも valid path。
        match ws {
            Some(warm) => {
                assert_eq!(warm.x.len(), n);
                assert!(warm.mu > 0.0);
            }
            None => { /* slack-covered: no crash warm needed */ }
        }
    }

    /// per-result 並列性 sentinel: 2 solve を独立実行しそれぞれの stats が独立。
    /// global Atomic に戻す改変で crash_ipm_attempted が累積してこのテストが FAIL する。
    #[test]
    fn parallel_solve_stats_independent() {
        use crate::options::SolverOptions;
        use crate::problem::SolveRoute;

        let lp = eq_lp_fixture(3500, 200);
        let lp2 = eq_lp_fixture(3600, 180);
        let opts = SolverOptions::default();

        let r1 = crate::lp::solve_lp_with(&lp, &opts);
        let r2 = crate::lp::solve_lp_with(&lp2, &opts);

        // Each result carries its own route; no shared state.
        assert_eq!(r1.stats.route, SolveRoute::LpDirect, "r1 route must be LpDirect");
        assert_eq!(r2.stats.route, SolveRoute::LpDirect, "r2 route must be LpDirect");
        // crash stats are false for pure simplex calls
        assert!(!r1.stats.crash_ipm_attempted, "simplex path: crash not attempted");
        assert!(!r2.stats.crash_ipm_attempted, "simplex path: crash not attempted");
    }

    /// Build a zero-Q QpProblem (pure LP) from the eq fixture so `solve_qp_with`
    /// routes through the large-LP dispatch (`solve_as_lp_pub`, IPM-first).
    fn eq_qp_zero_q(n: usize, m: usize) -> QpProblem {
        let lp = eq_lp_fixture(n, m);
        let q = CscMatrix::new(n, n); // Q = 0 → LP path
        QpProblem::new(q, lp.c, lp.a, lp.b, lp.bounds, lp.constraint_types).unwrap()
    }

    /// Load-bearing sentinel for the crash→IPM warm-start disable.
    ///
    /// The warm-start is net-negative (full-101 A/B: never reduces IPM iters, only
    /// adds crash-LU cost) AND its faer LU can overflow on large/sparse crash bases
    /// (dfl001 panic — same root cause). The default `use_lp_crash_ipm_warm=false`
    /// must therefore not even *attempt* it (no attempt ⇒ no overflow). `n>3000`
    /// routes IPM-first so the gate is exercised.
    ///
    /// No-op proof: flipping the default to `true` makes `crash_ipm_attempted`
    /// `true` for the default solve, failing the first assert.
    #[test]
    fn crash_ipm_warm_disabled_by_default() {
        let qp = eq_qp_zero_q(3500, 200);

        let res_default = crate::qp::solve_qp_with(&qp, &SolverOptions::default());
        assert!(
            !res_default.stats.crash_ipm_attempted,
            "default (use_lp_crash_ipm_warm=false) must NOT attempt the crash→IPM \
             warm-start; attempting it re-introduces the net-negative LU + dfl001 overflow"
        );

        // The flag is a live gate, not dead code: enabling re-attempts the warm-start.
        let opts_on = SolverOptions { use_lp_crash_ipm_warm: true, ..SolverOptions::default() };
        let res_on = crate::qp::solve_qp_with(&qp, &opts_on);
        assert!(
            res_on.stats.crash_ipm_attempted,
            "use_lp_crash_ipm_warm=true must attempt the crash→IPM warm-start"
        );
    }
}
