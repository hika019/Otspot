//! LP-specific entry point.
//!
//! Splits LP from the QP `Q.is_zero` dispatch so that LP-only paths
//! (simplex, future IPM-first / crash / postsolve) are owned by this
//! module. `solve_qp_with(Q=0)` keeps backward compat by forwarding
//! here; the two call sites are distinguishable via `SolverResult.stats.route`.

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveRoute, SolveStatus, SolverResult};

/// Solve an LP directly. Sets `result.stats.route = SolveRoute::LpDirect`.
///
/// Returns [`SolveStatus::NumericalError`] if `options` fails validation;
/// validation is performed by the underlying `simplex::solve_with`.
pub fn solve_lp_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    // Presolve-independent bound-consistency guard: an empty box (lb > ub) is
    // trivially infeasible and is reported here so correctness does not rely on
    // presolve running or on the simplex-internal empty-box check downstream.
    if crate::problem::first_infeasible_bound(&problem.bounds).is_some() {
        let mut result = SolverResult::infeasible();
        result.stats.route = SolveRoute::LpDirect;
        return result;
    }
    // Materialize timeout_secs → deadline HERE so the deadline_triggered clock
    // check below sees the same deadline the solve actually ran against
    // (a raw timeout_secs-only option set is clock-blind at this layer).
    let materialized = options.materialize_deadline();
    let options = materialized.as_ref().unwrap_or(options);
    let mut result = guard_lp_incumbent_claim(
        crate::simplex::solve_with(problem, options),
        problem,
        options,
    );
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        result.objective += problem.obj_offset;
    }
    result.stats.route = SolveRoute::LpDirect;
    result.stats.deadline_triggered =
        matches!(result.status, SolveStatus::Timeout) && options.external_stop_requested();
    result
}

/// LP 公開結果の最終ゲート: `SuboptimalSolution` を名乗れるのは、元問題で primal
/// feasible と検証できた点だけ。
///
/// simplex の停止経路 (`stall_status` / `honest_stall_result`) は「解ベクトルが空で
/// ないこと」だけを incumbent の条件にしていたため、eta ドリフトで bound を破った
/// 反復点や postsolve 再構成に失敗した点まで「解」として提示されうる。これは
/// [`SolveStatus::SuboptimalSolution`] の契約 (= 検証済みの feasible な点、最適性の
/// 証明のみ欠く) に反するので、検証を通らない点は品質を主張しない
/// [`SolveStatus::Stalled`] (解ベクトルが無ければ [`SolveStatus::MaxIterations`]) へ
/// 落とす。許容は postsolve 品質判定と同じ `lp_accept_primal_tol`。
///
/// `Optimal` 側の対応物は `qp::certificate::guard_lp_optimal` (証明書ゲート)。
fn guard_lp_incumbent_claim(
    mut result: SolverResult,
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    if result.status != SolveStatus::SuboptimalSolution {
        return result;
    }
    if result.solution.len() != problem.num_vars {
        result.status = SolveStatus::MaxIterations;
        return result;
    }
    let (primal_residual, bound_violation) = crate::simplex::lp_primal_residuals(problem, &result);
    if primal_residual > options.lp_accept_primal_tol()
        || bound_violation > options.lp_accept_primal_tol()
    {
        result.status = SolveStatus::Stalled;
    }
    result
}

/// LP entry from `solve_qp_with(Q=0)`. Sets `result.stats.route = SolveRoute::LpForwardedFromQp`.
pub(crate) fn solve_lp_forwarded_from_qp(
    problem: &LpProblem,
    options: &SolverOptions,
) -> SolverResult {
    let materialized = options.materialize_deadline();
    let options = materialized.as_ref().unwrap_or(options);
    let mut result = guard_lp_incumbent_claim(
        crate::simplex::solve_with(problem, options),
        problem,
        options,
    );
    if matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
    ) {
        result.objective += problem.obj_offset;
    }
    result.stats.route = SolveRoute::LpForwardedFromQp;
    result.stats.deadline_triggered =
        matches!(result.status, SolveStatus::Timeout) && options.external_stop_requested();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use otspot_num::sparse::CscMatrix;

    fn make_trivial_lp() -> LpProblem {
        // minimize x  s.t.  x <= 5,  x >= 0
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap()
    }

    /// `SuboptimalSolution` は「元問題で feasible と検証済みの点」の契約なので、
    /// bound を破った iterate は `Stalled` (品質主張なし) へ落ちる。
    ///
    /// オラクル (手計算): make_trivial_lp は x ≥ 0 / x ≤ 5。x = −3 は下界を 3 破り、
    /// 相対 bound violation は 3/(1+3) 相当で既定許容 (1e-6 級) を大きく超える。
    ///
    /// ## Sentinel (no-op-fail)
    /// `guard_lp_incumbent_claim` を素通しに戻すと `SuboptimalSolution` のままとなり FAIL。
    #[test]
    fn guard_demotes_bound_violating_incumbent_to_stalled() {
        let problem = make_trivial_lp();
        let options = SolverOptions::default();
        let claimed = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: -3.0,
            solution: vec![-3.0],
            ..Default::default()
        };
        let guarded = guard_lp_incumbent_claim(claimed, &problem, &options);
        assert_eq!(guarded.status, SolveStatus::Stalled, "{guarded:?}");
        assert_eq!(
            guarded.solution,
            vec![-3.0],
            "診断用 iterate は残す (status だけ降格する)"
        );
    }

    /// 制約 (x ≤ 5) を破った点も同様に `Stalled`。x = 9 は 4 超過。
    #[test]
    fn guard_demotes_row_violating_incumbent_to_stalled() {
        let problem = make_trivial_lp();
        let claimed = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 9.0,
            solution: vec![9.0],
            ..Default::default()
        };
        let guarded = guard_lp_incumbent_claim(claimed, &problem, &SolverOptions::default());
        assert_eq!(guarded.status, SolveStatus::Stalled, "{guarded:?}");
    }

    /// 逆方向の teeth: feasible な点を誤って降格しない (x = 2 は 0 ≤ 2 ≤ 5)。
    #[test]
    fn guard_keeps_feasible_incumbent_as_suboptimal() {
        let problem = make_trivial_lp();
        let claimed = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 2.0,
            solution: vec![2.0],
            ..Default::default()
        };
        let guarded = guard_lp_incumbent_claim(claimed, &problem, &SolverOptions::default());
        assert_eq!(
            guarded.status,
            SolveStatus::SuboptimalSolution,
            "{guarded:?}"
        );
    }

    /// 解ベクトルが無い (次元不一致) 場合は解を主張できないので `MaxIterations`。
    #[test]
    fn guard_without_solution_vector_reports_max_iterations() {
        let problem = make_trivial_lp();
        let claimed = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![],
            ..Default::default()
        };
        let guarded = guard_lp_incumbent_claim(claimed, &problem, &SolverOptions::default());
        assert_eq!(guarded.status, SolveStatus::MaxIterations, "{guarded:?}");
    }

    /// 他 status には触らない (Optimal は `guard_lp_optimal` の担当)。
    #[test]
    fn guard_leaves_non_suboptimal_status_untouched() {
        let problem = make_trivial_lp();
        for status in [
            SolveStatus::Optimal,
            SolveStatus::Timeout,
            SolveStatus::Stalled,
            SolveStatus::MaxIterations,
        ] {
            let claimed = SolverResult {
                status: status.clone(),
                objective: -3.0,
                solution: vec![-3.0], // bound 違反でも触らない
                ..Default::default()
            };
            let guarded = guard_lp_incumbent_claim(claimed, &problem, &SolverOptions::default());
            assert_eq!(guarded.status, status, "{status:?} を書き換えてはいけない");
        }
    }

    /// Timeout incumbent must include `problem.obj_offset`.
    ///
    /// Sentinel: removing `SolveStatus::Timeout` from the match in `solve_lp_with`
    /// causes `result.objective == 0.0` instead of 42.5 → FAIL.
    ///
    /// `cancel_flag = true` with `deadline = None` bypasses the pre-simplex
    /// INFINITY timeout (entry.rs only checks `deadline.is_some_and(...)`).
    /// The simplex loop's first-iteration cancel check fires → Timeout with
    /// initial BFS (x_decision = 0, c^T x = 0, sf.obj_offset = 0).
    #[test]
    fn test_lp_timeout_incumbent_includes_obj_offset() {
        use std::sync::{atomic::AtomicBool, Arc};

        let mut lp = make_trivial_lp();
        lp.obj_offset = 42.5;

        let opts = SolverOptions {
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            presolve: false,
            ..Default::default()
        };

        let result = solve_lp_with(&lp, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "cancel_flag=true must produce Timeout"
        );
        assert!(
            result.objective.is_finite(),
            "Timeout incumbent must have finite objective (not INFINITY); got {}",
            result.objective
        );
        assert!(
            (result.objective - 42.5).abs() < 1e-9,
            "Timeout incumbent must include obj_offset 42.5; got {} \
             (sentinel: removing Timeout from match yields 0.0 ≠ 42.5)",
            result.objective
        );
    }

    /// A chain LP with `Ge` constraints (forces artificial variables into the
    /// initial basis, routing through `dual_advanced`'s Big-M cold start) large
    /// enough that most artificials are still basic when `cancel_flag` fires
    /// mid-solve, matching the shape that reaches
    /// `dual_advanced::phase1::farkas_infeasibility_certified` /
    /// `primal::extract_farkas_certificate`'s per-row Farkas probe loops with
    /// `art_rows.len()` close to `m`.
    fn make_chain_lp_with_many_artificials(n: usize) -> LpProblem {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut b = Vec::new();
        for i in 0..(n - 1) {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
            rows.push(i);
            cols.push(i + 1);
            vals.push(1.0);
            b.push(((i % 5) + 1) as f64);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, n - 1, n).unwrap();
        let c: Vec<f64> = (0..n).map(|i| ((i % 7) + 1) as f64).collect();
        LpProblem::new_general(
            c,
            a,
            b,
            vec![ConstraintType::Ge; n - 1],
            vec![(0.0, 10.0); n],
            None,
        )
        .unwrap()
    }

    /// `farkas_infeasibility_certified` (`dual_advanced::phase1`) and
    /// `extract_farkas_certificate` (`primal`) each verify a Farkas certificate
    /// with a `for &row in &art_rows { ... }` loop that does one BTRAN solve
    /// plus an O(n) certificate check per remaining artificial row -- and
    /// neither checked `deadline` or `cancel_flag` inside that loop, only at
    /// entry. A preset `cancel_flag=true` (Phase 1 bails on its very first
    /// iteration, leaving nearly every artificial still basic --
    /// `art_rows.len()` close to `m`) went unnoticed until the whole O(m)
    /// probe loop finished on its own. Deterministic preset (not a delayed
    /// background thread + sleep) so the assertion isn't a race against
    /// however long the solve happens to take to reach the probe loop.
    ///
    /// n=6000 (not the smaller sizes used elsewhere in this file) because
    /// `[profile.test] opt-level = 3` (this workspace's Cargo.toml) makes the
    /// whole per-row probe cheap enough at smaller n that even the *unfixed*
    /// loop finished in well under a second -- this size was chosen by
    /// actually reverting the fix and increasing n until the regression
    /// reproduced under `cargo test`'s own profile, not just under `--profile
    /// dev`. Measured (opt-level=3, this machine): 0.05s fixed vs 1.9s
    /// reverted. Sentinel confirmed by reverting and re-running.
    ///
    /// 1.5s bound, not 0.05s (Codex PR #31 audit P3): a >30x margin, chosen
    /// over a counter-based rewrite because the preset-flag setup is already
    /// deterministic (no background thread to race) and >30x dwarfs the
    /// contention slowdown actually seen on this project's CI (~9% overshoot
    /// on a since-fixed, near-zero-margin `qp_phase2.rs` test).
    #[test]
    fn farkas_certificate_probe_loop_honors_cancel_flag_preset() {
        use std::sync::{atomic::AtomicBool, Arc};
        use std::time::{Duration, Instant};

        let lp = make_chain_lp_with_many_artificials(6000);
        let opts = SolverOptions {
            cancel_flag: Some(Arc::new(AtomicBool::new(true))),
            presolve: false,
            ..Default::default()
        };

        let t0 = Instant::now();
        let result = solve_lp_with(&lp, &opts);
        let elapsed = t0.elapsed();
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "cancel_flag=true must produce Timeout"
        );
        assert!(
            elapsed < Duration::from_millis(1500),
            "preset cancel_flag=true took {elapsed:?} to stop the solve -- \
             Farkas certificate probe loop is not honoring cancel_flag \
             (pre-fix measured 1.9s, opt-level=3, this machine)"
        );
    }

    /// Invalid options produce NumericalError via `solve_lp_with`.
    ///
    /// Validation is performed by `simplex::solve_with` (the load-bearing sentinel
    /// lives in `simplex::entry::invalid_options_rejected_at_simplex_entry`).
    #[test]
    fn invalid_options_rejected_at_lp_entry() {
        let lp = make_trivial_lp();
        let cases: &[(&str, SolverOptions)] = &[
            (
                "nan primal_tol",
                SolverOptions {
                    primal_tol: f64::NAN,
                    ..Default::default()
                },
            ),
            (
                "inf primal_tol",
                SolverOptions {
                    primal_tol: f64::INFINITY,
                    ..Default::default()
                },
            ),
            (
                "neg timeout_secs",
                SolverOptions {
                    timeout_secs: Some(-0.5),
                    ..Default::default()
                },
            ),
            (
                "zero threads",
                SolverOptions {
                    threads: 0,
                    ..Default::default()
                },
            ),
            (
                "nan dual_tol",
                SolverOptions {
                    dual_tol: f64::NAN,
                    ..Default::default()
                },
            ),
        ];
        for (label, opts) in cases {
            let result = solve_lp_with(&lp, opts);
            assert_eq!(
                result.status,
                SolveStatus::NumericalError,
                "solve_lp_with with {label} must return NumericalError"
            );
        }
    }

    fn cx(lp: &LpProblem, sol: &[f64]) -> f64 {
        lp.c.iter().zip(sol).map(|(c, x)| c * x).sum::<f64>() + lp.obj_offset
    }

    fn solve_lp_no_presolve(lp: &LpProblem) -> SolverResult {
        let opts = SolverOptions {
            presolve: false,
            ..Default::default()
        };
        solve_lp_with(lp, &opts)
    }

    /// Sentinel: an LP with an empty variable box (lb > ub) must solve to
    /// Infeasible on the direct LP entry, with presolve either ON or OFF —
    /// never a construction Err, panic, or false Optimal.
    ///
    /// Reverting the `first_infeasible_bound` guard in `solve_lp_with` leaves
    /// the presolve-ON case still Infeasible (presolve/simplex catch it), so the
    /// guard's own contribution is exercised by asserting BOTH toggles agree.
    #[test]
    fn lp_empty_box_lb_gt_ub_is_infeasible() {
        // min x  s.t.  x <= 10 (Le),  x ∈ [5, 3]  (empty box)
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(5.0, 3.0)],
            None,
        )
        .expect("lb>ub box must be ACCEPTED at construction");
        for presolve in [true, false] {
            let opts = SolverOptions {
                presolve,
                ..Default::default()
            };
            let res = solve_lp_with(&lp, &opts);
            assert_eq!(
                res.status,
                SolveStatus::Infeasible,
                "LP empty box must be Infeasible (presolve={presolve}), got {:?}",
                res.status
            );
        }
    }

    /// Reported objective must equal `c·x` of the returned solution for an LP
    /// with a NONZERO lower bound that routes through the Big-M Phase I path.
    ///
    /// `min x  s.t.  x >= 5 (Ge),  x ∈ [3, ∞)` → optimum x=5, obj=5.
    /// The Ge constraint forces artificials ⇒ `big_m_cold_start`. The standard
    /// form shifts x = 3 + x', so `sf.obj_offset = c·lb = 3`.
    ///
    /// BUG (a4200da): `phase1.rs` recomputes `obj_orig = c·solution` from the
    /// un-shifted solution (already = c·x = 5) and then ADDS `sf.obj_offset`
    /// again ⇒ reports 8. The solution (x=5) is correct; only the scalar is wrong.
    /// This test FAILS until the Big-M path stops double-adding `sf.obj_offset`.
    /// Expected after fix: reported objective == 5.
    #[test]
    fn bigm_nonzero_lb_objective_double_count() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Ge],
            vec![(3.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let res = solve_lp_no_presolve(&lp);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.solution[0] - 5.0).abs() < 1e-6,
            "solution must be x=5; got {}",
            res.solution[0]
        );
        assert!(
            (res.objective - cx(&lp, &res.solution)).abs() < 1e-6,
            "reported objective {} must equal c·x {} (Big-M path double-counts \
             sf.obj_offset = c·lb = 3 ⇒ reports 8 instead of 5)",
            res.objective,
            cx(&lp, &res.solution)
        );
    }

    /// Control: Le-only LP with nonzero lower bound routes through the Le-only
    /// cold-start path, which correctly adds `sf.obj_offset` to the SHIFTED
    /// `basic_obj`. `min x s.t. x <= 10, x ∈ [3, ∞)` → x=3, obj=3. PASSES.
    /// Sentinel for the scope of the Big-M bug (this path must stay correct).
    #[test]
    fn le_only_nonzero_lb_objective_correct() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(3.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let res = solve_lp_no_presolve(&lp);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.objective - cx(&lp, &res.solution)).abs() < 1e-6,
            "Le-only path: reported {} must equal c·x {}",
            res.objective,
            cx(&lp, &res.solution)
        );
    }

    /// Control: bounded LP (finite ub) with nonzero lower bound routes through
    /// the BFRT bounded path, which is also correct. `min x s.t. x <= 10,
    /// x ∈ [3, 8]` → x=3, obj=3. PASSES. Sentinel for the bounded path.
    #[test]
    fn bounded_nonzero_lb_objective_correct() {
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![10.0],
            vec![ConstraintType::Le],
            vec![(3.0, 8.0)],
            None,
        )
        .unwrap();
        let res = solve_lp_no_presolve(&lp);
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.objective - cx(&lp, &res.solution)).abs() < 1e-6,
            "bounded path: reported {} must equal c·x {}",
            res.objective,
            cx(&lp, &res.solution)
        );
    }

    /// Klee-Minty LP: the classic worst-case-for-simplex construction
    /// (`max sum_j 2^(n-j) x_j s.t. 2*sum_{k<i} 2^(i-k) x_k + x_i <= 5^i`,
    /// `x >= 0`), recast as a minimization of the negated objective. Forces
    /// many more pivots from a cold start than a "nice" LP of the same size
    /// — needed so a `max_iters` cap set below the natural iteration count
    /// is distinguishable from one that's simply irrelevant.
    fn klee_minty_lp(n: usize) -> LpProblem {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for k in 0..i {
                rows.push(i);
                cols.push(k);
                vals.push(2.0f64.powi((i - k + 1) as i32));
            }
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let b: Vec<f64> = (1..=n).map(|i| 5.0f64.powi(i as i32)).collect();
        let c: Vec<f64> = (1..=n).map(|j| -(2.0f64.powi((n - j) as i32))).collect();
        LpProblem::new_general(
            c,
            a,
            b,
            vec![ConstraintType::Le; n],
            vec![(0.0, f64::INFINITY); n],
            None,
        )
        .unwrap()
    }

    /// SENTINEL (P2-4/max_iters plumbing, real solver end-to-end): P2-3 in the
    /// review — `SolverOptions::max_iters` must be honored by the actual
    /// dispatch chain (`solve_lp_with` → `simplex::solve_with` → whichever
    /// simplex core `SimplexMethod::Auto` selects), not only by the
    /// low-level helpers unit-tested directly in `bounded_core::tests`.
    ///
    /// Solves the same LP twice: once uncapped to measure its natural
    /// iteration count `n`, once with `max_iters = Some(n / 2)`. The capped
    /// solve must report `iterations <= n / 2` (the cap was honored, not
    /// silently ignored) and a non-`Optimal` status (an artificially
    /// truncated solve cannot have a certified optimum) — `stop_status` maps
    /// this internal (non-wall-clock) stop to `SuboptimalSolution` /
    /// `MaxIterations`, never `Timeout`.
    ///
    /// No-op proof: reverting any of the `options.max_iters` checks added to
    /// `dual_advanced::bounded_core`/`dual_advanced::core`/`primal::core`
    /// makes the capped solve run to completion regardless of `max_iters`,
    /// reporting `Optimal` with `iterations == n` — failing both assertions.
    #[test]
    fn max_iters_is_honored_by_the_real_solver_end_to_end() {
        let lp = klee_minty_lp(12);
        let opts_uncapped = SolverOptions {
            presolve: false,
            ..Default::default()
        };
        let uncapped = solve_lp_with(&lp, &opts_uncapped);
        assert_eq!(
            uncapped.status,
            SolveStatus::Optimal,
            "test premise: the uncapped solve must reach Optimal"
        );
        let n = uncapped.iterations;
        assert!(
            n >= 2,
            "test premise: klee_minty_lp must need at least 2 iterations \
             (a too-easy LP can't distinguish 'cap honored' from 'cap \
             irrelevant'); got n={n}"
        );

        let cap = (n / 2) as u64;
        let capped = solve_lp_with(
            &lp,
            &SolverOptions {
                max_iters: Some(cap),
                presolve: false,
                ..Default::default()
            },
        );
        assert!(
            capped.iterations as u64 <= cap,
            "max_iters={cap} must bound reported iterations; got {}",
            capped.iterations
        );
        assert_ne!(
            capped.status,
            SolveStatus::Optimal,
            "a solve truncated at max_iters={cap} (< natural n={n}) cannot \
             have reached a certified optimum"
        );
        assert_ne!(
            capped.status,
            SolveStatus::Timeout,
            "max_iters exhaustion is an internal budget decision, not the \
             external wall-clock deadline — must not report Timeout"
        );
    }
}
