//! Fast per-phase degeneration detector.
//!
//! Full suite bench runs (`skill bench`) take 30 minutes to multiple hours, so
//! they are not something to reach for after every small change. This file
//! covers a small, hand-picked cross-section of LP / QP / conic problems,
//! each solved through a production entry point and checked `status ==
//! Optimal` plus the objective within 1e-5 relative of an expected value:
//! LP/QP via `parse_qps`→`solve_qp_with` (through
//! `otspot_dev::screening::screen_single`) against
//! `data/baseline_objectives/*.csv`; conic via `solve_socp` against a
//! closed-form oracle (see that test's doc comment for why). Runtime is
//! milliseconds per problem, so it runs on every `nextest` pass (not
//! `#[ignore]`d) and turns a phase-boundary check into seconds instead of a
//! full bench.
//!
//! This does NOT replace `lp_coverage_screen_all` (all 90 Netlib LPs,
//! `#[ignore]`, ~5-6 min) or the `--profile heavy` sweeps in
//! `otspot-io/tests/presolve_correctness_sweep.rs` — those remain the
//! authority for full-corpus regression. This file is a cheap tripwire for
//! the common case: 5 LPs, 5 QPs, 3 synthetic SOCPs, chosen because each
//! solves in well under 1s and has a precise expected value.
//!
//! Data (`data/*`) is a gitignored symlink farm; unlike the "skip gracefully
//! when absent" convention used by some real-data tests (e.g.
//! `cbf_feasibility.rs`), every LP/QP problem here is a *hard* requirement: a
//! missing file panics loudly instead of silently reducing the executed-count
//! to zero. CI runs the `lib-only` nextest profile (kind(lib)+kind(bin) only,
//! see `.config/nextest.toml`), which does not build this binary at all, so
//! CI is unaffected either way; a local `--profile default` run with `data/`
//! absent must fail loudly, not report a false green.
//!
//! The conic leg deliberately does NOT read `data/cblib_socp`: CI's
//! integration job only fetches the `--ci-subset` of `download_all_bench_data.sh`
//! (109 `lp_problems` + 138 `maros_meszaros`, covering the LP/QP legs above),
//! and CBLIB is a separate, manually-run `scripts/cblib_download.sh` that is
//! never part of that subset or its `bench-data-v1` cache — a `data/cblib_socp`
//! hard-require here would panic on every CI run. Instead it solves small
//! synthetic SOCPs with a closed-form (Cauchy–Schwarz) optimum through the
//! same `solve_socp` production entry point, so it needs no data files and
//! runs everywhere; the real CBLIB corpus (`data/cblib_socp`,
//! `cbf_cblib_integration.rs`, `cbf_feasibility.rs`) remains the full-corpus
//! conic regression, checked at bench checkpoints, not by this file.

use otspot_core::conic::{solve_socp, ConeSpec, ConicOptions, ConicProblem};
use otspot_core::options::SolverOptions;
use otspot_core::problem::SolveStatus;
use otspot_core::sparse::CscMatrix;
use otspot_dev::bench_utils::{load_baseline_objectives, obj_within_tol};
use otspot_dev::screening::{screen_single, ScreenVerdict};
use std::path::Path;
use std::time::Instant;

/// Relative objective tolerance (matches the solver's own eps=1e-6 convergence
/// target with headroom; see CLAUDE.md bench section).
const REL_TOL: f64 = 1e-5;
/// Per-problem solver timeout. Actual solves finish in single-digit
/// milliseconds; this is generous headroom, not a tuned expectation.
const SOLVE_TIMEOUT_SECS: f64 = 10.0;
/// Wall-clock budget per problem enforced by this file (selection criterion:
/// each representative problem must solve in under 1s).
const PER_PROBLEM_BUDGET_SECS: f64 = 1.0;

/// Runs the production `parse_qps` → `solve_qp_with` path (via
/// `screen_single`, shared with `lp_coverage_screen.rs` / `qps_benchmark`)
/// over a small named subset, asserting `Optimal` + baseline-objective match
/// for every problem. Returns `(problems_checked, total_wall_secs)`.
///
/// Panics (not skips) when the data directory, a named `.QPS` file, or the
/// baseline CSV is missing, so a data-absent environment cannot report a
/// false "0 problems, all green" pass. Also hard-requires every name to have
/// a baseline entry: `screen_single`'s `(Optimal, None)` arm silently treats
/// a missing baseline as a pass (by design, for its other callers that sweep
/// whole directories without a curated name list), so without this guard a
/// deleted CSV row or a typo'd name would go from "checked against baseline"
/// to "no-op pass" without failing anything here.
fn run_screened_set(
    dir: &str,
    baseline_csv: &str,
    names: &[&str],
    timeout_secs: f64,
) -> (usize, f64) {
    let dir_path = Path::new(dir);
    assert!(
        dir_path.exists(),
        "{dir} not found — bench data missing (data/ symlink absent in this worktree?)"
    );
    let baseline = load_baseline_objectives(Path::new(baseline_csv))
        .unwrap_or_else(|e| panic!("baseline CSV {baseline_csv} unreadable: {e}"));
    assert!(
        !baseline.is_empty(),
        "baseline CSV {baseline_csv} parsed to zero entries"
    );

    let mut opts = SolverOptions::default();
    opts.ipm.eps = 1e-6;
    opts.timeout_secs = Some(timeout_secs);

    let mut checked = 0usize;
    let mut total_secs = 0.0f64;
    for &name in names {
        let path = dir_path.join(format!("{name}.QPS"));
        assert!(
            path.exists(),
            "required regression fixture missing: {} (data/ absent — see skill bench)",
            path.display()
        );
        assert!(
            baseline.contains_key(name),
            "{name}: no baseline entry in {baseline_csv} (typo, or the CSV row was deleted — \
             screen_single treats a missing baseline as an automatic pass, so this guard \
             prevents a silent false green)"
        );
        let entry = screen_single(&path, name, &opts, &baseline, REL_TOL);
        total_secs += entry.elapsed_secs;
        assert!(
            matches!(entry.verdict, ScreenVerdict::Optimal),
            "{name}: expected Optimal + baseline match, got {:?} ({:.3}s)",
            entry.verdict,
            entry.elapsed_secs
        );
        assert!(
            entry.elapsed_secs < PER_PROBLEM_BUDGET_SECS,
            "{name}: took {:.3}s, expected < {PER_PROBLEM_BUDGET_SECS}s \
             (representative problems must stay fast; pick a smaller instance)",
            entry.elapsed_secs
        );
        checked += 1;
    }
    (checked, total_secs)
}

const LP_NAMES: [&str; 5] = ["afiro", "adlittle", "sc50a", "blend", "share2b"];
/// Independent of `LP_NAMES`'s own length: a compile-time tripwire so an edit
/// that shrinks (or empties) the array without also updating this count fails
/// the build, rather than the runtime `assert_eq!(checked, LP_NAMES.len())`
/// silently comparing the array against itself and passing with 0 problems.
const LP_PROBLEM_COUNT: usize = 5;
const _: () = assert!(LP_NAMES.len() == LP_PROBLEM_COUNT);

/// LP phase tripwire: 5 small Netlib LPs against `netlib_lp.csv` (official
/// MINOS 5.3 reference values). Same problems as
/// `presolve_correctness_sweep::presolve_invariance_curated_clean` (minus
/// `israel`, plus `share2b`), but checking a different property: objective
/// vs. the external baseline, not ON/OFF presolve invariance.
#[test]
fn fast_regression_lp_matches_netlib_baseline() {
    let (checked, total_secs) = run_screened_set(
        "data/lp_problems",
        "data/baseline_objectives/netlib_lp.csv",
        &LP_NAMES,
        SOLVE_TIMEOUT_SECS,
    );
    assert_eq!(
        checked,
        LP_NAMES.len(),
        "must execute all {} representative LP problems (no silent 0-run pass)",
        LP_NAMES.len()
    );
    eprintln!("[fast-baseline-regression] LP: {checked} problems in {total_secs:.3}s total");
}

const QP_NAMES: [&str; 5] = ["HS21", "HS35", "HS35MOD", "HS51", "HS52"];
/// See `LP_PROBLEM_COUNT` for why this is independent of `QP_NAMES.len()`.
const QP_PROBLEM_COUNT: usize = 5;
const _: () = assert!(QP_NAMES.len() == QP_PROBLEM_COUNT);

/// QP phase tripwire: 5 small Hock-Schittkowski QPs (Maros-Mészáros suite)
/// against `maros_meszaros.csv`. These five carry the high-precision
/// `clarabel_0.11.1_tol1e-12` reference rows (12 significant digits), unlike
/// most of that CSV's 2-3 digit self-measured rows, so a 1e-5 relative
/// tolerance is meaningful rather than accidentally loose.
#[test]
fn fast_regression_qp_matches_maros_meszaros_baseline() {
    let (checked, total_secs) = run_screened_set(
        "data/maros_meszaros",
        "data/baseline_objectives/maros_meszaros.csv",
        &QP_NAMES,
        SOLVE_TIMEOUT_SECS,
    );
    assert_eq!(
        checked,
        QP_NAMES.len(),
        "must execute all {} representative QP problems (no silent 0-run pass)",
        QP_NAMES.len()
    );
    eprintln!("[fast-baseline-regression] QP: {checked} problems in {total_secs:.3}s total");
}

/// Matches the solver's bench-standard eps=1e-6 convergence target (see
/// CLAUDE.md bench section), consistent with `REL_TOL`'s headroom.
const CONIC_TOL: f64 = 1e-6;

/// Builds `min c^T x` s.t. `||x||_2 <= 1` (the `k`-dimensional unit ball) as a
/// `ConicProblem`: no equality rows, one SOC block of dimension `k+1`, `s = h
/// - G x = (1, x_0, .., x_{k-1})` via `h[0] = 1` and `G[i+1, i] = -1`.
///
/// Independent oracle (Cauchy-Schwarz, not the solver): over the unit ball
/// `c^T x` is minimized at `x* = -c/||c||`, giving `obj* = -||c||`.
fn unit_ball_socp(c: &[f64]) -> (ConicProblem, f64) {
    let k = c.len();
    let mut rows = Vec::with_capacity(k);
    let mut cols = Vec::with_capacity(k);
    let mut vals = Vec::with_capacity(k);
    for i in 0..k {
        rows.push(i + 1);
        cols.push(i);
        vals.push(-1.0);
    }
    let g = CscMatrix::from_triplets(&rows, &cols, &vals, k + 1, k).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, k).unwrap();
    let mut h = vec![0.0; k + 1];
    h[0] = 1.0;
    let prob = ConicProblem {
        c: c.to_vec(),
        a,
        b: vec![],
        g,
        h,
        cone: ConeSpec {
            l: 0,
            soc: vec![k + 1],
        },
    };
    let norm_c = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    (prob, -norm_c)
}

/// 3 cost vectors of increasing dimension (2, 3, 4 vars) — enough to catch a
/// dimension-dependent regression while staying independent of any data file.
const CONIC_SYNTHETIC_COSTS: [&[f64]; 3] = [&[3.0, 4.0], &[1.0, 2.0, 2.0], &[1.0, 1.0, 1.0, 1.0]];
/// See `LP_PROBLEM_COUNT` for why this is independent of
/// `CONIC_SYNTHETIC_COSTS.len()`.
const CONIC_PROBLEM_COUNT: usize = 3;
const _: () = assert!(CONIC_SYNTHETIC_COSTS.len() == CONIC_PROBLEM_COUNT);

/// Conic phase tripwire: synthetic unit-ball SOCPs solved via the production
/// `solve_socp` entry point, checked against the closed-form optimum above.
/// No `data/` dependency (see module doc for why the real CBLIB corpus is not
/// used here), so the executed-count guard is just "ran all 3 vectors".
#[test]
fn fast_regression_conic_matches_closed_form_unit_ball() {
    let opts = ConicOptions {
        tol: CONIC_TOL,
        ..ConicOptions::default()
    };

    let mut checked = 0usize;
    let mut total_secs = 0.0f64;
    for &c in &CONIC_SYNTHETIC_COSTS {
        let (prob, known_obj) = unit_ball_socp(c);

        let start = Instant::now();
        let res = solve_socp(&prob, &opts);
        let elapsed = start.elapsed().as_secs_f64();
        total_secs += elapsed;

        assert_eq!(
            res.status,
            SolveStatus::Optimal,
            "n={}: status {:?} ({:.3}s)",
            c.len(),
            res.status,
            elapsed
        );
        eprintln!(
            "[fast-baseline-regression] conic n={}: obj={:.6e} known={:.6e} ({:.3}s)",
            c.len(),
            res.objective,
            known_obj,
            elapsed
        );
        assert!(
            obj_within_tol(res.objective, known_obj, REL_TOL),
            "n={}: obj {:.8e} != known {:.8e} (tol={:.1e})",
            c.len(),
            res.objective,
            known_obj,
            REL_TOL
        );
        assert!(
            elapsed < PER_PROBLEM_BUDGET_SECS,
            "n={}: took {elapsed:.3}s, expected < {PER_PROBLEM_BUDGET_SECS}s",
            c.len()
        );
        checked += 1;
    }
    assert_eq!(
        checked,
        CONIC_SYNTHETIC_COSTS.len(),
        "must execute all {} representative conic problems (no silent 0-run pass)",
        CONIC_SYNTHETIC_COSTS.len()
    );
    eprintln!("[fast-baseline-regression] conic: {checked} problems in {total_secs:.3}s total");
}
