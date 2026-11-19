//! Public-path sentinels for the coefficient-aware forcing margin
//! (PR #89 P1 class, `mip::root_presolve` / `presolve::transforms::forcing`).
//!
//! The unit-level guards for this finding lived inside `mip::root_presolve`
//! and inspected `reduce()` internals (`map.pinned`). This file re-anchors
//! the regression on the PUBLIC solver path (`solve_milp_with_stats` with
//! default options — the shape every user reaches): a row whose entire
//! activity window is below an absolute forcing tolerance must not pin its
//! members, and the published objective must equal the hand-computed oracle
//! below, not a solver self-report. The `root_presolve_*` counter asserts
//! carry the CI sentinel-gate overlay contract: the counters are new on this
//! branch, so the base checkout fails to compile this file (red on base),
//! while a margin revert turns the objective / `cols_removed` asserts red
//! (see the per-test `sentinel:` lines for the exact revert-fail claim).
//!
//! Test names mirror the unit guards in `mip::root_presolve::tests`. The
//! genuine zero-slack forcing control is kept here as an UNMARKED test: it
//! is green on base by design, and a `/// sentinel:` marker would claim a
//! revert-fail it does not make (the gate requires every marked test to be
//! red on base).

use otspot_core::options::{MipConfig, SolverOptions};
use otspot_core::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot_core::sparse::CscMatrix;
use otspot_core::{solve_milp_with_stats, MilpProblem, MipStats, SolverResult};

/// Same fixture builder as the unit guards: single-row binary MILP.
fn build_milp(
    c: Vec<f64>,
    triplets: &[(usize, usize, f64)],
    m: usize,
    b: Vec<f64>,
    ctypes: Vec<ConstraintType>,
    bounds: Vec<(f64, f64)>,
    ints: Vec<usize>,
) -> MilpProblem {
    let n = c.len();
    let mut rows = Vec::with_capacity(triplets.len());
    let mut cols = Vec::with_capacity(triplets.len());
    let mut vals = Vec::with_capacity(triplets.len());
    for &(r, j, v) in triplets {
        rows.push(r);
        cols.push(j);
        vals.push(v);
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let lp = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();
    MilpProblem::new(lp, ints).unwrap()
}

/// The public solve path with default user options (measured root_presolve
/// counters for these fixtures: rows_removed 1, fallbacks 0).
fn solve_public(p: &MilpProblem) -> (SolverResult, MipStats) {
    solve_milp_with_stats(p, &SolverOptions::default(), &MipConfig::default())
}

/// Single-row model `min −x−y` over binary x, y with one row; `mk` keeps the
/// mirror variants (sense / coefficient / rhs) in one place.
fn single_row(b: f64, ctype: ConstraintType, coeff: f64) -> MilpProblem {
    build_milp(
        vec![-1.0, -1.0],
        &[(0, 0, coeff), (0, 1, coeff)],
        1,
        vec![b],
        vec![ctype],
        vec![(0.0, 1.0), (0.0, 1.0)],
        vec![0, 1],
    )
}

/// Root reduction must have fired and published without falling back. The
/// counters are branch-new: referencing them is also what makes this file
/// red on the base checkout (the whole module they measure does not exist
/// there — see the module docs).
fn assert_reduced_without_fallback(stats: &MipStats) {
    assert_eq!(
        stats.root_presolve_fallbacks, 0,
        "the reduced search's postsolved point must pass independent \
         verification; a non-zero fallback would mean the objective asserts \
         below measure the unreduced re-run instead"
    );
    assert!(
        stats.root_presolve_rows_removed >= 1,
        "the row of every fixture here is removed by the root reduction \
         (measured: 1); 0 means the stage did not run"
    );
}

/// A row that is merely redundant at 1e-10 coefficient scale must not force
/// its members to endpoints (public-path twin of the unit guard of the same
/// name in `mip::root_presolve::tests`).
///
/// Independent oracle: the model is `min −x−y` over binary x, y with
/// `1e-10·x + 1e-10·y ≤ 2e-10`. The maximum activity over the box is
/// `2e-10 = rhs`, so EVERY box point is feasible and the optimum is the
/// corner (1, 1) with objective −2. The pre-fix absolute margin
/// (`rhs − min_act ≤ ROOT_PRESOLVE_ACT_ABS_TOL` — 5× the whole row here)
/// pinned (0, 0) and published Optimal obj 0.
///
/// sentinel: restoring `rhs − min_act ≤ ROOT_PRESOLVE_ACT_ABS_TOL` in the
/// forcing test reproduces the false (0, 0) pin (`cols_removed == 0`
/// flips to 2), the row removal and the published obj-0 point (objective
/// `-2.0` flips to 0) → FAIL.
#[test]
fn tiny_coefficient_redundant_row_does_not_force_endpoints() {
    let (res, stats) = solve_public(&single_row(2e-10, ConstraintType::Le, 1e-10));
    assert_reduced_without_fallback(&stats);
    assert_eq!(
        stats.root_presolve_cols_removed, 0,
        "a row whose slack (2e-10) is 5× the row itself must pin no member \
         (the absolute 1e-9 margin removed both columns as forced)"
    );
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(
        res.objective, -2.0,
        "oracle: the whole box satisfies the row (max activity 2e-10 = rhs), \
         so min −x−y attains −2 at (1, 1); the false pin published 0"
    );
    assert_eq!(res.solution, vec![1.0, 1.0]);
}

/// The mirrored forcing paths (Ge side and both Eq sides of the margin) are
/// scale-aware too, end-to-end on the public path.
///
/// sentinel: restoring `(max_act − rhs) ≤ ACT_ABS` (Ge) or the
/// `(min_act − rhs).abs() / (max_act − rhs).abs() ≤ ACT_ABS` pair (Eq)
/// reproduces the mirrored false pins — the Ge redundant case publishes
/// obj 0 instead of the oracle −2 and the Eq interior case pins away from
/// (at least) its exact optimum −1 → FAIL.
#[test]
fn tiny_scale_forcing_mirrors_are_scale_aware() {
    // Ge mirror of the repro: −1e-10·x − 1e-10·y ≥ −2e-10 ⟺ x + y ≤ 2 —
    // every box point satisfies it. Oracle: optimum (1, 1), objective −2;
    // the old absolute margin read the max-activity slack as forcing and
    // published (0, 0) with obj 0.
    let (res, stats) = solve_public(&single_row(-2e-10, ConstraintType::Ge, -1e-10));
    assert_reduced_without_fallback(&stats);
    assert_eq!(
        stats.root_presolve_cols_removed, 0,
        "redundant Ge row pins none"
    );
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(
        res.objective, -2.0,
        "oracle: x + y ≤ 2 is redundant over the binary box, min −x−y = −2"
    );
    assert_eq!(res.solution, vec![1.0, 1.0]);

    // Ge genuine forcing at the same scale: −1e-10·x − 1e-10·y ≥ 0 ⟺
    // x + y ≤ 0 — (0, 0) is the ONLY feasible point; oracle objective 0.
    // The counter assert pins the ROOT forcing itself (measured
    // cols_removed = 2); losing it, B&B would still find (0, 0) and the
    // objectives alone would not notice.
    let (res, stats) = solve_public(&single_row(0.0, ConstraintType::Ge, -1e-10));
    assert_reduced_without_fallback(&stats);
    assert_eq!(
        stats.root_presolve_cols_removed, 2,
        "zero-slack Ge row must force BOTH members at the root"
    );
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(res.objective, 0.0);
    assert_eq!(res.solution, vec![0.0, 0.0], "x + y ≤ 0 forces (0, 0)");

    // Eq interior: 1e-10·x + 1e-10·y = 1e-10 ⟺ x + y = 1 — the rhs sits
    // strictly inside the activity window [0, 2e-10], so neither extreme
    // may pin. The old `(min_act − rhs).abs() ≤ ACT_ABS` pinned (0, 0) and
    // published obj 0. Post-fix the row may legitimately weaken to its ≥
    // side (the activity-margin family, whose absolute floor is the
    // solver's row-feasibility contract and NOT part of this finding), so
    // the published objective must do no worse than the exact optimum −1
    // of the original model — never the pinned 0. Measured here: the
    // weakened row solves to (1, 1), obj −2, with no member removed.
    let (res, stats) = solve_public(&single_row(1e-10, ConstraintType::Eq, 1e-10));
    assert_reduced_without_fallback(&stats);
    assert_eq!(
        stats.root_presolve_cols_removed, 0,
        "an rhs strictly inside the activity window must pin no member"
    );
    assert_eq!(res.status, SolveStatus::Optimal);
    assert!(
        res.objective <= -1.0,
        "the false (0, 0) pin published obj 0; the weakened row must do no \
         worse than the exact optimum −1 (got {})",
        res.objective
    );

    // Eq genuine forcing: 1e-10·x + 1e-10·y = 2e-10 ⟺ x + y = 2 — the
    // max-activity endpoint equals the rhs exactly: forced (measured
    // cols_removed = 2). Oracle: (1, 1), objective −2.
    let (res, stats) = solve_public(&single_row(2e-10, ConstraintType::Eq, 1e-10));
    assert_reduced_without_fallback(&stats);
    assert_eq!(
        stats.root_presolve_cols_removed, 2,
        "zero-slack Eq row must force BOTH members at the root"
    );
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(res.objective, -2.0);
    assert_eq!(res.solution, vec![1.0, 1.0], "x + y = 2 forces (1, 1)");
}

/// Over-tightening control on the public path (port of the unit-level
/// `genuine_forcing_rows_still_force_endpoints` published-objective half):
/// GENUINE zero-slack forcing keeps these published objectives whatever the
/// margin formula becomes.
///
/// NOT marked with `/// sentinel:` — green on base and under margin reverts
/// is its POINT (the MILP root propagates zero-slack rows to the same pins
/// regardless of the row_pass margin, as the unit control's doc records);
/// a marker would claim a revert-fail it does not make.
#[test]
fn genuine_forcing_rows_still_force_endpoints_public() {
    // Le: x + y ≤ 0 over the binary box has exactly one feasible point.
    let (res, stats) = solve_public(&single_row(0.0, ConstraintType::Le, 1.0));
    assert_reduced_without_fallback(&stats);
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(res.objective, 0.0);
    assert_eq!(res.solution, vec![0.0, 0.0]);

    // Eq forced from below: x + y = 0 → the single point (0, 0).
    let (res, stats) = solve_public(&single_row(0.0, ConstraintType::Eq, 1.0));
    assert_reduced_without_fallback(&stats);
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(res.objective, 0.0);

    // Eq forced from above: x + y = 2 → the single point (1, 1), obj −2.
    let (res, stats) = solve_public(&single_row(2.0, ConstraintType::Eq, 1.0));
    assert_reduced_without_fallback(&stats);
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(res.objective, -2.0);
    assert_eq!(res.solution, vec![1.0, 1.0]);
}
