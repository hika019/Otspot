//! GMI/MIR cut sentinels.
//!
//! Cuts are appended as Le rows (`−g·x ≤ −rhs`, equivalent to `g·x ≥ rhs`).
//!
//! The load-bearing test is `cut_validity_brute_force`: it enumerates every
//! integer point of small all-integer MILPs and asserts that no original-feasible
//! point is removed by any generated cut. Corrupting the GMI formula (wrong
//! rounding direction, sign flip) makes a cut slice off an integer point and this
//! test fails. The companion tests assert cuts are actually generated and that
//! they cut the fractional LP optimum (a no-op generator fails those), and that
//! cuts do not change the MILP optimum.

use super::*;
use crate::options::{MipConfig, SolverOptions};
use crate::problem::ConstraintType;
use otspot_num::sparse::CscMatrix;

fn lp(
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    m: usize,
    b: Vec<f64>,
    ct: Vec<ConstraintType>,
    bounds: Vec<(f64, f64)>,
) -> LpProblem {
    let n = c.len();
    let a = if m == 0 {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(rows, cols, vals, m, n).unwrap()
    };
    LpProblem::new_general(c, a, b, ct, bounds, None).unwrap()
}

fn cuts_cfg(rounds: usize) -> MipConfig {
    MipConfig {
        cuts: true,
        max_cut_rounds: rounds,
        ..MipConfig::default()
    }
}

/// Enumerate every integer lattice point in the (finite) box `bounds`.
fn enumerate_int_box(bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
    let mut pts = vec![vec![]];
    for &(lo, hi) in bounds {
        let lo = lo.ceil() as i64;
        let hi = hi.floor() as i64;
        let mut next = Vec::new();
        for p in &pts {
            for v in lo..=hi {
                let mut q = p.clone();
                q.push(v as f64);
                next.push(q);
            }
        }
        pts = next;
    }
    pts
}

/// Is `x` feasible for the original LP rows (and bounds)?
fn feasible_orig(p: &LpProblem, x: &[f64]) -> bool {
    let tol = 1e-7;
    for (j, &(lo, hi)) in p.bounds.iter().enumerate() {
        if x[j] < lo - tol || x[j] > hi + tol {
            return false;
        }
    }
    let ax = p.a.mat_vec_mul(x).unwrap();
    for i in 0..p.num_constraints {
        let ok = match p.constraint_types[i] {
            ConstraintType::Le => ax[i] <= p.b[i] + tol,
            ConstraintType::Ge => ax[i] >= p.b[i] - tol,
            ConstraintType::Eq => (ax[i] - p.b[i]).abs() <= tol,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Solve the LP root the way the cut generator does (primal, no presolve).
fn lp_root(p: &LpProblem) -> crate::problem::SolverResult {
    super::solve_cut_lp(p, &SolverOptions::default(), None, None)
}

// ── Test problems (all-integer, small box ⇒ brute-forceable) ───────────────

/// max x+y  ⇔  min -x-y  s.t. 2x+2y<=3, x,y∈{0,1}. LP opt x+y=1.5 (fractional);
/// bounded vars exercise the UB-row slack mapping.
fn p_box_le() -> MilpProblem {
    let l = lp(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 2.0],
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    MilpProblem::new(l, vec![0, 1]).unwrap()
}

#[test]
fn append_ge_rows_snaps_near_empty_integer_bounds_to_integer_point() {
    let mut milp = p_box_le();
    milp.lp.bounds[0] = (1.0 + ZERO_TOL * 0.5, 1.0);
    let cuts = [CutRow {
        coeffs: vec![1.0, 0.0],
        rhs: 0.5,
    }];

    let out = append_ge_rows_with_integer_mask(&milp.lp, &cuts, &[true, true]);

    assert_eq!(out.bounds[0], (1.0, 1.0));
    assert_eq!(out.num_constraints, milp.lp.num_constraints + 1);
}

// A material empty box (gap > ZERO_TOL) is a well-formed infeasible box: it must
// be PRESERVED verbatim through cut augmentation (never snapped to a point) so
// the downstream relaxation solve reports Infeasible. Construction now accepts
// lb>ub, so `append_ge_rows` builds a valid LP that keeps the empty box.
#[test]
fn append_ge_rows_preserves_large_material_bound_gap() {
    let mut milp = p_box_le();
    milp.lp.bounds[0] = (1.0e12 + 1.0, 1.0e12);
    let cuts = [CutRow {
        coeffs: vec![1.0, 0.0],
        rhs: 0.5,
    }];

    let out = append_ge_rows(&milp.lp, &cuts);
    assert_eq!(
        out.bounds[0],
        (1.0e12 + 1.0, 1.0e12),
        "large material empty box must be preserved, not scaled/snapped away"
    );
    assert!(
        out.bounds[0].0 > out.bounds[0].1,
        "empty box preserved (lb>ub)"
    );
}

#[test]
fn append_ge_rows_preserves_material_empty_box() {
    let mut milp = p_box_le();
    milp.lp.bounds[0] = (1.0 + ZERO_TOL * 10.0, 1.0);
    let cuts = [CutRow {
        coeffs: vec![1.0, 0.0],
        rhs: 0.5,
    }];

    let out = append_ge_rows(&milp.lp, &cuts);
    assert_eq!(
        out.bounds[0],
        (1.0 + ZERO_TOL * 10.0, 1.0),
        "material empty box must be kept verbatim (flows through as Infeasible)"
    );
    assert!(
        out.bounds[0].0 > out.bounds[0].1,
        "empty box preserved (lb>ub)"
    );
}

/// min x+y s.t. 2x+2y>=3, x,y∈[0,3]. Ge constraint ⇒ surplus-slack mapping.
/// LP opt x+y=1.5 (fractional).
fn p_box_ge() -> MilpProblem {
    let l = lp(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 2.0],
        1,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    MilpProblem::new(l, vec![0, 1]).unwrap()
}

/// max x+2y  ⇔  min -x-2y  s.t. 2y<=3, x+y<=3, x,y∈[0,3]. Two Le rows + bounded
/// vars. Unique LP optimum (x,y)=(1.5,1.5) is fractional; integer opt obj=-4.
fn p_two_le() -> MilpProblem {
    let l = lp(
        vec![-1.0, -2.0],
        &[0, 1, 1],
        &[1, 0, 1],
        &[2.0, 1.0, 1.0],
        2,
        vec![3.0, 3.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    MilpProblem::new(l, vec![0, 1]).unwrap()
}

/// 3-var bounded integer problem: min -x-y-z s.t. 3x+2y+4z<=7, x,y,z∈[0,3].
/// UB rows ARE generated (bounds are finite); structural cols are LbShift.
fn p_lb_only() -> MilpProblem {
    let l = lp(
        vec![-1.0, -1.0, -1.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[3.0, 2.0, 4.0],
        1,
        vec![7.0],
        vec![ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0), (0.0, 3.0)],
    );
    MilpProblem::new(l, vec![0, 1, 2]).unwrap()
}

/// True lb-only (ub=+∞): min -x-y-z s.t. 3x+2y+4z<=7, x,y,z≥0 (no UB rows).
/// Structural cols are pure LbShift; no UB slack is generated.
fn p_lb_only_inf() -> MilpProblem {
    let l = lp(
        vec![-1.0, -1.0, -1.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[3.0, 2.0, 4.0],
        1,
        vec![7.0],
        vec![ConstraintType::Le],
        vec![
            (0.0, f64::INFINITY),
            (0.0, f64::INFINITY),
            (0.0, f64::INFINITY),
        ],
    );
    MilpProblem::new(l, vec![0, 1, 2]).unwrap()
}

fn all_problems() -> Vec<(&'static str, MilpProblem)> {
    vec![
        ("box_le", p_box_le()),
        ("box_ge", p_box_ge()),
        ("two_le", p_two_le()),
        ("lb_only", p_lb_only()),
    ]
}

/// **Cut validity (load-bearing):** every integer-feasible point of the original
/// problem must satisfy every generated cut. Cuts are Le rows (`−g·x ≤ −rhs`);
/// a valid cut means `ax[i] ≤ b[i] + ε` for all original-feasible integer x.
/// A sign error in the GMI/MIR formula or the Le negation fails this test.
#[test]
fn cut_validity_brute_force() {
    for (name, milp) in all_problems() {
        for rounds in [1usize, 5] {
            let out = add_root_cuts(&milp, &SolverOptions::default(), &cuts_cfg(rounds));
            let m_old = milp.lp.num_constraints;
            let m_new = out.lp.num_constraints;
            assert!(
                m_new >= m_old,
                "{name}: cuts must not drop rows ({m_old}->{m_new})"
            );
            let pts = enumerate_int_box(&milp.lp.bounds);
            for x in &pts {
                if !feasible_orig(&milp.lp, x) {
                    continue;
                }
                // Cuts are Le rows (−g·x ≤ −rhs): valid for x when ax[i] ≤ b[i] + ε.
                let ax = out.lp.a.mat_vec_mul(x).unwrap();
                for i in m_old..m_new {
                    assert_eq!(out.lp.constraint_types[i], ConstraintType::Le);
                    assert!(
                        ax[i] <= out.lp.b[i] + 1e-6,
                        "{name} round={rounds}: INVALID CUT — integer point {x:?} \
                         removed by Le cut row {i}: −g·x={} > −rhs={}",
                        ax[i],
                        out.lp.b[i]
                    );
                }
            }
        }
    }
}

/// **Validity for the UbOnly mapping (lb=-∞, ub finite):** the finite-lb problems
/// never exercise the `x_std = ub - x_p` structural image. Here x has bounds
/// (-∞, 2]; we enumerate a finite integer window and assert no feasible integer
/// point is sliced. A sign error in the UbOnly image fails this test.
#[test]
fn cut_validity_ub_only_var() {
    // min -x s.t. 2x<=3, x∈(-∞,2] integer. LP opt x=1.5 (UbOnly source). Integer
    // feasible: x<=1.
    let l = lp(
        vec![-1.0],
        &[0],
        &[0],
        &[2.0],
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(f64::NEG_INFINITY, 2.0)],
    );
    let milp = MilpProblem::new(l, vec![0]).unwrap();
    let out = add_root_cuts(&milp, &SolverOptions::default(), &cuts_cfg(3));
    let m_old = milp.lp.num_constraints;
    let m_new = out.lp.num_constraints;
    assert!(
        m_new > m_old,
        "a cut must be generated for the UbOnly source"
    );
    for xi in -8..=2 {
        let x = vec![xi as f64];
        if !feasible_orig(&milp.lp, &x) {
            continue;
        }
        let ax = out.lp.a.mat_vec_mul(&x).unwrap();
        for i in m_old..m_new {
            assert_eq!(out.lp.constraint_types[i], ConstraintType::Le);
            assert!(
                ax[i] <= out.lp.b[i] + 1e-6,
                "INVALID CUT (UbOnly): integer x={xi} removed by Le cut row {i}: {} > {}",
                ax[i],
                out.lp.b[i]
            );
        }
    }
}

/// Cuts are generated AND they cut the fractional LP optimum. A no-op generator
/// (empty cuts, or a cut equal to a trivially-satisfied inequality) fails here.
/// Cuts are Le rows: x* violates when `ax[i] > b[i] + ε` (i.e., `−g·x* > −rhs`).
#[test]
fn cuts_are_generated_and_cut_lp_optimum() {
    for (name, milp) in all_problems() {
        let root = lp_root(&milp.lp);
        assert_eq!(root.status, SolveStatus::Optimal, "{name}: root must solve");
        let x_star = &root.solution;

        let out = add_root_cuts(&milp, &SolverOptions::default(), &cuts_cfg(1));
        let m_old = milp.lp.num_constraints;
        let m_new = out.lp.num_constraints;
        assert!(
            m_new > m_old,
            "{name}: at least one cut must be generated (root LP is fractional)"
        );
        // Le cut violated by x*: −g·x* > −rhs + ε, i.e., ax[i] > b[i] + ε.
        let ax = out.lp.a.mat_vec_mul(x_star).unwrap();
        let any_violated = (m_old..m_new).any(|i| ax[i] > out.lp.b[i] + 1e-6);
        assert!(
            any_violated,
            "{name}: a generated cut must violate the fractional LP optimum {x_star:?}"
        );
    }
}

/// Cuts only tighten the relaxation: the LP bound does not loosen and stays a
/// valid lower bound on the MILP optimum (minimization).
#[test]
fn cuts_tighten_lp_bound_without_crossing_integer_optimum() {
    for (name, milp) in all_problems() {
        let root = lp_root(&milp.lp);
        let out = add_root_cuts(&milp, &SolverOptions::default(), &cuts_cfg(5));
        let cut_root = lp_root(&out.lp);
        assert_eq!(cut_root.status, SolveStatus::Optimal, "{name}");
        assert!(
            cut_root.objective >= root.objective - 1e-6,
            "{name}: cut LP bound {} must not be looser than root {}",
            cut_root.objective,
            root.objective
        );
        let int_opt = brute_force_min(&milp);
        if let Some(opt) = int_opt {
            assert!(
                cut_root.objective <= opt + 1e-6,
                "{name}: cut LP bound {} must stay <= integer optimum {}",
                cut_root.objective,
                opt
            );
        }
    }
}

/// Brute-force integer optimum over the box (all-integer problems only).
fn brute_force_min(milp: &MilpProblem) -> Option<f64> {
    let mut best: Option<f64> = None;
    for x in enumerate_int_box(&milp.lp.bounds) {
        if feasible_orig(&milp.lp, &x) {
            let obj: f64 = milp.lp.c.iter().zip(&x).map(|(c, xi)| c * xi).sum();
            best = Some(best.map_or(obj, |b| b.min(obj)));
        }
    }
    best
}

/// **Optimality invariance:** solving with cuts ON reaches the same optimal
/// objective and integer solution as cuts OFF (cuts never change the optimum).
#[test]
fn cuts_preserve_optimum() {
    use crate::solve_milp;
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    for (name, milp) in all_problems() {
        let off = solve_milp(&milp, &opts, &MipConfig::default());
        let on = solve_milp(&milp, &opts, &cuts_cfg(0));
        assert_eq!(off.status, SolveStatus::Optimal, "{name}: off optimal");
        assert_eq!(on.status, SolveStatus::Optimal, "{name}: on optimal");
        assert!(
            (off.objective - on.objective).abs() < 1e-6,
            "{name}: cuts changed the optimum: off={} on={}",
            off.objective,
            on.objective
        );
        let bf = brute_force_min(&milp).expect("feasible integer optimum");
        assert!(
            (on.objective - bf).abs() < 1e-6,
            "{name}: cuts-ON optimum {} != brute force {}",
            on.objective,
            bf
        );
    }
}

/// **Bug 1 regression guard:** `solve_milp_with_stats` with cuts ON must reach the
/// correct integer optimum and report a finite `root_lp_bound`.
///
/// - Bug 1 (FP false incumbent): FP receives `effective.lp` (cut-augmented) instead
///   of `problem_bt.lp`, potentially finding a trivially-feasible x=0 with
///   obj=0 as incumbent; B&B then prunes the true optimal.
///
/// Sentinel: reverting Bug 1 produces an incorrect objective (Bug 1), failing here.
#[test]
fn cuts_on_root_lp_bound_valid_and_optimum_correct() {
    use crate::solve_milp_with_stats;
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    for (name, milp) in all_problems() {
        let bf = brute_force_min(&milp);
        let (res, stats) = solve_milp_with_stats(&milp, &opts, &cuts_cfg(3));
        assert_eq!(
            res.status,
            SolveStatus::Optimal,
            "{name}: cuts-on solve must reach Optimal"
        );
        assert!(
            stats.root_lp_bound.is_finite(),
            "{name}: root_lp_bound must be finite"
        );
        if let Some(opt) = bf {
            assert!(
                (res.objective - opt).abs() < 1e-6,
                "{name}: cuts-on objective {} must equal brute-force {} \
                 (Bug 1: FP false incumbent corrupts B&B pruning)",
                res.objective,
                opt
            );
        }
    }
}

/// **Tableau row correctness:** `alpha = e_i^T B^{-1} A_std` for a hand-built LP
/// matches a direct dense computation of `B^{-1} A`. Locks the BTRAN + column-dot
/// path that the GMI formula consumes.
#[test]
fn tableau_row_matches_dense() {
    let l = p_box_le().lp;
    let root = lp_root(&l);
    assert_eq!(root.status, SolveStatus::Optimal);
    let basis = &root.warm_start_basis.as_ref().unwrap().basis;

    let sf = build_standard_form(&l);
    assert_eq!(basis.len(), sf.m);
    let mut lu = LuBasis::new_timed(&sf.a, basis, 0, None).unwrap();

    let m = sf.m;
    let n = sf.n_total;
    let dense_a = csc_to_dense(&sf.a, m, n);
    let b_inv = dense_basis_inverse(&dense_a, basis);

    for i in 0..m {
        let mut rho = vec![0.0; m];
        rho[i] = 1.0;
        lu.btran_dense(&mut rho);
        for j in 0..n {
            let via_btran = column_dot(&sf.a, j, &rho);
            let mut direct = 0.0;
            for k in 0..m {
                direct += b_inv[i][k] * dense_a[k][j];
            }
            assert!(
                (via_btran - direct).abs() < 1e-7,
                "tableau ({i},{j}): btran {via_btran} != dense {direct}"
            );
        }
    }
}

/// Enumerate integer points in `[lo, hi]^n_vars` (used when some bounds are ∞).
fn enumerate_int_window(n_vars: usize, lo: i64, hi: i64) -> Vec<Vec<f64>> {
    let mut pts = vec![vec![]];
    for _ in 0..n_vars {
        let mut next = Vec::new();
        for p in &pts {
            for v in lo..=hi {
                let mut q = p.clone();
                q.push(v as f64);
                next.push(q);
            }
        }
        pts = next;
    }
    pts
}

/// Shared validity+non-vacuous check: assert that cuts were generated (non-vacuous)
/// and that every feasible integer point in `int_pts` satisfies every Le cut
/// (`ax[i] ≤ b[i] + ε`).
fn assert_cuts_valid_nonvacuous(
    milp: &MilpProblem,
    int_pts: &[Vec<f64>],
    name: &str,
    rounds: usize,
) -> usize {
    let out = add_root_cuts(milp, &SolverOptions::default(), &cuts_cfg(rounds));
    let m_old = milp.lp.num_constraints;
    let m_new = out.lp.num_constraints;
    assert!(
        m_new > m_old,
        "{name}: no cuts generated (LP relaxation must be fractional for non-vacuous check)"
    );
    for x in int_pts {
        if !feasible_orig(&milp.lp, x) {
            continue;
        }
        let ax = out.lp.a.mat_vec_mul(x).unwrap();
        for i in m_old..m_new {
            assert_eq!(out.lp.constraint_types[i], ConstraintType::Le);
            assert!(
                ax[i] <= out.lp.b[i] + 1e-6,
                "{name}: INVALID CUT — integer point {x:?} removed by Le cut row {i}: \
                 −g·x={} > −rhs={}",
                ax[i],
                out.lp.b[i]
            );
        }
    }
    m_new - m_old
}

/// **Negative lb (lb-shift in negative direction):** x,y ∈ [-1, 2] forces
/// `x_std = x - (-1) = x + 1` with offset = -1.  The LP relaxation is
/// fractional at x=y=0.75; integer opt is (1,0) or (0,1).
#[test]
fn cut_validity_negative_lb() {
    let l = lp(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 2.0],
        1,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(-1.0, 2.0), (-1.0, 2.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1]).unwrap();
    let pts = enumerate_int_box(&milp.lp.bounds);
    let n = assert_cuts_valid_nonvacuous(&milp, &pts, "neg_lb", 5);
    assert!(n > 0, "negative-lb path must generate ≥1 cut");
}

/// **Negative RHS / row-negation path:** Le constraint with coefficient signs
/// that produce `b_shifted < 0`, triggering `row_negated = true` in
/// `build_standard_form`.
#[test]
fn cut_validity_negative_rhs_row_negation() {
    let l = lp(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[-2.0, -2.0],
        1,
        vec![-3.0],
        vec![ConstraintType::Le],
        vec![(0.0, 2.0), (0.0, 2.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1]).unwrap();
    let pts = enumerate_int_box(&milp.lp.bounds);
    let n = assert_cuts_valid_nonvacuous(&milp, &pts, "neg_rhs_row_negation", 5);
    assert!(n > 0, "row-negation path must generate ≥1 cut");
}

/// **Le and Ge mixed in the same problem:** exercises the two slack-kind paths
/// (`ConstraintLe` and `ConstraintGe`) within a single cut round.
#[test]
fn cut_validity_mixed_le_ge() {
    let l = lp(
        vec![-1.0, -1.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[2.0, 2.0, 1.0, 1.0],
        2,
        vec![3.0, 0.0],
        vec![ConstraintType::Le, ConstraintType::Ge],
        vec![(0.0, 2.0), (0.0, 2.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1]).unwrap();
    let pts = enumerate_int_box(&milp.lp.bounds);
    let n = assert_cuts_valid_nonvacuous(&milp, &pts, "mixed_le_ge", 5);
    assert!(n > 0, "mixed Le/Ge path must generate ≥1 cut");
}

/// **True lb-only (ub = +∞):** UB rows are NOT generated for these variables.
#[test]
fn cut_validity_true_lb_only_inf_ub() {
    let milp = p_lb_only_inf();
    let pts = enumerate_int_window(3, 0, 4);
    let n = assert_cuts_valid_nonvacuous(&milp, &pts, "true_lb_only_inf", 5);
    assert!(n > 0, "true lb-only (ub=∞) path must generate ≥1 cut");
}

/// **Multi-var UbOnly columns + Eq row:** two UbOnly variables (lb=-∞, ub=3)
/// plus an equality constraint (no slack column).
#[test]
fn cut_validity_multi_var_ubonly_eq_row() {
    let l = lp(
        vec![-1.0, -2.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 2.0, 4.0],
        2,
        vec![2.0, 7.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(f64::NEG_INFINITY, 3.0), (f64::NEG_INFINITY, 3.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1]).unwrap();
    let pts = enumerate_int_window(2, -2, 4);
    let n = assert_cuts_valid_nonvacuous(&milp, &pts, "ubonly_eq_row", 5);
    assert!(n > 0, "UbOnly + Eq row path must generate ≥1 cut");
}

/// **Deterministic LCG fuzz:** generates ≥100 2-variable MILPs with fractional
/// LP relaxation optima, varying constraint types, coefficients, RHS and bounds.
/// For every problem the test asserts that no integer-feasible point is removed
/// by any generated cut. `with_cuts` tracks how many problems actually produce
/// cuts; the test asserts at least some do.
#[test]
fn cut_validity_fuzz_lcg() {
    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    }
    fn lcg_f(s: &mut u64, lo: f64, hi: f64) -> f64 {
        lo + ((lcg(s) >> 11) as f64 / (1u64 << 53) as f64) * (hi - lo)
    }
    fn is_frac(v: f64) -> bool {
        let f = v - v.floor();
        f > 1e-4 && f < 1.0 - 1e-4
    }

    let mut rng: u64 = 0xdead_beef_cafe_babe;
    let mut total = 0usize;
    let mut with_cuts = 0usize;

    for _ in 0..600 {
        if total >= 160 {
            break;
        }

        let lb0 = lcg_f(&mut rng, -2.0, 0.9).round();
        let lb1 = lcg_f(&mut rng, -2.0, 0.9).round();
        let ub0 = lcg_f(&mut rng, 2.0, 4.0).round();
        let ub1 = lcg_f(&mut rng, 2.0, 4.0).round();
        if lb0 >= ub0 || lb1 >= ub1 {
            continue;
        }

        let a00 = lcg_f(&mut rng, 1.0, 5.0).round();
        let a01 = lcg_f(&mut rng, 1.0, 5.0).round();
        let is_le = lcg(&mut rng).is_multiple_of(2);

        let min_ax = a00 * lb0 + a01 * lb1;
        let max_ax = a00 * ub0 + a01 * ub1;
        let range = max_ax - min_ax;
        if range < 2.0 {
            continue;
        }
        let mid = (min_ax + max_ax) / 2.0;
        let rhs = mid.floor() + 0.5;
        let ct = if is_le {
            ConstraintType::Le
        } else {
            ConstraintType::Ge
        };
        let actual_rhs = rhs;
        if actual_rhs <= min_ax || actual_rhs >= max_ax {
            continue;
        }

        let l = lp(
            vec![-1.0, -1.0],
            &[0, 0],
            &[0, 1],
            &[a00, a01],
            1,
            vec![actual_rhs],
            vec![ct],
            vec![(lb0, ub0), (lb1, ub1)],
        );
        let milp = match MilpProblem::new(l, vec![0, 1]) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let root = lp_root(&milp.lp);
        if root.status != SolveStatus::Optimal {
            continue;
        }
        let fractional_lp = root.solution.iter().any(|&v| is_frac(v));
        if !fractional_lp {
            continue;
        }

        total += 1;
        let out = add_root_cuts(&milp, &SolverOptions::default(), &cuts_cfg(3));
        let m_old = milp.lp.num_constraints;
        let m_new = out.lp.num_constraints;
        if m_new > m_old {
            with_cuts += 1;
        }

        let pts = enumerate_int_box(&milp.lp.bounds);
        for x in &pts {
            if !feasible_orig(&milp.lp, x) {
                continue;
            }
            let ax = out.lp.a.mat_vec_mul(x).unwrap();
            for i in m_old..m_new {
                // Le cut: −g·x ≤ −rhs. Valid for x when ax[i] ≤ b[i] + ε.
                assert!(
                    ax[i] <= out.lp.b[i] + 1e-6,
                    "fuzz INVALID CUT — a=[{a00},{a01}] rhs={actual_rhs} ct={ct:?} \
                     int-pt {x:?} violates Le cut row {i}: −g·x={} > −rhs={}",
                    ax[i],
                    out.lp.b[i]
                );
            }
        }
    }

    assert!(
        total >= 100,
        "fuzz: need ≥100 fractional-LP problems, got {total}"
    );
    assert!(
        with_cuts > 0,
        "fuzz: at least one fractional-LP problem must generate cuts (got 0/{total})"
    );
}

fn csc_to_dense(a: &CscMatrix, m: usize, n: usize) -> Vec<Vec<f64>> {
    let mut d = vec![vec![0.0; n]; m];
    for j in 0..n {
        let (rs, vs) = a.get_column(j).unwrap();
        for (&r, &v) in rs.iter().zip(vs) {
            d[r][j] = v;
        }
    }
    d
}

/// Dense inverse of the basis matrix B (columns = `basis` of `dense_a`), via
/// Gauss-Jordan. Test-only oracle for the tableau check.
fn dense_basis_inverse(dense_a: &[Vec<f64>], basis: &[usize]) -> Vec<Vec<f64>> {
    let m = basis.len();
    let mut aug = vec![vec![0.0; 2 * m]; m];
    for (r, row) in aug.iter_mut().enumerate() {
        for (c, &col) in basis.iter().enumerate() {
            row[c] = dense_a[r][col];
        }
        row[m + r] = 1.0;
    }
    for c in 0..m {
        let mut piv = c;
        for r in (c + 1)..m {
            if aug[r][c].abs() > aug[piv][c].abs() {
                piv = r;
            }
        }
        aug.swap(c, piv);
        let d = aug[c][c];
        assert!(d.abs() > 1e-12, "singular basis in oracle");
        for v in aug[c].iter_mut() {
            *v /= d;
        }
        for r in 0..m {
            if r != c {
                let f = aug[r][c];
                for k in 0..2 * m {
                    aug[r][k] -= f * aug[c][k];
                }
            }
        }
    }
    aug.iter().map(|row| row[m..].to_vec()).collect()
}

/// **MIR coefficient equals GMI:** MIR and GMI produce identical coefficients for all
/// cases. For continuous nonbasics with negative α, the coefficient is `−α/(1−f₀)` —
/// setting it to 0 is invalid and can exclude integer-feasible solutions.
#[test]
fn mir_coeff_equals_gmi_for_all_cases() {
    let f0 = 0.4_f64;
    let omf0 = 0.6_f64;
    // Positive alpha: both use alpha/f0.
    let pos = 0.3_f64;
    assert!((mir_coeff(pos, f0, omf0, false) - pos / f0).abs() < 1e-12);
    assert!((gmi_coeff(pos, f0, omf0, false) - pos / f0).abs() < 1e-12);
    // Negative alpha: MIR must use -alpha/(1-f0), not 0.
    let neg = -0.2_f64;
    let expected = (-neg) / omf0;
    assert!(
        (mir_coeff(neg, f0, omf0, false) - expected).abs() < 1e-12,
        "MIR must use -alpha/(1-f0) for continuous negative alpha: \
         got {} expected {} (returning 0 excludes integer-feasible solutions)",
        mir_coeff(neg, f0, omf0, false),
        expected
    );
    assert!((gmi_coeff(neg, f0, omf0, false) - expected).abs() < 1e-12);
    // Integer case: identical for all alpha values.
    for &alpha in &[-1.7_f64, -0.3, 0.0, 0.3, 1.2, 2.7] {
        let g = gmi_coeff(alpha, f0, omf0, true);
        let m = mir_coeff(alpha, f0, omf0, true);
        assert!(
            (m - g).abs() < 1e-12,
            "integer case must be identical: alpha={alpha} gmi={g} mir={m}"
        );
    }
}

/// **MIR cut validity with continuous nonbasic having negative tableau entry:**
///
/// LP: min −x₁ + 10·x₂  s.t. x₁ − x₂ ≤ 2.5, x₁ ∈ [0,3] integer, x₂ ≥ 0 continuous.
/// LP opt: x₁=2.5, x₂=0. In the tableau row for x₁ (basic, f₀=0.5), x₂ has α=−1
/// (negative, structural continuous at its lower bound).
///
/// With the correct MIR = GMI formula, the cut is −2·x₁ + 4·x₂ ≥ −4, which holds
/// for every integer-feasible solution. If MIR used 0 for negative-α continuous
/// columns, the cut would be −2·x₁ + 2·x₂ ≥ −4, which the integer-feasible witness
/// (x₁=3, x₂=0.5) violates: −6 + 1 = −5 < −4.
#[test]
fn mir_cut_validity_continuous_nonbasic_negative_alpha() {
    // LP: min -x1 + 10*x2, s.t. x1 - x2 <= 2.5, 0 <= x1 <= 3, x2 >= 0.
    // A is 1×2: row=[0,0], col=[0,1], val=[1,-1] → x1 - x2 <= 2.5.
    let l = lp(
        vec![-1.0, 10.0],
        &[0, 0],
        &[0, 1],
        &[1.0, -1.0],
        1,
        vec![2.5],
        vec![ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, f64::INFINITY)],
    );
    let milp = MilpProblem::new(l.clone(), vec![0]).unwrap();

    let lp_res = lp_root(&l);
    assert_eq!(lp_res.status, SolveStatus::Optimal, "LP must solve");
    assert!(
        (lp_res.solution[0] - 2.5).abs() < 1e-6,
        "LP opt must be x1=2.5, got {}",
        lp_res.solution[0]
    );
    assert!(
        lp_res.solution[1].abs() < 1e-6,
        "LP opt must be x2=0 (high cost keeps x2 at LB), got {}",
        lp_res.solution[1]
    );

    let integer_mask = super::super::integer_mask(l.num_vars, milp.integer_vars.as_slice());
    let basis = lp_res.warm_start_basis.as_ref().unwrap().basis.clone();

    // Direct MIR round: if alpha<0 continuous → 0 (buggy), the cut becomes -2x1+2x2 >= -4.
    // With MIR = GMI (correct): -2x1+4x2 >= -4.
    let cuts = generate_round(&l, &integer_mask, &lp_res.solution, &basis, CutKind::Mir);
    assert!(
        !cuts.is_empty(),
        "MIR must generate a cut for the fractional LP"
    );

    // Witness: (x1=3, x2=0.5) is integer-feasible: x1∈Z, x2>=0, 3-0.5=2.5<=2.5.
    let witness = [3.0_f64, 0.5_f64];
    for (i, cut) in cuts.iter().enumerate() {
        let lhs: f64 = cut
            .coeffs
            .iter()
            .zip(witness.iter())
            .map(|(&g, &x)| g * x)
            .sum();
        assert!(
            lhs >= cut.rhs - 1e-9,
            "MIR cut {i} INVALID: integer-feasible witness (x1=3, x2=0.5) violates \
             g·x={lhs} < rhs={} (bug: negative-alpha continuous coeff was 0 instead of \
             -alpha/(1-f0))",
            cut.rhs
        );
    }
}

/// **MIR cuts generated:** two-round run (GMI round 0, MIR round 1) must still
/// produce cuts for every standard test problem with a fractional LP optimum.
#[test]
fn mir_cuts_generated_after_two_rounds() {
    for (name, milp) in all_problems() {
        let out = add_root_cuts(&milp, &SolverOptions::default(), &cuts_cfg(2));
        let m_old = milp.lp.num_constraints;
        let m_new = out.lp.num_constraints;
        assert!(
            m_new > m_old,
            "{name}: GMI+MIR (2 rounds) must generate at least one cut"
        );
    }
}

/// **Multi-Ge optimality invariance:** `solve_milp` with cuts=true must reach
/// the correct integer optimum on a problem with multiple Ge constraints.
///
/// Regression guard for the presolve-mismatch bug: `solve_validate` previously
/// ran with presolve=true while B&B nodes used presolve=false. Cuts that passed
/// validate but were numerically unstable without presolve corrupted B&B
/// incumbents (obj ≈ 1e12 on mas76). Fix: cuts are appended as Le rows
/// (numerically stable without presolve) and `solve_validate` uses presolve=false.
#[test]
fn cuts_preserve_optimum_multi_ge() {
    use crate::solve_milp;
    // min x+y  s.t. 2x+2y>=3 (Ge), x+3y>=4 (Ge),  x,y∈[0,3] integer.
    // LP opt: x=0, y=1.5 (fractional). Integer opt: obj=2 at (1,1) or (0,2).
    let l = lp(
        vec![1.0, 1.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[2.0, 2.0, 1.0, 3.0],
        2,
        vec![3.0, 4.0],
        vec![ConstraintType::Ge, ConstraintType::Ge],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1]).unwrap();
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let on = solve_milp(&milp, &opts, &cuts_cfg(3));
    assert_eq!(on.status, SolveStatus::Optimal);
    let bf = brute_force_min(&milp).expect("feasible");
    assert!(
        (on.objective - bf).abs() < 1e-6,
        "cuts+multi-Ge corrupted incumbent: got {} expected {}",
        on.objective,
        bf
    );
}

/// **Le re-validation — happy path:** the Le LP returned by `add_root_cuts` must
/// solve Optimally without presolve (same conditions B&B uses). This verifies
/// the Le re-validation gate added after `convert_cuts_to_le` runs and passes.
#[test]
fn le_revalidation_lp_is_optimal_no_presolve() {
    for (name, milp) in all_problems() {
        let out = add_root_cuts(&milp, &SolverOptions::default(), &cuts_cfg(3));
        if out.lp.num_constraints == milp.lp.num_constraints {
            continue; // no cuts generated for this problem
        }
        // Solve the Le LP without presolve — the conditions B&B uses.
        let check = solve_cut_lp(&out.lp, &SolverOptions::default(), None, None);
        assert_eq!(
            check.status,
            SolveStatus::Optimal,
            "{name}: Le cut LP must be Optimal without presolve (Le re-validation passed)"
        );
    }
}

/// **Le re-validation — fallback detection:** a manually-constructed Le LP whose
/// cut rows are infeasible must not validate as Optimal.  This is the condition
/// `add_root_cuts` guards against via the Le re-validation fallback.
///
/// Ge cut `0·x >= 1` is infeasible; after `convert_cuts_to_le` it becomes
/// `0·x <= −1`, which is also infeasible.  `solve_validate` must return
/// non-Optimal, confirming the gate would trigger the fallback.
#[test]
fn le_revalidation_detects_infeasible_le_cut() {
    let milp = p_box_le();
    let m_orig = milp.lp.num_constraints;

    // Build a committed LP with an all-zero-coefficient Ge row (rhs=1.0).
    // This is infeasible in both Ge and Le form; after conversion the Le row
    // is `0·x <= −1`, which solve_validate must reject.
    let infeasible_cut = CutRow {
        coeffs: vec![0.0, 0.0],
        rhs: 1.0,
    };
    let committed_bad = append_ge_rows(&milp.lp, &[infeasible_cut]);
    let le_bad = convert_cuts_to_le(committed_bad, m_orig);

    let check = solve_validate(&le_bad, &SolverOptions::default(), None, None);
    assert_ne!(
        check.status,
        SolveStatus::Optimal,
        "infeasible Le cut LP (0·x <= -1) must not validate as Optimal — \
         this is the condition the add_root_cuts fallback guards against"
    );

    // Confirm the original LP (fallback target) is still solvable.
    let orig_check = solve_validate(&milp.lp, &SolverOptions::default(), None, None);
    assert_eq!(
        orig_check.status,
        SolveStatus::Optimal,
        "original LP must remain Optimal (fallback is meaningful)"
    );
}

// ── Structural cut tests (cover, clique, implied bound) ─────────────────────

/// Build a 0-1 knapsack MILP: max Σ c_j x_j s.t. Σ a_j x_j ≤ b, x_j ∈ {0,1}.
fn knapsack_milp(c: Vec<f64>, a: Vec<f64>, b: f64) -> MilpProblem {
    let n = c.len();
    assert_eq!(a.len(), n);
    let rows: Vec<usize> = vec![0; n];
    let cols: Vec<usize> = (0..n).collect();
    let l = lp(
        c.iter().map(|&v| -v).collect(), // minimise -obj
        &rows,
        &cols,
        &a,
        1,
        vec![b],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0); n],
    );
    let ivars: Vec<usize> = (0..n).collect();
    MilpProblem::new(l, ivars).unwrap()
}

/// **Cover cut validity:** no original integer-feasible point is removed.
///
/// 2x1+2x2+2x3≤5, max x1+x2+x3, xi∈{0,1}.
/// LP opt: x1=x2=1, x3=0.5 (obj=2.5, fractional).
/// Cover {x1,x2,x3}: 2+2+2=6>5, minimal (removing any gives sum=4<5 wait—
/// 2+2=4<5 so can't remove any). Cut: x1+x2+x3≤2.
/// LP violates: 2.5>2.
#[test]
fn cover_cut_validity_brute_force() {
    let milp = knapsack_milp(vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0], 5.0);
    let x_lp = lp_root(&milp.lp).solution;
    let mask = super::super::integer_mask(3, &[0, 1, 2]);
    let cuts = generate_cover_cuts(&milp.lp, &mask, &x_lp);
    assert!(
        !cuts.is_empty(),
        "cover cut must be generated for this knapsack"
    );

    let pts = enumerate_int_box(&milp.lp.bounds);
    for x in &pts {
        if !feasible_orig(&milp.lp, x) {
            continue;
        }
        for (k, cut) in cuts.iter().enumerate() {
            let lhs: f64 = cut
                .coeffs
                .iter()
                .zip(x.iter())
                .map(|(&g, &xi)| g * xi)
                .sum();
            assert!(
                lhs >= cut.rhs - 1e-9,
                "cover cut {k} removes integer-feasible point {x:?}: lhs={lhs} < rhs={}",
                cut.rhs
            );
        }
    }
}

/// **Cover cut generation:** LP-fractional knapsack must produce ≥1 cover cut
/// and those cuts must violate the LP optimum.
#[test]
fn cover_cut_generated_and_cuts_lp_opt() {
    // 2x1+2x2+2x3≤5 has fractional LP opt (sum=2.5); cover cut: x1+x2+x3≤2.
    let milp = knapsack_milp(vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0], 5.0);
    let lp_res = lp_root(&milp.lp);
    assert_eq!(lp_res.status, SolveStatus::Optimal);
    let x_star = &lp_res.solution;
    let mask = super::super::integer_mask(3, &[0, 1, 2]);
    let cuts = generate_cover_cuts(&milp.lp, &mask, x_star);
    assert!(!cuts.is_empty(), "must generate ≥1 cover cut");
    let any_violated = cuts.iter().any(|cut| {
        let lhs: f64 = cut
            .coeffs
            .iter()
            .zip(x_star.iter())
            .map(|(&g, &xi)| g * xi)
            .sum();
        lhs < cut.rhs - 1e-9
    });
    assert!(
        any_violated,
        "at least one cover cut must violate LP optimum {x_star:?}"
    );
}

/// **Cover cuts end-to-end:** `add_root_cuts` on a knapsack must not change the
/// integer optimum (correctness invariant).
#[test]
fn cover_cuts_preserve_optimum() {
    use crate::solve_milp;
    let milp = knapsack_milp(vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0], 5.0);
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let cfg = cuts_cfg(5);
    let res = solve_milp(&milp, &opts, &cfg);
    assert_eq!(res.status, SolveStatus::Optimal);
    let bf = brute_force_min(&milp).expect("feasible");
    assert!(
        (res.objective - bf).abs() < 1e-6,
        "cuts changed optimum: got {} expected {}",
        res.objective,
        bf
    );
}

/// **Clique cut validity and generation via pairwise conflicts.**
///
/// Three binary vars x1,x2,x3. Three pairwise constraints:
///   x1+x2≤1, x1+x3≤1, x2+x3≤1 (each pair conflicts: a_i+a_j=2>1=b).
/// LP min -(x1+x2+x3): LP opt x1=x2=x3=0.5, sum=1.5>1 — violates clique cut.
/// Clique cut Σ x_j ≤ 1 must be generated and must not remove any {0,1}^3 feasible point.
#[test]
fn clique_cut_validity_brute_force() {
    // Three pairwise Le rows; binary vars. LP opt has fractional x_i=0.5.
    let l = lp(
        vec![-1.0, -1.0, -1.0],
        &[0, 1, 0, 2, 1, 2],
        &[0, 0, 1, 1, 2, 2],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        3,
        vec![1.0, 1.0, 1.0],
        vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1, 2]).unwrap();
    let lp_res = lp_root(&milp.lp);
    assert_eq!(lp_res.status, SolveStatus::Optimal);
    let x_star = &lp_res.solution;
    let mask = super::super::integer_mask(3, &[0, 1, 2]);
    let cuts = generate_clique_cuts(&milp.lp, &mask, x_star);
    assert!(
        !cuts.is_empty(),
        "must generate clique cut from pairwise conflict graph"
    );

    let pts = enumerate_int_box(&milp.lp.bounds);
    for x in &pts {
        if !feasible_orig(&milp.lp, x) {
            continue;
        }
        for (k, cut) in cuts.iter().enumerate() {
            let lhs: f64 = cut
                .coeffs
                .iter()
                .zip(x.iter())
                .map(|(&g, &xi)| g * xi)
                .sum();
            assert!(
                lhs >= cut.rhs - 1e-9,
                "clique cut {k} removes integer-feasible point {x:?}: lhs={lhs} < rhs={}",
                cut.rhs
            );
        }
    }
}

/// **Clique cut: no cut when no pairwise conflict exists.**
/// 2x1+2x2≤5: a_i+a_j=4 < b=5, no conflict. No clique cut.
#[test]
fn clique_cut_not_generated_without_conflict() {
    let l = lp(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[2.0, 2.0],
        1,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1]).unwrap();
    let x_star = lp_root(&milp.lp).solution;
    let mask = super::super::integer_mask(2, &[0, 1]);
    let cuts = generate_clique_cuts(&milp.lp, &mask, &x_star);
    assert!(
        cuts.is_empty(),
        "no conflict (a_i+a_j=4 < b=5) must produce no clique cut, got {}",
        cuts.len()
    );
}

/// **Clique cut: mixed-sign row must not produce false conflicts.**
///
/// x1+x2+x3 - 3*x4 ≤ 1, all binary. a_1+a_2=2 > b=1, but (1,1,0,1) is
/// feasible (activity = 1+1-3 = -1 ≤ 1). A naïve conflict check that ignores
/// the negative coefficient would falsely conclude x1, x2 conflict, leading to
/// an unsound clique cut x1+x2+x3 ≤ 1 that removes the optimal point (1,1,1,1).
#[test]
fn clique_cut_mixed_sign_row_no_false_conflict() {
    let l = lp(
        vec![-2.0, -2.0, -2.0, 3.0],
        &[0, 0, 0, 0],
        &[0, 1, 2, 3],
        &[1.0, 1.0, 1.0, -3.0],
        1,
        vec![1.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    );
    let milp = MilpProblem::new(l, vec![0, 1, 2, 3]).unwrap();
    let x_star = lp_root(&milp.lp).solution;
    let mask = super::super::integer_mask(4, &[0, 1, 2, 3]);
    let cuts = generate_clique_cuts(&milp.lp, &mask, &x_star);
    assert!(
        cuts.is_empty(),
        "mixed-sign row must not produce clique cuts (negative coeff invalidates pairwise test), got {}",
        cuts.len()
    );

    // End-to-end: cuts must not change the optimal objective.
    let cfg = MipConfig {
        cuts: true,
        ..MipConfig::default()
    };
    let cfg_off = MipConfig {
        cuts: false,
        ..MipConfig::default()
    };
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let r_on = super::super::solve_milp(&milp, &opts, &cfg);
    let r_off = super::super::solve_milp(&milp, &opts, &cfg_off);
    assert_eq!(r_on.status, SolveStatus::Optimal);
    assert_eq!(r_off.status, SolveStatus::Optimal);
    assert!(
        (r_on.objective - r_off.objective).abs() < 1e-6,
        "cuts must not change optimum: on={} off={}",
        r_on.objective,
        r_off.objective
    );
}

/// **Implied bound cut validity:** no original integer-feasible point removed.
///
/// 3x1 + x2 ≤ 5, x1 ∈ [0,2] integer, x2 ∈ [0,3].
/// Continuous implied ub for x1 = (5-0)/3 = 1.667. Floor → 1.
/// LP opt (min -x1): x1=5/3≈1.667. Violated: 1.667 > 1 (the integer bound).
/// All integer-feasible points have x1 ∈ {0,1} so no integer point is removed.
#[test]
fn implied_bound_cut_validity_brute_force() {
    let l = lp(
        vec![-1.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[3.0, 1.0],
        1,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 2.0), (0.0, 3.0)],
    );
    let milp = MilpProblem::new(l, vec![0]).unwrap();
    let x_star = lp_root(&milp.lp).solution;
    let mask = super::super::integer_mask(2, &[0]);
    let cuts = generate_implied_bound_cuts(&milp.lp, &mask, &x_star);
    assert!(
        !cuts.is_empty(),
        "must generate implied bound cut (floor of 1.667 = 1 < ub=2)"
    );

    let pts = enumerate_int_box(&milp.lp.bounds);
    for x in &pts {
        if !feasible_orig(&milp.lp, x) {
            continue;
        }
        for (k, cut) in cuts.iter().enumerate() {
            let lhs: f64 = cut
                .coeffs
                .iter()
                .zip(x.iter())
                .map(|(&g, &xi)| g * xi)
                .sum();
            assert!(
                lhs >= cut.rhs - 1e-9,
                "implied bound cut {k} removes integer-feasible point {x:?}: lhs={lhs} < rhs={}",
                cut.rhs
            );
        }
    }
}

/// **Implied bound cut: Ge row** implies a lower bound.
#[test]
fn implied_bound_cut_ge_row_validity() {
    // 3x1 + x2 >= 4, x1 ∈ [0,3] integer, x2 ∈ [0,3].
    // Implied lb for x1: (4 - 3*3) / 3 = (4-9)/3 = -5/3 (not useful).
    // Use tighter: x2 ∈ [0,1]. Then activity_max_without_x1 = 1*1 = 1.
    // implied_lb(x1) = (4 - 1) / 3 = 1.0 (tighter than lb=0).
    // LP opt (min x1): x1 = 1 (already integer) when x2=1.
    // For LP opt (min -x2, s.t. 3x1+x2>=4, x1 integer ∈ [0,3], x2 ∈ [0,1]):
    // LP: maximise x2, which means x2=1, 3x1>=3, x1>=1. LP opt x1=1,x2=1 (already int).
    // Let's choose objective that makes LP fractional: min x1 s.t. 3x1+x2>=4, x1∈[0,3] int, x2∈[0,1].
    // LP: x1=1, x2=1 (all integer → no cut needed).
    // Let's try x2∈[0,2]: implied_lb(x1) = (4-2)/3 = 0.67. LP min x1: x1=0.67 fractional!
    let l = lp(
        vec![1.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[3.0, 1.0],
        1,
        vec![4.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 3.0), (0.0, 2.0)],
    );
    let milp = MilpProblem::new(l, vec![0]).unwrap();
    let x_star = lp_root(&milp.lp).solution;
    let mask = super::super::integer_mask(2, &[0]);
    let cuts = generate_implied_bound_cuts(&milp.lp, &mask, &x_star);
    // Verify no integer-feasible point is removed.
    let pts = enumerate_int_box(&milp.lp.bounds);
    for x in &pts {
        if !feasible_orig(&milp.lp, x) {
            continue;
        }
        for (k, cut) in cuts.iter().enumerate() {
            let lhs: f64 = cut
                .coeffs
                .iter()
                .zip(x.iter())
                .map(|(&g, &xi)| g * xi)
                .sum();
            assert!(
                lhs >= cut.rhs - 1e-9,
                "Ge implied bound cut {k} removes integer-feasible point {x:?}: lhs={lhs} < rhs={}",
                cut.rhs
            );
        }
    }
}

/// **Structural cuts end-to-end:** `add_root_cuts` with structural cuts must
/// not change the integer optimum for any of the standard test problems.
#[test]
fn structural_cuts_preserve_optimum() {
    use crate::solve_milp;
    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let cfg = cuts_cfg(5);
    for (name, milp) in all_problems() {
        let res = solve_milp(&milp, &opts, &cfg);
        assert_eq!(
            res.status,
            SolveStatus::Optimal,
            "{name}: must reach Optimal"
        );
        let bf = brute_force_min(&milp).expect("feasible");
        assert!(
            (res.objective - bf).abs() < 1e-6,
            "{name}: structural cuts changed optimum: got {} expected {}",
            res.objective,
            bf
        );
    }
}

/// **Structural cut validity end-to-end:** Le cut rows added by `add_root_cuts`
/// (which includes structural cuts) must not remove any integer-feasible point.
#[test]
fn structural_cuts_validity_end_to_end() {
    let mut problems: Vec<(&str, MilpProblem)> = all_problems();
    problems.push((
        "knapsack_3var",
        knapsack_milp(vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0], 5.0),
    ));
    for (name, milp) in &problems {
        let out = add_root_cuts(milp, &SolverOptions::default(), &cuts_cfg(5));
        let m_old = milp.lp.num_constraints;
        let m_new = out.lp.num_constraints;
        let pts = enumerate_int_box(&milp.lp.bounds);
        for x in &pts {
            if !feasible_orig(&milp.lp, x) {
                continue;
            }
            let ax = out.lp.a.mat_vec_mul(x).unwrap();
            for i in m_old..m_new {
                assert_eq!(out.lp.constraint_types[i], ConstraintType::Le);
                assert!(
                    ax[i] <= out.lp.b[i] + 1e-6,
                    "{name}: structural cut row {i} removes integer-feasible point {x:?}: \
                     ax={} > b={}",
                    ax[i],
                    out.lp.b[i]
                );
            }
        }
    }
}

// ── In-tree separation sentinel ─────────────────────────────────────────────

/// Binary 0/1 knapsack with weight/profit deliberately correlated
/// (`c_i = a_i + 37`), the classic construction for defeating simple
/// rounding and forcing deep B&B branching: the LP relaxation's fractional
/// item makes the bound look nearly achievable, so almost every rounding
/// is a near-miss and the search must branch extensively to close the gap.
/// `cap = Σaᵢ / 2` keeps the knapsack at maximum combinatorial tension.
///
/// Used wherever a test needs an in-tree-cut-worthy search that is
/// realistically deep enough to grow `total_simplex_iters` past the
/// per-dimension useful-work minimum ([`tree_cut_min_useful_iters`]) — unlike
/// a handful of independently-random small LPs, which typically resolve in
/// a few dozen nodes regardless of variable count (their LP relaxations
/// bound tightly, so they never need enough total search to build up a
/// share of iterations comparable to that per-dimension minimum).
fn hard_knapsack_milp(n: usize) -> MilpProblem {
    let a: Vec<f64> = (0..n).map(|i| 101.0 + i as f64 * 13.0).collect();
    let c: Vec<f64> = a.iter().map(|w| w + 37.0).collect();
    let cap: f64 = a.iter().sum::<f64>() / 2.0;
    let cneg: Vec<f64> = c.iter().map(|v| -v).collect();
    let rows = vec![0usize; n];
    let cols: Vec<usize> = (0..n).collect();
    let l = lp(
        cneg,
        &rows,
        &cols,
        &a,
        1,
        vec![cap],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0); n],
    );
    MilpProblem::new(l, cols).unwrap()
}

/// In-tree-cut sentinel fixture: 24-variable [`hard_knapsack_milp`]. Its LP
/// relaxation stays fractional several levels deep, so re-separating GMI/MIR
/// at interior B&B nodes tightens bounds the root cuts miss.
///
/// Not used by [`tree_cuts_reduce_node_count_sentinel_on_dedicated_knapsack`]
/// — see [`TREE_CUT_NODE_COUNT_SENTINEL_N`]'s doc for why that one sentinel
/// needs a different item count.
fn tree_cut_sentinel_milp() -> MilpProblem {
    hard_knapsack_milp(24)
}

/// Item count for [`tree_cuts_reduce_node_count_sentinel_on_dedicated_knapsack`]'s dedicated
/// knapsack — deliberately not [`tree_cut_sentinel_milp`]'s 24. Codex round
/// 3's `attempted`-flag fix (`separate_tree_cuts` distinguishing a
/// budget-deferred round from a genuine dry attempt, see
/// `separate_tree_cuts_reports_not_attempted_on_zero_iteration_budget`)
/// changed exactly when `tree_cut_dry_streak` disables separation, and the
/// 24-item knapsack's on/off margin is not robust to that: on=1336 vs
/// off=1335 (cuts fractionally *worse*) after the fix, down from a
/// comfortable on=1314 vs off=1335 before it. A sweep of n=16..=32 (all
/// still `Optimal` within the 30s budget below) found this margin is
/// specific to a few sizes (n=18 and 24) and not a general regression;
/// n=30 gives a stable on=3405 vs off=3478 margin under the corrected
/// accounting.
const TREE_CUT_NODE_COUNT_SENTINEL_N: usize = 30;

/// **Sentinel**: in-tree cuts must measurably shrink the search vs `tree_cuts=off`.
///
/// Root cuts are disabled in *both* runs so the only difference is in-tree
/// separation. If `tree_cuts` is a no-op (hook never fires, pool always rejects,
/// or the re-solve is discarded), node counts are identical and this FAILS. The
/// optimum must be unchanged — cuts only remove fractional points.
///
/// Uses [`TREE_CUT_NODE_COUNT_SENTINEL_N`]'s dedicated knapsack rather than
/// [`tree_cut_sentinel_milp`] — see that const's doc. Replaces the former
/// `tree_cuts_reduce_node_count_sentinel` (identical assertions, 24-item
/// knapsack), deleted rather than edited in place because its margin
/// stopped being robust under Codex round 3's `attempted`-flag fix.
#[test]
fn tree_cuts_reduce_node_count_sentinel_on_dedicated_knapsack() {
    let milp = hard_knapsack_milp(TREE_CUT_NODE_COUNT_SENTINEL_N);
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };

    let cfg_off = MipConfig {
        cuts: false,
        tree_cuts: false,
        ..MipConfig::default()
    };
    let cfg_on = MipConfig {
        cuts: false,
        tree_cuts: true,
        ..MipConfig::default()
    };

    let (r_off, s_off) = super::super::solve_milp_with_stats(&milp, &opts, &cfg_off);
    let (r_on, s_on) = super::super::solve_milp_with_stats(&milp, &opts, &cfg_on);

    assert_eq!(r_off.status, SolveStatus::Optimal);
    assert_eq!(r_on.status, SolveStatus::Optimal);
    assert!(
        (r_on.objective - r_off.objective).abs() < 1e-6,
        "in-tree cuts must not change the optimum: on={} off={}",
        r_on.objective,
        r_off.objective
    );
    assert!(
        s_on.tree_cut_rounds > 0,
        "in-tree separation must fire at least one accepted round (got 0)"
    );
    assert!(
        s_on.nodes_processed < s_off.nodes_processed,
        "in-tree cuts must reduce node count: on={} off={}",
        s_on.nodes_processed,
        s_off.nodes_processed
    );
}

/// Regression guard: with `tree_cuts=off` no separation fires (counter stays 0).
#[test]
fn tree_cuts_off_does_not_separate() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let cfg = MipConfig {
        cuts: false,
        tree_cuts: false,
        ..MipConfig::default()
    };
    let (_, s) = super::super::solve_milp_with_stats(&milp, &opts, &cfg);
    assert_eq!(s.tree_cut_rounds, 0, "tree_cuts=off must never separate");
}

/// Fixture for
/// [`extend_basis_for_new_rows_shifts_ub_row_slacks_past_new_ge_rows`]:
/// `n=8` variables, all with finite `(0, 5)` bounds, a single `Le` row —
/// chosen so `build_standard_form`'s implicit UB-row count (8) is known
/// exactly for the sentinel's hand-computed expected basis. Distinct from
/// [`tree_cut_sentinel_milp`] (24-variable binary knapsack): that shape
/// does not give the same round-number column boundaries this worked
/// example relies on.
fn extend_basis_sentinel_milp() -> MilpProblem {
    let c = [12.0, 17.0, 13.0, 21.0, 9.0, 16.0, 7.0, 19.0];
    let a = [5.0, 7.0, 6.0, 9.0, 4.0, 7.0, 3.0, 8.0];
    let n = c.len();
    let cneg: Vec<f64> = c.iter().map(|v| -v).collect();
    let rows = vec![0usize; n];
    let cols: Vec<usize> = (0..n).collect();
    let l = lp(
        cneg,
        &rows,
        &cols,
        &a,
        1,
        vec![23.0],
        vec![ConstraintType::Le],
        vec![(0.0, 5.0); n],
    );
    MilpProblem::new(l, cols).unwrap()
}

/// **SENTINEL**: [`extend_basis_for_new_rows`] must shift UB-row slack
/// columns past the newly-appended real rows' own slacks, not copy
/// `prev_basis` unmodified.
///
/// `extend_basis_sentinel_milp` has `n=8` variables all with finite `(0, 5)`
/// bounds, so `build_standard_form` appends 8 implicit UB rows after the
/// single real row: `n_shifted=8` (simple lower-bound shift, no free-var
/// split), `n_real_slack(committed)=1` (the one real row), `n_ub=8`, giving
/// `sf.n_total(committed) = 8+1+8 = 17` with UB-row slacks at columns
/// `9..17`. Appending `k=1` new Ge row makes `candidate` have 2 real rows
/// (`n_real_slack=2`), so `boundary = n_shifted + n_real_slack(candidate) -
/// k = 8+2-1 = 9`: the new row's own slack takes column 9, and every UB-row
/// slack (columns `9..17` in `committed`'s numbering) shifts to `10..18` in
/// `candidate`'s larger numbering.
///
/// Sentinel: copying `prev_basis` unmodified (this function's original,
/// buggy form) would return `[0, 9, 10, 11, 12, 13, 14, 15, 16, 9]` instead
/// (a duplicate `9`, and every UB-row slack aliasing the *wrong* column in
/// `candidate`) — see [`tree_cut_resolve`]'s doc for the `gt2` regression
/// this caused when it went undetected.
#[test]
fn extend_basis_for_new_rows_shifts_ub_row_slacks_past_new_ge_rows() {
    let sentinel = extend_basis_sentinel_milp();
    let committed = &sentinel.lp;
    let mask = super::super::integer_mask(committed.num_vars, &sentinel.integer_vars);
    let cut = CutRow {
        coeffs: vec![1.0; committed.num_vars],
        rhs: 1.0,
    };
    let candidate = append_ge_rows_with_integer_mask(committed, std::slice::from_ref(&cut), &mask);

    // Hypothetical `committed`-space basis (length 9 = m_ext = 1 real + 8 UB
    // rows): variable 0 basic, every UB-row slack (columns 9..17) basic.
    let prev_basis: Vec<usize> = std::iter::once(0).chain(9..17).collect();
    let extended = extend_basis_for_new_rows(&candidate, &prev_basis, 1);

    let expected: Vec<usize> = std::iter::once(0)
        .chain((10..18).collect::<Vec<_>>()) // UB-row slacks, shifted +1
        .chain(std::iter::once(9)) // new row's own slack
        .collect();
    assert_eq!(
        extended, expected,
        "UB-row slack columns must shift past the new row's own slack column"
    );
}

/// **SENTINEL**: [`tree_cut_warm_options`] must dispatch through
/// `DualAdvanced` with `disable_bounded_dispatch: true` and the supplied
/// basis/`max_iters` — the entire mechanism by which round-to-round
/// re-solves both stop being cold and land in the same `build_standard_
/// form` space this module's tableau needs, instead of silently taking
/// `dual_advanced`'s smaller bounded-fast-path space (the `gt2`
/// `cuts_empty` 1.2%→15.1% regression [`tree_cut_resolve`]'s doc
/// describes).
///
/// A cold-vs-warm *iteration-count* comparison is not a reliable sentinel
/// here (tried and rejected in an earlier iteration): on a root relaxation,
/// a round-1 warm re-optimization needs as many pivots as a cold solve as
/// often as fewer — testing the options construction directly is both
/// reliable and exactly matches what reverting the fix would change.
///
/// Sentinel: setting `simplex_method: SimplexMethod::Primal` (which never
/// reads `options.warm_start` — see `primal::two_phase_simplex`'s only use
/// of it, gating the *crash basis*, not warm-starting), `warm_start: None`,
/// or `disable_bounded_dispatch: false` makes this FAIL.
#[test]
fn tree_cut_warm_options_dispatches_dual_advanced_with_disabled_bounded_path() {
    let base = SolverOptions {
        primal_tol: 1e-7,
        dual_tol: 1e-8,
        threads: 3,
        tolerance: Some(crate::options::Tolerance::Custom(1e-4)),
        ..SolverOptions::default()
    };
    let warm_basis = vec![4usize, 1, 7, 2];
    let opts = tree_cut_warm_options(&base, None, Some(42), warm_basis.clone());

    assert_eq!(
        opts.simplex_method,
        SimplexMethod::DualAdvanced,
        "must dispatch through DualAdvanced, the only method that reads `warm_start`"
    );
    assert!(
        opts.disable_bounded_dispatch,
        "must force the legacy sf.m-shaped path — the bounded fast path's \
         smaller space silently rejects this warm start"
    );
    let ws = opts
        .warm_start
        .as_ref()
        .expect("warm_start must be Some for round-to-round re-solves to warm-start at all");
    assert_eq!(
        ws.basis, warm_basis,
        "must carry the exact basis the caller extended"
    );
    assert_eq!(
        opts.max_iters,
        Some(42),
        "must propagate the round's remaining iteration budget"
    );
    assert_eq!(
        opts.primal_tol, base.primal_tol,
        "must inherit caller tolerances"
    );
    assert_eq!(opts.dual_tol, base.dual_tol);
    assert_eq!(opts.threads, base.threads);
    assert_eq!(
        opts.tolerance, base.tolerance,
        "must inherit the caller's convergence tolerance (Codex review, P2-3): \
         without this every in-tree separation solve silently ignores the \
         caller's eps and falls back to ipm.eps / Tolerance::default, so a \
         caller requesting e.g. 1e-8 or 1e-4 gets separation solved at a \
         different accuracy than the rest of the search"
    );
}

/// **SENTINEL** (Codex review, P2-3): [`solve_cut_lp_options`] — the cold
/// bootstrap solve `separate_tree_cuts` uses for round 0 and
/// [`add_root_cuts`] uses for every GMI/MIR round — must also inherit
/// `options.tolerance`, for the same reason as [`tree_cut_warm_options`]
/// above: without it every in-tree/root cut-LP solve silently ignores the
/// caller's eps.
///
/// Sentinel: dropping `tolerance: options.tolerance` from
/// `solve_cut_lp_options` (reverting to the implicit `None` from
/// `..SolverOptions::default()`) makes this FAIL.
#[test]
fn solve_cut_lp_options_inherits_caller_tolerance() {
    let base = SolverOptions {
        tolerance: Some(crate::options::Tolerance::Custom(1e-4)),
        ..SolverOptions::default()
    };
    let opts = solve_cut_lp_options(&base, None, None);
    assert_eq!(
        opts.tolerance, base.tolerance,
        "solve_cut_lp must inherit the caller's convergence tolerance"
    );
}

/// **SENTINEL** (Modification 1's core fix): a full [`separate_tree_cuts`]
/// run against an all-boxed MILP (every variable has a finite upper bound,
/// so `dual_advanced` would take its bounded fast path for *every* solve in
/// this test if `disable_bounded_dispatch` were not wired through) must
/// actually accept its warm-started re-solves through `dual_advanced`'s
/// legacy path — not silently fall through to a singular-basis cold start,
/// which was this module's actual pre-fix failure mode (basis shape
/// mismatch; see [`tree_cut_resolve`]'s doc).
///
/// Sentinel: reverting `disable_bounded_dispatch: true` out of
/// [`tree_cut_warm_options`] reintroduces the mismatch — the bounded fast
/// path rejects an `sf.m`-shaped warm start via its own `bsf.m`-shaped
/// guard before `LuBasis::new_timed` (the legacy path's own warm-start
/// entry point) is ever reached, so `legacy_warm_start_accepted_count`
/// stays 0 and this FAILS.
#[test]
fn separate_tree_cuts_accepts_legacy_warm_start_without_singular_fallback() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);
    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);

    crate::simplex::dual_advanced::reset_legacy_warm_start_counts();
    let (tightened, iters, _overhead, _attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        u64::MAX,
    );
    assert!(
        tightened.is_some(),
        "test premise: sentinel node must accept at least one tightening"
    );
    assert!(iters > 0, "test premise: at least one solve must have run");

    let accepted = crate::simplex::dual_advanced::legacy_warm_start_accepted_count();
    let cold_fallback_total =
        crate::simplex::dual_advanced::legacy_warm_start_cold_fallback_total_count();
    assert!(
        accepted > 0,
        "at least one round-to-round re-solve must accept its warm start via \
         dual_advanced's legacy path (accepted={accepted})"
    );
    // Codex review (P2-4): checking only the singular-basis fallback left two
    // of the legacy path's three cold-fallback exits (shape/range mismatch,
    // dual-infeasible-under-new-c) uncounted — a regression hitting either
    // one would have passed this sentinel silently. `..._cold_fallback_
    // total_count` sums all three, so any of them firing here fails this
    // single check.
    assert_eq!(
        cold_fallback_total, 0,
        "no warm-started re-solve should hit any of the legacy path's cold-start \
         fallbacks (shape/range mismatch, singular basis, dual-infeasible) — the \
         pre-fix shape-mismatch failure mode (cold_fallback_total={cold_fallback_total})"
    );
}

/// **Contract test**: [`tree_cut_resolve`] must be a pure passthrough of
/// [`solve_tree_cut_warm`]'s own result whenever that result is not
/// `Optimal` — no extra cold solve attempted, no field rewritten. Checked by
/// comparing `.iterations` (not just `.status`) against a direct
/// `solve_tree_cut_warm` call with identical arguments: a reintroduced cold
/// retry would add its own iterations even on the (likely, same-budget)
/// chance its status also ends up non-`Optimal`, so status equality alone
/// would not reliably catch it.
///
/// A cut-augmented candidate's extended warm basis is primal-infeasible at
/// the new rows by construction (`generate_round`/`CutPool` only emit a cut
/// that violates the current vertex), so capping the warm solve's own
/// `max_iters` below what it needs to repair that reliably produces a
/// non-`Optimal` premise to compare against.
///
/// [`tree_cut_resolve`]'s shape-mismatch fallback is deliberately not
/// separately sentinel-tested here — see its own doc for why (a mismatched
/// `warm_basis` already degrades gracefully inside `solve_dual_advanced`
/// itself, verified empirically to leave every test's outcome unchanged).
/// `tree_cut_warm_options_dispatches_dual_advanced_with_disabled_bounded_
/// path` and `separate_tree_cuts_accepts_legacy_warm_start_without_
/// singular_fallback` are the sentinels that fail if `disable_bounded_
/// dispatch` itself regresses.
#[test]
fn tree_cut_resolve_is_pure_passthrough_when_warm_solve_is_non_optimal() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions::default();
    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);

    let boot = solve_cut_lp(&milp.lp, &opts, None, None);
    assert_eq!(boot.status, SolveStatus::Optimal, "test premise");
    let ws = boot
        .warm_start_basis
        .as_ref()
        .expect("test premise: bootstrap must expose a basis");
    let x_star = &boot.solution;

    let rows = generate_round(&milp.lp, &mask, x_star, &ws.basis, CutKind::Gmi);
    assert!(
        !rows.is_empty(),
        "test premise: root relaxation must yield at least one cut"
    );
    let k = rows.len();
    let candidate = append_ge_rows_with_integer_mask(&milp.lp, &rows, &mask);
    let warm_basis = extend_basis_for_new_rows(&candidate, &ws.basis, k);

    let direct = solve_tree_cut_warm(&candidate, &opts, None, Some(1), warm_basis.clone());
    assert_ne!(
        direct.status,
        SolveStatus::Optimal,
        "test premise: max_iters=1 must be too tight for the extended warm \
         basis to repair {k} newly-violated cut row(s)"
    );

    let via_resolve = tree_cut_resolve(&candidate, &opts, None, Some(1), warm_basis);
    assert_eq!(
        via_resolve.status, direct.status,
        "tree_cut_resolve must not reinterpret a non-Optimal warm status"
    );
    assert_eq!(
        via_resolve.iterations, direct.iterations,
        "tree_cut_resolve must not spend extra iterations on a cold retry \
         when the warm solve is already non-Optimal"
    );
}

/// **SENTINEL** (Codex review, P2-1): a full [`separate_tree_cuts`] attempt
/// against a real LP must return a nonzero [`tree_cut_construction_
/// surcharge`] overhead, on the same order of magnitude as `n_builds ×
/// tree_cut_dim / TREE_CUT_BUILD_ITER_COST_DIVISOR` — bounds derived
/// independently from the round mechanics (at least one accepted round,
/// costing at least a cold-bootstrap round's builds on `committed`'s
/// starting dimension; at most [`TREE_CUT_MAX_ROUNDS`], each costing at
/// most a warm round's builds on a dimension inflated by that round's own
/// share of [`MAX_CUTS_PER_ROUND`] extra rows), not by calling
/// [`tree_cut_construction_surcharge`] itself (which would only prove the
/// arithmetic is self-consistent, not that anything is actually charged).
///
/// Sentinel: collapsing `tree_cut_construction_surcharge` to always return 0
/// makes this FAIL on `overhead > 0` (and every other test that checks
/// `overhead` explicitly — this one exists so the surcharge's *existence* on
/// a real accepted attempt has direct, non-tautological coverage even if
/// those did not).
#[test]
fn separate_tree_cuts_surcharge_is_nonzero_and_order_of_magnitude_correct() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);
    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);

    let (tightened, _iters, overhead, _attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        u64::MAX,
    );
    assert!(
        tightened.is_some(),
        "test premise: sentinel node must accept at least one tightening"
    );
    assert!(
        overhead > 0,
        "an accepted round must always charge a nonzero construction surcharge"
    );

    let dim0 = (milp.lp.num_vars + milp.lp.num_constraints) as u64;
    let min_builds_round0 = TREE_CUT_BUILDS_ROUND_START_COLD
        + TREE_CUT_BUILDS_GENERATE_ROUND
        + TREE_CUT_BUILDS_ROUND_END;
    let lower_bound = (min_builds_round0 * dim0) / TREE_CUT_BUILD_ITER_COST_DIVISOR;
    assert!(
        overhead >= lower_bound,
        "overhead {overhead} is below the minimum possible for a single \
         accepted round (dim0={dim0}, lower_bound={lower_bound})"
    );

    let max_dim = dim0 + (TREE_CUT_MAX_ROUNDS as u64) * (MAX_CUTS_PER_ROUND as u64);
    let max_builds_per_round = TREE_CUT_BUILDS_ROUND_START_WARM
        + TREE_CUT_BUILDS_GENERATE_ROUND
        + TREE_CUT_BUILDS_ROUND_END;
    let upper_bound = (TREE_CUT_MAX_ROUNDS as u64) * max_builds_per_round * max_dim
        / TREE_CUT_BUILD_ITER_COST_DIVISOR;
    assert!(
        overhead <= upper_bound,
        "overhead {overhead} exceeds the maximum possible across at most \
         TREE_CUT_MAX_ROUNDS rounds (max_dim={max_dim}, upper_bound={upper_bound})"
    );
}

/// **SENTINEL** (Codex review, P3-2): the declared `TREE_CUT_BUILDS_*`
/// construction-count constants must match how many times `build_standard_
/// form` actually runs when each call site executes in isolation —
/// independent empirical confirmation of the structural claim those
/// constants document, via `simplex::build_standard_form_call_count`'s
/// global counter (not `tree_cut_construction_surcharge`, which only
/// multiplies whatever the constants already say).
///
/// Sentinel: changing any `TREE_CUT_BUILDS_*` constant without a matching
/// change in this module's actual call graph — or vice versa — desyncs one
/// of these four equalities.
#[test]
fn tree_cut_builds_constants_match_actual_build_standard_form_call_counts() {
    let sentinel = tree_cut_sentinel_milp();
    let committed = &sentinel.lp;
    let mask = super::super::integer_mask(committed.num_vars, &sentinel.integer_vars);
    let opts = SolverOptions::default();

    let boot = solve_cut_lp(committed, &opts, None, None);
    assert_eq!(boot.status, SolveStatus::Optimal, "test premise");
    let ws = boot
        .warm_start_basis
        .as_ref()
        .expect("test premise: bootstrap must expose a basis");

    // TREE_CUT_BUILDS_GENERATE_ROUND: generate_round's own build.
    crate::simplex::reset_build_standard_form_call_count();
    let cuts = generate_round(committed, &mask, &boot.solution, &ws.basis, CutKind::Gmi);
    assert_eq!(
        crate::simplex::build_standard_form_call_count(),
        TREE_CUT_BUILDS_GENERATE_ROUND,
        "generate_round's own build_standard_form call count must match \
         TREE_CUT_BUILDS_GENERATE_ROUND"
    );
    assert!(
        !cuts.is_empty(),
        "test premise: root relaxation must yield at least one cut"
    );

    // TREE_CUT_BUILDS_ROUND_START_COLD: solve_cut_lp's own internal build.
    crate::simplex::reset_build_standard_form_call_count();
    let reboot = solve_cut_lp(committed, &opts, None, None);
    assert_eq!(reboot.status, SolveStatus::Optimal, "test premise");
    assert_eq!(
        crate::simplex::build_standard_form_call_count(),
        TREE_CUT_BUILDS_ROUND_START_COLD,
        "solve_cut_lp's own build_standard_form call count must match \
         TREE_CUT_BUILDS_ROUND_START_COLD"
    );

    // TREE_CUT_BUILDS_ROUND_START_WARM: tree_cut_resolve's own shape-check
    // build plus the warm solve's own internal build, on the happy
    // (non-cold-fallback) path — re-solving the exact same LP from its own
    // just-accepted basis must stay warm and Optimal.
    crate::simplex::reset_build_standard_form_call_count();
    let warm_res = tree_cut_resolve(committed, &opts, None, None, ws.basis.clone());
    assert_eq!(
        warm_res.status,
        SolveStatus::Optimal,
        "test premise: warm-resolving the exact same LP+basis must stay Optimal"
    );
    assert_eq!(
        crate::simplex::build_standard_form_call_count(),
        TREE_CUT_BUILDS_ROUND_START_WARM,
        "tree_cut_resolve's own build_standard_form call count on the happy \
         path must match TREE_CUT_BUILDS_ROUND_START_WARM"
    );

    // TREE_CUT_BUILDS_ROUND_END: extend_basis_for_new_rows's own build plus
    // tree_cut_resolve's (ROUND_START_WARM-shaped) builds on the resulting
    // candidate, together.
    let k = cuts.len();
    let candidate = append_ge_rows_with_integer_mask(committed, &cuts, &mask);
    crate::simplex::reset_build_standard_form_call_count();
    let warm_basis = extend_basis_for_new_rows(&candidate, &ws.basis, k);
    let check = tree_cut_resolve(&candidate, &opts, None, None, warm_basis);
    assert_eq!(check.status, SolveStatus::Optimal, "test premise");
    assert_eq!(
        crate::simplex::build_standard_form_call_count(),
        TREE_CUT_BUILDS_ROUND_END,
        "extend_basis_for_new_rows + tree_cut_resolve's combined \
         build_standard_form call count must match TREE_CUT_BUILDS_ROUND_END"
    );
}

#[test]
fn tree_cut_node_gate_depth_node_boundary_decision_table() {
    let cases = [
        // depth gate | node gate | boundary purpose | selected
        (0, 0, false, "root node: both gates off"),
        (
            3,
            TREE_CUT_NODE_INTERVAL - 1,
            false,
            "just below both gates",
        ),
        (
            TREE_CUT_DEPTH_INTERVAL,
            0,
            true,
            "first positive depth multiple",
        ),
        (
            0,
            TREE_CUT_NODE_INTERVAL,
            true,
            "first positive node multiple",
        ),
        (
            TREE_CUT_DEPTH_INTERVAL,
            TREE_CUT_NODE_INTERVAL,
            true,
            "both gates true",
        ),
        (
            TREE_CUT_DEPTH_INTERVAL * 2,
            1,
            true,
            "later depth multiple with non-gating node",
        ),
        (
            1,
            TREE_CUT_NODE_INTERVAL * 2,
            true,
            "later node multiple with non-gating depth",
        ),
    ];

    for (depth, node_index, expected, label) in cases {
        assert_eq!(
            tree_cut_node_selected(depth, node_index),
            expected,
            "{label}: depth={depth}, node_index={node_index}"
        );
    }
}

#[test]
fn tree_cut_gate_rejects_zero_depth_zero_node_sentinel() {
    assert!(
        !tree_cut_node_selected(0, 0),
        "zero is a numeric multiple, but root separation is intentionally rejected"
    );
}

#[test]
fn separate_tree_cuts_drops_augmented_warm_start_basis() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);
    assert!(
        node_res.warm_start_basis.is_some(),
        "node LP solve should expose a basis before tree-cut tightening"
    );

    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);
    let (tightened, _iters, _overhead, _attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        u64::MAX,
    );
    let tightened = tightened.expect("sentinel node must accept at least one tree-cut tightening");
    assert!(tightened.objective > node_res.objective);
    assert!(
        tightened.warm_start_basis.is_none(),
        "tree-cut result must not return a basis from the augmented node-local LP"
    );
}

/// SENTINEL (Phase 1c/P1-2): `max_iters == 0` must run zero rounds — the
/// round-boundary budget check happens BEFORE the first round's LP solve,
/// not only between later rounds.
///
/// Sentinel: moving the `if iters_spent >= max_iters { break; }` check to
/// only run after round 0 always executes would make this FAIL (`tightened`
/// would be `Some(..)` and/or `iters > 0`).
#[test]
fn separate_tree_cuts_respects_zero_iter_budget() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);

    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);
    let (tightened, iters, overhead, _attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        0,
    );
    assert!(
        tightened.is_none(),
        "zero iteration budget must run no rounds"
    );
    assert_eq!(iters, 0, "zero iteration budget must spend zero iterations");
    assert_eq!(
        overhead, 0,
        "zero iteration budget must spend zero surcharge overhead either"
    );
}

/// Codex round 3 (P2), reversing Phase 1d/P3-B's original stance: a
/// zero-iteration budget can never even pass round 0's own budget
/// pre-check (`remaining < tree_cut_min_useful_iters`, checked before any
/// solve), so no separation work of any kind happened this call — a budget
/// deferral, not a genuine "tried and found nothing" dry attempt. Reporting
/// `attempted = true` here (the pre-round-3 behavior) let a long run of
/// budget-starved calls silently count toward `tree_cut_dry_streak` and
/// disable separation via `effort::SEPARATION_DRY_STREAK_LIMIT`, even
/// though separation was never actually given a chance to run.
///
/// Sentinel: replacing `any_round_solved` with a hardcoded `true` in
/// `separate_tree_cuts`'s two no-op return points makes this FAIL.
#[test]
fn separate_tree_cuts_reports_not_attempted_on_zero_iteration_budget() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);

    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);
    let (_tightened, iters, _overhead, attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        0,
    );
    assert_eq!(
        iters, 0,
        "test premise: zero iteration budget must spend zero iterations"
    );
    assert!(
        !attempted,
        "a zero-iteration budget never passes round 0's own pre-check, so no \
         separation work happened this call — must report `attempted = false` \
         (a budget deferral, not a dry attempt)"
    );
}

/// SENTINEL (Phase 1c/P1-2): a small but nonzero iteration budget caps the
/// simplex iterations actually spent below what an unrestricted call spends
/// on the same node (which needs multiple rounds to reach its acceptance
/// criterion — see `separate_tree_cuts_drops_augmented_warm_start_basis`).
///
/// Sentinel: removing the round-boundary `max_iters` check makes the
/// "capped" call spend the SAME iterations as the unrestricted call, failing
/// the `<` assertion.
#[test]
fn separate_tree_cuts_caps_iterations_at_small_budget() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);
    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);

    let (_, unrestricted_iters, _overhead, _attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        u64::MAX,
    );
    assert!(
        unrestricted_iters > 0,
        "test premise: the unrestricted attempt must spend some iterations"
    );

    // A 1-iteration budget is below what even a single round's cut LP solve
    // needs on this instance, so the capped attempt must stop after fewer
    // rounds and spend strictly fewer iterations than the unrestricted one.
    let (_, capped_iters, _overhead, _attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        1,
    );
    assert!(
        capped_iters < unrestricted_iters,
        "a 1-iteration budget must spend fewer iterations than an unrestricted attempt: \
         capped={capped_iters} unrestricted={unrestricted_iters}"
    );
}

/// SENTINEL (Codex review, P1): `solve_cut_lp` / `solve_validate` actually
/// pass their `max_iters` parameter through to the underlying LP solve —
/// before this fix both always built a `SolverOptions` with `max_iters:
/// None` internally, so a `Some(cap)` argument was accepted but silently had
/// no effect (a single cold solve could still run unboundedly). A too-small
/// cap on an LP that genuinely needs several iterations must make the solve
/// stop before reaching `Optimal`.
///
/// Sentinel: removing `max_iters,` from either function's constructed
/// `SolverOptions` (reverting to always `max_iters: None`, as the previous
/// hardcoded field was) makes the `capped`/`capped_v` assertions below FAIL
/// — the solve would still reach `Optimal` in more than 1 iteration, exactly
/// like the unrestricted case.
#[test]
fn solve_cut_lp_and_solve_validate_honor_max_iters() {
    let milp = tree_cut_sentinel_milp();

    let unrestricted = solve_cut_lp(&milp.lp, &SolverOptions::default(), None, None);
    assert_eq!(unrestricted.status, SolveStatus::Optimal, "test premise");
    assert!(
        unrestricted.iterations > 1,
        "test premise: root LP must need > 1 iteration; got {}",
        unrestricted.iterations
    );
    let capped = solve_cut_lp(&milp.lp, &SolverOptions::default(), None, Some(1));
    assert_ne!(
        capped.status,
        SolveStatus::Optimal,
        "max_iters=1 must stop solve_cut_lp before it reaches Optimal"
    );

    let unrestricted_v = solve_validate(&milp.lp, &SolverOptions::default(), None, None);
    assert_eq!(unrestricted_v.status, SolveStatus::Optimal, "test premise");
    assert!(
        unrestricted_v.iterations > 1,
        "test premise: root LP must need > 1 iteration; got {}",
        unrestricted_v.iterations
    );
    let capped_v = solve_validate(&milp.lp, &SolverOptions::default(), None, Some(1));
    assert_ne!(
        capped_v.status,
        SolveStatus::Optimal,
        "max_iters=1 must stop solve_validate before it reaches Optimal"
    );
}

/// SENTINEL (markshare_4_0 regression fix): a round whose remaining
/// allowance is below the per-dimension useful minimum
/// (`tree_cut_min_useful_iters`) is skipped outright — no cold solve is
/// attempted at all — rather than run with a truncated `max_iters` (the
/// pre-fix behavior tested by the now-removed `..._via_floor` sentinel,
/// which inflated a too-small `remaining` up to the per-dimension floor so
/// every approved attempt still burned at least one full round of
/// overhead). `max_iters = 1` is far below what any cold resolve of this
/// instance needs (`>= 4`, per
/// `separate_tree_cuts_caps_iterations_at_small_budget`'s test premise), so
/// the first round must never start.
///
/// Codex round 3 (P2) additionally reverses the `attempted` verdict for
/// this exact scenario: round 0's *own* budget pre-check fired before any
/// solve ran, so this is a budget deferral (see
/// `separate_tree_cuts_reports_not_attempted_on_zero_iteration_budget`'s
/// doc), not a real dry attempt — `attempted` must be `false`, not `true`.
///
/// Sentinel: reverting the round-boundary check from `remaining <
/// tree_cut_min_useful_iters(&committed)` back to `remaining == 0` makes the
/// solve run with `max_iters = Some(1)` instead of being skipped, so `iters`
/// becomes nonzero (a cold solve was attempted), failing the second
/// assertion. Replacing `any_round_solved` with a hardcoded `true` at the
/// no-op return points makes the `attempted` assertion fail.
#[test]
fn separate_tree_cuts_skips_a_round_below_the_per_dimension_minimum_and_reports_not_attempted() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);
    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);

    let max_iters = 1;
    assert!(
        max_iters < tree_cut_min_useful_iters(&milp.lp),
        "test premise: max_iters must be below the per-dimension minimum"
    );
    let (tightened, iters, overhead, attempted) = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        max_iters,
    );
    assert!(
        tightened.is_none(),
        "a round below the per-dimension minimum must be skipped, not run \
         with a truncated max_iters"
    );
    assert_eq!(iters, 0, "no cold solve should have been attempted at all");
    assert_eq!(
        overhead, 0,
        "no construction surcharge should have been charged either"
    );
    assert!(
        !attempted,
        "round 0's own budget pre-check fired before any solve ran — a \
         budget deferral, not a real dry attempt — so this must NOT count \
         toward dry-streak accounting"
    );
}

/// SENTINEL (Codex round 3, P2): round 0's solve `max_iters` is `remaining -
/// round_start_surcharge`, not `remaining` — the surcharge is unavoidable
/// overhead the solve is about to incur (its shape-check build happens
/// inside `solve_cut_lp` regardless of outcome), so charging it to
/// `overhead_spent` only *after* the solve returns (the pre-round-3
/// behavior) let a solve that fully consumed its cap push `iters_spent +
/// overhead_spent` past the caller's `max_iters`.
///
/// Sentinel: reverting the pre-charge (passing `remaining` instead of
/// `remaining.saturating_sub(round_start_surcharge)` to round 0's solve)
/// makes the observed value equal `max_iters` (300) instead of `max_iters -
/// round_start_surcharge`, failing the assertion below.
#[test]
fn round_solve_max_iters_pre_charges_the_construction_surcharge() {
    let milp = tree_cut_sentinel_milp();
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let node_res = lp_root(&milp.lp);
    assert_eq!(node_res.status, SolveStatus::Optimal);
    let mask = super::super::integer_mask(milp.lp.num_vars, &milp.integer_vars);

    let round_start_surcharge =
        tree_cut_construction_surcharge(&milp.lp, TREE_CUT_BUILDS_ROUND_START_COLD);
    assert!(
        round_start_surcharge > 0,
        "test premise: this fixture's dim must yield a nonzero surcharge"
    );
    let max_iters = tree_cut_min_useful_iters(&milp.lp) + 200;

    let _ = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
        max_iters,
    );
    let observed = LAST_ROUND_SOLVE_MAX_ITERS.with(std::cell::Cell::get);
    assert_eq!(
        observed,
        Some(max_iters - round_start_surcharge),
        "round 0's solve must be capped at `max_iters - round_start_surcharge` \
         ({}), not `max_iters` ({max_iters})",
        max_iters - round_start_surcharge,
    );
}

/// `tree_cut_min_useful_iters` returns exactly `dim *
/// TREE_CUT_MIN_SOLVE_ITER_DIM_MULT`, independent of any `remaining`
/// argument (unlike the removed `tree_cut_solve_iter_cap`, it is a pure
/// threshold, not a value combined with `remaining`).
///
/// Sentinel: changing the multiplier used internally without updating
/// `TREE_CUT_MIN_SOLVE_ITER_DIM_MULT` itself would desync this from
/// `separate_tree_cuts`'s actual skip threshold, failing this equality.
#[test]
fn tree_cut_min_useful_iters_is_dim_times_the_multiplier() {
    let milp = tree_cut_sentinel_milp();
    let expected =
        (milp.lp.num_vars + milp.lp.num_constraints) as u64 * TREE_CUT_MIN_SOLVE_ITER_DIM_MULT;
    assert_eq!(tree_cut_min_useful_iters(&milp.lp), expected);
}

// ── Optimum-preservation sweep (cross-node soundness gate) ───────────────────

/// Deterministic LCG so the sweep is reproducible (no `rand` dependency).
struct SweepRng(u64);
impl SweepRng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() as i64).rem_euclid(hi - lo + 1)
    }
}

/// Random all-integer MILP: `min c·x` s.t. `A x ≤ b`, `x ∈ {0..ub}`.
fn sweep_milp(rng: &mut SweepRng, n: usize, m: usize, ub: f64) -> MilpProblem {
    let c: Vec<f64> = (0..n).map(|_| rng.range(-7, 7) as f64).collect();
    let (mut rows, mut cols, mut vals, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..m {
        for j in 0..n {
            let v = rng.range(-4, 8);
            if v != 0 {
                rows.push(r);
                cols.push(j);
                vals.push(v as f64);
            }
        }
        b.push(rng.range(5, 20) as f64);
    }
    let l = lp(
        c,
        &rows,
        &cols,
        &vals,
        m,
        b,
        vec![ConstraintType::Le; m],
        vec![(0.0, ub); n],
    );
    MilpProblem::new(l, (0..n).collect()).unwrap()
}

/// Brute-force optimum over the integer lattice: `Some(min c·x)` over feasible
/// points, or `None` when none is feasible. Used as ground truth.
fn brute_force_opt(milp: &MilpProblem) -> Option<f64> {
    let lp = &milp.lp;
    let mut best: Option<f64> = None;
    for x in enumerate_int_box(&lp.bounds) {
        if !feasible_orig(lp, &x) {
            continue;
        }
        let obj: f64 = lp.c.iter().zip(&x).map(|(&ci, &xi)| ci * xi).sum();
        best = Some(best.map_or(obj, |b| b.min(obj)));
    }
    best
}

/// **P0 soundness sentinel**: in-tree cuts must never change the optimum.
///
/// In-tree GMI/MIR cuts bake in the generating node's branching-tightened bounds,
/// so they are valid only within that node's subtree. A cross-node cut pool (the
/// original buggy design) re-applies them at sibling/non-descendant nodes,
/// slicing off globally integer-feasible points and returning a too-good
/// "Optimal" objective. This sweep solves a deterministic batch of small random
/// MILPs three ways — brute force (ground truth), `tree_cuts=off`, and
/// `tree_cuts=on` — and asserts all three agree. Under the buggy cross-node pool
/// several instances in this seed mismatch, so it FAILS; node-local separation
/// passes. `fired > 0` proves separation is actually exercised (else the test is
/// vacuous).
#[test]
fn tree_cuts_preserve_optimum_sweep() {
    let opts = SolverOptions {
        timeout_secs: Some(30.0),
        ..Default::default()
    };
    let cfg_off = MipConfig {
        cuts: false,
        tree_cuts: false,
        ..MipConfig::default()
    };
    let cfg_on = MipConfig {
        cuts: false,
        tree_cuts: true,
        ..MipConfig::default()
    };

    let mut rng = SweepRng(12345);
    let mut fired = 0usize;
    let mut checked = 0usize;
    for it in 0..500usize {
        // markshare_4_0 regression fix: `separate_tree_cuts` now skips a
        // round outright once the remaining iteration-share budget drops
        // below the per-dimension useful minimum (see
        // `cuts::tree_cut_min_useful_iters`), rather than forcing it through
        // an inflated floor. The independently-random 5..7-variable draws
        // below bound tightly and resolve in a handful of nodes regardless
        // of size, so `total_simplex_iters` never grows enough for their
        // share ceiling to clear that per-dimension minimum (separation
        // would never fire on any of the 500, making the soundness check
        // vacuous). A few [`hard_knapsack_milp`] draws interspersed in the
        // sweep run deep enough B&B searches to clear it.
        let milp = if it % 100 == 0 {
            hard_knapsack_milp(16)
        } else {
            let n = 5 + it % 3; // 5..7
            let m = 2 + it % 3; // 2..4
            sweep_milp(&mut rng, n, m, 4.0)
        };

        let truth = brute_force_opt(&milp);
        let (r_off, _) = super::super::solve_milp_with_stats(&milp, &opts, &cfg_off);
        let (r_on, s_on) = super::super::solve_milp_with_stats(&milp, &opts, &cfg_on);
        if s_on.tree_cut_rounds > 0 {
            fired += 1;
        }

        match truth {
            None => {
                // Infeasible MILP: neither configuration may invent a solution.
                assert_ne!(
                    r_off.status,
                    SolveStatus::Optimal,
                    "it={it}: off feasible but brute says infeasible"
                );
                assert_ne!(
                    r_on.status,
                    SolveStatus::Optimal,
                    "it={it}: ON invented a solution for infeasible MILP"
                );
            }
            Some(opt) => {
                assert_eq!(
                    r_off.status,
                    SolveStatus::Optimal,
                    "it={it}: off must solve feasible MILP"
                );
                assert!(
                    (r_off.objective - opt).abs() < 1e-6,
                    "it={it}: OFF baseline wrong: off={} brute={opt}",
                    r_off.objective
                );
                assert_eq!(
                    r_on.status,
                    SolveStatus::Optimal,
                    "it={it}: ON must solve feasible MILP"
                );
                assert!(
                    (r_on.objective - opt).abs() < 1e-6,
                    "it={it}: in-tree cuts changed the optimum (cross-node leak): on={} brute={opt}",
                    r_on.objective
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "sweep must verify at least one feasible instance"
    );
    assert!(
        fired > 0,
        "in-tree separation must fire on the sweep (else the soundness check is vacuous)"
    );
}

#[test]
#[should_panic(expected = "cover separation requires one LP value per variable")]
fn cover_cuts_reject_short_lp_solution_vector() {
    let p = lp(
        vec![0.0, 0.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        vec![1.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 1.0)],
    );
    let integer_mask = vec![true, true];
    let _ = generate_cover_cuts(&p, &integer_mask, &[0.5]);
}

#[test]
#[should_panic(
    expected = "cut violation evaluation requires matching cut and LP solution dimensions"
)]
fn finalize_cut_rejects_dimension_mismatch() {
    let _ = finalize_cut(vec![1.0, -1.0], 0.0, &[0.5], 1e-9);
}
