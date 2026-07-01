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
use crate::sparse::CscMatrix;

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
    super::solve_cut_lp(p, &SolverOptions::default(), None)
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

#[test]
#[should_panic(expected = "cut-augmented LP is valid")]
fn append_ge_rows_does_not_scale_away_large_bound_gap() {
    let mut milp = p_box_le();
    milp.lp.bounds[0] = (1.0e12 + 1.0, 1.0e12);
    let cuts = [CutRow {
        coeffs: vec![1.0, 0.0],
        rhs: 0.5,
    }];

    let _ = append_ge_rows(&milp.lp, &cuts);
}

#[test]
#[should_panic(expected = "cut-augmented LP is valid")]
fn append_ge_rows_keeps_material_invalid_bounds_as_error() {
    let mut milp = p_box_le();
    milp.lp.bounds[0] = (1.0 + ZERO_TOL * 10.0, 1.0);
    let cuts = [CutRow {
        coeffs: vec![1.0, 0.0],
        rhs: 0.5,
    }];

    let _ = append_ge_rows(&milp.lp, &cuts);
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
        let check = solve_cut_lp(&out.lp, &SolverOptions::default(), None);
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

    let check = solve_validate(&le_bad, &SolverOptions::default(), None);
    assert_ne!(
        check.status,
        SolveStatus::Optimal,
        "infeasible Le cut LP (0·x <= -1) must not validate as Optimal — \
         this is the condition the add_root_cuts fallback guards against"
    );

    // Confirm the original LP (fallback target) is still solvable.
    let orig_check = solve_validate(&milp.lp, &SolverOptions::default(), None);
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

/// Knapsack-style general-integer MILP used as the in-tree-cut sentinel.
///
/// max Σ c_j x_j  s.t.  Σ a_j x_j ≤ 23,  x_j ∈ {0..5} integer (min of −c).
/// Its LP relaxation stays fractional several levels deep, so re-separating
/// GMI/MIR at interior B&B nodes tightens bounds the root cuts miss.
fn tree_cut_sentinel_milp() -> MilpProblem {
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

/// **Sentinel**: in-tree cuts must measurably shrink the search vs `tree_cuts=off`.
///
/// Root cuts are disabled in *both* runs so the only difference is in-tree
/// separation. If `tree_cuts` is a no-op (hook never fires, pool always rejects,
/// or the re-solve is discarded), node counts are identical and this FAILS. The
/// optimum must be unchanged — cuts only remove fractional points.
#[test]
fn tree_cuts_reduce_node_count_sentinel() {
    let milp = tree_cut_sentinel_milp();
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
    let tightened = separate_tree_cuts(
        &milp.lp,
        &mask,
        &opts,
        &node_res,
        TREE_CUT_DEPTH_INTERVAL,
        1,
    )
    .expect("sentinel node must accept at least one tree-cut tightening");
    assert!(tightened.objective > node_res.objective);
    assert!(
        tightened.warm_start_basis.is_none(),
        "tree-cut result must not return a basis from the augmented node-local LP"
    );
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
        let n = 5 + it % 3; // 5..7
        let m = 2 + it % 3; // 2..4
        let milp = sweep_milp(&mut rng, n, m, 4.0);

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
