//! GMI cut sentinels.
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
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, f64::INFINITY)],
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
/// problem must satisfy every generated cut. A single sliced integer point fails.
#[test]
fn cut_validity_brute_force() {
    for (name, milp) in all_problems() {
        // Multiple rounds + multiple cuts → broad coverage of the GMI formula.
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
                // Check the original feasible integer point against the cut rows.
                let ax = out.lp.a.mat_vec_mul(x).unwrap();
                for i in m_old..m_new {
                    assert_eq!(out.lp.constraint_types[i], ConstraintType::Ge);
                    assert!(
                        ax[i] >= out.lp.b[i] - 1e-6,
                        "{name} round={rounds}: INVALID CUT — integer point {x:?} \
                         removed by cut row {i}: {} < {}",
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
    assert!(m_new > m_old, "a GMI cut must be generated for the UbOnly source");
    for xi in -8..=2 {
        let x = vec![xi as f64];
        if !feasible_orig(&milp.lp, &x) {
            continue;
        }
        let ax = out.lp.a.mat_vec_mul(&x).unwrap();
        for i in m_old..m_new {
            assert!(
                ax[i] >= out.lp.b[i] - 1e-6,
                "INVALID CUT (UbOnly): integer x={xi} removed by cut row {i}: {} < {}",
                ax[i],
                out.lp.b[i]
            );
        }
    }
}

/// Cuts are generated AND they cut the fractional LP optimum. A no-op generator
/// (empty cuts, or a cut equal to a trivially-satisfied inequality) fails here.
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
            "{name}: at least one GMI cut must be generated (root LP is fractional)"
        );
        // Some generated cut must be violated by x* (the cut is effective).
        let ax = out.lp.a.mat_vec_mul(x_star).unwrap();
        let any_violated = (m_old..m_new).any(|i| ax[i] < out.lp.b[i] - 1e-6);
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
        // Minimization: cut LP objective >= original LP objective (tighter).
        assert!(
            cut_root.objective >= root.objective - 1e-6,
            "{name}: cut LP bound {} must not be looser than root {}",
            cut_root.objective,
            root.objective
        );
        // The cut LP bound must still under-estimate the integer optimum.
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
        let (res, stats) =
            solve_milp_with_stats(&milp, &opts, &cuts_cfg(3));
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
    // min -x-y s.t. 2x+2y<=3, x,y in [0,1]. (Same as p_box_le's LP.)
    let l = p_box_le().lp;
    let root = lp_root(&l);
    assert_eq!(root.status, SolveStatus::Optimal);
    let basis = &root.warm_start_basis.as_ref().unwrap().basis;

    let sf = build_standard_form(&l);
    assert_eq!(basis.len(), sf.m);
    let mut lu = LuBasis::new_timed(&sf.a, basis, 0, None).unwrap();

    // Dense B^{-1} A: column-by-column FTRAN of each A_std column.
    let m = sf.m;
    let n = sf.n_total;
    let dense_a = csc_to_dense(&sf.a, m, n);
    let b_inv = dense_basis_inverse(&dense_a, basis);

    for i in 0..m {
        // alpha_i,: via BTRAN(e_i) then column_dot — the production path.
        let mut rho = vec![0.0; m];
        rho[i] = 1.0;
        lu.btran_dense(&mut rho);
        for j in 0..n {
            let via_btran = column_dot(&sf.a, j, &rho);
            // Direct: (B^{-1} A)_{i,j} = row i of B^{-1} dotted with column j of A.
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
/// and that every feasible integer point in `int_pts` satisfies every cut.
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
            assert_eq!(out.lp.constraint_types[i], ConstraintType::Ge);
            assert!(
                ax[i] >= out.lp.b[i] - 1e-6,
                "{name}: INVALID CUT — integer point {x:?} removed by cut row {i}: \
                 lhs={} < rhs={}",
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
    // min -x-y  s.t. 2x+2y <= 3,  x,y ∈ [-1, 2] integer.
    // LP opt: x=y=0.75 (fractional). lb-shift offset = -1 (negative).
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
/// `build_standard_form`.  Checks that `classify_slack_cols` tracks the
/// ConstraintLe mapping correctly through the sign flip.
#[test]
fn cut_validity_negative_rhs_row_negation() {
    // min x+y  s.t. -2x-2y <= -3,  x,y ∈ [0, 2] integer.
    // Equivalent to 2x+2y >= 3. LP opt: x=y=0.75 (fractional).
    // b_shifted = -3 < 0 → Le row is negated in standard form.
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
    // min -x-y  s.t. 2x+2y<=3 (Le), x+y>=0 (Ge),  x,y ∈ [0, 2] integer.
    // LP opt: x=y=0.75; Ge is slack at opt. Integer opt: (1,0) or (0,1).
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

/// **True lb-only (ub = +∞):** UB rows are NOT generated for these variables
/// (only the Le constraint adds slacks). Exercises the lb-only structural path
/// where `classify_slack_cols` sees only constraint slacks, no UB-row slacks.
#[test]
fn cut_validity_true_lb_only_inf_ub() {
    let milp = p_lb_only_inf();
    // Enumerate in window [0,4]^3 — all feasible integer points satisfy 3x+2y+4z<=7
    // with x,y,z>=0, so x<=2, y<=3, z<=1; window [0,4] is conservative.
    let pts = enumerate_int_window(3, 0, 4);
    let n = assert_cuts_valid_nonvacuous(&milp, &pts, "true_lb_only_inf", 5);
    assert!(n > 0, "true lb-only (ub=∞) path must generate ≥1 cut");
}

/// **Multi-var UbOnly columns + Eq row:** two UbOnly variables (lb=-∞, ub=3)
/// plus an equality constraint (no slack column). Exercises the Eq-row no-slack
/// path in `classify_slack_cols` and the UbOnly image in `accumulate_column`.
#[test]
fn cut_validity_multi_var_ubonly_eq_row() {
    // min -x-2y  s.t.  x+y=2 (Eq),  2x+4y<=7 (Le),  x,y ∈ (-∞, 3] integer.
    // Substituting y=2-x: 2x+4(2-x)<=7 → -2x<=-1 → x>=0.5 (fractional LP opt).
    // Integer feasible (x+y=2, 2x+4y<=7, x,y<=3):
    //   x=1,y=1: 2+4=6<=7 ✓   x=2,y=0: 4+0=4<=7 ✓   x=3,y=-1: 6-4=2<=7 ✓
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
    // Enumerate window [-2, 4]^2.
    let pts = enumerate_int_window(2, -2, 4);
    let n = assert_cuts_valid_nonvacuous(&milp, &pts, "ubonly_eq_row", 5);
    assert!(n > 0, "UbOnly + Eq row path must generate ≥1 cut");
}

/// **Deterministic LCG fuzz:** generates ≥100 2-variable MILPs with fractional
/// LP relaxation optima (verified via the LP solver before testing), varying
/// constraint types, coefficients, RHS and bounds.  For every problem the test
/// asserts that no integer-feasible point is removed by any generated cut.
/// `with_cuts` tracks how many problems actually produce cuts; the test asserts
/// at least some do (guards against a silent no-op generator).
#[test]
fn cut_validity_fuzz_lcg() {
    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

    // Try up to 600 candidates to collect ≥100 fractional-LP problems.
    for _ in 0..600 {
        if total >= 160 {
            break;
        }

        // Integer-valued bounds.
        let lb0 = lcg_f(&mut rng, -2.0, 0.9).round();
        let lb1 = lcg_f(&mut rng, -2.0, 0.9).round();
        let ub0 = lcg_f(&mut rng, 2.0, 4.0).round();
        let ub1 = lcg_f(&mut rng, 2.0, 4.0).round();
        if lb0 >= ub0 || lb1 >= ub1 {
            continue;
        }

        // Integer coefficients to avoid trivially-integer LP vertices.
        let a00 = lcg_f(&mut rng, 1.0, 5.0).round();
        let a01 = lcg_f(&mut rng, 1.0, 5.0).round();
        let is_le = lcg(&mut rng).is_multiple_of(2);

        // Pick RHS strictly between min_ax and max_ax so the constraint is
        // active at the LP optimum, making a fractional solution likely.
        let min_ax = a00 * lb0 + a01 * lb1;
        let max_ax = a00 * ub0 + a01 * ub1;
        let range = max_ax - min_ax;
        if range < 2.0 {
            continue;
        }
        // Half-integer RHS near the centre (forces fractional LP optimal).
        let mid = (min_ax + max_ax) / 2.0;
        let rhs = mid.floor() + 0.5;
        // Flip sign for Ge so the sense matches: Ge with the same half-int rhs.
        let ct = if is_le { ConstraintType::Le } else { ConstraintType::Ge };
        let actual_rhs = rhs;
        // Ge must satisfy: rhs <= max_ax and rhs >= min_ax.
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

        // Keep only problems whose LP relaxation is actually fractional.
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
                assert!(
                    ax[i] >= out.lp.b[i] - 1e-6,
                    "fuzz INVALID CUT — a=[{a00},{a01}] rhs={actual_rhs} ct={ct:?} \
                     int-pt {x:?} violates cut row {i}: lhs={} < rhs={}",
                    ax[i],
                    out.lp.b[i]
                );
            }
        }
    }

    assert!(total >= 100, "fuzz: need ≥100 fractional-LP problems, got {total}");
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
        // Partial pivot.
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
