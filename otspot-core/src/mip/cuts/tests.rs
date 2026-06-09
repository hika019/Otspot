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
    super::solve_cut_lp(p, &SolverOptions::default())
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

/// 3-var lb-only (ub=inf) integer problem: constraint slacks are continuous in
/// GMI, structural cols are pure lb-shifts (no UB rows).
/// min -x-y-z s.t. 3x+2y+4z<=7, x,y,z>=0 int (bounded above for enumeration).
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
