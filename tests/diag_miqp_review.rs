//! Independent convex-MIQP correctness checks (#17 review).
//!
//! Complements the diagonal-Q brute-force tests by covering **off-diagonal PSD Q**
//! (full quadratic form), maximize (concave→convex / convex→reject), binary, and a
//! tight integrality-gap stress that targets the ε-suboptimal-relaxation lower
//! bound: if the QP relaxation objective were used as an *invalid* (over-estimating)
//! bound it would over-prune and miss the true integer optimum here.
//!
//! Q is generated as L·Lᵀ (+ridge) so it is provably PSD and well-conditioned; the
//! brute force enumerates every integer point with the full quadratic form, so the
//! true optimum is not hardcoded.

use otspot::options::{MipConfig, SolverOptions};
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::{solve_miqp_with_stats, CscMatrix, MiqpProblem, Model, QpProblem};

fn opts() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(30.0);
    o
}

/// Deterministic LCG (same as the MILP fuzz) for reproducibility.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct MiqpCase {
    /// Dense symmetric n×n Q (full quadratic form, objective 1/2 xᵀQx + cᵀx).
    q: Vec<Vec<f64>>,
    c: Vec<f64>,
    cons: Vec<(Vec<f64>, ConstraintType, f64)>,
    bounds: Vec<(i64, i64)>,
}

fn build(case: &MiqpCase) -> MiqpProblem {
    let n = case.c.len();
    // full-symmetric CSC storage of Q
    let (mut qr, mut qc, mut qv) = (vec![], vec![], vec![]);
    for i in 0..n {
        for j in 0..n {
            if case.q[i][j] != 0.0 {
                qr.push(i);
                qc.push(j);
                qv.push(case.q[i][j]);
            }
        }
    }
    let q = CscMatrix::from_triplets(&qr, &qc, &qv, n, n).unwrap();
    let m = case.cons.len();
    let (mut rows, mut cols, mut vals, mut b, mut ct) = (vec![], vec![], vec![], vec![], vec![]);
    for (i, (coeffs, c, rhs)) in case.cons.iter().enumerate() {
        for (j, &v) in coeffs.iter().enumerate() {
            if v != 0.0 {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        b.push(*rhs);
        ct.push(*c);
    }
    let a =
        if m == 0 { CscMatrix::new(0, n) } else { CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap() };
    let bounds: Vec<(f64, f64)> = case.bounds.iter().map(|&(l, u)| (l as f64, u as f64)).collect();
    let qp = QpProblem::new(q, case.c.clone(), a, b, bounds, ct).unwrap();
    MiqpProblem::new(qp, (0..n).collect()).unwrap()
}

fn quad_obj(case: &MiqpCase, x: &[f64]) -> f64 {
    let n = x.len();
    let mut q = 0.0;
    for i in 0..n {
        for j in 0..n {
            q += case.q[i][j] * x[i] * x[j];
        }
    }
    0.5 * q + (0..n).map(|i| case.c[i] * x[i]).sum::<f64>()
}

/// True integer optimum via full enumeration. None = infeasible.
fn brute_force(case: &MiqpCase) -> Option<(f64, Vec<f64>)> {
    let n = case.c.len();
    let mut idx: Vec<i64> = case.bounds.iter().map(|&(l, _)| l).collect();
    let mut best: Option<(f64, Vec<f64>)> = None;
    loop {
        let x: Vec<f64> = idx.iter().map(|&v| v as f64).collect();
        let feasible = case.cons.iter().all(|(coeffs, ct, rhs)| {
            let lhs: f64 = coeffs.iter().zip(&x).map(|(a, b)| a * b).sum();
            match ct {
                ConstraintType::Le => lhs <= rhs + 1e-9,
                ConstraintType::Ge => lhs >= rhs - 1e-9,
                ConstraintType::Eq => (lhs - rhs).abs() <= 1e-9,
                _ => unreachable!(),
            }
        });
        if feasible {
            let o = quad_obj(case, &x);
            if best.as_ref().is_none_or(|(bo, _)| o < *bo) {
                best = Some((o, x.clone()));
            }
        }
        let mut k = 0;
        loop {
            if k == n {
                return best;
            }
            idx[k] += 1;
            if idx[k] <= case.bounds[k].1 {
                break;
            }
            idx[k] = case.bounds[k].0;
            k += 1;
        }
    }
}

#[test]
fn offdiag_psd_bruteforce_cases() {
    use ConstraintType::*;
    // Q = [[2,1],[1,2]] (eig 1,3 PSD). min 1/2 xᵀQx + cᵀx.
    // NOTE: the box-only / no-linear-constraint case (continuous opt at (2,2), obj -12)
    // is split out into `miqp_boxonly_offdiag_relaxation_stall_repro` below because it
    // currently FAILS (P1 silent-wrong: the QP IPM stalls on box-only off-diagonal QPs).
    let cases = vec![
        // coupling + Ge constraint forcing off-corner.
        MiqpCase {
            q: vec![vec![4.0, 2.0], vec![2.0, 4.0]],
            c: vec![0.0, 0.0],
            cons: vec![(vec![1.0, 1.0], Ge, 3.0)],
            bounds: vec![(0, 5), (0, 5)],
        },
        // equality-pinned (forces fixed-point evaluation in the leaves).
        MiqpCase {
            q: vec![vec![2.0, 1.0], vec![1.0, 2.0]],
            c: vec![-1.0, -4.0],
            cons: vec![(vec![1.0, 1.0], Eq, 4.0)],
            bounds: vec![(0, 4), (0, 4)],
        },
        // 3-var coupled Q = diag(2)+0.5 off-diagonals (PSD, diag-dominant).
        MiqpCase {
            q: vec![
                vec![3.0, 1.0, 0.5],
                vec![1.0, 3.0, 1.0],
                vec![0.5, 1.0, 3.0],
            ],
            c: vec![-2.0, -3.0, -1.0],
            cons: vec![(vec![1.0, 1.0, 1.0], Le, 5.0)],
            bounds: vec![(0, 3), (0, 3), (0, 3)],
        },
    ];
    for (i, case) in cases.iter().enumerate() {
        let truth = brute_force(case);
        let (r, stats) = solve_miqp_with_stats(&build(case), &opts(), &MipConfig::default());
        match truth {
            Some((opt, xstar)) => {
                assert_eq!(r.status, SolveStatus::Optimal, "case {i} status {:?}", r.status);
                assert!(
                    (r.objective - opt).abs() < 1e-3,
                    "case {i}: solver {} != truth {} (x*={:?})",
                    r.objective,
                    opt,
                    xstar
                );
                // recompute objective from the reported (rounded) integer solution
                let recomputed = quad_obj(case, &r.solution.iter().map(|v| v.round()).collect::<Vec<_>>());
                assert!(
                    (recomputed - opt).abs() < 1e-3,
                    "case {i}: recomputed {} != truth {}",
                    recomputed,
                    opt
                );
                println!("[offdiag {i}] opt={opt} nodes={} pruned={}", stats.nodes_processed, stats.pruned);
            }
            None => assert_eq!(r.status, SolveStatus::Infeasible, "case {i}"),
        }
    }
}

/// REPRODUCER for a P1 silent-wrong MIQP answer.
///
/// min x²+xy+y²-6x-6y over integers [0,4]² (Q=[[2,1],[1,2]], c=[-6,-6], NO linear
/// constraints). Continuous optimum is the integer point (2,2), obj -12 → the MIQP
/// optimum is -12. The solver instead returns -11 at (1,3)/(3,2) **with status
/// Optimal** (false "proven optimal").
///
/// Root cause: the QP IPM stalls on this box-only off-diagonal convex QP and returns
/// `SuboptimalSolution` at ~(0.09, 2.82), obj -9.23 (true relaxation min -12). The
/// MIQP driver trusts that suboptimal *primal* objective as a *lower* bound
/// (`node_lb = parent.max(res.objective)`), but a suboptimal primal value is an
/// UPPER bound on the relaxation optimum, so the node holding (2,2) is over-pruned.
/// Adding ANY linear constraint, or a diagonal Q, makes the QP converge and the bug
/// disappears — which is why the diagonal-Q fuzz never caught it.
///
/// Un-ignored (#17 fix): the driver now trusts only Optimal relaxation objectives as
/// lower bounds and bisects integer boxes when the QP relaxation stalls, so the search
/// reaches the true integer optimum via exact fixed-point leaves instead of over-pruning.
#[test]
fn miqp_boxonly_offdiag_relaxation_stall_repro() {
    let case = MiqpCase {
        q: vec![vec![2.0, 1.0], vec![1.0, 2.0]],
        c: vec![-6.0, -6.0],
        cons: vec![],
        bounds: vec![(0, 4), (0, 4)],
    };
    let (opt, xstar) = brute_force(&case).unwrap();
    assert!((opt - (-12.0)).abs() < 1e-9, "brute truth {opt} (expected -12 at {xstar:?})");
    let (r, _) = solve_miqp_with_stats(&build(&case), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        (r.objective - (-12.0)).abs() < 1e-3,
        "MIQP returned {} (x={:?}); true optimum -12 at (2,2)",
        r.objective,
        r.solution
    );
}

/// Box-only off-diagonal MIQP fuzz: the class that triggered the P1. Every case has
/// NO linear constraints (so the QP IPM is most likely to stall) and an optimum often
/// far from the origin. Asserts the solver matches brute force AND returns a genuine
/// `Optimal` for these fully-resolvable pure-integer problems — guarding BOTH against
/// re-introducing over-pruning AND against over-conservative SuboptimalSolution.
#[test]
fn fuzz_boxonly_offdiag_miqp_optimum_and_status() {
    let mut rng = Lcg(0x0BAD_C0DE_F00D_1234);
    let mut feasible = 0;
    for trial in 0..120 {
        let n = rng.range(2, 3) as usize;
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                l[i][j] = rng.unit() * 2.0 - 1.0;
            }
        }
        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                q[i][j] = (0..n).map(|k| l[i][k] * l[j][k]).sum();
            }
            q[i][i] += 0.4;
        }
        // larger linear term → optimum pushed away from the IPM's interior start.
        let c: Vec<f64> = (0..n).map(|_| rng.range(-8, 8) as f64).collect();
        let bounds: Vec<(i64, i64)> = (0..n).map(|_| (0, rng.range(3, 5))).collect();
        let case = MiqpCase { q, c, cons: vec![], bounds }; // NO constraints (P1 trigger)
        let truth = brute_force(&case);
        let (r, _s) = solve_miqp_with_stats(&build(&case), &opts(), &MipConfig::default());
        match truth {
            Some((opt, xstar)) => {
                feasible += 1;
                assert!(
                    (r.objective - opt).abs() < 2e-3,
                    "trial {trial}: solver {} != truth {} (x*={:?})",
                    r.objective,
                    opt,
                    xstar
                );
                // box-only pure-integer is always fully resolvable → genuine Optimal.
                assert_eq!(
                    r.status,
                    SolveStatus::Optimal,
                    "trial {trial}: fully-resolvable box-only MIQP must be Optimal, got {:?}",
                    r.status
                );
            }
            None => assert_eq!(r.status, SolveStatus::Infeasible, "trial {trial}"),
        }
    }
    println!("box-only offdiag MIQP fuzz: {feasible} feasible — all true optima + genuine Optimal");
    assert!(feasible > 30, "expected many feasible, got {feasible}");
}

#[test]
fn fuzz_offdiag_psd_miqp() {
    use ConstraintType::*;
    let mut rng = Lcg(0xC0FFEE_1234_5678);
    let mut feasible = 0;
    let cfg = MipConfig::default();
    for trial in 0..150 {
        let n = rng.range(2, 3) as usize;
        // Q = L Lᵀ + ridge·I → symmetric PSD, well-conditioned.
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                l[i][j] = rng.unit() * 2.0 - 1.0;
            }
        }
        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += l[i][k] * l[j][k];
                }
                q[i][j] = s;
            }
            q[i][i] += 0.5; // ridge: strictly PD, well-conditioned
        }
        let c: Vec<f64> = (0..n).map(|_| rng.range(-4, 4) as f64).collect();
        let bounds: Vec<(i64, i64)> =
            (0..n).map(|_| { let lo = rng.range(-1, 1); (lo, lo + rng.range(1, 3)) }).collect();
        let m = rng.range(0, 2) as usize;
        let cons: Vec<(Vec<f64>, ConstraintType, f64)> = (0..m)
            .map(|_| {
                let coeffs: Vec<f64> = (0..n).map(|_| rng.range(-2, 2) as f64).collect();
                let ct = match rng.range(0, 2) { 0 => Le, 1 => Ge, _ => Eq };
                (coeffs, ct, rng.range(-3, 4) as f64)
            })
            .collect();
        let case = MiqpCase { q, c, cons, bounds };
        let truth = brute_force(&case);
        let (r, _s) = solve_miqp_with_stats(&build(&case), &opts(), &cfg);
        match truth {
            Some((opt, xstar)) => {
                feasible += 1;
                assert_eq!(
                    r.status,
                    SolveStatus::Optimal,
                    "trial {trial}: feasible (opt={opt}) but solver {:?}",
                    r.status
                );
                assert!(
                    (r.objective - opt).abs() < 2e-3,
                    "trial {trial}: solver {} != truth {} (x*={:?})",
                    r.objective,
                    opt,
                    xstar
                );
            }
            None => assert_eq!(
                r.status,
                SolveStatus::Infeasible,
                "trial {trial}: brute-force infeasible but solver {:?}",
                r.status
            ),
        }
    }
    println!("offdiag MIQP fuzz: {feasible} feasible — all matched brute force");
    assert!(feasible > 30, "expected many feasible cases, got {feasible}");
}

#[test]
fn tight_integrality_gap_not_overpruned() {
    // min 1/2·2·(x-2.5)^2 + ... constructed so two integer points (x=2 and x=3)
    // have *equal* objective and the relaxation lower bound sits just below them.
    // If the ε-suboptimal relaxation objective were used as an over-estimating bound,
    // the node holding the true optimum could be fathomed → wrong answer.
    // min x^2 - 5x  (Q=2, c=-5): continuous min at x=2.5, integer min at x=2 or 3 (=-6).
    let case = MiqpCase {
        q: vec![vec![2.0]],
        c: vec![-5.0],
        cons: vec![],
        bounds: vec![(0, 5)],
    };
    let (truth, _) = brute_force(&case).unwrap();
    assert!((truth - (-6.0)).abs() < 1e-9, "brute truth {truth}");
    let (r, _) = solve_miqp_with_stats(&build(&case), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - (-6.0)).abs() < 1e-3, "obj={} (expected -6)", r.objective);
    let xr = r.solution[0].round();
    assert!(xr == 2.0 || xr == 3.0, "x={}", r.solution[0]);
}

// --- Model API: maximize concave (convex problem) vs maximize convex (rejected) ---

#[test]
fn maximize_concave_miqp_matches_truth() {
    // maximize -(x-3)^2 - (y-2)^2 + ... i.e. concave (NSD Q) → convex minimization.
    // maximize -x^2 + 6x - y^2 + 4y, x,y integer in [0,5], x+y<=4.
    // Unconstrained concave max at (3,2); x+y<=4 binds → enumerate: best integer (3,1):
    //   -9+18 -1+4 = 12 ; (2,2): -4+12 -4+8 = 12 ; (3,1)=12,(2,2)=12,(4,0)=-16+24+0=8.
    // true max = 12.
    let mut m = Model::new("concave");
    let x = m.add_int_var("x", 0.0, 5.0);
    let y = m.add_int_var("y", 0.0, 5.0);
    m.add_constraint((x + y).leq(4.0));
    m.set_diagonal_q(&[-2.0, -2.0]); // objective 1/2 xᵀQx = -x^2 - y^2
    m.maximize(6.0 * x + 4.0 * y);
    let r = m.solve().unwrap();
    assert!((r.objective() - 12.0).abs() < 1e-3, "obj={} (expected 12)", r.objective());
}

#[test]
fn maximize_convex_miqp_rejected_not_silent() {
    // maximize a convex (PSD Q) quadratic is non-convex → must error, never silent.
    let mut m = Model::new("max_convex");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.set_diagonal_q(&[2.0]); // PSD; maximize → -Q NSD → non-convex
    m.maximize(x);
    let err = m.solve().unwrap_err();
    assert!(format!("{err:?}").contains("Non-convex"), "expected Non-convex error, got {err:?}");
}

#[test]
fn binary_miqp_bruteforce() {
    use ConstraintType::*;
    // binary {0,1} vars, coupled PSD Q.
    let case = MiqpCase {
        q: vec![vec![2.0, 1.0], vec![1.0, 2.0]],
        c: vec![-3.0, -2.0],
        cons: vec![(vec![1.0, 1.0], Le, 1.0)], // at most one selected
        bounds: vec![(0, 1), (0, 1)],
    };
    let (opt, xstar) = brute_force(&case).unwrap();
    let (r, _) = solve_miqp_with_stats(&build(&case), &opts(), &MipConfig::default());
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.objective - opt).abs() < 1e-3, "solver {} != truth {} x*={:?}", r.objective, opt, xstar);
}

/// Point 3 (#17 fix): mixed integer + CONTINUOUS where the continuous relaxation
/// stalls must NOT be reported as a false `Optimal`. Here x is integer [0,2] and
/// (y,z) are continuous with an off-diagonal coupling Q-block whose box-only sub-QP
/// stalls (same pathology as the P1). After x is fixed the continuous region cannot
/// be bisected, so that region is left unresolved (`proof_uncertain`) → the solver
/// must return the incumbent as `SuboptimalSolution`, never a disguised `Optimal`.
/// The returned point must still be feasible with a consistent objective.
#[test]
fn mixed_continuous_stall_no_false_optimal() {
    // Q full-symmetric 3×3: tiny x penalty + [[2,1],[1,2]] coupling on (y,z).
    let q = CscMatrix::from_triplets(
        &[0, 1, 1, 2, 2],
        &[0, 1, 2, 1, 2],
        &[0.001, 2.0, 1.0, 1.0, 2.0],
        3,
        3,
    )
    .unwrap();
    let a = CscMatrix::new(0, 3);
    let qp = QpProblem::new(
        q,
        vec![0.0, -6.0, -6.0],
        a,
        vec![],
        vec![(0.0, 2.0), (0.0, 4.0), (0.0, 4.0)],
        vec![],
    )
    .unwrap();
    let miqp = MiqpProblem::new(qp, vec![0]).unwrap(); // only x is integer
    let (r, _) = solve_miqp_with_stats(&miqp, &opts(), &MipConfig::default());
    // True optimum is -12 (x=0, (y,z)=(2,2)). The continuous (y,z) sub-QP stalls, so
    // the solver may not *prove* optimality — but it must NOT claim a false Optimal.
    assert_ne!(
        r.status,
        SolveStatus::Infeasible,
        "feasible problem must not be Infeasible"
    );
    if r.status == SolveStatus::Optimal {
        // If it does claim Optimal, the value must actually be the true optimum.
        assert!(
            (r.objective - (-12.0)).abs() < 1e-2,
            "false Optimal at {} (true -12)",
            r.objective
        );
    } else {
        // Honest non-Optimal (SuboptimalSolution / MaxIterations / Timeout): the
        // returned incumbent, if any, must be a feasible near-optimal point (≥ true opt).
        assert!(
            matches!(
                r.status,
                SolveStatus::SuboptimalSolution
                    | SolveStatus::MaxIterations
                    | SolveStatus::Timeout
            ),
            "unexpected status {:?}",
            r.status
        );
        if !r.solution.is_empty() {
            assert!(r.objective >= -12.0 - 1e-2, "incumbent {} below true optimum", r.objective);
            assert!((r.solution[0].round() - r.solution[0]).abs() < 1e-6, "x must be integral");
        }
    }
}
