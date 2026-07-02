//! Correctness tests for the conic (SOCP/QCQP) solver.

use super::*;
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::sparse::CscMatrix;

fn csc(rows: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut v = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val != 0.0 {
                r.push(i);
                c.push(j);
                v.push(val);
            }
        }
    }
    CscMatrix::from_triplets(&r, &c, &v, nrows, ncols).unwrap()
}

struct Lcg(u64);
impl Lcg {
    fn next_f(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((self.0 >> 11) as f64) / ((1u64 << 53) as f64);
        lo + (hi - lo) * u
    }
}

#[test]
fn tiny_socp_hand_optimum() {
    // min x0 s.t. ||x1|| <= x0, x1 = 1  => x0* = 1.
    let g = csc(&[vec![-1.0, 0.0], vec![0.0, -1.0]], 2, 2);
    let a = csc(&[vec![0.0, 1.0]], 1, 2);
    let prob = ConicProblem {
        c: vec![1.0, 0.0],
        a,
        b: vec![1.0],
        g,
        h: vec![0.0, 0.0],
        cone: ConeSpec { l: 0, soc: vec![2] },
    };
    let res = solve_socp(&prob, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!((res.objective - 1.0).abs() < 1e-6, "obj={}", res.objective);
    assert!((res.x[0] - 1.0).abs() < 1e-5);
}

#[test]
fn lp_as_socp_matches_simplex() {
    for seed in 0..6u64 {
        let mut rng = Lcg(seed.wrapping_mul(2654435761).wrapping_add(12345));
        let n = 3usize;
        let mineq = 2usize;
        let ub = 5.0;
        let c: Vec<f64> = (0..n).map(|_| rng.next_f(-2.0, 2.0)).collect();
        let mut arows = Vec::new();
        let mut b = Vec::new();
        for _ in 0..mineq {
            let row: Vec<f64> = (0..n).map(|_| rng.next_f(0.0, 1.0)).collect();
            arows.push(row);
            b.push(rng.next_f(3.0, 6.0));
        }
        // LP reference: min c^T x s.t. A x <= b, 0 <= x <= ub.
        let a_lp = csc(&arows, mineq, n);
        let lp = LpProblem::new_general(
            c.clone(),
            a_lp,
            b.clone(),
            vec![ConstraintType::Le; mineq],
            vec![(0.0, ub); n],
            None,
        )
        .unwrap();
        let lp_res = crate::lp::solve_lp_with(&lp, &crate::options::SolverOptions::default());
        assert_eq!(lp_res.status, SolveStatus::Optimal, "seed {seed}");

        // Conic: orthant rows [A x <= b ; x <= ub ; -x <= 0].
        let mut grows = Vec::new();
        let mut h = Vec::new();
        for i in 0..mineq {
            grows.push(arows[i].clone());
            h.push(b[i]);
        }
        for j in 0..n {
            let mut row = vec![0.0; n];
            row[j] = 1.0;
            grows.push(row);
            h.push(ub);
        }
        for j in 0..n {
            let mut row = vec![0.0; n];
            row[j] = -1.0;
            grows.push(row);
            h.push(0.0);
        }
        let m = grows.len();
        let g = csc(&grows, m, n);
        let prob = ConicProblem {
            c: c.clone(),
            a: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b: vec![],
            g,
            h,
            cone: ConeSpec { l: m, soc: vec![] },
        };
        let res = solve_socp(&prob, &ConicOptions::default());
        assert_eq!(
            res.status,
            SolveStatus::Optimal,
            "seed {seed} conic {res:?}"
        );
        assert!(
            (res.objective - lp_res.objective).abs() < 1e-5,
            "seed {seed}: conic {} vs lp {}",
            res.objective,
            lp_res.objective
        );
    }
}

#[test]
fn convex_qp_box_matches_closed_form() {
    // min (1/2) sum p_j x_j^2 + q_j x_j, 0 <= x <= ub, unconstrained opt interior.
    for seed in 0..5u64 {
        let mut rng = Lcg(seed.wrapping_mul(97).wrapping_add(7));
        let n = 3usize;
        let ub = 10.0;
        let mut pdiag = vec![0.0; n];
        let mut q = vec![0.0; n];
        let mut xstar = vec![0.0; n];
        for j in 0..n {
            let pj = rng.next_f(0.5, 3.0);
            let xj = rng.next_f(1.0, 4.0); // interior target
            pdiag[j] = pj;
            q[j] = -pj * xj; // so unconstrained min at x = xj
            xstar[j] = xj;
        }
        let mut prows = vec![vec![0.0; n]; n];
        for j in 0..n {
            prows[j][j] = pdiag[j];
        }
        // linear ineq: x <= ub, -x <= 0
        let mut grows = Vec::new();
        let mut h = Vec::new();
        for j in 0..n {
            let mut r = vec![0.0; n];
            r[j] = 1.0;
            grows.push(r);
            h.push(ub);
        }
        for j in 0..n {
            let mut r = vec![0.0; n];
            r[j] = -1.0;
            grows.push(r);
            h.push(0.0);
        }
        let qp = QcqpProblem {
            n,
            p0: Some(csc(&prows, n, n)),
            q0: q.clone(),
            quad: vec![],
            g_lin: csc(&grows, grows.len(), n),
            h_lin: h,
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
        };
        let res = solve_qcqp(&qp, &ConicOptions::default());
        assert_eq!(res.status, SolveStatus::Optimal, "seed {seed}: {res:?}");
        let mut obj_star = 0.0;
        for j in 0..n {
            obj_star += 0.5 * pdiag[j] * xstar[j] * xstar[j] + q[j] * xstar[j];
        }
        assert!(
            (res.objective - obj_star).abs() < 1e-4,
            "seed {seed}: qcqp {} vs closed {}",
            res.objective,
            obj_star
        );
        for j in 0..n {
            assert!((res.x[j] - xstar[j]).abs() < 1e-3, "seed {seed} x{j}");
        }
    }
}

#[test]
fn convex_qcqp_ball_constraint() {
    // min -x0 - x1  s.t.  x0^2 + x1^2 <= 1.  Optimum at (1/sqrt2, 1/sqrt2), obj = -sqrt2.
    let n = 2usize;
    // (1/2) x^T (2 I) x - 1 <= 0  => P = 2I, q = 0, r = -1.
    let p = csc(&[vec![2.0, 0.0], vec![0.0, 2.0]], 2, 2);
    let qp = QcqpProblem {
        n,
        p0: None,
        q0: vec![-1.0, -1.0],
        quad: vec![QuadConstraint {
            p,
            q: vec![0.0, 0.0],
            r: -1.0,
        }],
        g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        h_lin: vec![],
        a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b_eq: vec![],
    };
    let res = solve_qcqp(&qp, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    let want = -(2.0_f64).sqrt();
    assert!(
        (res.objective - want).abs() < 1e-4,
        "obj={} want={}",
        res.objective,
        want
    );
}

#[test]
fn unbounded_lp_detected() {
    // min -x0 s.t. x0 >= 0  (orthant: -x0 <= 0)  => unbounded below.
    let g = csc(&[vec![-1.0]], 1, 1);
    let prob = ConicProblem {
        c: vec![-1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
        b: vec![],
        g,
        h: vec![0.0],
        cone: ConeSpec { l: 1, soc: vec![] },
    };
    let res = solve_socp(&prob, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Unbounded, "{res:?}");
}

#[test]
fn infeasible_lp_detected() {
    // x0 <= -1 (row x0 + s = -1, s>=0) and x0 >= 0 (-x0 + s = 0) => infeasible.
    let g = csc(&[vec![1.0], vec![-1.0]], 2, 1);
    let prob = ConicProblem {
        c: vec![0.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
        b: vec![],
        g,
        h: vec![-1.0, 0.0],
        cone: ConeSpec { l: 2, soc: vec![] },
    };
    let res = solve_socp(&prob, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Infeasible, "{res:?}");
}

#[test]
fn misocp_ball_integer_optimum() {
    // max x0 + x1  (min -x0 -x1)  s.t. ||(x0,x1)|| <= sqrt(2.5), x in {0,1,2}^2.
    // Continuous opt ~ (1.118,1.118); integer opt (1,1), obj = -2.
    let n = 2usize;
    let r = 2.5_f64.sqrt();
    // SOC dim 3: s = (r, x0, x1) in Q3.
    let g = csc(&[vec![0.0, 0.0], vec![-1.0, 0.0], vec![0.0, -1.0]], 3, 2);
    let base = ConicProblem {
        c: vec![-1.0, -1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b: vec![],
        g,
        h: vec![r, 0.0, 0.0],
        cone: ConeSpec { l: 0, soc: vec![3] },
    };
    let prob = MisocpProblem {
        base,
        integers: vec![0, 1],
        int_lb: vec![0.0, 0.0],
        int_ub: vec![2.0, 2.0],
    };
    let res = solve_misocp(&prob, &ConicOptions::default(), &BbOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!(
        (res.objective - (-2.0)).abs() < 1e-4,
        "obj={}",
        res.objective
    );
    assert!((res.x[0] - 1.0).abs() < 1e-4 && (res.x[1] - 1.0).abs() < 1e-4);
}

#[test]
fn miqcp_integer_ball_matches_enumeration() {
    // min -x0 - 2 x1  s.t. x0^2 + x1^2 <= 5, x integer in [0,3]^2.
    // Enumerate: feasible ints with x0^2+x1^2<=5; maximise x0+2x1.
    let n = 2usize;
    let p = csc(&[vec![2.0, 0.0], vec![0.0, 2.0]], 2, 2); // (1/2)x^T(2I)x = x0^2+x1^2
    let qp = QcqpProblem {
        n,
        p0: None,
        q0: vec![-1.0, -2.0],
        quad: vec![QuadConstraint {
            p,
            q: vec![0.0, 0.0],
            r: -5.0,
        }],
        g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        h_lin: vec![],
        a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b_eq: vec![],
    };
    // brute force reference
    let mut best = f64::INFINITY;
    for x0 in 0..=3 {
        for x1 in 0..=3 {
            if (x0 * x0 + x1 * x1) as f64 <= 5.0 {
                let obj = -(x0 as f64) - 2.0 * (x1 as f64);
                if obj < best {
                    best = obj;
                }
            }
        }
    }
    let res = solve_miqcp(
        &qp,
        &[0, 1],
        &[0.0, 0.0],
        &[3.0, 3.0],
        &ConicOptions::default(),
        &BbOptions::default(),
    );
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!(
        (res.objective - best).abs() < 1e-3,
        "miqcp {} vs brute {}",
        res.objective,
        best
    );
}

/// Benchmark suite across SOCP / QCQP / MISOCP / MIQCP. Writes a CSV report to
/// `/tmp/conic_bench_results.csv`. Run with `--ignored`.
#[test]
#[ignore = "benchmark: run with --ignored; writes /tmp/conic_bench_results.csv"]
fn bench_conic_suite() {
    use std::fmt::Write as _;
    use std::time::Instant;

    let mut csv = String::from(
        "class,name,n,m,status,objective,expected,iters_or_nodes,pres,dres,gap,micros\n",
    );

    // ---- SOCP: min c^T x s.t. ||x||_2 <= 1  => obj = -||c||. ----
    for &n in &[2usize, 5, 10, 20, 40] {
        let mut rng = Lcg((n as u64) * 1009 + 1);
        let c: Vec<f64> = (0..n).map(|_| rng.next_f(-1.0, 1.0)).collect();
        let mut grows = Vec::new();
        let mut h = Vec::new();
        grows.push(vec![0.0; n]);
        h.push(1.0);
        for j in 0..n {
            let mut r = vec![0.0; n];
            r[j] = -1.0;
            grows.push(r);
            h.push(0.0);
        }
        let g = csc(&grows, n + 1, n);
        let prob = ConicProblem {
            c: c.clone(),
            a: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b: vec![],
            g,
            h,
            cone: ConeSpec {
                l: 0,
                soc: vec![n + 1],
            },
        };
        let t0 = Instant::now();
        let res = solve_socp(&prob, &ConicOptions::default());
        let us = t0.elapsed().as_micros();
        let expected = -(c.iter().map(|v| v * v).sum::<f64>().sqrt());
        assert_eq!(res.status, SolveStatus::Optimal, "socp n={n}");
        assert!((res.objective - expected).abs() < 1e-5, "socp n={n}");
        writeln!(
            csv,
            "SOCP,unit_ball_{n},{n},{},{:?},{:.6},{:.6},{},{:.2e},{:.2e},{:.2e},{}",
            prob.m(),
            res.status,
            res.objective,
            expected,
            res.iterations,
            res.residuals.0,
            res.residuals.1,
            res.residuals.2,
            us
        )
        .unwrap();
    }

    // ---- QCQP: diagonal box QP, closed-form optimum. ----
    for &n in &[3usize, 8, 16, 32] {
        let mut rng = Lcg((n as u64) * 7919 + 3);
        let mut prows = vec![vec![0.0; n]; n];
        let mut q = vec![0.0; n];
        let mut expected = 0.0;
        for j in 0..n {
            let pj = rng.next_f(0.5, 3.0);
            let xj = rng.next_f(1.0, 4.0);
            prows[j][j] = pj;
            q[j] = -pj * xj;
            expected += 0.5 * pj * xj * xj + q[j] * xj;
        }
        let mut grows = Vec::new();
        let mut h = Vec::new();
        for j in 0..n {
            let mut r = vec![0.0; n];
            r[j] = 1.0;
            grows.push(r);
            h.push(10.0);
            let mut r2 = vec![0.0; n];
            r2[j] = -1.0;
            grows.push(r2);
            h.push(0.0);
        }
        let qp = QcqpProblem {
            n,
            p0: Some(csc(&prows, n, n)),
            q0: q.clone(),
            quad: vec![],
            g_lin: csc(&grows, grows.len(), n),
            h_lin: h,
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
        };
        let t0 = Instant::now();
        let res = solve_qcqp(&qp, &ConicOptions::default());
        let us = t0.elapsed().as_micros();
        assert_eq!(res.status, SolveStatus::Optimal, "qcqp n={n}");
        assert!((res.objective - expected).abs() < 1e-3, "qcqp n={n}");
        writeln!(
            csv,
            "QCQP,box_diag_{n},{n},-,{:?},{:.6},{:.6},{},-,-,-,{}",
            res.status, res.objective, expected, res.iterations, us
        )
        .unwrap();
    }

    // ---- MISOCP / MIQCP: integer point in a ball, brute-force reference. ----
    for &(r2, k) in &[(5.0f64, 3i64), (8.0, 3), (13.0, 4)] {
        let n = 2usize;
        let p = csc(&[vec![2.0, 0.0], vec![0.0, 2.0]], 2, 2);
        let qp = QcqpProblem {
            n,
            p0: None,
            q0: vec![-1.0, -2.0],
            quad: vec![QuadConstraint {
                p,
                q: vec![0.0, 0.0],
                r: -r2,
            }],
            g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            h_lin: vec![],
            a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
            b_eq: vec![],
        };
        let mut best = f64::INFINITY;
        for x0 in 0..=k {
            for x1 in 0..=k {
                if (x0 * x0 + x1 * x1) as f64 <= r2 {
                    let obj = -(x0 as f64) - 2.0 * (x1 as f64);
                    if obj < best {
                        best = obj;
                    }
                }
            }
        }
        let t0 = Instant::now();
        let res = solve_miqcp(
            &qp,
            &[0, 1],
            &[0.0, 0.0],
            &[k as f64, k as f64],
            &ConicOptions::default(),
            &BbOptions::default(),
        );
        let us = t0.elapsed().as_micros();
        assert_eq!(res.status, SolveStatus::Optimal, "miqcp r2={r2}");
        assert!((res.objective - best).abs() < 1e-3, "miqcp r2={r2}");
        writeln!(
            csv,
            "MIQCP,int_ball_r2_{r2},{n},-,{:?},{:.6},{:.6},{},-,-,-,{}",
            res.status, res.objective, best, res.nodes, us
        )
        .unwrap();
    }

    std::fs::write("/tmp/conic_bench_results.csv", &csv).unwrap();
}

#[test]
fn nonconvex_bilinear_box_global_min() {
    use super::nonconvex::*;
    // min x0*x1  over [-1,1]^2.  (1/2)x^T P x with P=[[0,1],[1,0]] = x0*x1.
    // Global min = -1 at (-1,1) or (1,-1).
    let n = 2usize;
    let p = csc(&[vec![0.0, 1.0], vec![1.0, 0.0]], 2, 2);
    let qp = NonconvexQcqp {
        n,
        p0: Some(p),
        q0: vec![0.0, 0.0],
        quad: vec![],
        g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        h_lin: vec![],
        a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b_eq: vec![],
        lb: vec![-1.0, -1.0],
        ub: vec![1.0, 1.0],
    };
    let res = solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!(
        (res.objective - (-1.0)).abs() < 1e-3,
        "obj={}",
        res.objective
    );
}

#[test]
fn nonconvex_concave_max_box() {
    use super::nonconvex::*;
    // min -x0^2 over [0,1]  => global min -1 at x0=1.  P0 = [[-2]] => (1/2)(-2)x^2 = -x^2.
    let n = 1usize;
    let p = csc(&[vec![-2.0]], 1, 1);
    let qp = NonconvexQcqp {
        n,
        p0: Some(p),
        q0: vec![0.0],
        quad: vec![],
        g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        h_lin: vec![],
        a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b_eq: vec![],
        lb: vec![0.0],
        ub: vec![1.0],
    };
    let res = solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!(
        (res.objective - (-1.0)).abs() < 1e-3,
        "obj={}",
        res.objective
    );
}

#[test]
fn nonconvex_constraint_hyperbola() {
    use super::nonconvex::*;
    // min x0 + x1 s.t. x0*x1 >= 1, x in [0.1,3]^2.  Global min = 2 at (1,1).
    // x0*x1 >= 1  <=>  -x0*x1 + 1 <= 0.  P = [[0,-1],[-1,0]] => (1/2)x^TPx = -x0*x1.
    let n = 2usize;
    let p = csc(&[vec![0.0, -1.0], vec![-1.0, 0.0]], 2, 2);
    let qp = NonconvexQcqp {
        n,
        p0: None,
        q0: vec![1.0, 1.0],
        quad: vec![GQuadConstraint {
            p,
            q: vec![0.0, 0.0],
            r: 1.0,
        }],
        g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        h_lin: vec![],
        a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b_eq: vec![],
        lb: vec![0.1, 0.1],
        ub: vec![3.0, 3.0],
    };
    let res = solve_global_qcqp(&qp, &ConicOptions::default(), &GlobalOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!((res.objective - 2.0).abs() < 5e-3, "obj={}", res.objective);
}

#[test]
fn qp_problem_bridge_solves_convex_qcqp_constraint() {
    use crate::qp::{QcqpMatrix, QpProblem};
    // min -x0-x1  s.t. x0^2+x1^2 <= 1, x>=0.  Optimum (1/sqrt2,1/sqrt2).
    let n = 2usize;
    let q_obj = CscMatrix::new(n, n);
    let c = vec![-1.0, -1.0];
    // one linear row is zero; the quadratic matrix carries the true constraint.
    let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let mut qc = QcqpMatrix::new(n);
    // constraint uses 1/2 x^T Qc x <= 1, so Qc = 2I.
    qc.triplets.push((0, 0, 2.0));
    qc.triplets.push((1, 1, 2.0));
    let mut qp = QpProblem::new(q_obj, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();
    qp.set_quadratic_constraints(vec![qc]).unwrap();
    let res = solve_qp_problem_as_qcqp(&qp, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!(
        (res.objective + 2.0_f64.sqrt()).abs() < 1e-4,
        "obj={}",
        res.objective
    );
}

#[test]
fn qp_problem_bridge_rejects_quadratic_ge() {
    use crate::qp::{QcqpMatrix, QpProblem};
    let n = 1usize;
    let q_obj = CscMatrix::new(n, n);
    let a = CscMatrix::from_triplets(&[], &[], &[], 1, n).unwrap();
    let mut qc = QcqpMatrix::new(n);
    qc.triplets.push((0, 0, 2.0));
    let mut qp = QpProblem::new(
        q_obj,
        vec![0.0],
        a,
        vec![1.0],
        vec![(0.0, 2.0)],
        vec![ConstraintType::Ge],
    )
    .unwrap();
    qp.set_quadratic_constraints(vec![qc]).unwrap();
    let res = solve_qp_problem_as_qcqp(&qp, &ConicOptions::default());
    assert!(matches!(res.status, SolveStatus::NotSupported(_)));
}

#[test]
fn nonconvex_miqcp_integer_bilinear() {
    use super::nonconvex::*;
    // min x0*x1  s.t. x0,x1 integer in [-2,2].  Global min = -4 at (-2,2)/(2,-2).
    let n = 2usize;
    let p = csc(&[vec![0.0, 1.0], vec![1.0, 0.0]], 2, 2); // (1/2)x^TPx = x0*x1
    let qp = NonconvexQcqp {
        n,
        p0: Some(p),
        q0: vec![0.0, 0.0],
        quad: vec![],
        g_lin: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        h_lin: vec![],
        a_eq: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b_eq: vec![],
        lb: vec![-2.0, -2.0],
        ub: vec![2.0, 2.0],
    };
    let res = solve_global_miqcp(
        &qp,
        &[0, 1],
        &ConicOptions::default(),
        &GlobalOptions::default(),
    );
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!(
        (res.objective - (-4.0)).abs() < 1e-3,
        "obj={}",
        res.objective
    );
    for k in 0..2 {
        assert!(
            (res.x[k] - res.x[k].round()).abs() < 1e-6,
            "x{k} not integral"
        );
    }
}
