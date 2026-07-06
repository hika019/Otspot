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

#[test]
fn unbounded_certificate_is_improving_ray() {
    // min -x0 s.t. x0 >= 0  => unbounded; ray d has c·d < 0.
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
    assert_eq!(res.status, SolveStatus::Unbounded);
    let d = res.primal_ray.expect("unbounded must carry a ray");
    let cd: f64 = prob.c.iter().zip(&d).map(|(a, b)| a * b).sum();
    assert!(cd < -1e-6, "ray must be improving: c·d={cd}");
}

#[test]
fn infeasible_certificate_is_farkas() {
    // x0 <= -1 and x0 >= 0  => infeasible.  Certificate (y,z): b·y + h·z < 0,
    // A^T y + G^T z ≈ 0, z >= 0.
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
    assert_eq!(res.status, SolveStatus::Infeasible);
    let (_y, z) = res
        .infeas_cert
        .expect("infeasible must carry a Farkas certificate");
    // z in orthant dual (>= 0).
    for &zi in &z {
        assert!(zi >= -1e-6, "z must lie in K*: {zi}");
    }
    // h·z < 0 (b empty here) certifies infeasibility direction.
    let hz: f64 = prob.h.iter().zip(&z).map(|(a, b)| a * b).sum();
    assert!(hz < 1e-6, "Farkas value h·z should be <= 0-ish: {hz}");
}

// ---------------------------------------------------------------------------
// MISOCP branch-and-bound status classification sentinels.
//
// These pin the state table of `solve_misocp`: an inconclusive search (node
// numerical failures, node limit, deadline) must never be promoted to a
// proven `Infeasible` / `Optimal`.
// ---------------------------------------------------------------------------

/// `min x` s.t. `x >= 1/2`, integer `x ∈ [0, 2]`. The root relaxation is
/// fractional (x = 1/2); the down child (x fixed to 0) is infeasible and the
/// up child yields the integer optimum x = 1.
fn half_int_lp() -> MisocpProblem {
    let g = csc(&[vec![-1.0]], 1, 1);
    let base = ConicProblem {
        c: vec![1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
        b: vec![],
        g,
        h: vec![-0.5],
        cone: ConeSpec { l: 1, soc: vec![] },
    };
    MisocpProblem {
        base,
        integers: vec![0],
        int_lb: vec![0.0],
        int_ub: vec![2.0],
    }
}

/// `max x0 + x1` s.t. `||(x0, x1)|| <= sqrt(2.5)`, integers in `[0, 2]^2`
/// (same instance as `misocp_ball_integer_optimum`; integer optimum (1, 1)).
fn ball_int_misocp() -> MisocpProblem {
    let r = 2.5_f64.sqrt();
    let g = csc(&[vec![0.0, 0.0], vec![-1.0, 0.0], vec![0.0, -1.0]], 3, 2);
    let base = ConicProblem {
        c: vec![-1.0, -1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
        b: vec![],
        g,
        h: vec![r, 0.0, 0.0],
        cone: ConeSpec { l: 0, soc: vec![3] },
    };
    MisocpProblem {
        base,
        integers: vec![0, 1],
        int_lb: vec![0.0, 0.0],
        int_ub: vec![2.0, 2.0],
    }
}

/// `x0 <= -1` and `x0 >= 0` (proven infeasible) with one integer variable.
fn infeasible_int_lp() -> MisocpProblem {
    let g = csc(&[vec![1.0, 0.0], vec![-1.0, 0.0]], 2, 2);
    let base = ConicProblem {
        c: vec![0.0, 0.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
        b: vec![],
        g,
        h: vec![-1.0, 0.0],
        cone: ConeSpec { l: 2, soc: vec![] },
    };
    MisocpProblem {
        base,
        integers: vec![1],
        int_lb: vec![0.0],
        int_ub: vec![5.0],
    }
}

#[test]
fn misocp_unresolved_nodes_do_not_prove_infeasibility() {
    // max_iter = 0 makes every node relaxation return MaxIterations without a
    // certificate: nothing was proven, so the empty search must report
    // NumericalError, never a false Infeasible (the pre-fix behaviour).
    let opts = ConicOptions {
        max_iter: 0,
        ..ConicOptions::default()
    };
    let res = solve_misocp(&half_int_lp(), &opts, &BbOptions::default());
    assert_eq!(res.status, SolveStatus::NumericalError, "{res:?}");
    assert!(res.x.is_empty());
}

#[test]
fn misocp_certified_infeasible_returns_infeasible() {
    // Every leaf is pruned by a Farkas certificate: Infeasible is proven.
    let res = solve_misocp(
        &infeasible_int_lp(),
        &ConicOptions::default(),
        &BbOptions::default(),
    );
    assert_eq!(res.status, SolveStatus::Infeasible, "{res:?}");
    assert!(res.x.is_empty());
}

#[test]
fn misocp_exhaustive_search_without_failures_is_optimal() {
    // Incumbent + fully exhausted search + zero node failures => Optimal.
    let res = solve_misocp(
        &half_int_lp(),
        &ConicOptions::default(),
        &BbOptions::default(),
    );
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!((res.objective - 1.0).abs() < 1e-6, "obj={}", res.objective);
    assert!((res.x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn misocp_node_limit_with_incumbent_is_suboptimal() {
    // Node order: root (fractional), then the up child (integer incumbent
    // x = 1); the down child is still pending when max_nodes = 2 hits, so the
    // incumbent is not proven optimal.
    let bb = BbOptions {
        max_nodes: 2,
        ..BbOptions::default()
    };
    let res = solve_misocp(&half_int_lp(), &ConicOptions::default(), &bb);
    assert_eq!(res.status, SolveStatus::SuboptimalSolution, "{res:?}");
    assert!((res.x[0] - 1.0).abs() < 1e-6);
}

#[test]
fn misocp_node_limit_without_incumbent_is_max_iterations() {
    // max_nodes = 0 explores nothing; even on an infeasible instance the
    // search proved nothing, so it must report MaxIterations, never a false
    // Infeasible (the pre-fix behaviour).
    let bb = BbOptions {
        max_nodes: 0,
        ..BbOptions::default()
    };
    let res = solve_misocp(&infeasible_int_lp(), &ConicOptions::default(), &bb);
    assert_eq!(res.status, SolveStatus::MaxIterations, "{res:?}");
    assert_eq!(res.nodes, 0);
    assert!(res.x.is_empty());
}

/// Node iteration budget for `misocp_incumbent_with_failed_node_is_suboptimal`,
/// calibrated on `ball_int_misocp`: every feasible node relaxation converges
/// within 9 IPM iterations while the infeasible up child (x0 fixed at 2)
/// needs 16 to form its Farkas certificate. Any value in `9..=15` makes that
/// one node fail without a certificate while the incumbent is still found.
const BALL_NODE_FAILURE_ITER_BUDGET: usize = 12;

#[test]
fn misocp_incumbent_with_failed_node_is_suboptimal() {
    // Incumbent found, but one node relaxation failed without a certificate:
    // the search is not exhaustive, so Optimal must not be claimed.
    let opts = ConicOptions {
        max_iter: BALL_NODE_FAILURE_ITER_BUDGET,
        ..ConicOptions::default()
    };
    let res = solve_misocp(&ball_int_misocp(), &opts, &BbOptions::default());
    assert_eq!(res.status, SolveStatus::SuboptimalSolution, "{res:?}");
    assert!(
        (res.objective - (-2.0)).abs() < 1e-4,
        "obj={}",
        res.objective
    );
}

#[test]
fn misocp_expired_deadline_returns_timeout() {
    // A deadline that has already passed stops the search before any node:
    // Timeout, no incumbent, and definitely not Infeasible.
    let bb = BbOptions {
        deadline: Some(std::time::Instant::now()),
        ..BbOptions::default()
    };
    let res = solve_misocp(&half_int_lp(), &ConicOptions::default(), &bb);
    assert_eq!(res.status, SolveStatus::Timeout, "{res:?}");
    assert_eq!(res.nodes, 0);
    assert!(res.x.is_empty());
}

#[test]
fn misocp_mid_search_deadline_keeps_incumbent() {
    // Wider instance (37 nodes): max x0+x1+x2 in a ball of r^2 = 30, integers
    // in [0, 5]^3. Deadline at half the measured full runtime T. Timing
    // margins (measured, same-process warm run): the incumbent lands by node
    // 2 of 37 (~T/18), so missing it needs a ~9x mid-test slowdown of the
    // second run relative to the first; finishing the whole search before T/2
    // needs the second run to be 2x faster than the just-completed identical
    // warm run. 45/45 trials landed mid-search (nodes 15..28).
    let n = 3usize;
    let r2 = 30.0_f64;
    let mut grows = vec![vec![0.0; n]];
    let mut h = vec![r2.sqrt()];
    for j in 0..n {
        let mut row = vec![0.0; n];
        row[j] = -1.0;
        grows.push(row);
        h.push(0.0);
    }
    let g = csc(&grows, n + 1, n);
    let base = ConicProblem {
        c: vec![-1.0; n],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap(),
        b: vec![],
        g,
        h,
        cone: ConeSpec {
            l: 0,
            soc: vec![n + 1],
        },
    };
    let prob = MisocpProblem {
        base,
        integers: vec![0, 1, 2],
        int_lb: vec![0.0; n],
        int_ub: vec![5.0; n],
    };
    let t0 = std::time::Instant::now();
    let full = solve_misocp(&prob, &ConicOptions::default(), &BbOptions::default());
    let full_time = t0.elapsed();
    assert!(!full.x.is_empty(), "calibration run must find an incumbent");
    let bb = BbOptions {
        deadline: Some(std::time::Instant::now() + full_time / 2),
        ..BbOptions::default()
    };
    let res = solve_misocp(&prob, &ConicOptions::default(), &bb);
    assert_eq!(res.status, SolveStatus::Timeout, "{res:?}");
    assert!(!res.x.is_empty(), "incumbent must be preserved on timeout");
    assert!(res.nodes < full.nodes, "deadline must actually cut the search");
}

// ---------------------------------------------------------------------------
// IPM Farkas-certificate sentinels for degenerate (fixed-variable) nodes.
// ---------------------------------------------------------------------------

/// Verify a Farkas certificate `(y, z)`: `z ∈ K*`, `A^T y + G^T z ≈ 0`, and
/// `b·y + h·z < 0`.
fn assert_farkas(prob: &ConicProblem, y: &[f64], z: &[f64]) {
    let tol = 1e-6;
    for (i, &zi) in z.iter().enumerate().take(prob.cone.l) {
        assert!(zi >= -tol, "z[{i}] = {zi} must lie in K*");
    }
    let mut off = prob.cone.l;
    for &d in &prob.cone.soc {
        let nr = z[off + 1..off + d]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(z[off] >= nr - tol, "SOC block at {off} not in K*");
        off += d;
    }
    let mut ray = vec![0.0; prob.n()];
    let at = prob.a.transpose();
    let gt = prob.g.transpose();
    let aty = at.mat_vec_mul(y).unwrap();
    let gtz = gt.mat_vec_mul(z).unwrap();
    for i in 0..prob.n() {
        ray[i] = aty[i] + gtz[i];
    }
    let res = ray.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(res < tol, "A^T y + G^T z must vanish, got {res}");
    let val = prob.b.iter().zip(y).map(|(a, b)| a * b).sum::<f64>()
        + prob.h.iter().zip(z).map(|(a, b)| a * b).sum::<f64>();
    assert!(val < -tol, "b·y + h·z must be negative, got {val}");
}

#[test]
fn socp_degenerate_fixed_var_infeasible_gets_certificate() {
    // B&B down-child shape: x fixed to 0 by branching bounds while the base
    // problem requires x >= 1/2. Before the per-iteration certificate check
    // the IPM diverged to non-finite iterates and reported MaxIterations /
    // NumericalError, so branch-and-bound could not prune this node soundly.
    let g = csc(&[vec![-1.0], vec![1.0], vec![-1.0]], 3, 1);
    let prob = ConicProblem {
        c: vec![1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
        b: vec![],
        g,
        h: vec![-0.5, 0.0, 0.0],
        cone: ConeSpec { l: 3, soc: vec![] },
    };
    let res = solve_socp(&prob, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Infeasible, "{res:?}");
    let (y, z) = res.infeas_cert.expect("must carry a Farkas certificate");
    assert_farkas(&prob, &y, &z);
}

#[test]
fn socp_fixed_var_soc_infeasible_gets_certificate() {
    // B&B up-child shape: x0 fixed to 2 by branching bounds while the SOC
    // ball only allows x0 <= sqrt(2.5). Same failure mode as above but with
    // the conflict inside the SOC block.
    let r = 2.5_f64.sqrt();
    let rows = vec![
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, -1.0],
        vec![0.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.0, -1.0],
    ];
    let h = vec![2.0, -2.0, 2.0, 0.0, r, 0.0, 0.0];
    let g = csc(&rows, 7, 2);
    let prob = ConicProblem {
        c: vec![-1.0, -1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
        b: vec![],
        g,
        h,
        cone: ConeSpec { l: 4, soc: vec![3] },
    };
    let res = solve_socp(&prob, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Infeasible, "{res:?}");
    let (y, z) = res.infeas_cert.expect("must carry a Farkas certificate");
    assert_farkas(&prob, &y, &z);
}

#[test]
fn cone_membership_check_covers_all_branches() {
    let blk = super::cone::Blocks::new(&ConeSpec { l: 2, soc: vec![3] });
    let tol = 1e-9;
    // Interior point: inside.
    assert!(super::cone::in_cone(&blk, &[1.0, 2.0, 2.0, 1.0, 1.0], tol));
    // Orthant violation.
    assert!(!super::cone::in_cone(&blk, &[-1.0, 2.0, 2.0, 1.0, 1.0], tol));
    // SOC violation (head < ||rest||).
    assert!(!super::cone::in_cone(&blk, &[1.0, 2.0, 1.0, 1.0, 1.0], tol));
    // Boundary within tolerance: accepted.
    assert!(super::cone::in_cone(
        &blk,
        &[0.0, 0.0, 1.0, 1.0, 0.0],
        tol
    ));
}

// ---------------------------------------------------------------------------
// Scale-invariance sentinels for the certificate checks. The Farkas /
// improving-ray tests must be immune to problem data scale: a large bound V
// must neither fake a certificate (false Infeasible / Unbounded) nor mask a
// genuine one.
// ---------------------------------------------------------------------------

/// `min -x` s.t. `0 <= x <= v` (optimal x = v, objective -v).
fn box_lp(v: f64) -> ConicProblem {
    let g = csc(&[vec![1.0], vec![-1.0]], 2, 1);
    ConicProblem {
        c: vec![-1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
        b: vec![],
        g,
        h: vec![v, 0.0],
        cone: ConeSpec { l: 2, soc: vec![] },
    }
}

#[test]
fn socp_large_bound_box_scale_sweep_is_optimal() {
    for v in [1e6f64, 1e9, 1e10] {
        let res = solve_socp(&box_lp(v), &ConicOptions::default());
        assert_eq!(res.status, SolveStatus::Optimal, "V={v:e}: {res:?}");
        assert!(
            (res.objective + v).abs() <= 1e-6 * v,
            "V={v:e}: obj={}",
            res.objective
        );
    }
}

#[test]
fn socp_extreme_scale_never_falsely_conclusive() {
    // At V >= 1e12 the dense IPM stalls short of the 1e-9 relative tolerance
    // (measured: objective approaches -V monotonically, MaxIterations). The
    // stall is honest; what must never happen is a fake proof: the old
    // magnitude heuristics (cx < -1e11 && ||x|| > 1e8) declared these
    // trivially bounded boxes Unbounded.
    for v in [1e12f64, 1e14] {
        let res = solve_socp(&box_lp(v), &ConicOptions::default());
        assert_ne!(res.status, SolveStatus::Unbounded, "V={v:e}: {res:?}");
        assert_ne!(res.status, SolveStatus::Infeasible, "V={v:e}: {res:?}");
        assert!(res.primal_ray.is_none(), "V={v:e}: unproven ray");
        assert!(res.infeas_cert.is_none(), "V={v:e}: unproven cert");
    }
}

#[test]
fn misocp_fixed_integer_at_large_scale_solves() {
    // Feasible root with the integer fixed at 1e10 by its bounds. The fixed
    // pair used to be encoded as `x <= v`, `-x <= -v` orthant rows whose
    // empty slack interior stalled the IPM, and the scale-dependent Farkas
    // ratio then promoted the stall to a false Infeasible.
    let v = 1e10f64;
    let g = csc(&[vec![1.0]], 1, 1);
    let base = ConicProblem {
        c: vec![1.0],
        a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
        b: vec![],
        g,
        h: vec![2.0 * v],
        cone: ConeSpec { l: 1, soc: vec![] },
    };
    let mp = MisocpProblem {
        base,
        integers: vec![0],
        int_lb: vec![v],
        int_ub: vec![v],
    };
    let res = solve_misocp(&mp, &ConicOptions::default(), &BbOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "{res:?}");
    assert!(
        (res.x[0] - v).abs() <= 1e-6 * v,
        "x={} want {v:e}",
        res.x[0]
    );
}

#[test]
fn socp_large_scale_true_infeasible_gets_certificate() {
    // Reverse-direction sentinel: tightening the Farkas verification must not
    // blind it. `x <= V` vs `x >= V + margin` at V = 1e10, with the margin at
    // 1e-6 relative (well above the 1e-9 verification tolerance) and at 1e0
    // relative.
    let v = 1e10f64;
    for margin in [1e-6 * v, v] {
        let g = csc(&[vec![1.0], vec![-1.0]], 2, 1);
        let prob = ConicProblem {
            c: vec![0.0],
            a: CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(),
            b: vec![],
            g,
            h: vec![v, -(v + margin)],
            cone: ConeSpec { l: 2, soc: vec![] },
        };
        let res = solve_socp(&prob, &ConicOptions::default());
        assert_eq!(
            res.status,
            SolveStatus::Infeasible,
            "margin={margin:e}: {res:?}"
        );
        let (y, z) = res
            .infeas_cert
            .expect("true infeasibility must carry a certificate");
        assert_farkas(&prob, &y, &z);
    }
}

#[test]
fn nt_scaling_block_diagonal_invariants() {
    // Independent recomputation of the Nesterov--Todd scaling invariants for
    // the block-diagonal `Scaling` representation: `W z = Winv s = lambda`,
    // `W` and `Winv` are mutual inverses, and `s = W^2 z` (equivalently
    // `z = Winv^2 s`). Exercises the orthant part and two differently-sized
    // SOC blocks so both the diagonal and dense-block code paths are
    // checked.
    use super::cone::{self, Blocks};

    let cone_spec = ConeSpec {
        l: 2,
        soc: vec![3, 4],
    };
    let blk = Blocks::new(&cone_spec);
    // Strictly interior points: orthant coordinates > 0, SOC blocks with
    // `s[0] > ||s[1:]||` (and likewise for z).
    let s = vec![2.0, 5.0, 3.0, 0.5, 0.4, 4.0, 1.0, 0.5, 0.3];
    let z = vec![1.0, 3.0, 1.0, 0.2, 0.1, 2.0, 0.3, 0.2, 0.1];
    assert_eq!(s.len(), blk.dim());
    assert_eq!(z.len(), blk.dim());

    let sc = cone::nt_scaling(&blk, &s, &z);

    // W z == Winv s == lambda.
    let wz = sc.apply_w(&blk, &z);
    let winv_s = sc.apply_winv(&blk, &s);
    for i in 0..blk.dim() {
        assert!(
            (wz[i] - winv_s[i]).abs() < 1e-9,
            "W z != Winv s at {i}: {} vs {}",
            wz[i],
            winv_s[i]
        );
    }

    // W and Winv are mutual inverses on arbitrary vectors.
    let probe = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, -0.5, 1.5, -2.5];
    let roundtrip_1 = sc.apply_w(&blk, &sc.apply_winv(&blk, &probe));
    let roundtrip_2 = sc.apply_winv(&blk, &sc.apply_w(&blk, &probe));
    for i in 0..blk.dim() {
        assert!(
            (roundtrip_1[i] - probe[i]).abs() < 1e-8,
            "W(Winv v) != v at {i}"
        );
        assert!(
            (roundtrip_2[i] - probe[i]).abs() < 1e-8,
            "Winv(W v) != v at {i}"
        );
    }

    // s == W^2 z: applying the linear operator `W` twice is `W^2` exactly.
    let w2z = sc.apply_w(&blk, &sc.apply_w(&blk, &z));
    for i in 0..blk.dim() {
        assert!(
            (w2z[i] - s[i]).abs() < 1e-8,
            "W^2 z != s at {i}: {} vs {}",
            w2z[i],
            s[i]
        );
    }

    // z == Winv^2 s, the complementary identity.
    let winv2s = sc.apply_winv(&blk, &sc.apply_winv(&blk, &s));
    for i in 0..blk.dim() {
        assert!(
            (winv2s[i] - z[i]).abs() < 1e-8,
            "Winv^2 s != z at {i}: {} vs {}",
            winv2s[i],
            z[i]
        );
    }
}

/// Checks the NT-scaling invariants `W z == Winv s`, `s == W^2 z`, and
/// `z == Winv^2 s` for a given cone configuration and interior pair.
fn assert_nt_invariants(cone_spec: &ConeSpec, s: &[f64], z: &[f64]) {
    use super::cone::{self, Blocks};

    let blk = Blocks::new(cone_spec);
    assert_eq!(s.len(), blk.dim());
    assert_eq!(z.len(), blk.dim());
    let sc = cone::nt_scaling(&blk, s, z);
    let wz = sc.apply_w(&blk, z);
    let winv_s = sc.apply_winv(&blk, s);
    let w2z = sc.apply_w(&blk, &wz);
    let winv2s = sc.apply_winv(&blk, &winv_s);
    for i in 0..blk.dim() {
        assert!(
            (wz[i] - winv_s[i]).abs() < 1e-9,
            "W z != Winv s at {i}: {} vs {}",
            wz[i],
            winv_s[i]
        );
        assert!(
            (w2z[i] - s[i]).abs() < 1e-8,
            "W^2 z != s at {i}: {} vs {}",
            w2z[i],
            s[i]
        );
        assert!(
            (winv2s[i] - z[i]).abs() < 1e-8,
            "Winv^2 s != z at {i}: {} vs {}",
            winv2s[i],
            z[i]
        );
    }
}

#[test]
fn nt_scaling_invariants_edge_configs() {
    // Orthant-only: no SOC blocks, purely diagonal scaling.
    assert_nt_invariants(
        &ConeSpec { l: 3, soc: vec![] },
        &[2.0, 0.5, 7.0],
        &[1.0, 4.0, 0.25],
    );
    // SOC-only: no orthant part.
    assert_nt_invariants(
        &ConeSpec { l: 0, soc: vec![3] },
        &[3.0, 1.0, 0.5],
        &[2.0, 0.4, 0.3],
    );
    // dim=1 SOC blocks: Q_1 = R_+, degenerate tail-free case.
    assert_nt_invariants(
        &ConeSpec { l: 0, soc: vec![1, 1] },
        &[4.0, 0.5],
        &[1.0, 2.0],
    );
    // Mixed with a dim=1 SOC between larger blocks.
    assert_nt_invariants(
        &ConeSpec {
            l: 1,
            soc: vec![2, 1, 3],
        },
        &[3.0, 2.0, 1.0, 5.0, 2.5, 0.5, 1.0],
        &[0.5, 1.5, 0.5, 0.25, 1.0, 0.2, 0.3],
    );
    // Empty cone (m = 0): all operators act on the empty vector.
    assert_nt_invariants(&ConeSpec { l: 0, soc: vec![] }, &[], &[]);
}

#[test]
fn empty_cone_equality_only_socp() {
    // m = 0: purely equality-constrained problem passes through the conic
    // IPM with every cone-block loop empty.
    // min x0 + x1 s.t. x0 = 3, x1 = 4.
    let prob = ConicProblem {
        c: vec![1.0, 1.0],
        a: CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap(),
        b: vec![3.0, 4.0],
        g: CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap(),
        h: vec![],
        cone: ConeSpec { l: 0, soc: vec![] },
    };
    let res = solve_socp(&prob, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!((res.objective - 7.0).abs() < 1e-6, "obj={}", res.objective);
    assert!((res.x[0] - 3.0).abs() < 1e-6);
    assert!((res.x[1] - 4.0).abs() < 1e-6);
}
