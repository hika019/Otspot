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
