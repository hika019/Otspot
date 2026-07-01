//! Convex QCQP support via reformulation to an SOCP.
//!
//! Problem form:
//! ```text
//! minimize    (1/2) x^T P0 x + q0^T x
//! subject to  (1/2) x^T Pi x + qi^T x + ri <= 0   (i = 1..M)
//!             Gl x <= hl                           (linear inequalities)
//!             Ae x = be                            (linear equalities)
//! ```
//! Each `Pi` (and `P0`) must be symmetric positive semidefinite (convex case).

use super::{solve_socp, ConeSpec, ConicOptions, ConicProblem, ConicResult};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// A convex quadratic constraint `(1/2) x^T P x + q^T x + r <= 0` with `P` PSD.
#[derive(Debug, Clone)]
pub struct QuadConstraint {
    /// PSD matrix `P` (`n x n`).
    pub p: CscMatrix,
    /// Linear term `q` (length `n`).
    pub q: Vec<f64>,
    /// Constant `r`.
    pub r: f64,
}

/// A convex QCQP.
#[derive(Debug, Clone)]
pub struct QcqpProblem {
    /// Number of variables.
    pub n: usize,
    /// Optional PSD objective matrix `P0` (`n x n`); `None` = linear objective.
    pub p0: Option<CscMatrix>,
    /// Objective linear term (length `n`).
    pub q0: Vec<f64>,
    /// Quadratic constraints.
    pub quad: Vec<QuadConstraint>,
    /// Linear inequality matrix `Gl` (`ml x n`), `Gl x <= hl`.
    pub g_lin: CscMatrix,
    /// Linear inequality rhs.
    pub h_lin: Vec<f64>,
    /// Linear equality matrix `Ae` (`p x n`).
    pub a_eq: CscMatrix,
    /// Linear equality rhs.
    pub b_eq: Vec<f64>,
}

fn dense(a: &CscMatrix) -> Vec<Vec<f64>> {
    let mut d = vec![vec![0.0; a.ncols()]; a.nrows()];
    let cp = a.col_ptr();
    let ri = a.row_ind();
    let va = a.values();
    for j in 0..a.ncols() {
        for k in cp[j]..cp[j + 1] {
            d[ri[k]][j] = va[k];
        }
    }
    d
}

/// Upper Cholesky factor `R` (`n x n`) with `P = R^T R`, or `None` if not PD.
fn cholesky_upper(p: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0; n]; n]; // lower
    for i in 0..n {
        for j in 0..=i {
            let mut sum = p[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 1e-14 {
                    // Allow tiny PSD jitter; hard-fail on clearly negative.
                    if sum < -1e-9 {
                        return None;
                    }
                    l[i][j] = 1e-7;
                } else {
                    l[i][j] = sum.sqrt();
                }
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // R = L^T
    let mut r = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            r[j][i] = l[i][j];
        }
    }
    Some(r)
}

struct Triplets {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl Triplets {
    fn new() -> Self {
        Triplets {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }
    fn push(&mut self, r: usize, c: usize, v: f64) {
        if v != 0.0 {
            self.rows.push(r);
            self.cols.push(c);
            self.vals.push(v);
        }
    }
}

/// Convert this convex QCQP into an equivalent standard-form SOCP.
///
/// A trailing epigraph variable `t` is appended when the objective is
/// quadratic; the SOCP variable vector is `[x (n) , t?]`.
pub fn to_conic(qp: &QcqpProblem) -> Result<(ConicProblem, usize), String> {
    let n = qp.n;
    let has_quad_obj = qp.p0.is_some();
    let nvar = n + if has_quad_obj { 1 } else { 0 };

    // Objective c.
    let mut c = vec![0.0; nvar];
    if has_quad_obj {
        c[n] = 1.0; // minimize t
    } else {
        c[..n].copy_from_slice(&qp.q0);
    }

    // Equalities: A_eq padded to nvar columns.
    let ae = dense(&qp.a_eq);
    let p_eq = qp.b_eq.len();
    let mut at = Triplets::new();
    for (i, row) in ae.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            at.push(i, j, v);
        }
    }
    let a = CscMatrix::from_triplets(&at.rows, &at.cols, &at.vals, p_eq, nvar)
        .map_err(|e| format!("A build: {e:?}"))?;

    // Conic rows: orthant (linear inequalities) then SOC blocks.
    let gl = dense(&qp.g_lin);
    let ml = qp.h_lin.len();
    let mut gt = Triplets::new();
    let mut h: Vec<f64> = Vec::new();
    // Orthant: Gl x + s = hl, s >= 0.
    for (i, row) in gl.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            gt.push(i, j, v);
        }
        h.push(qp.h_lin[i]);
    }
    let mut row_off = ml;
    let mut soc: Vec<usize> = Vec::new();

    // Helper to append a convex quadratic "(1/2)xtPx + q^T x + r <= 0" as SOC.
    // With P = R^T R, u = R x:  ||u||^2 <= 2 a b, a = -(q^T x + r), b = 1.
    // SOC block s = [a+b; sqrt2 u; a-b], dim = k+2 (k = n rows of R).
    let append_quad = |gt: &mut Triplets,
                       h: &mut Vec<f64>,
                       row_off: &mut usize,
                       soc: &mut Vec<usize>,
                       rmat: &[Vec<f64>],
                       q: &[f64],
                       qt_coef: Option<usize>,
                       r: f64| {
        let k = n; // R is n x n
        let dim = k + 2;
        let base = *row_off;
        // s0 = a + b = 1 - r - q^T x  (+ (-1)*t if qt_coef)
        //   s = h - G x  => G row = q (and +1 for t coeff), h = 1 - r
        for (j, &qv) in q.iter().enumerate() {
            gt.push(base, j, qv);
        }
        if let Some(tj) = qt_coef {
            gt.push(base, tj, -1.0); // q^T x - t : coefficient of t is -1
        }
        h.push(1.0 - r);
        // s_{1..k} = sqrt2 * R x  => G row = -sqrt2 R, h = 0
        let s2 = std::f64::consts::SQRT_2;
        for (ri, rrow) in rmat.iter().enumerate() {
            for (j, &rv) in rrow.iter().enumerate() {
                gt.push(base + 1 + ri, j, -s2 * rv);
            }
            h.push(0.0);
        }
        // s_last = a - b = -r - 1 - q^T x  => G row = q (+ -1 t), h = -r - 1
        for (j, &qv) in q.iter().enumerate() {
            gt.push(base + 1 + k, j, qv);
        }
        if let Some(tj) = qt_coef {
            gt.push(base + 1 + k, tj, -1.0);
        }
        h.push(-r - 1.0);
        *row_off += dim;
        soc.push(dim);
    };

    // Objective epigraph as first SOC (if quadratic).
    if let Some(p0) = &qp.p0 {
        let p0d = dense(p0);
        let r0 = cholesky_upper(&p0d, n).ok_or("P0 not PSD (nonconvex)")?;
        append_quad(
            &mut gt,
            &mut h,
            &mut row_off,
            &mut soc,
            &r0,
            &qp.q0,
            Some(n),
            0.0,
        );
    }
    // Quadratic constraints.
    for (ci, qc) in qp.quad.iter().enumerate() {
        let pd = dense(&qc.p);
        let rr =
            cholesky_upper(&pd, n).ok_or_else(|| format!("P{} not PSD (nonconvex)", ci + 1))?;
        append_quad(
            &mut gt,
            &mut h,
            &mut row_off,
            &mut soc,
            &rr,
            &qc.q,
            None,
            qc.r,
        );
    }

    let m = h.len();
    let g = CscMatrix::from_triplets(&gt.rows, &gt.cols, &gt.vals, m, nvar)
        .map_err(|e| format!("G build: {e:?}"))?;
    let cone = ConeSpec { l: ml, soc };
    let prob = ConicProblem {
        c,
        a,
        b: qp.b_eq.clone(),
        g,
        h,
        cone,
    };
    Ok((prob, nvar))
}

/// Result of a QCQP solve, with `x` restricted to the original variables.
#[derive(Debug, Clone)]
pub struct QcqpResult {
    /// Status.
    pub status: SolveStatus,
    /// Objective value of the original QCQP.
    pub objective: f64,
    /// Solution for the original `n` variables.
    pub x: Vec<f64>,
    /// Iterations.
    pub iterations: usize,
}

/// Solve a convex QCQP by reformulating to an SOCP.
pub fn solve_qcqp(qp: &QcqpProblem, opts: &ConicOptions) -> QcqpResult {
    let (conic, _nvar) = match to_conic(qp) {
        Ok(v) => v,
        Err(e) => {
            return QcqpResult {
                status: SolveStatus::NotSupported(e),
                objective: f64::NAN,
                x: vec![0.0; qp.n],
                iterations: 0,
            }
        }
    };
    let res: ConicResult = solve_socp(&conic, opts);
    let x = res.x[..qp.n].to_vec();
    // Recompute the true QCQP objective from x.
    let objective = qcqp_objective(qp, &x);
    QcqpResult {
        status: res.status,
        objective,
        x,
        iterations: res.iterations,
    }
}

fn qcqp_objective(qp: &QcqpProblem, x: &[f64]) -> f64 {
    let mut obj = 0.0;
    for (j, &qv) in qp.q0.iter().enumerate() {
        obj += qv * x[j];
    }
    if let Some(p0) = &qp.p0 {
        let px = p0.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; qp.n]);
        let mut xpx = 0.0;
        for j in 0..qp.n {
            xpx += x[j] * px[j];
        }
        obj += 0.5 * xpx;
    }
    obj
}
