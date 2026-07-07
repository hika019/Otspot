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

/// Widens `m` to `new_ncols` columns by appending all-zero columns.
///
/// `O(ncols)`: reuses `m`'s row/value storage as-is rather than rebuilding via
/// triplets, since the appended columns contribute no entries.
fn widen_cols(m: &CscMatrix, new_ncols: usize) -> CscMatrix {
    debug_assert!(new_ncols >= m.ncols());
    let mut col_ptr = m.col_ptr().to_vec();
    let nnz = *col_ptr
        .last()
        .expect("col_ptr always has ncols+1 >= 1 entries");
    col_ptr.resize(new_ncols + 1, nnz);
    CscMatrix {
        col_ptr,
        row_ind: m.row_ind().to_vec(),
        values: m.values().to_vec(),
        nrows: m.nrows(),
        ncols: new_ncols,
    }
}

/// Pivots at or below this are numerically zero (clamped for stability).
const CHOL_PIVOT_ZERO_TOL: f64 = 1e-14;
/// Pivots below this are clearly negative: the matrix is not PSD.
const CHOL_PIVOT_INDEFINITE_TOL: f64 = -1e-9;
/// Replacement value for clamped pivots.
const CHOL_PIVOT_CLAMP: f64 = 1e-7;

/// Sparse left-looking Cholesky of a PSD-with-jitter matrix `p` (`n x n`,
/// both triangles stored, as `QcqpProblem::p0`/`QuadConstraint::p` are).
///
/// Returns column `j` of the lower factor `L` (`p = L L^T`) as `(row, value)`
/// pairs with `row >= j`, for every `j`. Time and memory are `O(nnz(L))`
/// rather than the `O(n^2)` of a dense factorization: `L` stays as sparse as
/// `p` itself when no fill-in occurs (e.g. `nnz(L) = O(n)` for diagonal `p`,
/// the case that drove the QPLIB DCQ bridge OOM this replaces).
///
/// Pivot handling matches the dense Cholesky it replaces: a pivot in
/// `(CHOL_PIVOT_INDEFINITE_TOL, CHOL_PIVOT_ZERO_TOL]` is clamped to
/// `CHOL_PIVOT_CLAMP` (the returned bool is `true` only when that pivot was
/// negative); a pivot below `CHOL_PIVOT_INDEFINITE_TOL` rejects `p` as not
/// PSD (`Err`).
/// Columns of a sparse lower-triangular Cholesky factor: `column[j]` holds the
/// `(row, value)` entries of `L`'s column `j`, `row >= j`.
type SparseCholCols = Vec<Vec<(usize, f64)>>;

fn sparse_cholesky_lower(p: &CscMatrix, n: usize) -> Result<(SparseCholCols, bool), ()> {
    let mut l_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut clamped = false;

    // Reused sparse accumulator for the column currently being eliminated:
    // `acc[r]` holds the running value for row `r >= j`; `touched` lists the
    // rows to reset to 0.0 before the next column (avoids an O(n) clear).
    let mut acc = vec![0.0f64; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut touched_mark = vec![false; n];

    for j in 0..n {
        for &r in &touched {
            acc[r] = 0.0;
            touched_mark[r] = false;
        }
        touched.clear();

        let (rows, vals) = p.get_column(j).map_err(|_| ())?;
        for (&r, &v) in rows.iter().zip(vals) {
            if r >= j {
                if !touched_mark[r] {
                    touched_mark[r] = true;
                    touched.push(r);
                }
                acc[r] += v;
            }
        }

        // Subtract L[:,k] * L[j,k] for every earlier column k<j with a
        // nonzero at row j — `row_to_cols[j]` was populated when those
        // columns were finalized, in ascending k order.
        for &k in &row_to_cols[j] {
            let ljk = l_cols[k]
                .iter()
                .find(|&&(r, _)| r == j)
                .map(|&(_, v)| v)
                .expect("row_to_cols[j] only lists columns with a stored L[j,k]");
            for &(r, v) in &l_cols[k] {
                if r >= j {
                    if !touched_mark[r] {
                        touched_mark[r] = true;
                        touched.push(r);
                    }
                    acc[r] -= v * ljk;
                }
            }
        }

        let pivot = acc[j];
        let ljj = if pivot <= CHOL_PIVOT_ZERO_TOL {
            if pivot < CHOL_PIVOT_INDEFINITE_TOL {
                return Err(());
            }
            if pivot < 0.0 {
                clamped = true;
            }
            CHOL_PIVOT_CLAMP
        } else {
            pivot.sqrt()
        };

        let mut col_j = Vec::with_capacity(touched.len());
        col_j.push((j, ljj));
        for &r in &touched {
            if r > j {
                let val = acc[r] / ljj;
                if val != 0.0 {
                    col_j.push((r, val));
                    row_to_cols[r].push(j);
                }
            }
        }
        l_cols[j] = col_j;
    }

    Ok((l_cols, clamped))
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
///
/// The third tuple element is `true` when any Cholesky factorization clamped
/// a negative jitter-band pivot (see `sparse_cholesky_lower`): the SOCP is
/// then only an approximation of the QCQP and convexity is unproven.
pub fn to_conic(qp: &QcqpProblem) -> Result<(ConicProblem, usize, bool), String> {
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

    // Equalities: A_eq padded to nvar columns. No extra rows, so widening the
    // column count reuses `a_eq`'s storage directly (O(nnz), no triplet pass).
    debug_assert_eq!(
        qp.a_eq.nrows(),
        qp.b_eq.len(),
        "QcqpProblem invariant: a_eq row count must match b_eq length"
    );
    let a = widen_cols(&qp.a_eq, nvar);

    // Conic rows: orthant (linear inequalities) then SOC blocks.
    let ml = qp.h_lin.len();
    debug_assert_eq!(
        qp.g_lin.nrows(),
        ml,
        "QcqpProblem invariant: g_lin row count must match h_lin length"
    );
    let mut gt = Triplets::new();
    let mut h: Vec<f64> = Vec::with_capacity(ml);
    // Orthant: Gl x + s = hl, s >= 0. Transcribed directly from g_lin's own
    // sparse storage (its row space already matches rows 0..ml of G).
    for j in 0..n {
        for idx in qp.g_lin.col_ptr()[j]..qp.g_lin.col_ptr()[j + 1] {
            gt.push(qp.g_lin.row_ind()[idx], j, qp.g_lin.values()[idx]);
        }
    }
    h.extend_from_slice(&qp.h_lin);
    let mut row_off = ml;
    let mut soc: Vec<usize> = Vec::new();

    // Helper to append a convex quadratic "(1/2)xtPx + q^T x + r <= 0" as SOC.
    // With P = R^T R, u = R x:  ||u||^2 <= 2 a b, a = -(q^T x + r), b = 1.
    // SOC block s = [a+b; sqrt2 u; a-b], dim = k+2 (k = n rows of R).
    //
    // `l_cols` is the sparse Cholesky factor of P (`l_cols[i]` = column `i` of
    // `L`, entries `(row, val)` with `row >= i`); since `R = L^T`, row `i` of
    // `R` is exactly column `i` of `L`, so only its nonzero entries are pushed.
    let append_quad = |gt: &mut Triplets,
                       h: &mut Vec<f64>,
                       row_off: &mut usize,
                       soc: &mut Vec<usize>,
                       l_cols: &[Vec<(usize, f64)>],
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
        for (i, col) in l_cols.iter().enumerate() {
            for &(row, val) in col {
                gt.push(base + 1 + i, row, -s2 * val);
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

    let mut convexity_unproven = false;

    // Objective epigraph as first SOC (if quadratic).
    if let Some(p0) = &qp.p0 {
        let (l0, clamped) = sparse_cholesky_lower(p0, n).map_err(|_| "P0 not PSD (nonconvex)")?;
        convexity_unproven |= clamped;
        append_quad(
            &mut gt,
            &mut h,
            &mut row_off,
            &mut soc,
            &l0,
            &qp.q0,
            Some(n),
            0.0,
        );
    }
    // Quadratic constraints.
    for (ci, qc) in qp.quad.iter().enumerate() {
        let (lk, clamped) = sparse_cholesky_lower(&qc.p, n)
            .map_err(|_| format!("P{} not PSD (nonconvex)", ci + 1))?;
        convexity_unproven |= clamped;
        append_quad(
            &mut gt,
            &mut h,
            &mut row_off,
            &mut soc,
            &lk,
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
    Ok((prob, nvar, convexity_unproven))
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
    /// A Cholesky negative-pivot clamp occurred during the SOCP reformulation:
    /// the solved SOCP only approximates the QCQP and the result (whatever the
    /// status) does not certify anything about the original problem.
    pub convexity_unproven: bool,
}

/// Solve a convex QCQP by reformulating to an SOCP.
pub fn solve_qcqp(qp: &QcqpProblem, opts: &ConicOptions) -> QcqpResult {
    let (conic, _nvar, convexity_unproven) = match to_conic(qp) {
        Ok(v) => v,
        Err(e) => {
            return QcqpResult {
                status: SolveStatus::NotSupported(e),
                objective: f64::NAN,
                x: vec![0.0; qp.n],
                iterations: 0,
                convexity_unproven: true,
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
        convexity_unproven,
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

use crate::problem::ConstraintType;
use crate::qp::{QcqpMatrix, QpProblem};

pub(crate) fn qcqp_matrix_to_csc(q: &QcqpMatrix) -> CscMatrix {
    let mut r = Vec::with_capacity(q.triplets.len());
    let mut c = Vec::with_capacity(q.triplets.len());
    let mut v = Vec::with_capacity(q.triplets.len());
    for &(i, j, val) in &q.triplets {
        if val != 0.0 {
            r.push(i);
            c.push(j);
            v.push(val);
        }
    }
    CscMatrix::from_triplets(&r, &c, &v, q.n, q.n).expect("QCQP triplets were validated")
}

/// Convert a core [`QpProblem`] (including QPLIB QCQP output) to a convex
/// conic [`QcqpProblem`]. Linear constraints are split into `<=` and `=`, `>=`
/// rows are sign-flipped, and finite variable bounds become linear inequalities.
///
/// Quadratic constraints must be of `<=` type for this convex conic bridge.
/// Equal/greater quadratic constraints are rejected because a PSD quadratic
/// equality/greater-than set is generally nonconvex.
pub fn qcqp_from_qp_problem(src: &QpProblem) -> Result<QcqpProblem, String> {
    let n = src.num_vars;
    // Row-major view of A (O(nnz)): column k of `a_rows` is row k of `src.a`,
    // giving sparse per-constraint access without ever densifying A (O(m*n)
    // for the QPLIB DCQ problems that drove this bridge's OOM).
    let a_rows = src.a.transpose();
    let mut gl = Triplets::new();
    let mut hl: Vec<f64> = Vec::new();
    let mut ae = Triplets::new();
    let mut be: Vec<f64> = Vec::new();
    let mut quad = Vec::new();
    let mut gl_count = 0usize;
    let mut ae_count = 0usize;

    let has_qc = !src.quadratic_constraints.is_empty();
    for k in 0..src.num_constraints {
        let qmat = if has_qc {
            Some(&src.quadratic_constraints[k])
        } else {
            None
        };
        let (row_idx, row_val) = a_rows
            .get_column(k)
            .map_err(|e| format!("A row {k}: {e:?}"))?;
        match src.constraint_types[k] {
            ConstraintType::Le => {
                if let Some(qc) = qmat {
                    if qc.nnz() > 0 {
                        let mut row = vec![0.0; n];
                        for (&j, &v) in row_idx.iter().zip(row_val) {
                            row[j] = v;
                        }
                        quad.push(QuadConstraint {
                            p: qcqp_matrix_to_csc(qc),
                            q: row,
                            r: -src.b[k],
                        });
                        continue;
                    }
                }
                for (&j, &v) in row_idx.iter().zip(row_val) {
                    gl.push(gl_count, j, v);
                }
                hl.push(src.b[k]);
                gl_count += 1;
            }
            ConstraintType::Ge => {
                if qmat.is_some_and(|qc| qc.nnz() > 0) {
                    return Err(
                        "quadratic >= constraints are nonconvex for the convex QCQP bridge".into(),
                    );
                }
                for (&j, &v) in row_idx.iter().zip(row_val) {
                    gl.push(gl_count, j, -v);
                }
                hl.push(-src.b[k]);
                gl_count += 1;
            }
            ConstraintType::Eq => {
                if qmat.is_some_and(|qc| qc.nnz() > 0) {
                    return Err("quadratic equality constraints are not supported by the convex QCQP bridge".into());
                }
                for (&j, &v) in row_idx.iter().zip(row_val) {
                    ae.push(ae_count, j, v);
                }
                be.push(src.b[k]);
                ae_count += 1;
            }
        }
    }
    // Variable bounds as linear inequalities.
    for j in 0..n {
        let (lb, ub) = src.bounds[j];
        if ub.is_finite() {
            gl.push(gl_count, j, 1.0);
            hl.push(ub);
            gl_count += 1;
        }
        if lb.is_finite() {
            gl.push(gl_count, j, -1.0);
            hl.push(-lb);
            gl_count += 1;
        }
    }

    let g_lin = CscMatrix::from_triplets(&gl.rows, &gl.cols, &gl.vals, gl_count, n)
        .map_err(|e| format!("G_lin build: {e:?}"))?;
    let a_eq = CscMatrix::from_triplets(&ae.rows, &ae.cols, &ae.vals, ae_count, n)
        .map_err(|e| format!("A_eq build: {e:?}"))?;
    Ok(QcqpProblem {
        n,
        p0: Some(src.q.clone()),
        q0: src.c.clone(),
        quad,
        g_lin,
        h_lin: hl,
        a_eq,
        b_eq: be,
    })
}

/// Solve a convex QCQP represented by [`QpProblem`] through the conic bridge.
pub fn solve_qp_problem_as_qcqp(src: &QpProblem, opts: &ConicOptions) -> QcqpResult {
    match qcqp_from_qp_problem(src) {
        Ok(qp) => solve_qcqp(&qp, opts),
        Err(e) => QcqpResult {
            status: SolveStatus::NotSupported(e),
            objective: f64::NAN,
            x: vec![0.0; src.num_vars],
            iterations: 0,
            convexity_unproven: true,
        },
    }
}
