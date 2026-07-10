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

/// Safety factor multiplying the `n · f64::EPSILON` Cholesky backward-error
/// growth bound, used to build every scale-relative pivot/residual tolerance
/// below.
///
/// # Why the tolerances must be scale-relative
///
/// The pivot and off-diagonal residual tested at column `j` are the
/// *post-elimination* quantities `acc[j] = M[j][j] - Σ_{k<j} L[j][k]^2` and
/// `acc[r] = M[r][j] - Σ_{k<j} L[r][k] L[j][k]`, not the raw input entries.
/// Their floating-point rounding floor scales with `‖M‖`, so an *absolute*
/// tolerance (the earlier `1e-9`) falsely rejects a genuine PSD matrix once
/// `‖M‖ · eps` exceeds it: e.g. a rank-1 `v v^T` with `‖v‖ ~ 1e7` has
/// `‖M‖ ~ 1e14` and a pure-rounding Schur residual `~ 1e-2`, seven orders
/// above `1e-9`, yet is (to double precision) exactly PSD.
///
/// # The bound (Higham, *ASNA* 2nd ed., Thm 10.3–10.4)
///
/// The computed Cholesky factor satisfies `L̂ L̂ᵀ = M + ΔM` with
/// `|ΔM[r][j]| <= γ_{j} · Σ_{k<j} |L̂[r][k]| |L̂[j][k]|`, where
/// `γ_k = k·u / (1 - k·u)` and `u = f64::EPSILON / 2` is the unit roundoff.
/// At a zero pivot the exact residual is zero, so `acc[r]` *is* this error.
/// Cauchy–Schwarz plus `Σ_{k<j} L̂[·][k]^2 <= M[·][·] = orig_diag[·]` give
/// ```text
/// |acc[r]| <= γ_j · sqrt(orig_diag[r] · orig_diag[j])
///          <= n · f64::EPSILON · sqrt(orig_diag[r] · orig_diag[j]),
/// ```
/// using `γ_j <= γ_n <= 2·n·u = n · f64::EPSILON` (valid for `n·u <= 1/2`,
/// i.e. all representable `n`). The diagonal case `r = j` gives the pivot
/// floor `n · f64::EPSILON · orig_diag[j]`. The `n` factor is exactly the
/// elimination-depth error propagation (`γ_n`); it is not a fudge factor.
///
/// This is a *tolerance design*, not a proof of correctness: the intermediate
/// step `Σ_{k<j} L̂[·][k]^2 <= orig_diag[·]` holds in exact arithmetic but in
/// floating point can be exceeded by a `(1 + O(eps))` factor, and Higham's
/// `γ` bound is itself worst-case. `CHOL_ROUNDING_REL_FACTOR` is the
/// engineering headroom absorbing both (plus the `γ_{n+1}` vs `γ_n` slack and
/// non-adversarial accumulation). Its exact value is not delicate: rounding
/// residuals sit at `~ n·eps·sqrt(...)` while a genuine indefiniteness signal
/// is `~ O(1)·sqrt(...)`, ~15 orders larger, so any factor in `[1, 1e6]`
/// separates them. `8` leaves comfortable margin on both sides (measured:
/// the `v v^T` rounding residual `1.56e-2` vs tolerance `0.59`, a 37×
/// margin; a `det = -0.01` indefinite residual `0.1` vs tolerance `5e-15`).
///
/// Degenerate case: when a zero-pivot column also has `orig_diag[j] = 0` (a
/// structurally zero diagonal, e.g. `[[0,1],[1,0]]`), the off-diagonal
/// tolerance `sqrt(orig_diag[j]·orig_diag[r]) = 0`, so the test degenerates
/// to an *exact* `acc[r] != 0` rejection. That is correct, not a gap: for a
/// PSD matrix `M[j][j] = 0` forces row `j ≡ 0` exactly, so the true rounding
/// floor there really is zero and any nonzero residual is genuine
/// indefiniteness.
const CHOL_ROUNDING_REL_FACTOR: f64 = 8.0;
/// Absolute floor for the "treat a positive pivot as an exact zero and drop
/// the factor column" (rank-deficiency) decision, used when the column's
/// original diagonal is `~0` so the relative floor `n·eps·orig_diag[j]`
/// vanishes. Preserves the small-scale exact-drop behavior (`diag(1,0)`,
/// zero matrix) that `20a0d960` relies on to keep unbounded directions free.
const CHOL_PIVOT_ZERO_ABS_FLOOR: f64 = 1e-14;
/// Absolute floor for the "pivot is genuinely negative → not PSD" rejection,
/// used when the column's original diagonal is `~0`. Preserves the
/// small-scale jitter band: an `O(1)` matrix with a `-1e-10` diagonal entry
/// is accepted (clamped), not rejected. At large scale the relative floor
/// `n·eps·orig_diag[j]` dominates this constant, so it never causes the
/// scale-driven false rejection that motivated relativizing.
const CHOL_PIVOT_INDEFINITE_ABS_FLOOR: f64 = 1e-9;

/// Columns of a sparse lower-triangular Cholesky factor: `column[j]` holds the
/// `(row, value)` entries of `L`'s column `j`, `row >= j`.
type SparseCholCols = Vec<Vec<(usize, f64)>>;

/// Sparse left-looking Cholesky of a PSD-with-jitter matrix `p` (`n x n`,
/// both triangles stored, as `QcqpProblem::p0`/`QuadConstraint::p` are).
///
/// Returns column `j` of the lower factor `L` (`p = L L^T`) as `(row, value)`
/// pairs with `row >= j`, for every `j`. Time and memory are `O(nnz(L))`
/// rather than the `O(n^2)` of a dense factorization: `L` stays as sparse as
/// `p` itself when no fill-in occurs (e.g. `nnz(L) = O(n)` for diagonal `p`,
/// the case that drove the QPLIB DCQ bridge OOM this replaces).
///
/// Pivot handling (all tolerances scale-relative — see
/// `CHOL_ROUNDING_REL_FACTOR`). Let `noise_j = CHOL_ROUNDING_REL_FACTOR ·
/// n · eps · orig_diag[j]` be the rounding floor of the post-elimination
/// pivot at column `j`.
/// - `pivot < -max(CHOL_PIVOT_INDEFINITE_ABS_FLOOR, noise_j)`: the pivot is
///   negative beyond rounding — not PSD (`Err`).
/// - `pivot <= max(CHOL_PIVOT_ZERO_ABS_FLOOR, noise_j)`: rank-deficient PSD
///   *iff* every off-diagonal residual `acc[r]` (r > j) is within
///   `CHOL_ROUNDING_REL_FACTOR · n · eps · sqrt(orig_diag[j]·orig_diag[r])`
///   of zero; then the factor column is set to zero exactly (matching SOC
///   row vanishes, preserving any unbounded direction). A residual beyond
///   that bound makes the `{j,r}` 2x2 principal minor certifiably negative
///   (`M[j][j] ≈ 0 ⟹ minor = -M[r][j]^2 < 0`), so the matrix is rejected
///   (`Err`) no matter how small the pivot is.
/// - otherwise `pivot.sqrt()`.
///
/// The returned bool (`convexity_unproven`) is `true` only when a pivot was
/// negative beyond the rounding floor but within the jitter band
/// (`pivot < -noise_j`, i.e. genuine small-scale indefiniteness clamped to
/// zero); within-rounding residuals and non-negative pivots are treated as
/// PSD-to-precision and leave it `false`.
fn sparse_cholesky_lower(p: &CscMatrix, n: usize) -> Result<(SparseCholCols, bool), ()> {
    let mut l_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut clamped = false;

    // Original diagonal of `p` (both triangles stored, so column `j`'s stored
    // entry at row `j` is `M[j][j]`; duplicate triplets accumulate, matching
    // how `acc` sums the raw column below). Sets the scale of every rounding
    // tolerance (see `CHOL_ROUNDING_REL_FACTOR`).
    let mut orig_diag = vec![0.0f64; n];
    for j in 0..n {
        let (rows, vals) = p.get_column(j).map_err(|_| ())?;
        for (&r, &v) in rows.iter().zip(vals) {
            if r == j {
                orig_diag[j] += v;
            }
        }
    }

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
        // Rounding floor of the post-elimination pivot at column `j`; scales
        // with the original diagonal so the decision tracks the matrix's true
        // (relative) definiteness rather than the factorization's absolute
        // rounding cancellation (see `CHOL_ROUNDING_REL_FACTOR`).
        let od_j = orig_diag[j].max(0.0);
        let noise_j = CHOL_ROUNDING_REL_FACTOR * (n as f64) * f64::EPSILON * od_j;
        let zero_tol = noise_j.max(CHOL_PIVOT_ZERO_ABS_FLOOR);
        let indef_tol = noise_j.max(CHOL_PIVOT_INDEFINITE_ABS_FLOOR);
        let ljj = if pivot <= zero_tol {
            if pivot < -indef_tol {
                return Err(());
            }
            // A zero/jitter-band pivot only certifies a rank-deficient PSD
            // direction if the off-diagonal entries sharing this column are
            // also within rounding of zero: for PSD `M`, `M[j][j] = 0` forces
            // `M[r][j] = 0`, so a residual beyond its own rounding floor
            // `n·eps·sqrt(orig_diag[j]·orig_diag[r])` proves indefiniteness
            // rather than noise. Reject rather than silently drop a genuinely
            // nonzero Schur-complement entry.
            for &r in &touched {
                if r > j {
                    let offdiag_tol = CHOL_ROUNDING_REL_FACTOR
                        * (n as f64)
                        * f64::EPSILON
                        * (od_j * orig_diag[r].max(0.0)).sqrt();
                    if acc[r].abs() > offdiag_tol {
                        return Err(());
                    }
                }
            }
            // Accepted as rank-deficient PSD (within rounding): drop the whole
            // factor column to zero, removing the SOC row instead of adding
            // ~sqrt(pivot) curvature that would bound an unbounded direction.
            // Flag as approximate (unproven) only when the pivot is negative
            // beyond the rounding floor — a genuine (if tiny) indefiniteness
            // clamped to zero. A non-negative pivot or a within-rounding
            // residual is PSD-to-precision and leaves the flag clear.
            if pivot < -noise_j {
                clamped = true;
            }
            0.0
        } else {
            pivot.sqrt()
        };

        let mut col_j = Vec::with_capacity(touched.len());
        col_j.push((j, ljj));
        if ljj != 0.0 {
            for &r in &touched {
                if r > j {
                    let val = acc[r] / ljj;
                    if val != 0.0 {
                        col_j.push((r, val));
                        row_to_cols[r].push(j);
                    }
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

/// Reject a [`QcqpProblem`] whose public fields disagree on the variable
/// count `n`. `QcqpProblem` is a public struct with no constructor, so callers
/// can build a mis-sized instance; without this check `to_conic` would panic
/// (`copy_from_slice`, indexing) or silently drop a mis-sized objective matrix
/// via `qcqp_objective`'s fallback.
fn validate_dims(qp: &QcqpProblem) -> Result<(), String> {
    let n = qp.n;
    if qp.q0.len() != n {
        return Err(format!("q0 length {} != n {}", qp.q0.len(), n));
    }
    if let Some(p0) = &qp.p0 {
        if p0.nrows() != n || p0.ncols() != n {
            return Err(format!(
                "P0 is {}x{}, expected {n}x{n}",
                p0.nrows(),
                p0.ncols()
            ));
        }
    }
    for (i, qc) in qp.quad.iter().enumerate() {
        if qc.p.nrows() != n || qc.p.ncols() != n {
            return Err(format!(
                "quad[{i}].p is {}x{}, expected {n}x{n}",
                qc.p.nrows(),
                qc.p.ncols()
            ));
        }
        if qc.q.len() != n {
            return Err(format!("quad[{i}].q length {} != n {n}", qc.q.len()));
        }
    }
    if qp.g_lin.ncols() != n {
        return Err(format!("g_lin has {} cols, expected {n}", qp.g_lin.ncols()));
    }
    if qp.g_lin.nrows() != qp.h_lin.len() {
        return Err(format!(
            "g_lin has {} rows but h_lin has {}",
            qp.g_lin.nrows(),
            qp.h_lin.len()
        ));
    }
    if qp.a_eq.ncols() != n {
        return Err(format!("a_eq has {} cols, expected {n}", qp.a_eq.ncols()));
    }
    if qp.a_eq.nrows() != qp.b_eq.len() {
        return Err(format!(
            "a_eq has {} rows but b_eq has {}",
            qp.a_eq.nrows(),
            qp.b_eq.len()
        ));
    }
    Ok(())
}

/// Convert this convex QCQP into an equivalent standard-form SOCP.
///
/// A trailing epigraph variable `t` is appended when the objective is
/// quadratic; the SOCP variable vector is `[x (n) , t?]`.
///
/// The third tuple element is `true` when any Cholesky factorization clamped
/// a negative jitter-band pivot to zero (a genuine small-scale indefiniteness
/// within the rounding floor — see `sparse_cholesky_lower`): the SOCP is then
/// only an approximation of the QCQP and convexity is unproven.
pub fn to_conic(qp: &QcqpProblem) -> Result<(ConicProblem, usize, bool), String> {
    validate_dims(qp)?;
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
    let a = widen_cols(&qp.a_eq, nvar);

    // Conic rows: orthant (linear inequalities) then SOC blocks.
    let ml = qp.h_lin.len();
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
    /// A Cholesky negative jitter-band pivot was clamped to zero during the
    /// SOCP reformulation (genuine small-scale indefiniteness within the
    /// rounding floor): the solved SOCP only approximates the QCQP and the
    /// result (whatever the status) does not certify anything about the
    /// original problem.
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
/// A quadratic `>=` row is sign-flipped into the equivalent `<=` form (`P ->
/// -P`) exactly like a linear one; whether that is actually convex is left to
/// `to_conic`'s per-constraint Cholesky (accepts `-P` PSD, rejects it
/// indefinite as `NotSupported`, same as any other quadratic `<=` row) rather
/// than assumed nonconvex up front. Quadratic equality constraints are still
/// rejected: a PSD quadratic equality set is generally nonconvex.
pub fn qcqp_from_qp_problem(src: &QpProblem) -> Result<QcqpProblem, String> {
    let n = src.num_vars;
    // `QpProblem::quadratic_constraints` is a public field (PR #25 review):
    // `set_quadratic_constraints` enforces "empty or num_constraints", but a
    // caller can still assign the field directly and violate it, and the loop
    // below indexes `[k]` for every `k < num_constraints` unconditionally.
    if !src.quadratic_constraints.is_empty()
        && src.quadratic_constraints.len() != src.num_constraints
    {
        return Err(format!(
            "quadratic_constraints length must be 0 or {}, got {}",
            src.num_constraints,
            src.quadratic_constraints.len()
        ));
    }
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
                if let Some(qc) = qmat {
                    if qc.nnz() > 0 {
                        // (1/2)x^T Q x + a^T x >= b  <=>  (1/2)x^T(-Q)x + (-a)^T x + b <= 0.
                        // Whether `-Q` is actually PSD (so this row's
                        // superlevel set is convex) is left to `to_conic`'s
                        // per-constraint Cholesky, exactly like any other
                        // quadratic `<=` row.
                        let mut row = vec![0.0; n];
                        for (&j, &v) in row_idx.iter().zip(row_val) {
                            row[j] = -v;
                        }
                        quad.push(QuadConstraint {
                            p: qcqp_matrix_to_csc(qc).scale_values(-1.0),
                            q: row,
                            r: src.b[k],
                        });
                        continue;
                    }
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
    // Variable bounds. A fixed bound (`lb == ub`) becomes an equality row
    // `x_j = lb`: the inequality pair `x_j <= ub`, `-x_j <= -lb` has no
    // strictly feasible slack, which stalls the interior-point method (the
    // MISOCP path special-cases fixed bounds for the same reason).
    for j in 0..n {
        let (lb, ub) = src.bounds[j];
        if lb.is_finite() && ub.is_finite() && lb == ub {
            ae.push(ae_count, j, 1.0);
            be.push(lb);
            ae_count += 1;
            continue;
        }
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
        // A structurally zero objective matrix is a *linear* objective: keep
        // `p0 = None` so `to_conic` skips the quadratic epigraph reformulation
        // (an epigraph plus pivot handling would perturb the linear optimum).
        p0: (!src.is_zero_q()).then(|| src.q.clone()),
        q0: src.c.clone(),
        quad,
        g_lin,
        h_lin: hl,
        a_eq,
        b_eq: be,
    })
}

/// Distinct global variable indices touched by `triplets` (both row and
/// col), sorted ascending and deduplicated: `O(nnz log nnz)`.
fn touched_vars(triplets: &[(usize, usize, f64)]) -> Vec<usize> {
    let mut vars: Vec<usize> = Vec::with_capacity(2 * triplets.len());
    for &(r, c, _) in triplets {
        vars.push(r);
        vars.push(c);
    }
    vars.sort_unstable();
    vars.dedup();
    vars
}

/// Sparse Cholesky of a PSD-with-jitter matrix given as symmetric COO
/// `triplets` (both triangles stored, `QcqpMatrix`'s convention) over an
/// implicit `n_global x n_global` space, doing all work over just the
/// touched variables: `O(nnz)` time and memory instead of `O(n_global)`.
///
/// Exact, not an approximation: rows and columns outside `triplets`' support
/// are identically zero, so eliminating only the induced principal submatrix
/// (index-compressed, order-preserving) reproduces the full-space
/// `sparse_cholesky_lower` factor, whose every other row is the zero vector.
///
/// Returns `(l_cols, clamped)`: `l_cols[i]` is column `i` of that submatrix's
/// lower factor `L` as `(global_row, value)` pairs, with identically-zero
/// columns dropped (a zero row of `R` changes neither `||Rx||^2` nor the
/// feasible set); `clamped` matches `sparse_cholesky_lower`'s jitter-clamp
/// contract.
fn touched_cholesky(triplets: &[(usize, usize, f64)]) -> Result<(SparseCholCols, bool), ()> {
    let nz: Vec<(usize, usize, f64)> = triplets
        .iter()
        .copied()
        .filter(|&(_, _, v)| v != 0.0)
        .collect();
    let local_to_global = touched_vars(&nz);
    let k = local_to_global.len();
    let mut local_rows = Vec::with_capacity(nz.len());
    let mut local_cols = Vec::with_capacity(nz.len());
    let mut vals = Vec::with_capacity(nz.len());
    for &(r, c, v) in &nz {
        let lr = local_to_global
            .binary_search(&r)
            .expect("touched_vars collects every triplet row");
        let lc = local_to_global
            .binary_search(&c)
            .expect("touched_vars collects every triplet col");
        local_rows.push(lr);
        local_cols.push(lc);
        vals.push(v);
    }
    let local_p =
        CscMatrix::from_triplets(&local_rows, &local_cols, &vals, k, k).map_err(|_| ())?;
    let (l_local, clamped) = sparse_cholesky_lower(&local_p, k)?;
    let l_global: SparseCholCols = l_local
        .into_iter()
        .filter(|col| col.iter().any(|&(_, v)| v != 0.0))
        .map(|col| {
            col.into_iter()
                .filter(|&(_, v)| v != 0.0)
                .map(|(local_row, val)| (local_to_global[local_row], val))
                .collect()
        })
        .collect();
    Ok((l_global, clamped))
}

/// Extracts `(row, col, val)` triplets from a `CscMatrix`: `O(ncols + nnz)`
/// time (must visit every column-pointer slot to know which are empty), but
/// only `O(nnz)` output. Used for the objective Hessian `src.q`, a single
/// per-problem matrix whose own `O(n)` `col_ptr` is paid once — unlike the
/// per-constraint case `qp_problem_to_conic` otherwise avoids entirely.
fn csc_to_triplets(m: &CscMatrix) -> Vec<(usize, usize, f64)> {
    let mut t = Vec::with_capacity(m.nnz());
    for j in 0..m.ncols() {
        let (rows, vals) = m.get_column(j).expect("j < ncols by loop bound");
        for (&r, &v) in rows.iter().zip(vals) {
            t.push((r, j, v));
        }
    }
    t
}

/// Dense linear-term vector to sparse `(index, value)` pairs, dropping
/// zeros.
fn sparse_nz(v: &[f64]) -> Vec<(usize, f64)> {
    v.iter()
        .enumerate()
        .filter(|&(_, &x)| x != 0.0)
        .map(|(j, &x)| (j, x))
        .collect()
}

/// Appends a convex quadratic `(1/2) x^T P x + q^T x + r <= 0` (the
/// objective epigraph when `qt_coef` is `Some`) as one SOC block, from a
/// *compact* Cholesky factor (`touched_cholesky`'s output: `l_cols.len()`
/// tracks the constraint's own touched-variable rank, not the problem's
/// global variable count) and a sparse linear term. Same embedding formula
/// as `to_conic`'s `append_quad`: `s = [a+b; sqrt2 * R x; a-b]`, `dim =
/// l_cols.len() + 2`.
fn append_quad_compact(
    gt: &mut Triplets,
    h: &mut Vec<f64>,
    row_off: &mut usize,
    soc: &mut Vec<usize>,
    l_cols: &[Vec<(usize, f64)>],
    q_nz: &[(usize, f64)],
    qt_coef: Option<usize>,
    r: f64,
) {
    let k = l_cols.len();
    let dim = k + 2;
    let base = *row_off;
    for &(j, qv) in q_nz {
        gt.push(base, j, qv);
    }
    if let Some(tj) = qt_coef {
        gt.push(base, tj, -1.0);
    }
    h.push(1.0 - r);
    let s2 = std::f64::consts::SQRT_2;
    for (i, col) in l_cols.iter().enumerate() {
        for &(row, val) in col {
            gt.push(base + 1 + i, row, -s2 * val);
        }
        h.push(0.0);
    }
    for &(j, qv) in q_nz {
        gt.push(base + 1 + k, j, qv);
    }
    if let Some(tj) = qt_coef {
        gt.push(base + 1 + k, tj, -1.0);
    }
    h.push(-r - 1.0);
    *row_off += dim;
    soc.push(dim);
}

/// Convert a `QpProblem` (QPLIB QCQP form) directly into an SOCP, without
/// materializing an intermediate `QcqpProblem`/`Vec<QuadConstraint>`.
///
/// `qcqp_from_qp_problem` + `to_conic` costs `O(n)` per quadratic constraint
/// just to *store* it: `QuadConstraint::p`'s CSC `col_ptr` and `q`'s dense
/// `Vec<f64>` are sized to the global variable count `n` — a public
/// cross-crate contract via `otspot-model`, so neither can shrink to the
/// constraint's own touched-variable count. All `m` are held live at once, so
/// QPLIB_8585 (`n=99999`, `m=49999`, `nnz=1` each) needs tens of GB before any
/// solving starts, versus `O(nnz) = O(m)` here.
///
/// This builds each constraint's SOC block from its own sparse
/// `QcqpMatrix::triplets` and sparse linear coefficients, one at a time.
/// Semantics mirror `qcqp_from_qp_problem` + `to_conic` exactly: same row
/// transcription (`Le` kept, `Ge` sign-flipped -- including a quadratic `Ge`
/// row's matrix, screened for convexity by `touched_cholesky` like any other
/// quadratic row -- `Eq`/quadratic-`Eq` rejected, finite bounds as inequality
/// rows, a fixed bound as an equality row), same SOC embedding formula, same
/// error messages.
pub(crate) fn qp_problem_to_conic(src: &QpProblem) -> Result<(ConicProblem, usize, bool), String> {
    let n = src.num_vars;
    // `QpProblem::quadratic_constraints` is a public field (PR #25 review):
    // `set_quadratic_constraints` enforces "empty or num_constraints", but a
    // caller can still assign the field directly and violate it, and the loop
    // below indexes `[k]` for every `k < num_constraints` unconditionally.
    if !src.quadratic_constraints.is_empty()
        && src.quadratic_constraints.len() != src.num_constraints
    {
        return Err(format!(
            "quadratic_constraints length must be 0 or {}, got {}",
            src.num_constraints,
            src.quadratic_constraints.len()
        ));
    }
    let has_quad_obj = !src.is_zero_q();
    let nvar = n + if has_quad_obj { 1 } else { 0 };

    let mut c = vec![0.0; nvar];
    if has_quad_obj {
        c[n] = 1.0;
    } else {
        c[..n].copy_from_slice(&src.c);
    }

    let a_rows = src.a.transpose();
    // `gt`/`h` accumulate the orthant rows first (indices 0..ml), then the
    // SOC block rows are appended directly after — no intermediate `g_lin`
    // `CscMatrix` is built or re-scanned.
    let mut gt = Triplets::new();
    let mut h: Vec<f64> = Vec::new();
    let mut ae = Triplets::new();
    let mut be: Vec<f64> = Vec::new();
    let mut gl_count = 0usize;
    let mut ae_count = 0usize;

    struct StagedQuad<'a> {
        triplets: std::borrow::Cow<'a, [(usize, usize, f64)]>,
        q_nz: Vec<(usize, f64)>,
        r: f64,
    }
    let mut staged: Vec<StagedQuad> = Vec::new();

    let has_qc = !src.quadratic_constraints.is_empty();
    for k in 0..src.num_constraints {
        let qmat = has_qc
            .then(|| &src.quadratic_constraints[k])
            .filter(|qc| qc.nnz() > 0);
        let (row_idx, row_val) = a_rows
            .get_column(k)
            .map_err(|e| format!("A row {k}: {e:?}"))?;
        match src.constraint_types[k] {
            ConstraintType::Le => {
                if let Some(qc) = qmat {
                    let q_nz: Vec<(usize, f64)> =
                        row_idx.iter().zip(row_val).map(|(&j, &v)| (j, v)).collect();
                    staged.push(StagedQuad {
                        triplets: std::borrow::Cow::Borrowed(&qc.triplets),
                        q_nz,
                        r: -src.b[k],
                    });
                    continue;
                }
                for (&j, &v) in row_idx.iter().zip(row_val) {
                    gt.push(gl_count, j, v);
                }
                h.push(src.b[k]);
                gl_count += 1;
            }
            ConstraintType::Ge => {
                if let Some(qc) = qmat {
                    // (1/2)x^T Q x + a^T x >= b  <=>  (1/2)x^T(-Q)x + (-a)^T x + b <= 0.
                    // `touched_cholesky` below screens `-Q` for PSD-ness
                    // exactly like any other quadratic `<=` row; an
                    // indefinite `-Q` still rejects with the same
                    // "not PSD (nonconvex)" error as before.
                    let q_nz: Vec<(usize, f64)> = row_idx
                        .iter()
                        .zip(row_val)
                        .map(|(&j, &v)| (j, -v))
                        .collect();
                    let neg_triplets: Vec<(usize, usize, f64)> =
                        qc.triplets.iter().map(|&(r, c, v)| (r, c, -v)).collect();
                    staged.push(StagedQuad {
                        triplets: std::borrow::Cow::Owned(neg_triplets),
                        q_nz,
                        r: src.b[k],
                    });
                    continue;
                }
                for (&j, &v) in row_idx.iter().zip(row_val) {
                    gt.push(gl_count, j, -v);
                }
                h.push(-src.b[k]);
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
    for j in 0..n {
        let (lb, ub) = src.bounds[j];
        if lb.is_finite() && ub.is_finite() && lb == ub {
            ae.push(ae_count, j, 1.0);
            be.push(lb);
            ae_count += 1;
            continue;
        }
        if ub.is_finite() {
            gt.push(gl_count, j, 1.0);
            h.push(ub);
            gl_count += 1;
        }
        if lb.is_finite() {
            gt.push(gl_count, j, -1.0);
            h.push(-lb);
            gl_count += 1;
        }
    }

    let a_eq_narrow = CscMatrix::from_triplets(&ae.rows, &ae.cols, &ae.vals, ae_count, n)
        .map_err(|e| format!("A_eq build: {e:?}"))?;
    let a = widen_cols(&a_eq_narrow, nvar);

    let ml = gl_count;
    let mut row_off = ml;
    let mut soc: Vec<usize> = Vec::new();
    let mut convexity_unproven = false;

    if has_quad_obj {
        let p0_triplets = csc_to_triplets(&src.q);
        let (l0, clamped) =
            touched_cholesky(&p0_triplets).map_err(|_| "P0 not PSD (nonconvex)".to_string())?;
        convexity_unproven |= clamped;
        let q0_nz = sparse_nz(&src.c);
        append_quad_compact(
            &mut gt,
            &mut h,
            &mut row_off,
            &mut soc,
            &l0,
            &q0_nz,
            Some(n),
            0.0,
        );
    }
    for (ci, sq) in staged.iter().enumerate() {
        let (lk, clamped) = touched_cholesky(sq.triplets.as_ref())
            .map_err(|_| format!("P{} not PSD (nonconvex)", ci + 1))?;
        convexity_unproven |= clamped;
        append_quad_compact(
            &mut gt,
            &mut h,
            &mut row_off,
            &mut soc,
            &lk,
            &sq.q_nz,
            None,
            sq.r,
        );
    }

    let m = h.len();
    let g = CscMatrix::from_triplets(&gt.rows, &gt.cols, &gt.vals, m, nvar)
        .map_err(|e| format!("G build: {e:?}"))?;
    let cone = ConeSpec { l: ml, soc };
    let prob = ConicProblem {
        c,
        a,
        b: be,
        g,
        h,
        cone,
    };
    Ok((prob, nvar, convexity_unproven))
}

/// Recomputes `1/2 x^T Q x + c^T x` for a [`QpProblem`] at a candidate `x`
/// (mirrors `qcqp_objective`, but reads `src.q`/`src.c` directly rather than
/// a `QcqpProblem`'s `p0`/`q0` — used by `solve_qp_problem_as_qcqp` since it
/// no longer builds a `QcqpProblem` intermediate).
fn qp_problem_objective(src: &QpProblem, x: &[f64]) -> f64 {
    let mut obj = 0.0;
    for (j, &cv) in src.c.iter().enumerate() {
        obj += cv * x[j];
    }
    if !src.is_zero_q() {
        let px = src
            .q
            .mat_vec_mul(x)
            .unwrap_or_else(|_| vec![0.0; src.num_vars]);
        let mut xpx = 0.0;
        for j in 0..src.num_vars {
            xpx += x[j] * px[j];
        }
        obj += 0.5 * xpx;
    }
    obj
}

/// Solve a convex QCQP represented by [`QpProblem`] through the conic bridge.
///
/// Each quadratic constraint's SOC block is built directly from its own sparse
/// triplets, so peak memory tracks the problem's total `nnz` rather than
/// `n * m` over `m` constraints of `n` variables.
pub fn solve_qp_problem_as_qcqp(src: &QpProblem, opts: &ConicOptions) -> QcqpResult {
    let (conic, _nvar, convexity_unproven) = match qp_problem_to_conic(src) {
        Ok(v) => v,
        Err(e) => {
            return QcqpResult {
                status: SolveStatus::NotSupported(e),
                objective: f64::NAN,
                x: vec![0.0; src.num_vars],
                iterations: 0,
                convexity_unproven: true,
            }
        }
    };
    let res: ConicResult = solve_socp(&conic, opts);
    let x = res.x[..src.num_vars].to_vec();
    let objective = qp_problem_objective(src, &x);
    QcqpResult {
        status: res.status,
        objective,
        x,
        iterations: res.iterations,
        convexity_unproven,
    }
}
