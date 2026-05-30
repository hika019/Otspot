use otspot_core::mip::{MilpProblem, MiqpProblem};
use otspot_core::problem::{ConstraintType, LpProblem};
use otspot_core::qp::{QcqpMatrix, QpProblem};
use otspot_core::sparse::CscMatrix;

use super::token_stream::TokenStream;
use super::{QplibError, QplibProblem};

/// Relative tolerance for the QPLIB declared-infinity marker.
///
/// QPLIB files include an explicit `inf_val` field.  Any bound `x` satisfying
/// `|x| >= QPLIB_INF_REL_TOL * inf_val` is treated as ±∞.  The 1% margin
/// absorbs rounding during file generation without falsely classifying
/// finite bounds as infinite.
///
/// Source: QPLIB format (Furini et al., *Math. Prog. Computation* 2019, §3) defines the
/// `inf_val` marker concept; the 1% margin (0.99) is an implementation convention,
/// not explicitly specified in the paper.
const QPLIB_INF_REL_TOL: f64 = 0.99;

pub(super) fn parse_token_stream(mut ts: TokenStream) -> Result<QplibProblem, QplibError> {
    // Problem name (skip)
    let _name = ts.read_string()?;

    // Problem type: 3 chars (Objective, Variables, Constraints)
    let prob_type = ts.read_string()?;
    if prob_type.len() != 3 {
        return Err(QplibError::ParseError(format!(
            "Problem type must be 3 characters, got '{}'",
            prob_type
        )));
    }
    let type_bytes = prob_type.as_bytes();
    let _obj_char = type_bytes[0] as char;
    let var_char = type_bytes[1] as char;
    let con_char = type_bytes[2] as char;

    let var_binary = var_char == 'B';
    let var_integer = var_char == 'I';
    match var_char {
        'C' | 'B' | 'I' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Variable type '{}' not supported (C/B/I supported; M/G/S mixed-integer unsupported). Type={}",
                c, prob_type
            )));
        }
    }

    match con_char {
        'L' | 'B' | 'N' | 'Q' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Constraint type '{}' not supported (only L/B/N/Q supported). Type={}",
                c, prob_type
            )));
        }
    }

    let objsense = ts.read_string()?.to_lowercase();
    let maximize = matches!(objsense.as_str(), "maximize" | "max");

    let n = ts.read_usize()?;
    let m = match con_char {
        'L' | 'N' | 'Q' => ts.read_usize()?,
        _ => 0, // 'B': no m field
    };

    // Objective quadratic terms (lower-triangular, symmetrized)
    let nqobj = ts.read_usize()?;
    if nqobj > n.saturating_mul(n.saturating_add(1)) / 2 {
        return Err(QplibError::ParseError(format!(
            "nqobj {} exceeds n*(n+1)/2={} (n={})",
            nqobj,
            n.saturating_mul(n.saturating_add(1)) / 2,
            n
        )));
    }

    let mut q_triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(nqobj * 2);
    for _ in 0..nqobj {
        let i = ts.read_index_1based(n, "Q row")?;
        let j = ts.read_index_1based(n, "Q col")?;
        let v = ts.read_f64()?;
        q_triplets.push((i, j, v));
        if i != j {
            q_triplets.push((j, i, v));
        }
    }

    // Objective linear terms
    let default_b0 = ts.read_f64()?;
    let mut c = vec![default_b0; n];
    let n_nondefault_b0 = ts.read_usize()?;
    for _ in 0..n_nondefault_b0 {
        let i = ts.read_index_1based(n, "linear obj index")?;
        let v = ts.read_f64()?;
        c[i] = v;
    }

    let q0 = ts.read_f64()?;
    if !q0.is_finite() {
        return Err(QplibError::ParseError(format!(
            "objective constant q0 is not finite: {}",
            q0
        )));
    }

    // Constraint quadratic terms (QCQ only)
    let mut con_q_triplets: Vec<Vec<(usize, usize, f64)>> = if con_char == 'Q' {
        vec![vec![]; m]
    } else {
        vec![]
    };
    if con_char == 'Q' {
        let n_con_quad_terms = ts.read_usize()?;
        for _ in 0..n_con_quad_terms {
            let k = ts.read_index_1based(m, "constraint quad index")?;
            let i = ts.read_index_1based(n, "constraint quad row")?;
            let j = ts.read_index_1based(n, "constraint quad col")?;
            let v = ts.read_f64()?;
            con_q_triplets[k].push((i, j, v));
            if i != j {
                con_q_triplets[k].push((j, i, v));
            }
        }
    }

    // Constraint linear terms (L/N/Q types)
    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();
    if matches!(con_char, 'L' | 'N' | 'Q') {
        let n_con_lin_terms = ts.read_usize()?;
        if n_con_lin_terms > n.saturating_mul(m) {
            return Err(QplibError::ParseError(format!(
                "n_con_lin_terms {} exceeds n*m={}",
                n_con_lin_terms,
                n.saturating_mul(m)
            )));
        }
        a_triplets = Vec::with_capacity(n_con_lin_terms);
        for _ in 0..n_con_lin_terms {
            let k = ts.read_index_1based(m, "constraint index")?;
            let i = ts.read_index_1based(n, "variable index")?;
            let v = ts.read_f64()?;
            a_triplets.push((k, i, v));
        }
    }

    // Infinity value
    let inf_val = ts.read_f64()?;
    let is_pos_inf = |x: f64| x >= inf_val * QPLIB_INF_REL_TOL;
    let is_neg_inf = |x: f64| x <= -inf_val * QPLIB_INF_REL_TOL;

    // Constraint bounds (L/N/Q types)
    let mut lb_con = vec![f64::NEG_INFINITY; m];
    let mut ub_con = vec![f64::INFINITY; m];
    if matches!(con_char, 'L' | 'N' | 'Q') {
        let lb_con_default = ts.read_f64()?;
        let n_nondefault_lb_con = ts.read_usize()?;
        lb_con = vec![lb_con_default; m];
        for _ in 0..n_nondefault_lb_con {
            let k = ts.read_index_1based(m, "lb_con index")?;
            let v = ts.read_f64()?;
            lb_con[k] = v;
        }

        let ub_con_default = ts.read_f64()?;
        let n_nondefault_ub_con = ts.read_usize()?;
        ub_con = vec![ub_con_default; m];
        for _ in 0..n_nondefault_ub_con {
            let k = ts.read_index_1based(m, "ub_con index")?;
            let v = ts.read_f64()?;
            ub_con[k] = v;
        }
    }

    // Variable bounds
    // Binary ('B'): implicit [0,1]; Continuous/Integer: explicit in file.
    let (lb_var, ub_var) = if var_binary {
        (vec![0.0_f64; n], vec![1.0_f64; n])
    } else {
        let lb_var_default = ts.read_f64()?;
        let n_nondefault_lb_var = ts.read_usize()?;
        let mut lb_var = vec![lb_var_default; n];
        for _ in 0..n_nondefault_lb_var {
            let i = ts.read_index_1based(n, "lb_var index")?;
            let v = ts.read_f64()?;
            lb_var[i] = v;
        }
        let ub_var_default = ts.read_f64()?;
        let n_nondefault_ub_var = ts.read_usize()?;
        let mut ub_var = vec![ub_var_default; n];
        for _ in 0..n_nondefault_ub_var {
            let i = ts.read_index_1based(n, "ub_var index")?;
            let v = ts.read_f64()?;
            ub_var[i] = v;
        }
        (lb_var, ub_var)
    };

    // Remaining fields (initial point, dual values, names) are ignored.

    // ── Build QpProblem ───────────────────────────────────────────────────────

    let sign = if maximize { -1.0 } else { 1.0 };

    let q = {
        let q_rows: Vec<usize> = q_triplets.iter().map(|&(r, _, _)| r).collect();
        let q_cols: Vec<usize> = q_triplets.iter().map(|&(_, c, _)| c).collect();
        let q_vals: Vec<f64> = q_triplets.iter().map(|&(_, _, v)| sign * v).collect();
        drop(q_triplets);
        if q_rows.is_empty() {
            CscMatrix::new(n, n)
        } else {
            CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n)
                .map_err(|e| QplibError::ParseError(format!("Q matrix error: {}", e)))?
        }
    };

    if maximize {
        for v in &mut c {
            *v = -*v;
        }
    }

    // Expand lb_con[k] <= a[k]^T x <= ub_con[k] → Ax <= b rows.
    let mut aug_ub_row: Vec<Option<usize>> = vec![None; m];
    let mut aug_lb_row: Vec<Option<usize>> = vec![None; m];
    let mut b_vec: Vec<f64> = Vec::new();
    let mut constraint_types: Vec<ConstraintType> = Vec::new();

    for k in 0..m {
        let lb = lb_con[k];
        let ub = ub_con[k];
        if !is_pos_inf(ub) && !is_neg_inf(lb) && (lb - ub).abs() < 1e-15 {
            // Equality constraint: store as single Eq row.
            aug_ub_row[k] = Some(b_vec.len());
            b_vec.push(ub);
            constraint_types.push(ConstraintType::Eq);
        } else {
            if !is_pos_inf(ub) {
                aug_ub_row[k] = Some(b_vec.len());
                b_vec.push(ub);
                constraint_types.push(ConstraintType::Le);
            }
            if !is_neg_inf(lb) {
                aug_lb_row[k] = Some(b_vec.len());
                b_vec.push(-lb);
                constraint_types.push(ConstraintType::Le);
            }
        }
    }

    let m_aug = b_vec.len();

    let a_mat = {
        let cap = a_triplets.len();
        let mut a_rows: Vec<usize> = Vec::with_capacity(cap);
        let mut a_cols: Vec<usize> = Vec::with_capacity(cap);
        let mut a_vals: Vec<f64> = Vec::with_capacity(cap);

        for &(con_idx, var_idx, val) in &a_triplets {
            if let Some(aug_row) = aug_ub_row[con_idx] {
                a_rows.push(aug_row);
                a_cols.push(var_idx);
                a_vals.push(val);
            }
            if let Some(aug_row) = aug_lb_row[con_idx] {
                a_rows.push(aug_row);
                a_cols.push(var_idx);
                a_vals.push(-val);
            }
        }
        drop(a_triplets);

        if a_rows.is_empty() {
            CscMatrix::new(m_aug, n)
        } else {
            CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_aug, n)
                .map_err(|e| QplibError::ParseError(format!("A matrix error: {}", e)))?
        }
    };

    let bounds: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let lb = if is_neg_inf(lb_var[i]) {
                f64::NEG_INFINITY
            } else {
                lb_var[i]
            };
            let ub = if is_pos_inf(ub_var[i]) {
                f64::INFINITY
            } else {
                ub_var[i]
            };
            (lb, ub)
        })
        .collect();

    // Quadratic constraint matrices (QCQP only).
    // Stored as QcqpMatrix (COO triplets) to avoid O(n) col_ptr per constraint.
    let quadratic_constraints = if con_char == 'Q' {
        let mut qc: Vec<QcqpMatrix> = vec![QcqpMatrix::new(n); m_aug];
        for k in 0..m {
            let trips = &con_q_triplets[k];
            if trips.is_empty() {
                continue;
            }
            if let Some(aug_row) = aug_ub_row[k] {
                qc[aug_row].triplets = trips.clone();
            }
            if let Some(aug_row) = aug_lb_row[k] {
                qc[aug_row].triplets = trips.iter().map(|&(r, c, v)| (r, c, -v)).collect();
            }
        }
        qc
    } else {
        vec![]
    };

    let q0_offset = if maximize { -q0 } else { q0 };

    let mut prob = QpProblem::new(q, c, a_mat, b_vec, bounds, constraint_types)
        .map_err(|e| QplibError::ParseError(e.to_string()))?;
    prob.quadratic_constraints = quadratic_constraints;
    prob.obj_offset = q0_offset;

    if var_binary || var_integer {
        let integer_vars: Vec<usize> = (0..n).collect();
        if prob.q.nnz() == 0 {
            let mut lp = LpProblem::new_general(
                prob.c,
                prob.a,
                prob.b,
                prob.constraint_types,
                prob.bounds,
                None,
            )
            .map_err(|e: otspot_core::error::SolverError| QplibError::ParseError(e.to_string()))?;
            lp.obj_offset = q0_offset;
            let milp = MilpProblem::new(lp, integer_vars).map_err(
                |e: otspot_core::mip::MipProblemError| QplibError::ParseError(e.to_string()),
            )?;
            Ok(QplibProblem::Milp(milp))
        } else {
            let miqp = MiqpProblem::new(prob, integer_vars).map_err(
                |e: otspot_core::mip::MipProblemError| QplibError::ParseError(e.to_string()),
            )?;
            Ok(QplibProblem::Miqp(miqp))
        }
    } else {
        Ok(QplibProblem::Qp(prob))
    }
}
