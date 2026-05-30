// Bug-reproduction tests that exercise parser + solver together.
// Moved from otspot-core (which used crate::io internally) to break the source duplication.
// Uses otspot-io parsers + public otspot-core solver APIs.

use otspot_core::linalg::ldl;
use otspot_core::options::SolverOptions;
use otspot_core::problem::SolveStatus;
use otspot_core::qp::{
    ipm_solver::{kkt::kkt_residual_rel, outcome::ProblemView, solve_ipm},
    solve_qp_with,
};
use otspot_core::sparse::CscMatrix;
use otspot_io::{mps, qps};
use std::path::Path;
use std::time::Instant;

/// Build Q + eps*I using public CscMatrix API (triplet dedup sums duplicates).
fn build_q_with_diag_reg(q: &CscMatrix, eps: f64) -> CscMatrix {
    let n = q.ncols();
    let col_ptr = q.col_ptr();
    let row_ind = q.row_ind();
    let values = q.values();
    let mut rows: Vec<usize> = Vec::with_capacity(values.len() + n);
    let mut cols: Vec<usize> = Vec::with_capacity(values.len() + n);
    let mut vals: Vec<f64> = Vec::with_capacity(values.len() + n);
    for col in 0..n {
        for ptr in col_ptr[col]..col_ptr[col + 1] {
            rows.push(row_ind[ptr]);
            cols.push(col);
            vals.push(values[ptr]);
        }
    }
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(eps);
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).expect("build_q_with_diag_reg")
}

/// UBH1 (n=18009): verify Q non-PSD detection via sparse LDL.
///
/// Diagnostic test — no assertions; prints LDL factorization result per eps.
#[test]
fn test_ubh1_q_psd_diagnose() {
    let path = Path::new("data/maros_meszaros/UBH1.QPS");
    if !path.exists() {
        eprintln!("UBH1.QPS not found, skipping");
        return;
    }
    let prob = qps::parse_qps(path).expect("parse UBH1");
    eprintln!(
        "UBH1: n={}, m={}, Q.nnz={}",
        prob.num_vars,
        prob.num_constraints,
        prob.q.values().len()
    );
    for eps in &[0.0_f64, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-3, 1.0] {
        let q_reg = build_q_with_diag_reg(&prob.q, *eps);
        let t = Instant::now();
        match ldl::factorize(&q_reg) {
            Ok(_) => eprintln!(
                "  eps={:.0e}: factorize OK (Q+εI PSD), {:.2}s",
                eps,
                t.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!(
                "  eps={:.0e}: factorize FAILED ({:?}), {:.2}s",
                eps,
                e,
                t.elapsed().as_secs_f64()
            ),
        }
    }
}

/// HS268 (n=5, m=5): dual residual component diagnosis.
///
/// Diagnostic test — no assertions; prints KKT residuals and reconstructed duals.
#[test]
#[allow(clippy::needless_range_loop)]
fn test_hs268_dual_residual_diagnose() {
    let path = Path::new("data/maros_meszaros/HS268.QPS");
    if !path.exists() {
        eprintln!("HS268.QPS not found, skipping");
        return;
    }
    let prob = qps::parse_qps(path).expect("parse HS268");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(5.0);
    let result = solve_qp_with(&prob, &opts);
    eprintln!(
        "HS268 status={:?} obj={:.6e}",
        result.status, result.objective
    );
    let x = &result.solution;
    let y = &result.dual_solution;
    let bd = &result.bound_duals;
    eprintln!("  x = {:?}", x);
    eprintln!("  y = {:?}", y);
    eprintln!("  bound_duals = {:?} (len={})", bd, bd.len());
    let qx = prob.q.mat_vec_mul(x).unwrap();
    let aty = if !y.is_empty() {
        prob.a.transpose().mat_vec_mul(y).unwrap()
    } else {
        vec![0.0; prob.num_vars]
    };
    for j in 0..prob.num_vars {
        let r = qx[j] + prob.c[j] + aty[j];
        eprintln!(
            "    j={}: Qx={:.3e} c={:.3e} (A^Ty)={:.3e} sum={:.3e}",
            j, qx[j], prob.c[j], aty[j], r
        );
    }
    let n = prob.num_vars;
    let m = prob.num_constraints;
    let col_ptr = prob.a.col_ptr();
    let row_ind = prob.a.row_ind();
    let values = prob.a.values();
    let mut at_dense = vec![vec![0.0_f64; m]; n];
    for j in 0..n {
        for k in col_ptr[j]..col_ptr[j + 1] {
            let i = row_ind[k];
            let v = values[k];
            if i < m {
                at_dense[j][i] = v;
            }
        }
    }
    let rhs: Vec<f64> = (0..n).map(|j| -(qx[j] + prob.c[j])).collect();
    let mut aug = at_dense.clone();
    let mut b = rhs.clone();
    for k in 0..n.min(m) {
        let mut max_row = k;
        for i in (k + 1)..n {
            if aug[i][k].abs() > aug[max_row][k].abs() {
                max_row = i;
            }
        }
        aug.swap(k, max_row);
        b.swap(k, max_row);
        if aug[k][k].abs() < 1e-15 {
            eprintln!("  singular at k={}", k);
            return;
        }
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..m {
                aug[i][j] -= factor * aug[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut y_recon = vec![0.0_f64; m];
    for k in (0..n.min(m)).rev() {
        let mut sum = b[k];
        for j in (k + 1)..m {
            sum -= aug[k][j] * y_recon[j];
        }
        y_recon[k] = sum / aug[k][k];
    }
    eprintln!("  reconstructed y (LSQ): {:?}", y_recon);
    eprintln!("  ratio (solver_y / recon_y):");
    for i in 0..m.min(y.len()) {
        if y_recon[i].abs() > 1e-15 {
            eprintln!("    i={}: ratio={:.4}", i, y[i] / y_recon[i]);
        }
    }
}

/// HS21: compare full QP solver vs raw IPM on the same problem.
///
/// Diagnostic test — no assertions; prints KKT residuals for both.
#[test]
fn test_ipm_hs21_cmp_full_solver() {
    let path = Path::new("data/maros_meszaros/HS21.QPS");
    if !path.exists() {
        eprintln!("HS21.QPS not found, skipping");
        return;
    }
    let prob = qps::parse_qps(path).expect("parse HS21");
    let opts = SolverOptions::default();
    let v1 = solve_qp_with(&prob, &opts);
    let v2 = solve_ipm(&prob, &opts);
    eprintln!("=== v1 ===");
    eprintln!(
        "  status={:?} obj={} iters={}",
        v1.status, v1.objective, v1.iterations
    );
    eprintln!("=== v2 ===");
    eprintln!(
        "  status={:?} obj={} iters={}",
        v2.status, v2.objective, v2.iterations
    );
    let view = ProblemView {
        q: &prob.q,
        a: &prob.a,
        c: &prob.c,
        b: &prob.b,
        bounds: &prob.bounds,
        constraint_types: &prob.constraint_types,
        eliminated_cols: &[],
    };
    let v1_kkt = kkt_residual_rel(&view, &v1.solution, &v1.dual_solution, &v1.bound_duals);
    let v2_kkt = kkt_residual_rel(&view, &v2.solution, &v2.dual_solution, &v2.bound_duals);
    eprintln!("v1 KKT_rel={:.3e}", v1_kkt);
    eprintln!("v2 KKT_rel={:.3e}", v2_kkt);
}

/// scsd6: LP with 147 equality constraints.  Must not return NumericalError.
#[test]
fn test_scsd6_equality_constraints() {
    let path = Path::new("data/lp_problems/scsd6.QPS");
    if !path.exists() {
        return;
    }
    let content = std::fs::read_to_string(path).unwrap();
    let lp = mps::parse_mps(&content).unwrap();

    use otspot_core::options::SimplexMethod;
    let methods = [
        ("Auto", SimplexMethod::Auto),
        ("Primal", SimplexMethod::Primal),
        ("Dual", SimplexMethod::Dual),
    ];
    let results: Vec<_> = methods
        .iter()
        .map(|(name, method)| {
            let mut opts = SolverOptions::default();
            opts.simplex_method = *method;
            opts.presolve = false;
            let result = otspot_core::lp::solve_lp_with(&lp, &opts);
            eprintln!(
                "scsd6 {} -> {:?} obj={:.3e}",
                name, result.status, result.objective
            );
            (*name, result.status)
        })
        .collect();

    for (name, status) in &results {
        assert_ne!(
            *status,
            SolveStatus::NumericalError,
            "scsd6 {} returned NumericalError",
            name
        );
    }
}
