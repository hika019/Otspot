use super::super::*;
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// 不定 Q (対角負値) → 慣性修正 IPM で NonConvex を返さないこと。
#[test]
fn test_qp_nonconvex_indefinite_q() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1.0, 1.0, 1.0], 3, 3).unwrap();
    let c = vec![0.0, 0.0, 0.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert!(
        !matches!(result.status, SolveStatus::NonConvex(_)),
        "got {:?}", result.status
    );
    assert!(
        matches!(
            result.status,
            SolveStatus::LocallyOptimal | SolveStatus::Optimal
            | SolveStatus::Unbounded | SolveStatus::Timeout
            | SolveStatus::SuboptimalSolution | SolveStatus::NumericalError
        ),
        "got {:?}", result.status
    );
}

/// 不定 Q + bounds → LocallyOptimal/Optimal/Suboptimal。
#[test]
fn test_qp_nonconvex_with_bounds() {
    let q = CscMatrix::from_triplets(
        &[0, 1],
        &[0, 1],
        &[-2.0, 2.0],
        2,
        2,
    ).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let b = vec![];
    let bounds = vec![(-1.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds.clone()).unwrap();

    let opts = SolverOptions { timeout_secs: Some(10.0), ..Default::default() };
    let result = solve_qp_with(&problem, &opts);

    assert!(
        !matches!(result.status, SolveStatus::NonConvex(_)),
        "got {:?}", result.status
    );
    assert!(
        matches!(result.status, SolveStatus::LocallyOptimal | SolveStatus::Optimal
            | SolveStatus::SuboptimalSolution | SolveStatus::Timeout),
        "got {:?}", result.status
    );
    if !result.solution.is_empty() {
        for (&xi, &(lb, ub)) in result.solution.iter().zip(bounds.iter()) {
            assert!(xi >= lb - 1e-4 && xi <= ub + 1e-4);
        }
    }
}

/// 半正定値 Q (min eig=0) は PSD 判定。
#[test]
fn test_qp_psd_semidefinite_q() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.0, 1.0, 1.0], 3, 3).unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// SolveStatus::NonConvex の Display。
#[test]
fn test_solve_status_display_nonconvex() {
    let msg = "Q matrix is indefinite".to_string();
    let status = SolveStatus::NonConvex(msg.clone());
    assert_eq!(format!("{}", status), format!("NonConvex({})", msg));
}

/// n>1000 対角負値 → NonPSD 検出。
#[test]
fn test_qp_nonconvex_large_diagonal_negative() {
    let n = 1001_usize;
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = std::iter::once(-1.0_f64)
        .chain(std::iter::repeat(1.0_f64).take(n - 1))
        .collect();
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    assert!(!check_q_positive_semidefinite(&q));
}

/// n>1000 対角全正値 → PSD (偽陽性防止)。
#[test]
fn test_qp_psd_large_diagonal_positive() {
    let n = 1001_usize;
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    let vals: Vec<f64> = vec![1.0_f64; n];
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// 閾値 ‖Q‖_max × 1e-6 内の僅かな負対角値は PSD 扱い (QPS encoding noise)。
#[test]
fn test_qp_diagonal_boundary_below_threshold() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-11_f64, 1.0, 1.0], 3, 3)
        .unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// noise floor (Q[0,0]=-1e-7, ‖Q‖_max=1) は PSD。
#[test]
fn test_qp_diagonal_boundary_at_noise_floor() {
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-7_f64, 1.0, 1.0], 3, 3).unwrap();
    assert!(check_q_positive_semidefinite(&q));
}

/// 閾値 |‖Q‖_max × 1e-6| 超 (Q[0,0]=-1e-4) → NonConvex。
#[test]
fn test_qp_diagonal_boundary_above_threshold() {
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[-1e-4_f64, 1.0, 1.0], 3, 3).unwrap();
    assert!(!check_q_positive_semidefinite(&q));
}

/// UBH1 (n=18009) の Q が sparse LDL で non-PSD と判定されるかを実証 (n>1000 で
/// dense Cholesky skip のため対角正値だけでは検出不能)。
#[test]
fn test_ubh1_q_psd_diagnose() {
    use crate::io::qps::parse_qps;
    use crate::linalg::ldl;
    use std::path::Path;
    use std::time::Instant;

    let path = Path::new("data/maros_meszaros/UBH1.QPS");
    if !path.exists() {
        eprintln!("UBH1.QPS not found, skipping");
        return;
    }
    let prob = parse_qps(path).expect("parse UBH1");
    eprintln!(
        "UBH1: n={}, m={}, Q.nnz={}",
        prob.num_vars,
        prob.num_constraints,
        prob.q.values.len()
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

/// HS268 (n=5, m=5) で IPPMM 出力の dual 残差を成分ごと表示する診断テスト。
#[test]
fn test_hs268_dual_residual_diagnose() {
    use crate::io::qps::parse_qps;
    use crate::options::SolverOptions;
    use std::path::Path;

    let path = Path::new("data/maros_meszaros/HS268.QPS");
    if !path.exists() {
        eprintln!("HS268.QPS not found, skipping");
        return;
    }
    let prob = parse_qps(path).expect("parse HS268");
    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
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
    // 各成分の KKT 残差: Qx + c + A^T y + bound_contrib
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
    let mut at_dense = vec![vec![0.0_f64; m]; n];
    for j in 0..n {
        for k in prob.a.col_ptr[j]..prob.a.col_ptr[j + 1] {
            let i = prob.a.row_ind[k];
            let v = prob.a.values[k];
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

/// Q の対角に ε を加算した CSC を返す (UBH1 PSD 診断用)。
#[cfg(test)]
fn build_q_with_diag_reg(q: &CscMatrix, eps_q: f64) -> CscMatrix {
    let n = q.ncols;
    let mut new_col_ptr = vec![0_usize; n + 1];
    let mut new_row_ind: Vec<usize> = Vec::with_capacity(q.values.len() + n);
    let mut new_values: Vec<f64> = Vec::with_capacity(q.values.len() + n);
    for col in 0..n {
        new_col_ptr[col] = new_row_ind.len();
        let start = q.col_ptr[col];
        let end = q.col_ptr[col + 1];
        let mut diag_added = false;
        for ptr in start..end {
            let row = q.row_ind[ptr];
            let val = q.values[ptr];
            if row == col {
                new_row_ind.push(row);
                new_values.push(val + eps_q);
                diag_added = true;
            } else {
                new_row_ind.push(row);
                new_values.push(val);
            }
        }
        if !diag_added {
            new_row_ind.push(col);
            new_values.push(eps_q);
        }
    }
    new_col_ptr[n] = new_row_ind.len();
    CscMatrix {
        col_ptr: new_col_ptr,
        row_ind: new_row_ind,
        values: new_values,
        nrows: n,
        ncols: n,
    }
}
