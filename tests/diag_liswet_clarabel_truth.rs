//! LISWET-family 12 問の Clarabel 真値取得 diag (baseline 再記録用)。
//! Clarabel (tol=1e-12, max_iter=100k) を独立 reference として objective を出力。

use solver::io::qps::parse_qps;
use solver::QpProblem;
use solver::problem::ConstraintType;
use clarabel::algebra::CscMatrix as ClCsc;
use clarabel::solver::{DefaultSolver, DefaultSettings, IPSolver, SolverStatus, SupportedConeT};

const STRICT_TOL: f64 = 1e-12;
const STRICT_MAX_ITER: u32 = 100_000;

fn build_clarabel(prob: &QpProblem) -> (ClCsc<f64>, Vec<f64>, ClCsc<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
    let n = prob.num_vars;
    let m = prob.num_constraints;
    let n_lb = prob.bounds.iter().filter(|&&(lb, _): &&(f64, f64)| lb.is_finite()).count();
    let n_ub = prob.bounds.iter().filter(|&&(_, ub): &&(f64, f64)| ub.is_finite()).count();

    let mut row_ord: Vec<(usize, ConstraintType)> =
        (0..m).map(|i| (i, prob.constraint_types[i])).collect();
    row_ord.sort_by_key(|&(_, ct)| match ct { ConstraintType::Eq => 0, _ => 1 });
    let n_eq = row_ord.iter().filter(|&&(_, ct)| ct == ConstraintType::Eq).count();
    let n_le_ge = m - n_eq;

    let mut row_pos = vec![0_usize; m];
    for (new_row, &(orig_row, _)) in row_ord.iter().enumerate() {
        row_pos[orig_row] = new_row;
    }

    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let total_rows = m + n_lb + n_ub;
    let mut b_clar = vec![0.0_f64; total_rows];

    for j in 0..n {
        for ptr in prob.a.col_ptr[j]..prob.a.col_ptr[j + 1] {
            let orig_row = prob.a.row_ind[ptr];
            let val = prob.a.values[ptr];
            let new_row = row_pos[orig_row];
            let ct = prob.constraint_types[orig_row];
            match ct {
                ConstraintType::Ge => triplets.push((new_row, j, -val)),
                _ => triplets.push((new_row, j, val)),
            }
        }
    }
    for (orig_row, ct) in prob.constraint_types.iter().enumerate() {
        let new_row = row_pos[orig_row];
        match ct {
            ConstraintType::Ge => b_clar[new_row] = -prob.b[orig_row],
            _ => b_clar[new_row] = prob.b[orig_row],
        }
    }
    let mut bound_row = m;
    for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() {
            triplets.push((bound_row, j, -1.0));
            b_clar[bound_row] = -lb;
            bound_row += 1;
        }
    }
    for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
        if ub.is_finite() {
            triplets.push((bound_row, j, 1.0));
            b_clar[bound_row] = ub;
            bound_row += 1;
        }
    }

    triplets.sort_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &triplets {
        col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    let mut row_ind = vec![0_usize; triplets.len()];
    let mut values = vec![0.0_f64; triplets.len()];
    let mut cursor = col_ptr.clone();
    for &(r, c, v) in &triplets {
        let pos = cursor[c];
        row_ind[pos] = r;
        values[pos] = v;
        cursor[c] += 1;
    }
    let a_clar = ClCsc::new(total_rows, n, col_ptr, row_ind, values);

    let mut p_triplets: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..n {
        for ptr in prob.q.col_ptr[j]..prob.q.col_ptr[j + 1] {
            let i = prob.q.row_ind[ptr];
            if i <= j {
                p_triplets.push((i, j, prob.q.values[ptr]));
            }
        }
    }
    p_triplets.sort_by_key(|&(r, c, _)| (c, r));
    let mut p_col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &p_triplets {
        p_col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        p_col_ptr[j + 1] += p_col_ptr[j];
    }
    let mut p_row_ind = vec![0_usize; p_triplets.len()];
    let mut p_values = vec![0.0_f64; p_triplets.len()];
    let mut p_cursor = p_col_ptr.clone();
    for &(r, c, v) in &p_triplets {
        let pos = p_cursor[c];
        p_row_ind[pos] = r;
        p_values[pos] = v;
        p_cursor[c] += 1;
    }
    let p_clar = ClCsc::new(n, n, p_col_ptr, p_row_ind, p_values);

    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    if n_eq > 0 {
        cones.push(SupportedConeT::ZeroConeT(n_eq));
    }
    if n_le_ge + n_lb + n_ub > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(n_le_ge + n_lb + n_ub));
    }

    (p_clar, prob.c.clone(), a_clar, b_clar, cones)
}

fn compute_internal_obj(prob: &QpProblem, x: &[f64]) -> f64 {
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    0.5 * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + prob.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>()
}

/// Clarabel strict 解で LISWET-family 12 問の真値出力。
/// Clarabel が convergence しなかったら fail。
#[test]
#[ignore = "diag: Clarabel strict tol=1e-12 / max_iter=100k で LISWET-family 12 問の真値取得"]
fn diag_liswet_family_clarabel_truth() {
    let names = [
        "LISWET1", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6",
        "LISWET7", "LISWET8", "LISWET9", "LISWET10", "LISWET11", "LISWET12",
    ];

    let mut results: Vec<(String, String, f64, f64, u32)> = Vec::new();
    let mut failed: Vec<String> = Vec::new();

    for name in &names {
        let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
        assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path);
        let prob = parse_qps(&path).expect("parse failed");
        let (p, q, a, b, cones) = build_clarabel(&prob);

        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.tol_gap_abs = STRICT_TOL;
        settings.tol_gap_rel = STRICT_TOL;
        settings.tol_feas = STRICT_TOL;
        settings.max_iter = STRICT_MAX_ITER;

        let mut solver = match DefaultSolver::new(&p, &q, &a, &b, &cones, settings) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[{}] Clarabel setup error: {:?}", name, e);
                failed.push(format!("{} (setup)", name));
                continue;
            }
        };
        solver.solve();

        let status = format!("{:?}", solver.info.status);
        let cost = solver.info.cost_primal;
        let iters = solver.info.iterations;
        let internal = compute_internal_obj(&prob, &solver.solution.x);

        eprintln!(
            "[{}] status={} iters={} cost_primal={:.10e} internal(via Q,c)={:.10e} obj_offset={:.6e}",
            name, status, iters, cost, internal, prob.obj_offset
        );

        results.push((name.to_string(), status.clone(), cost, internal, iters));

        if !matches!(
            solver.info.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) {
            failed.push(format!("{} ({})", name, status));
        }
    }

    eprintln!("\n========= SUMMARY (LISWET-family Clarabel strict ground truth) =========");
    eprintln!(
        "{:10} {:18} {:>16} {:>16} {:>8}",
        "problem", "status", "cost_primal", "internal_obj", "iters"
    );
    for (name, status, cost, internal, iters) in &results {
        eprintln!(
            "{:10} {:18} {:16.6e} {:16.6e} {:>8}",
            name, status, cost, internal, iters
        );
    }

    assert!(
        failed.is_empty(),
        "Clarabel strict did not converge on: {:?}",
        failed
    );
}
