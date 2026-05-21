//! 小規模 Maros 凸 QP に対する Clarabel cross-check 個別 test + 厳密 assertion。
//! 各 test は QpProblem を parse し、Clarabel 参照解と internal obj で比較 (rel < 1e-4)。

use clarabel::algebra::CscMatrix as ClCsc;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
};
use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::solve_qp_with;
use solver::QpProblem;

const CROSS_CHECK_TIMEOUT_SECS: f64 = 60.0;
/// 目的関数 (internal, offset 除く) の相対許容: bench eps=1e-6 と Clarabel
/// strict tol_feas=1e-9 を考慮し、1e-4 を許容上限とする (qp-survey の既存
/// `clarabel_cross_check::deep_check` も同値)。
const CROSS_OBJ_REL_TOL: f64 = 1e-4;

/// Clarabel 形式に問題変換 (clarabel_cross_check.rs から複製・整理)。
fn build_clarabel(
    prob: &QpProblem,
) -> (
    ClCsc<f64>,
    Vec<f64>,
    ClCsc<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let n = prob.num_vars;
    let m = prob.num_constraints;
    let n_lb = prob
        .bounds
        .iter()
        .filter(|&&(lb, _): &&(f64, f64)| lb.is_finite())
        .count();
    let n_ub = prob
        .bounds
        .iter()
        .filter(|&&(_, ub): &&(f64, f64)| ub.is_finite())
        .count();

    let mut row_ord: Vec<(usize, ConstraintType)> =
        (0..m).map(|i| (i, prob.constraint_types[i])).collect();
    row_ord.sort_by_key(|&(_, ct)| match ct {
        ConstraintType::Eq => 0,
        _ => 1,
    });
    let n_eq = row_ord
        .iter()
        .filter(|&&(_, ct)| ct == ConstraintType::Eq)
        .count();
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

    // P upper triangular (Clarabel 規約)
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

fn solve_clarabel(prob: &QpProblem) -> Option<(f64, Vec<f64>)> {
    let (p, q, a, b, cones) = build_clarabel(prob);
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.tol_gap_abs = 1e-9;
    settings.tol_gap_rel = 1e-9;
    settings.tol_feas = 1e-9;
    settings.max_iter = 5000;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).ok()?;
    solver.solve();
    if matches!(
        solver.info.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        Some((solver.info.cost_primal, solver.solution.x.clone()))
    } else {
        None
    }
}

/// 本ソルバの objective は obj_offset 加算済みのため、internal (offset なし) で比較。
fn compute_internal_obj(prob: &QpProblem, x: &[f64]) -> f64 {
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    0.5 * qx
        .iter()
        .zip(x.iter())
        .map(|(&qi, &xi)| qi * xi)
        .sum::<f64>()
        + prob.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>()
}

/// 1 問題分の cross-check 本体。data が無ければ panic (CLAUDE.md「SKIP 禁止」)。
/// 個別 test がデータ欠落で flaky にならないよう、各 test の冒頭で path 存在を確認。
fn cross_check_problem(name: &str) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    assert!(
        path.exists(),
        "{}: data file missing at {:?}, run scripts/setup_extra_benches.sh",
        name, path
    );
    let prob = parse_qps(&path).unwrap_or_else(|e| panic!("{}: parse failed: {:?}", name, e));

    let cl = solve_clarabel(&prob)
        .unwrap_or_else(|| panic!("{}: Clarabel reference failed (Solved/AlmostSolved 期待)", name));
    let cl_internal = compute_internal_obj(&prob, &cl.1);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(CROSS_CHECK_TIMEOUT_SECS);
    let r = solve_qp_with(&prob, &opts);

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "{}: solver status must be Optimal, got {:?} (clarabel ok obj={:.6e})",
        name, r.status, cl.0
    );
    let our_internal = compute_internal_obj(&prob, &r.solution);
    let diff = (our_internal - cl_internal).abs();
    let scale = our_internal.abs().max(cl_internal.abs()).max(1.0);
    let rel = diff / scale;
    assert!(
        rel < CROSS_OBJ_REL_TOL,
        "{}: internal obj mismatch. ours={:.6e} clarabel={:.6e} rel={:.3e} (tol={:.1e})",
        name, our_internal, cl_internal, rel, CROSS_OBJ_REL_TOL
    );
}

// =============================================================================
// Small problems (n <= 10)
// =============================================================================

#[test]
fn cross_hs21() { cross_check_problem("HS21"); }

#[test]
fn cross_hs35() { cross_check_problem("HS35"); }

#[test]
fn cross_hs35mod() { cross_check_problem("HS35MOD"); }

#[test]
fn cross_hs76() { cross_check_problem("HS76"); }

#[test]
fn cross_hs268() { cross_check_problem("HS268"); }

#[test]
fn cross_s268() { cross_check_problem("S268"); }

#[test]
fn cross_zecevic2() { cross_check_problem("ZECEVIC2"); }

#[test]
fn cross_tame() { cross_check_problem("TAME"); }

#[test]
fn cross_genhs28() { cross_check_problem("GENHS28"); }

// =============================================================================
// Medium problems (sparse, n=100..500)
// =============================================================================

#[test]
fn cross_qadlittl() { cross_check_problem("QADLITTL"); }

#[test]
fn cross_qsc205() { cross_check_problem("QSC205"); }

#[test]
fn cross_qscagr7() { cross_check_problem("QSCAGR7"); }

#[test]
fn cross_dualc1() { cross_check_problem("DUALC1"); }

#[test]
fn cross_dualc5() { cross_check_problem("DUALC5"); }

#[test]
fn cross_dual1() { cross_check_problem("DUAL1"); }

#[test]
fn cross_dual2() { cross_check_problem("DUAL2"); }

#[test]
fn cross_dual3() { cross_check_problem("DUAL3"); }

// =============================================================================
// 等式系 (n=200~)
// =============================================================================

#[test]
fn cross_aug2d() { cross_check_problem("AUG2D"); }

#[test]
fn cross_aug2dc() { cross_check_problem("AUG2DC"); }
