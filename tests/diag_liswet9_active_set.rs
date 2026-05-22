//! LISWET wrong-basin 観測 diag: Clarabel strict 解と本 solver 解を per-row で
//! 比較し、active set 一致率 / slack 分布を出力する (assertion なし)。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::ConstraintType;
use otspot::qp::solve_qp_with;
use otspot::QpProblem;

use clarabel::algebra::CscMatrix as ClCsc;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};

const STRICT_TOL: f64 = 1e-12;
const STRICT_MAX_ITER: u32 = 100_000;
const ACTIVE_TOL: f64 = 1e-6;

fn build_clarabel(
    prob: &QpProblem,
) -> (ClCsc<f64>, Vec<f64>, ClCsc<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
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

/// Solve QP with Clarabel strict (tol=1e-12) and return (x_ref, y_ref, status, obj).
fn solve_clarabel_strict(prob: &QpProblem) -> Option<(Vec<f64>, Vec<f64>, String, f64)> {
    let (p, q, a, b, cones) = build_clarabel(prob);
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.tol_gap_abs = STRICT_TOL;
    settings.tol_gap_rel = STRICT_TOL;
    settings.tol_feas = STRICT_TOL;
    settings.max_iter = STRICT_MAX_ITER;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).ok()?;
    solver.solve();
    let status = format!("{:?}", solver.info.status);
    if !matches!(
        solver.info.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        eprintln!("  Clarabel status: {}", status);
        return Some((solver.solution.x.clone(), solver.solution.z.clone(), status, solver.info.cost_primal));
    }
    Some((solver.solution.x.clone(), solver.solution.z.clone(), status, solver.info.cost_primal))
}

/// 各 row i の slack r_i を計算: Le 行は b_i - a_i^T x、Ge 行は a_i^T x - b_i、Eq 行は |a_i^T x - b_i|。
fn compute_row_slacks(prob: &QpProblem, x: &[f64]) -> Vec<f64> {
    let m = prob.num_constraints;
    let mut ax = vec![0.0_f64; m];
    for j in 0..prob.num_vars {
        for ptr in prob.a.col_ptr[j]..prob.a.col_ptr[j + 1] {
            let i = prob.a.row_ind[ptr];
            ax[i] += prob.a.values[ptr] * x[j];
        }
    }
    (0..m)
        .map(|i| match prob.constraint_types[i] {
            ConstraintType::Le => prob.b[i] - ax[i],
            ConstraintType::Ge => ax[i] - prob.b[i],
            ConstraintType::Eq => (ax[i] - prob.b[i]).abs(),
            _ => 0.0,
        })
        .collect()
}

/// objective recompute (without offset).
fn obj_internal(prob: &QpProblem, x: &[f64]) -> f64 {
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    0.5 * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + prob.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>()
}

fn diff_active_set(
    name: &str,
    slack_ref: &[f64],
    slack_ours: &[f64],
) {
    let m = slack_ref.len();
    let active_ref: Vec<bool> = slack_ref.iter().map(|&v| v.abs() <= ACTIVE_TOL).collect();
    let active_ours: Vec<bool> = slack_ours.iter().map(|&v| v.abs() <= ACTIVE_TOL).collect();
    let n_ref = active_ref.iter().filter(|&&v| v).count();
    let n_ours = active_ours.iter().filter(|&&v| v).count();
    let only_ref: usize = active_ref.iter().zip(active_ours.iter())
        .filter(|&(&r, &o)| r && !o).count();
    let only_ours: usize = active_ref.iter().zip(active_ours.iter())
        .filter(|&(&r, &o)| !r && o).count();
    let both = active_ref.iter().zip(active_ours.iter())
        .filter(|&(&r, &o)| r && o).count();
    let jaccard = if n_ref + n_ours - both > 0 {
        both as f64 / (n_ref + n_ours - both) as f64
    } else {
        1.0
    };

    // viol = wrong side (slack < -ACTIVE_TOL)
    let viol_ref: usize = slack_ref.iter().filter(|&&v| v < -ACTIVE_TOL).count();
    let viol_ours: usize = slack_ours.iter().filter(|&&v| v < -ACTIVE_TOL).count();
    let max_viol_ref = slack_ref.iter().map(|&v| (-v).max(0.0)).fold(0.0_f64, f64::max);
    let max_viol_ours = slack_ours.iter().map(|&v| (-v).max(0.0)).fold(0.0_f64, f64::max);

    eprintln!(
        "[{}] m={} active_ref={} active_ours={} both={} only_ref={} only_ours={} jaccard={:.4}",
        name, m, n_ref, n_ours, both, only_ref, only_ours, jaccard,
    );
    eprintln!(
        "       viol_ref={} (max={:.3e}) viol_ours={} (max={:.3e})",
        viol_ref, max_viol_ref, viol_ours, max_viol_ours,
    );
}

#[test]
#[ignore = "diag: LISWET-family active-set diff observation"]
fn diag_liswet_family_active_set_diff() {
    let names = [
        "LISWET1", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6",
        "LISWET7", "LISWET8", "LISWET9", "LISWET10", "LISWET11", "LISWET12",
    ];

    eprintln!("\n========= LISWET-family active set diff =========");
    eprintln!("ACTIVE_TOL = {:.0e}", ACTIVE_TOL);

    for name in &names {
        let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
        assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path);
        let prob = parse_qps(&path).expect("parse");

        // Clarabel strict
        let (x_ref, _y_ref, cl_status, cl_obj) = match solve_clarabel_strict(&prob) {
            Some(v) => v,
            None => {
                eprintln!("[{}] Clarabel setup error", name);
                continue;
            }
        };
        let internal_ref = obj_internal(&prob, &x_ref);

        // Ours
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(60.0);
        let res = solve_qp_with(&prob, &opts);
        let internal_ours = obj_internal(&prob, &res.solution);
        let denom = internal_ref.abs().max(internal_ours.abs()).max(1.0);
        let rel_err = (internal_ref - internal_ours).abs() / denom;

        eprintln!(
            "\n[{}] n={} m={} status_ours={:?} cl_status={}",
            name, prob.num_vars, prob.num_constraints, res.status, cl_status,
        );
        eprintln!(
            "       obj_ref={:.6e} (cl_cost={:.6e}) obj_ours={:.6e} rel_err={:.3e}",
            internal_ref, cl_obj, internal_ours, rel_err,
        );

        let slack_ref = compute_row_slacks(&prob, &x_ref);
        let slack_ours = compute_row_slacks(&prob, &res.solution);
        diff_active_set(name, &slack_ref, &slack_ours);

        // |x_ref - x_ours| 分布
        let mut max_dx = 0.0_f64;
        let mut sum_dx = 0.0_f64;
        for (a, b) in x_ref.iter().zip(res.solution.iter()) {
            let d = (a - b).abs();
            max_dx = max_dx.max(d);
            sum_dx += d;
        }
        let mean_dx = sum_dx / x_ref.len() as f64;
        eprintln!(
            "       |x_ref - x_ours|: max={:.3e} mean={:.3e}",
            max_dx, mean_dx,
        );
    }
}

/// max_iter=1,2,5,10,49 で LISWET9 を解き、iter ごとの x 進化 / active-set 変化を観測。
#[test]
#[ignore = "diag: LISWET9 early-iter active-set evolution"]
fn diag_liswet9_early_iter_evolution() {
    let path = std::path::PathBuf::from("data/maros_meszaros/LISWET9.QPS");
    assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path);
    let prob = parse_qps(&path).expect("parse");
    let (x_ref, _y_ref, _cl_status, _cl_obj) = solve_clarabel_strict(&prob).expect("clarabel");
    let slack_ref = compute_row_slacks(&prob, &x_ref);

    eprintln!("\n========= LISWET9 early-iter active-set evolution =========");

    for &max_it in &[1_usize, 2, 5, 10, 20, 49, 100] {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(60.0);
        opts.ipm.max_iter = max_it;
        let res = solve_qp_with(&prob, &opts);
        let slack_ours = compute_row_slacks(&prob, &res.solution);

        let viol: usize = slack_ours.iter().filter(|&&v| v < -ACTIVE_TOL).count();
        let active: usize = slack_ours.iter().filter(|&&v| v.abs() <= ACTIVE_TOL).count();
        let mean_x: f64 = res.solution.iter().map(|v| v.abs()).sum::<f64>() / res.solution.len() as f64;
        let max_x: f64 = res.solution.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let internal = obj_internal(&prob, &res.solution);

        // active vs ref jaccard
        let active_ref: Vec<bool> = slack_ref.iter().map(|&v| v.abs() <= ACTIVE_TOL).collect();
        let active_ours: Vec<bool> = slack_ours.iter().map(|&v| v.abs() <= ACTIVE_TOL).collect();
        let n_ref = active_ref.iter().filter(|&&v| v).count();
        let n_ours = active_ours.iter().filter(|&&v| v).count();
        let both = active_ref.iter().zip(active_ours.iter())
            .filter(|&(&r, &o)| r && o).count();
        let jaccard = if n_ref + n_ours - both > 0 {
            both as f64 / (n_ref + n_ours - both) as f64
        } else { 1.0 };

        eprintln!(
            "max_iter={:>4} status={:?} obj={:.6e} viol={:5} active={:5} jaccard_vs_ref={:.4} mean|x|={:.3e} max|x|={:.3e}",
            max_it, res.status, internal, viol, active, jaccard, mean_x, max_x,
        );
    }
}
