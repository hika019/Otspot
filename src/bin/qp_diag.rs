//! QP 診断バイナリ: 1 問題を solve_qp_with で解いて、
//! `result.solution`, `result.dual_solution`, `result.bound_duals` から
//! 元空間 KKT 残差を bench と同じ式で再計算し、内訳を表示する。
//!
//! 使い方: `qp_diag <path/to/problem.QPS> [solver=ipm|ippmm_new|concurrent]`
//! 環境変数:
//!   DIAG_NO_RUIZ=1     — Ruiz scaling 無効化
//!   DIAG_NO_PRESOLVE=1 — presolve 無効化
//!
//! T1.2 QRECIPE / T1.4 Catastrophic 系の root-cause 診断で利用 (2026-04-30)。

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::path::PathBuf;
use solver::io::qps::parse_qps;
use solver::options::{QpSolverChoice, SolverOptions};
use solver::problem::ConstraintType;
use solver::qp::solve_qp_with;
use solver::QpProblem;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path/to/file.QPS> [solver]", args[0]);
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let solver_name = args.get(2).cloned().unwrap_or_else(|| "ippmm_new".to_string());
    let solver_choice = match solver_name.as_str() {
        "ipm" => QpSolverChoice::Ipm,
        "ippmm_new" => QpSolverChoice::IpPmmNew,
        "concurrent" => QpSolverChoice::Concurrent,
        _ => panic!("unknown solver: {}", solver_name),
    };

    let prob_box = parse_qps(&path).expect("parse failed");
    let prob: &QpProblem = &prob_box;
    let mut opts = SolverOptions::default();
    let timeout = std::env::var("DIAG_TIMEOUT").ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(30.0);
    opts.timeout_secs = Some(timeout);
    opts.qp_solver = solver_choice;
    if std::env::var("DIAG_NO_RUIZ").ok().as_deref() == Some("1") {
        opts.use_ruiz_scaling = false;
        println!("[DIAG] Ruiz scaling DISABLED");
    }
    if std::env::var("DIAG_NO_PRESOLVE").ok().as_deref() == Some("1") {
        opts.presolve = false;
        println!("[DIAG] presolve DISABLED");
    }
    let result = solve_qp_with(prob, &opts);

    println!("status={:?} obj={:.6e} iters={}", result.status, result.objective, result.iterations);
    // primal residual も併記 (T1.3 LISWET 系の診断用)
    if !result.solution.is_empty() && prob.num_constraints > 0 {
        let ax = prob.a.mat_vec_mul(&result.solution).expect("Ax");
        let mut max_pf = 0.0_f64;
        let mut max_ax = 0.0_f64;
        let mut max_b = 0.0_f64;
        for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
            let v = match prob.constraint_types[i] {
                solver::problem::ConstraintType::Eq => (ax_i - b_i).abs(),
                solver::problem::ConstraintType::Ge => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            };
            let _ = i;
            max_pf = max_pf.max(v);
            max_ax = max_ax.max(ax_i.abs());
            max_b = max_b.max(b_i.abs());
        }
        let pfn = max_pf / (1.0 + max_ax.max(max_b));
        // 違反制約の分布を集計
        let mut n_violated_above_eps = 0_usize;
        let mut n_violated = 0_usize;
        for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
            let v = match prob.constraint_types[i] {
                solver::problem::ConstraintType::Eq => (ax_i - b_i).abs(),
                solver::problem::ConstraintType::Ge => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            };
            if v > 1e-12 { n_violated += 1; }
            if v > 1e-6  { n_violated_above_eps += 1; }
        }
        println!("primal: pf_abs={:.3e} pf_rel={:.3e} max_ax={:.3e} max_b={:.3e}",
            max_pf, pfn, max_ax, max_b);
        println!("  violations: >1e-12: {} >1e-6: {} (total m={})",
            n_violated, n_violated_above_eps, prob.num_constraints);
    }
    println!("n={} m={} n_lb={} n_ub={}",
        prob.num_vars, prob.num_constraints,
        prob.bounds.iter().filter(|&&(lb, _): &&(f64, f64)| lb.is_finite()).count(),
        prob.bounds.iter().filter(|&&(_, ub): &&(f64, f64)| ub.is_finite()).count(),
    );
    println!("|x|_inf={:.3e} |y|_inf={:.3e} |z|_inf={:.3e} (z=bound_duals.len={})",
        result.solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs())),
        result.dual_solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs())),
        result.bound_duals.iter().fold(0.0_f64, |a, &v| a.max(v.abs())),
        result.bound_duals.len(),
    );

    // 元空間 KKT 残差を bench と同じ式で再計算
    let n = prob.num_vars;
    let qx = prob.q.mat_vec_mul(&result.solution).expect("Qx");
    let aty: Vec<f64> = if prob.a.nrows > 0 && !result.dual_solution.is_empty() {
        prob.a.transpose().mat_vec_mul(&result.dual_solution).expect("Aty")
    } else {
        vec![0.0; n]
    };

    // bound_contrib: -y_lb (lb 有限) + y_ub (ub 有限)
    let mut bound_contrib = vec![0.0_f64; n];
    let mut bd_idx = 0usize;
    for (j, &(lb, _ub)) in prob.bounds.iter().enumerate() {
        let lb: f64 = lb;
        if lb.is_finite() && bd_idx < result.bound_duals.len() {
            bound_contrib[j] -= result.bound_duals[bd_idx];
            bd_idx += 1;
        }
    }
    for (j, &(_lb, ub)) in prob.bounds.iter().enumerate() {
        let ub: f64 = ub;
        if ub.is_finite() && bd_idx < result.bound_duals.len() {
            bound_contrib[j] += result.bound_duals[bd_idx];
            bd_idx += 1;
        }
    }

    // FX/EmptyCol を skip した bench 形式
    let mut max_r_skip = 0.0_f64;
    let mut max_r_full = 0.0_f64;
    let mut argmax_full = 0usize;
    let mut argmax_skip = 0usize;
    let mut max_qx = 0.0_f64;
    let mut max_c = 0.0_f64;
    let mut max_aty = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    let mut n_fx_skipped = 0;
    let mut n_empty_skipped = 0;
    for j in 0..n {
        let (lb_j, ub_j): (f64, f64) = prob.bounds[j];
        let r = (qx[j] + aty[j] + bound_contrib[j] + prob.c[j]).abs();
        if r > max_r_full {
            max_r_full = r;
            argmax_full = j;
        }
        let is_fx = lb_j.is_finite() && ub_j.is_finite() && (lb_j - ub_j).abs() < 1e-12;
        let is_empty_col = prob.a.col_ptr.len() > j + 1
            && prob.a.col_ptr[j + 1] - prob.a.col_ptr[j] == 0;
        if is_fx { n_fx_skipped += 1; continue; }
        if is_empty_col { n_empty_skipped += 1; continue; }
        if r > max_r_skip {
            max_r_skip = r;
            argmax_skip = j;
        }
        max_qx = max_qx.max(qx[j].abs());
        max_c = max_c.max(prob.c[j].abs());
        max_aty = max_aty.max(aty[j].abs());
        max_bnd = max_bnd.max(bound_contrib[j].abs());
    }
    let scale = 1.0 + max_qx.max(max_c).max(max_aty).max(max_bnd);
    println!();
    println!("=== bench-style dfeas (skip FX/EmptyCol) ===");
    println!("max_r_full={:.3e} argmax={} (no skip)", max_r_full, argmax_full);
    println!("max_r_skip={:.3e} argmax={} (n_fx_skip={} n_empty_skip={})",
        max_r_skip, argmax_skip, n_fx_skipped, n_empty_skipped);
    println!("max_qx={:.3e} max_c={:.3e} max_aty={:.3e} max_bnd={:.3e}",
        max_qx, max_c, max_aty, max_bnd);
    println!("scale=1+max(...)={:.3e} dfeas_rel={:.3e}", scale, max_r_skip / scale);

    // worst variable の内訳
    let j = argmax_skip;
    let (lb, ub) = prob.bounds[j];
    let ct_count: usize = (0..prob.num_constraints).filter(|&i| {
        prob.a.col_ptr[j+1].saturating_sub(prob.a.col_ptr[j]) > 0
        && (prob.a.col_ptr[j]..prob.a.col_ptr[j+1]).any(|k| prob.a.row_ind[k] == i)
    }).count();
    println!();
    println!("worst variable j={}: x={:.6e} bounds=({},{}) refs_in_A={}",
        j, result.solution[j], lb, ub, ct_count);
    println!("  qx_j={:.3e} c_j={:.3e} aty_j={:.3e} bound_contrib_j={:.3e}",
        qx[j], prob.c[j], aty[j], bound_contrib[j]);
    println!("  sum (residual)={:.3e}", qx[j] + prob.c[j] + aty[j] + bound_contrib[j]);
    // bound_duals の中身を表示 (idx j_in_lb_only_layout)
    // QADLITTL は全 lb 有限・ub 無限なので bound_duals[j] は y_lb[j]
    let bd_lb_idx_for_j = {
        let mut count = 0usize;
        let mut found = None;
        for (k, &(lb_k, _)) in prob.bounds.iter().enumerate() {
            let lb_k: f64 = lb_k;
            if lb_k.is_finite() {
                if k == j { found = Some(count); break; }
                count += 1;
            }
        }
        found
    };
    if let Some(idx) = bd_lb_idx_for_j {
        let val = result.bound_duals.get(idx).copied().unwrap_or(0.0);
        println!("  bound_duals[lb_idx={}] = {:.6e}", idx, val);
    }
    // 最大絶対値の bound_dual と x の slack
    let mut max_y_lb = 0.0_f64;
    let mut max_y_lb_idx = 0usize;
    let mut bd_idx2 = 0usize;
    for (k, &(lb_k, _)) in prob.bounds.iter().enumerate() {
        let lb_k: f64 = lb_k;
        if lb_k.is_finite() {
            let v = result.bound_duals.get(bd_idx2).copied().unwrap_or(0.0);
            if v.abs() > max_y_lb {
                max_y_lb = v.abs();
                max_y_lb_idx = k;
            }
            bd_idx2 += 1;
        }
    }
    let x_at_max = result.solution.get(max_y_lb_idx).copied().unwrap_or(0.0);
    println!("  max y_lb = {:.3e} at var={}, x[{}]={:.6e}, lb=0",
        max_y_lb, max_y_lb_idx, max_y_lb_idx, x_at_max);

    // x の値ごとの分布
    let mut zero_count = 0;
    let mut at_lb_count = 0;
    let mut interior_count = 0;
    for (k, &xk) in result.solution.iter().enumerate() {
        let xk: f64 = xk;
        let lb_k: f64 = prob.bounds[k].0;
        if lb_k.is_finite() {
            let s_k = xk - lb_k;
            if s_k.abs() < 1e-12 { zero_count += 1; }
            else if s_k.abs() < 1e-6 { at_lb_count += 1; }
            else { interior_count += 1; }
        }
    }
    println!("  x distribution (lb-finite vars): exactly_at_lb={} near_lb<1e-6={} interior={}",
        zero_count, at_lb_count, interior_count);

    // 等式 / 不等式 / Eq の数
    let n_eq = prob.constraint_types.iter().filter(|&&ct| ct == ConstraintType::Eq).count();
    let n_le = prob.constraint_types.iter().filter(|&&ct| ct == ConstraintType::Le).count();
    let n_ge = prob.constraint_types.iter().filter(|&&ct| ct == ConstraintType::Ge).count();
    println!();
    println!("constraint mix: Eq={} Le={} Ge={}", n_eq, n_le, n_ge);

    // Q が 0 か
    let q_max = prob.q.values.iter().fold(0.0_f64, |a: f64, &v: &f64| a.max(v.abs()));
    println!("Q nonzeros={} max|Q|={:.3e} (Q=0 if {})",
        prob.q.values.len(), q_max, prob.q.values.iter().all(|&v| v == 0.0));
}
