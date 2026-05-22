//! QP 診断バイナリ: 1 問題を solve_qp_with で解いて、
//! `result.solution`, `result.dual_solution`, `result.bound_duals` から
//! 元空間 KKT 残差を bench と同じ式で再計算し、内訳を表示する。
//!
//! 使い方: `qp_diag <path/to/problem.QPS>`
//! 環境変数:
//!   DIAG_NO_RUIZ=1     — Ruiz scaling 無効化
//!   DIAG_NO_PRESOLVE=1 — presolve 無効化

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::path::PathBuf;
use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::ConstraintType;
use otspot::qp::solve_qp_with;
use otspot::QpProblem;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path/to/file.QPS>", args[0]);
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);

    let prob_box = parse_qps(&path).expect("parse failed");
    let prob: &QpProblem = &prob_box;
    let mut opts = SolverOptions::default();
    let timeout = std::env::var("DIAG_TIMEOUT").ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(30.0);
    opts.timeout_secs = Some(timeout);
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
    if result.solution.len() != prob.num_vars {
        println!(
            "[DIAG] solution unavailable: len={} expected={} dual_len={} bound_duals_len={}",
            result.solution.len(),
            prob.num_vars,
            result.dual_solution.len(),
            result.bound_duals.len(),
        );
        return;
    }
    // primal residual も併記 (T1.3 LISWET 系の診断用)
    if !result.solution.is_empty() && prob.num_constraints > 0 {
        let ax = prob.a.mat_vec_mul(&result.solution).expect("Ax");
        let mut max_pf = 0.0_f64;
        let mut max_ax = 0.0_f64;
        let mut max_b = 0.0_f64;
        for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
            let v = match prob.constraint_types[i] {
                otspot::problem::ConstraintType::Eq => (ax_i - b_i).abs(),
                otspot::problem::ConstraintType::Ge => (b_i - ax_i).max(0.0),
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
                otspot::problem::ConstraintType::Eq => (ax_i - b_i).abs(),
                otspot::problem::ConstraintType::Ge => (b_i - ax_i).max(0.0),
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
        result.solution.iter().fold(0.0_f64, |a: f64, &v: &f64| a.max(v.abs())),
        result.dual_solution.iter().fold(0.0_f64, |a: f64, &v: &f64| a.max(v.abs())),
        result.bound_duals.iter().fold(0.0_f64, |a: f64, &v: &f64| a.max(v.abs())),
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

    // FX/EmptyCol を skip した bench 形式 (成分相対化 = bench の dfr と同式)
    let mut max_r_full = 0.0_f64;
    let mut argmax_full = 0usize;
    let mut max_qx = 0.0_f64;
    let mut max_c = 0.0_f64;
    let mut max_aty = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    let mut n_fx_skipped = 0;
    let mut n_empty_skipped = 0;
    // 成分相対化: bench の compute_dfeas_orig と同式
    let mut dfeas_abs = 0.0_f64;
    let mut dfeas_rel_componentwise = 0.0_f64;
    let mut argmax_componentwise = 0usize;
    // 上位 20 件を表示するためのバッファ
    let mut per_col: Vec<(usize, f64, f64, f64, f64, f64, f64)> = Vec::new(); // (j, r_rel, r_abs, qx, aty, bnd, c)
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
        dfeas_abs = dfeas_abs.max(r);
        let scale_j = 1.0 + qx[j].abs() + aty[j].abs() + bound_contrib[j].abs() + prob.c[j].abs();
        let r_rel = r / scale_j;
        if r_rel > dfeas_rel_componentwise {
            dfeas_rel_componentwise = r_rel;
            argmax_componentwise = j;
        }
        max_qx = max_qx.max(qx[j].abs());
        max_c = max_c.max(prob.c[j].abs());
        max_aty = max_aty.max(aty[j].abs());
        max_bnd = max_bnd.max(bound_contrib[j].abs());
        per_col.push((j, r_rel, r, qx[j], aty[j], bound_contrib[j], prob.c[j]));
    }
    let scale_global = 1.0 + max_qx.max(max_c).max(max_aty).max(max_bnd);
    // 成分相対化でソート
    per_col.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!();
    println!("=== bench-style dfeas (skip FX/EmptyCol) ===");
    println!("max_r_full={:.3e} argmax={} (no skip, n_fx_skip={} n_empty_skip={})",
        max_r_full, argmax_full, n_fx_skipped, n_empty_skipped);
    println!("max_qx={:.3e} max_c={:.3e} max_aty={:.3e} max_bnd={:.3e}",
        max_qx, max_c, max_aty, max_bnd);
    println!("global: scale={:.3e} dfeas_abs={:.3e} dfeas_rel_global={:.3e}",
        scale_global, dfeas_abs, dfeas_abs / scale_global);
    println!("componentwise (bench dfr): dfeas_rel={:.3e} argmax={}",
        dfeas_rel_componentwise, argmax_componentwise);
    println!();
    println!("=== top 20 by component-wise residual ===");
    for k in 0..per_col.len().min(20) {
        let (j, r_rel, r, qxj, atyj, bndj, cj) = per_col[k];
        let scale_j = 1.0 + qxj.abs() + atyj.abs() + bndj.abs() + cj.abs();
        let (lbj, ubj) = prob.bounds[j];
        println!("  j={} r_rel={:.3e} r={:.3e} scale={:.3e} | qx={:.3e} aty={:.3e} bnd={:.3e} c={:.3e} | bounds=[{:.3e},{:.3e}] x={:.3e}",
            j, r_rel, r, scale_j, qxj, atyj, bndj, cj, lbj, ubj, result.solution[j]);
    }

    // worst variable の内訳 (成分相対化 argmax)
    let j = argmax_componentwise;
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
    for k in prob.a.col_ptr[j]..prob.a.col_ptr[j + 1] {
        let row = prob.a.row_ind[k];
        let aij = prob.a.values[k];
        let yi = result.dual_solution.get(row).copied().unwrap_or(0.0);
        let mut row_lhs = 0.0_f64;
        let mut row_nnz = 0usize;
        let mut row_terms = Vec::new();
        for col in 0..prob.num_vars {
            for kk in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
                if prob.a.row_ind[kk] == row {
                    let coeff = prob.a.values[kk];
                    let xcol = result.solution.get(col).copied().unwrap_or(0.0);
                    row_lhs += coeff * xcol;
                    row_nnz += 1;
                    if row_terms.len() < 6 {
                        row_terms.push(format!("col{}:{:.3e}*{:.3e}", col, coeff, xcol));
                    }
                }
            }
        }
        println!(
            "  A[row={}, j]={:.6e} ct={:?} y[row]={:.6e} contrib={:.6e} row_lhs={:.6e} b={:.6e} row_nnz={} terms=[{}]",
            row,
            aij,
            prob.constraint_types[row],
            yi,
            aij * yi,
            row_lhs,
            prob.b[row],
            row_nnz,
            row_terms.join(", ")
        );
    }
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
