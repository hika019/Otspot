//! 1000s 級 TIMEOUT/FAIL 問題群の真因分類診断。
//!
//! 各 problem の status / iterations / pf / df / timing / 違反列を観測し
//! class (dual 退化 / cleanup coupling / cycling / 性能不足 / presolve エラー)
//! を heuristic で推定する。全 test `#[ignore]`、個別実行用。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use std::path::Path;

const BOUND_TOL: f64 = 1e-6;

/// 1 問の classification 診断結果を print する。
fn classify(qp_path: &str, eps: f64, timeout_s: f64) -> bool {
    let path = Path::new(qp_path);
    assert!(path.exists(), "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行", qp_path);
    let prob = parse_qps(path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_s);
    let t0 = std::time::Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let solve_elapsed = t0.elapsed().as_secs_f64();

    let n = prob.c.len();
    let m = prob.num_constraints;

    // dfeas 違反列の集計
    let mut viol_at_lb = 0usize;
    let mut viol_at_ub = 0usize;
    let mut max_viol = 0.0_f64;
    let mut max_viol_rel = 0.0_f64;
    let mut zero_c_violators = 0usize;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed { continue; }
        let x = if j < r.solution.len() { r.solution[j] } else { continue; };
        let rc = if j < r.reduced_costs.len() { r.reduced_costs[j] } else { continue; };
        let at_lb = lb.is_finite() && (x - lb).abs() < BOUND_TOL;
        let at_ub = ub.is_finite() && (x - ub).abs() < BOUND_TOL;
        let viol = if at_lb && !at_ub { f64::max(0.0, -rc) }
            else if at_ub && !at_lb { f64::max(0.0, rc) }
            else { 0.0 };
        if viol > 1e-6 {
            if at_lb { viol_at_lb += 1; }
            if at_ub { viol_at_ub += 1; }
            if prob.c[j].abs() < 1e-12 { zero_c_violators += 1; }
            let scale = 1.0 + rc.abs() + prob.c[j].abs();
            max_viol = max_viol.max(viol);
            max_viol_rel = max_viol_rel.max(viol / scale);
        }
    }
    // pfeas 違反 (||Ax - b||_inf)
    let mut max_pf = 0.0_f64;
    if !r.solution.is_empty() {
        let mut ax = vec![0.0_f64; m];
        for j in 0..n.min(r.solution.len()) {
            if let Ok((rows, vals)) = prob.a.get_column(j) {
                for k in 0..rows.len() {
                    ax[rows[k]] += vals[k] * r.solution[j];
                }
            }
        }
        for (i, &ax_i) in ax.iter().enumerate() {
            let viol = match prob.constraint_types[i] {
                otspot::problem::ConstraintType::Eq => (ax_i - prob.b[i]).abs(),
                otspot::problem::ConstraintType::Le => (ax_i - prob.b[i]).max(0.0),
                otspot::problem::ConstraintType::Ge => (prob.b[i] - ax_i).max(0.0),
                _ => 0.0,
            };
            max_pf = max_pf.max(viol);
        }
    }
    let timing = r.timing_breakdown.as_ref();
    let (t_pre, t_sol, t_post) = match timing {
        Some(t) => (t.presolve_us as f64 / 1e6, t.solve_us as f64 / 1e6, t.postsolve_us as f64 / 1e6),
        None => (f64::NAN, f64::NAN, f64::NAN),
    };

    eprintln!("==== {} (n={} m={}) ====", qp_path, n, m);
    eprintln!("  status={:?}  obj={:.4e}  elapsed={:.1}s  iters={}",
        r.status, r.objective, solve_elapsed, r.iterations);
    eprintln!("  timing: presolve={:.2}s solve={:.2}s postsolve={:.2}s",
        t_pre, t_sol, t_post);
    eprintln!("  pf_max={:.2e}  df_max={:.2e}  df_rel_max={:.2e}",
        max_pf, max_viol, max_viol_rel);
    eprintln!("  violators: at_lb={} at_ub={}  zero_c_violators={}",
        viol_at_lb, viol_at_ub, zero_c_violators);

    let class: &str = match &r.status {
        SolveStatus::Optimal => {
            if max_viol_rel < eps {
                "PASS"
            } else if zero_c_violators > 0 && (viol_at_lb + viol_at_ub) > 0 {
                "Class A/B (dual 退化)"
            } else if max_viol_rel > 0.0 {
                "Class A 変種 (dual 違反)"
            } else {
                "Optimal but check"
            }
        }
        SolveStatus::Timeout => "Class D (convergence 不到達)",
        SolveStatus::NumericalError => "Class C/E (Phase I 失敗 or presolve エラー)",
        SolveStatus::Infeasible => "FAIL: Infeasible 誤判定",
        SolveStatus::Unbounded => "FAIL: Unbounded 誤判定",
        SolveStatus::MaxIterations => "Class D (iter 上限到達)",
        SolveStatus::SuboptimalSolution => "Class D (収束 partial)",
        SolveStatus::LocallyOptimal => "Class D (LP では起きるべきでない)",
        SolveStatus::NonConvex(_) => "FAIL: LP に NonConvex 返却 (バグ)",
        _ => "未知 status",
    };
    eprintln!("  CLASS ESTIMATE: {}", class);
    eprintln!();
    matches!(r.status, SolveStatus::Optimal) && max_viol_rel < eps
}

#[test] #[ignore = "diag (~30s)"] fn classify_greenbea() {
    classify("data/lp_problems/greenbea.QPS", 1e-6, 60.0);
}

#[test] #[ignore = "diag (~60s)"] fn classify_d6cube() {
    classify("data/lp_problems/d6cube.QPS", 1e-6, 90.0);
}

#[test] #[ignore = "diag (~120s)"] fn classify_dfl001() {
    classify("data/lp_problems/dfl001.QPS", 1e-6, 200.0);
}

#[test] #[ignore = "diag (~200s)"] fn classify_pds_10() {
    classify("data/lp_problems/pds-10.QPS", 1e-6, 300.0);
}

#[test] #[ignore = "diag (~300s)"] fn classify_cre_b() {
    classify("data/lp_problems/cre-b.QPS", 1e-6, 600.0);
}

#[test] #[ignore = "diag (~600s)"] fn classify_pds_20() {
    classify("data/lp_problems/pds-20.QPS", 1e-6, 1000.0);
}

#[test] #[ignore = "diag (very heavy)"] fn classify_ken_13() {
    classify("data/lp_problems/ken-13.QPS", 1e-6, 600.0);
}

#[test] #[ignore = "diag (very heavy)"] fn classify_ken_18() {
    classify("data/lp_problems/ken-18.QPS", 1e-6, 1000.0);
}

#[test] #[ignore = "diag (~120s)"] fn classify_rail582() {
    classify("data/lp_problems_hard/rail582.QPS", 1e-6, 300.0);
}

#[test] #[ignore = "diag (~120s)"] fn classify_n3700() {
    classify("data/lp_problems_hard/n3700.QPS", 1e-6, 300.0);
}

#[test] #[ignore = "diag (~120s)"] fn classify_sgpf5y6() {
    classify("data/lp_problems_hard/sgpf5y6.QPS", 1e-6, 300.0);
}

#[test] #[ignore = "diag (~120s)"] fn classify_watson_2() {
    classify("data/lp_problems_hard/watson_2.QPS", 1e-6, 300.0);
}
