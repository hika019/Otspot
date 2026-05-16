//! 診断: afiro で y (dual_solution) の偏りを観測する
//!
//! 事実: e61f27b 以降 afiro (32x27) で DFEAS_FAIL。a1d42b1 では PASS。
//! presolve OFF と ON で y を比較し、どちらが偏るかを切り分ける。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::{ConstraintType, LpProblem};
use solver::qp::solve_qp_with;
use solver::QpProblem;
use std::path::Path;

fn make_lp(qp: &QpProblem) -> LpProblem {
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .unwrap()
}

fn max_pf(lp: &LpProblem, x: &[f64]) -> f64 {
    let m = lp.num_constraints;
    let n = lp.num_vars.min(x.len());
    let mut ax = vec![0.0f64; m];
    for j in 0..n {
        if let Ok((rows, vals)) = lp.a.get_column(j) {
            for k in 0..rows.len() {
                let row = rows[k];
                ax[row] += vals[k] * x[j];
            }
        }
    }
    let mut v_max = 0.0f64;
    for i in 0..m {
        let v = match lp.constraint_types[i] {
            ConstraintType::Eq => (ax[i] - lp.b[i]).abs(),
            ConstraintType::Le => (ax[i] - lp.b[i]).max(0.0),
            ConstraintType::Ge => (lp.b[i] - ax[i]).max(0.0),
            _ => 0.0,
        };
        if v > v_max {
            v_max = v;
        }
    }
    v_max
}

fn kkt_residual(qp: &QpProblem, y: &[f64], rc: &[f64]) -> (Vec<f64>, f64) {
    let n = qp.c.len();
    let mut diff = vec![0.0f64; n];
    let mut max_diff = 0.0f64;
    for j in 0..n {
        let mut ct_y = qp.c[j];
        if let Ok((rows, vals)) = qp.a.get_column(j) {
            for k in 0..rows.len() {
                let row = rows[k];
                ct_y -= vals[k] * y[row];
            }
        }
        let d = ct_y - rc[j];
        diff[j] = d;
        if d.abs() > max_diff {
            max_diff = d.abs();
        }
    }
    (diff, max_diff)
}

#[test]
fn diag_afiro_y_presolve_off_vs_on() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    if !path.exists() {
        eprintln!("[SKIP] afiro.QPS not found at {}", path.display());
        return;
    }
    let qp = parse_qps(path).expect("parse afiro");
    let lp = make_lp(&qp);
    let n = lp.num_vars;
    let m = lp.num_constraints;
    println!("afiro: n={} m={}", n, m);

    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    opts_off.timeout_secs = Some(30.0);
    let r_off = solve_qp_with(&qp, &opts_off);

    let mut opts_on = SolverOptions::default();
    opts_on.presolve = true;
    opts_on.timeout_secs = Some(30.0);
    let r_on = solve_qp_with(&qp, &opts_on);

    println!("status off={:?} on={:?}", r_off.status, r_on.status);
    println!("obj    off={:e} on={:e}", r_off.objective, r_on.objective);
    println!("pf     off={:e} on={:e}", max_pf(&lp, &r_off.solution), max_pf(&lp, &r_on.solution));

    let max_y_off = r_off.dual_solution.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let max_y_on = r_on.dual_solution.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    println!("max|y| off={:e} on={:e}", max_y_off, max_y_on);

    let (_diff_off, kkt_off) = kkt_residual(&qp, &r_off.dual_solution, &r_off.reduced_costs);
    let (_diff_on, kkt_on) = kkt_residual(&qp, &r_on.dual_solution, &r_on.reduced_costs);
    println!("KKT (c - A^T y - rc) max abs:  off={:e} on={:e}", kkt_off, kkt_on);

    // bench DFEAS_FAIL 判定相当: df = max(0, -rc[j]) (LP 経路, bound_duals empty)
    let df_off = r_off.reduced_costs.iter().fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
    let df_on = r_on.reduced_costs.iter().fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
    println!("df (bench DFEAS judge): off={:e} on={:e}", df_off, df_on);

    // slack[12] 観測 (RedundantConstraint で削除された行)
    let mut ax_off = vec![0.0f64; m];
    let mut ax_on = vec![0.0f64; m];
    for j in 0..n {
        if let Ok((rows, vals)) = lp.a.get_column(j) {
            for k in 0..rows.len() {
                let row = rows[k];
                ax_off[row] += vals[k] * r_off.solution[j];
                ax_on[row] += vals[k] * r_on.solution[j];
            }
        }
    }
    for &i in &[12usize, 0, 1, 2, 3, 4] {
        if i < m {
            let slack_off = lp.b[i] - ax_off[i];
            let slack_on = lp.b[i] - ax_on[i];
            println!("row {:3}: type={:?} b={} ax_off={} slack_off={:e} y_off={} | ax_on={} slack_on={:e} y_on={}",
                i, lp.constraint_types[i], lp.b[i], ax_off[i], slack_off, r_off.dual_solution[i],
                ax_on[i], slack_on, r_on.dual_solution[i]);
        }
    }

    println!("y[off][0..10] = {:?}", &r_off.dual_solution[..10.min(m)]);
    println!("y[on ][0..10] = {:?}", &r_on.dual_solution[..10.min(m)]);
    println!("rc[off][0..10] = {:?}", &r_off.reduced_costs[..10.min(n)]);
    println!("rc[on ][0..10] = {:?}", &r_on.reduced_costs[..10.min(n)]);

    // y の差分
    let mut max_y_diff = 0.0f64;
    let mut argmax_i = 0usize;
    for i in 0..m {
        let d = (r_off.dual_solution[i] - r_on.dual_solution[i]).abs();
        if d > max_y_diff {
            max_y_diff = d;
            argmax_i = i;
        }
    }
    println!(
        "max|y_off - y_on| = {:e} at i={} (y_off={} y_on={})",
        max_y_diff, argmax_i, r_off.dual_solution[argmax_i], r_on.dual_solution[argmax_i]
    );
}
