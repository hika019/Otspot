//! 診断: afiro で y (dual_solution) の偏りを観測する
//!
//! 事実: e61f27b 以降 afiro (32x27) で DFEAS_FAIL。a1d42b1 では PASS。
//! presolve OFF と ON で y を比較し、どちらが偏るかを切り分ける。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::{ConstraintType, LpProblem};
use solver::qp::solve_qp_with;
use solver::{solve_with, QpProblem};
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

/// presolve ON で解いた afiro が LP の dual feasibility (rc>=0) と
/// KKT 残差 ≈ 0 の両方を満たすことを要求する TDD テスト。
///
/// 現状 (旧方式 rc): rc>=0 PASS、KKT FAIL (削除行 12 の y=0 が KKT 破る)
/// 新方式 rc (revert 後): rc>=0 FAIL (-0.325)、KKT PASS (機械的に 0)
/// 真の修正 = 各 transform で y を KKT 整合に復元 → 両方 PASS
const KKT_TOL: f64 = 1e-6;
const RC_NONNEG_TOL: f64 = 1e-6;

/// LP dual feasibility (bound 考慮版):
/// x[j] at lb → rc[j] >= 0
/// x[j] at ub → rc[j] <= 0
/// interior   → rc[j] ≈ 0
/// free / fixed → 任意
const BOUND_TOL: f64 = 1e-6;

fn check_lp_dual_kkt(qp_path: &str) {
    let path = Path::new(qp_path);
    if !path.exists() {
        eprintln!("[SKIP] {} not found", qp_path);
        return;
    }
    let qp = parse_qps(path).expect("parse failed");

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(30.0);
    let r = solve_qp_with(&qp, &opts);

    let n = qp.c.len();
    for j in 0..n.min(r.reduced_costs.len()).min(r.solution.len()) {
        let x = r.solution[j];
        let (lb, ub) = qp.bounds[j];
        let rc = r.reduced_costs[j];
        let at_lb = lb.is_finite() && (x - lb).abs() < BOUND_TOL;
        let at_ub = ub.is_finite() && (x - ub).abs() < BOUND_TOL;
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed { continue; }
        if at_lb && !at_ub {
            assert!(rc >= -RC_NONNEG_TOL,
                "[{}] x[{}]={} at lb={} なのに rc={} < -{}",
                qp_path, j, x, lb, rc, RC_NONNEG_TOL);
        } else if at_ub && !at_lb {
            assert!(rc <= RC_NONNEG_TOL,
                "[{}] x[{}]={} at ub={} なのに rc={} > {}",
                qp_path, j, x, ub, rc, RC_NONNEG_TOL);
        }
        // interior は rc ≈ 0 を厳格に要求しない (退化解で簡単に壊れる)
    }

    let (_diff, kkt_max) = kkt_residual(&qp, &r.dual_solution, &r.reduced_costs);
    assert!(
        kkt_max < KKT_TOL,
        "[{}] KKT 残差 |c - A^T y - rc|_∞ = {} >= {}",
        qp_path, kkt_max, KKT_TOL
    );
}

#[test]
fn test_afiro_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/afiro.QPS");
}

#[test]
fn test_blend_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/blend.QPS");
}

#[test]
fn test_adlittle_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/adlittle.QPS");
}

#[test]
fn test_agg_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/agg.QPS");
}

#[test]
fn test_sc50a_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/sc50a.QPS");
}

#[test]
fn test_kb2_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/kb2.QPS");
}

#[test]
fn test_recipe_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/recipe.QPS");
}

#[test]
fn test_scorpion_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/scorpion.QPS");
}

#[test]
fn test_share1b_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/share1b.QPS");
}

#[test]
fn test_brandy_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/brandy.QPS");
}

#[test]
fn test_scfxm1_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/scfxm1.QPS");
}

// QP 調査結果に基づく Tier 1 (Large Coeff Scaling / Implied Bounds Guards) 対象問題。
// これらは現状で fail or TIMEOUT。実装後 GREEN になるべき問題。
#[test]
fn test_etamacro_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/etamacro.QPS");
}

#[test]
fn test_bandm_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/bandm.QPS");
}

#[test]
fn test_beaconfd_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/beaconfd.QPS");
}

/// 大規模 LP の timing_breakdown を出力 (ignored、 cargo nextest run -- --ignored で実行)。
/// 「どこに時間が掛かっているか」事実観測用。
#[test]
#[ignore = "diag (heavy: 12 LP × 15s timeout、要 data/lp_problems/)、timing 観測のみ"]
fn diag_large_lp_timing_breakdown() {
    let problems = [
        "data/lp_problems/cre-b.QPS",
        "data/lp_problems/cre-d.QPS",
        "data/lp_problems/dfl001.QPS",
        "data/lp_problems/ken-11.QPS",
        "data/lp_problems/d2q06c.QPS",
        "data/lp_problems/d6cube.QPS",
        "data/lp_problems/greenbea.QPS",
        "data/lp_problems/pilot.QPS",
        "data/lp_problems/maros-r7.QPS",
        "data/lp_problems/pilot87.QPS",
        "data/lp_problems/perold.QPS",
        "data/lp_problems/pds-20.QPS",
    ];
    println!("{:<24} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "problem", "presolve", "solve", "postsolve", "total_ms", "status");
    for p in &problems {
        let path = Path::new(p);
        if !path.exists() { continue; }
        let qp = parse_qps(path).expect("parse");
        let lp = make_lp(&qp);
        let mut opts = SolverOptions::default();
        opts.presolve = true;
        opts.timeout_secs = Some(15.0);
        let r = solve_with(&lp, &opts);
        let name = path.file_stem().unwrap().to_string_lossy();
        if let Some(tb) = r.timing_breakdown {
            let total_ms = (tb.presolve_us + tb.solve_us + tb.postsolve_us) as f64 / 1000.0;
            println!("{:<24} {:>10} {:>10} {:>10} {:>10.1} {:>12?}",
                name, tb.presolve_us, tb.solve_us, tb.postsolve_us, total_ms, r.status);
        } else {
            println!("{:<24} {:>10} {:>10} {:>10} {:>10} {:>12?}",
                name, "-", "-", "-", "-", r.status);
        }
    }
}

/// TimingBreakdown が presolve ON で填まること、 各 phase Duration > 0 を要求。
#[test]
fn test_timing_breakdown_recorded_for_presolved_lp() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    if !path.exists() { eprintln!("[SKIP]"); return; }
    let qp = parse_qps(path).expect("parse");
    let lp = make_lp(&qp);
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(10.0);
    let r = solve_with(&lp, &opts);
    assert!(r.timing_breakdown.is_some(), "timing_breakdown must be Some when presolve reduced");
    let tb = r.timing_breakdown.unwrap();
    // presolve / solve は実時間で μs 単位、 0 は不自然
    assert!(tb.presolve_us > 0, "presolve_us={}", tb.presolve_us);
    assert!(tb.solve_us > 0, "solve_us={}", tb.solve_us);
    // postsolve は cleanup LP で時間掛かる
    assert!(tb.postsolve_us > 0, "postsolve_us={}", tb.postsolve_us);
}

/// scorpion の y を presolve OFF / ON で比較し、cleanup LP の必要性を観察
#[test]
fn diag_scorpion_y_off_vs_on() {
    let path = Path::new("data/lp_problems/scorpion.QPS");
    if !path.exists() { eprintln!("[SKIP]"); return; }
    let qp = parse_qps(path).expect("parse");
    let lp = make_lp(&qp);
    let n = lp.num_vars;
    let m = lp.num_constraints;
    println!("scorpion: n={} m={}", n, m);

    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    opts_off.timeout_secs = Some(30.0);
    let r_off = solve_qp_with(&qp, &opts_off);

    let mut opts_on = SolverOptions::default();
    opts_on.presolve = true;
    opts_on.timeout_secs = Some(30.0);
    let r_on = solve_qp_with(&qp, &opts_on);

    println!("status off={:?} on={:?}", r_off.status, r_on.status);
    println!("pf off={:e} on={:e}", max_pf(&lp, &r_off.solution), max_pf(&lp, &r_on.solution));
    let max_y_off = r_off.dual_solution.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let max_y_on = r_on.dual_solution.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    println!("max|y| off={:e} on={:e}", max_y_off, max_y_on);

    let df_off = r_off.reduced_costs.iter().fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
    let df_on = r_on.reduced_costs.iter().fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
    println!("df (max(0,-rc)) off={:e} on={:e}", df_off, df_on);

    let mut max_y_diff = 0.0f64;
    let mut argmax_i = 0usize;
    for i in 0..m {
        let d = (r_off.dual_solution[i] - r_on.dual_solution[i]).abs();
        if d > max_y_diff { max_y_diff = d; argmax_i = i; }
    }
    println!("max|y_off - y_on| = {:e} at i={} (y_off={} y_on={})",
        max_y_diff, argmax_i, r_off.dual_solution[argmax_i], r_on.dual_solution[argmax_i]);

    let mut min_rc_on = f64::INFINITY;
    let mut argmin_j = 0usize;
    for j in 0..n {
        if r_on.reduced_costs[j] < min_rc_on { min_rc_on = r_on.reduced_costs[j]; argmin_j = j; }
    }
    println!("min rc on = {} at j={} (x[j]={}, bounds={:?})",
        min_rc_on, argmin_j, r_on.solution[argmin_j], qp.bounds[argmin_j]);
}
