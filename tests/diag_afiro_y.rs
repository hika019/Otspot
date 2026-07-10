//! 診断: afiro で y (dual_solution) の偏りを観測する
//!
//! 事実: ある時点以降 afiro (32x27) で DFEAS_FAIL。その直前の commit では PASS
//! (bisect 元の commit hash は squash/rebase で追跡不能)。
//! presolve OFF と ON で y を比較し、どちらが偏るかを切り分ける。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem};
use otspot::qp::solve_qp_with;
use otspot::{solve_with, QpProblem};
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
    for (j, &x_j) in x.iter().enumerate().take(n) {
        if let Ok((rows, vals)) = lp.a.get_column(j) {
            for k in 0..rows.len() {
                let row = rows[k];
                ax[row] += vals[k] * x_j;
            }
        }
    }
    let mut v_max = 0.0f64;
    for (i, &ax_i) in ax.iter().enumerate() {
        let v = match lp.constraint_types[i] {
            ConstraintType::Eq => (ax_i - lp.b[i]).abs(),
            ConstraintType::Le => (ax_i - lp.b[i]).max(0.0),
            ConstraintType::Ge => (lp.b[i] - ax_i).max(0.0),
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
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
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
    println!(
        "pf     off={:e} on={:e}",
        max_pf(&lp, &r_off.solution),
        max_pf(&lp, &r_on.solution)
    );

    let max_y_off = r_off
        .dual_solution
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs()));
    let max_y_on = r_on
        .dual_solution
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs()));
    println!("max|y| off={:e} on={:e}", max_y_off, max_y_on);

    let (_diff_off, kkt_off) = kkt_residual(&qp, &r_off.dual_solution, &r_off.reduced_costs);
    let (_diff_on, kkt_on) = kkt_residual(&qp, &r_on.dual_solution, &r_on.reduced_costs);
    println!(
        "KKT (c - A^T y - rc) max abs:  off={:e} on={:e}",
        kkt_off, kkt_on
    );

    // bench DFEAS_FAIL 判定相当: df = max(0, -rc[j]) (LP 経路, bound_duals empty)
    let df_off = r_off
        .reduced_costs
        .iter()
        .fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
    let df_on = r_on
        .reduced_costs
        .iter()
        .fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
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
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        qp_path
    );
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
        if fixed {
            continue;
        }
        if at_lb && !at_ub {
            assert!(
                rc >= -RC_NONNEG_TOL,
                "[{}] x[{}]={} at lb={} なのに rc={} < -{}",
                qp_path,
                j,
                x,
                lb,
                rc,
                RC_NONNEG_TOL
            );
        } else if at_ub && !at_lb {
            assert!(
                rc <= RC_NONNEG_TOL,
                "[{}] x[{}]={} at ub={} なのに rc={} > {}",
                qp_path,
                j,
                x,
                ub,
                rc,
                RC_NONNEG_TOL
            );
        }
        // interior は rc ≈ 0 を厳格に要求しない (退化解で簡単に壊れる)
    }

    let (_diff, kkt_max) = kkt_residual(&qp, &r.dual_solution, &r.reduced_costs);
    assert!(
        kkt_max < KKT_TOL,
        "[{}] KKT 残差 |c - A^T y - rc|_∞ = {} >= {}",
        qp_path,
        kkt_max,
        KKT_TOL
    );
}

#[test]
fn test_afiro_presolve_on_dual_feasibility_and_kkt() {
    check_lp_dual_kkt("data/lp_problems/afiro.QPS");
}

/// Safety sentinel: the QP cert EmptyCol mask (eliminated_cols at
/// attempt.rs) must never hide an AFIRO stationarity violation.
///
/// The `kkt_residual_rel` skip is narrow: it fires only for columns that are
/// eliminated AND A-empty AND Q-empty (LP-style fully isolated). AFIRO (an LP →
/// Q is all-zero, so every column is Q-empty) has **zero** columns that are also
/// A-empty: every variable participates in a constraint. Hence the mask provably
/// skips no AFIRO column, and any genuine false-Optimal stays exposed to
/// prove_optimal / guard_qp_optimal. This pins that structural premise so a future
/// change to the skip condition that started skipping A-non-empty columns would
/// fail here. Combined with the QP-path solve, AFIRO must not be NumericalError.
#[test]
fn test_afiro_qp_cert_mask_skips_no_column() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
    let qp = parse_qps(path).expect("parse afiro");
    let n = qp.num_vars;

    let mut struct_empty = 0usize;
    for j in 0..n {
        let a_empty =
            qp.a.get_column(j)
                .map(|(r, _)| r.is_empty())
                .unwrap_or(true);
        let q_empty =
            qp.q.get_column(j)
                .map(|(r, _)| r.is_empty())
                .unwrap_or(true);
        if a_empty && q_empty {
            struct_empty += 1;
        }
    }
    assert_eq!(
        struct_empty, 0,
        "AFIRO must have 0 (A-empty AND Q-empty) columns so the narrow EmptyCol \
         mask cannot skip any AFIRO column (got {struct_empty})",
    );

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(30.0);
    let r = solve_qp_with(&qp, &opts);
    assert_ne!(
        r.status,
        otspot::problem::SolveStatus::NumericalError,
        "AFIRO QP cert path must not catastrophically demote (status={:?})",
        r.status,
    );
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

/// TimingBreakdown が presolve ON で填まること、 各 phase Duration > 0 を要求。
#[test]
fn test_timing_breakdown_recorded_for_presolved_lp() {
    let path = Path::new("data/lp_problems/afiro.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
    let qp = parse_qps(path).expect("parse");
    let lp = make_lp(&qp);
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(10.0);
    let r = solve_with(&lp, &opts);
    assert!(
        r.timing_breakdown.is_some(),
        "timing_breakdown must be Some when presolve reduced"
    );
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
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        path.display()
    );
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
    println!(
        "pf off={:e} on={:e}",
        max_pf(&lp, &r_off.solution),
        max_pf(&lp, &r_on.solution)
    );
    let max_y_off = r_off
        .dual_solution
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs()));
    let max_y_on = r_on
        .dual_solution
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs()));
    println!("max|y| off={:e} on={:e}", max_y_off, max_y_on);

    let df_off = r_off
        .reduced_costs
        .iter()
        .fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
    let df_on = r_on
        .reduced_costs
        .iter()
        .fold(0.0f64, |a, &rc| a.max(f64::max(0.0, -rc)));
    println!("df (max(0,-rc)) off={:e} on={:e}", df_off, df_on);

    // dual_solution が空だと下の loop で OOB するため、先に Optimal を assert。
    assert_eq!(
        r_off.status,
        otspot::problem::SolveStatus::Optimal,
        "scorpion presolve=OFF must be Optimal (got {:?}); regression in simplex without presolve",
        r_off.status
    );
    assert_eq!(
        r_on.status,
        otspot::problem::SolveStatus::Optimal,
        "scorpion presolve=ON must be Optimal (got {:?}); regression in postsolve/cleanup LP",
        r_on.status
    );
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

    let mut min_rc_on = f64::INFINITY;
    let mut argmin_j = 0usize;
    for j in 0..n {
        if r_on.reduced_costs[j] < min_rc_on {
            min_rc_on = r_on.reduced_costs[j];
            argmin_j = j;
        }
    }
    println!(
        "min rc on = {} at j={} (x[j]={}, bounds={:?})",
        min_rc_on, argmin_j, r_on.solution[argmin_j], qp.bounds[argmin_j]
    );
}
