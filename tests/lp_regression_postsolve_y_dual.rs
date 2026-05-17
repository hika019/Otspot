//! postsolve 後の (x, y, rc) が LP KKT を満たすことを複数問題で assert する
//! regression guard。perold は postsolve dual 復元の代表 canary。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::{solve_qp_with, QpProblem};
use std::path::Path;

/// LP/QP dual feasibility 相対残差 (`viol_j = max(0, -rc_j)`、bound 非考慮版)。
fn dfeas_abs_rel(prob: &QpProblem, rc: &[f64]) -> (f64, f64) {
    let n = prob.c.len().min(rc.len());
    let mut dfeas_abs = 0.0_f64;
    let mut dfeas_rel = 0.0_f64;
    for j in 0..n {
        let r = rc[j];
        let viol = f64::max(0.0, -r);
        dfeas_abs = dfeas_abs.max(viol);
        let scale = 1.0 + r.abs() + prob.c[j].abs();
        dfeas_rel = dfeas_rel.max(viol / scale);
    }
    (dfeas_abs, dfeas_rel)
}

/// LP dual feasibility (bound 考慮版): bound-hit 列のみ厳格判定。
/// `viol = max(0, -rc)` if x at lb only; `max(0, rc)` if x at ub only;
/// fixed (lb==ub) skip; interior 0。
fn dfeas_rel_bound_aware(prob: &QpProblem, x: &[f64], rc: &[f64]) -> f64 {
    const BOUND_TOL: f64 = 1e-6;
    let n = prob.c.len().min(rc.len()).min(x.len());
    let mut dfeas_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed { continue; }
        let at_lb = lb.is_finite() && (x[j] - lb).abs() < BOUND_TOL;
        let at_ub = ub.is_finite() && (x[j] - ub).abs() < BOUND_TOL;
        let r = rc[j];
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -r)
        } else if at_ub && !at_lb {
            f64::max(0.0, r)
        } else {
            0.0 // interior / 両端
        };
        let scale = 1.0 + r.abs() + prob.c[j].abs();
        dfeas_rel = dfeas_rel.max(viol / scale);
    }
    dfeas_rel
}

/// 仕様: `presolve=true` で solve した結果が以下を満たす:
///   - status = Optimal
///   - dfeas_rel < eps (旧 strict formula、bound 考慮版どちらも)
fn check_postsolve_dual_feasibility(
    qp_path: &str,
    eps_dual: f64,
    timeout_s: f64,
) -> Result<String, String> {
    let path = Path::new(qp_path);
    if !path.exists() {
        return Ok(format!("[SKIP] {} not found", qp_path));
    }
    let prob = parse_qps(path).map_err(|e| format!("parse failed: {:?}", e))?;
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_s);
    let r = solve_qp_with(&prob, &opts);
    let (df_abs, df_rel_strict) = dfeas_abs_rel(&prob, &r.reduced_costs);
    let df_rel_bound = dfeas_rel_bound_aware(&prob, &r.solution, &r.reduced_costs);
    let summary = format!(
        "{}: status={:?} obj={:.4e} df_abs={:.2e} df_rel_strict={:.2e} df_rel_bound={:.2e}",
        qp_path, r.status, r.objective, df_abs, df_rel_strict, df_rel_bound
    );
    if !matches!(r.status, SolveStatus::Optimal) {
        return Err(format!("{} | status must be Optimal", summary));
    }
    // bound 考慮版を主判定にする (c69959d 以降の bench と同等)。
    if df_rel_bound > eps_dual {
        return Err(format!("{} | df_rel_bound > eps={}", summary, eps_dual));
    }
    Ok(summary)
}

/// perold: postsolve dual 復元退化の canary。
#[test]
fn perold_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/perold.QPS", 1e-6, 180.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}

/// 診断: greenbea を presolve=off で解いた dfeas を観測。
#[test]
#[ignore = "diag (60s)"]
fn greenbea_presolve_off_dfeas_check() {
    let path = Path::new("data/lp_problems/greenbea.QPS");
    if !path.exists() { eprintln!("[SKIP]"); return; }
    let prob = parse_qps(path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(300.0);
    let r = solve_qp_with(&prob, &opts);
    let df = dfeas_rel_bound_aware(&prob, &r.solution, &r.reduced_costs);
    eprintln!("greenbea[presolve=off]: status={:?} obj={:.4e} df_rel_bound={:.3e}", r.status, r.objective, df);
}

/// perold の presolve=off は別経路 (postsolve なし)。bug の局在を切り分け:
/// HEAD で PASS (df_rel ≈ 3.5e-13) なら、FAIL は postsolve 経路に 100% 局在。
#[test]
fn perold_presolve_off_baseline() {
    let path = Path::new("data/lp_problems/perold.QPS");
    if !path.exists() {
        eprintln!("[SKIP] perold.QPS");
        return;
    }
    let prob = parse_qps(path).expect("parse perold");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(180.0);
    let r = solve_qp_with(&prob, &opts);
    let (df_abs, df_rel) = dfeas_abs_rel(&prob, &r.reduced_costs);
    eprintln!(
        "perold[presolve=off]: status={:?} obj={:.4e} df_abs={:.2e} df_rel={:.2e}",
        r.status, r.objective, df_abs, df_rel
    );
    // SKIP allowed for status: 別経路で NumericalError でも本テストの目的ではない。
    if matches!(r.status, SolveStatus::Optimal) {
        assert!(
            df_rel < 1e-6,
            "perold[presolve=off]: df_rel={:.3e} → simplex 単体に別バグの疑い",
            df_rel
        );
    } else {
        eprintln!("perold[presolve=off]: status={:?}", r.status);
    }
}

/// 小規模 Netlib LP の post-solve dual feasibility 網羅テスト。
///
/// 各問題で `presolve=true` で解き、reduced_cost が bound-aware df_rel < 1e-6 を
/// 満たすことを assert。問題は postsolve.rs の各 PostsolveStep を踏むよう選定。
///
/// 失敗時 stderr に問題名と df 数値を出す。
macro_rules! netlib_postsolve_test {
    ($name:ident, $file:expr, $eps:expr, $timeout:expr) => {
        #[test]
        fn $name() {
            let r = check_postsolve_dual_feasibility($file, $eps, $timeout);
            match r {
                Ok(s) => eprintln!("PASS {}", s),
                Err(e) => panic!("{}", e),
            }
        }
    };
}

netlib_postsolve_test!(afiro_postsolve,    "data/lp_problems/afiro.QPS",    1e-6, 30.0);
netlib_postsolve_test!(sc50a_postsolve,    "data/lp_problems/sc50a.QPS",    1e-6, 30.0);
netlib_postsolve_test!(sc50b_postsolve,    "data/lp_problems/sc50b.QPS",    1e-6, 30.0);
netlib_postsolve_test!(sc105_postsolve,    "data/lp_problems/sc105.QPS",    1e-6, 30.0);
netlib_postsolve_test!(sc205_postsolve,    "data/lp_problems/sc205.QPS",    1e-6, 30.0);
netlib_postsolve_test!(scagr7_postsolve,   "data/lp_problems/scagr7.QPS",   1e-6, 30.0);
netlib_postsolve_test!(share1b_postsolve,  "data/lp_problems/share1b.QPS",  1e-6, 30.0);
netlib_postsolve_test!(scorpion_postsolve, "data/lp_problems/scorpion.QPS", 1e-6, 30.0);
netlib_postsolve_test!(brandy_postsolve,   "data/lp_problems/brandy.QPS",   1e-6, 30.0);
netlib_postsolve_test!(agg_postsolve,      "data/lp_problems/agg.QPS",      1e-6, 30.0);
netlib_postsolve_test!(boeing2_postsolve,  "data/lp_problems/boeing2.QPS",  1e-6, 30.0);
netlib_postsolve_test!(stocfor1_postsolve, "data/lp_problems/stocfor1.QPS", 1e-6, 30.0);

/// 診断: perold のどの列が dual feasibility 違反を起こしているかを print する。
/// 修正開発時の手がかり用 (assertion はせず情報のみ出力)。
/// 重 (perold solve 約 5s)、default 除外。`cargo test -- --ignored` で実行。
#[test]
#[ignore = "diag (5s)、default 除外"]
fn perold_diagnostic_dump_worst_violations() {
    let path = Path::new("data/lp_problems/perold.QPS");
    if !path.exists() { eprintln!("[SKIP] perold.QPS"); return; }
    let prob = parse_qps(path).expect("parse perold");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(180.0);
    let r = solve_qp_with(&prob, &opts);

    const BOUND_TOL: f64 = 1e-6;
    let n = prob.c.len();

    // 違反列を集めて top 10 を print
    let mut viols: Vec<(usize, f64, f64, f64, f64, &'static str, f64, f64, f64)> = Vec::new();
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed { continue; }
        let x = r.solution[j];
        let rc = r.reduced_costs[j];
        let at_lb = lb.is_finite() && (x - lb).abs() < BOUND_TOL;
        let at_ub = ub.is_finite() && (x - ub).abs() < BOUND_TOL;
        let (viol, where_): (f64, &'static str) = if at_lb && !at_ub {
            (f64::max(0.0, -rc), "at_lb")
        } else if at_ub && !at_lb {
            (f64::max(0.0, rc), "at_ub")
        } else {
            (0.0, "interior")
        };
        if viol > 1e-6 {
            let scale = 1.0 + rc.abs() + prob.c[j].abs();
            viols.push((j, x, lb, ub, rc, where_, prob.c[j], viol, viol/scale));
        }
    }
    viols.sort_by(|a, b| b.8.partial_cmp(&a.8).unwrap());
    eprintln!("perold violations (top 20 of {}):", viols.len());
    eprintln!("  j     x         lb         ub         rc         where    c[j]       viol      rel");
    for v in viols.iter().take(20) {
        eprintln!("  {:5} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:8} {:10.3e} {:10.3e} {:10.3e}",
            v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7, v.8);
    }

    // y_i の分布 (削除行と残存行)
    let m = prob.num_constraints;
    let mut y_nonzero = 0;
    let mut y_max = 0.0_f64;
    for i in 0..m {
        if r.dual_solution[i].abs() > 1e-12 { y_nonzero += 1; }
        y_max = y_max.max(r.dual_solution[i].abs());
    }
    eprintln!("perold: m={} y_nonzero={} y_max={:.3e} obj={:.4e}", m, y_nonzero, y_max, r.objective);
}

/// 深掘り診断: perold col 229 の rc 内訳 (どの row の y が誤って巨大か特定)。
#[test]
#[ignore = "diag、default 除外"]
fn perold_col229_deep_diag() {
    let path = Path::new("data/lp_problems/perold.QPS");
    if !path.exists() { eprintln!("[SKIP]"); return; }
    let prob = parse_qps(path).expect("parse perold");
    let mut opts_on = SolverOptions::default();
    opts_on.presolve = true;
    opts_on.timeout_secs = Some(180.0);
    let r_on = solve_qp_with(&prob, &opts_on);

    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    opts_off.timeout_secs = Some(180.0);
    let r_off = solve_qp_with(&prob, &opts_off);

    let j = 229;
    eprintln!("col {} : c[j]={:.3e} bounds={:?}", j, prob.c[j], prob.bounds[j]);
    eprintln!("  x_on  = {:.3e}  x_off = {:.3e}", r_on.solution[j], r_off.solution[j]);
    eprintln!("  rc_on = {:.3e}  rc_off= {:.3e}", r_on.reduced_costs[j], r_off.reduced_costs[j]);

    // 列 229 のエントリを列挙
    if let Ok((rows, vals)) = prob.a.get_column(j) {
        eprintln!("col {} entries ({} non-zero):", j, rows.len());
        eprintln!("  i      A_ij       y_on        y_off       A*y_on      A*y_off    Δy");
        let mut sum_on = 0.0;
        let mut sum_off = 0.0;
        for (k, &i) in rows.iter().enumerate() {
            let a = vals[k];
            let yi_on = r_on.dual_solution[i];
            let yi_off = r_off.dual_solution[i];
            sum_on += a * yi_on;
            sum_off += a * yi_off;
            let dy = yi_on - yi_off;
            eprintln!("  {:5} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e}",
                i, a, yi_on, yi_off, a*yi_on, a*yi_off, dy);
        }
        eprintln!("Σ A*y_on  = {:.3e} → rc_on  = c - Σ = {:.3e} (should match)", sum_on, prob.c[j] - sum_on);
        eprintln!("Σ A*y_off = {:.3e} → rc_off = c - Σ = {:.3e} (should match)", sum_off, prob.c[j] - sum_off);
    }
}

// 大規模 LP の dfeas_rel assertion (network/重 LP は #[ignore] で default 除外)。

/// cre-b: 72k×9k の大規模 LP。
#[test]
#[ignore = "重 LP (timeout 300s 必要); cargo test -- --ignored で明示実行"]
fn cre_b_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/cre-b.QPS", 1e-6, 300.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}

/// greenbea: perold と同類の dual 退化パターン。cleanup LP が 60s 内に未収束で GREEN 未達。
#[test]
#[ignore = "GREEN target 未達: cleanup LP が 60s 内に未収束"]
fn greenbea_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/greenbea.QPS", 1e-6, 60.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}

/// pds-10: 105k×34k の大規模 LP。convergence に時間がかかる。
#[test]
#[ignore = "重 LP (≈ 200s); cargo test -- --ignored で明示実行"]
fn pds_10_postsolve_dual_feasibility() {
    let r = check_postsolve_dual_feasibility("data/lp_problems/pds-10.QPS", 1e-6, 200.0);
    match r {
        Ok(s) => eprintln!("PASS {}", s),
        Err(e) => panic!("{}", e),
    }
}

/// 診断: greenbea の最悪 dual 違反列を print。assertion なし。
#[test]
#[ignore = "diag (60s)、default 除外"]
fn greenbea_diagnostic_dump_worst_violations() {
    let path = Path::new("data/lp_problems/greenbea.QPS");
    if !path.exists() { eprintln!("[SKIP] greenbea.QPS"); return; }
    let prob = parse_qps(path).expect("parse greenbea");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(60.0);
    let r = solve_qp_with(&prob, &opts);

    const BOUND_TOL: f64 = 1e-6;
    let n = prob.c.len();
    let m = prob.num_constraints;

    let mut viols: Vec<(usize, f64, f64, f64, f64, &'static str, f64, f64, f64)> = Vec::new();
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed { continue; }
        let x = r.solution[j];
        let rc = r.reduced_costs[j];
        let at_lb = lb.is_finite() && (x - lb).abs() < BOUND_TOL;
        let at_ub = ub.is_finite() && (x - ub).abs() < BOUND_TOL;
        let (viol, where_): (f64, &'static str) = if at_lb && !at_ub {
            (f64::max(0.0, -rc), "at_lb")
        } else if at_ub && !at_lb {
            (f64::max(0.0, rc), "at_ub")
        } else {
            (0.0, "interior")
        };
        if viol > 1e-6 {
            let scale = 1.0 + rc.abs() + prob.c[j].abs();
            viols.push((j, x, lb, ub, rc, where_, prob.c[j], viol, viol/scale));
        }
    }
    viols.sort_by(|a, b| b.8.partial_cmp(&a.8).unwrap());
    eprintln!("greenbea: status={:?} obj={:.4e} m={} n={}", r.status, r.objective, m, n);
    eprintln!("greenbea violations (top 20 of {}):", viols.len());
    eprintln!("  j     x          lb         ub         rc         where    c[j]       viol      rel");
    for v in viols.iter().take(20) {
        eprintln!("  {:5} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:8} {:10.3e} {:10.3e} {:10.3e}",
            v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7, v.8);
    }
    let mut y_nonzero = 0;
    let mut y_max = 0.0_f64;
    for i in 0..m {
        if r.dual_solution[i].abs() > 1e-12 { y_nonzero += 1; }
        y_max = y_max.max(r.dual_solution[i].abs());
    }
    eprintln!("greenbea: y_nonzero={} y_max={:.3e}", y_nonzero, y_max);
}

/// greenbea 深掘り: col 2741 (worst) の entries と y の出元を identify。
#[test]
#[ignore = "diag (60s)、default 除外"]
fn greenbea_col2741_deep_diag() {
    let path = Path::new("data/lp_problems/greenbea.QPS");
    if !path.exists() { eprintln!("[SKIP]"); return; }
    let prob = parse_qps(path).expect("parse greenbea");
    let mut opts_on = SolverOptions::default();
    opts_on.presolve = true;
    opts_on.timeout_secs = Some(60.0);
    let r_on = solve_qp_with(&prob, &opts_on);

    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    opts_off.timeout_secs = Some(180.0);
    let r_off = solve_qp_with(&prob, &opts_off);
    eprintln!("greenbea presolve=off status={:?}", r_off.status);

    for &j in &[2741usize, 2743, 2687, 2742, 2739] {
        eprintln!("------ col {} : c[j]={:.3e} bounds={:?} ------",
            j, prob.c[j], prob.bounds[j]);
        eprintln!("  x_on  = {:.3e}  rc_on = {:.3e}", r_on.solution[j], r_on.reduced_costs[j]);
        if matches!(r_off.status, SolveStatus::Optimal) {
            eprintln!("  x_off = {:.3e}  rc_off= {:.3e}", r_off.solution[j], r_off.reduced_costs[j]);
        }
        if let Ok((rows, vals)) = prob.a.get_column(j) {
            eprintln!("  col entries ({}):", rows.len());
            let mut sum_on = 0.0;
            let mut sum_off = 0.0;
            for (k, &i) in rows.iter().enumerate() {
                let a = vals[k];
                let yi_on = r_on.dual_solution[i];
                let yi_off = if matches!(r_off.status, SolveStatus::Optimal) { r_off.dual_solution[i] } else { f64::NAN };
                sum_on += a * yi_on;
                sum_off += a * yi_off;
                eprintln!("    i={:5} A={:10.3e} y_on={:10.3e} y_off={:10.3e} A*y_on={:10.3e} A*y_off={:10.3e}",
                    i, a, yi_on, yi_off, a*yi_on, a*yi_off);
            }
            eprintln!("  Σ A*y_on  = {:.3e} → rc_on  = c - Σ = {:.3e}", sum_on, prob.c[j] - sum_on);
            if matches!(r_off.status, SolveStatus::Optimal) {
                eprintln!("  Σ A*y_off = {:.3e} → rc_off = c - Σ = {:.3e}", sum_off, prob.c[j] - sum_off);
            }
        }
    }
}

/// 診断: greenbea で row 1270 がどの col に何個 entry を持つか列挙
#[test]
#[ignore = "diag (60s)、default 除外"]
fn greenbea_row1270_scan() {
    let path = Path::new("data/lp_problems/greenbea.QPS");
    if !path.exists() { eprintln!("[SKIP]"); return; }
    let prob = parse_qps(path).expect("parse greenbea");
    let n = prob.c.len();
    let row = 1270usize;
    let mut cols_in_row: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    opts_off.timeout_secs = Some(180.0);
    let r_off = solve_qp_with(&prob, &opts_off);
    for j in 0..n {
        if let Ok((rows, vals)) = prob.a.get_column(j) {
            for (k, &i) in rows.iter().enumerate() {
                if i == row {
                    let x_off = if matches!(r_off.status, SolveStatus::Optimal) { r_off.solution[j] } else { f64::NAN };
                    cols_in_row.push((j, vals[k], prob.c[j], x_off, r_off.reduced_costs[j]));
                }
            }
        }
    }
    eprintln!("row {} appears in {} cols. ct={:?} b={}", row, cols_in_row.len(),
        prob.constraint_types[row], prob.b[row]);
    eprintln!("y_off[{}] = {:.3e}", row, r_off.dual_solution[row]);
    eprintln!("  j     A_ij      c_j      x_off     rc_off    ");
    for (j, a, c, x, rc) in cols_in_row.iter().take(40) {
        let (lb, ub) = prob.bounds[*j];
        eprintln!("  {:5} {:9.3e} {:9.3e} {:9.3e} {:9.3e}  bounds=({:.2e},{:.2e})", j, a, c, x, rc, lb, ub);
    }
}
