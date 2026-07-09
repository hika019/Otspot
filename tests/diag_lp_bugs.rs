//! 診断テスト: capri/forplan/scsd8/wood1p の根本原因調査
//!
//! 各問題について事実を記録する:
//! - presolve off vs on で解が変わるか
//! - presolve 後の問題サイズ・bounds・b の範囲
//! - scsd8/wood1p: simplex が失敗するか、IPM fallback で解けるか
//! - 各解が制約を満たすか (feasibility 確認)

use otspot::io::mps::parse_mps_file;
use otspot::io::qps::parse_qps;
use otspot::options::{SimplexMethod, SolverOptions};
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::qp::solve_qp_with;
use otspot::{solve, solve_with};
use std::path::Path;
use std::time::Instant;

/// 制約違反の最大値を計算する (Ax に対する)
fn max_violation_lp(x: &[f64], prob: &LpProblem) -> f64 {
    let m = prob.num_constraints;
    let n = prob.num_vars.min(x.len());
    let mut ax = vec![0.0f64; m];
    for (j, &x_j) in x.iter().enumerate().take(n) {
        if let Ok((rows, vals)) = prob.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                ax[row] += vals[k] * x_j;
            }
        }
    }
    let mut max_viol = 0.0f64;
    for (i, &ax_i) in ax.iter().enumerate() {
        let viol = match prob.constraint_types[i] {
            ConstraintType::Eq => (ax_i - prob.b[i]).abs(),
            ConstraintType::Le => (ax_i - prob.b[i]).max(0.0),
            ConstraintType::Ge => (prob.b[i] - ax_i).max(0.0),
            _ => 0.0,
        };
        if viol > max_viol {
            max_viol = viol;
        }
    }
    max_viol
}

/// capri: presolve off vs on の事実確認
#[test]
fn diag_capri_presolve_vs_no_presolve() {
    let mps_path = Path::new("tests/netlib/capri.mps");
    assert!(
        mps_path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        mps_path.display()
    );
    let prob = parse_mps_file(mps_path).expect("parse capri.mps failed");
    println!("capri: n={}, m={}", prob.num_vars, prob.num_constraints);

    // constraint type 分布
    let n_eq = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Eq)
        .count();
    let n_le = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Le)
        .count();
    let n_ge = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Ge)
        .count();
    println!("  constraints: Eq={}, Le={}, Ge={}", n_eq, n_le, n_ge);

    // bounds 分布
    let n_fixed = prob
        .bounds
        .iter()
        .filter(|&&(lo, hi)| (lo - hi).abs() < 1e-12)
        .count();
    let n_finite_lb_nonzero = prob
        .bounds
        .iter()
        .filter(|&&(lo, _)| lo.is_finite() && lo.abs() > 1e-12)
        .count();
    let n_finite_ub = prob
        .bounds
        .iter()
        .filter(|&&(_, hi)| hi.is_finite())
        .count();
    let n_fr = prob
        .bounds
        .iter()
        .filter(|&&(lo, hi)| lo == f64::NEG_INFINITY && hi == f64::INFINITY)
        .count();
    println!(
        "  bounds: fixed={}, finite_lb(nonzero)={}, finite_ub={}, free={}",
        n_fixed, n_finite_lb_nonzero, n_finite_ub, n_fr
    );

    // lb の最大値
    let max_lb = prob
        .bounds
        .iter()
        .filter_map(|&(lo, _)| if lo.is_finite() { Some(lo) } else { None })
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lb = prob
        .bounds
        .iter()
        .filter_map(|&(lo, _)| if lo.is_finite() { Some(lo) } else { None })
        .fold(f64::INFINITY, f64::min);
    println!("  lb range: [{:.4}, {:.4}]", min_lb, max_lb);

    // b の範囲
    let max_b = prob.b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_b = prob.b.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("  b range: [{:.4e}, {:.4e}]", min_b, max_b);

    // Step 2: presolve off で解く
    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    let t0 = Instant::now();
    let result_off = solve_with(&prob, &opts_off);
    println!(
        "capri presolve=OFF: status={:?}, obj={:.6e}, time={:.3}s",
        result_off.status,
        result_off.objective,
        t0.elapsed().as_secs_f64()
    );
    if !result_off.solution.is_empty() {
        let viol = max_violation_lp(&result_off.solution, &prob);
        println!("  max_violation={:.2e}", viol);
    }

    // Step 3: presolve on で解く
    let mut opts_on = SolverOptions::default();
    opts_on.presolve = true;
    let t1 = Instant::now();
    let result_on = solve_with(&prob, &opts_on);
    println!(
        "capri presolve=ON:  status={:?}, obj={:.6e}, time={:.3}s",
        result_on.status,
        result_on.objective,
        t1.elapsed().as_secs_f64()
    );
    if !result_on.solution.is_empty() {
        let viol = max_violation_lp(&result_on.solution, &prob);
        println!("  max_violation={:.2e}", viol);
    }

    // presolve on の解が正しいか確認 (expected: 2690.0129138)
    let expected = 2690.012914;
    if result_on.status == SolveStatus::Optimal {
        let rel = (result_on.objective - expected).abs() / expected.abs();
        println!("  rel_err from expected={:.2e}", rel);
        assert!(rel < 1e-3, "capri presolve=ON obj err: {:.2e}", rel);
    }
}

/// forplan: presolve off vs on の事実確認
#[test]
fn diag_forplan_presolve_vs_no_presolve() {
    let qps_path = Path::new("data/lp_problems/forplan.QPS");
    assert!(
        qps_path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        qps_path.display()
    );
    let prob_raw = parse_qps(qps_path).expect("parse forplan.QPS failed");

    // QpProblem から LpProblem を構築
    let lp = LpProblem::new_general(
        prob_raw.c.clone(),
        prob_raw.a.clone(),
        prob_raw.b.clone(),
        prob_raw.constraint_types.clone(),
        prob_raw.bounds.clone(),
        None,
    )
    .expect("forplan LpProblem construction failed");

    println!("forplan: n={}, m={}", lp.num_vars, lp.num_constraints);

    let n_eq = lp
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Eq)
        .count();
    let n_le = lp
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Le)
        .count();
    let n_ge = lp
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Ge)
        .count();
    println!("  constraints: Eq={}, Le={}, Ge={}", n_eq, n_le, n_ge);

    let n_fixed = lp
        .bounds
        .iter()
        .filter(|&&(lo, hi)| (lo - hi).abs() < 1e-12)
        .count();
    let n_finite_lb_nonzero = lp
        .bounds
        .iter()
        .filter(|&&(lo, _)| lo.is_finite() && lo.abs() > 1e-12)
        .count();
    let n_finite_ub = lp.bounds.iter().filter(|&&(_, hi)| hi.is_finite()).count();
    let n_fr = lp
        .bounds
        .iter()
        .filter(|&&(lo, hi)| lo == f64::NEG_INFINITY && hi == f64::INFINITY)
        .count();
    println!(
        "  bounds: fixed={}, finite_lb(nonzero)={}, finite_ub={}, free={}",
        n_fixed, n_finite_lb_nonzero, n_finite_ub, n_fr
    );

    let max_lb = lp
        .bounds
        .iter()
        .filter_map(|&(lo, _)| if lo.is_finite() { Some(lo) } else { None })
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lb = lp
        .bounds
        .iter()
        .filter_map(|&(lo, _)| if lo.is_finite() { Some(lo) } else { None })
        .fold(f64::INFINITY, f64::min);
    println!("  lb range: [{:.4}, {:.4}]", min_lb, max_lb);

    let max_b = lp.b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_b = lp.b.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("  b range: [{:.4e}, {:.4e}]", min_b, max_b);

    // presolve off
    let mut opts_off = SolverOptions::default();
    opts_off.presolve = false;
    let t0 = Instant::now();
    let result_off = solve_with(&lp, &opts_off);
    println!(
        "forplan presolve=OFF: status={:?}, obj={:.6e}, time={:.3}s",
        result_off.status,
        result_off.objective,
        t0.elapsed().as_secs_f64()
    );
    if !result_off.solution.is_empty() {
        let viol = max_violation_lp(&result_off.solution, &lp);
        println!("  max_violation={:.2e}", viol);
    }

    // presolve on
    let mut opts_on = SolverOptions::default();
    opts_on.presolve = true;
    let t1 = Instant::now();
    let result_on = solve_with(&lp, &opts_on);
    println!(
        "forplan presolve=ON:  status={:?}, obj={:.6e}, time={:.3}s",
        result_on.status,
        result_on.objective,
        t1.elapsed().as_secs_f64()
    );
    if !result_on.solution.is_empty() {
        let viol = max_violation_lp(&result_on.solution, &lp);
        println!("  max_violation={:.2e}", viol);
    }

    // expected: -6.6421873953e+02
    let expected = -6.6421873953e+02;
    if result_on.status == SolveStatus::Optimal {
        let rel = (result_on.objective - expected).abs() / expected.abs().max(1.0);
        println!("  rel_err from expected={:.2e}", rel);
        assert!(rel < 1e-3, "forplan presolve=ON obj err: {:.2e}", rel);
    }
}

/// scsd8: simplex (Primal/Dual) の挙動確認
/// - 全制約が Eq → 人工変数多数 → Phase I degeneracy
#[test]
fn diag_scsd8_simplex_behavior() {
    let qps_path = Path::new("data/lp_problems/scsd8.QPS");
    assert!(
        qps_path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        qps_path.display()
    );
    let prob = parse_qps(qps_path).expect("parse scsd8.QPS failed");
    println!("scsd8: n={}, m={}", prob.a.ncols(), prob.b.len());

    let n_eq = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Eq)
        .count();
    let n_le = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Le)
        .count();
    let n_ge = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Ge)
        .count();
    println!("  constraints: Eq={}, Le={}, Ge={}", n_eq, n_le, n_ge);

    let max_b = prob.b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_b = prob.b.iter().cloned().fold(f64::INFINITY, f64::min);
    let n_b_nonzero = prob.b.iter().filter(|&&bi| bi.abs() > 1e-12).count();
    println!(
        "  b range: [{:.4e}, {:.4e}], b≠0 count={}",
        min_b, max_b, n_b_nonzero
    );

    let n_fixed = prob
        .bounds
        .iter()
        .filter(|&&(lo, hi)| (lo - hi).abs() < 1e-12)
        .count();
    let n_fr = prob
        .bounds
        .iter()
        .filter(|&&(lo, hi)| lo == f64::NEG_INFINITY && hi == f64::INFINITY)
        .count();
    println!("  bounds: fixed={}, free={}", n_fixed, n_fr);

    // プリマル単体法で実行 (デフォルト)
    let mut opts_primal = SolverOptions::default();
    opts_primal.timeout_secs = Some(30.0);
    opts_primal.simplex_method = SimplexMethod::Primal;
    let t0 = Instant::now();
    let result_primal = solve_qp_with(&prob, &opts_primal);
    println!(
        "scsd8 Primal Simplex: status={:?}, obj={:.6e}, time={:.3}s",
        result_primal.status,
        result_primal.objective,
        t0.elapsed().as_secs_f64()
    );

    // Default (Simplex → IPM fallback)
    let mut opts_default = SolverOptions::default();
    opts_default.timeout_secs = Some(30.0);
    let t1 = Instant::now();
    let result_default = solve_qp_with(&prob, &opts_default);
    println!(
        "scsd8 Default:        status={:?}, obj={:.6e}, time={:.3}s",
        result_default.status,
        result_default.objective,
        t1.elapsed().as_secs_f64()
    );

    // feasibility 確認 (scsd8 は QpProblem なので LpProblem に変換)
    let lp = LpProblem::new_general(
        prob.c.clone(),
        prob.a.clone(),
        prob.b.clone(),
        prob.constraint_types.clone(),
        prob.bounds.clone(),
        None,
    )
    .expect("scsd8 LpProblem construction failed");

    if !result_default.solution.is_empty() {
        let viol = max_violation_lp(&result_default.solution, &lp);
        println!("  Default max_violation={:.2e}", viol);
    }

    // scsd8 expected: 9.04723801E+02
    let expected = 9.04723801e+02;
    if result_default.status == SolveStatus::Optimal {
        let rel = (result_default.objective - expected).abs() / expected.abs().max(1.0);
        println!("  Default rel_err from expected={:.2e}", rel);
    }
}

/// wood1p: simplex の挙動確認
/// - 243 Eq + 1 Ge の問題
#[test]
fn diag_wood1p_simplex_behavior() {
    let qps_path = Path::new("data/lp_problems/wood1p.QPS");
    assert!(
        qps_path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        qps_path.display()
    );
    let prob = parse_qps(qps_path).expect("parse wood1p.QPS failed");
    println!("wood1p: n={}, m={}", prob.a.ncols(), prob.b.len());

    let n_eq = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Eq)
        .count();
    let n_le = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Le)
        .count();
    let n_ge = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Ge)
        .count();
    println!("  constraints: Eq={}, Le={}, Ge={}", n_eq, n_le, n_ge);

    // b の値の範囲 (Eq制約のbが0かを確認)
    let n_b_zero = prob.b.iter().filter(|&&bi| bi.abs() < 1e-12).count();
    let max_b = prob.b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_b = prob.b.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "  b range: [{:.4e}, {:.4e}], b≈0 count={}",
        min_b, max_b, n_b_zero
    );

    let n_fixed = prob
        .bounds
        .iter()
        .filter(|&&(lo, hi)| (lo - hi).abs() < 1e-12)
        .count();
    let n_fr = prob
        .bounds
        .iter()
        .filter(|&&(lo, hi)| lo == f64::NEG_INFINITY && hi == f64::INFINITY)
        .count();
    let n_default = prob
        .bounds
        .iter()
        .filter(|&&(lo, hi)| lo == 0.0 && hi == f64::INFINITY)
        .count();
    println!(
        "  bounds: fixed={}, free={}, default(0,∞)={}",
        n_fixed, n_fr, n_default
    );

    // Default solve (Simplex → IPM fallback if SingularBasis)
    let mut opts_default = SolverOptions::default();
    opts_default.timeout_secs = Some(60.0);
    let t0 = Instant::now();
    let result_default = solve_qp_with(&prob, &opts_default);
    println!(
        "wood1p Default:       status={:?}, obj={:.6e}, time={:.3}s",
        result_default.status,
        result_default.objective,
        t0.elapsed().as_secs_f64()
    );

    // feasibility 確認
    let lp = LpProblem::new_general(
        prob.c.clone(),
        prob.a.clone(),
        prob.b.clone(),
        prob.constraint_types.clone(),
        prob.bounds.clone(),
        None,
    )
    .expect("wood1p LpProblem construction failed");

    if !result_default.solution.is_empty() {
        let viol = max_violation_lp(&result_default.solution, &lp);
        println!("  Default max_violation={:.2e}", viol);
    }

    // wood1p expected: 1.44290241E+00
    let expected = 1.44290241e+00;
    if result_default.status == SolveStatus::Optimal {
        let rel = (result_default.objective - expected).abs() / expected.abs().max(1.0);
        println!("  rel_err from expected={:.2e}", rel);
    }
}

/// boeing1: 現在の状態確認 (case_c 修正後)
#[test]
fn diag_boeing1_current_state() {
    let mps_path = Path::new("tests/netlib/boeing1.mps");
    assert!(
        mps_path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        mps_path.display()
    );
    let prob = parse_mps_file(mps_path).expect("parse boeing1.mps failed");
    println!("boeing1: n={}, m={}", prob.num_vars, prob.num_constraints);

    let n_eq = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Eq)
        .count();
    let n_le = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Le)
        .count();
    let n_ge = prob
        .constraint_types
        .iter()
        .filter(|&&ct| ct == ConstraintType::Ge)
        .count();
    println!("  constraints: Eq={}, Le={}, Ge={}", n_eq, n_le, n_ge);

    let t0 = Instant::now();
    let result = solve(&prob);
    println!(
        "boeing1: status={:?}, obj={:.6e}, time={:.3}s",
        result.status,
        result.objective,
        t0.elapsed().as_secs_f64()
    );

    let expected = -3.3521356751e+02;
    if !result.solution.is_empty() {
        let viol = max_violation_lp(&result.solution, &prob);
        println!("  max_violation={:.2e}", viol);
        if viol > 1e-4 {
            println!(
                "  INFEASIBLE SOLUTION! max_viol={:.2e} (expected < 1e-4)",
                viol
            );
        } else {
            println!("  solution is feasible");
        }
    }
    if result.status == SolveStatus::Optimal {
        let rel = (result.objective - expected).abs() / expected.abs().max(1.0);
        println!("  rel_err from expected={:.2e}", rel);
        if rel > 1e-3 {
            println!("  WARNING: obj mismatch (expected {:.6e})", expected);
        } else {
            println!("  objective matches expected");
        }
    }
}

/// Bland 則の適用範囲確認 (コードレビュー)
/// - ratio test のみに Bland 則が適用されているか
/// - entering variable の選択には steepest-edge pricing が使われているか
#[test]
fn diag_bland_rule_coverage() {
    // これはコード構造を確認するテストではなく、
    // scsd8 を実行して SingularBasis (cycling) が起きるかを確認する。
    // デフォルト: Simplex が SingularBasis で失敗 → solve_as_lp が IPM fallback
    // Primal only: SingularBasis を返すはず
    let qps_path = Path::new("data/lp_problems/scsd8.QPS");
    assert!(
        qps_path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        qps_path.display()
    );
    let prob = parse_qps(qps_path).expect("parse scsd8.QPS failed");

    // Simplex のみ (IPM fallback を起こさないよう presolve=false, simplex only)
    // solve_qp_with は Q=0 → solve_as_lp → Simplex → IPM fallback
    // SimplexBackend 直接呼び出しには公開 API が必要
    // ここでは solve_qp_with デフォルト (Simplex → IPM fallback) の結果を観察する
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(15.0);
    opts.simplex_method = SimplexMethod::Primal;

    // Primal simplex のみで解く (IPM fallback は solve_as_lp で起きるはずだが)
    let t0 = Instant::now();
    let result = solve_qp_with(&prob, &opts);
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "scsd8 Primal-only: status={:?}, obj={:.6e}, time={:.3}s",
        result.status, result.objective, elapsed
    );

    // SingularBasis が NumericalError として伝搬し、IPM fallback で Optimal になるはず
    // (solve_as_lp 内の fallback)
    match result.status {
        SolveStatus::Optimal => println!("  -> Optimal (via IPM fallback or simplex success)"),
        SolveStatus::NumericalError => {
            println!("  -> NumericalError (simplex failed, no IPM fallback triggered)")
        }
        _ => println!("  -> unexpected status: {:?}", result.status),
    }
}

/// modszk1 Primal: 旧ベースラインで10-18sで解けていた
#[test]
fn diag_modszk1_primal_baseline() {
    let qps_path = std::path::Path::new("data/lp_problems/modszk1.QPS");
    assert!(
        qps_path.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        qps_path.display()
    );
    let prob = otspot::io::qps::parse_qps(qps_path).expect("parse modszk1 failed");
    let known_obj = 3.21049143e2_f64;
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    opts.simplex_method = SimplexMethod::Primal;
    let t0 = std::time::Instant::now();
    let result = otspot::qp::solve_qp_with(&prob, &opts);
    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "modszk1 Primal: status={:?} obj={:.6e} t={:.2}s",
        result.status, result.objective, elapsed
    );
    if result.status == SolveStatus::Optimal {
        let rel_err = (result.objective - known_obj).abs() / known_obj.abs().max(1.0);
        println!("  rel_err={:.2e}", rel_err);
    }
}
