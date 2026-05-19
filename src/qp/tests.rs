use super::*;
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

mod dual_recovery;
mod psd_nonconvex;
mod smoke;

const EPS: f64 = 1e-2;

fn assert_close(a: f64, b: f64, eps: f64, name: &str) {
    assert!(
        (a - b).abs() < eps,
        "{}: expected {:.8}, got {:.8} (diff={:.2e})",
        name,
        b,
        a,
        (a - b).abs()
    );
}

/// 大行ノルム制約での Ruiz scaling 耐性 (元空間で pfeas 評価)。
#[test]
fn test_presolve_pfeas_large_row_norm() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0], 1, 1).unwrap();
    let b = vec![500.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    let ax = problem.a.mat_vec_mul(&result.solution).unwrap();
    let pfeas = ax
        .iter()
        .zip(problem.b.iter())
        .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
        .fold(0.0_f64, f64::max);
    let norm_b = problem
        .b
        .iter()
        .fold(0.0_f64, |a, &bi| a.max(bi.abs()))
        .max(1.0);
    let eps = opts.ipm_eps();
    assert!(pfeas < eps * (1.0 + norm_b), "pfeas={pfeas:.2e}");
}

/// bounds 付き問題で post-postsolve bfeas check が誤降格しないこと。
#[test]
fn test_presolve_bfeas_bounded_problem() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    let x = result.solution[0];
    assert!(x >= -1e-4, "x >= lb=0, got {x}");
    assert!(x <= 1.0 + 1e-4, "x <= ub=1, got {x}");
}

/// 正常解で post-postsolve pfeas+bfeas check が Optimal を維持。
#[test]
fn test_presolve_pfeas_bfeas_ok() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![1.0];
    let bounds = vec![(0.0_f64, 0.5_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
}

/// presolve=true で post-unscaling check が正常問題に影響しないこと。
#[test]
fn test_solve_qp_with_presolve_path_verified() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    assert!(opts.presolve);
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    let eps = 1e-3_f64;
    assert!((result.solution[0] - 0.5).abs() < eps);
    assert!((result.solution[1] - 0.5).abs() < eps);
}


/// solve_as_lp が NumericalError を返さないこと。
#[test]
fn test_qp001_solve_as_lp_no_numerical_error() {
    let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![4.0];
    let bounds = vec![(0.0f64, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_ne!(result.status, SolveStatus::NumericalError);
}

/// timeout_secs=None で有限ステップ収束。
#[test]
fn test_a2t03_qp_no_deadline_converges() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        timeout_secs: None,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
}

/// cancel_flag 事前設定で Timeout。
#[test]
fn test_a3c02_cancel_flag_preset_qp_returns_timeout() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let cancel = Arc::new(AtomicBool::new(true));
    let opts = SolverOptions {
        cancel_flag: Some(Arc::clone(&cancel)),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

/// presolve 有無で解が一致 (透過性)。
#[test]
fn test_a4p01_presolve_transparency_qp() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts_with = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let opts_without = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result_with = solve_qp_with(&problem, &opts_with);
    let result_without = solve_qp_with(&problem, &opts_without);
    assert_eq!(result_with.status, SolveStatus::Optimal);
    assert_eq!(result_without.status, SolveStatus::Optimal);
    assert!((result_with.solution[0] - result_without.solution[0]).abs() < 1e-3);
    assert!((result_with.solution[1] - result_without.solution[1]).abs() < 1e-3);
}

/// n>1000 では Cholesky skip。対角負値は検出、非対角の非 PSD は skip (既知制限)。
#[test]
fn test_a6i03_nonconvex_skip_for_large_n() {
    let n = 1001usize;
    let mut rows = vec![0usize];
    let mut cols = vec![0usize];
    let mut vals = vec![-1e-3_f64];
    for i in 1..n {
        rows.push(i);
        cols.push(i);
        vals.push(1.0);
    }
    let q1 = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    assert!(!check_q_positive_semidefinite(&q1));

    let mut rows2: Vec<usize> = (0..n).collect();
    let mut cols2: Vec<usize> = (0..n).collect();
    let mut vals2: Vec<f64> = vec![1.0; n];
    rows2.push(0);
    cols2.push(1);
    vals2.push(-2.0);
    let q2 = CscMatrix::from_triplets(&rows2, &cols2, &vals2, n, n).unwrap();
    assert!(check_q_positive_semidefinite(&q2));
}

/// A7-CS02: concurrent solver スレッド安全性（cancel_flag 経由の停止）
#[cfg(feature = "parallel")]
#[test]
fn test_a7cs02_concurrent_cancel_flag_thread_safety() {
    // SPEC: A7-CS02
    // concurrent solver で Optimal を発見したとき cancel_flag でリソースリーク・
    // データ競合なしに停止することを確認（10回繰り返してクラッシュなし）
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    for _ in 0..10 {
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal);
    }
}

/// 全スレッド Timeout → Timeout。
#[cfg(feature = "parallel")]
#[test]
fn test_a7cs03_concurrent_all_timeout_returns_timeout() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        timeout_secs: Some(0.0),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

/// concurrent solver で cancel_flag=true → Timeout。
#[cfg(feature = "parallel")]
#[test]
fn test_a3c01_cancel_flag_concurrent_returns_timeout() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let cancel = Arc::new(AtomicBool::new(true));
    let opts = SolverOptions {
        cancel_flag: Some(Arc::clone(&cancel)),
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Timeout);
}

/// presolve OFF 基準線。
#[test]
fn test_postsolve_t1_presolve_off_baseline() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
    let b = vec![4.0, 3.0];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-8_f64;
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.objective).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 4.0).abs() < tol);
    assert!((result.slack[1] - 3.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), n);
    assert!((result.reduced_costs[0] - 2.0).abs() < tol);
    assert!((result.reduced_costs[1] - 3.0).abs() < tol);
}

/// FixedVar + col_map リマップ (rc[2]=0 で展開されること)。
#[test]
fn test_postsolve_t2_fixed_var_col_map() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 2.0], 2, n)
        .unwrap();
    let b = vec![4.0, 6.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-6_f64;
    assert_eq!(result.solution.len(), 3);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.solution[2] - 5.0).abs() < tol);
    assert!((result.objective - 5.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[0] - 2.0).abs() < tol);
    assert!((result.reduced_costs[1] - 3.0).abs() < tol);
    assert!((result.reduced_costs[2] - 1.0).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 4.0).abs() < tol);
    assert!((result.slack[1] - 6.0).abs() < tol);
    // 自由変数 (x, y) のみ複ementarity 検査 (固定 z は lb/ub の dual を持ち得る)。
    for j in 0..2 {
        assert!((result.solution[j] * result.reduced_costs[j]).abs() < 1e-7);
    }
}

/// SingletonRow + row_map: x=2 (Eq) + y≤3。
#[test]
fn test_postsolve_t3_singleton_row() {
    use crate::problem::ConstraintType;
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    // x=2 (Eq), y<=3 (Le)
    let rows = &[0usize, 1usize];
    let cols = &[0usize, 1usize];
    let vals = &[1.0, 1.0];
    let a = CscMatrix::from_triplets(rows, cols, vals, 2, n).unwrap();
    let b = vec![2.0, 3.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Eq, ConstraintType::Le],
    )
    .unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-6_f64;
    assert_eq!(result.solution.len(), 2);
    assert!((result.solution[0] - 2.0).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0]).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 2);
}

/// Ruiz + FixedVar 複合。
#[test]
fn test_postsolve_t4_ruiz_fixed_var() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[10.0, 1.0, 1.0], 2, n).unwrap();
    let b = vec![10.0, 3.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (5.0, 5.0)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-6_f64;
    assert_eq!(result.solution.len(), 3);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.solution[2] - 5.0).abs() < tol);
    assert!((result.objective - 5.0).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 10.0).abs() < tol);
    assert!((result.slack[1] - 3.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[2] - 1.0).abs() < tol);
}

/// LCS (1e7 係数) + Ruiz + FixedVar: slack を元空間 b-Ax で再計算する精度確認。
#[test]
fn test_postsolve_t5_lcs_ruiz_fixed_var() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1e7, 1.0, 1.0, 1.0], 2, n)
        .unwrap();
    let b = vec![1e7, 2.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.5, 0.5)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let x = result.solution[0];
    let y = result.solution[1];
    assert_eq!(result.slack.len(), 2);
    let slack0_expected = 1e7 - 1e7 * x - y;
    let slack1_expected = 2.0 - x - y;
    let tol_rel = 1e-5_f64;
    assert!((result.slack[0] - slack0_expected).abs() <= tol_rel * slack0_expected.abs().max(1.0));
    assert!((result.slack[1] - slack1_expected).abs() <= tol_rel * slack1_expected.abs().max(1.0));
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[2] - 1.0).abs() < 1e-6);
}

/// EmptyCol (z 制約行ゼロ) → z=lb=0 に固定。
#[test]
fn test_postsolve_t6_empty_col() {
    let n = 3usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, n).unwrap();
    let b = vec![4.0, 3.0];
    let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, 3.0)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-8_f64;
    assert_eq!(result.solution.len(), 3);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
    assert!((result.solution[2]).abs() < tol);
    assert!((result.objective).abs() < tol);
    assert_eq!(result.slack.len(), 2);
    assert!((result.slack[0] - 4.0).abs() < tol);
    assert!((result.slack[1] - 3.0).abs() < tol);
    assert_eq!(result.reduced_costs.len(), 3);
    assert!((result.reduced_costs[2] - 1.0).abs() < tol);
}

/// QP IPM 経路では slack=[], reduced_costs=[]。
#[test]
fn test_postsolve_t7_qp_ipm_empty_slack_rc() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![2.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!(result.slack.is_empty());
    assert!(result.reduced_costs.is_empty());
}

/// 全変数 FixedVar。
#[test]
fn test_postsolve_e1_all_vars_fixed() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(1.0_f64, 1.0_f64), (2.0_f64, 2.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.solution.len(), 2);
    assert_eq!(result.reduced_costs.len(), 2);
    assert_eq!(result.slack.len(), 0);
}

/// 制約なし問題: slack=0, rc=n。
#[test]
fn test_postsolve_e2_no_constraints() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 3.0];
    let a = CscMatrix::new(0, n);
    let b: Vec<f64> = vec![];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let tol = 1e-8_f64;
    assert_eq!(result.slack.len(), 0);
    assert_eq!(result.reduced_costs.len(), n);
    assert!((result.solution[0]).abs() < tol);
    assert!((result.solution[1]).abs() < tol);
}

/// presolve=true でも reduction 発動なし → col_map identity。
#[test]
fn test_postsolve_e3_presolve_no_reduction() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![2.0];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.reduced_costs.len(), n);
    assert_eq!(result.slack.len(), 1);
    let tol = 1e-8_f64;
    assert!((result.slack[0] - 2.0).abs() < tol);
}

/// LCS 発動 + presolve 変数除去なし: slack を b-Ax 元空間再計算。
#[test]
fn test_postsolve_e4_lcs_no_presolve_elimination() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1e7, 1.0], 1, n).unwrap();
    let b = vec![1e7];
    let bounds = vec![(0.0, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let x = result.solution[0];
    let y = result.solution[1];
    assert_eq!(result.slack.len(), 1);
    let slack_expected = 1e7 - 1e7 * x - y;
    let tol_rel = 1e-5_f64;
    assert!((result.slack[0] - slack_expected).abs() <= tol_rel * slack_expected.abs().max(1.0));
    assert_eq!(result.reduced_costs.len(), n);
}

/// Q=0 (LP) で reduced_costs が理論値と一致 (Simplex 経路保持)。
#[test]
fn test_solve_as_lp_preserves_reduced_costs() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 2.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);

    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.reduced_costs.len(), n);
    assert_eq!(result.slack.len(), 1);
}

/// BD-T1: baseline (presolve OFF, 全変数 box) → bound_duals.len()=4。
#[test]
fn test_bd_t1_baseline_presolve_off() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, 5.0_f64); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 1e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0]).abs() < sol_tol);
    assert!((result.solution[1]).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 4);
    assert!(result.bound_duals[0] > tol);
    assert!(result.bound_duals[1] > tol);
    assert!(result.bound_duals[2].abs() < tol);
    assert!(result.bound_duals[3].abs() < tol);
}

/// BD-T2: FixedVar + bound_duals リマップ (z 除去 → bound_duals.len()=6, lb_x≠lb_y で順序検証)。
#[test]
fn test_bd_t2_fixed_var_remap_core() {
    let n = 3usize;
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
    let c = vec![2.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, 5.0_f64), (0.0_f64, 5.0_f64), (3.0_f64, 3.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 5e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0]).abs() < sol_tol);
    assert!((result.solution[1]).abs() < sol_tol);
    assert!((result.solution[2] - 3.0).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 6);
    assert!(result.bound_duals[0] > tol);
    assert!(result.bound_duals[1] > tol);
    // lb_x ≠ lb_y で変数順序バグを検出。
    assert!((result.bound_duals[0] - result.bound_duals[1]).abs() > tol);
    assert!((result.bound_duals[2]).abs() < tol);
    assert!(result.bound_duals[3].abs() < 5e-3);
    assert!(result.bound_duals[4].abs() < 5e-3);
    assert!((result.bound_duals[5]).abs() < tol);
    let dual = if result.dual_solution.is_empty() {
        0.0
    } else {
        result.dual_solution[0]
    };
    let kkt_x = 2.0 - dual - result.bound_duals[0] + result.bound_duals[3];
    assert!(kkt_x.abs() < 1e-3);
    let kkt_y = 1.0 - dual - result.bound_duals[1] + result.bound_duals[4];
    assert!(kkt_y.abs() < 1e-3);
}

/// BD-T3: FixedVar + lb_only 変数 → bound_duals.len()=3。
#[test]
fn test_bd_t3_fixed_var_lb_only() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, f64::INFINITY), (2.0_f64, 2.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.bound_duals.len(), 3);
}

/// BD-T4: EmptyCol の bound_duals を KKT で復元 (refit_bound_duals_kkt が 0 埋めを修復)。
#[test]
fn test_bd_t4_empty_col_kkt_recovered() {
    let n = 3usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
    let c = vec![-1.0, -1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![4.0];
    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0_f64, 3.0_f64),
    ];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.bound_duals.len(), 2);
    let z_lb = result.bound_duals[0];
    let z_ub = result.bound_duals[1];
    assert!((z_lb - 1.0).abs() < 1e-3, "z_lb={z_lb}");
    assert!(z_ub.abs() < 1e-3, "z_ub={z_ub}");
}

/// 全変数 ±∞ → bound_duals 空。
#[test]
fn test_bd_t5_unbounded_vars_empty() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!(result.bound_duals.is_empty());
}

/// BD-T6: FixedVar + ub 活性変数 (ub_dual 非ゼロ × presolve 残存)。
#[test]
fn test_bd_t6_ub_active_with_presolve() {
    let n = 3usize;
    let q =
        CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], n, n).unwrap();
    let c = vec![-1.0, -1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![10.0];
    let bounds = vec![(0.0_f64, 3.0_f64), (0.0_f64, 5.0_f64), (2.0_f64, 2.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: true,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 1e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0] - 3.0).abs() < sol_tol);
    assert!((result.solution[1] - 5.0).abs() < sol_tol);
    assert!((result.solution[2] - 2.0).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 6);
    assert!(result.bound_duals[0].abs() < tol);
    assert!(result.bound_duals[1].abs() < tol);
    assert!((result.bound_duals[2]).abs() < tol);
    assert!(result.bound_duals[3] > tol);
    assert!(result.bound_duals[4] > tol);
    assert!((result.bound_duals[5]).abs() < tol);
}

/// BD-T7: constraint active × lb_dual nonzero × KKT 照合 (x*=2, y*=1)。
#[test]
fn test_bd_t7_constraint_active_lb_dual_nonzero_kkt() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, n).unwrap();
    let b = vec![-3.0];
    let bounds = vec![(2.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..SolverOptions::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    let sol_tol = 1e-3_f64;
    let tol = 1e-4_f64;
    assert!((result.solution[0] - 2.0).abs() < sol_tol);
    assert!((result.solution[1] - 1.0).abs() < sol_tol);
    assert_eq!(result.bound_duals.len(), 2);
    let dual = if result.dual_solution.is_empty() {
        0.0
    } else {
        result.dual_solution[0]
    };
    assert!(dual > tol);
    assert!(result.bound_duals[0] > tol);
    assert!(result.bound_duals[1].abs() < tol);
    let kkt_x = result.solution[0] - dual - result.bound_duals[0];
    assert!(kkt_x.abs() < 1e-3);
    let kkt_y = result.solution[1] - dual - result.bound_duals[1];
    assert!(kkt_y.abs() < 1e-3);
}

/// row_infinity_norms 基本。
#[test]
fn test_row_infinity_norms_basic() {
    let a = CscMatrix::from_triplets(
        &[0, 1, 0],
        &[0, 1, 2],
        &[1.0, 2.5, -3.0],
        2,
        3,
    )
    .unwrap();
    let norms = a.row_infinity_norms();
    assert_eq!(norms.len(), 2);
    assert!((norms[0] - 3.0).abs() < 1e-15);
    assert!((norms[1] - 2.5).abs() < 1e-15);
}

/// 大/小係数行 mixed で行ノルム正規化 pfeas が偽 SubOptimal を防ぐ。
#[test]
fn test_pfeas_row_norm_mixed_scale() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1000.0], 2, 1).unwrap();
    let norms = a.row_infinity_norms();
    assert!((norms[0] - 1.0).abs() < 1e-15);
    assert!((norms[1] - 1000.0).abs() < 1e-15);

    let b: Vec<f64> = vec![1.0, 1000.0];
    let x_val: f64 = 1.0 + 1e-7;
    let ax: Vec<f64> = vec![x_val, 1000.0 * x_val];
    let eps: f64 = 1e-6;

    let pfeas_old = ax
        .iter()
        .zip(b.iter())
        .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
        .fold(0.0_f64, f64::max);
    assert!(pfeas_old > 1e-5);

    let pfeas_normalized = ax
        .iter()
        .zip(b.iter())
        .zip(norms.iter())
        .map(|((&ax_i, &b_i), &rn)| {
            let violation = (ax_i - b_i).max(0.0);
            violation / (1.0 + rn + b_i.abs())
        })
        .fold(0.0_f64, f64::max);
    assert!(pfeas_normalized < eps);
}

/// b=0 大係数行で正規化 pfeas が偽 SubOptimal を防ぐ。
#[test]
fn test_pfeas_row_norm_false_suboptimal_prevention() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1e6], 1, 1).unwrap();
    let norms = a.row_infinity_norms();
    assert!((norms[0] - 1e6).abs() < 1e-9);

    let b_val: f64 = 0.0;
    let ax_val: f64 = 1e6 * 1e-9;
    let eps: f64 = 1e-6;

    let norm_b = b_val.abs().max(1.0);
    let pfeas_old = (ax_val - b_val).abs();
    assert!(pfeas_old >= eps * (1.0 + norm_b));

    let pfeas_norm = (ax_val - b_val).abs() / (1.0 + norms[0] + b_val.abs());
    assert!(pfeas_norm < eps);
}

/// Ge 制約 (ConstraintType::Ge) で Optimal 到達。
#[test]
fn test_qp_ge_defensive() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Mixed Ge+Le 防御 (presolve=false でソルバ本体の正確さ; mixed presolve bug 既知)。
#[test]
fn test_qp_mixed_ge_le_defensive() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    // Row 0: x+y≥0.5 (Ge), Row 1: x-y≤1 (Le)
    let a =
        CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
            .unwrap();
    let b = vec![0.5, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Ge, ConstraintType::Le],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        presolve: false,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0, "D: wall-clock 6秒超過");
    assert_eq!(result.status, SolveStatus::Optimal, "D: status");
    assert_close(result.solution[0], 0.25, EPS, "D: x[0]");
    assert_close(result.solution[1], 0.25, EPS, "D: x[1]");
}

/// Concurrent Eq 制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_eq_constraint() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Concurrent Ge 制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_ge_constraint() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Concurrent Box 制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_box_constraint() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.0, EPS, "x[0]");
    assert_close(result.solution[1], 0.0, EPS, "x[1]");
}

/// Concurrent Mixed (Le+Eq)。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_mixed_constraint() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
    let b = vec![1.0, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Eq, ConstraintType::Le],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Concurrent 無制約。
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_unconstrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-2.0, -2.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 1.0, EPS, "x[0]");
    assert_close(result.solution[1], 1.0, EPS, "x[1]");
}

/// 全変数固定退化ケース (presolve=false で本体検証)。
#[test]
fn test_qp_all_vars_fixed() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b: Vec<f64> = vec![];
    let bounds = vec![(1.0_f64, 1.0_f64)];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

    let mut opts = SolverOptions {
        timeout_secs: Some(5.0),
        ..Default::default()
    };
    opts.presolve = false;
    let start = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    assert!(start.elapsed().as_secs_f64() < 6.0);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert_close(result.solution[0], 1.0, EPS, "x[0]");
}

/// SuboptimalSolution mapping: MaxIterations/NumericalError が外部に漏れないこと。
#[test]
fn test_suboptimal_to_optimal_mapping() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(2.0),
        ipm: crate::options::IpmOptions {
            max_iter: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_ne!(result.status, SolveStatus::MaxIterations);
    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
    ), "got {:?}", result.status);
}

/// MaxIterations が外部 API に漏れないこと。
#[test]
fn test_max_iterations_to_timeout_mapping() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        ipm: crate::options::IpmOptions {
            max_iter: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_ne!(result.status, SolveStatus::MaxIterations);
    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::Timeout | SolveStatus::SuboptimalSolution
    ), "got {:?}", result.status);
}

/// Eq 制約 presolve ON/OFF で解一致。
#[test]
fn test_presolve_qp_eq_on_off_consistency() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();

    let opts_on = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let mut opts_off = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    opts_off.presolve = false;

    let result_on = solve_qp_with(&problem, &opts_on);
    let result_off = solve_qp_with(&problem, &opts_off);

    assert_eq!(result_on.status, SolveStatus::Optimal);
    assert_eq!(result_off.status, SolveStatus::Optimal);
    assert!((result_on.solution[0] - result_off.solution[0]).abs() < 1e-4);
    assert!((result_on.solution[1] - result_off.solution[1]).abs() < 1e-4);
}

/// Box 制約 presolve ON/OFF で解一致。
#[test]
fn test_presolve_qp_box_on_off_consistency() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 2.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts_on = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let mut opts_off = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    opts_off.presolve = false;

    let result_on = solve_qp_with(&problem, &opts_on);
    let result_off = solve_qp_with(&problem, &opts_off);

    assert_eq!(result_on.status, SolveStatus::Optimal);
    assert_eq!(result_off.status, SolveStatus::Optimal);
    assert_close(result_on.solution[0], 0.0, EPS, "ON x[0]");
    assert_close(result_on.solution[1], 0.0, EPS, "ON x[1]");
    assert_close(result_off.solution[0], 0.0, EPS, "OFF x[0]");
    assert_close(result_off.solution[1], 0.0, EPS, "OFF x[1]");
}

/// Ge 制約 + presolve ON。
#[test]
fn test_qp_ge_constraint_with_presolve() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
}

/// Mixed (Ge+Le) presolve=false (mixed presolve バグ既知)。
#[test]
fn test_qp_mixed_ge_with_presolve() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a =
        CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
            .unwrap();
    let b = vec![0.5, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Ge, ConstraintType::Le],
    )
    .unwrap();

    let mut opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    opts.presolve = false;
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.25, EPS, "x[0]");
    assert_close(result.solution[1], 0.25, EPS, "x[1]");
}

/// Mixed (Ge+Le) presolve=ON + Ruiz=ON: pfeas Ge 違反検出 regression。
#[test]
fn test_qp_mixed_ge_le_presolve_ruiz_regression() {
    use crate::problem::ConstraintType;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a =
        CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, -1.0], 2, 2)
            .unwrap();
    let b = vec![0.5, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Ge, ConstraintType::Le],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert_close(result.solution[0], 0.25, EPS, "x[0]");
    assert_close(result.solution[1], 0.25, EPS, "x[1]");
    let pfeas = {
        let x = &result.solution;
        let ge_viol = (0.5_f64 - (x[0] + x[1])).max(0.0);
        let le_viol = (x[0] - x[1] - 1.0_f64).max(0.0);
        ge_viol.max(le_viol)
    };
    assert!(pfeas < 1e-6, "pfeas={:e}", pfeas);

    let opts_no_presolve = SolverOptions {
        timeout_secs: Some(10.0),
        presolve: false,
        ..Default::default()
    };
    let result_no_presolve = solve_qp_with(&problem, &opts_no_presolve);
    assert_eq!(result_no_presolve.status, SolveStatus::Optimal);
    assert_close(result_no_presolve.solution[0], 0.25, EPS, "no-presolve x[0]");
    assert_close(result_no_presolve.solution[1], 0.25, EPS, "no-presolve x[1]");
}

/// 正常解で dfeas check が Optimal を維持。
#[test]
fn test_dfeas_optimal_preserved() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
}

/// スケール不変性 (1e6 倍) で Optimal 維持。
#[test]
fn test_dfeas_scale_invariant() {
    let scale = 1e6_f64;
    let q = CscMatrix::from_triplets(
        &[0, 1],
        &[0, 1],
        &[2.0 * scale * scale, 2.0 * scale * scale],
        2,
        2,
    )
    .unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-scale, -scale], 1, 2).unwrap();
    let b = vec![-scale];
    let bounds = vec![(0.0, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert_close(result.solution[0], 0.5, 1e-4, "x[0]");
    assert_close(result.solution[1], 0.5, 1e-4, "x[1]");
}

/// dfeas 悪化解の SuboptimalSolution 降格 (check_dfeas_status 直接呼出)。
#[test]
fn test_dfeas_bad_solution_downgraded() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    // 最適 x=y=0, dfeas=0。bad: x=y=1 で Qx+c=[2,2], dfeas=2.0。
    let bad_x = vec![1.0, 1.0];
    let bad_y: Vec<f64> = vec![];
    let bad_bd: Vec<f64> = vec![];

    let status = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 1e-6);
    assert_eq!(status, SolveStatus::SuboptimalSolution);
    let status_ok = ipm_core::check_dfeas_status(&problem, &bad_x, &bad_y, &bad_bd, 10.0);
    assert_eq!(status_ok, SolveStatus::Optimal);

    let status_rel =
        ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 0.01);
    assert_eq!(status_rel, SolveStatus::SuboptimalSolution);
    let status_rel_ok =
        ipm_core::check_dfeas_status_relative(&problem, &bad_x, &bad_y, &bad_bd, 1.0);
    assert_eq!(status_rel_ok, SolveStatus::Optimal);
}

/// 大 KKT スケール (2e12) でも相対閾値が正規化。
#[test]
fn test_dfeas_relative_threshold_large_kkt() {
    let n = 1usize;
    let q = CscMatrix::from_triplets(&[0], &[0], &[2e12], n, n).unwrap();
    let c = vec![-1e6];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "got {:?}", result.status);
    assert!((result.solution[0] - 5e-7).abs() < 1e-9, "x*=5e-7, got {:.2e}", result.solution[0]);
}

/// 巨大項キャンセレーション (Qx ≈ -A^Ty): 成分相対なら正確に判定。
#[test]
fn test_dfeas_cancellation_pattern() {
    let n = 2usize;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let big_x = vec![5e9, 5e9];
    let empty_y: Vec<f64> = vec![];
    let empty_bd: Vec<f64> = vec![];
    let status =
        ipm_core::check_dfeas_status_relative(&problem, &big_x, &empty_y, &empty_bd, 0.01);
    assert_eq!(status, SolveStatus::SuboptimalSolution);

    let good_x = vec![1e-12, 1e-12];
    let status_good =
        ipm_core::check_dfeas_status_relative(&problem, &good_x, &empty_y, &empty_bd, 1e-8);
    assert_eq!(status_good, SolveStatus::Optimal);
}

/// REFIT-T1: lb 活性 + c>0 で y_lb = c を復元。
#[test]
fn test_refit_bound_duals_lb_only_active() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.5_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![],
        bound_duals: vec![0.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result);
    assert!((result.bound_duals[0] - 2.5).abs() < 1e-9, "got {}", result.bound_duals[0]);
}

/// REFIT-T2: ub 活性 + c<0 で y_ub = -c。
#[test]
fn test_refit_bound_duals_ub_only_active() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![-3.0_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, 5.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![5.0],
        dual_solution: vec![],
        bound_duals: vec![0.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result);
    assert!((result.bound_duals[0] - 3.0).abs() < 1e-9, "got {}", result.bound_duals[0]);
}

/// REFIT-T3: 内点では y_lb=y_ub=0 維持。
#[test]
fn test_refit_bound_duals_interior_keeps_zero() {
    let n = 1usize;
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], n, n).unwrap();
    let c = vec![-4.0_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0_f64, 5.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![2.0],
        dual_solution: vec![],
        bound_duals: vec![0.0, 0.0],
        objective: -4.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result);
    assert!(result.bound_duals[0].abs() < 1e-9);
    assert!(result.bound_duals[1].abs() < 1e-9);
}

/// REFIT-T4: KKT-guard が改善なし更新を revert (既値維持)。
#[test]
fn test_refit_bound_duals_kkt_guard_no_regression() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0_f64];
    let a = CscMatrix::new(0, n);
    let b = vec![];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![],
        bound_duals: vec![2.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result);
    assert!((result.bound_duals[0] - 2.0).abs() < 1e-9, "got {}", result.bound_duals[0]);
}

/// REFIT-T5: 制約あり (A^T y 非ゼロ) で bound_dual 計算。
#[test]
fn test_refit_bound_duals_with_constraint() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0_f64, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![5.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0, 0.0],
        dual_solution: vec![0.0],
        bound_duals: vec![0.0, 0.0],
        objective: 0.0,
        ..SolverResult::default()
    };
    refit_bound_duals_kkt(&problem, &mut result);
    assert!((result.bound_duals[0] - 1.0).abs() < 1e-9);
    assert!(result.bound_duals[1].abs() < 1e-9);
}

/// 不可能な正 Le dual を singleton column interval {0} に projection。
#[test]
fn test_project_duals_from_singleton_columns_clamps_infeasible_positive_le_dual() {
    let n = 2usize;
    let q = CscMatrix::new(n, n);
    let c = vec![0.0_f64, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0_f64, 1.0], 1, n).unwrap();
    let b = vec![0.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0, 0.0],
        dual_solution: vec![5.0],
        bound_duals: vec![0.0, 0.0],
        ..SolverResult::default()
    };

    project_duals_from_singleton_columns(&problem, &mut result);
    refit_bound_duals_kkt(&problem, &mut result);

    assert!(result.dual_solution[0].abs() < 1e-12);
    assert!(result.bound_duals.iter().all(|v| v.abs() < 1e-12));
}

/// lb-only singleton column の lower bound から y を必要値まで引き上げ。
#[test]
fn test_project_duals_from_singleton_columns_respects_lb_only_lower_bound() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![-2.0_f64];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
    let b = vec![0.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![0.0],
        dual_solution: vec![0.0],
        bound_duals: vec![0.0],
        ..SolverResult::default()
    };

    project_duals_from_singleton_columns(&problem, &mut result);
    refit_bound_duals_kkt(&problem, &mut result);

    assert!((result.dual_solution[0] - 2.0).abs() < 1e-12);
    assert!(result.bound_duals[0].abs() < 1e-12);
}

#[test]
fn test_zero_inactive_inequality_duals_clears_slack_le_rows() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![0.0_f64];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, n).unwrap();
    let b = vec![10.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![3.0],
        dual_solution: vec![7.0],
        bound_duals: vec![0.0],
        ..SolverResult::default()
    };

    zero_inactive_inequality_duals(&problem, &mut result);

    assert!(result.dual_solution[0].abs() < 1e-12);
}

#[test]
fn test_refine_dual_projected_gradient_uses_curvature_scaled_step() {
    let n = 1usize;
    let q = CscMatrix::new(n, n);
    let c = vec![-1.0_f64];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1000.0_f64], 1, n).unwrap();
    let b = vec![1.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![1.0e-3],
        dual_solution: vec![0.0],
        bound_duals: vec![0.0],
        ..SolverResult::default()
    };

    refine_dual_projected_gradient(&problem, &mut result, None);

    assert!((result.dual_solution[0] - 1.0e-3).abs() < 1e-9);
}

#[test]
fn test_refine_dual_worst_active_block_updates_row_and_bound_duals_together() {
    let q = CscMatrix::from_triplets(&[1], &[1], &[2.0_f64], 2, 2).unwrap();
    let c = vec![-1.0_f64, 0.0_f64];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 1.0_f64], 1, 2).unwrap();
    let b = vec![1.0_f64];
    let bounds = vec![(0.0_f64, f64::INFINITY), (0.0_f64, f64::INFINITY)];
    let problem =
        QpProblem::new(q, c, a, b, bounds, vec![crate::problem::ConstraintType::Eq]).unwrap();
    let view = crate::qp::ipm_solver::outcome::ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
    };
    let mut result = SolverResult {
        status: SolveStatus::Optimal,
        solution: vec![1.0_f64, 0.0_f64],
        dual_solution: vec![0.0_f64],
        bound_duals: vec![0.0_f64, 0.0_f64],
        ..SolverResult::default()
    };

    let pre = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );
    refine_dual_worst_active_block(&problem, &mut result, None);
    let post = crate::qp::ipm_solver::kkt::kkt_residual_rel(
        &view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    );

    assert!(post < pre);
    assert!(post < 1e-12);
    assert!((result.dual_solution[0] - 1.0).abs() < 1e-9);
    assert!(result.bound_duals[0].abs() < 1e-12);
    assert!((result.bound_duals[1] - 1.0).abs() < 1e-9);
}
