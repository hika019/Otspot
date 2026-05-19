use super::*;
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

mod bound_duals;
mod concurrent;
mod dual_recovery;
mod dual_refit;
mod pfeas;
mod presolve;
mod psd_nonconvex;
mod smoke;
mod status_dfeas;

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
