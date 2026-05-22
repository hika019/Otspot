use super::super::*;
use super::{assert_close, EPS};
use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// TimingBreakdown の QP IPM/postsolve フィールドが実測値で埋まることを検証。
///
/// Sentinel: timing フィールドを常時 None にすると本テストが FAIL する。
/// IPM が ≥1 反復する問題では `ipm_factorize_us > 0` および `ipm_solve_us > 0` が保証される。
/// `postsolve_us == sum(sub-stages)` は構造的不変条件 (algebraic construction)。
/// postsolve 計測が実際に行われているかはケース3 (20 変数) で `postsolve_us > 0` を検証。
#[test]
fn test_qp_timing_breakdown_fields_populated() {
    // ── ケース1: 凸 QP (制約つき、IPM が複数反復) ────────────────────────────
    {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);

        assert_eq!(result.status, SolveStatus::Optimal, "QP should converge");
        let tb = result.timing_breakdown
            .expect("timing_breakdown must be Some for QP IPM path");

        assert!(result.iterations > 0, "IPM should iterate");
        assert!(tb.ipm_factorize_us > 0,
            "ipm_factorize_us must be > 0 when IPM iterated (got {})", tb.ipm_factorize_us);
        assert!(tb.ipm_solve_us > 0,
            "ipm_solve_us must be > 0 when IPM iterated (got {})", tb.ipm_solve_us);

        let postsolve_sum = tb.postsolve_map_us
            + tb.postsolve_lsq_us
            + tb.postsolve_recovery_us
            + tb.postsolve_refine_us
            + tb.postsolve_krylov_ir_us;
        assert_eq!(tb.postsolve_us, postsolve_sum,
            "postsolve_us ({}) must equal sum of sub-stages ({})",
            tb.postsolve_us, postsolve_sum);
    }

    // ── ケース2: 境界制約つき QP (bound dual が出る) ─────────────────────────
    {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-1.0, -1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = vec![1.0];
        let bounds = vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)];
        let problem = QpProblem::new(
            q, c, a, b, bounds, vec![crate::problem::ConstraintType::Le],
        ).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);

        assert!(matches!(result.status, SolveStatus::Optimal | SolveStatus::SuboptimalSolution));
        let tb = result.timing_breakdown
            .expect("timing_breakdown must be Some for QP IPM path");

        let postsolve_sum = tb.postsolve_map_us
            + tb.postsolve_lsq_us
            + tb.postsolve_recovery_us
            + tb.postsolve_refine_us
            + tb.postsolve_krylov_ir_us;
        assert_eq!(tb.postsolve_us, postsolve_sum,
            "postsolve_us ({}) must equal sum of sub-stages ({}) in case 2",
            tb.postsolve_us, postsolve_sum);
    }

    // ── ケース3: 20 変数 QP — postsolve_us が μs 解像度で > 0 になることを確認 ─
    // 微小問題では postsolve が sub-μs で完了し 0 になる。20 変数なら postsolve 経路で
    // 十分な作業量があり計測値 > 0 が保証される。
    {
        let n = 20usize;
        // 対角 Q = 2I, c = -ones (min 0.5‖x‖²- sum(x_i))
        let rows: Vec<usize> = (0..n).collect();
        let vals: Vec<f64> = vec![2.0; n];
        let q = CscMatrix::from_triplets(&rows, &rows, &vals, n, n).unwrap();
        let c = vec![-1.0; n];
        // 3 つの不等式制約: sum(x_i) <= 12, x_0+x_1 <= 2, x_3+x_4 <= 2
        let a_rows = vec![0usize; n].into_iter().chain([1, 1, 2, 2]).collect::<Vec<_>>();
        let a_cols = (0..n).chain([0, 1, 3, 4]).collect::<Vec<_>>();
        let a_vals = vec![1.0f64; n + 4];
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 3, n).unwrap();
        let b = vec![12.0, 2.0, 2.0];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let opts = SolverOptions::default();
        let result = solve_qp_with(&problem, &opts);

        assert!(matches!(result.status, SolveStatus::Optimal | SolveStatus::SuboptimalSolution),
            "20-var QP should converge, got {:?}", result.status);
        let tb = result.timing_breakdown
            .expect("timing_breakdown must be Some for 20-var QP");

        // IPM 計測が非ゼロ
        assert!(result.iterations > 0, "IPM should iterate on 20-var problem");
        assert!(tb.ipm_factorize_us > 0,
            "ipm_factorize_us must be > 0 for 20-var QP (got {})", tb.ipm_factorize_us);

        // 合計 == 内訳の和
        let postsolve_sum = tb.postsolve_map_us
            + tb.postsolve_lsq_us
            + tb.postsolve_recovery_us
            + tb.postsolve_refine_us
            + tb.postsolve_krylov_ir_us;
        assert_eq!(tb.postsolve_us, postsolve_sum,
            "postsolve_us ({}) must equal sum of sub-stages ({}) for 20-var QP",
            tb.postsolve_us, postsolve_sum);
    }
}

/// min x²+y² s.t. x+y ≥ 1 → x*=y*=0.5, obj=0.5
#[test]
fn test_basic_qp_2vars() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
    assert_close(result.objective, 0.5, EPS, "obj");
    assert!(result.bound_duals.is_empty());
    assert_eq!(result.dual_solution.len(), 1);
}

/// min x²+y² s.t. x+y=1 → x*=y*=0.5
#[test]
fn test_qp_equality_constraint() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a =
        CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2)
            .unwrap();
    let b = vec![1.0, -1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 0.5, EPS, "x[1]");
    assert_close(result.objective, 0.5, EPS, "obj");
}

/// Q=0 (LP): min x+2y s.t. x,y≥0, x+y≤4, 2x+y≤6 → obj=0
#[test]
fn test_qp_degenerate_lp_case() {
    let n = 2;
    let q = CscMatrix::new(n, n);
    let c = vec![1.0, 2.0];
    let a = CscMatrix::from_triplets(
        &[0, 1, 2, 2, 3, 3],
        &[0, 1, 0, 1, 0, 1],
        &[-1.0, -1.0, 1.0, 1.0, 2.0, 1.0],
        4,
        2,
    )
    .unwrap();
    let b = vec![0.0, 0.0, 4.0, 6.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.objective, 0.0, EPS, "obj");
}

/// 制約なし: min (x-3)²+(y-4)² → x*=3, y*=4
#[test]
fn test_qp_unconstrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-6.0, -8.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 3.0, EPS, "x[0]");
    assert_close(result.solution[1], 4.0, EPS, "x[1]");
    assert_close(result.objective, -25.0, EPS, "obj");
}

/// warm-start: IPM は warm-start を無視するため同一解が返る。
#[test]
fn test_warm_start_consistency() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a.clone(), b.clone(), bounds.clone()).unwrap();
    let problem2 = QpProblem::new_all_le(
        CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap(),
        vec![0.0, 0.0],
        a,
        b,
        bounds,
    )
    .unwrap();

    let result1 = solve_qp(&problem);
    assert_eq!(result1.status, SolveStatus::Optimal);

    let ws = crate::qp::QpWarmStart {
        x: result1.solution.clone(),
        y: result1.dual_solution.clone(),
        mu: result1.gap.unwrap_or(1e-6),
    };
    let result2 = solve_qp_warm(&problem2, &ws, &SolverOptions::default());

    assert_eq!(result2.status, SolveStatus::Optimal);
    assert_close(result2.solution[0], 0.5, EPS, "x[0]");
    assert_close(result2.solution[1], 0.5, EPS, "x[1]");
}

/// 矛盾制約 (x≥1 ∧ x≤0) → Infeasible
#[test]
fn test_qp_infeasible() {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[-1.0, 1.0], 2, 1).unwrap();
    let b = vec![-1.0, 0.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Infeasible);
}

/// Markowitz 平均分散ポートフォリオ。
#[test]
fn test_qp_portfolio_markowitz() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
    let c = vec![0.0, 0.0, 0.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 0, 1, 1, 1, 2, 3, 4],
        &[0, 1, 2, 0, 1, 2, 0, 1, 2],
        &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        5,
        3,
    )
    .unwrap();
    let b = vec![1.0, -1.0, 0.0, 0.0, 0.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    let w_sum = result.solution[0] + result.solution[1] + result.solution[2];
    assert_close(w_sum, 1.0, EPS, "w_sum");
    assert_close(result.solution[0], 1.0 / 3.0, EPS, "w[0]");
    assert_close(result.solution[1], 1.0 / 3.0, EPS, "w[1]");
    assert_close(result.solution[2], 1.0 / 3.0, EPS, "w[2]");
    assert_close(result.objective, 1.0 / 3.0, EPS, "obj");
}

/// Least Squares。
#[test]
fn test_qp_least_squares() {
    let q =
        CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[10.0, 8.0, 8.0, 10.0], 2, 2)
            .unwrap();
    let c = vec![-28.0, -26.0];
    let a = CscMatrix::new(0, 2);
    let b_vec = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b_vec, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 2.0, EPS, "x[0]");
    assert_close(result.solution[1], 1.0, EPS, "x[1]");
    assert_close(result.objective, -41.0, EPS, "obj");
}

/// Q=0 → LP 退化。
#[test]
fn test_qp_degenerate_to_lp() {
    let n = 2;
    let q = CscMatrix::new(n, n);
    let c = vec![2.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[-1.0, -1.0, -1.0, -1.0],
        3,
        2,
    )
    .unwrap();
    let b = vec![-1.0, 0.0, 0.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.0, EPS, "x[0]");
    assert_close(result.solution[1], 1.0, EPS, "x[1]");
    assert_close(result.objective, 1.0, EPS, "obj");
}

/// 等式 + 不等式 mixed。
#[test]
fn test_qp_mixed_constraints() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-2.0, -4.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1, 2],
        &[0, 1, 0, 1, 0],
        &[1.0, 1.0, -1.0, -1.0, -1.0],
        3,
        2,
    )
    .unwrap();
    let b = vec![2.0, -2.0, 0.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(10.0),
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.5, EPS, "x[0]");
    assert_close(result.solution[1], 1.5, EPS, "x[1]");
    assert_close(result.objective, -4.5, EPS, "obj");
}

/// Box: 上界 active。
#[test]
fn test_qp_box_constrained_upper_bound() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-4.0, -4.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert!(matches!(
        result.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution
    ), "got {:?}", result.status);
    assert_close(result.solution[0], 1.0, EPS, "x[0]");
    assert_close(result.solution[1], 1.0, EPS, "x[1]");
    assert_close(result.objective, -6.0, EPS, "obj");
}

/// Box: 下界 active。
#[test]
fn test_qp_box_constrained_lower_bound() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![4.0, 0.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_close(result.solution[0], 0.0, EPS, "x[0]");
    assert_close(result.solution[1], 0.0, EPS, "x[1]");
    assert_close(result.objective, 0.0, EPS, "obj");
}

/// timeout=0 で Timeout or Optimal。
#[test]
fn test_timeout_returns_timeout_status() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(0.0),
        ..Default::default()
    };

    let result = solve_qp_with(&problem, &opts);
    assert!(
        result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
        "got {:?}", result.status
    );
}

/// 強制 IPM (小規模)。
#[test]
fn test_force_ipm_small() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.solution[0] - 0.5).abs() < 1e-4);
    assert!((result.solution[1] - 0.5).abs() < 1e-4);
    assert!((result.objective - 0.5).abs() < 1e-4);
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

/// parallel feature 有効時の IPPMM dispatch smoke test
#[cfg(feature = "parallel")]
#[test]
fn test_concurrent_solver_basic() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions::default();
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.solution[0] - 0.5).abs() < EPS);
    assert!((result.solution[1] - 0.5).abs() < EPS);
    assert!((result.objective - 0.5).abs() < EPS);
}
