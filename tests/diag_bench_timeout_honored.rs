//! bench harness の Timeout→Optimal silent wrap 撤廃 regression sentinel (task #52).
//!
//! 真因 (task #46 eps-stuck-investigator 観測):
//!   qps_benchmark / bench_qplib が `Timeout + 解あり` を SolveStatus::Optimal に
//!   silent 格上げ → Primal 半 deadline の低品質 incumbent が品質判定経路 (pfeas/dfeas)
//!   へ流れ込み PFEAS_FAIL として表示 → 真因 (Timeout) が観測者に隠れる。
//!
//! 期待挙動: Timeout は格上げ対象外、Timeout のまま bench に報告する。
//! SuboptimalSolution / LocallyOptimal は据置 (本来 KKT 近傍の正規 status)。

use solver::bench_utils::{apply_bench_status_promotion, BenchPromotionPolicy};
use solver::problem::{SolveStatus, SolverResult};

fn make(status: SolveStatus, solution: Vec<f64>, objective: f64) -> SolverResult {
    SolverResult { status, solution, objective, ..Default::default() }
}

/// regression sentinel: Timeout + 有効解 でも Optimal 格上げしない (task #52 真因対処)。
#[test]
fn timeout_with_solution_stays_timeout_qps_benchmark() {
    let r_in = make(SolveStatus::Timeout, vec![0.1, 0.2, 0.3], 1.0);
    let r_out = apply_bench_status_promotion(r_in, 3, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(
        r_out.status,
        SolveStatus::Timeout,
        "Timeout は Optimal に silent 格上げしてはいけない (task #46/#52)"
    );
}

#[test]
fn timeout_with_solution_stays_timeout_bench_qplib() {
    let r_in = make(SolveStatus::Timeout, vec![0.0; 5], -1.5);
    let r_out = apply_bench_status_promotion(r_in, 5, BenchPromotionPolicy::BenchQplib);
    assert_eq!(r_out.status, SolveStatus::Timeout);
}

#[test]
fn suboptimal_with_valid_solution_promoted_to_optimal() {
    let r_in = make(SolveStatus::SuboptimalSolution, vec![1.0; 4], 2.0);
    let r_out = apply_bench_status_promotion(r_in, 4, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(r_out.status, SolveStatus::Optimal);
}

#[test]
fn locally_optimal_with_valid_solution_promoted_to_optimal() {
    let r_in = make(SolveStatus::LocallyOptimal, vec![1.0; 4], 2.0);
    let r_out = apply_bench_status_promotion(r_in, 4, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(r_out.status, SolveStatus::Optimal);
}

#[test]
fn empty_solution_blocks_promotion() {
    let r_in = make(SolveStatus::SuboptimalSolution, vec![], 1.0);
    let r_out = apply_bench_status_promotion(r_in, 3, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(r_out.status, SolveStatus::SuboptimalSolution);
}

#[test]
fn solution_length_mismatch_blocks_promotion() {
    let r_in = make(SolveStatus::SuboptimalSolution, vec![0.0; 2], 1.0);
    let r_out = apply_bench_status_promotion(r_in, 3, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(r_out.status, SolveStatus::SuboptimalSolution);
}

#[test]
fn nan_objective_blocks_qplib_promotion() {
    let r_in = make(SolveStatus::SuboptimalSolution, vec![0.0; 3], f64::NAN);
    let r_out = apply_bench_status_promotion(r_in, 3, BenchPromotionPolicy::BenchQplib);
    assert_eq!(
        r_out.status,
        SolveStatus::SuboptimalSolution,
        "bench_qplib は obj 非有限なら格上げしない (obj 照合できないため)"
    );
}

#[test]
fn inf_objective_blocks_qplib_promotion() {
    let r_in = make(SolveStatus::LocallyOptimal, vec![0.0; 3], f64::INFINITY);
    let r_out = apply_bench_status_promotion(r_in, 3, BenchPromotionPolicy::BenchQplib);
    assert_eq!(r_out.status, SolveStatus::LocallyOptimal);
}

#[test]
fn optimal_passes_through_unchanged() {
    let r_in = make(SolveStatus::Optimal, vec![0.5; 2], 1.0);
    let r_out = apply_bench_status_promotion(r_in, 2, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(r_out.status, SolveStatus::Optimal);
}

#[test]
fn infeasible_unchanged() {
    let r_in = make(SolveStatus::Infeasible, vec![], f64::INFINITY);
    let r_out = apply_bench_status_promotion(r_in, 2, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(r_out.status, SolveStatus::Infeasible);
}

#[test]
fn unbounded_unchanged() {
    let r_in = make(SolveStatus::Unbounded, vec![], f64::NEG_INFINITY);
    let r_out = apply_bench_status_promotion(r_in, 2, BenchPromotionPolicy::BenchQplib);
    assert_eq!(r_out.status, SolveStatus::Unbounded);
}

#[test]
fn max_iterations_not_promoted() {
    // MaxIterations 系の status (もし存在しても) は格上げ対象外であることを念のため確認。
    // Timeout と同様、収束未達 status は honest に報告すべき。
    let r_in = make(SolveStatus::NumericalError, vec![1.0; 3], 1.0);
    let r_out = apply_bench_status_promotion(r_in, 3, BenchPromotionPolicy::QpsBenchmark);
    assert_eq!(r_out.status, SolveStatus::NumericalError);
}
