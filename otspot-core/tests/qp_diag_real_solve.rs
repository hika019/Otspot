//! P0 診断の現実 solve 経路スモーク: solve_qp_with のスコープで
//! SolveDiagnostics が公開経路から読め、集計構造 (排他的 bucket・回数分離) が
//! 実測値で破綻していないことを検証する。実時間の厳密一致は要求しない
//! (plan §3: 件数・分類・集計構造のテストまでが unit、wall 対応は診断実行で)。

use otspot_core::problem::SolveStatus;
use otspot_core::qp::diagnostics::AttemptKind;
use otspot_num::sparse::CscMatrix;

fn tiny_qp() -> otspot_core::qp::QpProblem {
    // min 0.5*2*x^2 - 4x = (x-2)^2 - 4, bounds [-10, 10] → x*=2, obj=-4。
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
    otspot_core::qp::QpProblem::new_all_le(
        q,
        vec![-4.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![(-10.0, 10.0)],
    )
    .unwrap()
}

#[test]
fn real_solve_publishes_diagnostics_with_exclusive_buckets() {
    let prob = tiny_qp();
    let result = otspot_core::qp::solve_qp(&prob);
    assert_eq!(result.status, SolveStatus::Optimal);

    let diag = otspot_core::qp::diagnostics::last_qp_solve_diagnostics()
        .expect("solve_qp_with は caller スレッドへ診断を引き継ぐ");
    assert!(!diag.attempts.is_empty(), "実行 attempt が最低 1 件");
    let pub_id = diag.published_attempt().expect("Optimal 採用 attempt あり");

    let published = diag.attempts.iter().find(|a| a.id == pub_id).unwrap();
    assert_eq!(published.kind, AttemptKind::Ladder);
    assert!(published.termination.is_some());
    assert!(published.iterations > 0);

    // 回数: IPM 反復 > 0、factorize 試行は反復以上、materialize+Schur 構築は
    // 組み立て呼出回数 (時間 bucket とは別枠であることは型分離で保証)。
    let counts = diag.total_counts();
    assert!(counts.ipm_iterations > 0);
    assert!(counts.kkt_factorize_attempts >= counts.ipm_iterations);
    assert!(counts.kkt_materialize_calls + counts.kkt_schur_builds >= 1);

    // 集計構造: 全階層で other < 0 (子の合計 > 親) が発生していないこと。
    // 二重計上を導入する revert はここで fail する sentinel。
    assert!(
        !diag.is_aggregation_inconsistent(),
        "集計不整合 attempt 数={}",
        diag.inconsistent_attempt_count()
    );
    assert!(diag.other_us() >= 0);

    // QP 経路で未設定だった TimingBreakdown presolve_us / solve_us が実測値で
    // 埋まっていること (plan §3: never-set fields の填充)。
    let tb = result
        .timing_breakdown
        .expect("QP solve は timing_breakdown を持つ");
    assert!(
        tb.solve_us > 0,
        "solve_us が QP 経路でまだ 0 のまま (未填充の回帰)"
    );
}
