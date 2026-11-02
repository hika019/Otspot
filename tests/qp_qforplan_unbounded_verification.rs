//! QFORPLAN 偽 Unbounded 誤判定の sentinel (eps 依存バグ、2026-11)。
//!
//! ## 症状 (修正前に FAIL した挙動)
//!
//! QFORPLAN (Maros-Meszaros, n=421 m=161) は **bounded + 有限最適解**
//! (HiGHS 1.15.1 オラクル obj = 7.4566314758e9、self-check 済み
//! `/tmp/opencode/inv_eps_anom/oracle4.py`; csv 基準 7.456631460955e9 と相対 ~2e-9)。
//! にもかかわらず eps=1e-8 では `FAIL:Unbounded`、eps=1e-6 では PASS という
//! eps 依存のステータス誤報告が起きていた。原因は (1) 検出器受理の ray/Farkas
//! 検証ゼロ、(2) 未証明 infeasibility incumbent の絶対優先 (no-presolve
//! fallback の真最適点の置換拒否)、(3) 閾値の絶対定数 (fp ノイズを ray と誤認)。
//!
//! ## TDD GREEN 基準
//!
//! - eps=1e-8: Unbounded / Infeasible を返さない (Optimal か、認証不足なら
//!   正直な Stalled 系ステータス)。Optimal のときは目的値はオラクル一致。
//! - eps=1e-6: 従来 PASS の維持 (Optimal + 目的値が基準と相対 1e-6 以内)。
//!
//! 修正コミット (検証義務化 / fallback 優先順位 / 閾値スケール化) を revert すると
//! eps=1e-8 レッグが `Unbounded` で FAIL する。
//!
//! ## CLAUDE.md
//!
//! data 欠落時は SKIP せず panic (検証空白を作らない原則)。

use otspot::io::qps::parse_qps;
use otspot::options::{SolverOptions, Tolerance};
use otspot::problem::{SolveStatus, SolverResult};
use otspot::qp::solve_qp_with;
use std::path::Path;

/// csv 基準値 (data/baseline_objectives/maros_meszaros.csv)。HiGHS オラクルとも
/// 相対 ~2e-9 で一致している。
const QFORPLAN_OBJ: f64 = 7.456_631_460_955e9;

fn solve_qforplan(eps: f64) -> SolverResult {
    let path = Path::new("data/maros_meszaros/QFORPLAN.QPS");
    assert!(
        path.exists(),
        "data missing: {} — Maros-Meszaros suite 必須 (scripts/download_all_bench_data.sh)",
        path.display()
    );
    let problem = parse_qps(path).expect("parse_qps QFORPLAN");
    let mut opts = SolverOptions::default();
    opts.ipm.eps = eps;
    opts.tolerance = Some(Tolerance::Custom(eps));
    solve_qp_with(&problem, &opts)
}

/// eps=1e-8: 検証なしの偽 Unbounded を返してはならない (sentinel 本体)。
#[test]
fn qforplan_eps_1e_8_must_not_report_false_unbounded() {
    let result = solve_qforplan(1e-8);
    assert!(
        !matches!(
            result.status,
            SolveStatus::Unbounded | SolveStatus::Infeasible
        ),
        "false structural verdict at eps=1e-8: {:?} — the problem is bounded \
         (HiGHS 1.15.1 oracle obj=7.4566314758e9); an unverified infeasibility \
         verdict must never be returned",
        result.status
    );
    if result.status == SolveStatus::Optimal {
        assert!(
            ((result.objective - QFORPLAN_OBJ) / QFORPLAN_OBJ).abs() < 1e-6,
            "claimed Optimal but obj {:.6e} deviates from reference {:.6e}",
            result.objective,
            QFORPLAN_OBJ
        );
    }
    eprintln!(
        "[qforplan 1e-8] status={:?} obj={:.6e} iters={}",
        result.status, result.objective, result.iterations
    );
}

/// eps=1e-6: 従来 PASS の維持 (修正前の唯一通っていた経路を壊さない)。
#[test]
fn qforplan_eps_1e_6_stays_optimal() {
    let result = solve_qforplan(1e-6);
    eprintln!(
        "[qforplan 1e-6] status={:?} obj={:.6e} iters={}",
        result.status, result.objective, result.iterations
    );
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "eps=1e-6 regressed from Optimal (obj={:.6e})",
        result.objective
    );
    assert!(
        ((result.objective - QFORPLAN_OBJ) / QFORPLAN_OBJ).abs() < 1e-6,
        "obj {:.6e} deviates from reference {:.6e}",
        result.objective,
        QFORPLAN_OBJ
    );
}
