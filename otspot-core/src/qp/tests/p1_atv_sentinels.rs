//! P1 sentinel: postsolve refine の反復経路で転置構築回数が 0 であること。
//!
//! カウンタは otspot-num の `transpose-count` feature (dev 経由で有効化) による
//! スレッドローカル計測。他テストの転置とはスレッド分離で衝突しない。

use super::super::*;
use super::p1_atv_golden::{fixture_guarded_then_worst_active, fixture_multi_reject_moving};
use otspot_num::sparse::{reset_transpose_constructions, transpose_constructions};

/// 計装の正の対照: `transpose-count` feature が実際に `CscMatrix::transpose()`
/// を観測していることを先に固定する。計装が外れると zero-count sentinel が
/// 黙って通るため、その silent-pass をここに紐付ける。
#[test]
fn p1_transpose_counter_observable() {
    reset_transpose_constructions();
    let a = otspot_num::sparse::CscMatrix::identity(2);
    let _t = a.transpose();
    assert_eq!(
        transpose_constructions(),
        1,
        "transpose-count instrumentation must observe CscMatrix::transpose(); \
         if this breaks, p1_postsolve_refine_constructs_zero_transposes \
         silently passes"
    );
}

/// P1 sentinel: 反復的后処理 (projected gradient refine / worst-active) が
/// 転置を 1 回も構築しない。revert (objective 内・曲率・worst-active 冒頭の
/// `transpose()` 復活) ではカウンタ > 0 で必ず fail する。
#[test]
fn p1_postsolve_refine_constructs_zero_transposes() {
    // line search 複数回不受理 (objective 評価 ≥ 17 回) を含む経路を走らせる。
    let (problem, mut result) = fixture_multi_reject_moving();
    reset_transpose_constructions();
    refine_dual_projected_gradient(&problem, &mut result, &[], None);
    assert_eq!(
        transpose_constructions(),
        0,
        "refine_dual_projected_gradient must not build a transpose per \
         objective/curvature evaluation (old path constructs 18 here, \
         independently re-measured by base-code revert)"
    );

    let (problem, mut result) = fixture_guarded_then_worst_active();
    reset_transpose_constructions();
    refine_dual_projected_gradient(&problem, &mut result, &[], None);
    assert_eq!(transpose_constructions(), 0, "PG must stay transpose-free");
    refine_dual_worst_active_block(&problem, &mut result, &[], None);
    assert_eq!(
        transpose_constructions(),
        0,
        "refine_dual_worst_active_block must not build a transpose at entry \
         (old path: 1 per call)"
    );
}
