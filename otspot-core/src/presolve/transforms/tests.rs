//! Unit + KKT-roundtrip tests covering presolve transforms.

use super::*;
use crate::error::SolverError;
use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;

#[allow(clippy::too_many_arguments)]
fn make_lp_general(
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    nrows: usize,
    ncols: usize,
    b: Vec<f64>,
    cts: Vec<ConstraintType>,
    bounds: Vec<(f64, f64)>,
) -> LpProblem {
    let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
    LpProblem::new_general(c, a, b, cts, bounds, None).unwrap()
}

fn make_lp(
    c: Vec<f64>,
    rows: &[usize],
    cols: &[usize],
    vals: &[f64],
    nrows: usize,
    ncols: usize,
    b: Vec<f64>,
) -> LpProblem {
    let n = c.len();
    make_lp_general(
        c,
        rows,
        cols,
        vals,
        nrows,
        ncols,
        b,
        vec![ConstraintType::Le; nrows],
        vec![(0.0, f64::INFINITY); n],
    )
}

// -----------------------------------------------------------
// 1. Fixed variable removal
// -----------------------------------------------------------
#[test]
fn test_fixed_variable_removal() {
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(2.0, 2.0), (0.0, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert_eq!(result.reduced_problem.num_vars, 0);
    assert_eq!(result.reduced_problem.num_constraints, 0);
    assert!(result.was_reduced);
    assert!((result.obj_offset - 2.0).abs() < 1e-10);
}

#[test]
fn test_fixed_infeasible() {
    // lb > ub is now rejected at construction time (InvalidBounds), not by presolve.
    let a = CscMatrix::new(0, 1);
    let res = LpProblem::new_general(vec![1.0], a, vec![], vec![], vec![(3.0, 2.0)], None);
    assert!(
        matches!(res, Err(SolverError::InvalidBounds { index: 0, lb, ub }) if lb == 3.0 && ub == 2.0),
        "lb > ub must be rejected at construction"
    );
}

#[test]
fn test_presolve_detects_lb_gt_ub() {
    // Construction now rejects lb > ub, but presolve's bound-consistency check
    // (step1_fixed_variable) is still reachable in production when a transform
    // *tightens* a valid bound past its opposite. Inject lb > ub post-construction
    // (valid build → mutate public field) to keep that detection path covered.
    let mut lp = make_lp_general(
        vec![1.0],
        &[],
        &[],
        &[],
        0,
        1,
        vec![],
        vec![],
        vec![(0.0, 1.0)],
    );
    lp.bounds[0] = (3.0, 2.0); // lb > ub injected after the constructor check
    assert!(
        matches!(run_presolve(&lp, None), Err(PresolveStatus::Infeasible)),
        "presolve must report Infeasible for lb > ub bounds"
    );
}

// -----------------------------------------------------------
// 2. Empty row/column removal
// -----------------------------------------------------------
#[test]
fn test_empty_row_feasible() {
    let lp = make_lp_general(
        vec![1.0],
        &[1],
        &[0],
        &[1.0],
        2,
        1,
        vec![5.0, 3.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert_eq!(result.reduced_problem.num_constraints, 0);
}

#[test]
fn test_empty_row_infeasible() {
    let lp = make_lp_general(
        vec![1.0],
        &[1],
        &[0],
        &[1.0],
        2,
        1,
        vec![-1.0, 3.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, f64::INFINITY)],
    );
    assert!(matches!(
        run_presolve(&lp, None),
        Err(PresolveStatus::Infeasible)
    ));
}

#[test]
fn test_empty_column_min_with_finite_lb() {
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        CscMatrix::new(0, 2),
        vec![],
        vec![],
        vec![(0.0, f64::INFINITY), (1.0, f64::INFINITY)],
        None,
    )
    .unwrap();
    let result = run_presolve(&lp, None).unwrap();
    assert_eq!(result.reduced_problem.num_vars, 0);
    assert!((result.obj_offset - 1.0).abs() < 1e-10);
}

#[test]
fn test_empty_column_unbounded() {
    let lp = LpProblem::new_general(
        vec![-1.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![],
        vec![(0.0, f64::INFINITY)],
        None,
    )
    .unwrap();
    assert!(matches!(
        run_presolve(&lp, None),
        Err(PresolveStatus::Unbounded)
    ));
}

// -----------------------------------------------------------
// 3. Singleton row (Eq)
// -----------------------------------------------------------
#[test]
fn test_singleton_row_eq() {
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 1, 1],
        &[0, 0, 1],
        &[2.0, 1.0, 1.0],
        2,
        2,
        vec![6.0, 10.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert_eq!(result.reduced_problem.num_vars, 0);
    assert_eq!(result.reduced_problem.num_constraints, 0);
    assert!((result.obj_offset - 3.0).abs() < 1e-10);
}

#[test]
fn test_singleton_row_infeasible() {
    let lp = make_lp_general(
        vec![1.0],
        &[0],
        &[0],
        &[2.0],
        1,
        1,
        vec![6.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 1.0)],
    );
    assert!(matches!(
        run_presolve(&lp, None),
        Err(PresolveStatus::Infeasible)
    ));
}

// -----------------------------------------------------------
// 4. Redundant constraint removal
// -----------------------------------------------------------
#[test]
fn test_redundant_le() {
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0, 1, 2],
        &[0, 1, 0, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        2,
        vec![10.0, 3.0, 3.0],
        vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert_eq!(
        result.reduced_problem.num_constraints, 0,
        "all 3 constraints should be redundant"
    );
    assert_eq!(
        result.reduced_problem.num_vars, 0,
        "vars removed as empty cols after constraints gone"
    );

    // Use negative cost so dual fixing (Step 11) cannot collapse the LP:
    // c < 0 with Le a > 0 disqualifies neg-pressure, c < 0 fails pos-pressure cost gate.
    let lp2 = make_lp_general(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![2.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 10.0)],
    );
    let result2 = run_presolve(&lp2, None).unwrap();
    assert_eq!(
        result2.reduced_problem.num_constraints, 1,
        "x1+x2<=2 is not redundant"
    );
}

// -----------------------------------------------------------
// 5. Bounds tightening
// -----------------------------------------------------------
#[test]
fn test_bounds_tightening() {
    // Use negative cost: Step 11 dual fixing (which collapses x→0 when c≥0
    // and all Le coefs ≥0) does not apply here, so we observe pure Step 5.
    let lp = make_lp_general(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 10.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    let _ = result.was_reduced;
    assert_eq!(result.reduced_problem.num_vars, 2);
}

#[test]
fn test_bounds_tightening_negative_coeff_le_feasible() {
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, -1.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 3.0)],
    );
    assert!(
        run_presolve(&lp, None).is_ok(),
        "x - y <= 5 should be feasible"
    );
}

#[test]
fn test_bounds_tightening_negative_coeff_ge_feasible() {
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[-1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 5.0), (0.0, 8.0)],
    );
    assert!(
        run_presolve(&lp, None).is_ok(),
        "-x + y >= 3 should be feasible"
    );
}

// -----------------------------------------------------------
// Roundtrip
// -----------------------------------------------------------
#[test]
fn test_presolve_singleton_le_netlib_like() {
    let lp = make_lp(
        vec![-1.0, -1.0, -1.0],
        &[0, 0, 0, 1, 2, 3],
        &[0, 1, 2, 0, 1, 2],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        4,
        3,
        vec![4.0, 3.0, 3.0, 3.0],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced);
    // Singleton Le rows (1-3) absorbed into bounds; row 0 remains with tightened bounds.
    assert_eq!(result.reduced_problem.num_vars, 3);
    assert_eq!(result.reduced_problem.num_constraints, 1);
}

#[test]
fn test_pre001_deadline_fires_immediately() {
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(2.0, 2.0), (0.0, f64::INFINITY)],
    );
    let expired = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let result = run_presolve(&lp, Some(expired)).unwrap();
    assert!(
        !result.was_reduced,
        "期限切れ deadline では early-exit し was_reduced=false を返すこと"
    );
}

// -----------------------------------------------------------
// R6: Doubleton equation
// -----------------------------------------------------------
#[test]
fn presolve_doubleton_eq_basic() {
    // min x + y + z
    // s.t. x + y = 3        (Eq doubleton)
    //      x + y + z <= 10
    //      x in [0,5], y in [0,5], z in [0, inf)
    // x を消去 (pivot=x, others=y), 残: y, z, 制約: (y, z への変換)
    let lp = make_lp_general(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 1, 1, 1],
        &[0, 1, 0, 1, 2],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![3.0, 10.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0), (0.0, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    // x または y のいずれかが消去される。残り 2 vars, 1 制約 (or さらに縮小)
    assert!(result.was_reduced);
    // postsolve_stack に LinearSubstitution が含まれていることを確認
    let has_subst = result
        .postsolve_stack
        .iter()
        .any(|s| matches!(s, PostsolveStep::LinearSubstitution { .. }));
    assert!(
        has_subst,
        "Doubleton equation should produce LinearSubstitution"
    );
}

#[test]
fn presolve_doubleton_eq_solution_consistency() {
    // 同じ問題を presolve あり / なしで解いた解の "目的値" を obj_offset 含めて比較する
    // ここでは presolve のみ実行し、reduced + offset が元の最適値に一致するロジック検証
    //
    // min x + y
    // s.t. x + y = 4
    //      x in [0,3], y in [0,3]
    // 最適解: 任意の x+y=4 (例: x=1,y=3 or x=3,y=1)。最適値 = 4
    // presolve: x = 4 - y, x in [0,3] → y in [1,4] ∩ [0,3] = [1,3]
    //   reduced: min (4-y) + y = 4 over y in [1,3] → 縮約後 c[y]=0, offset=4
    //   reduced は 0変数 / 0制約 になり得る (cy=1-1=0, 制約はx+y<=. ここでは無いので)
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![4.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    // 目的値の総和は 4 (= obj_offset + reduced c^T x)
    // reduced c[y] = 0 (1 - 1*1 = 0), offset = 4 (1*4/1 = 4)
    assert!((result.obj_offset - 4.0).abs() < 1e-10, "obj_offset = 4");
}

#[test]
fn presolve_doubleton_eq_infeasible() {
    // x + y = 10, x in [0,3], y in [0,3] → 最大 6 < 10 → Infeasible
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![10.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 3.0), (0.0, 3.0)],
    );
    let res = run_presolve(&lp, None);
    assert!(matches!(res, Err(PresolveStatus::Infeasible)));
}

// -----------------------------------------------------------
// R15: Free variable substitution
// -----------------------------------------------------------
#[test]
fn presolve_free_var_subst_basic() {
    // min x + y + z
    // s.t. x + y + z = 5     (Eq)
    //      x + y <= 10
    //      z is free, x in [0,10], y in [0,10]
    // → z = 5 - x - y を Eq から代入 → Eq 消去、他制約に z 出現なし → 影響なし
    // 結果: vars = (x, y) のみ (z 消去), 制約 = 1 (Le)
    let lp = make_lp_general(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 0, 1, 1],
        &[0, 1, 2, 0, 1],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![5.0, 10.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced);
    let has_subst = result
        .postsolve_stack
        .iter()
        .any(|s| matches!(s, PostsolveStep::LinearSubstitution { .. }));
    assert!(
        has_subst,
        "Free var substitution should produce LinearSubstitution"
    );
    // z が消去されているはず
    assert!(
        result.col_map[2].is_none(),
        "z (col 2) should be eliminated"
    );
}

#[test]
fn presolve_free_var_subst_multi_constraint() {
    // min x + y + z
    // s.t. x + z = 4          (Eq, z 含む)
    //      y + z = 5          (Eq, z 含む)
    //      x in [0,10], y in [0,10], z free
    // → z = 4 - x を Eq#0 から代入 → Eq#0 消去, Eq#1: y + (4 - x) = 5 → y - x = 1
    let lp = make_lp_general(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 1, 1],
        &[0, 2, 1, 2],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![4.0, 5.0],
        vec![ConstraintType::Eq, ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced);
    // z は消去される. 制約は 1 (Eq) 残り
    assert!(result.col_map[2].is_none());
}

// -----------------------------------------------------------
// R5: Free singleton column
// -----------------------------------------------------------
#[test]
fn presolve_doubleton_dual_recovery_eq_le() {
    // Eq doubleton (x1+x2=6) + Le (x2<=5)。pivot=x1 で x1 を消去後、
    // dual 復元式: y_piv = (c_orig - Σ_{i ≠ piv} A_ij_orig * y_i) / pivot で
    // y[0] = 1.0 になることを確認。
    let lp = make_lp_general(
        vec![1.0, 2.0],
        &[0, 0, 1],
        &[0, 1, 1],
        &[1.0, 1.0, 1.0],
        2,
        2,
        vec![6.0, 5.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    // postsolve_stack に LinearSubstitution が含まれ、その c_orig が正しく保存されている
    let lin = result.postsolve_stack.iter().find_map(|s| match s {
        PostsolveStep::LinearSubstitution { c_orig, pivot, .. } => Some((*c_orig, *pivot)),
        _ => None,
    });
    assert!(lin.is_some(), "LinearSubstitution expected");
    let (c_orig, pivot) = lin.unwrap();
    // pivot=1 (x1 の係数), c_orig = c_x1 = 1
    assert!((pivot - 1.0).abs() < 1e-12);
    assert!(
        (c_orig - 1.0).abs() < 1e-12,
        "c_orig must capture pre-distribution c[x1]=1"
    );
}

#[test]
fn presolve_free_singleton_col_basic() {
    // min x + y + z
    // s.t. x + y >= 3
    //      x + z = 7        (Eq, z singleton 列 = z は他制約に出ない)
    //      x in [0,10], y in [0,10], z free
    // R5 (も R15 も両方適用条件) → z 消去 + Eq#1 消去
    let lp = make_lp_general(
        vec![1.0, 1.0, 1.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 2],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![3.0, 7.0],
        vec![ConstraintType::Ge, ConstraintType::Eq],
        vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced);
    assert!(result.col_map[2].is_none(), "z should be eliminated");
    assert!(result.row_map[1].is_none(), "Eq row should be eliminated");
}

// -----------------------------------------------------------
// Round-trip KKT tests: presolve→solve→postsolve cycle が原問題で
// primal/dual/objective を全て満たすことを assert する。
//
// 既存 test 群は run_presolve の構造的副作用 (num_vars, postsolve_stack,
// col_map) のみ検証していたため、postsolve の dual recovery が崩れても
// 検出できなかった (perold 等で実際に bug を漏らした)。
// -----------------------------------------------------------
mod roundtrip_kkt {
    use super::*;
    use crate::test_kkt::assert_kkt_optimal;

    /// Doubleton Eq の round-trip: x+y=4, x∈[0,3], y∈[0,3], min x+y → obj=4
    #[test]
    fn roundtrip_doubleton_eq_simple() {
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![4.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 3.0), (0.0, 3.0)],
        );
        assert_kkt_optimal(&lp, 4.0, "roundtrip_doubleton_eq_simple");
    }

    /// Doubleton Eq + 異なる係数: 2x+3y=12, x∈[0,4], y∈[0,4], min x+2y
    /// 代入: x = 6 - 1.5y, feasible: 4/3 ≤ y ≤ 4
    /// obj = (6-1.5y) + 2y = 6 + 0.5y → min y=4/3, x=4, obj = 6+2/3 = 20/3
    #[test]
    fn roundtrip_doubleton_eq_nonunit_coeffs() {
        let lp = make_lp_general(
            vec![1.0, 2.0],
            &[0, 0],
            &[0, 1],
            &[2.0, 3.0],
            1,
            2,
            vec![12.0],
            vec![ConstraintType::Eq],
            vec![(0.0, 4.0), (0.0, 4.0)],
        );
        assert_kkt_optimal(&lp, 20.0 / 3.0, "roundtrip_doubleton_eq_nonunit_coeffs");
    }

    /// Free var substitution: z free + Eq row で z を消去後 KKT 整合
    /// min x+y+z, x+y+z=5, x+y<=10, x,y∈[0,10], z free → z=5-x-y, obj=5
    #[test]
    fn roundtrip_free_var_subst() {
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0],
            &[0, 0, 0, 1, 1],
            &[0, 1, 2, 0, 1],
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            2,
            3,
            vec![5.0, 10.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        );
        assert_kkt_optimal(&lp, 5.0, "roundtrip_free_var_subst");
    }

    /// Free singleton col: z は singleton 列 + free。Eq 1 + Ge 1 の混在で
    /// postsolve が free col + Ge dual の符号慣例を正しく復元するか。
    /// min x+y+z, x+y>=3, x+z=7, x,y∈[0,10], z free → x=3, y=0, z=4 obj=7
    #[test]
    fn roundtrip_free_singleton_col() {
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 2],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            3,
            vec![3.0, 7.0],
            vec![ConstraintType::Ge, ConstraintType::Eq],
            vec![(0.0, 10.0), (0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY)],
        );
        // x+y>=3, x+z=7. min x+y+z = x+y + (7-x) = y+7 → minimize y → y=0
        // y=0: x>=3, z=7-x. min x+0+7-x = 7. 任意 x ∈ [3,7] feasible. obj=7
        assert_kkt_optimal(&lp, 7.0, "roundtrip_free_singleton_col");
    }

    /// Singleton row + bounds tightening: x0 = 5 fix で SingletonRow 経由
    /// y_0 を bound-aware に復元する経路 (perold class proxy)。
    /// min x0+x1+x2, x0=5 (Eq singleton), x1+x2=4 (Eq), x1∈[0,3], x2∈[0,3]
    /// → x0=5, x1+x2=4 minimize → 任意組合せ、obj = 5+4=9
    #[test]
    fn roundtrip_singleton_row_eq_with_doubleton() {
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0],
            &[0, 1, 1],
            &[0, 1, 2],
            &[1.0, 1.0, 1.0],
            2,
            3,
            vec![5.0, 4.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, 10.0), (0.0, 3.0), (0.0, 3.0)],
        );
        assert_kkt_optimal(&lp, 9.0, "roundtrip_singleton_row_eq_with_doubleton");
    }

    /// Redundant Le row + active Eq: Redundant が削除されても残りの Eq
    /// で KKT が成立し、削除行の y_i は bound-aware default (= 0) で
    /// 矛盾ないことを round-trip で検証。
    #[test]
    fn roundtrip_redundant_le_with_active_eq() {
        // x1+x2 <= 100 (Le, redundant: x1∈[0,3], x2∈[0,3])
        // x1+x2 = 4 (Eq, active)
        // min 2x1+x2, x1∈[0,3], x2∈[0,3]
        // → x1=1, x2=3 (cost x1 を最小化、x2 が cheaper): obj = 2+3 = 5
        //   x1=3, x2=1: obj = 6+1=7
        //   x1=0, x2=4: infeasible (x2>3)
        //   x1=1, x2=3: obj=5 (★)
        let lp = make_lp_general(
            vec![2.0, 1.0],
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            2,
            2,
            vec![100.0, 4.0],
            vec![ConstraintType::Le, ConstraintType::Eq],
            vec![(0.0, 3.0), (0.0, 3.0)],
        );
        assert_kkt_optimal(&lp, 5.0, "roundtrip_redundant_le_with_active_eq");
    }

    /// 全 transform 混在: doubleton + free var + singleton + redundant
    /// (presolve→postsolve の全体パスの cross 検証)
    #[test]
    fn roundtrip_mixed_transforms() {
        // min x1 + x2 + x3 + x4
        // x1 + x2     = 3    (Eq doubleton, x1∈[0,2], x2∈[0,2] active)
        // x3 + x4     = 2    (Eq doubleton, x3 free, x4∈[0,5])
        // x1 + x3    <= 100  (Le redundant)
        // → x1+x2=3 (x1=1,x2=2 や x1=2,x2=1)、x3+x4=2 (任意)、obj = 3+2 = 5
        let lp = make_lp_general(
            vec![1.0, 1.0, 1.0, 1.0],
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 2, 3, 0, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            3,
            4,
            vec![3.0, 2.0, 100.0],
            vec![ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Le],
            vec![
                (0.0, 2.0),
                (0.0, 2.0),
                (f64::NEG_INFINITY, f64::INFINITY),
                (0.0, 5.0),
            ],
        );
        assert_kkt_optimal(&lp, 5.0, "roundtrip_mixed_transforms");
    }

    /// Le → Ge round-trip: Ge は postsolve で符号反転、dual 符号慣例を
    /// 正しく復元できないと dfeas_rel_bound が劣化。
    #[test]
    fn roundtrip_ge_constraint_dual_sign() {
        // min x+y, x+y >= 3, x∈[0,5], y∈[0,5] → x+y=3 (任意)、obj=3
        let lp = make_lp_general(
            vec![1.0, 1.0],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![3.0],
            vec![ConstraintType::Ge],
            vec![(0.0, 5.0), (0.0, 5.0)],
        );
        assert_kkt_optimal(&lp, 3.0, "roundtrip_ge_constraint_dual_sign");
    }

    /// singleton Le (non-binding) の dual が complementarity により 0 であることを確認。
    #[test]
    fn test_singleton_le_nonbinding_dual_zero() {
        // min -x, s.t. x <= 10 (singleton Le, non-binding), x + 0*y <= 5, x,y in [0,5].
        // Presolve: x <= 10 doesn't tighten x_ub (already 5 < 10), row removed.
        // Optimal: x=5, y=0, obj=-5.
        // Row 0 slack = 10 - 5 = 5 > 0, so dual must be 0 (complementarity).
        let lp = make_lp_general(
            vec![-1.0, 0.0],
            &[0, 1, 1],
            &[0, 0, 1],
            &[1.0, 1.0, 0.0],
            2,
            2,
            vec![10.0, 5.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 5.0), (0.0, 5.0)],
        );
        assert_kkt_optimal(&lp, -5.0, "singleton_le_nonbinding_dual_zero");
    }
}

// -----------------------------------------------------------
// Step 7 free-variable substitution fill-in cascade (cont1/4/11 hang repro)
// -----------------------------------------------------------
//
// REGRESSION GUARD — currently EXPECTED TO FAIL.
//
// `step7_free_var_substitution` hangs on cont* (measured 2026-06-01: burns full
// 30s deadline after only 5708/~40k eliminations; `fill_in_exceeds_budget` never
// fired because its budget scales with the current row count, so an already-dense
// column is allowed to densify further).
//
// This synthetic reproduces the cascade at small scale: hub vars a_0..a_N chained
// by shared pivot rows; eliminating a_0 densifies all R load rows repeatedly.
// Cost grows super-linearly in R (measured: doubling R ≈ 3.5× time).
//
// EXPECTED AFTER FIX: must finish under MAX_PRESOLVE_SECS (~2.6s release → FAILS).
// SAFETY_DEADLINE is a non-hang guard; assertion is on wall-clock.
#[test]
fn test_step7_free_var_fillin_must_not_blow_up() {
    use std::time::{Duration, Instant};

    // Sized so the buggy path runs ~2.6s (release) — clearly over the bound —
    // while the safety deadline keeps the test bounded if it ever hangs.
    const N_HUBS: usize = 1500; // free chained variables a_0..a_N
    const N_LOADS: usize = 9000; // load rows that densify around the chain
    const MAX_PRESOLVE_SECS: f64 = 2.0; // post-fix budget (assertion; bad path ~2.6s)
    const SAFETY_DEADLINE_SECS: f64 = 8.0; // non-hang guard only

    let n_hub = N_HUBS + 1;
    let idx_g = n_hub; // single shared bounded variable
    let idx_d0 = n_hub + 1; // first load slack
    let ncols = n_hub + 1 + N_LOADS;
    let nrows = N_HUBS + N_LOADS; // N_HUBS pivot Eq rows + N_LOADS load Le rows

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    // Pivot rows: 2*a_i - a_{i+1} - 0.5*g = 0  (3 terms ⇒ not a doubleton, so
    // step6 skips it; a_i has the largest |coef| ⇒ step7 picks it to eliminate a_i).
    for i in 0..N_HUBS {
        rows.push(i);
        cols.push(i);
        vals.push(2.0);
        rows.push(i);
        cols.push(i + 1);
        vals.push(-1.0);
        rows.push(i);
        cols.push(idx_g);
        vals.push(-0.5);
    }
    // Load rows: a_0 + d_r <= 2.
    for r in 0..N_LOADS {
        let row = N_HUBS + r;
        rows.push(row);
        cols.push(0);
        vals.push(1.0);
        rows.push(row);
        cols.push(idx_d0 + r);
        vals.push(1.0);
    }

    let a = CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).unwrap();
    let c = vec![0.0; ncols];
    let mut b = vec![0.0; nrows];
    let mut cts = vec![ConstraintType::Eq; nrows];
    let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); ncols]; // hubs free
    bounds[idx_g] = (0.0, 1.0);
    for r in 0..N_LOADS {
        b[N_HUBS + r] = 2.0;
        cts[N_HUBS + r] = ConstraintType::Le;
        bounds[idx_d0 + r] = (0.0, 1.0);
    }
    let lp = LpProblem::new_general(c, a, b, cts, bounds, None).unwrap();

    let deadline = Some(Instant::now() + Duration::from_secs_f64(SAFETY_DEADLINE_SECS));
    let t0 = Instant::now();
    let _ = run_presolve(&lp, deadline).expect("presolve must not error on this LP");
    let elapsed = t0.elapsed().as_secs_f64();

    assert!(
        elapsed < MAX_PRESOLVE_SECS,
        "step7 free-var substitution fill-in cascade: presolve took {:.3}s on a \
         {}-var / {}-constraint LP (budget {:.1}s). The free-var elimination cost \
         grows super-linearly as columns densify — see fill_in_exceeds_budget / \
         eliminate_variable_via_eq_row in presolve/transforms.",
        elapsed,
        ncols,
        nrows,
        MAX_PRESOLVE_SECS
    );
}

// -----------------------------------------------------------
// Singleton Le/Ge — end-to-end
// -----------------------------------------------------------

#[test]
fn test_singleton_le_reduces_and_round_trips() {
    // min -x - y, s.t. 3x <= 6 (singleton Le), x + y <= 5, x,y in [0,5].
    // Singleton Le tightens x_ub to 2. Row 1 not redundant: max=2+5=7>5.
    let lp = make_lp_general(
        vec![-1.0, -1.0],
        &[0, 1, 1],
        &[0, 0, 1],
        &[3.0, 1.0, 1.0],
        2,
        2,
        vec![6.0, 5.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced, "singleton Le must reduce the problem");
    assert_eq!(
        result.reduced_problem.num_constraints, 1,
        "singleton Le row removed"
    );
}

#[test]
fn test_singleton_ge_reduces_and_round_trips() {
    // min -x - y, s.t. 2x >= 4 (singleton Ge), x + y <= 5, x,y in [0,5].
    // Singleton Ge tightens x_lb to 2. Row 1 not redundant: max=5+5=10>5.
    let lp = make_lp_general(
        vec![-1.0, -1.0],
        &[0, 1, 1],
        &[0, 0, 1],
        &[2.0, 1.0, 1.0],
        2,
        2,
        vec![4.0, 5.0],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 5.0), (0.0, 5.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced, "singleton Ge must reduce the problem");
    assert_eq!(
        result.reduced_problem.num_constraints, 1,
        "singleton Ge row removed"
    );
}

#[test]
fn test_singleton_le_infeasible_at_presolve() {
    // 2x <= 1, x in [3, 5] → implied ub = 0.5 < lb = 3 → infeasible.
    let lp = make_lp_general(
        vec![1.0],
        &[0],
        &[0],
        &[2.0],
        1,
        1,
        vec![1.0],
        vec![ConstraintType::Le],
        vec![(3.0, 5.0)],
    );
    assert!(matches!(
        run_presolve(&lp, None),
        Err(PresolveStatus::Infeasible)
    ));
}

// -----------------------------------------------------------
// Forcing row — end-to-end
// -----------------------------------------------------------

#[test]
fn test_forcing_le_all_positive_reduces() {
    // min -x - y, s.t. x + y <= 0, -x + z <= 5, x in [0,1], y in [0,2], z in [0,6].
    // Forcing: min = 0+0 = 0 >= 0. x→0, y→0. Row 0 removed, x,y fixed.
    // Row 1 becomes -0+z <= 5 (singleton Le, tightens z_ub to 5). Row 1 removed.
    // z with c=-0 but row gone → empty col. c(z)=0 so any value: fix at lb=0.
    // obj_offset = 0 (x=0, y=0, z=0).
    let lp = make_lp_general(
        vec![-1.0, -1.0, 0.0],
        &[0, 0, 1, 1],
        &[0, 1, 0, 2],
        &[1.0, 1.0, -1.0, 1.0],
        2,
        3,
        vec![0.0, 5.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 2.0), (0.0, 6.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced);
    // Forcing + subsequent reductions eliminate everything.
    assert_eq!(result.reduced_problem.num_vars, 0);
    assert_eq!(result.reduced_problem.num_constraints, 0);
}

#[test]
fn test_forcing_ge_reduces() {
    // x + y >= 3, x in [0,1], y in [0,2]. max=3 <= 3.
    // Forcing from max: x→1, y→2. Row removed.
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Ge],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced);
    assert_eq!(result.reduced_problem.num_vars, 0);
    assert!(
        (result.obj_offset - 3.0).abs() < 1e-10,
        "obj = 1*1 + 1*2 = 3"
    );
}

#[test]
fn test_forcing_eq_reduces() {
    // x + y = 3, x in [0,1], y in [0,2]. max=3 <= 3. Forced.
    let lp = make_lp_general(
        vec![1.0, 1.0],
        &[0, 0],
        &[0, 1],
        &[1.0, 1.0],
        1,
        2,
        vec![3.0],
        vec![ConstraintType::Eq],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    assert!(result.was_reduced);
    assert_eq!(result.reduced_problem.num_vars, 0);
    assert!((result.obj_offset - 3.0).abs() < 1e-10);
}

#[test]
fn test_forcing_not_triggered_when_slack() {
    // -x + y <= 10, x in [0,1], y in [0,2]. min=-1 < 10, not forcing.
    // Not redundant: max=0+2=2 <= 10 → redundant. Use tighter rhs.
    // 3x + 2y <= 5, x in [0,1], y in [0,2]. min=0 < 5, not forcing.
    // max=3+4=7 > 5, not redundant.
    let lp = make_lp_general(
        vec![-1.0, -1.0],
        &[0, 0],
        &[0, 1],
        &[3.0, 2.0],
        1,
        2,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0), (0.0, 2.0)],
    );
    let result = run_presolve(&lp, None).unwrap();
    // Not forcing, not redundant. Bounds tighten but no rows/cols removed.
    assert_eq!(result.reduced_problem.num_vars, 2, "vars stay");
    assert_eq!(result.reduced_problem.num_constraints, 1, "row stays");
}

// -----------------------------------------------------------
// P1-1: ForcingRow dual recovery with prior FixedVariable
// -----------------------------------------------------------
// When step1 (FixedVariable) removes x0 before step2b (ForcingRow) runs,
// LIFO replay processes ForcingRow first with solution[x0]=0 (not yet restored).
// is_row_nonbinding computed Ax using orig_problem (includes x0 with stale value),
// yielding an inflated slack → binding row misclassified as non-binding → dual=0.
// Fix: ForcingRow snapshot stores active entries at presolve time; postsolve uses
// these entries directly, bypassing the incomplete-solution binding check.

#[test]
fn test_forcing_row_dual_le_with_prior_fixed_variable() {
    // min -x1 - x2
    // x0 + x1 + x2 <= 5,  x0 in [5,5], x1 in [0,2], x2 in [0,2]
    //
    // step1: x0 fixed at 5 → modified_b = 5 - 5 = 0
    // step2b: x1 + x2 <= 0, min=0=rhs, forcing: x1→0, x2→0
    // At optimum: x0=5, x1=0, x2=0; row = 5+0+0 = 5 = rhs → binding.
    // KKT for x1 at lb=0 (Le row): rc_1 = -1 - y >= 0 → y <= -1.
    // KKT for x2 at lb=0: rc_2 = -1 - y >= 0 → y <= -1.  Dual = -1.
    let lp = make_lp_general(
        vec![0.0, -1.0, -1.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[1.0, 1.0, 1.0],
        1,
        3,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(5.0, 5.0), (0.0, 2.0), (0.0, 2.0)],
    );
    let result = crate::simplex::solve(&lp);
    assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
    assert!(
        (result.solution[0] - 5.0).abs() < 1e-6,
        "x0={}",
        result.solution[0]
    );
    assert!(result.solution[1].abs() < 1e-6, "x1={}", result.solution[1]);
    assert!(result.solution[2].abs() < 1e-6, "x2={}", result.solution[2]);
    // Le row is binding; dual must be -1.
    assert!(
        (result.dual_solution[0] - (-1.0)).abs() < 1e-6,
        "dual should be -1, got {}",
        result.dual_solution[0]
    );
}

#[test]
fn test_forcing_row_dual_ge_with_prior_fixed_variable() {
    // min x1 + x2
    // x0 + x1 + x2 >= 5,  x0 in [1,1], x1 in [0,2], x2 in [0,2]
    //
    // step1: x0 fixed at 1 → modified_b = 5 - 1 = 4
    // step2b: x1 + x2 >= 4, max=4=rhs, forcing: x1→2, x2→2
    // At optimum: x0=1, x1=2, x2=2; row = 1+2+2 = 5 = rhs → binding.
    // KKT for x1 at ub=2 (Ge row): rc_1 = 1 - y <= 0 → y >= 1.
    // KKT for x2 at ub=2: rc_2 = 1 - y <= 0 → y >= 1.  Dual = 1.
    let lp = make_lp_general(
        vec![0.0, 1.0, 1.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[1.0, 1.0, 1.0],
        1,
        3,
        vec![5.0],
        vec![ConstraintType::Ge],
        vec![(1.0, 1.0), (0.0, 2.0), (0.0, 2.0)],
    );
    let result = crate::simplex::solve(&lp);
    assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
    assert!(
        (result.solution[0] - 1.0).abs() < 1e-6,
        "x0={}",
        result.solution[0]
    );
    assert!(
        (result.solution[1] - 2.0).abs() < 1e-6,
        "x1={}",
        result.solution[1]
    );
    assert!(
        (result.solution[2] - 2.0).abs() < 1e-6,
        "x2={}",
        result.solution[2]
    );
    // Ge row is binding; dual must be 1.
    assert!(
        (result.dual_solution[0] - 1.0).abs() < 1e-6,
        "dual should be 1, got {}",
        result.dual_solution[0]
    );
}

#[test]
fn test_forcing_row_dual_le_multiple_prior_fixed_vars() {
    // min -x2 - x3
    // x0 + x1 + x2 + x3 <= 3,  x0 in [1,1], x1 in [2,2], x2 in [0,2], x3 in [0,2]
    //
    // step1: x0 fixed (modified_b = 2), x1 fixed (modified_b = 0)
    // step2b: x2 + x3 <= 0, min=0=rhs, forcing: x2→0, x3→0
    // At optimum: x0=1, x1=2, x2=0, x3=0; row = 3 = rhs → binding.
    // KKT for x2 at lb=0: rc_2 = -1 - y >= 0 → y <= -1.
    // KKT for x3 at lb=0: rc_3 = -1 - y >= 0 → y <= -1.  Dual = -1.
    let lp = make_lp_general(
        vec![0.0, 0.0, -1.0, -1.0],
        &[0, 0, 0, 0],
        &[0, 1, 2, 3],
        &[1.0, 1.0, 1.0, 1.0],
        1,
        4,
        vec![3.0],
        vec![ConstraintType::Le],
        vec![(1.0, 1.0), (2.0, 2.0), (0.0, 2.0), (0.0, 2.0)],
    );
    let result = crate::simplex::solve(&lp);
    assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
    assert!(
        (result.solution[0] - 1.0).abs() < 1e-6,
        "x0={}",
        result.solution[0]
    );
    assert!(
        (result.solution[1] - 2.0).abs() < 1e-6,
        "x1={}",
        result.solution[1]
    );
    assert!(result.solution[2].abs() < 1e-6, "x2={}", result.solution[2]);
    assert!(result.solution[3].abs() < 1e-6, "x3={}", result.solution[3]);
    // Le row is binding; dual must be -1.
    assert!(
        (result.dual_solution[0] - (-1.0)).abs() < 1e-6,
        "dual should be -1, got {}",
        result.dual_solution[0]
    );
}

#[test]
fn test_forcing_row_dual_eq_with_prior_fixed_variable() {
    // min -x1 - x2
    // x0 + x1 + x2 = 5,  x0 in [5,5], x1 in [0,2], x2 in [0,2]
    //
    // step1: x0 fixed at 5 → modified_b = 0
    // step2b: x1 + x2 = 0, min=0=rhs, forcing from below: x1→0, x2→0
    // At optimum: x0=5, x1=0, x2=0; row = 5 = rhs → binding.
    // Eq row dual unconstrained in sign.
    // KKT for x1 at lb=0: rc_1 = -1 - y >= 0 → y <= -1. Dual = -1.
    let lp = make_lp_general(
        vec![0.0, -1.0, -1.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[1.0, 1.0, 1.0],
        1,
        3,
        vec![5.0],
        vec![ConstraintType::Eq],
        vec![(5.0, 5.0), (0.0, 2.0), (0.0, 2.0)],
    );
    let result = crate::simplex::solve(&lp);
    assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
    assert!(
        (result.solution[0] - 5.0).abs() < 1e-6,
        "x0={}",
        result.solution[0]
    );
    assert!(result.solution[1].abs() < 1e-6, "x1={}", result.solution[1]);
    assert!(result.solution[2].abs() < 1e-6, "x2={}", result.solution[2]);
    assert!(
        (result.dual_solution[0] - (-1.0)).abs() < 1e-6,
        "Eq forcing-row dual should be -1, got {}",
        result.dual_solution[0]
    );
}

#[test]
fn test_forcing_row_dual_le_mixed_sign_with_prior_fixed_variable() {
    // min -x1 + x2
    // 2*x0 + x1 - x2 <= 4,  x0 in [1,1], x1 in [0,3], x2 in [0,3]
    //
    // step1: x0 fixed at 1 → modified_b = 4 - 2 = 2
    // step2b: x1 - x2 <= 2, min = 0 + (-1)*3 = -3 < 2, not forcing.
    // Try: 2*x0 + x1 - x2 <= -1,  x0 in [1,1], x1 in [0,3], x2 in [0,3]
    //   modified_b = -1 - 2 = -3. x1 - x2 <= -3. min = 0 + (-1)*3 = -3 >= -3 → forcing.
    //   x1→0 (pos coeff, to lb), x2→3 (neg coeff, to ub).
    // At optimum: x0=1, x1=0, x2=3; row = 2+0-3 = -1 = rhs → binding.
    // KKT for x1 at lb=0: rc_1 = -1 - 1*y >= 0 → y <= -1.
    // KKT for x2 at ub=3: rc_2 = 1 - (-1)*y <= 0 → 1 + y <= 0 → y <= -1. Dual = -1.
    let lp = make_lp_general(
        vec![0.0, -1.0, 1.0],
        &[0, 0, 0],
        &[0, 1, 2],
        &[2.0, 1.0, -1.0],
        1,
        3,
        vec![-1.0],
        vec![ConstraintType::Le],
        vec![(1.0, 1.0), (0.0, 3.0), (0.0, 3.0)],
    );
    let result = crate::simplex::solve(&lp);
    assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
    assert!(
        (result.solution[0] - 1.0).abs() < 1e-6,
        "x0={}",
        result.solution[0]
    );
    assert!(result.solution[1].abs() < 1e-6, "x1={}", result.solution[1]);
    assert!(
        (result.solution[2] - 3.0).abs() < 1e-6,
        "x2={}",
        result.solution[2]
    );
    assert!(
        (result.dual_solution[0] - (-1.0)).abs() < 1e-6,
        "dual should be -1, got {}",
        result.dual_solution[0]
    );
}

#[test]
fn test_forcing_row_dual_multirow_with_prior_fixed_variable() {
    // min -x1 - x2
    // row0: x0 + x1 + x2 <= 5,  x0 in [5,5], x1 in [0,2], x2 in [0,2]
    // row1: x1 + x2 <= 10 (slack, non-binding)
    //
    // step1: x0 fixed at 5 → row0 modified_b = 0
    // step2b: x1 + x2 <= 0, forcing: x1→0, x2→0
    // Row1 survives presolve. At optimum: x0=5, x1=0, x2=0.
    // Row0: 5+0+0 = 5 = rhs → binding, dual = -1.
    // Row1: 0+0 = 0 <= 10 → non-binding, dual = 0.
    // KKT for x1 at lb=0: rc_1 = -1 - y0 - y1 = -1 - (-1) - 0 = 0.  OK.
    let lp = make_lp_general(
        vec![0.0, -1.0, -1.0],
        &[0, 0, 0, 1, 1],
        &[0, 1, 2, 1, 2],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        2,
        3,
        vec![5.0, 10.0],
        vec![ConstraintType::Le, ConstraintType::Le],
        vec![(5.0, 5.0), (0.0, 2.0), (0.0, 2.0)],
    );
    let result = crate::simplex::solve(&lp);
    assert_eq!(result.status, crate::problem::SolveStatus::Optimal);
    assert!(
        (result.solution[0] - 5.0).abs() < 1e-6,
        "x0={}",
        result.solution[0]
    );
    assert!(result.solution[1].abs() < 1e-6, "x1={}", result.solution[1]);
    assert!(result.solution[2].abs() < 1e-6, "x2={}", result.solution[2]);
    assert!(
        (result.dual_solution[0] - (-1.0)).abs() < 1e-6,
        "row0 dual should be -1, got {}",
        result.dual_solution[0]
    );
    assert!(
        result.dual_solution[1].abs() < 1e-6,
        "row1 (non-binding) dual should be 0, got {}",
        result.dual_solution[1]
    );
}
