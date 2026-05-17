//! Task #17 mini-corpus — **bug class 1 & 2** の構造的最小再現。
//!
//! ## 対象 bug class
//!
//! **(1) dual 退化 col / SingletonRow + RedundantConstraint 経由 y 復元**
//!   - 構造的特徴: 列 j の A 列エントリが「全て」削除される行 (Singleton /
//!     Redundant) にあり、列 j の bound active 状態から逆算した y_i が
//!     **複数 row dual の連立** で初めて一意に決まる。
//!   - 大問題対応: perold col 229 (c[j]=0, row 84 のみに entry、interior)。
//!   - 旧バグ: cleanup LP の Phase I slack 最小化が「y_i = 0 解」を選んだ
//!     ために rc[j] = -y_i 規模の dual infeasibility を出していた。
//!   - 本 fix (66857c1): 3-way (y_loop / y_gs / y_cl) bound-aware dfeas 比較。
//!
//! **(2) cleanup LP tie-break / 非一意 dual で誤った最適解採用**
//!   - 構造的特徴: 削除行が 2 個以上 + 削除行間で y のスケールが連立して
//!     一意でない LP optimum (dual 退化)。
//!   - 旧バグ: cleanup LP が複数最適解の中で恣意的に選び、KKT を満たさない
//!     y を採用していた。
//!   - 本 fix: y_loop / y_gs / y_cl の中で bound-aware dfeas が最小の y を採用。
//!
//! ## このファイルのテスト方針
//!
//! - HEAD (fix/lp-bugs) では全て GREEN。
//! - 旧 commit (e61f27b 直後) では複数が FAIL。bisect で fix の真因が確定
//!   している。
//! - 各 test は LP を Rust リテラル (≤ 10 var) で組み、`solve_with(presolve=true)`
//!   の結果が **primal feasibility / dual feasibility / 目的関数値** の三本柱
//!   全てを満たすことを assert。

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, LpProblem, SolveStatus};
use solver::solve_with;
use solver::sparse::CscMatrix;

// =============================================================================
// Shared helpers (he file-local; integration test なので mod 共有しない)
// =============================================================================

/// LP KKT 残差判定の許容誤差。bench (`eps=1e-6`) と揃える (CLAUDE.md L42)。
const EPS_KKT: f64 = 1e-6;

/// LP optimum の目的関数値判定の相対許容誤差。obj 自身がゼロに近い場合の
/// 絶対判定用 floor は (1 + |obj_expected|) のスケールで吸収。
const EPS_OBJ_REL: f64 = 1e-6;

/// 全 mini test の単一実行 timeout。CLAUDE.md L16「test 1 つ 3 分以内」と
/// 「mini test は 1 秒以内」の team-lead 制約を両立。
const MINI_TIMEOUT_SECS: f64 = 5.0;

/// Bench `compute_dfeas_orig` (c69959d 以降) 同型の bound-aware dual feasibility。
///
/// fixed (lb==ub) は除外、at_lb のみで rc<0 を、at_ub のみで rc>0 を、
/// interior は rc=0 (両端 hit は判定除外、複合扱い) で違反量を測る。
fn dfeas_rel_bound(
    c: &[f64],
    bounds: &[(f64, f64)],
    x: &[f64],
    rc: &[f64],
) -> f64 {
    const BOUND_TOL: f64 = 1e-6;
    let n = c.len().min(rc.len()).min(x.len());
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = bounds[j];
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
            0.0
        };
        let scale = 1.0 + r.abs() + c[j].abs();
        max_rel = max_rel.max(viol / scale);
    }
    max_rel
}

/// 主実行可能性残差 (|Ax - b|_∞)。Eq/Le/Ge 別に違反方向のみ取る。
fn pfeas_abs(a: &CscMatrix, b: &[f64], cts: &[ConstraintType], x: &[f64]) -> f64 {
    let m = b.len();
    let mut ax = vec![0.0_f64; m];
    for j in 0..x.len() {
        if let Ok((rows, vals)) = a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                ax[row] += vals[k] * x[j];
            }
        }
    }
    let mut max_v = 0.0_f64;
    for i in 0..m {
        let v = match cts[i] {
            ConstraintType::Eq => (ax[i] - b[i]).abs(),
            ConstraintType::Le => (ax[i] - b[i]).max(0.0),
            ConstraintType::Ge => (b[i] - ax[i]).max(0.0),
            // ConstraintType は #[non_exhaustive] (problem/mod.rs L12 確認済)。
            // 将来追加された種別は 0 violation 扱いで bench 退化を起こさない方が安全。
            _ => 0.0,
        };
        if v > max_v { max_v = v; }
    }
    max_v
}

/// LP を solve し KKT 整合性を一括 assert する。失敗時 stderr に詳細 dump。
fn assert_kkt_optimal(lp: &LpProblem, expected_obj: f64, label: &'static str) {
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(MINI_TIMEOUT_SECS);
    let r = solve_with(lp, &opts);

    eprintln!(
        "[{}] status={:?} obj={:.6e} expected={:.6e}",
        label, r.status, r.objective, expected_obj
    );

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "[{}] expected Optimal, got {:?}", label, r.status
    );

    // (i) primal feasibility
    let pf = pfeas_abs(&lp.a, &lp.b, &lp.constraint_types, &r.solution);
    assert!(
        pf < EPS_KKT,
        "[{}] pfeas={:.3e} > eps={:.3e} (x={:?})",
        label, pf, EPS_KKT, &r.solution
    );

    // (ii) dual feasibility (bound-aware)
    let df = dfeas_rel_bound(&lp.c, &lp.bounds, &r.solution, &r.reduced_costs);
    assert!(
        df < EPS_KKT,
        "[{}] dfeas_rel_bound={:.3e} > eps={:.3e} | x={:?} rc={:?} y={:?}",
        label, df, EPS_KKT, &r.solution, &r.reduced_costs, &r.dual_solution
    );

    // (iii) 目的関数値
    let obj_err = (r.objective - expected_obj).abs() / (1.0 + expected_obj.abs());
    assert!(
        obj_err < EPS_OBJ_REL,
        "[{}] obj={:.9e} expected={:.9e} rel_err={:.3e} > {:.3e}",
        label, r.objective, expected_obj, obj_err, EPS_OBJ_REL
    );
}

// =============================================================================
// Bug class 1: dual 退化 col (perold col 229 proxy)
// =============================================================================

/// **構造的特徴**: c[j]=0 の列 j の A エントリが「SingletonRow 経由で削除
/// される 1 行」にしか無く、列 j は interior (非 bound-active)。元問題で
/// rc[j]=0 を満たす y_i は一意 (= 0) だが、cleanup LP の Phase I slack
/// 最小化はタイブレークで別の y を選びうる。
///
/// **元 bug**: `perold` col 229 (c[j]=0, row 84 のみ、interior)。
/// HEAD `4a1e305` で df_rel_bound ≈ 0.99、fix (66857c1 + 3-way 比較) で <1e-10。
///
/// **構成**: 4 var, 3 row。SingletonRow が x0 を即時 fix し、x0 の y_0 復元が
/// rc[x0]=0 (interior 扱い、bounds 広い) で一意決定されるパターン。
#[test]
fn bug1a_singleton_row_zero_cost_col() {
    // min 0*x0 + 1*x1 + 2*x2 + 1*x3
    // s.t. x0          = 5         (Eq, SingletonRow 対象 — x0 を 5 に fix)
    //      x1 + x2     = 4         (Eq, doubleton 残存)
    //      x2 + x3    <= 10        (Le, RedundantConstraint 候補)
    // bounds: x0 in [0, 10], x1, x2, x3 in [0, INF)
    //
    // Expected optimum: x0=5, x1=4, x2=0, x3=0 → obj = 0+4+0+0 = 4
    let c = vec![0.0, 1.0, 2.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 1, 1, 2, 2],   // row indices
        &[0, 1, 2, 2, 3],   // col indices
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        3, 4,
    ).unwrap();
    let b = vec![5.0, 4.0, 10.0];
    let cts = vec![ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Le];
    let bounds = vec![(0.0, 10.0), (0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, f64::INFINITY)];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug1a".into())).unwrap();
    assert_kkt_optimal(&lp, 4.0, "bug1a_singleton_row_zero_cost_col");
}

/// **構造的特徴**: c[j]=0 の列 j を持ち、列 j の唯一の A エントリが
/// RedundantConstraint で削除される Le 行にある。x0 は EmptyColumn 化し
/// 削除された RedundantConstraint 行の y を bound-aware 復元する必要がある。
///
/// **元 bug**: perold-class の「c[j]=0 + Redundant 行 only 列」を分離。
///
/// **構成**: 3 var, 3 row。
/// Row 0 (Le): x0+x1+x2 <= 1000 → x0 in [0,1], x1<=5, x2<=5 で row max=11<1000、
/// → RedundantConstraint で削除。x0 は EmptyColumn 化 (c[0]=0)。
/// Row 1 (Eq): x1 + x2 = 5。
/// Row 2 (Le): x1 <= 3。
#[test]
fn bug1b_redundant_constraint_zero_cost_col() {
    let c = vec![0.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 0, 1, 1, 2],
        &[0, 1, 2, 1, 2, 1],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        3, 3,
    ).unwrap();
    let b = vec![1000.0, 5.0, 3.0];
    let cts = vec![ConstraintType::Le, ConstraintType::Eq, ConstraintType::Le];
    // x0 bounds [0,1] (有限) で c[0]=0 → EmptyColumn 時 lb=0 にされる。
    let bounds = vec![(0.0, 1.0), (0.0, 5.0), (0.0, 5.0)];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug1b".into())).unwrap();
    // x0 自由 (c=0, redundant 1 行のみ); x1+x2=5, x1<=3 で min x1+x2 → 5
    assert_kkt_optimal(&lp, 5.0, "bug1b_redundant_constraint_zero_cost_col");
}

/// **構造的特徴**: c[j] が非ゼロ (=1.0) で、列 j の唯一のエントリが
/// RedundantConstraint で削除される Le 行にあり、x[j] は lower bound active。
/// 削除行の y_i = c[j]/a_ij = 1.0/1.0 = 1.0 でないと rc[j] ≥ 0 を満たさない
/// ケースを cleanup LP / Gauss-Seidel が正しく回せるか。
///
/// **元 bug**: 削除行の y_i = 0 デフォルトで rc[j] = c[j] = 1 > 0 になり一見
/// OK だが、bound active なので rc 符号一貫性は OK。本テストは regression
/// 防壁 (RedundantConstraint y 復元の境界条件確認)。
#[test]
fn bug1c_redundant_constraint_only_entry_at_lb() {
    // min  x0 + x1 + x2
    // s.t. x0 + x1      <= 100  (Le, Redundant: x0<=10, x1<=10, sum<=20<100)
    //      x1 + x2       = 3    (Eq)
    //      x1           <= 2    (Le, binding)
    // x0,x1,x2 in [0,10]. Optimal: x0=0, x1=0, x2=3 → obj = 3
    let c = vec![1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1, 2],
        &[0, 1, 1, 2, 1],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        3, 3,
    ).unwrap();
    let b = vec![100.0, 3.0, 2.0];
    let cts = vec![ConstraintType::Le, ConstraintType::Eq, ConstraintType::Le];
    let bounds = vec![(0.0, 10.0); 3];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug1c".into())).unwrap();
    assert_kkt_optimal(&lp, 3.0, "bug1c_redundant_constraint_only_entry_at_lb");
}

// =============================================================================
// Bug class 2: cleanup LP tie-break / 非一意 dual
// =============================================================================

/// **構造的特徴**: 2 つの SingletonRow が同じ y 自由度に作用し、cleanup LP の
/// 連立 LP は無数の解を持つが、bound-aware dfeas を満たす y は一意。
/// Phase I slack 最小化のタイ崩しで誤った y を選ばないか確認。
///
/// **構成**: 5 var, 4 row。x0, x1 が SingletonRow で削除され、両者の y_0, y_1 が
/// 残った列 (x2, x3, x4) の rc 制約を介して連立する。
#[test]
fn bug2a_two_singleton_rows_coupled_dual() {
    // min 0*x0 + 0*x1 + x2 + x3 + x4
    // s.t. x0                       = 7   (Eq SingletonRow)
    //              x1               = 3   (Eq SingletonRow)
    //                   x2 + x3     = 4   (Eq, 残存)
    //                        x3 + x4 = 2  (Eq, 残存)
    // 連立: x0=7, x1=3, x2+x3=4, x3+x4=2.
    //   Optimal: minimize x2+x3+x4. From rows: x2 = 4-x3, x4 = 2-x3.
    //   sum = (4-x3) + x3 + (2-x3) = 6 - x3. Maximize x3.
    //   x3 <= min(4, 2) = 2. So x3 = 2, x2 = 2, x4 = 0. obj = 2+2+0 = 4.
    // c[x0]=c[x1]=0; x0, x1 interior (lb=0, no ub binding); SingletonRow が削除。
    let c = vec![0.0, 0.0, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 1, 2, 2, 3, 3],
        &[0, 1, 2, 3, 3, 4],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        4, 5,
    ).unwrap();
    let b = vec![7.0, 3.0, 4.0, 2.0];
    let cts = vec![
        ConstraintType::Eq, ConstraintType::Eq,
        ConstraintType::Eq, ConstraintType::Eq,
    ];
    let bounds = vec![(0.0, f64::INFINITY); 5];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug2a".into())).unwrap();
    assert_kkt_optimal(&lp, 4.0, "bug2a_two_singleton_rows_coupled_dual");
}

/// **構造的特徴**: 削除行 1 行 + RedundantConstraint 1 行が同じ列 x0 に作用し、
/// x0 が bound active (lb)。y_0, y_1 の連立で rc[x0] >= 0 を満たす範囲が
/// interval (一意でない) になり、cleanup LP が tie-break で間違える可能性。
#[test]
fn bug2b_two_deleted_rows_bound_active_col() {
    // min x0 + x1 + x2
    // s.t. x0 + x1        = 4    (Eq, 残存 — x0, x1 共に活性)
    //      x0 + x2       <= 100  (Le, Redundant: x0<=2 + x2<=10 = 12 < 100)
    //      x2             = 1    (Eq SingletonRow on x2)
    // bounds: x0 in [0,2], x1 in [0,4], x2 in [0,10]
    // SingletonRow → x2=1 削除; Redundant → row 1 削除 (Σ row1 max = 2+10=12 < 100)
    // 残る: x0+x1 = 4, x0 in [0,2], x1 in [0,4]
    // min x0+x1 = 4 (Eq 固定) — 任意の (x0,x1) で同値。x0 で min は x0=0? 違う:
    //   obj = x0 + x1 + 1 (x2 fixed). x0+x1=4 だから obj = 5。x0 自由.
    //   ただし unique なため x0 = 0, x1 = 4 が cleanup の出口になる確率高い。
    //   (test の主目的は dfeas 整合性、x の一意性は問わない)
    let c = vec![1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1, 2],
        &[0, 1, 0, 2, 2],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        3, 3,
    ).unwrap();
    let b = vec![4.0, 100.0, 1.0];
    let cts = vec![ConstraintType::Eq, ConstraintType::Le, ConstraintType::Eq];
    let bounds = vec![(0.0, 2.0), (0.0, 4.0), (0.0, 10.0)];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug2b".into())).unwrap();
    assert_kkt_optimal(&lp, 5.0, "bug2b_two_deleted_rows_bound_active_col");
}

/// **構造的特徴**: LinearSubstitution (free 変数置換) + SingletonRow の混在。
/// 旧バグ (e61f27b): LinearSubstitution の y_piv 復元は追加されたが
/// SingletonRow / RedundantConstraint との連立整合性が保証されていなかった。
/// 本 fix (66857c1) で 3-way 比較に統一。
///
/// **構成**: 4 var (x0 free, others ≥ 0), 3 row。Doubleton Eq + SingletonRow。
#[test]
fn bug2c_linear_substitution_plus_singleton() {
    // min x0 + x1 + x2 + x3
    // s.t. x0 + x1       = 5    (Eq, doubleton — x0 free なら substituted)
    //      x0 - x2       = 1    (Eq, kept)
    //      x3            = 2    (Eq, SingletonRow)
    // bounds: x0 free (-INF, INF), x1, x2 >= 0, x3 in [0, 5]
    // From row 0: x0 = 5 - x1.
    // From row 1: (5 - x1) - x2 = 1 → x1 + x2 = 4.
    // Reduced (after substitution + SingletonRow): min (5-x1)+x1+x2+2 = 7 + x2
    //   s.t. x1 + x2 = 4, x1,x2 >= 0. min x2 → x2=0, x1=4. obj=7. x0=1.
    // Total obj = 1 + 4 + 0 + 2 = 7.
    let c = vec![1.0, 1.0, 1.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1, 2],
        &[0, 1, 0, 2, 3],
        &[1.0, 1.0, 1.0, -1.0, 1.0],
        3, 4,
    ).unwrap();
    let b = vec![5.0, 1.0, 2.0];
    let cts = vec![ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Eq];
    let bounds = vec![
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, 5.0),
    ];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug2c".into())).unwrap();
    assert_kkt_optimal(&lp, 7.0, "bug2c_linear_substitution_plus_singleton");
}

// =============================================================================
// Cross-cutting regression: presolve OFF baseline (postsolve 経路を bypass)
// =============================================================================

/// 真の真因が postsolve y 経路に局在することを示す cross-check:
/// bug1a 同型問題を presolve=false で解いても dfeas は GREEN。
/// presolve=true でも GREEN なら本 fix の効果あり; FAIL なら postsolve 退行。
#[test]
fn cross_check_bug1a_presolve_off_baseline() {
    let c = vec![0.0, 1.0, 2.0, 1.0];
    let a = CscMatrix::from_triplets(
        &[0, 1, 1, 2, 2],
        &[0, 1, 2, 2, 3],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
        3, 4,
    ).unwrap();
    let b = vec![5.0, 4.0, 10.0];
    let cts = vec![ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Le];
    let bounds = vec![(0.0, 10.0), (0.0, f64::INFINITY), (0.0, f64::INFINITY), (0.0, f64::INFINITY)];
    let lp = LpProblem::new_general(c, a, b, cts, bounds, Some("bug1a-pre-off".into())).unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(MINI_TIMEOUT_SECS);
    let r = solve_with(&lp, &opts);
    eprintln!(
        "[bug1a presolve=off] status={:?} obj={:.6e}",
        r.status, r.objective
    );
    assert_eq!(r.status, SolveStatus::Optimal);
    let df = dfeas_rel_bound(&lp.c, &lp.bounds, &r.solution, &r.reduced_costs);
    assert!(df < EPS_KKT, "presolve=off でも dfeas 違反: {:.3e}", df);
}
