//! Task #17 mini-corpus — **bug class 1 & 2** の構造的最小再現。
//!
//! ## 対象 bug class
//!
//! **(1) dual 退化 col / SingletonRow + RedundantConstraint 経由 y 復元**
//!   - 列 j の A 列エントリが「全て」削除される行 (Singleton /
//!     Redundant) にあり、列 j の bound active 状態から逆算した y_i が
//!     **複数 row dual の連立** で初めて一意に決まる。
//!   - 大問題対応: perold col 229 (c[j]=0, row 84 のみに entry、interior)。
//!   - 旧バグ: cleanup LP の Phase I slack 最小化が「y_i = 0 解」を選んだ
//!     ために rc[j] = -y_i 規模の dual infeasibility を出していた。
//!   - 本 fix (66857c1): 3-way (y_loop / y_gs / y_cl) bound-aware dfeas 比較。
//!
//! **(2) cleanup LP tie-break / 非一意 dual で誤った最適解採用**
//!   - 削除行が 2 個以上 + 削除行間で y のスケールが連立して
//!     一意でない LP optimum (dual 退化)。
//!   - 旧バグ: cleanup LP が複数最適解の中で恣意的に選び、KKT を満たさない
//!     y を採用していた。
//!   - 本 fix: y_loop / y_gs / y_cl の中で bound-aware dfeas が最小の y を採用。
//!
//! ## このファイルのテスト方針
//!
//! - HEAD では全 GREEN。
//! - 旧 commit (e61f27b 直後) では複数が FAIL。bisect で fix の真因が確定。
//! - 各 test は LP を Model API (`Model::add_var` + `add_constraint`) で組み、
//!   presolve=true (本 fix 経路) で primal feasibility / dual feasibility /
//!   目的関数値の三本柱を assert。LP path で拡張済の
//!   `ModelResult.dual_solution` / `reduced_costs` 経由で KKT 検証する。

use otspot::model::{constraint, Expression, Model, Variable};
use otspot::problem::ConstraintType;
use otspot::sparse::CscMatrix;

// =============================================================================
// Shared helpers
// =============================================================================

/// LP KKT 残差判定の許容誤差。bench (`eps=1e-6`) と揃える (CLAUDE.md L42)。
const EPS_KKT: f64 = 1e-6;

/// LP optimum の目的関数値判定の相対許容誤差。obj 自身がゼロに近い場合の
/// 絶対判定用 floor は (1 + |obj_expected|) のスケールで吸収。
const EPS_OBJ_REL: f64 = 1e-6;

/// 全 mini test の単一実行 timeout (CLAUDE.md L16: 1 test 3 分以内)。
const MINI_TIMEOUT_SECS: f64 = 5.0;

/// Bench `compute_dfeas_orig` 同型の bound-aware dual feasibility。
///
/// fixed (lb==ub) は除外、at_lb のみで rc<0 を、at_ub のみで rc>0 を、
/// interior は rc=0 (両端 hit は判定除外、複合扱い) で違反量を測る。
fn dfeas_rel_bound(c: &[f64], bounds: &[(f64, f64)], x: &[f64], rc: &[f64]) -> f64 {
    const BOUND_TOL: f64 = 1e-6;
    let n = c.len().min(rc.len()).min(x.len());
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = bounds[j];
        let fixed = lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_TOL;
        if fixed {
            continue;
        }
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
    for (j, &x_j) in x.iter().enumerate() {
        if let Ok((rows, vals)) = a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                ax[row] += vals[k] * x_j;
            }
        }
    }
    let mut max_v = 0.0_f64;
    for i in 0..m {
        let v = match cts[i] {
            ConstraintType::Eq => (ax[i] - b[i]).abs(),
            ConstraintType::Le => (ax[i] - b[i]).max(0.0),
            ConstraintType::Ge => (b[i] - ax[i]).max(0.0),
            // ConstraintType は #[non_exhaustive]: 将来追加された種別は
            // 違反 0 扱いで bench 退化を起こさない方が安全。
            _ => 0.0,
        };
        if v > max_v {
            max_v = v;
        }
    }
    max_v
}

/// 三項組形式の LP raw データ。regression sentinel として
/// 旧 bug data path (b=0 dual 退化 / cleanup LP tie-break) を sparse 構造ごと
/// 保持する目的のため、Model API に渡しつつ pfeas 検証用にも CSC を保持する。
struct LpData<'a> {
    name: &'a str,
    c: &'a [f64],
    rows: &'a [usize],
    cols: &'a [usize],
    vals: &'a [f64],
    b: &'a [f64],
    cts: &'a [ConstraintType],
    bounds: &'a [(f64, f64)],
}

/// Raw triplet 形式から Model を組み立てる。
///
/// Model API 経由でも元 LP と同じ A (sparse 構造)、c、bounds、cts、b を持つ
/// problem instance が構築される。Phase I + presolve 経路を bypass せず、
/// 3-way y_loop/y_gs/y_cl 比較を踏襲する。
fn build_model(data: &LpData<'_>) -> (Model, Vec<Variable>) {
    let mut model = Model::new(data.name);
    let vars: Vec<Variable> = data
        .bounds
        .iter()
        .enumerate()
        .map(|(i, &(lb, ub))| model.add_var(&format!("x{}", i), lb, ub))
        .collect();

    let m = data.b.len();
    let mut row_terms: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for k in 0..data.rows.len() {
        row_terms[data.rows[k]].push((data.cols[k], data.vals[k]));
    }
    for (i, row) in row_terms.iter().enumerate() {
        let mut expr = Expression::from_constant(0.0);
        for &(j, v) in row {
            expr = expr + v * vars[j];
        }
        let con = match data.cts[i] {
            ConstraintType::Eq => expr.eq_constraint(data.b[i]),
            ConstraintType::Le => expr.leq(data.b[i]),
            ConstraintType::Ge => expr.geq(data.b[i]),
            _ => panic!("[{}] unsupported ConstraintType", data.name),
        };
        model.add_constraint(con);
    }

    let mut obj = Expression::from_constant(0.0);
    for (i, var) in vars.iter().enumerate() {
        obj = obj + data.c[i] * *var;
    }
    model.minimize(obj);
    (model, vars)
}

/// LP を Model API で solve し KKT 整合性を一括 assert する。
///
/// presolve は `Model::set_presolve(true)` 経由で
/// 明示的に有効化。dual_solution / reduced_costs は LP path で
/// populate されるため `.as_ref().expect(..)` で取得して dfeas 検証に使う。
fn assert_kkt_optimal(data: &LpData<'_>, expected_obj: f64) {
    let (mut model, vars) = build_model(data);
    model.set_timeout(MINI_TIMEOUT_SECS);
    model.set_presolve(true);

    let r = model
        .solve()
        .unwrap_or_else(|e| panic!("[{}] expected Optimal, got {:?}", data.name, e));

    let x: Vec<f64> = vars.iter().map(|v| r[*v]).collect();
    let rc = r
        .reduced_costs
        .as_ref()
        .unwrap_or_else(|| panic!("[{}] reduced_costs must be populated on LP path", data.name));
    let dual = r
        .dual_solution
        .as_ref()
        .unwrap_or_else(|| panic!("[{}] dual_solution must be populated on LP path", data.name));

    eprintln!(
        "[{}] obj={:.6e} expected={:.6e}",
        data.name, r.objective_value, expected_obj
    );

    // (i) primal feasibility — 元 A 構造で評価 (Model 構築前の triplet)
    let a = CscMatrix::from_triplets(data.rows, data.cols, data.vals, data.b.len(), data.c.len())
        .expect("CscMatrix::from_triplets");
    let pf = pfeas_abs(&a, data.b, data.cts, &x);
    assert!(
        pf < EPS_KKT,
        "[{}] pfeas={:.3e} > eps={:.3e} (x={:?})",
        data.name,
        pf,
        EPS_KKT,
        &x
    );

    // (ii) dual feasibility (bound-aware) — 3-way fix の本丸検証
    let df = dfeas_rel_bound(data.c, data.bounds, &x, rc);
    assert!(
        df < EPS_KKT,
        "[{}] dfeas_rel_bound={:.3e} > eps={:.3e} | x={:?} rc={:?} y={:?}",
        data.name,
        df,
        EPS_KKT,
        &x,
        rc,
        dual
    );

    // (iii) 目的関数値
    let obj_err = (r.objective_value - expected_obj).abs() / (1.0 + expected_obj.abs());
    assert!(
        obj_err < EPS_OBJ_REL,
        "[{}] obj={:.9e} expected={:.9e} rel_err={:.3e} > {:.3e}",
        data.name,
        r.objective_value,
        expected_obj,
        obj_err,
        EPS_OBJ_REL
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
    // Expected optimum: x0=5, x1=4, x2=0, x3=0 → obj = 0+4+0+0 = 4
    let c = [0.0, 1.0, 2.0, 1.0];
    let rows = [0, 1, 1, 2, 2];
    let cols = [0, 1, 2, 2, 3];
    let vals = [1.0, 1.0, 1.0, 1.0, 1.0];
    let b = [5.0, 4.0, 10.0];
    let cts = [ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Le];
    let bounds = [
        (0.0, 10.0),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
    ];
    let data = LpData {
        name: "bug1a_singleton_row_zero_cost_col",
        c: &c,
        rows: &rows,
        cols: &cols,
        vals: &vals,
        b: &b,
        cts: &cts,
        bounds: &bounds,
    };
    assert_kkt_optimal(&data, 4.0);
}

/// **構造的特徴**: c[j]=0 の列 j を持ち、列 j の唯一の A エントリが
/// RedundantConstraint で削除される Le 行にある。x0 は EmptyColumn 化し
/// 削除された RedundantConstraint 行の y を bound-aware 復元する必要がある。
///
/// **元 bug**: perold-class の「c[j]=0 + Redundant 行 only 列」を分離。
#[test]
fn bug1b_redundant_constraint_zero_cost_col() {
    let c = [0.0, 1.0, 1.0];
    let rows = [0, 0, 0, 1, 1, 2];
    let cols = [0, 1, 2, 1, 2, 1];
    let vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let b = [1000.0, 5.0, 3.0];
    let cts = [ConstraintType::Le, ConstraintType::Eq, ConstraintType::Le];
    let bounds = [(0.0, 1.0), (0.0, 5.0), (0.0, 5.0)];
    let data = LpData {
        name: "bug1b_redundant_constraint_zero_cost_col",
        c: &c,
        rows: &rows,
        cols: &cols,
        vals: &vals,
        b: &b,
        cts: &cts,
        bounds: &bounds,
    };
    // x0 自由 (c=0, redundant 1 行のみ); x1+x2=5, x1<=3 で min x1+x2 → 5
    assert_kkt_optimal(&data, 5.0);
}

/// **構造的特徴**: c[j] が非ゼロ (=1.0) で、列 j の唯一のエントリが
/// RedundantConstraint で削除される Le 行にあり、x[j] は lower bound active。
/// 削除行の y_i = c[j]/a_ij = 1.0/1.0 = 1.0 でないと rc[j] ≥ 0 を満たさない
/// ケースを cleanup LP / Gauss-Seidel が正しく回せるか。
#[test]
fn bug1c_redundant_constraint_only_entry_at_lb() {
    // min  x0 + x1 + x2
    // s.t. x0 + x1      <= 100  (Le, Redundant: x0<=10, x1<=10, sum<=20<100)
    //      x1 + x2       = 3    (Eq)
    //      x1           <= 2    (Le, binding)
    // x0,x1,x2 in [0,10]. Optimal: x0=0, x1=0, x2=3 → obj = 3
    let c = [1.0, 1.0, 1.0];
    let rows = [0, 0, 1, 1, 2];
    let cols = [0, 1, 1, 2, 1];
    let vals = [1.0, 1.0, 1.0, 1.0, 1.0];
    let b = [100.0, 3.0, 2.0];
    let cts = [ConstraintType::Le, ConstraintType::Eq, ConstraintType::Le];
    let bounds = [(0.0, 10.0); 3];
    let data = LpData {
        name: "bug1c_redundant_constraint_only_entry_at_lb",
        c: &c,
        rows: &rows,
        cols: &cols,
        vals: &vals,
        b: &b,
        cts: &cts,
        bounds: &bounds,
    };
    assert_kkt_optimal(&data, 3.0);
}

// =============================================================================
// Bug class 2: cleanup LP tie-break / 非一意 dual
// =============================================================================

/// **構造的特徴**: 2 つの SingletonRow が同じ y 自由度に作用し、cleanup LP の
/// 連立 LP は無数の解を持つが、bound-aware dfeas を満たす y は一意。
/// Phase I slack 最小化のタイ崩しで誤った y を選ばないか確認。
#[test]
fn bug2a_two_singleton_rows_coupled_dual() {
    // min 0*x0 + 0*x1 + x2 + x3 + x4
    // s.t. x0                       = 7   (Eq SingletonRow)
    //              x1               = 3   (Eq SingletonRow)
    //                   x2 + x3     = 4   (Eq, 残存)
    //                        x3 + x4 = 2  (Eq, 残存)
    // x3 <= min(4, 2) = 2 で x3=2, x2=2, x4=0. obj=4.
    let c = [0.0, 0.0, 1.0, 1.0, 1.0];
    let rows = [0, 1, 2, 2, 3, 3];
    let cols = [0, 1, 2, 3, 3, 4];
    let vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let b = [7.0, 3.0, 4.0, 2.0];
    let cts = [
        ConstraintType::Eq,
        ConstraintType::Eq,
        ConstraintType::Eq,
        ConstraintType::Eq,
    ];
    let bounds = [(0.0, f64::INFINITY); 5];
    let data = LpData {
        name: "bug2a_two_singleton_rows_coupled_dual",
        c: &c,
        rows: &rows,
        cols: &cols,
        vals: &vals,
        b: &b,
        cts: &cts,
        bounds: &bounds,
    };
    assert_kkt_optimal(&data, 4.0);
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
    // 残る: x0+x1=4 → obj = (x0+x1) + 1 = 5 (test 主目的は dfeas 整合性)
    let c = [1.0, 1.0, 1.0];
    let rows = [0, 0, 1, 1, 2];
    let cols = [0, 1, 0, 2, 2];
    let vals = [1.0, 1.0, 1.0, 1.0, 1.0];
    let b = [4.0, 100.0, 1.0];
    let cts = [ConstraintType::Eq, ConstraintType::Le, ConstraintType::Eq];
    let bounds = [(0.0, 2.0), (0.0, 4.0), (0.0, 10.0)];
    let data = LpData {
        name: "bug2b_two_deleted_rows_bound_active_col",
        c: &c,
        rows: &rows,
        cols: &cols,
        vals: &vals,
        b: &b,
        cts: &cts,
        bounds: &bounds,
    };
    assert_kkt_optimal(&data, 5.0);
}

/// **構造的特徴**: LinearSubstitution (free 変数置換) + SingletonRow の混在。
/// 旧バグ (e61f27b): LinearSubstitution の y_piv 復元は追加されたが
/// SingletonRow / RedundantConstraint との連立整合性が保証されていなかった。
/// 本 fix (66857c1) で 3-way 比較に統一。
#[test]
fn bug2c_linear_substitution_plus_singleton() {
    // min x0 + x1 + x2 + x3
    // s.t. x0 + x1       = 5    (Eq, doubleton — x0 free なら substituted)
    //      x0 - x2       = 1    (Eq, kept)
    //      x3            = 2    (Eq, SingletonRow)
    // bounds: x0 free (-INF, INF), x1, x2 >= 0, x3 in [0, 5]
    // 解析解: x0=1, x1=4, x2=0, x3=2. Total obj = 7.
    let c = [1.0, 1.0, 1.0, 1.0];
    let rows = [0, 0, 1, 1, 2];
    let cols = [0, 1, 0, 2, 3];
    let vals = [1.0, 1.0, 1.0, -1.0, 1.0];
    let b = [5.0, 1.0, 2.0];
    let cts = [ConstraintType::Eq, ConstraintType::Eq, ConstraintType::Eq];
    let bounds = [
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, 5.0),
    ];
    let data = LpData {
        name: "bug2c_linear_substitution_plus_singleton",
        c: &c,
        rows: &rows,
        cols: &cols,
        vals: &vals,
        b: &b,
        cts: &cts,
        bounds: &bounds,
    };
    assert_kkt_optimal(&data, 7.0);
}

// =============================================================================
// Cross-cutting regression: presolve OFF baseline (postsolve 経路を bypass)
// =============================================================================

/// 真因が postsolve y 経路に局在することを示す cross-check:
/// bug1a 同型問題を presolve=false で解いても dfeas は GREEN。
/// presolve=true でも GREEN なら本 fix の効果あり; FAIL なら postsolve 退行。
#[test]
fn cross_check_bug1a_presolve_off_baseline() {
    let mut model = Model::new("bug1a-pre-off");
    let x0 = model.add_var("x0", 0.0, 10.0);
    let x1 = model.add_var("x1", 0.0, f64::INFINITY);
    let x2 = model.add_var("x2", 0.0, f64::INFINITY);
    let x3 = model.add_var("x3", 0.0, f64::INFINITY);
    model.add_constraint(constraint!(x0 == 5.0));
    model.add_constraint(constraint!((x1 + x2) == 4.0));
    model.add_constraint(constraint!((x2 + x3) <= 10.0));
    model.minimize(0.0 * x0 + 1.0 * x1 + 2.0 * x2 + 1.0 * x3);
    model.set_timeout(MINI_TIMEOUT_SECS);
    model.set_presolve(false);

    let r = model
        .solve()
        .expect("[bug1a presolve=off] expected Optimal");

    let x = [r[x0], r[x1], r[x2], r[x3]];
    let rc = r
        .reduced_costs
        .as_ref()
        .expect("[bug1a presolve=off] reduced_costs must be populated on LP path");
    eprintln!("[bug1a presolve=off] obj={:.6e}", r.objective_value);

    let c = [0.0, 1.0, 2.0, 1.0];
    let bounds = [
        (0.0, 10.0),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
        (0.0, f64::INFINITY),
    ];
    let df = dfeas_rel_bound(&c, &bounds, &x, rc);
    assert!(df < EPS_KKT, "presolve=off でも dfeas 違反: {:.3e}", df);
}
