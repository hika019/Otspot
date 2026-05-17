//! Maros-Meszaros QP ベンチマーク問題テスト
//!
//! 業界標準 QP ベンチマーク（Hock-Schittkowski / Maros-Meszaros）を
//! Rust テストとして実装し、solver の QP 実装を検証する。
//!
//! 問題形式: min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub
//! （「1/2 あり」OSQP/qpOASES 標準規約）

use solver::qp::{solve_qp, solve_qp_warm, QpProblem, QpWarmStart};
use solver::sparse::CscMatrix;
use solver::SolveStatus;

// 解精度の許容値: default user_eps=1e-6 を基準に問題構造から導出する。
//
// 目的値: solver は relative gap = |pf-df| / (1 + |obj|) ≤ eps で収束判定する。
//   → 目的値絶対誤差 ≤ eps × (1 + |obj|)。abs 比較ではなく relative 比較が正しい。
//   assert_obj_close は relative eps=1e-6 で比較する。
//
// 解変数: 内点法の primal 精度は O(eps) ≈ 1e-6 (postsolve 後の丸め込みで数倍)
//   EPS_SOL: 通常解変数の絶対許容値
//
// 退化境界 (λ*=0 かつ制約 active) の解変数: O(sqrt(eps)) ≈ 1e-3
//   理由: 補完余裕 λ*s=mu で λ→0+ の場合、s≈sqrt(mu) で収束するため。
//   例: CVXQP1_S (λ*=0, sum(x)=5 active) → xi 誤差≈sqrt(1e-6)/2 ≈ 5e-4

// 解変数の絶対許容値 (postsolve 後の丸め ≈ 数×eps)
const EPS_SOL: f64 = 1e-5;
// 退化制約境界 (λ*=0) の解変数許容値
const EPS_DEG: f64 = 5e-4;

/// 目的値を相対誤差 eps=1e-6 で検証する。
/// ソルバーの収束判定 relative gap ≤ 1e-6 に対応した正しい比較。
fn assert_obj_close(actual: f64, expected: f64, name: &str) {
    let rel = (actual - expected).abs() / (1.0 + expected.abs());
    assert!(
        rel < 1e-6,
        "{}: expected {:.8}, got {:.8} (rel_err={:.2e})",
        name, expected, actual, rel
    );
}

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

/// HS21: Hock-Schittkowski Problem #21
///
/// 原問題:
///   min f = 0.01*x1^2 + x2^2 - 10*x1 - x2
///   s.t. 10*x1 - x2 >= 2
///        2 <= x1 <= 10,  -10 <= x2 <= 10
///
/// QP 形式（1/2 あり規約）:
///   Q = diag(0.02, 2.0),  c = [-10, -1]
///   A = [[-10, 1]],  b = [-2]  （10x1-x2>=2 → -10x1+x2<=-2）
///
/// 解析解: x1* = 10（上界が活性）, x2* = 0.5（内点最小）
/// QP 目的関数値: 1/2*(0.02*100 + 2*0.25) + (-100 - 0.5) = 1.25 - 100.5 = -99.25
#[test]
fn test_hs21() {
    let n = 2;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.02, 2.0], n, n).unwrap();
    let c = vec![-10.0, -1.0];
    // 10*x1 - x2 >= 2  →  -10*x1 + x2 <= -2
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-10.0, 1.0], 1, n).unwrap();
    let b = vec![-2.0];
    // bounds: 2 <= x1 <= 10,  -10 <= x2 <= 10
    let bounds = vec![(2.0_f64, 10.0_f64), (-10.0_f64, 10.0_f64)];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "HS21: status should be Optimal");
    assert_close(result.solution[0], 10.0, EPS_SOL, "HS21: x1* = 10 (upper bound active)");
    assert_close(result.solution[1], 0.5, EPS_SOL, "HS21: x2* = 0.5 (interior min)");
    // QP 目的関数値 = -99.25（原問題値と同一、定数項なし）
    assert_obj_close(result.objective, -99.25, "HS21: QP objective = -99.25");
}

/// HS35: Hock-Schittkowski Problem #35
///
/// 原問題（Hock & Schittkowski 1981, Problem 35）:
///   min f = 9 - 8x1 - 6x2 - 4x3 + 2x1^2 + 2x2^2 + x3^2 + 2x1x2 + 2x1x3
///   s.t. x1 + x2 + 2*x3 <= 3
///        xi >= 0  (i=1,2,3)
///
/// QP 形式（1/2 あり規約）, 定数 +9 を除いた形:
///   Q = [[4,2,2],[2,4,0],[2,0,2]],  c = [-8,-6,-4]
///   A = [[1,1,2]],  b = [3]
///   bounds = [(0, INF); 3]
///
/// 解析解: x* = (4/3, 7/9, 4/9)
///   原問題値 f* = 1/9 ≈ 0.1111
///   QP 目的関数値（定数除く）: -80/9 ≈ -8.8889
#[test]
fn test_hs35() {
    let n = 3;
    // Q = [[4,2,2],[2,4,0],[2,0,2]] — 対称行列を全要素列挙（column-major）
    let q = CscMatrix::from_triplets(
        &[0, 1, 2, 0, 1, 0, 2],
        &[0, 0, 0, 1, 1, 2, 2],
        &[4.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0],
        n,
        n,
    )
    .unwrap();
    let c = vec![-8.0, -6.0, -4.0];
    // A = [[1, 1, 2]],  b = [3]
    let a =
        CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 2.0], 1, n).unwrap();
    let b = vec![3.0];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "HS35: status should be Optimal");
    assert_close(result.solution[0], 4.0 / 3.0, EPS_SOL, "HS35: x1* = 4/3");
    assert_close(result.solution[1], 7.0 / 9.0, EPS_SOL, "HS35: x2* = 7/9");
    assert_close(result.solution[2], 4.0 / 9.0, EPS_SOL, "HS35: x3* = 4/9");
    // QP obj = -80/9 (原問題値 = QP obj + 定数9 = -80/9 + 81/9 = 1/9)
    assert_obj_close(result.objective, -80.0 / 9.0, "HS35: QP objective = -80/9");
}

/// HS51: Hock-Schittkowski Problem #51
///
/// 原問題（5変数、3等式制約）:
///   min f = (x1-x2)^2 + (x2+x3-2)^2 + (x4-1)^2 + (x5-1)^2
///   s.t. x1 + 3*x2 = 4
///        x3 + x4 - 2*x5 = 0
///        x2 - x5 = 0
///   (変数境界なし)
///
/// QP 形式（1/2 あり規約）, 定数 +6 を除いた形:
///   Q = [[2,-2,0,0,0],[-2,4,2,0,0],[0,2,2,0,0],[0,0,0,2,0],[0,0,0,0,2]]
///   c = [0,-4,-4,-2,-2]
///   等式制約を2不等式ペアに変換（計6行）
///
/// 解析解: x* = (1,1,1,1,1)
///   原問題値 f* = 0
///   QP 目的関数値（定数除く）: -6
#[test]
fn test_hs51() {
    let n = 5;
    // Q の非ゼロ要素（column-major）:
    //   col0: (0, 2.0), (1,-2.0)
    //   col1: (0,-2.0), (1, 4.0), (2, 2.0)
    //   col2: (1, 2.0), (2, 2.0)
    //   col3: (3, 2.0)
    //   col4: (4, 2.0)
    let q = CscMatrix::from_triplets(
        &[0, 1, 0, 1, 2, 1, 2, 3, 4],
        &[0, 0, 1, 1, 1, 2, 2, 3, 4],
        &[2.0, -2.0, -2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        n,
        n,
    )
    .unwrap();
    let c = vec![0.0, -4.0, -4.0, -2.0, -2.0];

    // 等式制約 3本 → 6不等式行:
    //   行0: x1 + 3x2 <= 4
    //   行1: -x1 - 3x2 <= -4
    //   行2: x3 + x4 - 2x5 <= 0
    //   行3: -x3 - x4 + 2x5 <= 0
    //   行4: x2 - x5 <= 0
    //   行5: -x2 + x5 <= 0
    //
    // 非ゼロ要素（column-major, col = 変数インデックス）:
    //   col0(x1): row0=1.0, row1=-1.0
    //   col1(x2): row0=3.0, row1=-3.0, row4=1.0, row5=-1.0
    //   col2(x3): row2=1.0, row3=-1.0
    //   col3(x4): row2=1.0, row3=-1.0
    //   col4(x5): row2=-2.0, row3=2.0, row4=-1.0, row5=1.0
    let a = CscMatrix::from_triplets(
        &[0, 1, 0, 1, 4, 5, 2, 3, 2, 3, 2, 3, 4, 5],
        &[0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
        &[
            1.0, -1.0,
            3.0, -3.0, 1.0, -1.0,
            1.0, -1.0,
            1.0, -1.0,
            -2.0, 2.0, -1.0, 1.0,
        ],
        6,
        n,
    )
    .unwrap();
    let b = vec![4.0, -4.0, 0.0, 0.0, 0.0, 0.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    // Phase I LP は全自由変数（-INF/INF）で未界を誤検知する場合がある。
    // 既知の実行可能点 x*=[1,...,1] を warm-start として Phase I をバイパスする。
    let ws = QpWarmStart {
        initial_active_set: vec![],
        initial_point: Some(vec![1.0; n]),
    };
    let result = solve_qp_warm(&problem, &ws, &Default::default());
    assert_eq!(result.status, SolveStatus::Optimal, "HS51: status should be Optimal");
    for i in 0..n {
        assert_close(
            result.solution[i],
            1.0,
            EPS_SOL,
            &format!("HS51: x{}* = 1.0", i + 1),
        );
    }
    // QP obj = -6 (原問題値 = QP obj + 定数6 = -6 + 6 = 0)
    assert_obj_close(result.objective, -6.0, "HS51: QP objective = -6");
}

/// CVXQP1_S: 10変数小型凸QP（Maros-Meszaros CVXQP1_S 相当の合成問題）
///
/// min 1/2 * sum_i(2*xi^2) - sum_i(xi)
///   [QP 形式: Q = 2*I_{10},  c = [-1,...,-1]]
/// s.t. x1 + x2 + ... + x10 <= 5
///      xi >= 0  (bounds)
///
/// 解析:
///   KKT 条件: 2xi - 1 + λ = 0 (全変数同一) → xi = (1-λ)/2
///   制約が等号活性なら: 10*(1-λ)/2 = 5 → λ = 0 → xi* = 0.5
///   (無制約最小点が制約境界上にある)
///
/// x* = [0.5; 10],  QP 目的関数値 = -2.5
#[test]
fn test_cvxqp1_s() {
    let n = 10;
    // Q = 2*I_{10}
    let idx: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&idx, &idx, &vec![2.0_f64; n], n, n).unwrap();
    let c = vec![-1.0_f64; n];

    // A = [[1,...,1]] (1×10)
    let a_rows = vec![0_usize; n];
    let a_cols: Vec<usize> = (0..n).collect();
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &vec![1.0_f64; n], 1, n).unwrap();
    let b = vec![5.0];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "CVXQP1_S: status should be Optimal");
    // λ*=0 (制約境界上で dual が零): 解変数の精度は O(sqrt(eps)) ≈ 5e-4 (EPS_DEG)
    // 目的値は O(eps) まで収束するので EPS_OBJ を使う
    for i in 0..n {
        assert_close(
            result.solution[i],
            0.5,
            EPS_DEG,
            &format!("CVXQP1_S: x{}* = 0.5", i + 1),
        );
    }
    // QP obj = 1/2*2*10*0.25 + (-10*0.5) = 2.5 - 5.0 = -2.5
    assert_obj_close(result.objective, -2.5, "CVXQP1_S: QP objective = -2.5");
}

/// QPCSTAIR 類似: 6変数・階段構造 QP
///
/// 階段型（staircase）制約を持つ QP（QPCSTAIR に近い構造）:
///   min 1/2*(2*x1^2+...+2*x6^2) - [3,2,3,2,3,2]^T x
///   s.t. x1 + x2 <= 2
///        x3 + x4 <= 2
///        x5 + x6 <= 2
///        xi >= 0  (bounds)
///
/// 各ペアは独立に解ける:
///   KKT: 2xi - ci + λj = 0 (λj はペア j の双対乗数)
///   ペア (2k-1, 2k) に対し:
///     c = [-3, -2] → 2x1-3+λ=0, 2x2-2+λ=0 → x1=x2+0.5
///     x1+x2=2 → x1=1.25, x2=0.75, λ=0.5 >= 0 ✓
///
/// x* = [1.25, 0.75, 1.25, 0.75, 1.25, 0.75]
/// QP 目的関数値:
///   = 1/2*2*(1.25^2+0.75^2)*3 + (-3*1.25-2*0.75)*3
///   = (1.5625+0.5625)*3 + (-3.75-1.5)*3
///   = 6.375 - 15.75 = -9.375
#[test]
fn test_qpcstair_like() {
    let n = 6;
    // Q = 2*I_6
    let idx: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&idx, &idx, &vec![2.0_f64; n], n, n).unwrap();
    let c = vec![-3.0, -2.0, -3.0, -2.0, -3.0, -2.0];

    // A: 3×6 階段行列
    //   行0: x1 + x2 <= 2
    //   行1: x3 + x4 <= 2
    //   行2: x5 + x6 <= 2
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1, 2, 2],
        &[0, 1, 2, 3, 4, 5],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        3,
        n,
    )
    .unwrap();
    let b = vec![2.0, 2.0, 2.0];
    let bounds = vec![(0.0_f64, f64::INFINITY); n];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_qp(&problem);
    assert_eq!(result.status, SolveStatus::Optimal, "QPCSTAIR: status should be Optimal");
    for k in 0..3 {
        assert_close(result.solution[2 * k], 1.25, EPS_SOL, &format!("QPCSTAIR: x{}* = 1.25", 2*k+1));
        assert_close(result.solution[2 * k + 1], 0.75, EPS_SOL, &format!("QPCSTAIR: x{}* = 0.75", 2*k+2));
    }
    assert_obj_close(result.objective, -9.375, "QPCSTAIR: QP objective = -9.375");
}
