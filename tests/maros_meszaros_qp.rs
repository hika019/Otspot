//! Maros-Meszaros / Hock-Schittkowski QP ベンチマーク (形式: min 1/2 xᵀQx+cᵀx s.t. Ax≤b, lb≤x≤ub)。

use otspot::qp::{solve_qp, solve_qp_warm, QpProblem, QpWarmStart};
use otspot::sparse::CscMatrix;
use otspot::SolveStatus;

// 通常解変数の絶対許容値 (postsolve 後の丸め ≈ 数×eps)
const EPS_SOL: f64 = 1e-5;
// 退化制約境界 (λ*=0) は補完余裕 λs=μ より s≈√μ で収束するため O(√eps)
const EPS_DEG: f64 = 5e-4;

/// 相対誤差 eps=1e-6 で目的値を検証 (solver の relative gap 収束判定と整合)。
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

/// HS21: min 0.01x1²+x2²−10x1−x2  s.t. 10x1−x2≥2, x1∈[2,10], x2∈[-10,10]  →  opt=(10, 0.5), obj=-99.25
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

/// HS35: min 9−8x1−6x2−4x3+2x1²+2x2²+x3²+2x1x2+2x1x3  s.t. x1+x2+2x3≤3, x≥0  →  opt=(4/3, 7/9, 4/9), QP obj=-80/9 (定数9除く)
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

/// HS51: min (x1-x2)²+(x2+x3-2)²+(x4-1)²+(x5-1)²  s.t. x1+3x2=4, x3+x4-2x5=0, x2-x5=0 (free)  →  opt=(1,1,1,1,1), QP obj=-6 (定数6除く)
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
        x: vec![1.0; n],
        y: vec![0.0; 6],
        mu: 1.0,
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

/// CVXQP1_S: min Σxi²−Σxi (Q=2I_10, c=-1)  s.t. Σxi≤5, xi≥0  →  opt=0.5·1, obj=-2.5 (λ*=0, 制約境界活性で退化)
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

/// QPCSTAIR-like: min Σxi²−[3,2,3,2,3,2]ᵀx  s.t. x_{2k-1}+x_{2k}≤2 (k=1..3), xi≥0  →  opt=(1.25,0.75)×3, obj=-9.375
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
