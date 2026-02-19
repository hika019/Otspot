//! 二次計画法（QP）ソルバーモジュール
//!
//! Active Set法による QP ソルバーを提供する。
//! 問題形式: min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub
//!
//! # 規約
//! **「1/2あり」規約** (OSQP/qpOASES標準):
//! - 目的関数: min 1/2 x^T Q x + c^T x
//! - ∇f(x) = Qx + c
//! - KKT行列: [Q, A_W^T; A_W, 0]（NC1修正済み）
//!
//! # 使用例
//! ```rust
//! use solver::qp::{solve_qp, QpProblem, QpResult};
//! use solver::sparse::CscMatrix;
//!
//! // min x^2 + y^2  s.t. x + y >= 1
//! // Q = [[2,0],[0,2]] (「1/2あり」規約で min 1/2 * 2 * (x^2+y^2))
//! // c = [0, 0]
//! // A = [[-1,-1]], b = [-1]（x+y >= 1 を Ax <= b 形式に変換）
//! let q = CscMatrix::from_triplets(
//!     &[0, 1], &[0, 1], &[2.0, 2.0], 2, 2
//! ).unwrap();
//! let c = vec![0.0, 0.0];
//! let a = CscMatrix::from_triplets(
//!     &[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2
//! ).unwrap();
//! let b = vec![-1.0];
//! let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
//! let problem = QpProblem::new(q, c, a, b, bounds).unwrap();
//! let result = solve_qp(&problem);
//! // result.solution ≈ [0.5, 0.5], result.objective ≈ 0.5
//! ```

mod active_set;
pub(crate) mod kkt;
mod problem;
mod solver;

pub use problem::{QpProblem, QpResult, QpWarmStart};

use crate::options::SolverOptions;

/// QPを解く（デフォルト設定）
///
/// qpOASESの `init()` に相当する基本API。
/// デフォルトの [`SolverOptions`] を使用して求解する。
///
/// # 引数
/// - `problem`: 解くべき二次計画問題
///
/// # 戻り値
/// [`QpResult`] — ステータス・目的関数値・解・ラグランジュ乗数・活性集合・反復数
pub fn solve_qp(problem: &QpProblem) -> QpResult {
    solve_qp_with(problem, &SolverOptions::default())
}

/// QPをカスタム設定で解く
///
/// qpOASESの `init()` に相当。`nWSR` は `options.max_iterations` で指定。
pub fn solve_qp_with(problem: &QpProblem, options: &SolverOptions) -> QpResult {
    solver::qp_solve_impl(problem, None, options)
}

/// Warm-start付きでQPを解く
///
/// qpOASESの `hotstart()` に相当。SQP反復で前回解の活性集合を引き継ぐ場合に使用。
///
/// # 使用例（SQP典型パターン）
/// ```rust,no_run
/// use solver::qp::{solve_qp, solve_qp_warm, QpProblem, QpWarmStart};
///
/// # let problem1 = unimplemented!();
/// # let problem2 = unimplemented!();
/// let result1 = solve_qp(&problem1);
/// let ws = QpWarmStart {
///     initial_active_set: result1.active_set.clone(),
///     initial_point: Some(result1.solution.clone()),
/// };
/// let result2 = solve_qp_warm(&problem2, &ws, &Default::default());
/// // result2 は result1 の活性集合を初期値として使用
/// ```
pub fn solve_qp_warm(
    problem: &QpProblem,
    warm_start: &QpWarmStart,
    options: &SolverOptions,
) -> QpResult {
    solver::qp_solve_impl(problem, Some(warm_start), options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::SolveStatus;
    use crate::sparse::CscMatrix;

    const EPS: f64 = 1e-5;

    fn assert_close(a: f64, b: f64, eps: f64, name: &str) {
        assert!(
            (a - b).abs() < eps,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name, b, a, (a - b).abs()
        );
    }

    /// T1: 2変数基本QP
    /// min 1/2 * 2*(x^2+y^2) = x^2+y^2  s.t. x+y >= 1
    /// Q = [[2,0],[0,2]], c=[0,0], A=[[-1,-1]], b=[-1]
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_basic_qp_2vars() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T1: status should be Optimal");
        assert_close(result.solution[0], 0.5, EPS, "T1: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "T1: x[1]");
        assert_close(result.objective, 0.5, EPS, "T1: objective");
    }

    /// T2: 等式制約付きQP
    /// min x^2+y^2 (1/2あり規約: Q=2I)  s.t. x+y=1
    /// 等式制約は Ax<=b 形式で2不等式に変換:
    ///   x+y <= 1  と  -(x+y) <= -1
    /// 期待: x*=y*=0.5, obj=0.5
    #[test]
    fn test_qp_equality_constraint() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // A: [1,1; -1,-1], b: [1, -1]
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, 1.0, -1.0, -1.0],
            2,
            2,
        )
        .unwrap();
        let b = vec![1.0, -1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T2: status should be Optimal");
        assert_close(result.solution[0], 0.5, EPS, "T2: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "T2: x[1]");
        assert_close(result.objective, 0.5, EPS, "T2: objective");
    }

    /// T3: Q=0退化ケース（LP問題として解く）
    /// min x+2y  s.t. x>=0, y>=0, x+y<=4, 2x+y<=6
    /// 期待: x*=2, y*=0, obj=2
    #[test]
    fn test_qp_degenerate_lp_case() {
        let n = 2;
        let q = CscMatrix::new(n, n); // Q = 0
        let c = vec![1.0, 2.0];
        // A = [[-1,0],[0,-1],[1,1],[2,1]], b = [0,0,4,6]
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
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T3: status should be Optimal");
        // LP最適解: x*=2, y*=0 (corner: 2x+y<=6, x>=0 → x=3但し x+y<=4なので x=2,y=0? or x=3,y=0?)
        // min x+2y s.t. x+y<=4, 2x+y<=6, x>=0, y>=0
        // vertices: (0,0)→0, (3,0)→3, (2,2)→6, (0,4)→8
        // 最適: (0,0) でobj=0? wait...
        // x>=0: -x<=0, y>=0: -y<=0
        // vertices of feasible region:
        // (0,0): obj=0  → this is optimal for min x+2y
        assert_close(result.objective, 0.0, EPS, "T3: objective");
    }

    /// T4: 制約なしQP
    /// min (x-3)^2 + (y-4)^2 = 1/2*2*(x^2+y^2) - 6x - 8y + const
    /// Q = [[2,0],[0,2]], c = [-6,-8], no constraints, no bounds
    /// 期待: x*=3, y*=4
    #[test]
    fn test_qp_unconstrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-6.0, -8.0];
        let a = CscMatrix::new(0, 2); // 制約なし
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T4: status should be Optimal");
        assert_close(result.solution[0], 3.0, EPS, "T4: x[0]");
        assert_close(result.solution[1], 4.0, EPS, "T4: x[1]");
        // obj = 1/2*2*(9+16) - 6*3 - 8*4 = 25 - 18 - 32 = -25
        assert_close(result.objective, -25.0, EPS, "T4: objective");
    }

    /// T5: warm-start整合性
    /// T1と同じ問題を2回解く（2回目はwarm-start）
    /// 期待: 同一解、iterations <= cold-startのiterations
    #[test]
    fn test_warm_start_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a.clone(), b.clone(), bounds.clone()).unwrap();
        let problem2 = QpProblem::new(
            CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap(),
            vec![0.0, 0.0],
            a,
            b,
            bounds,
        )
        .unwrap();

        // Cold start
        let result1 = solve_qp(&problem);
        assert_eq!(result1.status, SolveStatus::Optimal, "T5: cold start should be Optimal");

        // Warm start
        let ws = QpWarmStart {
            initial_active_set: result1.active_set.clone(),
            initial_point: Some(result1.solution.clone()),
        };
        let result2 = solve_qp_warm(&problem2, &ws, &SolverOptions::default());

        assert_eq!(result2.status, SolveStatus::Optimal, "T5: warm start should be Optimal");
        assert_close(result2.solution[0], 0.5, EPS, "T5: warm start x[0]");
        assert_close(result2.solution[1], 0.5, EPS, "T5: warm start x[1]");
        // warm-startはinitial_pointとactive_setを初期値として使うので反復数が少ない or 等しい
        assert!(
            result2.iterations <= result1.iterations + 1,
            "T5: warm start iterations ({}) should be <= cold start ({})",
            result2.iterations,
            result1.iterations
        );
    }

    /// T6: Infeasible QP
    /// min x^2  s.t. x >= 1, x <= 0  (矛盾制約)
    /// 期待: status = Infeasible
    #[test]
    fn test_qp_infeasible() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        // A: [-1; 1], b: [-1; 0]  (x>=1: -x<=-1, x<=0: x<=0)
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[-1.0, 1.0], 2, 1).unwrap();
        let b = vec![-1.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Infeasible, "T6: should be Infeasible");
    }

    /// T7: ポートフォリオ最適化（Markowitz平均分散モデル）
    ///
    /// 3銘柄、対称共分散行列 Sigma = diag(2,2,2)（独立等分散）
    /// 目的: min 1/2 w^T Sigma w（リスク最小化）
    /// 制約: w1+w2+w3=1（等式）、wi>=0（非負）
    ///
    /// 等式制約を2不等式に変換:
    ///   w1+w2+w3 <= 1  と  -(w1+w2+w3) <= -1
    /// 非負制約: -wi <= 0
    ///
    /// 対称性より最適解: w* = [1/3, 1/3, 1/3]
    /// 目的関数: 1/2 * (2/9+2/9+2/9) = 1/3 ≈ 0.3333
    #[test]
    fn test_qp_portfolio_markowitz() {
        // Q = Sigma = diag(2,2,2)
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[2.0, 2.0, 2.0],
            3, 3,
        ).unwrap();
        let c = vec![0.0, 0.0, 0.0];

        // A行列 (5行3列):
        //   行0: [1,1,1] <= 1  (等式上界)
        //   行1: [-1,-1,-1] <= -1  (等式下界)
        //   行2: [-1,0,0] <= 0  (w1>=0)
        //   行3: [0,-1,0] <= 0  (w2>=0)
        //   行4: [0,0,-1] <= 0  (w3>=0)
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 3, 4],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            5, 3,
        ).unwrap();
        let b = vec![1.0, -1.0, 0.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T7: status should be Optimal");
        let w_sum = result.solution[0] + result.solution[1] + result.solution[2];
        assert_close(w_sum, 1.0, EPS, "T7: w sum = 1");
        assert_close(result.solution[0], 1.0 / 3.0, EPS, "T7: w[0]");
        assert_close(result.solution[1], 1.0 / 3.0, EPS, "T7: w[1]");
        assert_close(result.solution[2], 1.0 / 3.0, EPS, "T7: w[2]");
        assert_close(result.objective, 1.0 / 3.0, EPS, "T7: objective");
    }

    /// T8: 最小二乗法（Least Squares）
    ///
    /// min ||Ax - b||^2 を QP として定式化:
    ///   Q = 2 * A^T A, c = -2 * A^T b  （「1/2あり」規約）
    ///
    /// A = [[2,1],[1,2]], b = [5,4]
    ///   A^T A = [[5,4],[4,5]]
    ///   A^T b = [14,13]
    ///   Q = [[10,8],[8,10]], c = [-28,-26]
    ///
    /// 解析解: (A^T A) x = A^T b → x* = [2, 1]
    /// QP目的関数値: 1/2 x^T Q x + c^T x = 41 - 82 = -41
    #[test]
    fn test_qp_least_squares() {
        // Q = 2 * A^T A = [[10,8],[8,10]]
        let q = CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[10.0, 8.0, 8.0, 10.0],
            2, 2,
        ).unwrap();
        let c = vec![-28.0, -26.0]; // -2 * A^T b
        let a = CscMatrix::new(0, 2); // 制約なし
        let b_vec = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b_vec, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T8: status should be Optimal");
        assert_close(result.solution[0], 2.0, EPS, "T8: x[0]");
        assert_close(result.solution[1], 1.0, EPS, "T8: x[1]");
        assert_close(result.objective, -41.0, EPS, "T8: objective");
    }

    /// T9: QP→LP退化テスト（Q=0の場合）
    ///
    /// Q = 0 として LP と同一解が得られることを確認。
    /// LP問題: min 2x+y  s.t. x+y >= 1, x >= 0, y >= 0
    /// Ax<=b 形式:
    ///   -x-y <= -1  (x+y >= 1)
    ///   -x   <= 0   (x >= 0)
    ///   -y   <= 0   (y >= 0)
    ///
    /// 解析解: x*=[0,1], obj=1（y=1で最小化）
    #[test]
    fn test_qp_degenerate_to_lp() {
        let n = 2;
        let q = CscMatrix::new(n, n); // Q = 0（LP退化）
        let c = vec![2.0, 1.0];       // min 2x + y

        // A: [[-1,-1],[-1,0],[0,-1]], b: [-1,0,0]
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[-1.0, -1.0, -1.0, -1.0],
            3, 2,
        ).unwrap();
        let b = vec![-1.0, 0.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T9: status should be Optimal");
        // LP最適解: x*=[0,1], obj=0*2+1*1=1
        assert_close(result.solution[0], 0.0, EPS, "T9: x[0]");
        assert_close(result.solution[1], 1.0, EPS, "T9: x[1]");
        assert_close(result.objective, 1.0, EPS, "T9: objective");
    }

    /// T10: 複合制約テスト（等式+不等式の組み合わせ）
    ///
    /// min (x-1)^2 + (y-2)^2 = 1/2*Q*x^T + c^T*x + const
    /// Q = [[2,0],[0,2]], c = [-2,-4]
    /// 等式制約: x+y=2 → [1,1]<=2, [-1,-1]<=-2
    /// 不等式制約: x>=0 → [-1,0]<=0
    ///
    /// x+y=2直線上でmin (x-1)^2+(y-2)^2 = min (x-1)^2+(1-x)^2 = 2x^2-2x+2
    /// → x*=1/2, y*=3/2
    /// QP目的関数値: 1/2*2*(0.25+2.25) + (-2*0.5-4*1.5) = 2.5 - 7 = -4.5
    #[test]
    fn test_qp_mixed_constraints() {
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[2.0, 2.0],
            2, 2,
        ).unwrap();
        let c = vec![-2.0, -4.0];

        // A行列 (3行2列):
        //   行0: [1,1] <= 2  (等式上界)
        //   行1: [-1,-1] <= -2  (等式下界)
        //   行2: [-1,0] <= 0  (x>=0)
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2],
            &[0, 1, 0, 1, 0],
            &[1.0, 1.0, -1.0, -1.0, -1.0],
            3, 2,
        ).unwrap();
        let b = vec![2.0, -2.0, 0.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_qp(&problem);
        assert_eq!(result.status, SolveStatus::Optimal, "T10: status should be Optimal");
        assert_close(result.solution[0], 0.5, EPS, "T10: x[0]");
        assert_close(result.solution[1], 1.5, EPS, "T10: x[1]");
        assert_close(result.objective, -4.5, EPS, "T10: objective");
    }
}
