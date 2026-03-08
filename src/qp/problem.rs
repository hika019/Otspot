//! QP問題のデータ構造定義
//!
//! 二次計画問題 min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub の
//! 構造体と求解結果を定義する。

use crate::problem::SolveStatus;
use crate::sparse::CscMatrix;

/// [`QpProblem::new`] が返す専用エラー型
#[non_exhaustive]
#[derive(Debug)]
pub enum QpProblemError {
    /// 行列・ベクトルの次元が不一致
    DimensionMismatch(String),
}

impl std::fmt::Display for QpProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QpProblemError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {}", msg),
        }
    }
}

impl std::error::Error for QpProblemError {}

/// 二次計画問題: min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub
#[derive(Debug, Clone)]
pub struct QpProblem {
    /// 目的関数二次項: n×n PSD行列（全要素格納）
    pub q: CscMatrix,
    /// 目的関数線形項: n次元ベクトル
    pub c: Vec<f64>,
    /// 制約行列: m×n（CSC形式）
    pub a: CscMatrix,
    /// 制約右辺: m次元ベクトル（Ax <= b）
    pub b: Vec<f64>,
    /// 変数境界 (lb, ub): n個。lb = -INF/ub = +INF は無制限
    pub bounds: Vec<(f64, f64)>,
    /// 変数数
    pub num_vars: usize,
    /// 制約数
    pub num_constraints: usize,
}

impl QpProblem {
    /// QP問題を生成する（次元チェック付き）
    pub fn new(
        q: CscMatrix,
        c: Vec<f64>,
        a: CscMatrix,
        b: Vec<f64>,
        bounds: Vec<(f64, f64)>,
    ) -> Result<Self, QpProblemError> {
        let n = c.len();
        let m = b.len();
        if q.nrows != n || q.ncols != n {
            return Err(QpProblemError::DimensionMismatch(
                format!("Q must be {}x{}, got {}x{}", n, n, q.nrows, q.ncols)
            ));
        }
        if a.nrows != m || a.ncols != n {
            return Err(QpProblemError::DimensionMismatch(
                format!("A must be {}x{}, got {}x{}", m, n, a.nrows, a.ncols)
            ));
        }
        if bounds.len() != n {
            return Err(QpProblemError::DimensionMismatch(
                format!("bounds length must be {}, got {}", n, bounds.len())
            ));
        }
        Ok(QpProblem { q, c, a, b, bounds, num_vars: n, num_constraints: m })
    }

    /// Q が全ゼロかどうかを検査する（LP退化ケース判定）
    pub fn is_zero_q(&self) -> bool {
        self.q.values.iter().all(|&v| v.abs() < 1e-12)
    }

    /// Q が対角行列かどうかを検査する
    pub fn is_diagonal_q(&self) -> bool {
        for col in 0..self.num_vars {
            let start = self.q.col_ptr[col];
            let end = self.q.col_ptr[col + 1];
            for k in start..end {
                let row = self.q.row_ind[k];
                if row != col && self.q.values[k].abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }

    /// 対角要素のベクトルを返す（対角行列のみ有効）
    #[allow(dead_code)]
    pub fn diagonal_q(&self) -> Vec<f64> {
        let mut diag = vec![0.0; self.num_vars];
        for (col, diag_val) in diag.iter_mut().enumerate().take(self.num_vars) {
            let start = self.q.col_ptr[col];
            let end = self.q.col_ptr[col + 1];
            for k in start..end {
                if self.q.row_ind[k] == col {
                    *diag_val = self.q.values[k];
                }
            }
        }
        diag
    }
}

/// QP求解結果（`SolverResult` の型エイリアス）
///
/// # Deprecated
///
/// `SolverResult` に統合された。`crate::problem::SolverResult` を直接使用すること。
#[deprecated(since = "0.1.0", note = "use SolverResult (LP/QP unified result type) instead")]
#[allow(unused)]
pub type QpResult = crate::problem::SolverResult;

impl crate::problem::SolverResult {
    /// Infeasible結果を生成（QP用）
    pub fn infeasible() -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::Infeasible,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
            ..Default::default()
        }
    }

    /// MaxIterations結果を生成（QP用）。真のイテレーション上限到達時のみ使用すること。
    pub fn max_iterations(x: Vec<f64>, obj: f64, active: Vec<usize>, iters: usize) -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::MaxIterations,
            objective: obj,
            solution: x,
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: active,
            iterations: iters,
            ..Default::default()
        }
    }

    /// NumericalError結果を生成（KKT分解失敗・Q特異・Phase1数値困難時）
    pub fn numerical_error() -> Self {
        crate::problem::SolverResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
            ..Default::default()
        }
    }
}

/// Warm-start情報: SQP等で前反復の活性集合を次反復に引き継ぐ
#[derive(Debug, Clone)]
pub struct QpWarmStart {
    /// 初期活性制約インデックス（QpResult.active_set から取得して渡す）
    pub initial_active_set: Vec<usize>,
    /// 初期点 x_0（None のときは LP Phase I で計算）
    pub initial_point: Option<Vec<f64>>,
}
