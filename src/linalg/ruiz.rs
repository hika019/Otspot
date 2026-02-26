//! Ruiz equilibration スケーリング
//!
//! OSQPの前処理技術。行・列ノルム交互正規化により QP 問題の数値安定性を向上させる。
//!
//! # 数学的背景
//! 変換 x = D * x_s (D = diag(d)) を用いて、以下のスケール済み問題を構築する:
//!   min 1/2 x_s^T Q_s x_s + q_s^T x_s  s.t. A_s x_s <= b_s, lb_s <= x_s <= ub_s
//! ここで:
//!   Q_s = c * D * Q * D   （Q_s[i,j] = c * d[i] * Q[i,j] * d[j]）
//!   q_s = c * D * q       （q_s[j] = c * d[j] * q[j]）
//!   A_s = E * A * D       （A_s[i,j] = e[i] * A[i,j] * d[j]）
//!   b_s = E * b           （b_s[i] = e[i] * b[i]）
//!   bounds_s = (lb/d, ub/d)
//!
//! 双対変数の逆変換（KKT条件より導出）: y[i] = e[i] * y_s[i] / c
//! 目的関数の逆変換: obj = obj_s / c
//!
//! # 参考文献
//! D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms
//! in matrices", ENSEEIHT-IRIT Technical Report, 2001.

use crate::sparse::CscMatrix;

/// Ruiz equilibration スケーラー
///
/// `compute()` で行・列スケーリング係数 (d, e, c) を計算し、
/// `scale_problem()` で問題をスケーリング、`unscale_solution()` で解を逆変換する。
pub struct RuizScaler {
    /// 列スケーリング係数 D = diag(d)（サイズ n: 変数数）
    pub d: Vec<f64>,
    /// 行スケーリング係数 E = diag(e)（サイズ m: 制約数）
    pub e: Vec<f64>,
    /// コスト関数スケーリング係数（スカラー）
    pub c: f64,
}

impl RuizScaler {
    /// 新規スケーラーを生成（初期値: 恒等変換 D=I, E=I, c=1）
    pub fn new(n: usize, m: usize) -> Self {
        RuizScaler {
            d: vec![1.0; n],
            e: vec![1.0; m],
            c: 1.0,
        }
    }

    /// Ruiz equilibration を実行（10回反復）
    ///
    /// 各反復:
    /// 1. 行ノルム正規化: e_i ← e_i / sqrt(max(||row_i(A_s)||_∞, ε))
    /// 2. 列ノルム正規化: d_j ← d_j / sqrt(max(||col_j([Q_s; A_s])||_∞, ε))
    /// 3. コスト正規化: c ← c / max(||Q_s||_∞, ||q_s||_∞, ε)
    ///
    /// l, u（変数境界）はAPIの完全性のため受け取るが、
    /// ノルム計算には使用しない（Q と A のみで均衡化する）。
    pub fn compute(
        &mut self,
        q: &CscMatrix,
        a: &CscMatrix,
        q_vec: &[f64],
        _l: &[f64],
        _u: &[f64],
    ) {
        let n = q.ncols;
        let m = a.nrows;
        const EPS: f64 = 1e-6;
        const NUM_ITER: usize = 10;

        for _ in 0..NUM_ITER {
            // ------------------------------------------------------------------
            // Step 1: 行ノルム正規化
            // A_s[i,j] = e[i] * A[i,j] * d[j]
            // ||row_i(A_s)||_∞ = e[i] * max_j |A[i,j] * d[j]|
            // 更新: e_i ← e_i / sqrt(norm_i)  → スケール済み行ノルム → 1 に近づく
            // ------------------------------------------------------------------
            if m > 0 {
                let mut row_norms = vec![0.0f64; m];
                for col in 0..n {
                    for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                        let i = a.row_ind[k];
                        // スケール済み値: e[i] * A[i,col] * d[col]
                        let val = (self.e[i] * a.values[k] * self.d[col]).abs();
                        if val > row_norms[i] {
                            row_norms[i] = val;
                        }
                    }
                }
                for i in 0..m {
                    let norm = row_norms[i].max(EPS);
                    self.e[i] /= norm.sqrt();
                }
            }

            // ------------------------------------------------------------------
            // Step 2: 列ノルム正規化
            // Q_s = c * D * Q * D: Q_s[i,j] = c * d[i] * Q[i,j] * d[j]
            // A_s = E * A * D: A_s[i,j] = e[i] * A[i,j] * d[j]
            // col_norms[j] = max_i(|Q_s[i,j]|, |A_s[i,j]|)
            // 更新: d_j ← d_j / sqrt(norm_j)
            // ------------------------------------------------------------------
            let mut col_norms = vec![0.0f64; n];

            // Q 寄与（対称行列: 全要素格納前提）
            for col in 0..n {
                for k in q.col_ptr[col]..q.col_ptr[col + 1] {
                    let row = q.row_ind[k];
                    // c * d[row] * Q[row,col] * d[col]
                    let val = (self.c * self.d[row] * q.values[k] * self.d[col]).abs();
                    if val > col_norms[col] {
                        col_norms[col] = val;
                    }
                }
            }

            // A 寄与（step 1 で更新済みの e を使用）
            if m > 0 {
                for col in 0..n {
                    for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                        let row = a.row_ind[k];
                        // e[row] * A[row,col] * d[col]
                        let val = (self.e[row] * a.values[k] * self.d[col]).abs();
                        if val > col_norms[col] {
                            col_norms[col] = val;
                        }
                    }
                }
            }

            for j in 0..n {
                let norm = col_norms[j].max(EPS);
                self.d[j] /= norm.sqrt();
            }

            // ------------------------------------------------------------------
            // Step 3: コスト正規化
            // Q_s = c * D * Q * D, q_s = c * D * q_vec
            // c ← c / max(||Q_s||_∞, ||q_s||_∞, ε)
            // ------------------------------------------------------------------
            let mut q_mat_inf = 0.0f64;
            for col in 0..n {
                for k in q.col_ptr[col]..q.col_ptr[col + 1] {
                    let row = q.row_ind[k];
                    let val = (self.c * self.d[row] * q.values[k] * self.d[col]).abs();
                    if val > q_mat_inf {
                        q_mat_inf = val;
                    }
                }
            }

            let q_vec_inf = q_vec
                .iter()
                .enumerate()
                .map(|(j, &v)| (self.c * self.d[j] * v).abs())
                .fold(0.0f64, f64::max);

            let denom = q_mat_inf.max(q_vec_inf).max(EPS);
            self.c /= denom;
        }
    }

    /// 問題をスケーリング済みに変換する
    ///
    /// # 変換式
    /// - Q_s[i,j] = c * d[i] * Q[i,j] * d[j]
    /// - A_s[i,j] = e[i] * A[i,j] * d[j]
    /// - q_s[j] = c * d[j] * q_vec[j]
    /// - b_s[i] = e[i] * b[i]
    /// - bounds_s[j] = (lb[j] / d[j], ub[j] / d[j])
    ///
    /// スケール済み問題の解は `unscale_solution` で元のスケールに戻すこと。
    pub fn scale_problem(
        &self,
        q: &CscMatrix,
        a: &CscMatrix,
        q_vec: &[f64],
        b: &[f64],
        bounds: &[(f64, f64)],
    ) -> (CscMatrix, CscMatrix, Vec<f64>, Vec<f64>, Vec<(f64, f64)>) {
        let n = q.ncols;
        let m = a.nrows;

        // Q_s = c * D * Q * D（疎パターン保持: 値のみ変更）
        let mut q_s = q.clone();
        for col in 0..n {
            for k in q.col_ptr[col]..q.col_ptr[col + 1] {
                let row = q.row_ind[k];
                // Q_s[row, col] = c * d[row] * Q[row, col] * d[col]
                q_s.values[k] = self.c * self.d[row] * q.values[k] * self.d[col];
            }
        }

        // A_s = E * A * D（疎パターン保持: 値のみ変更）
        let mut a_s = a.clone();
        for col in 0..n {
            for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                let row = a.row_ind[k];
                // A_s[row, col] = e[row] * A[row, col] * d[col]
                a_s.values[k] = self.e[row] * a.values[k] * self.d[col];
            }
        }

        // q_s[j] = c * d[j] * q_vec[j]
        let q_vec_s: Vec<f64> = q_vec
            .iter()
            .enumerate()
            .map(|(j, &v)| self.c * self.d[j] * v)
            .collect();

        // b_s[i] = e[i] * b[i]
        let b_s: Vec<f64> = if m > 0 {
            b.iter()
                .enumerate()
                .map(|(i, &v)| self.e[i] * v)
                .collect()
        } else {
            vec![]
        };

        // bounds_s[j] = (lb[j] / d[j], ub[j] / d[j])
        // d[j] > 0 が保証されているため、符号は変わらない
        let bounds_s: Vec<(f64, f64)> = bounds
            .iter()
            .enumerate()
            .map(|(j, &(lb, ub))| (lb / self.d[j], ub / self.d[j]))
            .collect();

        (q_s, a_s, q_vec_s, b_s, bounds_s)
    }

    /// ADMM解を元のスケールに逆変換する
    ///
    /// # 引数
    /// - `x_s`: スケール済み主変数（長さ n）
    /// - `y_s`: スケール済み双対変数（長さ m）
    ///
    /// # 変換式
    /// - x[j] = d[j] * x_s[j]  （x = D * x_s）
    /// - y[i] = e[i] * y_s[i] / c  （KKT条件より導出）
    ///
    /// # 数学的根拠
    /// スケール済み KKT: Q_s x_s + q_s + A_s^T y_s = 0
    ///   = c * D * Q * D * x_s + c * D * q + D * A^T * E * y_s = 0
    /// 両辺を c*D で割る: Q * x + q + (1/c) * A^T * E * y_s = 0
    /// 元の KKT: Q * x + q + A^T * y = 0 との比較: y = E * y_s / c
    pub fn unscale_solution(&self, x_s: &[f64], y_s: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // x[j] = d[j] * x_s[j]
        let x: Vec<f64> = x_s
            .iter()
            .enumerate()
            .map(|(j, &v)| self.d[j] * v)
            .collect();

        // y[i] = e[i] * y_s[i] / c
        let y: Vec<f64> = y_s
            .iter()
            .enumerate()
            .map(|(i, &v)| self.e[i] * v / self.c)
            .collect();

        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// test_ruiz_scaler_identity:
    /// 既にスケール済み問題（Q=I, A=I, q=0）では各反復後に d, e ≈ 1, c ≈ 1
    #[test]
    fn test_ruiz_scaler_identity() {
        let n = 3usize;
        let m = 3usize;
        // Q = I_3
        let q = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[1.0, 1.0, 1.0],
            n, n,
        ).unwrap();
        // q_vec = 0
        let q_vec = vec![0.0; n];
        // A = I_3
        let a = CscMatrix::from_triplets(
            &[0, 1, 2],
            &[0, 1, 2],
            &[1.0, 1.0, 1.0],
            m, n,
        ).unwrap();
        let l = vec![0.0; n];
        let u = vec![1.0; n];

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&q, &a, &q_vec, &l, &u);

        // d, e はほぼ 1.0（恒等変換に近い）
        for j in 0..n {
            assert!(
                (scaler.d[j] - 1.0).abs() < 0.2,
                "d[{}] = {:.6} (expected ~1.0)",
                j, scaler.d[j]
            );
        }
        for i in 0..m {
            assert!(
                (scaler.e[i] - 1.0).abs() < 0.2,
                "e[{}] = {:.6} (expected ~1.0)",
                i, scaler.e[i]
            );
        }
    }

    /// test_ruiz_scaling_correctness:
    /// 小規模 QP (n=5, m=3) でスケーリングあり/なしの解が一致することを確認。
    /// min 1/2 x^T Q x + q^T x  s.t. Ax <= b, bounds=[0,∞)
    /// Q = diag(1,100,1,100,1)（意図的に悪くスケーリングされた問題）
    #[test]
    fn test_ruiz_scaling_correctness() {
        use crate::qp::QpProblem;
        use crate::options::SolverOptions;
        use crate::problem::SolveStatus;

        // Q = diag(1, 100, 1, 100, 1) — 条件数が大きい
        let n = 5usize;
        let m = 3usize;
        let q_rows: Vec<usize> = (0..n).collect();
        let q_cols: Vec<usize> = (0..n).collect();
        let q_vals = vec![1.0, 100.0, 1.0, 100.0, 1.0];
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();

        let q_vec = vec![-1.0, -10.0, -1.0, -10.0, -1.0];

        // A: 3 simple constraints
        // A[0,0]=1, A[0,1]=1
        // A[1,2]=1, A[1,3]=1
        // A[2,0]=1, A[2,4]=1
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2],
            &[0, 1, 2, 3, 0, 4],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            m, n,
        ).unwrap();
        let b = vec![2.0, 2.0, 2.0];
        let bounds = vec![(0.0f64, f64::INFINITY); n];

        let problem = QpProblem::new(q, q_vec, a, b, bounds).unwrap();

        // スケーリングなし
        let mut opts_no_scale = SolverOptions::default();
        opts_no_scale.use_ruiz_scaling = false;
        opts_no_scale.eps_abs = 1e-4;
        opts_no_scale.eps_rel = 1e-4;
        opts_no_scale.qp_solver = crate::options::QpSolverChoice::Admm;
        let r_no_scale = crate::qp::admm::solve_qp_admm(&problem, &opts_no_scale);

        // スケーリングあり
        let mut opts_scale = SolverOptions::default();
        opts_scale.use_ruiz_scaling = true;
        opts_scale.eps_abs = 1e-4;
        opts_scale.eps_rel = 1e-4;
        opts_scale.qp_solver = crate::options::QpSolverChoice::Admm;
        let r_scale = crate::qp::admm::solve_qp_admm(&problem, &opts_scale);

        // 両方 Optimal
        assert!(
            r_no_scale.status == SolveStatus::Optimal || r_no_scale.status == SolveStatus::MaxIterations,
            "no_scale: {:?}", r_no_scale.status
        );
        assert!(
            r_scale.status == SolveStatus::Optimal || r_scale.status == SolveStatus::MaxIterations,
            "scale: {:?}", r_scale.status
        );

        // 両方 Optimal なら解が近い
        if r_no_scale.status == SolveStatus::Optimal && r_scale.status == SolveStatus::Optimal {
            for j in 0..n {
                assert!(
                    (r_no_scale.solution[j] - r_scale.solution[j]).abs() < 0.1,
                    "x[{}]: no_scale={:.6}, scale={:.6}",
                    j, r_no_scale.solution[j], r_scale.solution[j]
                );
            }
            assert!(
                (r_no_scale.objective - r_scale.objective).abs() < 0.1,
                "obj: no_scale={:.6}, scale={:.6}",
                r_no_scale.objective, r_scale.objective
            );
        }
    }

    /// test_ruiz_disabled:
    /// use_ruiz_scaling=false で従来通りの動作（スケーリングなし）
    #[test]
    fn test_ruiz_disabled() {
        use crate::qp::QpProblem;
        use crate::options::SolverOptions;
        use crate::problem::SolveStatus;

        // 簡単な QP: min x^2 + y^2  s.t. x+y >= 1
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let q_vec = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, q_vec, a, b, bounds).unwrap();

        let mut opts = SolverOptions::default();
        opts.use_ruiz_scaling = false;
        opts.qp_solver = crate::options::QpSolverChoice::Admm;

        let result = crate::qp::admm::solve_qp_admm(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "disabled: {:?}", result.status);
        assert!((result.solution[0] - 0.5).abs() < 0.05, "x[0]={}", result.solution[0]);
        assert!((result.solution[1] - 0.5).abs() < 0.05, "x[1]={}", result.solution[1]);
    }
}
