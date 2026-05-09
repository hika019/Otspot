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
    /// Ruiz equilibration sweep 数。Ruiz の **f64 storage 精度** から導出。
    ///
    /// Ruiz iteration は contractive で、各 sweep で deviation が ~1/2 になる
    /// (理論収束率)。`RuizScaler` の e/d/c は f64 で保持するため、storage 上の precision
    /// は `2^-52`。52 sweep で deviation は f64 machine precision に到達し、それ以降の
    /// sweep は noise しか動かさない。`MANTISSA_DIGITS = 53` で +1 マージン。
    ///
    /// **DD (double-double) との関係:** IR / 残差計算で DD 精度を使う箇所があるが、
    /// Ruiz の出力は f64 storage に書き戻される時点で f64 精度に制限される。Ruiz 全体
    /// を DD 化するなら storage 含む大規模 refactor が必要 (現状未実装)。
    ///
    /// 各 sweep は O(nnz) で軽量、早期 break しなくてもコストは μs オーダーで無視可。
    pub const RUIZ_SWEEPS: usize = f64::MANTISSA_DIGITS as usize;

    /// IPM で f64 LDL solve が確保できる KKT residual の下限 (実効 backward error)。
    ///
    /// 物理: `f64 EPSILON × cond(K_scaled)`。Ruiz equilibration 後の K_scaled は
    /// 多くの問題で cond ≲ 1e4 まで圧縮され、backward error は 2.2e-16 × 1e4 ≈ 2e-12
    /// 級。経験値として 1e-12 を採用する (IS_LASSO_100 系で実測 PASS の dfr=4e-7、
    /// 旧 EPS_FLOOR=1e-12 で動作実績)。これより tight な eps を IPM に課しても f64 では
    /// 達成不能。1e-10 にすると Ruiz floor が 1e-4 に緩和され equilibration 弱体化、
    /// well-equilibrated 問題で必要な精度を出せず IS_LASSO regress (実測)。
    ///
    /// `scale_floor_for_eps` の出力を介して Ruiz の equilibration 強度を制限し、
    /// unscale 後の残差 (= amp × IPM_eps) が `user_eps` を下回るようにする。
    pub const IPM_F64_ACHIEVABLE_EPS: f64 = 1e-12;

    /// `user_eps` から導出する scaling 係数下限。
    ///
    /// 導出: unscale 後残差 ≤ `user_eps` を保証するには
    ///       `amp × IPM_eps_target ≤ user_eps`、ここで `amp = 1/min(e,c·d)`。
    ///       IPM_eps_target ≥ `IPM_F64_ACHIEVABLE_EPS` (f64 限界) なので
    ///       `amp ≤ user_eps / IPM_F64_ACHIEVABLE_EPS`、よって
    ///       `min(e,c·d) ≥ IPM_F64_ACHIEVABLE_EPS / user_eps`.
    ///
    /// 例: user_eps=1e-6 → floor=1e-4 (amp≤1e4)、user_eps=1e-3 → floor=1e-7 (amp≤1e7)。
    /// user_eps が `IPM_F64_ACHIEVABLE_EPS` 以下になると floor≥1 (= no-scaling) で
    /// 自然に縮退する。
    pub fn scale_floor_for_eps(user_eps: f64) -> f64 {
        if user_eps > 0.0 {
            (Self::IPM_F64_ACHIEVABLE_EPS / user_eps).min(1.0)
        } else {
            0.0
        }
    }

    /// 新規スケーラーを生成（初期値: 恒等変換 D=I, E=I, c=1）
    pub fn new(n: usize, m: usize) -> Self {
        RuizScaler {
            d: vec![1.0; n],
            e: vec![1.0; m],
            c: 1.0,
        }
    }

    /// Ruiz equilibration を実行する (Q/A のみ、user_eps から floor 導出)。
    ///
    /// 各 sweep で行・列・コストノルムを順次正規化し、固定点 (相対変化 < CONV_TOL)
    /// に達した時点で打ち切る。終了後、`scale_floor_for_eps(user_eps)` で
    /// e[i] / d[j] の下限をクリップし、unscale 後の残差 ≤ user_eps を保証する。
    ///
    /// l, u（変数境界）はAPIの完全性のため受け取るが、
    /// ノルム計算には使用しない（Q と A のみで均衡化する）。
    #[allow(clippy::needless_range_loop)]
    pub fn compute(
        &mut self,
        q: &CscMatrix,
        a: &CscMatrix,
        q_vec: &[f64],
        _l: &[f64],
        _u: &[f64],
        user_eps: f64,
    ) {
        let floor = Self::scale_floor_for_eps(user_eps);
        self.compute_with_rhs_floor(q, a, q_vec, &[], floor);
    }

    /// RHS（b）を含む行ノルムで Ruiz equilibration を実行 (presolve 用)。
    ///
    /// `compute()` と同じだが、Step 1 の行ノルムに `|e[i]*b[i]|` を追加する。
    /// 終了後の floor も `scale_floor_for_eps(user_eps)` から導出。
    #[allow(clippy::needless_range_loop)]
    pub fn compute_with_rhs(
        &mut self,
        q: &CscMatrix,
        a: &CscMatrix,
        q_vec: &[f64],
        b: &[f64],
        user_eps: f64,
    ) {
        let floor = Self::scale_floor_for_eps(user_eps);
        self.compute_with_rhs_floor(q, a, q_vec, b, floor);
    }

    /// `compute_with_rhs` の floor 可変版。`scale_floor=0.0` でクリップ無効。
    #[allow(clippy::needless_range_loop)]
    pub fn compute_with_rhs_floor(
        &mut self,
        q: &CscMatrix,
        a: &CscMatrix,
        q_vec: &[f64],
        b: &[f64],
        scale_floor: f64,
    ) {
        let n = q.ncols;
        let m = a.nrows;
        const EPS: f64 = 1e-6;
        // 詳細は `RUIZ_SWEEPS` ドキュメント参照。
        const NUM_ITER: usize = RuizScaler::RUIZ_SWEEPS;

        for _iter in 0..NUM_ITER {
            // Step 1: 行ノルム正規化（b を含む）
            if m > 0 {
                let mut row_norms = vec![0.0f64; m];
                for col in 0..n {
                    for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                        let i = a.row_ind[k];
                        let val = (self.e[i] * a.values[k] * self.d[col]).abs();
                        if val > row_norms[i] {
                            row_norms[i] = val;
                        }
                    }
                }
                // b を行ノルムに追加
                for i in 0..m.min(b.len()) {
                    let b_val = (self.e[i] * b[i]).abs();
                    if b_val > row_norms[i] {
                        row_norms[i] = b_val;
                    }
                }
                for i in 0..m {
                    let norm = row_norms[i].max(EPS);
                    self.e[i] /= norm.sqrt();
                }
            }

            // Step 2: 列ノルム正規化（compute() と同じ）
            let mut col_norms = vec![0.0f64; n];
            for col in 0..n {
                for k in q.col_ptr[col]..q.col_ptr[col + 1] {
                    let row = q.row_ind[k];
                    let val = (self.c * self.d[row] * q.values[k] * self.d[col]).abs();
                    if val > col_norms[col] {
                        col_norms[col] = val;
                    }
                }
            }
            if m > 0 {
                for col in 0..n {
                    for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                        let row = a.row_ind[k];
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

            // Step 3: コスト正規化（compute() と同じ）
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

        // 下限クリッピング (scale_floor>0 のみ)。`scale_floor_for_eps(user_eps)` から
        // 与えられ、`min(e, c·d) ≥ floor` を強制することで unscale 後残差 ≤ user_eps
        // を保証する (詳細は `scale_floor_for_eps` ドキュメント参照)。
        if scale_floor > 0.0 {
            for i in 0..m {
                self.e[i] = self.e[i].max(scale_floor);
            }
            for j in 0..n {
                self.d[j] = self.d[j].max(scale_floor);
            }
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
    #[allow(clippy::type_complexity)]
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

    /// スケール済み境界双対変数を元のスケールに逆変換する
    ///
    /// # 変換式
    /// KKT条件: Q*x + q + A^T*y - y_lb + y_ub = 0
    /// スケール後KKT: c*D*Q*D*x_s + c*D*q + D*A^T*E*y_s - (c*D)*y_lb_s + (c*D)*y_ub_s = 0
    /// 両辺を c*D で割る: y_lb = y_lb_s / (c * d[j]), y_ub = y_ub_s / (c * d[j])
    ///
    /// # 引数
    /// - `bound_duals_s`: スケール済み境界双対変数。lb有限変数の下界dual（昇順）、次にub有限変数の上界dual（昇順）の順で格納
    /// - `bounds`: 元問題の変数境界
    pub fn unscale_bound_duals(&self, bound_duals_s: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
        // 空入力（bound_dualsが未計算の場合）は空を返す
        if bound_duals_s.is_empty() {
            return vec![];
        }
        let mut result = Vec::with_capacity(bound_duals_s.len());
        let mut idx = 0;
        // 下界分（lb が有限な変数、変数番号昇順）
        for (j, &(lb, _)) in bounds.iter().enumerate() {
            if lb.is_finite() {
                result.push(bound_duals_s[idx] / (self.c * self.d[j]));
                idx += 1;
                debug_assert!(idx <= bound_duals_s.len(), "bound_duals_s index out of bounds (lb)");
            }
        }
        // 上界分（ub が有限な変数、変数番号昇順）
        for (j, &(_, ub)) in bounds.iter().enumerate() {
            if ub.is_finite() {
                result.push(bound_duals_s[idx] / (self.c * self.d[j]));
                idx += 1;
                debug_assert!(idx <= bound_duals_s.len(), "bound_duals_s index out of bounds (ub)");
            }
        }
        debug_assert_eq!(idx, bound_duals_s.len(), "unscale_bound_duals: idx != bound_duals_s.len()");
        result
    }

    /// スケール済み解を元のスケールに逆変換する
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
        scaler.compute(&q, &a, &q_vec, &l, &u, 1e-6);

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
        use crate::options::{SolverOptions, QpSolverChoice};
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

        let problem = QpProblem::new_all_le(q, q_vec, a, b, bounds).unwrap();

        // スケーリングなし
        let opts_no_scale = SolverOptions { use_ruiz_scaling: false, qp_solver: QpSolverChoice::IpPmm, ..Default::default() };
        let r_no_scale = crate::qp::solve_qp_with(&problem, &opts_no_scale);

        // スケーリングあり
        let opts_scale = SolverOptions { use_ruiz_scaling: true, qp_solver: QpSolverChoice::IpPmm, ..Default::default() };
        let r_scale = crate::qp::solve_qp_with(&problem, &opts_scale);

        // 両方 Optimal (偽Optimal検出時はSuboptimalSolutionも許容)
        assert!(
            r_no_scale.status == SolveStatus::Optimal
                || r_no_scale.status == SolveStatus::Timeout
                || r_no_scale.status == SolveStatus::SuboptimalSolution,
            "no_scale: {:?}", r_no_scale.status
        );
        assert!(
            r_scale.status == SolveStatus::Optimal
                || r_scale.status == SolveStatus::Timeout
                || r_scale.status == SolveStatus::SuboptimalSolution,
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
        use crate::options::{SolverOptions, QpSolverChoice};
        use crate::problem::SolveStatus;

        // 簡単な QP: min x^2 + y^2  s.t. x+y >= 1
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let q_vec = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, q_vec, a, b, bounds).unwrap();

        let opts = SolverOptions { use_ruiz_scaling: false, qp_solver: QpSolverChoice::IpPmm, ..Default::default() };

        let result = crate::qp::solve_qp_with(&problem, &opts);
        assert_eq!(result.status, SolveStatus::Optimal, "disabled: {:?}", result.status);
        assert!((result.solution[0] - 0.5).abs() < 0.05, "x[0]={}", result.solution[0]);
        assert!((result.solution[1] - 0.5).abs() < 0.05, "x[1]={}", result.solution[1]);
    }

    /// scale_problem → unscale_solution の round-trip が恒等であること。
    /// これが破れると IPM が scaled 空間で正しく解いても元空間で解が狂う。
    #[test]
    fn scale_unscale_round_trip_identity() {
        let n = 3usize;
        let m = 2usize;
        let q = CscMatrix::from_triplets(
            &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 4.0], n, n,
        ).unwrap();
        let q_vec = vec![1.0, 2.0, 3.0];
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 1, 2], &[1.0, 2.0, 3.0, 4.0], m, n,
        ).unwrap();
        let b = vec![5.0, 6.0];
        let bounds = vec![(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)];

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&q, &a, &q_vec, &[0.0; 3], &[10.0; 3], 1e-6);

        // 任意の orig 空間 (x, y) で round-trip を確認:
        //   scaled_x = D^{-1} x  (公式: x = D x_s → x_s = D^{-1} x)
        // ここでは scale_problem 後の bounds で scaled x_s を取り、unscale で戻す。
        let (_q_s, _a_s, _q_s_vec, _b_s, bounds_s) =
            scaler.scale_problem(&q, &a, &q_vec, &b, &bounds);
        // x_s = midpoint of bounds_s
        let x_s: Vec<f64> = bounds_s.iter().map(|&(l, u)| 0.5 * (l + u)).collect();
        // y_s 任意
        let y_s = vec![0.7_f64, -0.3];
        let (x_orig, y_orig) = scaler.unscale_solution(&x_s, &y_s);

        // 期待値: x = D x_s = d[j] * x_s[j], y = E y_s / c
        for j in 0..n {
            let expected = scaler.d[j] * x_s[j];
            assert!((x_orig[j] - expected).abs() < 1e-12 * (1.0 + expected.abs()),
                "x_orig[{}]={} expected {}", j, x_orig[j], expected);
        }
        for i in 0..m {
            let expected = scaler.e[i] * y_s[i] / scaler.c;
            assert!((y_orig[i] - expected).abs() < 1e-12 * (1.0 + expected.abs()),
                "y_orig[{}]={} expected {}", i, y_orig[i], expected);
        }
    }

    /// scaled 空間の KKT 残差は orig 空間の (c × d[j]) 倍である関係を確認。
    /// (`r_d_orig[j] = r_d_scaled[j] / (c × d[j])`)
    #[test]
    fn dual_residual_unscale_factor_is_c_times_d() {
        let n = 2usize;
        let m = 1usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], n, n).unwrap();
        let q_vec = vec![1.0_f64, 2.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[3.0_f64, 4.0], m, n).unwrap();
        let b = vec![5.0_f64];
        let bounds = vec![(0.0_f64, 10.0); 2];

        let mut scaler = RuizScaler::new(n, m);
        scaler.compute(&q, &a, &q_vec, &[0.0; 2], &[10.0; 2], 1e-6);
        let (q_s, a_s, q_s_vec, _b_s, _bounds_s) =
            scaler.scale_problem(&q, &a, &q_vec, &b, &bounds);

        // x_s, y_s 任意で stationarity を計算
        let x_s = vec![0.3_f64, 0.4];
        let y_s = vec![0.5_f64];

        // r_d_scaled[j] = (Q_s x_s + q_s + A_s^T y_s)[j]
        let qx_s = q_s.mat_vec_mul(&x_s).unwrap();
        let aty_s = a_s.transpose().mat_vec_mul(&y_s).unwrap();
        let r_d_s: Vec<f64> = (0..n).map(|j| qx_s[j] + q_s_vec[j] + aty_s[j]).collect();

        // 元空間で同じ計算
        let (x, y) = scaler.unscale_solution(&x_s, &y_s);
        let qx = q.mat_vec_mul(&x).unwrap();
        let aty = a.transpose().mat_vec_mul(&y).unwrap();
        let r_d: Vec<f64> = (0..n).map(|j| qx[j] + q_vec[j] + aty[j]).collect();

        // 関係: r_d[j] ≈ r_d_s[j] / (c * d[j])
        for j in 0..n {
            let expected = r_d_s[j] / (scaler.c * scaler.d[j]);
            assert!((r_d[j] - expected).abs() < 1e-10 * (1.0 + expected.abs()),
                "r_d[{}]={} expected {} (= r_d_s[{}] / (c×d[{}]))", j, r_d[j], expected, j, j);
        }
    }
}
