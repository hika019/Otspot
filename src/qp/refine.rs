//! Iterative Refinement for QP solutions (cmd_330)
//!
//! E2: Primal-only refinement — post-postsolve検証でSuboptimalSolutionと判定された解に対して
//! 原問題空間で正規方程式 A^T*A*δx = -A^T*r_p を解き、primal feasibilityを改善する。
//!
//! cmd_332 C3対応: 疎A^T*A構築によりnサイズ制限を撤廃。

use crate::linalg::ldl;
use crate::sparse::CscMatrix;
use super::problem::QpProblem;

/// Primal-only iterative refinement (E2)
///
/// SuboptimalSolution判定後に呼び出す。A^T*A*δx = -A^T*r_p を解いてxを補正する。
/// pfeas < eps*(1+||b||_inf) を満たせば `true` を返す。それ以外は `false`。
///
/// # 引数
/// - `problem`: 元問題（原問題空間、postsolve済み）
/// - `x`: 補正対象のprimal解（in/out）
/// - `_y`: dual解（E2では不使用、将来拡張用）
/// - `_z`: bound dual（E2では不使用、将来拡張用）
/// - `max_steps`: 最大refinementステップ数
/// - `eps`: 収束判定閾値（solve_qp_withのipm_eps()を渡す）
pub fn iterative_refine(
    problem: &QpProblem,
    x: &mut [f64],
    _y: &mut [f64],
    _z: &mut [f64],
    max_steps: usize,
    eps: f64,
) -> bool {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    // 制約なしの問題はスキップ（正規方程式が定義されない）
    if m == 0 {
        return false;
    }

    let norm_b = problem.b.iter().fold(0.0_f64, |a, &bi| a.max(bi.abs())).max(1.0);

    // 疎A^T*A (n×n 上三角CSC) を事前構築（refinementループ内で再利用）
    // C1正則化εI付き（正定値性を保証）
    let ata_csc = match build_ata_upper_csc(&problem.a, n, m) {
        Some(m) => m,
        None => return false,
    };

    // A^T*A を LDL^T 因子化（factorize は正定値行列を要求）
    let factor = match ldl::factorize(&ata_csc) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let mut prev_pfeas = f64::INFINITY;
    for _ in 0..max_steps {
        // Step 1: r_p = A*x - b
        let ax = match problem.a.mat_vec_mul(x) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let r_p: Vec<f64> = ax.iter().zip(problem.b.iter())
            .map(|(&ax_i, &b_i)| ax_i - b_i)
            .collect();

        // Step 2: pfeas収束チェック（Ge対応: violation = max(b-ax, 0) = max(-r_p, 0)）
        let pfeas = r_p.iter().zip(problem.constraint_types.iter())
            .map(|(&r, ct)| match ct {
                crate::problem::ConstraintType::Eq => r.abs(),
                crate::problem::ConstraintType::Ge => (-r).max(0.0),
                _ => r.max(0.0),
            })
            .fold(0.0_f64, f64::max);
        if pfeas < eps * (1.0 + norm_b) {
            return true;
        }

        // 対策D: 発散防止 — pfeasが10%超増加したら打ち切り [cmd_337]
        if pfeas > prev_pfeas * 1.1 {
            break;
        }
        prev_pfeas = pfeas;

        // Step 3: rhs = -A^T * r_p
        // (A^T * r_p)[i] = sum_k A[k][i] * r_p[k]
        // A の CSC形式で列iのエントリを使って計算
        let rhs: Vec<f64> = (0..n).map(|col| {
            let val: f64 = (problem.a.col_ptr[col]..problem.a.col_ptr[col + 1])
                .map(|ptr| problem.a.values[ptr] * r_p[problem.a.row_ind[ptr]])
                .sum();
            -val
        }).collect();

        // Step 4: A^T*A * δx = rhs を解く
        let mut dx = vec![0.0f64; n];
        factor.solve(&rhs, &mut dx);

        // Step 5: x += δx、boundsをclamp
        for i in 0..n {
            x[i] += dx[i];
            if let Some(&(lb, ub)) = problem.bounds.get(i) {
                if lb.is_finite() {
                    x[i] = x[i].max(lb);
                }
                if ub.is_finite() {
                    x[i] = x[i].min(ub);
                }
            }
        }
    }

    // 最終収束チェック（max_steps到達後、Ge対応）
    if let Ok(ax) = problem.a.mat_vec_mul(x) {
        let pfeas = ax.iter().zip(problem.b.iter()).zip(problem.constraint_types.iter())
            .map(|((&ax_i, &b_i), ct)| match ct {
                crate::problem::ConstraintType::Eq => (ax_i - b_i).abs(),
                crate::problem::ConstraintType::Ge => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            })
            .fold(0.0_f64, f64::max);
        if pfeas < eps * (1.0 + norm_b) {
            return true;
        }
    }

    false
}

/// A^T*A の上三角 CSC 行列を疎形式で構築する（C1正則化εI付き）
///
/// A は m×n の CSC 行列（行インデックスは列内でソート済み）。
/// A^T*A[j][k] = sum_i A[i][j]*A[i][k] を疎パターンで計算し、上三角CSCで返す。
///
/// # C1: 明示的対角正則化
/// ε = 1e-12 * max(対角最大値, 1.0) を全対角成分に加算する。
/// A^T*Aがrank-deficient（冗長制約等）でも安定したLDL因子化を保証し、
/// faerの暗黙的正則化への依存を排除する。
///
/// # C3: MAX_N_FOR_REFINE撤廃
/// 旧実装は密行列A^T*Aを構築していたため、n>300をスキップしていた。
/// 本実装は疎パターンのみ保持するため、nサイズ制限がない。
/// 計算量: O(nnz(A)^2 / m + nnz(A^T*A))（疎行列では密行列より大幅に削減）。
///
/// 失敗時（ゼロ行列等）は None を返す。
fn build_ata_upper_csc(a: &CscMatrix, n: usize, m: usize) -> Option<CscMatrix> {
    use std::collections::BTreeSet;

    // row_to_cols[i] = 行iで非ゼロの列インデックスのリスト（ソート済み）
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
    for col in 0..n {
        for ptr in a.col_ptr[col]..a.col_ptr[col + 1] {
            let row = a.row_ind[ptr];
            if row < m {
                row_to_cols[row].push(col);
            }
        }
    }

    // シンボリックフェーズ: A^T*Aの非ゼロパターン（上三角）を収集
    // pattern_set[k] = A^T*Aの列kにおける行インデックス集合（j <= k）
    // 対角成分は常に含める（C1正則化でεを加算するため）
    let mut pattern_set: Vec<BTreeSet<usize>> = (0..n).map(|k| {
        let mut s = BTreeSet::new();
        s.insert(k);
        s
    }).collect();

    for cols in &row_to_cols {
        for &j in cols.iter() {
            for &k in cols.iter() {
                if j <= k {
                    pattern_set[k].insert(j);
                }
            }
        }
    }

    // 数値フェーズ: ソート済み行インデックスのマージジョインでdot積を計算
    let mut col_ptr = vec![0usize; n + 1];
    let mut row_ind_out: Vec<usize> = Vec::new();
    let mut values_out: Vec<f64> = Vec::new();
    let mut diag_positions: Vec<usize> = vec![0; n]; // 各列kの対角成分のvalues_out内位置

    for k in 0..n {
        col_ptr[k] = row_ind_out.len();
        for &j in &pattern_set[k] {
            // A^T*A[j,k] = Σ_i A[i,j] * A[i,k]
            // 列j,kのソート済み行インデックスをマージジョインで計算（O(nnz_j + nnz_k)）
            let (j_start, j_end) = (a.col_ptr[j], a.col_ptr[j + 1]);
            let (k_start, k_end) = (a.col_ptr[k], a.col_ptr[k + 1]);

            let mut dot = 0.0;
            let mut pj = j_start;
            let mut pk = k_start;
            while pj < j_end && pk < k_end {
                match a.row_ind[pj].cmp(&a.row_ind[pk]) {
                    std::cmp::Ordering::Less => pj += 1,
                    std::cmp::Ordering::Greater => pk += 1,
                    std::cmp::Ordering::Equal => {
                        dot += a.values[pj] * a.values[pk];
                        pj += 1;
                        pk += 1;
                    }
                }
            }

            if j == k {
                diag_positions[k] = values_out.len();
            }
            row_ind_out.push(j);
            values_out.push(dot);
        }
    }
    col_ptr[n] = row_ind_out.len();

    // 対角成分を常に挿入するため values_out は必ず n 要素以上存在する
    if values_out.is_empty() {
        return None;
    }

    // C1: 明示的対角正則化 εI を追加
    // ε = 1e-12 * max(対角最大値の絶対値, 1.0)
    let diag_max = diag_positions.iter()
        .map(|&p| values_out[p].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let eps_reg = 1e-12 * diag_max;
    for &p in &diag_positions {
        values_out[p] += eps_reg;
    }

    Some(CscMatrix { col_ptr, row_ind: row_ind_out, values: values_out, nrows: n, ncols: n })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    /// IPM-T14: 半正定値A^T*A（rank-deficient制約）でrefinementが収束することを確認
    ///
    /// 問題: min x^2 + y^2  s.t. x + y >= 1.0 (A=[-1,-1], b=[-1.0])
    /// A^T*A = [[1,1],[1,1]] は半正定値（rank=1）。
    /// C1の対角正則化εIにより確実にPDとなりLDL因子化が安定する。
    /// 初期解: x=[0.3, 0.3] → pfeas = 0.4 > eps → 補正後収束
    #[test]
    fn test_ipm_t14_suboptimal_converges() {
        // min x^2 + y^2  s.t. x + y >= 1  (n=2, m=1)
        // Ax <= b 形式: -x - y <= -1 → A = [[-1,-1]], b = [-1]
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 初期解 x=[0.3, 0.3]: pfeas = 0.4 > eps
        let mut x = vec![0.3, 0.3];
        let mut y = vec![0.0];
        let mut z = vec![0.0, 0.0];
        let result = iterative_refine(&problem, &mut x, &mut y, &mut z, 3, 1e-6);
        // C1正則化により確実にresult=true（rank-deficientでも収束する）
        assert!(result, "IPM-T14: C1対角正則化後、rank-deficient A^T*Aでもrefinementは収束すべき");
        let norm_b = 1.0_f64.max(1.0);
        let pfeas = (-x[0] - x[1] - (-1.0_f64)).max(0.0);
        assert!(pfeas < 1e-6 * (1.0 + norm_b),
            "IPM-T14: 収束後pfeas={pfeas:.2e}は閾値以下であるべき");
    }

    /// IPM-T14b: フル列ランクのAに対してrefinementが機能することを確認
    ///
    /// 問題: min x^2  s.t. x >= 0.5 (A=[-1], b=[-0.5])
    /// 初期解 x=[0.0] → pfeas = -0.0 - (-0.5) = 0.5 > eps
    /// 補正後: A^T*A = [1], rhs = -A^T*r_p = -(-1)*0.5 = 0.5
    /// δx = 0.5, x_new = 0.5 → pfeas = 0 → 収束
    #[test]
    fn test_ipm_t14b_refinement_converges() {
        // min x^2  s.t. x >= 0.5 → -x <= -0.5
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let c = vec![0.0];
        let a = CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 1).unwrap();
        let b = vec![-0.5];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 初期解 x=0.0: pfeas = (-1*0.0) - (-0.5) = 0.5 > eps
        let mut x = vec![0.0];
        let mut y = vec![0.0];
        let mut z = vec![0.0];
        let result = iterative_refine(&problem, &mut x, &mut y, &mut z, 3, 1e-6);

        assert!(result, "IPM-T14b: 1次元問題でrefinementは収束すべき");
        // x ≈ 0.5 になるはず
        let pfeas = (-x[0] - (-0.5_f64)).max(0.0);
        assert!(pfeas < 1e-6 * (1.0 + 0.5), "IPM-T14b: pfeasが収束閾値以下になるべき: pfeas={pfeas:.2e}");
    }

    /// IPM-T14c: MAX_N_FOR_REFINE撤廃後、n>300でも疎A^T*Aにより動作することを確認
    ///
    /// 旧実装では n=301 > MAX_N_FOR_REFINE=300 でスキップ（false）していたが、
    /// C3疎A^T*A導入後は制約があればrefinementが実行される。
    #[test]
    fn test_ipm_t14c_large_n_works() {
        let n = 301;
        let q = CscMatrix::from_triplets(
            &(0..n).collect::<Vec<_>>(),
            &(0..n).collect::<Vec<_>>(),
            &vec![2.0; n],
            n, n,
        ).unwrap();
        let c = vec![0.0; n];
        // 制約: x[0] >= 0.5 → A=[-e_0], b=[-0.5]
        let a = CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, n).unwrap();
        let b = vec![-0.5];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

        // 初期解 x[0]=0.0: pfeas = (-1*0.0) - (-0.5) = 0.5 > eps
        let mut x = vec![0.0; n];
        let mut y = vec![0.0];
        let mut z = vec![0.0; n];
        // MAX_N_FOR_REFINE撤廃後はn=301でも実行され収束するはず
        let result = iterative_refine(&problem, &mut x, &mut y, &mut z, 3, 1e-6);
        assert!(result, "IPM-T14c: MAX_N_FOR_REFINE撤廃後、n=301でも疎A^T*Aによりrefinementは動作すべき");
    }
}
