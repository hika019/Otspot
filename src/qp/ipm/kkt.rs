//! IPM KKT 行列構築ユーティリティ
//!
//! - 拡張制約行列構築 (`build_extended_constraints`)
//! - augmented KKT system 構築 (`build_augmented_system`)
//! - 疎行列-ベクトル演算ヘルパー

use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// 疎行列-ベクトル演算
// ---------------------------------------------------------------------------

/// out = A * x（上書き）
#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmv(a: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..a.ncols {
        let xv = x[col];
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            out[a.row_ind[k]] += a.values[k] * xv;
        }
    }
}

/// out = A^T * v（上書き）
#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmtv(a: &CscMatrix, v: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|o| *o = 0.0);
    for col in 0..a.ncols {
        let mut s = 0.0;
        for k in a.col_ptr[col]..a.col_ptr[col + 1] {
            s += a.values[k] * v[a.row_ind[k]];
        }
        out[col] = s;
    }
}

/// out = Q * x（全要素格納の対称 Q に対応）
#[inline]
#[allow(clippy::needless_range_loop)]
pub(crate) fn spmv_q(q: &CscMatrix, x: &[f64], out: &mut [f64]) {
    out.iter_mut().for_each(|v| *v = 0.0);
    for col in 0..q.ncols {
        let xv = x[col];
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            out[q.row_ind[k]] += q.values[k] * xv;
        }
    }
}

/// ||v||_∞
#[inline]
pub(crate) fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

// ---------------------------------------------------------------------------
// 拡張制約行列構築
// ---------------------------------------------------------------------------

/// Ax <= b + lb/ub 境界を含む拡張制約を構築する
///
/// 戻り値: (A_ext, b_ext, m_ext, m_orig, n_lb)
/// 順序: [original inequalities | lower bound rows | upper bound rows]
pub(crate) fn build_extended_constraints(
    problem: &QpProblem,
) -> (CscMatrix, Vec<f64>, usize, usize, usize) {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    let n_lb: usize = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub: usize = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    let m_ext = m + n_lb + n_ub;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut b_ext = Vec::with_capacity(m_ext);

    // 元の不等式制約 A x <= b
    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            rows.push(problem.a.row_ind[k]);
            cols.push(col);
            vals.push(problem.a.values[k]);
        }
    }
    b_ext.extend_from_slice(&problem.b);

    // 下界制約: x_j >= lb_j → -x_j <= -lb_j
    let mut lb_row = m;
    for (j, &(lb, _)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() {
            rows.push(lb_row);
            cols.push(j);
            vals.push(-1.0);
            b_ext.push(-lb);
            lb_row += 1;
        }
    }

    // 上界制約: x_j <= ub_j
    let mut ub_row = m + n_lb;
    for (j, &(_, ub)) in problem.bounds.iter().enumerate() {
        if ub.is_finite() {
            rows.push(ub_row);
            cols.push(j);
            vals.push(1.0);
            b_ext.push(ub);
            ub_row += 1;
        }
    }

    let a_ext = if m_ext == 0 || rows.is_empty() {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, m_ext, n).unwrap()
    };

    (a_ext, b_ext, m_ext, m, n_lb)
}

// ---------------------------------------------------------------------------
// augmented KKT system 構築
// ---------------------------------------------------------------------------

/// augmented KKT system の上三角 CSC を構築する
///
/// ```text
/// [ Q + δ_p·I     A_ext^T        ] [dx]   [rx]
/// [ A_ext      -(Σ + δ_d·I)      ] [dy] = [ry]
/// ```
///
/// サイズ: (n + m_ext) × (n + m_ext)。上三角のみ CSC として返す。
///
/// # 利点（Schur complement との比較）
/// - 条件数 κ ≈ κ(A)（Schur complement の κ(A)^2 ではない）
/// - IP-PMM 正則化で quasidefinite 保証 → LDLT 常に成功
/// - スパーシティ保存（A^T D^{-1} A で fill-in しない）
#[allow(clippy::needless_range_loop)]
pub(crate) fn build_augmented_system(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    sigma_vec: &[f64],
    delta_p: f64,
    delta_d: f64,
) -> CscMatrix {
    let n = q.nrows;
    let m_ext = a_ext.nrows;
    let total = n + m_ext;

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Part 1: Q + δ_p·I (上三角のみ)
    let mut diag_added = vec![false; n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                let v = q.values[k] + if row == col { delta_p } else { 0.0 };
                rows.push(row);
                cols.push(col);
                vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    for i in 0..n {
        if !diag_added[i] {
            rows.push(i);
            cols.push(i);
            vals.push(delta_p);
        }
    }

    // Part 2: A_ext^T ブロック（右上、row < col 保証）
    for j in 0..n {
        for idx in a_ext.col_ptr[j]..a_ext.col_ptr[j + 1] {
            let k = a_ext.row_ind[idx];
            let v = a_ext.values[idx];
            rows.push(j);
            cols.push(n + k);
            vals.push(v);
        }
    }

    // Part 3: -(Σ + δ_d)·I 対角ブロック（インデックス n..n+m_ext）
    for k in 0..m_ext {
        rows.push(n + k);
        cols.push(n + k);
        vals.push(-(sigma_vec[k] + delta_d));
    }

    if rows.is_empty() {
        CscMatrix::new(total, total)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, total, total).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Schur complement 構築
// ---------------------------------------------------------------------------

/// M = Q + δ_p·I + A_ext^T D^{-1} A_ext の上三角 CSC を構築する
///
/// M は正定値（IP-PMM 正則化により保証）なので
/// `ldl::factorize_with_deadline` で分解できる。
///
/// # 注意
/// 密行列（n×n）で蓄積するため n が大きい場合メモリが O(n²) 必要。
/// LDL_THRESHOLD (20_000) でゲートされており大問題には使われない。
#[allow(clippy::needless_range_loop)]
pub(crate) fn build_schur_complement(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    d_inv: &[f64],
    delta_p: f64,
    cancel: &AtomicBool,
) -> Option<CscMatrix> {
    let n = q.nrows;
    let m_ext = a_ext.nrows;

    // 密行列で蓄積（n が小さい問題用）
    let mut m_dense = vec![0.0f64; n * n];

    // Q を加算（全要素格納 → 対称）
    for col in 0..n {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            m_dense[row * n + col] += q.values[k];
            if row != col {
                m_dense[col * n + row] += q.values[k];
            }
        }
    }

    // δ_p·I を加算
    for i in 0..n {
        m_dense[i * n + i] += delta_p;
    }

    // A_ext^T D^{-1} A_ext を加算
    // 行 i のエントリを事前構築
    let mut row_data: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m_ext];
    for col in 0..n {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            let row = a_ext.row_ind[k];
            row_data[row].push((col, a_ext.values[k]));
        }
    }

    for i in 0..m_ext {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let d = d_inv[i];
        let row_i = &row_data[i];
        for &(p, vp) in row_i {
            for &(q_col, vq) in row_i {
                m_dense[p * n + q_col] += d * vp * vq;
            }
        }
    }

    // 上三角のみ triplet として抽出
    let mut out_rows = Vec::new();
    let mut out_cols = Vec::new();
    let mut out_vals = Vec::new();
    for p in 0..n {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        for q in p..n {
            let v = m_dense[p * n + q];
            if v != 0.0 {
                out_rows.push(p);
                out_cols.push(q);
                out_vals.push(v);
            }
        }
    }

    if out_rows.is_empty() {
        // Q=0, A=0 のエッジケース: δ_p I
        let diag_rows: Vec<usize> = (0..n).collect();
        let diag_cols: Vec<usize> = (0..n).collect();
        let diag_vals = vec![delta_p; n];
        Some(CscMatrix::from_triplets(&diag_rows, &diag_cols, &diag_vals, n, n).unwrap())
    } else {
        Some(CscMatrix::from_triplets(&out_rows, &out_cols, &out_vals, n, n).unwrap())
    }
}

