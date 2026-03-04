//! IPM KKT 行列構築ユーティリティ
//!
//! - 拡張制約行列構築 (`build_extended_constraints`)
//! - augmented KKT system 構築 (`build_augmented_system`)
//! - 疎行列-ベクトル演算ヘルパー
//! - CGパス用 matrix-free 演算

use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "parallel")]
use crate::linalg::ldl::LdlFactorizationAmd;
#[cfg(feature = "parallel")]
use std::time::Instant;

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
/// LDL_THRESHOLD (5000) でゲートされており大問題には使われない。
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

// ---------------------------------------------------------------------------
// CGパス用ヘルパー
// ---------------------------------------------------------------------------

/// M·v を計算する（matrix-free）
///
/// M = Q + δ_p I + A_ext^T D^{-1} A_ext
pub(crate) fn mv_ipm_apply(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    d_inv: &[f64],
    delta_p: f64,
    v: &[f64],
    out: &mut [f64],
) {
    let n = v.len();
    let m_ext = d_inv.len();

    spmv_q(q, v, out);
    for i in 0..n {
        out[i] += delta_p * v[i];
    }

    if m_ext == 0 {
        return;
    }

    let mut av = vec![0.0f64; m_ext];
    spmv(a_ext, v, &mut av);
    for i in 0..m_ext {
        av[i] *= d_inv[i];
    }

    let mut at_av = vec![0.0f64; n];
    spmtv(a_ext, &av, &mut at_av);
    for i in 0..n {
        out[i] += at_av[i];
    }
}

/// Jacobi（対角）前処理ベクトルを計算する
///
/// m_inv[j] = 1 / diag(M)[j]
#[allow(clippy::needless_range_loop)]
pub(crate) fn compute_jacobi_precond_ipm(
    q: &CscMatrix,
    a_ext: &CscMatrix,
    d_inv: &[f64],
    delta_p: f64,
) -> Vec<f64> {
    let n = q.nrows;
    let mut diag = vec![delta_p; n];

    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col {
                diag[col] += q.values[k];
            }
        }
    }

    for col in 0..a_ext.ncols {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            let row = a_ext.row_ind[k];
            let v = a_ext.values[k];
            diag[col] += d_inv[row] * v * v;
        }
    }

    diag.iter()
        .map(|&d| if d.abs() < super::JACOBI_MIN_DIAG { 1.0 } else { 1.0 / d })
        .collect()
}

// ---------------------------------------------------------------------------
// 制約前処理 Schur 補元構築
// ---------------------------------------------------------------------------

/// 制約前処理の Schur 補元 S_orig = D_orig + A_orig (Q+δI)^{-1} A_orig^T を構築する
///
/// m_orig は元の問題の制約数（境界制約を除く）。拡張制約行列 a_ext の先頭 m_orig 行のみを
/// 用いて m_orig×m_orig の Schur 補元を構築する。境界制約行は Jacobi 前処理で扱う。
///
/// # 引数
/// - `q_fac`: Q + δ_p·I の LDL+AMD 分解（`factorize_with_amd` で事前構築）
/// - `a_ext`: 拡張制約行列（m_ext×n CSC）
/// - `d_vec`: D = Σ + δ_d·I の対角（m_ext 要素）
/// - `m_orig`: 元の問題制約数（<= m_ext）
/// - `cancel`: キャンセルフラグ（10 列ごとにチェック）
/// - `deadline`: タイムアウト期限（10 列ごとにチェック）
///
/// # 返り値
/// S_orig の上三角 CSC 行列。m_orig > 5000 の場合、またはタイムアウト/キャンセル時は `None`。
#[cfg(feature = "parallel")]
#[allow(clippy::needless_range_loop)]
pub(crate) fn build_constraint_schur(
    q_fac: &LdlFactorizationAmd,
    a_ext: &CscMatrix,
    d_vec: &[f64],
    m_orig: usize,
    cancel: &AtomicBool,
    deadline: Option<Instant>,
) -> Option<CscMatrix> {
    let m_ext = a_ext.nrows;
    let n = a_ext.ncols;

    // m_orig > 5000: 密行列 m_orig² が大きすぎるため CG パスに委譲
    if m_orig > 5000 {
        return None;
    }

    if m_orig == 0 {
        return None;
    }

    // CSC 列インデックスから行インデックスに変換（A の行ごとの非零エントリリスト）
    // 先頭 m_orig 行のみ収集（境界制約行は不要）
    let mut row_data: Vec<Vec<(usize, f64)>> = vec![vec![]; m_orig];
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            let row = a_ext.row_ind[k];
            if row < m_orig {
                row_data[row].push((col, a_ext.values[k]));
            }
        }
    }

    // 密行列 m_orig×m_orig（row-major: s_dense[i*m_orig + j] = S[i,j]）
    let mut s_dense = vec![0.0f64; m_orig * m_orig];

    // D_orig の対角成分を初期値として設定
    for i in 0..m_orig {
        s_dense[i * m_orig + i] = d_vec[i];
    }

    // 作業バッファ（各反復で再利用）
    let mut a_j = vec![0.0f64; n];
    let mut z_j = vec![0.0f64; n];
    // col_j はフルサイズ (m_ext) で計算し、先頭 m_orig 分のみ使用
    let mut col_j = vec![0.0f64; m_ext];

    for j in 0..m_orig {
        // 10 列ごとにキャンセル/デッドラインチェック
        if j % 10 == 0 {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return None;
            }
        }

        // 行 j を密ベクトルに展開（a_j = A_orig の j 行目, R^n）
        a_j.iter_mut().for_each(|v| *v = 0.0);
        for &(col, val) in &row_data[j] {
            a_j[col] = val;
        }

        // z_j = (Q+δI)^{-1} * a_j（LDL solve）
        q_fac.solve(&a_j, &mut z_j);

        // col_j = A_ext * z_j（全行計算、先頭 m_orig 行のみ使用）
        spmv(a_ext, &z_j, &mut col_j);

        // S の j 列に累積（先頭 m_orig 行のみ）
        for i in 0..m_orig {
            s_dense[i * m_orig + j] += col_j[i];
        }
    }

    // 上三角部分を triplet 形式で抽出
    let mut out_rows = Vec::new();
    let mut out_cols = Vec::new();
    let mut out_vals = Vec::new();
    for j in 0..m_orig {
        for i in 0..=j {
            let v = s_dense[i * m_orig + j];
            if v != 0.0 {
                out_rows.push(i);
                out_cols.push(j);
                out_vals.push(v);
            }
        }
    }

    if out_rows.is_empty() {
        // エッジケース: D のみの対角行列
        let diag_rows: Vec<usize> = (0..m_orig).collect();
        let diag_cols: Vec<usize> = (0..m_orig).collect();
        let diag_vals: Vec<f64> = d_vec[..m_orig].to_vec();
        Some(CscMatrix::from_triplets(&diag_rows, &diag_cols, &diag_vals, m_orig, m_orig).unwrap())
    } else {
        Some(CscMatrix::from_triplets(&out_rows, &out_cols, &out_vals, m_orig, m_orig).unwrap())
    }
}
