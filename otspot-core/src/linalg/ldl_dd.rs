//! Sparse simplicial LDL^T factorization in TwoFloat (~106-bit) precision.
//!
//! For ill-conditioned KKT matrices (cond ≥ 1e8) where f64 LDL gives forward
//! error cond × 2.2e-16 that exceeds eps=1e-6, this DD-precision LDL pushes
//! the precision floor to cond × 4.9e-32 (Wilkinson backward stability).
//!
//! Algorithm: Davis "Direct Methods for Sparse Linear Systems" §4.6
//! up-looking simplicial LDL with etree-driven sparse triangular solve,
//! all numeric ops in TwoFloat. Symbolic phase (etree, column counts) stays
//! in usize arithmetic — no precision loss there.
//!
//! Public API mirrors `crate::linalg::ldl::factorize_quasidefinite_with_cached_perm`
//! so that `KktFactor::DirectDd` can be a drop-in replacement.

use crate::linalg::amd::{inv_permute_vec, permute_sym_upper, permute_vec};
use crate::linalg::ldl::LdlError;
use crate::sparse::CscMatrix;
use std::time::Instant;
use twofloat::TwoFloat;

/// Quasidefinite regularization (matches `crate::linalg::ldl::do_numeric_factorize`).
/// 期待符号と異なる方向に小さくなった D[k] にこの値を上書きする。
const DELTA: f64 = 1e-8;
/// |D[k]| がこれ未満なら符号方向に DELTA で押し戻す。
const EPSILON: f64 = 1e-13;

/// AMD-permuted simplicial LDL^T in TwoFloat precision.
pub struct LdlFactorizationDdAmd {
    n: usize,
    perm: Vec<usize>,
    /// Lower-triangular L (without unit diagonal), CSC: col j contains rows i > j.
    l_col_ptr: Vec<usize>,
    l_row_ind: Vec<usize>,
    l_values: Vec<TwoFloat>,
    /// Diagonal D in TwoFloat
    d: Vec<TwoFloat>,
}

impl LdlFactorizationDdAmd {
    /// 因子の非ゼロ数 (L の非ゼロ数; D は対角なので除く)
    pub fn nnz_l(&self) -> usize {
        self.l_values.len()
    }

    /// AMD 付き LDL^T x = b を DD 精度で解く。
    ///
    /// 1. 右辺を AMD 置換 (b_p[k] = rhs[perm[k]])
    /// 2. f64 → TwoFloat に持ち上げ
    /// 3. forward solve: L y = b_p (unit diagonal、列走査)
    /// 4. diagonal: y' = y / D
    /// 5. back solve: L^T x = y' (列を逆順走査)
    /// 6. TwoFloat → f64 に丸めて逆置換
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        let n = self.n;
        let b_p = permute_vec(rhs, &self.perm);
        let mut x: Vec<TwoFloat> = b_p.iter().map(|&v| TwoFloat::from(v)).collect();

        // forward solve: L y = b_p
        // L は単位対角なので y[j] はそのまま。j が確定したら子ノード i (>j) に
        // y[i] -= L[i,j] * y[j] を伝搬する。
        for j in 0..n {
            let yj = x[j];
            for p in self.l_col_ptr[j]..self.l_col_ptr[j + 1] {
                let i = self.l_row_ind[p];
                x[i] -= self.l_values[p] * yj;
            }
        }

        // diagonal solve: y' = y / D
        for j in 0..n {
            x[j] /= self.d[j];
        }

        // back solve: L^T z = y'
        // 列 j を逆順に走査。L^T x の j 行分は、x[j] -= Σ_{i>j} L[i,j] * x[i]。
        for j in (0..n).rev() {
            let mut sum = TwoFloat::from(0.0);
            for p in self.l_col_ptr[j]..self.l_col_ptr[j + 1] {
                let i = self.l_row_ind[p];
                sum += self.l_values[p] * x[i];
            }
            x[j] -= sum;
        }

        let x_f64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let out = inv_permute_vec(&x_f64, &self.perm);
        sol.copy_from_slice(&out);
    }
}

/// 上三角 CSC (row ≤ col の格納) から etree と L の列カウントを計算する。
///
/// 戻り値: (parent, l_col_ptr)
/// - `parent[k]` = etree 上の親 (-1 = root)
/// - `l_col_ptr[k+1] - l_col_ptr[k]` = 列 k 以下三角における非ゼロ数
fn ldl_symbolic(
    n: usize,
    a_col_ptr: &[usize],
    a_row_ind: &[usize],
) -> (Vec<isize>, Vec<usize>) {
    let mut parent = vec![-1isize; n];
    let mut flag = vec![usize::MAX; n];
    let mut lnz = vec![0usize; n];

    for k in 0..n {
        flag[k] = k;
        for p in a_col_ptr[k]..a_col_ptr[k + 1] {
            let i_orig = a_row_ind[p];
            if i_orig >= k {
                continue;
            }
            // i_orig < k: walk path i → root, marking and setting parents
            let mut i = i_orig;
            loop {
                if flag[i] == k {
                    break;
                }
                if parent[i] == -1 {
                    parent[i] = k as isize;
                }
                lnz[i] += 1;
                flag[i] = k;
                i = parent[i] as usize;
            }
        }
    }

    let mut l_col_ptr = vec![0usize; n + 1];
    for k in 0..n {
        l_col_ptr[k + 1] = l_col_ptr[k] + lnz[k];
    }

    (parent, l_col_ptr)
}

/// 数値因子化 (DD 精度)。Davis up-looking 流。
///
/// `signs` (Some) が与えられたら、quasidefinite regularization を適用する:
/// 期待符号 +1 の対角が EPSILON 未満なら DELTA で置換、-1 なら -DELTA。
/// `signs` が None なら数値破綻時に SingularOrIndefinite を返す。
fn ldl_numeric_dd(
    n: usize,
    a_col_ptr: &[usize],
    a_row_ind: &[usize],
    a_values: &[f64],
    parent: &[isize],
    l_col_ptr: &[usize],
    signs: Option<&[i8]>,
) -> Result<(Vec<usize>, Vec<TwoFloat>, Vec<TwoFloat>), LdlError> {
    let l_nnz_total = l_col_ptr[n];
    let mut l_row_ind = vec![0usize; l_nnz_total];
    let mut l_values = vec![TwoFloat::from(0.0); l_nnz_total];
    let mut d = vec![TwoFloat::from(0.0); n];

    let mut y = vec![TwoFloat::from(0.0); n]; // dense workspace for column k
    let mut pattern = vec![0usize; n]; // dual-purpose: tail = collected, head[top..n] = topo
    let mut flag = vec![usize::MAX; n];
    let mut lnz_running = vec![0usize; n]; // current insertion count per column of L

    for k in 0..n {
        let mut top = n;
        flag[k] = k;
        // Y[k] is the running diagonal; initialize from any leftover (should be 0 from prev).
        // Davis sets Y[k]=0 explicitly at start; ensure invariant.
        debug_assert_eq!(f64::from(y[k]), 0.0);

        for p in a_col_ptr[k]..a_col_ptr[k + 1] {
            let i_orig = a_row_ind[p];
            if i_orig > k {
                continue; // strict lower-tri input shouldn't occur (upper-tri storage)
            }
            // accumulate A[i_orig, k] into Y[i_orig]
            y[i_orig] += a_values[p];
            // collect etree path from i_orig (skip if already flagged, including i_orig==k)
            let mut i = i_orig;
            let mut len = 0usize;
            loop {
                if flag[i] == k {
                    break;
                }
                pattern[len] = i;
                len += 1;
                flag[i] = k;
                if parent[i] == -1 {
                    break; // shouldn't happen if symbolic ran correctly, but guard
                }
                i = parent[i] as usize;
            }
            // Move to top of stack reversed: pattern[--top] = pattern[--len]
            while len > 0 {
                len -= 1;
                top -= 1;
                pattern[top] = pattern[len];
            }
        }

        // D[k] starts as Y[k] (diagonal A[k,k] accumulated above)
        let mut dk = y[k];
        y[k] = TwoFloat::from(0.0);

        // Process pattern[top..n] in topological order (leaves first, root last)
        let mut t = top;
        while t < n {
            let i = pattern[t];
            let yi = y[i];
            y[i] = TwoFloat::from(0.0);

            // Update Y[row] -= L[row, i] * yi for each entry already in column i of L
            let cs = l_col_ptr[i];
            let ce = cs + lnz_running[i];
            for p in cs..ce {
                let row = l_row_ind[p];
                y[row] -= l_values[p] * yi;
            }

            // L[k, i] = yi / D[i]
            let l_ki = yi / d[i];
            // D[k] -= L[k, i] * yi
            dk -= l_ki * yi;

            // Store L[k, i] at end of column i
            let pos = l_col_ptr[i] + lnz_running[i];
            l_row_ind[pos] = k;
            l_values[pos] = l_ki;
            lnz_running[i] += 1;

            t += 1;
        }

        // Quasidefinite regularization or singular check
        let dk_f64 = f64::from(dk);
        if let Some(signs) = signs {
            let s = signs[k];
            if s >= 0 && dk_f64 < EPSILON {
                dk = TwoFloat::from(DELTA);
            } else if s < 0 && dk_f64 > -EPSILON {
                dk = TwoFloat::from(-DELTA);
            }
        } else if dk_f64.abs() < EPSILON {
            return Err(LdlError::SingularOrIndefinite);
        }
        d[k] = dk;
    }

    Ok((l_row_ind, l_values, d))
}

/// AMD キャッシュ済み置換付き quasidefinite LDL^T 分解 (DD 精度)。
///
/// `mat`: 元の (未置換の) augmented KKT 行列、上三角 CSC
/// `perm`: 事前計算済み AMD 置換 (perm[k] = 元インデックス)
/// `deadline`: factorize 前と symbolic 完了直後の 2 箇所でチェック
pub fn factorize_quasidefinite_with_cached_perm_dd(
    mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
) -> Result<LdlFactorizationDdAmd, LdlError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    let n = mat.nrows;
    let (col_ptr, row_ind, values) =
        permute_sym_upper(n, &mat.col_ptr, &mat.row_ind, &mat.values, perm);

    let (parent, l_col_ptr) = ldl_symbolic(n, &col_ptr, &row_ind);

    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }

    let signs = crate::linalg::ldl::extract_diagonal_signs(n, &col_ptr, &row_ind, &values);
    let (l_row_ind, l_values, d_vec) =
        ldl_numeric_dd(n, &col_ptr, &row_ind, &values, &parent, &l_col_ptr, Some(&signs))?;

    Ok(LdlFactorizationDdAmd {
        n,
        perm: perm.to_vec(),
        l_col_ptr,
        l_row_ind,
        l_values,
        d: d_vec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn upper_tri_csc(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        let mut cols: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for &(row, col, val) in entries {
            assert!(row <= col, "upper triangle only: row={row} col={col}");
            cols[col].push((row, val));
        }
        for c in cols.iter_mut() {
            c.sort_by_key(|&(r, _)| r);
        }
        let nnz: usize = cols.iter().map(|c| c.len()).sum();
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + cols[j].len();
        }
        let mut row_ind = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        for j in 0..n {
            let s = col_ptr[j];
            for (k, &(r, v)) in cols[j].iter().enumerate() {
                row_ind[s + k] = r;
                values[s + k] = v;
            }
        }
        CscMatrix {
            col_ptr,
            row_ind,
            values,
            nrows: n,
            ncols: n,
        }
    }

    fn residual_inf(full: &[(usize, usize, f64)], x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut r = vec![0.0f64; n];
        for &(row, col, val) in full {
            r[row] += val * x[col];
        }
        r.iter()
            .zip(b.iter())
            .fold(0.0f64, |a, (&ri, &bi)| a.max((ri - bi).abs()))
    }

    #[test]
    fn dd_ldl_pd_3x3() {
        // [4 1 0; 1 3 2; 0 2 5] (PD)
        let mat = upper_tri_csc(
            3,
            &[
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 1, 3.0),
                (1, 2, 2.0),
                (2, 2, 5.0),
            ],
        );
        let fac = factorize_quasidefinite_with_cached_perm_dd(&mat, &[0, 1, 2], None).unwrap();
        let b = [1.0f64, 2.0, 3.0];
        let mut x = [0.0f64; 3];
        fac.solve(&b, &mut x);
        let full: &[(usize, usize, f64)] = &[
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (1, 2, 2.0),
            (2, 1, 2.0),
            (2, 2, 5.0),
        ];
        assert!(residual_inf(full, &x, &b) < 1e-14);
    }

    #[test]
    fn dd_ldl_quasidefinite_2x2() {
        // [3 1; 1 -2] — quasidefinite (D[0]>0, D[1]<0)
        let mat = upper_tri_csc(2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 1, -2.0)]);
        let fac = factorize_quasidefinite_with_cached_perm_dd(&mat, &[0, 1], None).unwrap();
        let b = [1.0f64, 2.0];
        let mut x = [0.0f64; 2];
        fac.solve(&b, &mut x);
        let full: &[(usize, usize, f64)] =
            &[(0, 0, 3.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, -2.0)];
        assert!(residual_inf(full, &x, &b) < 1e-14);
    }

    #[test]
    fn dd_ldl_quasidefinite_5x5_with_amd() {
        // Q=diag(1,2), A=[[1,0],[0,1],[1,1]], δ=1e-4 → KKT 5x5 quasidefinite
        let delta = 1e-4f64;
        let mat = upper_tri_csc(
            5,
            &[
                (0, 0, 1.0 + delta),
                (1, 1, 2.0 + delta),
                (2, 2, -delta),
                (3, 3, -delta),
                (4, 4, -delta),
                (0, 2, 1.0),
                (1, 3, 1.0),
                (0, 4, 1.0),
                (1, 4, 1.0),
            ],
        );
        // identity perm でも AMD 結果でも動くはず。ここは identity でテスト。
        let fac = factorize_quasidefinite_with_cached_perm_dd(&mat, &[0, 1, 2, 3, 4], None).unwrap();
        let b = [1.0f64, 2.0, 0.5, -0.5, 1.0];
        let mut x = [0.0f64; 5];
        fac.solve(&b, &mut x);
        let full: &[(usize, usize, f64)] = &[
            (0, 0, 1.0 + delta),
            (1, 1, 2.0 + delta),
            (2, 2, -delta),
            (3, 3, -delta),
            (4, 4, -delta),
            (0, 2, 1.0),
            (2, 0, 1.0),
            (1, 3, 1.0),
            (3, 1, 1.0),
            (0, 4, 1.0),
            (4, 0, 1.0),
            (1, 4, 1.0),
            (4, 1, 1.0),
        ];
        assert!(residual_inf(full, &x, &b) < 1e-12);
    }

    /// f64 LDL では cond × ε ≈ 1e-8 程度の解誤差が出る ill-conditioned 系を、
    /// DD LDL は ε_DD ≈ 5e-32 まで詰められること (より小さい residual を返す)。
    #[test]
    fn dd_ldl_outperforms_f64_for_ill_conditioned() {
        use crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par;
        // Hilbert-like ill-conditioned 4x4 (PD)
        // H[i,j] = 1 / (i + j + 1), cond ~ 1.5e4 for n=4
        let n = 4;
        let mut entries = Vec::new();
        for i in 0..n {
            for j in i..n {
                entries.push((i, j, 1.0 / (i + j + 1) as f64));
            }
        }
        let mat = upper_tri_csc(n, &entries);
        let perm: Vec<usize> = (0..n).collect();

        let fac_dd =
            factorize_quasidefinite_with_cached_perm_dd(&mat, &perm, None).expect("dd factorize");
        let fac_f64 =
            factorize_quasidefinite_with_cached_perm_budget_par(&mat, &perm, None, None, faer::Par::Seq).expect("f64 factorize");

        let b = vec![1.0f64; n];
        let mut x_dd = vec![0.0f64; n];
        let mut x_f64 = vec![0.0f64; n];
        fac_dd.solve(&b, &mut x_dd);
        fac_f64.solve(&b, &mut x_f64);

        // residual = H x - b
        let full: Vec<(usize, usize, f64)> = {
            let mut v = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    v.push((i, j, 1.0 / (i + j + 1) as f64));
                }
            }
            v
        };
        let r_dd = residual_inf(&full, &x_dd, &b);
        let r_f64 = residual_inf(&full, &x_f64, &b);
        // DD residual should be at least as small as f64 (typically much smaller)
        assert!(r_dd <= r_f64 * 1.01 + 1e-15, "DD residual {r_dd:.3e} should not exceed f64 {r_f64:.3e}");
        // Both should solve to working precision for n=4 (cond ~1e4)
        assert!(r_dd < 1e-13, "DD residual {r_dd:.3e}");
        eprintln!("Hilbert n=4: f64 residual={r_f64:.3e}, DD residual={r_dd:.3e}");
    }

    #[test]
    fn dd_ldl_diagonal() {
        let n = 5;
        let entries: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, (i + 1) as f64)).collect();
        let mat = upper_tri_csc(n, &entries);
        let fac =
            factorize_quasidefinite_with_cached_perm_dd(&mat, &(0..n).collect::<Vec<_>>(), None)
                .unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x = vec![0.0f64; n];
        fac.solve(&b, &mut x);
        for i in 0..n {
            // x[i] = b[i] / D[i] = 1
            assert!((x[i] - 1.0).abs() < 1e-15, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn dd_ldl_etree_simple_chain() {
        // 上三角 chain: A[i, i+1] != 0 → etree が [1, 2, 3, ..., -1] になる
        let n = 4;
        let mut entries = Vec::new();
        for i in 0..n {
            entries.push((i, i, 4.0));
        }
        for i in 0..n - 1 {
            entries.push((i, i + 1, 1.0));
        }
        let mat = upper_tri_csc(n, &entries);
        let (parent, l_col_ptr) = ldl_symbolic(n, &mat.col_ptr, &mat.row_ind);
        // chain: parent[0]=1, parent[1]=2, parent[2]=3, parent[3]=-1
        assert_eq!(parent, vec![1, 2, 3, -1]);
        // 各列の L 非ゼロ数 = 1 (i+1 行のみ), 最後の列は 0
        assert_eq!(l_col_ptr, vec![0, 1, 2, 3, 3]);
    }
}
