//! AMD (Approximate Minimum Degree) 再順序化
//!
//! 対称疎行列の LDL^T 分解前に適用し、フィルイン (fill-in) を最小化する
//! 列並べ替えを計算する。
//! faer の AMD-2 アルゴリズム（`faer::sparse::linalg::amd::order`）を内部で使用する。
//! O(n + nnz) の高速 AMD で、独自 SMD 実装を置き換えた。

use std::time::Instant;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::amd;
use faer::sparse::SymbolicSparseColMatRef;

/// AMD 再順序化を計算する（deadline 付き）。
///
/// # 引数
/// - `n`: 行列サイズ
/// - `col_ptr`: 上三角 CSC 形式の列ポインタ（対角を含んでも含まなくてもよい）
/// - `row_ind`: 上三角 CSC 形式の行インデックス
/// - `deadline`: タイムアウト時刻。超過した場合は残りノードを自然順で埋めた有効な置換を返す。
///
/// # 戻り値
/// 置換ベクトル `perm` で `perm[k] = i` は
/// 消去ステップ `k` に元のノード `i` を割り当てることを意味する。
/// AMD 再順序化を計算する（deadline 付き）。
///
/// deadline が指定された場合、faer AMD-2 呼び出し前に1回チェックし、
/// 超過時は identity 置換 (0..n) を返す。
/// faer amd::order() がエラーを返した場合も identity fallback を返す。
pub fn amd_with_deadline(n: usize, col_ptr: &[usize], row_ind: &[usize], deadline: Option<Instant>) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }

    // deadline チェック（faer AMD は O(n+nnz) で高速なため呼び出し前1回で十分）
    if let Some(dl) = deadline {
        if Instant::now() >= dl {
            return (0..n).collect();
        }
    }

    let nnz = col_ptr[n];

    // faer AMD 呼び出し
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];

    let a = unsafe {
        SymbolicSparseColMatRef::<usize>::new_unchecked(
            n, n, col_ptr, None, row_ind,
        )
    };

    let req = amd::order_scratch::<usize>(n, nnz);
    let mut mem = MemBuffer::new(req);
    let stack = MemStack::new(&mut mem);

    if amd::order(&mut perm, &mut perm_inv, a, amd::Control::default(), stack).is_err() {
        return (0..n).collect(); // identity fallback（deadline超過時と同じパターン）
    }

    perm
}

/// 置換ベクトルの逆置換を計算する。
///
/// `perm[k] = i` → `inv_perm[i] = k`
pub fn inverse_perm(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut inv = vec![0usize; n];
    for (k, &i) in perm.iter().enumerate() {
        inv[i] = k;
    }
    inv
}

/// 対称行列を置換する: P A P^T の上三角 CSC を返す。
///
/// `perm[k] = i` は新ノード k が元ノード i に対応することを意味する。
/// 元の上三角 A の各エントリ A[i,j] (i≤j) が
/// P A P^T[inv_perm\[i\], inv_perm\[j\]] にマップされる。
///
/// # 戻り値
/// `(new_col_ptr, new_row_ind, new_values)` — PAP^T の上三角 CSC
pub fn permute_sym_upper(
    n: usize,
    col_ptr: &[usize],
    row_ind: &[usize],
    values: &[f64],
    perm: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let inv_perm = inverse_perm(perm);

    // 各エントリを (new_row, new_col, val) に変換し上三角に正規化
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..n {
        let new_j = inv_perm[j];
        for idx in col_ptr[j]..col_ptr[j + 1] {
            let i = row_ind[idx];
            let v = values[idx];
            let new_i = inv_perm[i];
            // 上三角に正規化: row <= col
            let (r, c) = if new_i <= new_j { (new_i, new_j) } else { (new_j, new_i) };
            entries.push((r, c, v));
        }
    }

    // 列優先、列内行昇順でソート
    entries.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    let nnz = entries.len();
    let mut new_col_ptr = vec![0usize; n + 1];
    for &(_, c, _) in &entries {
        new_col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        new_col_ptr[j + 1] += new_col_ptr[j];
    }
    let mut new_row_ind = vec![0usize; nnz];
    let mut new_values = vec![0.0f64; nnz];
    for (idx, &(r, _, v)) in entries.iter().enumerate() {
        new_row_ind[idx] = r;
        new_values[idx] = v;
    }

    (new_col_ptr, new_row_ind, new_values)
}

/// ベクトルを前方置換する: `(Pv)[k] = v[perm[k]]`
pub fn permute_vec(v: &[f64], perm: &[usize]) -> Vec<f64> {
    perm.iter().map(|&i| v[i]).collect()
}

/// ベクトルを逆置換する: `(P^T v)[perm[k]] = v[k]`
pub fn inv_permute_vec(v: &[f64], perm: &[usize]) -> Vec<f64> {
    let n = v.len();
    let mut out = vec![0.0f64; n];
    for (k, &i) in perm.iter().enumerate() {
        out[i] = v[k];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amd_n1() {
        // n=1のケース: identity permutation が返ること
        let col_ptr = vec![0usize, 0];
        let row_ind: Vec<usize> = vec![];
        let perm = amd_with_deadline(1, &col_ptr, &row_ind, None);
        assert_eq!(perm, vec![0]);
    }

    #[test]
    fn test_amd_star_graph() {
        // 星グラフ: ノード0が中心（次数4）、ノード1-4が葉（次数1）
        // 隣接: (0,1),(0,2),(0,3),(0,4) — 上三角CSC（対角なし）
        // AMDは最初に次数最小（1）の葉を消去する。中心ノード0（次数4）は最初には選ばれない
        let n = 5;
        let col_ptr = vec![0, 0, 1, 2, 3, 4];
        let row_ind = vec![0, 0, 0, 0]; // col1:(0,1), col2:(0,2), col3:(0,3), col4:(0,4)

        let perm = amd_with_deadline(n, &col_ptr, &row_ind, None);

        assert_eq!(perm.len(), n);
        // 中心ノード0は最初に消去されない（初期次数4 > 葉の次数1）
        assert_ne!(perm[0], 0, "Central node 0 should not be eliminated first");
        // 最初に消去されるのは葉ノード（次数1）のいずれか
        assert!(perm[0] >= 1 && perm[0] <= 4, "First eliminated node should be a leaf");
        // 全て有効な順列であること
        let mut check = perm.clone();
        check.sort_unstable();
        assert_eq!(check, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn test_amd_returns_valid_permutation() {
        // 任意の帯行列で有効な順列が返ることを確認
        let n = 4;
        // 帯行列上三角（対角なし）: (0,1),(1,2),(2,3)
        let col_ptr = vec![0, 0, 1, 2, 3];
        let row_ind = vec![0, 1, 2];

        let perm = amd_with_deadline(n, &col_ptr, &row_ind, None);

        assert_eq!(perm.len(), n);
        let mut check = perm.clone();
        check.sort_unstable();
        assert_eq!(check, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_amd_empty() {
        let perm = amd_with_deadline(0, &[0], &[], None);
        assert!(perm.is_empty());
    }

    #[test]
    fn test_inverse_perm() {
        let perm = vec![2, 0, 3, 1];
        let inv = inverse_perm(&perm);
        // perm[0]=2 → inv[2]=0
        // perm[1]=0 → inv[0]=1
        // perm[2]=3 → inv[3]=2
        // perm[3]=1 → inv[1]=3
        assert_eq!(inv, vec![1, 3, 0, 2]);
    }

    #[test]
    fn test_permute_sym_upper_swap() {
        // A = [[4,1],[1,3]] (上三角CSC)
        // perm = [1,0]: 新ノード0=元ノード1, 新ノード1=元ノード0
        // PAP^T = [[3,1],[1,4]] (上三角)
        let n = 2;
        let col_ptr = vec![0, 1, 3];
        let row_ind = vec![0, 0, 1]; // col0:(0,0)=4; col1:(0,1)=1,(1,1)=3
        let values = vec![4.0, 1.0, 3.0];
        let perm = vec![1, 0];

        let (new_col_ptr, new_row_ind, new_values) =
            permute_sym_upper(n, &col_ptr, &row_ind, &values, &perm);

        // PAP^T[0,0]=A[1,1]=3, PAP^T[0,1]=A[0,1]=1, PAP^T[1,1]=A[0,0]=4
        assert_eq!(new_col_ptr, vec![0, 1, 3]);
        assert_eq!(new_row_ind, vec![0, 0, 1]);
        let eps = 1e-14;
        assert!((new_values[0] - 3.0).abs() < eps, "A_p[0,0]={}", new_values[0]);
        assert!((new_values[1] - 1.0).abs() < eps, "A_p[0,1]={}", new_values[1]);
        assert!((new_values[2] - 4.0).abs() < eps, "A_p[1,1]={}", new_values[2]);
    }

    #[test]
    fn test_permute_vec() {
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let perm = vec![2, 0, 3, 1];
        // (Pv)[0]=v[2]=30, (Pv)[1]=v[0]=10, (Pv)[2]=v[3]=40, (Pv)[3]=v[1]=20
        let result = permute_vec(&v, &perm);
        assert_eq!(result, vec![30.0, 10.0, 40.0, 20.0]);
    }

    #[test]
    fn test_inv_permute_vec() {
        let v = vec![30.0, 10.0, 40.0, 20.0];
        let perm = vec![2, 0, 3, 1];
        // out[perm[0]]=v[0] → out[2]=30
        // out[perm[1]]=v[1] → out[0]=10
        // out[perm[2]]=v[2] → out[3]=40
        // out[perm[3]]=v[3] → out[1]=20
        let result = inv_permute_vec(&v, &perm);
        assert_eq!(result, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_permute_inv_permute_roundtrip() {
        // Pv を inv_permute_vec すると元の v に戻ること
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let perm = vec![3, 0, 4, 2, 1];
        let pv = permute_vec(&v, &perm);
        let recovered = inv_permute_vec(&pv, &perm);
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-14, "roundtrip failed: {} != {}", a, b);
        }
    }
}
