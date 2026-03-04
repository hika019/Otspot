//! AMD (Approximate Minimum Degree) 再順序化
//!
//! 対称疎行列の LDL^T 分解前に適用し、フィルイン (fill-in) を最小化する
//! 列並べ替えを計算する。
//! Simplified Minimum Degree アルゴリズム
//! (George & Liu 1981 / Davis "Direct Methods for Sparse Linear Systems" 参照)。

use std::time::Instant;

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
/// deadline 超過時は部分的な AMD 順序 + identity フォールバックの有効な置換を返す。
///
/// # アルゴリズム
/// Simplified Minimum Degree (SMD):
/// 1. 対称グラフの隣接リストを構築（上三角入力から対称化、対角除外）
/// 2. n 回繰り返し（100回ごとに deadline チェック）:
///    a. 最小次数（未消去隣接数）のノードを選択
///    b. 置換に記録し消去済みにマーク
///    c. 全隣接ノードペア間にフィルインエッジを追加（完全グラフ化）
///    d. 影響を受けたノードの次数を再計算
pub fn amd(n: usize, col_ptr: &[usize], row_ind: &[usize]) -> Vec<usize> {
    amd_with_deadline(n, col_ptr, row_ind, None)
}

/// AMD 再順序化を計算する（deadline 付き）。
///
/// deadline が指定された場合、100イテレーションごとにチェックし、
/// 超過時は残りノードを自然順で埋めた有効な置換を返す。
pub fn amd_with_deadline(n: usize, col_ptr: &[usize], row_ind: &[usize], deadline: Option<Instant>) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }

    // 対称隣接リスト構築（対角除外）
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for j in 0..n {
        for &i in &row_ind[col_ptr[j]..col_ptr[j + 1]] {
            if i != j {
                // 上三角入力: i < j が前提。対称グラフに両方向追加
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }

    let mut eliminated = vec![false; n];
    let mut degree: Vec<usize> = adj.iter().map(|a| a.len()).collect();
    let mut perm = vec![0usize; n];

    for (slot_idx, slot) in perm.iter_mut().enumerate() {
        // 毎イテレーション deadline チェック（fill-in後は1イテレーションが数秒になりうるため）
        // deadline=None の場合は if let が短絡評価されるためオーバーヘッドはゼロ
        if let Some(d) = deadline {
            if Instant::now() >= d {
                // deadline 超過: 残りの未消去ノードを自然順で埋めて有効な置換を返す
                let mut remaining_slot = slot_idx;
                for (i, &elim) in eliminated.iter().enumerate() {
                    if !elim {
                        perm[remaining_slot] = i;
                        remaining_slot += 1;
                    }
                }
                return perm;
            }
        }

        // 最小次数の未消去ノードを線形探索
        let min_node = (0..n)
            .filter(|&i| !eliminated[i])
            .min_by_key(|&i| degree[i])
            .unwrap();

        *slot = min_node;
        eliminated[min_node] = true;

        // 未消去の隣接ノード一覧を収集
        let neighbors: Vec<usize> = adj[min_node]
            .iter()
            .copied()
            .filter(|&nb| !eliminated[nb])
            .collect();

        // フィルインエッジを追加: 全隣接ノードペア間を完全グラフ化
        // adj は Vec<Vec<usize>> であり異なるインデックスへの逐次アクセスは安全
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let u = neighbors[i];
                let v = neighbors[j];
                adj[u].push(v);
                adj[v].push(u);
            }
        }
        // 追加後に sort & dedup（重複除去）
        for &u in &neighbors {
            adj[u].sort_unstable();
            adj[u].dedup();
        }

        // 影響を受けたノードの次数を再計算（消去済みを除外）
        for &u in &neighbors {
            degree[u] = adj[u].iter().filter(|&&nb| !eliminated[nb]).count();
        }
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
    fn test_amd_star_graph() {
        // 星グラフ: ノード0が中心（次数4）、ノード1-4が葉（次数1）
        // 隣接: (0,1),(0,2),(0,3),(0,4) — 上三角CSC（対角なし）
        // AMDは最初に次数最小（1）の葉を消去する。中心ノード0（次数4）は最初には選ばれない
        let n = 5;
        let col_ptr = vec![0, 0, 1, 2, 3, 4];
        let row_ind = vec![0, 0, 0, 0]; // col1:(0,1), col2:(0,2), col3:(0,3), col4:(0,4)

        let perm = amd(n, &col_ptr, &row_ind);

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

        let perm = amd(n, &col_ptr, &row_ind);

        assert_eq!(perm.len(), n);
        let mut check = perm.clone();
        check.sort_unstable();
        assert_eq!(check, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_amd_empty() {
        let perm = amd(0, &[0], &[]);
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
