use crate::error::SolverError;
use crate::tolerances::DROP_TOL;
use std::collections::HashMap;

type CompressedFormat = (Vec<usize>, Vec<usize>, Vec<f64>);

/// COO トリプレットを圧縮形式（CSC/CSR 共通）に変換する内部ヘルパー
///
/// `major_indices` が主軸（CSC では列、CSR では行）、`minor_indices` が副軸。
/// 重複エントリの加算・境界チェック・DROP_TOL フィルタリング・ソートを行い、
/// `(major_ptr, minor_ind, values)` を返す。
pub(super) fn build_compressed_format(
    n_major: usize,
    n_minor: usize,
    major_indices: &[usize],
    minor_indices: &[usize],
    vals: &[f64],
) -> Result<CompressedFormat, SolverError> {
    let mut map: HashMap<(usize, usize), f64> = HashMap::new();
    for i in 0..major_indices.len() {
        let maj = major_indices[i];
        let min = minor_indices[i];
        if maj >= n_major {
            return Err(SolverError::IndexOutOfBounds { context: "major", index: maj, bound: n_major });
        }
        if min >= n_minor {
            return Err(SolverError::IndexOutOfBounds { context: "minor", index: min, bound: n_minor });
        }
        *map.entry((maj, min)).or_insert(0.0) += vals[i];
    }

    let mut triplets: Vec<(usize, usize, f64)> = map
        .into_iter()
        .filter(|(_, v)| v.abs() > DROP_TOL)
        .map(|((maj, min), v)| (maj, min, v))
        .collect();
    triplets.sort_by_key(|&(maj, min, _)| (maj, min));

    let mut major_ptr = vec![0; n_major + 1];
    let mut minor_ind = Vec::new();
    let mut values = Vec::new();

    let mut current_major = 0;
    for (maj, min, v) in triplets {
        while current_major < maj {
            current_major += 1;
            major_ptr[current_major] = minor_ind.len();
        }
        minor_ind.push(min);
        values.push(v);
    }
    while current_major < n_major {
        current_major += 1;
        major_ptr[current_major] = minor_ind.len();
    }

    Ok((major_ptr, minor_ind, values))
}
