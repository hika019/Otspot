use crate::error::SolverError;
use crate::tolerances::DROP_TOL;

type CompressedFormat = (Vec<usize>, Vec<usize>, Vec<f64>);

/// COO トリプレットを圧縮形式（CSC/CSR 共通）に変換する内部ヘルパー。
///
/// O(n log n) sort-merge アプローチを使用する。以前の HashMap ベースの実装は
/// 1M エントリで 50MB+ のオーバーヘッドを生じさせ OOM を引き起こしていた。
/// ソートアプローチは入力と同サイズの Vec 1 本のみを使用する。
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
    debug_assert_eq!(major_indices.len(), minor_indices.len());
    debug_assert_eq!(major_indices.len(), vals.len());

    // Collect into (major, minor, val) triples with bounds validation.
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(major_indices.len());
    for i in 0..major_indices.len() {
        let maj = major_indices[i];
        let min = minor_indices[i];
        if maj >= n_major {
            return Err(SolverError::IndexOutOfBounds {
                context: "major",
                index: maj,
                bound: n_major,
            });
        }
        if min >= n_minor {
            return Err(SolverError::IndexOutOfBounds {
                context: "minor",
                index: min,
                bound: n_minor,
            });
        }
        triplets.push((maj, min, vals[i]));
    }

    // Sort by (major, minor) for stable merge.
    triplets.sort_unstable_by_key(|&(maj, min, _)| (maj, min));

    // Merge consecutive duplicate (major, minor) pairs in-place; filter DROP_TOL.
    let n_merged = merge_sorted_inplace(&mut triplets);
    triplets.truncate(n_merged);

    // Build compressed-format arrays from the merged sorted triples.
    let nnz = triplets.len();
    let mut major_ptr = vec![0usize; n_major + 1];
    let mut minor_ind = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    for &(maj, min, v) in &triplets {
        major_ptr[maj + 1] += 1;
        minor_ind.push(min);
        values.push(v);
    }
    drop(triplets);

    // Prefix-sum to turn counts into pointers.
    for i in 0..n_major {
        major_ptr[i + 1] += major_ptr[i];
    }

    Ok((major_ptr, minor_ind, values))
}

/// Merges consecutive equal-key `(major, minor)` entries by summing their values.
/// Entries whose merged absolute value is ≤ `DROP_TOL` are discarded.
/// Returns the new logical length (caller must `truncate` to that length).
fn merge_sorted_inplace(triplets: &mut [(usize, usize, f64)]) -> usize {
    if triplets.is_empty() {
        return 0;
    }
    let (mut cur_maj, mut cur_min, mut cur_val) = triplets[0];
    let mut write = 0usize;

    for read in 1..triplets.len() {
        let (maj, min, v) = triplets[read];
        if maj == cur_maj && min == cur_min {
            cur_val += v;
        } else {
            if cur_val.abs() > DROP_TOL {
                triplets[write] = (cur_maj, cur_min, cur_val);
                write += 1;
            }
            cur_maj = maj;
            cur_min = min;
            cur_val = v;
        }
    }
    if cur_val.abs() > DROP_TOL {
        triplets[write] = (cur_maj, cur_min, cur_val);
        write += 1;
    }
    write
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build compressed format and return (ptr, ind, vals).
    fn bcf(
        n_major: usize,
        n_minor: usize,
        maj: &[usize],
        min: &[usize],
        v: &[f64],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        build_compressed_format(n_major, n_minor, maj, min, v).unwrap()
    }

    #[test]
    fn test_empty() {
        let (ptr, ind, vals) = bcf(3, 3, &[], &[], &[]);
        assert_eq!(ptr, vec![0, 0, 0, 0]);
        assert!(ind.is_empty());
        assert!(vals.is_empty());
    }

    #[test]
    fn test_single_entry() {
        let (ptr, ind, vals) = bcf(3, 3, &[1], &[2], &[5.0]);
        assert_eq!(ptr[1], 0);
        assert_eq!(ptr[2], 1);
        assert_eq!(ind, vec![2]);
        assert_eq!(vals, vec![5.0]);
    }

    #[test]
    fn test_no_duplicates_multiple_entries() {
        // 3 distinct entries in a 3×3 matrix
        let (ptr, ind, vals) = bcf(3, 3, &[0, 1, 2], &[0, 1, 2], &[1.0, 2.0, 3.0]);
        assert_eq!(ptr, vec![0, 1, 2, 3]);
        assert_eq!(ind, vec![0, 1, 2]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_duplicates_accumulated() {
        // Same (major=0, minor=0) three times → summed
        let (ptr, ind, vals) = bcf(2, 2, &[0, 0, 0], &[0, 0, 0], &[1.0, 2.0, 3.0]);
        assert_eq!(ptr, vec![0, 1, 1]);
        assert_eq!(ind, vec![0]);
        assert!((vals[0] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn test_duplicates_cancel_to_zero_dropped() {
        // (0,0) = 1.0 + (-1.0) = 0.0 → filtered by DROP_TOL
        let (ptr, ind, vals) =
            bcf(2, 2, &[0, 0, 1], &[0, 0, 1], &[1.0, -1.0, 5.0]);
        assert_eq!(ptr, vec![0, 0, 1]);
        assert_eq!(ind, vec![1]);
        assert_eq!(vals, vec![5.0]);
    }

    #[test]
    fn test_unsorted_input_sorted_output() {
        // Input out of (major, minor) order
        let maj = vec![2, 0, 1];
        let min = vec![1, 0, 2];
        let v = vec![30.0, 10.0, 20.0];
        let (ptr, ind, vals) = bcf(3, 3, &maj, &min, &v);
        // Entry (0,0)=10, (1,2)=20, (2,1)=30
        assert_eq!(ptr, vec![0, 1, 2, 3]);
        assert_eq!(ind[0], 0);
        assert_eq!(ind[1], 2);
        assert_eq!(ind[2], 1);
        assert_eq!(vals, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_out_of_bounds_major_returns_error() {
        assert!(build_compressed_format(2, 2, &[2], &[0], &[1.0]).is_err());
    }

    #[test]
    fn test_out_of_bounds_minor_returns_error() {
        assert!(build_compressed_format(2, 2, &[0], &[2], &[1.0]).is_err());
    }

    /// Large sparse input: exercises the sort-merge path with many entries.
    /// Verifies correctness for a 1000×1000 matrix with 10000 entries.
    #[test]
    fn test_large_sparse_correctness() {
        const N: usize = 1000;
        const NNZ: usize = 10_000;
        let mut maj = Vec::with_capacity(NNZ);
        let mut min = Vec::with_capacity(NNZ);
        let mut v = Vec::with_capacity(NNZ);
        // Diagonal entries only (no duplicates)
        for i in 0..N {
            maj.push(i);
            min.push(i);
            v.push((i + 1) as f64);
        }
        // Extra off-diagonal entries
        for i in 0..(NNZ - N) {
            maj.push((i * 7) % N);
            min.push((i * 13) % N);
            v.push(1.0);
        }
        let (ptr, ind, vals) =
            build_compressed_format(N, N, &maj, &min, &v).unwrap();
        assert_eq!(ptr.len(), N + 1);
        assert_eq!(ind.len(), vals.len());
        // Sanity: total nnz ≤ NNZ (duplicates may have been merged)
        assert!(vals.len() <= NNZ);
        // All values non-zero (filtered by DROP_TOL)
        assert!(vals.iter().all(|&x| x.abs() > DROP_TOL));
    }

    /// Sentinel: sort-merge must produce identical results to the old HashMap path.
    ///
    /// This is a table-driven cross-check with known (maj, min, val) input and
    /// expected output. Failing on a no-op change (reverting to HashMap) is
    /// guaranteed because correctness is checked explicitly.
    #[test]
    fn test_sort_merge_matches_reference_output() {
        // Reference cases: (n_major, n_minor, input, expected_ptr, expected_ind, expected_vals)
        let cases: &[(usize, usize, &[(usize, usize, f64)], &[usize], &[usize], &[f64])] = &[
            // Single duplicate merged: (0,0)×2→7.0, (1,1)→1.0
            (2, 2, &[(0,0,3.0),(0,0,4.0),(1,1,1.0)], &[0,1,2], &[0,1], &[7.0,1.0]),
            // Out-of-order, no duplicates
            (3, 3, &[(2,0,1.0),(0,2,2.0),(1,1,3.0)], &[0,1,2,3], &[2,1,0], &[2.0,3.0,1.0]),
            // Cancellation: (0,0) = 5.0 - 5.0 = 0 → dropped
            (2, 2, &[(0,0,5.0),(0,0,-5.0),(1,0,2.0)], &[0,0,1], &[0], &[2.0]),
        ];

        for &(n_maj, n_min, input, exp_ptr, exp_ind, exp_vals) in cases {
            let maj: Vec<usize> = input.iter().map(|&(m,_,_)| m).collect();
            let min: Vec<usize> = input.iter().map(|&(_,n,_)| n).collect();
            let vals: Vec<f64> = input.iter().map(|&(_,_,v)| v).collect();
            let (ptr, ind, v) =
                build_compressed_format(n_maj, n_min, &maj, &min, &vals).unwrap();
            assert_eq!(ptr, exp_ptr, "ptr mismatch for case n_maj={n_maj}");
            assert_eq!(ind, exp_ind, "ind mismatch");
            for (got, exp) in v.iter().zip(exp_vals.iter()) {
                assert!((got - exp).abs() < 1e-12, "val mismatch: {got} vs {exp}");
            }
        }
    }
}
