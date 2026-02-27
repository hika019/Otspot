//! テスト共通ユーティリティ（basis モジュール内）
//!
//! `lu.rs` と `mod.rs` のテスト双方で使用するヘルパー関数を一元化する。

use crate::sparse::CscMatrix;
use crate::tolerances::DROP_TOL;

/// 二つのスライスが許容誤差 `tol` 以内で一致するかを検証する。
pub fn assert_vec_near(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "Vector lengths differ: {} vs {}",
        a.len(),
        b.len()
    );
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() < tol,
            "Mismatch at index {}: {} vs {} (diff={})",
            i,
            a[i],
            b[i],
            (a[i] - b[i]).abs()
        );
    }
}

/// 密行列スライスから CSC 行列を生成する。
pub fn dense_to_csc(dense: &[Vec<f64>], nrows: usize, ncols: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for (i, row) in dense.iter().enumerate().take(nrows) {
        for (j, &v) in row.iter().enumerate().take(ncols) {
            if v.abs() > DROP_TOL {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).unwrap()
}
