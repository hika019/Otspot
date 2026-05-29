//! `SolverOptions.psd_check_max_n` の soundness sentinel。
//!
//! `solve_miqp` は `is_q_psd_by_cholesky` で Q の凸性を確認し、非凸 Q を拒否する。
//! `psd_check_max_n = Some(n)` を設定すると `Q.nrows > n` のチェックをスキップし、
//! convex として扱う。これは O(n²) メモリを抑えるためのトレードオフだが、
//! 非凸 Q を誤って受け入れる soundness 穴になる。
//!
//! ## sentinel 検出力 (no-op proof)
//!
//! - `solve_miqp` default opts → NonConvex 返却 (正常ガード)
//! - `solve_miqp` + `psd_check_max_n=Some(1000)` → NonConvex を返さない (穴の確認)
//! - `is_convex_with_limit` の unit test は `mip/problem.rs` 内に配置

use otspot::options::{MipConfig, SolverOptions};
use otspot::problem::SolveStatus;
use otspot::solve_miqp_with_stats;
use otspot::{CscMatrix, MiqpProblem, QpProblem};

/// n=1001 の不定値 Q を作る。
/// 対角はすべて正 (1.0)、Q[0,1]=2.0 で 左上 2x2 block が不定 (eigenvalues: -1, 3)。
fn indefinite_q_n1001() -> QpProblem {
    let n = 1001_usize;
    let mut rows = vec![];
    let mut cols = vec![];
    let mut vals = vec![];
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(1.0_f64);
    }
    // 上三角に off-diagonal: Q[0,1] = 2.0 → Q[0..2,0..2] = [[1,2],[2,1]], eigenvalues -1/3
    rows.push(0);
    cols.push(1);
    vals.push(2.0_f64);
    let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
    let a = CscMatrix::new(0, n);
    let bounds = vec![(0.0_f64, 10.0_f64); n];
    QpProblem::new_all_le(q, vec![0.0; n], a, vec![], bounds).unwrap()
}

/// default opts (psd_check_max_n=None) では NonConvex ガードが機能する。
///
/// **Sentinel**: `solve_miqp_with_stats` の convex ガードを削除すると
/// NonConvex が返らず、このテストが FAIL する。
#[test]
fn solve_miqp_default_rejects_indefinite_n1001() {
    let qp = indefinite_q_n1001();
    let m = MiqpProblem::new(qp, vec![0]).unwrap();
    let (result, _) = solve_miqp_with_stats(&m, &SolverOptions::default(), &MipConfig::default());
    assert!(
        matches!(result.status, SolveStatus::NonConvex(_)),
        "default opts で indefinite Q → NonConvex 必須 (got {:?})",
        result.status
    );
}

/// `psd_check_max_n=Some(1000)` では n=1001 indefinite Q が NonConvex で拒否されない (soundness 穴)。
///
/// **Sentinel**: `is_convex_with_limit` の size check を削除すると
/// Some(1000) でも `is_q_psd_by_cholesky` が走り false が返る
/// → NonConvex になり、このテストが FAIL する。
#[test]
fn solve_miqp_psd_limit_1000_skips_nonconvex_check_n1001() {
    let qp = indefinite_q_n1001();
    let m = MiqpProblem::new(qp, vec![0]).unwrap();
    let mut opts = SolverOptions::default();
    opts.psd_check_max_n = Some(1000);
    opts.timeout_secs = Some(5.0);
    let (result, _) = solve_miqp_with_stats(&m, &opts, &MipConfig::default());
    assert!(
        !matches!(result.status, SolveStatus::NonConvex(_)),
        "psd_check_max_n=Some(1000): n=1001 > 1000 → skip → NonConvex を返してはならない"
    );
}
