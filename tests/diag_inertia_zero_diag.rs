//! task#37 sentinel: zero-diagonal indefinite Q を `compute_inertia_correction`
//! が PSD と誤判定し、IPM が `q_is_indefinite=false` で `Optimal` を返す
//! 退化を防ぐ end-to-end test。
//!
//! ## 真因 (bug)
//! `is_q_psd_by_cholesky` の旧実装は LDL^T `ZeroPivot { index }` 経路で
//! `D[0..index]` が全て非負なら PSD と判定。Q=[[0,1],[1,0]] のような
//! 零対角 indefinite は index=0 で ZeroPivot し、prior D が空 → PSD 誤判定。
//! 結果 `compute_inertia_correction` が 0 を返し、IPM が status=Optimal を
//! claim する (本来 LocallyOptimal が正)。
//!
//! ## sentinel 設計
//! 既知 indefinite QP を solve、status が `LocallyOptimal` (= IPM が非凸を
//! 認識) に降格されることを確認。複数 data pattern (bilinear / mixed /
//! 3変数 chain) で誤判定の data 依存を排除。
//!
//! ## no-op proof
//! `src/linalg/ldl.rs` の `is_q_psd_by_cholesky` で `shift = 0.0` に書換 →
//! 該当 sentinel が FAIL (status=Optimal)。fix 復帰で PASS。検証済。

use solver::options::SolverOptions;
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;
use solver::SolveStatus;

fn solve(p: &QpProblem) -> SolveStatus {
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let r = solve_qp_with(p, &opts);
    r.status
}

/// Q=[[0,1],[1,0]] (full-symmetric), λ=±1 indefinite, box [-1,1]².
/// 局所解候補: (±1, ±1) corner, obj=±1。IPM cold は (0,0) saddle 周辺 →
/// 慣性補正必須。status は LocallyOptimal (or その他 indefinite 認識 status)
/// であるべき。
#[test]
fn sentinel_bilinear_zero_diag_status_not_optimal() {
    let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2]).unwrap();
    let status = solve(&p);
    assert!(
        !matches!(status, SolveStatus::Optimal),
        "bilinear Q=[[0,1],[1,0]] is indefinite (λ=±1); solver must NOT claim Optimal, got {:?}",
        status
    );
}

/// Q=[[0,1],[1,-1]] mixed zero+negative diag, indefinite.
/// 旧バグでは LDL^T pre-check で row 1 の diag=-1 < 0 → indefinite 直で
/// 検出済み (zero-diag 寄与なし) なので、これは control case。
#[test]
fn sentinel_mixed_zero_neg_diag_status_not_optimal() {
    let q = CscMatrix::from_triplets(
        &[0, 1, 1],
        &[1, 0, 1],
        &[1.0, 1.0, -1.0],
        2,
        2,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2]).unwrap();
    let status = solve(&p);
    assert!(
        !matches!(status, SolveStatus::Optimal),
        "Q=[[0,1],[1,-1]] indefinite, got {:?}",
        status
    );
}

/// 3 変数 zero-diag chain: Q[0,1]=Q[1,2]=1 のみ。
/// row sums = (1, 2, 1), 全対角 0 → 旧バグの主要 trigger pattern
/// (LDL^T pre-check 通過、ZeroPivot at col 0)。
#[test]
fn sentinel_zero_diag_chain_3var_status_not_optimal() {
    let q = CscMatrix::from_triplets(
        &[0, 1, 1, 2],
        &[1, 0, 2, 1],
        &[1.0, 1.0, 1.0, 1.0],
        3,
        3,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
    let p = QpProblem::new_all_le(
        q,
        vec![0.0; 3],
        a,
        vec![],
        vec![(-1.0, 1.0); 3],
    )
    .unwrap();
    let status = solve(&p);
    assert!(
        !matches!(status, SolveStatus::Optimal),
        "Q chain (zero diag, off=1) is indefinite (3 distinct eigenvalues), got {:?}",
        status
    );
}

/// control: 真の PSD Q は引き続き Optimal を返す (over-correction で
/// LocallyOptimal に過剰降格しないことを確認)。
#[test]
fn sentinel_psd_q_remains_optimal() {
    // Q=[[2,1],[1,2]] PD (λ=1, 3), box [-1,1]²。
    let q = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[2.0, 1.0, 1.0, 2.0],
        2,
        2,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2]).unwrap();
    let status = solve(&p);
    assert!(
        matches!(status, SolveStatus::Optimal),
        "PSD Q must remain Optimal (no over-correction), got {:?}",
        status
    );
}
