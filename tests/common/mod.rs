//! 宣言的 LP/QP problem builder。math (`min cᵀx s.t. Ax{≤,=,≥}b, lb≤x≤ub`) と
//! 1:1 対応する API で `CscMatrix::from_triplets` 直書きの可読性問題を解消する。
//!
//! 設計: Option C (関数 helper) — math 引数順そのまま、追加状態なし。
//! 拡張時 (mixed / general Q) も同 module に関数を追加するだけで吸収。

#![allow(dead_code)]

use solver::problem::{ConstraintType, LpProblem};
use solver::qp::QpProblem;
use solver::sparse::CscMatrix;

pub const INF: f64 = f64::INFINITY;
pub const NEG_INF: f64 = f64::NEG_INFINITY;

/// 1 制約行: `(kind, coefs, rhs)`。`coefs.len() == n_vars` 必須。
pub type Row<'a> = (ConstraintType, &'a [f64], f64);

/// `min cᵀx s.t. <rows>, lb ≤ x ≤ ub` の LP を `QpProblem` (Q=0) として構築。
pub fn lp(c: &[f64], rows: &[Row<'_>], bounds: &[(f64, f64)]) -> QpProblem {
    build_qp(None, c, rows, bounds)
}

/// `min ½ xᵀ diag(q) x + cᵀx s.t. <rows>, lb ≤ x ≤ ub` の QP。
/// Q 対角 (現行 test は全て対角) のみ扱う。off-diag 必要時は別 helper を追加。
pub fn qp_diag(
    q_diag: &[f64],
    c: &[f64],
    rows: &[Row<'_>],
    bounds: &[(f64, f64)],
) -> QpProblem {
    build_qp(Some(q_diag), c, rows, bounds)
}

/// `LpProblem` を欲しい test (`LpProblem::new_general` 経由 path) 用。
/// `lp` と signature 同形、return type だけ差し替え。
pub fn lp_problem(
    c: &[f64],
    rows: &[Row<'_>],
    bounds: &[(f64, f64)],
    name: Option<&str>,
) -> LpProblem {
    let n = c.len();
    let (a, b, cts) = build_a_b_cts(n, rows);
    LpProblem::new_general(
        c.to_vec(),
        a,
        b,
        cts,
        bounds.to_vec(),
        name.map(str::to_owned),
    )
    .expect("lp_problem: dimensions must match")
}

pub fn le<'a>(coefs: &'a [f64], rhs: f64) -> Row<'a> {
    (ConstraintType::Le, coefs, rhs)
}
pub fn ge<'a>(coefs: &'a [f64], rhs: f64) -> Row<'a> {
    (ConstraintType::Ge, coefs, rhs)
}
pub fn eq<'a>(coefs: &'a [f64], rhs: f64) -> Row<'a> {
    (ConstraintType::Eq, coefs, rhs)
}

fn build_qp(
    q_diag: Option<&[f64]>,
    c: &[f64],
    rows: &[Row<'_>],
    bounds: &[(f64, f64)],
) -> QpProblem {
    let n = c.len();
    assert_eq!(
        bounds.len(),
        n,
        "bounds length must equal |c|={n}, got {}",
        bounds.len()
    );

    let q = match q_diag {
        None => CscMatrix::new(n, n),
        Some(d) => {
            assert_eq!(d.len(), n, "q_diag length must equal n={n}, got {}", d.len());
            let mut r = Vec::with_capacity(n);
            let mut col = Vec::with_capacity(n);
            let mut v = Vec::with_capacity(n);
            for (i, &val) in d.iter().enumerate() {
                if val != 0.0 {
                    r.push(i);
                    col.push(i);
                    v.push(val);
                }
            }
            CscMatrix::from_triplets(&r, &col, &v, n, n).expect("q_diag CSC build")
        }
    };

    let (a, b, cts) = build_a_b_cts(n, rows);
    QpProblem::new(q, c.to_vec(), a, b, bounds.to_vec(), cts).expect("qp build: dimensions")
}

fn build_a_b_cts(
    n: usize,
    rows: &[Row<'_>],
) -> (CscMatrix, Vec<f64>, Vec<ConstraintType>) {
    let m = rows.len();
    for (i, (_, coefs, _)) in rows.iter().enumerate() {
        assert_eq!(
            coefs.len(),
            n,
            "row {i} coefs must have {n} entries, got {}",
            coefs.len()
        );
    }

    let a = if m == 0 {
        CscMatrix::new(0, n)
    } else {
        // col-major triplet (CscMatrix::from_triplets が CSC を期待)
        let mut rs = Vec::new();
        let mut cs = Vec::new();
        let mut vs = Vec::new();
        for j in 0..n {
            for (i, (_, coefs, _)) in rows.iter().enumerate() {
                if coefs[j] != 0.0 {
                    rs.push(i);
                    cs.push(j);
                    vs.push(coefs[j]);
                }
            }
        }
        CscMatrix::from_triplets(&rs, &cs, &vs, m, n).expect("a CSC build")
    };

    let b: Vec<f64> = rows.iter().map(|(_, _, r)| *r).collect();
    let cts: Vec<ConstraintType> = rows.iter().map(|(k, _, _)| *k).collect();
    (a, b, cts)
}
