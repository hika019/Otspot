//! §3.1 分類器と Phase 0 実測 b1 表 (`/tmp/opencode/qcqp_phase0/REPORT.md`) の
//! 突合 sentinel。4 実データ問 (2546 / 8595 / 8605 / 8685) を実パーサ経由で
//! `solve_qp_with` に入力し、G1 の拒否メッセージが報告する bound 要求集合 `R`
//! のサイズを抽出して、Phase 0 の b1 との一致/乖離を固定する。
//!
//! ## 突合の判定基準 (計画書 §3.1「PSD 判定は sparse_cholesky_lower と契約を
//! 共有、convexity_unproven は凸扱いしない」)
//!
//! Phase 0 の b1 は外部スクリプトの Gershgorin→eigvalsh (tol=1e-9·|S|max) で
//! 測った値で、repo 側分類器 (Gershgorin 十分条件 ∨ sparse_cholesky の
//! `Ok ∧ ¬clamped`) は意図的に保守側へずれる。乖離は「計画書が認めた想定内」
//! であり一致は要求しないが、どの経路でずれたかは本テストと下のノートで固定する。
//!
//! ## QPLIB_2546 の決着 (2026-11-13 実測)
//!
//! b1 (Phase 0) = 415 (二次制約側の双線形変数のみ)。repo 分類器の |R| = 799。
//! 差分 384 は目的側の変数: 目的 P₀ (n=1015・二次 808 項・support 424 変数) が
//! Gershgorin 十分条件に失敗し、`sparse_cholesky_lower` も support 部分行列
//! (424×424) に対して **Err (認証可能な負の 2×2 小行列)** を返す (2026-11-13
//! 実測、判定コードは qcqp_classify.rs のモジュール doc に記録)。よって目的は非凸行
//! として規則 3 で対角外ペアの両端点 384 変数が R に入る (目的変数 424 と制約
//! 変数 415 の共通は §0.2 のとおり 0。残り 40 変数は非ゼロ非対角ペアの端点で
//! なく、対角係数は実測 neg_diag=0 で規則 2 より R 外)。eigvalsh (tol=1e-9·|S|max) はこの負小行列を tol 内とみなして凸と
//! 数えるが、repo 契約 (`convexity_unproven` を凸扱いしない) はこれを凸にしない
//! — 想定内の乖離。

use otspot::io::qplib::{parse_qplib, QplibProblem};
use otspot::options::SolverOptions;
use otspot::problem::{SolveStatus, SolverResult};
use otspot::qp::{solve_qp_with, QpProblem};
use otspot::sparse::CscMatrix;
use std::path::Path;

const QPLIB_DIR: &str = "data/qplib";

/// `require_finite_bounds` の拒否メッセージから (R 内の非有限数, |R|) を読む。
/// メッセージ書式は `qcqp_classify::require_finite_bounds` 側のテスト
/// (`require_finite_bounds_rejects_unbounded_required_variable`) と共有。
fn r_gate_counts(res: &SolverResult) -> Option<(usize, usize)> {
    let SolveStatus::NotSupported(msg) = &res.status else {
        return None;
    };
    if !msg.contains("required variables have a non-finite bound") {
        return None;
    }
    // "… {bad} of the {r} required variables have a non-finite bound …"
    let head = msg.split(" required variables").next()?;
    let (bad, r) = head.split_once(" of the ")?;
    Some((
        bad.rsplit([' ', '\n'])
            .next()?
            .parse()
            .unwrap_or_else(|e| panic!("bad count in {msg}: {e}")),
        r.rsplit([' ', '\n'])
            .next()?
            .parse()
            .unwrap_or_else(|e| panic!("|R| count in {msg}: {e}")),
    ))
}

fn load_qplib_q(id: &str) -> QpProblem {
    let path = Path::new(QPLIB_DIR).join(format!("QPLIB_{id}.qplib"));
    match parse_qplib(&path).unwrap_or_else(|e| panic!("parse {id}: {e}")) {
        QplibProblem::Qp(p) => p,
        other => panic!("expected continuous QP for {id}, got {other:?}"),
    }
}

fn solve_expect_structural_rejection(problem: &QpProblem, id: &str, label: &str) -> SolverResult {
    let res = solve_qp_with(problem, &SolverOptions::default());
    assert!(
        matches!(res.status, SolveStatus::NotSupported(_)),
        "{id}/{label}: expected NotSupported gate, got {:?}",
        res.status
    );
    res
}

/// 目的 Q をゼロ行列に置き換えた変種 (R の目的由来分の分離用)。線形 c は
/// R に効かない (§3.1「線形項にしか現れない変数は不要」) ので目的二次部だけ
/// を消せば R は制約由来分だけになる。
fn without_objective_quadratic(problem: &QpProblem) -> QpProblem {
    let mut p = problem.clone();
    p.q = CscMatrix::new(p.num_vars, p.num_vars);
    p
}

/// 突合本体。各行: (問, Phase 0 b1, 実測 |R| 全体, 実測 |R| 制約由来のみ)。
/// 2546 のみ分類器が保守側へ食い違い (上のノート)、残りは b1 と一致する。
///
/// Sentinel: 全変数 finite 判定 (G1 前のゲート) を戻すと、メッセージが旧書式に
/// なり `r_gate_counts` が None → panic で fail する (計画 §5 sentinel 1 の
/// 実データ版)。逆に分類器が非対角規則 (規則 3) を落とすと |R| が減少して
/// assert が崩れる。
#[test]
fn qcqp_phase0_bound_requirement_reconciliation() {
    // (id, phase0_b1, r_full, r_constraints_only)
    let cases: [(&str, usize, usize, usize); 4] = [
        ("2546", 415, 799, 415),
        ("8595", 2500, 2500, 2500),
        ("8605", 5000, 5000, 5000),
        ("8685", 772, 772, 772),
    ];
    for (id, b1, r_full, r_quad) in cases {
        let problem = load_qplib_q(id);
        let res = solve_expect_structural_rejection(&problem, id, "raw");
        let (bad_full, r) = r_gate_counts(&res)
            .unwrap_or_else(|| panic!("{id}: raw bounds did not hit the R gate: {:?}", res.status));
        assert_eq!(r, r_full, "{id}: |R| from the classifier");
        assert_eq!(
            bad_full, r_full,
            "{id}: every R variable has a raw non-finite bound in these instances"
        );

        let no_obj = without_objective_quadratic(&problem);
        let res = solve_expect_structural_rejection(&no_obj, id, "no-obj-quad");
        let (_, r_con) = r_gate_counts(&res)
            .unwrap_or_else(|| panic!("{id}: no-obj variant did not hit the R gate"));
        assert_eq!(
            r_con, r_quad,
            "{id}: constraint-derived |R| must match Phase 0 b1 ({b1})"
        );
        assert_eq!(
            r_con, b1,
            "{id}: constraint-derived |R| != Phase 0 b1 ({b1}) — classifier drift"
        );
        eprintln!(
            "[qcqp_phase0_reconcile] {id}: b1={b1} |R|={r_full} (constraints {r_con}, \
             objective-derived {})",
            r_full - r_con
        );
    }
}

/// 2546 乖離の機構固定: 分類器 |R| 799 = 制約由来 415 (Phase 0 b1 と一致) +
/// 目的 P₀ 由来 384。目的由来分はファイル上の独立変数で、raw と no-obj 変種
/// の差分として直接観測する (二次制約行は両変種で同一なので、差分は純粋に
/// 目的 P₀ が規則 3 で追加する対角外ペアの端点である)。obj-only への追加
/// 変種は `has_qcqp_constraints()` が QCQP ルートへの dispatch を取り消す
/// (二次行が全滅すると QP/IPM 経路へ移り、ゲート自体を通らない) ため使えない。
#[test]
fn qcqp_2546_objective_split_adds_rule3_endpoints() {
    let problem = load_qplib_q("2546");
    let raw = solve_expect_structural_rejection(&problem, "2546", "raw");
    let no_obj = without_objective_quadratic(&problem);
    let stripped = solve_expect_structural_rejection(&no_obj, "2546", "no-obj-quad");
    let (_, r_full) = r_gate_counts(&raw).expect("raw hits the R gate");
    let (_, r_con) = r_gate_counts(&stripped).expect("no-obj variant hits the R gate");
    assert_eq!(r_con, 415, "constraint-derived part equals Phase 0 b1");
    assert_eq!(
        r_full,
        r_con + 384,
        "objective P₀ adds 384 rule-3 endpoints"
    );
    // 目的変数と二次制約変数は共通しない (計画書 §0.2): 制約由来 415 は
    // 目的の凸扱い成否と無関係に不変でよい (上の r_con 固定)。
}
