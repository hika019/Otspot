//! LISWET-family (Maros-Meszaros) 12 問の Clarabel strict 真値取得 (diag)。
//!
//! 背景: baseline CSV の LISWET1/7/8/9/10/11/12 行は 2026-05-17 世代
//! (tag `clarabel_strict_tol1e-12`) の値で、当時のパーサの QUADOBJ 読み込み
//! が ½ スケールのまま Clarabel に渡していたため、問題の大域下界
//! (Q=I 対角なので -0.5·||c||², .mat の c と現行 QPS の c は一致実績あり)
//! を下回る実現不可能な値が CSV に残っている (例: LISWET7 は CSV -5.008e3
//! に対して下界 -2525.26)。ソルバ側は
//! tests/liswet7_11_obj_independent_check.rs で独立オラクル検証済み
//! (ours ≡ Clarabel 同一パース問題、rel_diff < 1%)。
//!
//! 本テストは現行パーサ + 現行 data/ で Clarabel strict (tol=1e-12,
//! max_iter=100k) を全 12 問にかけ、CSV 再生成用の真値を出力する。
//! 出力の cost_primal は obj_offset を含まない Clarabel 値。obj_offset が
//! 非零の問題では cost_primal + obj_offset を baseline に使うこと
//! (LISWET family は全問 QPS に RHS セクションがなく obj_offset=0)。
#![allow(clippy::field_reassign_with_default)]

use otspot::io::qps::parse_qps;

use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus};

use std::path::PathBuf;

#[path = "helpers/clarabel_utils.rs"]
mod clarabel_helper;
use clarabel_helper::build_clarabel;

const STRICT_TOL: f64 = 1e-12;
const STRICT_MAX_ITER: u32 = 100_000;

/// diag テスト: convergence 失敗は assert で落とす (真値取得が目的のため)。
#[test]
#[ignore = "diag (現行パーサ/データでの LISWET-family 12 問 Clarabel strict 真値取得; baseline 再生成用)"]
fn diag_liswet_family_clarabel_truth() {
    let names = [
        "LISWET1", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6", "LISWET7", "LISWET8",
        "LISWET9", "LISWET10", "LISWET11", "LISWET12",
    ];

    let mut failed: Vec<String> = Vec::new();

    eprintln!(
        "\n========= LISWET-family Clarabel strict truth (現行パーサ + 現行 data/) ========="
    );
    eprintln!(
        "{:10} {:16} {:>8} {:>20} {:>20} {:>12}",
        "problem", "status", "iters", "cost_primal", "cost+offset", "obj_offset"
    );

    for name in names {
        let path = PathBuf::from(format!("data/maros_meszaros/{name}.QPS"));
        if !path.exists() {
            eprintln!("[{name}] DATA MISSING: {path:?}");
            failed.push(format!("{name} (missing data)"));
            continue;
        }

        let prob = parse_qps(&path).expect("parse failed");
        let (p, q, a, b, cones) = build_clarabel(&prob);

        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.tol_gap_abs = STRICT_TOL;
        settings.tol_gap_rel = STRICT_TOL;
        settings.tol_feas = STRICT_TOL;
        settings.max_iter = STRICT_MAX_ITER;

        let mut solver =
            DefaultSolver::new(&p, &q, &a, &b, &cones, settings).expect("clarabel setup failed");
        solver.solve();

        let status = format!("{:?}", solver.info.status);
        let cost = solver.info.cost_primal;
        let total = cost + prob.obj_offset;
        let iters = solver.info.iterations;

        eprintln!(
            "{:10} {:16} {:>8} {:>20.12e} {:>20.12e} {:>12.4e}",
            name, status, iters, cost, total, prob.obj_offset
        );

        if !matches!(
            solver.info.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ) {
            failed.push(format!("{name} ({status})"));
        }
    }

    assert!(
        failed.is_empty(),
        "Clarabel strict did not converge on: {failed:?}"
    );
}
