//! 小規模 Maros 凸 QP に対する Clarabel cross-check 個別 test + 厳密 assertion。
//! 各 test は QpProblem を parse し、Clarabel 参照解と internal obj で比較 (rel < 1e-4)。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;

#[path = "helpers/clarabel_utils.rs"]
mod clarabel_helper;
use clarabel_helper::{compute_internal_obj, solve_clarabel};

const CROSS_CHECK_TIMEOUT_SECS: f64 = 60.0;
/// 目的関数 (internal, offset 除く) の相対許容: bench eps=1e-6 と Clarabel
/// strict tol_feas=1e-9 を考慮し、1e-4 を許容上限とする (qp-survey の既存
/// `clarabel_cross_check::deep_check` も同値)。
const CROSS_OBJ_REL_TOL: f64 = 1e-4;

/// 1 問題分の cross-check 本体。data が無ければ panic (CLAUDE.md「SKIP 禁止」)。
/// 個別 test がデータ欠落で flaky にならないよう、各 test の冒頭で path 存在を確認。
fn cross_check_problem(name: &str) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    assert!(
        path.exists(),
        "{}: data file missing at {:?}, run scripts/setup_extra_benches.sh",
        name,
        path
    );
    let prob = parse_qps(&path).unwrap_or_else(|e| panic!("{}: parse failed: {:?}", name, e));

    let cl = solve_clarabel(&prob).unwrap_or_else(|| {
        panic!(
            "{}: Clarabel reference failed (Solved/AlmostSolved 期待)",
            name
        )
    });
    let cl_internal = compute_internal_obj(&prob, &cl.1);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(CROSS_CHECK_TIMEOUT_SECS);
    let r = solve_qp_with(&prob, &opts);

    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "{}: solver status must be Optimal, got {:?} (clarabel ok obj={:.6e})",
        name,
        r.status,
        cl.0
    );
    let our_internal = compute_internal_obj(&prob, &r.solution);
    let diff = (our_internal - cl_internal).abs();
    let scale = our_internal.abs().max(cl_internal.abs()).max(1.0);
    let rel = diff / scale;
    assert!(
        rel < CROSS_OBJ_REL_TOL,
        "{}: internal obj mismatch. ours={:.6e} clarabel={:.6e} rel={:.3e} (tol={:.1e})",
        name,
        our_internal,
        cl_internal,
        rel,
        CROSS_OBJ_REL_TOL
    );
}

// =============================================================================
// Small problems (n <= 10)
// =============================================================================

#[test]
fn cross_hs21() {
    cross_check_problem("HS21");
}

#[test]
fn cross_hs35() {
    cross_check_problem("HS35");
}

#[test]
fn cross_hs35mod() {
    cross_check_problem("HS35MOD");
}

#[test]
fn cross_hs76() {
    cross_check_problem("HS76");
}

#[test]
fn cross_hs268() {
    cross_check_problem("HS268");
}

#[test]
fn cross_s268() {
    cross_check_problem("S268");
}

#[test]
fn cross_zecevic2() {
    cross_check_problem("ZECEVIC2");
}

#[test]
fn cross_tame() {
    cross_check_problem("TAME");
}

#[test]
fn cross_genhs28() {
    cross_check_problem("GENHS28");
}

// =============================================================================
// Medium problems (sparse, n=100..500)
// =============================================================================

#[test]
fn cross_qadlittl() {
    cross_check_problem("QADLITTL");
}

#[test]
fn cross_qsc205() {
    cross_check_problem("QSC205");
}

#[test]
fn cross_qscagr7() {
    cross_check_problem("QSCAGR7");
}

#[test]
fn cross_dualc1() {
    cross_check_problem("DUALC1");
}

#[test]
fn cross_dualc5() {
    cross_check_problem("DUALC5");
}

#[test]
fn cross_dual1() {
    cross_check_problem("DUAL1");
}

#[test]
fn cross_dual2() {
    cross_check_problem("DUAL2");
}

#[test]
fn cross_dual3() {
    cross_check_problem("DUAL3");
}

// =============================================================================
// 等式系 (n=200~)
// =============================================================================

#[test]
fn cross_aug2d() {
    cross_check_problem("AUG2D");
}

#[test]
fn cross_aug2dc() {
    cross_check_problem("AUG2DC");
}
