//! LISWET7 / LISWET11 偽 Optimal 証跡 sentinel (bug-frontier 2026-10-18)。
//!
//! 独立オラクル (HiGHS 1.14 生 QPS 直読み + 自前パーサ KKT 検証 + 弱双対下界) で
//! 確定した真値と、`prove_optimal` の infeasibility-mass witness
//! (`certificate.rs::DUAL_MASS_VIOL_FLOOR`) による棄却を integration で pin する。
//!
//! - LISWET7  真値 p* = −2026.419109214 (全行 active + 再帰双対 y ≥ 0 証明 +
//!   HiGHS 主双対 gap −4e-11)。旧返解 −2500.257302 (x̂ ≈ −c アトラクタ,
//!   viol 9.87e-8 × 9997 行) は不等式行質量を破る実行不能点なので
//!   Optimal を主張できない (witness が導入する条件)。
//! - LISWET11 真値 p* = −2476.226042863 (HiGHS Optimal + 自前 A^T y ≥ 0 +
//!   d(y) 下界 gap 6.1e-7)。baseline CSV の −5.015e3 は弱双対性
//!   (feasible x の obj ≥ max_y d(y)) より低い元素和要求で不可能値
//!   (`data/baseline_objectives/maros_meszaros.csv` の訂正は lead 報告)。
//!
//! 追記 (2026-10-19): 独立再裁定 (同一手順 + float128 弱双対下界/実行可能点上界の
//! 挟み込み) でも両真値を再確認 — 判定区間幅は LISWET7 ≤2.0e-11, LISWET11 ≤6.1e-7。
//! Clarabel 0.11.1 tol=1e-12 の AlmostSolved 反復点の値 (−2483.194968631 /
//! −2489.222682544) は大域下界を下回る到達不能値 (primal 違反 3.5e-7 × 7169 行 /
//! 3.5e-7 × 2710 行) で、baseline 参照値には使えない。
//!
//! bench (`scripts/bench_parallel.sh`) と同条件の eps / timeout を使う。
//! 解実装 heavy を tier-2 で走らせる (`--profile heavy --run-ignored all`)。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::{solve_qp_with, QpProblem};
use std::path::Path;

/// LISWET7 偽 mint (`obj = −0.5‖c‖²`, x̂ ≈ −c) は Optimal を主張してはならない。
///
/// **Sentinel (no-op-fail)**: `prove_optimal` の infeasible_mass 判定を外すと
/// attempt ladder の悪 mint (m̄ = 6.97e-8 > DUAL_MASS_VIOL_FLOOR) が Optimal に
/// 昇格し、status == Optimal で本テストが FAIL する。
#[test]
#[ignore = "tier-2"]
fn liswet7_false_optimal_rejected() {
    let prob = maros_qp("LISWET7.QPS");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(120.0);
    let res = solve_qp_with(&prob, &opts);
    assert_ne!(
        res.status,
        SolveStatus::Optimal,
        "LISWET7 返解 obj={:.10e} (viol 9997 行 × 9.87e-8, 弱双対下界 −2026.419109) \
         は Optimal を主張できない (infeasibility-mass witness が棄却する)",
        res.objective
    );
}

/// LISWET11 は certified 真値 −2476.226042863 に 1e-6 で到達する Optimal を返す。
///
/// **Sentinel (no-op-fail)**: witness のスコープ/床を狭めて正しい (F,1000) mint
/// (gap=2.237e-7, m̄=1.758e-12) を誤棄却すると Optimal が失われて FAIL する。
#[test]
#[ignore = "tier-2"]
fn liswet11_optimal_objective_matches_certified_truth() {
    // 独立オラクル確定値 (module doc の手順: HiGHS 生 QPS 直解 + 自前パーサ KKT)。
    const LISWET11_TRUTH: f64 = -2476.226042863;
    let prob = maros_qp("LISWET11.QPS");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(120.0);
    let res = solve_qp_with(&prob, &opts);
    assert_eq!(
        res.status,
        SolveStatus::Optimal,
        "LISWET11 は 1e-6 証明可能な正しい mint を持つ (got {:?}, obj={:.10e})",
        res.status,
        res.objective
    );
    let rel = (res.objective - LISWET11_TRUTH).abs() / LISWET11_TRUTH.abs().max(1.0);
    assert!(
        rel <= 1e-6,
        "LISWET11 obj={:.10e} は 独立オラクル真値 {:.9e} から rel={rel:.3e} (≤1e-6) \
         外れている (CSV −5.015e3 は不可能値: 返解が異常化していない実測)",
        res.objective,
        LISWET11_TRUTH
    );
}

fn maros_qp(name: &str) -> QpProblem {
    let path = maros_path(name);
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
        path.display()
    );
    parse_qps(&path).expect("parse")
}

fn maros_path(name: &str) -> std::path::PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("data/maros_meszaros").join(name)
}
