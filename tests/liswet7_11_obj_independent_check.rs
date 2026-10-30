//! LISWET11 の目的値照合。参照値は原 QPS (目的定数 2525.75 込み) を HiGHS で解き、
//! `scripts/highs_baseline_truth.py` で弱双対検算した値 (92ce6cc2 Era の HiGHS 検算値。
//! 現行 baseline CSV は exact_rational_kkt 認証値で絶対差 3.2e-10 — ベンチ tol 1e-6 内
//! だが同値ではない。docs/baseline-certificate-contract.md 参照)。
//!
//! Clarabel (tol=1e-12) は LISWET11 で AlmostSolved 36.527 (参照値比 26% 誤差) に止まり、
//! 独立オラクルに使えない。旧 .mat 変換データ (c が 1 ulp ずれ、目的定数なし) では ours が
//! Optimal を返したが、原データでは吸引点 (≈ 25.19) で Stalled になる。このテストは
//! 偽 Optimal (吸引点での収束宣言) を検出する番兵で、Optimal を主張した場合に限り
//! 目的値が参照値に一致することを要求する。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::qp::solve_qp_with;
use otspot::SolveStatus;

/// 目的定数込みの検算済み最適値。
const LISWET11_TRUTH_WITH_CONSTANT: f64 = 4.952395713834e1;
/// bench の目的値照合と同じ相対許容 (eps=1e-6)。
const OBJ_REL_TOL: f64 = 1e-6;
/// 実測 20.5 s (release, 他プロセス並走下) で Stalled 終了する。その約 3 倍の余裕を取りつつ
/// 1 テスト 3 分以内の上限に収まる値。
const TIMEOUT_SECS: f64 = 60.0;

#[test]
fn liswet11_optimal_claim_matches_certified_value() {
    let path = std::path::PathBuf::from("data/maros_meszaros/LISWET11.QPS");
    assert!(
        path.exists(),
        "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
        path
    );
    let prob = parse_qps(&path).expect("parse");
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_SECS);
    let ours = solve_qp_with(&prob, &opts);
    let rel = (ours.objective - LISWET11_TRUTH_WITH_CONSTANT).abs()
        / LISWET11_TRUTH_WITH_CONSTANT.abs().max(1.0);
    eprintln!(
        "LISWET11: status={:?} obj={:.10e} truth={LISWET11_TRUTH_WITH_CONSTANT:.10e} rel={rel:.3e}",
        ours.status, ours.objective
    );
    if ours.status == SolveStatus::Optimal {
        assert!(
            rel <= OBJ_REL_TOL,
            "LISWET11: Optimal obj={:.10e} but certified optimum is \
             {LISWET11_TRUTH_WITH_CONSTANT:.10e} (rel={rel:.3e})",
            ours.objective
        );
    }
}
