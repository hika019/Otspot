//! LISWET7 delta_matrix floor regression guard (bug-frontier 2026-08-02)
//!
//! `rho_matrix`/`delta_matrix` (otspot-core/src/qp/ipm_core/ippmm/iter.rs) は
//! `pmm.rho`/`pmm.delta` を反復ごとに使うが、旧実装はここに固定
//! `options.ipm.delta_min` (DEFAULT_IPM_DELTA_MIN=1e-8、DEFAULT_IPM_EPS=1e-6 用に
//! 校正) を `.max()` で課していた。eps=1e-8 では reg_limit が REG_LIMIT_MIN
//! (1e-14) まで下がっても行列側の正則化が 1e-8 に恒久的に張り付き、近縮退
//! 制約行 (LISWET7 の二階差分行 [-0.5,1.0,-0.5] 等) の primal residual が
//! delta_matrix=1e-8 による `dy_i×delta_matrix=r_p_i` の恒等式で 6.216e-8 に
//! 凍結する副作用があった (実測: r_p_i=-6.216e-8, dy_i=6.216, delta_matrix=1e-8
//! で厳密一致、TwoFloat DD 残差評価でも同一値 → 評価精度でなく正則化設計が真因)。
//!
//! 固定 delta_min を撤去し、REG_LIMIT_MIN に連動する reg_limit 経路一本化に
//! よってこの床が解消することを確認する。
//!
//! Sentinel: `rho_matrix = pmm.rho.max(options.ipm.delta_min)` /
//! `delta_matrix = pmm.delta.max(options.ipm.delta_min)` に revert すると
//! pfn は旧床 (bench 実測 9.8e-8) 近辺に張り付き、本テストの閾値
//! (5e-8, 旧床の約半分) を割れず fail する。

use otspot::io::qps::parse_qps;
use otspot::options::{IpmOptions, SolverOptions, Tolerance};
use otspot::problem::ConstraintType;
use otspot::qp::{solve_qp_with, QpProblem};
use std::path::Path;

/// bench (`compute_pfeas_normalized`) と同型の componentwise primal feasibility:
///   `max_i violation_i / (1 + |Ax_i| + |b_i|)`
fn pfeas_normalized(prob: &QpProblem, x: &[f64]) -> f64 {
    if prob.num_constraints == 0 || x.len() != prob.num_vars {
        return f64::NAN;
    }
    let ax = prob.a.mat_vec_mul(x).expect("Ax must succeed");
    let mut max_rel = 0.0_f64;
    for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
        let viol = match prob.constraint_types[i] {
            ConstraintType::Eq => (ax_i - b_i).abs(),
            ConstraintType::Ge => (b_i - ax_i).max(0.0),
            _ => (ax_i - b_i).max(0.0), // Le or future variants
        };
        let scale_i = 1.0 + ax_i.abs() + b_i.abs();
        let rel_i = viol / scale_i;
        if rel_i > max_rel {
            max_rel = rel_i;
        }
    }
    max_rel
}

fn maros_path(name: &str) -> std::path::PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("data/maros_meszaros").join(name)
}

fn solve_with_eps(prob: &QpProblem, user_eps: f64, timeout_secs: f64) -> otspot::SolverResult {
    let mut opts = SolverOptions::default();
    opts.tolerance = Some(Tolerance::Custom(user_eps));
    opts.ipm = IpmOptions {
        eps: user_eps,
        ..IpmOptions::default()
    };
    opts.timeout_secs = Some(timeout_secs);
    solve_qp_with(prob, &opts)
}

/// fix 前: pfn は 9.8e-8 (bench 実測、以下 5e-8 を割れず fail) に張り付く。
/// fix 後: reg_limit が REG_LIMIT_MIN=1e-14 まで正しく行列に反映され、
/// 500 iter 以内に pfn は 6.216e-8 の旧床を明確に下回る (実測 2.7e-8 で
/// なお減少中)。5e-8 は旧床 9.8e-8 の約半分、実測 2.7e-8 に約2倍の余裕を残す。
#[test]
fn liswet7_pfn_breaks_delta_matrix_floor_at_eps_1e8() {
    let path = maros_path("LISWET7.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
        path.display()
    );
    let prob = parse_qps(&path).expect("parse LISWET7");
    let result = solve_with_eps(&prob, 1e-8, 60.0);
    assert_eq!(
        result.solution.len(),
        prob.num_vars,
        "LISWET7 eps=1e-8 must return an original-space diagnostic iterate, got status={:?}",
        result.status
    );
    let pfn = pfeas_normalized(&prob, &result.solution);
    assert!(
        pfn < 5e-8,
        "LISWET7 eps=1e-8 pfn={:.3e} status={:?} must break the delta_matrix=delta_min(1e-8) \
         floor (旧床 9.8e-8 実測; delta_matrix が options.ipm.delta_min に revert すると \
         この閾値を割れず fail する)",
        pfn,
        result.status
    );
}

/// 退行防止: eps=1e-6 (デフォルト) では fix 前後で挙動が変わらないこと。
/// delta_min=1e-8 は元々 eps=1e-6 に対し 100倍マージンがあったため、fix 前でも
/// この eps では床に到達しなかった (LISWET7 は eps=1e-6 で正常収束する)。
#[test]
fn liswet7_pfn_unaffected_at_default_eps_1e6() {
    let path = maros_path("LISWET7.QPS");
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
        path.display()
    );
    let prob = parse_qps(&path).expect("parse LISWET7");
    let result = solve_with_eps(&prob, 1e-6, 60.0);
    let pfn = pfeas_normalized(&prob, &result.solution);
    assert!(
        pfn < 1e-6,
        "LISWET7 eps=1e-6 pfn={:.3e} status={:?} must stay within user_eps (no regression \
         expected at default eps — delta_matrix floor removal must not affect loose eps)",
        pfn,
        result.status
    );
}
