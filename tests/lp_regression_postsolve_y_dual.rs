//! 回帰テスト: e61f27b "presolve に R6/R15/R5 追加" が壊した postsolve dual 復元
//!
//! ## 真因 (bisect 2026-05-17 by bisecter agent)
//!
//! commit e61f27b で `src/presolve/postsolve.rs:111` の reduced_cost 計算を
//!   旧: `vec![0.0; n]` + col_map で reduced 空間 rc を展開 (削除変数=0)
//!   新: `c.clone()` から `c[j] - Σ A_ij * y_i` を全変数で再計算
//! に変更したが、`y_i` (dual_solution[row]) は postsolve すべての step で
//! KKT 整合に復元されている前提に立つ。e61f27b は LinearSubstitution の y
//! のみ復元を追加し、RedundantCons / SingletonRow / 既存 transform 群の y
//! 復元の網羅性は保証されていない。
//!
//! 結果: 一部の行で y=0 のまま残り、その行を持つ列 j の rc は
//!   rc[j] = c[j] - Σ A_ij * y_i ≈ c[j] (大きく非ゼロ)
//! となり、bench `compute_dfeas_orig` の `viol = max(0, -rc)` で巨大な
//! dual residual が出る。
//!
//! ## 観測 (perold 1376×625, eps=1e-6, concurrent solver, cargo clean+build):
//! - ae81dea (e61f27b の親): obj=-9.38e3 pf=2.9e-8 **df=5.8e-11** PASS
//! - e61f27b              : obj=-9.38e3 pf=1.2e-8 **df=5.2e2 dfr=1.0** DFEAS_FAIL
//! - HEAD (4a1e305)       : obj=-9.38e3 pf=2.2e-10 **df=3.1e1 dfr=9.7e-1** DFEAS_FAIL
//!   (1269322 faer LU で solve 速度が 50s → 2.3s に短縮、bug は同じ)
//!
//! ## このテストの目的
//! - HEAD で FAIL、e61f27b 修正後で PASS することを保証する回帰防壁。
//! - bench `src/bin/qps_benchmark.rs::compute_dfeas_orig` と同型の dfeas_rel を
//!   直接 Rust API で計算するため bench script (stale binary bug あり) に依存しない。
//! - 実行時間: HEAD で perold ≈ 2-3s (fast-fail)、修正後でも < 60s (TDD 3 分制約内)。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::{solve_qp_with, QpProblem};
use std::path::Path;

/// bench `compute_dfeas_orig` と同型: LP/QP の dual feasibility 相対残差。
///
/// returns `(dfeas_abs, dfeas_rel)`:
///   dfeas_abs = max_j max(0, -rc_j)
///   dfeas_rel = max_j max(0, -rc_j) / (1 + |rc_j| + |c_j|)
///
/// 注: bound_duals が空の LP simplex 経路を想定した formula (b8de691 〜 c69959d
/// 直前の bench 判定式)。c69959d 以降は at_lb/at_ub 場合分けで緩和されているが、
/// perold の DFEAS_FAIL は緩和後 (新 judge) でも残るほど巨大なので、ここでは
/// 旧 formula で十分検出可能。
fn dfeas_abs_rel(prob: &QpProblem, rc: &[f64]) -> (f64, f64) {
    let n = prob.c.len().min(rc.len());
    let mut dfeas_abs = 0.0_f64;
    let mut dfeas_rel = 0.0_f64;
    for j in 0..n {
        let r = rc[j];
        let viol = f64::max(0.0, -r);
        dfeas_abs = dfeas_abs.max(viol);
        let scale = 1.0 + r.abs() + prob.c[j].abs();
        dfeas_rel = dfeas_rel.max(viol / scale);
    }
    (dfeas_abs, dfeas_rel)
}

/// presolve postsolve が y を全行 KKT 整合に復元できているか確認する。
///
/// `c - A^T y - rc = 0` (LP の dual 最適性) を最大絶対残差で測る。
/// e61f27b の rc 計算 `rc = c - A^T y` は機械的に kkt_residual = 0 を満たすが、
/// その y 自体が破綻していると `c - A^T y` (= rc) が dual feasibility を破る。
fn kkt_residual_max(prob: &QpProblem, y: &[f64], rc: &[f64]) -> f64 {
    let n = prob.c.len();
    let mut max_diff = 0.0_f64;
    for j in 0..n {
        let mut ct_y = prob.c[j];
        if let Ok((rows, vals)) = prob.a.get_column(j) {
            for k in 0..rows.len() {
                ct_y -= vals[k] * y[rows[k]];
            }
        }
        let d = ct_y - rc[j];
        if d.abs() > max_diff {
            max_diff = d.abs();
        }
    }
    max_diff
}

fn solve_perold() -> (QpProblem, solver::problem::SolverResult) {
    let path = Path::new("data/lp_problems/perold.QPS");
    if !path.exists() {
        panic!(
            "data/lp_problems/perold.QPS not found at {}; \
             scripts/netlib_lp_download.sh で取得すること",
            path.display()
        );
    }
    let prob = parse_qps(path).expect("parse perold");
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(180.0); // HEAD で fast-fail (3s)、good で ~50s
    let result = solve_qp_with(&prob, &opts);
    (prob, result)
}

/// 回帰本体: perold が Optimal を返し obj 一致 + dual feasibility が eps を満たす。
///
/// 期待値:
///   status = Optimal
///   obj ≈ -9.38e3 (Netlib 公式 -9.3807580773e+03)
///   dfeas_rel < 1e-6 (bench eps=1e-6 と一致)
///
/// HEAD (e61f27b 以降) で FAIL: dfeas_rel ≈ 0.97
/// 修正後で PASS: dfeas_rel ≈ 5e-11 (a1d42b1 / ae81dea で観測した水準)
#[test]
fn perold_postsolve_dual_feasibility_regression() {
    let (prob, r) = solve_perold();
    let (df_abs, df_rel) = dfeas_abs_rel(&prob, &r.reduced_costs);
    let kkt = kkt_residual_max(&prob, &r.dual_solution, &r.reduced_costs);

    eprintln!(
        "perold: status={:?} obj={:.4e} df_abs={:.2e} df_rel={:.2e} kkt_resid={:.2e}",
        r.status, r.objective, df_abs, df_rel, kkt
    );

    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "perold: status must be Optimal, got {:?}",
        r.status
    );
    let obj_expected = -9.3807580773e3;
    let obj_err = (r.objective - obj_expected).abs() / obj_expected.abs().max(1.0);
    assert!(
        obj_err < 1e-4,
        "perold: obj={:.6e} expected {:.6e} rel_err={:.2e}",
        r.objective, obj_expected, obj_err
    );
    assert!(
        df_rel < 1e-6,
        "perold: dfeas_rel={:.3e} must be < 1e-6 (df_abs={:.2e} kkt_resid={:.2e}). \
         e61f27b で壊れた reduced_cost = c - A^T y の y 復元網羅性を疑え。",
        df_rel, df_abs, kkt
    );
}

/// presolve OFF の場合は postsolve dual 復元経路を通らないため必ず PASS する。
///
/// 真因が postsolve y 復元にあることを 1 テストで切り分ける TDD assertion。
/// HEAD で presolve=false → perold PASS、presolve=true → DFEAS_FAIL なら
/// 真因は完全に postsolve 経路に局在する。
#[test]
fn perold_presolve_off_passes() {
    let path = Path::new("data/lp_problems/perold.QPS");
    if !path.exists() {
        eprintln!("[SKIP] perold.QPS not found");
        return;
    }
    let prob = parse_qps(path).expect("parse perold");
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(180.0);
    let r = solve_qp_with(&prob, &opts);
    let (df_abs, df_rel) = dfeas_abs_rel(&prob, &r.reduced_costs);

    eprintln!(
        "perold[presolve=off]: status={:?} obj={:.4e} df_abs={:.2e} df_rel={:.2e}",
        r.status, r.objective, df_abs, df_rel
    );

    assert!(
        matches!(r.status, SolveStatus::Optimal),
        "perold[presolve=off]: status must be Optimal, got {:?}",
        r.status
    );
    // presolve OFF なら e61f27b の経路を通らない → df_rel < 1e-6 が必須。
    // 万一ここも FAIL なら simplex 側にも別バグあり。
    assert!(
        df_rel < 1e-6,
        "perold[presolve=off]: dfeas_rel={:.3e} → simplex 単体に別バグの疑い",
        df_rel
    );
}
