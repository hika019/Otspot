//! maros STALLED 10 問の停止ゲート内訳 diag (bug-frontier 2026-10-18)。
//!
//! bench (eps=1e-6, timeout=1000s, jobs=6) で STALLED になる 10 問の bench 出力
//! pf (pfn) は 1.1e-12..3.9e-7 と小さく、primal 可行性が共通ボトルネックでは
//! ない。`otspot::qp::ipm_solver::kkt` の 5 残差関数 (sol に公開済み) を用いて
//! 返却 iterate の元空間残差を per-problem に測り、`IpmOutcome::satisfies_eps`
//! のどのゲートで user_eps=1e-6 を割れないかを確定する:
//!   kkt_residual_rel / primal_residual_rel / bound_violation /
//!   complementarity (rel と componentwise の max) / duality gap。
//! satisfaction 閾値は kkt/pres/bv/comp <= 1e-6 (gap は < 0.1, prove_optimal で
//! さらに <= user_eps)。双対目的の式は core/duality_gap.rs と同型 (pub(crate) の
//! ため diag 側で再掲; 変更時は本式と core 両方を更新すること)。

#![allow(clippy::field_reassign_with_default)]

use otspot::io::qps::parse_qps;
use otspot::options::{IpmOptions, SolverOptions, Tolerance};
use otspot::qp::ipm_solver::kkt::{
    bound_violation, complementarity_componentwise_rel, complementarity_residual_rel,
    kkt_residual_rel, primal_residual_rel,
};
use otspot::qp::ipm_solver::outcome::ProblemView;
use otspot::qp::solve_qp_with;
use otspot::QpProblem;

/// core.rs `compute_duality_gap_rel` と同型の双対ギャップ (元空間)。
/// QP 双対目的: dual_obj = -1/2 x'Qx - b'y + lb'z_lb - ub'z_ub。
/// FX (lb=ub) は停留性 qx+c+aty で解析置換 (core 側と同一)。
///
/// FX 判定閾値: otspot-core tolerances.rs の FX_TOL (1e-12) と同一値。
/// core 側の FX_TOL は pub(crate) のため diag 側で同値の const を置く。
const FX_TOL: f64 = 1e-12;
fn diag_duality_gap_rel(prob: &QpProblem, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    let n = prob.num_vars;
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    let aty = if prob.a.nrows() > 0 && !y.is_empty() {
        prob.a.transpose().mat_vec_mul(y).expect("A'y")
    } else {
        vec![0.0_f64; n]
    };
    let xqx = qx.iter().zip(x).map(|(&a, &b)| a * b).sum::<f64>();
    let cx = prob.c.iter().zip(x).map(|(&a, &b)| a * b).sum::<f64>();
    let primal_obj = 0.5 * xqx + cx + prob.obj_offset;

    let mut by = 0.0_f64;
    for (&bi, &yi) in prob.b.iter().zip(y.iter()) {
        by += bi * yi;
    }

    // FX (lb=ub) は postsolve で z_lb,z_ub が 0 埋めされるため、
    // val * (z_lb - z_ub) = val * (qx + c + aty) で解析的に置換 (core 同型)。
    let mut bnd_term = 0.0_f64;
    let mut lb_idx = 0_usize;
    let mut ub_idx = prob
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    for (j, &(lb, ub)) in prob.bounds.iter().enumerate() {
        let lb_finite = lb.is_finite();
        let ub_finite = ub.is_finite();
        let is_fx = lb_finite && ub_finite && (lb - ub).abs() < FX_TOL;
        if is_fx {
            let stat_no_bnd = qx[j] + prob.c[j] + aty[j];
            bnd_term += lb * stat_no_bnd;
            if lb_finite {
                lb_idx += 1;
            }
            if ub_finite {
                ub_idx += 1;
            }
        } else {
            if lb_finite {
                if lb_idx >= z.len() {
                    return f64::INFINITY;
                }
                bnd_term += lb * z[lb_idx];
                lb_idx += 1;
            }
            if ub_finite {
                if ub_idx >= z.len() {
                    return f64::INFINITY;
                }
                bnd_term -= ub * z[ub_idx];
                ub_idx += 1;
            }
        }
    }
    let dual_obj = -0.5 * xqx - by + bnd_term + prob.obj_offset;
    let gap_abs = (primal_obj - dual_obj).abs();
    let denom = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    if denom > 0.0 && gap_abs.is_finite() {
        gap_abs / denom
    } else {
        f64::INFINITY
    }
}

/// 1 問分の残差内訳を出力する (bench 条件同一: eps=1e-6, timeout 1000s)。
/// 1 問分の残差内訳を出力する。
/// `user_eps`/`ruiz_override`/`timeout` を変えて比較できる (eps ladder / ruiz off)。
/// 既定 (1e-6, None, 1000.0) が bench 条件。
fn report_variant(name: &str, user_eps: f64, ruiz_override: Option<bool>, timeout: f64) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{name}.QPS"));
    assert!(path.exists(), "{path:?} not found");
    let prob = parse_qps(&path).expect("parse failed");

    let mut opts = SolverOptions::default();
    opts.tolerance = Some(Tolerance::Custom(user_eps));
    opts.ipm = IpmOptions {
        eps: user_eps,
        ..IpmOptions::default()
    };
    if let Some(use_ruiz) = ruiz_override {
        opts.use_ruiz_scaling = use_ruiz;
    }
    opts.timeout_secs = Some(timeout);
    let t0 = std::time::Instant::now();
    let r = solve_qp_with(&prob, &opts);
    let dt = t0.elapsed().as_secs_f64();

    let view = ProblemView {
        q: &prob.q,
        a: &prob.a,
        c: &prob.c,
        b: &prob.b,
        bounds: &prob.bounds,
        constraint_types: &prob.constraint_types,
        eliminated_cols: &vec![false; prob.num_vars],
    };
    let x = r.solution.as_slice();
    let y = r.dual_solution.as_slice();
    let z = r.bound_duals.as_slice();

    let kkt = kkt_residual_rel(&view, x, y, z);
    let pres = primal_residual_rel(&view, x);
    let bv = bound_violation(&prob.bounds, x);
    let comp = complementarity_residual_rel(&view, x, y, z)
        .max(complementarity_componentwise_rel(&view, x, y, z));
    let gap = diag_duality_gap_rel(&prob, x, y, z);

    // satisfies_eps (eps=1e-6) のどの項で落ちるか (gap は PROMOTION_GAP_TOL=0.1)。
    let mut fails: Vec<&str> = Vec::new();
    if kkt > 1e-6 {
        fails.push("kkt");
    }
    if pres > 1e-6 {
        fails.push("pres");
    }
    if bv > 1e-6 {
        fails.push("bv");
    }
    if comp > 1e-6 {
        fails.push("comp");
    }
    if gap >= 0.1 {
        fails.push("gap>=0.1");
    }
    let gate = if fails.is_empty() {
        "ALL_PASS (satisfies_eps 通過 → prove_optimal 側の問題)"
    } else {
        &fails.join(",")
    };

    let ruiz_tag = if opts.use_ruiz_scaling {
        "ruiz-on"
    } else {
        "ruiz-OFF"
    };
    eprintln!(
        "{:10} {:8} eps={:6.0e} {:16} iters={:>4} obj={:>15.8e} kkt={:.3e} pres={:.3e} bv={:.3e} comp={:.3e} gap={:.3e} total_s={:>6.1} fails=[{}]",
        name,
        ruiz_tag,
        user_eps,
        format!("{:?}", r.status),
        r.iterations,
        r.objective,
        kkt,
        pres,
        bv,
        comp,
        gap,
        dt,
        gate
    );
}

macro_rules! stalled_diag {
    ($fn_name:ident, $name:literal) => {
        #[test]
        #[ignore = "diag (maros STALLED 問の 5 残差内訳; 観測用であり失敗しない)"]
        fn $fn_name() {
            report_variant($name, 1e-6, None, 1000.0);
        }
    };
}

stalled_diag!(maros_stalled_liswet1, "LISWET1");
// LISWET7 は STALLED ではないが、ladder 構成変更の regression 監視対象
// (loose lane (base/10, 1) を証明に使う。lane death で skip されると転落する)。
stalled_diag!(maros_stalled_liswet7, "LISWET7");
stalled_diag!(maros_stalled_liswet8, "LISWET8");
stalled_diag!(maros_stalled_liswet9, "LISWET9");
stalled_diag!(maros_stalled_liswet10, "LISWET10");
stalled_diag!(maros_stalled_liswet12, "LISWET12");
stalled_diag!(maros_stalled_aug2dcqp, "AUG2DCQP");
stalled_diag!(maros_stalled_qpcboei2, "QPCBOEI2");
stalled_diag!(maros_stalled_stadat1, "STADAT1");
stalled_diag!(maros_stalled_values, "VALUES");
stalled_diag!(maros_stalled_yao, "YAO");

// ---------------------------------------------------------------------------
// eps ladder + ruiz-off 比較 diag (2026-10-18)。
//
// 問 1: comp 残差は user_eps に追随して下がるか、それとも固定床で凍結するか。
//  追随   → inner scaled eps 収束が早すぎる系 (tighten 深さの問題)
//  凍結   → unscale/正則化由来の精度床 (reg_limit / delta floor 系)
//
// 問 2: LISWET8/9/10/12 の kkt/gap 崩壊 (kkt≈0.9, gap≈0.36) が ruiz lane
//   (unscale の双対復元) 由来か。ruiz off で kkt/gap が正常化するなら、
//   崩壊点は ruiz unscale ないし ruiz 空間の IPM 探索にある。
//
// timeout は 1 問あたり 120s 区切り (diag 用; bench の 1000s ではない)。
// ---------------------------------------------------------------------------

/// comp 追随チェック: bench 時間の短い comp-only 3 問 (A组代表)。
#[test]
#[ignore = "diag (comp の eps 追随チェック; 観測用)"]
fn maros_eps_ladder_fast() {
    const NAMES: [&str; 3] = ["QPCBOEI2", "AUG2DCQP", "YAO"];
    const EPS: [f64; 3] = [1e-6, 1e-8, 1e-10];
    for name in NAMES {
        for &eps in &EPS {
            report_variant(name, eps, None, 120.0);
        }
    }
}

/// LISWET 家族 off-lane の kkt 崩壊代表 1 問: eps × ruiz on/off。
#[test]
#[ignore = "diag (LISWET8 の ruiz on/off 比較; 観測用)"]
fn maros_ruiz_liswet8() {
    const EPS: [f64; 2] = [1e-6, 1e-8];
    for &eps in &EPS {
        for &ruiz in &[true, false] {
            report_variant("LISWET8", eps, Some(ruiz), 120.0);
        }
    }
}

/// 双対崩壊 3 問 (kkt pristine + gap 大) の ruiz on/off。
#[test]
#[ignore = "diag (LISWET9/10/12 の ruiz on/off 比較; 観測用)"]
fn maros_ruiz_liswet9_10_12() {
    const NAMES: [&str; 3] = ["LISWET9", "LISWET10", "LISWET12"];
    const EPS: [f64; 2] = [1e-6, 1e-8];
    for name in NAMES {
        for &eps in &EPS {
            for &ruiz in &[true, false] {
                report_variant(name, eps, Some(ruiz), 120.0);
            }
        }
    }
}

/// kkt-only 系 (VALUES, m=1, iters=210) の eps × ruiz on/off。
#[test]
#[ignore = "diag (VALUES の eps ラダーと ruiz on/off; 観測用)"]
fn maros_eps_ladder_values() {
    const EPS: [f64; 3] = [1e-6, 1e-8, 1e-10];
    for &eps in &EPS {
        for &ruiz in &[true, false] {
            report_variant("VALUES", eps, Some(ruiz), 120.0);
        }
    }
}

/// Ruiz スケールの実測 (solve 無し・一瞬)。
///
/// unscale 双対は y_orig = e_i·y_s / c、eps は inner 側で eps/amplification に締められ
/// IPM_EPS_NOISE_FLOOR の床と衝突しうる。e_i の極値は dual 崩壊/精度床の候補行を示す。
#[test]
#[ignore = "diag (Ruiz スケール範囲の計測; 観測用)"]
fn maros_ruiz_scale_profile() {
    use otspot::num::linalg::ruiz::RuizScaler;

    const NAMES: [&str; 10] = [
        "LISWET1", "LISWET8", "LISWET9", "LISWET10", "LISWET12", "AUG2DCQP", "QPCBOEI2", "STADAT1",
        "VALUES", "YAO",
    ];

    eprintln!(
        "{:10} {:6} {:12} {:12} {:12} {:12} {:12} {:10}",
        "problem", "n/m", "e_min", "e_max", "d_min", "d_max", "c", "amplif"
    );

    for name in NAMES {
        let path = std::path::PathBuf::from(format!("data/maros_meszaros/{name}.QPS"));
        let prob = parse_qps(&path).expect("parse failed");
        let n = prob.num_vars;
        let m = prob.num_constraints;
        let mut scaler = RuizScaler::new(n, m);
        scaler.compute_with_rhs(&prob.q, &prob.a, &prob.c, &[]);
        let e_min = scaler.e.iter().cloned().fold(f64::INFINITY, f64::min);
        let e_max = scaler.e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let d_min = scaler.d.iter().cloned().fold(f64::INFINITY, f64::min);
        let d_max = scaler.d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "{:10} {:3}/{:<3} {:12.3e} {:12.3e} {:12.3e} {:12.3e} {:12.3e} {:10.3e}",
            name,
            n,
            m,
            e_min,
            e_max,
            d_min,
            d_max,
            scaler.c,
            1f64.max((1.0 / e_min).max(1.0 / (scaler.c * d_min)))
        );
    }
}
