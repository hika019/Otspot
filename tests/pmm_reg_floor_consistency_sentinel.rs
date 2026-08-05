//! KKT 行列正則化床の適用範囲に関する退化ガード (bug-frontier 2026-08-05)
//!
//! 床 (`KKT_PIVOT_FLOOR` = 1e-8) は行列側 (`rho_matrix` / `delta_matrix`) にだけ
//! 掛かり、PMM 残差 (`r_d − ρ(x−x_ref)` / `r_p − δ(y−y_ref)`) には掛からない。
//! この非対称は LP では長年の実運用で検証済みだが、QP へ広げると解いている系が
//! 「prox 中心がほぼ現在点」の別の PMM 部分問題にすり替わり、ステップ後の残差が
//! `ρ_matrix · ‖dx‖_∞` に張り付く。
//!
//! 1babe8bc は床のゲートを Gershgorin の `λ_min(Q)` 下界で行っていたため、
//! Q の対角にゼロ行が 1 つでもあれば QP にも満額の床が掛かっていた
//! (Maros-Meszaros 138 問中 106 問)。実測 (QGFRDXPN @ eps=1e-6, jobs=1):
//! `nr_d` が `1e-8 × ‖dx‖_∞ = 1e-8 × 11.83 = 1.183e-7` で iter 132〜182 の
//! 50 反復ぶん凍結し residual_stall → STALLED / pfn=1.3e-5。
//!
//! 本 file の 2 test は revert 感度が異なる。実測 (eps=1e-6, timeout 既定):
//!
//! | revert 対象 | QGFRDXPN | QSHELL |
//! |---|---|---|
//! | A: 床ゲートを Gershgorin 下界へ戻す | **FAIL** Stalled pfn=1.334e-5 | PASS 19.76s |
//! | B: `charged_iterations` を granted 課金へ戻す | PASS | PASS 18.89s |
//! | A + B 同時 | **FAIL** | **FAIL** 19.90s Stalled pfn=1.455e-11 |
//!
//! すなわち QGFRDXPN 側は A 単独の sentinel、QSHELL 側は **A と B の同時 revert
//! でのみ落ちる連言 sentinel** である (どちらか一方が直っていれば QSHELL は
//! 収束経路に戻る)。B 単独の sentinel は `attempt.rs` の
//! `failed_extension_keeps_no_presolve_fallback_reachable` が担う。

use otspot::io::qps::parse_qps;
use otspot::options::{IpmOptions, SolverOptions, Tolerance};
use otspot::problem::{ConstraintType, SolveStatus};
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
            _ => (ax_i - b_i).max(0.0),
        };
        let rel_i = viol / (1.0 + ax_i.abs() + b_i.abs());
        if rel_i > max_rel {
            max_rel = rel_i;
        }
    }
    max_rel
}

fn maros_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/maros_meszaros")
        .join(name)
}

fn solve_maros(name: &str, user_eps: f64, timeout_secs: f64) -> (QpProblem, otspot::SolverResult) {
    let path = maros_path(name);
    assert!(
        path.exists(),
        "{} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
        path.display()
    );
    let prob = parse_qps(&path).unwrap_or_else(|e| panic!("parse {name}: {e:?}"));
    let mut opts = SolverOptions::default();
    opts.tolerance = Some(Tolerance::Custom(user_eps));
    opts.ipm = IpmOptions {
        eps: user_eps,
        ..IpmOptions::default()
    };
    opts.timeout_secs = Some(timeout_secs);
    let result = solve_qp_with(&prob, &opts);
    (prob, result)
}

/// QP へ床が掛かると `nr_d` が `ρ_matrix·‖dx‖` に凍結して STALLED になる。
/// 床を LP 経路に限定すれば収束する。
#[test]
fn qgfrdxpn_converges_when_the_pivot_floor_is_limited_to_lp() {
    let (prob, result) = solve_maros("QGFRDXPN.QPS", 1e-6, 120.0);
    let pfn = pfeas_normalized(&prob, &result.solution);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "QGFRDXPN eps=1e-6 must converge; pfn={pfn:.3e} (床を Gershgorin ゲートへ \
         revert すると nr_d が 1e-8·‖dx‖=1.183e-7 に凍結し STALLED/pfn=1.3e-5 になる)"
    );
    assert!(
        pfn < 1e-6,
        "QGFRDXPN pfn={pfn:.3e} must be within user_eps=1e-6 (実測 6.1e-12)"
    );
}

/// 失敗した延長 attempt が iter 予算を食い潰すと、QSHELL を実際に解いていた
/// no-presolve fallback が消えて STALLED に退化する
/// (`charged_iterations` の unit sentinel と対になる live 再現)。
#[test]
fn qshell_reaches_optimal_through_the_no_presolve_fallback() {
    let (prob, result) = solve_maros("QSHELL.QPS", 1e-6, 300.0);
    let pfn = pfeas_normalized(&prob, &result.solution);
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "QSHELL eps=1e-6 must converge; pfn={pfn:.3e} (床ゲートと延長 attempt の \
         課金を同時に revert すると 3 attempt が bit 同一 IterationLimit になり \
         延長 attempt が予算を使い切って STALLED iters=1099 になる。片方だけの \
         revert では落ちない — module doc の表を参照)"
    );
    assert!(
        pfn < 1e-6,
        "QSHELL pfn={pfn:.3e} must be within user_eps=1e-6 (実測 5.8e-11)"
    );
}
