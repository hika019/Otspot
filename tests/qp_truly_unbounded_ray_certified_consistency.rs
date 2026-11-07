//! 真に非有界な QP が eps に依らず Optimal を名乗らない sentinel (2026-11)。
//!
//! ## 置き換え経緯 (既存テストの assert 編集禁止に対応)
//!
//! 本ファイルは tests/qp_truly_unbounded_eps_consistency.rs (削除済み) の
//! 置き換えである。旧ファイルは shared 証書フラグ `infeasibility_certified`
//! を assert していたが、証書フラグの ray / Farkas 分離 (PR #71 review 指摘)
//! で旧名は廃止され、旧 assert はコンパイル不能になった。AGENTS.md 実装規約
//! 「既存テストの assert・ロジックは変更しない (削除 + 新規追加で表現)」に
//! 従い、旧 3 ケースは削除 (git 履歴に残る) し、同一意味を新フラグ
//! `stats.unbounded_ray_certified` 上で読む新規ケースとして本ファイルに
//! 追加した。旧→新テスト名対応:
//!
//! - `truly_unbounded_delta_5e_8_never_optimal_cross_eps`
//!   → `truly_unbounded_delta_5e_8_never_optimal_ray_certified_cross_eps`
//! - `truly_unbounded_delta_eq_eps_is_deterministic_unbounded`
//!   → `truly_unbounded_delta_eq_eps_ray_certified_deterministic_unbounded`
//! - `truly_unbounded_delta_1e_9_never_optimal_cross_eps`
//!   → `truly_unbounded_delta_1e_9_never_optimal_ray_certified_cross_eps`
//!
//! ## 台帳 (手計算オラクル、ソルバ非依存)
//!
//! インスタンス U1: min (0, -delta)'x  s.t. x1 = 1 (Eq 1 行)、Q = 0、
//! x1, x2 とも自由境界。実行可能集合は F = {(1, t) : t ∈ R}。方向
//! dx = e2 は A·dx = 0、Q·dx = 0、c'dx = -delta < 0 を満たす退行方向
//! (recession ray) なので c'x(t) = -delta·t → -∞。**いかなる eps でも
//! 最適解は存在しない** — 有限 iterate 上での KKT 残差 (定数成分 δ) が
//! 要求 eps 以下という理由だけの Optimal は常にオラクル偽。
//!
//! ## 修正前の症状 (調査台帳 /tmp/opencode/eps_diverge_repro/REPORT.md)
//!
//! - delta = 5e-8: eps 1e-6 / 1e-7 で Optimal (収束判定が検出器より先に
//!   break するレース)、eps 1e-8 のみ検証済み Unbounded。
//! - delta = 1e-9: eps 1e-6, 1e-7, 1e-8 の全てで Optimal (distrust gate が
//!   検証前の commit を Stall に前借り → finalize が restored iterate に
//!   Optimal を打刻)。
//!
//! ## GREEN 基準 (修正後)
//!
//! 全 eps で `Unbounded` かつ `stats.unbounded_ray_certified == true`
//! (ray 側証書。ray / Farkas の独立性は tests/qp_cert_flag_split_sentinel.rs
//! が pin する)。eps-free な独立検証器 (verify_unbounded_ray) が唯一の審判
//! であり、eps 依存ヒューリスティック (distrust gate / 収束判定) は未証明の
//! 検出器主張にのみ作用する。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot::qp::ipm_solver::solve_ipm;
use otspot::qp::QpProblem;
use otspot::sparse::CscMatrix;

fn inf() -> f64 {
    f64::INFINITY
}

/// U1: min -delta*x2 s.t. x1 = 1 (Eq)、x1, x2 自由。手計算 ray dx = e2。
fn u1_truly_unbounded(delta: f64) -> QpProblem {
    QpProblem::new(
        CscMatrix::new(2, 2),
        vec![0.0, -delta],
        CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 2).unwrap(),
        vec![1.0],
        vec![(-inf(), inf()), (-inf(), inf())],
        vec![ConstraintType::Eq],
    )
    .unwrap()
}

fn solve_u1(delta: f64, eps: f64) -> SolverResult {
    let problem = u1_truly_unbounded(delta);
    let mut opts = SolverOptions::default();
    opts.ipm.eps = eps;
    opts.presolve = false;
    opts.timeout_secs = Some(4.0);
    solve_ipm(&problem, &opts)
}

/// delta = 5e-8: 収束判定 vs 検出器 commit のレース経路。修正前のコード
/// (3d2df0cf) では eps=1e-6 / 1e-7 で Optimal を返し FAIL。修正後の単独
/// revert では受理部 verify-first と ladder 守衛の二重防御のため GREEN のまま、
/// 両方同時に revert して FAIL (台帳は commit に届く全経路の審判順の修正を要求)。
#[test]
fn truly_unbounded_delta_5e_8_never_optimal_ray_certified_cross_eps() {
    for &eps in &[1e-6_f64, 1e-7, 1e-8] {
        let r = solve_u1(5e-8, eps);
        assert_ne!(
            r.status,
            SolveStatus::Optimal,
            "delta=5e-8 eps={eps:.0e}: Optimal is oracle-false (hand-proved ray \
             dx=e2: c'x(1,t) = -5e-8*t -> -inf for all eps)"
        );
        assert_eq!(
            r.status,
            SolveStatus::Unbounded,
            "delta=5e-8 eps={eps:.0e}: expected certified Unbounded, obj={:e} iters={}",
            r.objective,
            r.iterations
        );
        assert!(
            r.stats.unbounded_ray_certified,
            "delta=5e-8 eps={eps:.0e}: Unbounded must carry the ray certificate"
        );
        eprintln!(
            "[U1 5e-8] eps={eps:.0e} -> {:?} ray_certified={} iters={}",
            r.status, r.stats.unbounded_ray_certified, r.iterations
        );
    }
}

/// delta = 1e-7 = eps (台帳 §"Borderline non-determinism": 修正前は同一 solve が
/// Optimal / 検証済み Unbounded を交互に返したレース窓)。修正後は全試行で
/// 検証済み Unbounded に決定すること。
#[test]
fn truly_unbounded_delta_eq_eps_ray_certified_deterministic_unbounded() {
    for rep in 0..5 {
        let r = solve_u1(1e-7, 1e-7);
        assert_eq!(
            r.status,
            SolveStatus::Unbounded,
            "delta=eps race (rep {rep}): verdict must be deterministic \
             certified Unbounded, got {:?} obj={:e} iters={}",
            r.status,
            r.objective,
            r.iterations
        );
        assert!(
            r.stats.unbounded_ray_certified,
            "delta=eps race (rep {rep}): Unbounded must carry the ray certificate"
        );
    }
}

/// delta = 1e-9: mint 層 (ladder の prove_optimal 早期 break) が検証済み ray
/// 判定を打ち消す経路。`finite_mint_may_replace` 守衛の revert で全 eps レッグが
/// Optimal を返し FAIL (実測)。
#[test]
fn truly_unbounded_delta_1e_9_never_optimal_ray_certified_cross_eps() {
    for &eps in &[1e-6_f64, 1e-7, 1e-8] {
        let r = solve_u1(1e-9, eps);
        assert_ne!(
            r.status,
            SolveStatus::Optimal,
            "delta=1e-9 eps={eps:.0e}: Optimal is oracle-false (no KKT point \
             exists; stationarity residual is exactly delta on the free ray)"
        );
        assert_eq!(
            r.status,
            SolveStatus::Unbounded,
            "delta=1e-9 eps={eps:.0e}: expected certified Unbounded, obj={:e} iters={}",
            r.objective,
            r.iterations
        );
        assert!(
            r.stats.unbounded_ray_certified,
            "delta=1e-9 eps={eps:.0e}: Unbounded must carry the ray certificate"
        );
        eprintln!(
            "[U1 1e-9] eps={eps:.0e} -> {:?} ray_certified={} iters={}",
            r.status, r.stats.unbounded_ray_certified, r.iterations
        );
    }
}
