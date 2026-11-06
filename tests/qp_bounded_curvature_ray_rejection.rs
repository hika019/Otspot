//! 有界 PD QP の偽 certified Unbounded sentinel (Codex PR #71 P1、2026-11)。
//!
//! ## 台帳 (手計算オラクル、ソルバ非依存)
//!
//! インスタンス P1: min 0.5·xᵀQx + cᵀx、Q = [[1,-1],[-1,1+1e-9]]、
//! c = (-1e-7,-1e-7)、s.t. x1 - x2 = 0 (Eq)、x は自由境界。制約より実行可能
//! 集合は直線 x = (s, s)。代入で
//!
//! ```text
//! xᵀQx = s² - s² - s² + (1+1e-9)·s² = 1e-9·s²   (展開による厳密等式)
//! obj(s) = 5e-10·s² - 2e-7·s
//! ```
//!
//! 下に凸二次の唯一の最小点は s* = 2e-7 / 1e-9 = 200、
//! obj* = 5e-10·40000 - 2e-7·200 = 2e-5 - 4e-5 = **-2e-5**。検算:
//! obj(200) = -2e-5、obj(1000) = +3e-4 > obj(0) = 0。s→±∞ で obj → +∞ なので
//! **問題は有界** — いかなる Unbounded / Infeasible もオラクル偽
//! (x=(0,0) が実行可能)。
//!
//! ## 修正前の症状 (台帳 /tmp/opencode/codex_p1_probe/REPORT.md)
//!
//! 方向 dx=(1,1) は Q·dx = (0, 1.0e-9) が平坦性 tol 1e-8·‖Q_i‖∞·‖dx‖₁ = 2e-8
//! を通り、cᵀdx = -2e-7 < 0 と退行実行可能性も満たすため、非正曲率検査のない
//! `verify_unbounded_ray` が ray と誤受理し、presolve × eps の 6 構成すべてで
//! certified Unbounded を返した。曲率検査 (条件 5) の revert で本テストは FAIL。
//!
//! ## GREEN 基準 (修正後、実測で pin)
//!
//! 全構成で Unbounded を返さず、目的値はオラクル -2e-5 に次節の許容誤差で
//! 一致する。実測 (シングルスレッド決定論): eps 1e-6/1e-7 の presolve 両値と
//! (presolve=false, eps=1e-8) は Optimal、(presolve=true, eps=1e-8) は
//! SuboptimalSolution (KKT 残差 eps 未達の検証済み incumbent)。
//!
//! ## 目的値許容誤差 1e-9 の導出
//!
//! 上の厳密等式より obj(s) - obj* = 5e-10·(s-200)²。IPM は eps ≤ 1e-6 の残差で
//! 停止し実測 s ≈ 199.863 なので偏差 ≤ 5e-10·0.15² ≈ 1.1e-11。pin 値 1e-9 は
//! 実測最大偏差 1.11e-11 の約 90 倍 (= 位置なら |s-200| ≤ √2 相当) の決定論的
//! 余裕を持ちながら、修正前の誤答 obj=0 (x=0 早期収束) を |Δ| = 2e-5 =
//! 許容誤差の 2e4 倍で明確に区別する。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot::qp::ipm_solver::solve_ipm;
use otspot::qp::QpProblem;
use otspot::sparse::CscMatrix;

/// 手計算オラクル obj* = -2e-5 (ヘッダ台帳参照)。
const ORACLE_OBJ: f64 = -2e-5;

/// 目的値の許容誤差。導出はヘッダ「目的値許容誤差 1e-9 の導出」参照。
const OBJ_TOL: f64 = 1e-9;

fn inf() -> f64 {
    f64::INFINITY
}

/// P1: Q の (2,2) 成分にのみ 1e-9 の近平な正定値摂動を持つ有界問題。
fn p1_bounded() -> QpProblem {
    QpProblem::new(
        CscMatrix::from_triplets(
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[1.0, -1.0, -1.0, 1.0 + 1e-9],
            2,
            2,
        )
        .unwrap(),
        vec![-1e-7, -1e-7],
        CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, -1.0], 1, 2).unwrap(),
        vec![0.0],
        vec![(-inf(), inf()), (-inf(), inf())],
        vec![ConstraintType::Eq],
    )
    .unwrap()
}

fn solve_p1(eps: f64, presolve: bool) -> SolverResult {
    let problem = p1_bounded();
    let mut opts = SolverOptions::default();
    opts.ipm.eps = eps;
    opts.presolve = presolve;
    opts.timeout_secs = Some(10.0);
    solve_ipm(&problem, &opts)
}

/// soundness 本体: 有界問題 (手計算 obj* = -2e-5) がいかなる presolve × eps
/// 構成でも certified Unbounded を返さない。曲率検査の revert では全構成が
/// `Unbounded` (certified=true) を返し FAIL (台帳 TASK 2 で 6/6 実測)。
#[test]
fn bounded_p1_never_certified_unbounded_cross_eps_presolve() {
    for &presolve in &[false, true] {
        for &eps in &[1e-6_f64, 1e-7, 1e-8] {
            let r = solve_p1(eps, presolve);
            assert_ne!(
                r.status,
                SolveStatus::Unbounded,
                "presolve={presolve} eps={eps:.0e}: Unbounded is oracle-false \
                 (substitution-proved unique min obj* = -2e-5 at x=(200,200)); \
                 got obj={:e} iters={}",
                r.objective,
                r.iterations
            );
            assert_ne!(
                r.status,
                SolveStatus::Infeasible,
                "presolve={presolve} eps={eps:.0e}: x=(0,0) is feasible, so \
                 Infeasible is oracle-false; got obj={:e}",
                r.objective
            );
            assert!(
                !(r.status == SolveStatus::Unbounded && r.stats.infeasibility_certified),
                "presolve={presolve} eps={eps:.0e}: certified Unbounded must never \
                 appear on a bounded instance"
            );
        }
    }
}

/// 修正後の挙動を 6 構成それぞれで正確に pin (上記 GREEN 基準の実測値) +
/// 目的値が手計算オラクルに許容誤差 1e-9 以内。旧実装の誤答 obj=0 (Optimal
/// 早期収束) もこの tolerances で同時に弾く。
#[test]
fn bounded_p1_status_and_objective_match_oracle_across_configs() {
    let expected: [(&str, f64, bool, SolveStatus); 6] = [
        ("presolve=false eps=1e-6", 1e-6, false, SolveStatus::Optimal),
        ("presolve=false eps=1e-7", 1e-7, false, SolveStatus::Optimal),
        ("presolve=false eps=1e-8", 1e-8, false, SolveStatus::Optimal),
        ("presolve=true eps=1e-6", 1e-6, true, SolveStatus::Optimal),
        ("presolve=true eps=1e-7", 1e-7, true, SolveStatus::Optimal),
        (
            "presolve=true eps=1e-8",
            1e-8,
            true,
            SolveStatus::SuboptimalSolution,
        ),
    ];
    for (label, eps, presolve, want_status) in expected {
        let r = solve_p1(eps, presolve);
        assert_eq!(
            r.status, want_status,
            "{label}: post-fix status must match the measured ledger \
             (obj={:e} iters={})",
            r.objective, r.iterations
        );
        assert!(
            (r.objective - ORACLE_OBJ).abs() <= OBJ_TOL,
            "{label}: objective must match hand oracle -2e-5 within {OBJ_TOL:e} \
             (derivation in file header); got {:e} (Δ={:e})",
            r.objective,
            (r.objective - ORACLE_OBJ).abs()
        );
        eprintln!(
            "[P1 bounded] {label} -> {:?} obj={:e} iters={}",
            r.status, r.objective, r.iterations
        );
    }
}
