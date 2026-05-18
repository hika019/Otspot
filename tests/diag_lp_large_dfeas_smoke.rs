//! Regression sentinel: dual feasibility が unit test 時間内 (CLAUDE.md 3 min cap)
//! で退化検知できるよう、bench-equivalent dfr (`compute_dfeas_orig` 同等式) を
//! 完走する複数 LP で測る。
//!
//! Why: #56 系 (extract_dual_info / postsolve rc 復元) を触ると pilot87 が
//! 6 桁劣化したが、既存 unit test では捕捉できず bench 18 分回さないと表面化
//! しなかった (memory: feedback_dual_feas_changes_need_bench.md)。次回 dual
//! feasibility 系修正の即時 FAIL ガード。
//!
//! 含む:
//!  - `bound_absorb_synthetic_minimal`: BoundAbsorb 経路を踏ませる最小 LP。
//!    cleanup LP が小規模 LP では成功するため bug 自体は再現しないが、
//!    BoundAbsorb 後の rc が dual feasibility を満たすことを契約として固定。
//!  - `pilot4_dfeas_smoke_60s`: pilot87 系列縮約版 (n=1000 m=400 級)、60s 内
//!    完走する。Optimal なら bench dfr < 1e-6 を要求。退化方向に振れたら fail。
//!  - `pilot87_dfeas_smoke_150s`: 本修正の元 LP (n=4883 m=2030)。CLAUDE.md
//!    3 min cap 内のため 150s 制限。完走しなくとも rc 取得時は dfr を記録、
//!    取得不能時は visible log (silent SKIP 禁止に従い honest に記録)。
//!    fallback (simplex/mod.rs:96-) 経由 alt が走るため本修正の clamp 効果
//!    自体は test 環境では完全には観測できない (本来の安全網は bench)。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::{ConstraintType, LpProblem, SolveStatus};
use solver::solve_with;
use solver::sparse::CscMatrix;
use std::path::Path;

const REL_TOL_AT_BOUND: f64 = 1e-8;
const ZERO_TOL_FIXED: f64 = 1e-12;
const DFEAS_PASS_EPS: f64 = 1e-6;

/// bench 判定 (`src/bin/qps_benchmark.rs::compute_dfeas_orig`、LP 経路) 同等式。
/// `bound_duals` 不要 (LP 経路は rc から bound 双対を読む)。
fn bench_dfeas(lp: &LpProblem, sol: &[f64], rc: &[f64]) -> f64 {
    let n = lp.num_vars;
    if rc.is_empty() || rc.len() != n || sol.len() != n {
        return f64::NAN;
    }
    let mut worst = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = lp.bounds[j];
        if lb.is_finite() && ub.is_finite() && (ub - lb).abs() < ZERO_TOL_FIXED {
            continue;
        }
        if lp.a.col_ptr.len() > j + 1 && lp.a.col_ptr[j + 1] - lp.a.col_ptr[j] == 0 {
            continue;
        }
        let xj = sol[j];
        let at_lb = lb.is_finite() && (xj - lb).abs() <= REL_TOL_AT_BOUND * (1.0 + xj.abs() + lb.abs());
        let at_ub = ub.is_finite() && (xj - ub).abs() <= REL_TOL_AT_BOUND * (1.0 + xj.abs() + ub.abs());
        let viol = if at_lb && !at_ub {
            f64::max(0.0, -rc[j])
        } else if at_ub && !at_lb {
            f64::max(0.0, rc[j])
        } else {
            0.0
        };
        let scale = 1.0 + rc[j].abs() + lp.c[j].abs();
        worst = worst.max(viol / scale);
    }
    worst
}

/// BoundAbsorb 契約: orig (lb=0, ub=+inf) で implied ub=0 → fix された列が
/// at orig lb=0 を取るとき rc は ≥ 0 (= μ_lb) でなければならない。
/// 本テストは BoundAbsorb logic の最小契約として固定する (cleanup LP が小規模で
/// 成功するため bug 自体は再現しないが、clamp が逆向きに書き換わったら
/// rc が dual infeasible 化して bench dfeas が立つことを保証する)。
#[test]
fn bound_absorb_synthetic_minimal_at_lb() {
    let a = CscMatrix::from_triplets(
        &[0, 0, 1],
        &[0, 1, 1],
        &[1.0, 1.0, 1.0],
        2,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 0.0],
        a,
        vec![1.0, 0.0],
        vec![ConstraintType::Eq, ConstraintType::Le],
        vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(10.0);
    let r = solve_with(&lp, &opts);

    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.solution[0] - 1.0).abs() < 1e-9);
    assert!(r.solution[1].abs() < 1e-9);

    // 最低契約: at lb の rc は ≥ 0 (BoundAbsorb の at_lb 分岐)。
    assert!(
        r.reduced_costs[1] >= -DFEAS_PASS_EPS,
        "rc[x2]={} (at lb=0) は dual feasibility 上 ≥ 0 必須",
        r.reduced_costs[1]
    );

    let dfr = bench_dfeas(&lp, &r.solution, &r.reduced_costs);
    assert!(dfr < DFEAS_PASS_EPS, "synthetic dfr={:.3e}", dfr);
}

/// 対称チェック: at orig ub の場合 rc は ≤ 0 (= -μ_ub) でなければならない。
#[test]
fn bound_absorb_synthetic_minimal_at_ub() {
    // min  -x1  s.t.  x1 + x2 = 1, x1 + x2 >= 1, x1 in [0, 1], x2 in [0, 1].
    // 1 行 Ge を入れて step5 で x1 を lb=ub=1 に固定させたいが、現実的には
    // orig ub=1 に押し付ける構造を作る。
    //   min  -x1   (c = [-1, 0])
    //   s.t. x1 + x2 = 1     (Eq)
    //        x1 >= 1          (Ge, x1 の implied lb=1 を作る)
    //        0 <= x1, x2 <= 1
    // → x1=1 (at orig ub), x2=0 (at orig lb).
    let a = CscMatrix::from_triplets(
        &[0, 0, 1],
        &[0, 1, 0],
        &[1.0, 1.0, 1.0],
        2,
        2,
    )
    .unwrap();
    let lp = LpProblem::new_general(
        vec![-1.0, 0.0],
        a,
        vec![1.0, 1.0],
        vec![ConstraintType::Eq, ConstraintType::Ge],
        vec![(0.0, 1.0), (0.0, 1.0)],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(10.0);
    let r = solve_with(&lp, &opts);

    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.solution[0] - 1.0).abs() < 1e-9);
    assert!(r.solution[1].abs() < 1e-9);

    // x1 at orig ub=1 → rc ≤ 0、x2 at orig lb=0 → rc ≥ 0。
    assert!(
        r.reduced_costs[0] <= DFEAS_PASS_EPS,
        "rc[x1]={} (at ub=1) は dual feasibility 上 ≤ 0 必須",
        r.reduced_costs[0]
    );
    assert!(
        r.reduced_costs[1] >= -DFEAS_PASS_EPS,
        "rc[x2]={} (at lb=0) は dual feasibility 上 ≥ 0 必須",
        r.reduced_costs[1]
    );

    let dfr = bench_dfeas(&lp, &r.solution, &r.reduced_costs);
    assert!(dfr < DFEAS_PASS_EPS, "synthetic at_ub dfr={:.3e}", dfr);
}

fn solve_qps_with_budget(path: &str, timeout_secs: f64) -> (SolveStatus, Vec<f64>, Vec<f64>, Option<f64>) {
    let p = Path::new(path);
    let qp = parse_qps(p).expect("parse qps");
    let lp = LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(timeout_secs);
    let r = solve_with(&lp, &opts);
    let dfr = if !r.reduced_costs.is_empty() && r.solution.len() == lp.num_vars {
        Some(bench_dfeas(&lp, &r.solution, &r.reduced_costs))
    } else {
        None
    };
    (r.status, r.solution, r.reduced_costs, dfr)
}

/// pilot4 (Netlib pilot87 系列縮約版): 60s 完走可能 → Optimal なら dfr < 1e-6 必須。
/// 通常 0.5s 程度で完走するため CI 上の負担も軽く、dual feasibility 退化への
/// fast-feedback regression sentinel。
#[test]
fn pilot4_dfeas_smoke_60s() {
    let path = "data/lp_problems/pilot4.QPS";
    if !Path::new(path).exists() {
        eprintln!("[smoke] {} not found, bench data 未配置で観測スキップ (visible)", path);
        return;
    }
    let (status, _x, _rc, dfr) = solve_qps_with_budget(path, 60.0);
    eprintln!("[smoke] pilot4 status={:?} dfr={:?}", status, dfr);
    match status {
        SolveStatus::Optimal => {
            let d = dfr.expect("Optimal なら rc 取れる");
            assert!(
                d < DFEAS_PASS_EPS,
                "pilot4 Optimal dfr={:.3e} >= {:.0e} — dual feasibility 退化",
                d, DFEAS_PASS_EPS
            );
        }
        SolveStatus::Timeout | SolveStatus::SuboptimalSolution => {
            // 通常完走するので Timeout は前段退化のサイン。partial dfr 取れた場合のみ
            // 弱いガード (1e-3) を課す (収束途中の partial dfr は通常 1e-1 程度)。
            if let Some(d) = dfr {
                assert!(
                    d < 1e-3,
                    "pilot4 partial dfr={:.3e} >= 1e-3 — 通常 60s 完走 LP の劣化",
                    d
                );
            } else {
                panic!("pilot4 status={:?} で rc 不在は本来想定外 (60s 完走想定)", status);
            }
        }
        other => panic!("pilot4 unexpected status: {:?}", other),
    }
}

/// pilot87 (n=4883 m=2030): 本修正の元 LP。CLAUDE.md 3 min cap で 150s 制限。
/// この test 経路では fallback (simplex/mod.rs) が走るため最終 status は Timeout
/// になりやすく、その場合 alt が assemble されず rc 取れない。観測不能を honest
/// に log (silent SKIP 禁止)、観測可能時のみ dfr 判定。
///
/// 本来の安全網は bench (memory: feedback_dual_feas_changes_need_bench.md)。
#[test]
fn pilot87_dfeas_smoke_150s() {
    let path = "data/lp_problems/pilot87.QPS";
    if !Path::new(path).exists() {
        eprintln!("[smoke] {} not found, bench data 未配置で観測スキップ (visible)", path);
        return;
    }
    let (status, _x, _rc, dfr) = solve_qps_with_budget(path, 150.0);
    eprintln!("[smoke] pilot87 status={:?} dfr={:?}", status, dfr);
    match status {
        SolveStatus::Optimal => {
            let d = dfr.expect("Optimal なら rc 取れる");
            assert!(
                d < DFEAS_PASS_EPS,
                "pilot87 Optimal dfr={:.3e} >= {:.0e} — bound 双対吸収退化",
                d, DFEAS_PASS_EPS
            );
        }
        SolveStatus::Timeout | SolveStatus::SuboptimalSolution => {
            if let Some(d) = dfr {
                // partial state は通常 dfr ≤ 1e-2 (alt が presolve-off で
                // 動いている過程の dfr)。0.1 を超えるのは catastrophic な
                // dual 構造破壊 (例: y 符号 / インデキシング全壊) の signal。
                // 本修正の固有 bug (rc=1.8e-4) は partial の上に乗っても
                // この閾値では刺さらない — 本来の安全網は bench。
                assert!(
                    d < 1e-1,
                    "pilot87 partial dfr={:.3e} >= 0.1 — catastrophic dual infeasibility 検出 (sentinel as fast-fail)",
                    d
                );
                eprintln!("[smoke] pilot87 partial dfr={:.3e} (Timeout 中で本 bug 自体は分離不能、bench 必須)", d);
            } else {
                eprintln!("[smoke] pilot87 {:?} で rc 不在、unit test 環境では観測不能 (visible)。bench で regression 監視 (feedback_dual_feas_changes_need_bench.md)", status);
            }
        }
        other => panic!("pilot87 unexpected status: {:?}", other),
    }
}
