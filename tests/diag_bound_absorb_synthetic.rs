//! `BoundAbsorb` (presolve/postsolve.rs:426) の 4 分岐を最小合成 LP で固定する。
//!
//! 分岐: AtLb / AtUb / interior-skip / truly_fixed-skip。
//! 各 test は単独 LP で対応分岐を踏ませ、契約 (clamp 後 rc が dfeas 上整合 or
//! clamp なしで raw `c − A^T y` 維持) を assert する。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::solve_with;
use otspot::sparse::CscMatrix;

/// bench 同等 dual feasibility 判定の eps (CLAUDE.md bench 標準 1e-6)。
const BENCH_DFEAS_EPS: f64 = 1e-6;
/// at-bound 判定: absolute 1e-6 (合成 LP は unit-scale のため本実装の relative 化と結果一致、
/// large-scale fixture では本実装の at_lb_tol を使う)。
const BOUND_ACTIVE_REL_TOL: f64 = 1e-6;
/// 解値の許容 (合成 LP は cleanup LP で十分小さく出る)。
const SOLUTION_TOL: f64 = 1e-9;
/// 「rc に clamp が走らなかった」の同値判定 (interior / truly_fixed branch)。
/// 浮動小数同一性は厳しすぎ、recompute 経路上の最後の f64 演算誤差を許容。
const RC_RAW_MATCH_TOL: f64 = 1e-12;

/// raw `c − A^T y` を返り値の dual_solution から復元 (clamp 前の rc)。
fn raw_rc(lp: &LpProblem, dual_solution: &[f64]) -> Vec<f64> {
    let n = lp.num_vars;
    let mut rc = lp.c.clone();
    for (j, slot) in rc.iter_mut().enumerate().take(n) {
        if let Ok((rows, vals)) = lp.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                *slot -= vals[k] * dual_solution[row];
            }
        }
    }
    rc
}

/// bench `compute_dfeas_orig` と同等式 (LP 経路、bound dual を rc から読む)。
fn bench_dfeas(lp: &LpProblem, sol: &[f64], rc: &[f64]) -> f64 {
    let n = lp.num_vars;
    if rc.len() != n || sol.len() != n {
        return f64::NAN;
    }
    let mut worst = 0.0f64;
    for j in 0..n {
        let (lb, ub) = lp.bounds[j];
        if lb.is_finite() && ub.is_finite() && (ub - lb).abs() < BOUND_ACTIVE_REL_TOL {
            continue;
        }
        let xj = sol[j];
        let at_lb = lb.is_finite() && (xj - lb).abs() < BOUND_ACTIVE_REL_TOL;
        let at_ub = ub.is_finite() && (xj - ub).abs() < BOUND_ACTIVE_REL_TOL;
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

/// AtLb 分岐: orig (lb=0, ub=+inf)、presolve が ub→0 タイト化 → x=0=orig_lb に
/// fix。 BoundAbsorb は `rc = max(rc, 0)` を適用、dual feasibility 上 rc≥0 が契約。
#[test]
fn bound_absorb_at_lb() {
    //   min  x1
    //   s.t. x1 + x2 = 1          (Eq)
    //                x2 <= 0      (Le → x2 の implied ub=0、x2 fix at orig lb=0)
    //        0 <= x1, x2 < +inf
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, 1.0], 2, 2).unwrap();
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
    assert!((r.solution[0] - 1.0).abs() < SOLUTION_TOL);
    assert!(r.solution[1].abs() < SOLUTION_TOL);

    // x2 at orig lb=0 → AtLb 契約: rc ≥ 0。
    assert!(
        r.reduced_costs[1] >= -BENCH_DFEAS_EPS,
        "AtLb 契約違反: rc[x2]={} (at orig_lb=0)",
        r.reduced_costs[1]
    );
    let dfr = bench_dfeas(&lp, &r.solution, &r.reduced_costs);
    assert!(dfr < BENCH_DFEAS_EPS, "dfr={:.3e}", dfr);
}

/// AtUb 分岐: orig (lb=0, ub=1)、presolve が lb→1 タイト化 → x=1=orig_ub に fix。
/// BoundAbsorb は `rc = min(rc, 0)` を適用、dual feasibility 上 rc≤0 が契約。
#[test]
fn bound_absorb_at_ub() {
    //   min  -x1
    //   s.t. x1 + x2 = 1     (Eq)
    //        x1      >= 1    (Ge → x1 の implied lb=1、x1 fix at orig ub=1)
    //        0 <= x1, x2 <= 1
    let a = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 0], &[1.0, 1.0, 1.0], 2, 2).unwrap();
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
    assert!((r.solution[0] - 1.0).abs() < SOLUTION_TOL);
    assert!(r.solution[1].abs() < SOLUTION_TOL);

    // x1 at orig ub=1 → AtUb 契約: rc ≤ 0。
    assert!(
        r.reduced_costs[0] <= BENCH_DFEAS_EPS,
        "AtUb 契約違反: rc[x1]={} (at orig_ub=1)",
        r.reduced_costs[0]
    );
    let dfr = bench_dfeas(&lp, &r.solution, &r.reduced_costs);
    assert!(dfr < BENCH_DFEAS_EPS, "dfr={:.3e}", dfr);
}

/// interior-skip 分岐: orig (0, 100)、presolve で両 bound タイト化 → fixed at 50
/// (orig 内部)。BoundAbsorb は対象外 (None)、rc は raw `c − A^T y` のまま。
/// bandm/beaconfd 等で BoundAbsorb を無効化した効果を unit test 化したもの。
///
/// 2 列 (c=+, c=−) を同時に interior-fix させ、誤クランプを両方向で検出する:
/// `max(rc,0)` (誤 AtLb) は負側 col の rc を 0 に書き換え、`min(rc,0)` (誤 AtUb)
/// は正側 col の rc を 0 に書き換える。両方の rc==raw 等値が保たれることで
/// 正しい None 判定を確認する。
#[test]
fn bound_absorb_interior_skip() {
    //   min  x1 − x2
    //   s.t. x1      <= 50     (Le → ub=50)
    //        x1      >= 50     (Ge → lb=50, x1 fixed at 50 ∈ (0,100) interior)
    //             x2 <= 30     (Le → ub=30)
    //             x2 >= 30     (Ge → lb=30, x2 fixed at 30 ∈ (0,80) interior)
    //        0 <= x1 <= 100, 0 <= x2 <= 80
    let a = CscMatrix::from_triplets(&[0, 1, 2, 3], &[0, 0, 1, 1], &[1.0, 1.0, 1.0, 1.0], 4, 2)
        .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, -1.0],
        a,
        vec![50.0, 50.0, 30.0, 30.0],
        vec![
            ConstraintType::Le,
            ConstraintType::Ge,
            ConstraintType::Le,
            ConstraintType::Ge,
        ],
        vec![(0.0, 100.0), (0.0, 80.0)],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(10.0);
    let r = solve_with(&lp, &opts);
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.solution[0] - 50.0).abs() < SOLUTION_TOL);
    assert!((r.solution[1] - 30.0).abs() < SOLUTION_TOL);

    // 両 col とも interior-fixed → BoundAbsorb None、rc == raw が契約。
    let raw = raw_rc(&lp, &r.dual_solution);
    for (j, &raw_j) in raw.iter().enumerate() {
        assert!(
            (r.reduced_costs[j] - raw_j).abs() < RC_RAW_MATCH_TOL,
            "interior-skip 違反: rc[x{}]={} != raw={} (clamp 誤適用)",
            j + 1,
            r.reduced_costs[j],
            raw_j,
        );
    }
    let dfr = bench_dfeas(&lp, &r.solution, &r.reduced_costs);
    assert!(dfr < BENCH_DFEAS_EPS, "dfr={:.3e}", dfr);
}

/// truly_fixed-skip 分岐: orig lb=ub (元から fixed)、step1 即 fix。
/// `BoundAbsorb` は早期 `continue` で対象外、rc は raw のまま。
/// 2 列 (c=+, c=−) で誤クランプを両方向で検出する (interior-skip と同戦略)。
#[test]
fn bound_absorb_truly_fixed_skip() {
    //   min  x1 − x2 + x3
    //   s.t. x1 + x2 + x3 = 12
    //        x1 in [5,5] (truly fixed), x2 in [3,3] (truly fixed), x3 in [0,+inf]
    let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, -1.0, 1.0],
        a,
        vec![12.0],
        vec![ConstraintType::Eq],
        vec![(5.0, 5.0), (3.0, 3.0), (0.0, f64::INFINITY)],
        None,
    )
    .unwrap();

    let mut opts = SolverOptions::default();
    opts.presolve = true;
    opts.timeout_secs = Some(10.0);
    let r = solve_with(&lp, &opts);
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!((r.solution[0] - 5.0).abs() < SOLUTION_TOL);
    assert!((r.solution[1] - 3.0).abs() < SOLUTION_TOL);
    assert!((r.solution[2] - 4.0).abs() < SOLUTION_TOL);

    // x1, x2 とも元から fixed → BoundAbsorb 早期 continue、rc == raw が契約。
    let raw = raw_rc(&lp, &r.dual_solution);
    for (j, &raw_j) in raw.iter().enumerate().take(2) {
        assert!(
            (r.reduced_costs[j] - raw_j).abs() < RC_RAW_MATCH_TOL,
            "truly_fixed-skip 違反: rc[x{}]={} != raw={} (clamp 誤適用)",
            j + 1,
            r.reduced_costs[j],
            raw_j,
        );
    }
    let dfr = bench_dfeas(&lp, &r.solution, &r.reduced_costs);
    assert!(dfr < BENCH_DFEAS_EPS, "dfr={:.3e}", dfr);
}
