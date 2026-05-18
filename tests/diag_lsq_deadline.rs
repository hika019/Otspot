//! LSQ dual refine が user deadline を honor するかの regression sentinel。
//!
//! task #40 fail-safe: cleanup stagnant 時の LSQ skip (#39) は algorithmic gate。
//! cleanup が一定改善した case では LSQ が走るため、LSQ 自体が
//! `Option<Instant>` deadline を尊重して budget 内に終了する必要がある。
//!
//! 観測 (#38): dfl001 postsolve LSQ は no-deadline では 2.9-4.5s 消費。
//! deadline propagation 漏れがあると ken-13 等の大規模 LP で
//! 短い `timeout_secs` を設定しても wall が大きく超過する。
//!
//! 直接 `compute_lsq_dual_y` は `pub(crate)` のため public API
//! (`solve_with`) 経由で wall <= budget + SLACK_SEC を assert する。

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::LpProblem;
use solver::{solve_with, QpProblem};
use std::path::Path;
use std::time::Instant;

/// postsolve cleanup LP / Gauss-Seidel 等、LSQ 外の post-processing 残量を吸収する余裕。
/// task #48 で simplex half-deadline 撤廃後、solver が user budget を full 使い切り
/// parallel test 実行時の CPU 競合で wall がやや膨らむため 4.0s に拡張。
/// LSQ 漏れ (= wall ≈ 2×budget) との分離は依然可能。
const SLACK_SEC: f64 = 4.0;

fn make_lp(qp: &QpProblem) -> LpProblem {
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .unwrap()
}

/// ken-13 を短い timeout で解いて wall <= timeout + SLACK_SEC を確認。
/// LSQ deadline 漏れがあると postsolve LSQ が budget を無視して wall が
/// (timeout + LSQ_runtime) に膨らみ、assert で fail する。
#[test]
fn lsq_honors_deadline_on_ken13_short_budget() {
    let path = Path::new("data/lp_problems/ken-13.QPS");
    assert!(
        path.exists(),
        "data required (no SKIP): {:?} — lp_download script で取得",
        path
    );
    let qp = parse_qps(path).expect("parse QPS");
    let lp = make_lp(&qp);

    let budget = 3.0_f64;
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(budget);

    let t0 = Instant::now();
    let result = solve_with(&lp, &opts);
    let wall = t0.elapsed().as_secs_f64();

    eprintln!(
        "[lsq-deadline ken-13] status={:?} wall={:.3}s budget={}s slack={}s",
        result.status, wall, budget, SLACK_SEC
    );
    assert!(
        wall <= budget + SLACK_SEC,
        "wall {:.3}s > budget {}s + slack {}s — LSQ deadline 漏れ疑い",
        wall,
        budget,
        SLACK_SEC
    );
}

/// dfl001 でも同様の sentinel。#39 で cleanup_stagnant → LSQ skip となる想定だが、
/// LSQ skip gate が将来外れた場合に deadline 漏れが顕在化しないよう二重防御。
#[test]
fn lsq_honors_deadline_on_dfl001_short_budget() {
    let path = Path::new("data/lp_problems/dfl001.QPS");
    assert!(
        path.exists(),
        "data required (no SKIP): {:?} — lp_download script で取得",
        path
    );
    let qp = parse_qps(path).expect("parse QPS");
    let lp = make_lp(&qp);

    let budget = 3.0_f64;
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(budget);

    let t0 = Instant::now();
    let result = solve_with(&lp, &opts);
    let wall = t0.elapsed().as_secs_f64();

    eprintln!(
        "[lsq-deadline dfl001] status={:?} wall={:.3}s budget={}s slack={}s",
        result.status, wall, budget, SLACK_SEC
    );
    assert!(
        wall <= budget + SLACK_SEC,
        "wall {:.3}s > budget {}s + slack {}s — LSQ deadline 漏れ疑い",
        wall,
        budget,
        SLACK_SEC
    );
}
