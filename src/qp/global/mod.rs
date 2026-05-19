//! Phase 3 spatial Branch-and-Bound scaffolding (#6 非凸 QP 大域最適化)。
//!
//! # scope
//! Phase 3 は **scaffolding** = BB tree / 分枝 / incumbent / 単純 pruning だけ。
//! 下界は box 上 interval arithmetic = 制約 Ax=b 無視で緩い。実用 (gap_tol=1e-3
//! で実問題確実 hit) は Phase 4 (α-BB) 必須。
//!
//! # API
//! [`solve_qp_global`] を [`crate::qp::solve_qp_with`] とは別の明示 entry として提供。
//! `SolverOptions::global_optimization` が Some でも `solve_qp_with` は dispatch しない
//! (= 既存 QP user の wall を桁違いに伸ばさない安全装置)。
//!
//! # 戻り値の status
//! - `Optimal`: queue 空 + 全 leaf が ε-feasible → 大域 ε-optimal 証明済み
//! - `LocallyOptimal`: deadline / max_nodes / max_depth で打ち切り、incumbent あり (gap 未保証)
//! - `Timeout`: deadline で打ち切り、incumbent 未発見
//! - root と同じ status: root が Infeasible / NumericalError / Unbounded だった場合

pub(crate) mod bound;
pub(crate) mod bound_alpha_bb;
pub mod bound_mccormick;
pub(crate) mod branch;
pub(crate) mod node;
pub(crate) mod pruning;
pub(crate) mod tree;

use crate::options::{GlobalOptimizationConfig, QpWarmStart, SolverOptions};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use std::time::{Duration, Instant};

use bound::{interval_quadratic_bounds, is_feasible_result, solve_local_upper_bound};
use bound_alpha_bb::{alpha_bb_lower_bound, gershgorin_alpha};
use bound_mccormick::mccormick_lower_bound;
use branch::{select_branching_variable, split_node};
use node::BBNode;
use pruning::{should_prune, within_gap};
use tree::BBTree;

/// 大域最適化 entry。
///
/// 入力: convex / nonconvex QP (`QpProblem`) + 共通 solver options + 大域設定。
/// 出力: 大域 ε-optimal incumbent (`SolveStatus::Optimal`) or 打ち切り incumbent
/// (`LocallyOptimal` / `Timeout` / 入口失敗の伝播)。
///
/// 各 node の local solve は `solve_qp_with` 経由 = inertia 補正付き IPM
/// + warm start で parent 解継承。下界 default は α-BB (`bound_alpha_bb`、Phase 4)、
/// `use_alpha_bb=false` で interval_quadratic_bounds (Phase 3 fallback) に切替可。
/// BB 探索の統計 (テスト sentinel 用、production API には含めない)。
/// `nodes_processed`: solve_local_upper_bound 呼び出し総回数 (root 含む)。
/// `max_depth_seen`: 探索 tree 内で到達した最大 depth。
/// `pruned`: 子展開前に枝刈で discard した node 数。
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalStats {
    pub nodes_processed: usize,
    pub max_depth_seen: usize,
    pub pruned: usize,
}

pub fn solve_qp_global(
    problem: &QpProblem,
    options: &SolverOptions,
    cfg: &GlobalOptimizationConfig,
) -> SolverResult {
    solve_qp_global_with_stats(problem, options, cfg).0
}

/// テスト sentinel 用: result とともに探索統計を返す。
/// public で sentinel test (pruning no-op 検出) からのみ参照される。
pub fn solve_qp_global_with_stats(
    problem: &QpProblem,
    options: &SolverOptions,
    cfg: &GlobalOptimizationConfig,
) -> (SolverResult, GlobalStats) {
    // deadline 計算: options.deadline 優先、無ければ timeout_secs から固定。
    let deadline = options.deadline.or_else(|| {
        options
            .timeout_secs
            .map(|s| Instant::now() + Duration::from_secs_f64(s))
    });
    let mut shared_opts = options.clone();
    shared_opts.deadline = deadline;
    shared_opts.timeout_secs = None;
    shared_opts.multistart = None;
    shared_opts.global_optimization = None;

    let root_bounds = problem.bounds.clone();

    let mut stats = GlobalStats::default();

    // 1. root local solve (= 初期 incumbent 候補)
    let root_solve = solve_local_upper_bound(problem, &root_bounds, &shared_opts, None);
    if !is_feasible_result(&root_solve.status) {
        // root が解けない (Infeasible / NumericalError / Unbounded / NonConvex / Timeout)
        // → そのまま伝播。
        return (root_solve, stats);
    }

    // Phase 4 α-BB: 全 node で共通の α (Q only). use_alpha_bb=false なら 0 で実質無効化。
    let alpha = if cfg.use_alpha_bb {
        gershgorin_alpha(&problem.q)
    } else {
        0.0
    };

    let root_lb = compute_node_lower_bound(
        problem,
        &root_bounds,
        alpha,
        &shared_opts,
        deadline,
        cfg.use_alpha_bb,
        cfg.use_mccormick,
    );

    let mut state = SearchState::new(root_solve);
    stats.nodes_processed = 1;

    // root が ε-optimal なら即終了 (queue 不要)。
    if within_gap(state.incumbent_obj, root_lb, cfg.gap_tol) {
        return (state.finalize_proven(root_lb), stats);
    }

    let mut tree = BBTree::new();

    // root 分枝。分枝不能 (= 全変数 infinite bound or width <= MIN_BRANCH_BOX_WIDTH)
    // のとき: 下界が incumbent と gap_tol 以内なら proof 済み、
    // そうでなければ証明不能 → LocallyOptimal (= 大域証明できない)。
    let root_node = BBNode::root(root_bounds, root_lb);
    let root_x = state.incumbent_sol.clone();
    match select_branching_variable(&root_node, &root_x) {
        None => {
            return if within_gap(state.incumbent_obj, root_lb, cfg.gap_tol) {
                (state.finalize_proven(root_lb), stats)
            } else {
                (
                    state.finalize_unproven(root_lb, stats.nodes_processed, 0, cfg),
                    stats,
                )
            };
        }
        Some(j) => {
            let warm = state.build_warm();
            let (l, r) = split_node(&root_node, j, root_x[j], warm);
            tree.push(l);
            tree.push(r);
        }
    }

    let mut max_depth_breached = false;

    while let Some(node) = tree.pop() {
        if deadline_hit(&deadline) {
            break;
        }
        if stats.nodes_processed >= cfg.max_nodes {
            break;
        }

        // 親から継承 lb で再 prune (incumbent が更新されている可能性)
        if should_prune(node.lower_bound, Some(state.incumbent_obj), cfg.gap_tol) {
            stats.pruned += 1;
            continue;
        }

        // 自前で再計算した lb (Phase 4/5: interval + α-BB + McCormick の max) で tight 化、再 prune
        let local_lb = compute_node_lower_bound(
            problem,
            &node.var_bounds,
            alpha,
            &shared_opts,
            deadline,
            cfg.use_alpha_bb,
            cfg.use_mccormick,
        );
        let node_lb = local_lb.max(node.lower_bound);
        if should_prune(node_lb, Some(state.incumbent_obj), cfg.gap_tol) {
            stats.pruned += 1;
            continue;
        }

        stats.nodes_processed += 1;
        if node.depth > stats.max_depth_seen {
            stats.max_depth_seen = node.depth;
        }

        let res = solve_local_upper_bound(
            problem,
            &node.var_bounds,
            &shared_opts,
            node.warm.as_ref(),
        );
        if !is_feasible_result(&res.status) {
            // この box は infeasible / numerical issue → discard (上の region は
            // 他 branch に任せる; 下界 ≥ 0 補正は Phase 4 で α-BB と併せて検討)。
            continue;
        }

        // incumbent 更新 (より小さい obj 発見)
        let improved = res.objective < state.incumbent_obj;
        if improved {
            state.update_incumbent(&res);
        }

        // 分枝
        if node.depth + 1 > cfg.max_depth {
            // 深さ上限超過 → 子を展開しない = unproven region 残存
            max_depth_breached = true;
            continue;
        }
        if let Some(j) = select_branching_variable(&node, &res.solution) {
            let warm = build_warm_from(&res);
            let (left, right) = split_node(&node, j, res.solution[j], warm);
            tree.push(left);
            tree.push(right);
        }
        // 分枝不能 (= node 内で x* が midpoint 一致) → leaf 確定、proof は incumbent 比で取れる
    }

    // 終了条件分岐:
    // - queue 空 AND max_depth 未超過 AND deadline/max_nodes 未到達 → proven
    // - それ以外 → 未証明 (incumbent あれば LocallyOptimal)
    let halted_early = !tree.is_empty()
        || max_depth_breached
        || deadline_hit(&deadline)
        || stats.nodes_processed >= cfg.max_nodes;

    let result = if halted_early {
        // 未探索領域の下界 (queue に残った node の最小 lb)
        let remaining_lb = tree.best_lower_bound().unwrap_or(f64::INFINITY);
        let proven = within_gap(state.incumbent_obj, remaining_lb, cfg.gap_tol);
        let inc_obj = state.incumbent_obj;
        if proven {
            let lb_for_proof = remaining_lb.min(inc_obj);
            state.finalize_proven(lb_for_proof)
        } else {
            state.finalize_unproven(
                remaining_lb,
                stats.nodes_processed,
                stats.max_depth_seen,
                cfg,
            )
        }
    } else {
        // queue 空 = 全探索完了 → incumbent_obj が global
        let inc_obj = state.incumbent_obj;
        state.finalize_proven(inc_obj)
    };
    (result, stats)
}

fn deadline_hit(deadline: &Option<Instant>) -> bool {
    deadline.map_or(false, |d| Instant::now() >= d)
}

/// 当該 box に対する lower bound。
/// 戦略: interval lb (cheap) + α-BB lb (1 凸 IPM solve) + McCormick lb (1 LP solve) の **max**。
/// 3 経路はいずれも valid lower bound のため `max` を取ることで一方が tight な方を採用
/// (= ロスなし)。各経路は `use_*` flag で個別に skip 可能。
fn compute_node_lower_bound(
    problem: &QpProblem,
    bounds: &[(f64, f64)],
    alpha: f64,
    base_opts: &SolverOptions,
    deadline: Option<Instant>,
    use_alpha_bb: bool,
    use_mccormick: bool,
) -> f64 {
    let (interval_lb, _) = interval_quadratic_bounds(problem, bounds);
    let mut lb = interval_lb;
    if use_alpha_bb {
        if let Some(ab_lb) = alpha_bb_lower_bound(problem, bounds, alpha, base_opts, deadline) {
            lb = lb.max(ab_lb);
        }
    }
    if use_mccormick {
        if let Some(mc_lb) = mccormick_lower_bound(problem, bounds, base_opts, deadline) {
            lb = lb.max(mc_lb);
        }
    }
    lb
}

fn build_warm_from(res: &SolverResult) -> Option<QpWarmStart> {
    if res.solution.is_empty() {
        return None;
    }
    Some(QpWarmStart {
        x: res.solution.clone(),
        y: res.dual_solution.clone(),
        mu: res.gap.unwrap_or(1e-6).max(1e-10),
    })
}

/// search state encapsulation: incumbent + 最終 result の組み立てを 1 箇所に集約。
struct SearchState {
    incumbent_result: SolverResult,
    incumbent_obj: f64,
    incumbent_sol: Vec<f64>,
}

impl SearchState {
    fn new(root: SolverResult) -> Self {
        let obj = root.objective;
        let sol = root.solution.clone();
        Self {
            incumbent_result: root,
            incumbent_obj: obj,
            incumbent_sol: sol,
        }
    }

    fn build_warm(&self) -> Option<QpWarmStart> {
        build_warm_from(&self.incumbent_result)
    }

    fn update_incumbent(&mut self, res: &SolverResult) {
        self.incumbent_obj = res.objective;
        self.incumbent_sol = res.solution.clone();
        self.incumbent_result = res.clone();
    }

    fn finalize_proven(mut self, lower_bound: f64) -> SolverResult {
        self.incumbent_result.status = SolveStatus::Optimal;
        log::debug!(
            "QP global proven: obj={:.6e} lb={:.6e}",
            self.incumbent_obj, lower_bound
        );
        self.incumbent_result
    }

    fn finalize_unproven(
        mut self,
        lower_bound: f64,
        nodes: usize,
        depth: usize,
        cfg: &GlobalOptimizationConfig,
    ) -> SolverResult {
        // 大域最適性は未確定 → LocallyOptimal に降格 (Optimal はあくまで proof 済み枠)
        self.incumbent_result.status = SolveStatus::LocallyOptimal;
        let gap = self.incumbent_obj - lower_bound;
        log::debug!(
            "QP global unproven: obj={:.6e} lb={:.6e} gap={:.3e} nodes={} depth={} tol={:.0e}",
            self.incumbent_obj, lower_bound, gap, nodes, depth, cfg.gap_tol
        );
        self.incumbent_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn diag_concave_1d(bnd: f64) -> QpProblem {
        // f = -x², box [-bnd, bnd] → global min = -bnd² at corners
        let q = CscMatrix::from_triplets(&[0], &[0], &[-2.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        QpProblem::new_all_le(q, vec![0.0], a, vec![], vec![(-bnd, bnd)]).unwrap()
    }

    fn opts(timeout: f64) -> SolverOptions {
        let mut o = SolverOptions::default();
        o.timeout_secs = Some(timeout);
        o
    }

    #[test]
    fn solve_qp_global_finds_corner_minimum_concave_1d() {
        let p = diag_concave_1d(2.0);
        let cfg = GlobalOptimizationConfig::default();
        let r = solve_qp_global(&p, &opts(5.0), &cfg);
        assert!(
            matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal),
            "expected Optimal/LocallyOptimal, got {:?}",
            r.status
        );
        // global = -4 at x=±2. Local IPM cold solve typically gets stuck at x=0 (saddle).
        assert!(
            r.objective < -3.99,
            "expected global ≈ -4, got obj={:.4}",
            r.objective
        );
    }

    #[test]
    fn solve_qp_global_cold_vs_global_separation() {
        // 大域: x=±2 → -4。cold IPM だと saddle x=0 (obj=0) に固着するケース。
        let p = diag_concave_1d(2.0);
        let cold = crate::qp::solve_qp_with(&p, &opts(5.0));
        let global = solve_qp_global(&p, &opts(5.0), &GlobalOptimizationConfig::default());
        // 大域結果は cold より厳密に良い (= global の方が小さい)
        assert!(
            global.objective <= cold.objective + 1e-6,
            "global ({}) should be ≤ cold ({})",
            global.objective,
            cold.objective
        );
        assert!(
            global.objective < -3.99,
            "global should reach corner, got {}",
            global.objective
        );
    }
}
