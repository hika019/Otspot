//! Phase 3 spatial Branch-and-Bound (非凸 QP 大域最適化)。
//!
//! [`solve_qp_global`] を `solve_qp_with` と別 entry で提供する (既存 QP user の
//! wall を桁違いに伸ばさない安全装置)。下界は box 上の interval arithmetic で
//! 制約を無視するため緩い — 実用には Phase 4 (α-BB) 必須。
//!
//! 戻り status: PSD なら `Optimal` / `LocallyOptimal`、indefinite なら
//! `NonconvexGlobal` / `NonconvexLocal`、deadline は `Timeout`、root が
//! Infeasible/NumericalError/Unbounded ならそのまま伝播。

pub(crate) mod bound;
pub(crate) mod bound_alpha_bb;
pub(crate) mod bound_mccormick;
pub(crate) mod branch;
pub(crate) mod node;
pub(crate) mod pruning;
pub(crate) mod tree;

use crate::linalg::timeout::deadline_reached;
use crate::options::{GlobalOptimizationConfig, QpWarmStart, SolverOptions};
use crate::problem::certificate::BoundGapCertificate;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::certificate::prove_optimal;
use crate::qp::ipm_solver::core::compute_duality_gap_rel;
use crate::qp::ipm_solver::kkt::{
    bound_violation as kkt_bound_violation, complementarity_residual_rel as kkt_comp_residual,
    kkt_residual_rel, primal_residual_rel as kkt_primal_residual,
};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::kkt_resid::dual_sign_violation as kkt_dual_sign_violation;
use crate::qp::problem::QpProblem;
use std::time::{Duration, Instant};

use bound::{interval_quadratic_bounds, is_feasible_result, solve_local_upper_bound};
use bound_alpha_bb::{alpha_bb_lower_bound, gershgorin_alpha};
use bound_mccormick::mccormick_lower_bound;
use branch::{select_branching_variable, split_node};
use node::BBNode;
use pruning::{should_prune, within_gap};
use tree::BBTree;

/// SuboptimalSolution な polish 結果を KKT 残差で採用するときの user_eps に対する倍率。
///
/// duality_gap のみ `user_eps` を僅かに超えて SuboptimalSolution になった polish を
/// dual recovery 目的で採用するための緩和係数。根拠: regression threshold
/// `EPS_KKT_NONCONVEX_LOCAL = 1e-3` に対して user_eps=1e-6 での十分な margin を確保。
const POLISH_KKT_ACCEPT_FACTOR: f64 = 100.0;

/// KKT 許容閾値の絶対上限。
///
/// `user_eps * POLISH_KKT_ACCEPT_FACTOR` が大きい (user_eps=1e-4 で 1e-2) 場合でも
/// regression threshold `EPS_KKT_NONCONVEX_LOCAL = 1e-3` を超えないよう制限する。
const POLISH_KKT_ABS_CAP: f64 = 1e-3;

/// polish solve の fallback timeout (秒)。
///
/// `polish_incumbent_duals` は B&B `deadline` の残時間を優先継承し、残時間が
/// 0 (B&B budget 枯渇) の場合のみこの値を fresh budget として用いる。
/// 5.0 sec は polish IPM の典型 iteration 数 (~10) と problem 規模 (B&B が
/// 解ける範囲 = 数百変数) に対する経験値で、収束に十分なゆとりを持つ。
/// B&B が timeout 前に正常終了した場合は残時間継承により timeout_secs 契約を
/// 破らない。
const POLISH_TIMEOUT_SECS: f64 = 5.0;

/// 大域最適化 entry。
///
/// 入力: convex / nonconvex QP (`QpProblem`) + 共通 solver options + 大域設定。
/// 出力: 大域 ε-optimal incumbent (`SolveStatus::Optimal`) or 打ち切り incumbent
/// (`LocallyOptimal` / `Timeout` / 入口失敗の伝播)。
///
/// 各 node の local solve は `solve_qp_with` 経由 = inertia 補正付き IPM
/// + warm start で parent 解継承。下界 default は α-BB (`bound_alpha_bb`、Phase 4)、
///   `use_alpha_bb=false` で interval_quadratic_bounds (Phase 3 fallback) に切替可。
///   BB 探索の統計 (テスト sentinel 用、production API には含めない)。
///   `nodes_processed`: solve_local_upper_bound 呼び出し総回数 (root 含む)。
///   `max_depth_seen`: 探索 tree 内で到達した最大 depth。
///   `pruned`: 子展開前に枝刈で discard した node 数。
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
    if options.validate().is_err() {
        return (SolverResult::numerical_error(), GlobalStats::default());
    }
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

    // status 分岐用: Q が indefinite かどうかを Gershgorin で判定。
    // gershgorin_alpha は対角 - off-diag 行和の最小値の絶対値 (Q が PSD 範囲なら 0)。
    // use_alpha_bb=false でも判定だけは行う (status 判別は探索戦略に依存させない)。
    let q_indefinite = is_q_indefinite(problem);

    let (root_lb, _) = compute_node_lower_bound(
        problem,
        &root_bounds,
        alpha,
        &shared_opts,
        deadline,
        cfg.use_alpha_bb,
        cfg.use_mccormick,
        None,
        cfg.gap_tol,
        None,
    );

    let mut state = SearchState::new(root_solve);
    stats.nodes_processed = 1;
    let user_eps = shared_opts.ipm_eps();

    if within_gap(state.incumbent_obj, root_lb, cfg.gap_tol) {
        state.polish_incumbent_duals(problem, &shared_opts, cfg.gap_tol, q_indefinite);
        return (
            state.finalize_proven(problem, root_lb, q_indefinite, cfg.gap_tol, user_eps),
            stats,
        );
    }

    let mut tree = BBTree::new();

    // root 分枝。分枝不能 (= 全変数 infinite bound or width <= MIN_BRANCH_BOX_WIDTH)
    // のとき: 下界が incumbent と gap_tol 以内なら proof 済み、
    // そうでなければ証明不能 → LocallyOptimal (= 大域証明できない)。
    let root_node = BBNode::root(root_bounds, root_lb);
    let root_x = state.incumbent_sol.clone();
    match select_branching_variable(&root_node, &root_x) {
        None => {
            if within_gap(state.incumbent_obj, root_lb, cfg.gap_tol) {
                state.polish_incumbent_duals(problem, &shared_opts, cfg.gap_tol, q_indefinite);
                return (
                    state.finalize_proven(problem, root_lb, q_indefinite, cfg.gap_tol, user_eps),
                    stats,
                );
            }
            return (
                state.finalize_unproven(root_lb, stats.nodes_processed, 0, cfg, q_indefinite),
                stats,
            );
        }
        Some(j) => {
            let warm = state.build_warm();
            let (l, r) = split_node(&root_node, j, root_x[j], warm, None);
            tree.push(l);
            tree.push(r);
        }
    }

    let mut max_depth_breached = false;
    // 深さ上限で破棄した node の node_lb の min を保持する。これが未探索領域の下界に
    // なるため remaining_lb に畳み込む必要がある。
    let mut depth_discard_lb: f64 = f64::INFINITY;

    while let Some(node) = tree.pop() {
        if deadline_reached(deadline) {
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
        let (local_lb, ab_warm_for_children) = compute_node_lower_bound(
            problem,
            &node.var_bounds,
            alpha,
            &shared_opts,
            deadline,
            cfg.use_alpha_bb,
            cfg.use_mccormick,
            Some(state.incumbent_obj),
            cfg.gap_tol,
            node.alpha_bb_warm.clone(),
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

        let res =
            solve_local_upper_bound(problem, &node.var_bounds, &shared_opts, node.warm.as_ref());
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
            // 深さ上限超過 → 子を展開しない = unproven region 残存。
            // この node の lb を depth_discard_lb に畳み込む (remaining_lb に反映する)。
            max_depth_breached = true;
            depth_discard_lb = depth_discard_lb.min(node_lb);
            continue;
        }
        if let Some(j) = select_branching_variable(&node, &res.solution) {
            let warm = build_warm_from(&res);
            let (left, right) = split_node(&node, j, res.solution[j], warm, ab_warm_for_children);
            tree.push(left);
            tree.push(right);
        }
        // 分枝不能 (= node 内で x* が midpoint 一致) → leaf 確定、proof は incumbent 比で取れる
    }

    // B&B incumbent の sub-box dual を元問題に整合させる (bound comp 修復)。
    state.polish_incumbent_duals(problem, &shared_opts, cfg.gap_tol, q_indefinite);

    // 終了条件分岐:
    // - queue 空 AND max_depth 未超過 AND deadline/max_nodes 未到達 → proven
    // - それ以外 → 未証明 (incumbent あれば LocallyOptimal)
    let halted_early = !tree.is_empty()
        || max_depth_breached
        || deadline_reached(deadline)
        || stats.nodes_processed >= cfg.max_nodes;

    let result = if halted_early {
        // 未探索領域の下界: queue に残った node の最小 lb と、深さ上限で破棄した
        // node の lb の両方を考慮する。どちらの領域も「未証明」であるため min を取る。
        let remaining_lb = tree
            .best_lower_bound()
            .unwrap_or(f64::INFINITY)
            .min(depth_discard_lb);
        let proven = within_gap(state.incumbent_obj, remaining_lb, cfg.gap_tol);
        let inc_obj = state.incumbent_obj;
        if proven {
            let lb_for_proof = remaining_lb.min(inc_obj);
            state.finalize_proven(problem, lb_for_proof, q_indefinite, cfg.gap_tol, user_eps)
        } else {
            state.finalize_unproven(
                remaining_lb,
                stats.nodes_processed,
                stats.max_depth_seen,
                cfg,
                q_indefinite,
            )
        }
    } else {
        // queue 空 = 全探索完了 → incumbent_obj が global
        let inc_obj = state.incumbent_obj;
        state.finalize_proven(problem, inc_obj, q_indefinite, cfg.gap_tol, user_eps)
    };
    (result, stats)
}

/// Q が indefinite (= 少なくとも 1 つの負固有値が Gershgorin で証明可能) か。
///
/// `gershgorin_alpha` は対角項 - off-diag 絶対値和の最小値が負のとき正値を返す
/// (= α-BB の δ 補正量、Q が PSD 範囲内なら 0)。これを「PSD でない疑いあり」
/// = caller 視点では nonconvex 確実、と扱う (Gershgorin は十分条件、必要ではない)。
fn is_q_indefinite(problem: &QpProblem) -> bool {
    gershgorin_alpha(&problem.q) > 0.0
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
    incumbent_obj: Option<f64>,
    gap_tol: f64,
    alpha_bb_warm_in: Option<QpWarmStart>,
) -> (f64, Option<QpWarmStart>) {
    let (interval_lb, _) = interval_quadratic_bounds(problem, bounds);
    let mut lb = interval_lb;
    let mut ab_warm_out: Option<QpWarmStart> = None;
    if should_prune(lb, incumbent_obj, gap_tol) {
        return (lb, None);
    }
    if use_alpha_bb {
        if let Some((ab_lb, ab_warm)) = alpha_bb_lower_bound(
            problem,
            bounds,
            alpha,
            base_opts,
            deadline,
            alpha_bb_warm_in,
        ) {
            lb = lb.max(ab_lb);
            ab_warm_out = ab_warm;
        }
    }
    if use_mccormick {
        if let Some(mc_lb) = mccormick_lower_bound(problem, bounds, base_opts, deadline) {
            lb = lb.max(mc_lb);
        }
    }
    (lb, ab_warm_out)
}

fn build_warm_from(res: &SolverResult) -> Option<QpWarmStart> {
    if res.solution.is_empty() {
        return None;
    }
    Some(QpWarmStart {
        x: res.solution.clone(),
        y: res.dual_solution.clone(),
        mu: res
            .final_residuals
            .map(|(_, _, g)| g)
            .unwrap_or(1e-6)
            .max(1e-10),
    })
}

/// polish した解の採用可否を判定する (通常パス)。
///
/// 採用条件:
/// 1. `status` が収束済み (Optimal / LocallyOptimal) であること。
/// 2. `polished_obj` が有限かつ `incumbent_obj` より悪化していないこと。
fn is_polish_acceptable(
    status: &SolveStatus,
    polished_obj: f64,
    incumbent_obj: f64,
    gap_tol: f64,
) -> bool {
    let converged = matches!(status, SolveStatus::Optimal | SolveStatus::LocallyOptimal);
    if !converged || !polished_obj.is_finite() {
        return false;
    }
    let scale = 1.0_f64.max(incumbent_obj.abs());
    polished_obj <= incumbent_obj + gap_tol * scale
}

/// Structural EmptyCol mask: `eliminated_cols[j] = true` iff column `j` has
/// no non-zero entries in either `Q` or `A` (LP-style isolated variable).
///
/// This mirrors `attempt.rs`'s presolve col_map mask but derives it from
/// the CSC sparsity pattern directly, so it is valid for any box-restriction
/// of the same problem (B&B only changes bounds, never Q or A).
fn structural_empty_col_mask(problem: &QpProblem) -> Vec<bool> {
    let n = problem.num_vars;
    let a_ncols = problem.a.col_ptr.len().saturating_sub(1);
    let q_ncols = problem.q.col_ptr.len().saturating_sub(1);
    (0..n)
        .map(|j| {
            let a_empty = j >= a_ncols || problem.a.col_ptr[j + 1] == problem.a.col_ptr[j];
            let q_empty = j >= q_ncols || problem.q.col_ptr[j + 1] == problem.q.col_ptr[j];
            a_empty && q_empty
        })
        .collect()
}

/// SuboptimalSolution な polish 結果を KKT 残差で採用可否を追加判定する。
///
/// `prove_optimal` の duality_gap チェックが `user_eps` を僅かに上回り SuboptimalSolution
/// になった場合でも、KKT 残差が `user_eps * POLISH_KKT_ACCEPT_FACTOR` 以下なら dual
/// recovery 目的の polish として採用する。KKT 残差を独立に再計算し、gap のみ不合格な
/// 収束済み解と、真に収束不足の解を区別する。
fn is_polish_suboptimal_acceptable(
    polished: &SolverResult,
    problem: &QpProblem,
    incumbent_obj: f64,
    gap_tol: f64,
    user_eps: f64,
) -> bool {
    if !matches!(polished.status, SolveStatus::SuboptimalSolution) {
        return false;
    }
    if !polished.objective.is_finite() {
        return false;
    }
    let scale = 1.0_f64.max(incumbent_obj.abs());
    if polished.objective > incumbent_obj + gap_tol * scale {
        return false;
    }
    // dimension guard — mirrors prove_optimal (certificate.rs ~L64)
    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    if polished.solution.len() != problem.num_vars
        || polished.dual_solution.len() != problem.num_constraints
        || polished.bound_duals.len() != n_lb + n_ub
    {
        return false;
    }
    let kkt_tol = (user_eps * POLISH_KKT_ACCEPT_FACTOR).min(POLISH_KKT_ABS_CAP);
    let eliminated_cols = structural_empty_col_mask(problem);
    let view = ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &eliminated_cols,
    };
    let kkt = kkt_residual_rel(
        &view,
        &polished.solution,
        &polished.dual_solution,
        &polished.bound_duals,
    );
    let pf = kkt_primal_residual(&view, &polished.solution);
    let bv = kkt_bound_violation(&problem.bounds, &polished.solution);
    let comp = kkt_comp_residual(
        &view,
        &polished.solution,
        &polished.dual_solution,
        &polished.bound_duals,
    );
    let dsign = kkt_dual_sign_violation(
        &problem.constraint_types,
        &polished.dual_solution,
        &problem.bounds,
        &polished.bound_duals,
    );
    kkt <= kkt_tol && pf <= kkt_tol && bv <= kkt_tol && comp <= kkt_tol && dsign <= kkt_tol
}

/// 非凸 B&B 向け KKT recovery accept: 目的関数制約なしで IPM 収束 + 全 KKT を確認する。
///
/// sub-box incumbent は bound comp を原問題基準で破ることがある。polish が
/// Optimal/LocallyOptimal で全 KKT を満たし、かつ目的が incumbent から
/// `gap_tol * scale` 以上悪化していなければ採用する。obj 悪化 reject は
/// 非凸で polish が異なる local に流れた場合の incumbent 退化を防ぐ。
fn is_polish_kkt_recovery(
    polished: &SolverResult,
    problem: &QpProblem,
    incumbent_obj: f64,
    gap_tol: f64,
    user_eps: f64,
) -> bool {
    if !matches!(
        polished.status,
        SolveStatus::Optimal | SolveStatus::LocallyOptimal
    ) {
        return false;
    }
    if !polished.objective.is_finite() {
        return false;
    }
    let scale = 1.0_f64.max(incumbent_obj.abs());
    if polished.objective > incumbent_obj + gap_tol * scale {
        return false;
    }
    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let n_ub = problem
        .bounds
        .iter()
        .filter(|&&(_, ub)| ub.is_finite())
        .count();
    if polished.solution.len() != problem.num_vars
        || polished.dual_solution.len() != problem.num_constraints
        || polished.bound_duals.len() != n_lb + n_ub
    {
        return false;
    }
    let kkt_tol = (user_eps * POLISH_KKT_ACCEPT_FACTOR).min(POLISH_KKT_ABS_CAP);
    let eliminated_cols = structural_empty_col_mask(problem);
    let view = ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &eliminated_cols,
    };
    let kkt = kkt_residual_rel(
        &view,
        &polished.solution,
        &polished.dual_solution,
        &polished.bound_duals,
    );
    let pf = kkt_primal_residual(&view, &polished.solution);
    let bv = kkt_bound_violation(&problem.bounds, &polished.solution);
    let comp = kkt_comp_residual(
        &view,
        &polished.solution,
        &polished.dual_solution,
        &polished.bound_duals,
    );
    let dsign = kkt_dual_sign_violation(
        &problem.constraint_types,
        &polished.dual_solution,
        &problem.bounds,
        &polished.bound_duals,
    );
    kkt <= kkt_tol && pf <= kkt_tol && bv <= kkt_tol && comp <= kkt_tol && dsign <= kkt_tol
}

/// search state encapsulation: incumbent + 最終 result の組み立てを 1 箇所に集約。
struct SearchState {
    incumbent_result: SolverResult,
    incumbent_obj: f64,
    incumbent_sol: Vec<f64>,
    /// true when B&B found a sub-box incumbent better than root.
    incumbent_updated: bool,
}

impl SearchState {
    fn new(root: SolverResult) -> Self {
        let obj = root.objective;
        let sol = root.solution.clone();
        Self {
            incumbent_result: root,
            incumbent_obj: obj,
            incumbent_sol: sol,
            incumbent_updated: false,
        }
    }

    fn build_warm(&self) -> Option<QpWarmStart> {
        build_warm_from(&self.incumbent_result)
    }

    fn update_incumbent(&mut self, res: &SolverResult) {
        self.incumbent_obj = res.objective;
        self.incumbent_sol = res.solution.clone();
        self.incumbent_result = res.clone();
        self.incumbent_updated = true;
    }

    /// Dual recovery polish: re-solves on original bounds to fix sub-box-contaminated duals.
    ///
    /// Skipped only when root already returned Optimal/LocallyOptimal on original bounds.
    /// SuboptimalSolution root carries barrier-contaminated bound duals and must not be skipped.
    fn polish_incumbent_duals(
        &mut self,
        problem: &QpProblem,
        base_opts: &SolverOptions,
        gap_tol: f64,
        relax_for_nonconvex: bool,
    ) {
        if !self.incumbent_updated
            && matches!(
                self.incumbent_result.status,
                SolveStatus::Optimal | SolveStatus::LocallyOptimal
            )
        {
            return;
        }
        let Some(warm) = build_warm_from(&self.incumbent_result) else {
            return;
        };
        if warm.x.len() != problem.num_vars {
            return;
        }
        let mut opts = base_opts.clone();
        opts.warm_start_qp = Some(warm);
        opts.multistart = None;
        opts.global_optimization = None;
        // B&B 残時間を優先継承し、枯渇時のみ POLISH_TIMEOUT_SECS の fresh budget を使う
        // (timeout_secs 契約破り回避 + budget 枯渇時 fallback 両立)。
        let now = Instant::now();
        let polish_deadline = match base_opts.deadline {
            Some(d) if d > now => d,
            _ => now + Duration::from_secs_f64(POLISH_TIMEOUT_SECS),
        };
        opts.deadline = Some(polish_deadline);
        opts.timeout_secs = None;
        let user_eps = base_opts.ipm_eps();
        let polished = crate::qp::solve_qp_with(problem, &opts);
        if is_polish_acceptable(
            &polished.status,
            polished.objective,
            self.incumbent_obj,
            gap_tol,
        ) || is_polish_suboptimal_acceptable(
            &polished,
            problem,
            self.incumbent_obj,
            gap_tol,
            user_eps,
        ) || (relax_for_nonconvex
            && is_polish_kkt_recovery(&polished, problem, self.incumbent_obj, gap_tol, user_eps))
        {
            self.update_incumbent(&polished);
        }
    }

    /// Q が indefinite なら `NonconvexGlobal`、convex なら `Optimal` を set。
    ///
    /// B&B bound-gap closure だけでなく `prove_optimal` による全 KKT 条件
    /// (stationarity / primal_feasibility / bound_feasibility / complementarity /
    /// dual_sign / duality_gap) を検証する。検証に失敗した場合は
    /// LocallyOptimal / NonconvexLocal へ降格し証明書は付与しない。
    ///
    /// ## sentinel (no-op-fail)
    /// このメソッドの `prove_optimal` 呼び出しを除去すると、
    /// `finalize_proven_dual_gate_table` テストが FAIL する。
    fn finalize_proven(
        mut self,
        problem: &QpProblem,
        lower_bound: f64,
        q_indefinite: bool,
        gap_tol: f64,
        user_eps: f64,
    ) -> SolverResult {
        let eliminated_cols = structural_empty_col_mask(problem);
        let view = ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &eliminated_cols,
        };
        let duality_gap_rel = self
            .incumbent_result
            .duality_gap_rel
            .unwrap_or_else(|| compute_duality_gap_rel(problem, &self.incumbent_result));
        let cert_result = {
            let x = &self.incumbent_result.solution;
            let y = &self.incumbent_result.dual_solution;
            let z = &self.incumbent_result.bound_duals;
            prove_optimal(&view, x, y, z, duality_gap_rel, user_eps)
        };

        match cert_result {
            Ok(opt_cert) => {
                let scale = 1.0_f64.max(self.incumbent_obj.abs());
                let gap_rel = (self.incumbent_obj - lower_bound) / scale;
                self.incumbent_result.bound_gap_cert = Some(BoundGapCertificate::new(
                    self.incumbent_obj,
                    lower_bound,
                    gap_rel,
                    gap_tol,
                ));
                self.incumbent_result.opt_cert = Some(opt_cert);
                self.incumbent_result.status = if q_indefinite {
                    SolveStatus::NonconvexGlobal
                } else {
                    SolveStatus::Optimal
                };
                log::debug!(
                    "QP global proven: status={} obj={:.6e} lb={:.6e} gap_rel={:.3e}",
                    self.incumbent_result.status,
                    self.incumbent_obj,
                    lower_bound,
                    gap_rel
                );
            }
            Err(not_proven) => {
                self.incumbent_result.status = if q_indefinite {
                    SolveStatus::NonconvexLocal
                } else {
                    SolveStatus::LocallyOptimal
                };
                log::debug!(
                    "QP global gap-closed but KKT failed ({:?}): demoted to {}",
                    not_proven.failing_conditions,
                    self.incumbent_result.status,
                );
            }
        }
        self.incumbent_result
    }

    /// Q が indefinite なら `NonconvexLocal`、convex なら `LocallyOptimal` を set。
    /// (= IPM 単発 inertia 補正 `LocallyOptimal` と BB 打切 `NonconvexLocal` を分離)
    fn finalize_unproven(
        mut self,
        lower_bound: f64,
        nodes: usize,
        depth: usize,
        cfg: &GlobalOptimizationConfig,
        q_indefinite: bool,
    ) -> SolverResult {
        self.incumbent_result.status = if q_indefinite {
            SolveStatus::NonconvexLocal
        } else {
            SolveStatus::LocallyOptimal
        };
        let gap = self.incumbent_obj - lower_bound;
        log::debug!(
            "QP global unproven: status={} obj={:.6e} lb={:.6e} gap={:.3e} nodes={} depth={} tol={:.0e}",
            self.incumbent_result.status, self.incumbent_obj, lower_bound, gap, nodes, depth, cfg.gap_tol
        );
        self.incumbent_result
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use crate::test_kkt::assert_solver_invariants_qp;

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
            matches!(
                r.status,
                SolveStatus::Optimal
                    | SolveStatus::LocallyOptimal
                    | SolveStatus::NonconvexGlobal
                    | SolveStatus::NonconvexLocal
            ),
            "expected Optimal/Locally/NonconvexGlobal/NonconvexLocal, got {:?}",
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

    // ---- status 区別 sentinel ----------------------------------
    //
    // 観測: BB driver の return path で Q が convex (PSD) か indefinite かに応じて
    // `Optimal` vs `NonconvexGlobal` / `LocallyOptimal` vs `NonconvexLocal` が
    // 切り替わることを fact 検証 (no-op proof: finalize_proven / finalize_unproven
    // を全て `Optimal` 固定にすると下記 sentinel は FAIL する = mutation 検出)。

    fn diag_convex_1d(bnd: f64) -> QpProblem {
        // f = x², box [-bnd, bnd] → global min = 0 at x=0 (PSD)
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        QpProblem::new_all_le(q, vec![0.0], a, vec![], vec![(-bnd, bnd)]).unwrap()
    }

    #[test]
    fn convex_q_yields_optimal_not_nonconvex_global() {
        // PSD Q → BB が即収束 → Optimal (NonconvexGlobal でない)
        let p = diag_convex_1d(3.0);
        let r = solve_qp_global(&p, &opts(2.0), &GlobalOptimizationConfig::default());
        assert!(
            matches!(r.status, SolveStatus::Optimal),
            "convex Q must yield Optimal, got {:?}",
            r.status
        );
        assert_solver_invariants_qp(&r, &p);
    }

    #[test]
    fn indefinite_q_proven_yields_nonconvex_global() {
        // indefinite Q (-x²) + 十分な budget → NonconvexGlobal が出ることを確認。
        // 1D concave は root 即 corner = corner で proof 完了。
        let p = diag_concave_1d(2.0);
        let r = solve_qp_global(&p, &opts(5.0), &GlobalOptimizationConfig::default());
        assert!(
            matches!(r.status, SolveStatus::NonconvexGlobal),
            "indefinite Q + proven must yield NonconvexGlobal, got {:?}",
            r.status
        );
    }

    #[test]
    fn indefinite_q_unproven_yields_nonconvex_local() {
        // indefinite Q + 極小 budget (max_nodes=1, max_depth=1) → proof 取れず
        // → NonconvexLocal が出る。
        // 2D concave (= bowl 逆さ) + 各軸 [-1,1] を box にして root 分枝が必要に。
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0), (-1.0, 1.0)])
            .unwrap();
        // gap_tol を非現実的に厳しく (1e-12) + max_nodes=1 で proof 不能化
        let cfg = GlobalOptimizationConfig {
            gap_tol: 1e-12,
            max_depth: 1,
            max_nodes: 1,
            ..GlobalOptimizationConfig::default()
        };
        let r = solve_qp_global(&p, &opts(5.0), &cfg);
        assert!(
            matches!(r.status, SolveStatus::NonconvexLocal),
            "indefinite Q + unproven must yield NonconvexLocal, got {:?}",
            r.status
        );
    }

    #[test]
    fn is_q_indefinite_distinguishes_psd_and_indefinite() {
        // gershgorin_alpha(Q) > 0 を Q indefinite と判定する直接検証 (Status 分岐の root)
        let psd = diag_convex_1d(1.0);
        let indef = diag_concave_1d(1.0);
        assert!(!is_q_indefinite(&psd), "x² should be PSD");
        assert!(is_q_indefinite(&indef), "-x² should be indefinite");
    }

    // ---- BoundGapCertificate sentinels -----------------------------------------

    /// Proven QP global (convex Q) result carries BoundGapCertificate.
    ///
    /// Sentinel: removing `self.incumbent_result.bound_gap_cert = Some(...)` from
    /// `finalize_proven` leaves cert as `None` → this test FAILS.
    #[test]
    fn qp_global_proven_convex_has_bound_gap_cert() {
        let p = diag_convex_1d(3.0);
        let r = solve_qp_global(&p, &opts(2.0), &GlobalOptimizationConfig::default());
        assert!(matches!(r.status, SolveStatus::Optimal));
        let cert = r
            .bound_gap_cert
            .as_ref()
            .expect("proven QP global (Optimal) must carry BoundGapCertificate");
        assert!(
            cert.gap_rel() <= cert.gap_tol() + 1e-10,
            "gap_rel={:.3e} must be ≤ gap_tol={:.3e}",
            cert.gap_rel(),
            cert.gap_tol()
        );
    }

    /// Proven QP global (indefinite Q) result carries BoundGapCertificate.
    #[test]
    fn qp_global_proven_nonconvex_has_bound_gap_cert() {
        let p = diag_concave_1d(2.0);
        let r = solve_qp_global(&p, &opts(5.0), &GlobalOptimizationConfig::default());
        assert!(matches!(r.status, SolveStatus::NonconvexGlobal));
        let cert = r
            .bound_gap_cert
            .as_ref()
            .expect("proven QP global (NonconvexGlobal) must carry BoundGapCertificate");
        assert!(cert.gap_rel() <= cert.gap_tol() + 1e-10);
    }

    /// Unproven QP global result has no BoundGapCertificate.
    ///
    /// Sentinel: attaching cert unconditionally in `finalize_unproven` causes
    /// NonconvexLocal/LocallyOptimal to have Some(cert) → this test FAILS.
    #[test]
    fn qp_global_unproven_has_no_bound_gap_cert() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0), (-1.0, 1.0)])
            .unwrap();
        let cfg = GlobalOptimizationConfig {
            gap_tol: 1e-12,
            max_depth: 1,
            max_nodes: 1,
            ..GlobalOptimizationConfig::default()
        };
        let r = solve_qp_global(&p, &opts(5.0), &cfg);
        assert!(
            matches!(
                r.status,
                SolveStatus::NonconvexLocal | SolveStatus::LocallyOptimal
            ),
            "expected unproven status, got {:?}",
            r.status
        );
        assert!(
            r.bound_gap_cert.is_none(),
            "unproven must have no BoundGapCertificate"
        );
    }

    /// depth 超過 node の lb が remaining_lb に畳み込まれ、偽 proven を阻止する。
    ///
    /// Sentinel: `depth_discard_lb = depth_discard_lb.min(node_lb)` を除去すると
    /// depth 破棄後にキューが空になり `remaining_lb = f64::INFINITY` →
    /// `within_gap(inc_obj, ∞) = true` → NonconvexGlobal + cert が mint される (偽 proven)。
    /// この修正により remaining_lb = depth_discard_lb (≈ -2) になり、
    /// `within_gap(0, -2, 1e-12) = false` → NonconvexLocal、cert なし。
    #[test]
    fn depth_exceeded_lb_folds_into_remaining_lb_blocks_false_cert() {
        // 2D 凹 QP (Q=diag(-2,-2), [-1,1]²): IPM は x=0 に固着 (obj=0)、
        // コーナー最小値 = -2 には未収束。interval 下界 = -2。
        // max_depth=1 で深さ 1 のノードが depth_exceeded → depth_discard_lb=-2。
        // use_alpha_bb=false で alpha_bb が lb を 0 に引き上げないようにする。
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0), (-1.0, 1.0)])
            .unwrap();
        let cfg = GlobalOptimizationConfig {
            gap_tol: 1e-12,
            max_depth: 1,
            max_nodes: 10_000,
            use_alpha_bb: false,
            use_mccormick: false,
            ..GlobalOptimizationConfig::default()
        };
        let r = solve_qp_global(&p, &opts(10.0), &cfg);
        assert!(
            matches!(r.status, SolveStatus::NonconvexLocal),
            "depth-exceeded lb must block false proven: expected NonconvexLocal, got {:?}",
            r.status
        );
        assert!(
            r.bound_gap_cert.is_none(),
            "depth-exceeded unproven must have no BoundGapCertificate"
        );
    }

    /// 分枝 node 由来 incumbent の双対が元問題に整合する (相補性違反なし)。
    ///
    /// 3 変数 nonconvex QP (Q=diag(1,-1,-1)、A 第 1 行のみ非零、Le×3、box [-0.5,0.5]³)。
    /// 大域最小 x≈[0.2,-0.5,-0.5] は var0 が interior。B&B はこの incumbent を var0 を
    /// ub≈0.2 へ分枝した node で発見するため、polish なしでは `z_ub[0]` に分枝境界由来の
    /// 大きな bound dual が残り、元問題基準で `z_ub[0]·(ub−x0) ≈ 0.42` の相補性違反になる。
    ///
    /// Sentinel: `state.polish_incumbent_duals(...)` 呼び出しを除去すると相補性残差が
    /// `EPS_KKT` を超え FAIL する (= no-op proof)。`assert_solver_invariants_qp` は
    /// `NonconvexLocal` を skip するため、この相補性 gate がカバーする。
    #[test]
    fn branched_incumbent_duals_reconciled_to_original_box() {
        use crate::problem::ConstraintType;
        use crate::qp::ipm_solver::kkt::complementarity_residual_rel;
        use crate::qp::ipm_solver::outcome::ProblemView;
        use crate::test_kkt::EPS_KKT;

        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, -1.0, -1.0], 3, 3).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, 0.6], 3, 3).unwrap();
        let p = QpProblem::new(
            q,
            vec![0.0, 0.0, 0.0],
            a,
            vec![-0.5, 0.5, 1.0],
            vec![(-0.5, 0.5); 3],
            vec![ConstraintType::Le; 3],
        )
        .unwrap();
        let cfg = GlobalOptimizationConfig::default();
        let r = solve_qp_global(&p, &opts(8.0), &cfg);
        // 大域最小 (x0 interior の corner solution) に到達していること。
        assert!(
            (r.objective - (-0.23)).abs() < 1e-2,
            "expected global ≈ -0.23, got obj={:.4} status={:?}",
            r.objective,
            r.status
        );
        let view = ProblemView::from_problem(&p);
        let comp =
            complementarity_residual_rel(&view, &r.solution, &r.dual_solution, &r.bound_duals);
        assert!(
            comp < EPS_KKT,
            "branched-incumbent duals must satisfy original-box complementarity: comp={:.3e} > {:.3e} (status={:?})",
            comp,
            EPS_KKT,
            r.status
        );
    }

    // ---- polish guard sentinels ------------------------------------------------

    /// P2-a: polish は収束済み (Optimal/LocallyOptimal) のみ採用。
    ///
    /// Sentinel: `is_polish_acceptable` の `converged` 判定を除去すると、
    /// MaxIterations / SuboptimalSolution でも true を返すようになりこのテストが FAIL する。
    #[test]
    fn polish_acceptance_rejects_unconverged_status() {
        // 収束済み → 採用可
        assert!(is_polish_acceptable(&SolveStatus::Optimal, 0.0, 0.0, 1e-6));
        assert!(is_polish_acceptable(
            &SolveStatus::LocallyOptimal,
            0.0,
            0.0,
            1e-6
        ));
        // 未収束 → 棄却
        assert!(!is_polish_acceptable(
            &SolveStatus::MaxIterations,
            0.0,
            0.0,
            1e-6
        ));
        assert!(!is_polish_acceptable(
            &SolveStatus::SuboptimalSolution,
            0.0,
            0.0,
            1e-6
        ));
        // その他の失敗 status も棄却
        assert!(!is_polish_acceptable(
            &SolveStatus::Infeasible,
            0.0,
            0.0,
            1e-6
        ));
        assert!(!is_polish_acceptable(
            &SolveStatus::NumericalError,
            0.0,
            0.0,
            1e-6
        ));
        assert!(!is_polish_acceptable(&SolveStatus::Timeout, 0.0, 0.0, 1e-6));
    }

    /// P2-b: polish は obj が悪化した場合 (min なので polished_obj > incumbent_obj + tol) を棄却。
    ///
    /// Sentinel: 片側 guard を abs 判定 (`|polished - incumbent| <= tol`) に戻すと、
    /// 悪化ケース (`polished_obj > incumbent_obj + tol`) でも true を返しこのテストが FAIL する。
    #[test]
    fn polish_acceptance_rejects_worse_obj() {
        let gap_tol = 1e-4_f64;
        // incumbent_obj = -1.0 → scale = 1.0, 許容上限 = -1.0 + 1e-4
        let inc = -1.0_f64;
        let scale = 1.0_f64.max(inc.abs());
        let tol = gap_tol * scale; // 1e-4

        // 同点 → 採用可
        assert!(is_polish_acceptable(
            &SolveStatus::Optimal,
            inc,
            inc,
            gap_tol
        ));
        // 改善 (より小さい) → 採用可
        assert!(is_polish_acceptable(
            &SolveStatus::Optimal,
            inc - 0.5,
            inc,
            gap_tol
        ));
        // tol 以内の微小悪化 → 採用可 (dual 数値誤差)
        assert!(is_polish_acceptable(
            &SolveStatus::Optimal,
            inc + tol * 0.5,
            inc,
            gap_tol
        ));
        // tol を超える悪化 → 棄却
        assert!(!is_polish_acceptable(
            &SolveStatus::Optimal,
            inc + tol + 1e-10,
            inc,
            gap_tol
        ));
        // 明確な悪化 → 棄却
        assert!(!is_polish_acceptable(
            &SolveStatus::Optimal,
            0.0,
            inc,
            gap_tol
        ));
        assert!(!is_polish_acceptable(
            &SolveStatus::Optimal,
            1.0,
            inc,
            gap_tol
        ));

        // incumbent_obj = 0.0 → scale = 1.0, 許容上限 = 0 + 1e-4
        let inc = 0.0_f64;
        let scale = 1.0_f64.max(inc.abs());
        let tol = gap_tol * scale;
        assert!(is_polish_acceptable(
            &SolveStatus::Optimal,
            0.0,
            inc,
            gap_tol
        ));
        assert!(is_polish_acceptable(
            &SolveStatus::Optimal,
            -0.5,
            inc,
            gap_tol
        ));
        assert!(!is_polish_acceptable(
            &SolveStatus::Optimal,
            tol + 1e-10,
            inc,
            gap_tol
        ));

        // incumbent_obj = 100.0 → scale = 100.0, 許容上限 = 100.0 + 1e-2
        let inc = 100.0_f64;
        let scale = 1.0_f64.max(inc.abs());
        let tol = gap_tol * scale; // 1e-2
        assert!(is_polish_acceptable(
            &SolveStatus::Optimal,
            inc + tol * 0.5,
            inc,
            gap_tol
        ));
        assert!(!is_polish_acceptable(
            &SolveStatus::Optimal,
            inc + tol + 1e-10,
            inc,
            gap_tol
        ));
    }

    /// Invalid options are rejected at the global entry with NumericalError — not panic.
    ///
    /// Sentinel: removing `validate()` from `solve_qp_global_with_stats` causes
    /// negative `timeout_secs` to reach `Duration::from_secs_f64`, which **panics**.
    /// With the guard present, NumericalError is returned instead.
    #[test]
    fn invalid_options_rejected_at_global_entry() {
        let p = diag_concave_1d(2.0);
        let cfg = GlobalOptimizationConfig::default();
        let cases: &[(&str, SolverOptions)] = &[
            (
                "neg timeout_secs",
                SolverOptions {
                    timeout_secs: Some(-1.0),
                    ..Default::default()
                },
            ),
            (
                "inf timeout_secs",
                SolverOptions {
                    timeout_secs: Some(f64::INFINITY),
                    ..Default::default()
                },
            ),
            (
                "nan primal_tol",
                SolverOptions {
                    primal_tol: f64::NAN,
                    ..Default::default()
                },
            ),
            (
                "zero threads",
                SolverOptions {
                    threads: 0,
                    ..Default::default()
                },
            ),
        ];
        for (label, opts) in cases {
            let result = solve_qp_global(&p, opts, &cfg);
            assert_eq!(
                result.status,
                SolveStatus::NumericalError,
                "solve_qp_global with {label} must return NumericalError (not panic)"
            );
        }
    }

    // ---- is_polish_suboptimal_acceptable sentinels ----------------------------

    /// P2-a sentinel: dual_sign gate の no-op-fail 検証。
    ///
    /// stationarity/primal/bound/complementarity は全て kkt_tol 以下だが、
    /// Le 制約の dual が負 (wrong-sign) で dual_sign_violation が kkt_tol を超える場合、
    /// `is_polish_suboptimal_acceptable` は false を返す。
    ///
    /// Sentinel: `&& dsign <= kkt_tol` を除去すると true を返し、このテストが FAIL する
    /// (= no-op で FAIL する真の sentinel)。
    #[test]
    fn is_polish_suboptimal_acceptable_rejects_wrong_sign_duals() {
        use crate::problem::ConstraintType;

        // 1 変数、1 Le 制約、A = 0 行列 → stationarity/primal/comp は全て 0
        // bounds = (-inf, +inf) → bound_duals は空、bound_violation = 0
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let problem = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![0.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();

        // dual = -0.5: Le 制約に対して wrong-sign
        // dsign = 0.5 / (1 + 0.5) ≈ 0.333 >> kkt_tol (= (1e-6 * 100).min(1e-3) = 1e-4)
        let polished = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![0.0],
            dual_solution: vec![-0.5],
            bound_duals: vec![],
            ..SolverResult::default()
        };

        assert!(
            !is_polish_suboptimal_acceptable(&polished, &problem, 0.0, 0.1, 1e-6),
            "wrong-sign dual (y = -0.5 for Le constraint) must be rejected by dual_sign gate",
        );
    }

    /// P2-b sentinel: dimension guard — 次元不一致は false 返却。
    ///
    /// solution.len や dual_solution.len が problem 次元と合わない場合、
    /// 残差計算前に棄却する。
    #[test]
    fn is_polish_suboptimal_acceptable_rejects_mismatched_dimensions() {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap();
        let problem = QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![0.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
            vec![ConstraintType::Le],
        )
        .unwrap();

        // solution の長さが 1 (正しくは 2) → 次元不整合
        let polished_short_sol = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![0.0], // wrong: should be len 2
            dual_solution: vec![0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        assert!(
            !is_polish_suboptimal_acceptable(&polished_short_sol, &problem, 0.0, 0.1, 1e-6),
            "mismatched solution dimension must be rejected",
        );

        // dual_solution の長さが 0 (正しくは 1) → 次元不整合
        let polished_short_dual = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![0.0, 0.0],
            dual_solution: vec![], // wrong: should be len 1
            bound_duals: vec![],
            ..SolverResult::default()
        };
        assert!(
            !is_polish_suboptimal_acceptable(&polished_short_dual, &problem, 0.0, 0.1, 1e-6),
            "mismatched dual_solution dimension must be rejected",
        );
    }

    /// P2 sentinel: dimension guard rejects wrong bound_duals length before reaching
    /// kkt_dual_sign_violation.
    ///
    /// Sentinel: removing `|| polished.bound_duals.len() != n_lb + n_ub` from the
    /// dimension guard in `is_polish_suboptimal_acceptable` allows wrong-length z to
    /// reach `kkt_dual_sign_violation`, which returns 0.0 (z[0]=0.0 is non-violating:
    /// ≥0 check passes, viol=0) and the function would return true — FAIL.
    #[test]
    fn is_polish_suboptimal_acceptable_rejects_mismatched_bound_duals() {
        use crate::problem::ConstraintType;

        // 1 variable, lb=0 (finite), ub=∞ → n_lb=1, n_ub=0 → expected bound_duals.len()=1
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let problem = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![],
            vec![(0.0_f64, f64::INFINITY)],
            vec![ConstraintType::Le; 0],
        )
        .unwrap();

        // correct bound_duals len = 1 (n_lb=1, n_ub=0)
        // pass len=2 → mismatch → must return false
        let polished_wrong_bd = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![0.0],
            dual_solution: vec![],
            bound_duals: vec![0.0, 0.0], // wrong: should be len 1
            ..SolverResult::default()
        };
        assert!(
            !is_polish_suboptimal_acceptable(&polished_wrong_bd, &problem, 0.0, 0.1, 1e-6),
            "mismatched bound_duals length must be rejected by dimension guard",
        );
    }

    // ---- is_polish_kkt_recovery sentinels -------------------------------------

    /// 5 軸 sentinel: `is_polish_kkt_recovery` の accept/reject 全 gate を検証。
    ///
    /// Fixture: 1 var、A = 0、bounds = (-∞, +∞)、Le 制約 1 本、dual = 0
    /// → 全 KKT 残差は 0 で構成上 trivially feasible。各 axis を 1 つずつ破る。
    ///
    /// ## Sentinel (no-op-fail proof)
    /// - axis 1 (status): Optimal/LocallyOptimal 以外を許容に書き換えると case 1 が FAIL
    /// - axis 2 (dim): 次元 guard を除去すると case 2 が FAIL
    /// - axis 3 (KKT): wrong-sign dual を許容すると case 3 が FAIL
    /// - axis 5 (obj guard): `polished.objective > incumbent_obj + gap_tol*scale` の reject
    ///   を `if polished.objective < 0.0 { return false; }` (常に通過) に書き換えると
    ///   case 5 が FAIL する → axis 5 が load-bearing であることを確認 (P1-A 検証用)
    #[test]
    fn is_polish_kkt_recovery_five_axis_gates() {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let problem = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![0.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Le],
        )
        .unwrap();

        let incumbent_obj = 0.0_f64;
        let gap_tol = 0.1_f64;
        let user_eps = 1e-6_f64;

        // axis 4 (accept 通過): valid Optimal + 全 KKT 0 + obj 同点 → true
        let valid = SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![0.0],
            dual_solution: vec![0.0],
            bound_duals: vec![],
            ..SolverResult::default()
        };
        assert!(
            is_polish_kkt_recovery(&valid, &problem, incumbent_obj, gap_tol, user_eps),
            "axis 4 (accept): all gates pass must return true",
        );

        // axis 1 (status 棄却): SuboptimalSolution / MaxIterations / Timeout → false
        for bad_status in [
            SolveStatus::SuboptimalSolution,
            SolveStatus::MaxIterations,
            SolveStatus::Timeout,
            SolveStatus::NumericalError,
            SolveStatus::Infeasible,
        ] {
            let polished = SolverResult {
                status: bad_status.clone(),
                ..valid.clone()
            };
            assert!(
                !is_polish_kkt_recovery(&polished, &problem, incumbent_obj, gap_tol, user_eps),
                "axis 1 (status): {:?} must be rejected",
                bad_status,
            );
        }

        // axis 2 (dim mismatch): solution.len / dual_solution.len 不一致 → false
        let polished_short_sol = SolverResult {
            solution: vec![],
            ..valid.clone()
        };
        assert!(
            !is_polish_kkt_recovery(
                &polished_short_sol,
                &problem,
                incumbent_obj,
                gap_tol,
                user_eps
            ),
            "axis 2 (dim): wrong solution.len must be rejected",
        );
        let polished_short_dual = SolverResult {
            dual_solution: vec![],
            ..valid.clone()
        };
        assert!(
            !is_polish_kkt_recovery(
                &polished_short_dual,
                &problem,
                incumbent_obj,
                gap_tol,
                user_eps
            ),
            "axis 2 (dim): wrong dual_solution.len must be rejected",
        );

        // axis 3 (KKT failing): Le 制約に対して wrong-sign dual → dsign violation
        let polished_wrong_sign = SolverResult {
            dual_solution: vec![-0.5],
            ..valid.clone()
        };
        assert!(
            !is_polish_kkt_recovery(
                &polished_wrong_sign,
                &problem,
                incumbent_obj,
                gap_tol,
                user_eps
            ),
            "axis 3 (KKT): wrong-sign dual must be rejected",
        );

        // axis 5 (obj 悪化 reject, P1-A): polished.objective が incumbent + tol を上回る
        // → 非凸で polish が悪い local に流れたケース、incumbent 退化を防ぐ。
        // incumbent_obj=0, gap_tol=0.1, scale=1.0 → threshold=0.1。obj=1.0 → reject。
        let polished_worse = SolverResult {
            objective: 1.0,
            ..valid.clone()
        };
        assert!(
            !is_polish_kkt_recovery(
                &polished_worse,
                &problem,
                incumbent_obj,
                gap_tol,
                user_eps
            ),
            "axis 5 (obj guard): polished obj {} > incumbent + gap_tol*scale = {} must be rejected (P1-A)",
            polished_worse.objective,
            incumbent_obj + gap_tol * 1.0_f64.max(incumbent_obj.abs()),
        );

        // axis 5 二重確認: scale-aware 動作。incumbent=-10 → scale=10 → threshold=-10+1=-9.
        // obj=-9.5 (改善) → accept。obj=-8 (悪化、scale*tol 超過) → reject。
        let polished_within_tol = SolverResult {
            objective: -9.5,
            ..valid.clone()
        };
        assert!(
            is_polish_kkt_recovery(&polished_within_tol, &problem, -10.0, gap_tol, user_eps),
            "axis 5: improvement within scaled tol must be accepted",
        );
        let polished_outside_tol = SolverResult {
            objective: -8.0,
            ..valid.clone()
        };
        assert!(
            !is_polish_kkt_recovery(&polished_outside_tol, &problem, -10.0, gap_tol, user_eps),
            "axis 5: obj outside scaled tol (-8 > -10 + 0.1*10 = -9) must be rejected",
        );
    }

    // ---- finalize_proven dual-quality gate sentinels --------------------------

    /// Sentinel: 4 combinations of (convex/indefinite) × (good-dual/bad-dual).
    /// Removing the `prove_optimal` call from `finalize_proven` would always stamp
    /// Optimal/NonconvexGlobal regardless of dual quality; the bad-dual rows then
    /// FAIL their assertion (no-op-fail requirement, gate is load-bearing).
    #[test]
    fn finalize_proven_dual_gate_table() {
        // Convex: min x², box [-1, 1]
        let q_conv = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
        let a_empty = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let p_convex = QpProblem::new_all_le(
            q_conv,
            vec![0.0_f64],
            a_empty.clone(),
            vec![],
            vec![(-1.0_f64, 1.0_f64)],
        )
        .unwrap();

        // Indefinite: min -x², box [-1, 1]
        let q_indef = CscMatrix::from_triplets(&[0], &[0], &[-2.0_f64], 1, 1).unwrap();
        let p_indef = QpProblem::new_all_le(
            q_indef,
            vec![0.0_f64],
            a_empty,
            vec![],
            vec![(-1.0_f64, 1.0_f64)],
        )
        .unwrap();

        let user_eps = 1e-6_f64;
        let gap_tol = 1e-6_f64;

        // ── convex-good-dual: x=0, z=[0,0], gap=0 → Optimal ─────────────────
        let good_conv = SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![0.0_f64],
            dual_solution: vec![],
            bound_duals: vec![0.0_f64, 0.0_f64], // [z_lb, z_ub]
            duality_gap_rel: Some(0.0),
            ..Default::default()
        };
        let r =
            SearchState::new(good_conv).finalize_proven(&p_convex, 0.0, false, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::Optimal,
            "convex-good-dual must be Optimal"
        );
        assert!(
            r.bound_gap_cert.is_some(),
            "Optimal must carry bound_gap_cert"
        );
        assert!(r.opt_cert.is_some(), "Optimal must carry opt_cert");

        // ── convex-bad-dual: z=[100,-100], large gap → LocallyOptimal ────────
        // Sentinel: without the gate this row returns Optimal, failing the assertion.
        let bad_conv = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![0.0_f64],
            dual_solution: vec![],
            bound_duals: vec![100.0_f64, -100.0_f64], // wrong sign + stationarity violation
            duality_gap_rel: Some(0.5),
            ..Default::default()
        };
        let r =
            SearchState::new(bad_conv).finalize_proven(&p_convex, 0.0, false, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::LocallyOptimal,
            "convex-bad-dual must be demoted to LocallyOptimal"
        );
        assert!(
            r.bound_gap_cert.is_none(),
            "demoted must have no bound_gap_cert"
        );
        assert!(r.opt_cert.is_none(), "demoted must have no opt_cert");

        // ── indefinite-good-dual: x=1 (ub active), z=[0,2] → NonconvexGlobal
        let good_indef = SolverResult {
            status: SolveStatus::Optimal,
            objective: -1.0,
            solution: vec![1.0_f64],
            dual_solution: vec![],
            bound_duals: vec![0.0_f64, 2.0_f64], // z_lb=0, z_ub=2 (stationarity: -2+2=0)
            duality_gap_rel: Some(0.0),
            ..Default::default()
        };
        let r =
            SearchState::new(good_indef).finalize_proven(&p_indef, -1.0, true, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::NonconvexGlobal,
            "indefinite-good-dual must be NonconvexGlobal"
        );
        assert!(
            r.bound_gap_cert.is_some(),
            "NonconvexGlobal must carry bound_gap_cert"
        );
        assert!(r.opt_cert.is_some(), "NonconvexGlobal must carry opt_cert");

        // ── indefinite-bad-dual: z=[50,50] → stationarity fails → NonconvexLocal
        // Sentinel: without the gate this row returns NonconvexGlobal, failing the assertion.
        let bad_indef = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: -1.0,
            solution: vec![1.0_f64],
            dual_solution: vec![],
            bound_duals: vec![50.0_f64, 50.0_f64], // stationarity: -2 - 50 + 50 = -2 ≠ 0
            duality_gap_rel: Some(0.5),
            ..Default::default()
        };
        let r =
            SearchState::new(bad_indef).finalize_proven(&p_indef, -1.0, true, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::NonconvexLocal,
            "indefinite-bad-dual must be demoted to NonconvexLocal"
        );
        assert!(
            r.bound_gap_cert.is_none(),
            "demoted must have no bound_gap_cert"
        );
        assert!(r.opt_cert.is_none(), "demoted must have no opt_cert");
    }

    /// P1 regression: `finalize_proven` must not false-demote a valid incumbent when
    /// presolve eliminated an EmptyCol variable.
    ///
    /// ## Setup
    /// Problem: `min -x₀² + x₁`, Q=diag([-2,0]), c=[0,1], A=∅, x₀∈[-1,1], x₁∈[0,1].
    /// x₁ is EmptyCol (Q[:,1]=0, A[:,1]=0, c[1]=1>0 → presolve fixes x₁=lb=0).
    ///
    /// KKT at (x₀=1, x₁=0):
    ///   stationarity x₀: (-2)·1 + (-z_lb_x0 + z_ub_x0) = -2 + 2 = 0   ✓  (z_ub=2,z_lb=0)
    ///   stationarity x₁: 0 + c[1] + 0 = 1.0  (spurious if x₁ not skipped)
    ///
    /// With `eliminated_cols=&[]` (bug): kkt for x₁ = 1.0 ≫ eps → false-demote → NonconvexLocal.
    /// With structural mask (fix): x₁ has a_empty∧q_empty → skipped → kkt=0 → NonconvexGlobal.
    ///
    /// ## Sentinel (no-op-fail)
    /// Changing `structural_empty_col_mask` to return `vec![false; n]` (= disable the mask)
    /// causes this test to FAIL: kkt for x₁ = 1.0 → prove_optimal rejects → NonconvexLocal.
    #[test]
    fn finalize_proven_empty_col_not_false_demoted() {
        // Problem with EmptyCol x₁ (c[1]=1.0 > 0 → postsolve sets x₁=lb=0, z=0 by convention)
        let q = CscMatrix::from_triplets(&[0], &[0], &[-2.0_f64], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0_f64, 1.0_f64], // c[1]=1.0: spurious stationarity = 1.0 without mask
            a,
            vec![],
            vec![(-1.0_f64, 1.0_f64), (0.0_f64, 1.0_f64)],
        )
        .unwrap();

        // KKT-valid solution: x₀=1 (ub active), x₁=0 (EmptyCol fixed at lb).
        // bound_duals = [z_lb_x0=0, z_lb_x1=0, z_ub_x0=2, z_ub_x1=0]
        // stationarity x₀: Q[0,0]·1 + c[0] - z_lb_x0 + z_ub_x0 = -2 + 0 + 2 = 0  ✓
        // duality_gap = 0: primal=-1, dual=-0.5·(-2)·1 - 1·2 = 1-2 = -1  ✓
        let incumbent = SolverResult {
            status: SolveStatus::Optimal,
            objective: -1.0,
            solution: vec![1.0_f64, 0.0_f64],
            dual_solution: vec![],
            bound_duals: vec![0.0_f64, 0.0_f64, 2.0_f64, 0.0_f64],
            duality_gap_rel: Some(0.0),
            ..Default::default()
        };

        let user_eps = 1e-6_f64;
        let gap_tol = 1e-6_f64;

        let r =
            SearchState::new(incumbent).finalize_proven(&problem, -1.0, true, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::NonconvexGlobal,
            "EmptyCol incumbent must not be false-demoted: expected NonconvexGlobal, got {:?}. \
             Sentinel: structural_empty_col_mask returning vec![false; n] causes this FAIL \
             because kkt for x₁ (c=1,z=0) gives 1.0 ≫ eps.",
            r.status,
        );
        assert!(r.opt_cert.is_some(), "NonconvexGlobal must carry opt_cert");
    }

    /// Regression: proptest seed a46bde58 — PD Q (Gershgorin false positive) must satisfy KKT.
    ///
    /// The a46bde58 problem has a truly PD Q (Cholesky succeeds) but Gershgorin reports
    /// indefinite (Q[0,0]=0.16 < off-diag sum 0.358). The global solver must return
    /// complementarity < 1e-3 for NonconvexLocal/NonconvexGlobal status.
    #[test]
    fn proptest_seed_a46bde58_kkt_regression() {
        use crate::problem::ConstraintType;

        let rows = vec![0usize, 1, 2, 0, 1, 2, 0, 1, 2];
        let cols = vec![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let vals = vec![
            0.16000000000000003_f64,
            -0.03915063848637796,
            0.3192145885469365,
            -0.03915063848637796,
            0.460208173753392,
            -0.17436676978450188,
            0.3192145885469365,
            -0.17436676978450188,
            1.0576743304356357,
        ];
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        let c = vec![
            0.653536572287863_f64,
            -0.010836684577960307,
            -1.445105979349165,
        ];
        let a_rows = vec![1usize];
        let a_cols = vec![0usize];
        let a_vals = vec![0.3965134170122774_f64];
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 2, 3).unwrap();
        let b = vec![2.477268994387253_f64, -0.7050675637248502];
        let bounds = vec![
            (-0.5_f64, 0.5),
            (-2.519765539465491, 2.519765539465491),
            (-0.6799197663837497, 0.6799197663837497),
        ];
        let cts = vec![ConstraintType::Le, ConstraintType::Ge];
        let problem = QpProblem::new(q, c, a, b, bounds, cts).unwrap();

        let mut o = SolverOptions::default();
        o.timeout_secs = Some(15.0);
        let cfg = GlobalOptimizationConfig::default();
        let res = solve_qp_global(&problem, &o, &cfg);

        assert!(
            matches!(
                res.status,
                SolveStatus::NonconvexLocal | SolveStatus::NonconvexGlobal
            ),
            "expected NonconvexLocal/NonconvexGlobal, got {:?}",
            res.status
        );

        let elim = structural_empty_col_mask(&problem);
        let view = ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &elim,
        };
        let comp = kkt_comp_residual(&view, &res.solution, &res.dual_solution, &res.bound_duals);
        let stat = kkt_residual_rel(&view, &res.solution, &res.dual_solution, &res.bound_duals);
        let pf = kkt_primal_residual(&view, &res.solution);
        assert!(
            comp < 1e-3,
            "a46bde58: complementarity={:.3e} >= 1e-3 (status={:?} stat={:.3e} pf={:.3e})",
            comp,
            res.status,
            stat,
            pf,
        );
    }
}
