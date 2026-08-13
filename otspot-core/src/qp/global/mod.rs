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
pub(crate) mod dual_recovery;
pub(crate) mod node;
pub(crate) mod pruning;
pub(crate) mod tree;

use crate::options::{GlobalOptimizationConfig, QpWarmStart, SolverOptions};
use crate::problem::certificate::BoundGapCertificate;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::certificate::prove_optimal;
use crate::qp::ipm_solver::core::compute_duality_gap_rel;
use crate::qp::ipm_solver::kkt::{
    bound_violation as kkt_bound_violation,
    complementarity_componentwise_rel as kkt_comp_componentwise,
    complementarity_residual_rel as kkt_comp_residual, kkt_residual_rel,
    primal_residual_rel as kkt_primal_residual,
};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::kkt_resid::dual_sign_violation as kkt_dual_sign_violation;
use crate::qp::problem::QpProblem;
use std::time::{Duration, Instant};

use bound::{
    interval_quadratic_bounds, is_feasible_result, is_verified_feasible_point,
    solve_local_upper_bound,
};
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

#[cfg(test)]
thread_local! {
    /// Test-only cancel injection for the B&B loop's own per-iteration stop
    /// check (the loop previously broke only on `deadline_reached`,
    /// never observing `cancel_flag`/`external_stop_requested` directly —
    /// a Ctrl-C racing in mid-search, far from the wall-clock deadline, was
    /// invisible to the loop's own control flow. It still eventually drained
    /// because every already-queued node's *local solve* independently
    /// honors `cancel_flag` via `solve_qp_with`, but only after popping and
    /// discarding the entire backlog one node at a time). Counts calls to
    /// `test_maybe_cancel_at_loop_top`; once the count reaches
    /// `CANCEL_AFTER_LOOP_ITER`, flips the real `cancel_flag` `AtomicBool`
    /// a test threads through `SolverOptions::cancel_flag`.
    static LOOP_ITER_COUNT: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
    static CANCEL_AFTER_LOOP_ITER: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    /// Test-only cancel injection isolated to `finalize_search_outcome`'s own
    /// `external_stop_requested()` backstop, independent of the loop-top
    /// check above: fired exactly once, right after the B&B loop/polish have
    /// already finished cleanly (queue drained, nothing discarded), to
    /// simulate a cancel racing in during that narrow window. Verifies the
    /// backstop demotes an otherwise-clean proof to unproven rather than
    /// minting a `BoundGapCertificate` over a search that observed a stop.
    static CANCEL_BEFORE_FINALIZE: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn test_maybe_cancel_at_loop_top(opts: &SolverOptions) {
    let count = LOOP_ITER_COUNT.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if CANCEL_AFTER_LOOP_ITER.with(std::cell::Cell::get) == Some(count) {
        if let Some(flag) = &opts.cancel_flag {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
fn test_maybe_cancel_before_finalize(opts: &SolverOptions) {
    if CANCEL_BEFORE_FINALIZE.with(std::cell::Cell::get) {
        if let Some(flag) = &opts.cancel_flag {
            flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

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
///   `remaining_lb`: 終了時点の未探索領域下界 (`tree.best_lower_bound()` と
///   `discard_lb` の min)。全探索完了 (`!halted_early`) なら未探索領域が無いため
///   `f64::INFINITY`。`within_gap`/`BoundGapCertificate` の guard 状態に左右されない
///   直接値のため、fold 系修正 (discard_lb への畳み込み) をピンポイントで検証できる。
#[derive(Debug, Clone, Copy, Default)]
pub struct GlobalStats {
    pub nodes_processed: usize,
    pub max_depth_seen: usize,
    pub pruned: usize,
    pub remaining_lb: f64,
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
    // Presolve-independent bound-consistency guard for this separate public
    // entry: an empty box (lb > ub) is trivially infeasible and must be reported
    // before the spatial B&B touches the box (branching / α-BB clamps assume
    // lb <= ub). Mirrors the guards in `solve_lp_with` / `dispatch_solve_qp`.
    if crate::problem::first_infeasible_bound(&problem.bounds).is_some() {
        return (SolverResult::infeasible(), GlobalStats::default());
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

    // 1. root local solve (= 初期 incumbent 候補)。上界 incumbent は「feasible な
    // 点の目的値」で健全なので、非収束 status (Stalled 等) でも点そのものが
    // feasible と検証できれば採用する (status は信用しない)。
    let root_solve = solve_local_upper_bound(problem, &root_bounds, &shared_opts, None);
    let root_solve = match classify_root_usability(root_solve, problem, &shared_opts) {
        Ok(usable) => usable,
        Err(fallback) => return (*fallback, stats),
    };

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
        // root incumbent が gap 以内 = 未探索領域なしで全証明完了。`finalize_search_
        // outcome` の `!halted_early` 経路と同じ「open region なし」sentinel を公開し、
        // GlobalStats::default() 由来の捏造 `0.0` を返さない。
        stats.remaining_lb = f64::INFINITY;
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
            let result = finalize_unbranchable_root(
                state,
                problem,
                &shared_opts,
                &mut stats,
                root_lb,
                cfg,
                q_indefinite,
                user_eps,
            );
            return (result, stats);
        }
        Some(j) => {
            let warm = state.build_warm();
            let (l, r) = split_node(&root_node, j, root_x[j], warm, None);
            tree.push(l);
            tree.push(r);
        }
    }

    // 深さ上限超過、または local solve が unusable (Infeasible 以外) で discard した
    // node があるか。どちらも未探索領域を残すため queue 空だけでは完全探索と言えない。
    let mut search_incomplete = false;
    // 上記で discard した node の node_lb の min を保持する。これが未探索領域の下界に
    // なるため remaining_lb に畳み込む必要がある。
    let mut discard_lb: f64 = f64::INFINITY;

    while let Some(node) = tree.pop() {
        #[cfg(test)]
        test_maybe_cancel_at_loop_top(&shared_opts);
        if shared_opts.external_stop_requested() {
            fold_interrupted_node(&node, &mut search_incomplete, &mut discard_lb);
            break;
        }
        if stats.nodes_processed >= cfg.max_nodes {
            fold_interrupted_node(&node, &mut search_incomplete, &mut discard_lb);
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
        if !is_node_result_usable(&res, problem, user_eps) {
            if !node_discard_is_conclusive(&res.status) {
                // 「box に解があるか不明」なだけで空の証明ではないため、node_lb を
                // discard_lb に畳み込み未探索領域として残す (完全探索を偽装しない)。
                search_incomplete = true;
                discard_lb = discard_lb.min(node_lb);
            }
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
            // この node の lb を discard_lb に畳み込む (remaining_lb に反映する)。
            search_incomplete = true;
            discard_lb = discard_lb.min(node_lb);
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

    #[cfg(test)]
    test_maybe_cancel_before_finalize(&shared_opts);

    let result = finalize_search_outcome(
        problem,
        &tree,
        state,
        &mut stats,
        discard_lb,
        search_incomplete,
        &shared_opts,
        cfg,
        q_indefinite,
        user_eps,
    );
    (result, stats)
}

/// polish / dual recovery の sub-solve が使う deadline。
///
/// B&B の残時間を優先継承し、枯渇時のみ [`POLISH_TIMEOUT_SECS`] の fresh budget を
/// 使う (`timeout_secs` 契約破りの回避と、budget 枯渇時の fallback の両立)。
/// deadline 到達で B&B が終了した経路では `base_opts.deadline` は既に過去なので、
/// これを補修せずに sub-solve へ渡すと即 Timeout になる。
fn polish_deadline(base_opts: &SolverOptions) -> Instant {
    let now = Instant::now();
    match base_opts.deadline {
        Some(d) if d > now => d,
        _ => now + Duration::from_secs_f64(POLISH_TIMEOUT_SECS),
    }
}

/// 分枝不能な root の終端処理。
///
/// gap 以内なら未探索領域なしで証明完了 (`remaining_lb = ∞` sentinel)。そうでなければ
/// root box 全体が未証明領域として残るので、その下界 (`root_lb`) を公開する
/// (`GlobalStats::default()` 由来の `0.0` は「documented 下界」でも
/// 「open region なし」でもない捏造値)。
#[allow(clippy::too_many_arguments)]
fn finalize_unbranchable_root(
    mut state: SearchState,
    problem: &QpProblem,
    shared_opts: &SolverOptions,
    stats: &mut GlobalStats,
    root_lb: f64,
    cfg: &GlobalOptimizationConfig,
    q_indefinite: bool,
    user_eps: f64,
) -> SolverResult {
    // 証明可否によらず先に polish/復元を通す (B&B ループ側の終端と同じ順序)。
    // 未証明経路だけ乗数の復元を飛ばすと、局所最適だが barrier 由来の乗数を持つ
    // root が `local_kkt_within` で `FeasiblePoint` へ落ちる。
    state.polish_incumbent_duals(problem, shared_opts, cfg.gap_tol, q_indefinite);
    if within_gap(state.incumbent_obj, root_lb, cfg.gap_tol) {
        stats.remaining_lb = f64::INFINITY;
        return state.finalize_proven(problem, root_lb, q_indefinite, cfg.gap_tol, user_eps);
    }
    stats.remaining_lb = root_lb;
    state.finalize_unproven(
        problem,
        root_lb,
        stats.nodes_processed,
        0,
        cfg,
        q_indefinite,
        user_eps,
    )
}

/// Builds the final `SolverResult` from the B&B loop's exit state: queue
/// non-empty, `search_incomplete`, or deadline/cancel/`max_nodes` reached all
/// mean `halted_early`, in which case `remaining_lb` (folded into `stats.
/// remaining_lb` — see `GlobalStats`'s doc) decides `finalize_proven` vs
/// `finalize_unproven`. Otherwise the queue drained cleanly and
/// `incumbent_obj` is the global optimum.
///
/// `opts.external_stop_requested()` (横展開, mirrors `conic::nonconvex::global_core`'s
/// terminal-classification backstop) is a defense-in-depth backstop, not the
/// primary cancel detection: the loop's own top-of-iteration check already
/// folds a mid-search cancel into `search_incomplete` before this function is
/// ever called. This second check only matters for the narrow race where the
/// tree drains exactly as cancel fires, in which case treating the search as
/// `halted_early` (rather than a clean, fully-proven completion) is the
/// conservative choice.
#[allow(clippy::too_many_arguments)]
fn finalize_search_outcome(
    problem: &QpProblem,
    tree: &BBTree,
    state: SearchState,
    stats: &mut GlobalStats,
    discard_lb: f64,
    search_incomplete: bool,
    opts: &SolverOptions,
    cfg: &GlobalOptimizationConfig,
    q_indefinite: bool,
    user_eps: f64,
) -> SolverResult {
    let halted_early = !tree.is_empty()
        || search_incomplete
        || opts.external_stop_requested()
        || stats.nodes_processed >= cfg.max_nodes;

    if !halted_early {
        // queue 空 = 全探索完了 → 未探索領域なし、incumbent_obj が global。
        stats.remaining_lb = f64::INFINITY;
        let inc_obj = state.incumbent_obj;
        return state.finalize_proven(problem, inc_obj, q_indefinite, cfg.gap_tol, user_eps);
    }
    // 未探索領域の下界: queue に残った node の最小 lb と、深さ上限/discard で破棄
    // した node の lb の両方を考慮する。どちらの領域も「未証明」であるため min を取る。
    let remaining_lb = tree
        .best_lower_bound()
        .unwrap_or(f64::INFINITY)
        .min(discard_lb);
    stats.remaining_lb = remaining_lb;
    let proven = within_gap(state.incumbent_obj, remaining_lb, cfg.gap_tol);
    let inc_obj = state.incumbent_obj;
    if proven {
        let lb_for_proof = remaining_lb.min(inc_obj);
        state.finalize_proven(problem, lb_for_proof, q_indefinite, cfg.gap_tol, user_eps)
    } else {
        state.finalize_unproven(
            problem,
            remaining_lb,
            stats.nodes_processed,
            stats.max_depth_seen,
            cfg,
            q_indefinite,
            user_eps,
        )
    }
}

/// Folds `node`'s inherited `lower_bound` into `discard_lb` and marks the
/// search incomplete, for the loop-top `deadline_reached`/`max_nodes` breaks
/// in `solve_qp_global_with_stats`. `tree.pop()` is best-bound-first, so
/// `node` holds the smallest lower_bound of everything still unexplored at
/// this instant (including whatever remains in `tree`); discarding it
/// without folding would leave `remaining_lb` (computed from `tree.
/// best_lower_bound()`, which no longer sees this node) skewed optimistic,
/// which can make an unproven gap look closed. Mirrors `mip::
/// check_stop_conditions`, which already folds the just-popped node's bound
/// before its own deadline/max_nodes break.
fn fold_interrupted_node(node: &BBNode, search_incomplete: &mut bool, discard_lb: &mut f64) {
    *search_incomplete = true;
    *discard_lb = discard_lb.min(node.lower_bound);
}

/// Classifies the root local solve's usability as a starting incumbent.
/// `Ok` continues the B&B; `Err(fallback)` is the exact early-return result
/// for `solve_qp_global_with_stats`'s two direct-return paths.
///
/// `is_feasible_result`/`is_verified_feasible_point` trust `status` alone
/// (Optimal/LocallyOptimal/SuboptimalSolution) or a verified point, without
/// checking `objective`/`solution` are finite (Codex review, P2, follow-up
/// to `within_gap`'s false-Optimal fix) — mirrors `qcqp_route::
/// is_clean_convex_outcome`'s `Optimal` invariant. A root that claims
/// feasible but is non-finite must not reach `SearchState::new` (which
/// `assert!`s this — see its doc): report it honestly as `NumericalError`
/// rather than forwarding the raw corrupt result (whose `status` would still
/// read `Optimal`/`SuboptimalSolution`). A root that never claimed feasible
/// (Infeasible/NumericalError/Unbounded/NonConvex/Timeout) propagates as-is.
fn classify_root_usability(
    root_solve: SolverResult,
    problem: &QpProblem,
    opts: &SolverOptions,
) -> Result<SolverResult, Box<SolverResult>> {
    let claims_feasible = is_feasible_result(&root_solve.status)
        || is_verified_feasible_point(problem, &root_solve.solution, opts.ipm_eps());
    if !claims_feasible {
        return Err(Box::new(root_solve));
    }
    if !root_solve.is_finite_candidate() {
        return Err(Box::new(SolverResult::numerical_error()));
    }
    Ok(root_solve)
}

/// Whether a node's local upper-bound solve is usable to adopt as an
/// improving incumbent: independently re-verifies feasibility (status-
/// trusted or point-verified) AND finiteness (`is_finite_candidate`) —
/// same gap as `classify_root_usability` (Codex review, P2), so a node
/// result claiming Optimal/SuboptimalSolution but non-finite is unusable
/// (folded into `discard_lb`/`search_incomplete` at the call site, exactly
/// like any other unusable status) rather than handed to `update_incumbent`.
fn is_node_result_usable(res: &SolverResult, problem: &QpProblem, eps: f64) -> bool {
    res.is_finite_candidate()
        && (is_feasible_result(&res.status)
            || is_verified_feasible_point(problem, &res.solution, eps))
}

/// unusable node (`!res_usable`) の discard が「完全探索」扱いにできるか。
///
/// `Infeasible` は A x <= b (+ box) の実行不可能証明 (Q に依存しない線形可否判定) で
/// あり、この box には本当に解が存在しない — node_lb を畳み込む必要はない。
/// それ以外 (`NumericalError`/`Timeout`/`NonConvex`/未検証 iterate の `Stalled`・
/// `MaxIterations`) は「box に解があるか不明」なだけで空の証明ではないため
/// `false` を返し、呼び出し側に node_lb の畳み込みを要求する。
fn node_discard_is_conclusive(status: &SolveStatus) -> bool {
    matches!(status, SolveStatus::Infeasible)
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
    let a_ncols = problem.a.col_ptr().len().saturating_sub(1);
    let q_ncols = problem.q.col_ptr().len().saturating_sub(1);
    (0..n)
        .map(|j| {
            let a_empty = j >= a_ncols || problem.a.col_ptr()[j + 1] == problem.a.col_ptr()[j];
            let q_empty = j >= q_ncols || problem.q.col_ptr()[j + 1] == problem.q.col_ptr()[j];
            a_empty && q_empty
        })
        .collect()
}

/// 局所最適性の主張に必要な KKT 5 条件 (stationarity / primal feasibility /
/// bound feasibility / complementarity / dual sign) を元問題空間で検証する。
///
/// `prove_optimal` との違いは duality gap を含まないこと: gap は大域証明の条件で
/// あり、`LocallyOptimal` / `NonconvexLocal` の主張には要らない。閾値は
/// `(user_eps * POLISH_KKT_ACCEPT_FACTOR).min(POLISH_KKT_ABS_CAP)`。
fn local_kkt_within(problem: &QpProblem, res: &SolverResult, user_eps: f64) -> bool {
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
    if res.solution.len() != problem.num_vars
        || res.dual_solution.len() != problem.num_constraints
        || res.bound_duals.len() != n_lb + n_ub
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
    let kkt = kkt_residual_rel(&view, &res.solution, &res.dual_solution, &res.bound_duals);
    let pf = kkt_primal_residual(&view, &res.solution);
    let bv = kkt_bound_violation(&problem.bounds, &res.solution);
    // `prove_optimal` と同じ基準: 全体スケールで正規化した残差だけでは、大きな正当な
    // 乗数が分母を膨らませて人工 bound 由来の違反 (成分単位では O(1)) を隠せる。
    let comp = kkt_comp_residual(&view, &res.solution, &res.dual_solution, &res.bound_duals).max(
        kkt_comp_componentwise(&view, &res.solution, &res.dual_solution, &res.bound_duals),
    );
    let dsign = kkt_dual_sign_violation(
        &problem.constraint_types,
        &res.dual_solution,
        &problem.bounds,
        &res.bound_duals,
    );
    kkt <= kkt_tol && pf <= kkt_tol && bv <= kkt_tol && comp <= kkt_tol && dsign <= kkt_tol
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
    local_kkt_within(problem, polished, user_eps)
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
    local_kkt_within(problem, polished, user_eps)
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
    /// Codex review (P2, follow-up to `within_gap`'s false-Optimal fix):
    /// `root` must already be `is_finite_candidate()` — enforced by an
    /// `assert!` rather than silently degrading, because this constructor
    /// has exactly one production call site (`solve_qp_global_with_stats`,
    /// immediately after its `root_solve.is_finite_candidate()` gate); a
    /// non-finite `root` reaching here means that gate itself regressed,
    /// which is a caller bug, not a data-dependent condition to route
    /// around. Every existing caller (including all `#[cfg(test)]` call
    /// sites) already passes a finite root.
    fn new(root: SolverResult) -> Self {
        assert!(
            root.is_finite_candidate(),
            "SearchState::new requires a finite-candidate root (objective and every \
             solution component finite); got objective={} solution={:?} — caller must \
             gate via SolverResult::is_finite_candidate() before construction",
            root.objective,
            root.solution
        );
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

    /// Adopt `res` as the new incumbent. Returns whether it was actually
    /// adopted.
    ///
    /// Codex review (P2): rejects (no-op) a non-`is_finite_candidate()` `res`
    /// — mirrors `mip::MipState::consider`. The one production call site
    /// (the B&B loop in `solve_qp_global_with_stats`) already gates on
    /// `res.is_finite_candidate()` via `res_usable` before calling this, so
    /// this is defense-in-depth against a future call site that forgets to.
    fn update_incumbent(&mut self, res: &SolverResult) -> bool {
        if !res.is_finite_candidate() {
            return false;
        }
        self.incumbent_obj = res.objective;
        self.incumbent_sol = res.solution.clone();
        self.incumbent_result = res.clone();
        self.incumbent_updated = true;
        true
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
        opts.deadline = Some(polish_deadline(base_opts));
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
        self.recover_incumbent_duals_in_place(problem, base_opts, user_eps);
    }

    /// 再解 polish で直らなかった sub-box 乗数を、`x` を固定したまま復元する。
    ///
    /// 非凸では polish の再解が別の local optimum へ滑って棄却されることがあり、
    /// そのとき incumbent には node の (元問題では無効な) 乗数が残る。`x` は動かさず
    /// 元問題の active set 上で乗数を解き直し、局所 KKT を満たす場合だけ差し替える
    /// (目的値・主解は不変なので incumbent の品質を落とさない)。
    fn recover_incumbent_duals_in_place(
        &mut self,
        problem: &QpProblem,
        base_opts: &SolverOptions,
        user_eps: f64,
    ) {
        if local_kkt_within(problem, &self.incumbent_result, user_eps) {
            return;
        }
        // deadline はここで取り直す: 先行する polish の再解が残り budget を
        // 使い切っていることがあり、その状態を継承すると復元 LP が即 Timeout する
        // (復元が最も要る場面ほど失敗する)。
        let mut opts = base_opts.clone();
        opts.deadline = Some(polish_deadline(base_opts));
        opts.timeout_secs = None;
        let Some(recovered) = dual_recovery::recover_duals_at_fixed_x(
            problem,
            &self.incumbent_result.solution,
            &opts,
            user_eps,
        ) else {
            return;
        };
        let mut candidate = self.incumbent_result.clone();
        candidate.dual_solution = recovered.y;
        candidate.bound_duals = recovered.bound_duals;
        // 乗数を入れ替えたので、node 解が持っていた duality gap は無効。残すと
        // `finalize_proven` が古い値を優先し、復元後の乗数と無関係な gap で
        // 証明済み/未証明を判定してしまう。
        candidate.duality_gap_rel = None;
        if local_kkt_within(problem, &candidate, user_eps) {
            self.incumbent_result = candidate;
        }
    }

    /// Q が indefinite なら `NonconvexGlobal`、convex なら `Optimal` を set。
    ///
    /// B&B bound-gap closure だけでなく `prove_optimal` による全 KKT 条件
    /// (stationarity / primal_feasibility / bound_feasibility / complementarity /
    /// dual_sign / duality_gap) を検証する。検証に失敗した場合、incumbent が
    /// 品質ゲート (`is_feasible_result`) を通っていれば LocallyOptimal /
    /// NonconvexLocal へ降格、通っていなければ (Stalled/MaxIterations 由来の
    /// feasibility-only 点) `FeasiblePoint` を set する。いずれも証明書は付与しない。
    /// (`finalize_unproven` と対称の品質ゲート — gap が閉じたかどうかは incumbent
    /// 自体の検証状態を変えない。)
    ///
    /// ## sentinel (no-op-fail)
    /// このメソッドの `prove_optimal` 呼び出しを除去すると、
    /// `finalize_proven_bad_dual_demotes_to_feasible_point` テストが FAIL する。
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
                // 降格先の `LocallyOptimal`/`NonconvexLocal` も局所最適性の主張なので、
                // status ゲートに加えて元問題空間の局所 KKT を要求する
                // (`finalize_unproven` と同じ contract)。
                let local_ok = local_kkt_within(problem, &self.incumbent_result, user_eps);
                self.incumbent_result.status =
                    if !is_feasible_result(&self.incumbent_result.status) || !local_ok {
                        SolveStatus::FeasiblePoint
                    } else if q_indefinite {
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

    /// incumbent が品質ゲート (`is_feasible_result`: Optimal/LocallyOptimal/
    /// SuboptimalSolution) を通り、かつ元問題空間で局所 KKT 条件
    /// ([`local_kkt_within`]) を満たしていれば、Q が indefinite なら
    /// `NonconvexLocal`、convex なら `LocallyOptimal` を set。どちらか一方でも
    /// 欠けた incumbent (feasibility 検証のみで採用された Stalled/MaxIterations
    /// 由来の点、または node の sub-box 乗数しか持たない点) には局所最適性を
    /// 主張できないため `FeasiblePoint` を set する。
    /// (= IPM 単発 inertia 補正 `LocallyOptimal` と BB 打切 `NonconvexLocal` を分離)
    ///
    /// status だけを見て `NonconvexLocal` を刻むと、分枝で加えた人工 bound の乗数を
    /// 持つ incumbent が「局所最適解」を名乗る (`prop_nonconvex_qp_kkt_invariants_
    /// constrained` が KKT max 2.06e-1 で検出した欠陥)。乗数の復元は
    /// `recover_incumbent_duals_in_place` が先に試み、それでも満たせない点だけが
    /// ここで降格する。
    fn finalize_unproven(
        mut self,
        problem: &QpProblem,
        lower_bound: f64,
        nodes: usize,
        depth: usize,
        cfg: &GlobalOptimizationConfig,
        q_indefinite: bool,
        user_eps: f64,
    ) -> SolverResult {
        let local_ok = local_kkt_within(problem, &self.incumbent_result, user_eps);
        self.incumbent_result.status =
            if !is_feasible_result(&self.incumbent_result.status) || !local_ok {
                SolveStatus::FeasiblePoint
            } else if q_indefinite {
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
    use crate::test_kkt::assert_solver_invariants_qp;
    use otspot_num::sparse::CscMatrix;

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

    /// A root-level gap closure leaves no open region, so the public statistic
    /// must use the documented `+inf` sentinel rather than `Default`'s `0.0`.
    ///
    /// Sentinel: removing the root-return assignment reports exactly `0.0`.
    #[test]
    fn root_gap_exit_reports_no_open_region_remaining_lb() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let p = QpProblem::new_all_le(q, vec![0.0], a, vec![], vec![(-1.0, 1.0)]).unwrap();

        let (_, stats) =
            solve_qp_global_with_stats(&p, &opts(5.0), &GlobalOptimizationConfig::default());

        assert_eq!(stats.nodes_processed, 1);
        assert!(
            stats.remaining_lb.is_infinite() && stats.remaining_lb.is_sign_positive(),
            "root proof leaves no open region, expected +inf; got {}",
            stats.remaining_lb
        );
    }

    /// If the root box cannot be split but its gap remains open, the whole
    /// root box is the remaining unproven region and its actual lower bound
    /// must be exposed.
    ///
    /// Sentinel: removing the assignment reports `0.0` instead of the
    /// negative root lower bound.
    #[test]
    fn unbranchable_root_exit_reports_root_lower_bound() {
        let p = diag_concave_1d(5e-7);
        let cfg = GlobalOptimizationConfig {
            gap_tol: 1e-16,
            use_alpha_bb: false,
            use_mccormick: false,
            ..GlobalOptimizationConfig::default()
        };

        let (_, stats) = solve_qp_global_with_stats(&p, &opts(5.0), &cfg);

        assert_eq!(stats.nodes_processed, 1);
        assert!(
            stats.remaining_lb < 0.0,
            "unbranchable root's interval lower bound must be retained; got {}",
            stats.remaining_lb
        );
        assert!(
            (stats.remaining_lb + 2.5e-13).abs() <= 1e-25,
            "expected exact root lower bound -2.5e-13, got {}",
            stats.remaining_lb
        );
    }

    /// Sentinel: the global (nonconvex B&B) public entry must report Infeasible
    /// for an empty variable box (lb > ub), never a panic or a spurious incumbent
    /// from the spatial branch/α-BB clamps. The entry's own `first_infeasible_bound`
    /// guard is first-line; the root local solve routes through the guarded
    /// `solve_qp_with` as a transitive backup. Reverting the whole fix (the
    /// `is_valid_bound_pair` relaxation) makes construction reject the box and this
    /// test's `.expect(...)` panic instead.
    #[test]
    fn solve_qp_global_empty_box_lb_gt_ub_is_infeasible() {
        // f = -x²  on the empty box [5, 3].
        let q = CscMatrix::from_triplets(&[0], &[0], &[-2.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let p = QpProblem::new_all_le(q, vec![0.0], a, vec![], vec![(5.0, 3.0)])
            .expect("lb>ub box must be ACCEPTED at construction");
        let r = solve_qp_global(&p, &opts(5.0), &GlobalOptimizationConfig::default());
        assert_eq!(
            r.status,
            SolveStatus::Infeasible,
            "global empty box must be Infeasible, got {:?}",
            r.status
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

    /// SENTINEL (task 7, review of `fix/inf-incumbent-optimal`): the loop's
    /// `deadline_reached`/`max_nodes` breaks discard the just-popped node
    /// without folding its `lower_bound` into `discard_lb`. `tree.pop()` is
    /// best-bound-first, so that node holds the smallest pending bound of
    /// everything unexplored; dropping it lets `remaining_lb` skew optimistic
    /// (`mip::check_stop_conditions` already folds the popped bound before
    /// its own break — this closes the same gap here).
    ///
    /// Asserts `GlobalStats::remaining_lb` directly, not `res.status`/
    /// `bound_gap_cert`: a status-based version was found *vacuous* once
    /// tentatively merged with `fix/inf-incumbent-optimal` (07c8b954), whose
    /// `within_gap` guard independently demotes a `+inf` `remaining_lb` to
    /// "not proven" regardless of whether this fold ran. `remaining_lb` is
    /// the raw pre-`within_gap` value, so this assertion is guard-independent.
    ///
    /// Problem: `Q=diag(-2,-2)` on box `[-1,1]²` plus constraint `x0>=0.1`
    /// (invisible to the interval bound). Root's local solve is trapped at
    /// the interior saddle (`objective=-1`), splitting at `x=0`. The
    /// `x∈[-1,0]` child is entirely infeasible — a conclusive discard that
    /// still counts toward `nodes_processed` — so the `x∈[0,1]` sibling's
    /// `max_nodes=2` check fires as the *last* tree node, `discard_lb` still
    /// `+inf`; `root_lb = -2` is exact, so the equality check below is exact.
    ///
    /// Verified in 3 states (task report): solo, and tentatively merged with
    /// 07c8b954 (`git merge --no-commit --no-ff`, never committed) — both
    /// revert-fail (`remaining_lb == +inf`) identically; both pass
    /// (`remaining_lb == -2.0`) with the fold applied.
    #[test]
    fn interrupt_break_folds_popped_node_bound_into_remaining_lb() {
        use crate::problem::ConstraintType;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        // x0 >= 0.1, i.e. -x0 <= -0.1.
        let a = CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 2).unwrap();
        let p = QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![-0.1],
            vec![(-1.0, 1.0), (-1.0, 1.0)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let cfg = GlobalOptimizationConfig {
            gap_tol: 1e-12,
            max_depth: 30,
            max_nodes: 2,
            use_alpha_bb: false,
            use_mccormick: false,
            ..GlobalOptimizationConfig::default()
        };
        let (_, stats) = solve_qp_global_with_stats(&p, &opts(10.0), &cfg);

        assert_eq!(stats.nodes_processed, 2);
        assert_eq!(
            stats.remaining_lb, -2.0,
            "the last-standing sibling's own inherited bound (root_lb = -2, \
             exact) must be folded into remaining_lb; got {} (+inf means the \
             fold in the max_nodes break never ran)",
            stats.remaining_lb
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
    }

    /// B&B が deadline 到達で終了した経路では `base_opts.deadline` は既に過去。
    /// polish / dual recovery の sub-solve にそのまま渡すと即 Timeout になるため、
    /// 枯渇時は fresh budget へ補修されなければならない。
    ///
    /// ## Sentinel (no-op-fail)
    /// `polish_deadline` を `base_opts.deadline` の素通しに戻すと、期限切れケースの
    /// assert が FAIL する。
    #[test]
    fn polish_deadline_repairs_exhausted_budget() {
        let mut expired = SolverOptions::default();
        expired.deadline = Some(Instant::now() - Duration::from_secs(1));
        let repaired = polish_deadline(&expired);
        assert!(
            repaired > Instant::now(),
            "期限切れ deadline は fresh budget へ補修されるべき"
        );

        // 残時間があるときは B&B の deadline をそのまま継承する (契約破り防止)。
        let live_deadline = Instant::now() + Duration::from_secs(3600);
        let mut live = SolverOptions::default();
        live.deadline = Some(live_deadline);
        assert_eq!(polish_deadline(&live), live_deadline);
    }

    /// proptest seed 43b5a909 の問題。x* = (lb₀, 0.4960) は元問題の KKT 点だが、
    /// そこから再解 polish を掛けると別の (より悪い) local optimum へ滑るため、
    /// 乗数は「x を固定した復元」でしか直せない。
    fn seed_43b5a909_problem() -> QpProblem {
        use crate::problem::ConstraintType;

        let q = CscMatrix::from_triplets(
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[
                -1.326916674369138_f64,
                0.5594909414228733,
                0.5594909414228733,
                -0.029627559427708422,
            ],
            2,
            2,
        )
        .unwrap();
        let a = CscMatrix::from_triplets(
            &[0, 0],
            &[0, 1],
            &[0.9008102281303486_f64, -0.5517737255870893],
            2,
            2,
        )
        .unwrap();
        QpProblem::new(
            q,
            vec![0.5093837208820018_f64, 0.5260703864092237],
            a,
            vec![-1.9876786522125578_f64, 0.5],
            vec![
                (-1.9027544688666718_f64, 1.9027544688666718),
                (-1.2237729585496602, 1.2237729585496602),
            ],
            vec![ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap()
    }

    /// 期限切れ deadline のまま終わった B&B でも、sub-box 乗数は復元されて
    /// 局所 KKT を満たす (復元 LP が期限切れ deadline を継承しないこと)。
    ///
    /// incumbent は seed 43b5a909 の実測値: 行 0 が active なのに y = 0、非活性な
    /// x₁ 上界に z = 0.553 が残った node 解。再解 polish はここから別の local へ
    /// 滑って棄却されるので、直せるのは固定 x の復元だけ。
    ///
    /// ## Sentinel (no-op-fail)
    /// `polish_incumbent_duals` が復元へ `base_opts` (期限切れ deadline) を渡す
    /// 実装に戻すと、復元 LP が即 Timeout になり FAIL する。
    #[test]
    fn polish_recovers_duals_under_exhausted_deadline() {
        let problem = seed_43b5a909_problem();
        let contaminated = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: -3.641986_f64,
            solution: vec![-1.9027544688666718_f64, 0.49596048587182223],
            dual_solution: vec![0.0, 0.0],
            bound_duals: vec![3.311665752000553_f64, 0.0, 0.0, 0.5531976014425323],
            ..Default::default()
        };
        assert!(
            !local_kkt_within(&problem, &contaminated, 1e-6),
            "test premise: 汚染された乗数は局所 KKT を満たさない"
        );

        let mut state = SearchState::new(contaminated);
        state.incumbent_updated = true; // node 由来 incumbent = polish 対象
        let mut opts = SolverOptions::default();
        opts.deadline = Some(Instant::now() - Duration::from_secs(1));
        state.polish_incumbent_duals(&problem, &opts, 1e-6, true);
        assert!(
            local_kkt_within(&problem, &state.incumbent_result, 1e-6),
            "期限切れ budget でも乗数を復元すべき: y = {:?} z = {:?}",
            state.incumbent_result.dual_solution,
            state.incumbent_result.bound_duals
        );
    }

    /// 大きな正当乗数がスケールを膨らませ、人工 bound 由来の complementarity 違反を
    /// 全体正規化残差が隠す構成 (Codex P1)。
    ///
    /// n=2, Q=diag(2,2), 制約行なし、x=[1, 0.5] (x₀ は上界 active、x₁ は内点)。
    /// z_ub = [1e5, 1] とし、c を手計算で stationarity ちょうど 0 に合わせる:
    ///   x₀: 2·1 + c₀ + 1e5 = 0 → c₀ = −100002
    ///   x₁: 2·0.5 + c₁ + 1  = 0 → c₁ = −2
    /// 非活性な x₁ 上界の乗数 1 は complementarity 1·|0.5−1| = 0.5 の違反だが、
    /// 全体正規化 (|zᵀx| ≈ 1e5) では 5e-6 に薄まる。成分単位では O(0.5) のまま。
    ///
    /// ## Sentinel (no-op-fail)
    /// `local_kkt_within` から componentwise complementarity を外すと通ってしまい
    /// この test が FAIL する。
    #[test]
    fn local_kkt_within_catches_componentwise_complementarity() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0_f64, 2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![-100002.0_f64, -2.0],
            a,
            vec![],
            vec![(0.0_f64, 1.0), (0.0, 1.0)],
        )
        .unwrap();
        let res = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![1.0_f64, 0.5],
            dual_solution: vec![],
            // layout: [z_lb0, z_lb1, z_ub0, z_ub1]
            bound_duals: vec![0.0_f64, 0.0, 1e5, 1.0],
            ..Default::default()
        };
        assert!(
            !local_kkt_within(&problem, &res, 1e-6),
            "非活性 bound の乗数 1 (成分 comp 0.5) を見逃してはならない"
        );
    }

    /// 乗数を復元したら sub-box 由来の `duality_gap_rel` は捨てる (Codex P1)。
    /// 残すと `finalize_proven` が復元後の乗数と無関係な古い gap で判定する。
    ///
    /// ## Sentinel (no-op-fail)
    /// `candidate.duality_gap_rel = None` を外すと stale な `Some(0.0)` が残り FAIL する。
    #[test]
    fn recovered_duals_drop_stale_duality_gap() {
        let problem = seed_43b5a909_problem();
        let contaminated = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: -3.641986_f64,
            solution: vec![-1.9027544688666718_f64, 0.49596048587182223],
            dual_solution: vec![0.0, 0.0],
            bound_duals: vec![3.311665752000553_f64, 0.0, 0.0, 0.5531976014425323],
            duality_gap_rel: Some(0.0), // sub-box solve 由来の stale な値
            ..Default::default()
        };
        let mut state = SearchState::new(contaminated);
        state.incumbent_updated = true;
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(10.0);
        state.polish_incumbent_duals(&problem, &opts, 1e-6, true);

        assert!(
            local_kkt_within(&problem, &state.incumbent_result, 1e-6),
            "test premise: 乗数が復元されていること"
        );
        assert!(
            state.incumbent_result.duality_gap_rel.is_none(),
            "復元後は stale gap を持ち越さない: {:?}",
            state.incumbent_result.duality_gap_rel
        );
    }

    /// min x², box [−1, 1] (Q = [2])。x* = 0 が内点最適で g = 0、乗数はすべて 0。
    fn convex_box_1d() -> QpProblem {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        QpProblem::new_all_le(q, vec![0.0], a, vec![], vec![(-1.0, 1.0)]).unwrap()
    }

    /// `finalize_proven` が `prove_optimal` に落ちたとき、乗数が局所 KKT すら
    /// 満たさない incumbent は `FeasiblePoint` まで降格する (局所最適性も主張しない)。
    ///
    /// ## Sentinel (no-op-fail)
    /// `finalize_proven` から `prove_optimal` 呼び出しを外すと両行が Optimal /
    /// NonconvexGlobal になり FAIL する。`local_kkt_within` ゲートを外すと
    /// LocallyOptimal / NonconvexLocal になり FAIL する。
    #[test]
    fn finalize_proven_bad_dual_demotes_to_feasible_point() {
        let user_eps = 1e-6_f64;
        let gap_tol = 1e-6_f64;

        // convex: x=0, z=[100,-100] → 符号違反 + stationarity 違反、gap も大。
        let bad_conv = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: 0.0,
            solution: vec![0.0_f64],
            dual_solution: vec![],
            bound_duals: vec![100.0_f64, -100.0_f64],
            duality_gap_rel: Some(0.5),
            ..Default::default()
        };
        let r = SearchState::new(bad_conv).finalize_proven(
            &convex_box_1d(),
            0.0,
            false,
            gap_tol,
            user_eps,
        );
        assert_eq!(
            r.status,
            SolveStatus::FeasiblePoint,
            "convex-bad-dual must be demoted to FeasiblePoint"
        );
        assert!(
            r.bound_gap_cert.is_none(),
            "demoted must have no bound_gap_cert"
        );
        assert!(r.opt_cert.is_none(), "demoted must have no opt_cert");

        // indefinite: x=1, z=[50,50] → stationarity −2 − 50 + 50 = −2 ≠ 0 (手計算)。
        let bad_indef = SolverResult {
            status: SolveStatus::SuboptimalSolution,
            objective: -1.0,
            solution: vec![1.0_f64],
            dual_solution: vec![],
            bound_duals: vec![50.0_f64, 50.0_f64],
            duality_gap_rel: Some(0.5),
            ..Default::default()
        };
        let r = SearchState::new(bad_indef).finalize_proven(
            &indefinite_box_1d(),
            -1.0,
            true,
            gap_tol,
            user_eps,
        );
        assert_eq!(
            r.status,
            SolveStatus::FeasiblePoint,
            "indefinite-bad-dual must be demoted to FeasiblePoint"
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

    /// min −x², box [−1, 1] (Q = [−2])。x* = 1 で上界 active、g = −2 なので
    /// 手計算オラクルは `bound_duals = [z_lb, z_ub] = [0, 2]` (stationarity −2 + 2 = 0)。
    /// `z_ub` を 0 にすると残差 2 が残る。
    fn indefinite_box_1d() -> QpProblem {
        let q = CscMatrix::from_triplets(&[0], &[0], &[-2.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        QpProblem::new_all_le(q, vec![0.0], a, vec![], vec![(-1.0, 1.0)]).unwrap()
    }

    /// Sentinel (P2-4): `finalize_unproven` の incumbent が品質ゲート
    /// (`is_feasible_result`) を通っていない (= Stalled/MaxIterations 由来の
    /// feasibility-only 点) 場合、LocallyOptimal/NonconvexLocal ではなく
    /// `FeasiblePoint` を返さなければならない。
    #[test]
    fn finalize_unproven_feasibility_only_incumbent_yields_feasible_point() {
        let cfg = GlobalOptimizationConfig::default();
        let problem = indefinite_box_1d();

        for status in [SolveStatus::Stalled, SolveStatus::MaxIterations] {
            for q_indefinite in [false, true] {
                let unverified = SolverResult {
                    status: status.clone(),
                    objective: -1.0,
                    solution: vec![1.0_f64],
                    bound_duals: vec![0.0, 2.0],
                    ..Default::default()
                };
                let r = SearchState::new(unverified).finalize_unproven(
                    &problem,
                    -1.0,
                    1,
                    0,
                    &cfg,
                    q_indefinite,
                    1e-6,
                );
                assert_eq!(
                    r.status,
                    SolveStatus::FeasiblePoint,
                    "{status:?} incumbent (q_indefinite={q_indefinite}) never passed a \
                     quality gate; must not claim LocallyOptimal/NonconvexLocal, got {:?}",
                    r.status,
                );
            }
        }
    }

    /// 品質ゲートを通り、かつ元問題空間で局所 KKT を満たす incumbent は
    /// LocallyOptimal / NonconvexLocal を主張できる。
    #[test]
    fn finalize_unproven_kkt_consistent_incumbent_claims_local_optimality() {
        let cfg = GlobalOptimizationConfig::default();
        let problem = indefinite_box_1d();

        for (q_indefinite, expected) in [
            (false, SolveStatus::LocallyOptimal),
            (true, SolveStatus::NonconvexLocal),
        ] {
            let verified = SolverResult {
                status: SolveStatus::SuboptimalSolution,
                objective: -1.0,
                solution: vec![1.0_f64],
                bound_duals: vec![0.0, 2.0],
                ..Default::default()
            };
            let r = SearchState::new(verified).finalize_unproven(
                &problem,
                -1.0,
                1,
                0,
                &cfg,
                q_indefinite,
                1e-6,
            );
            assert_eq!(
                r.status, expected,
                "KKT 整合な quality-gated incumbent は {expected:?} を名乗れる, got {:?}",
                r.status,
            );
        }
    }

    /// Sentinel: 品質ゲートを通っていても、乗数が元問題の KKT を満たさない incumbent
    /// (= 分枝で加えた人工 bound の乗数しか持たない node 解) は局所最適性を主張できない。
    /// `finalize_unproven` の `local_kkt_within` ゲートを外すと NonconvexLocal /
    /// LocallyOptimal が返り、この test が FAIL する。
    #[test]
    fn finalize_unproven_kkt_inconsistent_incumbent_yields_feasible_point() {
        let cfg = GlobalOptimizationConfig::default();
        let problem = indefinite_box_1d();

        for q_indefinite in [false, true] {
            // z_ub = 0 では stationarity 残差が |g| = 2 残る (手計算)。
            let contaminated = SolverResult {
                status: SolveStatus::SuboptimalSolution,
                objective: -1.0,
                solution: vec![1.0_f64],
                bound_duals: vec![0.0, 0.0],
                ..Default::default()
            };
            let r = SearchState::new(contaminated).finalize_unproven(
                &problem,
                -1.0,
                1,
                0,
                &cfg,
                q_indefinite,
                1e-6,
            );
            assert_eq!(
                r.status,
                SolveStatus::FeasiblePoint,
                "KKT を満たさない乗数の incumbent (q_indefinite={q_indefinite}) は \
                 局所最適性を主張してはならない, got {:?}",
                r.status,
            );
        }
    }

    /// Sentinel (P1-a, `finalize_unproven` と対称): `finalize_proven` の
    /// `Err(not_proven)` 分岐 (gap は閉じたが KKT 証明に失敗) が、品質ゲート
    /// (`is_feasible_result`) を通っていない incumbent (Stalled/MaxIterations 由来の
    /// feasibility-only 点) にも LocallyOptimal/NonconvexLocal を mint していた。
    /// gap closure は incumbent の検証状態を変えないため `FeasiblePoint` を返す
    /// べき。品質ゲートを通った incumbent (`finalize_proven_dual_gate_table` の
    /// bad-dual 行) は従来通り LocallyOptimal/NonconvexLocal のまま。
    #[test]
    fn finalize_proven_kkt_fail_on_feasibility_only_incumbent_yields_feasible_point() {
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
        // wrong-sign z_ub (-100) → dual_sign 違反 → prove_optimal は必ず Err。
        let bad_dual = vec![100.0_f64, -100.0_f64];

        for status in [SolveStatus::Stalled, SolveStatus::MaxIterations] {
            let unverified_conv = SolverResult {
                status: status.clone(),
                objective: 0.0,
                solution: vec![0.0_f64],
                dual_solution: vec![],
                bound_duals: bad_dual.clone(),
                duality_gap_rel: Some(0.5),
                ..Default::default()
            };
            let r = SearchState::new(unverified_conv)
                .finalize_proven(&p_convex, 0.0, false, gap_tol, user_eps);
            assert_eq!(
                r.status,
                SolveStatus::FeasiblePoint,
                "{status:?} incumbent (convex, KKT-fail) must not claim LocallyOptimal, got {:?}",
                r.status,
            );
            assert!(
                r.bound_gap_cert.is_none(),
                "FeasiblePoint must have no BoundGapCertificate"
            );
            assert!(r.opt_cert.is_none(), "FeasiblePoint must have no opt_cert");

            let unverified_indef = SolverResult {
                status: status.clone(),
                objective: 0.0,
                solution: vec![0.0_f64],
                dual_solution: vec![],
                bound_duals: bad_dual.clone(),
                duality_gap_rel: Some(0.5),
                ..Default::default()
            };
            let r = SearchState::new(unverified_indef)
                .finalize_proven(&p_indef, 0.0, true, gap_tol, user_eps);
            assert_eq!(
                r.status,
                SolveStatus::FeasiblePoint,
                "{status:?} incumbent (indefinite, KKT-fail) must not claim NonconvexLocal, got {:?}",
                r.status,
            );
        }
    }

    /// SENTINEL (P2, Codex review follow-up to `within_gap`'s false-Optimal
    /// fix): `SearchState::new` — the sole point every top-level solve seeds
    /// its starting incumbent from — must never accept a non-finite
    /// candidate. Mirrors `mip::MipState::consider`'s guard and
    /// `qcqp_route::is_clean_convex_outcome`'s `Optimal` invariant
    /// (`objective.is_finite() && x.iter().all(finite)`).
    ///
    /// `new`'s one production call site (`solve_qp_global_with_stats`) now
    /// gates on `root_solve.is_finite_candidate()` before calling `new` at
    /// all, so this exercises `new`'s own internal enforcement directly,
    /// independent of that gate — defense-in-depth against a future caller
    /// that forgets it. Revert the `assert!` in `SearchState::new` to see
    /// this stop panicking (verified).
    #[test]
    #[should_panic(expected = "finite-candidate")]
    fn search_state_new_rejects_non_finite_objective() {
        let poisoned = SolverResult {
            status: SolveStatus::Optimal,
            objective: f64::INFINITY,
            solution: vec![0.0_f64],
            ..Default::default()
        };
        let _ = SearchState::new(poisoned);
    }

    /// SENTINEL companion: `SearchState::new` must also reject a candidate
    /// whose objective is finite but whose solution carries a non-finite
    /// component (the other half of `is_finite_candidate()`).
    #[test]
    #[should_panic(expected = "finite-candidate")]
    fn search_state_new_rejects_non_finite_solution_component() {
        let poisoned = SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![f64::NAN],
            ..Default::default()
        };
        let _ = SearchState::new(poisoned);
    }

    /// SENTINEL (P2): `update_incumbent` — the B&B loop's per-node incumbent
    /// adoption — must reject a non-finite candidate rather than adopting it,
    /// leaving the last-known-good incumbent untouched. Its one production
    /// call site already gates on `res.is_finite_candidate()` via `res_usable`
    /// before calling this, so this is defense-in-depth, tested directly.
    ///
    /// Revert the `is_finite_candidate()` check in `update_incumbent` to see
    /// this fail: the poisoned candidate gets adopted (`updated == true`,
    /// `incumbent_obj == f64::INFINITY`) (verified).
    #[test]
    fn update_incumbent_rejects_non_finite_candidate() {
        let good_root = SolverResult {
            status: SolveStatus::Optimal,
            objective: 5.0,
            solution: vec![1.0_f64],
            ..Default::default()
        };
        let mut state = SearchState::new(good_root);
        let poisoned = SolverResult {
            status: SolveStatus::Optimal,
            objective: f64::INFINITY,
            solution: vec![0.0_f64],
            ..Default::default()
        };
        let updated = state.update_incumbent(&poisoned);
        assert!(
            !updated,
            "a non-finite candidate must be rejected, not adopted"
        );
        assert_eq!(
            state.incumbent_obj, 5.0,
            "incumbent must remain the last finite value"
        );
        assert_eq!(state.incumbent_sol, vec![1.0_f64]);
        assert!(!state.incumbent_updated);
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

    /// Regression: proptest seed 43b5a909 — B&B incumbent の sub-box 乗数が
    /// そのまま返り、`NonconvexLocal` を名乗りながら KKT max 2.063e-1 だった欠陥。
    ///
    /// x* = (lb₀, 0.4960) は元問題の KKT 点で、root の単発 local solve は
    /// KKT 2.68e-9 の乗数 (y₀ ≈ −1.0026, z_lb₀ ≈ 2.4085) を出せる。B&B は同じ x を
    /// node 解として採用したため、行 0 が active であるにもかかわらず y = 0 で、
    /// 非活性な x₁ の上界に z = 0.553 が残っていた。
    ///
    /// ## Sentinel (no-op-fail)
    /// `polish_incumbent_duals` の `recover_incumbent_duals_in_place` 呼び出しを外すと、
    /// 乗数が復元されず `finalize_*` の `local_kkt_within` ゲートで `FeasiblePoint` に
    /// 降格するため、下の status assert が FAIL する (実測)。ゲートも併せて外すと
    /// status は `NonconvexLocal` に戻るが complementarity 3.774e-2 で FAIL する (実測)。
    #[test]
    fn proptest_seed_43b5a909_subbox_duals_recovered() {
        let problem = seed_43b5a909_problem();

        let mut o = SolverOptions::default();
        o.timeout_secs = Some(15.0);
        let res = solve_qp_global(&problem, &o, &GlobalOptimizationConfig::default());

        assert!(
            matches!(
                res.status,
                SolveStatus::NonconvexLocal | SolveStatus::NonconvexGlobal
            ),
            "乗数が復元できる KKT 点なので最適性を主張できるはず, got {:?}",
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
        let stat = kkt_residual_rel(&view, &res.solution, &res.dual_solution, &res.bound_duals);
        let comp = kkt_comp_residual(&view, &res.solution, &res.dual_solution, &res.bound_duals);
        let dsign = kkt_dual_sign_violation(
            &problem.constraint_types,
            &res.dual_solution,
            &problem.bounds,
            &res.bound_duals,
        );
        let pf = kkt_primal_residual(&view, &res.solution);
        let bv = kkt_bound_violation(&problem.bounds, &res.solution);
        for (name, value) in [
            ("stationarity", stat),
            ("complementarity", comp),
            ("dual_sign", dsign),
            ("primal_feasibility", pf),
            ("bound_feasibility", bv),
        ] {
            assert!(
                value < 1e-3,
                "43b5a909: {name}={value:.3e} >= 1e-3 (status={:?})",
                res.status
            );
        }
    }

    /// Sentinel: `node_discard_is_conclusive` は `Infeasible` (線形制約の厳密な
    /// 実行不可能証明) のみ `true` を返す。それ以外の unusable status (診断
    /// iterate しか持たない Stalled/MaxIterations、外的停止の Timeout、
    /// solver 内部破綻の NumericalError、box 情報だけでは判定できない Unbounded/
    /// NonConvex) は「box に解があるか不明」なだけなので `false` (= node_lb を
    /// 畳み込む必要あり) を返す。独立オラクル: Infeasible だけが Q に依存しない
    /// 線形実行可能性の厳密判定であり、他は全て探索・数値の都合による打ち切り。
    #[test]
    fn node_discard_is_conclusive_only_for_infeasible() {
        assert!(node_discard_is_conclusive(&SolveStatus::Infeasible));
        for s in [
            SolveStatus::MaxIterations,
            SolveStatus::Stalled,
            SolveStatus::Timeout,
            SolveStatus::NumericalError,
            SolveStatus::Unbounded,
            SolveStatus::NonConvex("x".into()),
        ] {
            assert!(
                !node_discard_is_conclusive(&s),
                "{s:?} must require folding node_lb (not a conclusive infeasibility proof)"
            );
        }
    }

    /// P1-1 の前提となる実測: 実行可能領域を持つ box でも `solve_local_upper_bound`
    /// は非収束 iterate (primal_residual_rel が user_eps を大きく超える) を返しうる
    /// (status=MaxIterations、Infeasible ではない)。この box は本当は解を持つため
    /// (n=40 中 39 変数は自由、target はその box 内で到達可能な値を選定)、旧実装が
    /// これを無音 discard していたら偽の「完全探索」を生みうる、という P1-1 の
    /// 前提が絵空事でないことを示す。
    #[test]
    fn unusable_node_can_be_non_infeasible_with_large_primal_residual() {
        let n = 40usize;
        let mut qi = vec![];
        let mut qj = vec![];
        let mut qv = vec![];
        for i in 0..n {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            qi.push(i);
            qj.push(i);
            qv.push(sign * (1.0 + ((i * 37 + 11) % 997) as f64 * 1e3));
        }
        let q = CscMatrix::from_triplets(&qi, &qj, &qv, n, n).unwrap();
        let c = vec![0.0; n];
        let a_cols: Vec<usize> = (0..n).collect();
        let a_rows = vec![0usize; n];
        let a_vals: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 53 + 7) % 11) as f64).collect();
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, 1, n).unwrap();
        // target は box [-10,10]^n 内で achievable (max |Ax| ~ sum(a_vals)*10 > 2000)
        // かつ naive start x0=0 からは大きく外れる (feasible だが到達に反復を要する)。
        let target: f64 = -2000.0;
        let bounds = vec![(-10.0, 10.0); n];
        let p = QpProblem::new(
            q,
            c,
            a,
            vec![target],
            bounds,
            vec![crate::problem::ConstraintType::Eq],
        )
        .unwrap();
        let mut o = SolverOptions::default();
        o.ipm.max_iter = 30;
        o.timeout_secs = Some(15.0);
        let res = solve_local_upper_bound(&p, &p.bounds.clone(), &o, None);
        assert_eq!(
            res.status,
            SolveStatus::MaxIterations,
            "expected a genuine non-convergent terminal status, got {:?}",
            res.status
        );
        assert_ne!(
            res.status,
            SolveStatus::Infeasible,
            "this box is feasible; a real solver must not report Infeasible here"
        );
        let view = ProblemView::from_problem(&p);
        let pf_rel = crate::qp::ipm_solver::kkt::primal_residual_rel(&view, &res.solution);
        assert!(
            pf_rel > 1e-3,
            "expected large primal residual demonstrating a genuinely unusable \
             (not just borderline) iterate, got pf_rel={pf_rel:.3e}"
        );
    }

    // ---- cancel_flag honored by the B&B loop's own control flow ----
    //
    // Fact established by grep before this fix: `otspot-core/src/qp/global/`
    // had zero references to `cancel_flag`/`stop_requested`/`external_stop`
    // anywhere; the loop broke only on `deadline_reached`/`max_nodes`. Each
    // node's own local/α-BB/McCormick sub-solve already honors `cancel_flag`
    // transitively (they all route through `solve_qp_with`, which clones
    // `base_opts` — Arc-sharing `cancel_flag` — before solving), so
    // cancellation was never silently *ignored*: a cancelled search still
    // eventually drained, one already-queued node at a time, each falling
    // back to a fast `Timeout`-cancelled discard. The bug is unresponsiveness:
    // the loop's own break conditions never observed the signal directly.

    fn x0_ge_0p1_concave_2d() -> QpProblem {
        use crate::problem::ConstraintType;
        // Q=diag(-2,-2) on box [-1,1]^2, plus x0 >= 0.1 (i.e. -x0 <= -0.1).
        // Same fixture as `interrupt_break_folds_popped_node_bound_into_remaining_lb`.
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[-1.0], 1, 2).unwrap();
        QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![-0.1],
            vec![(-1.0, 1.0), (-1.0, 1.0)],
            vec![ConstraintType::Le],
        )
        .unwrap()
    }

    /// Baseline/control for `qp_global_loop_honors_cancel_flag_not_only_deadline`:
    /// same fixture, generous `max_nodes`/`timeout_secs`, no cancel injected.
    /// The `x0>=0.1` constraint is invisible to the interval/α-BB bounds, so
    /// the infeasible-left-child region keeps getting re-split (each split
    /// discovers infeasibility only once box-restricted local solves are
    /// attempted) before the search fully exhausts. Value confirmed by this
    /// test itself (not hand-derived): establishes that this fixture keeps
    /// the loop busy well past root when nothing interrupts it, so the
    /// sentinel's contrast (1 vs 10) is not an artifact of a fixture that
    /// would have stopped at 1 anyway.
    #[test]
    fn qp_global_loop_baseline_processes_both_children_without_cancel() {
        let p = x0_ge_0p1_concave_2d();
        let cfg = GlobalOptimizationConfig {
            gap_tol: 1e-12,
            max_depth: 30,
            max_nodes: 1000,
            use_alpha_bb: false,
            use_mccormick: false,
            ..GlobalOptimizationConfig::default()
        };
        let (_, stats) = solve_qp_global_with_stats(&p, &opts(10.0), &cfg);
        assert_eq!(
            stats.nodes_processed, 10,
            "baseline (no cancel): fixture must exhaust well past 1 node when \
             nothing interrupts the loop, got {}",
            stats.nodes_processed
        );
    }

    /// SENTINEL: the B&B loop's top-of-iteration stop check must
    /// observe `cancel_flag` directly, not only `deadline_reached`. Same
    /// fixture/config as the baseline above (`max_nodes=1000`, `timeout_secs
    /// =10.0`, both far from firing), but cancel is injected right as the
    /// very first node is popped from the tree (after root, at loop
    /// iteration 1). A cancel-aware loop folds that node and breaks
    /// immediately, leaving `nodes_processed` at 1 (root only) —
    /// dramatically less than the baseline's 10, and unaffected by the
    /// (deliberately irrelevant) generous `max_nodes`/`timeout_secs` budget.
    ///
    /// Revert-fail (confirmed): reverting the loop-top check from
    /// `shared_opts.external_stop_requested()` back to
    /// `deadline_reached(deadline)` makes this FAIL with `nodes_processed ==
    /// 3` — the loop still eventually stops (each already-queued node's own
    /// `solve_qp_with` call independently honors `cancel_flag`, so no further
    /// branching occurs past the point cancel fires), but only after
    /// draining the small backlog already queued at that instant, not
    /// immediately as a cancel-aware loop must.
    #[test]
    fn qp_global_loop_honors_cancel_flag_not_only_deadline() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let p = x0_ge_0p1_concave_2d();
        let cfg = GlobalOptimizationConfig {
            gap_tol: 1e-12,
            max_depth: 30,
            max_nodes: 1000,
            use_alpha_bb: false,
            use_mccormick: false,
            ..GlobalOptimizationConfig::default()
        };

        LOOP_ITER_COUNT.with(|c| c.set(0));
        CANCEL_AFTER_LOOP_ITER.with(|c| c.set(Some(1)));

        let cancel = Arc::new(AtomicBool::new(false));
        let o = SolverOptions {
            timeout_secs: Some(10.0),
            cancel_flag: Some(Arc::clone(&cancel)),
            ..SolverOptions::default()
        };
        let (_, stats) = solve_qp_global_with_stats(&p, &o, &cfg);

        CANCEL_AFTER_LOOP_ITER.with(|c| c.set(None));

        assert!(
            cancel.load(Ordering::Relaxed),
            "test harness must have actually fired the injected cancel"
        );
        assert_eq!(
            stats.nodes_processed, 1,
            "cancel injected at the first loop iteration must break the loop \
             before processing any tree node beyond root, got {}",
            stats.nodes_processed
        );
    }

    /// SENTINEL (isolated from the loop-top check above): once the
    /// B&B loop drains cleanly (queue empty, nothing discarded),
    /// `finalize_search_outcome`'s own `external_stop_requested()` backstop
    /// must still demote an otherwise-fully-proven result if cancel was
    /// observed right at that boundary — mirrors the "discard an
    /// already-proven conclusion once a stop is observed" contract
    /// (`conic::nonconvex::global_core`'s backstop). Without it, a
    /// genuinely-complete-but-cancelled search takes `finalize_search_
    /// outcome`'s early `!halted_early` return, which calls `finalize_proven`
    /// unconditionally.
    ///
    /// Fixture: `diag_concave_1d(2.0)`, same as `indefinite_q_proven_yields_
    /// nonconvex_global` / `qp_global_proven_nonconvex_has_bound_gap_cert`
    /// (this test's uncancelled control, both assert `NonconvexGlobal` +
    /// cert). Root branches, one child finds the exact corner, the sibling
    /// is pruned by the resulting tight incumbent — a clean, fully-exhausted
    /// search. Cancel is injected *after* the loop/polish finish, isolating
    /// this backstop from the loop-top check.
    ///
    /// Revert-fail: removing `opts.external_stop_requested()` from
    /// `halted_early` makes this FAIL with `NonconvexGlobal` + `Some(cert)`
    /// — identical to the uncancelled control.
    #[test]
    fn qp_global_finalize_backstop_demotes_cancelled_complete_search() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let p = diag_concave_1d(2.0);
        let cfg = GlobalOptimizationConfig::default();

        CANCEL_BEFORE_FINALIZE.with(|c| c.set(true));

        let cancel = Arc::new(AtomicBool::new(false));
        let o = SolverOptions {
            timeout_secs: Some(5.0),
            cancel_flag: Some(Arc::clone(&cancel)),
            ..SolverOptions::default()
        };
        let r = solve_qp_global(&p, &o, &cfg);

        CANCEL_BEFORE_FINALIZE.with(|c| c.set(false));

        assert_eq!(
            r.status,
            SolveStatus::NonconvexLocal,
            "a cancel observed right before finalization must demote the \
             otherwise-proven result to unproven, got {:?}",
            r.status
        );
        assert!(
            r.bound_gap_cert.is_none(),
            "a cancelled-at-finalization result must not carry a \
             BoundGapCertificate"
        );
    }
}
