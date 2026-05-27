//! Phase 3 spatial Branch-and-Bound scaffolding (非凸 QP 大域最適化)。
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
//! Q の凸性 (Gershgorin 由来 `alpha == 0.0` を PSD と判定) で分岐:
//! - **Q PSD (convex):**
//!   - `Optimal`: BB 探索完了 (root tight or queue 空) → 凸 QP として global ε-optimal
//!   - `LocallyOptimal`: 早期打切 (gap 未証明)。convex Q では IPM 単発で global 達成しても
//!     budget 不足で proof が間に合わなかった希少ケース。
//! - **Q indefinite (nonconvex):**
//!   - `NonconvexGlobal`: BB 探索完了 → indefinite Q 上で ε-global 証明済み
//!   - `NonconvexLocal`: 早期打切 → incumbent あり、global proof なし (caller は探索打切と
//!     IPM 単発 `LocallyOptimal` を区別できる)
//! - `Timeout`: deadline で打ち切り、incumbent 未発見
//! - root と同じ status: root が Infeasible / NumericalError / Unbounded だった場合

pub(crate) mod bound;
pub(crate) mod bound_alpha_bb;
pub(crate) mod bound_mccormick;
pub(crate) mod branch;
pub(crate) mod node;
pub(crate) mod pruning;
pub(crate) mod tree;

use crate::options::{GlobalOptimizationConfig, QpWarmStart, SolverOptions};
use crate::problem::{SolveStatus, SolverResult};
use crate::problem::certificate::BoundGapCertificate;
use crate::qp::certificate::prove_optimal;
use crate::qp::problem::QpProblem;
use crate::qp::ipm_solver::kkt::{
    bound_violation as kkt_bound_violation,
    complementarity_residual_rel as kkt_comp_residual,
    kkt_residual_rel,
    primal_residual_rel as kkt_primal_residual,
};
use crate::qp::kkt_resid::dual_sign_violation as kkt_dual_sign_violation;
use crate::qp::ipm_solver::outcome::ProblemView;
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
    let user_eps = shared_opts.ipm_eps();

    // root が ε-optimal なら即終了 (queue 不要)。
    if within_gap(state.incumbent_obj, root_lb, cfg.gap_tol) {
        return (state.finalize_proven(problem, root_lb, q_indefinite, cfg.gap_tol, user_eps), stats);
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
                (state.finalize_proven(problem, root_lb, q_indefinite, cfg.gap_tol, user_eps), stats)
            } else {
                (
                    state.finalize_unproven(
                        root_lb,
                        stats.nodes_processed,
                        0,
                        cfg,
                        q_indefinite,
                    ),
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
    // 深さ上限で破棄した node の node_lb の min を保持する。これが未探索領域の下界に
    // なるため remaining_lb に畳み込む必要がある。
    let mut depth_discard_lb: f64 = f64::INFINITY;

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
            // 深さ上限超過 → 子を展開しない = unproven region 残存。
            // この node の lb を depth_discard_lb に畳み込む (remaining_lb に反映する)。
            max_depth_breached = true;
            depth_discard_lb = depth_discard_lb.min(node_lb);
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

    // incumbent が分枝 node 由来の場合、その双対は sub-box 基準で回収されているため
    // 元問題に整合させる (interior 変数への分枝境界 dual = 相補性違反を除去)。
    state.polish_incumbent_duals(problem, &shared_opts, cfg.gap_tol);

    // 終了条件分岐:
    // - queue 空 AND max_depth 未超過 AND deadline/max_nodes 未到達 → proven
    // - それ以外 → 未証明 (incumbent あれば LocallyOptimal)
    let halted_early = !tree.is_empty()
        || max_depth_breached
        || deadline_hit(&deadline)
        || stats.nodes_processed >= cfg.max_nodes;

    let result = if halted_early {
        // 未探索領域の下界: queue に残った node の最小 lb と、深さ上限で破棄した
        // node の lb の両方を考慮する。どちらの領域も「未証明」であるため min を取る。
        let remaining_lb = tree.best_lower_bound().unwrap_or(f64::INFINITY).min(depth_discard_lb);
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
    let n_lb = problem.bounds.iter().filter(|&&(lb, _)| lb.is_finite()).count();
    let n_ub = problem.bounds.iter().filter(|&&(_, ub)| ub.is_finite()).count();
    if polished.solution.len() != problem.num_vars
        || polished.dual_solution.len() != problem.num_constraints
        || polished.bound_duals.len() != n_lb + n_ub
    {
        return false;
    }
    let kkt_tol = (user_eps * POLISH_KKT_ACCEPT_FACTOR).min(POLISH_KKT_ABS_CAP);
    let view = ProblemView {
        q: &problem.q,
        a: &problem.a,
        c: &problem.c,
        b: &problem.b,
        bounds: &problem.bounds,
        constraint_types: &problem.constraint_types,
        eliminated_cols: &[],
    };
    let kkt = kkt_residual_rel(&view, &polished.solution, &polished.dual_solution, &polished.bound_duals);
    let pf = kkt_primal_residual(&view, &polished.solution);
    let bv = kkt_bound_violation(&problem.bounds, &polished.solution);
    let comp = kkt_comp_residual(&view, &polished.solution, &polished.dual_solution, &polished.bound_duals);
    let dsign = kkt_dual_sign_violation(
        &problem.constraint_types,
        &polished.dual_solution,
        &problem.bounds,
        &polished.bound_duals,
    );
    kkt <= kkt_tol && pf <= kkt_tol && bv <= kkt_tol && comp <= kkt_tol && dsign <= kkt_tol
}

/// Relative QP duality gap from KKT data: `|p - d| / max(|p|, |d|, 1)`.
///
/// `p - d = x'Qx + c'x + y'b - z_lb'lb + z_ub'ub`
///
/// `bound_duals` format: finite-lb multipliers first (variable order),
/// then finite-ub multipliers (variable order), matching the IPM packing.
/// Returns `f64::INFINITY` on any arithmetic failure.
fn compute_duality_gap_rel(
    problem: &QpProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    primal_obj: f64,
) -> f64 {
    let qx = match problem.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let xqx: f64 = x.iter().zip(&qx).map(|(xi, qi)| xi * qi).sum();
    let cx: f64 = problem.c.iter().zip(x).map(|(&ci, &xi)| ci * xi).sum();
    let yb: f64 = y.iter().zip(&problem.b).map(|(&yi, &bi)| yi * bi).sum();

    let n_lb_finite: usize = problem.bounds.iter().filter(|&&(lb, _)| lb.is_finite()).count();
    let mut lb_bnd = 0.0_f64;
    let mut ub_bnd = 0.0_f64;
    let (mut lb_idx, mut ub_idx) = (0usize, n_lb_finite);
    for &(lb, ub) in &problem.bounds {
        if lb.is_finite() {
            if lb_idx < z.len() {
                lb_bnd += z[lb_idx] * lb;
            }
            lb_idx += 1;
        }
        if ub.is_finite() {
            if ub_idx < z.len() {
                ub_bnd += z[ub_idx] * ub;
            }
            ub_idx += 1;
        }
    }

    let gap_raw = xqx + cx + yb - lb_bnd + ub_bnd;
    let gap_abs = gap_raw.abs();
    let dual_obj = primal_obj - gap_raw;
    let denom = primal_obj.abs().max(dual_obj.abs()).max(1.0);
    if gap_abs.is_finite() {
        gap_abs / denom
    } else {
        f64::INFINITY
    }
}

/// search state encapsulation: incumbent + 最終 result の組み立てを 1 箇所に集約。
struct SearchState {
    incumbent_result: SolverResult,
    incumbent_obj: f64,
    incumbent_sol: Vec<f64>,
    /// B&B ループで root 以外の incumbent が見つかった場合 true。
    /// false のまま = root solve の解がそのまま incumbent = 元問題 box で回収済み。
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

    /// 探索終了後の dual recovery polish。
    ///
    /// B&B incumbent の双対は分枝後の sub-box に対して回収されるため、元問題で
    /// interior な変数にも分枝境界由来の bound dual が残り、元問題基準の相補性
    /// (`z_j·(x_j − bnd_j) = 0`) を破る。incumbent を warm start に固定して **元問題の
    /// box** で局所 QP を解き直し、元問題に整合した双対を回収する。
    ///
    /// warm を境界張り付きでも採用する点が [`solve_local_upper_bound`] と異なる
    /// (探索中は saddle 再固着回避のため境界 warm を捨てるが、最終 polish では
    /// incumbent corner に錨を打つのが目的)。obj は gap_tol 内に保たれ proof 妥当性を
    /// 維持 (duals を整合化)。収束済み (Optimal/LocallyOptimal) かつ obj が悪化しない
    /// 場合のみ採用し、未収束 or obj 悪化は棄却して incumbent を保持する。
    /// root incumbent (分枝なし) は既に元問題 box で回収済みのため skip。
    fn polish_incumbent_duals(&mut self, problem: &QpProblem, base_opts: &SolverOptions, gap_tol: f64) {
        if !self.incumbent_updated {
            // root solve 結果は元問題 box で回収済み; polish は冗長。
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
        let user_eps = base_opts.ipm_eps();
        let polished = crate::qp::solve_qp_with(problem, &opts);
        if is_polish_acceptable(&polished.status, polished.objective, self.incumbent_obj, gap_tol) {
            self.update_incumbent(&polished);
        } else if is_polish_suboptimal_acceptable(&polished, problem, self.incumbent_obj, gap_tol, user_eps) {
            // SuboptimalSolution でも KKT が十分なら dual recovery として採用。
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
        let view = ProblemView {
            q: &problem.q,
            a: &problem.a,
            c: &problem.c,
            b: &problem.b,
            bounds: &problem.bounds,
            constraint_types: &problem.constraint_types,
            eliminated_cols: &[],
        };
        let cert_result = {
            let x = &self.incumbent_result.solution;
            let y = &self.incumbent_result.dual_solution;
            let z = &self.incumbent_result.bound_duals;
            let duality_gap_rel = self.incumbent_result.duality_gap_rel.unwrap_or_else(|| {
                compute_duality_gap_rel(problem, x, y, z, self.incumbent_obj)
            });
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
                    self.incumbent_result.status, self.incumbent_obj, lower_bound, gap_rel
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
        let q = CscMatrix::from_triplets(
            &[0, 1],
            &[0, 1],
            &[-2.0, -2.0],
            2,
            2,
        )
        .unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let p = QpProblem::new_all_le(
            q,
            vec![0.0, 0.0],
            a,
            vec![],
            vec![(-1.0, 1.0), (-1.0, 1.0)],
        )
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
        let cert = r.bound_gap_cert.as_ref()
            .expect("proven QP global (Optimal) must carry BoundGapCertificate");
        assert!(
            cert.gap_rel() <= cert.gap_tol() + 1e-10,
            "gap_rel={:.3e} must be ≤ gap_tol={:.3e}",
            cert.gap_rel(), cert.gap_tol()
        );
    }

    /// Proven QP global (indefinite Q) result carries BoundGapCertificate.
    #[test]
    fn qp_global_proven_nonconvex_has_bound_gap_cert() {
        let p = diag_concave_1d(2.0);
        let r = solve_qp_global(&p, &opts(5.0), &GlobalOptimizationConfig::default());
        assert!(matches!(r.status, SolveStatus::NonconvexGlobal));
        let cert = r.bound_gap_cert.as_ref()
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
        let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0), (-1.0, 1.0)]).unwrap();
        let cfg = GlobalOptimizationConfig { gap_tol: 1e-12, max_depth: 1, max_nodes: 1, ..GlobalOptimizationConfig::default() };
        let r = solve_qp_global(&p, &opts(5.0), &cfg);
        assert!(
            matches!(r.status, SolveStatus::NonconvexLocal | SolveStatus::LocallyOptimal),
            "expected unproven status, got {:?}", r.status
        );
        assert!(r.bound_gap_cert.is_none(), "unproven must have no BoundGapCertificate");
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
        let p = QpProblem::new_all_le(
            q,
            vec![0.0, 0.0],
            a,
            vec![],
            vec![(-1.0, 1.0), (-1.0, 1.0)],
        ).unwrap();
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
        assert!(is_polish_acceptable(&SolveStatus::Optimal,        0.0, 0.0, 1e-6));
        assert!(is_polish_acceptable(&SolveStatus::LocallyOptimal, 0.0, 0.0, 1e-6));
        // 未収束 → 棄却
        assert!(!is_polish_acceptable(&SolveStatus::MaxIterations,     0.0, 0.0, 1e-6));
        assert!(!is_polish_acceptable(&SolveStatus::SuboptimalSolution, 0.0, 0.0, 1e-6));
        // その他の失敗 status も棄却
        assert!(!is_polish_acceptable(&SolveStatus::Infeasible,      0.0, 0.0, 1e-6));
        assert!(!is_polish_acceptable(&SolveStatus::NumericalError,  0.0, 0.0, 1e-6));
        assert!(!is_polish_acceptable(&SolveStatus::Timeout,         0.0, 0.0, 1e-6));
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
        assert!(is_polish_acceptable(&SolveStatus::Optimal, inc,          inc, gap_tol));
        // 改善 (より小さい) → 採用可
        assert!(is_polish_acceptable(&SolveStatus::Optimal, inc - 0.5,    inc, gap_tol));
        // tol 以内の微小悪化 → 採用可 (dual 数値誤差)
        assert!(is_polish_acceptable(&SolveStatus::Optimal, inc + tol * 0.5, inc, gap_tol));
        // tol を超える悪化 → 棄却
        assert!(!is_polish_acceptable(&SolveStatus::Optimal, inc + tol + 1e-10, inc, gap_tol));
        // 明確な悪化 → 棄却
        assert!(!is_polish_acceptable(&SolveStatus::Optimal, 0.0,          inc, gap_tol));
        assert!(!is_polish_acceptable(&SolveStatus::Optimal, 1.0,          inc, gap_tol));

        // incumbent_obj = 0.0 → scale = 1.0, 許容上限 = 0 + 1e-4
        let inc = 0.0_f64;
        let scale = 1.0_f64.max(inc.abs());
        let tol = gap_tol * scale;
        assert!(is_polish_acceptable(&SolveStatus::Optimal, 0.0,        inc, gap_tol));
        assert!(is_polish_acceptable(&SolveStatus::Optimal, -0.5,       inc, gap_tol));
        assert!(!is_polish_acceptable(&SolveStatus::Optimal, tol + 1e-10, inc, gap_tol));

        // incumbent_obj = 100.0 → scale = 100.0, 許容上限 = 100.0 + 1e-2
        let inc = 100.0_f64;
        let scale = 1.0_f64.max(inc.abs());
        let tol = gap_tol * scale; // 1e-2
        assert!(is_polish_acceptable(&SolveStatus::Optimal, inc + tol * 0.5, inc, gap_tol));
        assert!(!is_polish_acceptable(&SolveStatus::Optimal, inc + tol + 1e-10, inc, gap_tol));
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
            ("neg timeout_secs", SolverOptions { timeout_secs: Some(-1.0), ..Default::default() }),
            ("inf timeout_secs", SolverOptions { timeout_secs: Some(f64::INFINITY), ..Default::default() }),
            ("nan primal_tol", SolverOptions { primal_tol: f64::NAN, ..Default::default() }),
            ("zero threads", SolverOptions { threads: 0, ..Default::default() }),
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
            solution: vec![0.0],           // wrong: should be len 2
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
            dual_solution: vec![],         // wrong: should be len 1
            bound_duals: vec![],
            ..SolverResult::default()
        };
        assert!(
            !is_polish_suboptimal_acceptable(&polished_short_dual, &problem, 0.0, 0.1, 1e-6),
            "mismatched dual_solution dimension must be rejected",
        );
    }

    // ---- finalize_proven dual-quality gate sentinels --------------------------

    /// Table-driven: 4 combinations of (convex/indefinite) × (good-dual/bad-dual).
    ///
    /// ## Sentinel (no-op-fail requirement)
    /// Removing the `prove_optimal` call from `finalize_proven` causes this function
    /// to ALWAYS stamp the Optimal/NonconvexGlobal status regardless of dual quality.
    /// The two bad-dual rows (`convex-bad-dual` → LocallyOptimal and
    /// `indefinite-bad-dual` → NonconvexLocal) would then receive Optimal/NonconvexGlobal
    /// and the assertions FAIL — confirming the gate is load-bearing.
    ///
    /// ## KKT math for test fixtures
    ///
    /// Convex problem: `min x²`, `Q=[[2]]`, `c=[0]`, no constraints, bounds `[-1,1]`.
    /// - Good dual: `x=0` (interior). `z=[z_lb=0, z_ub=0]`.
    ///   Stationarity: `2·0 + 0 - 0 + 0 = 0` ✓, `duality_gap=0` ✓.
    /// - Bad dual: `x=0`, `z=[100, -100]`.
    ///   Stationarity: `-100 + (-100) = -200 ≠ 0` ✗, `dual_sign_violation` for `z_ub < 0` ✗.
    ///
    /// Indefinite problem: `min -x²`, `Q=[[-2]]`, `c=[0]`, no constraints, bounds `[-1,1]`.
    /// - Good dual: `x=1` (ub active). `z=[z_lb=0, z_ub=2]`.
    ///   Stationarity: `-2·1 - 0 + 2 = 0` ✓, complementarity `z_ub·(1-1)=0` ✓.
    /// - Bad dual: `x=1`, `z=[50, 50]`.
    ///   Stationarity: `-2 - 50 + 50 = -2 ≠ 0` ✗.
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
        let r = SearchState::new(good_conv)
            .finalize_proven(&p_convex, 0.0, false, gap_tol, user_eps);
        assert_eq!(r.status, SolveStatus::Optimal, "convex-good-dual must be Optimal");
        assert!(r.bound_gap_cert.is_some(), "Optimal must carry bound_gap_cert");
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
        let r = SearchState::new(bad_conv)
            .finalize_proven(&p_convex, 0.0, false, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::LocallyOptimal,
            "convex-bad-dual must be demoted to LocallyOptimal"
        );
        assert!(r.bound_gap_cert.is_none(), "demoted must have no bound_gap_cert");
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
        let r = SearchState::new(good_indef)
            .finalize_proven(&p_indef, -1.0, true, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::NonconvexGlobal,
            "indefinite-good-dual must be NonconvexGlobal"
        );
        assert!(r.bound_gap_cert.is_some(), "NonconvexGlobal must carry bound_gap_cert");
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
        let r = SearchState::new(bad_indef)
            .finalize_proven(&p_indef, -1.0, true, gap_tol, user_eps);
        assert_eq!(
            r.status,
            SolveStatus::NonconvexLocal,
            "indefinite-bad-dual must be demoted to NonconvexLocal"
        );
        assert!(r.bound_gap_cert.is_none(), "demoted must have no bound_gap_cert");
        assert!(r.opt_cert.is_none(), "demoted must have no opt_cert");
    }
}
