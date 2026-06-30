//! 元空間 post-processing 3 段階: (1) primal projection, (2) y/z 交互 refit (+ IRLS),
//! (3) saddle-point Krylov IR + 2nd primal projection。

use crate::options::SolverOptions;
use crate::problem::SolverResult;
use crate::qp::ipm_solver::kkt::{
    bound_violation, complementarity_componentwise_rel, complementarity_residual_rel,
    kkt_residual_rel, primal_residual_rel,
};
use crate::qp::ipm_solver::outcome::ProblemView;
use crate::qp::kkt_resid::dual_sign_violation;
use crate::qp::problem::QpProblem;

/// primal projection の LDL 因子化に対する時間予算ガード。memory budget は factorize
/// 経路が別途見る (max_l_nnz_from_budget) が、予算内に収まっても巨大問題では
/// 因子化自体が分単位かかり deadline を空費する。これは「分単位 factorize を
/// post-processing 段で行うか否か」の時間 proxy ガード (n+m で判定)。
use crate::tolerances::LARGE_PROBLEM_THRESHOLD;
/// 進捗 stall の絶対 floor。残差が ~0 近傍 (相対閾値が underflow する regime) で
/// sub-floor の丸め noise を吸収しループを終端させる。残差が大きい regime では
/// `REFIT_REL_STALL` 相対項が支配するため、この floor は near-zero 専用。
const REFIT_PROGRESS_EPS: f64 = 1e-12;
/// 相対進捗 stall 閾値。refit/IRLS の KKT 残差は各 sub-step の revert guard により
/// 単調非増加。1 反復の改善 `prev - cur` が現残差の本割合を下回ると、f64 蓄積丸め
/// (ill-cond 系では KKT が f64 限界に張り付き相対 noise ~1e-10 しか動かない) と
/// 区別できない実質ゼロ進捗とみなし break する。絶対 `REFIT_PROGRESS_EPS` 単独では
/// 大残差 (~0.1) で drop≈1e-11 > 1e-12 を「進捗あり」と誤判定し deadline まで無駄
/// 反復していた。真の (低速含む) 線形収束は相対 drop が桁違いに大きい (>=1e-4) ため
/// 早期 break しない。1e-8 は noise floor (~1e-10) の 100x 上、収束 drop の桁下。
const REFIT_REL_STALL: f64 = 1e-8;
const IRLS_INNER_MAX_ITERS: usize = 30;
const KRYLOV_MAX_ITERS: usize = 400;
const KKT_SKIP_MARGIN: f64 = 100.0;

/// refit/IRLS の進捗 stall 判定。`prev_kkt`/`current_kkt` は revert guard により
/// `current_kkt <= prev_kkt`。改善 drop が「相対 (`REFIT_REL_STALL · prev_kkt`)」と
/// 「絶対 floor (`REFIT_PROGRESS_EPS`)」の大きい方を下回ったら stall とみなす。
fn refit_progress_stalled(prev_kkt: f64, current_kkt: f64) -> bool {
    let threshold = (REFIT_REL_STALL * prev_kkt).max(REFIT_PROGRESS_EPS);
    current_kkt + threshold >= prev_kkt
}

pub(super) fn allow_primal_projection(orig_problem: &QpProblem) -> bool {
    let problem_size = orig_problem.num_vars + orig_problem.num_constraints;
    problem_size <= LARGE_PROBLEM_THRESHOLD
}

/// IPM 出口で既に証明条件を満たした Optimal なら post-processing skip。
///
/// kkt + primal に加え、complementarity と duality gap も確認する。Krylov IR は
/// kkt/pres だけでなく comp/gap も改善するため、これらが未収束の場合に skip すると
/// prove_optimal が失敗して SuboptimalSolution になる。
pub(super) fn kkt_already_passes(
    orig_problem: &QpProblem,
    final_sol: &SolverResult,
    eliminated_cols: &[bool],
    ipm_status_optimal: bool,
    user_eps: f64,
) -> bool {
    if final_sol.solution.is_empty() || orig_problem.num_constraints == 0 || !ipm_status_optimal {
        return false;
    }
    let skip_tol = user_eps / KKT_SKIP_MARGIN;
    let view = build_view(orig_problem, eliminated_cols);
    let kkt0 = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    if kkt0 >= skip_tol {
        return false;
    }
    let pres0 = primal_residual_rel(&view, &final_sol.solution);
    if pres0 >= skip_tol {
        return false;
    }
    let bv = bound_violation(orig_problem.bounds.as_slice(), &final_sol.solution);
    if bv > user_eps {
        return false;
    }
    let comp = complementarity_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    )
    .max(complementarity_componentwise_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    ));
    if comp > user_eps {
        return false;
    }
    let dsign = dual_sign_violation(
        &orig_problem.constraint_types,
        &final_sol.dual_solution,
        orig_problem.bounds.as_slice(),
        &final_sol.bound_duals,
    );
    if dsign > user_eps {
        return false;
    }
    let gap = super::duality_gap::compute_duality_gap_rel(orig_problem, final_sol);
    gap <= user_eps
}

/// Post-processing stage 1+2: primal projection + y/z 交互 refit + IRLS。
/// 各 step は KKT-guard 付きで悪化時 revert。
pub(super) fn refine_post_processing(
    orig_problem: &QpProblem,
    final_sol: &mut SolverResult,
    eliminated_cols: &[bool],
    opts: &SolverOptions,
    allow_primal: bool,
) -> f64 {
    let user_eps = opts.ipm_eps();
    let view = build_view(orig_problem, eliminated_cols);

    // (1) primal projection: 違反制約に対して x を最小ノルム射影。
    if allow_primal {
        let pre_x = final_sol.solution.clone();
        let pre_pres = primal_residual_rel(&view, &final_sol.solution);
        crate::qp::refine_primal_lsq(orig_problem, final_sol, opts.deadline);
        let post_pres = primal_residual_rel(&view, &final_sol.solution);
        if post_pres > pre_pres {
            final_sol.solution = pre_x;
        } else {
            // x 改善時は z を新 x に合わせて refit。
            crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, user_eps);
        }
    }

    // (2) y/z 交互 refit。
    let diag = DiagPostsolve::new();
    let mut current_kkt = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    let mut refit_iters = 0usize;
    loop {
        if opts
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            diag.note(&format!("refit-loop break: deadline at iter {refit_iters}"));
            break;
        }
        let prev_kkt = current_kkt;

        let pre_dual_step = final_sol.clone();
        let t = diag.tic();
        crate::qp::refine_dual_lsq(orig_problem, final_sol, eliminated_cols, opts.deadline);
        diag.acc("refit.dual_lsq", t);
        let t = diag.tic();
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        diag.acc("refit.zero+project", t);
        let t = diag.tic();
        crate::qp::refine_dual_projected_gradient(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        diag.acc("refit.proj_grad", t);
        let t = diag.tic();
        crate::qp::refine_dual_worst_active_block(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        diag.acc("refit.worst_active", t);
        let post_kkt = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_kkt <= current_kkt {
            current_kkt = post_kkt;
        } else {
            *final_sol = pre_dual_step;
        }

        let pre_z = final_sol.bound_duals.clone();
        let t = diag.tic();
        crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, user_eps);
        diag.acc("refit.refit_z", t);
        let post_kkt = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_kkt <= current_kkt {
            current_kkt = post_kkt;
        } else {
            final_sol.bound_duals = pre_z;
        }

        refit_iters += 1;
        diag.trajectory("refit", refit_iters, prev_kkt, current_kkt);
        if refit_progress_stalled(prev_kkt, current_kkt) {
            diag.note(&format!(
                "refit-loop break: stall at iter {refit_iters} (prev={prev_kkt:.6e} cur={current_kkt:.6e})"
            ));
            break;
        }
    }
    diag.report("refit-loop", refit_iters);

    // 標準 LSQ が componentwise eps を満たさない場合 IRLS で L∞ 風 y を試行。
    let mut irls_iters = 0usize;
    loop {
        if current_kkt <= user_eps {
            break;
        }
        if opts
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            diag.note(&format!("irls-loop break: deadline at iter {irls_iters}"));
            break;
        }
        let prev_kkt = current_kkt;

        let pre_dual_step = final_sol.clone();
        let t = diag.tic();
        crate::qp::refine_dual_lsq_irls(
            orig_problem,
            final_sol,
            eliminated_cols,
            user_eps,
            IRLS_INNER_MAX_ITERS,
            opts.deadline,
        );
        diag.acc("irls.dual_lsq_irls", t);
        let t = diag.tic();
        crate::qp::zero_inactive_inequality_duals(orig_problem, final_sol);
        crate::qp::project_duals_from_singleton_columns(orig_problem, final_sol);
        crate::qp::refine_dual_projected_gradient(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        crate::qp::refine_dual_worst_active_block(
            orig_problem,
            final_sol,
            eliminated_cols,
            opts.deadline,
        );
        diag.acc("irls.proj+worst", t);
        let post_kkt_irls = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        if post_kkt_irls < current_kkt {
            current_kkt = post_kkt_irls;
            let pre_z = final_sol.bound_duals.clone();
            crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, user_eps);
            let post_kkt_z = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if post_kkt_z <= current_kkt {
                current_kkt = post_kkt_z;
            } else {
                final_sol.bound_duals = pre_z;
            }
        } else {
            *final_sol = pre_dual_step;
            diag.note(&format!("irls-loop break: no progress at iter {irls_iters}"));
            break;
        }

        irls_iters += 1;
        diag.trajectory("irls", irls_iters, prev_kkt, current_kkt);
        if refit_progress_stalled(prev_kkt, current_kkt) {
            diag.note(&format!(
                "irls-loop break: stall at iter {irls_iters} (prev={prev_kkt:.6e} cur={current_kkt:.6e})"
            ));
            break;
        }
    }
    diag.report("irls-loop", irls_iters);

    current_kkt
}

/// Env-gated (OTSPOT_DIAG_POSTSOLVE=1) diagnostic accumulator for postsolve
/// refine loops. No-op unless the env var is set; isolated diagnostic scaffolding.
struct DiagPostsolve {
    on: bool,
    acc: std::cell::RefCell<std::collections::BTreeMap<&'static str, (std::time::Duration, u64)>>,
}
impl DiagPostsolve {
    fn new() -> Self {
        DiagPostsolve {
            on: std::env::var("OTSPOT_DIAG_POSTSOLVE").is_ok(),
            acc: std::cell::RefCell::new(std::collections::BTreeMap::new()),
        }
    }
    fn tic(&self) -> Option<std::time::Instant> {
        if self.on {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }
    fn acc(&self, key: &'static str, t: Option<std::time::Instant>) {
        if let Some(t0) = t {
            let mut m = self.acc.borrow_mut();
            let e = m.entry(key).or_insert((std::time::Duration::ZERO, 0));
            e.0 += t0.elapsed();
            e.1 += 1;
        }
    }
    #[allow(clippy::print_stderr)] // env-gated diagnostic trace
    fn trajectory(&self, tag: &str, iter: usize, prev: f64, cur: f64) {
        if self.on && (iter <= 5 || iter.is_multiple_of(50)) {
            eprintln!("[diag {tag}] iter={iter} kkt {prev:.6e} -> {cur:.6e} (drop={:.3e})", prev - cur);
        }
    }
    #[allow(clippy::print_stderr)] // env-gated diagnostic trace
    fn note(&self, msg: &str) {
        if self.on {
            eprintln!("[diag] {msg}");
        }
    }
    #[allow(clippy::print_stderr)] // env-gated diagnostic trace
    fn report(&self, tag: &str, iters: usize) {
        if !self.on {
            return;
        }
        eprintln!("[diag {tag}] total_iters={iters}");
        for (k, (d, calls)) in self.acc.borrow().iter() {
            eprintln!("  {k:24} = {:8.2}s  calls={calls}", d.as_secs_f64());
        }
    }
}

/// Clear inactive inequality duals and refit bound duals when doing so improves
/// the full dual certificate without worsening the duality gap.
pub(super) fn cleanup_inactive_dual_complementarity(
    orig_problem: &QpProblem,
    final_sol: &mut SolverResult,
    eliminated_cols: &[bool],
    user_eps: f64,
) {
    if final_sol.solution.is_empty() || orig_problem.num_constraints == 0 {
        return;
    }
    let view = build_view(orig_problem, eliminated_cols);
    let before = dual_certificate_residual_max(orig_problem, &view, final_sol);
    let gap_before = super::compute_duality_gap_rel(orig_problem, final_sol);
    let mut candidate = final_sol.clone();
    crate::qp::zero_inactive_inequality_duals(orig_problem, &mut candidate);
    crate::qp::refit_bound_duals_kkt(orig_problem, &mut candidate, user_eps);
    let after = dual_certificate_residual_max(orig_problem, &view, &candidate);
    let gap_after = super::compute_duality_gap_rel(orig_problem, &candidate);
    if after.is_finite() && after <= before && gap_after.is_finite() && gap_after <= gap_before {
        *final_sol = candidate;
    }
}

fn dual_certificate_residual_max(
    orig_problem: &QpProblem,
    view: &ProblemView<'_>,
    result: &SolverResult,
) -> f64 {
    kkt_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    )
    .max(complementarity_residual_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    ))
    .max(complementarity_componentwise_rel(
        view,
        &result.solution,
        &result.dual_solution,
        &result.bound_duals,
    ))
    .max(dual_sign_violation(
        &orig_problem.constraint_types,
        &result.dual_solution,
        orig_problem.bounds.as_slice(),
        &result.bound_duals,
    ))
}

/// Post-processing stage 3: saddle-point Krylov IR (K [dx;dy] = -[r_d;r_p]) +
/// pres 残留時の 2nd primal projection (KKT-guard 付き)。
pub(super) fn refine_krylov_and_projection(
    orig_problem: &QpProblem,
    final_sol: &mut SolverResult,
    eliminated_cols: &[bool],
    opts: &SolverOptions,
    allow_primal: bool,
) {
    if final_sol.solution.is_empty() || orig_problem.num_constraints == 0 {
        return;
    }
    let view = build_view(orig_problem, eliminated_cols);
    let user_eps = opts.ipm_eps();
    let target_pf = user_eps;
    crate::qp::refine_kkt_iterative(
        orig_problem,
        final_sol,
        eliminated_cols,
        KRYLOV_MAX_ITERS,
        target_pf,
        opts.deadline,
    );

    // Stage 3a: extended-precision IR with TwoFloat accumulation.
    // Runs only when standard Krylov IR stalled above eps and the option is enabled.
    if opts.ipm.extended_ir {
        let kkt_post_std = kkt_residual_rel(
            &view,
            &final_sol.solution,
            &final_sol.dual_solution,
            &final_sol.bound_duals,
        );
        let pres_post_std = primal_residual_rel(&view, &final_sol.solution);
        if kkt_post_std > user_eps || pres_post_std > user_eps {
            crate::qp::refine_kkt_extended_precision(
                orig_problem,
                final_sol,
                eliminated_cols,
                target_pf,
                opts.deadline,
            );
        }
    }

    // (3b) KKT IR 後に pres > eps なら primal projection を 1 回追加。
    // 採用条件: pres 改善 AND kkt <= user_eps を厳守 (df 退行防止)。
    if !allow_primal {
        return;
    }
    if opts
        .deadline
        .is_some_and(|d| std::time::Instant::now() >= d)
    {
        return;
    }
    let pres_post_ir = primal_residual_rel(&view, &final_sol.solution);
    let kkt_post_ir = kkt_residual_rel(
        &view,
        &final_sol.solution,
        &final_sol.dual_solution,
        &final_sol.bound_duals,
    );
    if pres_post_ir > user_eps && kkt_post_ir <= user_eps {
        let pre_sol2 = final_sol.clone();
        crate::qp::refine_primal_lsq(orig_problem, final_sol, opts.deadline);
        let post_pres2 = primal_residual_rel(&view, &final_sol.solution);
        if post_pres2 < pres_post_ir {
            crate::qp::refit_bound_duals_kkt(orig_problem, final_sol, user_eps);
            let kkt_after2 = kkt_residual_rel(
                &view,
                &final_sol.solution,
                &final_sol.dual_solution,
                &final_sol.bound_duals,
            );
            if kkt_after2 > user_eps {
                *final_sol = pre_sol2;
            }
        } else {
            *final_sol = pre_sol2;
        }
    }
}

fn build_view<'a>(orig_problem: &'a QpProblem, eliminated_cols: &'a [bool]) -> ProblemView<'a> {
    ProblemView {
        q: &orig_problem.q,
        a: &orig_problem.a,
        c: &orig_problem.c,
        b: &orig_problem.b,
        bounds: &orig_problem.bounds,
        constraint_types: &orig_problem.constraint_types,
        eliminated_cols,
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod gate_predicate_tests {
    use super::{build_view, kkt_already_passes, kkt_residual_rel, primal_residual_rel};
    use crate::options::SolverOptions;
    use crate::problem::ConstraintType;
    use crate::qp::ipm_solver::kkt::{
        complementarity_componentwise_rel, complementarity_residual_rel,
    };
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    /// min 0.5·diag·Σx² s.t. Σx = rhs, x free. Solved deterministically.
    fn solved(n: usize, diag: f64, rhs: f64) -> (QpProblem, crate::problem::SolverResult) {
        let idx: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&idx, &idx, &vec![diag; n], n, n).unwrap();
        let a = CscMatrix::from_triplets(&vec![0usize; n], &idx, &vec![1.0; n], 1, n).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0; n],
            a,
            vec![rhs],
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let mut opts = SolverOptions::default();
        opts.ipm.eps = 1e-6;
        let res = crate::qp::solve_qp_with(&prob, &opts);
        (prob, res)
    }

    /// Converged optimal solution → predicate true (gate fires).
    #[test]
    fn already_passes_true_for_converged() {
        for &(n, diag, rhs) in &[(3usize, 1.0, 2.0), (5, 4.0, 1.0)] {
            let (prob, res) = solved(n, diag, rhs);
            assert!(
                kkt_already_passes(&prob, &res, &[], true, 1e-6),
                "n={n} diag={diag}: exact optimal must already pass"
            );
        }
    }

    /// Same solution but the dual is corrupted → stationarity residual ≫ eps →
    /// predicate false → the gate must NOT skip the IR.
    #[test]
    fn already_passes_false_when_dual_violates_stationarity() {
        let (prob, mut res) = solved(4, 2.0, 1.0);
        assert!(kkt_already_passes(&prob, &res, &[], true, 1e-6));
        for y in res.dual_solution.iter_mut() {
            *y += 100.0;
        }
        assert!(
            !kkt_already_passes(&prob, &res, &[], true, 1e-6),
            "corrupted dual breaks stationarity → must not be considered already-passing"
        );
    }

    /// Non-Optimal IPM status → predicate false even with an exact solution
    /// (the gate only skips for confirmed-Optimal results).
    #[test]
    fn already_passes_false_when_status_not_optimal() {
        let (prob, res) = solved(3, 1.0, 2.0);
        assert!(kkt_already_passes(&prob, &res, &[], true, 1e-6));
        assert!(
            !kkt_already_passes(&prob, &res, &[], false, 1e-6),
            "status != Optimal must gate the skip off"
        );
    }

    /// Stationarity holds (kkt = 0) but the primal constraint is violated
    /// (pres ≫ eps). Gives the `pres < eps` conjunct teeth: dropping it from
    /// `kkt_already_passes` would return `true` here and fail this assertion.
    ///
    /// For min 0.5·Σx² s.t. Σx = rhs (x free), the point x=0, y=0 satisfies
    /// stationarity (Qx + c − Aᵀy = 0) exactly, yet Σx = 0 ≠ rhs.
    #[test]
    fn already_passes_false_when_primal_infeasible_despite_stationarity() {
        for &(n, rhs) in &[(4usize, 5.0_f64), (6, -3.0)] {
            let idx: Vec<usize> = (0..n).collect();
            let q = CscMatrix::from_triplets(&idx, &idx, &vec![1.0; n], n, n).unwrap();
            let a = CscMatrix::from_triplets(&vec![0usize; n], &idx, &vec![1.0; n], 1, n).unwrap();
            let prob = QpProblem::new(
                q,
                vec![0.0; n],
                a,
                vec![rhs],
                vec![(f64::NEG_INFINITY, f64::INFINITY); n],
                vec![ConstraintType::Eq],
            )
            .unwrap();
            let res = crate::problem::SolverResult {
                solution: vec![0.0; n],
                dual_solution: vec![0.0; 1],
                bound_duals: vec![],
                ..Default::default()
            };
            // sanity: stationarity residual is ~0 but primal residual is large.
            let view = build_view(&prob, &[]);
            assert!(
                kkt_residual_rel(&view, &res.solution, &res.dual_solution, &res.bound_duals) < 1e-6
            );
            assert!(primal_residual_rel(&view, &res.solution) > 1e-6);
            assert!(
                !kkt_already_passes(&prob, &res, &[], true, 1e-6),
                "n={n} rhs={rhs}: primal-infeasible point must not be already-passing \
                 (the `pres < eps` conjunct must hold)"
            );
        }
    }

    /// kkt + pres both hold exactly, but bound dual is corrupted to give large
    /// complementarity → comp ≫ eps → predicate must return false (sentinel for
    /// the new comp check in kkt_already_passes).
    ///
    /// Problem: min 0.5·x², Ax=1 (Eq), x ≥ 0.
    /// Optimal: x=1, z_lb=0, y=-1  (stationarity: Qx+c+Aᵀy−z_lb = 1+0−1−0 = 0).
    ///
    /// Corrupt: z_lb=1e-3, y=z_lb−1=−0.999 so stationarity still holds exactly
    ///   (1 + 0 + Aᵀ·y − z_lb = 1 − 0.999 − 1e-3 = 0), but
    ///   comp = z_lb·(x−lb)/scale = 1e-3·1/~3.5 ≈ 2.9e-4 ≫ user_eps = 1e-6.
    ///
    /// Important: this case must fail under the "kkt+pres only" gate (the pre-fix
    /// behaviour) so that reverting kkt_already_passes to check only kkt+pres causes
    /// this test to FAIL (proving the sentinel is load-bearing for the comp check).
    #[test]
    fn already_passes_false_when_comp_fails() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0],
            a,
            vec![1.0],
            vec![(0.0_f64, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();

        // Baseline: optimal solution (z_lb=0, comp=0) must pass the gate.
        let mut res = crate::problem::SolverResult {
            solution: vec![1.0],
            dual_solution: vec![-1.0],
            bound_duals: vec![0.0],
            ..Default::default()
        };
        assert!(
            kkt_already_passes(&prob, &res, &[], true, 1e-6),
            "baseline optimal (z_lb=0, comp=0) must pass the gate"
        );

        // Corrupt: z_lb=1e-3, y=z_lb-1=-0.999.
        // Stationarity: Qx+c+Aᵀy−z_lb = 1+(−0.999)−1e-3 = 0 → kkt=0.
        // Primal: Ax=1=b → pres=0.
        // Complementarity: z_lb·(x−lb)/scale = 1e-3·1/~3.5 ≈ 2.9e-4 ≫ 1e-6.
        // Gate must return false (comp fails); reverting to kkt+pres-only would
        // return true (kkt=pres=0), causing this assertion to fail → sentinel fires.
        res.dual_solution = vec![-0.999_f64];
        res.bound_duals = vec![1e-3_f64];
        assert!(
            !kkt_already_passes(&prob, &res, &[], true, 1e-6),
            "z_lb=1e-3, y=-0.999: stationarity holds but comp≈2.9e-4≫1e-6 → gate must NOT skip IR"
        );
    }

    #[test]
    fn already_passes_false_when_only_componentwise_comp_fails() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![-1.0e12_f64],
            a,
            vec![1.0_f64],
            vec![(0.0_f64, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let res = crate::problem::SolverResult {
            solution: vec![1.0_f64],
            dual_solution: vec![1.0e12_f64 + 1.0e-3_f64],
            bound_duals: vec![1.0e-3_f64],
            ..Default::default()
        };
        let view = build_view(&prob, &[]);
        let aggregate = complementarity_residual_rel(
            &view,
            &res.solution,
            &res.dual_solution,
            &res.bound_duals,
        );
        let componentwise = complementarity_componentwise_rel(
            &view,
            &res.solution,
            &res.dual_solution,
            &res.bound_duals,
        );
        assert!(
            aggregate < 1e-6,
            "fixture must keep aggregate comp below eps, got {aggregate:.3e}"
        );
        assert!(
            componentwise > 1e-6,
            "fixture must expose componentwise comp above eps, got {componentwise:.3e}"
        );
        assert!(
            !kkt_already_passes(&prob, &res, &[], true, 1e-6),
            "componentwise comp failure must prevent post-processing skip"
        );
    }

    /// Sentinel Fix-2: dual-sign violation alone (kkt/pres/bv/comp/gap all pass within eps)
    /// must force kkt_already_passes to return false so Stage-2 IR is NOT skipped.
    ///
    /// Problem: min 0.5·x² − 2·x, Le: x ≤ 1, x ∈ [0,1]. Baseline optimum x=1, y_Le=1.
    /// Corrupt: y_Le=−1e-4 (Le sign violation), z_ub=1+1e-4. Then stat/pres/bv/comp/gap
    /// all evaluate to 0 within eps — only dual_sign sees y_Le<0, violation ≈1e-4 ≫1e-6.
    ///
    /// Reverting Fix 2 (removing the dual_sign check from kkt_already_passes) makes the gate
    /// return true here (all other checks pass), causing this assertion to fail → sentinel fires.
    #[test]
    fn already_passes_false_when_dual_sign_violated() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![-2.0_f64],
            a,
            vec![1.0_f64],
            vec![(0.0_f64, 1.0_f64)],
            vec![ConstraintType::Le],
        )
        .unwrap();

        // Baseline: x=1, y_Le=1, z_lb=0, z_ub=0.
        // bound_duals layout: n_lb_finite=1, n_ub_finite=1 → [z_lb, z_ub].
        // stat: 1−2+1+(−0+0)=0, pres=0, bv=0, comp=0, gap=0, dsign=0 → gate passes.
        let baseline = crate::problem::SolverResult {
            solution: vec![1.0_f64],
            dual_solution: vec![1.0_f64],
            bound_duals: vec![0.0_f64, 0.0_f64],
            ..Default::default()
        };
        assert!(
            kkt_already_passes(&prob, &baseline, &[], true, 1e-6),
            "baseline optimal (y_Le=1, z_lb=0, z_ub=0) must pass the gate"
        );

        // Corrupt: y_Le=−1e-4, z_ub=1+1e-4. stat=0, comp=0, gap=0, but dsign≈1e-4≫1e-6.
        // Without Fix 2 the gate returns true (kkt/pres/bv/comp/gap all pass); sentinel fires.
        let corrupt = crate::problem::SolverResult {
            solution: vec![1.0_f64],
            dual_solution: vec![-1e-4_f64],
            bound_duals: vec![0.0_f64, 1.0_f64 + 1e-4_f64],
            ..Default::default()
        };
        assert!(
            !kkt_already_passes(&prob, &corrupt, &[], true, 1e-6),
            "y_Le=−1e-4 violates Le sign (dsign≈1e-4≫1e-6) → gate must NOT skip Stage-2 IR; \
             reverting the dual_sign check from kkt_already_passes would return true here → sentinel fires"
        );
    }

    /// LISWET7 regression: kkt < skip_tol だが pres ∈ [skip_tol, user_eps) → false を返す。
    ///
    /// ill-conditioned QP (LISWET7: cond(AAᵀ) ≈ 1e16) では f64 算術が kkt ≈ 3e-11,
    /// pres ≈ 9e-8 の偽収束残差を生成する。旧コード (threshold = user_eps) では kkt も pres も
    /// user_eps = 1e-6 未満として skip → prove_optimal が偽 Optimal を返していた。
    /// fix では skip_tol = user_eps / KKT_SKIP_MARGIN (100) に厳格化し refinement を強制。
    ///
    /// 構成: min 0.5·x², s.t. x = 1 (Eq), x ∈ (-∞,+∞)。
    /// 停留性 (Qx + c + Aᵀy = 0): x + y = 0 → y = -x。
    /// テスト点: x = 1 + δ, y = -(1 + δ) → kkt = 0 (完全停留), pres = δ/(3+δ) ≈ δ/3。
    ///
    /// Sentinel: pres 閾値を skip_tol → user_eps に戻すと pres=6.7e-8 < user_eps なので
    /// kkt_already_passes が true を返し、このアサーションが失敗する → fix が load-bearing。
    #[test]
    fn already_passes_false_when_pres_between_skip_tol_and_eps() {
        let user_eps = 1e-6_f64;
        let skip_tol = user_eps / super::KKT_SKIP_MARGIN;

        // min 0.5·x², s.t. x = 1, x free
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0_f64],
            a,
            vec![1.0_f64],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
            vec![ConstraintType::Eq],
        )
        .unwrap();

        // δ = 2e-7: pres = δ/(1 + |1+δ| + |1|) ≈ δ/3 ≈ 6.7e-8 ∈ [skip_tol, user_eps)
        // y = -(1+δ) により停留性 x + y = (1+δ) + (-(1+δ)) = 0 → kkt = 0
        let delta = 2e-7_f64;
        let x = 1.0_f64 + delta;
        let y = -(1.0_f64 + delta);
        let res = crate::problem::SolverResult {
            solution: vec![x],
            dual_solution: vec![y],
            bound_duals: vec![],
            ..Default::default()
        };

        let view = build_view(&prob, &[]);
        let kkt = kkt_residual_rel(&view, &res.solution, &res.dual_solution, &res.bound_duals);
        let pres = primal_residual_rel(&view, &res.solution);

        assert!(
            kkt < skip_tol,
            "fixture: kkt={kkt:.3e} must be < skip_tol={skip_tol:.3e}"
        );
        assert!(
            pres >= skip_tol && pres < user_eps,
            "fixture: pres={pres:.3e} must be in [skip_tol={skip_tol:.3e}, user_eps={user_eps:.3e})"
        );
        assert!(
            !kkt_already_passes(&prob, &res, &[], true, user_eps),
            "kkt={kkt:.3e} < skip_tol but pres={pres:.3e} ≥ skip_tol \
             → post-processing must NOT be skipped (LISWET7 false-Optimal regression)"
        );
    }

    /// Degenerate inputs short-circuit to false: empty solution and m=0.
    #[test]
    fn already_passes_false_for_degenerate_inputs() {
        // empty solution
        let (prob, mut res) = solved(3, 1.0, 2.0);
        res.solution = vec![];
        assert!(
            !kkt_already_passes(&prob, &res, &[], true, 1e-6),
            "empty solution must short-circuit to false"
        );
        // m = 0 (no constraints): there is nothing the IR refines.
        let n = 3usize;
        let idx: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&idx, &idx, &vec![1.0; n], n, n).unwrap();
        let a = CscMatrix::new(0, n);
        let prob0 = QpProblem::new(
            q,
            vec![0.0; n],
            a,
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
            vec![],
        )
        .unwrap();
        let res0 = crate::problem::SolverResult {
            solution: vec![0.0; n],
            ..Default::default()
        };
        assert!(
            !kkt_already_passes(&prob0, &res0, &[], true, 1e-6),
            "m=0 must short-circuit to false"
        );
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod stall_gate_tests {
    //! Sentinels for the relative progress stall gate (`refit_progress_stalled`).
    //!
    //! The refit/IRLS loops break when one iteration's KKT improvement is
    //! indistinguishable from f64 accumulation noise. The gate combines a
    //! relative term (`REFIT_REL_STALL · prev`) — for the ill-conditioned regime
    //! where KKT is pinned at the f64 limit (~0.1) and only relative noise
    //! (~1e-10) moves — with an absolute floor (`REFIT_PROGRESS_EPS`) for the
    //! near-zero regime. Reverting to the absolute-only gate re-introduces the
    //! 979s LISWET spin (see `detects_noise_floor_flat_residual`).
    use super::{
        build_view, cleanup_inactive_dual_complementarity, dual_certificate_residual_max,
        kkt_residual_rel, refine_post_processing, refit_progress_stalled, REFIT_PROGRESS_EPS,
    };
    use crate::options::SolverOptions;
    use crate::problem::{ConstraintType, SolverResult};
    use crate::qp::problem::QpProblem;
    use crate::sparse::CscMatrix;

    /// ill-cond regime: KKT pinned at the f64 limit (~0.1), per-iteration drop is
    /// only relative noise (~1e-11 abs, ~1e-10 rel). The relative gate must flag
    /// this as stall. Reverting to the absolute 1e-12 floor would NOT (1e-11 >
    /// 1e-12) → the loop spins to the deadline (the original 979s LISWET hang).
    #[test]
    fn detects_noise_floor_flat_residual() {
        let prev = 1e-1_f64;
        let cur = prev - 1e-11_f64; // relative drop ~1e-10, absolute drop 1e-11
        assert!(
            refit_progress_stalled(prev, cur),
            "flat residual at noise floor must be stall (abs drop {:.1e}, rel {:.1e})",
            prev - cur,
            (prev - cur) / prev
        );
        // Pin the failure mode of the abandoned absolute-only gate: with a 1e-12
        // floor the same point is (wrongly) seen as progress.
        assert!(
            cur + REFIT_PROGRESS_EPS < prev,
            "absolute-only gate would treat this noise drop as progress → spin"
        );
    }

    /// Genuine progress — even slow linear convergence — must NOT be flagged as
    /// stall. If `REFIT_REL_STALL` were set too aggressively the loop would break
    /// early and degrade the solution.
    #[test]
    fn keeps_going_on_meaningful_relative_progress() {
        // (prev, cur): relative drops 0.5, 1e-4, 1e-5 — all ≫ REFIT_REL_STALL.
        for &(prev, cur) in &[
            (1e-1_f64, 5e-2_f64),
            (1.0_f64, 1.0_f64 - 1e-4_f64),
            (1e3_f64, 1e3_f64 - 1e-2_f64),
        ] {
            assert!(
                !refit_progress_stalled(prev, cur),
                "relative drop {:.1e} is meaningful → must continue",
                (prev - cur) / prev
            );
        }
    }

    /// near-zero regime: the relative threshold underflows (`REFIT_REL_STALL·prev`
    /// ≪ floor), so the absolute floor must still terminate the loop. Removing the
    /// `.max(REFIT_PROGRESS_EPS)` floor (pure-relative gate) would spin here.
    #[test]
    fn absolute_floor_terminates_near_zero() {
        let prev = 1e-12_f64;
        let cur = prev - 1e-13_f64;
        assert!(
            refit_progress_stalled(prev, cur),
            "near-zero residual: absolute floor must detect stall"
        );
        // Exact no-change is always stall regardless of magnitude.
        assert!(refit_progress_stalled(0.5, 0.5));
    }

    #[test]
    fn cleanup_inactive_duals_repairs_complementarity_without_moving_primal() {
        let q = CscMatrix::from_triplets(
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[0, 0, 0, 1, 1, 1, 2, 2, 2],
            &[
                0.4180730178857338,
                -0.3388321916091331,
                -0.11128811645712924,
                -0.3388321916091331,
                -0.998339628454278,
                0.7899096104935973,
                -0.11128811645712924,
                0.7899096104935973,
                0.16528704821825724,
            ],
            3,
            3,
        )
        .unwrap();
        let a = CscMatrix::from_triplets(
            &[0, 2, 2, 1, 2],
            &[0, 0, 1, 2, 2],
            &[
                0.20362059179708952,
                -0.49156994635882434,
                0.6899251194067307,
                0.35089945702726577,
                -0.20609706471316191,
            ],
            3,
            3,
        )
        .unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.9627433629541278, 1.1223543296642184, 0.7793851438408713],
            a,
            vec![-2.8845159230823487, -2.3497565244678635, 2.748374601409619],
            vec![
                (-2.770611501419692, 2.770611501419692),
                (-0.8857981031127659, 0.8857981031127659),
                (-0.9049511230872764, 0.9049511230872764),
            ],
            vec![ConstraintType::Ge, ConstraintType::Ge, ConstraintType::Le],
        )
        .unwrap();
        let x = vec![-2.770611501419515, -0.8857981031127654, -0.9049511230872724];
        let mut sol = SolverResult {
            solution: x.clone(),
            dual_solution: vec![
                -1.0081132368485388,
                -0.6795226652062781,
                4.512057531342496e-8,
            ],
            bound_duals: vec![
                3.3676306743224416e-8,
                2.230624487910973,
                1.1878225070205417e-9,
                0.0,
                0.0,
                0.0,
            ],
            ..Default::default()
        };
        let view = build_view(&prob, &[]);
        let before = dual_certificate_residual_max(&prob, &view, &sol);
        assert!(
            before > 0.5,
            "fixture must expose the inactive-row dual bug"
        );

        cleanup_inactive_dual_complementarity(&prob, &mut sol, &[], 1e-6);
        let after = dual_certificate_residual_max(&prob, &view, &sol);

        assert!(
            after < 1e-10,
            "cleanup must repair KKT certificate, got {after:.3e}"
        );
        assert_eq!(sol.solution, x, "dual cleanup must not move primal x");
        assert!(sol.dual_solution.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn cleanup_rejects_candidate_that_improves_complementarity_but_worsens_gap() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::new(1, 1);
        let prob = QpProblem::new(
            q,
            vec![1.0_f64],
            a,
            vec![1.0_f64],
            vec![(1.0_f64, 3.0_f64)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let mut sol = SolverResult {
            solution: vec![2.0_f64],
            dual_solution: vec![0.0_f64],
            bound_duals: vec![100.0_f64, 98.0_f64 / 3.0_f64],
            ..Default::default()
        };
        let original = sol.clone();
        let view = build_view(&prob, &[]);
        let before = dual_certificate_residual_max(&prob, &view, &sol);
        let gap_before = super::super::compute_duality_gap_rel(&prob, &sol);

        let mut candidate = sol.clone();
        crate::qp::zero_inactive_inequality_duals(&prob, &mut candidate);
        crate::qp::refit_bound_duals_kkt(&prob, &mut candidate, 1e-6);
        let after = dual_certificate_residual_max(&prob, &view, &candidate);
        let gap_after = super::super::compute_duality_gap_rel(&prob, &candidate);

        assert!(
            after < before,
            "fixture must improve the non-gap certificate ({before:.3e} -> {after:.3e})"
        );
        assert!(
            gap_after > gap_before,
            "fixture must worsen duality gap ({gap_before:.3e} -> {gap_after:.3e})"
        );

        cleanup_inactive_dual_complementarity(&prob, &mut sol, &[], 1e-6);
        assert_eq!(
            sol.dual_solution, original.dual_solution,
            "gap-worsening cleanup candidate must be rejected"
        );
        assert_eq!(
            sol.bound_duals, original.bound_duals,
            "gap-worsening bound-dual refit must be rejected"
        );
    }

    /// Convergence case unchanged: a well-conditioned Eq QP whose dual starts
    /// perturbed. The refit loop must drive KKT below user_eps (genuine
    /// convergence is NOT cut short by the relative gate) while leaving the primal
    /// `x` untouched (allow_primal=false). The relative gate only changes
    /// behaviour when KKT is pinned ≫ floor with noise-level drops; once KKT
    /// converges toward zero the absolute floor governs — identical to the old
    /// gate. An over-aggressive `REFIT_REL_STALL` would break before y is refined,
    /// leaving KKT large → this test fails.
    #[test]
    fn refit_converges_and_preserves_primal() {
        let n = 4usize;
        let idx: Vec<usize> = (0..n).collect();
        // min 0.5·Σx² s.t. Σx = 2 (Eq), x free. Optimum x_i = 0.5, dual fixed by
        // stationarity. x is already optimal; only the dual needs refitting.
        let q = CscMatrix::from_triplets(&idx, &idx, &vec![1.0_f64; n], n, n).unwrap();
        let a = CscMatrix::from_triplets(&vec![0usize; n], &idx, &vec![1.0_f64; n], 1, n).unwrap();
        let prob = QpProblem::new(
            q,
            vec![0.0_f64; n],
            a,
            vec![2.0_f64],
            vec![(f64::NEG_INFINITY, f64::INFINITY); n],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let x_opt = vec![0.5_f64; n];
        let mut sol = SolverResult {
            solution: x_opt.clone(),
            dual_solution: vec![0.0_f64], // perturbed away from the optimal dual
            bound_duals: vec![],
            ..Default::default()
        };
        let mut opts = SolverOptions::default();
        opts.ipm.eps = 1e-6;

        let view = build_view(&prob, &[]);
        let kkt_before = kkt_residual_rel(
            &view,
            &sol.solution,
            &sol.dual_solution,
            &sol.bound_duals,
        );
        assert!(kkt_before > 1e-3, "fixture must start with a real dual residual");

        let kkt_after = refine_post_processing(&prob, &mut sol, &[], &opts, false);

        assert!(
            kkt_after <= opts.ipm.eps,
            "refit must converge below user_eps (kkt {kkt_before:.3e} -> {kkt_after:.3e}); \
             an over-aggressive relative gate would break early here"
        );
        for (i, &xi) in sol.solution.iter().enumerate() {
            assert!(
                (xi - x_opt[i]).abs() < 1e-12,
                "primal x must be untouched: x[{i}]={xi} != {}",
                x_opt[i]
            );
        }
    }
}
