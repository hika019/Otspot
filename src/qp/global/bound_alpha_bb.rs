//! α-BB lower bound (#7 Phase 4 非凸 QP 大域最適化)。
//!
//! ## 原理
//! Maranas-Floudas (1995) underestimator:
//!
//!   L(x; α, l, u) = f(x) + α Σ_i (x_i − l_i)(x_i − u_i)
//!
//! `(x_i − l_i)(x_i − u_i) ≤ 0` on box `[l, u]`、端点で `= 0` なので `L ≤ f` 全域、
//! corner では `L = f`。Hessian は `∇²L = Q + 2α·I` で、`2α ≥ −λ_min(Q)` を満たせば PSD
//! (= L は box 上の convex underestimator)。
//!
//! 凸化問題は box constraint + 既存線形制約 `Ax {=,≤,≥} b` をそのまま使い、
//! 既存 IpPMM (`solve_qp_with`) を呼ぶだけで box+linear で global min が得られる。
//! その obj 値 = 元 non-convex f の box 上 lower bound。
//!
//! ## α 計算
//! raw Gershgorin で δ s.t. `Q + δ·I` PSD を計算 (`gershgorin_alpha`)、`α = δ / 2`
//! で `Q + 2α·I` PSD = α-BB の要求を満たす。LDL^T 経路は α-BB に不要なオーバヘッドで、
//! 凸化 lb の保守性を素直に表せる raw Gershgorin を独立実装する。
//!
//! ## semi-infinite box
//! `(x_i − l_i)(x_i − u_i)` 項は有限境界を要求する。l_i や u_i が ±∞ の変数があれば
//! α-BB underestimator は box 上で `-∞` まで落ちうるので意味のある lb にならない。
//! その場合 `None` を返し、caller (= mod.rs) は interval lb / `-∞` に fall back する。

use std::time::Instant;

use crate::linalg::gershgorin::psd_shift_from_gershgorin;
use crate::options::SolverOptions;
use crate::problem::SolverResult;
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

use super::bound::{all_bounds_finite, is_feasible_result};

/// α-BB underestimator `Q + 2α·I` が PSD となる最小 α を Gershgorin で取得。
///
/// `α = max(0, max_j(R_j − Q[j,j])) / 2 = psd_shift_from_gershgorin(Q) / 2`。
/// 共通 helper は `linalg::gershgorin` を参照。
pub(crate) fn gershgorin_alpha(q: &CscMatrix) -> f64 {
    0.5 * psd_shift_from_gershgorin(q)
}

/// `Q` の対角に `value` を加えた新しい CSC を返す。
///
/// Q の対角 entry が存在しない列にも `value` の単独 entry を生成する。
/// `from_triplets` の重複 entry sum 仕様 (csc.rs::test_from_triplets_duplicate_entries)
/// を利用して既存 entry と新 diag 寄与を統合する。
fn add_scalar_to_diagonal(q: &CscMatrix, value: f64) -> CscMatrix {
    let n = q.nrows;
    debug_assert_eq!(q.nrows, q.ncols, "Q must be square");
    let mut rows = Vec::with_capacity(q.values.len() + n);
    let mut cols = Vec::with_capacity(q.values.len() + n);
    let mut vals = Vec::with_capacity(q.values.len() + n);
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            rows.push(q.row_ind[k]);
            cols.push(col);
            vals.push(q.values[k]);
        }
        rows.push(col);
        cols.push(col);
        vals.push(value);
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).expect("from_triplets diag add")
}

/// α-BB underestimator 凸化問題を構築。
///
/// `L(x) = 0.5 x'(Q + 2α·I)x + (c − α(l+u))' x + (obj_offset + α Σ l_i u_i)`。
/// box / 線形制約 / constraint type は元と同一。
fn build_convex_relaxation(
    problem: &QpProblem,
    node_bounds: &[(f64, f64)],
    alpha: f64,
) -> QpProblem {
    let n = problem.num_vars;
    debug_assert_eq!(node_bounds.len(), n);
    let mut sub = problem.clone();
    sub.q = add_scalar_to_diagonal(&problem.q, 2.0 * alpha);
    let mut new_offset = problem.obj_offset;
    for i in 0..n {
        let (l, u) = node_bounds[i];
        sub.c[i] = problem.c[i] - alpha * (l + u);
        new_offset += alpha * l * u;
    }
    sub.obj_offset = new_offset;
    sub.bounds = node_bounds.to_vec();
    sub
}

/// α-BB lower bound on the given box。
///
/// 戻り値 `Some(lb)`: convex relaxation が feasible 解を返した = lb は元問題下界として有効。
/// 戻り値 `None`: semi-infinite box / α=0 (= Q PSD で α-BB は trivial、caller 側で
/// `solve_local_upper_bound` の obj が同時に lb になる) / convex solve が infeasible 等で
/// lb を得られない。caller は interval lb (or `-∞`) へ fall back する。
///
/// convex relaxation は cold solve で実行 (= warm 継承なし、`opts.warm_start_qp = None`)。
/// 元 non-convex の warm は凸化後の最適解と一致せず再固着 risk があるため。
pub(crate) fn alpha_bb_lower_bound(
    problem: &QpProblem,
    node_bounds: &[(f64, f64)],
    alpha: f64,
    base_opts: &SolverOptions,
    deadline: Option<Instant>,
) -> Option<f64> {
    if alpha <= 0.0 {
        return None;
    }
    if !all_bounds_finite(node_bounds) {
        return None;
    }
    let sub = build_convex_relaxation(problem, node_bounds, alpha);
    let mut opts = base_opts.clone();
    opts.multistart = None;
    opts.global_optimization = None;
    // sub-solve hygiene: caller の warm hint は凸化後の解空間で意味が変わるため全消去
    opts.warm_start = None;
    opts.warm_start_qp = None;
    opts.warm_start_lp = None;
    opts.deadline = deadline;
    let res: SolverResult = crate::qp::solve_qp_with(&sub, &opts);
    if !is_feasible_result(&res.status) {
        return None;
    }
    Some(res.objective)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    fn build_problem(
        diag: &[f64],
        c: &[f64],
        bounds: Vec<(f64, f64)>,
    ) -> QpProblem {
        let n = diag.len();
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&rows, &cols, diag, n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        QpProblem::new(q, c.to_vec(), a, vec![], bounds, vec![]).unwrap()
    }

    #[test]
    fn gershgorin_alpha_zero_on_psd() {
        // Q = diag(2, 2) → PSD → α = 0
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        assert_eq!(gershgorin_alpha(&q), 0.0);
    }

    #[test]
    fn gershgorin_alpha_positive_on_concave_diag() {
        // Q = diag(-2, -2) → λ_min = -2 → δ = 2 → α = 1
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        assert!((gershgorin_alpha(&q) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gershgorin_alpha_covers_off_diag_full_symmetric() {
        // bilinear Q full-symmetric: [[0,1],[1,0]] → Gershgorin row sum = 1 → δ ≥ 1 → α ≥ 0.5
        let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
        let a = gershgorin_alpha(&q);
        assert!(a >= 0.5 - 1e-12, "alpha must be at least 0.5, got {a}");
    }

    #[test]
    fn add_scalar_to_diagonal_inserts_missing_diag() {
        // Q with no diag entries → adding 3.0 yields pure diag matrix
        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
        let r = add_scalar_to_diagonal(&q, 3.0);
        assert_eq!(r.nnz(), 2);
        for col in 0..2 {
            let (ri, vi) = r.get_column(col).unwrap();
            assert_eq!(ri, &[col]);
            assert!((vi[0] - 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn add_scalar_to_diagonal_sums_existing_diag() {
        // Q diag=[1.0, 2.0]; add 5.0 → diag=[6.0, 7.0]
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 2.0], 2, 2).unwrap();
        let r = add_scalar_to_diagonal(&q, 5.0);
        let (_r0, v0) = r.get_column(0).unwrap();
        let (_r1, v1) = r.get_column(1).unwrap();
        assert!((v0[0] - 6.0).abs() < 1e-12);
        assert!((v1[0] - 7.0).abs() < 1e-12);
    }

    /// Evaluate L(x) = 0.5 x'Qx + c'x + offset using mat_vec_mul (storage agnostic).
    fn eval_obj(p: &QpProblem, x: &[f64]) -> f64 {
        let qx = p.q.mat_vec_mul(x).expect("mat_vec_mul");
        let xtqx: f64 = x.iter().zip(qx.iter()).map(|(xi, qxi)| xi * qxi).sum();
        let ctx: f64 = x.iter().zip(p.c.iter()).map(|(xi, ci)| xi * ci).sum();
        0.5 * xtqx + ctx + p.obj_offset
    }

    #[test]
    fn relaxation_matches_original_at_corners() {
        // f(x) = -x², box [-2, 2], corner x=2: f(2)=-4. L(2)=-4 because (x-l)(x-u)=0 at x=2.
        let problem = build_problem(&[-2.0], &[0.0], vec![(-2.0, 2.0)]);
        let alpha = gershgorin_alpha(&problem.q);
        assert!(alpha > 0.0);
        let sub = build_convex_relaxation(&problem, &problem.bounds, alpha);
        let f_orig = eval_obj(&problem, &[2.0]);
        let l_relax = eval_obj(&sub, &[2.0]);
        assert!(
            (l_relax - f_orig).abs() < 1e-9,
            "L at corner should equal f: f={f_orig}, L={l_relax}"
        );
        // Same on lower corner
        let f_lo = eval_obj(&problem, &[-2.0]);
        let l_lo = eval_obj(&sub, &[-2.0]);
        assert!((l_lo - f_lo).abs() < 1e-9, "L at lower corner: f={f_lo}, L={l_lo}");
    }

    #[test]
    fn relaxation_underestimates_strictly_in_interior() {
        // -x² on [-2, 2]: f(0) = 0. L(0) should be strictly negative.
        let problem = build_problem(&[-2.0], &[0.0], vec![(-2.0, 2.0)]);
        let alpha = gershgorin_alpha(&problem.q);
        let sub = build_convex_relaxation(&problem, &problem.bounds, alpha);
        let f_mid = eval_obj(&problem, &[0.0]);
        let l_mid = eval_obj(&sub, &[0.0]);
        assert!(
            l_mid <= f_mid + 1e-12,
            "L({{0}})={l_mid} must underestimate f({{0}})={f_mid}"
        );
        // For this concave -x², L(0) - f(0) = α*l*u = 1.0 * (-2) * 2 = -4
        assert!(
            (l_mid - f_mid - (-4.0)).abs() < 1e-9,
            "L(0)-f(0) should be -4, got {}",
            l_mid - f_mid
        );
    }

    #[test]
    fn alpha_bb_returns_none_for_infinite_box() {
        let problem = build_problem(&[-2.0], &[0.0], vec![(f64::NEG_INFINITY, 1.0)]);
        let alpha = 1.0;
        let opts = SolverOptions::default();
        assert!(alpha_bb_lower_bound(&problem, &problem.bounds, alpha, &opts, None).is_none());
    }

    #[test]
    fn alpha_bb_returns_none_for_zero_alpha() {
        let problem = build_problem(&[2.0], &[0.0], vec![(-1.0, 1.0)]);
        let opts = SolverOptions::default();
        assert!(alpha_bb_lower_bound(&problem, &problem.bounds, 0.0, &opts, None).is_none());
    }

    #[test]
    fn alpha_bb_yields_finite_lb_for_concave_box() {
        // f = -x², box [-2, 2], global min = -4. α-BB lb ≤ -4 (valid lb) but tighter than -∞.
        let problem = build_problem(&[-2.0], &[0.0], vec![(-2.0, 2.0)]);
        let alpha = gershgorin_alpha(&problem.q);
        let opts = SolverOptions::default();
        let lb = alpha_bb_lower_bound(&problem, &problem.bounds, alpha, &opts, None)
            .expect("convex relaxation must solve on bounded concave 1d");
        // lb must be a valid lower bound (≤ -4) and finite
        assert!(lb.is_finite(), "lb must be finite, got {lb}");
        assert!(lb <= -4.0 + 1e-6, "lb must be ≤ global -4, got {lb}");
    }

    #[test]
    fn alpha_bb_lb_tightens_as_box_shrinks() {
        // 同じ凸化、box 縮小で lb は上がる (枝刈効果の数学的根拠)。
        let problem_wide = build_problem(&[-2.0], &[0.0], vec![(-2.0, 2.0)]);
        let alpha = gershgorin_alpha(&problem_wide.q);
        let opts = SolverOptions::default();
        let lb_wide = alpha_bb_lower_bound(&problem_wide, &[(-2.0, 2.0)], alpha, &opts, None)
            .expect("wide");
        let lb_narrow = alpha_bb_lower_bound(&problem_wide, &[(0.0, 1.0)], alpha, &opts, None)
            .expect("narrow");
        assert!(
            lb_narrow >= lb_wide - 1e-9,
            "narrow lb ({lb_narrow}) should not be worse than wide ({lb_wide})"
        );
    }

    #[test]
    fn alpha_bb_with_linear_eq_constraint() {
        // x+y=1, min -x²-y² with x,y in [0,1]: global = -1 at corner (1,0) or (0,1).
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, -2.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let p = QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![1.0],
            vec![(0.0, 1.0); 2],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let alpha = gershgorin_alpha(&p.q);
        let opts = SolverOptions::default();
        let lb = alpha_bb_lower_bound(&p, &p.bounds, alpha, &opts, None)
            .expect("constrained α-BB must solve");
        assert!(lb.is_finite() && lb <= -1.0 + 1e-6,
            "lb must be ≤ -1 (global) and finite, got {lb}");
    }
}
