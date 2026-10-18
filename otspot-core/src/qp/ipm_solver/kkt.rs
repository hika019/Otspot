//! 元空間 KKT 残差 (bench compute_dfeas_orig と同形・成分相対化)。

use super::outcome::ProblemView;
use crate::problem::ConstraintType;
use crate::qp::kkt_resid::{self, dd_impl};
use crate::tolerances::{any_nonfinite, FX_TOL};

/// 成分相対化 stationarity max_j |r_j|/(1+|Qx_j|+|c_j|+|Aᵀy_j|+|z_j|) を DD 精度で計算。
/// FX (lb≈ub) は postsolve 慣例で除外。
///
/// EmptyCol skip は `eliminated_cols[j]==true` **AND** orig 行列で A 列空 **AND** Q 列空
/// の三条件 (= LP-style 完全孤立 var) のみ narrow:
/// - 非凸 QP の 26 eliminated 群 (A 非空 / Q 空) は skip されず measurement に出る
///   (これらを除くと refine 系の進捗 oracle として誤動作)。
/// - linear-only var (A 空 / Q 非空 / c≠0) は skip されず stationarity 露出維持。
/// - LP-style 完全孤立 var (A 空 / Q 空) のみ skip し cancellation noise 抑制。
pub fn kkt_residual_rel(prob: &ProblemView, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    let n = prob.bounds.len();
    if x.len() != n {
        return f64::INFINITY;
    }
    if !kkt_resid::bound_duals_valid_for_residual(prob.bounds, z) {
        return f64::INFINITY;
    }
    let qx_dd = dd_impl::qx(prob.q, x);
    let aty_dd = dd_impl::aty(prob.a, y, n);
    let bound_contrib = kkt_resid::bound_contrib(prob.bounds, z);
    let use_elim_mask = prob.eliminated_cols.len() == n;
    let a_col_count = prob.a.col_ptr().len().saturating_sub(1);
    let q_col_count = prob.q.col_ptr().len().saturating_sub(1);
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
            continue;
        }
        if use_elim_mask && prob.eliminated_cols[j] {
            let a_empty = j >= a_col_count || prob.a.col_ptr()[j + 1] == prob.a.col_ptr()[j];
            let q_empty = j >= q_col_count || prob.q.col_ptr()[j + 1] == prob.q.col_ptr()[j];
            if a_empty && q_empty {
                continue;
            }
        }
        let r_dd =
            qx_dd[j] + TwoFloat::from(prob.c[j]) + aty_dd[j] + TwoFloat::from(bound_contrib[j]);
        let r = f64::from(r_dd).abs();
        let qx_j = f64::from(qx_dd[j]).abs();
        let aty_j = f64::from(aty_dd[j]).abs();
        let scale_j = 1.0 + qx_j + prob.c[j].abs() + aty_j + bound_contrib[j].abs();
        let rel_j = r / scale_j;
        // 非有限成分は残差を評価できない = ゲートを通してはいけない。`rel > max_rel`
        // は NaN で偽になり 0 を返してしまうため、`INFINITY` を明示して落とす。
        if !rel_j.is_finite() {
            return f64::INFINITY;
        }
        if rel_j > max_rel {
            max_rel = rel_j;
        }
    }
    max_rel
}

/// 成分相対化 primal 残差。A·x は cancellation 対策で DD 積算。
pub fn primal_residual_rel(prob: &ProblemView, x: &[f64]) -> f64 {
    if prob.a.nrows() == 0 {
        return 0.0;
    }
    let ax_dd = dd_impl::ax(prob.a, x);
    let viols = dd_impl::constraint_violations(&ax_dd, prob.b, prob.constraint_types);
    let mut max_rel = 0.0_f64;
    for (i, &v) in viols.iter().enumerate() {
        let ax_i_abs = f64::from(ax_dd[i]).abs();
        let scale_i = 1.0 + ax_i_abs + prob.b[i].abs();
        let rel_i = v / scale_i;
        if !rel_i.is_finite() {
            return f64::INFINITY;
        }
        if rel_i > max_rel {
            max_rel = rel_i;
        }
    }
    max_rel
}

/// 実行不能質量計量 (infeasibility-mass witness) — 偽 Optimal の不可能性証拠。
///
/// `m̄ = Σ_ineq viol⁺ᵢ·|yᵢ| / Σ_ineq |yᵢ|` (Le/Ge 行のみ寄与、Eq 行は符号拘束が
/// ないため対象外で分子・分母とも除外)。
///
/// `dual_sign` 条件を通過した y (Le: y≥0 / Ge: y≤0) のもとでは、真に実行可能な
/// 任意の x が不等式行すべての viol⁺ = 0 を満たすので分子は厳密に 0 になる。
/// m̄ > 0 は返した点 (x, y) の対が不等式制約を実行不能方向に破っていることの
/// 直接証拠となる。実:m̄ の分子 = |−yᵀ(Ax − b)| の不等式質量部分であり、
/// 証明済み双対下界 d(ŷ) と主目的の乖離 (LISWET7 偽 mint の −2.13e-3) に
/// 直結する。max-scale primal_feasibility (per-row 相対化) は数万行の微小
/// 違反 (1e-7 準位 × 1 万行) を捉えないため、この加重平均が独立の検出器になる
/// (`prove_optimal` 内で絶対床 `certificate.rs::DUAL_MASS_VIOL_FLOOR` と比較)。
///
/// 正規化は双対質量 (Σ|yᵢ|) による加重平均で、符号拘束のない Eq 行の巨大 dual は
/// 証跡を作れないため分母にも入れない。収集項がすべて非負なので f64 Σ は
/// cancellation を起こさない (値がそのまま足し合わさる)。分母 0
/// (不等式行の dual が全 0) は証跡を主張できる情報がないため vacuous として 0 を返す。
///
/// 非有限入力は INFINITY (実行不能性を評価できない = ゲートを通してはならない)。
/// y.len() != a.nrows() も INFINITY。呼び出し側 (`prove_optimal`) は
/// dimension guard で b/ct 長 == n_rows を担保済み。
pub fn infeasible_dual_mass_rel(prob: &ProblemView, x: &[f64], y: &[f64]) -> f64 {
    let m = prob.a.nrows();
    if m == 0 {
        return 0.0;
    }
    if y.len() != m || any_nonfinite(x) {
        return f64::INFINITY;
    }
    let ax_dd = dd_impl::ax(prob.a, x);
    let viols = dd_impl::constraint_violations(&ax_dd, prob.b, prob.constraint_types);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    #[allow(unreachable_patterns)] // ConstraintType is #[non_exhaustive].
    for (i, &cti) in prob.constraint_types.iter().enumerate() {
        if i >= viols.len() {
            break;
        }
        match cti {
            ConstraintType::Le | ConstraintType::Ge => {}
            _ => continue,
        }
        let yi = y[i].abs();
        let viol = viols[i];
        if !yi.is_finite() || !viol.is_finite() {
            return f64::INFINITY;
        }
        // m̄ は非負質量の比率しか意味を持たない。NaN/±Inf は上の is_finite で
        // INFINITY に置き換え済みなので、ここに到達する値は必ず非負有限。
        num += viol * yi;
        den += yi;
    }
    if den == 0.0 {
        return 0.0;
    }
    num / den
}

/// 元空間 KKT complementarity 残差 (問題全体スケール正規化)。
///
/// stationarity / primal feas / bound feas に加えて KKT 4 条件目を gating する。
/// 欠落していると LISWET9 / YAO のように feasible だが optimal でない点を
/// Optimal と誤判定する (`y_i` が inactive 制約で大、`slack_i` が小、積が中程度)。
///
/// `complementarity = max(|y_i · slack_i|, |z_j · (x_j - bnd_j)|)`
///
/// 正規化分母は「IPM 双対対の自然スケール」: `|y^T b|`, `|y^T Ax|`, `|z^T x|`, `|c^T x|`,
/// `|0.5 x^T Q x|`, `1` の最大。これは双対関数 `b^T y - lb^T z_lb + ub^T z_ub` と
/// 主目的の典型的大きさで、IPM 反復で `y·s + z·(x-bnd)` がこの基準に対して 0 に
/// 収束する。bench `compute_pfeas_normalized` 流の `1 + ||·||∞` 正規化と同型で、
/// 巨大 dual と数値ゼロ slack の積 (= O(|y|² × machine_eps)) を問題全体スケールで
/// 押し下げ、真の complementarity 違反 (gap が obj スケールに匹敵) のみ捕捉する。
///
/// 等式制約は y·0=0 (primal feas で担保) のためスキップ。FX (lb≈ub) は postsolve で
/// z=0 埋めされ slack=0 にもなるため自動的に 0 寄与。
pub fn complementarity_residual_rel(prob: &ProblemView, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    let ax_dd = dd_impl::ax(prob.a, x);
    let z_full = if z.is_empty() {
        Vec::new()
    } else {
        let Some(z_full) = kkt_resid::bound_duals_full_layout(prob.bounds, z) else {
            return f64::INFINITY;
        };
        z_full
    };

    let yb: f64 = y.iter().zip(prob.b.iter()).map(|(&yi, &bi)| yi * bi).sum();
    let yax: f64 = y
        .iter()
        .zip(ax_dd.iter())
        .map(|(&yi, &ax_dd_i)| yi * f64::from(ax_dd_i))
        .sum();
    let zx: f64 = {
        let mut s = 0.0_f64;
        let mut idx = 0_usize;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && idx < z_full.len() {
                s += z_full[idx] * x[j];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && idx < z_full.len() {
                s += z_full[idx] * x[j];
                idx += 1;
            }
        }
        s
    };
    let cx: f64 = prob.c.iter().zip(x.iter()).map(|(&c, &xi)| c * xi).sum();
    let xqx: f64 = {
        let qx = prob.q.mat_vec_mul(x).expect(
            "q.ncols() == x.len() == num_vars: QpProblem::new() enforces \
             q.ncols() == num_vars (struct-literal construction bypasses this -- \
             all QpProblem fields are pub); callers guard x.len() == num_vars \
             before building ProblemView",
        );
        qx.iter().zip(x.iter()).map(|(&q, &xi)| q * xi).sum()
    };
    let scale = 1.0 + yb.abs() + yax.abs() + zx.abs() + cx.abs() + (0.5 * xqx).abs();

    let comp_i = dd_impl::comp_ineq_products(&ax_dd, prob.b, prob.constraint_types, y);
    let comp_b = kkt_resid::comp_bound_products(prob.bounds, x, z);
    let max_abs = comp_i
        .iter()
        .chain(comp_b.iter())
        .fold(0.0_f64, |a, &b| a.max(b));

    max_abs / scale
}

/// Component-wise relative complementarity residual.
///
/// This catches a single inactive row/bound carrying a nonzero dual even when the
/// problem-level complementarity scale is large enough to hide it.
pub fn complementarity_componentwise_rel(
    prob: &ProblemView,
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> f64 {
    let ax_dd = dd_impl::ax(prob.a, x);
    let comp_i = dd_impl::comp_ineq_products(&ax_dd, prob.b, prob.constraint_types, y);
    let mut worst = 0.0_f64;
    for (i, &prod) in comp_i.iter().enumerate() {
        if prod == 0.0 {
            continue;
        }
        let ax_i = f64::from(ax_dd[i]);
        let scale = 1.0 + y[i].abs() * (ax_i.abs() + prob.b[i].abs());
        worst = worst.max(prod / scale);
    }

    let comp_b = kkt_resid::comp_bound_products(prob.bounds, x, z);
    let z_full = if z.is_empty() {
        Vec::new()
    } else {
        let Some(z_full) = kkt_resid::bound_duals_full_layout(prob.bounds, z) else {
            return f64::INFINITY;
        };
        z_full
    };
    let mut idx = 0_usize;
    for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() && idx < z_full.len() {
            let scale = 1.0 + z_full[idx].abs() * (x[j].abs() + lb.abs());
            worst = worst.max(comp_b[idx] / scale);
            idx += 1;
        }
    }
    for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
        if ub.is_finite() && idx < z_full.len() {
            let scale = 1.0 + z_full[idx].abs() * (x[j].abs() + ub.abs());
            worst = worst.max(comp_b[idx] / scale);
            idx += 1;
        }
    }
    worst
}

/// 成分相対化 bounds 違反 max_j violation_j/(1+|x_j|+|bound_j|)。
///
/// Returns `f64::INFINITY` if any element of `x` is non-finite (NaN or ±Inf).
pub fn bound_violation(bounds: &[(f64, f64)], x: &[f64]) -> f64 {
    if any_nonfinite(x) {
        return f64::INFINITY;
    }
    let mut max_rel = 0.0_f64;
    for (&xi, &(lb, ub)) in x.iter().zip(bounds.iter()) {
        let lo = if lb.is_finite() {
            (lb - xi).max(0.0)
        } else {
            0.0
        };
        let hi = if ub.is_finite() {
            (xi - ub).max(0.0)
        } else {
            0.0
        };
        let v = lo.max(hi);
        let bnd = if lb.is_finite() && ub.is_finite() {
            lb.abs().max(ub.abs())
        } else if lb.is_finite() {
            lb.abs()
        } else if ub.is_finite() {
            ub.abs()
        } else {
            0.0
        };
        let scale_j = 1.0 + xi.abs() + bnd;
        let rel_j = v / scale_j;
        if !rel_j.is_finite() {
            return f64::INFINITY;
        }
        if rel_j > max_rel {
            max_rel = rel_j;
        }
    }
    max_rel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use otspot_num::sparse::CscMatrix;

    /// 相対残差の集約は、成分が非有限になったら「違反なし (0)」ではなく
    /// `INFINITY` を返さなければならない。
    ///
    /// `rel > max_rel` / `f64::max` は NaN 側で偽になるため、素朴に書くと
    /// オーバーフローした行 (`inf/inf = NaN`) が残差 0 として報告され、
    /// 上位の受理ゲートを素通りする。
    ///
    /// ## Sentinel (no-op-fail)
    /// 各集約ループの `if !rel.is_finite() { return f64::INFINITY }` を外すと、
    /// 下の assert が 0.0 を観測して FAIL する。
    #[test]
    fn residual_aggregation_reports_infinity_on_overflowing_activity() {
        // 1 行 1 列、係数 2.0。x = 1e308 で A·x はオーバーフローする。
        let a = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
        let q = CscMatrix::new(1, 1);
        let bounds = [(f64::NEG_INFINITY, f64::INFINITY)];
        let cts = [ConstraintType::Le];
        let b = [5.0_f64];
        let c = [0.0_f64];
        let view = ProblemView {
            q: &q,
            a: &a,
            c: &c,
            b: &b,
            bounds: &bounds,
            constraint_types: &cts,
            eliminated_cols: &[],
        };
        let x = [1e308_f64];
        let pres = primal_residual_rel(&view, &x);
        assert!(
            !pres.is_finite(),
            "オーバーフローした activity は非有限として報告すべき, got {pres}"
        );
    }

    fn build_view<'a>(
        q: &'a CscMatrix,
        a: &'a CscMatrix,
        c: &'a [f64],
        b: &'a [f64],
        bounds: &'a [(f64, f64)],
        cts: &'a [ConstraintType],
    ) -> ProblemView<'a> {
        ProblemView {
            q,
            a,
            c,
            b,
            bounds,
            constraint_types: cts,
            eliminated_cols: &[],
        }
    }

    fn build_view_with_mask<'a>(
        q: &'a CscMatrix,
        a: &'a CscMatrix,
        c: &'a [f64],
        b: &'a [f64],
        bounds: &'a [(f64, f64)],
        cts: &'a [ConstraintType],
        mask: &'a [bool],
    ) -> ProblemView<'a> {
        ProblemView {
            q,
            a,
            c,
            b,
            bounds,
            constraint_types: cts,
            eliminated_cols: mask,
        }
    }

    /// `q` is 3x3 but `x` has 2 entries. Must panic, not silently zero-fill.
    #[test]
    #[should_panic(expected = "q.ncols() == x.len() == num_vars")]
    fn complementarity_residual_rel_panics_on_dimension_mismatch() {
        let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[1.0, 1.0, 1.0], 3, 3).unwrap();
        let a = CscMatrix::new(0, 2);
        let c = vec![0.0, 0.0];
        let b: Vec<f64> = vec![];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let cts: Vec<ConstraintType> = vec![];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);
        let x = vec![0.5, 0.5];
        let y: Vec<f64> = vec![];
        let z: Vec<f64> = vec![];
        let _ = complementarity_residual_rel(&view, &x, &y, &z);
    }

    /// f64 で消える 1.0 residual を DD が拾うこと。
    #[test]
    fn kkt_residual_rel_uses_dd_to_avoid_f64_cancellation() {
        let a = CscMatrix::from_triplets(&[0, 1, 2], &[0, 0, 0], &[1.0_f64, 1.0e16, -1.0e16], 3, 1)
            .unwrap();
        let q = CscMatrix::new(1, 1);
        let c = vec![0.0_f64];
        let b = vec![0.0_f64; 3];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let cts = vec![ConstraintType::Eq; 3];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        let x = vec![0.0_f64];
        let y = vec![1.0_f64, 1.0, 1.0];
        let z: Vec<f64> = vec![];

        let aty_f64 = a.transpose().mat_vec_mul(&y).unwrap();
        assert_eq!(aty_f64[0], 0.0);

        let r = kkt_residual_rel(&view, &x, &y, &z);
        assert!(r > 0.4 && r < 0.6, "got r={:.3e}", r);
    }

    #[test]
    fn kkt_residual_rel_rejects_malformed_bound_duals_without_nan_masking() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::new(0, 1);
        let c = vec![0.0_f64];
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64)];
        let cts = vec![];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        let r = kkt_residual_rel(&view, &[0.5], &[], &[0.0]);
        assert!(
            r.is_infinite() && r > 0.0,
            "malformed bound_duals must remain a stationarity failure, got {r}"
        );
    }

    #[test]
    fn kkt_residual_rel_rejects_malformed_bound_duals_before_scaling() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        let x = vec![1.0e150_f64];
        let c = vec![-1.0e150_f64];
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0e200_f64)];
        let cts = vec![];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        let r = kkt_residual_rel(&view, &x, &[], &[0.0]);
        assert!(
            r.is_infinite() && r > 0.0,
            "invalid bound-dual layout must fail before component scaling, got {r}"
        );
    }

    #[test]
    fn complementarity_residual_rel_rejects_malformed_bound_duals_before_scaling() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        let x = vec![1.0e150_f64];
        let c = vec![-1.0e150_f64];
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0e200_f64)];
        let cts = vec![];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        let r = complementarity_residual_rel(&view, &x, &[], &[0.0]);
        assert!(
            r.is_infinite() && r > 0.0,
            "invalid bound-dual layout must fail before residual scaling, got {r}"
        );
    }

    #[test]
    fn complementarity_componentwise_rel_maps_fixed_omitted_bound_duals() {
        let q = CscMatrix::new(3, 3);
        let a = CscMatrix::new(0, 3);
        let c = vec![0.0_f64; 3];
        let b = vec![];
        let bounds = vec![
            (1.0_f64, 1.0_f64),
            (0.0, f64::INFINITY),
            (0.0, f64::INFINITY),
        ];
        let cts = vec![];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        let r = complementarity_componentwise_rel(&view, &[1.0, 0.0, 10.0], &[], &[0.0, 2.0]);
        assert!(
            (r - 20.0 / 21.0).abs() < 1e-12,
            "third-column product must use third-column scale, got {r}"
        );
    }

    #[test]
    fn primal_residual_rel_uses_dd_to_avoid_f64_cancellation() {
        let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0_f64, 1.0e16, -1.0e16], 1, 3)
            .unwrap();
        let q = CscMatrix::new(3, 3);
        let c = vec![0.0_f64; 3];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
        let cts = vec![ConstraintType::Eq];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        let x = vec![1.0_f64, 1.0, 1.0];

        let ax_f64 = a.mat_vec_mul(&x).unwrap();
        assert_eq!(ax_f64[0], 0.0);

        let r = primal_residual_rel(&view, &x);
        assert!(r > 0.4 && r < 0.6, "got r={:.3e}", r);
    }

    /// FX 列は KKT 評価から自動除外、LP-style 完全孤立 var (A 空 AND Q 空 AND mask=true)
    /// は明示 mask 経由で除外される。Path B narrow の skip 対象はこの一群のみ。
    #[test]
    fn kkt_residual_rel_excludes_fx_and_empty_col() {
        let q = CscMatrix::new(3, 3);
        // 列 1 を LP-style 完全孤立 (A 空 / Q 空) にするため A の非零は col 2 に置く。
        let c = vec![1e10_f64, 1e10, 0.0];
        let a = CscMatrix::from_triplets(&[0], &[2], &[1.0], 1, 3).unwrap();
        let b = vec![0.0];
        let bounds = vec![(1.0, 1.0), (0.0, f64::INFINITY), (0.0, f64::INFINITY)];
        let cts = vec![ConstraintType::Eq];
        // 列 0 = FX, 列 1 = LP-style 完全孤立 + mask=true, 列 2 = A 非空 (skip 対象外)。
        let mask = vec![false, true, false];
        let view = build_view_with_mask(&q, &a, &c, &b, &bounds, &cts, &mask);

        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0];
        let z = vec![0.0, 0.0];
        let r = kkt_residual_rel(&view, &x, &y, &z);
        assert!(r.abs() < 1e-15, "got r={:.3e}", r);
    }

    /// 真因 sentinel: QGFRDXPN-like 26 eliminated var (mask=true / A 非空 / Q 空 / c≠0)
    /// は Path B narrow で skip されず stationarity に出る。
    /// 旧 effc242 (Q 条件なし) では skip され r≈0 となり refine 進捗 oracle 退化した。
    #[test]
    fn kkt_residual_rel_qgfrdxpn_eliminated_var_not_skipped() {
        // n=1, A 列非空, Q 空, c=1, mask=true, y=0 → r_j = c = 1.0
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let cts = vec![ConstraintType::Eq];
        let mask = vec![true];
        let view = build_view_with_mask(&q, &a, &c, &b, &bounds, &cts, &mask);

        let x = vec![0.0_f64];
        let y = vec![0.0_f64];
        let z: Vec<f64> = vec![];
        let r = kkt_residual_rel(&view, &x, &y, &z);
        // raw r = 0 + 1 + 0 + 0 = 1, scale = 1+0+1+0+0 = 2 → rel = 0.5
        assert!(
            r > 0.4 && r < 0.6,
            "26 eliminated var (A 非空) は skip 対象外、r 露出必須 (got rel={:.3e})",
            r,
        );
    }

    /// control sentinel: linear-only var (A 空 / Q 非空 / mask=true) も Path B narrow で
    /// skip されず stationarity 露出を維持する。Q 空条件単独だと skip してしまうのを防ぐ。
    #[test]
    fn kkt_residual_rel_linear_only_var_not_skipped_with_mask() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[-2.0_f64], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        let c = vec![1.0_f64];
        let b: Vec<f64> = vec![];
        let bounds = vec![(-2.0_f64, 2.0_f64)];
        let cts: Vec<ConstraintType> = vec![];
        let mask = vec![true];
        let view = build_view_with_mask(&q, &a, &c, &b, &bounds, &cts, &mask);

        let x = vec![-2.0_f64];
        let y: Vec<f64> = vec![];
        let z = vec![0.0_f64, 0.0];
        let r = kkt_residual_rel(&view, &x, &y, &z);
        assert!(
            r > 0.5,
            "linear-only var (Q 非空) は mask=true でも skip されないこと (got rel={:.3e})",
            r,
        );
    }

    /// no-op proof: Path B narrow の Q 空条件を取り除いた挙動 (= effc242 の skip 規約) では
    /// QGFRDXPN-like fixture の stationarity が 0 と評価されてしまう。同一入力に手で
    /// "Q 空条件なし" の判定を当てて FAIL 相当の値を assert する。
    #[test]
    fn kkt_residual_rel_path_b_noop_proof() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 1, 1).unwrap();
        let c = vec![1.0_f64];
        let b = vec![0.0_f64];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let cts = vec![ConstraintType::Eq];

        // mask 空 → effc242 経路と同等に skip されないことを base line として確認。
        let view_no_mask = build_view(&q, &a, &c, &b, &bounds, &cts);
        let r_no_mask = kkt_residual_rel(&view_no_mask, &[0.0_f64], &[0.0_f64], &[]);
        assert!(
            r_no_mask > 0.4,
            "mask 空 base line で r が露出 (got {:.3e})",
            r_no_mask
        );

        // 同 fixture を mask=true で覆っても Path B narrow なら A 非空のため skip されず r 露出。
        let mask = vec![true];
        let view_masked = build_view_with_mask(&q, &a, &c, &b, &bounds, &cts, &mask);
        let r_masked = kkt_residual_rel(&view_masked, &[0.0_f64], &[0.0_f64], &[]);
        assert!(
            (r_masked - r_no_mask).abs() < 1e-15,
            "Path B: 26 eliminated 群は mask 有無で同値、effc242 (Q 条件なし) なら r_masked=0 で乖離 (no_mask={:.3e}, masked={:.3e})",
            r_no_mask, r_masked,
        );
    }

    /// A.2 sentinel: NaN in x must return INFINITY, not suppress violation as 0.0.
    /// Without the `x.iter().any(|v| !v.is_finite())` guard, `f64::NAN.max(0.0)==0.0`
    /// silently masks any bound violation → sentinel fails.
    #[test]
    fn bound_violation_nan_x_returns_infinity() {
        let bounds = vec![(0.0_f64, 1.0_f64)];
        let x = vec![f64::NAN];
        let v = bound_violation(&bounds, &x);
        assert!(
            v.is_infinite() && v > 0.0,
            "NaN x must give +INFINITY, got {v}"
        );
    }

    /// mask 未供給 (= IPM 経路) の場合: A 空 / Q 非空 の linear-only var は skip されず
    /// stationarity に出る。これが真因 (旧 A-only heuristic はこの r を隠していた)。
    #[test]
    fn kkt_residual_rel_no_mask_exposes_linear_only_var() {
        // n=1, A 空, Q diag=(-2), c=1, bounds=(-2, 2)
        let q = CscMatrix::from_triplets(&[0], &[0], &[-2.0_f64], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        let c = vec![1.0_f64];
        let b: Vec<f64> = vec![];
        let bounds = vec![(-2.0_f64, 2.0_f64)];
        let cts: Vec<ConstraintType> = vec![];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        // x=-2 (lb 当て), bd=[0,0] (誤って 0 埋め)
        // raw r = Qx+c+bc = -2*(-2)+1+0-0 = 5、scale = 1 + |Qx| + |c| + |aty| + |bc| = 6
        // rel = 5/6 ≈ 0.833。旧 buggy heuristic では skip され rel=0 と評価されていた。
        let x = vec![-2.0_f64];
        let y: Vec<f64> = vec![];
        let z = vec![0.0_f64, 0.0]; // [z_lb, z_ub] 両方 0
        let r = kkt_residual_rel(&view, &x, &y, &z);
        assert!(
            r > 0.5,
            "linear-only var の stationarity が露出するべき (got rel={:.3e})",
            r
        );
    }

    // ── infeasible_dual_mass_rel (LISWET7 偽 Optimal 証跡) ────────────────────
    //
    // m = 3 行 fixture: [Le: x ≤ 1], [Ge: x ≥ 1], [Eq: x == 1]。x=1 が唯一の
    // 実行可能点 (全行充足)。x < 1 は Ge を (1−x)+ で破る; Eq 違反も生じるが
    // Eq 行は証跡対象外 -> 巨大 Eq dual でも訳に立たない。
    type MassFixture = (
        CscMatrix,
        Vec<f64>,
        Vec<f64>,
        Vec<(f64, f64)>,
        Vec<ConstraintType>,
    );

    fn mass_fixture() -> MassFixture {
        let a = CscMatrix::from_triplets(&[0usize, 1, 2], &[0, 0, 0], &[1.0_f64, 1.0, 1.0], 3, 1)
            .unwrap();
        let c = vec![0.0_f64];
        let b = vec![1.0_f64, 1.0, 1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let ct = vec![ConstraintType::Le, ConstraintType::Ge, ConstraintType::Eq];
        (a, c, b, bounds, ct)
    }

    /// Le/Ge 行は |y| で重み付けされ、Eq 行 (巨大 dual を含む) は分子・分母とも除外
    /// される。手計算オラクル: x=0.5, y=[3,−5,100] → viol=[(1−0.5)_+=0 (Le 充足),
    /// (1−0.5)_+=0.5 (Ge 違反)] → m̄ = (0·3 + 0.5·5)/(3+5) = 5/16。
    #[test]
    fn infeasible_dual_mass_rel_weighted_by_dual_magnitude() {
        let (a, c, b, bounds, ct) = mass_fixture();
        let q = CscMatrix::new(1, 1);
        let view = build_view(&q, &a, &c, &b, &bounds, &ct);
        let x = vec![0.5_f64];
        let y = vec![3.0_f64, -5.0, 100.0];
        let m = infeasible_dual_mass_rel(&view, &x, &y);
        assert!(
            (m - 5.0 / 16.0).abs() < 1e-12,
            "x=0.5, y=[3,-5,100]: m̄ must be (0·3+0.5·5)/8 = 5/16, got {m}"
        );

        // x=0 (Ge 違反 viol=1, 充た Eq も証跡外) → m̄ = (0·3 + 1·5)/8 = 5/8
        let x = vec![0.0_f64];
        let m = infeasible_dual_mass_rel(&view, &x, &y);
        assert!(
            (m - 5.0 / 8.0).abs() < 1e-12,
            "x=0, y=[3,-5,100]: m̄ must be (0·3+1·5)/8 = 5/8, got {m}"
        );
    }

    /// Eq 行は証跡を作れない (符号拘束なし): 分子・分母とも Eq 行を除外するので、
    /// x が Eq を破っていてもその dual が巨大でも m̄ は 0。
    #[test]
    fn infeasible_dual_mass_rel_ignores_eq_duals() {
        let (a, c, b, bounds, ct) = mass_fixture();
        let q = CscMatrix::new(1, 1);
        let view = build_view(&q, &a, &c, &b, &bounds, &ct);
        let x = vec![0.0_f64];
        let y = vec![0.0_f64, 0.0, 1.0e6];
        let m = infeasible_dual_mass_rel(&view, &x, &y);
        assert_eq!(m, 0.0, "Eq-only dual mass must be vacuous, got {m}");
    }

    /// 不等式行の dual が全 0 (証跡を主張できる質量なし) は vacuous で 0。
    /// 実行可能な x (全不等式 viol⁺=0) は y ≠ 0 でも 0。
    #[test]
    fn infeasible_dual_mass_rel_vacuous_and_feasible_points() {
        let (a, c, b, bounds, ct) = mass_fixture();
        let q = CscMatrix::new(1, 1);
        let view = build_view(&q, &a, &c, &b, &bounds, &ct);
        let x = vec![0.0_f64];
        let m = infeasible_dual_mass_rel(&view, &x, &[0.0_f64, 0.0, 0.0]);
        assert_eq!(m, 0.0, "zero dual mass must be vacuous, got {m}");

        // 実行可能点 x=1: Le/Ge とも viol=0 → m̄=0 (Eq dual が巨大でも)
        let x = vec![1.0_f64];
        let y = vec![3.0_f64, -2.0, 100.0];
        let m = infeasible_dual_mass_rel(&view, &x, &y);
        assert_eq!(m, 0.0, "ineq-feasible x must give zero witness, got {m}");
    }

    /// y.len() != m → INFINITY (次元契約違反)。x に非有限成分があるときも
    /// INFINITY (viol 評価不能 = ゲート通過不可)。
    #[test]
    fn infeasible_dual_mass_rel_rejects_broken_inputs() {
        let (a, c, b, bounds, ct) = mass_fixture();
        let q = CscMatrix::new(1, 1);
        let view = build_view(&q, &a, &c, &b, &bounds, &ct);
        let x = vec![0.0_f64];
        for y in [vec![], vec![1.0_f64], vec![1.0_f64, -1.0, 2.0, 3.0]] {
            let m = infeasible_dual_mass_rel(&view, &x, &y);
            assert!(
                m.is_infinite() && m > 0.0,
                "y.len()={} must be INFINITY, got {m}",
                y.len()
            );
        }
        let y = vec![3.0_f64, -2.0, 100.0];
        for bad_x in [vec![f64::NAN], vec![f64::INFINITY]] {
            let m = infeasible_dual_mass_rel(&view, &bad_x, &y);
            assert!(
                m.is_infinite() && m > 0.0,
                "non-finite x must be INFINITY, got {m}"
            );
        }
    }
}
