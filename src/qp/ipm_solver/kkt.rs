//! 元空間 KKT 残差 (bench compute_dfeas_orig と同形・成分相対化)。

use crate::qp::kkt_resid::{self, dd_impl};
use crate::tolerances::FX_TOL;
use super::outcome::ProblemView;

/// 成分相対化 stationarity max_j |r_j|/(1+|Qx_j|+|c_j|+|Aᵀy_j|+|z_j|) を DD 精度で計算。
/// FX (lb≈ub) と `eliminated_cols[j]==true` の col は postsolve 慣例で除外。
/// 旧来の "A 列空のみ" heuristic は非凸 QP linear-only var (A 空 / Q 非空 / c≠0) を
/// 誤って skip する真因だったため廃止。EmptyCol 判定は presolve metadata 経由のみ。
pub fn kkt_residual_rel(prob: &ProblemView, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    let n = prob.bounds.len();
    if x.len() != n {
        return f64::INFINITY;
    }
    let qx_dd = dd_impl::qx(prob.q, x);
    let aty_dd = dd_impl::aty(prob.a, y, n);
    let bound_contrib = kkt_resid::bound_contrib(prob.bounds, z);
    let use_elim_mask = prob.eliminated_cols.len() == n;
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
            continue;
        }
        if use_elim_mask && prob.eliminated_cols[j] {
            continue;
        }
        let r_dd = qx_dd[j]
            + TwoFloat::from(prob.c[j])
            + aty_dd[j]
            + TwoFloat::from(bound_contrib[j]);
        let r = f64::from(r_dd).abs();
        let qx_j = f64::from(qx_dd[j]).abs();
        let aty_j = f64::from(aty_dd[j]).abs();
        let scale_j = 1.0 + qx_j + prob.c[j].abs() + aty_j + bound_contrib[j].abs();
        let rel_j = r / scale_j;
        if rel_j > max_rel {
            max_rel = rel_j;
        }
    }
    max_rel
}

/// 成分相対化 primal 残差。A·x は cancellation 対策で DD 積算。
pub fn primal_residual_rel(prob: &ProblemView, x: &[f64]) -> f64 {
    if prob.a.nrows == 0 {
        return 0.0;
    }
    let ax_dd = dd_impl::ax(prob.a, x);
    let viols = dd_impl::constraint_violations(&ax_dd, prob.b, prob.constraint_types);
    let mut max_rel = 0.0_f64;
    for (i, &v) in viols.iter().enumerate() {
        let ax_i_abs = f64::from(ax_dd[i]).abs();
        let scale_i = 1.0 + ax_i_abs + prob.b[i].abs();
        let rel_i = v / scale_i;
        if rel_i > max_rel {
            max_rel = rel_i;
        }
    }
    max_rel
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
pub fn complementarity_residual_rel(
    prob: &ProblemView,
    x: &[f64],
    y: &[f64],
    z: &[f64],
) -> f64 {
    let ax_dd = dd_impl::ax(prob.a, x);

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
            if lb.is_finite() && idx < z.len() {
                s += z[idx] * x[j];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && idx < z.len() {
                s += z[idx] * x[j];
                idx += 1;
            }
        }
        s
    };
    let cx: f64 = prob.c.iter().zip(x.iter()).map(|(&c, &xi)| c * xi).sum();
    let xqx: f64 = {
        let qx = prob.q.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; x.len()]);
        qx.iter().zip(x.iter()).map(|(&q, &xi)| q * xi).sum()
    };
    let scale = 1.0
        + yb.abs()
        + yax.abs()
        + zx.abs()
        + cx.abs()
        + (0.5 * xqx).abs();

    let comp_i = dd_impl::comp_ineq_products(&ax_dd, prob.b, prob.constraint_types, y);
    let comp_b = kkt_resid::comp_bound_products(prob.bounds, x, z);
    let max_abs = comp_i.iter().chain(comp_b.iter()).fold(0.0_f64, |a, &b| a.max(b));

    max_abs / scale
}

/// 成分相対化 bounds 違反 max_j violation_j/(1+|x_j|+|bound_j|)。
pub fn bound_violation(bounds: &[(f64, f64)], x: &[f64]) -> f64 {
    let mut max_rel = 0.0_f64;
    for (&xi, &(lb, ub)) in x.iter().zip(bounds.iter()) {
        let lo = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
        let hi = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
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
        if rel_j > max_rel {
            max_rel = rel_j;
        }
    }
    max_rel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use crate::problem::ConstraintType;

    fn build_view<'a>(
        q: &'a CscMatrix, a: &'a CscMatrix, c: &'a [f64], b: &'a [f64],
        bounds: &'a [(f64, f64)], cts: &'a [ConstraintType],
    ) -> ProblemView<'a> {
        ProblemView { q, a, c, b, bounds, constraint_types: cts, eliminated_cols: &[] }
    }

    fn build_view_with_mask<'a>(
        q: &'a CscMatrix, a: &'a CscMatrix, c: &'a [f64], b: &'a [f64],
        bounds: &'a [(f64, f64)], cts: &'a [ConstraintType], mask: &'a [bool],
    ) -> ProblemView<'a> {
        ProblemView { q, a, c, b, bounds, constraint_types: cts, eliminated_cols: mask }
    }

    /// f64 で消える 1.0 residual を DD が拾うこと。
    #[test]
    fn kkt_residual_rel_uses_dd_to_avoid_f64_cancellation() {
        let a = CscMatrix::from_triplets(
            &[0, 1, 2], &[0, 0, 0], &[1.0_f64, 1.0e16, -1.0e16], 3, 1,
        ).unwrap();
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
    fn primal_residual_rel_uses_dd_to_avoid_f64_cancellation() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 0], &[0, 1, 2], &[1.0_f64, 1.0e16, -1.0e16], 1, 3,
        ).unwrap();
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

    /// FX 列は KKT 評価から自動除外、EmptyCol は明示 mask 経由で除外される。
    #[test]
    fn kkt_residual_rel_excludes_fx_and_empty_col() {
        let q = CscMatrix::new(3, 3);
        let c = vec![1e10_f64, 1e10, 0.0];
        let a = CscMatrix::from_triplets(
            &[0], &[2], &[1.0], 1, 3,
        ).unwrap();
        let b = vec![0.0];
        let bounds = vec![(1.0, 1.0), (0.0, f64::INFINITY), (0.0, f64::INFINITY)];
        let cts = vec![ConstraintType::Eq];
        // 列 1 が presolve で削除された EmptyCol である情景を再現。
        let mask = vec![false, true, false];
        let view = build_view_with_mask(&q, &a, &c, &b, &bounds, &cts, &mask);

        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0];
        let z = vec![0.0, 0.0];
        let r = kkt_residual_rel(&view, &x, &y, &z);
        assert!(r.abs() < 1e-15, "got r={:.3e}", r);
    }

    /// mask 未供給 (= IPM 経路) の場合: A 空 / Q 非空 の linear-only var は skip されず
    /// stationarity に出る。これが #55 真因 (旧 A-only heuristic はこの r を隠していた)。
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
        assert!(r > 0.5, "linear-only var の stationarity が露出するべき (got rel={:.3e})", r);
    }
}
