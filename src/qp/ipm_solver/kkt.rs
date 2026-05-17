//! 元空間 KKT 残差計算 (bench `compute_dfeas_orig` と同形)。
//!
//! 設計書「元問題基準で報告」原則に従い、scaled 空間ではなく必ず元 problem.q / a / c で計算する。
//! **成分相対化** (max_j |r_j| / (1 + |Qx_j| + |c_j| + |A^T y_j| + |z_j|)) を採用。
//! 全体相対化 (OSQP 公式) は ill-scaled 問題で 1 成分のみ大きく外れた残差を見逃すため、
//! ユーザー指定 eps の保証として不十分。+1 オフセットにより微小成分での過剰判定も抑制する。

use crate::problem::ConstraintType;
use super::outcome::ProblemView;

/// FX (固定) 変数判定の許容差。lb と ub の差がこれ未満なら固定変数とみなす。
const FX_TOL: f64 = 1e-12;

/// 境界 dual から KKT stationarity の bound 寄与 (-y_lb + y_ub) を成分ごと計算する。
fn compute_bound_contrib(
    bounds: &[(f64, f64)],
    bound_duals: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut contrib = vec![0.0_f64; n];
    if bound_duals.is_empty() {
        return contrib;
    }
    let mut idx = 0usize;
    for (j, &(lb, _)) in bounds.iter().enumerate() {
        if lb.is_finite() && idx < bound_duals.len() {
            contrib[j] -= bound_duals[idx];
            idx += 1;
        }
    }
    for (j, &(_, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() && idx < bound_duals.len() {
            contrib[j] += bound_duals[idx];
            idx += 1;
        }
    }
    contrib
}

/// 元空間 KKT stationarity 残差 (成分相対化, 最大値)。
///
/// `max_j |r_j| / (1 + |Qx_j| + |c_j| + |A^T y_j| + |z_j|)`。
/// 全体相対化 (OSQP 公式) は ill-scaled 問題で 1 成分のみ大きく外れた残差を
/// 巨大スケールで割って eps を満たすように見せてしまう欠陥があり、ユーザー指定
/// 精度の保証として不十分なため成分相対化を採用する。+1 オフセットで微小成分の
/// 過剰判定を抑制。
///
/// **DD (TwoFloat) 精度** で計算する: ill-conditioned 問題 (QPILOTNO: cond≈3e12) で
/// f64 mat_vec のキャンセル誤差が真の残差を埋もれさせ、bench `compute_dfeas_orig` (DD)
/// と乖離する。同じ DD 演算で揃えないと採否判定 / quality_score が noise を相手にして
/// 誤った収束判定をする。
///
/// FX 変数 (lb≈ub) と EmptyCol 変数は postsolve 慣例で bound_dual=0 埋めされるため
/// KKT 評価から除外する (`compute_dfeas_orig` の除外条件と一致)。
pub fn kkt_residual_rel(prob: &ProblemView, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    let n = prob.bounds.len();
    if x.len() != n {
        return f64::INFINITY;
    }
    let zero_dd = TwoFloat::from(0.0);
    // qx[i] = sum_k Q[i, k] * x[k]
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = x[col];
        let cs = prob.q.col_ptr[col];
        let ce = prob.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = prob.q.row_ind[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(prob.q.values[k], xv);
        }
    }
    // aty[col] = sum_row A[row, col] * y[row]
    let aty_dd: Vec<TwoFloat> = if prob.a.nrows > 0 && !y.is_empty() {
        let mut acc: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let cs = prob.a.col_ptr[col];
            let ce = prob.a.col_ptr[col + 1];
            for k in cs..ce {
                let row = prob.a.row_ind[k];
                acc[col] = acc[col] + TwoFloat::new_mul(prob.a.values[k], y[row]);
            }
        }
        acc
    } else {
        vec![zero_dd; n]
    };
    let bound_contrib = compute_bound_contrib(prob.bounds, z, n);
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
            continue;
        }
        if prob.a.col_ptr.len() > j + 1
            && prob.a.col_ptr[j + 1] - prob.a.col_ptr[j] == 0
        {
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

/// 元空間 primal 残差 (成分相対化, 最大値)。
///
/// `max_i violation_i / (1 + |Ax_i| + |b_i|)` (制約型ごと violation を取る)。
/// A·x は DD で積算: f64 sum のキャンセル誤差で実 violation が見えなくなるのを防ぐ。
/// 成分相対化により ill-scaled 行列で 1 行のみ違反が大きい場合も検出する。
pub fn primal_residual_rel(prob: &ProblemView, x: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    if prob.a.nrows == 0 {
        return 0.0;
    }
    let m = prob.a.nrows;
    let zero_dd = TwoFloat::from(0.0);
    let mut ax_dd: Vec<TwoFloat> = vec![zero_dd; m];
    for col in 0..prob.a.ncols {
        let xv = x[col];
        for k in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
            ax_dd[prob.a.row_ind[k]] = ax_dd[prob.a.row_ind[k]] + TwoFloat::new_mul(prob.a.values[k], xv);
        }
    }
    let mut max_rel = 0.0_f64;
    for ((ax_i_dd, &b_i), ct) in ax_dd.iter()
        .zip(prob.b.iter())
        .zip(prob.constraint_types.iter())
    {
        let raw_dd = *ax_i_dd - TwoFloat::from(b_i);
        let raw = f64::from(raw_dd);
        let v = match ct {
            ConstraintType::Eq => raw.abs(),
            ConstraintType::Ge => (-raw).max(0.0),
            _ => raw.max(0.0),
        };
        let ax_i_abs = f64::from(*ax_i_dd).abs();
        let scale_i = 1.0 + ax_i_abs + b_i.abs();
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
    use twofloat::TwoFloat;
    let zero_dd = TwoFloat::from(0.0);

    // Ax DD (primal_residual_rel と同形)。
    let m = prob.a.nrows;
    let ax_dd: Vec<TwoFloat> = if m > 0 {
        let mut ax = vec![zero_dd; m];
        for col in 0..prob.a.ncols {
            let xv = x[col];
            for k in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
                ax[prob.a.row_ind[k]] = ax[prob.a.row_ind[k]] + TwoFloat::new_mul(prob.a.values[k], xv);
            }
        }
        ax
    } else {
        Vec::new()
    };

    // 双対対スケール: max(|y·b|, |y·Ax|, |z·x|, |c·x|, |0.5 x·Qx|, 1)
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

    let mut max_abs = 0.0_f64;

    // inequality complementarity
    if m > 0 && !y.is_empty() {
        for (i, ct) in prob.constraint_types.iter().enumerate() {
            let slack_dd = match ct {
                ConstraintType::Le => TwoFloat::from(prob.b[i]) - ax_dd[i],
                ConstraintType::Ge => ax_dd[i] - TwoFloat::from(prob.b[i]),
                ConstraintType::Eq => continue,
            };
            let prod = (f64::from(slack_dd) * y[i]).abs();
            if prod > max_abs {
                max_abs = prod;
            }
        }
    }

    // bound complementarity (postsolve の z=0/slack=0 は自動的に 0 寄与)
    if !z.is_empty() {
        let mut idx = 0_usize;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && idx < z.len() {
                let prod = (z[idx] * (x[j] - lb)).abs();
                if prod > max_abs {
                    max_abs = prod;
                }
                idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && idx < z.len() {
                let prod = (z[idx] * (ub - x[j])).abs();
                if prod > max_abs {
                    max_abs = prod;
                }
                idx += 1;
            }
        }
    }

    max_abs / scale
}

/// 元空間 bounds 違反 (成分相対化, 最大値)。
///
/// `max_j violation_j / (1 + |x_j| + |bound_j|)`。成分相対化により単一変数が
/// 大きく境界を超えても見逃さない。
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
        ProblemView { q, a, c, b, bounds, constraint_types: cts }
    }

    /// f64 sum のキャンセルで residual が見えなくなる入力で、kkt_residual_rel が
    /// DD 計算で真の値を返すことを確認する。
    ///
    /// 設計: A の col 0 に [1.0, 1e16, -1e16] を CSC 順で入れ、y=[1,1,1] で aty[0] を取る。
    /// f64 left-to-right sum (CSC walk): 0 + 1.0 = 1.0 → 1.0 + 1e16 = 1e16 (1.0 が ULP に
    /// 吸収) → 1e16 + (-1e16) = 0 (真値 1.0 が消える)。DD sum なら 1.0 が保たれる。
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

        // f64 mat_vec_mul は cancellation で 0 を返す (真値 1.0 を見失う)。
        let aty_f64 = a.transpose().mat_vec_mul(&y).unwrap();
        assert_eq!(aty_f64[0], 0.0, "f64 path loses the 1.0 residual via cancellation");

        // DD 経路の kkt_residual_rel は 真値 1.0 / scale を返す。scale = 1 + |aty|_dd ≈ 2。
        let r = kkt_residual_rel(&view, &x, &y, &z);
        assert!(r > 0.4 && r < 0.6, "DD reveals 1.0 residual / scale ≈ 0.5; got r={:.3e}", r);
    }

    /// primal_residual_rel も DD 計算であることを同形のキャンセル入力で確認する。
    #[test]
    fn primal_residual_rel_uses_dd_to_avoid_f64_cancellation() {
        // m=1, n=3。A = [[1.0, 1e16, -1e16]] (1 行)、x=[1,1,1]、b=[0]、Eq。
        // CSC col 走査順 (col 0 → col 1 → col 2): 0 + 1.0 → 1.0 + 1e16 → 1e16 + (-1e16) = 0。
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

        // f64 mat_vec_mul は 0 を返す (真の violation 1.0 が見えない)。
        let ax_f64 = a.mat_vec_mul(&x).unwrap();
        assert_eq!(ax_f64[0], 0.0, "f64 path loses the 1.0 violation via cancellation");

        // DD 経路の primal_residual_rel は 1.0 / scale を返す。scale = 1 + |Ax|_dd ≈ 2。
        let r = primal_residual_rel(&view, &x);
        assert!(r > 0.4 && r < 0.6, "DD reveals 1.0 violation / scale ≈ 0.5; got r={:.3e}", r);
    }

    /// FX (lb≈ub) と EmptyCol は KKT 評価から除外される慣例を確認。
    #[test]
    fn kkt_residual_rel_excludes_fx_and_empty_col() {
        // 3 列: col 0 = FX (lb=ub=1.0)、col 1 = empty (A 列に登場しない)、col 2 = 普通。
        let q = CscMatrix::new(3, 3);
        let c = vec![1e10_f64, 1e10, 0.0]; // FX/empty 列の c は意図的に大きく
        let a = CscMatrix::from_triplets(
            &[0], &[2], &[1.0], 1, 3,
        ).unwrap();
        let b = vec![0.0];
        let bounds = vec![(1.0, 1.0), (0.0, f64::INFINITY), (0.0, f64::INFINITY)];
        let cts = vec![ConstraintType::Eq];
        let view = build_view(&q, &a, &c, &b, &bounds, &cts);

        // x[0]=1 (固定)、x[1]=0、x[2]=0、y=[0]、z=[lb_for_col1, lb_for_col2, ub のうち有限分なし] → 0 埋め
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0];
        let z = vec![0.0, 0.0]; // 両 lb 有限 var (col 1, 2) の lb dual
        let r = kkt_residual_rel(&view, &x, &y, &z);
        // FX (col 0) は除外、empty col (col 1) も除外、col 2 のみ評価。c[2]=0、qx=0、aty=0、bnd=0 → r=0。
        assert!(r.abs() < 1e-15, "FX/empty col 除外で残差 0、got r={:.3e}", r);
    }
}
