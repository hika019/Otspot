//! 元空間 KKT 残差計算 (bench `compute_dfeas_orig` と同形)。
//!
//! 設計書「元問題基準で報告」原則に従い、scaled 空間ではなく必ず元 problem.q / a / c で計算する。
//! 成分相対化 (`r_j / (1 + |Qx_j| + |c_j| + |aty_j| + |bound_j|)`) で
//! ill-conditioned 問題でも妥当な精度判定が可能。

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
/// FX 変数は dual が postsolve で 0 埋めされる仕様のため評価から除外する。
pub fn kkt_residual_rel(prob: &ProblemView, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
    let n = prob.bounds.len();
    if x.len() != n {
        return f64::INFINITY;
    }
    let qx = match prob.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let aty = if prob.a.nrows > 0 && !y.is_empty() {
        match prob.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        }
    } else {
        vec![0.0; n]
    };
    let bound_contrib = compute_bound_contrib(prob.bounds, z, n);
    let mut max_rel = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
            continue;
        }
        let r = (qx[j] + prob.c[j] + aty[j] + bound_contrib[j]).abs();
        let scale = 1.0 + qx[j].abs() + prob.c[j].abs() + aty[j].abs() + bound_contrib[j].abs();
        max_rel = max_rel.max(r / scale);
    }
    max_rel
}

/// 元空間 primal 残差 (行ノルム正規化, 最大値)。
pub fn primal_residual_rel(prob: &ProblemView, x: &[f64]) -> f64 {
    if prob.a.nrows == 0 {
        return 0.0;
    }
    let ax = match prob.a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let row_norms = prob.a.row_infinity_norms();
    ax.iter()
        .zip(prob.b.iter())
        .zip(prob.constraint_types.iter())
        .zip(row_norms.iter())
        .map(|(((&ax_i, &b_i), ct), &rn)| {
            let v = match ct {
                ConstraintType::Eq => (ax_i - b_i).abs(),
                ConstraintType::Ge => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            };
            v / (1.0 + rn + b_i.abs())
        })
        .fold(0.0_f64, f64::max)
}

/// 元空間 bounds 違反 (絶対値, 最大値)。
pub fn bound_violation(bounds: &[(f64, f64)], x: &[f64]) -> f64 {
    x.iter()
        .zip(bounds.iter())
        .map(|(&xi, &(lb, ub))| {
            let lo = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
            let hi = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
            lo.max(hi)
        })
        .fold(0.0_f64, f64::max)
}
