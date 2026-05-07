//! 元空間 KKT 残差計算 (bench `compute_dfeas_orig` と同形)。
//!
//! 設計書「元問題基準で報告」原則に従い、scaled 空間ではなく必ず元 problem.q / a / c で計算する。
//! OSQP 式の全体相対化 (`||r||_∞ / (1 + max(||Qx||_∞, ||c||_∞, ||A^T y||_∞, ||z||_∞))`)
//! を採用し、ill-conditioned 問題でも妥当な精度判定が可能。

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

/// 元空間 KKT stationarity 残差 (OSQP 式: 全体相対化, 最大値)。
///
/// `||r||_∞ / (1 + max(||Qx||_∞, ||c||_∞, ||A^T y||_∞, ||z||_∞))` (OSQP 標準)。
/// 成分ごと正規化版は「他項が 0 に近い 1 変数」で過剰判定する欠陥があった。
///
/// **DD (TwoFloat) 精度** で計算する: ill-conditioned 問題 (QPILOTNO: cond≈3e12) で
/// f64 mat_vec のキャンセル誤差が真の残差を埋もれさせ、bench `compute_dfeas_orig` (DD)
/// と乖離する。同じ DD 演算で揃えないと Stage A/B/C/D の guard / 採否判定 / quality_score が
/// noise を相手にして誤った収束判定をする。
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
    let mut max_r = 0.0_f64;
    let mut max_qx = 0.0_f64;
    let mut max_c = 0.0_f64;
    let mut max_aty = 0.0_f64;
    let mut max_bnd = 0.0_f64;
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
        max_r = max_r.max(f64::from(r_dd).abs());
        max_qx = max_qx.max(f64::from(qx_dd[j]).abs());
        max_c = max_c.max(prob.c[j].abs());
        max_aty = max_aty.max(f64::from(aty_dd[j]).abs());
        max_bnd = max_bnd.max(bound_contrib[j].abs());
    }
    let scale = 1.0 + max_qx.max(max_c).max(max_aty).max(max_bnd);
    max_r / scale
}

/// 元空間 primal 残差 (OSQP 式: 全体相対化, 最大値)。
///
/// `||Ax - b||_∞ / (1 + max(||Ax||_∞, ||b||_∞))` (制約型ごと violation を取る)。
/// A·x は DD で積算: f64 sum のキャンセル誤差で実 violation が見えなくなるのを防ぐ。
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
    let mut max_v = 0.0_f64;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
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
        max_v = max_v.max(v);
        max_ax = max_ax.max(f64::from(*ax_i_dd).abs());
        max_b = max_b.max(b_i.abs());
    }
    let scale = 1.0 + max_ax.max(max_b);
    max_v / scale
}

/// 元空間 bounds 違反 (OSQP 式: 全体相対化, 最大値)。
///
/// `||violation||_∞ / (1 + max(||x||_∞, ||lb||_∞, ||ub||_∞))`。
pub fn bound_violation(bounds: &[(f64, f64)], x: &[f64]) -> f64 {
    let mut max_v = 0.0_f64;
    let mut max_x = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    for (&xi, &(lb, ub)) in x.iter().zip(bounds.iter()) {
        let lo = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
        let hi = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
        max_v = max_v.max(lo.max(hi));
        max_x = max_x.max(xi.abs());
        if lb.is_finite() {
            max_bnd = max_bnd.max(lb.abs());
        }
        if ub.is_finite() {
            max_bnd = max_bnd.max(ub.abs());
        }
    }
    let scale = 1.0 + max_x.max(max_bnd);
    max_v / scale
}
