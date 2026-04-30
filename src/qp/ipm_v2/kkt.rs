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
/// 旧実装は成分ごと正規化してから max を取ったが、これは「他項全部 0 に近い 1 変数」
/// で簡単に膨らむ過剰判定だった (Marginal 5件で dfr=1.5e-6 越えの主因)。
/// OSQP/Gurobi 等は `||r||_∞ / (1 + max(||Qx||_∞, ||c||_∞, ||A^T y||_∞, ||z||_∞))` と
/// 全体最大で正規化する標準形式を採用しており、本実装もそれに合わせる。
///
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
    let mut max_r = 0.0_f64;
    let mut max_qx = 0.0_f64;
    let mut max_c = 0.0_f64;
    let mut max_aty = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    for j in 0..n {
        let (lb, ub) = prob.bounds[j];
        // FX 変数 (lb≈ub): presolve 慣例で除去 → bound_dual=0、KKT 評価から除外。
        if lb.is_finite() && ub.is_finite() && (lb - ub).abs() < FX_TOL {
            continue;
        }
        // EmptyCol 変数 (制約 A に登場しない): presolve で除去、bound_dual=0 慣例。
        // この変数の stationarity = c[j] が 0 にならないため KKT 評価から除外する
        // (refit_z_active_set / dual_solve_kkt_lsq の skip と整合、BD-T4 等で必要)。
        if prob.a.col_ptr.len() > j + 1
            && prob.a.col_ptr[j + 1] - prob.a.col_ptr[j] == 0
        {
            continue;
        }
        let r = (qx[j] + prob.c[j] + aty[j] + bound_contrib[j]).abs();
        max_r = max_r.max(r);
        max_qx = max_qx.max(qx[j].abs());
        max_c = max_c.max(prob.c[j].abs());
        max_aty = max_aty.max(aty[j].abs());
        max_bnd = max_bnd.max(bound_contrib[j].abs());
    }
    let scale = 1.0 + max_qx.max(max_c).max(max_aty).max(max_bnd);
    max_r / scale
}

/// 元空間 primal 残差 (OSQP 式: 全体相対化, 最大値)。
///
/// 旧実装は行ごと `(1 + rn + |b_i|)` 正規化 → max で、行ノルム小の制約で過剰に厳しい。
/// OSQP 標準: `||Ax - b||_∞ / (1 + max(||Ax||_∞, ||b||_∞))` (制約型ごと violation を取る)。
pub fn primal_residual_rel(prob: &ProblemView, x: &[f64]) -> f64 {
    if prob.a.nrows == 0 {
        return 0.0;
    }
    let ax = match prob.a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let mut max_v = 0.0_f64;
    let mut max_ax = 0.0_f64;
    let mut max_b = 0.0_f64;
    for ((&ax_i, &b_i), ct) in ax.iter()
        .zip(prob.b.iter())
        .zip(prob.constraint_types.iter())
    {
        let v = match ct {
            ConstraintType::Eq => (ax_i - b_i).abs(),
            ConstraintType::Ge => (b_i - ax_i).max(0.0),
            _ => (ax_i - b_i).max(0.0),
        };
        max_v = max_v.max(v);
        max_ax = max_ax.max(ax_i.abs());
        max_b = max_b.max(b_i.abs());
    }
    let scale = 1.0 + max_ax.max(max_b);
    max_v / scale
}

/// 元空間 bounds 違反 (OSQP 式: 全体相対化, 最大値)。
///
/// 旧実装は絶対値 max でスケール無視 → bound が 1e10 の問題で 1e-6 違反が PASS にならない。
/// OSQP 標準: `||violation||_∞ / (1 + max(||x||_∞, ||lb||_∞, ||ub||_∞))`。
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
