//! QP KKT 残差の per-component primitives.
//!
//! 7 callers (bench_utils / ipm_solver::kkt 3 / qps_benchmark / diag_nonconvex_kkt /
//! verify_solutions) が散在的に持っていた Q·x / A^T·y / A·x / bound_contrib /
//! complementarity slack の重実装をここに集約する。caller 側は
//!   - 集約方法 (max abs / max componentwise rel / global rel / 構造体格納)
//!   - 経路 fork (LP rc 経路 / QP bound_dual 経路 / FX skip / EmptyCol skip)
//! を選ぶ責務のみ保持し、heavy mat-vec と bound iteration ロジックは
//! 重複しない。
//!
//! ## モジュール構成
//!
//! - `f64_impl`: 倍精度経路 (verify_solutions / diag_nonconvex_kkt / bench_utils 等)
//! - `dd_impl`:  TwoFloat 経路 (kkt_residual_rel / compute_dfeas_orig 等)
//!
//! generic trait 化はしない (band-aid 回避; f64 と DD は数値経路として明確に
//! 分離するほうが drift catch 性が高い)。

use crate::problem::ConstraintType;
use crate::sparse::CscMatrix;

/// Bound dual stationarity contribution per column: `−bd_lb + bd_ub`.
///
/// `bd` layout: `[lb-duals for lb-finite columns in column order, then ub-duals
/// for ub-finite columns]`. Returns zero vector when `bd.is_empty()`.
pub fn bound_contrib(bounds: &[(f64, f64)], bd: &[f64]) -> Vec<f64> {
    let n = bounds.len();
    let mut contrib = vec![0.0_f64; n];
    if bd.is_empty() {
        return contrib;
    }
    let mut idx = 0_usize;
    for (j, &(lb, _)) in bounds.iter().enumerate() {
        if lb.is_finite() && idx < bd.len() {
            contrib[j] -= bd[idx];
            idx += 1;
        }
    }
    for (j, &(_, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() && idx < bd.len() {
            contrib[j] += bd[idx];
            idx += 1;
        }
    }
    contrib
}

/// Raw bound complementarity products `|bd_j · (x_j − bnd_j)|`.
///
/// Output length = `bd.len()`, ordering matches [`bound_contrib`]: lb-half
/// followed by ub-half. caller が componentwise scale で割るか global scale で
/// 割るかは別。
pub fn comp_bound_products(bounds: &[(f64, f64)], x: &[f64], bd: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(bd.len());
    if bd.is_empty() {
        return out;
    }
    let mut idx = 0_usize;
    for (j, &(lb, _)) in bounds.iter().enumerate() {
        if lb.is_finite() && idx < bd.len() {
            out.push((bd[idx] * (x[j] - lb)).abs());
            idx += 1;
        }
    }
    for (j, &(_, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() && idx < bd.len() {
            out.push((bd[idx] * (ub - x[j])).abs());
            idx += 1;
        }
    }
    out
}

/// f64 経路の mat-vec / per-row 残差プリミティブ。
pub mod f64_impl {
    use super::*;

    /// Q·x (per-row sum). Q は IPM 規約に従い完全対称 CSC (上下三角両方格納) 前提。
    pub fn qx(q: &CscMatrix, x: &[f64]) -> Vec<f64> {
        q.mat_vec_mul(x).unwrap_or_else(|_| vec![0.0; q.nrows])
    }

    /// A^T·y (per-column sum), output 長 `n = a.ncols`。`y` が空 / `A` が 0 行なら zero。
    pub fn aty(a: &CscMatrix, y: &[f64], n: usize) -> Vec<f64> {
        if a.nrows == 0 || y.is_empty() {
            return vec![0.0; n];
        }
        a.transpose().mat_vec_mul(y).unwrap_or_else(|_| vec![0.0; n])
    }

    /// A·x (per-row sum). `A` が 0 行なら空 Vec。
    pub fn ax(a: &CscMatrix, x: &[f64]) -> Vec<f64> {
        if a.nrows == 0 {
            return Vec::new();
        }
        a.mat_vec_mul(x).unwrap_or_else(|_| Vec::new())
    }

    /// 制約タイプ別 per-row primal 違反 (`max(0, Ax−b)` Le / `max(0, b−Ax)` Ge / `|Ax−b|` Eq)。
    pub fn constraint_violations(ax: &[f64], b: &[f64], ct: &[ConstraintType]) -> Vec<f64> {
        let m = ct.len();
        let mut out = vec![0.0_f64; m];
        #[allow(unreachable_patterns)] // ConstraintType is #[non_exhaustive].
        for (i, cti) in ct.iter().enumerate() {
            if i >= ax.len() || i >= b.len() {
                continue;
            }
            out[i] = match cti {
                ConstraintType::Le => (ax[i] - b[i]).max(0.0),
                ConstraintType::Ge => (b[i] - ax[i]).max(0.0),
                ConstraintType::Eq => (ax[i] - b[i]).abs(),
                _ => 0.0,
            };
        }
        out
    }

    /// 不等式 complementarity 生積 `|y_i · slack_i|`. Eq 行は 0.
    pub fn comp_ineq_products(
        ax: &[f64], b: &[f64], ct: &[ConstraintType], y: &[f64],
    ) -> Vec<f64> {
        let m = ct.len();
        let mut out = vec![0.0_f64; m];
        if ax.is_empty() || y.is_empty() {
            return out;
        }
        #[allow(unreachable_patterns)] // ConstraintType is #[non_exhaustive].
        for (i, cti) in ct.iter().enumerate() {
            let slack = match cti {
                ConstraintType::Le => b[i] - ax[i],
                ConstraintType::Ge => ax[i] - b[i],
                ConstraintType::Eq => continue,
                _ => continue,
            };
            out[i] = (y[i] * slack).abs();
        }
        out
    }
}

/// double-double (TwoFloat) 経路。ill-scaled 行列 (Maros QPILOTNO 系) で
/// f64 cancellation noise を回避する。
pub mod dd_impl {
    use super::*;
    use twofloat::TwoFloat;

    /// Q·x DD per-row sum.
    pub fn qx(q: &CscMatrix, x: &[f64]) -> Vec<TwoFloat> {
        let n = q.nrows;
        let zero = TwoFloat::from(0.0);
        let mut out: Vec<TwoFloat> = vec![zero; n];
        for col in 0..q.ncols {
            let xv = x[col];
            for k in q.col_ptr[col]..q.col_ptr[col + 1] {
                let row = q.row_ind[k];
                out[row] = out[row] + TwoFloat::new_mul(q.values[k], xv);
            }
        }
        out
    }

    /// A^T·y DD per-column sum.
    pub fn aty(a: &CscMatrix, y: &[f64], n: usize) -> Vec<TwoFloat> {
        let zero = TwoFloat::from(0.0);
        if a.nrows == 0 || y.is_empty() {
            return vec![zero; n];
        }
        let mut out: Vec<TwoFloat> = vec![zero; n];
        for col in 0..a.ncols {
            for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                let row = a.row_ind[k];
                out[col] = out[col] + TwoFloat::new_mul(a.values[k], y[row]);
            }
        }
        out
    }

    /// A·x DD per-row sum.
    pub fn ax(a: &CscMatrix, x: &[f64]) -> Vec<TwoFloat> {
        if a.nrows == 0 {
            return Vec::new();
        }
        let zero = TwoFloat::from(0.0);
        let mut out: Vec<TwoFloat> = vec![zero; a.nrows];
        for col in 0..a.ncols {
            let xv = x[col];
            for k in a.col_ptr[col]..a.col_ptr[col + 1] {
                out[a.row_ind[k]] = out[a.row_ind[k]] + TwoFloat::new_mul(a.values[k], xv);
            }
        }
        out
    }

    /// per-row primal 違反, DD `Ax − b` を取って f64 に truncate. Le/Ge/Eq 別.
    pub fn constraint_violations(
        ax_dd: &[TwoFloat], b: &[f64], ct: &[ConstraintType],
    ) -> Vec<f64> {
        let m = ct.len();
        let mut out = vec![0.0_f64; m];
        #[allow(unreachable_patterns)] // ConstraintType is #[non_exhaustive].
        for (i, cti) in ct.iter().enumerate() {
            if i >= ax_dd.len() || i >= b.len() {
                continue;
            }
            let raw = f64::from(ax_dd[i] - TwoFloat::from(b[i]));
            out[i] = match cti {
                ConstraintType::Le => raw.max(0.0),
                ConstraintType::Ge => (-raw).max(0.0),
                ConstraintType::Eq => raw.abs(),
                _ => 0.0,
            };
        }
        out
    }

    /// 不等式 complementarity 生積 `|y_i · slack_i|`, slack を DD で計算.
    pub fn comp_ineq_products(
        ax_dd: &[TwoFloat], b: &[f64], ct: &[ConstraintType], y: &[f64],
    ) -> Vec<f64> {
        let m = ct.len();
        let mut out = vec![0.0_f64; m];
        if ax_dd.is_empty() || y.is_empty() {
            return out;
        }
        #[allow(unreachable_patterns)] // ConstraintType is #[non_exhaustive].
        for (i, cti) in ct.iter().enumerate() {
            let slack_dd = match cti {
                ConstraintType::Le => TwoFloat::from(b[i]) - ax_dd[i],
                ConstraintType::Ge => ax_dd[i] - TwoFloat::from(b[i]),
                ConstraintType::Eq => continue,
                _ => continue,
            };
            out[i] = (f64::from(slack_dd) * y[i]).abs();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_contrib_empty_bd_yields_zero() {
        let bounds = vec![(0.0, 1.0), (f64::NEG_INFINITY, f64::INFINITY)];
        let c = bound_contrib(&bounds, &[]);
        assert_eq!(c, vec![0.0, 0.0]);
    }

    #[test]
    fn bound_contrib_lb_then_ub_layout() {
        // bounds: col0 lb=0/ub=10 (両方 finite), col1 free, col2 lb=5 のみ
        // bd layout: [lb half: col0, col2], [ub half: col0]
        let bounds = vec![(0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY), (5.0, f64::INFINITY)];
        let bd = vec![1.0, 2.0, 3.0];
        let c = bound_contrib(&bounds, &bd);
        assert_eq!(c, vec![-1.0 + 3.0, 0.0, -2.0]);
    }

    #[test]
    fn comp_bound_products_lb_then_ub_layout() {
        let bounds = vec![(0.0, 10.0), (5.0, f64::INFINITY)];
        let x = vec![2.0, 7.0];
        let bd = vec![1.5, 0.5, 4.0];
        let p = comp_bound_products(&bounds, &x, &bd);
        assert_eq!(
            p,
            vec![
                (1.5_f64 * (2.0 - 0.0)).abs(),
                (0.5_f64 * (7.0 - 5.0)).abs(),
                (4.0_f64 * (10.0 - 2.0)).abs(),
            ]
        );
    }

    #[test]
    fn comp_bound_products_empty_bd() {
        let bounds = vec![(0.0, 1.0)];
        let x = vec![0.5];
        let p = comp_bound_products(&bounds, &x, &[]);
        assert!(p.is_empty());
    }

    #[test]
    fn f64_constraint_violations_le_ge_eq() {
        use ConstraintType::*;
        let ax = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 3.0, 3.0];
        let ct = vec![Le, Ge, Eq];
        let v = f64_impl::constraint_violations(&ax, &b, &ct);
        assert_eq!(v, vec![0.5, 1.0, 0.0]);
    }

    #[test]
    fn f64_constraint_violations_no_negative() {
        use ConstraintType::*;
        // Le 満足 (ax<b), Ge 満足 (ax>b) → 0
        let ax = vec![1.0, 5.0];
        let b = vec![2.0, 3.0];
        let ct = vec![Le, Ge];
        let v = f64_impl::constraint_violations(&ax, &b, &ct);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn dd_constraint_violations_recovers_from_f64_cancellation() {
        use twofloat::TwoFloat;
        use ConstraintType::*;
        // f64 で ax = 1.0 + 1e16 - 1e16 = 0 だが、DD なら 1.0 を保つ。b=0, Eq → 違反 1.0.
        let ax_dd = vec![TwoFloat::from(1.0_f64) + TwoFloat::new_mul(1.0e16, 1.0) - TwoFloat::new_mul(1.0e16, 1.0)];
        let b = vec![0.0];
        let ct = vec![Eq];
        let v = dd_impl::constraint_violations(&ax_dd, &b, &ct);
        assert!((v[0] - 1.0).abs() < 1e-12, "got {}", v[0]);
    }

    #[test]
    fn f64_comp_ineq_products_skip_eq() {
        use ConstraintType::*;
        let ax = vec![2.0, 5.0, 3.0];
        let b = vec![3.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 7.0];
        let ct = vec![Le, Ge, Eq];
        let p = f64_impl::comp_ineq_products(&ax, &b, &ct, &y);
        // Le: |1·(3−2)|=1, Ge: |2·(5−2)|=6, Eq: 0
        assert_eq!(p, vec![1.0, 6.0, 0.0]);
    }

    #[test]
    fn dd_qx_aty_ax_match_f64_on_well_conditioned() {
        use crate::sparse::CscMatrix;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0_f64, 3.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0_f64, 2.0], 1, 2).unwrap();
        let x = vec![5.0_f64, 7.0];
        let y = vec![4.0_f64];

        let qx_f = f64_impl::qx(&q, &x);
        let qx_d: Vec<f64> = dd_impl::qx(&q, &x).iter().map(|&v| f64::from(v)).collect();
        assert_eq!(qx_f, qx_d);

        let aty_f = f64_impl::aty(&a, &y, 2);
        let aty_d: Vec<f64> = dd_impl::aty(&a, &y, 2).iter().map(|&v| f64::from(v)).collect();
        assert_eq!(aty_f, aty_d);

        let ax_f = f64_impl::ax(&a, &x);
        let ax_d: Vec<f64> = dd_impl::ax(&a, &x).iter().map(|&v| f64::from(v)).collect();
        assert_eq!(ax_f, ax_d);
    }
}
