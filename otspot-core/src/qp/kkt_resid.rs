//! QP KKT 残差の per-component primitives.
//!
//! 7 callers (bench_utils / ipm_solver::kkt 3 / qps_benchmark / diag_nonconvex_kkt /
//! verify_solutions) が散在的に持っていた Q·x / A^T·y / A·x / bound_contrib /
//! complementarity slack の重実装をここに集約する。caller 側は
//!   - 集約方法 (max abs / max componentwise rel / global rel / 構造体格納)
//!   - 経路 fork (LP rc 経路 / QP bound_dual 経路 / FX skip / EmptyCol skip)
//!     を選ぶ責務のみ保持し、heavy mat-vec と bound iteration ロジックは
//!     重複しない。
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

/// KKT dual-sign violation (componentwise relative max).
///
/// Stationarity `Qx + c + Aᵀy + bound_contrib = 0` uses the convention:
/// - Le constraints: y_i ≥ 0 (violation if y_i < 0)
/// - Ge constraints: y_i ≤ 0 (violation if y_i > 0)
/// - Eq constraints: y_i free (no sign requirement)
/// - lb bound duals (first half of `z`): z_lb_j ≥ 0 (violation if z_lb_j < 0)
/// - ub bound duals (second half of `z`): z_ub_j ≥ 0 (violation if z_ub_j < 0)
///
/// Returns `max{ viol_k / (1 + |v_k|) }` over all sign-constrained components,
/// where `viol_k = max(0, wrong-sign part)`. Returns 0 when all sign constraints hold.
/// Scale invariant: scaling y and z by a positive scalar leaves the result unchanged.
pub fn dual_sign_violation(ct: &[ConstraintType], y: &[f64], bounds: &[(f64, f64)], z: &[f64]) -> f64 {
    let mut max_rel = 0.0_f64;

    // Constraint dual sign check.
    let m = ct.len().min(y.len());
    #[allow(unreachable_patterns)]
    for i in 0..m {
        let viol = match ct[i] {
            ConstraintType::Le => (-y[i]).max(0.0),   // must be >= 0
            ConstraintType::Ge => y[i].max(0.0),       // must be <= 0
            ConstraintType::Eq => 0.0,                 // free
            _ => 0.0,
        };
        if viol > 0.0 {
            let rel = viol / (1.0 + y[i].abs());
            if rel > max_rel { max_rel = rel; }
        }
    }

    // Bound dual sign check (lb half: >= 0, ub half: <= 0).
    // z layout mirrors bound_contrib: lb-finite columns first, then ub-finite columns.
    if z.is_empty() {
        return max_rel;
    }
    let mut idx = 0_usize;
    for &(lb, _) in bounds.iter() {
        if lb.is_finite() && idx < z.len() {
            let v = (-z[idx]).max(0.0); // z_lb must be >= 0
            if v > 0.0 {
                let rel = v / (1.0 + z[idx].abs());
                if rel > max_rel { max_rel = rel; }
            }
            idx += 1;
        }
    }
    for &(_, ub) in bounds.iter() {
        if ub.is_finite() && idx < z.len() {
            let v = (-z[idx]).max(0.0); // z_ub must be >= 0
            if v > 0.0 {
                let rel = v / (1.0 + z[idx].abs());
                if rel > max_rel { max_rel = rel; }
            }
            idx += 1;
        }
    }
    max_rel
}

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

    /// Per-component normalised primal feasibility (f64 path).
    ///
    /// `max_i violation_i / (1 + |Ax_i| + |b_i|)`.
    ///
    /// Internal utility for `otspot-dev`; not part of the stable public API.
    #[doc(hidden)]
    pub fn primal_residual_rel(a: &CscMatrix, b: &[f64], ct: &[ConstraintType], x: &[f64]) -> f64 {
        debug_assert_eq!(b.len(), ct.len(), "b and ct must have equal length");
        let ax = self::ax(a, x);
        if ax.is_empty() {
            return 0.0;
        }
        let viols = self::constraint_violations(&ax, b, ct);
        let mut max_rel = 0.0_f64;
        for (i, &v) in viols.iter().enumerate() {
            let scale_i = 1.0 + ax[i].abs() + b[i].abs();
            max_rel = max_rel.max(v / scale_i);
        }
        max_rel
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
                out[row] += TwoFloat::new_mul(q.values[k], xv);
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
                out[col] += TwoFloat::new_mul(a.values[k], y[row]);
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
                out[a.row_ind[k]] += TwoFloat::new_mul(a.values[k], xv);
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

    // ── dual_sign_violation tests ─────────────────────────────────────────────

    /// Table-driven: Le constraint — y must be >= 0.
    #[test]
    fn dual_sign_le_y_negative_is_violation() {
        use ConstraintType::*;
        // y_Le = -0.5 (should be >= 0) → violation
        let ct = vec![Le];
        let y = vec![-0.5_f64];
        let bounds: Vec<(f64, f64)> = vec![];
        let z: Vec<f64> = vec![];
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        assert!(v > 0.0, "Le with y=-0.5 should give violation > 0, got {}", v);
        // expected: 0.5 / (1 + 0.5) = 0.5/1.5 ≈ 0.333
        assert!((v - 0.5 / 1.5).abs() < 1e-12, "exact check: {}", v);
    }

    #[test]
    fn dual_sign_le_y_positive_no_violation() {
        use ConstraintType::*;
        let ct = vec![Le, Le];
        let y = vec![0.0_f64, 2.0];
        let bounds: Vec<(f64, f64)> = vec![];
        let z: Vec<f64> = vec![];
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        assert_eq!(v, 0.0, "Le with y>=0 must give 0");
    }

    /// Ge constraint — y must be <= 0.
    #[test]
    fn dual_sign_ge_y_positive_is_violation() {
        use ConstraintType::*;
        let ct = vec![Ge];
        let y = vec![1.0_f64];
        let bounds: Vec<(f64, f64)> = vec![];
        let z: Vec<f64> = vec![];
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        assert!(v > 0.0, "Ge with y=1.0 should be violation > 0, got {}", v);
        assert!((v - 1.0 / 2.0).abs() < 1e-12, "exact: {}", v);
    }

    #[test]
    fn dual_sign_ge_y_negative_no_violation() {
        use ConstraintType::*;
        let ct = vec![Ge];
        let y = vec![-3.0_f64];
        let bounds: Vec<(f64, f64)> = vec![];
        let z: Vec<f64> = vec![];
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        assert_eq!(v, 0.0, "Ge with y=-3 must give 0");
    }

    /// Eq constraint — y is free, no violation regardless of sign.
    #[test]
    fn dual_sign_eq_y_any_sign_no_violation() {
        use ConstraintType::*;
        for yi in [-100.0, -1.0, 0.0, 1.0, 100.0] {
            let ct = vec![Eq];
            let y = vec![yi];
            let v = dual_sign_violation(&ct, &y, &[], &[]);
            assert_eq!(v, 0.0, "Eq with y={yi} should give 0");
        }
    }

    /// Mixed Le/Ge/Eq: only the violating component contributes.
    #[test]
    fn dual_sign_mixed_constraints() {
        use ConstraintType::*;
        // Le y=0.5 (ok), Ge y=0.3 (violation), Eq y=-10 (ok)
        let ct = vec![Le, Ge, Eq];
        let y = vec![0.5, 0.3, -10.0];
        let v = dual_sign_violation(&ct, &y, &[], &[]);
        // Ge violation: 0.3 / (1 + 0.3) = 0.3/1.3 ≈ 0.2308
        let expected = 0.3 / 1.3;
        assert!((v - expected).abs() < 1e-12, "got {v}, expected {expected}");
    }

    /// z_lb must be >= 0: negative z_lb is a violation.
    #[test]
    fn dual_sign_z_lb_negative_is_violation() {
        use ConstraintType::*;
        let ct = vec![Le];
        let y = vec![0.5_f64]; // Le ok
        let bounds = vec![(0.0_f64, f64::INFINITY)]; // lb=0 finite, ub=inf
        let z = vec![-0.4_f64]; // lb-dual must be >= 0
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        // violation from z: 0.4 / (1 + 0.4) = 0.4/1.4 ≈ 0.286
        let expected = 0.4 / 1.4;
        assert!((v - expected).abs() < 1e-12, "got {v}");
    }

    /// z_ub must be >= 0: negative z_ub is a violation.
    #[test]
    fn dual_sign_z_ub_negative_is_violation() {
        let ct: Vec<ConstraintType> = vec![];
        let y: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, 1.0_f64)]; // ub finite
        let z = vec![-0.7_f64]; // ub-dual must be >= 0; negative is violation
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        let expected = 0.7 / 1.7;
        assert!((v - expected).abs() < 1e-12, "got {v}");
    }

    /// z_ub positive (correct sign) → no violation.
    #[test]
    fn dual_sign_z_ub_positive_no_violation() {
        let ct: Vec<ConstraintType> = vec![];
        let y: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, 1.0_f64)]; // ub finite
        let z = vec![0.7_f64]; // z_ub >= 0: correct sign
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        assert_eq!(v, 0.0, "positive z_ub must not be a violation, got {v}");
    }

    /// z = [] means no bound duals: no violation.
    #[test]
    fn dual_sign_empty_z_no_violation() {
        use ConstraintType::*;
        let ct = vec![Le];
        let y = vec![0.5_f64];
        let bounds = vec![(0.0, f64::INFINITY)];
        let v = dual_sign_violation(&ct, &y, &bounds, &[]);
        assert_eq!(v, 0.0);
    }

    /// Scale robustness: large and small violations both give bounded results in (0, 1].
    ///
    /// The `1 + |v|` denominator gives componentwise relative normalisation:
    /// violation / (1 + |violation|) is in [0, 1). Larger duals give smaller relative
    /// violations (closer to 1), but the value is always bounded.
    #[test]
    fn dual_sign_scale_robust() {
        use ConstraintType::*;
        // Ge violation: y > 0 for Ge constraint
        let ct = vec![Ge];
        for &yi in &[1e-6_f64, 1.0, 1e3, 1e9] {
            let v = dual_sign_violation(&ct, &[yi], &[], &[]);
            assert!(v > 0.0 && v < 1.0,
                "violation y={yi} must be in (0,1), got {v}");
            // As yi → ∞, violation → 1
            if yi > 100.0 {
                assert!(v > 0.99, "large yi={yi} should give v close to 1, got {v}");
            }
        }
        // No violation (correct sign) → always 0
        for &yi in &[-1e-6_f64, -1.0, -1e9] {
            let v = dual_sign_violation(&ct, &[yi], &[], &[]);
            assert_eq!(v, 0.0, "no violation for yi={yi}");
        }
    }

    /// All constraints satisfied (no violations) → 0.
    #[test]
    fn dual_sign_all_satisfied_returns_zero() {
        use ConstraintType::*;
        let ct = vec![Le, Ge, Eq, Le, Ge];
        let y = vec![1.0, -1.0, 0.5, 0.0, -2.0];
        let bounds = vec![
            (0.0_f64, 1.0_f64),  // lb+ub finite: 2 z entries
            (f64::NEG_INFINITY, f64::INFINITY),  // free: no z
        ];
        // z: lb-half=[z_lb_0], ub-half=[z_ub_0]
        // z_lb >= 0 ok, z_ub >= 0 ok (both bound duals non-negative)
        let z = vec![0.5_f64, 0.5]; // z_lb=0.5>=0 ok, z_ub=0.5>=0 ok
        let v = dual_sign_violation(&ct, &y, &bounds, &z);
        assert_eq!(v, 0.0, "all satisfied should give 0");
    }

    /// Empirical observation: solver returns z_ub >= 0 for active upper bound.
    ///
    /// min (x−10)^2 s.t. 0 ≤ x ≤ 5 → optimal x=5 (ub active).
    /// bound_duals layout: [z_lb (lb=0 finite), z_ub (ub=5 finite)].
    /// Stationarity: 2(x−10) + z_ub = 0 → z_ub = 2*(10−5) = 10 > 0.
    #[test]
    fn dual_sign_z_ub_observed_positive_at_active_ub() {
        use crate::qp::{QpProblem, solve_qp};
        use crate::sparse::CscMatrix;
        // min 1/2*(2)*x^2 + (-20)*x ≡ (x-10)^2 + const, 0 ≤ x ≤ 5
        let q = CscMatrix::from_triplets(&[0usize], &[0usize], &[2.0_f64], 1, 1).unwrap();
        let a = CscMatrix::new(0, 1);
        let prob = QpProblem::new(q, vec![-20.0], a, vec![], vec![(0.0, 5.0)], vec![]).unwrap();
        let result = solve_qp(&prob);
        // x* ≈ 5 (ub active)
        assert!((result.solution[0] - 5.0).abs() < 1e-4,
            "x should be ≈5, got {}", result.solution[0]);
        // z = [z_lb, z_ub]; z_ub must be > 0
        assert!(result.bound_duals.len() >= 2,
            "expected >=2 bound duals, got {}", result.bound_duals.len());
        let z_ub = result.bound_duals[1];
        assert!(z_ub > 1.0,
            "z_ub should be ≈10 (active ub dual), got {z_ub}");
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
