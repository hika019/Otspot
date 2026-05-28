//! Warm-start を受け取って interior 補正のみ適用する初期化経路。

use super::state::{
    warm_bound_margin, WARM_BOUND_REL_MARGIN, WARM_MU_MIN, WARM_SY_MIN,
};
use crate::problem::ConstraintType;
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

/// warm start から (x, y, s) を初期化し、有効なら μ を返す (none で cold start)。
///
/// 規約:
/// - `ws.x` 長さ n、`ws.y` 長さ m_orig (user 符号、Ge は内部で反転)、`ws.mu` スカラー
/// - bound row dual / slack は 1.0 で cold 初期化 (B&B でも bound multiplier は不安定)
pub(super) fn apply_qp_warm_start(
    ws: &crate::options::QpWarmStart,
    problem: &QpProblem,
    a_ext: &CscMatrix,
    b_ext: &[f64],
    is_eq_ext: &[bool],
    m_orig: usize,
    m_ext: usize,
    x: &mut [f64],
    y: &mut [f64],
    s: &mut [f64],
) -> Option<f64> {
    let n = problem.num_vars;
    if ws.x.len() != n || ws.y.len() != m_orig {
        return None;
    }
    let mu = ws.mu.max(WARM_MU_MIN);

    for j in 0..n {
        let xj = ws.x[j];
        let (lb, ub) = problem.bounds[j];
        x[j] = match (lb.is_finite(), ub.is_finite()) {
            (true, true) => {
                let range = ub - lb;
                // range × REL は box 幅相対 strict-interior。range が極小 (collapsing box)
                // のときだけ midpoint 退避させ、それ以外は scale 追従させる。
                let margin = range * WARM_BOUND_REL_MARGIN;
                if range > 2.0 * margin {
                    xj.clamp(lb + margin, ub - margin)
                } else {
                    0.5 * (lb + ub)
                }
            }
            (true, false) => xj.max(lb + warm_bound_margin(lb)),
            (false, true) => xj.min(ub - warm_bound_margin(ub)),
            (false, false) => xj,
        };
    }

    // 元制約 dual を内部符号 (Ge は -1 倍) に展開。
    for i in 0..m_orig {
        let yi = match problem.constraint_types[i] {
            ConstraintType::Ge => -ws.y[i],
            _ => ws.y[i],
        };
        y[i] = if is_eq_ext[i] { yi } else { yi.max(WARM_SY_MIN) };
    }

    // 自然な slack s = b_ext − A_ext·x (ineq は WARM_SY_MIN で boundary 退避)。
    let mut ax = vec![0.0_f64; m_ext];
    for col in 0..n {
        for k in a_ext.col_ptr[col]..a_ext.col_ptr[col + 1] {
            ax[a_ext.row_ind[k]] += a_ext.values[k] * x[col];
        }
    }
    for i in 0..m_ext {
        if is_eq_ext[i] {
            s[i] = 0.0;
        } else {
            s[i] = (b_ext[i] - ax[i]).max(WARM_SY_MIN);
        }
    }
    // bound 行 dual は中心パス s·y=μ から逆算 (x interior → y≈0、x active → y≈μ/ε 大)。
    // ユーザーが bound_duals を渡さない設計のため central path 関係で推定する。
    for i in m_orig..m_ext {
        y[i] = (mu / s[i]).max(WARM_SY_MIN);
    }

    Some(mu)
}
