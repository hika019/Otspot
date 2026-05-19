//! 下界 / 上界 計算 (#6 Phase 3 spatial B&B)。
//!
//! ## 下界 (lower bound)
//! Phase 3 は **interval arithmetic** = box 各次元の値域から f = c'x + 1/2 x'Q x の
//! 各項を独立に上下限化、和を取る。線形制約 Ax = b は無視するため **緩い** 下界
//! (Phase 4 で α-BB / McCormick に置換予定)。それでも box pruning に最低限の
//! discrimination 力はある (= ∞ よりは tight)。
//!
//! ## 上界 (upper bound)
//! IPM (非凸 inertia 補正付き) で local solve、解 obj = 当該 box 内 feasible point
//! の objective = incumbent 候補。warm start で parent 解継承で iter 削減。
//!
//! ## Q storage 規約
//! `QpProblem::q` は **full symmetric storage** (CSC で両半 (i,j) と (j,i) の両 entry)。
//! objective は solver 内部で `Q.mat_vec_mul(x)` を使って `0.5 x' Q x` を計算する
//! ため、Q は格納どおりに literal 解釈される。テスト fixture も両半 store が標準
//! (src/qp/tests.rs::test_qp_least_squares, tests/diag_qp_multistart_smoke.rs)。
//!
//! 区間 bound 計算は全 entry を走査し 0.5*v_ij の寄与を累積する。

use crate::options::{QpWarmStart, SolverOptions};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;

/// box [a, b] における x² の値域。
/// box が 0 を跨ぐなら min=0、それ以外は端点 min。
fn square_interval(a: f64, b: f64) -> (f64, f64) {
    let aa = a * a;
    let bb = b * b;
    let min = if a <= 0.0 && b >= 0.0 {
        0.0
    } else {
        aa.min(bb)
    };
    let max = aa.max(bb);
    (min, max)
}

/// box ([a1,b1] × [a2,b2]) における x*y の値域 = 4 端点の min/max。
fn product_interval(a1: f64, b1: f64, a2: f64, b2: f64) -> (f64, f64) {
    let c = [a1 * a2, a1 * b2, b1 * a2, b1 * b2];
    let mut lo = c[0];
    let mut hi = c[0];
    for &v in c.iter().skip(1) {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// 区間演算で f(x) = obj_offset + c'x + 1/2 x'Q x の box 上下限を返す。
/// 戻り値 = (lower, upper)。
/// 制約 Ax = b は無視するため "true" 下界より緩い。Phase 3 scaffolding 用。
pub(crate) fn interval_quadratic_bounds(
    problem: &QpProblem,
    bounds: &[(f64, f64)],
) -> (f64, f64) {
    let n = problem.num_vars;
    debug_assert_eq!(bounds.len(), n, "bounds length mismatch");

    let mut lo = problem.obj_offset;
    let mut hi = problem.obj_offset;

    // 線形項 c' x: 各 c_i * x_i の min/max を端点から
    for i in 0..n {
        let (a, b) = bounds[i];
        if !a.is_finite() || !b.is_finite() {
            // 無限境界 → 下界 -∞、上界 +∞ (= 当該 box では bound 不能)
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let p = problem.c[i] * a;
        let q = problem.c[i] * b;
        lo += p.min(q);
        hi += p.max(q);
    }

    // 二次項 0.5 x' Q x。Q は full symmetric storage 規約 (両半 entry あり)。
    // 各 entry の寄与 = 0.5 * v_ij * x_i * x_j (対角は x_i² の interval を直接、
    // 非対角は product interval)。lower-tri 半分も同様に加算され、両半合計で
    // 正しい 0.5 x'Qx 値域になる。
    for col in 0..n {
        let (a_c, b_c) = bounds[col];
        for k in problem.q.col_ptr[col]..problem.q.col_ptr[col + 1] {
            let row = problem.q.row_ind[k];
            let v = problem.q.values[k];
            let coeff = 0.5 * v;
            if row == col {
                let (x2_min, x2_max) = square_interval(a_c, b_c);
                let t1 = coeff * x2_min;
                let t2 = coeff * x2_max;
                lo += t1.min(t2);
                hi += t1.max(t2);
            } else {
                let (a_r, b_r) = bounds[row];
                let (p_min, p_max) = product_interval(a_r, b_r, a_c, b_c);
                let t1 = coeff * p_min;
                let t2 = coeff * p_max;
                lo += t1.min(t2);
                hi += t1.max(t2);
            }
        }
    }
    (lo, hi)
}

/// warm.x が新 box の境界に張り付いていると判定する相対 margin。
/// IPM tolerance より大きい、box width に対する相対比。
const WARM_BOUNDARY_REL_MARGIN: f64 = 1e-3;

/// parent warm を新 box で活用可能か判定。
/// warm.x のいずれかが新 box 境界に張り付くと IPM が saddle / stalled KKT に固着する
/// (concave-symmetric QP の典型病理) ため、その場合は warm を捨てて cold restart に倒す。
fn warm_is_safe_for_box(warm: &QpWarmStart, bounds: &[(f64, f64)]) -> bool {
    if warm.x.len() != bounds.len() {
        return false;
    }
    for (xi, &(lb, ub)) in warm.x.iter().zip(bounds.iter()) {
        if !lb.is_finite() || !ub.is_finite() {
            continue;
        }
        let width = ub - lb;
        if width <= 0.0 {
            return false;
        }
        let margin = WARM_BOUNDARY_REL_MARGIN * width;
        if *xi < lb + margin || *xi > ub - margin {
            return false;
        }
    }
    true
}

/// IPM local solve on the box subproblem → upper bound (incumbent candidate)。
///
/// `parent_warm` が Some なら interior point warm start で起動 (#12 利用)。
/// `node_bounds` で problem.bounds を差し替えた clone を solve。
/// silent SKIP しない: solver の status をそのまま返す。
/// `multistart` / `global_optimization` は **強制 None** (= 再入防止 + 子 solve は単発 local)。
///
/// warm.x が新 box 境界に張り付くケースでは warm を **捨てて cold** で起動
/// (= concave-symmetric saddle に再固着するのを避ける、Phase 3 退化対応)。
pub(crate) fn solve_local_upper_bound(
    problem: &QpProblem,
    node_bounds: &[(f64, f64)],
    base_opts: &SolverOptions,
    parent_warm: Option<&QpWarmStart>,
) -> SolverResult {
    let mut sub = problem.clone();
    sub.bounds = node_bounds.to_vec();
    let mut opts = base_opts.clone();
    opts.warm_start_qp = parent_warm.and_then(|w| {
        if warm_is_safe_for_box(w, node_bounds) {
            Some(w.clone())
        } else {
            None
        }
    });
    opts.multistart = None;
    opts.global_optimization = None;
    crate::qp::solve_qp_with(&sub, &opts)
}

/// box bounds が全変数で有限かつ `l ≤ u` か。
/// α-BB / McCormick underestimator は有限 box が前提 (semi-infinite では `(x-l)(x-u)`
/// や `w_{ij}` bound が定義できない)。caller は false なら interval / `-∞` に fall back する。
pub(crate) fn all_bounds_finite(bounds: &[(f64, f64)]) -> bool {
    bounds
        .iter()
        .all(|&(l, u)| l.is_finite() && u.is_finite() && u >= l)
}

/// 解 status が incumbent 候補として採用可能か。
/// Optimal / LocallyOptimal / SuboptimalSolution / MaxIterations は feasible point を持つ。
/// Infeasible / Unbounded / NumericalError / Timeout / NonConvex は incumbent 候補外。
pub(crate) fn is_feasible_result(status: &SolveStatus) -> bool {
    matches!(
        status,
        SolveStatus::Optimal
            | SolveStatus::LocallyOptimal
            | SolveStatus::SuboptimalSolution
            | SolveStatus::MaxIterations
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn build_diag_qp(diag: &[f64], c: &[f64], bounds: Vec<(f64, f64)>) -> QpProblem {
        let n = diag.len();
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        let q = CscMatrix::from_triplets(&rows, &cols, diag, n, n).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        QpProblem::new_all_le(q, c.to_vec(), a, vec![], bounds).unwrap()
    }

    #[test]
    fn square_interval_zero_crossing_yields_zero_min() {
        let (lo, hi) = square_interval(-2.0, 3.0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 9.0);
    }

    #[test]
    fn square_interval_positive_box() {
        let (lo, hi) = square_interval(1.0, 4.0);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 16.0);
    }

    #[test]
    fn square_interval_negative_box() {
        let (lo, hi) = square_interval(-4.0, -1.0);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 16.0);
    }

    #[test]
    fn product_interval_general() {
        let (lo, hi) = product_interval(-2.0, 1.0, -3.0, 4.0);
        // candidates: (-2)*(-3)=6, (-2)*4=-8, 1*(-3)=-3, 1*4=4
        assert_eq!(lo, -8.0);
        assert_eq!(hi, 6.0);
    }

    #[test]
    fn convex_diag_qp_interval_matches_box_min() {
        // f = 0.5*2*x² + 0*x = x², on [-2, 3] → min=0 at x=0, max=9 at x=3
        let p = build_diag_qp(&[2.0], &[0.0], vec![(-2.0, 3.0)]);
        let (lo, hi) = interval_quadratic_bounds(&p, &p.bounds);
        assert!((lo - 0.0).abs() < 1e-12, "lo={lo}");
        assert!((hi - 9.0).abs() < 1e-12, "hi={hi}");
    }

    #[test]
    fn concave_diag_qp_interval_min_at_corner() {
        // f = -x², on [-2, 3] → 0.5*q*x² with q=-2: -x²
        // min over box = -9 at x=3 (or -4 at x=-2; corner with larger |x|)
        let p = build_diag_qp(&[-2.0], &[0.0], vec![(-2.0, 3.0)]);
        let (lo, hi) = interval_quadratic_bounds(&p, &p.bounds);
        // interval: square in [0, 9], coeff=-1, range=[-9, 0]
        assert!((lo - (-9.0)).abs() < 1e-12, "lo={lo}");
        assert!((hi - 0.0).abs() < 1e-12, "hi={hi}");
    }

    #[test]
    fn linear_objective_contributes() {
        // f = c'x with c=[2,-3], x in [-1,1]^2: lo = -2+(-3) = -5, hi = 2+3=5
        let p = build_diag_qp(&[0.0, 0.0], &[2.0, -3.0], vec![(-1.0, 1.0); 2]);
        let (lo, hi) = interval_quadratic_bounds(&p, &p.bounds);
        assert!((lo - (-5.0)).abs() < 1e-12);
        assert!((hi - 5.0).abs() < 1e-12);
    }

    #[test]
    fn bilinear_off_diagonal_handled() {
        // Q full-symmetric (両半 entry): Q = [[0,1],[1,0]] → 0.5 x'Qx = x*y
        // On [-1,1]^2 → range [-1, 1].
        let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
        let p = QpProblem::new_all_le(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2])
            .unwrap();
        let (lo, hi) = interval_quadratic_bounds(&p, &p.bounds);
        assert!((lo - (-1.0)).abs() < 1e-12, "lo={lo}");
        assert!((hi - 1.0).abs() < 1e-12, "hi={hi}");
    }

    #[test]
    fn infinite_bound_returns_unbounded_interval() {
        let p = build_diag_qp(&[1.0], &[0.0], vec![(f64::NEG_INFINITY, 1.0)]);
        let (lo, hi) = interval_quadratic_bounds(&p, &p.bounds);
        assert!(lo.is_infinite() && lo < 0.0);
        assert!(hi.is_infinite() && hi > 0.0);
    }

    #[test]
    fn warm_safe_when_x_well_inside_box() {
        let warm = QpWarmStart {
            x: vec![0.5, -0.3],
            y: vec![],
            mu: 1e-6,
        };
        assert!(warm_is_safe_for_box(&warm, &[(0.0, 1.0), (-1.0, 1.0)]));
    }

    #[test]
    fn warm_unsafe_when_x_on_boundary() {
        // x[0] = 0.0001 is within 0.1% of lb=0.0 (width=1.0, margin=1e-3) → unsafe
        let warm = QpWarmStart {
            x: vec![0.0001, 0.0],
            y: vec![],
            mu: 1e-6,
        };
        assert!(!warm_is_safe_for_box(&warm, &[(0.0, 1.0), (-1.0, 1.0)]));
    }

    #[test]
    fn warm_unsafe_when_x_outside_box_post_branching() {
        // parent solved at (1.0, 0), child box now [-1, 0] for var 0 → x[0]=1.0 outside
        let warm = QpWarmStart {
            x: vec![1.0, 0.0],
            y: vec![],
            mu: 1e-6,
        };
        assert!(!warm_is_safe_for_box(&warm, &[(-1.0, 0.0), (-1.0, 1.0)]));
    }

    #[test]
    fn feasibility_classifier_covers_finite_obj_statuses() {
        for s in [
            SolveStatus::Optimal,
            SolveStatus::LocallyOptimal,
            SolveStatus::SuboptimalSolution,
            SolveStatus::MaxIterations,
        ] {
            assert!(is_feasible_result(&s), "{s:?} should be feasible");
        }
        for s in [
            SolveStatus::Infeasible,
            SolveStatus::Unbounded,
            SolveStatus::NumericalError,
            SolveStatus::Timeout,
            SolveStatus::NonConvex("x".into()),
        ] {
            assert!(!is_feasible_result(&s), "{s:?} should NOT be feasible");
        }
    }
}
