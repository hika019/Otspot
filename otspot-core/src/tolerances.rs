//! 数値許容値の一元管理モジュール
//!
//! ソルバー全体で使用する数値定数をここに集約する。
//! 定数値を変更する場合はこのファイルのみを編集すればよい。

/// 構造的ゼロ判定の閾値（絶対値がこれ以下の値をゼロとみなす）
pub const ZERO_TOL: f64 = 1e-12;

/// シンプレックス法の最適性・実行可能性判定の閾値
pub const PIVOT_TOL: f64 = 1e-8;

/// 行列構築時の微小値除去の閾値
pub const DROP_TOL: f64 = 1e-15;

/// シンプレックス法のピボット安定性しきい値
///
/// FTRAN 後の入基列で |d\[leaving_row\]| / max(|d|) < PIVOT_STABILITY_THRESHOLD の場合、
/// ピボット前に LU を再因子分解して eta 蓄積による数値誤差をリセットする。
/// 値: 最大列エントリの 1% 未満のピボットを「不安定」と判定。
pub const PIVOT_STABILITY_THRESHOLD: f64 = 0.01;

/// FX (固定) 変数判定: `|lb − ub| < FX_TOL` なら lb=ub の fixed 変数とみなす。
///
/// QP postsolve / refine / IPM stationarity 寄与の bound 評価で共用される。
pub const FX_TOL: f64 = 1e-12;

/// 相補性スラック判定の relative tolerance。
///
/// 行 i の primal slack が `COMP_SLACK_REL_TOL * (1 + |b_i| + |Ax_i|)` を超えれば
/// non-binding と判定し、KKT の `y_i · slack_i = 0` から `y_i = 0` を強制する。
/// LP postsolve の cleanup-LP / LSQ 経路、`recover_removed_row_dual` の
/// non-binding short-circuit が共用する。
pub const COMP_SLACK_REL_TOL: f64 = 1e-6;

/// 数値的「同等」(near-zero) 判定の relative tolerance。
///
/// `PIVOT_TOL.sqrt()` (= 1e-4 when PIVOT_TOL=1e-8)。
/// Wilkinson の経験則「LU 累積丸め誤差 ~ sqrt(機械精度)」に対応した派生値で、
/// magic ではなく PIVOT_TOL から構造的に導出される。
///
/// 用途: 解返却前の defense-in-depth check
///   - `|Ax - b| <= FEAS_REL_TOL * (1 + |b| + |Ax|)` (Eq 制約 feasibility)
///   - `|x - bound| <= FEAS_REL_TOL * (1 + |x| + |bound|)` (bound hit 判定)
///
/// 内部最適性判定 (PIVOT_TOL=1e-8) よりは緩いが、解返却前の false-positive
/// 検出としては十分厳しい。
pub fn feas_rel_tol() -> f64 {
    PIVOT_TOL.sqrt()
}

/// Relative tolerance for objective-value matching against a known reference.
///
/// `|obj − ref| / (1 + |ref|) < OBJ_MATCH_REL_TOL` is used by `obj_within_tol`
/// and the `known_optimal_obj` early-exit logic in lp_dispatch.
pub const OBJ_MATCH_REL_TOL: f64 = 1e-4;

/// Returns `true` when `obj` is within relative tolerance of `ref_obj`.
///
/// Criterion: `|obj − ref_obj| / (1 + |ref_obj|) < tol`.
/// Returns `false` if either value is non-finite.
pub fn obj_within_tol(obj: f64, ref_obj: f64, tol: f64) -> bool {
    if !obj.is_finite() || !ref_obj.is_finite() {
        return false;
    }
    (obj - ref_obj).abs() / (1.0 + ref_obj.abs()) < tol
}

/// Integer bound rounding tolerance.
///
/// When an implied bound for an integer variable is within this tolerance of an
/// integer value, it is rounded to that integer rather than away from it.  This
/// guards against floating-point drift: e.g. `0.3 / 0.1` evaluates to
/// `2.9999999999999996` in IEEE 754, so `floor` without tolerance would
/// incorrectly give `2` instead of `3`.
///
/// Chosen to be between `ZERO_TOL` (1e-12, structural zero) and `PIVOT_TOL`
/// (1e-8, solver primal tolerance), so that genuine non-integer implied bounds
/// are unaffected while float-arithmetic drift is absorbed.
pub const INT_ROUND_TOL: f64 = 1e-9;

/// アンダーフロー防止ガード閾値。行/列の最大絶対値がこれ以下の場合、
/// スケール係数の逆数計算によるオーバーフローを防ぐため 1.0 に固定する。
/// (1 / 1e-300 = 1e300 は表現可能だが、値として無意味なスケールを生む)
pub const UNDERFLOW_GUARD: f64 = 1e-300;

/// Minimum per-row scale factor for large-coefficient rescaling.
///
/// When a row contains entries with `|A[i,j]| > LARGE_A_COEFF_TRIGGER`, each
/// affected row is scaled by `σ_i = 1/√(max|A[i,*]|)`, clamped from below by
/// this value.  Clamping caps the per-stage amplification at `1/SCALING_SIGMA_FLOOR = 1e3`,
/// so the composite scaling (phase1 · phase2 · Ruiz) stays within the IPM's
/// achievable scaled accuracy.
pub const SCALING_SIGMA_FLOOR: f64 = 1e-3;

/// Trigger threshold for presolve row-scaling when `|A[i,j]|` exceeds this magnitude.
/// Units: A entry magnitude.
/// Used in: `presolve::qp_transforms::helpers::apply_large_coeff_rescaling`.
pub const LARGE_A_COEFF_TRIGGER: f64 = 1e6;

/// Dimensionless ratio threshold for Q diagonal range: if `q_pos_max / q_pos_min`
/// exceeds this value, IPM Q-diagonal scaling is applied to reduce conditioning.
/// Units: dimensionless ratio.
/// Used in: `qp::ipm_solver::attempt::try_q_diagonal_scaling`.
pub const Q_DIAG_RANGE_TRIGGER: f64 = 1e6;

/// Absolute drop threshold for Q off-diagonal pruning in `near_zero_q_removal`.
///
/// Off-diagonal entries `|Q[i,j]| < Q_OFFDIAG_ABS` are removed to improve
/// sparsity before IPM solve. Uses an absolute (not scale-relative) threshold
/// because empirical evidence shows that a purely relative threshold can leave
/// entries that change IPM trajectory and cause spurious Optimal convergence
/// (e.g. QPLIB_9002, 2026-05-30 audit#123).
pub const Q_OFFDIAG_ABS: f64 = 1e-10;

/// Relative tolerance for Q off-diagonal near-zero detection in `is_diagonal_q`
/// and `try_q_diagonal_scaling`.
///
/// An off-diagonal entry `|Q[i,j]|` is treated as structurally zero when
/// `|Q[i,j]| < Q_OFFDIAG_REL * q_abs_max + UNDERFLOW_GUARD`,
/// where `q_abs_max` is the global maximum absolute entry of Q.
///
/// Usage: `presolve/qp_transforms/helpers::is_diagonal_q` and
/// `qp/ipm_solver/attempt::try_q_diagonal_scaling`.
pub const Q_OFFDIAG_REL: f64 = 1e-12;

/// Absolute feasibility tolerance for the MIP fixed-point subproblem.
///
/// Used in `solve_fixed_point` when every variable is pinned to a single value
/// (zero-width box). The check `|Ax_k - b_k| <= FIXED_POINT_FEAS_TOL` is purely
/// absolute because x is already uniquely determined; there is no free scale.
///
/// Intentionally distinct from `COMP_SLACK_REL_TOL` (a relative scale factor
/// for LP postsolve): conflating them would cause LP postsolve tuning to silently
/// alter the MIP fixed-point feasibility gate.
pub const FIXED_POINT_FEAS_TOL: f64 = 1e-6;

/// Size gate shared across expensive post-processing sites.
///
/// Problems above this threshold skip high-cost operations (primal projection,
/// KKT refinement, presolve perturbation) to reserve budget for the IPM core.
///
/// Usage varies by site: some compare `n + m` against this value; others
/// check each dimension individually (`num_vars <= T && num_constraints <= T`).
pub const LARGE_PROBLEM_THRESHOLD: usize = 50_000;

/// Returns `true` if any element of `v` is non-finite (NaN or ±Inf).
pub(crate) fn any_nonfinite(v: &[f64]) -> bool {
    v.iter().any(|x| !x.is_finite())
}

