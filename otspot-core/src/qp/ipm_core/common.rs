//! IPM/IP-PMM 共通関数。

use super::kkt::{norm_inf, spmv};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use otspot_num::linalg::ldl;
use otspot_num::linalg::timeout::TimeoutCtx;
use otspot_num::sparse::CscMatrix;

/// 検出器・証明検査が使う problem-scale 相対閾値の基準係数。
///
/// 旧実装の絶対定数 `EPS_INF = 1e-8` は ‖c‖∞ ~ 2.5e4 の問題 (QFORPLAN 実測) で
/// fp ノイズレベルの傾斜 (cᵀdx/‖dx‖∞ = -6.9e-8) を「単調減少 ray」と誤認した
/// (2026-11 調査 /tmp/opencode/inv_eps_anom/REPORT.md §1)。LP 側 simplex Phase I の
/// `farkas_infeasibility_certified` が `tol = dual_tol * max(1, ‖b‖∞)` で
/// ‖b‖∞ スケールしているのと同じ構成へ揃える (tests/diag_klein3_dual_phase1.rs
/// が検証する certified 経路との整合)。`max(1, ·)` により全データ ≤ 1 の
/// 小規模問題では旧絶対閾値とビット同値 — 既存の検出力を落とさない。
const DETECT_REL_EPS: f64 = 1e-8;

/// 検出器が毎回 O(nnz) でノルムを再計算しないよう、solve 1 回分の特徴量を
/// 事前計算するスケールパック。閾値はすべて `DETECT_REL_EPS * max(1, 対応ノルム)`。
pub(crate) struct DetectionScales {
    /// max(1, ‖c‖∞)。降下条件 (cᵀdx) の基準。
    c: f64,
    /// max(1, ‖b_ext‖∞)。Farkas 厳密性 (bᵀdy) の基準 (LP Phase I 前例)。
    b: f64,
    /// max(1, ‖Q‖∞ 行和)。平坦条件 (‖Q·dx‖∞) の基準。
    q: f64,
    /// max(1, ‖A_ext‖∞ 行和)。退行条件 (‖A·dx‖∞) と Farkas 停留条件の基準。
    a: f64,
}

/// CSC 行和ノルム (行方向絶対値和の最大) — `row_infinity_norms` の最大。
fn row_sums_abs_max(m: &CscMatrix) -> f64 {
    m.row_infinity_norms()
        .into_iter()
        .fold(0.0_f64, |acc, s| acc.max(s))
}

impl DetectionScales {
    pub(crate) fn compute(problem: &QpProblem, a_ext: &CscMatrix, b_ext: &[f64]) -> Self {
        Self {
            c: 1.0_f64.max(norm_inf(&problem.c)),
            b: 1.0_f64.max(norm_inf(b_ext)),
            q: 1.0_f64.max(row_sums_abs_max(&problem.q)),
            a: 1.0_f64.max(row_sums_abs_max(a_ext)),
        }
    }

    #[inline]
    fn eps(&self, base: f64) -> f64 {
        DETECT_REL_EPS * base
    }
}

/// ステップ方向 (Δx, Δy) から infeasibility / unboundedness を検出する。
///
/// 閾値は `DetectionScales` 経由の problem-scale 相対 (上記 DETECT_REL_EPS doc 参照)。
/// 検出はヒューリスティックであり、確定判定には [`verify_unbounded_ray`] /
/// [`verify_infeasible_farkas`] の独立証明が別途必要 (受理ゲート側で強制)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_infeasible_or_unbounded(
    dx: &[f64],
    dy: &[f64],
    problem: &QpProblem,
    a_ext: &CscMatrix,
    m_orig: usize,
    m_ext: usize,
    iter: usize,
    delta_p: f64,
    scales: &DetectionScales,
) -> Option<SolveStatus> {
    const MIN_ITER: usize = 5;
    const MIN_DIR_NORM: f64 = 1e-3;

    if iter < MIN_ITER {
        return None;
    }
    let eps_a = scales.eps(scales.a);
    let eps_b = scales.eps(scales.b);
    let eps_c = scales.eps(scales.c);
    let eps_q = scales.eps(scales.q);

    let n = dx.len();

    if m_orig > 0 {
        let dy_orig = &dy[..m_orig];
        let norm_dy_inf = norm_inf(dy_orig);
        if norm_dy_inf > MIN_DIR_NORM {
            let norm_dy = norm_dy_inf;
            let mut at_dy = vec![0.0f64; n];
            for (j, at_dy_j) in at_dy.iter_mut().enumerate() {
                for ptr in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
                    let row = a_ext.row_ind()[ptr];
                    if row < m_orig {
                        *at_dy_j += a_ext.values()[ptr] * dy_orig[row];
                    }
                }
            }
            let cond_a = norm_inf(&at_dy) / norm_dy < eps_a;
            let b_dy: f64 = problem
                .b
                .iter()
                .zip(dy_orig.iter())
                .map(|(&bi, &dyi)| bi * dyi)
                .sum();
            let cond_b = b_dy / norm_dy < -eps_b;
            if cond_a && cond_b {
                return Some(SolveStatus::Infeasible);
            }
        }
    }

    if m_orig == 0 && m_ext > 0 {
        return None;
    }
    let norm_dx_inf = norm_inf(dx);
    if norm_dx_inf <= MIN_DIR_NORM {
        return None;
    }
    let norm_dx = norm_dx_inf;

    let is_lp = problem.q.values().iter().all(|&v| v == 0.0);
    let cond_obj = if is_lp {
        let c_dx: f64 = problem
            .c
            .iter()
            .zip(dx.iter())
            .map(|(&ci, &dxi)| ci * dxi)
            .sum();
        c_dx / norm_dx < -eps_c
    } else {
        let mut qdx = vec![0.0f64; n];
        spmv(&problem.q, dx, &mut qdx);
        for i in 0..n {
            qdx[i] += delta_p * dx[i];
        }
        let norm_qdx: f64 = qdx.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
        let c_dx: f64 = problem
            .c
            .iter()
            .zip(dx.iter())
            .map(|(&ci, &dxi)| ci * dxi)
            .sum();
        (norm_qdx / norm_dx < eps_q) && (c_dx / norm_dx < -eps_c)
    };
    if !cond_obj {
        return None;
    }

    if m_orig > 0 {
        let mut a_dx = vec![0.0f64; m_orig];
        for (j, &dxj) in dx.iter().enumerate() {
            for ptr in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
                let row = a_ext.row_ind()[ptr];
                if row < m_orig {
                    a_dx[row] += a_ext.values()[ptr] * dxj;
                }
            }
        }
        if norm_inf(&a_dx) / norm_dx >= eps_a {
            return None;
        }
    }

    Some(SolveStatus::Unbounded)
}

/// 検出器受理時に求める独立 ray 証明: 回収方向 (dx, ds) の ray
/// `(x + t·dx, s + t·ds)` が **全 t ≥ 0** で拡張系 `{A_ext x + s = b_ext, s ≥ 0}`
/// の実行可能性を保ち、目的関数が線形に減少することを検証する。
///
/// 条件 (すべて problem-scale 相対閾値、`DETECT_REL_EPS` doc 参照):
/// 1. 平坦性 `‖Q·dx‖∞ ≤ eps · ‖dx‖∞` — **delta_p 正則化を含まない生の Q**
///    (正則化項は問題の Q ではなく PMM の足場であり、平坦性の証拠にならない)。
/// 2. 降下 `cᵀdx ≤ -eps_c · ‖dx‖∞`。
/// 3. 退行実行可能性: 全 m_ext 行 (境界行を含む) で `|a_ext_i·dx + ds_i| ≈ 0`。
/// 4. 退行非負性: s ≥ 0 全行で `ds_i ≥ -eps` — s(t) = s₀ + t·ds が全 t ≥ 0 で ≥ 0。
///
/// 検出器 (`check_infeasible_or_unbounded`) との独立点: 検出器は (a) 境界行を
/// 一切見ない (rows < m_orig のみ)、(b) ds の符号条件を検査しない、
/// (c) 平坦性に delta_p を織り込む、(d) 絶対閾値 — のいずれもが偽陽性の原因
/// (QFORPLAN 実測: ‖Q·dx‖∞ = 1.0e-14 = fp ノイズ、cᵀdx/‖dx‖∞ = -6.9e-8 なのに
/// ‖c‖∞ ~ 2.5e4)。LP 側 simplex Phase I の `farkas_infeasibility_certified`
/// (tol = dual_tol · max(1, ‖b‖∞) で検証し失敗時は判定を撤回する) と同じ
/// 契約を QP 側 unboundedness に導入したもの。
pub(crate) fn verify_unbounded_ray(
    dx: &[f64],
    ds: &[f64],
    problem: &QpProblem,
    a_ext: &CscMatrix,
    is_eq_ext: &[bool],
    scales: &DetectionScales,
) -> bool {
    const MIN_DIR_NORM: f64 = 1e-3;
    let n = dx.len();
    let norm_dx = norm_inf(dx);
    if norm_dx < MIN_DIR_NORM {
        return false;
    }
    let norm_ds = norm_inf(ds);

    // 1. 平坦性: delta_p なしの Q·dx。
    let mut qdx = vec![0.0_f64; n];
    spmv(&problem.q, dx, &mut qdx);
    if norm_inf(&qdx) > scales.eps(scales.q) * norm_dx {
        return false;
    }

    // 2. 降下。
    let c_dx: f64 = problem
        .c
        .iter()
        .zip(dx.iter())
        .map(|(&ci, &dxi)| ci * dxi)
        .sum();
    if c_dx > -scales.eps(scales.c) * norm_dx {
        return false;
    }

    // 3+4. 退行実行可能性 (全行) と退行非負性 (s ≥ 0 行)。
    let mut a_dx = vec![0.0_f64; a_ext.nrows()];
    for (j, &dxj) in dx.iter().enumerate() {
        for ptr in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
            a_dx[a_ext.row_ind()[ptr]] += a_ext.values()[ptr] * dxj;
        }
    }
    let row_tol = scales.eps(scales.a) * norm_dx + DETECT_REL_EPS * norm_ds;
    for i in 0..a_ext.nrows() {
        if (a_dx[i] + ds[i]).abs() > row_tol {
            return false;
        }
        if !is_eq_ext[i] && ds[i] < -DETECT_REL_EPS * 1.0_f64.max(norm_ds) {
            return false;
        }
    }
    true
}

/// 検出器受理時に求める独立 Farkas 証明: 方向 dy を**拡張系全体**
/// `{A_ext x + s = b_ext, s ≥ 0}` に対する Farkas 証書 y として検証する。
///
/// 条件 (problem-scale 相対):
/// 1. 符号: s ≥ 0 全行 (境界行を含む) で `y_i ≥ -eps`。等式行の乗数は自由。
/// 2. 停留: 全列 j で `|A_extᵀy|_j ≤ eps_a · ‖y‖∞` (x は自由変数)。
/// 3. 厳密性: `b_extᵀy ≤ -eps_b · ‖y‖∞` — 境界行の寄与を含む全行合計。
///
/// 成立時は任意の実行可能点で bᵀy = yᵀ(Ax+s) = (Aᵀy)ᵀx + Σ yᵢsᵢ ≤ 0 となり、
/// 厳密性と矛盾 → 実行不可能。検出器 (rows < m_orig のみ・符号unchecked・
/// b_orig のみ) より条件が厳しく、LP 側 `farkas_direction_certified`
/// (`Aᵀy ≤ tol`, `bᵀy > tol`, tol = dual_tol · max(1, ‖b‖∞)) と同格の独立検証。
pub(crate) fn verify_infeasible_farkas(
    dy: &[f64],
    a_ext: &CscMatrix,
    b_ext: &[f64],
    is_eq_ext: &[bool],
    scales: &DetectionScales,
) -> bool {
    const MIN_DIR_NORM: f64 = 1e-3;
    let norm_dy = norm_inf(dy);
    if norm_dy < MIN_DIR_NORM {
        return false;
    }

    // 1. 符号: 不等式行 (s ≥ 0) の乗数は非負。
    let sign_tol = DETECT_REL_EPS * 1.0_f64.max(norm_dy);
    for (i, &dyi) in dy.iter().enumerate() {
        if !is_eq_ext[i] && dyi < -sign_tol {
            return false;
        }
    }

    // 2. 停留: A_extᵀy ≈ 0 (全行・全列)。
    for j in 0..a_ext.ncols() {
        let mut aty = 0.0_f64;
        for ptr in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
            aty += a_ext.values()[ptr] * dy[a_ext.row_ind()[ptr]];
        }
        if aty.abs() > scales.eps(scales.a) * norm_dy {
            return false;
        }
    }

    // 3. 厳密性: b_extᵀy < 0 (境界行寄与込み)。
    let b_dy: f64 = b_ext
        .iter()
        .zip(dy.iter())
        .map(|(&bi, &dyi)| bi * dyi)
        .sum();
    b_dy <= -scales.eps(scales.b) * norm_dy
}

/// α = min(1, τ · min_i { -v_i / Δv_i  for Δv_i < 0, skip_mask[i] == false } )
pub(crate) fn fraction_to_boundary_masked(
    v: &[f64],
    dv: &[f64],
    tau: f64,
    skip_mask: &[bool],
) -> f64 {
    let mut alpha = 1.0_f64;
    for (i, (&vi, &dvi)) in v.iter().zip(dv.iter()).enumerate() {
        if skip_mask[i] {
            continue;
        }
        if dvi < 0.0 {
            let step = tau * vi / (-dvi);
            if step < alpha {
                alpha = step;
            }
        }
    }
    alpha
}

/// 制約なし QP: Qx = -c（PD でなければ δ_p I で正則化）。
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_unconstrained(problem: &QpProblem, timeout_ctx: &TimeoutCtx) -> SolverResult {
    let n = problem.num_vars;

    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    if n == 0 {
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            iterations: 0,
            ..Default::default()
        };
    }

    let delta_p = 1e-7;
    let mut triplet_rows: Vec<usize> = Vec::new();
    let mut triplet_cols: Vec<usize> = Vec::new();
    let mut triplet_vals: Vec<f64> = Vec::new();
    let mut diag_added = vec![false; n];

    for col in 0..n {
        for k in problem.q.col_ptr()[col]..problem.q.col_ptr()[col + 1] {
            let row = problem.q.row_ind()[k];
            if row <= col {
                triplet_rows.push(row);
                triplet_cols.push(col);
                let v = problem.q.values()[k] + if row == col { delta_p } else { 0.0 };
                triplet_vals.push(v);
                if row == col {
                    diag_added[col] = true;
                }
            }
        }
    }
    for i in 0..n {
        if !diag_added[i] {
            triplet_rows.push(i);
            triplet_cols.push(i);
            triplet_vals.push(delta_p);
        }
    }

    let q_reg =
        CscMatrix::from_triplets(&triplet_rows, &triplet_cols, &triplet_vals, n, n).unwrap();

    match ldl::factorize(&q_reg) {
        Ok(fac) => {
            let rhs: Vec<f64> = problem.c.iter().map(|&ci| -ci).collect();
            let mut x = vec![0.0f64; n];
            fac.solve(&rhs, &mut x);

            let mut qx = vec![0.0f64; n];
            spmv(&problem.q, &x, &mut qx);
            let objective = 0.5
                * qx.iter()
                    .zip(x.iter())
                    .map(|(&qi, &xi)| qi * xi)
                    .sum::<f64>()
                + problem
                    .c
                    .iter()
                    .zip(x.iter())
                    .map(|(&ci, &xi)| ci * xi)
                    .sum::<f64>();

            SolverResult {
                status: SolveStatus::Optimal,
                objective,
                solution: x,
                dual_solution: vec![],
                bound_duals: vec![],
                iterations: 1,
                ..Default::default()
            }
        }
        Err(_) => numerical_error_result(n),
    }
}

pub(crate) fn timeout_result(n: usize) -> SolverResult {
    SolverResult {
        solution: vec![0.0; n],
        ..SolverResult::timeout()
    }
}

pub(crate) fn numerical_error_result(n: usize) -> SolverResult {
    SolverResult {
        status: SolveStatus::NumericalError,
        objective: f64::INFINITY,
        solution: vec![0.0; n],
        dual_solution: vec![],
        bound_duals: vec![],
        iterations: 0,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qp::ipm_core::kkt::build_extended_constraints;
    use crate::qp::problem::QpProblem;

    fn lp_like(problem: &QpProblem) -> (CscMatrix, Vec<f64>, Vec<bool>, DetectionScales) {
        let (a_ext, b_ext, _m_ext, _m_orig, _n_lb, is_eq_ext) = build_extended_constraints(problem);
        let scales = DetectionScales::compute(problem, &a_ext, &b_ext);
        (a_ext, b_ext, is_eq_ext, scales)
    }

    /// 真 ray が通る: min -x₁ (s.t. x₂ ≥ 0 のみ)。dx = e₁ は実行可能性を保ち
    /// 目的関数を -1/iter 単位で単調減少させる。
    #[test]
    fn verify_ray_accepts_true_unbounded_ray() {
        let q = CscMatrix::new(2, 2);
        let problem = QpProblem::new_all_le(
            q,
            vec![-1.0, 0.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
        )
        .unwrap();
        let (a_ext, _b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dx = vec![1.0, 0.0];
        let ds = vec![0.0; a_ext.nrows()];
        assert!(verify_unbounded_ray(
            &dx, &ds, &problem, &a_ext, &is_eq_ext, &scales
        ));
    }

    /// QFORPLAN 型偽陽性を拒否: ‖c‖∞ = 2.5e4 の問題で降下率が
    /// cᵀdx/‖dx‖∞ = -6.9e-8 (fp ノイズレベル) では ray と認定しない。
    /// 旧絶対閾値 (1e-8) なら発火していた値。
    #[test]
    fn verify_ray_rejects_noise_level_descent_on_large_c() {
        let q = CscMatrix::new(2, 2);
        let problem = QpProblem::new_all_le(
            q,
            vec![-6.9e-8, 2.5e4],
            CscMatrix::new(0, 2),
            vec![],
            vec![
                (f64::NEG_INFINITY, f64::INFINITY),
                (f64::NEG_INFINITY, f64::INFINITY),
            ],
        )
        .unwrap();
        let (a_ext, _b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dx = vec![1.0, 0.0];
        let ds = vec![0.0; a_ext.nrows()];
        assert!(!verify_unbounded_ray(
            &dx, &ds, &problem, &a_ext, &is_eq_ext, &scales
        ));
    }

    /// 有限上限からの退行不可: x₂ ≤ 1 に対し dx = e₂ (s₀=任意) は ray が有限境界を
    /// 突き破るため ds = -1 < 0 → 拒否。検出器は境界行の ds 符号を見ないため、
    /// この条件は受理時検証の独立性そのもの。
    #[test]
    fn verify_ray_rejects_direction_exiting_finite_bound() {
        let q = CscMatrix::new(2, 2);
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0, -1.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (f64::NEG_INFINITY, 1.0)],
        )
        .unwrap();
        let (a_ext, _b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dx = vec![0.0, 1.0];
        // 境界行 (x₂ ≤ 1) を dx = e₂ で突っ走る方向。ds 未補償なら残差で落ちる。
        let ds = vec![0.0; a_ext.nrows()];
        assert!(!verify_unbounded_ray(
            &dx, &ds, &problem, &a_ext, &is_eq_ext, &scales
        ));
        // 残差を補償した row-feasible な ds = -1 でも、退行非負性
        // (s(t) = s₀ - t ≥ 0 ∀t が成立しない) に反するため拒否。
        let ds_comp = vec![-1.0; a_ext.nrows()];
        assert!(!verify_unbounded_ray(
            &dx, &ds_comp, &problem, &a_ext, &is_eq_ext, &scales
        ));
    }

    /// 平坦性違反 (Q·dx ≠ 0) は ray でない: min x₁² s.t. x₂ free, c=-e₂ は
    /// x₂ 方向に平坦なので dx=e₂ は通るが、dx に e₁ 成分が混ざれば Q dx ≠ 0。
    #[test]
    fn verify_ray_rejects_nonflat_quadratic_direction() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 2, 2).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0, -1.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
        )
        .unwrap();
        let (a_ext, _b_ext, is_eq_ext, scales) = lp_like(&problem);
        let ds = vec![0.0; a_ext.nrows()];
        assert!(verify_unbounded_ray(
            &[0.0, 1.0],
            &ds,
            &problem,
            &a_ext,
            &is_eq_ext,
            &scales
        ));
        assert!(!verify_unbounded_ray(
            &[0.5, 1.0],
            &ds,
            &problem,
            &a_ext,
            &is_eq_ext,
            &scales
        ));
    }

    /// 真の Farkas 証書: {x ≤ 0, -x ≤ -1} (実行不可能) に y = (1, 1) ≥ 0、
    /// Aᵀy = 0、bᵀy = -1 < 0 が揃う。
    #[test]
    fn verify_farkas_accepts_valid_certificate() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, 1).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0],
            a,
            vec![0.0, -1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
        )
        .unwrap();
        let (a_ext, b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dy = vec![1.0, 1.0];
        assert!(verify_infeasible_farkas(
            &dy, &a_ext, &b_ext, &is_eq_ext, &scales
        ));
    }

    /// 符号条件違反の疑似証書を拒否: dy = (-1, 1) は Aᵀy = 0・bᵀy < 0 を満たして
    /// 見えても Farkas 証書ではない (検出器は符号を検査しない = 独立検証の要点)。
    #[test]
    fn verify_farkas_rejects_negative_multiplier() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, 1).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0],
            a,
            vec![0.0, -1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY)],
        )
        .unwrap();
        let (a_ext, b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dy = vec![-1.0, 1.0];
        assert!(!verify_infeasible_farkas(
            &dy, &a_ext, &b_ext, &is_eq_ext, &scales
        ));
    }
}
