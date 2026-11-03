//! IPM/IP-PMM 共通関数。

use super::kkt::{norm_inf, spmv};
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use otspot_num::linalg::ldl;
use otspot_num::linalg::timeout::TimeoutCtx;
use otspot_num::sparse::CscMatrix;

/// 検出器・証明検査が使う componentwise 後方誤差境界の基準係数。
///
/// 旧実装の絶対定数 `EPS_INF = 1e-8` は ‖c‖∞ ~ 2.5e4 の問題 (QFORPLAN 実測) で
/// fp ノイズレベルの傾斜 (cᵀdx/‖dx‖∞ = -6.9e-8) を「単調減少 ray」と誤認した
/// (2026-11 調査 /tmp/opencode/inv_eps_anom/REPORT.md §1)。一方で global な
/// max(‖c‖, ‖b‖, ‖Q‖行和, ‖A‖行和) 基準のスケールは、検査する残差と**無関係な
/// 大係数**が tolerance を緩め偽証書を通す (PR #66 レビュー反例: Q=diag(1e10,1)
/// の 1e10 が dx=e₂ 成分の平坦性 tol を 100 にする)。そこで各残差を、それが
/// 計算された内積自体の後方誤差境界 `DETECT_REL_EPS * ‖オペランド_i‖₁ * ‖方向‖₁`
/// で成分ごとに判定する (行・列・方向局所)。1e-8 は eps_machine・内積成長率に
/// 対する安全係数として十分大きい。LP 側 Phase I の
/// `farkas_infeasibility_certified` (tol = dual_tol・max(1, ‖b‖∞)) は元来
/// ベクトル 1 本分のスケールであり、この構成と矛盾しない。
const DETECT_REL_EPS: f64 = 1e-8;

/// 内積 `operand_i・dir` の後方誤差境界基準として、行列の行和・列和・ベクトル和を
/// solve 1 回分だけ事前計算するスケールパック (検出器は毎反復 O(nnz) を再計算しない)。
pub(crate) struct DetectionScales {
    /// ‖c‖₁。降下 (cᵀdx) の内積誤差基準。
    c1: f64,
    /// ‖b_ext‖₁。Farkas 厳密性 (b_extᵀy) の内積誤差基準。
    b1: f64,
    /// Q の行絶対値和 ‖Q_i‖₁ (平坦性 ‖Q·dx‖ の成分ごと基準)。
    q_rows: Vec<f64>,
    /// A_ext の行絶対値和 ‖A_ext_i‖₁ (退行実行可能性の行ごと基準)。
    a_rows: Vec<f64>,
    /// A_ext の列絶対値和 ‖A_ext^j‖₁ (停留・検出器 Aᵀ 条件の列ごと基準)。
    a_cols: Vec<f64>,
}

fn l1(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc + x.abs())
}

/// 内積残差の componentwise 許容値: `DETECT_REL_EPS * ‖オペランド_i‖₁ * ‖方向‖₁`。
#[inline]
fn dot_tol(operand_l1: f64, dir_l1: f64) -> f64 {
    DETECT_REL_EPS * operand_l1 * dir_l1
}

/// 符号条件の許容値。値自体に cancellation は無く、方向ベクトルの絶対値和に対する
/// 相対ノイズ + 小問題の床 (1.0) で決める。
#[inline]
fn sign_tol(dir_l1: f64) -> f64 {
    DETECT_REL_EPS * (1.0 + dir_l1)
}

/// CSC 行絶対値和ベクトル (`row_infinity_norms` の各行)。
fn row_abs_sums(m: &CscMatrix) -> Vec<f64> {
    m.row_infinity_norms()
}

/// CSC 列絶対値和ベクトル (CSC は列スライス直接和)。
fn col_abs_sums(m: &CscMatrix) -> Vec<f64> {
    (0..m.ncols())
        .map(|j| {
            m.values()[m.col_ptr()[j]..m.col_ptr()[j + 1]]
                .iter()
                .fold(0.0_f64, |acc, &v| acc + v.abs())
        })
        .collect()
}

impl DetectionScales {
    pub(crate) fn compute(problem: &QpProblem, a_ext: &CscMatrix, b_ext: &[f64]) -> Self {
        Self {
            c1: l1(&problem.c),
            b1: l1(b_ext),
            q_rows: row_abs_sums(&problem.q),
            a_rows: row_abs_sums(a_ext),
            a_cols: col_abs_sums(a_ext),
        }
    }
}

/// ステップ方向 (Δx, Δy) から infeasibility / unboundedness を検出する。
///
/// 閾値は `DetectionScales` 経由の componentwise 後方誤差境界 (DETECT_REL_EPS doc 参照)。
/// 平坦性は**元の Q** で評価する (正則化項 ρ·dx を織り込まない — 平坦性側の注参照)。
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
    scales: &DetectionScales,
) -> Option<SolveStatus> {
    const MIN_ITER: usize = 5;
    const MIN_DIR_NORM: f64 = 1e-3;

    if iter < MIN_ITER {
        return None;
    }

    let n = dx.len();

    if m_orig > 0 {
        let dy_orig = &dy[..m_orig];
        let dy1 = l1(dy_orig);
        if norm_inf(dy_orig) > MIN_DIR_NORM {
            let mut at_dy = vec![0.0f64; n];
            for (j, at_dy_j) in at_dy.iter_mut().enumerate() {
                for ptr in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
                    let row = a_ext.row_ind()[ptr];
                    if row < m_orig {
                        *at_dy_j += a_ext.values()[ptr] * dy_orig[row];
                    }
                }
            }
            // 停留近似: 全列で |Aᵀdy|_j がその列自身の後方誤差境界以内。
            let cond_a = (0..n).all(|j| at_dy[j].abs() <= dot_tol(scales.a_cols[j], dy1));
            let b_dy: f64 = problem
                .b
                .iter()
                .zip(dy_orig.iter())
                .map(|(&bi, &dyi)| bi * dyi)
                .sum();
            let cond_b = b_dy < -dot_tol(scales.b1, dy1);
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
    let dx1 = l1(dx);

    let is_lp = problem.q.values().iter().all(|&v| v == 0.0);
    let cond_obj = if is_lp {
        let c_dx: f64 = problem
            .c
            .iter()
            .zip(dx.iter())
            .map(|(&ci, &dxi)| ci * dxi)
            .sum();
        c_dx < -dot_tol(scales.c1, dx1)
    } else {
        let mut qdx = vec![0.0f64; n];
        spmv(&problem.q, dx, &mut qdx);
        // PR #66 round 2 (P1-3): 平坦性は元の Q で評価 (旧実装は (Q+ρI)·dx を
        // 織り込んでいた)。織り込みは Q 厳密ゼロ行 (componentwise tol =
        // 1e-8×0×‖dx‖₁ = ちょうど 0) に正則化アーティファクト ρ·dx — ρ 床
        // ≥1e-14 で常に非ゼロ — を持ち込み、真の nullspace 方向を検出器が
        // 棄却して検証器に届かない (tests 参照)。生 Q では真の退行行は
        // 正確に 0 なので tol=0 と両立する。
        let c_dx: f64 = problem
            .c
            .iter()
            .zip(dx.iter())
            .map(|(&ci, &dxi)| ci * dxi)
            .sum();
        let flat = (0..n).all(|i| qdx[i].abs() <= dot_tol(scales.q_rows[i], dx1));
        flat && c_dx < -dot_tol(scales.c1, dx1)
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
        if (0..m_orig).any(|i| a_dx[i].abs() >= dot_tol(scales.a_rows[i], dx1)) {
            return None;
        }
    }

    Some(SolveStatus::Unbounded)
}

/// 検出器受理時に求める独立 ray 証明: 回収方向 (dx, ds) の ray
/// `(x + t·dx, s + t·ds)` が **全 t ≥ 0** で拡張系 `{A_ext x + s = b_ext, s ≥ 0}`
/// の実行可能性を保ち、目的関数が線形に減少することを検証する。
///
/// 条件 (すべて componentwise 後方誤差境界、`DETECT_REL_EPS` doc 参照):
/// 1. 平坦性: 全行で `|Q·dx|_i ≤ eps · ‖Q_i‖₁ · ‖dx‖₁` — **delta_p 正則化を
///    含まない生の Q** (正則化項は問題の Q ではなく PMM の足場であり証拠にならない)。
/// 2. 降下 `cᵀdx < -eps · ‖c‖₁ · ‖dx‖₁`。
/// 3. 退行実行可能性: 全 m_ext 行 (境界行を含む) で
///    `|a_ext_i·dx + ds_i| ≤ eps · (‖a_ext_i‖₁·‖dx‖₁ + ‖ds‖₁)`。
/// 4. 退行非負性: s ≥ 0 全行で `ds_i ≥ -sign_tol(‖ds‖₁)` — s(t) が全 t ≥ 0 で ≥ 0。
///
/// 検出器 (`check_infeasible_or_unbounded`) との独立点: 検出器は (a) 境界行を
/// 見ない (rows < m_orig のみ)、(b) ds を構築・検査しない (s(t) の退行非負性が
/// 未検証) — のいずれかが偽陽性の原因 (QFORPLAN 実測: ‖Q·dx‖∞ =
/// 1.0e-14 = fp ノイズ、cᵀdx/‖dx‖∞ = -6.9e-8 なのに ‖c‖∞ ~ 2.5e4)。LP 側
/// simplex Phase I の `farkas_infeasibility_certified` (検証失敗時は判定を撤回する)
/// と同じ契約を QP 側 unboundedness に導入したもの。
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
    if norm_inf(dx) < MIN_DIR_NORM {
        return false;
    }
    let dx1 = l1(dx);
    let ds1 = l1(ds);

    // 1. 平坦性: delta_p なしの Q·dx を行ごとに判定。
    let mut qdx = vec![0.0_f64; n];
    spmv(&problem.q, dx, &mut qdx);
    if (0..n).any(|i| qdx[i].abs() > dot_tol(scales.q_rows[i], dx1)) {
        return false;
    }

    // 2. 降下。
    let c_dx: f64 = problem
        .c
        .iter()
        .zip(dx.iter())
        .map(|(&ci, &dxi)| ci * dxi)
        .sum();
    if c_dx >= -dot_tol(scales.c1, dx1) {
        return false;
    }

    // 3+4. 退行実行可能性 (全行) と退行非負性 (s ≥ 0 行)。
    let mut a_dx = vec![0.0_f64; a_ext.nrows()];
    for (j, &dxj) in dx.iter().enumerate() {
        for ptr in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
            a_dx[a_ext.row_ind()[ptr]] += a_ext.values()[ptr] * dxj;
        }
    }
    let ds_sign_tol = sign_tol(ds1);
    for i in 0..a_ext.nrows() {
        if (a_dx[i] + ds[i]).abs() > dot_tol(scales.a_rows[i], dx1) + DETECT_REL_EPS * ds1 {
            return false;
        }
        if !is_eq_ext[i] && ds[i] < -ds_sign_tol {
            return false;
        }
    }
    true
}

/// 検出器受理時に求める独立 Farkas 証明: 方向 dy を**拡張系全体**
/// `{A_ext x + s = b_ext, s ≥ 0}` に対する Farkas 証書 y として検証する。
///
/// 条件 (componentwise 後方誤差境界):
/// 1. 符号: s ≥ 0 全行 (境界行を含む) で `y_i ≥ -sign_tol(‖y‖₁)`。等式行は自由。
/// 2. 停留: 全列 j で `|A_extᵀy|_j ≤ eps · ‖A_ext^j‖₁ · ‖y‖₁` (列局所 —
///    無関係な大係数列が他列の残差を救わない)。
/// 3. 厳密性: `b_extᵀy < -eps · ‖b_ext‖₁ · ‖y‖₁` — 境界行の寄与を含む全行合計。
///
/// 成立時は任意の実行可能点で bᵀy = yᵀ(Ax+s) = (Aᵀy)ᵀx + Σ yᵢsᵢ ≤ 0 となり、
/// 厳密性と矛盾 → 実行不可能。検出器 (rows < m_orig のみ・符号 unchecked・
/// b_orig のみ) より条件が厳しく、LP 側 `farkas_direction_certified` と同格の
/// 独立検証。
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
    let y1 = l1(dy);

    // 1. 符号: 不等式行 (s ≥ 0) の乗数は非負。
    let dy_sign_tol = sign_tol(y1);
    for (i, &dyi) in dy.iter().enumerate() {
        if !is_eq_ext[i] && dyi < -dy_sign_tol {
            return false;
        }
    }

    // 2. 停留: A_extᵀy ≈ 0 を列ごとに判定。
    for j in 0..a_ext.ncols() {
        let mut aty = 0.0_f64;
        for ptr in a_ext.col_ptr()[j]..a_ext.col_ptr()[j + 1] {
            aty += a_ext.values()[ptr] * dy[a_ext.row_ind()[ptr]];
        }
        if aty.abs() > dot_tol(scales.a_cols[j], y1) {
            return false;
        }
    }

    // 3. 厳密性: b_extᵀy < 0 (境界行寄与込み)。
    let b_dy: f64 = b_ext
        .iter()
        .zip(dy.iter())
        .map(|(&bi, &dyi)| bi * dyi)
        .sum();
    b_dy < -dot_tol(scales.b1, y1)
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

    /// PR #66 レビュー (P1) 反例 a: global max(·‖Q‖行和) 基準では diag(1e10,1) の
    /// 無関係な 1e10 が dx=e₂ の平坦性 tolerance を 100 へ緩め、曲率 1 を持つ
    /// 方向が ray と誤認証される (min 0.5xᵀQx - x₂, x₁=0 は x₂*=1 で有限最小)。
    /// componentwise 行局所基準 (`dot_tol(‖Q_i‖₁, ‖dx‖₁)`) では行 2 の tol が 1e-8
    /// となり却下する。global 基準への revert で本テストは FAIL する sentinel。
    #[test]
    fn verify_ray_rejects_flatness_loosened_by_unrelated_huge_q_row() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e10, 1.0], 2, 2).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0, -1.0],
            CscMatrix::new(0, 2),
            vec![],
            vec![(0.0, 0.0), (f64::NEG_INFINITY, f64::INFINITY)],
        )
        .unwrap();
        let (a_ext, _b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dx = vec![0.0, 1.0];
        let ds = vec![0.0; a_ext.nrows()];
        assert!(
            !verify_unbounded_ray(&dx, &ds, &problem, &a_ext, &is_eq_ext, &scales),
            "column x2 has curvature 1 — e2 is not a recession direction; an unrelated \
             1e10 in row 1 must not loosen row 2's flatness tolerance"
        );
    }

    /// PR #66 レビュー (P1) 反例 b: global max(·‖A‖行和) 基準では diag(1e10,1) の
    /// 大係数が列 2 の停留残差 (Aᵀy)₂ = 1 を tol=100 で通し、実行可能系
    /// {1e10·x₁ ≤ 0, x₂ ≤ -1} (可行点 (0,-1)) が Infeasible と誤認証される。
    /// 列局所基準 (`dot_tol(‖A^j‖₁, ‖y‖₁)`) では列 2 の tol が 1e-8 で却下する。
    #[test]
    fn verify_farkas_rejects_stationarity_loosened_by_unrelated_huge_a_col() {
        let q = CscMatrix::new(2, 2);
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e10, 1.0], 2, 2).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0, 0.0],
            a,
            vec![0.0, -1.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
        )
        .unwrap();
        let (a_ext, b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dy = vec![0.0, 1.0];
        assert!(
            !verify_infeasible_farkas(&dy, &a_ext, &b_ext, &is_eq_ext, &scales),
            "(Aᵀy)₂ = 1 is a real nonzero certificate residual — the unrelated 1e10 \
             in column 1 must not absorb it; (0,-1) is feasible"
        );
    }

    /// PR #66 round 2 review (P1-3) 反例: 検出器の平坦性残差に正則化項 (Q+ρI)·dx を
    /// 織り込むと、Q の厳密ゼロ行では componentwise tol = 1e-8 × ‖Q_i‖₁ × ‖dx‖₁ =
    /// **ちょうど 0** なのに、織り込み項 ρ·dx が IPPMM の ρ 床 (≥1e-14) で常に 0 を
    /// 超え、真の nullspace 方向が検出器を通過して検証器に一切届かない
    /// (旧絶対閾値 1e-8 では ρ<1e-8 で受理されていた)。
    ///
    /// Q=diag(1,0), c=(0,-1)、x₁=0 (両側不等式)、x₂ 自由は真に unbounded
    /// (dx=e₂ が厳密 nullspace の退行方向)。平坦性を元の Q で評価する修正では
    /// 検出器 fire → 検証器 (生 Q 基準) 認証の双方が成立する。
    /// 修正前のビルド (delta_p 織り込み) では検出器 assert が FAIL する sentinel。
    #[test]
    fn verify_ray_accepts_exact_nullspace_direction_under_regularization() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0_f64], 2, 2).unwrap();
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, 2).unwrap();
        let problem = QpProblem::new_all_le(
            q,
            vec![0.0, -1.0],
            a,
            vec![0.0, 0.0],
            vec![(f64::NEG_INFINITY, f64::INFINITY); 2],
        )
        .unwrap();
        let (a_ext, _b_ext, is_eq_ext, scales) = lp_like(&problem);
        let dx = vec![0.0, 1.0];
        let dy = vec![0.0; a_ext.nrows()];
        assert_eq!(
            check_infeasible_or_unbounded(
                &dx,
                &dy,
                &problem,
                &a_ext,
                2,
                a_ext.nrows(),
                10,
                &scales,
            ),
            Some(SolveStatus::Unbounded),
            "e2 spans the exact nullspace of Q — the ρ (>= 1e-14) regularization artifact \
             must not veto it on the zero-Q row whose local tolerance is exactly 0"
        );
        let ds = vec![0.0; a_ext.nrows()];
        assert!(
            verify_unbounded_ray(&dx, &ds, &problem, &a_ext, &is_eq_ext, &scales),
            "verify_unbounded_ray evaluates raw Q and must certify the strict nullspace ray"
        );
    }
}
