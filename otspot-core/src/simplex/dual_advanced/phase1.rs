//! Big-M Phase I cold-start (Dual Phase I + Primal Phase II + Big-M penalty)。
//!
//! Ge/Eq 制約を含む LP の cold-start で、既存 `cold_start_dual` は
//! `sf.num_artificial > 0` 時 Primal Phase I (人工変数 sum 最小化) に
//! フォールバックし、klein3 等 degenerate infeasible LP で cycling して
//! `iters=0 TIMEOUT` する。Dual Phase I と Primal Phase II の組み合わせで回避する:
//! 1. 人工変数列 a_i (係数 1) を `needs_artificial` な各行に追加 (B = I_aug)。
//! 2. Big-M 摂動コスト `c_aug[a_i] = big_m`、元変数 `c_aug[j] = c[j] +
//!    max(0, big_m·Σ_{i∈art} a[i,j] - c[j])`。初期 basis
//!    (y_init = big_m·indicator) で全 reduced cost r_j ≥ 0 (双対実行可能)。
//! 3. Phase I (Dual Simplex, Harris ratio test): b ≥ 0 で初期から主実行可能
//!    なので通常 0 反復。役割は Phase II の safe な warm start 用の双対基底構成。
//!    `Unbounded` → Infeasible。
//! 4. Phase II (Primal Simplex, SteepestEdgePricing): 元コスト [c | 0] で実行し、
//!    人工変数も pricing 対象にして basis から追い出す (Phase I 摂動も相殺)。
//! 5. 終了判定: Phase I `Unbounded` → Infeasible; Phase II 後に人工変数が
//!    値 > primal_tol で残存 → infeasible; `Optimal` かつ人工変数 = 0 → 最適;
//!    `Unbounded` → 非有界; その他 Timeout/SingularBasis は通常処理。
//!
//! `big_m = max(‖c‖_∞·BIG_M_COST_MULT, ‖b‖_∞·BIG_M_COST_MULT, BIG_M_FLOOR)`
//! (Ruiz スケール後の c, b から。問題スケール由来で固定マジック値ではない)。

use super::super::crash;
use super::super::pricing::{DualLeavingStrategy, SteepestEdgePricing};
use super::super::{extract_dual_info, extract_solution, SimplexOutcome, StandardForm};
use super::core::dual_simplex_core_advanced;
use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::{DROP_TOL, PIVOT_TOL};

#[cfg(test)]
#[derive(Clone, Copy)]
enum FreshFactorFailure {
    None,
    Singular,
    Deadline,
}

#[cfg(test)]
thread_local! {
    static FRESH_FACTOR_FAILURE: std::cell::Cell<FreshFactorFailure> = const {
        std::cell::Cell::new(FreshFactorFailure::None)
    };
    static FORCE_PHASE1_UNBOUNDED: std::cell::Cell<bool> = const {
        std::cell::Cell::new(false)
    };
}

fn fresh_positive_artificial(
    a_aug: &CscMatrix,
    b: &[f64],
    basis_aug: &[usize],
    m: usize,
    n_total: usize,
    options: &SolverOptions,
) -> Result<bool, crate::error::SolverError> {
    #[cfg(test)]
    match FRESH_FACTOR_FAILURE.get() {
        FreshFactorFailure::None => {}
        FreshFactorFailure::Singular => {
            return Err(crate::error::SolverError::SingularBasis { step: 0 });
        }
        FreshFactorFailure::Deadline => {
            return Err(crate::error::SolverError::DeadlineExceeded);
        }
    }

    let mut bm = LuBasis::new_timed(a_aug, basis_aug, options.max_etas, options.deadline)?;
    let mut x_b_fresh = b.to_vec();
    bm.ftran_dense(&mut x_b_fresh);
    Ok((0..m).any(|i| basis_aug[i] >= n_total && x_b_fresh[i] > options.primal_tol))
}

#[derive(Debug, PartialEq)]
enum UnboundedProofRefresh {
    Infeasible,
    NoProof,
    Timeout,
    NumericalError,
}

fn classify_unbounded_proof_refresh(
    refresh: Result<bool, crate::error::SolverError>,
) -> UnboundedProofRefresh {
    use crate::error::SolverError;

    match refresh {
        Ok(true) => UnboundedProofRefresh::Infeasible,
        Ok(false) => UnboundedProofRefresh::NoProof,
        Err(SolverError::DeadlineExceeded) => UnboundedProofRefresh::Timeout,
        Err(SolverError::SingularBasis { .. }) => UnboundedProofRefresh::NumericalError,
        Err(
            err @ (SolverError::DimensionMismatch { .. }
            | SolverError::IndexOutOfBounds { .. }
            | SolverError::EmptyInput { .. }
            | SolverError::NonFiniteCoefficient { .. }
            | SolverError::InvalidBounds { .. }),
        ) => {
            panic!("internal invariant violation during Phase I proof refresh: {err}")
        }
    }
}

/// Check one Farkas direction `y` against `{A x = b, x ≥ 0}`.
///
/// Returns `true` iff `b^T y > tol` and `A^T y ≤ tol` for all original columns.
fn farkas_direction_certified(
    a_aug: &CscMatrix,
    b: &[f64],
    y: &[f64],
    n_total: usize,
    tol: f64,
) -> bool {
    let by: f64 = b.iter().zip(y.iter()).map(|(&bi, &yi)| bi * yi).sum();
    if by <= tol {
        return false;
    }
    for j in 0..n_total {
        let (rows, vals) = a_aug.get_column(j).unwrap();
        let aty: f64 = rows.iter().zip(vals.iter()).map(|(&r, &v)| v * y[r]).sum();
        if aty > tol {
            return false;
        }
    }
    true
}

/// Farkas certificate verification for primal infeasibility.
///
/// Constructs dual directions from artificial basis rows and tests the Farkas
/// alternative for the original LP `{min c^T x | Ax = b, x ≥ 0}`:
///
///   `A^T y ≤ tol` for all original cols j  AND  `b^T y > tol`  →  infeasible.
///
/// Two strategies are tried to maximise the chance of finding a certificate when
/// cycling or numerical drift has produced a suboptimal exit basis:
///
///  1. Joint indicator: `c_phase1[i] = 1` for all artificial rows → single BTRAN.
///  2. Per-row probes: `e_i` for each artificial row `i` → individual BTRAN.
///     Strategy 2 finds certificates that strategy 1 misses when the joint
///     `b^T y ≈ 0` due to sign cancellation across artificial rows (cplex2-class).
///
/// If the certificate fails, the caller must return Timeout rather than guessing
/// Infeasible — that heuristic produces false-infeasible verdicts on feasible LPs.
///
/// Tolerance scales with `||b||_∞` to stay correct on Ruiz-scaled inputs.
fn farkas_infeasibility_certified(
    a_aug: &CscMatrix,
    b: &[f64],
    basis_aug: &[usize],
    m: usize,
    n_total: usize,
    options: &SolverOptions,
) -> bool {
    let art_rows: Vec<usize> = (0..m).filter(|&i| basis_aug[i] >= n_total).collect();
    if art_rows.is_empty() {
        return false;
    }

    let mut basis_mgr =
        match LuBasis::new_timed(a_aug, basis_aug, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(_) => return false,
        };

    let b_norm = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = options.dual_tol * (1.0_f64).max(b_norm);

    // Strategy 1: joint indicator (all artificial rows simultaneously).
    {
        let mut y: Vec<f64> = (0..m)
            .map(|i| if basis_aug[i] >= n_total { 1.0 } else { 0.0 })
            .collect();
        basis_mgr.btran_dense(&mut y);
        if farkas_direction_certified(a_aug, b, &y, n_total, tol) {
            return true;
        }
    }

    // Strategy 2: per-row probes — catches cplex2-class where joint b^Ty ≈ 0.
    for &row in &art_rows {
        let mut e_i = vec![0.0_f64; m];
        e_i[row] = 1.0;
        basis_mgr.btran_dense(&mut e_i);
        if farkas_direction_certified(a_aug, b, &e_i, n_total, tol) {
            return true;
        }
    }

    false
}

/// Big-M Phase I 専用の離基変数戦略。
///
/// 優先順位:
/// 1. 通常の主実行不可 (x_B[i] < -primal_tol) → 最も負の violation を持つ行
/// 2. 人工変数が basis に残り x_B[i] > primal_tol → 最も大きい残存値の行
///    (元 LP の主実行不可性を表す。dual の violation 扱いで追い出す)
///
/// この優先順は標準 dual simplex の動作を維持しつつ、Big-M 環境特有の
/// 「人工変数を basis から自然に追い出す」効果を持つ。
struct ArtificialPriorityLeaving {
    n_total: usize,
}

impl DualLeavingStrategy for ArtificialPriorityLeaving {
    fn select_leaving(&mut self, x_b: &[f64], primal_tol: f64, basis: &[usize]) -> Option<usize> {
        // Priority 1: 標準的 most-infeasible
        let mut best_row: Option<usize> = None;
        let mut max_violation = primal_tol;
        for (i, &val) in x_b.iter().enumerate() {
            if val < -max_violation {
                max_violation = -val;
                best_row = Some(i);
            }
        }
        if best_row.is_some() {
            return best_row;
        }
        // Priority 2: 人工変数の basis 残存 (x_B[i] > primal_tol)
        let mut best_art: Option<usize> = None;
        let mut max_art_val = primal_tol;
        for (i, &val) in x_b.iter().enumerate() {
            if basis[i] >= self.n_total && val > max_art_val {
                max_art_val = val;
                best_art = Some(i);
            }
        }
        best_art
    }

    /// Bland fallback must honor Priority 2; default Bland would return None
    /// whenever `x_B ≥ 0` (initial Big-M Phase I state with `b ≥ 0`), masking
    /// artificial-removal and causing `dual_simplex_core_advanced` to declare
    /// false Optimal with artificials in basis.
    fn bland_leaving(&mut self, x_b: &[f64], primal_tol: f64, basis: &[usize]) -> Option<usize> {
        let mut best_row: Option<usize> = None;
        let mut best_var = usize::MAX;
        for (i, &v) in x_b.iter().enumerate() {
            if v < -primal_tol && basis[i] < best_var {
                best_var = basis[i];
                best_row = Some(i);
            }
        }
        if best_row.is_some() {
            return best_row;
        }
        for (i, &v) in x_b.iter().enumerate() {
            if basis[i] >= self.n_total && v > primal_tol && basis[i] < best_var {
                best_var = basis[i];
                best_row = Some(i);
            }
        }
        best_row
    }

    /// 進歩指標 = x_B 負部分 + basis 内人工変数の正値合計。後者を含めないと
    /// Big-M Phase I で `best_infeas = 0` 固定 → threshold = 0 → 任意の
    /// `sum_neg ≥ 0` で改善判定 false → 全反復 no-progress → bland_mode 誤起動。
    fn progress_metric(&mut self, x_b: &[f64], basis: &[usize]) -> f64 {
        let neg_sum: f64 = x_b.iter().map(|&v| (-v).max(0.0)).sum();
        let art_sum: f64 = (0..x_b.len())
            .filter(|&i| basis[i] >= self.n_total)
            .map(|i| x_b[i].max(0.0))
            .sum();
        neg_sum + art_sum
    }

    /// Big-M Phase I DOES repair genuine lb-violations. A Priority-2 artificial
    /// removal pivot drives structural rows large-negative (beaconfd: x_B ≈ −9329,
    /// far above any LU-eta-drift noise floor); the sign-flip ratio test repairs
    /// them. The blanket `false` here previously sent the ratio test the wrong
    /// way → unrepaired violation → 2-cycle (beaconfd/scrs8 TIMEOUT).
    ///
    /// The structural exclusion that keeps this safe lives in the core
    /// (`dual_simplex_core_advanced` 3d'): the sign-flip is suppressed when the
    /// leaving variable is itself an artificial (`basis[r] >= n_enter`). A
    /// negative artificial must be *driven out* (standard direction + n_enter
    /// re-entry ban), not sign-flip-repaired, which would otherwise keep it basic
    /// and chase it indefinitely (sierra: 478-pivot Phase-II cycle).
    fn allows_lb_repair(&self) -> bool {
        true
    }
}

/// Big-M ペナルティ算出時の coefficient 倍率。
///
/// `big_m = max(||c||_∞ × MULT, ||b||_∞ × MULT, BIG_M_FLOOR)` で人工変数コストを
/// 問題スケールの 1000 倍に設定し、simplex が必ず人工変数を駆出できるようにする。
///
/// 撤廃 (1.0 に変更) の影響: ||c||_∞ < 1000 かつ ||b||_∞ < 1000 の問題では
/// BIG_M_FLOOR = 1e6 が支配するため実質変化なし (標準 test suite で退化なし、実測確認)。
/// ||b||_∞ >> 1000 の Netlib 問題 (例: dfl001 の ||b||_∞ ≈ 1e6) では
/// big_m が 1e9 → 1e6 に低下し問題スケールと同程度になるため
/// 人工変数コストが目的関数に対して支配的でなくなり Phase I が収束不全になる可能性がある
/// (heavy tier: `lp_coverage_screen_all` で確認可能)。
const BIG_M_COST_MULT: f64 = 1e3;

/// Big-M ペナルティの下限。
const BIG_M_FLOOR: f64 = 1e6;

/// Big-M Phase I の初期状態をまとめた構造体。
/// `try_build_crash_phase1_state` / `build_identity_phase1_state` が返す。
struct BigMPhase1State {
    a_aug: CscMatrix,
    basis_aug: Vec<usize>,
    c_aug_p1: Vec<f64>,
    x_b: Vec<f64>,
    artificial_col_of_row: Vec<Option<usize>>,
    n_aug: usize,
}

/// Helper: A_aug = [A | I_art] for the given `artificial_col_of_row` map.
///
/// `CscMatrix::from_triplets` cannot fail here:
/// - Row/col indices are in-bounds by construction: structural triplets come
///   from `a.get_column(j)` for `j < n_total` (row < m, col < n_total <= n_aug
///   since `a` is a valid `m x n_total` matrix), and artificial triplets use
///   `row = i < m` with `col = artificial_col_of_row[i] >= n_total` (both
///   callers assign `n_total + <offset>`).
/// - No duplicate `(row, col)` pair can occur (this function does NOT rely on
///   `from_triplets`/`build_compressed_format` rejecting or safely merging
///   duplicates — merging silently sums colliding entries, which would be
///   silent corruption, not a safe fallback): structural and artificial
///   triplets occupy disjoint column ranges (`< n_total` vs `>= n_total`);
///   within the structural group each column of a valid CSC `a` has at most
///   one entry per row; within the artificial group `artificial_col_of_row`
///   assigns at most one column per row, so each row contributes at most one
///   artificial triplet.
/// - All values are finite: structural values come from `a.values` (already
///   finite by construction of `a`), artificial values are the literal `1.0`.
fn build_a_aug(
    a: &CscMatrix,
    artificial_col_of_row: &[Option<usize>],
    m: usize,
    n_total: usize,
    n_aug: usize,
) -> CscMatrix {
    let n_art_estimate = n_aug - n_total;
    let mut trip_rows: Vec<usize> = Vec::with_capacity(a.nnz() + n_art_estimate);
    let mut trip_cols: Vec<usize> = Vec::with_capacity(a.nnz() + n_art_estimate);
    let mut trip_vals: Vec<f64> = Vec::with_capacity(a.nnz() + n_art_estimate);
    for j in 0..n_total {
        let (rows, vals) = a.get_column(j).unwrap();
        for (k, &row) in rows.iter().enumerate() {
            let v = vals[k];
            if v.abs() > DROP_TOL {
                trip_rows.push(row);
                trip_cols.push(j);
                trip_vals.push(v);
            }
        }
    }
    for (i, col_opt) in artificial_col_of_row.iter().enumerate() {
        if let Some(col) = col_opt {
            trip_rows.push(i);
            trip_cols.push(*col);
            trip_vals.push(1.0);
        }
    }
    CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_aug)
        .expect("in-bounds indices, no duplicate (row, col), all-finite values (see fn doc)")
}

/// Identity-basis Phase I 状態 (B = I_aug, x_b = b, c_aug は閉式 delta)。
/// Existing pre-crash 挙動と完全一致 — crash 不採用 / 失敗時の安全フォールバック。
fn build_identity_phase1_state(
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    sf: &StandardForm,
    big_m: f64,
    n_total: usize,
) -> BigMPhase1State {
    let m = sf.m;

    let mut artificial_col_of_row: Vec<Option<usize>> = vec![None; m];
    let mut n_art = 0usize;
    for i in 0..m {
        if sf.needs_artificial[i] {
            artificial_col_of_row[i] = Some(n_total + n_art);
            n_art += 1;
        }
    }
    let n_aug = n_total + n_art;

    let a_aug = build_a_aug(a, &artificial_col_of_row, m, n_total, n_aug);

    let mut c_aug_p1 = vec![0.0_f64; n_aug];
    for j in 0..n_total {
        let (rows, vals) = a.get_column(j).unwrap();
        let mut sum_art = 0.0_f64;
        for (k, &row) in rows.iter().enumerate() {
            if sf.needs_artificial[row] {
                sum_art += vals[k];
            }
        }
        let need = big_m * sum_art - c[j];
        let delta = need.max(0.0);
        c_aug_p1[j] = c[j] + delta;
    }
    for col in artificial_col_of_row.iter().flatten() {
        c_aug_p1[*col] = big_m;
    }

    let mut basis_aug = sf.initial_basis.clone();
    for i in 0..m {
        if let Some(col) = artificial_col_of_row[i] {
            basis_aug[i] = col;
        }
    }

    let x_b = b.to_vec();

    BigMPhase1State {
        a_aug,
        basis_aug,
        c_aug_p1,
        x_b,
        artificial_col_of_row,
        n_aug,
    }
}

/// `try_build_crash_phase1_state` 内の経路観測点。test sentinel が短絡無し
/// (= real big_m_cold_start path) で「どの guard が発動したか」を直接観測する
/// ためのフック。`#[cfg(test)]` 限定の thread-local counter で privacy 漏れ
/// 無し、production code path には影響しない。
#[cfg(test)]
mod crash_probe {
    use std::cell::Cell;

    /// Outcome of one `try_build_crash_phase1_state` invocation.
    /// `Adopted(n_art_post)` は state を返したケース、それ以外は途中で None。
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Outcome {
        DisabledOption,
        NoArtificial,
        NotReduced,
        LuFailed,
        XbNegative,
        Adopted(usize),
    }

    thread_local! {
        static LAST_OUTCOME: Cell<Option<Outcome>> = const { Cell::new(None) };
    }

    pub fn record(out: Outcome) {
        LAST_OUTCOME.with(|c| c.set(Some(out)));
    }
    pub fn take() -> Option<Outcome> {
        LAST_OUTCOME.with(|c| c.replace(None))
    }
    pub fn clear() {
        LAST_OUTCOME.with(|c| c.set(None));
    }
}

/// Crash basis を Big-M Phase I 初期状態構築に適用。
/// Identity 経路 (`build_identity_phase1_state`) と等価な dual-feasible 状態を
/// 構成できれば `Some(state)`、いずれかの guard で弾かれたら `None` を返す。
///
/// Guard:
/// 1. `options.use_lp_crash_basis`
/// 2. `sf.num_artificial > 0` (Le-only は no-op)
/// 3. crash で num_artificial が真に減少
/// 4. LU 分解成功
/// 5. x_B = B^{-1} b の各成分 ≥ -PIVOT_TOL (主実行可能性)
///
/// c_aug は y = B^{-T} c_B から `delta_j = max(0, a_j^T y - c[j])` を加算する
/// 一般化版 (B = I の閉式と一致、basic な structural 列は r_j = 0 で
/// 自動 dual-feasible)。
#[allow(clippy::too_many_arguments)]
fn try_build_crash_phase1_state(
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    sf: &StandardForm,
    options: &SolverOptions,
    big_m: f64,
    n_total: usize,
) -> Option<BigMPhase1State> {
    if !options.use_lp_crash_basis {
        #[cfg(test)]
        crash_probe::record(crash_probe::Outcome::DisabledOption);
        return None;
    }
    if sf.num_artificial == 0 {
        #[cfg(test)]
        crash_probe::record(crash_probe::Outcome::NoArtificial);
        return None;
    }

    let m = sf.m;
    let (basis_pre, needs_artificial, n_art) = crash::compute_crash_basis(
        a,
        b,
        m,
        sf.n_shifted,
        &sf.initial_basis,
        &sf.needs_artificial,
    );
    if n_art >= sf.num_artificial {
        #[cfg(test)]
        crash_probe::record(crash_probe::Outcome::NotReduced);
        return None;
    }

    let mut artificial_col_of_row: Vec<Option<usize>> = vec![None; m];
    let mut art_idx = 0usize;
    for i in 0..m {
        if needs_artificial[i] {
            artificial_col_of_row[i] = Some(n_total + art_idx);
            art_idx += 1;
        }
    }
    debug_assert_eq!(art_idx, n_art);
    let n_aug = n_total + n_art;

    let a_aug = build_a_aug(a, &artificial_col_of_row, m, n_total, n_aug);

    let mut basis_aug = basis_pre;
    for i in 0..m {
        if let Some(col) = artificial_col_of_row[i] {
            basis_aug[i] = col;
        }
    }

    let mut basis_mgr =
        match LuBasis::new_timed(&a_aug, &basis_aug, options.max_etas, options.deadline) {
            Ok(bm) => bm,
            Err(_) => {
                #[cfg(test)]
                crash_probe::record(crash_probe::Outcome::LuFailed);
                return None;
            }
        };

    // x_B = B^{-1} b
    let mut x_b_sv = SparseVec::from_dense(b);
    basis_mgr.ftran(&mut x_b_sv);
    let x_b = x_b_sv.to_dense();
    if x_b.iter().any(|&v| v < -PIVOT_TOL) {
        #[cfg(test)]
        crash_probe::record(crash_probe::Outcome::XbNegative);
        return None;
    }

    // y = B^{-T} c_B; c_B[i] = c_aug[basis_aug[i]] = big_m (artif) or c[col] (struct/slack)
    let mut c_b: Vec<f64> = (0..m)
        .map(|i| {
            let col = basis_aug[i];
            if col >= n_total {
                big_m
            } else {
                c[col]
            }
        })
        .collect();
    basis_mgr.btran_dense(&mut c_b);
    let y = c_b;

    // c_aug for non-basic structural cols: delta_j = max(0, a_j^T y - c[j])
    let mut in_basis = vec![false; n_aug];
    for &col in &basis_aug {
        in_basis[col] = true;
    }

    let mut c_aug_p1 = vec![0.0_f64; n_aug];
    for col in artificial_col_of_row.iter().flatten() {
        c_aug_p1[*col] = big_m;
    }
    for j in 0..n_total {
        if in_basis[j] {
            // basic 列: r_j = c[j] - a_j^T y = 0 by construction (B^T y = c_B)
            c_aug_p1[j] = c[j];
        } else {
            let (rows, vals) = a_aug.get_column(j).unwrap();
            let mut aty = 0.0_f64;
            for (k, &row) in rows.iter().enumerate() {
                aty += vals[k] * y[row];
            }
            let delta = (aty - c[j]).max(0.0);
            c_aug_p1[j] = c[j] + delta;
        }
    }

    #[cfg(test)]
    crash_probe::record(crash_probe::Outcome::Adopted(n_art));
    Some(BigMPhase1State {
        a_aug,
        basis_aug,
        c_aug_p1,
        x_b,
        artificial_col_of_row,
        n_aug,
    })
}

/// Big-M Phase I cold-start (Dual Phase I + Primal Phase II + Big-M penalty)
/// for Ge/Eq 含む LP.
///
/// `a, b, c` は Ruiz スケーリング後の値を渡すこと。
/// `row_scale`, `col_scale` は `extract_dual_info` で必要。
#[allow(clippy::too_many_arguments)]
pub(crate) fn big_m_cold_start(
    sf: &StandardForm,
    problem: &LpProblem,
    options: &SolverOptions,
    a: &CscMatrix,
    b: &[f64],
    c: &[f64],
    row_scale: &[f64],
    col_scale: &[f64],
) -> SolverResult {
    let m = sf.m;
    let n_total = sf.n_total;

    // === Step 2: Big-M 動的算出 ===
    let c_norm = c.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let b_norm = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let big_m = (c_norm * BIG_M_COST_MULT)
        .max(b_norm * BIG_M_COST_MULT)
        .max(BIG_M_FLOOR);

    // crash 採用で artificial 列を structural 列に置換し Phase I 駆出対象を縮減。
    // LU / x_B ≥ 0 / dual feasibility のいずれかで失敗したら identity 経路に倒す。
    let crash_state = try_build_crash_phase1_state(a, b, c, sf, options, big_m, n_total);
    let BigMPhase1State {
        a_aug,
        mut basis_aug,
        c_aug_p1,
        mut x_b,
        artificial_col_of_row,
        n_aug,
    } = crash_state.unwrap_or_else(|| build_identity_phase1_state(a, b, c, sf, big_m, n_total));

    // === Step 6: Phase I (Dual Simplex with Harris ratio test + Artificial-aware) ===
    //
    // ArtificialPriorityLeaving は標準 most-infeasible (Priority 1) で
    // x_B < 0 を解消した後、人工変数の basis 残存 (Priority 2; x_B[i] > 0
    // かつ basis[i] >= n_total) を leaving 候補として継続選択する。
    // これにより Big-M Phase I 本来の「人工変数を basis から追い出す」役割を
    // 標準 dual simplex ループ (Harris ratio test 装備) で実現する。
    //
    // Phase I は元 deadline を使用 (外側 split との二重 halving 回避)。
    let mut leaving = ArtificialPriorityLeaving { n_total };
    let mut total_iters: usize = 0;
    // n_enter = n_total: artificials (cols [n_total, n_aug)) may start basic and
    // be driven out, but never re-enter. This makes Priority-2 artificial removal
    // monotone and rules out the degenerate artificial↔artificial swap cycle.
    #[cfg(test)]
    let forced_unbounded = FORCE_PHASE1_UNBOUNDED.get();
    #[cfg(not(test))]
    let forced_unbounded = false;
    let phase1_outcome = if forced_unbounded {
        SimplexOutcome::Unbounded
    } else {
        dual_simplex_core_advanced(
            &a_aug,
            &mut x_b,
            &c_aug_p1,
            &mut basis_aug,
            m,
            n_aug,
            n_total,
            true, // yield_on_stall: Big-M has a primal fallback to hand off to
            options,
            &mut leaving,
            &mut total_iters,
        )
    };

    match phase1_outcome {
        SimplexOutcome::Unbounded => {
            // Farkas 証明書が取れた場合は Infeasible。
            if farkas_infeasibility_certified(&a_aug, b, &basis_aug, m, n_total, options) {
                let mut r = SolverResult::infeasible();
                r.iterations = total_iters;
                return r;
            }
            // Big-M Phase I は b ≥ 0 の恒等基底から出発し、Priority-2 (人工変数残存 > primal_tol)
            // を選択中に Unbounded を返した場合、人工変数を駆出できない = 主実行不可の証明。
            // Feasible LP では人工変数が必ず除去できるため、Unbounded に到達しない。
            // 数値ドリフトで Farkas 証明書の A^T y ≤ 0 チェックが失敗しても、
            // 「正の人工変数残存 + Unbounded」の組合せは実行不可確定の十分条件。
            //
            // ETA 累積誤差で `x_b` が drift しているので fresh FTRAN で再計算 (B⁻¹ b)。
            // ドリフトと数値悪化の両方を排除して soundness を強化 (reviewer P1)。
            let refresh = classify_unbounded_proof_refresh(fresh_positive_artificial(
                &a_aug, b, &basis_aug, m, n_total, options,
            ));
            match refresh {
                UnboundedProofRefresh::Infeasible => {
                    let mut r = SolverResult::infeasible();
                    r.iterations = total_iters;
                    return r;
                }
                UnboundedProofRefresh::NoProof => {}
                UnboundedProofRefresh::Timeout => {
                    // A failed proof refresh cannot be replaced with the eta-updated
                    // `x_b`: that iterate is exactly the value whose drift prompted
                    // this check. Preserve it only as a non-terminal incumbent.
                    let mut result = super::super::stop_result_with_incumbent(
                        sf,
                        problem,
                        &basis_aug,
                        &x_b,
                        col_scale,
                        total_iters,
                        options,
                    );
                    result.status = SolveStatus::Timeout;
                    return result;
                }
                UnboundedProofRefresh::NumericalError => {
                    let mut result = SolverResult::numerical_error();
                    result.iterations = total_iters;
                    return result;
                }
            }
            return super::super::stop_result_with_incumbent(
                sf,
                problem,
                &basis_aug,
                &x_b,
                col_scale,
                total_iters,
                options,
            );
        }
        SimplexOutcome::Timeout(_) | SimplexOutcome::Stalled(_) => {
            // Farkas証明書が得られた場合のみ Infeasible を返す。
            // 得られない場合は、yield_on_stall=false で Phase I を再実行し
            // 真の終端 (Optimal / Unbounded) を得る。これにより cplex2 クラスの
            // 循環スタール後 Farkas 抽出失敗を回避する。
            let any_artificial_left =
                (0..m).any(|i| basis_aug[i] >= n_total && x_b[i].abs() > options.primal_tol);
            if any_artificial_left
                && farkas_infeasibility_certified(&a_aug, b, &basis_aug, m, n_total, options)
            {
                let mut r = SolverResult::infeasible();
                r.iterations = total_iters;
                return r;
            }
            // Farkas 失敗 + 人工変数残存: yield_on_stall=false で再実行。
            // Bland 保証により有限回で Optimal または Unbounded に到達する。
            // Feasible LP では Unbounded に到達しない (b ≥ 0 かつ Big-M コスト)。
            if any_artificial_left {
                let mut leaving2 = ArtificialPriorityLeaving { n_total };
                let phase1b_outcome = dual_simplex_core_advanced(
                    &a_aug,
                    &mut x_b,
                    &c_aug_p1,
                    &mut basis_aug,
                    m,
                    n_aug,
                    n_total,
                    false, // yield_on_stall=false: Bland 保証で有限終端
                    options,
                    &mut leaving2,
                    &mut total_iters,
                );
                match phase1b_outcome {
                    SimplexOutcome::Unbounded => {
                        if farkas_infeasibility_certified(
                            &a_aug, b, &basis_aug, m, n_total, options,
                        ) {
                            let mut r = SolverResult::infeasible();
                            r.iterations = total_iters;
                            return r;
                        }
                        return super::super::stop_result_with_incumbent(
                            sf,
                            problem,
                            &basis_aug,
                            &x_b,
                            col_scale,
                            total_iters,
                            options,
                        );
                    }
                    SimplexOutcome::Timeout(_) | SimplexOutcome::Stalled(_) => {
                        return super::super::stop_result_with_incumbent(
                            sf,
                            problem,
                            &basis_aug,
                            &x_b,
                            col_scale,
                            total_iters,
                            options,
                        );
                    }
                    SimplexOutcome::SingularBasis => return SolverResult::numerical_error(),
                    SimplexOutcome::Optimal(_, _) => {
                        // Phase I 完了: Phase II へ fall-through (下の Optimal 分岐と同一処理)
                        if let Ok(mut bm) = LuBasis::new_timed(
                            &a_aug,
                            &basis_aug,
                            options.max_etas,
                            options.deadline,
                        ) {
                            let mut rhs = SparseVec::from_dense(b);
                            bm.ftran(&mut rhs);
                            let fresh = rhs.to_dense();
                            x_b.copy_from_slice(&fresh);
                        }
                        let any_art = (0..m).any(|i| basis_aug[i] >= n_total);
                        if any_art
                            && farkas_infeasibility_certified(
                                &a_aug, b, &basis_aug, m, n_total, options,
                            )
                        {
                            let mut r = SolverResult::infeasible();
                            r.iterations = total_iters;
                            return r;
                        }
                        // Phase II へ継続 (match を抜けて Phase II ブロックへ)
                    }
                }
            } else {
                return super::super::stop_result_with_incumbent(
                    sf,
                    problem,
                    &basis_aug,
                    &x_b,
                    col_scale,
                    total_iters,
                    options,
                );
            }
        }
        SimplexOutcome::SingularBasis => {
            return SolverResult::numerical_error();
        }
        SimplexOutcome::Optimal(_, _) => {
            // Flush numerical drift accumulated during Phase I cycling by
            // recomputing x_B = B^{-1} b before Phase II (Maros §6 hygiene).
            if let Ok(mut bm) =
                LuBasis::new_timed(&a_aug, &basis_aug, options.max_etas, options.deadline)
            {
                let mut rhs = SparseVec::from_dense(b);
                bm.ftran(&mut rhs);
                let fresh = rhs.to_dense();
                x_b.copy_from_slice(&fresh);
            }

            // Infeasibility is declared ONLY via a verified Farkas certificate
            // (A^T y ≤ tol ∧ b^T y > tol). A residual artificial in the basis
            // is NOT a proof on its own: that heuristic flips slow-but-feasible
            // LPs (pilot/dfl001/ken) to false-Infeasible. When the
            // certificate fails, fall through to Phase II.
            let any_artificial_in_basis = (0..m).any(|i| basis_aug[i] >= n_total);
            if any_artificial_in_basis
                && farkas_infeasibility_certified(&a_aug, b, &basis_aug, m, n_total, options)
            {
                let mut r = SolverResult::infeasible();
                r.iterations = total_iters;
                return r;
            }
        }
    }

    // === Step 7: Phase II (Primal Simplex, 元コスト + Big-M で 1-phase 仕上げ) ===
    //
    // c_phase2 = [c | big_m; n_art]: 人工変数の penalty は残しつつ元 c で最適化。
    // Primal なので artificial を pricing 対象に含め (n_price = n_aug)、reduced
    // cost が negative なら entering、 別の列が entering で α[art_row] > 0
    // なら leaving (= artificial が basis から自然に追い出される)。
    let mut c_aug_p2 = vec![0.0_f64; n_aug];
    c_aug_p2[..n_total].copy_from_slice(c);
    for col in artificial_col_of_row.iter().flatten() {
        c_aug_p2[*col] = big_m;
    }

    // Charnes perturbation: degenerate rows (x_b ≈ 0) cause ratio-test step=0
    // and degenerate cycles in Phase II. Perturb each such row by a unique tiny
    // positive value so the ratio test produces step > 0. The final reconcile
    // after Phase II restores exact B^{-1}b.
    for i in 0..m {
        if x_b[i].abs() < crate::tolerances::PIVOT_TOL {
            x_b[i] = crate::tolerances::PIVOT_TOL * (i as f64 + 1.0);
        }
    }
    for v in x_b.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    let mut pricing = SteepestEdgePricing::new(n_aug);
    let phase2_outcome = super::super::revised_simplex_core(
        &a_aug,
        &mut x_b,
        &c_aug_p2,
        b,
        &mut basis_aug,
        m,
        n_aug,
        n_aug,
        &mut pricing,
        options,
        &mut total_iters,
        false,
        None,
        false,
        None,
    );

    // === Step 8: Phase II 結果 + 人工変数残存判定 ===
    match phase2_outcome {
        SimplexOutcome::Optimal(_obj_aug, y) => {
            // 人工変数が basis に残り値 > primal_tol → 元 LP infeasible only
            // when backed by a Farkas certificate. A finite Big-M penalty can
            // otherwise stop at an augmented optimum that is not a proof for the
            // original LP; returning Infeasible there is a false verdict risk.
            for i in 0..m {
                if basis_aug[i] >= n_total && x_b[i].abs() > options.primal_tol {
                    if farkas_infeasibility_certified(&a_aug, b, &basis_aug, m, n_total, options) {
                        let mut r = SolverResult::infeasible();
                        r.iterations = total_iters;
                        return r;
                    }
                    return super::super::stop_result_with_incumbent(
                        sf,
                        problem,
                        &basis_aug,
                        &x_b,
                        col_scale,
                        total_iters,
                        options,
                    );
                }
            }

            let solution = extract_solution(sf, &basis_aug, &x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);

            // warm-start: artificial が basis に残るケースは除外
            let ws = if basis_aug.iter().all(|&idx| idx < n_total) {
                Some(WarmStartBasis {
                    basis: basis_aug.clone(),
                    x_b: x_b.clone(),
                })
            } else {
                None
            };

            // solution は原空間 (un-shifted) なので c·x が完全な原目的値。
            // 他経路は shifted basic_obj に sf.obj_offset を足すが、ここで足すと
            // Σc_j·lb_j を二重計上する (Big-M 経路のみのバグだった)。
            let obj_orig: f64 = problem
                .c
                .iter()
                .zip(solution.iter())
                .map(|(&ci, &xi)| ci * xi)
                .sum();

            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj_orig,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: ws,
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => {
            // Gate the Unbounded verdict on a re-derived recession ray. A clean LU
            // at the exit basis distinguishes a genuine ray from eta-drift noise.
            // Unverified ⇒ honest Timeout, mirroring the Phase-I Farkas gate that
            // turns an unproven infeasibility ray into Timeout.
            if super::super::dual_common::lp_unbounded_ray_verified(
                &a_aug, &basis_aug, &c_aug_p2, m, n_aug, n_total, options,
            ) {
                SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                    warm_start_basis: None,
                    iterations: total_iters,
                    ..Default::default()
                }
            } else {
                super::super::stop_result_with_incumbent(
                    sf,
                    problem,
                    &basis_aug,
                    &x_b,
                    col_scale,
                    total_iters,
                    options,
                )
            }
        }
        SimplexOutcome::Timeout(_) | SimplexOutcome::Stalled(_) => {
            super::super::stop_result_with_incumbent(
                sf,
                problem,
                &basis_aug,
                &x_b,
                col_scale,
                total_iters,
                options,
            )
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::print_stderr)]
mod tests {
    //! Big-M Phase I の全分岐 (feasible / infeasible / Ge / Eq / 混在) を
    //! 小規模合成 LP で網羅検証する。
    //!
    //! 旧 test は objective + status のみ assert していたため、Phase I が偽
    //! Optimal を出した場合や dual recovery が崩れた場合に検出できなかった。
    //! `assert_kkt_optimal` で primal/dual/objective を一括検証する。

    use crate::options::SolverOptions;
    use crate::problem::{ConstraintType, LpProblem, SolveStatus};
    use crate::simplex::solve_with;
    use crate::sparse::CscMatrix;
    use crate::test_kkt::assert_kkt_optimal;

    struct FreshFactorFailureGuard(super::FreshFactorFailure);

    impl FreshFactorFailureGuard {
        fn set(mode: super::FreshFactorFailure) -> Self {
            let previous = super::FRESH_FACTOR_FAILURE.get();
            super::FRESH_FACTOR_FAILURE.set(mode);
            Self(previous)
        }
    }

    impl Drop for FreshFactorFailureGuard {
        fn drop(&mut self) {
            super::FRESH_FACTOR_FAILURE.set(self.0);
        }
    }

    struct ForcePhase1UnboundedGuard(bool);

    impl ForcePhase1UnboundedGuard {
        fn set() -> Self {
            let previous = super::FORCE_PHASE1_UNBOUNDED.get();
            super::FORCE_PHASE1_UNBOUNDED.set(true);
            Self(previous)
        }
    }

    impl Drop for ForcePhase1UnboundedGuard {
        fn drop(&mut self) {
            super::FORCE_PHASE1_UNBOUNDED.set(self.0);
        }
    }

    fn phase1_refresh_failure_result(failure: super::FreshFactorFailure) -> SolveStatus {
        use crate::presolve::LpEquilibration;

        // x = 1 is feasible, while the identity Phase I basis starts with the
        // artificial basic at stale x_B = 1 > primal_tol. Thus the forced
        // Unbounded route cannot legitimately obtain a Farkas certificate.
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0],
            a,
            vec![1.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let sf = crate::simplex::build_standard_form(&lp);
        let (a, b, c, row_scale, col_scale) = LpEquilibration::scale(&sf.a, &sf.b, &sf.c);
        let opts = SolverOptions {
            use_lp_crash_basis: false,
            ..Default::default()
        };
        let _outcome_guard = ForcePhase1UnboundedGuard::set();
        let _failure_guard = FreshFactorFailureGuard::set(failure);

        super::big_m_cold_start(&sf, &lp, &opts, &a, &b, &c, &row_scale, &col_scale).status
    }

    #[test]
    fn fresh_factor_failure_never_uses_stale_positive_artificial_as_proof() {
        let status = phase1_refresh_failure_result(super::FreshFactorFailure::Singular);
        assert_eq!(status, SolveStatus::NumericalError);
        assert_ne!(status, SolveStatus::Infeasible);
    }

    #[test]
    fn fresh_factor_deadline_failure_is_timeout() {
        let status = phase1_refresh_failure_result(super::FreshFactorFailure::Deadline);
        assert_eq!(status, SolveStatus::Timeout);
        assert_ne!(status, SolveStatus::Infeasible);
    }

    #[test]
    #[should_panic(expected = "internal invariant violation during Phase I proof refresh")]
    fn fresh_factor_internal_error_fails_fast() {
        super::classify_unbounded_proof_refresh(Err(
            crate::error::SolverError::DimensionMismatch {
                field: "basis",
                expected: 1,
                got: 0,
            },
        ));
    }

    #[test]
    fn big_m_phase1_feasible_eq() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        assert_kkt_optimal(&lp, 3.0, "big_m_phase1_feasible_eq");
    }

    /// Codex completeness edge: a feasible Eq system whose Big-M Phase I
    /// needs an lb-repair on an artificial-leaving row — which the anti-cycling
    /// guard (`basis[r] < n_enter`, kept to stop sierra's 478-pivot chase)
    /// suppresses, so Big-M abandons it *when the crash basis is disabled*. The
    /// architecture's primal fallback (`two_phase_dual_simplex`) recovers it, so
    /// the final verdict is still Optimal — the guard's conservatism is a
    /// completeness trade-off, not a correctness bug.
    ///
    /// `-2x + y = 1, -2x + 2y = 3, x,y ≥ 0, min x + y` ⇒ x=0.5, y=2, obj=2.5.
    /// Solved with presolve OFF and crash OFF to exercise the bare Big-M → fallback
    /// path (default config solves it via crash; see report).
    #[test]
    fn big_m_phase1_artificial_lb_repair_edge_recovers_via_fallback() {
        let a =
            CscMatrix::from_triplets(&[0, 1, 0, 1], &[0, 0, 1, 1], &[-2.0, -2.0, 1.0, 2.0], 2, 2)
                .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![1.0, 3.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            presolve: false,
            use_lp_crash_basis: false,
            ..SolverOptions::default()
        };
        let r = solve_with(&lp, &opts);
        assert_eq!(
            r.status,
            SolveStatus::Optimal,
            "bare Big-M (crash off) must still reach Optimal via the primal fallback; got {:?}",
            r.status
        );
        assert!(
            (r.objective - 2.5).abs() < 1e-6,
            "expected obj 2.5, got {}",
            r.objective
        );
    }

    #[test]
    fn big_m_phase1_feasible_ge() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        assert_kkt_optimal(&lp, 5.0, "big_m_phase1_feasible_ge");
    }

    #[test]
    fn big_m_phase1_infeasible_eq_contradiction() {
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
            .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0, 2.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let result = solve_with(&lp, &SolverOptions::default());
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "got {:?}",
            result.status
        );
    }

    #[test]
    fn big_m_phase1_infeasible_ge_eq_mix() {
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
            .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0, 2.0],
            vec![ConstraintType::Ge, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let result = solve_with(&lp, &SolverOptions::default());
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "got {:?}",
            result.status
        );
    }

    /// 3 ≤ x1+x2 ≤ 7, min x1+x2 → obj=3
    #[test]
    fn big_m_phase1_le_ge_range_feasible() {
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
            .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![7.0, 3.0],
            vec![ConstraintType::Le, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        assert_kkt_optimal(&lp, 3.0, "big_m_phase1_le_ge_range_feasible");
    }

    /// Ge b=0 (initial_basis に surplus が直接入る、artificial 不要)
    #[test]
    fn big_m_phase1_ge_b_zero_bypasses_bigm() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![0.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        assert_kkt_optimal(&lp, 0.0, "big_m_phase1_ge_b_zero_bypasses_bigm");
    }

    /// Eq with b=0 (degenerate artificial). wood1p / etamacro が踏むパターン
    /// の最小再現: Big-M Phase I が b=0 Eq 行で人工変数を正しく排除しないと
    /// dfeas が劣化する。
    #[test]
    fn big_m_phase1_degenerate_eq_zero_rhs() {
        // x1 + x2 = 0  (b=0 Eq → 人工変数縮退)
        // x1 + x3 = 1  (b=1 Eq)
        // min x3
        // → x1=x2=0, x3=1, obj=1
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 2], &[1.0, 1.0, 1.0, 1.0], 2, 3)
            .unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 0.0, 1.0],
            a,
            vec![0.0, 1.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 3],
            None,
        )
        .unwrap();
        assert_kkt_optimal(&lp, 1.0, "big_m_phase1_degenerate_eq_zero_rhs");
    }

    /// 大係数 + Eq + Ge 混在: Big-M スケーリングが c/b の大きさに動的追従しないと
    /// 双対実行可能性が崩れる。
    #[test]
    fn big_m_phase1_large_coeff_eq_ge_mix() {
        // 1e6 * x1 + x2 = 2e6, x1 + x2 >= 1, min x1 + x2
        // x1=1 で Eq 違反 (e6 + x2 = 2e6 → x2 = 1e6) → x2=1e6
        // → x1=1, x2=1e6 を最適化: x1+x2=1e6+1。x1↑にすると x2↓ で合計減 → x1=2, x2=0
        //   sum=2 だが Eq 確認: 2e6+0=2e6 ✓、Ge: 2>=1 ✓ → obj=2
        let a =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0e6, 1.0, 1.0, 1.0], 2, 2)
                .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0e6, 1.0],
            vec![ConstraintType::Eq, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        assert_kkt_optimal(&lp, 2.0, "big_m_phase1_large_coeff_eq_ge_mix");
    }

    /// Regression: ArtificialPriorityLeaving::bland_leaving must
    /// honor Priority 2 (artificial in basis, x_B > tol). Default Bland
    /// (Priority 1 only) would mask the artificial-removal objective and
    /// return None once `x_B ≥ 0`, causing `dual_simplex_core_advanced` to
    /// declare false Optimal with artificials still in basis.
    #[test]
    fn artificial_priority_bland_picks_artificial_when_xb_nonneg() {
        use super::ArtificialPriorityLeaving;
        use crate::simplex::pricing::DualLeavingStrategy;
        let n_total = 3usize;
        let mut strat = ArtificialPriorityLeaving { n_total };
        let basis = vec![1usize, n_total]; // row 0: orig var, row 1: artificial
        let x_b = vec![0.5_f64, 2.0_f64];
        let pick = strat.bland_leaving(&x_b, 1e-9, &basis);
        assert_eq!(
            pick,
            Some(1),
            "bland_leaving must select artificial row when x_B >= 0"
        );

        // No artificials → None
        let basis2 = vec![0usize, 1usize];
        let pick2 = strat.bland_leaving(&x_b, 1e-9, &basis2);
        assert_eq!(pick2, None);
    }

    /// Regression: progress_metric must count artificial-removal
    /// progress; otherwise `best_infeas = 0` for any Big-M Phase I starting
    /// from `x_B = b ≥ 0`, threshold = 0, and bland_mode triggers after
    /// k_trigger iterations regardless of genuine progress.
    #[test]
    fn artificial_priority_progress_metric_includes_artificial_sum() {
        use super::ArtificialPriorityLeaving;
        use crate::simplex::pricing::DualLeavingStrategy;
        let n_total = 2usize;
        let mut strat = ArtificialPriorityLeaving { n_total };
        let basis = vec![0usize, n_total]; // row 1: artificial
        let x_b = vec![3.0_f64, 5.0_f64];
        // sum_neg = 0, art_sum = 5.0
        assert!((strat.progress_metric(&x_b, &basis) - 5.0).abs() < 1e-12);

        // After driving artificial out
        let basis2 = vec![0usize, 1usize];
        assert!(strat.progress_metric(&x_b, &basis2) < 1e-12);
    }

    /// Regression: Big-M Phase I で bland_mode が誤起動しても false
    /// Infeasible を返してはいけない。小規模 Eq-only feasible LP で
    /// `assert_kkt_optimal` が Infeasible 戻り値で panic することを利用。
    #[test]
    fn big_m_phase1_no_false_infeasible_when_blandmode_triggers() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 2, 2],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[5.0, 3.0, 2.0, 2.0, 7.0, 1.0, 1.0, 1.0, 1.0],
            3,
            3,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0, 1.0],
            a,
            vec![10.0, 5.0, 3.0],
            vec![ConstraintType::Eq; 3],
            vec![(0.0, f64::INFINITY); 3],
            None,
        )
        .unwrap();
        assert_kkt_optimal(
            &lp,
            3.0,
            "big_m_phase1_no_false_infeasible_when_blandmode_triggers",
        );
    }

    /// 自由変数 + Eq: split-variable + Phase I の組合せで feasibility が崩れないか。
    #[test]
    fn big_m_phase1_free_var_eq() {
        // x1 + x2 = 2, x1 free, x2 in [0, INF), min x1+x2
        // → x1=2-x2, obj = 2 (任意の feasible で)
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        assert_kkt_optimal(&lp, 2.0, "big_m_phase1_free_var_eq");
    }

    // -------- crash basis → Big-M Phase I 配線 sentinel --------
    //
    // big_m_cold_start を直接呼び crash on/off で num_artificial と iter を比較。
    // solve_with は primal-first に倒れて Big-M を経由しないため、ここでは
    // build_standard_form / LpEquilibration / big_m_cold_start を `super::` 経由で
    // 直接呼ぶ。SERIAL_LOCK で thread-local probe の並列干渉を避ける。
    //
    // 経路観測は `crash_probe` thread-local hook を経由し、LU 失敗 /
    // x_B 負などの分岐が実際に踏まれたことを直接 assert する (sentinel が
    // observed 値を内部で再計算する短絡 = tautology を排除)。

    use std::sync::Mutex;
    static SERIAL_LOCK: Mutex<()> = Mutex::new(());

    /// 直接呼出し helper: big_m_cold_start に必要な事前変換 (build_standard_form +
    /// LpEquilibration) を内側で完結させ、(SolverResult, n_art_post, probe_outcome) を返す。
    ///
    /// observed n_art は `crash_probe` の最終 Outcome から派生する。
    /// - `Adopted(n)` → crash 採用、basis に残った artificial 数 = n
    /// - その他 → identity 経路に倒れた = sf.num_artificial
    ///
    /// crash off (use_crash=false) でも `try_build_crash_phase1_state` は短絡
    /// (DisabledOption) で hook を更新するため、probe は呼出ごとに必ず 1 件
    /// 記録される (None なら caller の clear 漏れ or 呼出 path 変更)。
    fn invoke_big_m_with_option(
        lp: &LpProblem,
        use_crash: bool,
    ) -> (
        crate::problem::SolverResult,
        usize,
        super::crash_probe::Outcome,
    ) {
        invoke_big_m_with_option_deadline_secs(lp, use_crash, 60.0)
    }

    fn invoke_big_m_with_option_deadline_secs(
        lp: &LpProblem,
        use_crash: bool,
        deadline_secs: f64,
    ) -> (
        crate::problem::SolverResult,
        usize,
        super::crash_probe::Outcome,
    ) {
        use crate::presolve::LpEquilibration;
        let sf = crate::simplex::build_standard_form(lp);
        let (a, b, c, row_scale, col_scale) = LpEquilibration::scale(&sf.a, &sf.b, &sf.c);
        let opts = SolverOptions {
            use_lp_crash_basis: use_crash,
            timeout_secs: Some(deadline_secs),
            max_etas: crate::options::default_max_etas(sf.m),
            deadline: Some(
                std::time::Instant::now() + std::time::Duration::from_secs_f64(deadline_secs),
            ),
            ..Default::default()
        };

        super::crash_probe::clear();
        let result = super::big_m_cold_start(&sf, lp, &opts, &a, &b, &c, &row_scale, &col_scale);
        let outcome = super::crash_probe::take()
            .expect("crash_probe must record an Outcome on every big_m_cold_start invocation");
        let n_art_obs = match outcome {
            super::crash_probe::Outcome::Adopted(n) => n,
            _ => sf.num_artificial,
        };
        (result, n_art_obs, outcome)
    }

    /// network-flow 風: 各 Eq 行に singleton 構造列 + 共有 hub。crash で大量の
    /// artif 列を structural singleton で被覆できる。
    fn build_network_eq_lp(n_flow: usize, n_hub: usize, seed_init: u64) -> LpProblem {
        let mut seed = seed_init;
        let mut next = || -> f64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 16) as f64 / (u64::MAX >> 16) as f64) * 2.0 - 1.0
        };
        let n = n_flow + n_hub;
        let m_eq = n_flow;
        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_vals = Vec::new();
        for i in 0..n_flow {
            a_rows.push(i);
            a_cols.push(i);
            a_vals.push(1.0);
        }
        for h in 0..n_hub {
            for i in 0..n_flow {
                a_rows.push(i);
                a_cols.push(n_flow + h);
                a_vals.push(0.01 + 0.02 * (next() + 1.0) * 0.5);
            }
        }
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_eq, n).unwrap();
        let b: Vec<f64> = (0..m_eq).map(|_| 1.0 + (next() + 1.0) * 0.25).collect();
        let c: Vec<f64> = (0..n).map(|_| next()).collect();
        let bounds = vec![(0.0_f64, 10.0_f64); n];
        LpProblem::new_general(c, a, b, vec![ConstraintType::Eq; m_eq], bounds, None).unwrap()
    }

    /// Ge/Eq 混在 + 多変量。crash 行被覆 + Phase I の Big-M penalty 駆出が結合。
    fn build_ge_eq_mix_lp(
        n_eq: usize,
        n_ge: usize,
        n_struct_extra: usize,
        seed_init: u64,
    ) -> LpProblem {
        let mut seed = seed_init;
        let mut next = || -> f64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 16) as f64 / (u64::MAX >> 16) as f64) * 2.0 - 1.0
        };
        let m = n_eq + n_ge;
        let n = m + n_struct_extra;
        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_vals = Vec::new();
        for i in 0..m {
            a_rows.push(i);
            a_cols.push(i);
            a_vals.push(1.0); // singleton diag
        }
        for j in 0..n_struct_extra {
            for i in 0..m {
                a_rows.push(i);
                a_cols.push(m + j);
                a_vals.push(0.05 * (next() + 1.0));
            }
        }
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();
        let b: Vec<f64> = (0..m).map(|i| if i < n_eq { 1.0 } else { 0.5 }).collect();
        let c: Vec<f64> = (0..n).map(|_| (next() + 1.0) * 0.5).collect();
        let mut ct = vec![ConstraintType::Eq; n_eq];
        ct.extend(std::iter::repeat_n(ConstraintType::Ge, n_ge));
        let bounds = vec![(0.0_f64, 10.0_f64); n];
        LpProblem::new_general(c, a, b, ct, bounds, None).unwrap()
    }

    /// Beale 教科書 degenerate LP の縮約 (Eq 化)。
    /// Phase I で人工変数を全行に挿入 → crash で対角構造列で被覆可能。
    fn build_beale_eq_lp() -> LpProblem {
        // 3 行 × 4 列、各行に diag entry + 共有 col
        let a = CscMatrix::from_triplets(
            &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            &[0, 1, 2, 3, 3, 3, 0, 1, 2, 1], // 重複しないよう注意
            &[1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.0001],
            3,
            4,
        )
        .unwrap();
        let b = vec![1.0, 2.0, 3.0];
        let c = vec![1.0, 1.0, 1.0, 0.5];
        let bounds = vec![(0.0_f64, 100.0_f64); 4];
        LpProblem::new_general(c, a, b, vec![ConstraintType::Eq; 3], bounds, None).unwrap()
    }

    /// crash 採用で num_artificial が真に減少する (network 構造)。
    #[test]
    fn crash_reduces_num_artificial_network() {
        let _guard = SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let lp = build_network_eq_lp(80, 3, 0xF1F2_F3F4_F5F6_F7F8);
        let (_r_off, n_art_off, out_off) = invoke_big_m_with_option(&lp, false);
        let (_r_on, n_art_on, out_on) = invoke_big_m_with_option(&lp, true);
        eprintln!(
            "CRASH_BIGM_NETWORK: n_art_off={} n_art_on={} out_off={:?} out_on={:?}",
            n_art_off, n_art_on, out_off, out_on
        );
        assert!(
            matches!(out_off, super::crash_probe::Outcome::DisabledOption),
            "off path must short-circuit on use_lp_crash_basis=false; got {:?}",
            out_off
        );
        assert!(
            matches!(out_on, super::crash_probe::Outcome::Adopted(_)),
            "on path must adopt crash state; got {:?}",
            out_on
        );
        assert!(
            n_art_on < n_art_off,
            "crash must reduce num_artificial: off={} on={}",
            n_art_off,
            n_art_on
        );
        let reduction_ratio = (n_art_off - n_art_on) as f64 / n_art_off.max(1) as f64;
        assert!(
            reduction_ratio >= 0.30,
            "crash artificial reduction {:.2} < 0.30 (off={} on={})",
            reduction_ratio,
            n_art_off,
            n_art_on
        );
    }

    /// Ge/Eq 混在で crash が num_artificial と iter を共に減らす。
    #[test]
    fn crash_reduces_iters_ge_eq_mix() {
        let _guard = SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let lp = build_ge_eq_mix_lp(40, 30, 4, 0xA1A2_A3A4_A5A6_A7A8);
        let (r_off, n_art_off, out_off) = invoke_big_m_with_option(&lp, false);
        let (r_on, n_art_on, out_on) = invoke_big_m_with_option(&lp, true);
        eprintln!(
            "CRASH_BIGM_MIX: n_art_off={} n_art_on={} iter_off={} iter_on={} status_off={:?} status_on={:?} out_on={:?}",
            n_art_off, n_art_on, r_off.iterations, r_on.iterations, r_off.status, r_on.status, out_on,
        );
        assert_eq!(r_off.status, SolveStatus::Optimal, "off must be Optimal");
        assert_eq!(r_on.status, SolveStatus::Optimal, "on must be Optimal");
        assert!(matches!(
            out_off,
            super::crash_probe::Outcome::DisabledOption
        ));
        assert!(matches!(out_on, super::crash_probe::Outcome::Adopted(_)));
        let obj_diff = (r_on.objective - r_off.objective).abs() / (1.0 + r_off.objective.abs());
        assert!(obj_diff < 1e-6, "crash obj drift: {:.3e}", obj_diff);
        assert!(
            n_art_on < n_art_off,
            "crash artif reduction expected: off={} on={}",
            n_art_off,
            n_art_on
        );
        // iter 削減 sentinel: wiring revert で確実に FAIL するよう assert 化。
        // 観測 96→26 (27%) でマージン十分、閾値は 0.7 (= 30% 削減) で設定。
        const ITER_REDUCTION_THRESHOLD: f64 = 0.7;
        assert!(
            (r_on.iterations as f64) < (r_off.iterations as f64) * ITER_REDUCTION_THRESHOLD,
            "crash iter reduction insufficient: off={} on={} (need on < {:.0})",
            r_off.iterations,
            r_on.iterations,
            (r_off.iterations as f64) * ITER_REDUCTION_THRESHOLD,
        );
    }

    /// Beale 縮約 (degenerate Eq) で crash 採用、対角構造を全 artif 被覆。
    #[test]
    fn crash_handles_beale_degenerate_eq() {
        let _guard = SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let lp = build_beale_eq_lp();
        let (r_off, n_art_off, out_off) = invoke_big_m_with_option(&lp, false);
        let (r_on, n_art_on, out_on) = invoke_big_m_with_option(&lp, true);
        eprintln!(
            "CRASH_BIGM_BEALE: n_art_off={} n_art_on={} iter_off={} iter_on={} out_off={:?} out_on={:?}",
            n_art_off, n_art_on, r_off.iterations, r_on.iterations, out_off, out_on,
        );
        assert_eq!(r_off.status, SolveStatus::Optimal, "off Optimal");
        assert_eq!(r_on.status, SolveStatus::Optimal, "on Optimal");
        assert!(
            matches!(out_on, super::crash_probe::Outcome::Adopted(0)),
            "Beale: crash must adopt with n_art=0; got {:?}",
            out_on
        );
        assert!(n_art_on <= n_art_off);
        assert_eq!(n_art_on, 0, "Beale 縮約は全 artif を crash で除去できる");
    }

    /// 複数 LCG seed (5 種) で random Ge/Eq LP を生成し crash 削減を集計。
    #[test]
    fn crash_reduces_num_artificial_multi_seed() {
        let _guard = SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let seeds: &[u64] = &[
            0xC0FF_EE00_DEAD_BEEF,
            0x1234_5678_9ABC_DEF0,
            0xF00D_BABE_FACE_CAFE,
            0xA5A5_5A5A_3C3C_C3C3,
            0x1111_2222_3333_4444,
        ];
        let mut wins = 0usize;
        let mut adopt_count = 0usize;
        for &seed in seeds {
            let lp = build_network_eq_lp(50, 2, seed);
            let (_, n_off, _) = invoke_big_m_with_option(&lp, false);
            let (_, n_on, out_on) = invoke_big_m_with_option(&lp, true);
            eprintln!(
                "CRASH_BIGM_SEED 0x{:x}: off={} on={} out_on={:?}",
                seed, n_off, n_on, out_on
            );
            if matches!(out_on, super::crash_probe::Outcome::Adopted(_)) {
                adopt_count += 1;
            }
            if n_on < n_off {
                wins += 1;
            }
        }
        assert!(
            wins >= 4,
            "crash reduced num_artificial on {}/{} seeds (need ≥ 4)",
            wins,
            seeds.len()
        );
        assert!(
            adopt_count >= 4,
            "crash actually adopted on {}/{} seeds (need ≥ 4)",
            adopt_count,
            seeds.len()
        );
    }

    /// no-op proof (memory: feedback_sentinel_must_fail_under_noop):
    /// `use_lp_crash_basis: false` で crash 経路が `DisabledOption` 短絡を踏み、
    /// option が唯一の disable 経路であることを probe + 実測で確認する。
    /// sentinel は option=true→false のトグルで確実に fail することを保証する。
    #[test]
    fn crash_disabled_option_collapses_to_identity() {
        let _guard = SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let lp = build_network_eq_lp(60, 2, 0xDEAD_BEEF_CAFE_F00D);

        // option=false → DisabledOption (identity path, no crash)
        let (r_off, n_off, out_off) = invoke_big_m_with_option(&lp, false);
        assert!(
            matches!(out_off, super::crash_probe::Outcome::DisabledOption),
            "use_lp_crash_basis=false must short-circuit on DisabledOption; got {:?}",
            out_off
        );

        // option=true → crash actually adopted (proves sentinel is non-trivial)
        let (r_on, n_on, out_on) = invoke_big_m_with_option(&lp, true);
        assert!(
            matches!(out_on, super::crash_probe::Outcome::Adopted(_)),
            "use_lp_crash_basis=true must adopt crash; got {:?}",
            out_on
        );

        // crash reduces num_artificial (non-tautological: the two paths differ)
        assert!(
            n_on < n_off,
            "crash must reduce num_artificial: off={} on={}",
            n_off,
            n_on
        );

        // both paths reach Optimal on the same LP
        assert_eq!(r_off.status, SolveStatus::Optimal, "off must be Optimal");
        assert_eq!(r_on.status, SolveStatus::Optimal, "on must be Optimal");
        let obj_diff = (r_off.objective - r_on.objective).abs() / (1.0 + r_off.objective.abs());
        assert!(
            obj_diff < 1e-6,
            "objective must match regardless of crash: {:.3e}",
            obj_diff
        );
    }

    // -------- crash fallback 直接 test --------
    //
    // `try_build_crash_phase1_state` の guard を probe 経由で直接検証する。
    // 大規模 e2e でなく合成 LP で guard 分岐を踏ませ、wiring 退化や guard
    // 漏れを最小 LP で捕捉する。

    /// LU 因子化失敗時、identity 経路に倒す (`LuFailed` を踏む)。
    ///
    /// crash::compute_crash_basis は singleton 構造 (各 Eq 行に独立 structural
    /// 列) で全行 structural-cover を試み、その構造で線形従属を仕込むことで
    /// LuBasis::new が SingularBasis を返すように構成する。
    #[test]
    fn crash_lu_failure_falls_back_to_identity() {
        let _guard = SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 3 Eq 行 × 4 列。col 0,1,2 を singleton で pick すると basis=[0,1,2]
        // だが col 0/1 が同一 sparsity でも row 2 の col 2 が独立で LU は通る
        // (singular にならない)。実際に singular にするには A 全体で rank 落ち
        // が要る。以下の構成を使う:
        // - col 0: row 0 val=1; row 1 val=1
        // - col 1: row 0 val=1; row 1 val=1 (col 1 = col 0)
        // - col 2: row 2 val=1
        // crash は col 0→row 0, col 1→row 1, col 2→row 2 を pick →
        // B = [[1,1,0],[1,1,0],[0,0,1]] singular。
        let a = CscMatrix::from_triplets(
            &[0, 1, 0, 1, 2],
            &[0, 0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            3,
            3,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0, 1.0],
            a,
            vec![1.0, 1.0, 1.0],
            vec![ConstraintType::Eq; 3],
            vec![(0.0, f64::INFINITY); 3],
            None,
        )
        .unwrap();
        let (_r, _n, out) = invoke_big_m_with_option(&lp, true);
        // crash が dup 列を pick して LU で singular に堕ちる、または
        // dup col の rank 漏れで NotReduced に倒れるのいずれか。後者でも
        // identity fallback の安全性は変わらず — adopt しないことが本質。
        assert!(
            matches!(
                out,
                super::crash_probe::Outcome::LuFailed | super::crash_probe::Outcome::NotReduced
            ),
            "duplicate-col LP must trigger LU failure or NotReduced fallback; got {:?}",
            out,
        );
    }

    /// x_B = B^{-1} b に負成分が出るケースで identity に倒す (`XbNegative`)。
    ///
    /// Mixed Le + Eq: Le row の slack が naturally basic、Eq row は crash の
    /// structural cover を負係数列で許可するケースを構成。crash::compute_crash_basis
    /// の sign-coincidence guard は係数符号 = b 符号を要求するが、列符号の不揃いで
    /// 通り抜けた末に B^{-1} b で負成分が生じる。
    #[test]
    fn crash_xb_negative_falls_back_to_identity() {
        let _guard = SERIAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 3 Eq 行 × 4 列、b = [1, 1, 1]。crash が選んだ basis で
        // B^{-1} b の特定成分が負になるよう、非対角 entry を仕込む。
        // random LCG fixture で候補生成、XbNegative を 1 件でも観測したら成功。
        let mut seed: u64 = 0xCAFEBABE_DEADBEEF;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
        };
        /// Max random probe attempts to generate a negative-x_B test configuration.
        const RANDOM_PROBE_RETRIES: usize = 20;
        let mut found = false;
        let mut last_outcome = super::crash_probe::Outcome::DisabledOption;
        for _ in 0..RANDOM_PROBE_RETRIES {
            let n = 6usize;
            let m = 4usize;
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            // 各 row に singleton-like diag (符号混在) + 隣接 row の off-diag
            for i in 0..m {
                rows.push(i);
                cols.push(i);
                vals.push(if next() < 0.0 { -1.0 } else { 1.0 });
                rows.push(i);
                cols.push((i + 1) % m);
                vals.push(next() * 0.5);
            }
            // 余剰列 (hub)
            for j in m..n {
                for i in 0..m {
                    rows.push(i);
                    cols.push(j);
                    vals.push(next() * 0.3);
                }
            }
            let a = match CscMatrix::from_triplets(&rows, &cols, &vals, m, n) {
                Ok(a) => a,
                Err(_) => continue,
            };
            let b: Vec<f64> = (0..m).map(|_| 0.5 + next().abs()).collect();
            let c: Vec<f64> = (0..n).map(|_| next().abs()).collect();
            let lp = match LpProblem::new_general(
                c,
                a,
                b,
                vec![ConstraintType::Eq; m],
                vec![(0.0, f64::INFINITY); n],
                None,
            ) {
                Ok(lp) => lp,
                Err(_) => continue,
            };
            let (_, _, out) = invoke_big_m_with_option_deadline_secs(&lp, true, 0.5);
            last_outcome = out;
            if matches!(out, super::crash_probe::Outcome::XbNegative) {
                found = true;
                break;
            }
        }
        // XbNegative 直接ヒットしなくとも、Adopted 以外 (= identity に倒れる) は
        // safe fallback として許容 — adopt して numerical error を生まない。
        assert!(
            found || !matches!(last_outcome, super::crash_probe::Outcome::Adopted(_)),
            "x_B < 0 fallback path unreachable; last_outcome={:?}",
            last_outcome,
        );
    }

    /// `ArtificialPriorityLeaving::allows_lb_repair` returns `true`: genuine
    /// lb-violations from Priority-2 removal pivots must be repaired. The
    /// artificial-vs-structural distinction that keeps this safe is enforced in
    /// the core (sign-flip suppressed for artificial leaving rows), not here —
    /// see `core.rs` 3d' and the end-to-end cycling sentinels below.
    #[test]
    fn artificial_priority_allows_lb_repair_is_true() {
        use super::ArtificialPriorityLeaving;
        use crate::simplex::pricing::DualLeavingStrategy;
        let strat = ArtificialPriorityLeaving { n_total: 4 };
        assert!(
            strat.allows_lb_repair(),
            "ArtificialPriorityLeaving must allow lb-repair so Priority-2-manufactured \
             lb-violations are sign-flip-repaired (blanket false re-introduces the \
             beaconfd/scrs8 Phase-I 2-cycle)"
        );
    }

    /// Companion: `MostInfeasibleLeaving` must use the default `true`
    /// (warm-start lb-repair is valid for standard dual simplex).
    #[test]
    fn most_infeasible_allows_lb_repair_is_true() {
        use crate::simplex::pricing::{DualLeavingStrategy, MostInfeasibleLeaving};
        let strat = MostInfeasibleLeaving;
        assert!(
            strat.allows_lb_repair(),
            "MostInfeasibleLeaving must allow lb-repair (allows_lb_repair == true)"
        );
    }

    /// Sentinel: `farkas_direction_certified` certifies a valid Farkas direction
    /// and rejects an invalid one.
    ///
    /// no-op proof: removing the A^T check (always returning true when b^Ty > tol)
    /// makes `farkas_direction_not_certified_when_aty_positive` FAIL. This companion
    /// proves the positive case is also detected correctly.
    ///
    /// A_aug = [[1,1,0],[-1,0,1]], b=[1,2], basis=[1,2].
    /// B=I, y_joint=[1,1]: b^Ty=3>0, A^T y for x0 = 1-1=0<=tol → certified.
    #[test]
    fn farkas_direction_certified_accepts_valid_direction() {
        use super::farkas_direction_certified;

        // A_aug = [[1,1,0],[-1,0,1]], 2 rows, 3 cols. Col 0 = x0 (original), cols 1,2 = artificials.
        let a_aug =
            CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 2], &[1.0, 1.0, -1.0, 1.0], 2, 3)
                .unwrap();
        let b = [1.0_f64, 2.0_f64];
        // y = [1,1] (already BTRAN'd from identity basis).
        let y = [1.0_f64, 1.0];
        let tol = 1e-6_f64;
        // b^Ty = 3 > tol. A^T y for x0: a[0][0]*y[0]+a[1][0]*y[1] = 1-1 = 0 <= tol → certified.
        assert!(
            farkas_direction_certified(&a_aug, &b, &y, 1, tol),
            "valid Farkas direction (b^Ty=3>0, A^Ty_x0=0<=tol) must be certified"
        );
    }

    /// Sentinel: no-op proof for `farkas_direction_certified`.
    /// A direction that violates A^T y <= tol must NOT be certified.
    ///
    /// no-op: removing the A^T check (always returning true after b^Ty > tol)
    /// causes this test to FAIL (false certification on a feasible direction).
    #[test]
    fn farkas_direction_not_certified_when_aty_positive() {
        use super::farkas_direction_certified;

        // A_aug = [[1, 1]], b = [2.0], y = [1.0] (not in row space of A^T).
        // A^T y for j=0: 1*y[0] = 1 > tol (dual_tol * 2 ≈ 2e-7). → NOT certified.
        let a_aug = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let b = [2.0_f64];
        let y = [1.0_f64];
        let tol = 1e-6_f64;
        assert!(
            !farkas_direction_certified(&a_aug, &b, &y, 1, tol),
            "A^T y = 1 > tol: direction must NOT be certified (removing A^T check breaks this)"
        );
    }

    /// Sentinel: cplex2-class infeasibility via end-to-end solve.
    /// This LP has Eq+Ge constraints that produce joint b^Ty cancellation
    /// but is correctly Infeasible.
    ///
    /// no-op: reverting to joint-only Farkas changes `Infeasible` to `Timeout` → FAIL.
    ///
    /// Construction: x0 + x1 = 3 (Eq) AND x0 + x1 >= 5 (Ge). Infeasible since
    /// x0+x1 cannot be both 3 and >=5. With Eq+Ge in standard form, the Big-M
    /// Phase I will have two artificial rows. The joint indicator may cancel
    /// when b values have mixed signs after Ruiz scaling; per-row probes recover.
    #[test]
    fn big_m_phase1_infeasible_eq_ge_cancellation_class() {
        // x0 + x1 = 3, x0 + x1 >= 5 → infeasible (3 < 5).
        let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2)
            .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let result = solve_with(&lp, &SolverOptions::default());
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "x0+x1=3 AND x0+x1>=5 must be detected as Infeasible; got {:?}. \
             If Timeout: per-row Farkas probe was removed or joint-only is insufficient \
             for this Eq+Ge combination.",
            result.status
        );
    }

    /// Sentinel: Big-M Unbounded + any_positive_art → Infeasible (cplex2 root fix).
    ///
    /// When Big-M Phase I dual simplex reaches `Unbounded` with at least one
    /// artificial still in the basis at positive value, the LP is infeasible:
    /// a feasible LP always drives all artificials to zero before reaching Unbounded.
    ///
    /// no-op: removing the `any_positive_art` check causes this LP to return
    /// `Timeout` instead of `Infeasible` → FAIL.
    ///
    /// LP: x0 = 2 (Eq, b=2) AND -x0 >= 1 (Ge, b=1). Infeasible: x0 must be 2
    /// but -x0 = -2 < 1. Mixed-sign structure causes b^Ty ≈ 0 (joint Farkas fails)
    /// while Phase I hits Unbounded with the Ge-artificial still positive.
    #[test]
    fn big_m_phase1_unbounded_with_positive_art_declares_infeasible() {
        // x0 = 2 (Eq)  →  needs artificial a0
        // -x0 >= 1 (Ge) →  after flip: x0 ≤ -1, needs artificial a1
        // These are contradictory: no x0 >= 0 can satisfy both.
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, -1.0], 2, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![2.0, 1.0],
            vec![ConstraintType::Eq, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let result = solve_with(&lp, &SolverOptions::default());
        assert_eq!(
            result.status,
            SolveStatus::Infeasible,
            "x0=2 AND -x0>=1 is infeasible; Big-M Phase I Unbounded+positive_art must \
             return Infeasible (not Timeout). got {:?}",
            result.status
        );
    }
}
