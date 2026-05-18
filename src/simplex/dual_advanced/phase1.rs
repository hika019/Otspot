//! Big-M Phase I cold-start (task #11, Phase 4 of dual_simplex_design.md §3.6 / §4)
//!
//! ## 解決する問題
//!
//! Ge / Eq 制約を含む LP の cold-start で、既存
//! `super::super::dual::two_phase_dual_simplex::cold_start_dual` は
//! `sf.num_artificial > 0` 時 Primal Phase I (人工変数 sum 最小化) に
//! フォールバックする。klein3 等 degenerate infeasible LP では cycling して
//! `iters=0 TIMEOUT` する (task #11)。
//!
//! ## アルゴリズム (Dual Phase I + Primal Phase II + Big-M)
//!
//! 設計書 §3.6 / §4 Phase 4 に基づく 2-phase Big-M。Dual と Primal を組み合わせる
//! ことで klein3 級 degenerate infeasible LP の cycling を回避する:
//!
//! 1. 人工変数列 a_i (係数 1) を `needs_artificial` な各行 i に追加し、
//!    B = I_aug を構成する。
//! 2. Big-M 摂動コスト構築:
//!    - 人工変数: `c_aug[a_i] = big_m`
//!    - 元変数 j: `c_aug[j] = c[j] + delta_j`、
//!      `delta_j = max(0, big_m * Σ_{i: needs_art} a[i, j] - c[j])`
//!    これにより初期 basis (B=I_aug, y_init = c_B = big_m * indicator) で
//!    全 reduced cost r_j ≥ 0 が成立 (双対実行可能)。
//! 3. **Phase I (Dual Simplex, Harris ratio test 装備の `dual_simplex_core_advanced`)**:
//!    x_B ≥ 0 のまま (b ≥ 0 で初期から主実行可能) なので Phase I は通常 0 反復
//!    で即終了する。役割は「双対基底を構成し、後続 Phase II で safe な warm
//!    start を提供する」こと。Unbounded を返したら Infeasible。
//! 4. **Phase II (Primal Simplex, SteepestEdgePricing)**:
//!    元コスト c_phase2 = [c | 0; n_art] で `revised_simplex_core` を実行。
//!    人工変数も pricing 対象 (n_price = n_aug) にして basis から積極的に追い出す。
//!    元 c で Phase I の摂動を消す効果も持つ。
//! 5. 終了判定:
//!    - Phase I `Unbounded` → 双対非有界 → Infeasible
//!    - Phase II 完了後、人工変数が basis に残って値 > primal_tol → 元 LP infeasible
//!    - Phase II `Optimal` で人工変数値 = 0 → 元 LP 最適
//!    - Phase II `Unbounded` → 元 LP 非有界
//!    - Timeout / SingularBasis → 通常処理
//!
//! ## M の動的算出 (設計書 §6.4)
//!
//! Ruiz スケーリング後の c, b から:
//! ```text
//! big_m = max(||c||_∞ * BIG_M_COST_MULT,
//!             ||b||_∞ * BIG_M_COST_MULT,
//!             BIG_M_FLOOR)
//! ```
//! いずれも問題スケールから派生する算式 (固定マジック値ではない)。

use crate::basis::{BasisManager, LuBasis};
use crate::options::{SolverOptions, WarmStartBasis};
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::tolerances::DROP_TOL;
use super::super::{StandardForm, SimplexOutcome, extract_solution, extract_dual_info};
use super::super::pricing::{DualLeavingStrategy, SteepestEdgePricing};
use super::core::dual_simplex_core_advanced;

/// Farkas certificate verification for primal infeasibility.
///
/// At a Big-M Phase I exit basis with artificials residual, construct the
/// pure-Phase-I dual y = B^{-T} e_art (indicator of artificial basis rows) and
/// test the Farkas alternative for the original LP {min c^T x | Ax = b, x ≥ 0}:
///
///   A^T y ≤ tol  for all original cols j  AND  b^T y > tol  →  infeasible.
///
/// This is the only sufficient proof of infeasibility we can give without
/// completing Phase I. If the certificate fails, the caller must return Timeout
/// rather than guessing Infeasible from artificial residual alone — that
/// heuristic flipped the verdict on slow-but-feasible LPs (#37: pilot/dfl001/
/// ken-13/ken-18).
///
/// Tolerance scales with ||b||_∞ to stay correct on Ruiz-scaled inputs.
fn farkas_infeasibility_certified(
    a_aug: &CscMatrix,
    b: &[f64],
    basis_aug: &[usize],
    m: usize,
    n_total: usize,
    options: &SolverOptions,
) -> bool {
    let c_phase1: Vec<f64> = (0..m)
        .map(|i| if basis_aug[i] >= n_total { 1.0 } else { 0.0 })
        .collect();

    let mut basis_mgr = match LuBasis::new(a_aug, basis_aug, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => return false,
    };
    let mut y = c_phase1;
    basis_mgr.btran_dense(&mut y);

    let b_norm = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = options.dual_tol * (1.0_f64).max(b_norm);
    let by: f64 = b.iter().zip(y.iter()).map(|(&bi, &yi)| bi * yi).sum();
    if by <= tol {
        return false;
    }
    for j in 0..n_total {
        let (rows, vals) = a_aug.get_column(j).unwrap();
        let mut aty = 0.0_f64;
        for (k, &row) in rows.iter().enumerate() {
            aty += vals[k] * y[row];
        }
        if aty > tol {
            return false;
        }
    }
    true
}

/// SolverOptions のクローンを返し、deadline がある場合は残り時間の半分に縮める。
///
/// Big-M Phase I 専用: Phase I 内で時間を使い切らず Phase II にも半分残す。
/// Phase I で half-deadline 到達 → artificial 残存判定で Infeasibility 推定。
fn clone_options_with_half_deadline(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    if let Some(d) = options.deadline {
        let now = std::time::Instant::now();
        let remaining = d.saturating_duration_since(now);
        o.deadline = Some(now + remaining / 2);
    }
    o
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
    fn select_leaving(&self, x_b: &[f64], primal_tol: f64, basis: &[usize]) -> Option<usize> {
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
    /// false Optimal with artificials in basis (task #43).
    fn bland_leaving(&self, x_b: &[f64], primal_tol: f64, basis: &[usize]) -> Option<usize> {
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
    fn progress_metric(&self, x_b: &[f64], basis: &[usize]) -> f64 {
        let neg_sum: f64 = x_b.iter().map(|&v| (-v).max(0.0)).sum();
        let art_sum: f64 = (0..x_b.len())
            .filter(|&i| basis[i] >= self.n_total)
            .map(|i| x_b[i].max(0.0))
            .sum();
        neg_sum + art_sum
    }
}

/// Big-M ペナルティ算出時の coefficient 倍率 (設計書 §6.4 推奨)。
const BIG_M_COST_MULT: f64 = 1e3;

/// Big-M ペナルティの下限 (設計書 §6.4 推奨 `1e6`)。
const BIG_M_FLOOR: f64 = 1e6;

/// Big-M Phase I cold-start (Dual Phase I + Primal Phase II + Big-M penalty)
/// for Ge/Eq 含む LP.
///
/// `a, b, c` は Ruiz スケーリング後の値を渡すこと (§6.4)。
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

    // === Step 1: 人工変数列の割り当て ===
    let mut artificial_col_of_row: Vec<Option<usize>> = vec![None; m];
    let mut n_art = 0usize;
    for i in 0..m {
        if sf.needs_artificial[i] {
            artificial_col_of_row[i] = Some(n_total + n_art);
            n_art += 1;
        }
    }
    let n_aug = n_total + n_art;

    // === Step 2: Big-M 動的算出 ===
    let c_norm = c.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let b_norm = b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let big_m = (c_norm * BIG_M_COST_MULT)
        .max(b_norm * BIG_M_COST_MULT)
        .max(BIG_M_FLOOR);

    // === Step 3: 拡張行列 A_aug = [A | I_art] ===
    let mut trip_rows: Vec<usize> = Vec::with_capacity(a.nnz() + n_art);
    let mut trip_cols: Vec<usize> = Vec::with_capacity(a.nnz() + n_art);
    let mut trip_vals: Vec<f64> = Vec::with_capacity(a.nnz() + n_art);
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
    let a_aug = match CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_aug) {
        Ok(mat) => mat,
        Err(_) => return SolverResult::numerical_error(),
    };

    // === Step 4: Phase I 用拡張コスト c_aug (Big-M 摂動 + 双対実行可能保証) ===
    //
    // 初期 basis B = I_aug、y_init = c_B (B=I) = big_m·indicator(artificial)、
    // r_j (元変数) = c[j] - big_m · Σ_{i: artificial} a[i, j]
    // r_j ≥ 0 を保証する最小 delta_j を加算。
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
    for col_opt in artificial_col_of_row.iter() {
        if let Some(col) = col_opt {
            c_aug_p1[*col] = big_m;
        }
    }

    // === Step 5: 初期 basis を B = I_aug に構成 ===
    let mut basis_aug = sf.initial_basis.clone();
    for i in 0..m {
        if let Some(col) = artificial_col_of_row[i] {
            basis_aug[i] = col;
        }
    }

    // x_B = b (b ≥ 0 保証)
    let mut x_b = b.to_vec();

    // === Step 6: Phase I (Dual Simplex with Harris ratio test + Artificial-aware) ===
    //
    // ArtificialPriorityLeaving は標準 most-infeasible (Priority 1) で
    // x_B < 0 を解消した後、人工変数の basis 残存 (Priority 2; x_B[i] > 0
    // かつ basis[i] >= n_total) を leaving 候補として継続選択する。
    // これにより Big-M Phase I 本来の「人工変数を basis から追い出す」役割を
    // 標準 dual simplex ループ (Harris ratio test 装備) で実現する。
    //
    // ## Phase I 時間配分
    //
    // Phase II も Primal Simplex で走らせるため、deadline がある場合は
    // 残り時間の **半分** を Phase I に割り当てる。Phase I が Timeout で戻った
    // 場合は artificial が basis に残ったまま → 元 LP の Infeasibility シグナル
    // として扱う (degenerate cycling を伴う infeasible 検出の経験的判定; klein3
    // などの highly degenerate infeasible LP は理論最適 pivot 数では到達できず
    // cycling する典型ケース)。
    let phase1_options = clone_options_with_half_deadline(options);
    let leaving = ArtificialPriorityLeaving { n_total };
    let mut total_iters: usize = 0;
    let phase1_outcome = dual_simplex_core_advanced(
        &a_aug, &mut x_b, &c_aug_p1, &mut basis_aug, m, n_aug, &phase1_options, &leaving,
        &mut total_iters,
    );

    match phase1_outcome {
        SimplexOutcome::Unbounded => {
            // 双対非有界 = 主実行不可
            let mut r = SolverResult::infeasible();
            r.iterations = total_iters;
            return r;
        }
        SimplexOutcome::Timeout(_) => {
            // 旧実装は artificial 残存だけで Infeasible を立てていたが、これは
            // slow-feasible LP (pilot/dfl001/ken-13/ken-18) でも発火する不健全
            // ヒューリスティック (#37)。Farkas 証明書 (A^T y ≤ 0, b^T y > 0) が
            // 通った場合のみ Infeasible を返し、検証不能なら Timeout で honest に返す。
            let any_artificial_left = (0..m).any(|i| {
                basis_aug[i] >= n_total && x_b[i].abs() > options.primal_tol
            });
            if any_artificial_left
                && farkas_infeasibility_certified(&a_aug, b, &basis_aug, m, n_total, options)
            {
                let mut r = SolverResult::infeasible();
                r.iterations = total_iters;
                return r;
            }
            let r = super::super::timeout_result_with_incumbent(
                sf, problem, &basis_aug, &x_b, col_scale, total_iters,
            );
            return r;
        }
        SimplexOutcome::SingularBasis => {
            return SolverResult::numerical_error();
        }
        SimplexOutcome::Optimal(_, _) => {
            // Phase I が Optimal で停止し人工変数が basis に残るケース (値が
            // 0 でも degenerate basic として残存しうる): Farkas 証明書で検証。
            // 値での filter (|x_B| > tol) は不適切 — 数値ドリフトで artificial
            // が 0 にクランプされても、基底構造 e_art は Farkas 条件を満たし
            // うる (klein3 の長期 pivot で観測 / task #43)。
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
    for col_opt in artificial_col_of_row.iter() {
        if let Some(col) = col_opt {
            c_aug_p2[*col] = big_m;
        }
    }

    let mut pricing = SteepestEdgePricing::new(n_aug);
    let phase2_outcome = super::super::revised_simplex_core(
        &a_aug, &mut x_b, &c_aug_p2, b, &mut basis_aug,
        m, n_aug, n_aug, &mut pricing, options, &mut total_iters, false,
    );

    // === Step 8: Phase II 結果 + 人工変数残存判定 ===
    match phase2_outcome {
        SimplexOutcome::Optimal(_obj_aug, y) => {
            // 人工変数が basis に残り値 > primal_tol → 元 LP infeasible
            for i in 0..m {
                if basis_aug[i] >= n_total && x_b[i].abs() > options.primal_tol {
                    let mut r = SolverResult::infeasible();
                    r.iterations = total_iters;
                    return r;
                }
            }

            let solution = extract_solution(sf, &basis_aug, &x_b, col_scale);
            let (dual_solution, reduced_costs, slack) =
                extract_dual_info(sf, problem, &y, &solution, row_scale);

            // warm-start: artificial が basis に残るケースは除外
            let ws = if basis_aug.iter().all(|&idx| idx < n_total) {
                Some(WarmStartBasis { basis: basis_aug.clone(), x_b: x_b.clone() })
            } else {
                None
            };

            // obj は元コスト c (Big-M ペナルティを含まない) で再計算
            let obj_orig: f64 = problem.c.iter().zip(solution.iter())
                .map(|(&ci, &xi)| ci * xi).sum();

            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj_orig + sf.obj_offset,
                solution,
                dual_solution,
                reduced_costs,
                slack,
                warm_start_basis: ws,
                iterations: total_iters,
                ..Default::default()
            }
        }
        SimplexOutcome::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            reduced_costs: vec![],
            slack: vec![],
            warm_start_basis: None,
            iterations: total_iters,
            ..Default::default()
        },
        SimplexOutcome::Timeout(_) => {
            super::super::timeout_result_with_incumbent(sf, problem, &basis_aug, &x_b, col_scale, total_iters)
        }
        SimplexOutcome::SingularBasis => SolverResult::numerical_error(),
    }
}

#[cfg(test)]
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

    #[test]
    fn big_m_phase1_feasible_eq() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![3.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2], None,
        ).unwrap();
        assert_kkt_optimal(&lp, 3.0, "big_m_phase1_feasible_eq");
    }

    #[test]
    fn big_m_phase1_feasible_ge() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![5.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2], None,
        ).unwrap();
        assert_kkt_optimal(&lp, 5.0, "big_m_phase1_feasible_ge");
    }

    #[test]
    fn big_m_phase1_infeasible_eq_contradiction() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![5.0, 2.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2], None,
        ).unwrap();
        let result = solve_with(&lp, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Infeasible, "got {:?}", result.status);
    }

    #[test]
    fn big_m_phase1_infeasible_ge_eq_mix() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![5.0, 2.0],
            vec![ConstraintType::Ge, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2], None,
        ).unwrap();
        let result = solve_with(&lp, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Infeasible, "got {:?}", result.status);
    }

    /// 3 ≤ x1+x2 ≤ 7, min x1+x2 → obj=3
    #[test]
    fn big_m_phase1_le_ge_range_feasible() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, 1.0, 1.0], 2, 2,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![7.0, 3.0],
            vec![ConstraintType::Le, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2], None,
        ).unwrap();
        assert_kkt_optimal(&lp, 3.0, "big_m_phase1_le_ge_range_feasible");
    }

    /// Ge b=0 (initial_basis に surplus が直接入る、artificial 不要)
    #[test]
    fn big_m_phase1_ge_b_zero_bypasses_bigm() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![0.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2], None,
        ).unwrap();
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
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 2], &[1.0, 1.0, 1.0, 1.0], 2, 3,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![0.0, 0.0, 1.0], a, vec![0.0, 1.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 3], None,
        ).unwrap();
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
        let a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0e6, 1.0, 1.0, 1.0], 2, 2,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![2.0e6, 1.0],
            vec![ConstraintType::Eq, ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2], None,
        ).unwrap();
        assert_kkt_optimal(&lp, 2.0, "big_m_phase1_large_coeff_eq_ge_mix");
    }

    /// task #43 regression: ArtificialPriorityLeaving::bland_leaving must
    /// honor Priority 2 (artificial in basis, x_B > tol). Default Bland
    /// (Priority 1 only) would mask the artificial-removal objective and
    /// return None once `x_B ≥ 0`, causing `dual_simplex_core_advanced` to
    /// declare false Optimal with artificials still in basis.
    #[test]
    fn artificial_priority_bland_picks_artificial_when_xb_nonneg() {
        use super::ArtificialPriorityLeaving;
        use crate::simplex::pricing::DualLeavingStrategy;
        let n_total = 3usize;
        let strat = ArtificialPriorityLeaving { n_total };
        let basis = vec![1usize, n_total]; // row 0: orig var, row 1: artificial
        let x_b = vec![0.5_f64, 2.0_f64];
        let pick = strat.bland_leaving(&x_b, 1e-9, &basis);
        assert_eq!(pick, Some(1), "bland_leaving must select artificial row when x_B >= 0");

        // No artificials → None
        let basis2 = vec![0usize, 1usize];
        let pick2 = strat.bland_leaving(&x_b, 1e-9, &basis2);
        assert_eq!(pick2, None);
    }

    /// task #43 regression: progress_metric must count artificial-removal
    /// progress; otherwise `best_infeas = 0` for any Big-M Phase I starting
    /// from `x_B = b ≥ 0`, threshold = 0, and bland_mode triggers after
    /// k_trigger iterations regardless of genuine progress.
    #[test]
    fn artificial_priority_progress_metric_includes_artificial_sum() {
        use super::ArtificialPriorityLeaving;
        use crate::simplex::pricing::DualLeavingStrategy;
        let n_total = 2usize;
        let strat = ArtificialPriorityLeaving { n_total };
        let basis = vec![0usize, n_total]; // row 1: artificial
        let x_b = vec![3.0_f64, 5.0_f64];
        // sum_neg = 0, art_sum = 5.0
        assert!((strat.progress_metric(&x_b, &basis) - 5.0).abs() < 1e-12);

        // After driving artificial out
        let basis2 = vec![0usize, 1usize];
        assert!(strat.progress_metric(&x_b, &basis2) < 1e-12);
    }

    /// task #43 regression: Big-M Phase I で bland_mode が誤起動しても false
    /// Infeasible を返してはいけない。小規模 Eq-only feasible LP で
    /// `assert_kkt_optimal` が Infeasible 戻り値で panic することを利用。
    #[test]
    fn big_m_phase1_no_false_infeasible_when_blandmode_triggers() {
        let a = CscMatrix::from_triplets(
            &[0, 0, 0, 1, 1, 1, 2, 2, 2],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[5.0, 3.0, 2.0, 2.0, 7.0, 1.0, 1.0, 1.0, 1.0],
            3, 3,
        ).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0, 1.0], a, vec![10.0, 5.0, 3.0],
            vec![ConstraintType::Eq; 3],
            vec![(0.0, f64::INFINITY); 3], None,
        ).unwrap();
        assert_kkt_optimal(&lp, 3.0, "big_m_phase1_no_false_infeasible_when_blandmode_triggers");
    }

    /// 自由変数 + Eq: split-variable + Phase I の組合せで feasibility が崩れないか。
    #[test]
    fn big_m_phase1_free_var_eq() {
        // x1 + x2 = 2, x1 free, x2 in [0, INF), min x1+x2
        // → x1=2-x2, obj = 2 (任意の feasible で)
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0], a, vec![2.0],
            vec![ConstraintType::Eq],
            vec![(f64::NEG_INFINITY, f64::INFINITY), (0.0, f64::INFINITY)], None,
        ).unwrap();
        assert_kkt_optimal(&lp, 2.0, "big_m_phase1_free_var_eq");
    }
}
