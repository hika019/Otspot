//! Postsolve（逆変換）モジュール
//!
//! Presolveで縮約した問題の解を元問題の解空間に復元する。
//! PostsolveStackを逆順（LIFO）に適用する。

use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::CscMatrix;
use crate::options::SolverOptions;
use super::transforms::{PostsolveStep, PresolveResult};
use std::time::Instant;

/// Cleanup LP を構築して解き、削除行の y_i を KKT 整合に決定する。
///
/// 構造:
/// - 変数: 削除行の y_i (k 個)、constraint type の符号慣例で bound 設定
/// - 制約: 未削除列 j で rc[j] = c[j] - Σ_i A_ij y_i = 0 or ≥ 0 or ≤ 0
///   - x[j] at lb: rc ≥ 0 → Σ A_ij y_i ≤ rc_known[j]
///   - x[j] at ub: rc ≤ 0 → Σ A_ij y_i ≥ rc_known[j]
///   - interior:   rc = 0 → Σ A_ij y_i = rc_known[j]
/// - 目的: min 0 (Phase I 風実行可能性のみ)
/// - LinearSubstitution の y_piv: free 変数 rc=0 を Eq 制約で含める (orig_col の rc が 0)
///
/// 戻り値: 削除行ごとの y_i 値 (None なら cleanup LP 構築失敗 or 解けず)
fn build_and_solve_cleanup_lp(
    orig_problem: &LpProblem,
    presolve_result: &PresolveResult,
    solution: &[f64],
    dual_solution_known: &[f64],
    deadline: Option<Instant>,
) -> Option<Vec<f64>> {
    // Inherit the parent deadline. If the parent has already lapsed, bail out
    // immediately and let the Gauss-Seidel fallback handle dual recovery.
    // When the parent has no deadline (Default options / interactive callers
    // that opted into unbounded runtime), we let the cleanup LP run without a
    // budget — the previous behaviour and required for the KKT-accuracy unit
    // tests in tests/diag_afiro_y.rs.
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return None;
        }
    }
    let n = orig_problem.num_vars;
    let m = orig_problem.num_constraints;
    let deleted_rows: Vec<usize> = (0..m)
        .filter(|&i| presolve_result.row_map[i].is_none())
        .collect();
    let k = deleted_rows.len();
    if k == 0 { return None; }

    let row_to_var: std::collections::HashMap<usize, usize> = deleted_rows
        .iter().enumerate().map(|(idx, &r)| (r, idx)).collect();

    // rc_known[j] = c[j] - Σ_{i: known} A_ij * ŷ_i
    let mut rc_known = orig_problem.c.clone();
    for j in 0..n {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (kk, &row) in rows.iter().enumerate() {
                if presolve_result.row_map[row].is_some() {
                    rc_known[j] -= vals[kk] * dual_solution_known[row];
                }
            }
        }
    }

    // 制約構築
    let mut tri_rows: Vec<usize> = Vec::new();
    let mut tri_cols: Vec<usize> = Vec::new();
    let mut tri_vals: Vec<f64> = Vec::new();
    let mut b_clean: Vec<f64> = Vec::new();
    let mut ct_clean: Vec<ConstraintType> = Vec::new();

    // (i) 未削除列 j の rc 符号制約
    for j in 0..n {
        let x_j = solution[j];
        let (lb_j, ub_j) = orig_problem.bounds[j];
        let at_lb = lb_j.is_finite() && (x_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        let at_ub = ub_j.is_finite() && (x_j - ub_j).abs() < BOUND_ACTIVE_TOL;
        let fixed = lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        if fixed { continue; }

        // 列 j の削除行エントリのみ抽出
        let mut col_terms: Vec<(usize, f64)> = Vec::new();
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (kk, &row) in rows.iter().enumerate() {
                if let Some(&var_idx) = row_to_var.get(&row) {
                    col_terms.push((var_idx, vals[kk]));
                }
            }
        }
        if col_terms.is_empty() { continue; }

        // 元の comment は「interior 列は退化で rc!=0 になり得るためスキップ」
        // としていたが、これは perold col 229 のように **削除行 84 にしか entry
        // を持たない非 bound-active 列** で削除行の y を under-determined にする。
        // LP 最適性 (complementary slackness):
        //   x[j] interior (basis 変数): rc[j] = 0
        //   x[j] at lb (non-basis):     rc[j] >= 0
        //   x[j] at ub (non-basis):     rc[j] <= 0
        // cleanup LP は Phase I slack で常に feasible 化されるので、interior 列を
        // Eq として含めても infeasible にはならない。primal が真に最適なら slack
        // も 0、退化していれば slack が違反量だけ非ゼロ。
        let ct = if at_lb && !at_ub {
            ConstraintType::Le
        } else if at_ub && !at_lb {
            ConstraintType::Ge
        } else {
            // interior (両 bound infinite or 両端非 active): rc = 0
            ConstraintType::Eq
        };
        let row_idx = b_clean.len();
        for &(var_idx, a) in &col_terms {
            tri_rows.push(row_idx);
            tri_cols.push(var_idx);
            tri_vals.push(a);
        }
        b_clean.push(rc_known[j]);
        ct_clean.push(ct);
    }

    // (ii) LinearSubstitution の y_piv (orig_col の rc = 0 = free 変数最適性)
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::LinearSubstitution { orig_col, .. } = step {
            let j = *orig_col;
            // rc[orig_col] = c[orig_col] - Σ_i A_{i,orig_col} * y_i = 0
            //   ⇔ Σ_{i: deleted} A_{i,j} * y_i = rc_known[j] (free 変数 KKT)
            let mut col_terms: Vec<(usize, f64)> = Vec::new();
            if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
                for (kk, &row) in rows.iter().enumerate() {
                    if let Some(&var_idx) = row_to_var.get(&row) {
                        col_terms.push((var_idx, vals[kk]));
                    }
                }
            }
            if col_terms.is_empty() { continue; }
            let row_idx = b_clean.len();
            for &(var_idx, a) in &col_terms {
                tri_rows.push(row_idx);
                tri_cols.push(var_idx);
                tri_vals.push(a);
            }
            b_clean.push(rc_known[j]);
            ct_clean.push(ConstraintType::Eq);
        }
    }

    if b_clean.is_empty() { return None; }

    // Phase I 風 slack 緩和で cleanup LP を常に feasible 化。
    // 各制約に slack 変数を追加し、目的を min Σ s に変更:
    //   Le: Σ a*y - s ≤ rc, s ≥ 0
    //   Ge: Σ a*y + s ≥ rc, s ≥ 0
    //   Eq: Σ a*y + s_pos - s_neg = rc, s_pos/s_neg ≥ 0
    // 最適値 0 なら厳密に rc 符号制約を満たす y が存在、0 でなければ違反量。
    let m_clean = b_clean.len();
    let mut slack_count = 0usize; // 追加 slack 変数の数 (Eq は 2、Le/Ge は 1)
    let mut slack_cols_per_row: Vec<(usize, Option<usize>)> = Vec::with_capacity(m_clean); // (s_idx, optional s_neg_idx for Eq)
    for ct in &ct_clean {
        match ct {
            ConstraintType::Eq => {
                let pos = k + slack_count;
                let neg = k + slack_count + 1;
                slack_cols_per_row.push((pos, Some(neg)));
                slack_count += 2;
            }
            _ => {
                slack_cols_per_row.push((k + slack_count, None));
                slack_count += 1;
            }
        }
    }
    // 各 slack 列を A に追加 (係数 ±1)
    for (row_idx, (s_pos, s_neg_opt)) in slack_cols_per_row.iter().enumerate() {
        let sign = match ct_clean[row_idx] {
            ConstraintType::Le => -1.0,  // a*y - s <= rc
            ConstraintType::Ge => 1.0,   // a*y + s >= rc
            ConstraintType::Eq => 1.0,   // a*y + s_pos - s_neg = rc
        };
        tri_rows.push(row_idx);
        tri_cols.push(*s_pos);
        tri_vals.push(sign);
        if let Some(s_neg) = s_neg_opt {
            tri_rows.push(row_idx);
            tri_cols.push(*s_neg);
            tri_vals.push(-1.0);
        }
    }
    let total_vars = k + slack_count;

    // 変数 bound: y は constraint type 符号慣例、slack は [0, ∞)
    let mut bounds_clean: Vec<(f64, f64)> = Vec::with_capacity(total_vars);
    for &i in &deleted_rows {
        match orig_problem.constraint_types[i] {
            ConstraintType::Le => bounds_clean.push((f64::NEG_INFINITY, 0.0)),
            ConstraintType::Ge => bounds_clean.push((0.0, f64::INFINITY)),
            ConstraintType::Eq => bounds_clean.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }
    for _ in 0..slack_count {
        bounds_clean.push((0.0, f64::INFINITY));
    }

    // 目的: min Σ slack (y の係数は 0)
    let mut c_clean = vec![0.0f64; total_vars];
    for j in k..total_vars { c_clean[j] = 1.0; }

    let a_clean = CscMatrix::from_triplets(
        &tri_rows, &tri_cols, &tri_vals, m_clean, total_vars
    ).ok()?;
    // Phase 2 でも b_clean / ct_clean を参照するため clone を保持
    let b_clean_keep = b_clean.clone();
    let ct_clean_keep = ct_clean.clone();
    let cleanup_lp = LpProblem::new_general(
        c_clean, a_clean, b_clean, ct_clean, bounds_clean, None
    ).ok()?;

    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.warm_start = None;
    // Inherit the parent deadline directly. The previous implementation set
    // `timeout_secs = CLEANUP_LP_TIMEOUT_SECS` (5 s, magic number), but that
    // value was only converted to `deadline` *inside* `simplex::solve_with`
    // after Ruiz scaling / standard-form build, so a 96k × 322k cleanup LP
    // (ken-18) easily spent minutes in setup before the budget was enforced.
    // Wiring the parent deadline through directly makes every inner step
    // (parse, scale, factorize, simplex iterate) check the same clock.
    // `deadline = None` is passed through unchanged — callers without a
    // budget opted into unbounded runtime.
    opts.deadline = deadline;
    let r1 = crate::simplex::solve_without_presolve(&cleanup_lp, &opts);
    let _ = (slack_count, m_clean);
    if r1.status != SolveStatus::Optimal || r1.solution.len() != total_vars {
        // Phase 1 失敗 → 上位の Gauss-Seidel フォールバックに任せる。
        return None;
    }
    let y_phase1: Vec<f64> = r1.solution[..k].to_vec();
    let slack_phase1: Vec<f64> = r1.solution[k..].to_vec();

    // -------------------------------------------------------------------------
    // Phase 2: タイ崩し (task #15)
    //
    // Phase 1 の cleanup LP は `min Σ slack` で feasibility 最大化のみ求める
    // ため、複数の y が同じ optimal slack を達成する **dual 退化** ケース
    // (perold col 229 / greenbea col 2741 等) で y は under-determined になる。
    // simplex のタイ崩しは LP 標準形の pivoting 順序依存で、|y| が極端に大きい
    // (perold で y=-30, greenbea で y=-3664) 別解を採用してしまい、後段で
    // dfeas を悪化させていた。
    //
    // 対処: Phase 2 で `min Σ |y_i - y_ref[i]|` を解く。slack は Phase 1 の
    // optimal 値で **固定** することで feasible 領域を維持し、その中で y を
    // y_ref (= postsolve loop の局所 KKT 復元 = dual_solution_known) に最も
    // 近づける。
    //
    // 構造的設計 (magic number 排除):
    //   - ε による重み付け (1 phase) ではなく Phase 2 LP を別解として解く
    //   - tie-break は別 LP の hard objective なので scale 依存なし
    //   - 計算量: cleanup LP 1 回追加 (Phase 2)。Phase 1 と同サイズ程度
    //
    // 旧 3-way 比較 (y_loop / y_gs / y_cl の dfeas 最小選択) は不要になる
    // (task #15 commit で撤廃予定)。
    //
    // 変数:  y_i (k 個、constraint type 符号 bound) + d_pos[i], d_neg[i]
    //        (各 k 個、>= 0)。合計 3k 個。
    // 制約:  (i) Phase 1 と同じ a*y 制約だが slack は Phase 1 値で吸収:
    //          Le: Σ a*y <= rc + slack_phase1[r]    (slack 抜き、RHS 緩和)
    //          Ge: Σ a*y >= rc - slack_phase1[r]
    //          Eq: Σ a*y = rc + slack_pos* - slack_neg*
    //       (ii) Tie-break Eq 行 (k 個): y_i - d_pos[i] + d_neg[i] = y_ref[i]
    // 目的:  min Σ (d_pos[i] + d_neg[i]) (= |y_i - y_ref[i]| を最小化)
    // タイ崩しの参照値 y_ref:
    //   y_gs (Gauss-Seidel 後) は kept 行で simplex y を保持し、deleted 行は
    //   local 復元値 (多くは 0)。これを y_ref にすると Phase 2 が deleted 行
    //   を 0 寄せ、kept 行は変化なし (cleanup LP は deleted 行のみ最適化なので
    //   無関係)。ゆえに y_ref = 0 と等価。明示的に 0 にする方が意図が明確。
    let y_ref: Vec<f64> = vec![0.0; k];
    let phase2_total_vars = 3 * k;
    let phase2_total_cons = m_clean + k;
    let mut p2_tri_rows: Vec<usize> = Vec::with_capacity(tri_rows.len() + 3 * k);
    let mut p2_tri_cols: Vec<usize> = Vec::with_capacity(tri_rows.len() + 3 * k);
    let mut p2_tri_vals: Vec<f64> = Vec::with_capacity(tri_rows.len() + 3 * k);
    let mut p2_b: Vec<f64> = Vec::with_capacity(phase2_total_cons);
    let mut p2_ct: Vec<ConstraintType> = Vec::with_capacity(phase2_total_cons);
    // (i) Phase 1 の a*y 制約を slack 抜き形で複製、RHS は Phase 1 slack で緩和
    for (orig_idx, (slack_pos, slack_neg_opt)) in slack_cols_per_row.iter().enumerate() {
        // a*y のエントリは tri_rows/tri_cols/tri_vals から orig_idx 行を抽出
        for (k_t, &row) in tri_rows.iter().enumerate() {
            if row != orig_idx { continue; }
            let col = tri_cols[k_t];
            if col >= k { continue; } // slack 列はスキップ
            p2_tri_rows.push(orig_idx);
            p2_tri_cols.push(col);
            p2_tri_vals.push(tri_vals[k_t]);
        }
        let s_p_val = slack_phase1[*slack_pos - k];
        // Phase 1 形:
        //   Le: a*y - s = rc 制約に変換可能 (slack 入り)、a*y = rc + s
        //   Ge: a*y + s = rc, a*y = rc - s
        //   Eq: a*y + s_pos - s_neg = rc, a*y = rc - s_pos + s_neg
        // Phase 2 (slack 固定): a*y を Phase 1 optimal の取れる範囲に縛る
        let rhs = match ct_clean_keep[orig_idx] {
            ConstraintType::Le => b_clean_keep[orig_idx] + s_p_val, // a*y <= rc + s
            ConstraintType::Ge => b_clean_keep[orig_idx] - s_p_val, // a*y >= rc - s
            ConstraintType::Eq => {
                let s_n_val = slack_phase1[slack_neg_opt.unwrap() - k];
                b_clean_keep[orig_idx] - s_p_val + s_n_val          // a*y = rc - s_p + s_n
            }
        };
        p2_b.push(rhs);
        p2_ct.push(ct_clean_keep[orig_idx].clone());
    }
    // (ii) Tie-break Eq 制約: y_i - d_pos[i] + d_neg[i] = y_ref[i]
    for i in 0..k {
        let row_idx = m_clean + i;
        p2_tri_rows.push(row_idx); p2_tri_cols.push(i);             p2_tri_vals.push(1.0);
        p2_tri_rows.push(row_idx); p2_tri_cols.push(k + i);         p2_tri_vals.push(-1.0);
        p2_tri_rows.push(row_idx); p2_tri_cols.push(2 * k + i);     p2_tri_vals.push(1.0);
        p2_b.push(y_ref[i]);
        p2_ct.push(ConstraintType::Eq);
    }
    // Phase 2 bounds: y は元 constraint type 符号、d_pos/d_neg は >= 0
    let mut p2_bounds: Vec<(f64, f64)> = Vec::with_capacity(phase2_total_vars);
    for &i in &deleted_rows {
        match orig_problem.constraint_types[i] {
            ConstraintType::Le => p2_bounds.push((f64::NEG_INFINITY, 0.0)),
            ConstraintType::Ge => p2_bounds.push((0.0, f64::INFINITY)),
            ConstraintType::Eq => p2_bounds.push((f64::NEG_INFINITY, f64::INFINITY)),
        }
    }
    for _ in 0..(2 * k) { p2_bounds.push((0.0, f64::INFINITY)); }
    // Phase 2 objective: min Σ (d_pos + d_neg)
    let mut p2_c = vec![0.0f64; phase2_total_vars];
    for j in k..(3 * k) { p2_c[j] = 1.0; }

    let p2_a = match CscMatrix::from_triplets(
        &p2_tri_rows, &p2_tri_cols, &p2_tri_vals, phase2_total_cons, phase2_total_vars
    ) {
        Ok(m) => m,
        Err(_) => return Some(y_phase1), // Phase 2 build 失敗 → Phase 1 採用
    };
    let p2_lp = match LpProblem::new_general(p2_c, p2_a, p2_b, p2_ct, p2_bounds, None) {
        Ok(l) => l,
        Err(_) => return Some(y_phase1),
    };
    let r2 = crate::simplex::solve_without_presolve(&p2_lp, &opts);
    if r2.status == SolveStatus::Optimal && r2.solution.len() == phase2_total_vars {
        Some(r2.solution[..k].to_vec())
    } else {
        // Phase 2 失敗 → Phase 1 採用 (3-way 比較に頼る)
        Some(y_phase1)
    }
}

/// CSC 形式の orig_problem.a から行 i のエントリ (j, A_ij) を列挙する。
/// O(nnz_total) の走査だが、一度だけ呼ばれる小規模問題用 (大規模では別途キャッシュ化)。
fn collect_row_entries(orig_problem: &LpProblem, i: usize) -> Vec<(usize, f64)> {
    let mut out = Vec::new();
    for j in 0..orig_problem.num_vars {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row == i {
                    out.push((j, vals[k]));
                }
            }
        }
    }
    out
}

/// bound active 判定の許容差 (x[j] が lb / ub にどれだけ近ければ active と見るか)。
const BOUND_ACTIVE_TOL: f64 = 1e-6;

/// 削除行 i の y_i を LP dual feasibility に整合的に復元する。
///
/// LP KKT (bound 考慮):
///   x[j] at lb (lb finite, |x-lb|<TOL): rc[j] >= 0 が必要
///   x[j] at ub (ub finite, |x-ub|<TOL): rc[j] <= 0 が必要
///   x[j] interior: rc[j] ≈ 0 が必要
///   x[j] fixed (lb==ub): rc[j] 任意
///
/// rc[j] = rc_at_y0[j] - A_ij * y_i (y_i 以外の y は確定済み) について、
/// 各列の必要符号から y_i の許容範囲 [min_y_i, max_y_i] を導出し、
/// 制約タイプ (Le: y<=0, Ge: y>=0, Eq: free) と交差して 0 に最も近い値を選ぶ。
fn recover_removed_row_dual(
    orig_problem: &LpProblem,
    i: usize,
    solution: &[f64],
    dual_solution: &[f64],
) -> f64 {
    let row_entries = collect_row_entries(orig_problem, i);
    let mut min_y_i = f64::NEG_INFINITY;
    let mut max_y_i = f64::INFINITY;
    for &(j, a_ij) in &row_entries {
        if a_ij.abs() < f64::EPSILON { continue; }
        let mut rc_at_y0 = orig_problem.c[j];
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                rc_at_y0 -= vals[k] * dual_solution[row];
            }
        }
        // bound active 判定 (x[j] が lb / ub のどちらに hit しているか)
        let x_j = solution[j];
        let (lb_j, ub_j) = orig_problem.bounds[j];
        let at_lb = lb_j.is_finite() && (x_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        let at_ub = ub_j.is_finite() && (x_j - ub_j).abs() < BOUND_ACTIVE_TOL;
        let fixed = lb_j.is_finite() && ub_j.is_finite() && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
        if fixed { continue; } // 固定変数の rc は LP 上自由
        // rc[j] = rc_at_y0 - a_ij * y_i に対する制約:
        //   at_lb && !at_ub: rc >= 0 → a_ij * y_i <= rc_at_y0
        //   at_ub && !at_lb: rc <= 0 → a_ij * y_i >= rc_at_y0
        //   interior:        rc == 0 → a_ij * y_i == rc_at_y0
        let bound_val = rc_at_y0 / a_ij;
        if at_lb && !at_ub {
            if a_ij > 0.0 {
                if bound_val < max_y_i { max_y_i = bound_val; }
            } else {
                if bound_val > min_y_i { min_y_i = bound_val; }
            }
        } else if at_ub && !at_lb {
            // 不等号が逆方向
            if a_ij > 0.0 {
                if bound_val > min_y_i { min_y_i = bound_val; }
            } else {
                if bound_val < max_y_i { max_y_i = bound_val; }
            }
        } else {
            // interior or 両端 hit: rc == 0 (等号制約)
            if bound_val < max_y_i { max_y_i = bound_val; }
            if bound_val > min_y_i { min_y_i = bound_val; }
        }
    }
    let (sign_lb, sign_ub) = match orig_problem.constraint_types[i] {
        ConstraintType::Le => (f64::NEG_INFINITY, 0.0),
        ConstraintType::Ge => (0.0, f64::INFINITY),
        ConstraintType::Eq => (f64::NEG_INFINITY, f64::INFINITY),
    };
    let lb_y = sign_lb.max(min_y_i);
    let ub_y = sign_ub.min(max_y_i);
    if lb_y <= ub_y {
        if lb_y <= 0.0 && ub_y >= 0.0 { 0.0 }
        else if ub_y < 0.0 { ub_y }
        else { lb_y }
    } else {
        0.0
    }
}

/// 縮約後の解を元問題の解空間に復元する。
///
/// # 引数
/// * `result` - 縮約後問題の SolverResult
/// * `presolve_result` - Presolve時に記録した変換情報
/// * `orig_problem` - 元の（縮約前の）LP問題（slack再計算に使用）
///
/// # 戻り値
/// 元問題の変数・制約数に合わせた SolverResult
pub fn run_postsolve(
    result: &SolverResult,
    presolve_result: &PresolveResult,
    orig_problem: &LpProblem,
    deadline: Option<Instant>,
) -> SolverResult {
    let n = presolve_result.orig_num_vars;
    let m = presolve_result.orig_num_constraints;

    // 縮約後問題の解を元変数空間に展開
    let mut solution = vec![0.0f64; n];
    let mut dual_solution = vec![0.0f64; m];

    // 縮約後のインデックスから元インデックスへ値をコピー
    for (j, &maybe_jj) in presolve_result.col_map.iter().enumerate() {
        if let Some(jj) = maybe_jj {
            if jj < result.solution.len() {
                solution[j] = result.solution[jj];
            }
        }
    }
    for (i, &maybe_ii) in presolve_result.row_map.iter().enumerate() {
        if let Some(ii) = maybe_ii {
            if ii < result.dual_solution.len() {
                dual_solution[i] = result.dual_solution[ii];
            }
        }
    }

    // PostsolveStack を逆順に適用して削除変数・制約を復元
    for step in presolve_result.postsolve_stack.iter().rev() {
        match step {
            PostsolveStep::FixedVariable { orig_col, value } => {
                solution[*orig_col] = *value;
            }
            PostsolveStep::EmptyColumn { orig_col, value } => {
                solution[*orig_col] = *value;
            }
            PostsolveStep::EmptyRow { orig_row } => {
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::SingletonRow { orig_col, orig_row, value, a_ij: _, c_j: _ } => {
                solution[*orig_col] = *value;
                // y_i を KKT 整合に復元 (RedundantConstraint と同様)。
                // bound active 時も含めて rc[j]>=0 を満たす y_i を選ぶ。
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::RedundantConstraint { orig_row } => {
                dual_solution[*orig_row] = recover_removed_row_dual(orig_problem, *orig_row, &solution, &dual_solution);
            }
            PostsolveStep::BoundsTightened { .. } => {
                // Bounds tightening は解の値そのものに影響しない（情報保持のみ）
            }
            PostsolveStep::LinearSubstitution {
                orig_col,
                orig_row,
                pivot,
                rhs,
                others,
                col_orig_entries,
                c_orig,
            } => {
                // --- Primal 復元: x_j = (rhs - Σ coeff_k * x_k) / pivot ---
                let mut sum_others = 0.0f64;
                for &(other_col, coeff) in others {
                    sum_others += coeff * solution[other_col];
                }
                solution[*orig_col] = (rhs - sum_others) / pivot;

                // --- Dual 復元: 消去された Eq 行 piv_row の y_piv ---
                // LinearSubstitution は free 変数 (R6/R15/R5) を Eq 行から消去する変換。
                // free 変数の最適性条件 rc[j] = 0 から:
                //   c_j_orig = Σ_i A_ij * y_i
                //   → y_piv = (c_orig - Σ_{i ≠ piv_row} A_ij * y_i) / pivot
                // col_orig_entries は分配前 (= 行 i の x_j 係数を 0 化する前) の
                // active な (i, A_ij^intermediate) snapshot (piv_row 以外)。
                if let Some(piv_row) = orig_row {
                    let mut sum_other_rows = 0.0f64;
                    for &(row_i, a_ij) in col_orig_entries {
                        if row_i == *piv_row {
                            continue;
                        }
                        sum_other_rows += a_ij * dual_solution[row_i];
                    }
                    dual_solution[*piv_row] = (c_orig - sum_other_rows) / pivot;
                }
            }
        }
    }

    // slack を元問題 b - Ax で再計算
    let mut slack = orig_problem.b.clone();
    for (j, &sol_j) in solution.iter().enumerate().take(n) {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * sol_j;
            }
        }
    }

    // Step 3: 2 経路 (Gauss-Seidel と Cleanup LP) で削除行の y 復元を計算し、
    // bound-aware dfeas (LP dual feasibility 違反 sup ノルム) が小さい方を採用する。
    //
    // 設計動機 (2026-05-17 task #10 真因対処):
    //   perold (col 229, c[j]=0, SingletonRow 経由) は cleanup LP が
    //   Phase I slack 最小化のタイ崩しで y=0 解 (KKT 整合) を捨て、
    //   別解 (y!=0、dual infeasible) を採用していた。LP optimum で y が
    //   一意でない (dual 退化) ケースでは、cleanup LP の解は dual feasible
    //   とは限らないため、Gauss-Seidel 経路と比較して KKT 整合性で勝った方を
    //   採用する。
    //
    // 旧実装は cleanup LP が解けたら無条件採用、失敗時のみ Gauss-Seidel に
    // フォールバックしていた。これは "cleanup LP 解は y_prev より厳格に良い"
    // 前提に立っていたが、perold で反例があった (dual 退化)。

    // (A) Loop 経路: postsolve reverse loop の直後の y (一度きり)。
    let y_loop = dual_solution.clone();

    // (A') Gauss-Seidel 経路: y_loop を起点に recover_removed_row_dual を反復収束。
    //     LinearSubstitution y_piv も同時に更新する。複数 deleted 行が結合した
    //     ケース (agg / scorpion 等) で反復改善する。
    //     大規模 LP (ken-18: 95k 削除行 × O(n) recover/row × 50 iter ≒ 10^11 ops) で
    //     postsolve が parent deadline を完全に無視して数百秒走る事故が発生していたため、
    //     outer loop 先頭と内側 1024 行ごとに deadline 確認し超過時は break する。
    let y_gs = {
        let mut y = y_loop.clone();
        let mut linsub_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for step in &presolve_result.postsolve_stack {
            if let PostsolveStep::LinearSubstitution { orig_row: Some(r), .. } = step {
                linsub_rows.insert(*r);
            }
        }
        let max_iter = 50;
        let conv_tol = 1e-12;
        'gs_outer: for _ in 0..max_iter {
            if deadline.is_some_and(|d| Instant::now() >= d) { break 'gs_outer; }
            let mut max_diff = 0.0f64;
            for i in 0..m {
                if presolve_result.row_map[i].is_some() { continue; }
                if linsub_rows.contains(&i) { continue; }
                if i & 0x3ff == 0 && deadline.is_some_and(|d| Instant::now() >= d) {
                    break 'gs_outer;
                }
                let new_y = recover_removed_row_dual(orig_problem, i, &solution, &y);
                let diff = (y[i] - new_y).abs();
                if diff > max_diff { max_diff = diff; }
                y[i] = new_y;
            }
            for step in &presolve_result.postsolve_stack {
                if let PostsolveStep::LinearSubstitution {
                    orig_row: Some(piv),
                    col_orig_entries,
                    c_orig,
                    pivot,
                    ..
                } = step {
                    let mut sum = 0.0f64;
                    for &(row_i, a_ij) in col_orig_entries {
                        if row_i == *piv { continue; }
                        sum += a_ij * y[row_i];
                    }
                    let new_y = (c_orig - sum) / pivot;
                    let diff = (y[*piv] - new_y).abs();
                    if diff > max_diff { max_diff = diff; }
                    y[*piv] = new_y;
                }
            }
            if max_diff < conv_tol { break 'gs_outer; }
        }
        y
    };

    // (B) Cleanup LP 経路: 削除行 y を Phase I 風 slack relaxation で一括解決。
    //     y_gs を rc_known 計算に使う (反復収束済みの方が rc_known の精度高い)。
    //     deadline を継承して ken-18 のような大規模問題で暴走しないようにする。
    let y_cl: Option<Vec<f64>> = build_and_solve_cleanup_lp(
        orig_problem, presolve_result, &solution, &y_gs, deadline,
    ).map(|y_clean| {
        let deleted_rows: Vec<usize> = (0..m)
            .filter(|&i| presolve_result.row_map[i].is_none())
            .collect();
        let mut y = y_gs.clone();
        for (var_idx, &i) in deleted_rows.iter().enumerate() {
            y[i] = y_clean[var_idx];
        }
        y
    });

    // (C) 採用判定: bound-aware dfeas が小さい方を選ぶ。
    //     dfeas_bound(y) = max_j viol_j で j: bound-active かつ
    //       - lb==ub の真の固定変数は除外
    //       - bound-tightening で FixedVar 化された列は rc=0 と見なして除外
    //         (後段の FixedVar rc=0 override と整合)
    //     viol_j = max(0, -rc_j) if at_lb only, max(0, rc_j) if at_ub only,
    //              0 otherwise。bench `compute_dfeas_orig` (c69959d 以降) と同型。
    let mut bound_tightened_fixed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::FixedVariable { orig_col, .. } = step {
            let (lb, ub) = orig_problem.bounds[*orig_col];
            let truly_fixed = lb.is_finite() && ub.is_finite()
                && (ub - lb).abs() < BOUND_ACTIVE_TOL;
            if !truly_fixed {
                bound_tightened_fixed.insert(*orig_col);
            }
        }
    }
    let dfeas_bound = |y: &[f64]| -> f64 {
        let mut max_viol = 0.0f64;
        for j in 0..n {
            if bound_tightened_fixed.contains(&j) { continue; }
            let (lb_j, ub_j) = orig_problem.bounds[j];
            let fixed = lb_j.is_finite() && ub_j.is_finite()
                && (ub_j - lb_j).abs() < BOUND_ACTIVE_TOL;
            if fixed { continue; }
            let at_lb = lb_j.is_finite() && (solution[j] - lb_j).abs() < BOUND_ACTIVE_TOL;
            let at_ub = ub_j.is_finite() && (solution[j] - ub_j).abs() < BOUND_ACTIVE_TOL;
            let mut rc = orig_problem.c[j];
            if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    rc -= vals[k] * y[row];
                }
            }
            let viol = if at_lb && !at_ub { f64::max(0.0, -rc) }
                else if at_ub && !at_lb { f64::max(0.0, rc) }
                else { 0.0 };
            if viol > max_viol { max_viol = viol; }
        }
        max_viol
    };

    // (B') task #20 (greenbea coupling 解消): cleanup LP 後の y を `compute_lsq_dual_y`
    //      経由で **全体 KKT 整合** に LSQ 射影する。
    //
    //      動機: cleanup LP は **deleted rows の y のみ** を最適化するが、kept rows の y は
    //      reduced LP の最適 dual で固定するため、kept-deleted の coupling が strong な場合
    //      (greenbea row 1270 et al.) に **infeasible** (Phase 1 slack ≈ 1e5) になる。
    //      LSQ 射影は kept/deleted の境界を意識せず A^T y ≈ -c を解くので、coupling を
    //      跨いで y 全体を調整できる。
    //
    //      実装: orig_problem (LpProblem) → 一時 QpProblem (Q=0) 変換、現在の y を持つ
    //      SolverResult を渡して `compute_lsq_dual_y` を呼び、得られた y を 4 番目の候補
    //      として df_bound 比較に組み込む。
    let y_lsq: Option<Vec<f64>> = {
        // size guard (compute_lsq_dual_y 内の 50_000 と整合)
        const LSQ_DUAL_SIZE_LIMIT: usize = 50_000;
        if n + m <= LSQ_DUAL_SIZE_LIMIT && m > 0 {
            // LpProblem を QpProblem (Q=0) に変換。共有 reference 不可なので clone。
            let q_empty = CscMatrix::new(n, n);
            let qp = crate::qp::QpProblem::new(
                q_empty,
                orig_problem.c.clone(),
                orig_problem.a.clone(),
                orig_problem.b.clone(),
                orig_problem.bounds.clone(),
                orig_problem.constraint_types.clone(),
            ).ok();
            qp.and_then(|qp| {
                // 現状の最良 y (y_cl があれば y_cl、なければ y_gs) を seed として渡す。
                // compute_lsq_dual_y は target = -(Qx + c + bound_contrib) から
                // A^T y ≈ target を LDL(A·A^T) で解く。LP (Q=0) では target = -c。
                let seed = y_cl.as_ref().cloned().unwrap_or_else(|| y_gs.clone());
                let tmp_result = crate::problem::SolverResult {
                    solution: solution.clone(),
                    dual_solution: seed,
                    ..Default::default()
                };
                crate::qp::compute_lsq_dual_y(&qp, &tmp_result)
            })
        } else {
            None
        }
    };

    // (C) 4-way 比較: y_loop / y_gs / y_cl / y_lsq のうち bound-aware dfeas が最小を採用。
    let df_loop = dfeas_bound(&y_loop);
    let df_gs = dfeas_bound(&y_gs);
    let (df_cl, _has_cl) = match &y_cl {
        Some(y) => (dfeas_bound(y), true),
        None => (f64::INFINITY, false),
    };
    let (df_lsq, _has_lsq) = match &y_lsq {
        Some(y) => (dfeas_bound(y), true),
        None => (f64::INFINITY, false),
    };
    // 最小 df の y を採用。同点は loop < gs < cl < lsq の優先 (計算量の小さい順)。
    let min_df = df_loop.min(df_gs).min(df_cl).min(df_lsq);
    if df_loop == min_df {
        dual_solution = y_loop;
    } else if df_gs == min_df {
        dual_solution = y_gs;
    } else if df_cl == min_df {
        dual_solution = y_cl.expect("df_cl finite implies Some");
    } else {
        dual_solution = y_lsq.expect("df_lsq finite implies Some");
    }

    // 被縮小費用は dual_solution が確定した後に元問題で再計算する:
    //   reduced_cost[j] = c[j] - Σ_i A_ij * y_i
    // y が KKT 整合に復元されていれば rc も KKT 整合になる (e61f27b 設計)。
    let mut reduced_costs = orig_problem.c.clone();
    for (j, rc) in reduced_costs.iter_mut().enumerate().take(n) {
        if let Ok((rows, vals)) = orig_problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                *rc -= vals[k] * dual_solution[row];
            }
        }
    }
    // Presolve で **bound tightening により固定化された** 変数の rc は bound dual
    // (mu_lb - mu_ub) で吸収される自由度があり KKT 整合性を rc=0 とみなせる。旧実装は
    // y のみで rc 再計算していたため、recipe j=76/107/138 等 (orig ub=20→presolve で
    // ub=0 に tightening されて FixedVar 化) で rc≠0 が報告され bench dfeas が違反
    // 検出していた。FixedVar として stack に push された col のみ rc=0 化する
    // (orig bounds で lb==ub の真の固定変数は既存挙動を保持、test_postsolve_t2 等)。
    for step in &presolve_result.postsolve_stack {
        if let PostsolveStep::FixedVariable { orig_col, .. } = step {
            let (lb, ub) = orig_problem.bounds[*orig_col];
            let truly_fixed = lb.is_finite() && ub.is_finite()
                && (ub - lb).abs() < BOUND_ACTIVE_TOL;
            if !truly_fixed && *orig_col < n {
                reduced_costs[*orig_col] = 0.0;
            }
        }
    }

    // 目的関数値 = 縮約後 objective + presolve で除いた変数の寄与
    let objective = result.objective + presolve_result.obj_offset;

    SolverResult {
        status: result.status.clone(),
        objective,
        solution,
        dual_solution,
        reduced_costs,
        slack,
        warm_start_basis: None, // presolve と warm-start の組み合わせは未対応
        iterations: result.iterations, // task #19: 縮約後 solve の iter を引き継ぐ
        ..Default::default()
    }
}
