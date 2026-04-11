//! QP Presolve Phase 2 モジュール（#19-21）
//!
//! Phase 1（#1-18）の縮約後問題に対してさらに高度な前処理を適用する。
//!
//! - #19: equality_constraint_qr()     — 等式制約冗長行を QR/Gaussian 消去で除去
//! - #20: near_zero_q_removal()        — Q 非対角の微小要素をゼロ化（疎性向上）
//! - #21: constraint_precond_refactor()— 制約行の前処理（行正規化）presolve に集約

use crate::options::SolverOptions;
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;
use crate::tolerances::ZERO_TOL;
use super::qp_transforms::{QpPresolveResult, QpPostsolveStep};

// ---------------------------------------------------------------------------
// #19: equality_constraint_qr — 等式制約の冗長行除去
// ---------------------------------------------------------------------------

/// 等式制約ペアを検出し、Gaussian 消去（部分ピボット）で線形独立な行を特定する。
/// 冗長な等式制約ペアを削除して問題を縮小する。
///
/// 適用条件: m > n*2 （小問題はスキップ）
/// PARAM: 適用閾値 m > n*2, 理由=QR分解コストが O(n²m) のため
///
/// 等式制約の検出: Le 制約 `A[i,*]x <= b[i]` と `A[j,*]x >= b[i]`
/// （= `-A[j,*]x <= -b[i]`）が対になる行を等式制約ペアとして認識する。
pub fn equality_constraint_qr(
    prob: &QpProblem,
    removed_rows: &mut [bool],
) {
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let n = prob.num_vars;
    let m = prob.num_constraints;

    // (C) 計算量が過大な問題では #19 をスキップして HANG を防止
    // QPLIB_8602 は m=105966, n≈1000 → 旧閾値 m>10000 でスキップ済み。
    // 計算量ベース閾値 m*n > 1e8 に変更: m=10000,n=100 の問題では実行可能になる。
    // PARAM: 閾値 1e8 = O(m_eq * n) の推定計算量ベース（Gaussian消去コスト）
    const QR_SKIP_SIZE_THRESHOLD: usize = 100_000_000;
    if m * n > QR_SKIP_SIZE_THRESHOLD || m <= n * 2 || n == 0 {
        return;
    }

    // 行ごとの非ゼロエントリを収集 (列順に格納されるため列優先でイテレート)
    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
    for j in 0..n {
        let start = prob.a.col_ptr[j];
        let end = prob.a.col_ptr[j + 1];
        for k in start..end {
            let row = prob.a.row_ind[k];
            if !removed_rows[row] {
                row_entries[row].push((j, prob.a.values[k]));
            }
        }
    }

    // (A) ハッシュベースのペア検出: O(m × nnz_per_row)
    // ペア条件: A[j,*] = -A[i,*] かつ b[j] = -b[i]
    //   → 同じ列パターン・同じ |b| の行をグループ化し、グループ内のみ比較

    // 列パターンハッシュ (値を除いた列インデックスのハッシュ)
    let col_pattern_hash = |entries: &[(usize, f64)]| -> u64 {
        let mut h = DefaultHasher::new();
        for &(col, _) in entries {
            col.hash(&mut h);
        }
        h.finish()
    };

    // グループキー: (nnz_count, col_pattern_hash, |b| 量子化)
    let mut groups: HashMap<(usize, u64, i64), Vec<usize>> = HashMap::new();
    for i in 0..m {
        if removed_rows[i] || row_entries[i].is_empty() {
            continue;
        }
        let ch = col_pattern_hash(&row_entries[i]);
        // PARAM: 1e9 — |b| 量子化スケール（経験値）。b 値の 10^-9 精度での量子化により
        // 同一制約の |b| 値が丸め誤差以内で一致する行をグループ化するためのハッシュキー。
        // HiGHS は b 値をハッシュキーに使わず係数正規化ハッシュのみ使用。本実装独自。
        // 注意: b > 9.22e9 のとき i64 飽和が発生する可能性。
        // 実用的なQP問題では通常b≤1e6程度なので実害は稀だが、スケーリング後の内部値に注意。
        // 飽和時は異なるb値の行が同一ハッシュに衝突するが、正確性への影響なし（QR消去で実値判定）。
        // 承認=家老承認済み（cmd_576）
        let bk = (prob.b[i].abs() * 1e9).round() as i64;
        groups.entry((row_entries[i].len(), ch, bk)).or_default().push(i);
    }

    // グループ内でペアを検出 (グループは通常2〜数行と小さい)
    let mut eq_pos_rows: Vec<usize> = Vec::new();
    let mut paired = vec![false; m];
    let mut pair_partner: Vec<usize> = vec![usize::MAX; m]; // i のペア相手 j

    for group in groups.values() {
        for &i in group {
            if paired[i] {
                continue;
            }
            for &j in group {
                if j <= i || paired[j] {
                    continue;
                }
                let entries_i = &row_entries[i];
                let entries_j = &row_entries[j];
                let b_i = prob.b[i];

                // A[j] = -A[i] かつ b[j] = -b[i] を厳密チェック
                if (b_i + prob.b[j]).abs() > ZERO_TOL * (1.0 + b_i.abs()) {
                    continue;
                }
                let is_neg = entries_i.iter().zip(entries_j.iter()).all(|((c1, v1), (c2, v2))| {
                    *c1 == *c2 && (v1 + v2).abs() < ZERO_TOL * (1.0 + v1.abs())
                });
                if is_neg {
                    eq_pos_rows.push(i);
                    paired[i] = true;
                    paired[j] = true;
                    pair_partner[i] = j;
                    break;
                }
            }
        }
    }

    let m_eq = eq_pos_rows.len();
    if m_eq == 0 {
        return;
    }

    // 等式制約行列 Aeq (m_eq x n) を密行列として構築
    let mut aeq = vec![vec![0.0f64; n]; m_eq];
    for (row_idx, &orig_row) in eq_pos_rows.iter().enumerate() {
        for &(col, val) in &row_entries[orig_row] {
            aeq[row_idx][col] = val;
        }
    }

    // Gaussian 消去（部分ピボット）で線形独立行を特定
    let mut pivot_rows: Vec<bool> = vec![false; m_eq];
    let mut pivot_count = 0usize;
    let mut used_pivot_col = vec![false; n];
    let mut work = aeq.clone();

    for col in 0..n {
        let mut max_val = 0.0f64;
        let mut max_row = usize::MAX;
        for row in 0..m_eq {
            if pivot_rows[row] {
                continue;
            }
            let v = work[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        // PARAM: 1e-10 — ピボット選択の最小値（実装的根拠）。EPS_Q と同値。
        // 小さすぎるピボットは数値不安定を引き起こすためスキップ。承認=家老承認済み（cmd_576）
        if max_row == usize::MAX || max_val < 1e-10 || used_pivot_col[col] {
            continue;
        }

        pivot_rows[max_row] = true;
        used_pivot_col[col] = true;
        pivot_count += 1;

        let pivot = work[max_row][col];
        for k in 0..m_eq {
            if k == max_row {
                continue;
            }
            let factor = work[k][col] / pivot;
            // PARAM: 1e-15 — ほぼゼロな因子のスキップ閾値（実装的根拠）。DROP_TOL と同値。
            // 数値誤差以下の消去は不要。承認=家老承認済み（cmd_576）
            if factor.abs() < 1e-15 {
                continue;
            }
            #[allow(clippy::needless_range_loop)]
            for c in 0..n {
                let delta = factor * work[max_row][c];
                work[k][c] -= delta;
            }
        }

        if pivot_count >= n {
            break;
        }
    }

    // 非ピボット行（冗長な等式制約ペア）を除去
    // pair_partner を使って O(m_eq) で完結（旧実装の O(m²) パートナー探索を廃止）
    for (row_idx, &orig_row) in eq_pos_rows.iter().enumerate() {
        if !pivot_rows[row_idx] {
            removed_rows[orig_row] = true;
            let partner = pair_partner[orig_row];
            if partner != usize::MAX {
                removed_rows[partner] = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// #20: near_zero_q_removal — Q 非対角微小要素のゼロ化
// ---------------------------------------------------------------------------

/// Q の非対角要素で |Q[i,j]| < eps_q のものをゼロ化する（疎性向上）。
///
/// PARAM: eps_q=1e-10, 理由=解の精度への影響は最小
pub fn near_zero_q_removal(q: &CscMatrix, n: usize) -> CscMatrix {
    const EPS_Q: f64 = 1e-10;

    let mut new_col_ptr = vec![0usize; n + 1];
    let mut new_row_ind: Vec<usize> = Vec::new();
    let mut new_values: Vec<f64> = Vec::new();

    for j in 0..n {
        let start = q.col_ptr[j];
        let end = q.col_ptr[j + 1];
        for k in start..end {
            let row = q.row_ind[k];
            let val = q.values[k];
            // 対角要素はゼロ化しない。非対角要素のみ eps_q でフィルタ。
            if row == j || val.abs() >= EPS_Q {
                new_row_ind.push(row);
                new_values.push(val);
            }
        }
        new_col_ptr[j + 1] = new_row_ind.len();
    }

    CscMatrix {
        nrows: q.nrows,
        ncols: n,
        col_ptr: new_col_ptr,
        row_ind: new_row_ind,
        values: new_values,
    }
}

// ---------------------------------------------------------------------------
// #21: constraint_precond_refactor — 制約前処理の presolve への集約
// ---------------------------------------------------------------------------

/// 制約行を行ノルムで正規化する（制約前処理の presolve への集約）。
///
/// 各行 i: σ_i = max|A[i,*]|。σ_i > 1 なら A[i,*] と b[i] を σ_i で割る。
///
/// これにより KKT 行列の数値安定性が改善し、IPM の収束が向上する。
///
/// 戻り値: 行スケール係数（逆変換に使用。双対変数 y_i *= σ_i で元スケールに戻る）
pub fn constraint_precond(
    a: &mut CscMatrix,
    b: &mut [f64],
) -> Vec<f64> {
    let m = a.nrows;
    let n = a.ncols;

    // 行ごとの max|A[i,*]|
    let mut row_max = vec![0.0f64; m];
    for col in 0..n {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            let row = a.row_ind[k];
            let v = a.values[k].abs();
            if v > row_max[row] { row_max[row] = v; }
        }
    }

    // σ_i = 1/row_max[i] for rows with row_max > 1.0
    let sigmas: Vec<f64> = row_max.iter().map(|&mx| {
        if mx > 1.0 + 1e-10 { 1.0 / mx } else { 1.0 }
    }).collect();

    let has_any = sigmas.iter().any(|&s| (s - 1.0).abs() > 1e-12);
    if !has_any {
        return sigmas;
    }

    // A の値をスケール: A[i,j] *= σ_i
    for col in 0..n {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            let row = a.row_ind[k];
            a.values[k] *= sigmas[row];
        }
    }

    // b[i] *= σ_i
    for i in 0..m {
        b[i] *= sigmas[i];
    }

    sigmas
}

// ---------------------------------------------------------------------------
// Phase 2 エントリポイント
// ---------------------------------------------------------------------------

/// QP Presolve Phase 2（#19-21 の技法を適用）
///
/// Phase 1 の `QpPresolveResult` を受け取り、縮約後問題をさらに前処理して返す。
///
/// - #19: 等式制約の QR 分解による冗長行除去
/// - #20: Q 非対角微小要素のゼロ化
/// - #21: 制約行正規化（IPM の収束改善）
pub fn run_qp_presolve_phase2(
    phase1_result: QpPresolveResult,
    _opts: &SolverOptions,
) -> QpPresolveResult {
    let prob = &phase1_result.reduced;
    let n = prob.num_vars;
    let m = prob.num_constraints;

    if n == 0 || m == 0 {
        return phase1_result;
    }

    // ==================================================================
    // #20: near_zero_q_removal() — Q 非対角要素のゼロ化
    // ==================================================================
    let q_cleaned = near_zero_q_removal(&prob.q, n);

    // ==================================================================
    // #19: equality_constraint_qr() — 等式制約の冗長行除去
    // 適用条件: m > n*2 の場合のみ実行
    // ==================================================================
    let mut removed_rows_phase2 = vec![false; m];
    {
        // #19 は prob の情報を元に removed_rows_phase2 を更新する
        // (q_cleaned は row 除去に影響しない)
        equality_constraint_qr(prob, &mut removed_rows_phase2);
    }

    // ==================================================================
    // 縮約後問題の再構築（#19 で除去した行を反映）
    // ==================================================================
    let any_removed = removed_rows_phase2.iter().any(|&b| b);

    let (a_new, b_new) = if any_removed {
        let mut new_row_map = vec![None; m];
        let mut new_row_idx = 0usize;
        for i in 0..m {
            if !removed_rows_phase2[i] {
                new_row_map[i] = Some(new_row_idx);
                new_row_idx += 1;
            }
        }
        let m_new = new_row_idx;

        // 新 A 行列（CSC）
        let mut trip_rows: Vec<usize> = Vec::new();
        let mut trip_cols: Vec<usize> = Vec::new();
        let mut trip_vals: Vec<f64> = Vec::new();
        for j in 0..n {
            let start = prob.a.col_ptr[j];
            let end = prob.a.col_ptr[j + 1];
            for k in start..end {
                let row = prob.a.row_ind[k];
                if let Some(ii) = new_row_map[row] {
                    trip_rows.push(ii);
                    trip_cols.push(j);
                    trip_vals.push(prob.a.values[k]);
                }
            }
        }
        let a_out = if trip_rows.is_empty() {
            CscMatrix::new(m_new, n)
        } else {
            CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n)
                .unwrap_or_else(|_| CscMatrix::new(m_new, n))
        };

        let b_out: Vec<f64> = (0..m)
            .filter(|&i| !removed_rows_phase2[i])
            .map(|i| prob.b[i])
            .collect();

        (a_out, b_out)
    } else {
        (prob.a.clone(), prob.b.clone())
    };

    // ==================================================================
    // #21: constraint_precond_refactor() — 制約行正規化
    // ==================================================================
    let mut a_precond = a_new;
    let mut b_precond = b_new;
    let sigmas = constraint_precond(&mut a_precond, &mut b_precond);

    // 縮約後問題を再構築（constraint_typesをremoved_rows_phase2でフィルタリング）
    let constraint_types_new: Vec<crate::problem::ConstraintType> = (0..m)
        .filter(|&i| !removed_rows_phase2[i])
        .map(|i| prob.constraint_types[i])
        .collect();
    let c_clone = prob.c.clone();
    let bounds_clone = prob.bounds.clone();
    let reduced_new = match QpProblem::new(q_cleaned, c_clone, a_precond, b_precond, bounds_clone, constraint_types_new) {
        Ok(p) => p,
        Err(_) => return phase1_result, // 構築失敗 → Phase 1 結果をそのまま返す
    };

    // constraint_precond の行スケーリングを postsolve_stack に記録（双対変数の逆変換に必要）
    // Phase1の LargeCoeffRowScale に加え、Phase2の constraint_precond も
    // postsolve_stack に積むことで、postsolve時に両方の行スケーリングが逆変換される。
    let mut result = QpPresolveResult {
        reduced: reduced_new,
        was_reduced: phase1_result.was_reduced || any_removed,
        ..phase1_result
    };
    let has_precond_scaling = sigmas.iter().any(|&s| (s - 1.0).abs() > 1e-12);
    if has_precond_scaling {
        result.postsolve_stack.push(QpPostsolveStep::LargeCoeffRowScale { row_scales: sigmas });
    }
    result
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::qp::QpProblem;
    use crate::sparse::CscMatrix;

    fn make_qp_simple(n: usize, m: usize) -> QpProblem {
        // 対角 Q=2I, c=0, A=I (truncated), b=1, bounds無限
        let q = CscMatrix::from_triplets(
            &(0..n).collect::<Vec<_>>(),
            &(0..n).collect::<Vec<_>>(),
            &vec![2.0; n],
            n, n,
        ).unwrap();
        let a_m = m.min(n);
        let a = CscMatrix::from_triplets(
            &(0..a_m).collect::<Vec<_>>(),
            &(0..a_m).collect::<Vec<_>>(),
            &vec![1.0; a_m],
            m, n,
        ).unwrap();
        let b = vec![1.0; m];
        QpProblem::new_all_le(q, vec![0.0; n], a, b, vec![(f64::NEG_INFINITY, f64::INFINITY); n]).unwrap()
    }

    #[test]
    fn test_near_zero_q_removal_removes_small_offdiag() {
        // Q = [[2.0, 1e-15], [1e-15, 2.0]] → 非対角を除去
        let q = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0, 1e-15, 1e-15, 2.0], 2, 2
        ).unwrap();
        let q_clean = near_zero_q_removal(&q, 2);
        // 非対角 (0,1)=(1,0) が除去されている
        let diag_count = q_clean.values.iter().zip(q_clean.row_ind.iter()).filter(|(_, &_r)| {
            // どの列かは不明なのでゼロ化された数を確認
            true
        }).count();
        // 非対角2要素が除去され対角2要素のみ残る
        assert_eq!(q_clean.values.len(), 2, "off-diag removed");
        let _ = diag_count;
    }

    #[test]
    fn test_constraint_precond_scales_large_rows() {
        // A行列の行1の係数が大きい場合にスケールされること
        let n = 2usize;
        let m = 2usize;
        let mut a = CscMatrix::from_triplets(
            &[0, 0, 1, 1], &[0, 1, 0, 1],
            &[1.0, 1.0, 1000.0, 1000.0],
            m, n,
        ).unwrap();
        let mut b = vec![1.0, 1000.0];
        let sigmas = constraint_precond(&mut a, &mut b);
        // 行0: max=1.0 → σ=1.0（変化なし）
        // 行1: max=1000.0 → σ=0.001
        assert!((sigmas[0] - 1.0).abs() < 1e-10, "row0 unchanged");
        assert!((sigmas[1] - 0.001).abs() < 1e-7, "row1 scaled: σ={}", sigmas[1]);
        // b[1] がスケールされていること
        assert!((b[1] - 1.0).abs() < 1e-7, "b[1] scaled: {}", b[1]);
    }

    #[test]
    fn test_run_qp_presolve_phase2_no_crash() {
        let prob = make_qp_simple(3, 2);
        let opts = SolverOptions::default();
        let phase1 = crate::presolve::run_qp_presolve_phase1(&prob, &opts);
        let _phase2 = run_qp_presolve_phase2(phase1, &opts);
        // クラッシュしなければ OK
    }

    #[test]
    fn test_equality_constraint_qr_redundant_removal() {
        // m=6, n=2: 3 等式制約ペア。うち2つは冗長（同一）。→ 1ペアのみ残す
        // 等式: x+y=1 (redundant pair: 2つ), x-y=0 (1つ)
        // Le 制約として:  x+y<=1, -(x+y)<=-1 × 2, x-y<=0, -(x-y)<=0
        // → m=6 > n*2=4 → QR 適用
        let n = 2usize;
        let m = 6usize;
        // rows 0,1: x+y<=1 と -(x+y)<=-1
        // rows 2,3: x+y<=1 と -(x+y)<=-1 (重複)
        // rows 4,5: x-y<=0 と -(x-y)<=0
        let a = CscMatrix::from_triplets(
            &[0,0, 1,1, 2,2, 3,3, 4,4, 5,5],
            &[0,1, 0,1, 0,1, 0,1, 0,1, 0,1],
            &[1.0,1.0, -1.0,-1.0, 1.0,1.0, -1.0,-1.0, 1.0,-1.0, -1.0,1.0],
            m, n,
        ).unwrap();
        let b = vec![1.0, -1.0, 1.0, -1.0, 0.0, 0.0];
        let q = CscMatrix::from_triplets(&[0,1], &[0,1], &[2.0,2.0], n, n).unwrap();
        let prob = QpProblem::new_all_le(q, vec![0.0;n], a, b, vec![(f64::NEG_INFINITY,f64::INFINITY);n]).unwrap();
        let mut removed = vec![false; m];
        equality_constraint_qr(&prob, &mut removed);
        // 少なくとも1行が除去されているべき（重複行）
        let removed_count = removed.iter().filter(|&&b| b).count();
        assert!(removed_count >= 2, "at least one redundant pair removed, got {}", removed_count);
    }
}
