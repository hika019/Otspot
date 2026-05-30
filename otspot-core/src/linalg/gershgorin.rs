//! Gershgorin 円定理ベースの λ_min(Q) 下界 helper。
//!
//! ## 公式
//! 対称行列 `Q` に対し `λ_min(Q) ≥ min_j (Q[j,j] − R_j)`、
//! ただし `R_j = Σ_{k≠j} |Q[j,k]|` (j 行非対角絶対値和)。
//!
//! ## 用途
//! - α-BB 凸化 (`qp::global::bound_alpha_bb::gershgorin_alpha`):
//!   `α = max(0, max_j(R_j − Q[j,j])) / 2`
//! - IPM 慣性修正 (`qp::ipm_core::kkt::compute_inertia_correction`):
//!   `δ_ic = max(0, max_j(R_j − Q[j,j]))`
//!
//! どちらも `max(0, max_j(R_j − Q[j,j])) = max(0, −λ_min) の Gershgorin 上界`
//! を共有するため本 helper に集約する。
//!
//! ## CSC 規約
//! 入力 `Q` は full-symmetric / 上三角 / 下三角 / 非対称 mixed (片側 entry に対称
//! pair が欠ける) のいずれも許容。各 off-diag entry は `(min(r,c), max(r,c))`
//! canonical pair に集約 (dedup) され、片側だけ存在しても両側存在しても
//! 同一の `R_j` を得る。両側 entry で値が異なる場合は `max(|v|)` を採用 (PSD 側に保守的)。
//! 対角は最後に書き込まれた値を採用 (CSC 慣例で 1 列 1 対角 entry を想定)。

use crate::sparse::CscMatrix;
use std::collections::HashMap;

/// `Q` の Gershgorin λ_min 下界から、PSD 化に必要な非負シフト
/// `max(0, max_j(R_j − Q[j,j])) = max(0, −λ_min_lower(Q))` を返す。
///
/// 戻り値 0 は「Gershgorin が `λ_min ≥ 0` を保証した = 何もシフト不要」を意味する
/// (Q が真に indefinite でも Gershgorin が保守的に non-negative を返す PSD 偽陽性は
/// 別途 `is_q_psd_by_cholesky` 等で扱う; 本 helper は素の Gershgorin に専念)。
pub(crate) fn psd_shift_from_gershgorin(q: &CscMatrix) -> f64 {
    let n = q.nrows;
    if n == 0 {
        return 0.0;
    }
    let mut diag = vec![0.0_f64; n];
    // 各 off-diag entry を (min(r,c), max(r,c)) canonical pair に集約。
    // upper/lower/full/mixed-asymmetric いずれの layout でも unordered pair
    // ごとに 1 度だけ R に反映。旧 `has_upper && has_lower → 1/2` heuristic は
    // 非対称 mixed (例: (0,1) upper と (2,1) lower で対称 pair なし) を full と
    // 誤認し誤値を返したため、canonical dedup に置換。
    let mut canonical: HashMap<(usize, usize), f64> = HashMap::new();
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            let val = q.values[k];
            if row == col {
                diag[col] = val;
            } else {
                let key = if row < col { (row, col) } else { (col, row) };
                let abs_val = val.abs();
                let entry = canonical.entry(key).or_insert(0.0);
                if abs_val > *entry {
                    *entry = abs_val;
                }
            }
        }
    }
    let mut row_offdiag_sum = vec![0.0_f64; n];
    for (&(i, j), &abs_val) in canonical.iter() {
        row_offdiag_sum[i] += abs_val;
        row_offdiag_sum[j] += abs_val;
    }
    let mut shift = 0.0_f64;
    for j in 0..n {
        let lower = diag[j] - row_offdiag_sum[j];
        if lower < 0.0 {
            shift = shift.max(-lower);
        }
    }
    shift
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upper_tri(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        let rows: Vec<usize> = entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    /// Lower-triangular CSC (row >= col) entry リストから CscMatrix を作る。
    fn lower_tri(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        for &(r, c, _) in entries {
            assert!(r >= c, "lower-tri requires row >= col, got ({r},{c})");
        }
        let rows: Vec<usize> = entries.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = entries.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = entries.iter().map(|&(_, _, v)| v).collect();
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    #[test]
    fn empty_matrix_returns_zero() {
        let q = CscMatrix::new(0, 0);
        assert_eq!(psd_shift_from_gershgorin(&q), 0.0);
    }

    #[test]
    fn diagonal_psd_returns_zero() {
        let q = upper_tri(2, &[(0, 0, 1.0), (1, 1, 2.0)]);
        assert_eq!(psd_shift_from_gershgorin(&q), 0.0);
    }

    #[test]
    fn diagonal_negative_returns_abs_min_diag() {
        // diag(-2, -3) → Gershgorin lower = (-2, -3), shift = 3
        let q = upper_tri(2, &[(0, 0, -2.0), (1, 1, -3.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn pure_bilinear_zero_diag_off_one() {
        // Q=[[0,1],[1,0]]: row sums = (1,1), diag=(0,0), Gershgorin lower=(-1,-1), shift=1
        let q = upper_tri(2, &[(0, 1, 1.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn full_symmetric_input_matches_upper() {
        // Full-symmetric: both (0,1) と (1,0) を入れても上三角と同 shift
        let q_full = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
        let q_upper = upper_tri(2, &[(0, 1, 1.0)]);
        assert_eq!(
            psd_shift_from_gershgorin(&q_full),
            psd_shift_from_gershgorin(&q_upper),
        );
    }

    #[test]
    fn mixed_zero_and_negative_diag() {
        // Q=[[0,1],[1,-1]]: diag=(0,-1), row sums=(1,1), Gershgorin=(-1,-2), shift=2
        let q = upper_tri(2, &[(0, 1, 1.0), (1, 1, -1.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn extreme_offdiag_dominates_diag() {
        // Q[0,0]=2, Q[0,3]=3, rest 0 (n=4)
        // row sums = (3,0,0,3), diag=(2,0,0,0), Gershgorin=(-1,0,0,-3), shift=3
        let q = upper_tri(4, &[(0, 0, 2.0), (0, 3, 3.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn psd_with_large_offdiag_returns_positive_shift_false_alarm() {
        // PD Q=[[1,1.1],[1.1,2]]: det=0.79 > 0 だが Gershgorin は λ_min ≥ -0.1 (誤検出)
        // helper は素 Gershgorin に専念し 0.1 を返す。caller 側で PSD 判定で吸収。
        let q = upper_tri(2, &[(0, 0, 1.0), (0, 1, 1.1), (1, 1, 2.0)]);
        assert!((psd_shift_from_gershgorin(&q) - 0.1).abs() < 1e-12);
    }

    /// Lower-triangular 入力 silent failure 防御: Q=[[0,1],[1,0]] を lower (1,0)=1 のみで
    /// 渡したとき、旧実装は off-diag をすべて取り零し shift=0 を返す。新実装は両側を
    /// 走査して shift=1 を出すこと (= upper-only / full-symmetric と同値)。
    #[test]
    fn lower_triangular_only_zero_diag_bilinear_matches_upper() {
        let q_lower = lower_tri(2, &[(1, 0, 1.0)]);
        let q_upper = upper_tri(2, &[(0, 1, 1.0)]);
        assert!((psd_shift_from_gershgorin(&q_lower) - 1.0).abs() < 1e-12);
        assert_eq!(
            psd_shift_from_gershgorin(&q_lower),
            psd_shift_from_gershgorin(&q_upper),
            "lower-only と upper-only は同じ shift を返すべき (対称化)"
        );
    }

    /// Lower-triangular only 非対角支配 indefinite: Q=[[1,2],[2,1]] を lower (1,0)=2 + diag のみで
    /// 渡したとき、shift = max(0, 2-1, 2-1) = 1。旧実装は shift=0 を返した。
    #[test]
    fn lower_triangular_only_offdiag_dominant_indefinite() {
        let q_lower = lower_tri(2, &[(0, 0, 1.0), (1, 0, 2.0), (1, 1, 1.0)]);
        assert!((psd_shift_from_gershgorin(&q_lower) - 1.0).abs() < 1e-12);
    }

    /// 3 layout (upper-only / lower-only / full-symmetric) で同じ抽象 Q に対して
    /// 同一 shift を返すこと。layout 判定 + 1/2 補正が full-symmetric 退化させていない
    /// ことを示す。
    #[test]
    fn three_layouts_agree_on_indefinite_q() {
        // 抽象 Q = [[1, 2, 0],[2, 1, -1],[0, -1, 1]]
        // 真 row sums = (2, 3, 1), diag = (1, 1, 1) → Gershgorin lower = (-1, -2, 0), shift = 2
        let upper = upper_tri(
            3,
            &[
                (0, 0, 1.0),
                (0, 1, 2.0),
                (1, 1, 1.0),
                (1, 2, -1.0),
                (2, 2, 1.0),
            ],
        );
        let lower = lower_tri(
            3,
            &[
                (0, 0, 1.0),
                (1, 0, 2.0),
                (1, 1, 1.0),
                (2, 1, -1.0),
                (2, 2, 1.0),
            ],
        );
        // full-symmetric: triplets を CscMatrix が col 並べ替えするので順序自由
        let full = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 1, 2, 2],
            &[0, 1, 0, 1, 2, 1, 2],
            &[1.0, 2.0, 2.0, 1.0, -1.0, -1.0, 1.0],
            3,
            3,
        )
        .unwrap();
        let s_upper = psd_shift_from_gershgorin(&upper);
        let s_lower = psd_shift_from_gershgorin(&lower);
        let s_full = psd_shift_from_gershgorin(&full);
        assert!((s_upper - 2.0).abs() < 1e-12, "upper shift = {s_upper}");
        assert!((s_lower - 2.0).abs() < 1e-12, "lower shift = {s_lower}");
        assert!((s_full - 2.0).abs() < 1e-12, "full shift = {s_full}");
    }

    /// 非対称 mixed layout: (0,1) upper と (2,1) lower で対称 pair なし。
    /// 抽象 Q = [[0,1,0],[1,0,1],[0,1,0]] (symmetric 化想定) → diag=(0,0,0),
    /// R=(1,2,1), Gershgorin lower=(-1,-2,-1), shift=2。
    /// 旧 `has_upper && has_lower → 1/2` heuristic は full と誤認し shift=1 (誤値) を返した。
    #[test]
    fn mixed_asymmetric_no_pair_canonicalizes() {
        let q = CscMatrix::from_triplets(&[0, 2], &[1, 1], &[1.0, 1.0], 3, 3).unwrap();
        let s = psd_shift_from_gershgorin(&q);
        assert!(
            (s - 2.0).abs() < 1e-12,
            "mixed-asymm shift = {s} (期待 2.0)"
        );
    }

    /// 非対称 mixed の正規化が full-symmetric 入力と一致することを示す。
    /// 上記 mixed 入力に欠けている対称 pair (1,0) と (1,2) を足した full 入力で
    /// 同一 shift を返すこと = canonical 化が完全。
    #[test]
    fn mixed_asymmetric_matches_full_symmetric() {
        let q_mixed = CscMatrix::from_triplets(&[0, 2], &[1, 1], &[1.0, 1.0], 3, 3).unwrap();
        let q_full =
            CscMatrix::from_triplets(&[0, 1, 2, 1], &[1, 0, 1, 2], &[1.0, 1.0, 1.0, 1.0], 3, 3)
                .unwrap();
        let s_mixed = psd_shift_from_gershgorin(&q_mixed);
        let s_full = psd_shift_from_gershgorin(&q_full);
        assert!(
            (s_mixed - s_full).abs() < 1e-12,
            "mixed={s_mixed} vs full={s_full}"
        );
    }

    /// 両側 entry で値不一致時は max(|v|) を採用 (PSD 側に保守的) すること。
    /// (0,1)=1.0 と (1,0)=3.0 → canonical (0,1) で max=3.0 → R=(3,3), shift=3。
    #[test]
    fn asymmetric_value_pair_takes_max_abs() {
        let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 3.0], 2, 2).unwrap();
        let s = psd_shift_from_gershgorin(&q);
        assert!((s - 3.0).abs() < 1e-12, "max-abs shift = {s} (期待 3.0)");
    }

    /// no-op proof: lower-triangular 対称化 fix を撤回 (= 旧 `row < col` only) すると
    /// `lower_triangular_only_zero_diag_bilinear_matches_upper` 等がどう FAIL するかを
    /// 単一 inline で機械検証する。`feedback_sentinel_must_fail_under_noop` 準拠:
    /// 新 sentinel が実装と coupled に PASS しているのではなく、fix 削除で確実に FAIL
    /// する性質を持つことを示す。
    #[test]
    fn no_op_proof_lower_tri_symmetrize_required() {
        // 旧 impl 相当 (row < col の片半のみ反映) を inline 再現。
        fn legacy_row_lt_col_only(q: &CscMatrix) -> f64 {
            let n = q.nrows;
            if n == 0 {
                return 0.0;
            }
            let mut diag = vec![0.0_f64; n];
            let mut row_sum = vec![0.0_f64; n];
            for col in 0..n {
                for k in q.col_ptr[col]..q.col_ptr[col + 1] {
                    let row = q.row_ind[k];
                    let val = q.values[k];
                    if row == col {
                        diag[col] = val;
                    } else if row < col {
                        let abs = val.abs();
                        row_sum[row] += abs;
                        row_sum[col] += abs;
                    }
                }
            }
            let mut shift = 0.0_f64;
            for j in 0..n {
                let lower = diag[j] - row_sum[j];
                if lower < 0.0 {
                    shift = shift.max(-lower);
                }
            }
            shift
        }
        let q_lower = lower_tri(2, &[(1, 0, 1.0)]);
        // 旧 impl: lower-only entry を取り零して shift=0 (= silent failure 再現)
        let legacy = legacy_row_lt_col_only(&q_lower);
        assert_eq!(
            legacy, 0.0,
            "旧 impl は lower-only off-diag を取り零し shift=0 (bug)"
        );
        // 新 impl: 正しく shift=1
        let fixed = psd_shift_from_gershgorin(&q_lower);
        assert!(
            (fixed - 1.0).abs() < 1e-12,
            "新 impl は対称化 shift=1 を返す"
        );
        // = 新 sentinel `lower_triangular_only_*` が fix 撤回で確実に FAIL する証拠
        assert!(
            (legacy - fixed).abs() > 0.5,
            "fix の有無で挙動が乖離 (legacy={legacy}, fixed={fixed}) = sentinel が active"
        );
    }

    /// no-op proof: canonical dedup fix を撤回 (= 旧 `has_upper && has_lower → 1/2`)
    /// すると `mixed_asymmetric_no_pair_canonicalizes` が確実に FAIL することを inline で
    /// 機械検証する。`feedback_sentinel_must_fail_under_noop` 準拠。
    #[test]
    fn no_op_proof_mixed_asymmetric_canonicalize_required() {
        // 旧 impl 相当 (#66 fix 後の has_upper && has_lower → 1/2 補正) を inline 再現。
        fn legacy_has_upper_lower_half(q: &CscMatrix) -> f64 {
            let n = q.nrows;
            if n == 0 {
                return 0.0;
            }
            let mut diag = vec![0.0_f64; n];
            let mut row_sum = vec![0.0_f64; n];
            let mut has_upper = false;
            let mut has_lower = false;
            for col in 0..n {
                for k in q.col_ptr[col]..q.col_ptr[col + 1] {
                    let row = q.row_ind[k];
                    let val = q.values[k];
                    if row == col {
                        diag[col] = val;
                    } else {
                        if row < col {
                            has_upper = true;
                        } else {
                            has_lower = true;
                        }
                        let abs = val.abs();
                        row_sum[row] += abs;
                        row_sum[col] += abs;
                    }
                }
            }
            if has_upper && has_lower {
                for r in row_sum.iter_mut() {
                    *r *= 0.5;
                }
            }
            let mut shift = 0.0_f64;
            for j in 0..n {
                let lower = diag[j] - row_sum[j];
                if lower < 0.0 {
                    shift = shift.max(-lower);
                }
            }
            shift
        }
        // 非対称 mixed: (0,1)=1 upper + (2,1)=1 lower, 対称 pair なし。
        // 抽象 Q = [[0,1,0],[1,0,1],[0,1,0]] → 真 shift=2。
        let q = CscMatrix::from_triplets(&[0, 2], &[1, 1], &[1.0, 1.0], 3, 3).unwrap();
        let legacy = legacy_has_upper_lower_half(&q);
        // 旧 impl: has_upper && has_lower=true → 誤って 1/2 補正、shift=1 (誤値)
        assert!(
            (legacy - 1.0).abs() < 1e-12,
            "旧 impl は mixed-asymm を full と誤認、shift={legacy}"
        );
        let fixed = psd_shift_from_gershgorin(&q);
        assert!(
            (fixed - 2.0).abs() < 1e-12,
            "新 impl: canonical dedup で shift={fixed}"
        );
        assert!(
            (legacy - fixed).abs() > 0.5,
            "fix の有無で乖離 (legacy={legacy}, fixed={fixed}) = sentinel が active"
        );
    }
}
