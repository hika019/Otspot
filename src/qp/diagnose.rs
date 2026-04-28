//! QP 問題の事前診断 API
//!
//! `diagnose()` は `solve()` 呼び出し前に問題定義の妥当性を確認するための軽量チェック。
//! 全チェックのコストは O(nnz_Q + nnz_A + n + m)。

use super::problem::QpProblem;

// ───────────────────────── 定数 ─────────────────────────

/// Q 対角要素の負判定閾値
const DIAG_TOL: f64 = 1e-10;
/// lb > ub 判定の許容誤差
const BOUND_TOL: f64 = 1e-10;
/// スケーリング比の警告閾値
/// PARAM: 根拠=IPM の KKT 行列条件数の経験的許容上限（1e8 超で数値的に不安定化しやすい）。
///        implied bounds ガード（qp_transforms.rs:1e8）と同値だが目的は独立
///        （こちらはユーザー向け診断警告のみ。bounds 更新には不使用）。
///        承認=家老承認済み
const SCALE_WARN_THRESHOLD: f64 = 1e8;
/// A 行の「非ゼロ」判定閾値（現在は nnz==0 で判定するため未使用だが定義を保持）
#[allow(dead_code)]
const ZERO_ROW_TOL: f64 = 1e-12;
/// b[i] < 0 判定閾値（ゼロ行の Error/Warning 分岐）
const ZERO_B_TOL: f64 = 1e-12;

// ───────────────────────── 型定義 ─────────────────────────

/// 重要度
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    /// 実行不可能または ill-posed。solve() は失敗する可能性が極めて高い
    Error,
    /// 数値的問題の可能性。solve() は不正確な結果を返すかもしれない
    Warning,
    /// 情報提供のみ。solve() への影響なし
    Info,
}

/// 診断コード
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticCode {
    /// Q の対角要素が負 → Q は PSD でない可能性が高い（Error）
    QNegativeDiagonal,
    /// Q が対称でない可能性がある（Warning）
    QNotSymmetric,
    /// 変数の下界 > 上界 → 実行不可能（Error）
    VariableBoundsConflict,
    /// スケーリングが不良（係数比 > SCALE_WARN_THRESHOLD）（Warning）
    PoorScaling,
    /// A の零行が存在（Warning: 冗長制約、または b[i]<0 なら Error）
    ZeroRowInA,
    /// 問題サイズ情報（Info）
    ProblemSize,
}

/// 個別の診断警告
#[derive(Debug, Clone)]
pub struct DiagnosticWarning {
    pub code: DiagnosticCode,
    pub severity: Severity,
    pub message: String,
    /// 問題のある変数インデックス（検出できた場合のみ Some）
    pub variable_index: Option<usize>,
    /// 問題のある制約インデックス（検出できた場合のみ Some）
    pub constraint_index: Option<usize>,
}

/// 問題サイズ情報
#[derive(Debug, Clone)]
pub struct ProblemInfo {
    pub n: usize,
    pub m: usize,
    pub nnz_q: usize,
    pub nnz_a: usize,
}

/// `diagnose()` の戻り値
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    /// 検出された警告リスト（検出順）
    pub warnings: Vec<DiagnosticWarning>,
    /// 問題サイズ情報（常に付与）
    pub info: ProblemInfo,
    /// Error レベルの警告が 1 件以上あれば true
    pub has_error: bool,
}

// ───────────────────────── ヘルパー ─────────────────────────

/// 値配列から `|max| / |min_nonzero|` を計算する。
/// 非ゼロ要素が 1 件未満なら `None`。
fn coefficient_ratio(values: &[f64]) -> Option<f64> {
    let mut max_v = 0.0_f64;
    let mut min_v = f64::INFINITY;
    for &v in values {
        let av = v.abs();
        if av > 1e-15 {
            if av > max_v {
                max_v = av;
            }
            if av < min_v {
                min_v = av;
            }
        }
    }
    if min_v == f64::INFINITY {
        None
    } else {
        Some(max_v / min_v)
    }
}

// ───────────────────────── 公開 API ─────────────────────────

/// QP 問題の事前診断を行い、不正・潜在的問題を報告する
///
/// `solve()` を呼ぶ前に問題定義の妥当性を確認するための軽量チェック。
/// 全チェックのコストは O(nnz_Q + nnz_A + n + m)。
///
/// # 引数
/// - `problem`: 診断対象の QpProblem
///
/// # 戻り値
/// [`DiagnosticReport`] — 問題サイズ情報・警告リスト・潜在的不実行可能性フラグ
pub fn diagnose(problem: &QpProblem) -> DiagnosticReport {
    let mut warnings: Vec<DiagnosticWarning> = Vec::new();

    // ── (1) Q 対角要素チェック（QNegativeDiagonal: Error） ──
    for col in 0..problem.num_vars {
        let start = problem.q.col_ptr[col];
        let end = problem.q.col_ptr[col + 1];
        for k in start..end {
            if problem.q.row_ind[k] == col && problem.q.values[k] < -DIAG_TOL {
                warnings.push(DiagnosticWarning {
                    code: DiagnosticCode::QNegativeDiagonal,
                    severity: Severity::Error,
                    message: format!(
                        "Q[{},{}] = {:.6e} < 0: Q is not PSD",
                        col, col, problem.q.values[k]
                    ),
                    variable_index: Some(col),
                    constraint_index: None,
                });
            }
        }
    }

    // ── (2) Q 対称性チェック（QNotSymmetric: Warning） ──
    // 下三角（row > col）に要素が存在するか確認する。
    // CSC 格納順で col_j の要素のうち row_ind[k] > col を探す。
    let mut found_lower = false;
    'outer: for col in 0..problem.num_vars {
        let start = problem.q.col_ptr[col];
        let end = problem.q.col_ptr[col + 1];
        for k in start..end {
            if problem.q.row_ind[k] > col {
                found_lower = true;
                break 'outer;
            }
        }
    }
    if found_lower {
        warnings.push(DiagnosticWarning {
            code: DiagnosticCode::QNotSymmetric,
            severity: Severity::Warning,
            message: "Q has sub-diagonal entries: input may not be upper-triangular or symmetric"
                .to_string(),
            variable_index: None,
            constraint_index: None,
        });
    }

    // ── (3) 変数 Bounds 整合性（VariableBoundsConflict: Error） ──
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        if lb > ub + BOUND_TOL {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::VariableBoundsConflict,
                severity: Severity::Error,
                message: format!(
                    "variable {}: lb ({:.6e}) > ub ({:.6e}): infeasible bounds",
                    j, lb, ub
                ),
                variable_index: Some(j),
                constraint_index: None,
            });
        }
    }

    // ── (4) スケーリング状態チェック（PoorScaling: Warning） ──
    // Q, A, c それぞれ個別にチェック
    if let Some(ratio) = coefficient_ratio(&problem.q.values) {
        if ratio > SCALE_WARN_THRESHOLD {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::PoorScaling,
                severity: Severity::Warning,
                message: format!(
                    "Q coefficient ratio = {:.2e} > {:.2e}: poor scaling may cause numerical issues",
                    ratio, SCALE_WARN_THRESHOLD
                ),
                variable_index: None,
                constraint_index: None,
            });
        }
    }
    if let Some(ratio) = coefficient_ratio(&problem.a.values) {
        if ratio > SCALE_WARN_THRESHOLD {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::PoorScaling,
                severity: Severity::Warning,
                message: format!(
                    "A coefficient ratio = {:.2e} > {:.2e}: poor scaling may cause numerical issues",
                    ratio, SCALE_WARN_THRESHOLD
                ),
                variable_index: None,
                constraint_index: None,
            });
        }
    }
    if let Some(ratio) = coefficient_ratio(&problem.c) {
        if ratio > SCALE_WARN_THRESHOLD {
            warnings.push(DiagnosticWarning {
                code: DiagnosticCode::PoorScaling,
                severity: Severity::Warning,
                message: format!(
                    "c coefficient ratio = {:.2e} > {:.2e}: poor scaling may cause numerical issues",
                    ratio, SCALE_WARN_THRESHOLD
                ),
                variable_index: None,
                constraint_index: None,
            });
        }
    }

    // ── (5) A 行列ゼロ行検出（ZeroRowInA: Warning/Error） ──
    if problem.num_constraints > 0 {
        let mut row_has_nonzero = vec![false; problem.num_constraints];
        for &row in &problem.a.row_ind {
            row_has_nonzero[row] = true;
        }
        for (i, &present) in row_has_nonzero.iter().enumerate() {
            if !present {
                let severity = if problem.b[i] < -ZERO_B_TOL {
                    Severity::Error
                } else {
                    Severity::Warning
                };
                let msg = if severity == Severity::Error {
                    format!(
                        "constraint {}: zero row in A with b[{}] = {:.6e} < 0: infeasible (0 <= {})",
                        i, i, problem.b[i], problem.b[i]
                    )
                } else {
                    format!(
                        "constraint {}: zero row in A with b[{}] = {:.6e} >= 0: redundant constraint",
                        i, i, problem.b[i]
                    )
                };
                warnings.push(DiagnosticWarning {
                    code: DiagnosticCode::ZeroRowInA,
                    severity,
                    message: msg,
                    variable_index: None,
                    constraint_index: Some(i),
                });
            }
        }
    }

    // ── (6) 問題サイズ情報（ProblemSize: Info） ──
    let info = ProblemInfo {
        n: problem.num_vars,
        m: problem.num_constraints,
        nnz_q: problem.q.nnz(),
        nnz_a: problem.a.nnz(),
    };
    warnings.push(DiagnosticWarning {
        code: DiagnosticCode::ProblemSize,
        severity: Severity::Info,
        message: format!(
            "problem size: n={}, m={}, nnz_Q={}, nnz_A={}",
            info.n, info.m, info.nnz_q, info.nnz_a
        ),
        variable_index: None,
        constraint_index: None,
    });

    let has_error = warnings.iter().any(|w| w.severity == Severity::Error);

    DiagnosticReport { warnings, info, has_error }
}

// ───────────────────────── テスト ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use crate::qp::problem::QpProblem;

    fn make_simple_problem() -> QpProblem {
        // min x^2 + y^2  s.t. x+y >= 1
        // Q = [[2,0],[0,2]], c=[0,0], A=[[-1,-1]], b=[-1]
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        QpProblem::new_all_le(q, c, a, b, bounds).unwrap()
    }

    // T1: 正常問題 → QNegativeDiagonal なし
    #[test]
    fn test_q_negative_diagonal_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        assert!(!report.has_error, "正常問題は has_error=false");
        let neg = report.warnings.iter().any(|w| w.code == DiagnosticCode::QNegativeDiagonal);
        assert!(!neg, "正常問題は QNegativeDiagonal なし");
    }

    // T2: Q 対角要素が負 → QNegativeDiagonal Error
    #[test]
    fn test_q_negative_diagonal_detected() {
        // Q = [[-2,0],[0,2]] → 対角(0,0)が負
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error, "has_error=true");
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::QNegativeDiagonal);
        assert!(w.is_some(), "QNegativeDiagonal が検出される");
        assert_eq!(w.unwrap().variable_index, Some(0));
    }

    // T3: 正常問題 → QNotSymmetric なし（対角行列は下三角要素なし）
    #[test]
    fn test_q_symmetric_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::QNotSymmetric);
        assert!(!w, "対角 Q は QNotSymmetric なし");
    }

    // T4: Q に下三角要素あり → QNotSymmetric Warning
    #[test]
    fn test_q_not_symmetric_detected() {
        // Q = [[2,1],[1,2]] を全要素格納（下三角あり）
        let q = CscMatrix::from_triplets(
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[2.0, 1.0, 1.0, 2.0],
            2, 2,
        ).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::QNotSymmetric);
        assert!(w.is_some(), "QNotSymmetric が検出される");
        assert_eq!(w.unwrap().severity, Severity::Warning);
    }

    // T5: 正常な bounds → VariableBoundsConflict なし
    #[test]
    fn test_bounds_conflict_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::VariableBoundsConflict);
        assert!(!w, "正常 bounds は VariableBoundsConflict なし");
    }

    // T6: lb > ub → VariableBoundsConflict Error
    #[test]
    fn test_bounds_conflict_detected() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        // x[1]: lb=2 > ub=1 → 矛盾
        let bounds = vec![(0.0, 1.0), (2.0, 1.0)];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error, "has_error=true");
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::VariableBoundsConflict);
        assert!(w.is_some(), "VariableBoundsConflict が検出される");
        assert_eq!(w.unwrap().variable_index, Some(1));
    }

    // T7: スケーリング良好 → PoorScaling なし
    #[test]
    fn test_poor_scaling_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::PoorScaling);
        assert!(!w, "正常スケーリングは PoorScaling なし");
    }

    // T8: Q のスケーリング不良 → PoorScaling Warning
    #[test]
    fn test_poor_scaling_detected() {
        // Q = [[1e10, 0],[0, 1]] → 比率 = 1e10 > 1e8
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1e10, 1.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::PoorScaling);
        assert!(w.is_some(), "PoorScaling が検出される");
        assert_eq!(w.unwrap().severity, Severity::Warning);
    }

    // T9: A にゼロ行なし → ZeroRowInA なし
    #[test]
    fn test_zero_row_in_a_clean() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        let w = report.warnings.iter().any(|w| w.code == DiagnosticCode::ZeroRowInA);
        assert!(!w, "ゼロ行なしは ZeroRowInA なし");
    }

    // T10: A にゼロ行 + b[i]>=0 → ZeroRowInA Warning
    #[test]
    fn test_zero_row_in_a_warning() {
        // A = [[0,0],[-1,-1]], b=[0,-1]  → 行0がゼロ行、b[0]=0 >= 0 → Warning
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        // 行0はゼロ行、行1に要素あり
        let a = CscMatrix::from_triplets(&[1, 1], &[0, 1], &[-1.0, -1.0], 2, 2).unwrap();
        let b = vec![0.0, -1.0]; // b[0]=0 (冗長), b[1]=-1
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::ZeroRowInA);
        assert!(w.is_some(), "ZeroRowInA が検出される");
        assert_eq!(w.unwrap().severity, Severity::Warning);
        assert_eq!(w.unwrap().constraint_index, Some(0));
    }

    // T11: A にゼロ行 + b[i]<0 → ZeroRowInA Error
    #[test]
    fn test_zero_row_in_a_error() {
        // A = [[0,0],[-1,-1]], b=[-1,-1]  → 行0がゼロ行、b[0]=-1 < 0 → Error
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[1, 1], &[0, 1], &[-1.0, -1.0], 2, 2).unwrap();
        let b = vec![-1.0, -1.0]; // b[0]=-1 < 0 → infeasible
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error, "has_error=true");
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::ZeroRowInA);
        assert!(w.is_some(), "ZeroRowInA が検出される");
        assert_eq!(w.unwrap().severity, Severity::Error);
    }

    // T12: ProblemSize Info は常に付与される
    #[test]
    fn test_problem_size_always_present() {
        let prob = make_simple_problem();
        let report = diagnose(&prob);
        // info フィールドで直接確認
        assert_eq!(report.info.n, 2);
        assert_eq!(report.info.m, 1);
        assert_eq!(report.info.nnz_q, 2);
        assert_eq!(report.info.nnz_a, 2);
        // warnings にも ProblemSize Info が含まれること
        let w = report.warnings.iter().find(|w| w.code == DiagnosticCode::ProblemSize);
        assert!(w.is_some(), "ProblemSize Info が常に付与される");
        assert_eq!(w.unwrap().severity, Severity::Info);
    }

    // T13: 複合チェック（複数 Error 同時検出）
    #[test]
    fn test_multiple_errors_combined() {
        // Q 対角負 + bounds 矛盾
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-1.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(5.0, 1.0), (0.0, 1.0)]; // x[0]: lb=5 > ub=1 → 矛盾
        let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let report = diagnose(&prob);
        assert!(report.has_error, "has_error=true");
        let errors: Vec<_> = report.warnings.iter()
            .filter(|w| w.severity == Severity::Error)
            .collect();
        assert!(errors.len() >= 2, "Error が 2 件以上検出される（実際: {}）", errors.len());
    }
}
