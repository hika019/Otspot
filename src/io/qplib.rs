//! QPLIBファイル形式パーサー
//!
//! QPLIB（Quadratic Programming Library）は https://qplib.zib.de/ で公開されている
//! QP問題の標準ベンチマークライブラリ。`.qplib` 形式は QPS（MPS系）とは別の形式。
//!
//! # QPLIBフォーマット概要
//!
//! ```text
//! QPLIB_XXXX            # 問題名
//! QCL                   # 問題タイプ（OVC: Objective, Variables, Constraints）
//! minimize              # objsense
//! n                     # 変数数
//! m                     # 制約数
//! nqobj                 # 目的関数の二次項数
//! i j coeff             # (下三角, 1-indexed) × nqobj行
//! default_b0            # 線形目的係数のデフォルト値
//! n_nondefault_b0       # 非デフォルト線形目的係数の数
//! i coeff               # × n_nondefault_b0行
//! q0                    # 目的定数（無視）
//! [QCQ のみ]
//! n_con_quad_terms      # 制約の二次項の総数
//! k i j coeff           # (k=制約, i=row, j=col, 1-indexed, 下三角) × n行
//! n_con_lin_terms       # 制約の線形項の総数
//! k i coeff             # (k=制約, i=変数, 1-indexed) × n_con_lin_terms行
//! INF                   # 無限大の定義値
//! lb_con_default        # 制約下界のデフォルト
//! n_nondefault_lb_con   # 非デフォルト制約下界数
//! k lb                  # × n行
//! ub_con_default        # 制約上界のデフォルト
//! n_nondefault_ub_con   # 非デフォルト制約上界数
//! k ub                  # × n行
//! lb_var_default        # 変数下界のデフォルト
//! n_nondefault_lb_var   # 非デフォルト変数下界数
//! i lb                  # × n行
//! ub_var_default        # 変数上界のデフォルト
//! n_nondefault_ub_var   # 非デフォルト変数上界数
//! i ub                  # × n行
//! ```
//!
//! # 対応問題タイプ
//!
//! - 変数タイプ: C（連続）のみ。B/M/I/G（整数・バイナリ）はスキップ
//! - 制約タイプ: L（線形）, B（境界のみ）, N（制約なし）, Q（二次制約）に対応
//! - 目的タイプ: L/D/Q すべて対応
//!
//! # 制約変換
//!
//! QPLIB は区間制約 lb <= a^T x <= ub を表現できる。
//! `QpProblem` は Ax <= b 形式のみサポートするため以下に変換:
//! - a^T x <= ub （ubが有限の場合）
//! - -a^T x <= -lb（lbが有限の場合）
//!
//! # 二次制約 (QCQ)
//!
//! C=Q の場合、各制約 k は 1/2 x^T Q_k x + a_k^T x {sense} rhs_k の形を持つ。
//! `QpProblem::quadratic_constraints[k]` に対称行列 Q_k を格納する。
//! QPLIB ファイルでは Q_k の下三角要素のみ記録され、パーサーが対称化する。

use crate::problem::ConstraintType;
use crate::qp::QpProblem;
use crate::sparse::CscMatrix;
use std::collections::HashMap;
use std::path::Path;

/// QPLIBパース中に発生するエラー
#[non_exhaustive]
#[derive(Debug)]
pub enum QplibError {
    /// ファイルI/Oエラー
    IoError(std::io::Error),
    /// パースエラー（メッセージ）
    ParseError(String),
    /// 対応していない問題タイプ（整数変数・二次制約等）
    UnsupportedType(String),
}

impl std::fmt::Display for QplibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QplibError::IoError(e) => write!(f, "I/O error: {}", e),
            QplibError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            QplibError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
        }
    }
}

impl std::error::Error for QplibError {}

impl From<std::io::Error> for QplibError {
    fn from(e: std::io::Error) -> Self {
        QplibError::IoError(e)
    }
}

/// ファイルパスからQPLIBファイルを読み込み `QpProblem` にパースする
pub fn parse_qplib(path: &Path) -> Result<QpProblem, QplibError> {
    let content = std::fs::read_to_string(path)?;
    parse_qplib_str(&content)
}

/// QPLIB形式の文字列を `QpProblem` にパースする
pub fn parse_qplib_str(input: &str) -> Result<QpProblem, QplibError> {
    let mut ts = TokenStream::from_str(input);

    // --- ヘッダー ---
    // 問題名（スキップ）
    let _name = ts.read_string()?;

    // 問題タイプ（OVC: Objective, Variables, Constraints）
    let prob_type = ts.read_string()?;
    if prob_type.len() != 3 {
        return Err(QplibError::ParseError(format!(
            "Problem type must be 3 characters, got '{}'",
            prob_type
        )));
    }
    let type_bytes = prob_type.as_bytes();
    let obj_char = type_bytes[0] as char;
    let var_char = type_bytes[1] as char;
    let con_char = type_bytes[2] as char;

    // 変数タイプ: C（連続）のみ
    if var_char != 'C' {
        return Err(QplibError::UnsupportedType(format!(
            "Variable type '{}' not supported (only C=continuous supported). Type={}",
            var_char, prob_type
        )));
    }

    // 制約タイプ: L/B/N/Q に対応
    match con_char {
        'L' | 'B' | 'N' | 'Q' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Constraint type '{}' not supported (only L/B/N/Q supported). Type={}",
                c, prob_type
            )));
        }
    }

    // objsense
    let objsense = ts.read_string()?.to_lowercase();
    let maximize = matches!(objsense.as_str(), "maximize" | "max");

    // 次元
    // 'L', 'N', 'Q': n と m を読む（制約あり）
    // 'B'（box）: m フィールド自体が存在しない
    let n = ts.read_usize()?;
    let m = match con_char {
        'L' | 'N' | 'Q' => ts.read_usize()?,
        _ => 0, // 'B': no m field in file
    };

    // --- 目的関数二次項 ---
    // 目的タイプが 'L'（線形）でも nqobj 行は存在する（0になる）
    let nqobj = if obj_char == 'L' {
        // 線形目的: 次のトークンが数値ならnqobj、そうでなければ0と仮定
        // ただし仕様上は0が格納されているはずなので読む
        ts.read_usize()?
    } else {
        ts.read_usize()?
    };

    // 下三角トリプレット（1-indexed, i >= j）→ 対称化
    let mut q_triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(nqobj * 2);
    for _ in 0..nqobj {
        let i = ts.read_index_1based(n, "Q row")?;
        let j = ts.read_index_1based(n, "Q col")?;
        let v = ts.read_f64()?;
        q_triplets.push((i, j, v));
        if i != j {
            q_triplets.push((j, i, v));
        }
    }

    // --- 目的関数線形項 ---
    let default_b0 = ts.read_f64()?;
    let mut c = vec![default_b0; n];
    let n_nondefault_b0 = ts.read_usize()?;
    for _ in 0..n_nondefault_b0 {
        let i = ts.read_index_1based(n, "linear obj index")?;
        let v = ts.read_f64()?;
        c[i] = v;
    }

    // 目的定数（無視）
    let _q0 = ts.read_f64()?;

    // --- 二次制約項（Q タイプのみ: Q_k 下三角トリプレット k i j val）---
    // per-constraint triplet lists; 対称化は後でまとめて実施
    let mut con_q_triplets: Vec<Vec<(usize, usize, f64)>> = vec![vec![]; m];
    if con_char == 'Q' {
        let n_con_quad_terms = ts.read_usize()?;
        for _ in 0..n_con_quad_terms {
            let k = ts.read_index_1based(m, "constraint quad index")?;
            let i = ts.read_index_1based(n, "constraint quad row")?;
            let j = ts.read_index_1based(n, "constraint quad col")?;
            let v = ts.read_f64()?;
            con_q_triplets[k].push((i, j, v));
            if i != j {
                con_q_triplets[k].push((j, i, v));
            }
        }
    }

    // --- 制約線形項（L/N/Q タイプ: ファイルに存在。B タイプ: 存在しない）---
    let mut a_triplets: HashMap<(usize, usize), f64> = HashMap::new();
    if matches!(con_char, 'L' | 'N' | 'Q') {
        let n_con_lin_terms = ts.read_usize()?;
        // k=constraint(1-indexed), i=variable(1-indexed), v=coefficient
        for _ in 0..n_con_lin_terms {
            let k = ts.read_index_1based(m, "constraint index")?;
            let i = ts.read_index_1based(n, "variable index")?;
            let v = ts.read_f64()?;
            *a_triplets.entry((k, i)).or_insert(0.0) += v;
        }
    }

    // --- 無限大の定義値 ---
    let inf_val = ts.read_f64()?;
    let is_pos_inf = |x: f64| x >= inf_val * 0.99;
    let is_neg_inf = |x: f64| x <= -inf_val * 0.99;

    // --- 制約下界・上界（L/N/Q タイプ: ファイルに存在。B タイプ: 存在しない）---
    let mut lb_con = vec![f64::NEG_INFINITY; m];
    let mut ub_con = vec![f64::INFINITY; m];
    if matches!(con_char, 'L' | 'N' | 'Q') {
        let lb_con_default = ts.read_f64()?;
        let n_nondefault_lb_con = ts.read_usize()?;
        lb_con = vec![lb_con_default; m];
        for _ in 0..n_nondefault_lb_con {
            let k = ts.read_index_1based(m, "lb_con index")?;
            let v = ts.read_f64()?;
            lb_con[k] = v;
        }

        let ub_con_default = ts.read_f64()?;
        let n_nondefault_ub_con = ts.read_usize()?;
        ub_con = vec![ub_con_default; m];
        for _ in 0..n_nondefault_ub_con {
            let k = ts.read_index_1based(m, "ub_con index")?;
            let v = ts.read_f64()?;
            ub_con[k] = v;
        }
    }

    // --- 変数下界 ---
    let lb_var_default = ts.read_f64()?;
    let n_nondefault_lb_var = ts.read_usize()?;
    let mut lb_var = vec![lb_var_default; n];
    for _ in 0..n_nondefault_lb_var {
        let i = ts.read_index_1based(n, "lb_var index")?;
        let v = ts.read_f64()?;
        lb_var[i] = v;
    }

    // --- 変数上界 ---
    let ub_var_default = ts.read_f64()?;
    let n_nondefault_ub_var = ts.read_usize()?;
    let mut ub_var = vec![ub_var_default; n];
    for _ in 0..n_nondefault_ub_var {
        let i = ts.read_index_1based(n, "ub_var index")?;
        let v = ts.read_f64()?;
        ub_var[i] = v;
    }

    // 残り（初期点・双対値・名前）は読み捨て

    // ============================================================
    // QpProblem 構築
    // ============================================================

    // Q行列（maximize の場合は符号反転）
    let sign = if maximize { -1.0 } else { 1.0 };

    let q_rows: Vec<usize> = q_triplets.iter().map(|&(r, _, _)| r).collect();
    let q_cols: Vec<usize> = q_triplets.iter().map(|&(_, c, _)| c).collect();
    let q_vals: Vec<f64> = q_triplets.iter().map(|&(_, _, v)| sign * v).collect();

    let q = if q_rows.is_empty() {
        CscMatrix::new(n, n)
    } else {
        CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n)
            .map_err(|e| QplibError::ParseError(format!("Q matrix error: {}", e)))?
    };

    // c（maximize の場合は符号反転）
    if maximize {
        for v in &mut c {
            *v = -*v;
        }
    }

    // 拡張制約行列の構築:
    // lb_con[k] <= a[k]^T x <= ub_con[k] を Ax <= b に変換
    //   a[k]^T x <= ub_con[k]   (ub が有限)
    //   -a[k]^T x <= -lb_con[k]  (lb が有限)
    let mut aug_ub_row: Vec<Option<usize>> = vec![None; m];
    let mut aug_lb_row: Vec<Option<usize>> = vec![None; m];
    let mut b_vec: Vec<f64> = Vec::new();
    let mut constraint_types: Vec<ConstraintType> = Vec::new();

    for k in 0..m {
        let lb = lb_con[k];
        let ub = ub_con[k];
        if !is_pos_inf(ub) && !is_neg_inf(lb) && (lb - ub).abs() < 1e-15 {
            // 等式制約: 1行Eqとして格納
            aug_ub_row[k] = Some(b_vec.len());
            b_vec.push(ub);
            constraint_types.push(ConstraintType::Eq);
        } else {
            // 範囲制約・不等式: 従来通り2Le展開
            if !is_pos_inf(ub) {
                aug_ub_row[k] = Some(b_vec.len());
                b_vec.push(ub);
                constraint_types.push(ConstraintType::Le);
            }
            if !is_neg_inf(lb) {
                aug_lb_row[k] = Some(b_vec.len());
                b_vec.push(-lb);
                constraint_types.push(ConstraintType::Le);
            }
        }
    }

    let m_aug = b_vec.len();
    let mut a_rows: Vec<usize> = Vec::new();
    let mut a_cols: Vec<usize> = Vec::new();
    let mut a_vals: Vec<f64> = Vec::new();

    for (&(con_idx, var_idx), &val) in &a_triplets {
        if let Some(aug_row) = aug_ub_row[con_idx] {
            a_rows.push(aug_row);
            a_cols.push(var_idx);
            a_vals.push(val);
        }
        if let Some(aug_row) = aug_lb_row[con_idx] {
            a_rows.push(aug_row);
            a_cols.push(var_idx);
            a_vals.push(-val);
        }
    }

    let a_mat = if a_rows.is_empty() {
        CscMatrix::new(m_aug, n)
    } else {
        CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_aug, n)
            .map_err(|e| QplibError::ParseError(format!("A matrix error: {}", e)))?
    };

    // 変数境界（無限大の変換）
    let bounds: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let lb = if is_neg_inf(lb_var[i]) { f64::NEG_INFINITY } else { lb_var[i] };
            let ub = if is_pos_inf(ub_var[i]) { f64::INFINITY } else { ub_var[i] };
            (lb, ub)
        })
        .collect();

    // 二次制約行列（QCQP のみ）
    // aug 行ごとに Q_k を配置:
    //   aug_ub_row[k] → Q_k（正符号）
    //   aug_lb_row[k] → -Q_k（lb 反転のため符号反転）
    let quadratic_constraints = if con_char == 'Q' {
        let mut qc = vec![CscMatrix::new(n, n); m_aug];
        for k in 0..m {
            let trips = &con_q_triplets[k];
            if trips.is_empty() {
                continue;
            }
            let rows: Vec<usize> = trips.iter().map(|&(r, _, _)| r).collect();
            let cols: Vec<usize> = trips.iter().map(|&(_, c, _)| c).collect();
            let vals: Vec<f64> = trips.iter().map(|&(_, _, v)| v).collect();
            let qk = CscMatrix::from_triplets(&rows, &cols, &vals, n, n)
                .map_err(|e| QplibError::ParseError(format!("Q_k[{}] matrix error: {}", k, e)))?;

            if let Some(aug_row) = aug_ub_row[k] {
                qc[aug_row] = qk.clone();
            }
            if let Some(aug_row) = aug_lb_row[k] {
                // lb 反転行: 1/2 x^T (-Q_k) x <= -lb_k
                let neg_vals: Vec<f64> = vals.iter().map(|v| -v).collect();
                let neg_qk = CscMatrix::from_triplets(&rows, &cols, &neg_vals, n, n)
                    .map_err(|e| QplibError::ParseError(format!("Q_k[{}] neg matrix error: {}", k, e)))?;
                qc[aug_row] = neg_qk;
            }
        }
        qc
    } else {
        vec![]
    };

    let mut prob = QpProblem::new(q, c, a_mat, b_vec, bounds, constraint_types)
        .map_err(|e| QplibError::ParseError(e.to_string()))?;
    prob.quadratic_constraints = quadratic_constraints;
    Ok(prob)
}

/// トークンストリーム（コメントを除去しながらフラットにトークン化）
struct TokenStream {
    tokens: Vec<String>,
    pos: usize,
}

impl TokenStream {
    fn from_str(input: &str) -> Self {
        let mut tokens = Vec::new();
        for line in input.lines() {
            // 行頭コメント（% または ! で始まる行）をスキップ
            let trimmed = line.trim();
            if trimmed.starts_with('%') || trimmed.starts_with('!') {
                continue;
            }
            // インラインコメント（# 以降）を除去
            let line = if let Some(idx) = line.find('#') {
                &line[..idx]
            } else {
                line
            };
            for token in line.split_whitespace() {
                tokens.push(token.to_string());
            }
        }
        TokenStream { tokens, pos: 0 }
    }

    fn next_str(&mut self) -> Option<&str> {
        if self.pos < self.tokens.len() {
            let t = &self.tokens[self.pos];
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn read_string(&mut self) -> Result<String, QplibError> {
        match self.next_str() {
            Some(t) => Ok(t.to_string()),
            None => Err(QplibError::ParseError("unexpected end of file (expected string)".to_string())),
        }
    }

    fn read_usize(&mut self) -> Result<usize, QplibError> {
        match self.next_str() {
            Some(t) => {
                // 浮動小数点として読んでから整数に変換（例: "1.0" → 1）
                if let Ok(u) = t.parse::<usize>() {
                    Ok(u)
                } else if let Ok(f) = t.parse::<f64>() {
                    Ok(f as usize)
                } else {
                    Err(QplibError::ParseError(format!(
                        "expected integer, got '{}'",
                        t
                    )))
                }
            }
            None => Err(QplibError::ParseError(
                "unexpected end of file (expected integer)".to_string(),
            )),
        }
    }

    fn read_f64(&mut self) -> Result<f64, QplibError> {
        match self.next_str() {
            Some(t) => t.parse::<f64>().map_err(|_| {
                QplibError::ParseError(format!("expected float, got '{}'", t))
            }),
            None => Err(QplibError::ParseError(
                "unexpected end of file (expected float)".to_string(),
            )),
        }
    }

    /// 1-indexedの整数を読み込み、0-indexedに変換して返す。
    /// 値が0またはmax_valを超える場合はParseErrorを返す。
    fn read_index_1based(&mut self, max_val: usize, context: &str) -> Result<usize, QplibError> {
        let raw = self.read_usize()?;
        if raw == 0 || raw > max_val {
            return Err(QplibError::ParseError(format!(
                "{}: index {} out of range (expected 1..={})",
                context, raw, max_val
            )));
        }
        Ok(raw - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 単純な2変数QP (QCL):
    /// min 1/2 * (x1^2 + x2^2)  s.t. x1 + x2 = 1, x1,x2 >= 0
    #[test]
    fn test_parse_qplib_simple() {
        let qplib = "\
SIMPLE_QP
QCL
minimize
2 # number of variables
1 # number of constraints
2 # number of quadratic terms in objective
1 1 1.0
2 2 1.0
0.0 # default linear obj coefficient
0 # number of non-default linear obj coefficients
0.0 # objective constant
2 # number of linear terms in all constraints
1 1 1.0
1 2 1.0
1.79769313486232E+308 # infinity
1.0 # default left-hand-side
0 # number of non-default left-hand-sides
1.0 # default right-hand-side
0 # number of non-default right-hand-sides
0.0 # default variable lower bound
0 # number of non-default variable lower bounds
1.79769313486232E+308 # default variable upper bound
0 # number of non-default variable upper bounds
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = parse_qplib_str(qplib).unwrap();
        assert_eq!(prob.num_vars, 2);
        // 等式制約 x1+x2=1 → 1行Eqとして保持（2Le展開しない）
        assert_eq!(prob.num_constraints, 1);
        assert_eq!(prob.constraint_types[0], crate::problem::ConstraintType::Eq);
        // Q = I_2（「1/2あり」規約で min 1/2*x^T*I*x）
        assert_eq!(prob.q.nnz(), 2);
    }

    /// 制約なしQP（QCN型）
    #[test]
    fn test_parse_qplib_unconstrained() {
        let qplib = "\
NO_CON
QCN
minimize
2 # vars
0 # constraints
2 # qobj
1 1 2.0
2 2 2.0
0.0 # default b0
0 # non-default b0
0.0 # obj constant
0 # no linear constraints
1.79769313486232E+308 # infinity
0.0 # default lhs
0
0.0 # default rhs
0
-1.79769313486232E+308 # default var lb
0
1.79769313486232E+308 # default var ub
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = parse_qplib_str(qplib).unwrap();
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 0);
        assert_eq!(prob.q.nnz(), 2);
    }

    /// QCQ: n=2, m=1, equality quadratic constraint with diagonal Q.
    ///
    /// min x1 + x2
    /// s.t. 1/2*(2*x1^2 + 4*x2^2) = 5, 0 <= xi <= 1
    ///
    /// Expected: 1 Eq aug row; quadratic_constraints[0] has nnz=2 (diagonal).
    #[test]
    fn test_parse_qcq_equality_diagonal_q() {
        // Fixture: n=2, m=1
        // Obj: L (no quadratic), c=[1,1], q0=0
        // Con Q: Q_1 = diag(2,4) lower-tri: (1,1,2.0),(2,2,4.0)
        // Con lin: 0 terms
        // inf=1e308; lb_con=ub_con=5.0 (equality)
        // var bounds: [0,1]
        let qplib = "\
QCQ_EQ_DIAG
QCQ
minimize
2 # n
1 # m
0 # nqobj
0.0 # default b0
2 # non-default b0
1 1.0
2 1.0
0.0 # q0
2 # n_con_quad_terms
1 1 1 2.0
1 2 2 4.0
0 # n_con_lin_terms
1.79769313486232E+308 # inf
5.0 # default lb_con
0 # non-default lb_con
5.0 # default ub_con
0 # non-default ub_con
0.0 # default lb_var
0 # non-default lb_var
1.0 # default ub_var
0 # non-default ub_var
0.0 # primal default
0
0.0 # dual default
0
0.0 # bound dual default
0
0 # non-default var names
0 # non-default con names
";
        let prob = parse_qplib_str(qplib).unwrap();
        assert_eq!(prob.num_vars, 2);
        // lb=ub=5 → 1 Eq row
        assert_eq!(prob.num_constraints, 1);
        assert_eq!(prob.constraint_types[0], crate::problem::ConstraintType::Eq);
        assert_eq!(prob.b[0], 5.0);
        // quadratic_constraints: 1 entry (one aug row), diagonal Q_1 has nnz=2
        assert_eq!(prob.quadratic_constraints.len(), 1);
        assert_eq!(prob.quadratic_constraints[0].nnz(), 2);
        // Verify Q_1 values (after symmetrization diagonal-only: no off-diag added)
        // CscMatrix nnz=2: entries at (0,0)=2.0 and (1,1)=4.0
        let qk = &prob.quadratic_constraints[0];
        assert_eq!(qk.nrows, 2);
        assert_eq!(qk.ncols, 2);
        // no quadratic_constraints on standard QCL/QCN (round-trip guard)
        assert!(prob.quadratic_constraints.iter().any(|q| q.nnz() > 0));
    }

    /// QCQ: n=3, m=2, mixed constraint types.
    ///
    /// Constraint 1: linear only  x1+x2 <= 4
    /// Constraint 2: quadratic    1/2*(x1^2 + x3^2) + x2 = 3
    ///
    /// Expected augmentation: con1→1 Le, con2→1 Eq; m_aug=2.
    /// quadratic_constraints[0] = empty matrix (con1 has no Q).
    /// quadratic_constraints[1] has nnz=2 (diagonal Q_2 for x1, x3).
    #[test]
    fn test_parse_qcq_mixed_linear_and_quadratic() {
        let qplib = "\
QCQ_MIXED
QCQ
minimize
3 # n
2 # m
3 # nqobj: 1/2*(2x1^2+2x2^2+2x3^2)
1 1 2.0
2 2 2.0
3 3 2.0
0.0 # default b0
0 # non-default b0
0.0 # q0
2 # n_con_quad_terms: Q_2 has (1,1,1.0),(3,3,1.0)
2 1 1 1.0
2 3 3 1.0
3 # n_con_lin_terms: con1: x1+x2, con2: x2
1 1 1.0
1 2 1.0
2 2 1.0
1.79769313486232E+308 # inf
-1.79769313486232E+308 # default lb_con
2 # non-default lb_con
1 -1.79769313486232E+308
2 3.0
4.0 # default ub_con
1 # non-default ub_con
2 3.0
0.0 # default lb_var
0
1.79769313486232E+308 # default ub_var
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = parse_qplib_str(qplib).unwrap();
        assert_eq!(prob.num_vars, 3);
        // con1: lb=-inf, ub=4 → 1 Le row; con2: lb=ub=3 → 1 Eq row
        assert_eq!(prob.num_constraints, 2);
        assert_eq!(prob.constraint_types[0], crate::problem::ConstraintType::Le);
        assert_eq!(prob.constraint_types[1], crate::problem::ConstraintType::Eq);
        assert_eq!(prob.b[0], 4.0);
        assert_eq!(prob.b[1], 3.0);
        // quadratic_constraints: 2 aug rows
        assert_eq!(prob.quadratic_constraints.len(), 2);
        // con1 has no Q → empty matrix
        assert_eq!(prob.quadratic_constraints[0].nnz(), 0);
        // con2 Q has nnz=2 (diagonal entries for x1 and x3)
        assert_eq!(prob.quadratic_constraints[1].nnz(), 2);
        // Verify dimensions
        assert_eq!(prob.quadratic_constraints[1].nrows, 3);
    }

    /// QCQ: n=4, m=3, range constraint expanding to ub+lb rows (sign-flip).
    ///
    /// Constraint 1: linear only, ub=5
    /// Constraint 2: Q_2=diag(2,0,0,0), range 1 <= 1/2*2x1^2 <= 3 → 2 Le rows
    /// Constraint 3: Q_3 off-diagonal (x1*x2 coupling), ub=10 only
    ///
    /// aug rows: [con1_ub, con2_ub, con2_lb, con3_ub] → m_aug=4
    /// quadratic_constraints[2] = -Q_2 (sign-flipped for lb row)
    #[test]
    fn test_parse_qcq_range_constraint_sign_flip() {
        let qplib = "\
QCQ_RANGE
QCQ
minimize
4 # n
3 # m
0 # nqobj
0.0 # default b0
0 # non-default b0
0.0 # q0
3 # n_con_quad_terms
2 1 1 2.0
3 1 1 1.0
3 2 1 0.5
1 # n_con_lin_terms
1 1 1.0
1.79769313486232E+308 # inf
-1.79769313486232E+308 # default lb_con
3 # non-default lb_con
1 -1.79769313486232E+308
2 1.0
3 -1.79769313486232E+308
5.0 # default ub_con
2 # non-default ub_con
2 3.0
3 10.0
0.0 # default lb_var
0
1.79769313486232E+308 # default ub_var
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = parse_qplib_str(qplib).unwrap();
        assert_eq!(prob.num_vars, 4);
        // con1: lb=-inf, ub=5 → 1 Le
        // con2: lb=1, ub=3 → 2 Le (ub and lb rows)
        // con3: lb=-inf, ub=10 → 1 Le
        assert_eq!(prob.num_constraints, 4);
        // Check all Le
        for ct in &prob.constraint_types {
            assert_eq!(*ct, crate::problem::ConstraintType::Le);
        }
        // b = [5, 3, -1, 10] (ub1=5, ub2=3, -lb2=-1, ub3=10)
        assert!((prob.b[0] - 5.0).abs() < 1e-12);
        assert!((prob.b[1] - 3.0).abs() < 1e-12);
        assert!((prob.b[2] - (-1.0)).abs() < 1e-12);
        assert!((prob.b[3] - 10.0).abs() < 1e-12);

        // quadratic_constraints has 4 aug entries
        assert_eq!(prob.quadratic_constraints.len(), 4);
        // con1 has no Q → empty
        assert_eq!(prob.quadratic_constraints[0].nnz(), 0);
        // con2 ub row: Q_2 = diag(2) for x1 → nnz=1
        assert_eq!(prob.quadratic_constraints[1].nnz(), 1);
        // con2 lb row: -Q_2, also nnz=1 but negated
        assert_eq!(prob.quadratic_constraints[2].nnz(), 1);
        // con3 ub row: Q_3 has (1,1) and off-diagonal (2,1),(1,2) after symmetrization → nnz=3
        assert_eq!(prob.quadratic_constraints[3].nnz(), 3);

        // Verify sign flip: con2_ub val positive, con2_lb val negative
        // Find (0,0) entry in Q[1] and Q[2]
        let q_ub = &prob.quadratic_constraints[1];
        let q_lb = &prob.quadratic_constraints[2];
        // Both are 4x4 with nnz=1 at (0,0)
        assert!(q_ub.values.iter().all(|&v| v > 0.0), "ub row Q_2 values must be positive");
        assert!(q_lb.values.iter().all(|&v| v < 0.0), "lb row Q_2 values must be negative (sign flip)");
    }

    /// QCL/QCN round-trip: quadratic_constraints must be empty for non-Q constraint types.
    #[test]
    fn test_qcl_no_quadratic_constraints() {
        let qplib = "\
QCL_ROUND_TRIP
QCL
minimize
2
1
2
1 1 1.0
2 2 1.0
0.0
0
0.0
2
1 1 1.0
1 2 1.0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
1.79769313486232E+308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = parse_qplib_str(qplib).unwrap();
        assert_eq!(prob.num_vars, 2);
        assert!(prob.quadratic_constraints.is_empty(),
            "QCL must produce empty quadratic_constraints");
    }

    /// 整数変数を含む問題（QIL）は拒否
    #[test]
    fn test_parse_qplib_reject_integer() {
        let qplib = "\
INT_QP
QIL
minimize
2
1
0
0.0
0
0.0
0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
1.79769313486232E+308
0
0.0
0
0.0
0
0.0
0
0
0
";
        assert!(matches!(
            parse_qplib_str(qplib),
            Err(QplibError::UnsupportedType(_))
        ));
    }

    // ── File-based tests: data/qplib_unsupported/ (QCQ instances) ──────────

    /// Helper: resolve path relative to the crate manifest directory.
    fn data_path(rel: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    /// QPLIB_1157: n=40, m=9 (8 equality constraints + 1 Le).
    ///
    /// Constraints 1-8 have lb=ub (equality); constraint 9 has lb=-inf.
    /// Quadratic terms exist only in constraint 9 (k=9, 778 lower-tri entries,
    /// nnz_sym = 40 diag + 738*2 off-diag = 1516).
    /// Constraints 1-8 have no Q_k (empty CscMatrix).
    ///
    /// Hand-verified from file:
    ///   lb/ub per constraint from L1966-L1984.
    ///   Q_9 diagonal (1,1)→(0,0): value=0.38 (L826).
    #[test]
    fn test_parse_qcq_file_1157_structure() {
        let path = data_path("data/qplib_unsupported/QPLIB_1157.qplib");
        if !path.exists() {
            return; // file not downloaded; skip
        }
        let prob = parse_qplib(path.as_path()).expect("QPLIB_1157 parse");

        assert_eq!(prob.num_vars, 40);
        assert_eq!(prob.num_constraints, 9);

        // Constraints 0-7: equality; constraint 8: Le
        for i in 0..8 {
            assert_eq!(prob.constraint_types[i], crate::problem::ConstraintType::Eq,
                "constraint {i} must be Eq");
        }
        assert_eq!(prob.constraint_types[8], crate::problem::ConstraintType::Le);

        // b vector hand-verified from file (lb=ub for Eq, ub for Le)
        let expected_b = [0.56_f64, -0.16, -0.4, -0.25, 0.45, 0.3, 0.99, 0.77, 16.22];
        for (i, &exp) in expected_b.iter().enumerate() {
            assert!((prob.b[i] - exp).abs() < 1e-10,
                "b[{i}]: expected {exp}, got {}", prob.b[i]);
        }

        // quadratic_constraints: 9 entries (one per aug row)
        assert_eq!(prob.quadratic_constraints.len(), 9);
        // Constraints 0-7 have no Q_k
        for i in 0..8 {
            assert_eq!(prob.quadratic_constraints[i].nnz(), 0,
                "Q_k[{i}] must be empty (no quadratic terms for constraints 1-8)");
        }
        // Constraint 9 (aug_row=8): full 40x40 Q_9, nnz=1516 after sym
        let qk = &prob.quadratic_constraints[8];
        assert_eq!(qk.nrows, 40);
        assert_eq!(qk.ncols, 40);
        assert_eq!(qk.nnz(), 1516,
            "Q_9 nnz: 40 diag + 738 off-diag*2 = 1516");
        // Diagonal (0,0) = 0.38 (file: 9 1 1 0.38)
        let col0_start = qk.col_ptr[0];
        assert_eq!(qk.row_ind[col0_start], 0, "first entry in col 0 must be diagonal");
        assert!((qk.values[col0_start] - 0.38).abs() < 1e-10,
            "Q_9[0,0] must be 0.38");
    }

    /// QPLIB_1353: n=50, m=6 (5 equality constraints + 1 Le).
    ///
    /// Constraints 1-5 have lb=ub; constraint 6 has lb=-inf.
    /// Quadratic terms only in constraint 6 (1211 lower-tri entries,
    /// nnz_sym = 50 diag + 1161*2 off-diag = 2372).
    ///
    /// Hand-verified from file:
    ///   lb/ub from L2795-L2807.
    ///   Q_6 diagonal (1,1)→(0,0): value=0.46 (L1282).
    #[test]
    fn test_parse_qcq_file_1353_structure() {
        let path = data_path("data/qplib_unsupported/QPLIB_1353.qplib");
        if !path.exists() {
            return;
        }
        let prob = parse_qplib(path.as_path()).expect("QPLIB_1353 parse");

        assert_eq!(prob.num_vars, 50);
        assert_eq!(prob.num_constraints, 6);

        // Constraints 0-4: Eq; constraint 5: Le
        for i in 0..5 {
            assert_eq!(prob.constraint_types[i], crate::problem::ConstraintType::Eq,
                "constraint {i} must be Eq");
        }
        assert_eq!(prob.constraint_types[5], crate::problem::ConstraintType::Le);

        let expected_b = [0.13_f64, -0.4, 0.1, -0.63, 0.57, 18.74];
        for (i, &exp) in expected_b.iter().enumerate() {
            assert!((prob.b[i] - exp).abs() < 1e-10,
                "b[{i}]: expected {exp}, got {}", prob.b[i]);
        }

        assert_eq!(prob.quadratic_constraints.len(), 6);
        for i in 0..5 {
            assert_eq!(prob.quadratic_constraints[i].nnz(), 0,
                "Q_k[{i}] must be empty");
        }
        let qk = &prob.quadratic_constraints[5];
        assert_eq!(qk.nrows, 50);
        assert_eq!(qk.ncols, 50);
        assert_eq!(qk.nnz(), 2372,
            "Q_6 nnz: 50 diag + 1161 off-diag*2 = 2372");
        // Diagonal (0,0) = 0.46 (file: 6 1 1 0.46)
        let col0_start = qk.col_ptr[0];
        assert_eq!(qk.row_ind[col0_start], 0);
        assert!((qk.values[col0_start] - 0.46).abs() < 1e-10,
            "Q_6[0,0] must be 0.46");
    }

    /// QPLIB_1055: n=40, m=20 (all Le, lb=-inf for all constraints).
    ///
    /// All 20 constraints have lb=-inf and finite ub → 20 Le rows.
    /// All 20 Q_k are fully dense lower-triangle (820 entries each),
    /// nnz_sym = 40 diag + 780*2 off-diag = 1600 per constraint.
    ///
    /// Hand-verified from file:
    ///   lb/ub from L18072-L18094: default lb=-inf, ub default=30.278.
    ///   Q_1 diagonal (1,1)→(0,0): value=0.839 (L870).
    #[test]
    fn test_parse_qcq_file_1055_all_le_dense_q() {
        let path = data_path("data/qplib_unsupported/QPLIB_1055.qplib");
        if !path.exists() {
            return;
        }
        let prob = parse_qplib(path.as_path()).expect("QPLIB_1055 parse");

        assert_eq!(prob.num_vars, 40);
        assert_eq!(prob.num_constraints, 20);

        // All constraints are Le (lb=-inf for all)
        for i in 0..20 {
            assert_eq!(prob.constraint_types[i], crate::problem::ConstraintType::Le,
                "constraint {i} must be Le");
        }

        // b[0]=71.197 (constraint 1 ub), b[19]=30.278 (constraint 20, default)
        assert!((prob.b[0] - 71.197).abs() < 1e-10, "b[0] must be 71.197");
        assert!((prob.b[19] - 30.278).abs() < 1e-10, "b[19] must be 30.278 (default)");

        assert_eq!(prob.quadratic_constraints.len(), 20);
        // All 20 Q_k are non-empty (full 40x40 lower-tri, 820 entries → 1600 after sym)
        for i in 0..20 {
            let qk = &prob.quadratic_constraints[i];
            assert_eq!(qk.nrows, 40);
            assert_eq!(qk.ncols, 40);
            assert_eq!(qk.nnz(), 1600,
                "Q_{i} nnz must be 1600 (full 40x40 lower-tri symmetrized)");
        }
        // Q_1[0,0] = 0.839 (file: 1 1 1 0.839)
        let qk0 = &prob.quadratic_constraints[0];
        let col0_start = qk0.col_ptr[0];
        assert_eq!(qk0.row_ind[col0_start], 0);
        assert!((qk0.values[col0_start] - 0.839).abs() < 1e-10,
            "Q_1[0,0] must be 0.839");
    }

    /// QPLIB_1493: n=40, m=5 (4 equality + 1 Le).
    ///
    /// Constraints 1-4 lb=ub; constraint 5 lb=-inf.
    /// Quadratic terms only in constraint 5 (792 lower-tri entries,
    /// nnz_sym = 37 diag + 755*2 off-diag = 1547).
    ///
    /// Hand-verified from file: Q_5[0,0] = 1.88 (L848).
    #[test]
    fn test_parse_qcq_file_1493_structure() {
        let path = data_path("data/qplib_unsupported/QPLIB_1493.qplib");
        if !path.exists() {
            return;
        }
        let prob = parse_qplib(path.as_path()).expect("QPLIB_1493 parse");

        assert_eq!(prob.num_vars, 40);
        assert_eq!(prob.num_constraints, 5);

        for i in 0..4 {
            assert_eq!(prob.constraint_types[i], crate::problem::ConstraintType::Eq,
                "constraint {i} must be Eq");
        }
        assert_eq!(prob.constraint_types[4], crate::problem::ConstraintType::Le);

        let expected_b = [-0.17_f64, 0.51, -0.41, -0.15, 67.98];
        for (i, &exp) in expected_b.iter().enumerate() {
            assert!((prob.b[i] - exp).abs() < 1e-10,
                "b[{i}]: expected {exp}, got {}", prob.b[i]);
        }

        assert_eq!(prob.quadratic_constraints.len(), 5);
        for i in 0..4 {
            assert_eq!(prob.quadratic_constraints[i].nnz(), 0,
                "Q_k[{i}] must be empty");
        }
        let qk = &prob.quadratic_constraints[4];
        assert_eq!(qk.nrows, 40);
        assert_eq!(qk.nnz(), 1547,
            "Q_5 nnz: 37 diag + 755 off-diag*2 = 1547");
        // Q_5[0,0] = 1.88 (file: 5 1 1 1.88)
        let col0_start = qk.col_ptr[0];
        assert_eq!(qk.row_ind[col0_start], 0);
        assert!((qk.values[col0_start] - 1.88).abs() < 1e-10,
            "Q_5[0,0] must be 1.88");
    }

    /// Regression: all 6 files in data/qplib_unsupported/ parse without error.
    #[test]
    fn test_parse_all_qcq_unsupported_files() {
        let dir = data_path("data/qplib_unsupported");
        if !dir.exists() {
            return;
        }
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("qplib") {
                continue;
            }
            parse_qplib(path.as_path()).unwrap_or_else(|e| {
                panic!("parse failed for {}: {e}", path.display());
            });
            count += 1;
        }
        assert!(count > 0, "no .qplib files found in data/qplib_unsupported/");
    }

    /// Regression: existing files in data/qplib/ parse without error.
    ///
    /// Files with binary/integer variables (CBL/QBI etc.) legitimately produce
    /// `UnsupportedType` — that is the expected, pre-existing behavior.
    /// `ParseError` or `IoError` on any file indicates a regression.
    #[test]
    fn test_parse_existing_qplib_files_regression() {
        let dir = data_path("data/qplib");
        if !dir.exists() {
            return;
        }
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("qplib") {
                continue;
            }
            match parse_qplib(path.as_path()) {
                Ok(_) => {}
                Err(QplibError::UnsupportedType(_)) => {} // expected for binary/integer files
                Err(e) => panic!("parse regression: {} failed with unexpected error: {e}", path.display()),
            }
            count += 1;
        }
        assert!(count > 0, "no .qplib files found in data/qplib/");
    }
}
