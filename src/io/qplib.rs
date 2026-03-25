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
//! - 制約タイプ: L（線形）, B（境界のみ）, N（制約なし）のみ。D/C/Q（二次制約）はスキップ
//! - 目的タイプ: L/D/Q すべて対応
//!
//! # 制約変換
//!
//! QPLIB は区間制約 lb <= a^T x <= ub を表現できる。
//! `QpProblem` は Ax <= b 形式のみサポートするため以下に変換:
//! - a^T x <= ub （ubが有限の場合）
//! - -a^T x <= -lb（lbが有限の場合）

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

    // 制約タイプ: L/B/N のみ
    match con_char {
        'L' | 'B' | 'N' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Constraint type '{}' not supported (only L/B/N supported). Type={}",
                c, prob_type
            )));
        }
    }

    // objsense
    let objsense = ts.read_string()?.to_lowercase();
    let maximize = matches!(objsense.as_str(), "maximize" | "max");

    // 次元
    // 制約タイプ 'L': n と m を読む（線形制約あり）
    // 'N'（無制約）: m=0 がファイルに存在するので読む
    // 'B'（box）: m フィールド自体が存在しない
    let n = ts.read_usize()?;
    let m = match con_char {
        'L' | 'N' => ts.read_usize()?,
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

    // --- 制約線形項（L/N タイプ: ファイルに存在。B タイプ: 存在しない）---
    let mut a_triplets: HashMap<(usize, usize), f64> = HashMap::new();
    if matches!(con_char, 'L' | 'N') {
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

    // --- 制約下界・上界（L/N タイプ: ファイルに存在。B タイプ: 存在しない）---
    let mut lb_con = vec![f64::NEG_INFINITY; m];
    let mut ub_con = vec![f64::INFINITY; m];
    if matches!(con_char, 'L' | 'N') {
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

    QpProblem::new(q, c, a_mat, b_vec, bounds, constraint_types)
        .map_err(|e| QplibError::ParseError(e.to_string()))
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
}
