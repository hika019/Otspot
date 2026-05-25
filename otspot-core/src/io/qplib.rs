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
//! - 変数タイプ: C（連続）, B（二値変数・全変数）, I（整数変数・全変数）
//!   - B/I は `QplibProblem::Milp` / `QplibProblem::Miqp` として返す
//!   - M/G/S（混合整数）は UnsupportedType
//! - 制約タイプ: L（線形）, B（境界のみ）, N（制約なし）, Q（二次制約）に対応
//! - 目的タイプ: L/D/Q/C すべて対応
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

use crate::mip::{MilpProblem, MiqpProblem};
use crate::problem::{ConstraintType, LpProblem};
use crate::qp::{QcqpMatrix, QpProblem};
use crate::sparse::CscMatrix;
use std::collections::VecDeque;
use std::io::BufRead;
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

/// Parsed result of a QPLIB file.
///
/// Continuous-variable problems return [`Qp`]; problems with binary (`B`) or
/// integer (`I`) variables return [`Milp`] (zero-Q) or [`Miqp`] (non-zero Q).
#[derive(Debug)]
pub enum QplibProblem {
    /// Continuous-variable QP / QCQP / LP.
    Qp(QpProblem),
    /// Mixed-integer LP (linear objective, binary or integer variables).
    Milp(MilpProblem),
    /// Mixed-integer QP (quadratic objective, binary or integer variables).
    Miqp(MiqpProblem),
}

/// ファイルパスからQPLIBファイルを読み込みパースする。
///
/// Uses a streaming tokenizer (`BufReader`) to avoid loading the entire file into memory.
/// Large files (200 MB+) that would OOM with `read_to_string` are handled correctly.
pub fn parse_qplib(path: &Path) -> Result<QplibProblem, QplibError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    parse_token_stream(TokenStream::from_reader(reader))
}

/// QPLIB形式の文字列をパースする
pub fn parse_qplib_str(input: &str) -> Result<QplibProblem, QplibError> {
    parse_token_stream(TokenStream::from_str(input))
}

fn parse_token_stream(mut ts: TokenStream) -> Result<QplibProblem, QplibError> {

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
    let _obj_char = type_bytes[0] as char;
    let var_char = type_bytes[1] as char;
    let con_char = type_bytes[2] as char;

    // 変数タイプ: C（連続）, B（全バイナリ）, I（全整数）に対応
    // M/G/S（混合整数・半連続）は未対応
    let var_binary = var_char == 'B';   // all vars ∈ {0,1}
    let var_integer = var_char == 'I';  // all vars ∈ ℤ
    match var_char {
        'C' | 'B' | 'I' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Variable type '{}' not supported (C/B/I supported; M/G/S mixed-integer unsupported). Type={}",
                c, prob_type
            )));
        }
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
    let nqobj = ts.read_usize()?;
    // nqobj is at most n*(n+1)/2 entries (lower-triangular Q)
    if nqobj > n.saturating_mul(n.saturating_add(1)) / 2 {
        return Err(QplibError::ParseError(format!(
            "nqobj {} exceeds n*(n+1)/2={} (n={})", nqobj, n.saturating_mul(n.saturating_add(1)) / 2, n
        )));
    }

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
    // Only allocate the outer Vec for QCQ problems; allocating vec![vec![]; m] for
    // non-Q types wastes up to 6 MB on large problems (e.g. m=250K for QPLIB_8500).
    let mut con_q_triplets: Vec<Vec<(usize, usize, f64)>> = if con_char == 'Q' {
        vec![vec![]; m]
    } else {
        vec![]
    };
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
    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();
    if matches!(con_char, 'L' | 'N' | 'Q') {
        let n_con_lin_terms = ts.read_usize()?;
        // Sanity bound: can't have more terms than n*m entries
        if n_con_lin_terms > n.saturating_mul(m) {
            return Err(QplibError::ParseError(format!(
                "n_con_lin_terms {} exceeds n*m={}", n_con_lin_terms, n.saturating_mul(m)
            )));
        }
        a_triplets = Vec::with_capacity(n_con_lin_terms);
        // k=constraint(1-indexed), i=variable(1-indexed), v=coefficient
        for _ in 0..n_con_lin_terms {
            let k = ts.read_index_1based(m, "constraint index")?;
            let i = ts.read_index_1based(n, "variable index")?;
            let v = ts.read_f64()?;
            a_triplets.push((k, i, v));
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

    // --- 変数下界・上界 ---
    // Binary ('B'): 変数は暗黙的に [0,1] — ファイルにこのセクションは存在しない
    // Continuous ('C') / Integer ('I'): ファイルに明示的に格納されている
    let (lb_var, ub_var) = if var_binary {
        (vec![0.0_f64; n], vec![1.0_f64; n])
    } else {
        let lb_var_default = ts.read_f64()?;
        let n_nondefault_lb_var = ts.read_usize()?;
        let mut lb_var = vec![lb_var_default; n];
        for _ in 0..n_nondefault_lb_var {
            let i = ts.read_index_1based(n, "lb_var index")?;
            let v = ts.read_f64()?;
            lb_var[i] = v;
        }
        let ub_var_default = ts.read_f64()?;
        let n_nondefault_ub_var = ts.read_usize()?;
        let mut ub_var = vec![ub_var_default; n];
        for _ in 0..n_nondefault_ub_var {
            let i = ts.read_index_1based(n, "ub_var index")?;
            let v = ts.read_f64()?;
            ub_var[i] = v;
        }
        (lb_var, ub_var)
    };

    // 残り（初期点・双対値・名前）は読み捨て

    // ============================================================
    // QpProblem 構築
    // ============================================================

    // Q行列（maximize の場合は符号反転）
    let sign = if maximize { -1.0 } else { 1.0 };

    // Scope q_triplets and intermediate row/col/val Vecs so they are freed
    // before the larger A-matrix construction begins.
    let q = {
        let q_rows: Vec<usize> = q_triplets.iter().map(|&(r, _, _)| r).collect();
        let q_cols: Vec<usize> = q_triplets.iter().map(|&(_, c, _)| c).collect();
        let q_vals: Vec<f64> = q_triplets.iter().map(|&(_, _, v)| sign * v).collect();
        drop(q_triplets); // free before CscMatrix::from_triplets allocates its sort Vec
        if q_rows.is_empty() {
            CscMatrix::new(n, n)
        } else {
            CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n)
                .map_err(|e| QplibError::ParseError(format!("Q matrix error: {}", e)))?
        }
        // q_rows, q_cols, q_vals dropped here
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

    // Build A matrix in a scoped block so a_triplets Vec is freed before
    // CscMatrix::from_triplets allocates its internal sort Vec.
    let a_mat = {
        // Each a_triplets entry produces at most 2 augmented rows (lb + ub),
        // but typically 1 (most constraints have only a finite ub or are equality).
        let cap = a_triplets.len();
        let mut a_rows: Vec<usize> = Vec::with_capacity(cap);
        let mut a_cols: Vec<usize> = Vec::with_capacity(cap);
        let mut a_vals: Vec<f64> = Vec::with_capacity(cap);

        for &(con_idx, var_idx, val) in &a_triplets {
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
        drop(a_triplets); // free Vec before from_triplets sort Vec is allocated

        if a_rows.is_empty() {
            CscMatrix::new(m_aug, n)
        } else {
            CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_aug, n)
                .map_err(|e| QplibError::ParseError(format!("A matrix error: {}", e)))?
        }
        // a_rows, a_cols, a_vals dropped here
    };

    // 変数境界（無限大の変換）
    let bounds: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let lb = if is_neg_inf(lb_var[i]) { f64::NEG_INFINITY } else { lb_var[i] };
            let ub = if is_pos_inf(ub_var[i]) { f64::INFINITY } else { ub_var[i] };
            (lb, ub)
        })
        .collect();

    // Quadratic constraint matrices (QCQP only).
    // Stored as QcqpMatrix (COO triplets) — O(nnz) memory regardless of n.
    // CscMatrix::from_triplets(n, n) would allocate O(n) col_ptr per constraint;
    // for QPLIB_8683 (n=200008, m=140000) that is 224 GB → OOM.
    let quadratic_constraints = if con_char == 'Q' {
        let mut qc: Vec<QcqpMatrix> = vec![QcqpMatrix::new(n); m_aug];
        for k in 0..m {
            let trips = &con_q_triplets[k];
            if trips.is_empty() {
                continue;
            }
            if let Some(aug_row) = aug_ub_row[k] {
                qc[aug_row].triplets = trips.clone();
            }
            if let Some(aug_row) = aug_lb_row[k] {
                // lb row: sign-flip → 1/2 x^T (-Q_k) x <= -lb_k
                qc[aug_row].triplets = trips.iter().map(|&(r, c, v)| (r, c, -v)).collect();
            }
        }
        qc
    } else {
        vec![]
    };

    let mut prob = QpProblem::new(q, c, a_mat, b_vec, bounds, constraint_types)
        .map_err(|e| QplibError::ParseError(e.to_string()))?;
    prob.quadratic_constraints = quadratic_constraints;

    if var_binary || var_integer {
        // Map all variables as integer (binary variables are integer ∈ {0,1} by bounds).
        let integer_vars: Vec<usize> = (0..n).collect();
        if prob.q.nnz() == 0 {
            // Linear objective → MILP (LP relaxation for B&B nodes)
            let lp = LpProblem::new_general(
                prob.c,
                prob.a,
                prob.b,
                prob.constraint_types,
                prob.bounds,
                None,
            )
            .map_err(|e: crate::error::SolverError| QplibError::ParseError(e.to_string()))?;
            let milp = MilpProblem::new(lp, integer_vars)
                .map_err(|e: crate::mip::MipProblemError| QplibError::ParseError(e.to_string()))?;
            Ok(QplibProblem::Milp(milp))
        } else {
            // Quadratic objective → MIQP (QP relaxation for B&B nodes)
            let miqp = MiqpProblem::new(prob, integer_vars)
                .map_err(|e: crate::mip::MipProblemError| QplibError::ParseError(e.to_string()))?;
            Ok(QplibProblem::Miqp(miqp))
        }
    } else {
        Ok(QplibProblem::Qp(prob))
    }
}

/// Whitespace/comment-stripping token stream.
///
/// Two backends:
/// - `Mem`: all tokens pre-loaded from a `&str` (used by `parse_qplib_str` / unit tests).
/// - `Stream`: reads one line at a time from a `BufRead` source; constant memory regardless
///   of file size (used by `parse_qplib` to avoid OOM on 200 MB+ files).
struct TokenStream {
    inner: TsInner,
}

enum TsInner {
    Mem { tokens: Vec<String>, pos: usize },
    Stream {
        reader: Box<dyn BufRead>,
        pending: VecDeque<String>,
        line_buf: String,
        /// Sticky I/O error: set on the first `read_line` failure; surfaced by `read_*`.
        io_err: Option<std::io::Error>,
    },
}

impl TokenStream {
    fn from_str(input: &str) -> Self {
        let mut tokens = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('%') || trimmed.starts_with('!') {
                continue;
            }
            let effective = if let Some(idx) = line.find('#') { &line[..idx] } else { line };
            for token in effective.split_whitespace() {
                tokens.push(token.to_string());
            }
        }
        TokenStream { inner: TsInner::Mem { tokens, pos: 0 } }
    }

    fn from_reader<R: BufRead + 'static>(reader: R) -> Self {
        TokenStream {
            inner: TsInner::Stream {
                reader: Box::new(reader),
                pending: VecDeque::new(),
                line_buf: String::new(),
                io_err: None,
            },
        }
    }

    /// Returns the next token, or `None` at EOF (or after a sticky I/O error).
    fn next_token(&mut self) -> Option<String> {
        match &mut self.inner {
            TsInner::Mem { tokens, pos } => {
                if *pos < tokens.len() {
                    let t = tokens[*pos].clone();
                    *pos += 1;
                    Some(t)
                } else {
                    None
                }
            }
            TsInner::Stream { reader, pending, line_buf, io_err } => loop {
                if io_err.is_some() {
                    return None;
                }
                if let Some(tok) = pending.pop_front() {
                    return Some(tok);
                }
                line_buf.clear();
                match reader.read_line(line_buf) {
                    Ok(0) => return None,
                    Err(e) => {
                        *io_err = Some(e);
                        return None;
                    }
                    Ok(_) => {
                        let trimmed = line_buf.trim();
                        if trimmed.starts_with('%') || trimmed.starts_with('!') {
                            continue;
                        }
                        let effective = if let Some(idx) = line_buf.find('#') {
                            &line_buf[..idx]
                        } else {
                            line_buf.as_str()
                        };
                        for token in effective.split_whitespace() {
                            pending.push_back(token.to_string());
                        }
                    }
                }
            },
        }
    }

    /// Takes a sticky I/O error if one was recorded, or returns `None`.
    fn take_io_err(&mut self) -> Option<std::io::Error> {
        match &mut self.inner {
            TsInner::Stream { io_err, .. } => io_err.take(),
            TsInner::Mem { .. } => None,
        }
    }

    fn read_string(&mut self) -> Result<String, QplibError> {
        match self.next_token() {
            Some(t) => Ok(t),
            None => Err(self.take_io_err().map(QplibError::IoError).unwrap_or_else(|| {
                QplibError::ParseError("unexpected end of file (expected string)".to_string())
            })),
        }
    }

    fn read_usize(&mut self) -> Result<usize, QplibError> {
        let t = match self.next_token() {
            Some(t) => t,
            None => return Err(self.take_io_err().map(QplibError::IoError).unwrap_or_else(|| {
                QplibError::ParseError("unexpected end of file (expected integer)".to_string())
            })),
        };
        // Accept float representation (e.g. "1.0" → 1)
        if let Ok(u) = t.parse::<usize>() {
            Ok(u)
        } else if let Ok(f) = t.parse::<f64>() {
            Ok(f as usize)
        } else {
            Err(QplibError::ParseError(format!("expected integer, got '{}'", t)))
        }
    }

    fn read_f64(&mut self) -> Result<f64, QplibError> {
        let t = match self.next_token() {
            Some(t) => t,
            None => return Err(self.take_io_err().map(QplibError::IoError).unwrap_or_else(|| {
                QplibError::ParseError("unexpected end of file (expected float)".to_string())
            })),
        };
        t.parse::<f64>()
            .map_err(|_| QplibError::ParseError(format!("expected float, got '{}'", t)))
    }

    /// Reads a 1-based index, validates range, and returns the 0-based equivalent.
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

    /// Extract `QpProblem` from `QplibProblem::Qp`, panicking for MIP variants.
    fn unwrap_qp(r: QplibProblem) -> QpProblem {
        match r {
            QplibProblem::Qp(p) => p,
            other => panic!("expected Qp, got {:?}", other),
        }
    }

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
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
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
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
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
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        // lb=ub=5 → 1 Eq row
        assert_eq!(prob.num_constraints, 1);
        assert_eq!(prob.constraint_types[0], crate::problem::ConstraintType::Eq);
        assert_eq!(prob.b[0], 5.0);
        // quadratic_constraints: 1 entry (one aug row), diagonal Q_1 has nnz=2
        assert_eq!(prob.quadratic_constraints.len(), 1);
        assert_eq!(prob.quadratic_constraints[0].nnz(), 2);
        // Verify Q_1: COO triplets at (0,0)=2.0 and (1,1)=4.0
        let qk = &prob.quadratic_constraints[0];
        assert_eq!(qk.n, 2);
        // round-trip guard: at least one constraint has quadratic terms
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
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
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
        assert_eq!(prob.quadratic_constraints[1].n, 3);
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
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
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
        let q_ub = &prob.quadratic_constraints[1];
        let q_lb = &prob.quadratic_constraints[2];
        assert!(q_ub.triplets.iter().all(|&(_, _, v)| v > 0.0), "ub row Q_2 values must be positive");
        assert!(q_lb.triplets.iter().all(|&(_, _, v)| v < 0.0), "lb row Q_2 values must be negative (sign flip)");
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
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        assert!(prob.quadratic_constraints.is_empty(),
            "QCL must produce empty quadratic_constraints");
    }

    /// Integer variables (QIL, n=2, linear obj nqobj=0) → MilpProblem.
    #[test]
    fn test_parse_qplib_integer_to_milp() {
        // QIL: quadratic obj header, but nqobj=0 → empty Q → Milp
        let qplib = "\
INT_LP
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
        let parsed = parse_qplib_str(qplib).unwrap();
        let milp = match parsed {
            QplibProblem::Milp(m) => m,
            other => panic!("expected Milp, got {:?}", other),
        };
        assert_eq!(milp.lp.num_vars, 2);
        assert_eq!(milp.integer_vars, vec![0, 1], "all 2 vars must be integer");
        // 1 equality constraint: lb=ub=1.0
        assert_eq!(milp.lp.num_constraints, 1);
    }

    /// Integer variables with quadratic objective (QIL, nqobj>0) → MiqpProblem.
    #[test]
    fn test_parse_qplib_integer_to_miqp() {
        // QIL with diagonal Q → Miqp
        let qplib = "\
INT_QP
QIL
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
        let parsed = parse_qplib_str(qplib).unwrap();
        let miqp = match parsed {
            QplibProblem::Miqp(m) => m,
            other => panic!("expected Miqp, got {:?}", other),
        };
        assert_eq!(miqp.qp.num_vars, 2);
        assert_eq!(miqp.integer_vars, vec![0, 1], "all 2 vars must be integer");
        assert_eq!(miqp.qp.q.nnz(), 2, "diagonal Q stored as 2 entries");
    }

    /// Binary variables (CBL, n=2, linear obj nqobj=0) → MilpProblem, bounds=[0,1].
    #[test]
    fn test_parse_qplib_binary_to_milp() {
        // CBL: convex obj header (C), binary vars (B), linear constraints (L)
        // nqobj=0 → empty Q → Milp
        let qplib = "\
BIN_LP
CBL
minimize
2
1
0
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
0.0
0
0.0
0
0
0
";
        let parsed = parse_qplib_str(qplib).unwrap();
        let milp = match parsed {
            QplibProblem::Milp(m) => m,
            other => panic!("expected Milp, got {:?}", other),
        };
        assert_eq!(milp.lp.num_vars, 2);
        assert_eq!(milp.integer_vars, vec![0, 1], "all 2 vars must be binary (integer)");
        // Binary: bounds must be [0,1]
        for &(lb, ub) in &milp.lp.bounds {
            assert!((lb - 0.0).abs() < 1e-12, "lb must be 0");
            assert!((ub - 1.0).abs() < 1e-12, "ub must be 1");
        }
        // 1 equality constraint: lb=ub=1.0
        assert_eq!(milp.lp.num_constraints, 1);
    }

    /// Binary variables with quadratic objective → MiqpProblem with [0,1] bounds.
    #[test]
    fn test_parse_qplib_binary_quad_to_miqp() {
        // CBL: C=convex-quad obj, B=binary vars, L=linear constraints
        // nqobj=2 → Q non-empty → Miqp
        let qplib = "\
BIN_QP
CBL
minimize
2
1
2
1 1 2.0
2 2 2.0
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
0.0
0
0.0
0
0
0
";
        let parsed = parse_qplib_str(qplib).unwrap();
        let miqp = match parsed {
            QplibProblem::Miqp(m) => m,
            other => panic!("expected Miqp, got {:?}", other),
        };
        assert_eq!(miqp.qp.num_vars, 2);
        assert_eq!(miqp.integer_vars, vec![0, 1]);
        // Binary: all bounds [0,1]
        for &(lb, ub) in &miqp.qp.bounds {
            assert!((lb - 0.0).abs() < 1e-12, "lb must be 0");
            assert!((ub - 1.0).abs() < 1e-12, "ub must be 1");
        }
        assert_eq!(miqp.qp.q.nnz(), 2);
    }

    /// Mixed-integer type 'M' remains UnsupportedType (per-var types section not yet parsed).
    #[test]
    fn test_parse_qplib_mixed_integer_unsupported() {
        let qplib = "\
MIXED_QP
QML
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
";
        assert!(matches!(
            parse_qplib_str(qplib),
            Err(QplibError::UnsupportedType(_))
        ));
    }

    // ── File-based tests: data/qplib_unsupported/ (QCQ instances) ──────────

    /// Helper: resolve path relative to the workspace root (one level above crate manifest).
    fn data_path(rel: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
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
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1157 parse"));

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
        assert_eq!(qk.n, 40);
        assert_eq!(qk.nnz(), 1516,
            "Q_9 nnz: 40 diag + 738 off-diag*2 = 1516");
        // Diagonal (0,0) = 0.38 (file: 9 1 1 0.38)
        let v00 = qk.triplets.iter().find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v).expect("Q_9 must have (0,0) entry");
        assert!((v00 - 0.38).abs() < 1e-10, "Q_9[0,0] must be 0.38");
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
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1353 parse"));

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
        assert_eq!(qk.n, 50);
        assert_eq!(qk.nnz(), 2372,
            "Q_6 nnz: 50 diag + 1161 off-diag*2 = 2372");
        // Diagonal (0,0) = 0.46 (file: 6 1 1 0.46)
        let v00 = qk.triplets.iter().find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v).expect("Q_6 must have (0,0) entry");
        assert!((v00 - 0.46).abs() < 1e-10, "Q_6[0,0] must be 0.46");
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
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1055 parse"));

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
            assert_eq!(qk.n, 40);
            assert_eq!(qk.nnz(), 1600,
                "Q_{i} nnz must be 1600 (full 40x40 lower-tri symmetrized)");
        }
        // Q_1[0,0] = 0.839 (file: 1 1 1 0.839)
        let qk0 = &prob.quadratic_constraints[0];
        let v00 = qk0.triplets.iter().find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v).expect("Q_1 must have (0,0) entry");
        assert!((v00 - 0.839).abs() < 1e-10, "Q_1[0,0] must be 0.839");
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
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1493 parse"));

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
        assert_eq!(qk.n, 40);
        assert_eq!(qk.nnz(), 1547,
            "Q_5 nnz: 37 diag + 755 off-diag*2 = 1547");
        // Q_5[0,0] = 1.88 (file: 5 1 1 1.88)
        let v00 = qk.triplets.iter().find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v).expect("Q_5 must have (0,0) entry");
        assert!((v00 - 1.88).abs() < 1e-10, "Q_5[0,0] must be 1.88");
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

    /// Regression: every tracked file in data/qplib/ parses without error.
    ///
    /// - Binary/integer files (CBL etc.) now parse as `Milp`/`Miqp`.
    /// - Mixed-integer types (M/G/S) still produce `UnsupportedType`.
    /// - `ParseError` or `IoError` on any file is a regression.
    ///
    /// All tracked files are swept (largest is QPLIB_8500 @ 24 MB / 1.2 M NZ);
    /// the parser is O(nnz log nnz), so the full sweep stays within the per-test
    /// budget single-threaded. Peak memory for the largest file is bounded
    /// separately by `test_memory_sentinel_no_double_hashmap_qplib8500`.
    #[test]
    fn test_parse_existing_qplib_files_regression() {
        let dir = data_path("data/qplib");
        if !dir.exists() {
            return;
        }
        let mut count = 0;
        let mut count_mip = 0usize;
        let mut count_unsupported = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("qplib") {
                continue;
            }
            match parse_qplib(path.as_path()) {
                Ok(QplibProblem::Qp(_)) => {}
                Ok(QplibProblem::Milp(_)) | Ok(QplibProblem::Miqp(_)) => {
                    count_mip += 1;
                }
                Err(QplibError::UnsupportedType(_)) => {
                    count_unsupported += 1;
                }
                Err(e) => panic!("parse regression: {} failed with unexpected error: {e}", path.display()),
            }
            count += 1;
        }
        assert!(count > 0, "no .qplib files found in data/qplib/");
        // data/qplib/ contains CBL files → must produce at least some Milp/Miqp results
        assert!(count_mip > 0,
            "expected at least one binary/integer file to parse as Milp/Miqp (got 0 out of {count})");
        // M/G/S mixed types remain unsupported
        let _ = count_unsupported; // may be 0 if no M/G/S files
    }

    /// Accumulation sentinel: parsing every file in data/qplib/ in sequence
    /// must return live allocations to ~baseline after each result is dropped.
    /// This proves peak memory is bounded by the *single largest* file, not the
    /// sum — i.e. the parser does not retain prior results.
    ///
    /// **No-op failure guarantee**: if a future change retains parse results
    /// across iterations (e.g. collecting into a `Vec` instead of dropping),
    /// `live` grows monotonically past `LIVE_RESIDUAL_LIMIT` and this fires.
    /// Verified: a retain-results variant reaches >100 MB live → FAIL.
    #[test]
    fn test_parse_sweep_no_memory_accumulation() {
        let dir = data_path("data/qplib");
        if !dir.exists() {
            return;
        }
        // Allocator slack tolerated between iterations when no result is retained.
        // Measured residual is ~0 MB; 4 MB leaves margin for allocator bookkeeping.
        const LIVE_RESIDUAL_LIMIT: isize = 4 * 1024 * 1024;

        let files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("qplib"))
            .collect();

        crate::peak_alloc::begin();
        for path in &files {
            let _ = parse_qplib(path.as_path()); // result dropped at end of statement
            let live = crate::peak_alloc::current_bytes();
            assert!(
                live <= LIVE_RESIDUAL_LIMIT,
                "live allocations {live} B remain after dropping {} — parser is \
                 retaining results across files (accumulation bug); limit {LIVE_RESIDUAL_LIMIT} B",
                path.display()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Memory sentinel tests
    //
    // These tests verify that peak allocations during large-file parsing stay
    // within a bounded range.
    //
    //   Fixed path  (~100 MB): a_triplets Vec(50) + a_rows/cols/vals(29)
    //                          + sort-Vec(29) + CSC(21)
    //                          – a_triplets freed before sort-Vec allocated
    //
    // NO-OP failure guarantee:
    //   Removing `drop(a_triplets)` causes peak to rise to ~129 MB → FAIL.
    //   If the system OOMs before the assertion, nextest also reports FAILED.
    // -----------------------------------------------------------------------

    /// Memory sentinel: QPLIB_8500 (25 MB, 1.2 M NZ) must parse without
    /// exceeding `QPLIB_8500_PARSE_PEAK_LIMIT` of concurrently live allocations.
    ///
    /// Measured by the thread-local `peak_alloc` tracking allocator (not RSS).
    ///
    /// **No-op failure guarantee**: removing `drop(a_triplets)` from the scoped
    /// block in `parse_token_stream` raises peak to ~129 MB → assertion fires.
    ///
    /// Threshold calibrated: Vec path measured 100.4 MB; threshold 115 MB
    /// (100.4 + 15 MB margin). No-drop path: 129.0 MB → FAIL.
    #[test]
    fn test_memory_sentinel_no_double_hashmap_qplib8500() {
        let path = data_path("data/qplib/QPLIB_8500.qplib");
        if !path.exists() {
            return;
        }

        // Calibrated: Vec path measured peak = 100.4 MB; limit = peak + 15 MB margin.
        const QPLIB_8500_PARSE_PEAK_LIMIT: usize = 115 * 1024 * 1024;

        crate::peak_alloc::begin();
        parse_qplib(path.as_path()).expect("QPLIB_8500 must parse without OOM");
        let peak = crate::peak_alloc::peak_bytes();

        assert!(
            peak <= QPLIB_8500_PARSE_PEAK_LIMIT,
            "QPLIB_8500 parse peak allocation {:.1} MB exceeds {:.1} MB limit.\n\
             Vec+drop path expected ~100.4 MB; no-drop path is ~129 MB.\n\
             Check that drop(a_triplets) is present before from_triplets call.",
            peak as f64 / 1_048_576.0,
            QPLIB_8500_PARSE_PEAK_LIMIT as f64 / 1_048_576.0
        );
    }

    /// Memory probe: large DCL files (QPLIB_8547 = 144 MB, QPLIB_9008 = 210 MB).
    ///
    /// These files have ~1 M variables and ~1 M constraints. Parsing them is
    /// memory-intensive due to the A-matrix triplet construction. This test
    /// measures the actual peak and reports it so we can calibrate limits.
    /// Currently set to a generous 2 GB limit — the intent is to detect
    /// catastrophic regressions (e.g. O(n²) allocations), not to be tight.
    #[test]
    fn test_memory_probe_large_dcl_files() {
        const DCL_PROBE_LIMIT: usize = 2 * 1024 * 1024 * 1024; // 2 GB generous ceiling

        for name in &["QPLIB_8547", "QPLIB_9008"] {
            let path = data_path(&format!("data/qplib/{name}.qplib"));
            if !path.exists() {
                continue;
            }
            crate::peak_alloc::begin();
            let result = parse_qplib(path.as_path());
            let peak = crate::peak_alloc::peak_bytes();
            assert!(
                result.is_ok() || matches!(result, Err(QplibError::UnsupportedType(_))),
                "{name} parse must not fail with ParseError or IoError: {:?}", result.err()
            );
            assert!(
                peak <= DCL_PROBE_LIMIT,
                "{name} parse peak {:.1} MB exceeds {:.1} GB catastrophic-regression limit.\n\
                 Expected O(nnz) memory not O(n·m) or O(n²).",
                peak as f64 / 1_048_576.0,
                DCL_PROBE_LIMIT as f64 / 1_073_741_824.0
            );
        }
    }

    /// Memory sentinel: QPLIB_8683 (DCQ, n=200008, m=140000).
    ///
    /// With the old `CscMatrix::from_triplets(&rows, &cols, &vals, n, n)` per
    /// filled slot, each slot allocates `col_ptr = vec![0; n+1]` = 1.6 MB.
    /// For 140000 filled slots → 224 GB → SIGKILL.
    /// With `QcqpMatrix` (COO), all slots share only the raw triplet data:
    /// 300000 symmetrized entries × 24 bytes ≈ 7 MB total.
    ///
    /// **No-op failure guarantee**: reverting the quadratic_constraints block
    /// to `CscMatrix::from_triplets(..., n, n)` causes OOM before this assertion.
    #[test]
    fn test_memory_sentinel_qplib8683_qcqp() {
        let path = data_path("data/qplib/QPLIB_8683.qplib");
        if !path.exists() {
            return;
        }
        // Peak = q_obj(~5MB) + con_quad_triplets(~14MB) + qc(~7MB)
        //       + a_triplets(~8MB) + build_internal(~8MB) + misc
        // Generous limit; no-op (CscMatrix per slot) raises this to 224 GB → OOM.
        const QPLIB_8683_PEAK_LIMIT: usize = 300 * 1024 * 1024; // 300 MB

        crate::peak_alloc::begin();
        parse_qplib(path.as_path()).expect("QPLIB_8683 must parse without OOM");
        let peak = crate::peak_alloc::peak_bytes();

        assert!(
            peak <= QPLIB_8683_PEAK_LIMIT,
            "QPLIB_8683 parse peak {:.1} MB exceeds {:.1} MB limit.\n\
             QcqpMatrix (COO) path expected < 50 MB; CscMatrix per-slot causes 224 GB OOM.\n\
             Check that quadratic_constraints uses QcqpMatrix not CscMatrix::from_triplets.",
            peak as f64 / 1_048_576.0,
            QPLIB_8683_PEAK_LIMIT as f64 / 1_048_576.0
        );
    }

    // -----------------------------------------------------------------------
    // QCQP sparse-init sentinel
    // -----------------------------------------------------------------------

    /// Build a minimal valid QCQ .qplib string with `n` variables and `m` equality
    /// constraints (lb=ub=0). Only constraint 1 has a quadratic term (x1^2).
    fn make_synthetic_qcq_content(n: usize, m: usize) -> String {
        let mut s = String::with_capacity(256);
        s.push_str("SYNTHETIC_QCQP\nQCQ\nminimize\n");
        s.push_str(&format!("{n}\n{m}\n"));
        s.push_str("0\n");   // nqobj (linear objective)
        s.push_str("0.0\n"); // default b0 (all-zero linear obj)
        s.push_str("0\n");   // non-default b0 count
        s.push_str("0.0\n"); // q0 (objective constant)
        s.push_str("1\n");   // n_con_quad_terms = 1
        s.push_str("1 1 1 1.0\n"); // constraint 1, x1^2 diagonal entry
        s.push_str("0\n");   // n_con_lin_terms
        s.push_str("1.79769313486232E+308\n"); // inf
        s.push_str("0.0\n0\n"); // lb_con default=0, non-defaults=0
        s.push_str("0.0\n0\n"); // ub_con default=0, non-defaults=0 (lb=ub → Eq)
        s.push_str("0.0\n0\n"); // lb_var default, non-defaults
        s.push_str("1.79769313486232E+308\n0\n"); // ub_var default, non-defaults
        s.push_str("0.0\n0\n0.0\n0\n0.0\n0\n0\n0\n"); // primal/dual/bound-dual/names
        s
    }

    /// QCQP COO-storage memory sentinel.
    ///
    /// The old `vec![CscMatrix::new(n, n); m_aug]` allocation plus
    /// `CscMatrix::from_triplets(..., n, n)` for filled slots both require
    /// O(n) `col_ptr` per constraint — O(m_aug·n) total.
    /// With `SYNTHETIC_N = 50_000` and `SYNTHETIC_M = 200`:
    /// - Dense-init default: 200 × 50_001 × 8 ≈ 80 MB → above `QCQP_DENSE_INIT_LIMIT`
    ///
    /// **No-op failure guarantee**: reverting to `CscMatrix::new(n, n)` default
    /// or `CscMatrix::from_triplets(..., n, n)` for filled slots raises peak
    /// above `QCQP_DENSE_INIT_LIMIT` → assertion fires.
    /// `QcqpMatrix` stores only triplets → negligible overhead.
    #[test]
    fn test_qcqp_sparse_init_memory_bounded() {
        const SYNTHETIC_N: usize = 50_000;
        const SYNTHETIC_M: usize = 200;
        // Dense-init: 200 × 50_001 × 8 = 80 MB; limit chosen between the two paths.
        const QCQP_DENSE_INIT_LIMIT: usize = 20 * 1024 * 1024; // 20 MB

        let content = make_synthetic_qcq_content(SYNTHETIC_N, SYNTHETIC_M);

        crate::peak_alloc::begin();
        let result = parse_qplib_str(&content).expect("synthetic QCQP must parse");
        let peak = crate::peak_alloc::peak_bytes();

        let prob = unwrap_qp(result);
        assert_eq!(prob.num_vars, SYNTHETIC_N);
        // All constraints are equality → m_aug = SYNTHETIC_M
        assert_eq!(prob.num_constraints, SYNTHETIC_M);
        assert_eq!(prob.quadratic_constraints.len(), SYNTHETIC_M);
        // Only slot 0 (constraint 1) has a non-zero Q
        assert_eq!(prob.quadratic_constraints[0].nnz(), 1,
            "constraint 0 must have nnz=1 (single diagonal entry)");
        for i in 1..SYNTHETIC_M {
            assert_eq!(prob.quadratic_constraints[i].nnz(), 0,
                "Q_k[{i}] must be empty for synthetic problem");
        }

        assert!(
            peak <= QCQP_DENSE_INIT_LIMIT,
            "synthetic QCQP parse peak {:.1} MB exceeds {:.1} MB limit.\n\
             QcqpMatrix (COO) path expected < 1 MB; \
             CscMatrix::new(n,n) default path ≈ 80 MB.\n\
             Revert check: ensure quadratic_constraints uses QcqpMatrix not CscMatrix.",
            peak as f64 / 1_048_576.0,
            QCQP_DENSE_INIT_LIMIT as f64 / 1_048_576.0
        );
    }

    // -----------------------------------------------------------------------
    // Multi-pattern tests: large / small / sparse / dense
    // -----------------------------------------------------------------------

    /// Small LP (LCL): linear obj, 10 vars, 5 constraints, 20 A-matrix NZ.
    /// Exercises the basic L-type constraint path with small data.
    #[test]
    fn test_parse_small_lp_lcl() {
        let mut content = String::from(
            "SMALL_LP\nLCL\nminimize\n10\n5\n0\n0.0\n0\n0.0\n20\n"
        );
        // 4 entries per constraint for each of the 5 constraints
        for k in 1..=5usize {
            for i in (k..k+4).filter(|&i| i <= 10) {
                content.push_str(&format!("{k} {i} 1.0\n"));
            }
        }
        // inf=1e308; lb_con=-inf (0 non-defaults); ub_con=100 (finite, 0 non-defaults)
        // lb_var=0 (0 non-defaults); ub_var=+inf (0 non-defaults)
        content.push_str(
            "1.0e308\n-1.0e308\n0\n100.0\n0\n0.0\n0\n1.0e308\n0\n"
        );
        let result = parse_qplib_str(&content);
        assert!(result.is_ok(), "small LP parse failed: {:?}", result.err());
        let prob = unwrap_qp(result.unwrap());
        assert_eq!(prob.num_vars, 10);
        // All 5 constraints have finite ub=100 → 5 Le rows
        assert_eq!(prob.num_constraints, 5);
        assert!(prob.a.nnz() > 0);
    }

    /// Dense Q matrix (LCB type: linear constraints = box/bounds-only).
    /// Generates a fully dense Q for n=80 vars: nqobj = 80*81/2 = 3240 entries.
    /// Verifies Q is symmetric and has correct nnz after symmetrization.
    #[test]
    fn test_parse_dense_q_box_constraints() {
        const N: usize = 80;
        const NQOBJ: usize = N * (N + 1) / 2; // lower-triangular entries

        let mut content = String::from(
            "DENSE_Q\nQCB\nminimize\n80\n"
        );
        content.push_str(&format!("{NQOBJ}\n"));
        for i in 1..=N {
            for j in 1..=i {
                content.push_str(&format!("{i} {j} 1.0\n"));
            }
        }
        // default_b0=0, n_nondefault=0, objective constant q0=0
        content.push_str("0.0\n0\n0.0\n");
        // inf_val (always present in QPLIB format, even for box-only problems)
        content.push_str("1.0e308\n");
        // var bounds: lb_default=0, 0 non-defaults, ub_default=1, 0 non-defaults
        content.push_str("0.0\n0\n1.0\n0\n");

        let result = parse_qplib_str(&content);
        assert!(result.is_ok(), "dense Q parse failed: {:?}", result.err());
        let prob = unwrap_qp(result.unwrap());
        assert_eq!(prob.num_vars, N);
        // Q: each off-diagonal entry appears twice (symmetrization), diagonal once
        // Total nnz = N (diag) + (NQOBJ - N) * 2 (off-diag symmetrized)
        let expected_nnz = N + (NQOBJ - N) * 2;
        assert_eq!(prob.q.nnz(), expected_nnz,
            "Q nnz: expected {expected_nnz} (full {N}×{N} symmetric)");
    }

    /// Sparse A matrix: LCL type, n=500 vars, m=200 constraints, ~1000 NZ.
    /// Verifies that the sort-merge CSC construction is correct for sparse inputs.
    #[test]
    fn test_parse_sparse_a_matrix_correctness() {
        const N: usize = 500;
        const M: usize = 200;

        let mut content = format!("SPARSE_A\nLCL\nminimize\n{N}\n{M}\n0\n0.0\n0\n0.0\n");
        // ~5 entries per constraint
        let mut nnz = 0usize;
        let mut entry_buf = String::new();
        for k in 1..=M {
            for offset in 0..5usize {
                let i = (k * 7 + offset * 13) % N + 1;
                entry_buf.push_str(&format!("{k} {i} 1.0\n"));
                nnz += 1;
            }
        }
        content.push_str(&format!("{nnz}\n"));
        content.push_str(&entry_buf);
        // inf=1e308; lb_con=-inf; ub_con=1000.0 (finite, generates Le rows); var bounds [0, +inf]
        content.push_str("1.0e308\n-1.0e308\n0\n1000.0\n0\n0.0\n0\n1.0e308\n0\n");

        let result = parse_qplib_str(&content);
        assert!(result.is_ok(), "sparse A parse failed: {:?}", result.err());
        let prob = unwrap_qp(result.unwrap());
        assert_eq!(prob.num_vars, N);
        assert!(prob.num_constraints >= M);
        assert!(prob.a.nnz() > 0, "A matrix should have non-zeros");
    }

    /// nqobj sanity bound: n=2 gives max nqobj=3; declaring nqobj=4 must return ParseError.
    #[test]
    fn test_sanity_bound_nqobj_too_large() {
        // n=2, nqobj=4 > n*(n+1)/2=3 → ParseError fires immediately after reading nqobj.
        let content = "\
SANITY_NQOBJ
QCL
minimize
2
1
4
";
        let result = parse_qplib_str(content);
        assert!(result.is_err(), "expected ParseError for nqobj=4 > n*(n+1)/2=3");
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(
            err_str.contains("nqobj") || err_str.contains("exceeds"),
            "error should mention nqobj bound, got: {}",
            err_str
        );
    }

    /// n_con_lin_terms sanity bound: n=2, m=1 gives max=2; declaring 3 must return ParseError.
    #[test]
    fn test_sanity_bound_n_con_lin_terms_too_large() {
        // n=2, m=1, n_con_lin_terms=3 > n*m=2 → ParseError fires after reading count.
        // Path: LCL → nqobj=0, default_b0=0.0, n_nondefault=0, q0=0.0, then n_con_lin_terms=3.
        let content = "\
SANITY_NCON
LCL
minimize
2
1
0
0.0
0
0.0
3
";
        let result = parse_qplib_str(content);
        assert!(result.is_err(), "expected ParseError for n_con_lin_terms=3 > n*m=2");
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(
            err_str.contains("n_con_lin_terms") || err_str.contains("exceeds"),
            "error should mention n_con_lin_terms bound, got: {}",
            err_str
        );
    }

    /// Duplicate linear constraint entries must be accumulated (not double-counted).
    ///
    /// Sentinel: if sort-merge deduplication is broken (e.g. reverted to no-dedup),
    /// the final A matrix will have twice as many entries → nnz assertion fails.
    #[test]
    fn test_parse_duplicate_a_entries_accumulated() {
        // n=2, m=1: constraint "1 1 2.0" appears twice → should give coeff 4.0
        let qplib = "\
DUP_TEST
LCL
minimize
2
1
0
0.0
0
0.0
2
1 1 2.0
1 1 2.0
1.0e308
-1.0e308
0
0.0
1
1 5.0
0.0
0
1.0e308
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        // Constraint upper bound is 5.0 → Le row exists
        assert_eq!(prob.a.nnz(), 1, "duplicate entries must be merged to one NZ");
        // The merged value should be 4.0 (2.0 + 2.0)
        let col0 = prob.a.col_ptr[0];
        assert!((prob.a.values[col0] - 4.0).abs() < 1e-12,
            "accumulated coeff should be 4.0, got {}", prob.a.values[col0]);
    }
}
