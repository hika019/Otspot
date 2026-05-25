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

