//! MPS ファイル形式パーサー (LP / MILP)。
//!
//! NAME / ROWS / COLUMNS / RHS / RANGES / BOUNDS / ENDATA セクションを、固定幅・
//! フリーフォーマットの両方を自動判別して解析する。COLUMNS の INTORG/INTEND マーカーと
//! BV/LI/UI 境界で整数変数を識別する。
//!
//! - [`parse_mps`] / [`parse_mps_file`]: `LpProblem` を返す (整数性は破棄＝LP relaxation)。
//! - [`parse_milp`] / [`parse_milp_file`]: 整数変数付きの `MilpProblem` を返す。

use crate::mip::MilpProblem;
use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;

/// MPS の INTORG/INTEND マーカーで囲まれ、かつ BOUNDS 指定が一切ない整数変数の
/// デフォルト上限。古典的な OSL/CPLEX 規約では「明示境界のない整数変数は二値」と
/// 解釈する (HiGHS の MPS リーダーと一致)。明示境界が 1 つでもあればこの既定は無効。
const INTEGER_DEFAULT_UPPER_BINARY: f64 = 1.0;

/// MPSファイルのパース中に発生するエラー
#[non_exhaustive]
#[derive(Debug)]
pub enum MpsError {
    /// ファイル読み込み時のI/Oエラー
    IoError(std::io::Error),
    /// 指定行のパースエラー（行番号とメッセージを含む）
    ParseError { line: usize, message: String },
    /// 必須セクションが欠落している
    MissingSection(String),
    /// 同じセクションが複数回出現した
    DuplicateSection(String),
    /// 無効な行タイプ文字が指定された
    InvalidRowType(char),
    /// 無効な上下限タイプ文字列が指定された
    InvalidBoundType(String),
    /// 未定義の行名または列名が参照された
    UndefinedReference { kind: String, name: String },
    /// INTORG マーカーが対応する INTEND で閉じられないまま COLUMNS を抜けた。
    /// 残り全列が無警告で整数化されるのを防ぐため明示エラーとする。
    UnclosedIntegerMarker,
}

impl std::fmt::Display for MpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MpsError::IoError(e) => write!(f, "I/O error: {}", e),
            MpsError::ParseError { line, message } => {
                write!(f, "Parse error at line {}: {}", line, message)
            }
            MpsError::MissingSection(s) => write!(f, "Missing required section: {}", s),
            MpsError::DuplicateSection(s) => write!(f, "Duplicate section: {}", s),
            MpsError::InvalidRowType(c) => write!(f, "Invalid row type: {}", c),
            MpsError::InvalidBoundType(s) => write!(f, "Invalid bound type: {}", s),
            MpsError::UndefinedReference { kind, name } => {
                write!(f, "Undefined {} reference: {}", kind, name)
            }
            MpsError::UnclosedIntegerMarker => {
                write!(f, "INTORG marker not closed by a matching INTEND in COLUMNS")
            }
        }
    }
}

impl std::error::Error for MpsError {}

impl From<std::io::Error> for MpsError {
    fn from(err: std::io::Error) -> Self {
        MpsError::IoError(err)
    }
}

/// ファイルパスからMPSファイルを読み込み、`LpProblem`としてパースする
///
/// Uses streaming I/O (`BufReader`) — peak memory is proportional to the largest
/// single line, not the entire file.
///
/// # Errors
///
/// - ファイルが存在しない、または読み取り権限がない場合は`MpsError::IoError`
/// - ファイル内容のパースに失敗した場合は各種`MpsError`バリアント
pub fn parse_mps_file(path: &Path) -> Result<LpProblem, MpsError> {
    let file = std::fs::File::open(path)?;
    parse_mps_reader(std::io::BufReader::new(file))
}

/// `BufRead` ストリームから MPS をパースし `LpProblem` を返す。
///
/// Large files can be parsed without loading the entire content into memory.
pub fn parse_mps_reader<R: BufRead>(reader: R) -> Result<LpProblem, MpsError> {
    let mut parser = MpsParser::new();
    let (lp, _integer_vars) = parser.parse_reader(reader)?;
    Ok(lp)
}

/// ファイルパスからMILPを読み込み、`MilpProblem`としてパースする
///
/// Uses streaming I/O (`BufReader`). INTORG/INTEND マーカーおよび BV/LI/UI 境界で
/// 識別した整数変数を保持する。整数変数が存在しない場合は整数集合が空の `MilpProblem`
/// (= LP) を返す。
///
/// # Errors
///
/// - ファイルが存在しない、または読み取り権限がない場合は`MpsError::IoError`
/// - ファイル内容のパースに失敗した場合は各種`MpsError`バリアント
pub fn parse_milp_file(path: &Path) -> Result<MilpProblem, MpsError> {
    let file = std::fs::File::open(path)?;
    parse_milp_reader(std::io::BufReader::new(file))
}

/// `BufRead` ストリームから MPS をパースし `MilpProblem` を返す。
pub fn parse_milp_reader<R: BufRead>(reader: R) -> Result<MilpProblem, MpsError> {
    let mut parser = MpsParser::new();
    let (lp, integer_vars) = parser.parse_reader(reader)?;
    MilpProblem::new(lp, integer_vars).map_err(|e| MpsError::ParseError {
        line: 0,
        message: e.to_string(),
    })
}

/// MPS形式の文字列を`LpProblem`にパースする
///
/// 固定幅フォーマットとフリーフォーマットの両方を自動判別して処理します。
///
/// # MILP ファイルを読む場合の挙動 (LP relaxation)
///
/// INTORG/INTEND マーカーや BV/LI/UI 境界を含む MILP ファイルを渡しても、本関数は
/// **整数性を破棄した LP relaxation** を返します。整数変数の境界は保持され、明示境界の
/// ない整数変数は二値 [0,1] に既定されます (`parse_milp` と同じ境界処理)。整数制約を
/// 保持したいときは [`parse_milp`] を使ってください。
///
/// # 引数
///
/// * `input` - MPSフォーマットの文字列
///
/// # 戻り値
///
/// 成功時は`LpProblem`、失敗時は`MpsError`を返す
///
/// # Examples
///
/// ```
/// use otspot::io::mps::parse_mps;
///
/// let mps = r"NAME          example
/// ROWS
///  N  obj
///  L  c1
/// COLUMNS
///     x1  obj  1.0  c1  2.0
/// RHS
///     rhs  c1  10.0
/// ENDATA
/// ";
/// let lp = parse_mps(mps).unwrap();
/// assert_eq!(lp.num_vars, 1);
/// assert_eq!(lp.num_constraints, 1);
/// ```
pub fn parse_mps(input: &str) -> Result<LpProblem, MpsError> {
    parse_mps_reader(std::io::Cursor::new(input.as_bytes()))
}

/// MPS形式の文字列を`MilpProblem`にパースする
///
/// `parse_mps` と同じパーサを用いるが、INTORG/INTEND マーカーおよび BV/LI/UI 境界で
/// 識別した整数変数を保持した [`MilpProblem`] を返す。整数変数が無ければ整数集合が空の
/// MILP (= LP relaxation) となる。
///
/// # Examples
///
/// ```
/// use otspot::io::mps::parse_milp;
///
/// let mps = r"NAME          milp
/// ROWS
///  N  obj
///  L  c1
/// COLUMNS
///     MARKER1   'MARKER'   'INTORG'
///     x1  obj  -1.0  c1  1.0
///     MARKER2   'MARKER'   'INTEND'
/// RHS
///     rhs  c1  10.5
/// BOUNDS
///  UP BND  x1  7.0
/// ENDATA
/// ";
/// let milp = parse_milp(mps).unwrap();
/// assert_eq!(milp.integer_vars, vec![0]);
/// ```
pub fn parse_milp(input: &str) -> Result<MilpProblem, MpsError> {
    parse_milp_reader(std::io::Cursor::new(input.as_bytes()))
}

/// MPSファイルのパース処理を管理する内部構造体
///
/// パース中間状態をフィールドに保持し、完了後に`LpProblem`を構築します。
struct MpsParser {
    /// 問題名（NAMEセクションから取得）
    problem_name: Option<String>,
    /// 行定義: (行名, 行タイプ) のリスト
    rows: Vec<(String, RowType)>,
    /// 列係数: (列名, 行名, 係数値) のリスト
    columns: Vec<(String, String, f64)>,
    /// 右辺値: 行名 → RHS値 のマップ
    rhs: HashMap<String, f64>,
    /// 幅指定: 行名 → RANGE値 のマップ
    ranges: HashMap<String, f64>,
    /// 上下限制約: (境界タイプ, 列名, 値) のリスト
    bounds: Vec<(BoundType, String, Option<f64>)>,
    /// 目的関数行の行名
    obj_row: Option<String>,
    /// 整数変数の列名集合。INTORG/INTEND マーカー領域内の列、または
    /// BV/LI/UI 境界タイプを持つ列を整数とみなす。
    integer_cols: HashSet<String>,
    /// COLUMNS パース中、現在 INTORG/INTEND マーカー領域内かどうか。
    in_integer_marker: bool,
}

/// COLUMNS セクション内の整数マーカー行の種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerMarker {
    /// `'MARKER' 'INTORG'`: 以降の列を整数として開始。
    Start,
    /// `'MARKER' 'INTEND'`: 整数領域の終了。
    End,
}

/// COLUMNS 行が整数マーカー (`'MARKER' 'INTORG'`/`'INTEND'`) かどうかを判定する。
///
/// `'MARKER'` トークンと `INTORG`/`INTEND` トークンの両方が揃った行のみマーカーと
/// みなす。クォートは除去して大文字小文字を無視する。両方揃わない通常の列名 (例えば
/// `INTORG` という名の列) を誤検出しないため、両トークンの共起を要求する。
fn integer_marker_kind(line: &str) -> Option<IntegerMarker> {
    let mut has_marker = false;
    let mut kind = None;
    for tok in line.split_whitespace() {
        match tok.trim_matches('\'').to_uppercase().as_str() {
            "MARKER" => has_marker = true,
            "INTORG" => kind = Some(IntegerMarker::Start),
            "INTEND" => kind = Some(IntegerMarker::End),
            _ => {}
        }
    }
    if has_marker {
        kind
    } else {
        None
    }
}

/// MPSファイルのROWSセクションにおける行タイプ
#[derive(Debug, Clone, Copy)]
enum RowType {
    /// 目的関数行（Nは"free"を意味する）
    N,
    /// 上限制約: Ax <= b
    L,
    /// 下限制約: Ax >= b
    G,
    /// 等式制約: Ax == b
    E,
}

/// MPSファイルのBOUNDSセクションにおける上下限タイプ
#[derive(Debug, Clone, Copy)]
enum BoundType {
    /// 下限値 (Lower bound): x >= value
    LO,
    /// 上限値 (Upper bound): x <= value
    UP,
    /// 固定値 (Fixed): x == value
    FX,
    /// 自由変数 (Free): -∞ <= x <= +∞
    FR,
    /// 下限マイナス無限大 (Minus infinity): x >= -∞
    MI,
    /// 2値変数 (Binary variable): x ∈ {0, 1}
    BV,
    /// Plus infinity: x <= +∞ (upper bound = +∞, lower bound unchanged at default 0)
    PL,
    /// 整数下限 (Lower integer): x >= value かつ x は整数。LO と同じ境界効果に加え
    /// 当該変数を整数として登録する。
    LI,
    /// 整数上限 (Upper integer): x <= value かつ x は整数。UP と同じ境界効果に加え
    /// 当該変数を整数として登録する。
    UI,
}

/// 固定幅MPSフォーマットかどうかを判定する
///
/// MPSの固定幅フォーマットは、列15（0-indexed: 14）が空白であることで識別する。
/// 以下のケースでは自由形式（false）として扱う:
/// - 空行
/// - 14文字以下の行（列15が存在しない）
/// - タブ文字は空白として扱うため `is_whitespace()` で正しく判定される
fn is_fixed_width_format(line: &str) -> bool {
    // chars().nth(14) は15文字未満の行でNoneを返すため、
    // 空行・短い行の境界ケースを自然に処理できる
    line.chars()
        .nth(14)
        .is_some_and(|c| c.is_whitespace())
}

impl MpsParser {
    /// 新しい空の`MpsParser`を生成する
    fn new() -> Self {
        Self {
            problem_name: None,
            rows: Vec::new(),
            columns: Vec::new(),
            rhs: HashMap::new(),
            ranges: HashMap::new(),
            bounds: Vec::new(),
            obj_row: None,
            integer_cols: HashSet::new(),
            in_integer_marker: false,
        }
    }

    /// MPS ストリームを行単位で読み込み、LP relaxation と整数変数インデックスを返す。
    ///
    /// Uses `BufRead::lines()` so only one line is held in memory at a time.
    /// Supports both file-backed `BufReader` and in-memory `Cursor`.
    fn parse_reader<R: BufRead>(&mut self, reader: R) -> Result<(LpProblem, Vec<usize>), MpsError> {
        let mut current_section = Section::None;
        let mut seen_sections = std::collections::HashSet::new();
        let mut line_num = 0;

        for line_result in reader.lines() {
            let line = line_result.map_err(MpsError::IoError)?;
            line_num += 1;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('*') {
                continue;
            }

            if !line.starts_with(' ') && !line.starts_with('\t') {
                if let Some(section) = Section::from_line(trimmed) {
                    if self.in_integer_marker && section != Section::Columns {
                        return Err(MpsError::UnclosedIntegerMarker);
                    }
                    if section != Section::Name && section != Section::EndData
                        && seen_sections.contains(&section)
                    {
                        return Err(MpsError::DuplicateSection(format!("{:?}", section)));
                    }
                    seen_sections.insert(section);
                    current_section = section;

                    if section == Section::Name && trimmed.len() > 4 {
                        let name_part = trimmed[4..].trim();
                        if !name_part.is_empty() {
                            self.problem_name = Some(name_part.to_string());
                        }
                    }
                    if section == Section::EndData {
                        break;
                    }
                    continue;
                }
            }

            match current_section {
                Section::None => {
                    return Err(MpsError::ParseError {
                        line: line_num,
                        message: "Line appears before any section header".to_string(),
                    });
                }
                Section::Name => {}
                Section::Rows => self.parse_rows_line(&line, line_num)?,
                Section::Columns => self.parse_columns_line(&line, line_num)?,
                Section::Rhs => self.parse_rhs_line(&line, line_num)?,
                Section::Ranges => self.parse_ranges_line(&line, line_num)?,
                Section::Bounds => self.parse_bounds_line(&line, line_num)?,
                Section::EndData => break,
            }
        }

        if !seen_sections.contains(&Section::EndData) {
            return Err(MpsError::MissingSection("ENDATA".to_string()));
        }
        if !seen_sections.contains(&Section::Rows) {
            return Err(MpsError::MissingSection("ROWS".to_string()));
        }
        if !seen_sections.contains(&Section::Columns) {
            return Err(MpsError::MissingSection("COLUMNS".to_string()));
        }

        self.build_lp_problem()
    }

    /// ROWSセクションの1行をパースして行定義を追加する
    ///
    /// 書式: `<行タイプ>  <行名>` （例: `L  constraint1`）
    /// 最初に現れるNタイプの行を目的関数行として記録します。
    fn parse_rows_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(MpsError::ParseError {
                line: line_num,
                message: "ROWS line must have at least 2 fields".to_string(),
            });
        }

        let row_type_str = parts[0];
        let row_name = parts[1].to_string();

        if row_type_str.len() != 1 {
            return Err(MpsError::ParseError {
                line: line_num,
                message: format!("Row type must be single character, got '{}'", row_type_str),
            });
        }

        let row_type_char = row_type_str.chars().next().unwrap();
        let row_type = match row_type_char {
            'N' => RowType::N,
            'L' => RowType::L,
            'G' => RowType::G,
            'E' => RowType::E,
            _ => return Err(MpsError::InvalidRowType(row_type_char)),
        };

        // 最初のNタイプ行を目的関数行として登録
        if matches!(row_type, RowType::N) && self.obj_row.is_none() {
            self.obj_row = Some(row_name.clone());
        }

        self.rows.push((row_name, row_type));
        Ok(())
    }

    /// COLUMNSセクションの1行をパースする
    ///
    /// 列位置を確認してフォーマットを自動判別し、
    /// 固定幅またはフリーフォーマットの対応関数に委譲します。
    fn parse_columns_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        // INTORG/INTEND マーカー行は係数を持たず、整数領域の開始/終了のみを切り替える。
        if let Some(kind) = integer_marker_kind(line) {
            self.in_integer_marker = matches!(kind, IntegerMarker::Start);
            return Ok(());
        }

        let is_fixed_width = is_fixed_width_format(line);

        if is_fixed_width {
            self.parse_columns_fixed(line, line_num)
        } else {
            self.parse_columns_free(line, line_num)
        }
    }

    /// 固定幅フォーマットのCOLUMNS行をパースする
    ///
    /// 固定フォーマットの列配置:
    /// - 列5〜12: 列名 (col_name)
    /// - 列15〜22: 行名1 (row_name)
    /// - 列25〜36: 係数値1 (value)
    /// - 列40〜47: 行名2（省略可）
    /// - 列50〜61: 係数値2（省略可）
    fn parse_columns_fixed(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        // 固定フォーマット: col5-12=col_name, col15-22=row_name, col25-36=value
        //                [col40-47=row_name2, col50-61=value2]
        let col_name = line.get(4..12).unwrap_or("").trim().to_string();
        let row_name1 = line.get(14..22).unwrap_or("").trim().to_string();
        let value1_str = line.get(24..36).unwrap_or("").trim();

        // 固定位置に (列名, 行名, 数値) が揃わなければ char-14 ヒューリスティックの誤判定。
        // 実体はフリーフォーマット (例: MIPLIB は名前が短く幅広パディングのため行名が
        // 列22以降にずれる) なのでフリー解析へ委譲する。これにより COLUMNS だけが固定
        // 解析で entry を取りこぼし、常にフリー解析の BOUNDS と列名が食い違う事故を防ぐ。
        if col_name.is_empty() || row_name1.is_empty() || value1_str.parse::<f64>().is_err() {
            return self.parse_columns_free(line, line_num);
        }
        if self.in_integer_marker {
            self.integer_cols.insert(col_name.clone());
        }

        let value1 = value1_str
            .parse::<f64>()
            .expect("value1_str parseable (checked above)");
        self.columns.push((col_name.clone(), row_name1, value1));

        // 2エントリ目（省略可）
        if line.len() >= 50 {
            let row_name2 = line.get(39..47).unwrap_or("").trim().to_string();
            let value2_str = line.get(49..61).unwrap_or("").trim();

            if !row_name2.is_empty() && !value2_str.is_empty() {
                let value2 = value2_str.parse::<f64>().map_err(|_| MpsError::ParseError {
                    line: line_num,
                    message: format!("Invalid numeric value: {}", value2_str),
                })?;
                self.columns.push((col_name.clone(), row_name2, value2));
            }
        }

        Ok(())
    }

    /// フリーフォーマットのCOLUMNS行をパースする
    ///
    /// 書式: `<列名>  <行名1>  <値1>  [<行名2>  <値2>]`
    /// 1行に最大2つの (行名, 値) ペアを記述できます。
    fn parse_columns_free(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(()); // 不完全な行はスキップ
        }

        let col_name = parts[0].to_string();
        if self.in_integer_marker {
            self.integer_cols.insert(col_name.clone());
        }

        // (行名, 値) のペアを順に処理
        for i in (1..parts.len()).step_by(2) {
            if i + 1 >= parts.len() {
                break;
            }
            let row_name = parts[i].to_string();
            let value = parts[i + 1].parse::<f64>().map_err(|_| MpsError::ParseError {
                line: line_num,
                message: format!("Invalid numeric value: {}", parts[i + 1]),
            })?;
            self.columns.push((col_name.clone(), row_name, value));
        }

        Ok(())
    }

    /// RHSセクションの1行をパースして右辺値を登録する
    ///
    /// 書式: `<RHS名>  <行名1>  <値1>  [<行名2>  <値2>]`
    /// RHS名は識別子として存在しますが、複数RHSは未サポートのため無視します。
    fn parse_rhs_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        // COLUMNSと同様のフォーマット
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }

        // RHS名（parts[0]）はスキップし、(行名, 値) ペアを処理
        for i in (1..parts.len()).step_by(2) {
            if i + 1 >= parts.len() {
                break;
            }
            let row_name = parts[i].to_string();
            let value = parts[i + 1].parse::<f64>().map_err(|_| MpsError::ParseError {
                line: line_num,
                message: format!("Invalid numeric value: {}", parts[i + 1]),
            })?;
            self.rhs.insert(row_name, value);
        }

        Ok(())
    }

    /// RANGESセクションの1行をパースして幅制約値を登録する
    ///
    /// 書式はRHSと同様です。RANGE値は制約の幅を定義し、
    /// `build_lp_problem`でRHS値の調整に使用されます。
    fn parse_ranges_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        // RHSと同様のフォーマット
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }

        for i in (1..parts.len()).step_by(2) {
            if i + 1 >= parts.len() {
                break;
            }
            let row_name = parts[i].to_string();
            let value = parts[i + 1].parse::<f64>().map_err(|_| MpsError::ParseError {
                line: line_num,
                message: format!("Invalid numeric value: {}", parts[i + 1]),
            })?;
            self.ranges.insert(row_name, value);
        }

        Ok(())
    }

    /// BOUNDSセクションの1行をパースして変数の上下限を登録する
    ///
    /// 書式: `<境界タイプ>  <境界名>  <列名>  [<値>]`
    /// FR・MI・BVタイプは値なしで指定できます。
    fn parse_bounds_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }

        let bound_type_str = parts[0];
        let _bound_name = parts[1]; // 境界名は通常無視される
        let col_name = parts[2].to_string();
        let value = if parts.len() >= 4 {
            Some(parts[3].parse::<f64>().map_err(|_| MpsError::ParseError {
                line: line_num,
                message: format!("Invalid numeric value: {}", parts[3]),
            })?)
        } else {
            None
        };

        let bound_type = match bound_type_str {
            "LO" => BoundType::LO,
            "UP" => BoundType::UP,
            "FX" => BoundType::FX,
            "FR" => BoundType::FR,
            "MI" => BoundType::MI,
            "BV" => BoundType::BV,
            "PL" => BoundType::PL,
            "LI" => BoundType::LI,
            "UI" => BoundType::UI,
            _ => return Err(MpsError::InvalidBoundType(bound_type_str.to_string())),
        };

        // BV/LI/UI は当該変数を整数として登録する (マーカー無しでも整数となる)。
        if matches!(bound_type, BoundType::BV | BoundType::LI | BoundType::UI) {
            self.integer_cols.insert(col_name.clone());
        }

        self.bounds.push((bound_type, col_name, value));
        Ok(())
    }

    /// パース済みデータから`LpProblem`と整数変数インデックスを構築する
    ///
    /// 処理の流れ:
    /// 1. 行名→インデックスマップを構築（目的関数行を除く）
    /// 2. RANGESを適用してRHS値を調整
    /// 3. 列名→インデックスマップを構築
    /// 4. 目的関数ベクトル`c`を構築
    /// 5. 制約行列`A`をCSC形式で構築
    /// 6. 変数の上下限ベクトルを構築
    /// 7. 整数変数インデックスを構築（明示境界のない整数変数に二値既定を適用）
    fn build_lp_problem(&self) -> Result<(LpProblem, Vec<usize>), MpsError> {
        // 行名 → インデックスのマップ構築（目的関数行を除外）
        let mut row_map = HashMap::new();
        let mut constraint_types = Vec::new();
        let mut rhs_vec = Vec::new();

        for (row_name, row_type) in &self.rows {
            if Some(row_name) == self.obj_row.as_ref() {
                continue; // 目的関数行はスキップ
            }

            let idx = row_map.len();
            row_map.insert(row_name.clone(), idx);

            let constraint_type = match row_type {
                RowType::L => ConstraintType::Le,
                RowType::G => ConstraintType::Ge,
                RowType::E => ConstraintType::Eq,
                RowType::N => continue, // 複数のN行はスキップ
            };
            constraint_types.push(constraint_type);

            let rhs_val = self.rhs.get(row_name).copied().unwrap_or(0.0);
            rhs_vec.push(rhs_val);
        }

        let base_num_constraints = row_map.len();

        // RANGESの適用: 区間制約（[lower, upper]）に変換する
        // 各RANGE行を上限（Le）と下限（Ge）の2制約に分割する行分割アプローチ
        //
        // 仕様（IBM MPS標準）:
        //   L制約: b - |r| <= Ax <= b
        //   G制約: b <= Ax <= b + |r|
        //   E制約(r>=0): b <= Ax <= b + |r|
        //   E制約(r<0):  b - |r| <= Ax <= b
        let mut range_extra_rows: Vec<(String, usize, f64)> = Vec::new();
        for (row_name, range_val) in &self.ranges {
            if let Some(&idx) = row_map.get(row_name) {
                let b = rhs_vec[idx];
                let abs_r = range_val.abs();

                let (lower, upper) = match constraint_types[idx] {
                    ConstraintType::Le => (b - abs_r, b),
                    ConstraintType::Ge => (b, b + abs_r),
                    ConstraintType::Eq => {
                        if *range_val >= 0.0 {
                            (b, b + abs_r)
                        } else {
                            (b - abs_r, b)
                        }
                    }
                };

                // 既存行をLe制約（上限側）に変更
                constraint_types[idx] = ConstraintType::Le;
                rhs_vec[idx] = upper;

                // 下限側の追加行を記録（行名, 元インデックス, lower_bound）
                range_extra_rows.push((row_name.clone(), idx, lower));
            }
        }

        // RANGE追加行のインデックスマップと制約情報を構築
        let mut range_row_map: HashMap<String, usize> = HashMap::new();
        for (row_name, _orig_idx, lower_bound) in &range_extra_rows {
            let new_idx = base_num_constraints + range_row_map.len();
            range_row_map.insert(row_name.clone(), new_idx);
            constraint_types.push(ConstraintType::Ge);
            rhs_vec.push(*lower_bound);
        }

        let num_constraints = base_num_constraints + range_row_map.len();

        // 列名 → インデックスのマップ構築
        let mut col_map = HashMap::new();
        for (col_name, _, _) in &self.columns {
            if !col_map.contains_key(col_name) {
                let idx = col_map.len();
                col_map.insert(col_name.clone(), idx);
            }
        }

        let num_vars = col_map.len();

        // 目的関数ベクトル c の構築
        let mut c = vec![0.0; num_vars];
        if let Some(obj_row_name) = &self.obj_row {
            for (col_name, row_name, value) in &self.columns {
                if row_name == obj_row_name {
                    if let Some(&col_idx) = col_map.get(col_name) {
                        c[col_idx] += *value;
                    }
                }
            }
        }

        // 制約行列 A をトリプレット形式で構築
        let mut triplets = Vec::new();
        for (col_name, row_name, value) in &self.columns {
            // 目的関数行はスキップ
            if Some(row_name) == self.obj_row.as_ref() {
                continue;
            }

            let col_idx = col_map.get(col_name).ok_or_else(|| MpsError::UndefinedReference {
                kind: "column".to_string(),
                name: col_name.clone(),
            })?;
            let row_idx = row_map.get(row_name).ok_or_else(|| MpsError::UndefinedReference {
                kind: "row".to_string(),
                name: row_name.clone(),
            })?;

            triplets.push((*row_idx, *col_idx, *value));

            // RANGE追加行（下限側Ge制約）にも同じ係数を複製
            if let Some(&range_row_idx) = range_row_map.get(row_name) {
                triplets.push((range_row_idx, *col_idx, *value));
            }
        }

        let rows: Vec<usize> = triplets.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();

        let a = CscMatrix::from_triplets(&rows, &cols, &vals, num_constraints, num_vars)
            .map_err(|e| MpsError::ParseError {
                line: 0,
                message: format!("Failed to build matrix: {}", e),
            })?;

        // 変数の上下限ベクトルを構築（デフォルト: [0, +∞)）
        let mut bounds = vec![(0.0, f64::INFINITY); num_vars];
        for (bound_type, col_name, value) in &self.bounds {
            let col_idx = col_map.get(col_name).ok_or_else(|| MpsError::UndefinedReference {
                kind: "column".to_string(),
                name: col_name.clone(),
            })?;

            match bound_type {
                BoundType::LO => {
                    bounds[*col_idx].0 = value.unwrap_or(0.0);
                }
                BoundType::UP => {
                    bounds[*col_idx].1 = value.unwrap_or(f64::INFINITY);
                }
                BoundType::FX => {
                    let val = value.unwrap_or(0.0);
                    bounds[*col_idx] = (val, val);
                }
                BoundType::FR => {
                    bounds[*col_idx] = (f64::NEG_INFINITY, f64::INFINITY);
                }
                BoundType::MI => {
                    bounds[*col_idx].0 = f64::NEG_INFINITY;
                }
                BoundType::BV => {
                    bounds[*col_idx] = (0.0, 1.0);
                }
                BoundType::PL => {
                    // PL: upper bound = +infinity (explicit, same as default).
                    // Lower bound remains unchanged (default 0).
                    bounds[*col_idx].1 = f64::INFINITY;
                }
                BoundType::LI => {
                    // LI: integer lower bound. Same numeric effect as LO; integrality
                    // is recorded separately at parse time.
                    bounds[*col_idx].0 = value.unwrap_or(0.0);
                }
                BoundType::UI => {
                    // UI: integer upper bound. Same numeric effect as UP; integrality
                    // is recorded separately at parse time.
                    bounds[*col_idx].1 = value.unwrap_or(f64::INFINITY);
                }
            }
        }

        // 整数変数インデックスを構築する。
        // 明示境界 (BOUNDS で当該列を参照する行) を持たない整数変数は、古典的 MPS 規約
        // (HiGHS と一致) に従い二値 [0, 1] とみなす。明示境界が 1 つでもあれば、その境界を
        // 尊重して既定上限 +∞ を維持する。
        let explicitly_bounded: HashSet<&String> =
            self.bounds.iter().map(|(_, name, _)| name).collect();

        let mut integer_vars: Vec<usize> = Vec::with_capacity(self.integer_cols.len());
        for col_name in &self.integer_cols {
            if let Some(&col_idx) = col_map.get(col_name) {
                if !explicitly_bounded.contains(col_name) {
                    bounds[col_idx].1 = INTEGER_DEFAULT_UPPER_BINARY;
                }
                integer_vars.push(col_idx);
            }
        }
        // HashSet 由来の非決定的順序を排し、決定論的な昇順に揃える。
        integer_vars.sort_unstable();

        let lp = LpProblem::new_general(
            c,
            a,
            rhs_vec,
            constraint_types,
            bounds,
            self.problem_name.clone(),
        )
        .map_err(|e| MpsError::ParseError {
            line: 0,
            message: e.to_string(),
        })?;

        Ok((lp, integer_vars))
    }
}

/// MPSファイルのセクション種別
///
/// パース中の現在処理セクションを追跡するために使用します。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Section {
    /// セクション未検出の初期状態
    None,
    /// NAMEセクション（問題名）
    Name,
    /// ROWSセクション（行タイプ定義）
    Rows,
    /// COLUMNSセクション（変数係数）
    Columns,
    /// RHSセクション（右辺値）
    Rhs,
    /// RANGESセクション（幅制約）
    Ranges,
    /// BOUNDSセクション（上下限）
    Bounds,
    /// ENDATAセクション（ファイル終端）
    EndData,
}

impl Section {
    /// 行文字列からセクション種別を判定する
    ///
    /// 大文字小文字を区別せず、行の先頭がセクション名と一致するか確認します。
    /// 一致しない場合は`None`を返します。
    fn from_line(line: &str) -> Option<Self> {
        let upper = line.to_uppercase();
        if upper.starts_with("NAME") {
            Some(Section::Name)
        } else if upper.starts_with("ROWS") {
            Some(Section::Rows)
        } else if upper.starts_with("COLUMNS") {
            Some(Section::Columns)
        } else if upper.starts_with("RHS") {
            Some(Section::Rhs)
        } else if upper.starts_with("RANGES") {
            Some(Section::Ranges)
        } else if upper.starts_with("BOUNDS") {
            Some(Section::Bounds)
        } else if upper.starts_with("ENDATA") {
            Some(Section::EndData)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal() {
        let mps = r"NAME          test
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  2.0
RHS
    rhs  c1  10.0
BOUNDS
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_vars, 1);
        assert_eq!(lp.num_constraints, 1);
        assert_eq!(lp.c, vec![1.0]);
        assert_eq!(lp.b, vec![10.0]);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le]);
        assert_eq!(lp.bounds, vec![(0.0, f64::INFINITY)]);
        assert_eq!(lp.name, Some("test".to_string()));
    }

    #[test]
    fn test_parse_equality() {
        let mps = r"NAME test2
ROWS
 N  obj
 E  eq1
COLUMNS
    x1  obj  2.0  eq1  1.0
RHS
    rhs  eq1  5.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 1);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Eq]);
        assert_eq!(lp.b, vec![5.0]);
    }

    #[test]
    fn test_parse_ge_constraint() {
        let mps = r"NAME test3
ROWS
 N  obj
 G  ge1
COLUMNS
    x1  obj  1.0  ge1  1.0
RHS
    rhs  ge1  3.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.constraint_types, vec![ConstraintType::Ge]);
        assert_eq!(lp.b, vec![3.0]);
    }

    #[test]
    fn test_parse_mixed_constraints() {
        let mps = r"NAME mixed
ROWS
 N  obj
 L  c1
 G  c2
 E  c3
COLUMNS
    x1  obj  1.0  c1  1.0
    x1  c2  2.0  c3  3.0
RHS
    rhs  c1  10.0  c2  20.0
    rhs  c3  30.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 3);
        assert_eq!(
            lp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Ge, ConstraintType::Eq]
        );
        assert_eq!(lp.b, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_parse_bounds_lo_up() {
        let mps = r"NAME bounds1
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 LO BND  x1  2.0
 UP BND  x1  8.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(2.0, 8.0)]);
    }

    #[test]
    fn test_parse_bounds_fx() {
        let mps = r"NAME bounds2
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 FX BND  x1  5.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(5.0, 5.0)]);
    }

    #[test]
    fn test_parse_bounds_fr() {
        let mps = r"NAME bounds3
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 FR BND  x1
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(f64::NEG_INFINITY, f64::INFINITY)]);
    }

    #[test]
    fn test_parse_bounds_mi() {
        let mps = r"NAME bounds4
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 MI BND  x1
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(f64::NEG_INFINITY, f64::INFINITY)]);
    }

    #[test]
    fn test_parse_ranges() {
        let mps = r"NAME ranges
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
RANGES
    rng  c1  5.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        // RANGE行は区間制約に展開される（Le上限 + Ge下限の2制約）
        // L制約 + range=5.0: upper=10.0（Le）, lower=10.0-5.0=5.0（Ge）
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.b[0], 10.0); // Le制約のRHS（上限）
        assert_eq!(lp.b[1], 5.0);  // Ge制約のRHS（下限）
    }

    #[test]
    fn test_parse_mps_accumulates_duplicate_objective_entries() {
        let mps = r"NAME dup_obj
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.5  c1  1.0
    x1  obj  2.5
RHS
    rhs  c1  10.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.c, vec![4.0]);
    }

    #[test]
    fn test_parse_multiple_rhs_entries() {
        let mps = r"NAME multi_rhs
ROWS
 N  obj
 L  c1
 L  c2
COLUMNS
    x1  obj  1.0  c1  1.0
    x2  obj  2.0  c2  1.0
RHS
    rhs  c1  10.0
    rhs  c2  20.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.b, vec![10.0, 20.0]);
    }

    #[test]
    fn test_parse_two_entries_per_line() {
        let mps = r"NAME two_per_line
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  2.0
RHS
    rhs  c1  10.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_vars, 1);
        // x1のc1における係数は2.0
        let (rows, vals) = lp.a.get_column(0).unwrap();
        assert_eq!(rows, &[0]);
        assert_eq!(vals, &[2.0]);
    }

    #[test]
    fn test_parse_error_missing_endata() {
        let mps = r"NAME test
ROWS
 N  obj
COLUMNS
    x1  obj  1.0
";
        let result = parse_mps(mps);
        assert!(result.is_err());
        match result {
            Err(MpsError::MissingSection(s)) => assert_eq!(s, "ENDATA"),
            _ => panic!("Expected MissingSection error"),
        }
    }

    #[test]
    fn test_parse_error_invalid_row_type() {
        let mps = r"NAME test
ROWS
 N  obj
 X  bad
COLUMNS
    x1  obj  1.0
ENDATA
";
        let result = parse_mps(mps);
        assert!(result.is_err());
        match result {
            Err(MpsError::InvalidRowType('X')) => {},
            _ => panic!("Expected InvalidRowType error"),
        }
    }

    /// Le制約 + 正のRANGE → [b-|r|, b] に変換されることを確認
    #[test]
    fn test_range_le_basic() {
        let mps = r"NAME range_le
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
RANGES
    rhs  c1  2.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        // Le c1 with b=10, r=2 → Le(upper=10) + Ge(lower=8)
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
        assert_eq!(lp.b[0], 10.0);
        assert_eq!(lp.b[1], 8.0);
    }

    /// Ge制約 + 正のRANGE → [b, b+|r|] に変換されることを確認
    #[test]
    fn test_range_ge_basic() {
        let mps = r"NAME range_ge
ROWS
 N  obj
 G  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  5.0
RANGES
    rhs  c1  3.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        // Ge c1 with b=5, r=3 → Le(upper=8) + Ge(lower=5)
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
        assert_eq!(lp.b[0], 8.0);
        assert_eq!(lp.b[1], 5.0);
    }

    /// Eq制約 + 正のRANGE → [b, b+|r|] に変換されることを確認
    #[test]
    fn test_range_eq_positive() {
        let mps = r"NAME range_eq_pos
ROWS
 N  obj
 E  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  7.0
RANGES
    rhs  c1  2.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        // Eq c1 with b=7, r=2 (r>=0) → Le(upper=9) + Ge(lower=7)
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
        assert_eq!(lp.b[0], 9.0);
        assert_eq!(lp.b[1], 7.0);
    }

    /// Eq制約 + 負のRANGE → [b-|r|, b] に変換されることを確認
    #[test]
    fn test_range_eq_negative() {
        let mps = r"NAME range_eq_neg
ROWS
 N  obj
 E  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  7.0
RANGES
    rhs  c1  -2.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        // Eq c1 with b=7, r=-2 (r<0) → Le(upper=7) + Ge(lower=5)
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
        assert_eq!(lp.b[0], 7.0);
        assert_eq!(lp.b[1], 5.0);
    }

    /// RANGE付き小規模LP を solve() で解き、最適値を検証
    /// minimize x1 + x2  s.t. 3 <= x1 + x2 <= 7, x1,x2 >= 0
    /// 最適解: x1=3, x2=0 (or x1=0, x2=3), 最適値=3
    #[test]
    fn test_range_solve_simple() {
        use crate::problem::SolveStatus;
        use crate::simplex::solve;

        let mps = r"NAME range_solve
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
    x2  obj  1.0  c1  1.0
RHS
    rhs  c1  7.0
RANGES
    rhs  c1  4.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        // Le c1 with b=7, r=4 → Le(upper=7) + Ge(lower=3)
        assert_eq!(lp.num_constraints, 2);
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal, "should reach Optimal");
        assert!(
            (result.objective - 3.0).abs() < 1e-6,
            "expected obj=3.0, got {}",
            result.objective
        );
    }

    // ──────────────────────────────────────────────
    // is_fixed_width_format のユニットテスト
    // ──────────────────────────────────────────────

    /// 典型的な固定幅行: 列15（index 14）がスペース → true
    #[test]
    fn test_is_fixed_width_typical_fixed() {
        // 列5-12: col_name, index 14 がスペース（固定幅の区切り）
        let fixed_line = "    x1          obj   1.0"; // index 14 = ' '
        assert!(
            is_fixed_width_format(fixed_line),
            "列14がスペースの行は固定幅と判定すべき"
        );
    }

    /// 典型的な自由形式行: 列15（index 14）がスペース以外 → false
    #[test]
    fn test_is_fixed_width_free_format() {
        // フリーフォーマット: "    x1  obj  1.0" のように詰まっている
        let line = "    x1  obj  1.0";
        assert!(
            !is_fixed_width_format(line),
            "フリーフォーマット行は固定幅と判定してはならない"
        );
    }

    /// 境界ケース: 14文字以下の行 → false（列15が存在しない）
    #[test]
    fn test_is_fixed_width_short_line() {
        assert!(!is_fixed_width_format(""), "空行はfalse");
        assert!(!is_fixed_width_format("    x1  c1 1"), "14文字以下はfalse");
        assert!(!is_fixed_width_format("12345678901234"), "ちょうど14文字もfalse");
    }

    /// タブ文字を含む行: タブはis_whitespace()でtrueになる → 固定幅判定に影響しない
    #[test]
    fn test_is_fixed_width_with_tab() {
        // 14文字の位置（index 14）にタブがある場合 → true
        let line_with_tab = "    x1        \tobj  1.0"; // index 14 = '\t'
        assert!(
            is_fixed_width_format(line_with_tab),
            "列14のタブは空白として扱い固定幅と判定すべき"
        );
    }

    // ──────────────────────────────────────────────
    // integer_marker_kind のユニットテスト
    // ──────────────────────────────────────────────

    #[test]
    fn test_integer_marker_kind_intorg_intend() {
        assert_eq!(
            integer_marker_kind("    M1 'MARKER' 'INTORG'"),
            Some(IntegerMarker::Start)
        );
        assert_eq!(
            integer_marker_kind("    M2 'MARKER' 'INTEND'"),
            Some(IntegerMarker::End)
        );
        // 大文字小文字・クォート無しでも検出する
        assert_eq!(
            integer_marker_kind("    m 'marker' intorg"),
            Some(IntegerMarker::Start)
        );
    }

    #[test]
    fn test_integer_marker_kind_non_marker() {
        // 通常の COLUMNS 行はマーカーでない
        assert_eq!(integer_marker_kind("    x1  obj  1.0  c1  2.0"), None);
        // 'MARKER' トークンを伴わない INTORG という名の列は誤検出しない
        assert_eq!(integer_marker_kind("    INTORG  obj  1.0"), None);
    }

    // ──────────────────────────────────────────────
    // MILP パース: 整数変数識別 + 境界規約
    // ──────────────────────────────────────────────

    /// マーカー整数で BOUNDS 指定が無い → 二値 [0, 1]、整数登録される。
    /// no-op proof: 二値既定 (INTEGER_DEFAULT_UPPER_BINARY) を外すと上限が +∞ となり
    /// このアサーションが落ちる。
    #[test]
    fn test_milp_marker_no_bounds_is_binary() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 1.0)]);
    }

    /// マーカー整数 + 明示 UP → [0, UP]。明示境界があるので二値既定は無効。
    #[test]
    fn test_milp_marker_with_up_bound() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
BOUNDS
 UP BND  x1  5.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 5.0)]);
    }

    /// マーカー整数 + 明示 LO のみ → [LO, +∞]。明示境界があるので二値既定は無効。
    #[test]
    fn test_milp_marker_with_lo_only() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
BOUNDS
 LO BND  x1  2.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(2.0, f64::INFINITY)]);
    }

    /// UI 境界はマーカー無しでも変数を整数化し、上限を設定する。
    #[test]
    fn test_milp_ui_bound_marks_integer() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
BOUNDS
 UI BND  x1  7.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 7.0)]);
    }

    /// LI 境界はマーカー無しでも変数を整数化し、下限を設定する。
    #[test]
    fn test_milp_li_bound_marks_integer() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.5
BOUNDS
 LI BND  x1  2.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(2.0, f64::INFINITY)]);
    }

    /// BV 境界は変数を整数化し [0, 1] を設定する。
    #[test]
    fn test_milp_bv_bound_marks_integer() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
BOUNDS
 BV BND  x1
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 1.0)]);
    }

    /// 混合整数: マーカー領域内の変数のみ整数、領域外は連続のまま。
    /// 二値既定はマーカー整数 (x1, 境界なし) のみに適用され、連続変数 x2 は [0,+∞]。
    #[test]
    fn test_milp_mixed_integer_continuous() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
    x2  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        // x1 (idx 0) のみ整数、x2 (idx 1) は連続。
        assert_eq!(milp.integer_vars, vec![0]);
        // x1 は境界なしマーカー整数 → 二値 [0,1]、x2 は連続既定 [0,+∞]。
        assert_eq!(milp.lp.bounds[0], (0.0, 1.0));
        assert_eq!(milp.lp.bounds[1], (0.0, f64::INFINITY));
    }

    /// `parse_mps` (LP path) は整数情報を破棄して LP relaxation を返す。
    /// 境界 (二値既定含む) は LP relaxation の正しい一部として保持される。
    #[test]
    fn test_parse_mps_returns_relaxation_dropping_integrality() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        // LP relaxation は連続。二値既定の [0,1] 境界は保持される。
        assert_eq!(lp.num_vars, 1);
        assert_eq!(lp.bounds, vec![(0.0, 1.0)]);
    }

    /// 純粋 LP ファイルはマーカーも整数境界も持たない → 整数集合は空。
    #[test]
    fn test_milp_pure_lp_has_no_integers() {
        let mps = r"NAME lp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert!(milp.integer_vars.is_empty());
        assert_eq!(milp.lp.bounds, vec![(0.0, f64::INFINITY)]);
    }

    /// 固定幅フォーマットのマーカー行も認識する。
    #[test]
    fn test_milp_fixed_format_marker() {
        // 固定幅: 列15 (index 14) が空白。マーカー行は split_whitespace で検出される。
        let mps = "NAME          milp\n\
ROWS\n\
 N  obj\n\
 L  c1\n\
COLUMNS\n    \
MARKER1                 'MARKER'                 'INTORG'\n    \
x1        c1        1.0   obj       -1.0\n    \
MARKER2                 'MARKER'                 'INTEND'\n\
RHS\n    \
rhs       c1        10.5\n\
ENDATA\n";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0], "固定幅マーカー領域の x1 が整数登録される");
    }

    /// 複数整数変数のインデックスが決定論的に昇順で返る。
    #[test]
    fn test_milp_integer_vars_sorted() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    a   obj  1.0  c1  1.0
    M1 'MARKER' 'INTORG'
    b   obj  1.0  c1  1.0
    c   obj  1.0  c1  1.0
    M2 'MARKER' 'INTEND'
    d   obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        // 列順: a=0, b=1, c=2, d=3。整数は b, c。
        assert_eq!(milp.integer_vars, vec![1, 2]);
    }

    /// MILP を実際に解いて最適値を検証する (HiGHS で独立に確認した期待値)。
    /// テーブル駆動で複数の境界規約パターンを網羅する。
    ///
    /// **load-bearing (no-op proof)**: 各ケースは整数強制によって最適値が LP relaxation
    /// と分岐するよう設計してある。マーカー検出を無効化すると当該変数が連続のままとなり、
    /// 期待 objective から外れる:
    /// - marker_no_bounds_binary: 二値既定が消え [0,+∞] 連続 → -1 ではなく -10.5
    /// - marker_up5_fractional: x1<=3.5 で連続なら x1=3.5 → -3 ではなく -3.5
    /// - marker_lo2: x1<=10.5 で連続なら x1=10.5 → -10 ではなく -10.5
    #[test]
    fn test_milp_solve_bound_conventions() {
        use crate::options::{MipConfig, SolverOptions};
        use crate::problem::SolveStatus;

        // (説明, BOUNDS セクション本体, 制約 c1 の RHS, 整数最適 objective)。
        // 全て min -x1 s.t. x1 <= rhs。HiGHS で独立検証済み。
        let cases: &[(&str, &str, f64, f64)] = &[
            // 境界なしマーカー整数 → 二値 [0,1] → x1=1 → obj=-1
            // (連続化すると x1<=10.5 で x1=10.5 → -10.5、よって load-bearing)
            ("marker_no_bounds_binary", "", 10.5, -1.0),
            // UP 5 → [0,5] 整数。x1<=3.5 → 整数 x1=3 → obj=-3 (連続なら x1=3.5 → -3.5)
            ("marker_up5_fractional", "BOUNDS\n UP BND  x1  5.0\n", 3.5, -3.0),
            // LO 2 のみ → [2,+inf] 整数。x1<=10.5 → 整数 x1=10 → -10 (連続なら x1=10.5 → -10.5)
            ("marker_lo2", "BOUNDS\n LO BND  x1  2.0\n", 10.5, -10.0),
        ];

        for (label, bounds_section, rhs, expected_obj) in cases {
            let mps = format!(
                "NAME milp\n\
ROWS\n N  obj\n L  c1\n\
COLUMNS\n    M1 'MARKER' 'INTORG'\n    x1  obj  -1.0  c1  1.0\n    M2 'MARKER' 'INTEND'\n\
RHS\n    rhs  c1  {rhs}\n\
{bounds_section}ENDATA\n"
            );
            let milp = parse_milp(&mps).unwrap();
            let opts = SolverOptions::default();
            let cfg = MipConfig::default();
            let res = crate::mip::solve_milp(&milp, &opts, &cfg);
            assert_eq!(res.status, SolveStatus::Optimal, "[{label}] should be Optimal");
            assert!(
                (res.objective - expected_obj).abs() < 1e-6,
                "[{label}] expected obj={expected_obj}, got {}",
                res.objective
            );
        }
    }

    /// UI 整数境界付き MILP を解く (load-bearing): x1 <= 3.5 の連続最適は分数 x1=3.5。
    /// UI による整数強制で x1=3, obj=-3。UI の整数マークを無効化すると連続 x1=3.5 →
    /// -3.5 となり期待値から外れる。HiGHS で -3 を独立確認済み。
    #[test]
    fn test_milp_solve_ui_bound() {
        use crate::options::{MipConfig, SolverOptions};
        use crate::problem::SolveStatus;

        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  3.5
BOUNDS
 UI BND  x1  7.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        let res = crate::mip::solve_milp(&milp, &SolverOptions::default(), &MipConfig::default());
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!((res.objective - (-3.0)).abs() < 1e-6, "expected -3, got {}", res.objective);
        assert!((res.solution[0] - 3.0).abs() < 1e-6, "x1 should be 3");
    }

    /// 閉じられない INTORG (INTEND 欠落) は明示エラーとする (P2②)。
    /// 放置すると INTORG 以降の全列が無警告で整数化される。
    #[test]
    fn test_milp_unclosed_intorg_errors() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
ENDATA
";
        let err = parse_milp(mps).unwrap_err();
        assert!(
            matches!(err, MpsError::UnclosedIntegerMarker),
            "unclosed INTORG must error, got {err:?}"
        );
        // LP path (parse_mps) も同じパーサなので同様にエラー。
        assert!(matches!(parse_mps(mps).unwrap_err(), MpsError::UnclosedIntegerMarker));
    }

    /// 回帰 (MIPLIB enlight_hard): 列名が短く行名が列22以降にずれる幅広パディング行は
    /// char-14 ヒューリスティックで固定幅と誤判定される。固定位置 [14:22] が空のため
    /// 旧実装は COLUMNS entry を無警告で取りこぼし、常にフリー解析の BOUNDS と列名が
    /// 食い違い `UndefinedReference` で parse 失敗していた。フリー解析フォールバックで解消。
    ///
    /// load-bearing: フォールバックを外すと x#1#1 が COLUMNS に登録されず、BOUNDS の
    /// 参照で `UndefinedReference` になり parse_milp が Err を返す (この test が落ちる)。
    #[test]
    fn test_columns_free_format_misclassified_as_fixed() {
        // 列名 "x#1#1" (5 文字) を 4 スペース後に置き、行名を列22から開始させる
        // (固定窓 [14:22] が空白になる = enlight_hard と同一レイアウト)。char[14] は
        // padding 内の空白なので is_fixed_width_format は true を返す。
        let pad = " ".repeat(22 - 4 - "x#1#1".len()); // 行名を列22開始に
        let mps = format!(
            "NAME wide\n\
ROWS\n N  obj\n L  c\n\
COLUMNS\n\
    x#1#1{pad}obj   -1.0\n\
    x#1#1{pad}c     1.0\n\
RHS\n    rhs{rpad}c     3.5\n\
BOUNDS\n UI BND  x#1#1  7\n\
ENDATA\n",
            rpad = " ".repeat(22 - 4 - "rhs".len()),
        );
        // パース成功 (UndefinedReference にならない)。
        let milp = parse_milp(&mps).expect("wide-padded free-format COLUMNS must parse");
        assert_eq!(milp.num_vars(), 1, "x#1#1 が 1 変数として登録される");
        assert_eq!(milp.integer_vars, vec![0], "UI 境界で整数登録");
        assert_eq!(milp.lp.bounds, vec![(0.0, 7.0)]);
        // 制約 c に x#1#1 の係数 1.0 が入っている (取りこぼしていない)。
        let (rows, vals) = milp.lp.a.get_column(0).unwrap();
        assert_eq!(rows, &[0]);
        assert_eq!(vals, &[1.0]);
    }

    /// 上記レイアウトを実際に解いて load-bearing 性を担保: min -x, x<=3.5, x∈ℤ[0,7]
    /// → 整数最適 x=3, obj=-3。列を取りこぼすと parse 失敗 or 解が変わる。
    #[test]
    fn test_columns_wide_padding_solves() {
        use crate::options::{MipConfig, SolverOptions};
        use crate::problem::SolveStatus;
        let pad = " ".repeat(22 - 4 - "x#1#1".len());
        let mps = format!(
            "NAME wide\n\
ROWS\n N  obj\n L  c\n\
COLUMNS\n\
    x#1#1{pad}obj   -1.0\n\
    x#1#1{pad}c     1.0\n\
RHS\n    rhs{rpad}c     3.5\n\
BOUNDS\n UI BND  x#1#1  7\n\
ENDATA\n",
            rpad = " ".repeat(22 - 4 - "rhs".len()),
        );
        let milp = parse_milp(&mps).unwrap();
        let res = crate::mip::solve_milp(&milp, &SolverOptions::default(), &MipConfig::default());
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!((res.objective - (-3.0)).abs() < 1e-6, "expected -3, got {}", res.objective);
    }

    /// 正しく INTEND で閉じれば後続列は連続のまま (整合性の確認)。
    #[test]
    fn test_milp_closed_intorg_following_cols_continuous() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  1.0  c1  1.0
    M2 'MARKER' 'INTEND'
    x2  obj  1.0  c1  1.0
    x3  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0], "x1 のみ整数、x2/x3 は連続");
    }

    // ─────────────────────────────────────────
    // Streaming API: round-trip + sentinel tests
    // ─────────────────────────────────────────

    /// Minimal fixture shared by streaming tests.
    const STREAM_MPS: &str = "NAME          stream\n\
ROWS\n N  obj\n L  c1\n\
COLUMNS\n    x1  obj  3.0  c1  1.0\n    x2  obj  5.0  c1  2.0\n\
RHS\n    rhs  c1  10.0\n\
ENDATA\n";

    /// `parse_mps_reader` produces identical result to `parse_mps` (round-trip).
    #[test]
    fn test_mps_reader_round_trip() {
        let expected = parse_mps(STREAM_MPS).unwrap();
        let got = parse_mps_reader(std::io::Cursor::new(STREAM_MPS.as_bytes())).unwrap();
        assert_eq!(got.num_vars, expected.num_vars);
        assert_eq!(got.num_constraints, expected.num_constraints);
        assert_eq!(got.c, expected.c);
        assert_eq!(got.b, expected.b);
        assert_eq!(got.bounds, expected.bounds);
    }

    /// `parse_milp_reader` produces identical result to `parse_milp` (round-trip).
    #[test]
    fn test_milp_reader_round_trip() {
        let mps = "NAME          m\nROWS\n N  obj\n L  c1\n\
COLUMNS\n    M1 'MARKER' 'INTORG'\n    x1  obj  -1.0  c1  1.0\n    M2 'MARKER' 'INTEND'\n\
RHS\n    rhs  c1  10.5\nENDATA\n";
        let expected = parse_milp(mps).unwrap();
        let got = parse_milp_reader(std::io::Cursor::new(mps.as_bytes())).unwrap();
        assert_eq!(got.integer_vars, expected.integer_vars);
        assert_eq!(got.lp.bounds, expected.lp.bounds);
    }

    /// Tracked fixture: parse netlib/afiro.mps via reader API and compare to string API.
    #[test]
    fn test_mps_reader_fixture_afiro() {
        let path = std::path::Path::new("tests/netlib/afiro.mps");
        if !path.exists() {
            return; // fixture not tracked in this worktree
        }
        let content = std::fs::read_to_string(path).unwrap();
        let expected = parse_mps(&content).unwrap();
        let file = std::fs::File::open(path).unwrap();
        let got = parse_mps_reader(std::io::BufReader::new(file)).unwrap();
        assert_eq!(got.num_vars, expected.num_vars, "num_vars mismatch");
        assert_eq!(got.num_constraints, expected.num_constraints, "num_constraints mismatch");
        assert_eq!(got.c, expected.c, "objective mismatch");
        assert_eq!(got.b, expected.b, "rhs mismatch");
    }

    // ─── Sentinel ────────────────────────────────────────────────────────────
    //
    // `parse_mps_reader` must call `BufRead::read_line` (via the `.lines()` iterator)
    // and NOT merely delegate to `read_to_string`.  `read_to_string` uses `fill_buf` /
    // `consume` directly and never calls the overridden `read_line` below.
    //
    // no-op proof:
    //   If `parse_mps_reader` is reverted to:
    //     let s = std::io::read_to_string(reader)?;
    //     parse_mps(&s)
    //   then `read_line` is not called → `line_call_count == 0` → the assert fails.

    use std::io::{self, Read};

    struct LineCountingReader<R: BufRead> {
        inner: R,
        pub line_call_count: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl<R: BufRead> Read for LineCountingReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.inner.read(buf)
        }
    }

    impl<R: BufRead> BufRead for LineCountingReader<R> {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.inner.fill_buf()
        }
        fn consume(&mut self, amt: usize) {
            self.inner.consume(amt)
        }
        fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
            let n = self.inner.read_line(buf)?;
            if n > 0 {
                self.line_call_count.set(self.line_call_count.get() + 1);
            }
            Ok(n)
        }
    }

    /// Sentinel: `parse_mps_reader` calls `read_line` once per content line (streaming).
    ///
    /// Fails when reverted to `read_to_string` because `read_to_string` never calls
    /// the `read_line` override — the counter stays 0.
    #[test]
    fn test_mps_reader_streaming_sentinel() {
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let reader = LineCountingReader {
            inner: std::io::Cursor::new(STREAM_MPS.as_bytes()),
            line_call_count: counter.clone(),
        };
        let lp = parse_mps_reader(reader).expect("parse must succeed");
        assert_eq!(lp.num_vars, 2);
        let expected_lines = STREAM_MPS.lines().count();
        assert!(
            counter.get() >= expected_lines,
            "streaming must call read_line at least {expected_lines} times, got {}",
            counter.get()
        );
    }
}
