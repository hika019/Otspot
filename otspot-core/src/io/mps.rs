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

// MpsError lives in crate::error; re-export for path stability.
pub use crate::error::MpsError;

/// MPS の INTORG/INTEND マーカーで囲まれ、かつ BOUNDS 指定が一切ない整数変数の
/// デフォルト上限。古典的な OSL/CPLEX 規約では「明示境界のない整数変数は二値」と
/// 解釈する (HiGHS の MPS リーダーと一致)。明示境界が 1 つでもあればこの既定は無効。
const INTEGER_DEFAULT_UPPER_BINARY: f64 = 1.0;

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
/// ```ignore
/// use otspot_core::io::mps::parse_mps;
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
/// ```ignore
/// use otspot_core::io::mps::parse_milp;
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

