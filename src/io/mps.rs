//! 線形計画問題のMPSファイル形式パーサー
//!
//! # MPSフォーマットとは
//!
//! MPS（Mathematical Programming System）は、線形計画問題（LP）や
//! 混合整数計画問題（MIP）を記述するための業界標準フォーマットです。
//! IBM社が1960年代に開発し、現在もほぼすべてのLPソルバーがサポートしています。
//!
//! # サポートするMPSセクション
//!
//! | セクション | 説明 |
//! |-----------|------|
//! | `NAME`    | 問題名（オプション） |
//! | `ROWS`    | 目的関数行・制約行の型定義 |
//! | `COLUMNS` | 変数係数の定義 |
//! | `RHS`     | 右辺値の定義 |
//! | `RANGES`  | 制約の幅指定（オプション） |
//! | `BOUNDS`  | 変数の上下限（オプション） |
//! | `ENDATA`  | ファイル終端マーカー |
//!
//! # フォーマットの種類
//!
//! - **固定幅フォーマット**: 各フィールドが固定列位置に配置される旧来の形式
//! - **フリーフォーマット**: 空白区切りで任意の列位置に配置できる現代的な形式
//!
//! 本モジュールは両方のフォーマットを自動判別して解析します。

use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;
use std::collections::HashMap;
use std::path::Path;

/// MPSファイルのパース中に発生するエラー
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
/// # 引数
///
/// * `path` - 読み込むMPSファイルのパス
///
/// # 戻り値
///
/// 成功時は`LpProblem`、失敗時は`MpsError`を返す
///
/// # Errors
///
/// - ファイルが存在しない、または読み取り権限がない場合は`MpsError::IoError`
/// - ファイル内容のパースに失敗した場合は各種`MpsError`バリアント
pub fn parse_mps_file(path: &Path) -> Result<LpProblem, MpsError> {
    let content = std::fs::read_to_string(path)?;
    parse_mps(&content)
}

/// MPS形式の文字列を`LpProblem`にパースする
///
/// 固定幅フォーマットとフリーフォーマットの両方を自動判別して処理します。
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
/// use solver::io::mps::parse_mps;
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
    let lines: Vec<&str> = input.lines().collect();
    let mut parser = MpsParser::new();
    parser.parse(&lines)
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
        }
    }

    /// MPSファイルの全行をセクションごとに解析し、`LpProblem`を返す
    ///
    /// セクションヘッダーの検出（先頭に空白がない行）と、
    /// データ行（先頭に空白がある行）の処理を区別します。
    fn parse(&mut self, lines: &[&str]) -> Result<LpProblem, MpsError> {
        let mut current_section = Section::None;
        let mut seen_sections = std::collections::HashSet::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            // 空行とコメント行（'*'で始まる行）をスキップ
            if trimmed.is_empty() || trimmed.starts_with('*') {
                continue;
            }

            // セクションヘッダーの検出（先頭が空白でない行がヘッダー候補）
            // データ行は先頭に空白があるため、この条件で区別できる
            if !line.starts_with(' ') && !line.starts_with('\t') {
                if let Some(section) = Section::from_line(trimmed) {
                    // NAMEとENDATAを除くセクションの重複チェック
                    if section != Section::Name && section != Section::EndData
                        && seen_sections.contains(&section) {
                        return Err(MpsError::DuplicateSection(format!("{:?}", section)));
                    }
                    // 出現済みセクションとして記録
                    seen_sections.insert(section);
                    current_section = section;

                    // NAMEセクション: 同一行から問題名を取得
                    // 書式: "NAME          problem_name"
                    if section == Section::Name && trimmed.len() > 4 {
                        let name_part = trimmed[4..].trim();
                        if !name_part.is_empty() {
                            self.problem_name = Some(name_part.to_string());
                        }
                    }
                    continue;
                }
            }

            // 現在のセクションに応じてデータ行を処理
            match current_section {
                Section::None => {
                    return Err(MpsError::ParseError {
                        line: line_num,
                        message: "Line appears before any section header".to_string(),
                    });
                }
                Section::Name => {
                    // NAMEはセクションヘッダー検出時に処理済み
                }
                Section::Rows => self.parse_rows_line(line, line_num)?,
                Section::Columns => self.parse_columns_line(line, line_num)?,
                Section::Rhs => self.parse_rhs_line(line, line_num)?,
                Section::Ranges => self.parse_ranges_line(line, line_num)?,
                Section::Bounds => self.parse_bounds_line(line, line_num)?,
                Section::EndData => break,
            }
        }

        // ENDATAセクションの存在チェック
        if !seen_sections.contains(&Section::EndData) {
            return Err(MpsError::MissingSection("ENDATA".to_string()));
        }

        // 必須セクション（ROWS・COLUMNS）の存在チェック
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

        if line.len() < 25 {
            return Ok(()); // 短すぎる行はスキップ
        }

        let col_name = line.get(4..12).unwrap_or("").trim().to_string();
        if col_name.is_empty() {
            return Ok(());
        }

        let row_name1 = line.get(14..22).unwrap_or("").trim().to_string();
        let value1_str = line.get(24..36).unwrap_or("").trim();

        if !row_name1.is_empty() && !value1_str.is_empty() {
            let value1 = value1_str.parse::<f64>().map_err(|_| MpsError::ParseError {
                line: line_num,
                message: format!("Invalid numeric value: {}", value1_str),
            })?;
            self.columns.push((col_name.clone(), row_name1, value1));
        }

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
            _ => return Err(MpsError::InvalidBoundType(bound_type_str.to_string())),
        };

        self.bounds.push((bound_type, col_name, value));
        Ok(())
    }

    /// パース済みデータから`LpProblem`を構築する
    ///
    /// 処理の流れ:
    /// 1. 行名→インデックスマップを構築（目的関数行を除く）
    /// 2. RANGESを適用してRHS値を調整
    /// 3. 列名→インデックスマップを構築
    /// 4. 目的関数ベクトル`c`を構築
    /// 5. 制約行列`A`をCSC形式で構築
    /// 6. 変数の上下限ベクトルを構築
    fn build_lp_problem(&self) -> Result<LpProblem, MpsError> {
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
                        c[col_idx] = *value;
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
            }
        }

        LpProblem::new_general(
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
        })
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
        // §6-4: RANGE行は区間制約に展開される（Le上限 + Ge下限の2制約）
        // L制約 + range=5.0: upper=10.0（Le）, lower=10.0-5.0=5.0（Ge）
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.b[0], 10.0); // Le制約のRHS（上限）
        assert_eq!(lp.b[1], 5.0);  // Ge制約のRHS（下限）
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
        // 列5-12: col_name, 列15: space（固定幅の区切り）
        // "    x1            obj  1.0" のように列14がスペース
        let line = "    x1        obj   1.0";
        //          0123456789012345...
        //          列14（0-indexed）= 'o' ではなく、スペースが来るケース
        // 実際に列14がスペースになる行を用意する
        let fixed_line = "    x1          obj   1.0"; // index 14 = ' '
        assert!(
            is_fixed_width_format(fixed_line),
            "列14がスペースの行は固定幅と判定すべき"
        );
        let _ = line; // unused warning回避
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
}
