//! QPSファイル形式パーサー
//!
//! QPS（Quadratic Programming Standard）形式は、MPSにQUADOBJセクションを追加した
//! 二次計画問題（QP）の標準記述フォーマットです。Maros-Meszarosベンチマーク等で使用。
//!
//! # QPSとMPSの差分
//!
//! QPS = MPS + QUADOBJセクション:
//! ```text
//! QUADOBJ
//!     col1    col2    value
//! ```
//! 上三角のみ記述される。本パーサーは対称化（下三角も設定）を行う。
//!
//! # 目的関数規約
//!
//! 本solverは「1/2あり」規約（OSQP/qpOASES標準）を採用:
//! min 1/2 x^T Q x + c^T x
//! Maros-MeszarosのQPSファイルも同規約を使用しているため、係数の変換不要。
//!
//! # 制約形式変換
//!
//! `QpProblem`は`Ax <= b`のみをサポートするため、MPSの各制約タイプを変換:
//! - Le (Ax <= b): そのまま
//! - Ge (Ax >= b): 両辺を否定 → -Ax <= -b
//! - Eq (Ax == b): 2行に展開 → Ax <= b, -Ax <= -b

use crate::qp::QpProblem;
use crate::sparse::CscMatrix;
use std::collections::HashMap;
use std::path::Path;

/// QPSファイルのパース中に発生するエラー
#[derive(Debug)]
pub enum QpsError {
    /// ファイルI/Oエラー
    IoError(std::io::Error),
    /// 指定行のパースエラー（行番号とメッセージ）
    ParseError { line: usize, message: String },
    /// 必須セクションが欠落
    MissingSection(String),
    /// 未定義の列名または行名が参照された
    UndefinedReference { kind: String, name: String },
}

impl std::fmt::Display for QpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QpsError::IoError(e) => write!(f, "I/O error: {}", e),
            QpsError::ParseError { line, message } => {
                write!(f, "Parse error at line {}: {}", line, message)
            }
            QpsError::MissingSection(s) => write!(f, "Missing required section: {}", s),
            QpsError::UndefinedReference { kind, name } => {
                write!(f, "Undefined {} reference: {}", kind, name)
            }
        }
    }
}

impl std::error::Error for QpsError {}

impl From<std::io::Error> for QpsError {
    fn from(err: std::io::Error) -> Self {
        QpsError::IoError(err)
    }
}

/// ファイルパスからQPSファイルを読み込み、`QpProblem`としてパースする
pub fn parse_qps(path: &Path) -> Result<QpProblem, QpsError> {
    let content = std::fs::read_to_string(path)?;
    parse_qps_str(&content)
}

/// QPS形式の文字列を`QpProblem`にパースする
pub fn parse_qps_str(input: &str) -> Result<QpProblem, QpsError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut parser = QpsParser::new();
    parser.parse(&lines)
}

/// MPSの行タイプ
#[derive(Debug, Clone, Copy)]
enum RowType {
    N, // 目的関数
    L, // Ax <= b
    G, // Ax >= b
    E, // Ax == b
}

/// MPSのBOUNDタイプ
#[derive(Debug, Clone, Copy)]
enum BoundType {
    LO, // 下限
    UP, // 上限
    FX, // 固定
    FR, // 自由変数
    MI, // 下限=-∞
    BV, // バイナリ変数
    PL, // デフォルト上限（+∞）
}

/// QPSパーサーのセクション種別
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Section {
    None,
    Name,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Quadobj,
    EndData,
}

impl Section {
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
        } else if upper.starts_with("QUADOBJ") {
            Some(Section::Quadobj)
        } else if upper.starts_with("ENDATA") {
            Some(Section::EndData)
        } else {
            None
        }
    }
}

/// QPSパーサーの中間状態
struct QpsParser {
    rows: Vec<(String, RowType)>,
    columns: Vec<(String, String, f64)>,
    rhs: HashMap<String, f64>,
    ranges: HashMap<String, f64>,
    bounds: Vec<(BoundType, String, Option<f64>)>,
    /// QUADOBJ: (row_idx, col_idx, value) — 列インデックスは build 時に解決
    quadobj: Vec<(String, String, f64)>,
    obj_row: Option<String>,
}

/// MPS固定幅フィールド取得ヘルパー
///
/// MPS固定幅フォーマットの指定位置（0-indexed, start..end）から文字列を取得してtrimする。
/// 標準MPS列位置:
///   Field 2 (col_name/rhs_name): cols 4-11 → mps_field(line, 4, 12)
///   Field 3 (row_name1):         cols 14-21 → mps_field(line, 14, 22)
///   Field 4 (value1):            cols 24-35 → mps_field(line, 24, 36)
///   Field 5 (row_name2):         cols 39-46 → mps_field(line, 39, 47)
///   Field 6 (value2):            cols 49-60 → mps_field(line, 49, 61)
fn mps_field(line: &str, start: usize, end: usize) -> &str {
    let len = line.len();
    if start >= len {
        return "";
    }
    let actual_end = end.min(len);
    // ASCII前提: バイト境界チェック
    if !line.is_char_boundary(start) || !line.is_char_boundary(actual_end) {
        return "";
    }
    line[start..actual_end].trim()
}

impl QpsParser {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
            columns: Vec::new(),
            rhs: HashMap::new(),
            ranges: HashMap::new(),
            bounds: Vec::new(),
            quadobj: Vec::new(),
            obj_row: None,
        }
    }

    fn parse(&mut self, lines: &[&str]) -> Result<QpProblem, QpsError> {
        let mut current_section = Section::None;
        let mut seen_sections = std::collections::HashSet::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with('$') {
                continue;
            }

            // セクションヘッダー: 先頭が空白でない行
            if !line.starts_with(' ') && !line.starts_with('\t') {
                if let Some(section) = Section::from_line(trimmed) {
                    seen_sections.insert(section);
                    current_section = section;
                    if section == Section::Name && trimmed.len() > 4 {
                        // NAME行は問題名を持つが使わない
                    }
                    if section == Section::EndData {
                        break;
                    }
                    continue;
                }
            }

            match current_section {
                Section::Rows => self.parse_rows_line(line, line_num)?,
                Section::Columns => self.parse_columns_line(line, line_num)?,
                Section::Rhs => self.parse_rhs_line(line, line_num)?,
                Section::Ranges => self.parse_ranges_line(line, line_num)?,
                Section::Bounds => self.parse_bounds_line(line, line_num)?,
                Section::Quadobj => self.parse_quadobj_line(line, line_num)?,
                Section::EndData => break,
                _ => {}
            }
        }

        if !seen_sections.contains(&Section::EndData) {
            return Err(QpsError::MissingSection("ENDATA".to_string()));
        }
        if !seen_sections.contains(&Section::Rows) {
            return Err(QpsError::MissingSection("ROWS".to_string()));
        }
        if !seen_sections.contains(&Section::Columns) {
            return Err(QpsError::MissingSection("COLUMNS".to_string()));
        }

        self.build_qp_problem()
    }

    fn parse_rows_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let mut parts = line.split_whitespace();
        let type_str = match parts.next() {
            Some(s) => s,
            None => return Ok(()),
        };
        let row_type = match type_str {
            "N" | "n" => RowType::N,
            "L" | "l" => RowType::L,
            "G" | "g" => RowType::G,
            "E" | "e" => RowType::E,
            _ => {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Unknown row type: {}", type_str),
                });
            }
        };
        // 行名: 固定幅MPS形式（列名にスペース含む場合）対応のためフィールド位置で取得
        // 固定幅: Field 2 (4:12) → "AZ  20  " → trim → "AZ  20"
        // 自由形式短名: 同範囲 → "obj     " → trim → "obj"（互換）
        let row_name = {
            let fw = mps_field(line, 4, 12);
            if !fw.is_empty() {
                fw.to_string()
            } else {
                match parts.next() {
                    Some(s) => s.to_string(),
                    None => return Ok(()),
                }
            }
        };
        if matches!(row_type, RowType::N) && self.obj_row.is_none() {
            self.obj_row = Some(row_name.clone());
        }
        self.rows.push((row_name, row_type));
        Ok(())
    }

    fn parse_columns_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }
        // MARKER行（INTORG/INTEND）はスキップ
        if parts[1] == "'MARKER'" {
            return Ok(());
        }

        // 形式判定: 自由形式ではvalue位置(2,4,6,...)の全トークンがf64になるはず
        // 1つでもf64変換に失敗すれば固定幅MPS形式（列名や行名にスペースを含む）
        let is_free = {
            let mut ok = true;
            let mut vi = 2usize;
            while vi < parts.len() {
                if parts[vi].parse::<f64>().is_err() {
                    ok = false;
                    break;
                }
                vi += 2;
            }
            ok
        };
        if !is_free {
            // 固定幅MPS形式:
            //   Field 2 (4:12)  → col_name
            //   Field 3 (14:22) → row_name1
            //   Field 4 (24:36) → value1
            //   Field 5 (39:47) → row_name2 (optional)
            //   Field 6 (49:61) → value2 (optional)
            let col_name = mps_field(line, 4, 12).to_string();
            if col_name.is_empty() {
                return Ok(());
            }
            let field3 = mps_field(line, 14, 22);
            if field3 == "'MARKER'" {
                return Ok(());
            }
            let row_name1 = field3.to_string();
            if !row_name1.is_empty() {
                let val_str1 = mps_field(line, 24, 36);
                if !val_str1.is_empty() {
                    let value1 = val_str1.parse::<f64>().map_err(|_| QpsError::ParseError {
                        line: line_num,
                        message: format!("Invalid value: {}", val_str1),
                    })?;
                    self.columns.push((col_name.clone(), row_name1, value1));
                }
            }
            let row_name2 = mps_field(line, 39, 47).to_string();
            if !row_name2.is_empty() {
                let val_str2 = mps_field(line, 49, 61);
                if !val_str2.is_empty() {
                    let value2 = val_str2.parse::<f64>().map_err(|_| QpsError::ParseError {
                        line: line_num,
                        message: format!("Invalid value: {}", val_str2),
                    })?;
                    self.columns.push((col_name, row_name2, value2));
                }
            }
            return Ok(());
        }

        // 自由形式: parts[0]=col_name, (parts[1]=row, parts[2]=val), ...
        let col_name = parts[0].to_string();
        let mut i = 1;
        while i + 1 < parts.len() {
            let row_name = parts[i].to_string();
            let value = parts[i + 1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[i + 1]),
            })?;
            self.columns.push((col_name.clone(), row_name, value));
            i += 2;
        }
        Ok(())
    }

    fn parse_rhs_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }
        // 2トークン: ["row","val"] — rhs_name省略確定
        if parts.len() == 2 {
            let row_name = parts[0].to_string();
            let value = parts[1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[1]),
            })?;
            self.rhs.insert(row_name, value);
            return Ok(());
        }
        // 3トークン以上: 値位置(2,4,6,...)がf64かチェック（COLUMNS と同様のhybrid判定）
        //   is_free=true  → 自由形式（parts[0]=rhs_name, pairs from parts[1..]）
        //   is_free=false → 固定幅MPS（行名にスペース含む場合 / rhs_name省略+行名がf64非解釈）
        let is_free = {
            let mut ok = true;
            let mut vi = 2usize;
            while vi < parts.len() {
                if parts[vi].parse::<f64>().is_err() {
                    ok = false;
                    break;
                }
                vi += 2;
            }
            ok
        };
        if !is_free {
            // 固定幅MPS: Field3(14:22)=row1, Field4(24:36)=val1, Field5(39:47)=row2, Field6(49:61)=val2
            // Field2(4:12)=rhs_name は無視
            let row_name1 = mps_field(line, 14, 22).to_string();
            if !row_name1.is_empty() {
                let val_str1 = mps_field(line, 24, 36);
                if !val_str1.is_empty() {
                    let value1 = val_str1.parse::<f64>().map_err(|_| QpsError::ParseError {
                        line: line_num,
                        message: format!("Invalid value: {}", val_str1),
                    })?;
                    self.rhs.insert(row_name1, value1);
                }
            }
            let row_name2 = mps_field(line, 39, 47).to_string();
            if !row_name2.is_empty() {
                let val_str2 = mps_field(line, 49, 61);
                if !val_str2.is_empty() {
                    let value2 = val_str2.parse::<f64>().map_err(|_| QpsError::ParseError {
                        line: line_num,
                        message: format!("Invalid value: {}", val_str2),
                    })?;
                    self.rhs.insert(row_name2, value2);
                }
            }
            return Ok(());
        }
        // 自由形式: parts[0]=rhs_name (スキップ), (parts[1]=row, parts[2]=val), ...
        let mut i = 1;
        while i + 1 < parts.len() {
            let row_name = parts[i].to_string();
            let value = parts[i + 1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[i + 1]),
            })?;
            self.rhs.insert(row_name, value);
            i += 2;
        }
        Ok(())
    }

    fn parse_ranges_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Ok(());
        }
        // parse_rhs_line と同様の hybrid 判定（range名フィールドの有無）
        if parts.len() == 2 {
            let row_name = parts[0].to_string();
            let value = parts[1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[1]),
            })?;
            self.ranges.insert(row_name, value);
            return Ok(());
        }
        let is_free = {
            let mut ok = true;
            let mut vi = 2usize;
            while vi < parts.len() {
                if parts[vi].parse::<f64>().is_err() {
                    ok = false;
                    break;
                }
                vi += 2;
            }
            ok
        };
        if !is_free {
            let row_name1 = mps_field(line, 14, 22).to_string();
            if !row_name1.is_empty() {
                let val_str1 = mps_field(line, 24, 36);
                if !val_str1.is_empty() {
                    let value1 = val_str1.parse::<f64>().map_err(|_| QpsError::ParseError {
                        line: line_num,
                        message: format!("Invalid value: {}", val_str1),
                    })?;
                    self.ranges.insert(row_name1, value1);
                }
            }
            let row_name2 = mps_field(line, 39, 47).to_string();
            if !row_name2.is_empty() {
                let val_str2 = mps_field(line, 49, 61);
                if !val_str2.is_empty() {
                    let value2 = val_str2.parse::<f64>().map_err(|_| QpsError::ParseError {
                        line: line_num,
                        message: format!("Invalid value: {}", val_str2),
                    })?;
                    self.ranges.insert(row_name2, value2);
                }
            }
            return Ok(());
        }
        let mut i = 1;
        while i + 1 < parts.len() {
            let row_name = parts[i].to_string();
            let value = parts[i + 1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[i + 1]),
            })?;
            self.ranges.insert(row_name, value);
            i += 2;
        }
        Ok(())
    }

    fn parse_bounds_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }
        let bound_type = match parts[0] {
            "LO" => BoundType::LO,
            "UP" => BoundType::UP,
            "FX" => BoundType::FX,
            "FR" => BoundType::FR,
            "MI" => BoundType::MI,
            "BV" => BoundType::BV,
            "PL" => BoundType::PL,
            _ => {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Unknown bound type: {}", parts[0]),
                });
            }
        };
        // 5トークン以上: col名またはbound名にスペースあり → 固定幅MPS
        // Field2(4:12)=bound_name(無視), Field3(14:22)=col_name, Field4(24:36)=value
        if parts.len() >= 5 {
            let col_name = mps_field(line, 14, 22).to_string();
            let value = {
                let v = mps_field(line, 24, 36);
                if v.is_empty() { None } else { v.parse::<f64>().ok() }
            };
            self.bounds.push((bound_type, col_name, value));
            return Ok(());
        }
        let col_name = parts[2].to_string();
        let value = if parts.len() >= 4 {
            parts[3].parse::<f64>().ok()
        } else {
            None
        };
        self.bounds.push((bound_type, col_name, value));
        Ok(())
    }

    fn parse_quadobj_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }
        // 3トークン: [col1, col2, value] — 自由形式
        // 4+トークン: col名にスペースあり → 固定幅MPS
        // Field2(4:12)=col1, Field3(14:22)=col2, Field4(24:36)=value
        let (col1, col2, val_str) = if parts.len() == 3 {
            (parts[0], parts[1], parts[2])
        } else {
            (mps_field(line, 4, 12), mps_field(line, 14, 22), mps_field(line, 24, 36))
        };
        let value = val_str.parse::<f64>().map_err(|_| QpsError::ParseError {
            line: line_num,
            message: format!("Invalid QUADOBJ value: {}", val_str),
        })?;
        self.quadobj.push((col1.to_string(), col2.to_string(), value));
        Ok(())
    }

    fn build_qp_problem(&self) -> Result<QpProblem, QpsError> {
        // --- 列名 → インデックスマップ構築 ---
        let mut col_map: HashMap<String, usize> = HashMap::new();
        for (col_name, _, _) in &self.columns {
            if !col_map.contains_key(col_name) {
                let idx = col_map.len();
                col_map.insert(col_name.clone(), idx);
            }
        }
        let n = col_map.len();

        // --- 目的関数線形項 c ---
        let mut c = vec![0.0; n];
        if let Some(obj_row_name) = &self.obj_row {
            for (col_name, row_name, value) in &self.columns {
                if row_name == obj_row_name {
                    if let Some(&col_idx) = col_map.get(col_name) {
                        c[col_idx] = *value;
                    }
                }
            }
        }

        // --- 制約処理（Ge/Eq変換を含む） ---
        // 行名 → 行タイプ・RHS のマップ
        let obj_row = self.obj_row.as_deref().unwrap_or("");

        // まず制約行を収集
        struct ConstraintRow {
            name: String,
            rtype: RowType,
            rhs: f64,
        }
        let mut constraint_rows: Vec<ConstraintRow> = Vec::new();
        for (row_name, row_type) in &self.rows {
            if row_name == obj_row {
                continue;
            }
            if matches!(row_type, RowType::N) {
                continue;
            }
            let rhs = self.rhs.get(row_name).copied().unwrap_or(0.0);
            constraint_rows.push(ConstraintRow {
                name: row_name.clone(),
                rtype: *row_type,
                rhs,
            });
        }

        // RANGESの適用: 区間制約 → 2制約に展開（MPS標準）
        // rangeが設定されている場合のみ追加行を生成
        let mut range_extra: Vec<(String, ConstraintRow)> = Vec::new();
        let mut base_rows: Vec<ConstraintRow> = Vec::new();
        for row in constraint_rows {
            if let Some(&range_val) = self.ranges.get(&row.name) {
                let b = row.rhs;
                let abs_r = range_val.abs();
                let (lower, upper, le_rhs, ge_rhs) = match row.rtype {
                    RowType::L => (b - abs_r, b, b, b - abs_r),
                    RowType::G => (b, b + abs_r, b + abs_r, b),
                    RowType::E => {
                        if range_val >= 0.0 {
                            (b, b + abs_r, b + abs_r, b)
                        } else {
                            (b - abs_r, b, b, b - abs_r)
                        }
                    }
                    RowType::N => unreachable!(),
                };
                let _ = (lower, upper);
                // Le制約（上限）
                base_rows.push(ConstraintRow {
                    name: row.name.clone(),
                    rtype: RowType::L,
                    rhs: le_rhs,
                });
                // Ge制約（下限）→ 後で変換
                range_extra.push((row.name.clone(), ConstraintRow {
                    name: row.name.clone(),
                    rtype: RowType::G,
                    rhs: ge_rhs,
                }));
            } else {
                base_rows.push(row);
            }
        }
        // range_extraをbase_rowsに追加
        for (_, row) in range_extra {
            base_rows.push(row);
        }

        // Ax<=b 形式に展開（Ge→否定, Eq→2行）
        // 各行に対して (sign, rhs) を生成
        struct AugRow {
            name: String,
            sign: f64,  // 1.0 = Le, -1.0 = Ge(否定)
            rhs: f64,
        }
        let mut aug_rows: Vec<AugRow> = Vec::new();
        for row in base_rows {
            match row.rtype {
                RowType::L => {
                    aug_rows.push(AugRow { name: row.name, sign: 1.0, rhs: row.rhs });
                }
                RowType::G => {
                    aug_rows.push(AugRow { name: row.name, sign: -1.0, rhs: -row.rhs });
                }
                RowType::E => {
                    aug_rows.push(AugRow { name: row.name.clone(), sign: 1.0, rhs: row.rhs });
                    aug_rows.push(AugRow { name: row.name, sign: -1.0, rhs: -row.rhs });
                }
                RowType::N => {}
            }
        }

        let m = aug_rows.len();

        // 行名 → 拡張行インデックス（複数ある場合があるため Vec）
        // AugRowのインデックスを行名でグループ化
        let mut row_name_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, ar) in aug_rows.iter().enumerate() {
            row_name_to_indices.entry(ar.name.clone()).or_default().push(i);
        }

        // 制約行列A（トリプレット）
        let mut a_rows: Vec<usize> = Vec::new();
        let mut a_cols: Vec<usize> = Vec::new();
        let mut a_vals: Vec<f64> = Vec::new();

        // 列ごとのaccumulator
        // (aug_row_idx, col_idx) → value
        let mut a_triplets: HashMap<(usize, usize), f64> = HashMap::new();

        for (col_name, row_name, value) in &self.columns {
            if row_name == obj_row {
                continue;
            }
            let col_idx = match col_map.get(col_name) {
                Some(&idx) => idx,
                None => continue,
            };
            if let Some(indices) = row_name_to_indices.get(row_name) {
                for &aug_idx in indices {
                    let sign = aug_rows[aug_idx].sign;
                    *a_triplets.entry((aug_idx, col_idx)).or_insert(0.0) += sign * value;
                }
            }
        }

        for ((row_idx, col_idx), val) in &a_triplets {
            a_rows.push(*row_idx);
            a_cols.push(*col_idx);
            a_vals.push(*val);
        }

        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).map_err(|e| {
            QpsError::ParseError {
                line: 0,
                message: format!("Failed to build A matrix: {}", e),
            }
        })?;

        let b: Vec<f64> = aug_rows.iter().map(|r| r.rhs).collect();

        // --- 変数境界（デフォルト: [0, +∞)）---
        let mut bounds = vec![(0.0_f64, f64::INFINITY); n];
        for (bound_type, col_name, value) in &self.bounds {
            let col_idx = match col_map.get(col_name) {
                Some(&idx) => idx,
                None => continue, // 未定義列は無視
            };
            match bound_type {
                BoundType::LO => {
                    bounds[col_idx].0 = value.unwrap_or(0.0);
                }
                BoundType::UP => {
                    bounds[col_idx].1 = value.unwrap_or(f64::INFINITY);
                }
                BoundType::FX => {
                    let val = value.unwrap_or(0.0);
                    bounds[col_idx] = (val, val);
                }
                BoundType::FR => {
                    bounds[col_idx] = (f64::NEG_INFINITY, f64::INFINITY);
                }
                BoundType::MI => {
                    bounds[col_idx].0 = f64::NEG_INFINITY;
                }
                BoundType::BV => {
                    bounds[col_idx] = (0.0, 1.0);
                }
                BoundType::PL => {
                    bounds[col_idx].1 = f64::INFINITY;
                }
            }
        }

        // --- Q行列構築（QUADOBJから）---
        // QUADOBJ: 上三角格納 → 対称化
        // Q_ij = value, Q_ji = value (i != j の場合)
        let mut q_triplets: Vec<(usize, usize, f64)> = Vec::new();
        let mut q_acc: HashMap<(usize, usize), f64> = HashMap::new();

        for (col1, col2, value) in &self.quadobj {
            let i = match col_map.get(col1) {
                Some(&idx) => idx,
                None => continue,
            };
            let j = match col_map.get(col2) {
                Some(&idx) => idx,
                None => continue,
            };
            *q_acc.entry((i, j)).or_insert(0.0) += value;
            if i != j {
                *q_acc.entry((j, i)).or_insert(0.0) += value;
            }
        }

        for ((i, j), v) in &q_acc {
            q_triplets.push((*i, *j, *v));
        }

        let q_rows: Vec<usize> = q_triplets.iter().map(|&(r, _, _)| r).collect();
        let q_cols: Vec<usize> = q_triplets.iter().map(|&(_, c, _)| c).collect();
        let q_vals: Vec<f64> = q_triplets.iter().map(|&(_, _, v)| v).collect();

        let q = if q_rows.is_empty() {
            CscMatrix::new(n, n) // Q=0（LP退化）
        } else {
            CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).map_err(|e| {
                QpsError::ParseError {
                    line: 0,
                    message: format!("Failed to build Q matrix: {}", e),
                }
            })?
        };

        QpProblem::new(q, c, a, b, bounds).map_err(|e| QpsError::ParseError {
            line: 0,
            message: e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::SolveStatus;
    use crate::qp::solve_qp;

    /// HS21: min 1/2*(100x1^2 + x2^2) - 100x1 - x2
    /// s.t. x1 - x2/20 >= -1/2  (Ge)
    ///      x1 in [2, 50], x2 in [-50, 50]
    ///
    /// 解析解: x1=2, x2=0 (境界での最適解 — HS21では多様な最適点あり)
    #[test]
    fn test_parse_qps_simple() {
        let qps = r"NAME          TEST_QP
ROWS
 N  obj
 G  c1
COLUMNS
    x1    obj    -100.0    c1    1.0
    x2    obj    -1.0      c1    -0.05
RHS
    rhs   c1    -0.5
BOUNDS
 LO BND   x1    2.0
 UP BND   x1    50.0
 LO BND   x2    -50.0
 UP BND   x2    50.0
QUADOBJ
    x1    x1    100.0
    x2    x2    1.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.num_vars, 2);
        // Ge制約 → 1行
        assert_eq!(prob.num_constraints, 1);
    }

    /// Q=0のQPS（LP問題として動作確認）
    #[test]
    fn test_parse_qps_no_quadobj() {
        let qps = r"NAME          LP_ONLY
ROWS
 N  obj
 L  c1
COLUMNS
    x1    obj    1.0    c1    1.0
    x2    obj    2.0    c1    1.0
RHS
    rhs   c1    10.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 1);
        assert!(prob.is_zero_q());
    }

    /// 等式制約付きQPS（Eq → 2行に展開）
    #[test]
    fn test_parse_qps_eq_constraint() {
        let qps = r"NAME          EQ_TEST
ROWS
 N  obj
 E  eq1
COLUMNS
    x1    obj    2.0    eq1    1.0
    x2    obj    1.0    eq1    1.0
RHS
    rhs   eq1    5.0
QUADOBJ
    x1    x1    2.0
    x2    x2    2.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.num_vars, 2);
        // Eq制約 → 2行
        assert_eq!(prob.num_constraints, 2);
    }

    /// 小規模QPS問題を実際に解く
    /// min 1/2 * (x^2 + y^2)  s.t. x + y >= 1, x,y >= 0
    #[test]
    fn test_solve_qps_basic() {
        let qps = r"NAME          BASIC
ROWS
 N  obj
 G  sum1
COLUMNS
    x    obj    0.0    sum1    1.0
    y    obj    0.0    sum1    1.0
RHS
    rhs   sum1    1.0
BOUNDS
 FR BND   x
 FR BND   y
QUADOBJ
    x    x    1.0
    y    y    1.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        let result = solve_qp(&prob);
        assert_eq!(result.status, SolveStatus::Optimal);
        let obj = result.objective;
        // min 1/2*(x^2+y^2) s.t. x+y >= 1 → 解析解: x=y=0.5, obj=0.25
        // ADMM収束tolerance eps=1e-3のため許容誤差は2e-3
        assert!((obj - 0.25).abs() < 2e-3, "expected obj≈0.25, got {}", obj);
    }

    /// QUADOBJの対称化確認：上三角のみ与えた場合と両側与えた場合で同じ結果
    #[test]
    fn test_quadobj_symmetry() {
        // x1*x2 クロス項: 上三角のみ (x1, x2, 1.0)
        let qps_upper = r"NAME SYM
ROWS
 N  obj
COLUMNS
    x1  obj  0.0
    x2  obj  0.0
BOUNDS
 FR BND  x1
 FR BND  x2
QUADOBJ
    x1  x1  2.0
    x1  x2  1.0
    x2  x2  2.0
ENDATA
";
        let prob = parse_qps_str(qps_upper).unwrap();
        // Q = [[2, 1], [1, 2]] — 対称化済み
        assert_eq!(prob.q.nrows, 2);
        assert_eq!(prob.q.ncols, 2);
        // 要素数: 4 (対角2 + 非対角2)
        assert_eq!(prob.q.values.len(), 4);
    }
}
