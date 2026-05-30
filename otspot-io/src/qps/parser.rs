use std::collections::{HashMap, HashSet};
use std::io::BufRead;

use otspot_core::problem::ConstraintType;
use otspot_core::qp::QpProblem;
use otspot_core::sparse::CscMatrix;

use crate::common::{RowType, mps_field, parse_mps_free_pairs, parse_mps_fixed_pairs};
use super::types::{BoundType, Section};
use super::QpsError;

pub(super) struct QpsParser {
    rows: Vec<(String, RowType)>,
    columns: Vec<(String, String, f64)>,
    rhs: HashMap<String, f64>,
    ranges: HashMap<String, f64>,
    bounds: Vec<(BoundType, String, Option<f64>)>,
    /// QUADOBJ entries: (col1, col2, value) in upper-triangular order.
    quadobj: Vec<(String, String, f64)>,
    /// Tracks normalized (min, max) key pairs seen in QUADOBJ to detect symmetric duplicates.
    quadobj_seen: HashSet<(String, String)>,
    obj_row: Option<String>,
    maximize: bool,
}

impl QpsParser {
    pub(super) fn new() -> Self {
        Self {
            rows: Vec::new(),
            columns: Vec::new(),
            rhs: HashMap::new(),
            ranges: HashMap::new(),
            bounds: Vec::new(),
            quadobj: Vec::new(),
            quadobj_seen: HashSet::new(),
            obj_row: None,
            maximize: false,
        }
    }

    pub(super) fn parse_reader<R: BufRead>(
        &mut self,
        reader: R,
    ) -> Result<QpProblem, QpsError> {
        let mut current_section = Section::None;
        let mut seen_sections = std::collections::HashSet::new();
        let mut line_num = 0;

        for line_result in reader.lines() {
            let line = line_result.map_err(QpsError::IoError)?;
            line_num += 1;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with('$') {
                continue;
            }

            if !line.starts_with(' ') && !line.starts_with('\t') {
                if let Some(section) = Section::from_line(trimmed) {
                    if section != Section::Name
                        && section != Section::EndData
                        && seen_sections.contains(&section)
                    {
                        return Err(QpsError::DuplicateSection(format!("{:?}", section)));
                    }
                    seen_sections.insert(section);
                    current_section = section;
                    if section == Section::EndData {
                        break;
                    }
                    continue;
                } else {
                    return Err(QpsError::ParseError {
                        line: line_num,
                        message: format!("Unrecognized section header: '{}'", trimmed),
                    });
                }
            }

            match current_section {
                Section::ObjSense => self.parse_objsense_line(&line, line_num)?,
                Section::Rows => self.parse_rows_line(&line, line_num)?,
                Section::Columns => self.parse_columns_line(&line, line_num)?,
                Section::Rhs => self.parse_rhs_line(&line, line_num)?,
                Section::Ranges => self.parse_ranges_line(&line, line_num)?,
                Section::Bounds => self.parse_bounds_line(&line, line_num)?,
                Section::Quadobj => self.parse_quadobj_line(&line, line_num)?,
                Section::EndData => break,
                Section::None | Section::Name => {}
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

    fn parse_objsense_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let upper = line.trim().to_uppercase();
        match upper.as_str() {
            "MAX" => self.maximize = true,
            "MIN" => self.maximize = false,
            _ => {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Invalid OBJSENSE value '{}'; expected MIN or MAX", line.trim()),
                });
            }
        }
        Ok(())
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
        let row_name = {
            let fw = mps_field(line, 4, 12);
            if !fw.is_empty() {
                fw.to_string()
            } else {
                match parts.next() {
                    Some(s) => s.to_string(),
                    None => {
                        return Err(QpsError::ParseError {
                            line: line_num,
                            message: "ROWS line missing row name".to_string(),
                        });
                    }
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
            return Err(QpsError::ParseError {
                line: line_num,
                message: "COLUMNS line requires at least 3 fields (col row value)".to_string(),
            });
        }
        if parts[1] == "'MARKER'" {
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
            let col_name = mps_field(line, 4, 12).to_string();
            if col_name.is_empty() {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: "COLUMNS fixed-format line missing column name at field 2".to_string(),
                });
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
                    if !value1.is_finite() {
                        return Err(QpsError::ParseError {
                            line: line_num,
                            message: format!("Non-finite COLUMNS value for col='{}' row='{}'", col_name, row_name1),
                        });
                    }
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
                    if !value2.is_finite() {
                        return Err(QpsError::ParseError {
                            line: line_num,
                            message: format!("Non-finite COLUMNS value for col='{}' row='{}'", col_name, row_name2),
                        });
                    }
                    self.columns.push((col_name, row_name2, value2));
                }
            }
            return Ok(());
        }

        let col_name = parts[0].to_string();
        let mut i = 1;
        while i + 1 < parts.len() {
            let row_name = parts[i].to_string();
            let value = parts[i + 1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[i + 1]),
            })?;
            if !value.is_finite() {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Non-finite COLUMNS value for col='{}' row='{}'", col_name, row_name),
                });
            }
            self.columns.push((col_name.clone(), row_name, value));
            i += 2;
        }
        Ok(())
    }

    fn parse_rhs_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(QpsError::ParseError {
                line: line_num,
                message: "RHS line requires at least 2 fields".to_string(),
            });
        }
        // 2-field shorthand: (row_name, value) without a preceding rhs_section_name.
        if parts.len() == 2 {
            let row_name = parts[0].to_string();
            let value = parts[1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[1]),
            })?;
            if !value.is_finite() {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Non-finite RHS value for row='{}'", row_name),
                });
            }
            self.rhs.insert(row_name, value);
            return Ok(());
        }
        let force_fixed =
            mps_field(line, 4, 12).is_empty() && !mps_field(line, 14, 22).is_empty();
        let is_free = if force_fixed {
            false
        } else {
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
        // finite check skipped for obj-row RHS (handled as obj_offset at build step)
        let pairs = if is_free {
            parse_mps_free_pairs(&parts, line_num, "RHS", false)
        } else {
            parse_mps_fixed_pairs(line, line_num, "RHS", false)
        }
        .map_err(|msg| QpsError::ParseError { line: line_num, message: msg })?;
        for (name, value) in pairs {
            self.rhs.insert(name, value);
        }
        Ok(())
    }

    fn parse_ranges_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(QpsError::ParseError {
                line: line_num,
                message: "RANGES line requires at least 2 fields".to_string(),
            });
        }
        // 2-field shorthand: (row_name, value) without a preceding rhs_section_name.
        if parts.len() == 2 {
            let row_name = parts[0].to_string();
            let value = parts[1].parse::<f64>().map_err(|_| QpsError::ParseError {
                line: line_num,
                message: format!("Invalid value: {}", parts[1]),
            })?;
            if !value.is_finite() {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Non-finite RANGES value for row='{}'", row_name),
                });
            }
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
        let pairs = if is_free {
            parse_mps_free_pairs(&parts, line_num, "RANGES", true)
        } else {
            parse_mps_fixed_pairs(line, line_num, "RANGES", true)
        }
        .map_err(|msg| QpsError::ParseError { line: line_num, message: msg })?;
        for (name, value) in pairs {
            self.ranges.insert(name, value);
        }
        Ok(())
    }

    fn parse_bounds_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(QpsError::ParseError {
                line: line_num,
                message: "BOUNDS line requires at least 3 fields (type name col)".to_string(),
            });
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
        if parts.len() >= 5 {
            let col_name = mps_field(line, 14, 22).to_string();
            let value = {
                let v = mps_field(line, 24, 36);
                if v.is_empty() {
                    None
                } else {
                    let parsed = v.parse::<f64>().ok();
                    if let Some(val) = parsed {
                        if !val.is_finite() {
                            return Err(QpsError::ParseError {
                                line: line_num,
                                message: format!("Non-finite BOUNDS value for col='{}'", col_name),
                            });
                        }
                    }
                    parsed
                }
            };
            self.bounds.push((bound_type, col_name, value));
            return Ok(());
        }
        let value_taking = !matches!(
            bound_type,
            BoundType::FR | BoundType::MI | BoundType::PL | BoundType::BV
        );
        let (col_name, value) = if !value_taking {
            (parts[2].to_string(), None)
        } else if parts.len() >= 4 {
            let raw = parts[3];
            let parsed = raw.parse::<f64>().ok();
            if let Some(val) = parsed {
                if !val.is_finite() {
                    return Err(QpsError::ParseError {
                        line: line_num,
                        message: format!("Non-finite BOUNDS value for col='{}'", parts[2]),
                    });
                }
            }
            (parts[2].to_string(), parsed)
        } else if let Ok(v) = parts[2].parse::<f64>() {
            if !v.is_finite() {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Non-finite BOUNDS value for col='{}'", parts[1]),
                });
            }
            (parts[1].to_string(), Some(v))
        } else {
            (parts[2].to_string(), None)
        };
        self.bounds.push((bound_type, col_name, value));
        Ok(())
    }

    fn parse_quadobj_line(&mut self, line: &str, line_num: usize) -> Result<(), QpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(QpsError::ParseError {
                line: line_num,
                message: "QUADOBJ line requires at least 3 fields (col1 col2 value)".to_string(),
            });
        }
        let (col1, col2, val_str) = if parts.len() == 3 {
            (parts[0], parts[1], parts[2])
        } else {
            (mps_field(line, 4, 12), mps_field(line, 14, 22), mps_field(line, 24, 36))
        };
        let value = val_str.parse::<f64>().map_err(|_| QpsError::ParseError {
            line: line_num,
            message: format!("Invalid QUADOBJ value: {}", val_str),
        })?;
        if !value.is_finite() {
            return Err(QpsError::ParseError {
                line: line_num,
                message: format!("Non-finite QUADOBJ value for ({}, {})", col1, col2),
            });
        }
        // Reject duplicate entries in QUADOBJ using a lexicographically normalized key.
        // This catches both (x1,x2) and (x2,x1) as duplicates, since both represent
        // the same upper-triangular Q entry after symmetrization.
        let key = if col1 <= col2 {
            (col1.to_string(), col2.to_string())
        } else {
            (col2.to_string(), col1.to_string())
        };
        if !self.quadobj_seen.insert(key) {
            return Err(QpsError::ParseError {
                line: line_num,
                message: format!("Duplicate QUADOBJ entry: ({}, {})", col1, col2),
            });
        }
        self.quadobj.push((col1.to_string(), col2.to_string(), value));
        Ok(())
    }

    fn build_qp_problem(&self) -> Result<QpProblem, QpsError> {
        let mut col_map: HashMap<String, usize> = HashMap::new();
        for (col_name, _, _) in &self.columns {
            if !col_map.contains_key(col_name) {
                let idx = col_map.len();
                col_map.insert(col_name.clone(), idx);
            }
        }
        let n = col_map.len();

        let mut c = vec![0.0; n];
        if let Some(obj_row_name) = &self.obj_row {
            for (col_name, row_name, value) in &self.columns {
                if row_name == obj_row_name {
                    if let Some(&col_idx) = col_map.get(col_name) {
                        c[col_idx] += *value;
                    }
                }
            }
        }

        let obj_row = self.obj_row.as_deref().unwrap_or("");

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

        let mut range_extra: Vec<(String, ConstraintRow)> = Vec::new();
        let mut base_rows: Vec<ConstraintRow> = Vec::new();
        for row in constraint_rows {
            if let Some(&range_val) = self.ranges.get(&row.name) {
                let b = row.rhs;
                let abs_r = range_val.abs();
                let (le_rhs, ge_rhs) = match row.rtype {
                    RowType::L => (b, b - abs_r),
                    RowType::G => (b + abs_r, b),
                    RowType::E => {
                        if range_val >= 0.0 {
                            (b + abs_r, b)
                        } else {
                            (b, b - abs_r)
                        }
                    }
                    RowType::N => unreachable!(),
                };
                base_rows.push(ConstraintRow {
                    name: row.name.clone(),
                    rtype: RowType::L,
                    rhs: le_rhs,
                });
                range_extra.push((row.name.clone(), ConstraintRow {
                    name: row.name.clone(),
                    rtype: RowType::G,
                    rhs: ge_rhs,
                }));
            } else {
                base_rows.push(row);
            }
        }
        for (_, row) in range_extra {
            base_rows.push(row);
        }

        struct AugRow {
            name: String,
            sign: f64,
            rhs: f64,
        }
        let mut aug_rows: Vec<AugRow> = Vec::new();
        let mut constraint_types: Vec<ConstraintType> = Vec::new();
        for row in base_rows {
            match row.rtype {
                RowType::L => {
                    aug_rows.push(AugRow { name: row.name, sign: 1.0, rhs: row.rhs });
                    constraint_types.push(ConstraintType::Le);
                }
                RowType::G => {
                    aug_rows.push(AugRow { name: row.name, sign: -1.0, rhs: -row.rhs });
                    constraint_types.push(ConstraintType::Le);
                }
                RowType::E => {
                    aug_rows.push(AugRow { name: row.name, sign: 1.0, rhs: row.rhs });
                    constraint_types.push(ConstraintType::Eq);
                }
                RowType::N => {}
            }
        }

        let m = aug_rows.len();

        let mut row_name_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, ar) in aug_rows.iter().enumerate() {
            row_name_to_indices.entry(ar.name.clone()).or_default().push(i);
        }

        let mut a_rows: Vec<usize> = Vec::new();
        let mut a_cols: Vec<usize> = Vec::new();
        let mut a_vals: Vec<f64> = Vec::new();

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
                    a_rows.push(aug_idx);
                    a_cols.push(col_idx);
                    a_vals.push(sign * value);
                }
            }
        }

        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).map_err(|e| {
            QpsError::ParseError {
                line: 0,
                message: format!("Failed to build A matrix: {}", e),
            }
        })?;

        let b: Vec<f64> = aug_rows.iter().map(|r| r.rhs).collect();

        let mut bounds = vec![(0.0_f64, f64::INFINITY); n];
        for (bound_type, col_name, value) in &self.bounds {
            let col_idx = match col_map.get(col_name) {
                Some(&idx) => idx,
                None => continue,
            };
            match bound_type {
                BoundType::LO => bounds[col_idx].0 = value.unwrap_or(0.0),
                BoundType::UP => bounds[col_idx].1 = value.unwrap_or(f64::INFINITY),
                BoundType::FX => {
                    let val = value.unwrap_or(0.0);
                    bounds[col_idx] = (val, val);
                }
                BoundType::FR => bounds[col_idx] = (f64::NEG_INFINITY, f64::INFINITY),
                BoundType::MI => bounds[col_idx].0 = f64::NEG_INFINITY,
                BoundType::BV => bounds[col_idx] = (0.0, 1.0),
                BoundType::PL => bounds[col_idx].1 = f64::INFINITY,
            }
        }

        // QUADOBJ: upper-triangular → symmetrize
        let mut q_rows: Vec<usize> = Vec::new();
        let mut q_cols: Vec<usize> = Vec::new();
        let mut q_vals: Vec<f64> = Vec::new();

        for (col1, col2, value) in &self.quadobj {
            let i = match col_map.get(col1) {
                Some(&idx) => idx,
                None => continue,
            };
            let j = match col_map.get(col2) {
                Some(&idx) => idx,
                None => continue,
            };
            q_rows.push(i); q_cols.push(j); q_vals.push(*value);
            if i != j {
                q_rows.push(j); q_cols.push(i); q_vals.push(*value);
            }
        }

        // Normalize MAX → MIN by negating objective (c and Q).
        if self.maximize {
            for v in &mut c {
                *v = -*v;
            }
            for v in &mut q_vals {
                *v = -*v;
            }
        }

        let q = if q_rows.is_empty() {
            CscMatrix::new(n, n)
        } else {
            CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).map_err(|e| {
                QpsError::ParseError {
                    line: 0,
                    message: format!("Failed to build Q matrix: {}", e),
                }
            })?
        };

        let obj_offset = match &self.obj_row {
            Some(obj_row_name) => self.rhs.get(obj_row_name).copied().unwrap_or(0.0),
            None => 0.0,
        };
        if !obj_offset.is_finite() {
            return Err(QpsError::InvalidObjectiveOffset(obj_offset));
        }

        let mut prob = QpProblem::new(q, c, a, b, bounds, constraint_types).map_err(|e| {
            QpsError::ParseError {
                line: 0,
                message: e.to_string(),
            }
        })?;
        prob.obj_offset = obj_offset;
        Ok(prob)
    }
}

// ── Public entry points ───────────────────────────────────────────────────────

pub fn parse_qps_reader<R: BufRead>(reader: R) -> Result<QpProblem, QpsError> {
    let mut parser = QpsParser::new();
    parser.parse_reader(reader)
}
