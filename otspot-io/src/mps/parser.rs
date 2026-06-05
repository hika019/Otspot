use std::collections::{HashMap, HashSet};
use std::io::BufRead;

pub use otspot_core::error::MpsError;
use otspot_core::mip::MilpProblem;
use otspot_core::problem::{ConstraintType, LpProblem};
use otspot_core::sparse::CscMatrix;

use super::types::{
    integer_marker_kind, BoundType, IntegerMarker, Section, INTEGER_DEFAULT_UPPER_BINARY,
};
use crate::common::{is_fixed_width_format, parse_mps_free_pairs, RowType};

pub(super) struct MpsParser {
    problem_name: Option<String>,
    rows: Vec<(String, RowType)>,
    columns: Vec<(String, String, f64)>,
    rhs: HashMap<String, f64>,
    ranges: HashMap<String, f64>,
    bounds: Vec<(BoundType, String, Option<f64>)>,
    obj_row: Option<String>,
    integer_cols: HashSet<String>,
    in_integer_marker: bool,
    maximize: bool,
}

impl MpsParser {
    pub(super) fn new() -> Self {
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
            maximize: false,
        }
    }

    /// Reads an MPS stream line-by-line; returns LP relaxation and integer var indices.
    pub(super) fn parse_reader<R: BufRead>(
        &mut self,
        reader: R,
    ) -> Result<(LpProblem, Vec<usize>), MpsError> {
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
                    if section != Section::Name
                        && section != Section::EndData
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
                Section::ObjSense => self.parse_objsense_line(&line, line_num)?,
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

    fn parse_objsense_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let upper = line.trim().to_uppercase();
        match upper.as_str() {
            "MAX" => self.maximize = true,
            "MIN" => self.maximize = false,
            _ => {
                return Err(MpsError::ParseError {
                    line: line_num,
                    message: format!(
                        "Invalid OBJSENSE value '{}'; expected MIN or MAX",
                        line.trim()
                    ),
                });
            }
        }
        Ok(())
    }

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

        if matches!(row_type, RowType::N) && self.obj_row.is_none() {
            self.obj_row = Some(row_name.clone());
        }

        self.rows.push((row_name, row_type));
        Ok(())
    }

    fn parse_columns_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        if let Some(kind) = integer_marker_kind(line) {
            self.in_integer_marker = matches!(kind, IntegerMarker::Start);
            return Ok(());
        }

        if is_fixed_width_format(line) {
            self.parse_columns_fixed(line, line_num)
        } else {
            self.parse_columns_free(line, line_num)
        }
    }

    fn parse_columns_fixed(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let col_name = line.get(4..12).unwrap_or("").trim().to_string();
        let row_name1 = line.get(14..22).unwrap_or("").trim().to_string();
        let value1_str = line.get(24..36).unwrap_or("").trim();

        // Fall back to free-format when fixed positions don't parse cleanly (e.g.
        // MIPLIB files where short col names push the row name past column 22).
        if col_name.is_empty() || row_name1.is_empty() || value1_str.parse::<f64>().is_err() {
            return self.parse_columns_free(line, line_num);
        }
        if self.in_integer_marker {
            self.integer_cols.insert(col_name.clone());
        }

        let value1 = value1_str
            .parse::<f64>()
            .expect("value1_str parseable (checked above)");
        if !value1.is_finite() {
            return Err(MpsError::ParseError {
                line: line_num,
                message: format!(
                    "Non-finite COLUMNS value for col='{}' row='{}'",
                    col_name, row_name1
                ),
            });
        }
        self.columns.push((col_name.clone(), row_name1, value1));

        if line.len() >= 50 {
            let row_name2 = line.get(39..47).unwrap_or("").trim().to_string();
            let value2_str = line.get(49..61).unwrap_or("").trim();

            if !row_name2.is_empty() && !value2_str.is_empty() {
                let value2 = value2_str
                    .parse::<f64>()
                    .map_err(|_| MpsError::ParseError {
                        line: line_num,
                        message: format!("Invalid numeric value: {}", value2_str),
                    })?;
                if !value2.is_finite() {
                    return Err(MpsError::ParseError {
                        line: line_num,
                        message: format!(
                            "Non-finite COLUMNS value for col='{}' row='{}'",
                            col_name, row_name2
                        ),
                    });
                }
                self.columns.push((col_name.clone(), row_name2, value2));
            }
        }

        Ok(())
    }

    fn parse_columns_free(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(MpsError::ParseError {
                line: line_num,
                message: "COLUMNS line requires at least 3 fields (col row value)".to_string(),
            });
        }

        let col_name = parts[0].to_string();
        if self.in_integer_marker {
            self.integer_cols.insert(col_name.clone());
        }

        for i in (1..parts.len()).step_by(2) {
            if i + 1 >= parts.len() {
                return Err(MpsError::ParseError {
                    line: line_num,
                    message: format!(
                        "odd trailing token '{}' in COLUMNS (row name without a value)",
                        parts[i]
                    ),
                });
            }
            let row_name = parts[i].to_string();
            let value = parts[i + 1]
                .parse::<f64>()
                .map_err(|_| MpsError::ParseError {
                    line: line_num,
                    message: format!("Invalid numeric value: {}", parts[i + 1]),
                })?;
            if !value.is_finite() {
                return Err(MpsError::ParseError {
                    line: line_num,
                    message: format!(
                        "Non-finite COLUMNS value for col='{}' row='{}'",
                        col_name, row_name
                    ),
                });
            }
            self.columns.push((col_name.clone(), row_name, value));
        }

        Ok(())
    }

    fn parse_rhs_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(MpsError::ParseError {
                line: line_num,
                message: "RHS line requires at least 3 fields (rhs_name row value)".to_string(),
            });
        }
        // Pass None: non-finite values are rejected for all rows including the N-row.
        // The obj_offset (N-row RHS) is extracted in build_lp_problem after all sections parse.
        let pairs = parse_mps_free_pairs(&parts, line_num, "RHS", None).map_err(|msg| {
            MpsError::ParseError {
                line: line_num,
                message: msg,
            }
        })?;
        for (name, value) in pairs {
            self.rhs.insert(name, value);
        }
        Ok(())
    }

    fn parse_ranges_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(MpsError::ParseError {
                line: line_num,
                message: "RANGES line requires at least 3 fields (rhs_name row value)".to_string(),
            });
        }
        let pairs = parse_mps_free_pairs(&parts, line_num, "RANGES", None).map_err(|msg| {
            MpsError::ParseError {
                line: line_num,
                message: msg,
            }
        })?;
        for (name, value) in pairs {
            self.ranges.insert(name, value);
        }
        Ok(())
    }

    fn parse_bounds_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(MpsError::ParseError {
                line: line_num,
                message: "BOUNDS line requires at least 3 fields (type name col)".to_string(),
            });
        }

        let bound_type_str = parts[0];
        let _bound_name = parts[1];
        let col_name = parts[2].to_string();
        let value = if parts.len() >= 4 {
            let v = parts[3].parse::<f64>().map_err(|_| MpsError::ParseError {
                line: line_num,
                message: format!("Invalid numeric value: {}", parts[3]),
            })?;
            if !v.is_finite() {
                return Err(MpsError::ParseError {
                    line: line_num,
                    message: format!("Non-finite BOUNDS value for col='{}'", col_name),
                });
            }
            Some(v)
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

        let value_required =
            matches!(bound_type, BoundType::LO | BoundType::UP | BoundType::FX | BoundType::LI | BoundType::UI);
        if value_required && value.is_none() {
            return Err(MpsError::ParseError {
                line: line_num,
                message: format!(
                    "BOUNDS type {} requires a value for col='{}'",
                    bound_type_str, col_name
                ),
            });
        }

        if matches!(bound_type, BoundType::BV | BoundType::LI | BoundType::UI) {
            self.integer_cols.insert(col_name.clone());
        }

        self.bounds.push((bound_type, col_name, value));
        Ok(())
    }

    fn build_lp_problem(&self) -> Result<(LpProblem, Vec<usize>), MpsError> {
        let mut row_map = HashMap::new();
        let mut constraint_types = Vec::new();
        let mut rhs_vec = Vec::new();

        for (row_name, row_type) in &self.rows {
            if Some(row_name) == self.obj_row.as_ref() {
                continue;
            }

            let idx = row_map.len();
            row_map.insert(row_name.clone(), idx);

            let constraint_type = match row_type {
                RowType::L => ConstraintType::Le,
                RowType::G => ConstraintType::Ge,
                RowType::E => ConstraintType::Eq,
                RowType::N => continue,
            };
            constraint_types.push(constraint_type);

            let rhs_val = self.rhs.get(row_name).copied().unwrap_or(0.0);
            rhs_vec.push(rhs_val);
        }

        let base_num_constraints = row_map.len();

        // RANGES: expand interval constraints to Le + Ge pairs (IBM MPS convention).
        //   L: b - |r| <= Ax <= b
        //   G: b <= Ax <= b + |r|
        //   E (r>=0): b <= Ax <= b + |r|
        //   E (r<0):  b - |r| <= Ax <= b
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
                    _ => continue,
                };

                constraint_types[idx] = ConstraintType::Le;
                rhs_vec[idx] = upper;
                range_extra_rows.push((row_name.clone(), idx, lower));
            }
        }

        let mut range_row_map: HashMap<String, usize> = HashMap::new();
        for (row_name, _orig_idx, lower_bound) in &range_extra_rows {
            let new_idx = base_num_constraints + range_row_map.len();
            range_row_map.insert(row_name.clone(), new_idx);
            constraint_types.push(ConstraintType::Ge);
            rhs_vec.push(*lower_bound);
        }

        let num_constraints = base_num_constraints + range_row_map.len();

        let mut col_map = HashMap::new();
        for (col_name, _, _) in &self.columns {
            if !col_map.contains_key(col_name) {
                let idx = col_map.len();
                col_map.insert(col_name.clone(), idx);
            }
        }

        let num_vars = col_map.len();

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
        // Normalize MAX → MIN by negating the objective.
        if self.maximize {
            for v in &mut c {
                *v = -*v;
            }
        }

        // Extract objective constant (N-row RHS); sign-flip mirrors the MAX→MIN transform above.
        // `raw` is finite by construction: every value inserted into `self.rhs` is validated
        // as a finite f64 at parse time (see `parse_rhs_section`).
        let obj_offset = if let Some(obj_row_name) = &self.obj_row {
            let raw = self.rhs.get(obj_row_name.as_str()).copied().unwrap_or(0.0);
            if self.maximize {
                -raw
            } else {
                raw
            }
        } else {
            0.0
        };

        let mut triplets = Vec::new();
        for (col_name, row_name, value) in &self.columns {
            if Some(row_name) == self.obj_row.as_ref() {
                continue;
            }

            let col_idx = col_map
                .get(col_name)
                .ok_or_else(|| MpsError::UndefinedReference {
                    kind: "column".to_string(),
                    name: col_name.clone(),
                })?;
            let row_idx = row_map
                .get(row_name)
                .ok_or_else(|| MpsError::UndefinedReference {
                    kind: "row".to_string(),
                    name: row_name.clone(),
                })?;

            triplets.push((*row_idx, *col_idx, *value));

            if let Some(&range_row_idx) = range_row_map.get(row_name) {
                triplets.push((range_row_idx, *col_idx, *value));
            }
        }

        let rows: Vec<usize> = triplets.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();

        let a = CscMatrix::from_triplets(&rows, &cols, &vals, num_constraints, num_vars).map_err(
            |e| MpsError::ParseError {
                line: 0,
                message: format!("Failed to build matrix: {}", e),
            },
        )?;

        let mut bounds = vec![(0.0, f64::INFINITY); num_vars];
        for (bound_type, col_name, value) in &self.bounds {
            let col_idx = col_map
                .get(col_name)
                .ok_or_else(|| MpsError::UndefinedReference {
                    kind: "column".to_string(),
                    name: col_name.clone(),
                })?;

            match bound_type {
                BoundType::LO => bounds[*col_idx].0 = value.unwrap_or(0.0),
                BoundType::UP => bounds[*col_idx].1 = value.unwrap_or(f64::INFINITY),
                BoundType::FX => {
                    let val = value.unwrap_or(0.0);
                    bounds[*col_idx] = (val, val);
                }
                BoundType::FR => bounds[*col_idx] = (f64::NEG_INFINITY, f64::INFINITY),
                BoundType::MI => bounds[*col_idx].0 = f64::NEG_INFINITY,
                BoundType::BV => bounds[*col_idx] = (0.0, 1.0),
                BoundType::PL => bounds[*col_idx].1 = f64::INFINITY,
                BoundType::LI => bounds[*col_idx].0 = value.unwrap_or(0.0),
                BoundType::UI => bounds[*col_idx].1 = value.unwrap_or(f64::INFINITY),
            }
        }

        // Integer variables without any explicit BOUNDS entry default to binary [0,1]
        // (classical OSL/CPLEX/HiGHS convention).
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
        integer_vars.sort_unstable();

        let mut lp = LpProblem::new_general(
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
        lp.obj_offset = obj_offset;

        Ok((lp, integer_vars))
    }
}

// ── Public entry points (used by mps/mod.rs) ─────────────────────────────────

pub fn parse_mps_reader<R: BufRead>(reader: R) -> Result<LpProblem, MpsError> {
    let mut parser = MpsParser::new();
    let (lp, _integer_vars) = parser.parse_reader(reader)?;
    Ok(lp)
}

pub fn parse_milp_reader<R: BufRead>(reader: R) -> Result<MilpProblem, MpsError> {
    let mut parser = MpsParser::new();
    let (lp, integer_vars) = parser.parse_reader(reader)?;
    MilpProblem::new(lp, integer_vars).map_err(|e| MpsError::ParseError {
        line: 0,
        message: e.to_string(),
    })
}
