//! MPS file format parser for Linear Programming problems
//!
//! Supports both fixed-width and free-format MPS files.

use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;
use std::collections::HashMap;
use std::path::Path;

/// Errors that can occur during MPS parsing
#[derive(Debug)]
pub enum MpsError {
    /// I/O error while reading file
    IoError(std::io::Error),
    /// Parse error on a specific line
    ParseError { line: usize, message: String },
    /// Missing required section
    MissingSection(String),
    /// Duplicate section encountered
    DuplicateSection(String),
    /// Invalid row type
    InvalidRowType(char),
    /// Invalid bound type
    InvalidBoundType(String),
    /// Reference to undefined row or column
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

/// Parse an MPS file from a file path
pub fn parse_mps_file(path: &Path) -> Result<LpProblem, MpsError> {
    let content = std::fs::read_to_string(path)?;
    parse_mps(&content)
}

/// Parse an MPS format string into an LpProblem
pub fn parse_mps(input: &str) -> Result<LpProblem, MpsError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut parser = MpsParser::new();
    parser.parse(&lines)
}

struct MpsParser {
    problem_name: Option<String>,
    rows: Vec<(String, RowType)>,      // (row_name, row_type)
    columns: Vec<(String, String, f64)>, // (col_name, row_name, value)
    rhs: HashMap<String, f64>,           // row_name -> rhs_value
    ranges: HashMap<String, f64>,        // row_name -> range_value
    bounds: Vec<(BoundType, String, Option<f64>)>, // (bound_type, col_name, value)
    obj_row: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum RowType {
    N,  // Objective (free)
    L,  // <=
    G,  // >=
    E,  // ==
}

#[derive(Debug, Clone, Copy)]
enum BoundType {
    LO, // Lower bound
    UP, // Upper bound
    FX, // Fixed value
    FR, // Free variable (-inf, +inf)
    MI, // Lower bound = -inf
    BV, // Binary (0 or 1)
}

impl MpsParser {
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

    fn parse(&mut self, lines: &[&str]) -> Result<LpProblem, MpsError> {
        let mut current_section = Section::None;
        let mut seen_sections = std::collections::HashSet::new();

        for (line_idx, line) in lines.iter().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with('*') {
                continue;
            }

            // Check for section headers (must start at column 1, no leading whitespace)
            // Data rows have leading whitespace, so they won't match
            if !line.starts_with(' ') && !line.starts_with('\t') {
                if let Some(section) = Section::from_line(trimmed) {
                    // Check for duplicate sections (except Name and EndData which can't duplicate)
                    if section != Section::Name && section != Section::EndData {
                        if seen_sections.contains(&section) {
                            return Err(MpsError::DuplicateSection(format!("{:?}", section)));
                        }
                    }
                    // Always insert into seen_sections for existence checking
                    seen_sections.insert(section);
                    current_section = section;

                    // Special handling for NAME section: extract name from same line
                    if section == Section::Name {
                        // NAME line format: "NAME          problem_name"
                        if trimmed.len() > 4 {
                            let name_part = trimmed[4..].trim();
                            if !name_part.is_empty() {
                                self.problem_name = Some(name_part.to_string());
                            }
                        }
                    }
                    continue;
                }
            }

            // Process line based on current section
            match current_section {
                Section::None => {
                    return Err(MpsError::ParseError {
                        line: line_num,
                        message: "Line appears before any section header".to_string(),
                    });
                }
                Section::Name => {
                    // NAME is already handled above when section header is detected
                }
                Section::Rows => self.parse_rows_line(line, line_num)?,
                Section::Columns => self.parse_columns_line(line, line_num)?,
                Section::Rhs => self.parse_rhs_line(line, line_num)?,
                Section::Ranges => self.parse_ranges_line(line, line_num)?,
                Section::Bounds => self.parse_bounds_line(line, line_num)?,
                Section::EndData => break,
            }
        }

        // Check for ENDATA
        if !seen_sections.contains(&Section::EndData) {
            return Err(MpsError::MissingSection("ENDATA".to_string()));
        }

        // Check for required sections
        if !seen_sections.contains(&Section::Rows) {
            return Err(MpsError::MissingSection("ROWS".to_string()));
        }
        if !seen_sections.contains(&Section::Columns) {
            return Err(MpsError::MissingSection("COLUMNS".to_string()));
        }

        self.build_lp_problem()
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
        // Detect format: if column 14 (0-indexed 13) is whitespace, it's fixed-width
        let is_fixed_width = line.len() > 14 && line.chars().nth(14).map_or(false, |c| c.is_whitespace());

        if is_fixed_width {
            self.parse_columns_fixed(line, line_num)
        } else {
            self.parse_columns_free(line, line_num)
        }
    }

    fn parse_columns_fixed(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        // Fixed format: col5-12=col_name, col15-22=row_name, col25-36=value
        //                [col40-47=row_name2, col50-61=value2]

        if line.len() < 25 {
            return Ok(()); // Skip if too short
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

        // Second entry (optional)
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

    fn parse_columns_free(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(()); // Skip incomplete lines
        }

        let col_name = parts[0].to_string();

        // Process pairs: (row_name, value)
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

    fn parse_rhs_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        // Similar format to COLUMNS
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }

        // Skip the RHS name (parts[0]), process pairs
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

    fn parse_ranges_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        // Same format as RHS
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

    fn parse_bounds_line(&mut self, line: &str, line_num: usize) -> Result<(), MpsError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Ok(());
        }

        let bound_type_str = parts[0];
        let _bound_name = parts[1]; // Usually ignored
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

    fn build_lp_problem(&self) -> Result<LpProblem, MpsError> {
        // Build row name -> index map (excluding objective)
        let mut row_map = HashMap::new();
        let mut constraint_types = Vec::new();
        let mut rhs_vec = Vec::new();

        for (row_name, row_type) in &self.rows {
            if Some(row_name) == self.obj_row.as_ref() {
                continue; // Skip objective row
            }

            let idx = row_map.len();
            row_map.insert(row_name.clone(), idx);

            let constraint_type = match row_type {
                RowType::L => ConstraintType::Le,
                RowType::G => ConstraintType::Ge,
                RowType::E => ConstraintType::Eq,
                RowType::N => continue, // Skip other N rows
            };
            constraint_types.push(constraint_type);

            let rhs_val = self.rhs.get(row_name).copied().unwrap_or(0.0);
            rhs_vec.push(rhs_val);
        }

        let num_constraints = row_map.len();

        // Apply RANGES
        for (row_name, range_val) in &self.ranges {
            if let Some(&idx) = row_map.get(row_name) {
                // RANGE interpretation: creates an interval
                // For L constraint: b <= Ax <= b + range
                // For G constraint: b - range <= Ax <= b
                // For E constraint: treat as L constraint with upper bound b + |range|
                match constraint_types[idx] {
                    ConstraintType::Le => {
                        // Keep as Le, rhs stays the same
                        // Upper bound would be rhs + range (not implemented in simplex yet)
                    }
                    ConstraintType::Ge => {
                        // Adjust rhs to b - range
                        rhs_vec[idx] -= range_val.abs();
                    }
                    ConstraintType::Eq => {
                        // Convert to Le with upper bound
                        constraint_types[idx] = ConstraintType::Le;
                        rhs_vec[idx] += range_val.abs();
                    }
                }
            }
        }

        // Build column name -> index map
        let mut col_map = HashMap::new();
        for (col_name, _, _) in &self.columns {
            if !col_map.contains_key(col_name) {
                let idx = col_map.len();
                col_map.insert(col_name.clone(), idx);
            }
        }

        let num_vars = col_map.len();

        // Build objective vector c
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

        // Build constraint matrix A
        let mut triplets = Vec::new();
        for (col_name, row_name, value) in &self.columns {
            // Skip objective row
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
        }

        let rows: Vec<usize> = triplets.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = triplets.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = triplets.iter().map(|&(_, _, v)| v).collect();

        let a = CscMatrix::from_triplets(&rows, &cols, &vals, num_constraints, num_vars)
            .map_err(|e| MpsError::ParseError {
                line: 0,
                message: format!("Failed to build matrix: {}", e),
            })?;

        // Build bounds vector
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
            message: e,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Section {
    None,
    Name,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
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
        assert_eq!(lp.num_constraints, 1);
        // Range doesn't change rhs for L constraints in this implementation
        assert_eq!(lp.b, vec![10.0]);
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
        // x1 has coefficient 2.0 in c1
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
}
