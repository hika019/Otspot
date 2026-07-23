use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Seek};

pub use otspot_core::error::MpsError;
use otspot_core::mip::MilpProblem;
use otspot_core::problem::{ConstraintType, LpProblem};
use otspot_core::sparse::CscMatrix;

use super::types::{
    integer_marker_kind, BoundType, IntegerMarker, Section, INTEGER_DEFAULT_UPPER_BINARY,
};
use crate::common::{
    parse_bounds_entry, parse_columns_entry, parse_objsense_value, parse_row_decl,
    parse_vector_entry, parse_with_format_fallback, Format, LineSource, ReaderSource, RowNameIndex,
    RowType, SectionState, VectorSectionState,
};

pub(super) struct MpsParser {
    format: Format,
    /// Line reached by this reading; reported alongside a failure so the caller
    /// can tell which of the two format readings engaged with more of the file.
    progress: usize,
    problem_name: Option<String>,
    rows: Vec<(String, RowType)>,
    row_names: HashSet<String>,
    columns: Vec<(String, String, f64)>,
    rhs: HashMap<String, f64>,
    ranges: HashMap<String, f64>,
    rhs_vectors: VectorSectionState,
    ranges_vectors: VectorSectionState,
    row_index: RowNameIndex,
    bounds: Vec<(BoundType, String, Option<f64>)>,
    obj_row: Option<String>,
    integer_cols: HashSet<String>,
    in_integer_marker: bool,
    maximize: bool,
}

impl MpsParser {
    fn new(format: Format) -> Self {
        Self {
            format,
            progress: 0,
            problem_name: None,
            rows: Vec::new(),
            row_names: HashSet::new(),
            columns: Vec::new(),
            rhs: HashMap::new(),
            ranges: HashMap::new(),
            rhs_vectors: VectorSectionState::new(),
            ranges_vectors: VectorSectionState::new(),
            row_index: RowNameIndex::new(),
            bounds: Vec::new(),
            obj_row: None,
            integer_cols: HashSet::new(),
            in_integer_marker: false,
            maximize: false,
        }
    }

    /// Reads an MPS source under one fixed layout; returns the LP relaxation
    /// and the integer variable indices.
    fn run<S: LineSource>(&mut self, source: &S) -> Result<(LpProblem, Vec<usize>), MpsError> {
        let mut state = SectionState::new(Section::None);

        source.visit_lines(MpsError::IoError, |line_num, line| {
            self.progress = line_num;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('*') {
                return Ok(true);
            }

            if !line.starts_with(' ') && !line.starts_with('\t') {
                if let Some(section) = Section::from_line(trimmed) {
                    if self.in_integer_marker && section != Section::Columns {
                        return Err(MpsError::UnclosedIntegerMarker);
                    }
                    let should_stop = state.advance(
                        section,
                        Section::Name,
                        Section::EndData,
                        MpsError::DuplicateSection,
                    )?;
                    self.parse_section_header(section, trimmed, line_num)?;
                    return Ok(!should_stop);
                }
            }

            let tokens: Vec<&str> = line.split_whitespace().collect();
            match state.current {
                Section::None => {
                    return Err(MpsError::ParseError {
                        line: line_num,
                        message: "Line appears before any section header".to_string(),
                    })
                }
                Section::Name => {}
                Section::ObjSense => self.parse_objsense_line(line, line_num)?,
                Section::Rows => self.parse_rows_line(line, &tokens, line_num)?,
                Section::Columns => self.parse_columns_line(line, &tokens, line_num)?,
                Section::Rhs => self.parse_vector_line(line, &tokens, line_num, "RHS")?,
                Section::Ranges => self.parse_vector_line(line, &tokens, line_num, "RANGES")?,
                Section::Bounds => self.parse_bounds_line(line, &tokens, line_num)?,
                Section::EndData => return Ok(false),
            }
            Ok(true)
        })?;

        state.require(
            &[
                (Section::EndData, "ENDATA"),
                (Section::Rows, "ROWS"),
                (Section::Columns, "COLUMNS"),
            ],
            MpsError::MissingSection,
        )?;

        self.validate_references()?;
        self.build_lp_problem()
    }

    /// Handles data carried on the section header line itself.
    ///
    /// `NAME` puts the problem name there, and `OBJSENSE` may put its value
    /// there (`OBJSENSE  MAX`) instead of on the following line — a legal form
    /// that both HiGHS and SCIP accept. Ignoring it would silently minimize a
    /// problem that asked to be maximized.
    fn parse_section_header(
        &mut self,
        section: Section,
        trimmed: &str,
        line_num: usize,
    ) -> Result<(), MpsError> {
        let keyword_len = match section {
            Section::Name => "NAME".len(),
            Section::ObjSense => "OBJSENSE".len(),
            _ => return Ok(()),
        };
        let Some(rest) = trimmed.get(keyword_len..).map(str::trim) else {
            return Ok(());
        };
        if rest.is_empty() {
            return Ok(());
        }
        match section {
            Section::Name => self.problem_name = Some(rest.to_string()),
            Section::ObjSense => self.parse_objsense_line(rest, line_num)?,
            _ => {}
        }
        Ok(())
    }

    fn parse_objsense_line(&mut self, value: &str, line_num: usize) -> Result<(), MpsError> {
        self.maximize = parse_objsense_value(value).map_err(|message| MpsError::ParseError {
            line: line_num,
            message,
        })?;
        Ok(())
    }

    fn parse_rows_line(
        &mut self,
        line: &str,
        tokens: &[&str],
        line_num: usize,
    ) -> Result<(), MpsError> {
        let (type_str, row_name) =
            parse_row_decl(line, tokens, self.format, line_num).map_err(|message| {
                MpsError::ParseError {
                    line: line_num,
                    message,
                }
            })?;

        if type_str.len() != 1 {
            return Err(MpsError::ParseError {
                line: line_num,
                message: format!("Row type must be single character, got '{}'", type_str),
            });
        }
        let type_char = type_str.chars().next().expect("len == 1 checked above");
        let row_type = match type_char.to_ascii_uppercase() {
            'N' => RowType::N,
            'L' => RowType::L,
            'G' => RowType::G,
            'E' => RowType::E,
            _ => return Err(MpsError::InvalidRowType(type_char)),
        };

        // Two rows sharing a name would make every reference to it ambiguous and
        // silently resolve to whichever landed in the index last.
        if !self.row_names.insert(row_name.clone()) {
            return Err(MpsError::ParseError {
                line: line_num,
                message: format!("ROWS: duplicate row name '{}'", row_name),
            });
        }
        if matches!(row_type, RowType::N) && self.obj_row.is_none() {
            self.obj_row = Some(row_name.clone());
        }
        self.rows.push((row_name, row_type));
        Ok(())
    }

    fn parse_columns_line(
        &mut self,
        line: &str,
        tokens: &[&str],
        line_num: usize,
    ) -> Result<(), MpsError> {
        if let Some(kind) = integer_marker_kind(line) {
            self.in_integer_marker = matches!(kind, IntegerMarker::Start);
            return Ok(());
        }

        let (col_name, pairs) =
            parse_columns_entry(line, tokens, self.format, line_num).map_err(|message| {
                MpsError::ParseError {
                    line: line_num,
                    message,
                }
            })?;

        if self.in_integer_marker {
            self.integer_cols.insert(col_name.clone());
        }
        for (row_name, value) in pairs {
            self.columns.push((col_name.clone(), row_name, value));
        }
        Ok(())
    }

    /// RHS and RANGES share one grammar; they differ only in the map they fill.
    ///
    /// A non-finite value is rejected for every row including the N-row: the
    /// objective offset (N-row RHS) is extracted in `build_lp_problem` once all
    /// sections are parsed.
    fn parse_vector_line(
        &mut self,
        line: &str,
        tokens: &[&str],
        line_num: usize,
        section: &str,
    ) -> Result<(), MpsError> {
        let (vector_name, pairs) = parse_vector_entry(
            line,
            tokens,
            self.format,
            &self.rows,
            &mut self.row_index,
            line_num,
            section,
            None,
        )
        .map_err(|message| MpsError::ParseError {
            line: line_num,
            message,
        })?;

        let (state, target) = if section == "RHS" {
            (&mut self.rhs_vectors, &mut self.rhs)
        } else {
            (&mut self.ranges_vectors, &mut self.ranges)
        };
        for (row_name, value) in pairs {
            state
                .record(target, section, vector_name.as_deref(), row_name, value)
                .map_err(|message| MpsError::ParseError {
                    line: line_num,
                    message,
                })?;
        }
        Ok(())
    }

    fn parse_bounds_line(
        &mut self,
        line: &str,
        tokens: &[&str],
        line_num: usize,
    ) -> Result<(), MpsError> {
        if tokens.is_empty() {
            return Err(MpsError::ParseError {
                line: line_num,
                message: "BOUNDS line is empty".to_string(),
            });
        }
        let bound_type = match tokens[0].to_uppercase().as_str() {
            "LO" => BoundType::LO,
            "UP" => BoundType::UP,
            "FX" => BoundType::FX,
            "FR" => BoundType::FR,
            "MI" => BoundType::MI,
            "BV" => BoundType::BV,
            "PL" => BoundType::PL,
            "LI" => BoundType::LI,
            "UI" => BoundType::UI,
            _ => return Err(MpsError::InvalidBoundType(tokens[0].to_string())),
        };
        let value_required = matches!(
            bound_type,
            BoundType::LO | BoundType::UP | BoundType::FX | BoundType::LI | BoundType::UI
        );

        let (col_name, value) =
            parse_bounds_entry(line, tokens, self.format, line_num, value_required).map_err(
                |message| MpsError::ParseError {
                    line: line_num,
                    message,
                },
            )?;

        if matches!(bound_type, BoundType::BV | BoundType::LI | BoundType::UI) {
            self.integer_cols.insert(col_name.clone());
        }
        self.bounds.push((bound_type, col_name, value));
        Ok(())
    }

    /// Every row/column name referenced by COLUMNS / RHS / RANGES / BOUNDS must
    /// have been declared. Dropping an unknown name silently is how a misread
    /// layout used to go unnoticed: reading a fixed-column line as free format
    /// invents names that match nothing, and discarding them quietly corrupts
    /// the model. Erroring out is also what lets the caller detect a wrong
    /// layout guess and retry the file as fixed-column.
    fn validate_references(&self) -> Result<(), MpsError> {
        let declared_cols: HashSet<&str> =
            self.columns.iter().map(|(c, _, _)| c.as_str()).collect();

        for row_name in self
            .columns
            .iter()
            .map(|(_, row, _)| row.as_str())
            .chain(self.rhs.keys().map(String::as_str))
            .chain(self.ranges.keys().map(String::as_str))
            .chain(self.rhs_vectors.referenced_rows())
            .chain(self.ranges_vectors.referenced_rows())
        {
            if !self.row_names.contains(row_name) {
                return Err(MpsError::UndefinedReference {
                    kind: "row".to_string(),
                    name: row_name.to_string(),
                });
            }
        }
        for (_, col_name, _) in &self.bounds {
            if !declared_cols.contains(col_name.as_str()) {
                return Err(MpsError::UndefinedReference {
                    kind: "column".to_string(),
                    name: col_name.clone(),
                });
            }
        }
        Ok(())
    }

    fn build_lp_problem(&self) -> Result<(LpProblem, Vec<usize>), MpsError> {
        // Only constraint rows get an index. N rows (the objective and any
        // further free row) are declared but never become constraints, so
        // giving them a `row_map` slot would desynchronise the row indices from
        // `constraint_types` / `rhs_vec`.
        let mut row_map: HashMap<&str, usize> = HashMap::new();
        let mut constraint_types = Vec::new();
        let mut rhs_vec = Vec::new();
        for (row_name, row_type) in &self.rows {
            if !row_type.is_constraint() {
                continue;
            }
            row_map.insert(row_name.as_str(), row_map.len());
            constraint_types.push(match row_type {
                RowType::L => ConstraintType::Le,
                RowType::G => ConstraintType::Ge,
                RowType::E => ConstraintType::Eq,
                RowType::N => unreachable!("N rows are skipped above"),
            });
            rhs_vec.push(self.rhs.get(row_name).copied().unwrap_or(0.0));
        }
        let base_num_constraints = row_map.len();

        // RANGES: expand interval constraints to Le + Ge pairs (IBM convention).
        //   L: b - |r| <= Ax <= b
        //   G: b <= Ax <= b + |r|
        //   E (r>=0): b <= Ax <= b + |r|
        //   E (r<0):  b - |r| <= Ax <= b
        //
        // Walk the rows in declaration order, not `self.ranges` (a HashMap):
        // iterating the map would order the appended rows by hash, making the
        // constraint indices — and therefore the built matrix — differ between
        // runs of the same input.
        let mut range_row_map: HashMap<&str, usize> = HashMap::new();
        let mut range_lowers: Vec<f64> = Vec::new();
        for (row_name, _) in &self.rows {
            let Some(&range_val) = self.ranges.get(row_name) else {
                continue;
            };
            // A RANGES entry on a declared free (N) row has no constraint to
            // widen; the standard ignores it.
            let Some(&idx) = row_map.get(row_name.as_str()) else {
                continue;
            };
            let b = rhs_vec[idx];
            let abs_r = range_val.abs();
            let (lower, upper) = match constraint_types[idx] {
                ConstraintType::Le => (b - abs_r, b),
                ConstraintType::Ge => (b, b + abs_r),
                ConstraintType::Eq => {
                    if range_val >= 0.0 {
                        (b, b + abs_r)
                    } else {
                        (b - abs_r, b)
                    }
                }
                _ => continue,
            };
            constraint_types[idx] = ConstraintType::Le;
            rhs_vec[idx] = upper;
            range_row_map.insert(row_name.as_str(), base_num_constraints + range_lowers.len());
            range_lowers.push(lower);
        }
        for lower in &range_lowers {
            constraint_types.push(ConstraintType::Ge);
            rhs_vec.push(*lower);
        }
        let num_constraints = base_num_constraints + range_lowers.len();

        let mut col_map: HashMap<&str, usize> = HashMap::new();
        for (col_name, _, _) in &self.columns {
            let next = col_map.len();
            col_map.entry(col_name.as_str()).or_insert(next);
        }
        let num_vars = col_map.len();

        let obj_row = self.obj_row.as_deref();
        let mut c = vec![0.0; num_vars];
        let mut triplets = Vec::new();
        for (col_name, row_name, value) in &self.columns {
            let col_idx = col_map[col_name.as_str()];
            if Some(row_name.as_str()) == obj_row {
                c[col_idx] += *value;
                continue;
            }
            // Declared free (N) rows other than the objective carry no
            // constraint; the standard ignores their coefficients.
            let Some(&row_idx) = row_map.get(row_name.as_str()) else {
                continue;
            };
            triplets.push((row_idx, col_idx, *value));
            if let Some(&range_row_idx) = range_row_map.get(row_name.as_str()) {
                triplets.push((range_row_idx, col_idx, *value));
            }
        }
        // Normalize MAX → MIN by negating the objective.
        if self.maximize {
            for v in &mut c {
                *v = -*v;
            }
        }

        // Objective constant (N-row RHS); the sign flip mirrors MAX → MIN above.
        // Finiteness is guaranteed at parse time.
        let obj_offset = match obj_row {
            Some(name) => {
                let raw = self.rhs.get(name).copied().unwrap_or(0.0);
                if self.maximize {
                    -raw
                } else {
                    raw
                }
            }
            None => 0.0,
        };

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
            let col_idx = col_map[col_name.as_str()];
            // LO/UP/FX/LI/UI are value_required at parse time
            // (`parse_bounds_entry`), so `value` is guaranteed `Some` here. The
            // `ok_or_else` is defense against that invariant regressing: it must
            // surface as an error, not silently fabricate 0.0/INFINITY.
            let require_value = || -> Result<f64, MpsError> {
                value.ok_or_else(|| MpsError::ParseError {
                    line: 0,
                    message: format!(
                        "BOUNDS: {:?} entry for column '{}' requires a value but none was \
                         recorded",
                        bound_type, col_name
                    ),
                })
            };
            match bound_type {
                BoundType::LO => bounds[col_idx].0 = require_value()?,
                BoundType::UP => bounds[col_idx].1 = require_value()?,
                BoundType::FX => {
                    let val = require_value()?;
                    bounds[col_idx] = (val, val);
                }
                BoundType::FR => bounds[col_idx] = (f64::NEG_INFINITY, f64::INFINITY),
                BoundType::MI => bounds[col_idx].0 = f64::NEG_INFINITY,
                BoundType::BV => bounds[col_idx] = (0.0, 1.0),
                BoundType::PL => bounds[col_idx].1 = f64::INFINITY,
                BoundType::LI => bounds[col_idx].0 = require_value()?,
                BoundType::UI => bounds[col_idx].1 = require_value()?,
            }
        }

        // Integer variables with no explicit BOUNDS entry default to binary
        // [0, 1] (classical OSL/CPLEX/HiGHS convention).
        let explicitly_bounded: HashSet<&str> = self
            .bounds
            .iter()
            .map(|(_, name, _)| name.as_str())
            .collect();
        let mut integer_vars: Vec<usize> = Vec::with_capacity(self.integer_cols.len());
        for col_name in &self.integer_cols {
            if let Some(&col_idx) = col_map.get(col_name.as_str()) {
                if !explicitly_bounded.contains(col_name.as_str()) {
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

/// Parse `source`, retrying as fixed-column MPS if the free-format read fails.
fn parse_source<S: LineSource>(source: &S) -> Result<(LpProblem, Vec<usize>), MpsError> {
    parse_with_format_fallback(source, |source, format| {
        let mut parser = MpsParser::new(format);
        parser.run(source).map_err(|e| (e, parser.progress))
    })
}

pub(super) fn parse_lp_source<S: LineSource>(source: &S) -> Result<LpProblem, MpsError> {
    let (lp, _integer_vars) = parse_source(source)?;
    Ok(lp)
}

pub(super) fn parse_milp_source<S: LineSource>(source: &S) -> Result<MilpProblem, MpsError> {
    let (lp, integer_vars) = parse_source(source)?;
    MilpProblem::new(lp, integer_vars).map_err(|e| MpsError::ParseError {
        line: 0,
        message: e.to_string(),
    })
}

// ── Public entry points (used by mps/mod.rs) ─────────────────────────────────

/// Parse an MPS stream, returning an LP relaxation.
///
/// The reader must be seekable: a file that turns out to be fixed-column is
/// re-read from the start, and rewinding is what lets that happen without
/// holding the input in memory.
pub fn parse_mps_reader<R: BufRead + Seek>(reader: R) -> Result<LpProblem, MpsError> {
    parse_lp_source(&ReaderSource::new(reader))
}

/// Parse an MPS stream, returning a `MilpProblem`. See [`parse_mps_reader`].
pub fn parse_milp_reader<R: BufRead + Seek>(reader: R) -> Result<MilpProblem, MpsError> {
    parse_milp_source(&ReaderSource::new(reader))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_bounds_entry` guarantees `value_required` bound types (LO, UP,
    /// FX, LI, UI) never carry `None` by the time they reach `self.bounds`.
    /// This bypasses that guard by constructing the parser's internal state
    /// directly, to prove `build_lp_problem` treats a violation of the
    /// invariant as a hard error instead of fabricating a bound.
    fn parser_with_dangling_bound(bound_type: BoundType) -> MpsParser {
        let mut parser = MpsParser::new(Format::Free);
        parser
            .columns
            .push(("x".to_string(), "obj".to_string(), 1.0));
        parser.bounds.push((bound_type, "x".to_string(), None));
        parser
    }

    #[test]
    fn build_lp_problem_rejects_missing_value_for_lo_bound() {
        let parser = parser_with_dangling_bound(BoundType::LO);
        let err = parser.build_lp_problem().unwrap_err();
        assert!(matches!(err, MpsError::ParseError { .. }), "{err:?}");
    }

    #[test]
    fn build_lp_problem_rejects_missing_value_for_up_bound() {
        let parser = parser_with_dangling_bound(BoundType::UP);
        let err = parser.build_lp_problem().unwrap_err();
        assert!(matches!(err, MpsError::ParseError { .. }), "{err:?}");
    }

    #[test]
    fn build_lp_problem_rejects_missing_value_for_fx_bound() {
        let parser = parser_with_dangling_bound(BoundType::FX);
        let err = parser.build_lp_problem().unwrap_err();
        assert!(matches!(err, MpsError::ParseError { .. }), "{err:?}");
    }

    #[test]
    fn build_lp_problem_rejects_missing_value_for_li_bound() {
        let parser = parser_with_dangling_bound(BoundType::LI);
        let err = parser.build_lp_problem().unwrap_err();
        assert!(matches!(err, MpsError::ParseError { .. }), "{err:?}");
    }

    #[test]
    fn build_lp_problem_rejects_missing_value_for_ui_bound() {
        let parser = parser_with_dangling_bound(BoundType::UI);
        let err = parser.build_lp_problem().unwrap_err();
        assert!(matches!(err, MpsError::ParseError { .. }), "{err:?}");
    }

    /// Valueless bound types must still succeed with `value = None`; this is
    /// their normal, well-formed state, not a violated invariant.
    #[test]
    fn build_lp_problem_accepts_valueless_bound_types() {
        for bound_type in [BoundType::FR, BoundType::MI, BoundType::BV, BoundType::PL] {
            let parser = parser_with_dangling_bound(bound_type);
            assert!(
                parser.build_lp_problem().is_ok(),
                "valueless bound type {bound_type:?} must not require a value"
            );
        }
    }
}
