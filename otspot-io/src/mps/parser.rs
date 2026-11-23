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
    objective_offset_from_rhs, parse_bounds_entry, parse_columns_entry, parse_objsense_value,
    parse_row_decl, parse_vector_entry, parse_with_format_fallback, Format, LineSource,
    NameRegistry, ReaderSource, RowType, SectionState, VectorSectionState, NO_INDEX,
};

/// Per-entry state uses interned name ids, not `String`s: `columns` holds
/// (column id, row id, value) fixed-size records and `bounds` likewise, so
/// parse-time residency per sparse entry no longer depends on how long the
/// referenced names are (the QPS parser mirrors this).
pub(super) struct MpsParser {
    format: Format,
    /// Line reached by this reading; reported alongside a failure so the caller
    /// can tell which of the two format readings engaged with more of the file.
    progress: usize,
    problem_name: Option<String>,
    rows: Vec<(u32, RowType)>,
    row_names: NameRegistry,
    col_names: NameRegistry,
    columns: Vec<(u32, u32, f64)>,
    rhs: HashMap<u32, f64>,
    ranges: HashMap<u32, f64>,
    rhs_vectors: VectorSectionState,
    ranges_vectors: VectorSectionState,
    bounds: Vec<(BoundType, u32, Option<f64>)>,
    obj_row: Option<u32>,
    integer_cols: HashSet<u32>,
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
            row_names: NameRegistry::new(),
            col_names: NameRegistry::new(),
            columns: Vec::new(),
            rhs: HashMap::new(),
            ranges: HashMap::new(),
            rhs_vectors: VectorSectionState::new(),
            ranges_vectors: VectorSectionState::new(),
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
        let (row_id, already_declared) = self.row_names.intern_declared(&row_name);
        if already_declared {
            return Err(MpsError::ParseError {
                line: line_num,
                message: format!("ROWS: duplicate row name '{}'", row_name),
            });
        }
        if matches!(row_type, RowType::N) && self.obj_row.is_none() {
            self.obj_row = Some(row_id);
        }
        self.rows.push((row_id, row_type));
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

        // A COLUMNS line always carries at least one (row, value) pair, so
        // marking the column declared here matches the old "declared = appears
        // in `columns`" set exactly.
        let col_id = self.col_names.intern_declared(&col_name).0;
        if self.in_integer_marker {
            self.integer_cols.insert(col_id);
        }
        for (row_name, value) in pairs {
            let row_id = self.row_names.intern_ref(&row_name);
            self.columns.push((col_id, row_id, value));
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
            &|name| self.row_names.is_declared_name(name),
            line_num,
            section,
            None,
        )
        .map_err(|message| MpsError::ParseError {
            line: line_num,
            message,
        })?;

        for (row_name, value) in pairs {
            let row_id = self.row_names.intern_ref(&row_name);
            let (state, target) = if section == "RHS" {
                (&mut self.rhs_vectors, &mut self.rhs)
            } else {
                (&mut self.ranges_vectors, &mut self.ranges)
            };
            state
                .record(
                    target,
                    section,
                    vector_name.as_deref(),
                    row_id,
                    &row_name,
                    value,
                )
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

        let col_id = self.col_names.intern_ref(&col_name);
        if matches!(bound_type, BoundType::BV | BoundType::LI | BoundType::UI) {
            self.integer_cols.insert(col_id);
        }
        self.bounds.push((bound_type, col_id, value));
        Ok(())
    }

    /// Every row/column name referenced by COLUMNS / RHS / RANGES / BOUNDS must
    /// have been declared. Dropping an unknown name silently is how a misread
    /// layout used to go unnoticed: reading a fixed-column line as free format
    /// invents names that match nothing, and discarding them quietly corrupts
    /// the model. Erroring out is also what lets the caller detect a wrong
    /// layout guess and retry the file as fixed-column.
    fn validate_references(&self) -> Result<(), MpsError> {
        for row_id in self
            .columns
            .iter()
            .map(|&(_, row, _)| row)
            .chain(self.rhs.keys().copied())
            .chain(self.ranges.keys().copied())
            .chain(self.rhs_vectors.referenced_rows())
            .chain(self.ranges_vectors.referenced_rows())
        {
            if !self.row_names.is_declared(row_id) {
                return Err(MpsError::UndefinedReference {
                    kind: "row".to_string(),
                    name: self.row_names.name(row_id).to_string(),
                });
            }
        }
        for &(_, col, _) in &self.bounds {
            if !self.col_names.is_declared(col) {
                return Err(MpsError::UndefinedReference {
                    kind: "column".to_string(),
                    name: self.col_names.name(col).to_string(),
                });
            }
        }
        Ok(())
    }

    fn build_lp_problem(&self) -> Result<(LpProblem, Vec<usize>), MpsError> {
        // Only constraint rows get an index. N rows (the objective and any
        // further free row) are declared but never become constraints, so
        // giving them a `row_index` slot would desynchronise the row indices
        // from `constraint_types` / `rhs_vec`.
        let mut row_index = vec![NO_INDEX; self.row_names.len()];
        let mut constraint_types = Vec::new();
        let mut rhs_vec = Vec::new();
        for &(row_id, row_type) in &self.rows {
            if !row_type.is_constraint() {
                continue;
            }
            row_index[row_id as usize] = constraint_types.len() as u32;
            constraint_types.push(match row_type {
                RowType::L => ConstraintType::Le,
                RowType::G => ConstraintType::Ge,
                RowType::E => ConstraintType::Eq,
                RowType::N => unreachable!("N rows are skipped above"),
            });
            rhs_vec.push(self.rhs.get(&row_id).copied().unwrap_or(0.0));
        }
        let base_num_constraints = constraint_types.len();

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
        let mut range_row_index = vec![NO_INDEX; self.row_names.len()];
        let mut range_lowers: Vec<f64> = Vec::new();
        for &(row_id, _) in &self.rows {
            let Some(&range_val) = self.ranges.get(&row_id) else {
                continue;
            };
            // A RANGES entry on a declared free (N) row has no constraint to
            // widen; the standard ignores it.
            let base_idx = row_index[row_id as usize];
            if base_idx == NO_INDEX {
                continue;
            }
            let base_idx = base_idx as usize;
            let b = rhs_vec[base_idx];
            let abs_r = range_val.abs();
            let (lower, upper) = match constraint_types[base_idx] {
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
            constraint_types[base_idx] = ConstraintType::Le;
            rhs_vec[base_idx] = upper;
            range_row_index[row_id as usize] = (base_num_constraints + range_lowers.len()) as u32;
            range_lowers.push(lower);
        }
        for lower in &range_lowers {
            constraint_types.push(ConstraintType::Ge);
            rhs_vec.push(*lower);
        }
        let num_constraints = base_num_constraints + range_lowers.len();

        // Compact column indices in first-appearance order — the numbering the
        // name-keyed `col_map` used to produce.
        let mut col_index = vec![NO_INDEX; self.col_names.len()];
        let mut num_vars = 0usize;
        for &(col, _, _) in &self.columns {
            if col_index[col as usize] == NO_INDEX {
                col_index[col as usize] = num_vars as u32;
                num_vars += 1;
            }
        }

        let obj_row = self.obj_row;
        let mut c = vec![0.0; num_vars];
        let mut a_rows: Vec<usize> = Vec::with_capacity(self.columns.len());
        let mut a_cols: Vec<usize> = Vec::with_capacity(self.columns.len());
        let mut a_vals: Vec<f64> = Vec::with_capacity(self.columns.len());
        for &(col, row, value) in &self.columns {
            let col_idx = col_index[col as usize] as usize;
            if Some(row) == obj_row {
                c[col_idx] += value;
                continue;
            }
            // Declared free (N) rows other than the objective carry no
            // constraint; the standard ignores their coefficients.
            if row_index[row as usize] == NO_INDEX {
                continue;
            }
            a_rows.push(row_index[row as usize] as usize);
            a_cols.push(col_idx);
            a_vals.push(value);
            if range_row_index[row as usize] != NO_INDEX {
                a_rows.push(range_row_index[row as usize] as usize);
                a_cols.push(col_idx);
                a_vals.push(value);
            }
        }
        // Normalize MAX → MIN by negating the objective.
        if self.maximize {
            for v in &mut c {
                *v = -*v;
            }
        }

        // Finiteness of the objective-row RHS is guaranteed at parse time.
        let obj_offset = match obj_row {
            Some(id) => {
                objective_offset_from_rhs(self.rhs.get(&id).copied().unwrap_or(0.0), self.maximize)
            }
            None => 0.0,
        };

        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, num_constraints, num_vars)
            .map_err(|e| MpsError::ParseError {
                line: 0,
                message: format!("Failed to build matrix: {}", e),
            })?;

        let mut bounds = vec![(0.0, f64::INFINITY); num_vars];
        for (bound_type, col, value) in &self.bounds {
            // Undefined column references were rejected by
            // `validate_references` before this runs in `run`, so the compact
            // index exists.
            let col_idx = col_index[*col as usize] as usize;
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
                        bound_type,
                        self.col_names.name(*col)
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
        let mut explicitly_bounded = vec![false; self.col_names.len()];
        for &(_, col, _) in &self.bounds {
            explicitly_bounded[col as usize] = true;
        }
        let mut integer_vars: Vec<usize> = Vec::with_capacity(self.integer_cols.len());
        for &col in &self.integer_cols {
            if col_index[col as usize] != NO_INDEX {
                let col_idx = col_index[col as usize] as usize;
                if !explicitly_bounded[col as usize] {
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
///
/// The retry re-streams the source from the start (`LineSource` is replayable
/// by contract); nothing from the failed free-format reading is carried over,
/// which is what lets the parser drop its intermediates as it goes instead of
/// holding them until a fallback decision is possible.
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
        let col_x = parser.col_names.intern_ref("x");
        let row_obj = parser.row_names.intern_ref("obj");
        parser.columns.push((col_x, row_obj, 1.0));
        parser.bounds.push((bound_type, col_x, None));
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

    /// Row / column counts and per-column row references of the synthetic MPS
    /// residency file (see the QPS parser's identically named constants for
    /// the rationale; the sizes are chosen so both sides peak in the MB range).
    const RESIDENCY_ROWS: usize = 256;
    const RESIDENCY_COLS: usize = 4096;
    const RESIDENCY_REFS_PER_COL: usize = 32;
    /// Extra bytes appended to every name in the long-name variant.
    const NAME_PAD: usize = 56;
    /// Allowed long-vs-short peak growth per entry: names are owned by the
    /// intern tables only, so name length may grow residency per *distinct
    /// name*, never per entry.
    const NAME_LENGTH_TOLERANCE_BYTES: usize = 10;
    /// Absolute per-entry budget plus fixed slack for the long-name peak.
    const PARSE_PEAK_BUDGET_BYTES: usize = 160;
    const PARSE_PEAK_FIXED_SLACK: usize = 2 * 1024 * 1024;

    /// A well-formed MPS text whose names carry `pad` filler bytes beyond the
    /// index (inside an INTORG block so `integer_cols` is exercised too), plus
    /// the number of COLUMNS entries it declares.
    fn synthetic_mps(pad: usize) -> (String, usize) {
        let row_name = |i: usize| format!("r{i:05}{}", "z".repeat(pad));
        let col_name = |j: usize| format!("c{j:05}{}", "z".repeat(pad));
        let mut text = String::from("NAME RESIDENCY\nROWS\n N obj\n");
        for i in 0..RESIDENCY_ROWS {
            text.push_str(&format!(" L {}\n", row_name(i)));
        }
        text.push_str("COLUMNS\n");
        text.push_str("    'x'    'MARKER'    'INTORG'\n");
        let mut nnz = 0usize;
        for j in 0..RESIDENCY_COLS {
            text.push_str(&format!(" {} obj 1.0", col_name(j)));
            nnz += 1;
            for k in 0..RESIDENCY_REFS_PER_COL {
                text.push_str(&format!(" {} 1.0", row_name((j + k) % RESIDENCY_ROWS)));
                nnz += 1;
            }
            text.push('\n');
        }
        text.push_str("    'x'    'MARKER'    'INTEND'\n");
        text.push_str("RHS\n rhs");
        for i in 0..RESIDENCY_ROWS {
            text.push_str(&format!(" {} 2.0", row_name(i)));
        }
        text.push('\n');
        text.push_str("BOUNDS\n");
        for j in 0..RESIDENCY_COLS / 2 {
            text.push_str(&format!(" UP bnd {} 10.0\n", col_name(j)));
        }
        text.push_str("ENDATA\n");
        (text, nnz)
    }

    /// Sentinel (residency, MPS side of the same fix as the QPS parser):
    /// parse-time allocation per sparse entry must not depend on name length.
    /// See the QPS parser's test of the same name for the full rationale.
    #[test]
    fn parse_retention_independent_of_name_length() {
        let (short, nnz) = synthetic_mps(0);
        let (long, nnz_long) = synthetic_mps(NAME_PAD);
        assert_eq!(nnz, nnz_long);

        crate::peak_alloc::begin();
        let parsed_short = crate::mps::parse_mps(&short).expect("short-name file must parse");
        let peak_short = crate::peak_alloc::peak_bytes();

        crate::peak_alloc::begin();
        let parsed_long = crate::mps::parse_mps(&long).expect("long-name file must parse");
        let peak_long = crate::peak_alloc::peak_bytes();

        assert_eq!(parsed_long.num_vars, parsed_short.num_vars);
        assert_eq!(parsed_long.num_constraints, parsed_short.num_constraints);
        assert_eq!(parsed_long.c, parsed_short.c);
        assert_eq!(parsed_long.b, parsed_short.b);
        assert_eq!(
            parsed_long.a.col_ptr().last(),
            parsed_short.a.col_ptr().last()
        );

        let tolerance = NAME_LENGTH_TOLERANCE_BYTES * nnz;
        assert!(
            peak_long.saturating_sub(peak_short) <= tolerance,
            "long-name parse peaked {} bytes above short-name; tolerance {} bytes \
             ({} B/entry × {nnz} entries). Per-entry residency must not depend on \
             name length — names belong in the intern table only.",
            peak_long.saturating_sub(peak_short),
            tolerance,
            NAME_LENGTH_TOLERANCE_BYTES,
        );
        let budget = PARSE_PEAK_BUDGET_BYTES * nnz + PARSE_PEAK_FIXED_SLACK;
        assert!(
            peak_long <= budget,
            "long-name parse peaked {} bytes, budget {} bytes ({} B/entry + {} fixed)",
            peak_long,
            budget,
            PARSE_PEAK_BUDGET_BYTES,
            PARSE_PEAK_FIXED_SLACK,
        );
    }
}
