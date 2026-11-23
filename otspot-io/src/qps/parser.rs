use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Seek};

use otspot_core::problem::ConstraintType;
use otspot_core::qp::QpProblem;
use otspot_core::sparse::CscMatrix;

use super::types::{BoundType, Section};
use super::QpsError;
use crate::common::{
    integer_marker_kind, objective_offset_from_rhs, parse_bounds_entry, parse_columns_entry,
    parse_objsense_value, parse_quadobj_entry, parse_row_decl, parse_vector_entry,
    parse_with_format_fallback, Format, LineSource, NameRegistry, ReaderSource, RowType,
    SectionState, VectorSectionState, NO_INDEX,
};

/// Bits to shift the smaller id by when packing a QUADOBJ dedup key — the
/// `(min, max)` pair of interned column ids — into one `u64`. Column ids are
/// `u32`, so the packing is injective on unordered pairs.
const QUADOBJ_KEY_SHIFT: u32 = 32;

/// Per-entry state uses interned name ids, not `String`s: `columns` holds
/// (column id, row id, value) fixed-size records and `bounds` / `quadobj`
/// likewise, so parse-time residency per sparse entry no longer depends on how
/// long the referenced names are.
pub(super) struct QpsParser {
    format: Format,
    /// Line reached by this reading; reported alongside a failure so the caller
    /// can tell which of the two format readings engaged with more of the file.
    progress: usize,
    rows: Vec<(u32, RowType)>,
    row_names: NameRegistry,
    col_names: NameRegistry,
    columns: Vec<(u32, u32, f64)>,
    rhs: HashMap<u32, f64>,
    ranges: HashMap<u32, f64>,
    rhs_vectors: VectorSectionState,
    ranges_vectors: VectorSectionState,
    bounds: Vec<(BoundType, u32, Option<f64>)>,
    /// QUADOBJ entries: (col1, col2, value) as column ids, in
    /// upper-triangular order.
    quadobj: Vec<(u32, u32, f64)>,
    /// Packed normalized (min, max) column-id keys seen in QUADOBJ, to detect
    /// symmetric duplicates.
    quadobj_seen: HashSet<u64>,
    obj_row: Option<u32>,
    maximize: bool,
}

/// A constraint row after RANGES expansion and `Ax <= b` / `Ax = b`
/// normalization: its declared row id, the sign its coefficients are scaled
/// by (G rows are negated), and its final rhs.
struct AugRow {
    name: u32,
    sign: f64,
    rhs: f64,
}

impl QpsParser {
    fn new(format: Format) -> Self {
        Self {
            format,
            progress: 0,
            rows: Vec::new(),
            row_names: NameRegistry::new(),
            col_names: NameRegistry::new(),
            columns: Vec::new(),
            rhs: HashMap::new(),
            ranges: HashMap::new(),
            rhs_vectors: VectorSectionState::new(),
            ranges_vectors: VectorSectionState::new(),
            bounds: Vec::new(),
            quadobj: Vec::new(),
            quadobj_seen: HashSet::new(),
            obj_row: None,
            maximize: false,
        }
    }
    fn run<S: LineSource>(&mut self, source: &S) -> Result<QpProblem, QpsError> {
        let mut state = SectionState::new(Section::None);

        source.visit_lines(QpsError::IoError, |line_num, line| {
            self.progress = line_num;
            let trimmed = line.trim();
            // `$` is only a comment marker in column 1 (the IBM fixed-column
            // dialect). A `$` behind leading blanks is the first field of a
            // data line: pang.QPS has a column legitimately named `$0AICCB`,
            // and testing the *trimmed* line skipped its COLUMNS lines,
            // silently dropping the variable (n=459 where the file declares 460).
            // `*` keeps the trimmed-line rule (an indented `*` opens a comment),
            // so a name legitimately starting with `*` would be dropped here and
            // in the MPS parser alike. Known asymmetry, no such line in the
            // bench data; tracked in https://github.com/hika019/Otspot/issues/62.
            if trimmed.is_empty() || trimmed.starts_with('*') || line.starts_with('$') {
                return Ok(true);
            }

            if !line.starts_with(' ') && !line.starts_with('\t') {
                let Some(section) = Section::from_line(trimmed) else {
                    return Err(QpsError::ParseError {
                        line: line_num,
                        message: format!("Unrecognized section header: '{}'", trimmed),
                    });
                };
                let should_stop = state.advance(
                    section,
                    Section::Name,
                    Section::EndData,
                    QpsError::DuplicateSection,
                )?;
                // `OBJSENSE  MAX` on the header line is legal; ignoring the value
                // would silently minimize a problem that asked to be maximized.
                if section == Section::ObjSense {
                    if let Some(rest) = trimmed.get("OBJSENSE".len()..).map(str::trim) {
                        if !rest.is_empty() {
                            self.parse_objsense_line(rest, line_num)?;
                        }
                    }
                }
                return Ok(!should_stop);
            }

            let tokens: Vec<&str> = line.split_whitespace().collect();
            match state.current {
                Section::ObjSense => self.parse_objsense_line(line, line_num)?,
                Section::Rows => self.parse_rows_line(line, &tokens, line_num)?,
                Section::Columns => self.parse_columns_line(line, &tokens, line_num)?,
                Section::Rhs => self.parse_vector_line(line, &tokens, line_num, "RHS")?,
                Section::Ranges => self.parse_vector_line(line, &tokens, line_num, "RANGES")?,
                Section::Bounds => self.parse_bounds_line(line, &tokens, line_num)?,
                Section::Quadobj => self.parse_quadobj_line(line, &tokens, line_num)?,
                Section::EndData => return Ok(false),
                Section::None | Section::Name => {}
            }
            Ok(true)
        })?;

        state.require(
            &[
                (Section::EndData, "ENDATA"),
                (Section::Rows, "ROWS"),
                (Section::Columns, "COLUMNS"),
            ],
            QpsError::MissingSection,
        )?;

        self.validate_references()?;
        self.build_qp_problem()
    }

    fn parse_objsense_line(&mut self, value: &str, line_num: usize) -> Result<(), QpsError> {
        self.maximize = parse_objsense_value(value).map_err(|message| QpsError::ParseError {
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
    ) -> Result<(), QpsError> {
        let (type_str, row_name) =
            parse_row_decl(line, tokens, self.format, line_num).map_err(|message| {
                QpsError::ParseError {
                    line: line_num,
                    message,
                }
            })?;
        let row_type = match type_str.to_uppercase().as_str() {
            "N" => RowType::N,
            "L" => RowType::L,
            "G" => RowType::G,
            "E" => RowType::E,
            _ => {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Unknown row type: {}", type_str),
                })
            }
        };
        // Two rows sharing a name would make every reference to it ambiguous and
        // silently resolve to whichever landed in the index last.
        let (row_id, already_declared) = self.row_names.intern_declared(&row_name);
        if already_declared {
            return Err(QpsError::ParseError {
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
    ) -> Result<(), QpsError> {
        // QPS keeps no integrality; a marker line declares no coefficients, so skip it.
        if integer_marker_kind(line).is_some() {
            return Ok(());
        }
        let (col_name, pairs) =
            parse_columns_entry(line, tokens, self.format, line_num).map_err(|message| {
                QpsError::ParseError {
                    line: line_num,
                    message,
                }
            })?;
        // A COLUMNS line always carries at least one (row, value) pair, so
        // marking the column declared here matches the old "declared = appears
        // in `columns`" set exactly.
        let col_id = self.col_names.intern_declared(&col_name).0;
        for (row_name, value) in pairs {
            let row_id = self.row_names.intern_ref(&row_name);
            self.columns.push((col_id, row_id, value));
        }
        Ok(())
    }

    /// RHS and RANGES share one grammar; they differ only in the map they fill.
    ///
    /// A non-finite RHS on the objective row is tolerated here and rejected in
    /// `build_qp_problem` as `InvalidObjectiveOffset`; RANGES has no objective
    /// entry, so it exempts nothing.
    fn parse_vector_line(
        &mut self,
        line: &str,
        tokens: &[&str],
        line_num: usize,
        section: &str,
    ) -> Result<(), QpsError> {
        let allow_nonfinite_for_row = if section == "RHS" {
            self.obj_row.map(|id| self.row_names.name(id).to_string())
        } else {
            None
        };
        let (vector_name, pairs) = parse_vector_entry(
            line,
            tokens,
            self.format,
            &|name| self.row_names.is_declared_name(name),
            line_num,
            section,
            allow_nonfinite_for_row.as_deref(),
        )
        .map_err(|message| QpsError::ParseError {
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
                .map_err(|message| QpsError::ParseError {
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
    ) -> Result<(), QpsError> {
        if tokens.is_empty() {
            return Err(QpsError::ParseError {
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
            _ => {
                return Err(QpsError::ParseError {
                    line: line_num,
                    message: format!("Unknown bound type: {}", tokens[0]),
                })
            }
        };
        let value_required = matches!(bound_type, BoundType::LO | BoundType::UP | BoundType::FX);

        let (col_name, value) =
            parse_bounds_entry(line, tokens, self.format, line_num, value_required).map_err(
                |message| QpsError::ParseError {
                    line: line_num,
                    message,
                },
            )?;
        let col_id = self.col_names.intern_ref(&col_name);
        self.bounds.push((bound_type, col_id, value));
        Ok(())
    }

    fn parse_quadobj_line(
        &mut self,
        line: &str,
        tokens: &[&str],
        line_num: usize,
    ) -> Result<(), QpsError> {
        let (col1, col2, value) = parse_quadobj_entry(line, tokens, self.format, line_num)
            .map_err(|message| QpsError::ParseError {
                line: line_num,
                message,
            })?;

        // Reject duplicates using a normalized key: both (x1,x2) and (x2,x1)
        // denote the same upper-triangular Q entry after symmetrization. The
        // key orders the interned ids instead of the name strings; ids are
        // assigned 1:1 to names, so the equivalence relation on unordered name
        // pairs is identical either way.
        let col1_id = self.col_names.intern_ref(&col1);
        let col2_id = self.col_names.intern_ref(&col2);
        let (lo, hi) = if col1_id <= col2_id {
            (col1_id, col2_id)
        } else {
            (col2_id, col1_id)
        };
        let key = ((lo as u64) << QUADOBJ_KEY_SHIFT) | hi as u64;
        if !self.quadobj_seen.insert(key) {
            return Err(QpsError::ParseError {
                line: line_num,
                message: format!("Duplicate QUADOBJ entry: ({}, {})", col1, col2),
            });
        }
        self.quadobj.push((col1_id, col2_id, value));
        Ok(())
    }

    /// Every row/column name referenced by COLUMNS / RHS / RANGES / BOUNDS /
    /// QUADOBJ must have been declared. Dropping an unknown name silently is
    /// how a misread layout used to go unnoticed: reading a fixed-column line
    /// as free format invents names that match nothing, and discarding them
    /// quietly corrupts the model. Erroring out is also what lets the caller
    /// detect a wrong layout guess and retry the file as fixed-column.
    fn validate_references(&self) -> Result<(), QpsError> {
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
                return Err(QpsError::UndefinedReference {
                    kind: "row".to_string(),
                    name: self.row_names.name(row_id).to_string(),
                });
            }
        }
        for col_id in self
            .bounds
            .iter()
            .map(|&(_, col, _)| col)
            .chain(self.quadobj.iter().flat_map(|&(c1, c2, _)| [c1, c2]))
        {
            if !self.col_names.is_declared(col_id) {
                return Err(QpsError::UndefinedReference {
                    kind: "column".to_string(),
                    name: self.col_names.name(col_id).to_string(),
                });
            }
        }
        Ok(())
    }

    /// Expand declared rows into the augmented constraint rows: constraint
    /// rows in declaration order (N rows — objective and any other free row —
    /// never become constraints), RANGES splitting each affected row into its
    /// Le and Ge halves, and G rows flipped to `Ax <= b` with `sign = -1`.
    /// Returns the augmented rows and their parallel constraint types.
    fn expand_constraint_rows(&self) -> (Vec<AugRow>, Vec<ConstraintType>) {
        struct ConstraintRow {
            name: u32,
            rtype: RowType,
            rhs: f64,
        }
        let mut constraint_rows: Vec<ConstraintRow> = Vec::new();
        for &(row_id, row_type) in &self.rows {
            if !row_type.is_constraint() {
                continue;
            }
            constraint_rows.push(ConstraintRow {
                name: row_id,
                rtype: row_type,
                rhs: self.rhs.get(&row_id).copied().unwrap_or(0.0),
            });
        }

        // RANGES splits a row into its Le and Ge halves.
        let mut base_rows: Vec<ConstraintRow> = Vec::new();
        let mut range_extra: Vec<ConstraintRow> = Vec::new();
        for row in constraint_rows {
            let Some(&range_val) = self.ranges.get(&row.name) else {
                base_rows.push(row);
                continue;
            };
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
                RowType::N => unreachable!("N rows are not constraint rows"),
            };
            base_rows.push(ConstraintRow {
                name: row.name,
                rtype: RowType::L,
                rhs: le_rhs,
            });
            range_extra.push(ConstraintRow {
                name: row.name,
                rtype: RowType::G,
                rhs: ge_rhs,
            });
        }
        base_rows.extend(range_extra);

        // Normalize to `Ax <= b` / `Ax = b` by flipping the sign of G rows.
        let mut aug_rows: Vec<AugRow> = Vec::new();
        let mut constraint_types: Vec<ConstraintType> = Vec::new();
        for row in base_rows {
            let (sign, rhs, ctype) = match row.rtype {
                RowType::L => (1.0, row.rhs, ConstraintType::Le),
                RowType::G => (-1.0, -row.rhs, ConstraintType::Le),
                RowType::E => (1.0, row.rhs, ConstraintType::Eq),
                RowType::N => unreachable!("N rows are not constraint rows"),
            };
            aug_rows.push(AugRow {
                name: row.name,
                sign,
                rhs,
            });
            constraint_types.push(ctype);
        }
        (aug_rows, constraint_types)
    }

    fn build_qp_problem(&self) -> Result<QpProblem, QpsError> {
        // Compact column indices in first-appearance order — the same numbering
        // the name-keyed `col_map` used to produce. `NO_INDEX` marks ids that
        // COLUMNS never put an entry for; entries in `self.columns` are all
        // indexed by construction.
        let mut col_index = vec![NO_INDEX; self.col_names.len()];
        let mut n = 0usize;
        for &(col, _, _) in &self.columns {
            if col_index[col as usize] == NO_INDEX {
                col_index[col as usize] = n as u32;
                n += 1;
            }
        }
        let obj_row = self.obj_row;

        let mut c = vec![0.0; n];
        for &(col, row, value) in &self.columns {
            if Some(row) == obj_row {
                c[col_index[col as usize] as usize] += value;
            }
        }

        // Constraint rows in declaration order, RANGES-expanded and
        // sign-normalized (see `expand_constraint_rows`).
        let (aug_rows, constraint_types) = self.expand_constraint_rows();
        let m = aug_rows.len();

        let mut row_name_to_indices: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, ar) in aug_rows.iter().enumerate() {
            row_name_to_indices.entry(ar.name).or_default().push(i);
        }

        let mut a_rows: Vec<usize> = Vec::with_capacity(self.columns.len());
        let mut a_cols: Vec<usize> = Vec::with_capacity(self.columns.len());
        let mut a_vals: Vec<f64> = Vec::with_capacity(self.columns.len());
        for &(col, row, value) in &self.columns {
            if Some(row) == obj_row {
                continue;
            }
            let col_idx = col_index[col as usize] as usize;
            // Declared free (N) rows other than the objective carry no
            // constraint; the standard ignores their coefficients.
            let Some(indices) = row_name_to_indices.get(&row) else {
                continue;
            };
            for &aug_idx in indices {
                a_rows.push(aug_idx);
                a_cols.push(col_idx);
                a_vals.push(aug_rows[aug_idx].sign * value);
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
        for (bound_type, col, value) in &self.bounds {
            // Undefined column references were rejected by
            // `validate_references` before this runs in `run`, so the compact
            // index exists.
            let col_idx = col_index[*col as usize] as usize;
            // LO/UP/FX are value_required at parse time (`parse_bounds_entry`),
            // so `value` is guaranteed `Some` here. The `ok_or_else` is defense
            // against that invariant regressing: it must surface as an error,
            // not silently fabricate 0.0/INFINITY.
            let require_value = || -> Result<f64, QpsError> {
                value.ok_or_else(|| QpsError::ParseError {
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
            }
        }

        // QUADOBJ is upper-triangular; symmetrize.
        let mut q_rows: Vec<usize> = Vec::new();
        let mut q_cols: Vec<usize> = Vec::new();
        let mut q_vals: Vec<f64> = Vec::new();
        for &(col1, col2, value) in &self.quadobj {
            let i = col_index[col1 as usize] as usize;
            let j = col_index[col2 as usize] as usize;
            q_rows.push(i);
            q_cols.push(j);
            q_vals.push(value);
            if i != j {
                q_rows.push(j);
                q_cols.push(i);
                q_vals.push(value);
            }
        }

        // Normalize MAX → MIN by negating the objective (c and Q).
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

        let obj_offset = match obj_row {
            Some(id) => {
                objective_offset_from_rhs(self.rhs.get(&id).copied().unwrap_or(0.0), self.maximize)
            }
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

/// Parse `source`, retrying as fixed-column MPS if the free-format read fails.
///
/// The retry re-streams the source from the start (`LineSource` is replayable
/// by contract); nothing from the failed free-format reading is carried over,
/// which is what lets the parser drop its intermediates as it goes instead of
/// holding them until a fallback decision is possible.
pub(super) fn parse_qps_source<S: LineSource>(source: &S) -> Result<QpProblem, QpsError> {
    parse_with_format_fallback(source, |source, format| {
        let mut parser = QpsParser::new(format);
        parser.run(source).map_err(|e| (e, parser.progress))
    })
}

// ── Public entry points ───────────────────────────────────────────────────────

/// Parse a QPS stream.
///
/// The reader must be seekable: a file that turns out to be fixed-column is
/// re-read from the start, and rewinding is what lets that happen without
/// holding the input in memory.
pub fn parse_qps_reader<R: BufRead + Seek>(reader: R) -> Result<QpProblem, QpsError> {
    parse_qps_source(&ReaderSource::new(reader))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_bounds_entry` guarantees `value_required` bound types (LO, UP,
    /// FX) never carry `None` by the time they reach `self.bounds`. This
    /// bypasses that guard by constructing the parser's internal state
    /// directly, to prove `build_qp_problem` treats a violation of the
    /// invariant as a hard error instead of fabricating a bound.
    fn parser_with_dangling_bound(bound_type: BoundType) -> QpsParser {
        let mut parser = QpsParser::new(Format::Free);
        let col_x = parser.col_names.intern_ref("x");
        let row_obj = parser.row_names.intern_ref("obj");
        parser.columns.push((col_x, row_obj, 1.0));
        parser.bounds.push((bound_type, col_x, None));
        parser
    }

    #[test]
    fn build_qp_problem_rejects_missing_value_for_lo_bound() {
        let parser = parser_with_dangling_bound(BoundType::LO);
        let err = parser.build_qp_problem().unwrap_err();
        assert!(matches!(err, QpsError::ParseError { .. }), "{err:?}");
    }

    #[test]
    fn build_qp_problem_rejects_missing_value_for_up_bound() {
        let parser = parser_with_dangling_bound(BoundType::UP);
        let err = parser.build_qp_problem().unwrap_err();
        assert!(matches!(err, QpsError::ParseError { .. }), "{err:?}");
    }

    #[test]
    fn build_qp_problem_rejects_missing_value_for_fx_bound() {
        let parser = parser_with_dangling_bound(BoundType::FX);
        let err = parser.build_qp_problem().unwrap_err();
        assert!(matches!(err, QpsError::ParseError { .. }), "{err:?}");
    }

    /// Valueless bound types must still succeed with `value = None`; this is
    /// their normal, well-formed state, not a violated invariant.
    #[test]
    fn build_qp_problem_accepts_valueless_bound_types() {
        for bound_type in [BoundType::FR, BoundType::MI, BoundType::BV, BoundType::PL] {
            let parser = parser_with_dangling_bound(bound_type);
            assert!(
                parser.build_qp_problem().is_ok(),
                "valueless bound type {bound_type:?} must not require a value"
            );
        }
    }

    /// Row count of the synthetic residency files.
    const RESIDENCY_ROWS: usize = 256;
    /// Column count of the synthetic residency files.
    const RESIDENCY_COLS: usize = 4096;
    /// Constraint-row references per column in the synthetic files (added to
    /// the one objective reference each column carries). High entry counts per
    /// distinct name make [`NAME_LENGTH_TOLERANCE_BYTES`] a tight per-entry
    /// budget relative to the (per-name) intern-table growth it must allow.
    const RESIDENCY_REFS_PER_COL: usize = 32;
    /// Extra bytes appended to every name in the long-name variant. Long
    /// enough that a per-entry `String` copy of the name is plainly visible in
    /// the peak, yet one file still fits comfortably in memory.
    const NAME_PAD: usize = 56;
    /// Per-entry slack the long-vs-short-name peak difference may consume.
    /// Legitimate growth from longer names enters only through the intern
    /// table (once per distinct name, ~`NAME_PAD` bytes each, ≤ rows+cols
    /// names), a small fraction of `RESIDENCY_REFS_PER_COL + 1` entries per
    /// column; 10 bytes per entry is ample for the table and forbids per-entry
    /// name retention.
    const NAME_LENGTH_TOLERANCE_BYTES: usize = 10;
    /// Absolute per-entry budget for the long-name parse peak: fixed-size
    /// intermediates plus the CSC build transients must stay well under this
    /// regardless of name length.
    const PARSE_PEAK_BUDGET_BYTES: usize = 160;
    /// Fixed headroom in [`PARSE_PEAK_BUDGET_BYTES`]-accounting for
    /// dimension-sized arrays and allocator bookkeeping, independent of nnz.
    const PARSE_PEAK_FIXED_SLACK: usize = 2 * 1024 * 1024;

    /// A well-formed QPS text whose names carry `pad` filler bytes beyond the
    /// index, plus the number of COLUMNS entries it declares.
    fn synthetic_qps(pad: usize) -> (String, usize) {
        let row_name = |i: usize| format!("r{i:05}{}", "z".repeat(pad));
        let col_name = |j: usize| format!("c{j:05}{}", "z".repeat(pad));
        let mut text = String::from("NAME RESIDENCY\nROWS\n N obj\n");
        for i in 0..RESIDENCY_ROWS {
            text.push_str(&format!(" L {}\n", row_name(i)));
        }
        text.push_str("COLUMNS\n");
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
        text.push_str("RHS\n rhs");
        for i in 0..RESIDENCY_ROWS {
            text.push_str(&format!(" {} 2.0", row_name(i)));
        }
        text.push('\n');
        text.push_str("QUADOBJ\n");
        for j in 0..RESIDENCY_COLS {
            text.push_str(&format!(" {} {} 2.0\n", col_name(j), col_name(j)));
        }
        text.push_str("ENDATA\n");
        (text, nnz)
    }

    /// Sentinel (residency): parse-time allocation per sparse entry must not
    /// depend on how long the referenced names are.
    ///
    /// Two structurally identical QPS files differ only by `NAME_PAD` filler
    /// bytes on every name. Names live only in the intern tables — one
    /// `String` per *distinct* name — so the long-name peak may exceed the
    /// short-name peak only by table-sized slack (tolerance below). The
    /// pre-interning parser held two name `String`s per COLUMNS entry (plus
    /// `(String, String)` keys in `quadobj_seen`), for a long-vs-short
    /// difference around `2 × NAME_PAD` per entry — an order of magnitude over
    /// the tolerance — so reverting to name-`String` intermediates fails this
    /// test. The absolute budget additionally catches any change that makes
    /// parse-time residency grow quadratically or re-retains names.
    #[test]
    fn parse_retention_independent_of_name_length() {
        use crate::qps::parse_qps_str;
        let (short, nnz) = synthetic_qps(0);
        let (long, nnz_long) = synthetic_qps(NAME_PAD);
        assert_eq!(nnz, nnz_long);

        crate::peak_alloc::begin();
        let parsed_short = parse_qps_str(&short).expect("short-name file must parse");
        let peak_short = crate::peak_alloc::peak_bytes();

        crate::peak_alloc::begin();
        let parsed_long = parse_qps_str(&long).expect("long-name file must parse");
        let peak_long = crate::peak_alloc::peak_bytes();

        // Padding must not have changed the model: both files describe the
        // same matrix, so the parsed problems agree.
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
