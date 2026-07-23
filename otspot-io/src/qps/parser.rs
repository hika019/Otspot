use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Seek};

use otspot_core::problem::ConstraintType;
use otspot_core::qp::QpProblem;
use otspot_core::sparse::CscMatrix;

use super::types::{BoundType, Section};
use super::QpsError;
use crate::common::{
    integer_marker_kind, parse_bounds_entry, parse_columns_entry, parse_objsense_value,
    parse_quadobj_entry, parse_row_decl, parse_vector_entry, parse_with_format_fallback, Format,
    LineSource, ReaderSource, RowNameIndex, RowType, SectionState, VectorSectionState,
};

pub(super) struct QpsParser {
    format: Format,
    /// Line reached by this reading; reported alongside a failure so the caller
    /// can tell which of the two format readings engaged with more of the file.
    progress: usize,
    rows: Vec<(String, RowType)>,
    row_names: HashSet<String>,
    columns: Vec<(String, String, f64)>,
    rhs: HashMap<String, f64>,
    ranges: HashMap<String, f64>,
    rhs_vectors: VectorSectionState,
    ranges_vectors: VectorSectionState,
    row_index: RowNameIndex,
    bounds: Vec<(BoundType, String, Option<f64>)>,
    /// QUADOBJ entries: (col1, col2, value) in upper-triangular order.
    quadobj: Vec<(String, String, f64)>,
    /// Normalized (min, max) key pairs seen in QUADOBJ, to detect symmetric duplicates.
    quadobj_seen: HashSet<(String, String)>,
    obj_row: Option<String>,
    maximize: bool,
}

impl QpsParser {
    fn new(format: Format) -> Self {
        Self {
            format,
            progress: 0,
            rows: Vec::new(),
            row_names: HashSet::new(),
            columns: Vec::new(),
            rhs: HashMap::new(),
            ranges: HashMap::new(),
            rhs_vectors: VectorSectionState::new(),
            ranges_vectors: VectorSectionState::new(),
            row_index: RowNameIndex::new(),
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
            if trimmed.is_empty() || trimmed.starts_with('*') || trimmed.starts_with('$') {
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
        if !self.row_names.insert(row_name.clone()) {
            return Err(QpsError::ParseError {
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
        for (row_name, value) in pairs {
            self.columns.push((col_name.clone(), row_name, value));
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
            self.obj_row.clone()
        } else {
            None
        };
        let (vector_name, pairs) = parse_vector_entry(
            line,
            tokens,
            self.format,
            &self.rows,
            &mut self.row_index,
            line_num,
            section,
            allow_nonfinite_for_row.as_deref(),
        )
        .map_err(|message| QpsError::ParseError {
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
        self.bounds.push((bound_type, col_name, value));
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

        // Reject duplicates using a lexicographically normalized key: both
        // (x1,x2) and (x2,x1) denote the same upper-triangular Q entry after
        // symmetrization.
        let key = if col1 <= col2 {
            (col1.clone(), col2.clone())
        } else {
            (col2.clone(), col1.clone())
        };
        if !self.quadobj_seen.insert(key) {
            return Err(QpsError::ParseError {
                line: line_num,
                message: format!("Duplicate QUADOBJ entry: ({}, {})", col1, col2),
            });
        }
        self.quadobj.push((col1, col2, value));
        Ok(())
    }

    /// Every row/column name referenced by COLUMNS / RHS / RANGES / BOUNDS /
    /// QUADOBJ must have been declared. Dropping an unknown name silently is
    /// how a misread layout used to go unnoticed: reading a fixed-column line
    /// as free format invents names that match nothing, and discarding them
    /// quietly corrupts the model. Erroring out is also what lets the caller
    /// detect a wrong layout guess and retry the file as fixed-column.
    fn validate_references(&self) -> Result<(), QpsError> {
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
                return Err(QpsError::UndefinedReference {
                    kind: "row".to_string(),
                    name: row_name.to_string(),
                });
            }
        }
        for col_name in self
            .bounds
            .iter()
            .map(|(_, col, _)| col)
            .chain(self.quadobj.iter().flat_map(|(c1, c2, _)| [c1, c2]))
        {
            if !declared_cols.contains(col_name.as_str()) {
                return Err(QpsError::UndefinedReference {
                    kind: "column".to_string(),
                    name: col_name.clone(),
                });
            }
        }
        Ok(())
    }

    fn build_qp_problem(&self) -> Result<QpProblem, QpsError> {
        let mut col_map: HashMap<&str, usize> = HashMap::new();
        for (col_name, _, _) in &self.columns {
            let next = col_map.len();
            col_map.entry(col_name.as_str()).or_insert(next);
        }
        let n = col_map.len();
        let obj_row = self.obj_row.as_deref();

        let mut c = vec![0.0; n];
        for (col_name, row_name, value) in &self.columns {
            if Some(row_name.as_str()) == obj_row {
                c[col_map[col_name.as_str()]] += *value;
            }
        }

        // Constraint rows in declaration order; N rows (objective and any other
        // free row) never become constraints.
        struct ConstraintRow<'a> {
            name: &'a str,
            rtype: RowType,
            rhs: f64,
        }
        let mut constraint_rows: Vec<ConstraintRow<'_>> = Vec::new();
        for (row_name, row_type) in &self.rows {
            if !row_type.is_constraint() {
                continue;
            }
            constraint_rows.push(ConstraintRow {
                name: row_name.as_str(),
                rtype: *row_type,
                rhs: self.rhs.get(row_name).copied().unwrap_or(0.0),
            });
        }

        // RANGES splits a row into its Le and Ge halves.
        let mut base_rows: Vec<ConstraintRow<'_>> = Vec::new();
        let mut range_extra: Vec<ConstraintRow<'_>> = Vec::new();
        for row in constraint_rows {
            let Some(&range_val) = self.ranges.get(row.name) else {
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
        struct AugRow<'a> {
            name: &'a str,
            sign: f64,
            rhs: f64,
        }
        let mut aug_rows: Vec<AugRow<'_>> = Vec::new();
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
        let m = aug_rows.len();

        let mut row_name_to_indices: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, ar) in aug_rows.iter().enumerate() {
            row_name_to_indices.entry(ar.name).or_default().push(i);
        }

        let mut a_rows: Vec<usize> = Vec::new();
        let mut a_cols: Vec<usize> = Vec::new();
        let mut a_vals: Vec<f64> = Vec::new();
        for (col_name, row_name, value) in &self.columns {
            if Some(row_name.as_str()) == obj_row {
                continue;
            }
            let col_idx = col_map[col_name.as_str()];
            // Declared free (N) rows other than the objective carry no
            // constraint; the standard ignores their coefficients.
            let Some(indices) = row_name_to_indices.get(row_name.as_str()) else {
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
        for (bound_type, col_name, value) in &self.bounds {
            let col_idx = col_map[col_name.as_str()];
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
            }
        }

        // QUADOBJ is upper-triangular; symmetrize.
        let mut q_rows: Vec<usize> = Vec::new();
        let mut q_cols: Vec<usize> = Vec::new();
        let mut q_vals: Vec<f64> = Vec::new();
        for (col1, col2, value) in &self.quadobj {
            let i = col_map[col1.as_str()];
            let j = col_map[col2.as_str()];
            q_rows.push(i);
            q_cols.push(j);
            q_vals.push(*value);
            if i != j {
                q_rows.push(j);
                q_cols.push(i);
                q_vals.push(*value);
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
        parser.columns.push(("x".to_string(), "obj".to_string(), 1.0));
        parser.bounds.push((bound_type, "x".to_string(), None));
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
}
