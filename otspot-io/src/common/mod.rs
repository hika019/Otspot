//! Shared MPS/QPS parsing primitives.
//!
//! Format determination is **per file, not per line**. A file is first read as
//! free format (whitespace-tokenized); if that read fails for any reason — a
//! malformed line, or a reference to a row/column that ROWS/COLUMNS never
//! declared — the whole file is re-read as strict fixed-column MPS. Every
//! section (ROWS / COLUMNS / RHS / RANGES / BOUNDS / QUADOBJ) obeys that one
//! decision, so a name containing embedded spaces (legal in fixed-column MPS,
//! e.g. Netlib `forplan`'s row `"BR   1 1"`) is read consistently everywhere,
//! or not at all.
//!
//! Per-line format guessing cannot work: a fixed-column line such as
//! `    RHS       BR   1 1            6.` tokenizes into an even number of
//! well-formed-looking tokens, so any local heuristic accepts it and silently
//! invents the rows `BR` and `1`. Only a whole-file decision, backed by hard
//! errors on undeclared names, rejects that reading.
//!
//! Both readings are strict, so the fallback cannot become a second way to
//! misread a file: the fixed-column reader requires the line to lie on the grid
//! (see `FIXED_GUTTERS`), so a free-format file with names longer than 8 bytes
//! is not silently re-read as fixed-column with every name truncated. Columns
//! 62+ (comment / sequence-number fields) are discarded before grid checks.

/// Parse an OBJSENSE value; returns `true` for MAX, `false` for MIN.
///
/// Both the abbreviated (`MAX`/`MIN`) and spelled-out (`MAXIMIZE`/`MINIMIZE`)
/// forms are accepted, as HiGHS and SCIP do.
pub(crate) fn parse_objsense_value(value: &str) -> Result<bool, String> {
    match value.trim().to_uppercase().as_str() {
        "MAX" | "MAXIMIZE" => Ok(true),
        "MIN" | "MINIMIZE" => Ok(false),
        other => Err(format!(
            "Invalid OBJSENSE value '{}'; expected MIN/MINIMIZE or MAX/MAXIMIZE",
            other
        )),
    }
}

/// Layout of a data line, decided once for the whole file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    /// Whitespace-tokenized. Names may not contain spaces.
    Free,
    /// Strict IBM fixed-column MPS. Names may contain spaces.
    Fixed,
}

/// Byte offsets (0-indexed, half-open) of the six IBM fixed-column MPS fields,
/// i.e. 1-indexed columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61.
///
/// Field roles by section:
/// - ROWS:         F1 = row type, F2 = row name
/// - COLUMNS:      F2 = column name, (F3, F4) and (F5, F6) = (row, value)
/// - RHS / RANGES: F2 = vector name (optional), (F3, F4) and (F5, F6) = (row, value)
/// - BOUNDS:       F1 = bound type, F2 = bound-set name, F3 = column, F4 = value
/// - QUADOBJ:      F2 = column 1, F3 = column 2, F4 = value
const FIELD_1: (usize, usize) = (1, 3);
const FIELD_2: (usize, usize) = (4, 12);
const FIELD_3: (usize, usize) = (14, 22);
const FIELD_4: (usize, usize) = (24, 36);
const FIELD_5: (usize, usize) = (39, 47);
const FIELD_6: (usize, usize) = (49, 61);

/// Token count of a well-formed free-format ROWS line (`type name`). A
/// fixed-column row name with embedded spaces yields more, which is precisely
/// the signal that routes the file to the fixed-column reader.
const FREE_ROWS_TOKENS: usize = 2;

/// Minimum token count of a free-format COLUMNS line (`col row value`).
const FREE_COLUMNS_MIN_TOKENS: usize = 3;

/// Token count of a free-format QUADOBJ line (`col1 col2 value`).
const FREE_QUADOBJ_TOKENS: usize = 3;

/// Bytes of a line that carry data. Everything from column 62 on is ignored:
/// columns 62-72 are the comment field and 73-80 the sequence-number field,
/// which the MPS standard (and GLPK, lp_solve, IBM OSL) discard.
const FIXED_DATA_LIMIT: usize = 61;

/// The blank gutters that separate the six fixed fields, as 0-indexed byte
/// ranges within the data portion of a line.
///
/// Requiring them to be blank is what keeps the fixed-column retry from becoming
/// a second way to misread a file. Silently truncating an overlong name would
/// let a *free-format* file with names longer than 8 bytes re-parse
/// "successfully" as fixed-column with every name clipped: distinct rows
/// `ROWLONGNAME1` and `ROWLONGNAME2` would both collapse to `ROWLONGN`, and a
/// typo'd reference would then resolve against the clipped name — a
/// plausible-but-wrong model with no diagnostic. Checking the gutters (rather
/// than only the bytes adjacent to each field) also leaves no unexamined byte
/// between fields.
const FIXED_GUTTERS: [(usize, usize); 6] = [(0, 1), (3, 4), (12, 14), (22, 24), (36, 39), (47, 49)];

/// The indicator field (columns 2-3). ROWS puts the row type here and BOUNDS the
/// bound type; every other section leaves it blank, and content there means the
/// line is not on the grid.
const FIXED_INDICATOR: (usize, usize) = FIELD_1;

/// A line's data portion (columns 1-61), verified to lie on the fixed-column
/// grid. Fields are read from it by byte offset, which is the only way to
/// recover a name containing embedded spaces.
pub(crate) struct FixedLine<'a> {
    data: &'a str,
    section: &'a str,
    line_num: usize,
}

/// Validate that `line` lies on the fixed-column grid and return its data
/// portion. `uses_indicator` says whether this section reads columns 2-3 (ROWS,
/// BOUNDS); when it does not, that field must be blank like any gutter.
pub(crate) fn fixed_line<'a>(
    line: &'a str,
    line_num: usize,
    section: &'a str,
    uses_indicator: bool,
) -> Result<FixedLine<'a>, String> {
    let mut limit = FIXED_DATA_LIMIT.min(line.len());
    while limit > 0 && !line.is_char_boundary(limit) {
        limit -= 1;
    }
    let data = &line[..limit];

    let mut blank_ranges: Vec<(usize, usize)> = FIXED_GUTTERS.to_vec();
    if !uses_indicator {
        blank_ranges.push(FIXED_INDICATOR);
    }
    for (start, end) in blank_ranges {
        for offset in start..end.min(data.len()) {
            if !data.as_bytes()[offset].is_ascii_whitespace() {
                return Err(format!(
                    "line {}: {} column {} must be blank — it separates the fixed-column \
                     fields, so content there means a name or value overflows its field and \
                     the line does not lie on the fixed-column MPS grid",
                    line_num,
                    section,
                    offset + 1,
                ));
            }
        }
    }
    Ok(FixedLine {
        data,
        section,
        line_num,
    })
}

impl FixedLine<'_> {
    /// Extract the field at 0-indexed byte offsets `start..end`, trimmed.
    ///
    /// The blank-gutter check already guarantees that both offsets fall on a
    /// character boundary (a multi-byte character reaching a field's edge puts a
    /// continuation byte in a gutter, which is not ASCII whitespace). The check
    /// below is therefore a guard, not a live path — but it errors rather than
    /// returning an empty name, so no reading can ever silently lose a name.
    fn field(&self, (start, end): (usize, usize)) -> Result<&str, String> {
        if start >= self.data.len() {
            return Ok("");
        }
        let actual_end = end.min(self.data.len());
        if !self.data.is_char_boundary(start) || !self.data.is_char_boundary(actual_end) {
            return Err(format!(
                "line {}: {} columns {}-{} split a multi-byte character; the line does not lie \
                 on the fixed-column MPS grid",
                self.line_num,
                self.section,
                start + 1,
                end
            ));
        }
        Ok(self.data[start..actual_end].trim())
    }
}

/// A source of lines that can be read more than once.
///
/// The fixed-column retry re-reads the whole input, so the source must be
/// replayable. Every implementation holds **one line at a time** — MPS files
/// reach the GiB range (`data/miplib_2017/square47.mps` is 1.4 GiB), so
/// buffering the input to enable a second pass is not an option.
pub(crate) trait LineSource {
    /// Feed each line (1-indexed, newline stripped) to `visit`, stopping early
    /// when it returns `Ok(false)`. May be called repeatedly.
    fn visit_lines<E>(
        &self,
        wrap_io: impl Fn(std::io::Error) -> E,
        visit: impl FnMut(usize, &str) -> Result<bool, E>,
    ) -> Result<(), E>;
}

/// Lines of an in-memory string.
pub(crate) struct TextSource<'a>(pub &'a str);

impl LineSource for TextSource<'_> {
    fn visit_lines<E>(
        &self,
        _wrap_io: impl Fn(std::io::Error) -> E,
        mut visit: impl FnMut(usize, &str) -> Result<bool, E>,
    ) -> Result<(), E> {
        for (i, line) in self.0.lines().enumerate() {
            if !visit(i + 1, line)? {
                break;
            }
        }
        Ok(())
    }
}

/// Lines of a file, re-opened and re-streamed on each pass.
pub(crate) struct FileSource(pub std::path::PathBuf);

impl LineSource for FileSource {
    fn visit_lines<E>(
        &self,
        wrap_io: impl Fn(std::io::Error) -> E,
        visit: impl FnMut(usize, &str) -> Result<bool, E>,
    ) -> Result<(), E> {
        let file = std::fs::File::open(&self.0).map_err(&wrap_io)?;
        stream_lines(std::io::BufReader::new(file), wrap_io, visit)
    }
}

/// Lines of a seekable reader, replayed from wherever the reader started.
///
/// `Seek` is what makes a second pass possible without buffering; a reader that
/// cannot rewind cannot be replayed, and buffering it would defeat the purpose.
///
/// The replay returns to the reader's position as first observed, not to
/// absolute 0, so a reader already positioned partway into a stream (an MPS
/// embedded in a larger file) is parsed from that point on both passes.
pub(crate) struct ReaderSource<R: std::io::BufRead + std::io::Seek> {
    reader: std::cell::RefCell<R>,
    start: std::cell::Cell<Option<u64>>,
}

impl<R: std::io::BufRead + std::io::Seek> ReaderSource<R> {
    pub(crate) fn new(reader: R) -> Self {
        Self {
            reader: std::cell::RefCell::new(reader),
            start: std::cell::Cell::new(None),
        }
    }
}

impl<R: std::io::BufRead + std::io::Seek> LineSource for ReaderSource<R> {
    fn visit_lines<E>(
        &self,
        wrap_io: impl Fn(std::io::Error) -> E,
        visit: impl FnMut(usize, &str) -> Result<bool, E>,
    ) -> Result<(), E> {
        let mut reader = self.reader.borrow_mut();
        match self.start.get() {
            None => self
                .start
                .set(Some(reader.stream_position().map_err(&wrap_io)?)),
            Some(start) => {
                reader
                    .seek(std::io::SeekFrom::Start(start))
                    .map_err(&wrap_io)?;
            }
        }
        stream_lines(&mut *reader, wrap_io, visit)
    }
}

/// Read `reader` line by line, reusing a single buffer so that only one line is
/// held at a time.
fn stream_lines<R: std::io::BufRead, E>(
    mut reader: R,
    wrap_io: impl Fn(std::io::Error) -> E,
    mut visit: impl FnMut(usize, &str) -> Result<bool, E>,
) -> Result<(), E> {
    let mut buf = String::new();
    let mut line_num = 0;
    loop {
        buf.clear();
        if reader.read_line(&mut buf).map_err(&wrap_io)? == 0 {
            return Ok(());
        }
        line_num += 1;
        let line = buf.trim_end_matches(['\n', '\r']);
        if !visit(line_num, line)? {
            return Ok(());
        }
    }
}

/// Read `source` as free format; on any failure re-read it as fixed-column.
///
/// `parse` reports, alongside a failure, how far into the file that reading got.
/// When both readings fail the one that got **further** is reported: it engaged
/// with more of the file, so its diagnostic describes the file's actual content
/// rather than the point where the wrong grammar gave up immediately. A tie
/// favours free format, the presumptive layout.
pub(crate) fn parse_with_format_fallback<T, E, S: LineSource>(
    source: &S,
    parse: impl Fn(&S, Format) -> Result<T, (E, usize)>,
) -> Result<T, E> {
    let (free_err, free_progress) = match parse(source, Format::Free) {
        Ok(parsed) => return Ok(parsed),
        Err(failure) => failure,
    };
    match parse(source, Format::Fixed) {
        Ok(parsed) => Ok(parsed),
        Err((fixed_err, fixed_progress)) => {
            if fixed_progress > free_progress {
                Err(fixed_err)
            } else {
                Err(free_err)
            }
        }
    }
}

/// Tracks the current section and the set of sections already seen.
///
/// Encapsulates duplicate-section detection and required-section checks common
/// to both the MPS and QPS parsers.
pub(crate) struct SectionState<S> {
    pub current: S,
    seen: std::collections::HashSet<S>,
}

impl<S: Copy + Eq + std::hash::Hash + std::fmt::Debug> SectionState<S> {
    pub fn new(initial: S) -> Self {
        Self {
            current: initial,
            seen: std::collections::HashSet::new(),
        }
    }

    /// Advance to `section`. Returns `true` when `section == enddata_section`
    /// (caller should stop). Returns `Err` on duplicate sections; `name_section`
    /// and `enddata_section` are exempt from the duplicate check.
    pub fn advance<E>(
        &mut self,
        section: S,
        name_section: S,
        enddata_section: S,
        make_dup_err: impl Fn(String) -> E,
    ) -> Result<bool, E> {
        if section != name_section && section != enddata_section && self.seen.contains(&section) {
            return Err(make_dup_err(format!("{:?}", section)));
        }
        self.seen.insert(section);
        self.current = section;
        Ok(section == enddata_section)
    }

    /// Verify that each `(section, name)` pair in `required` was seen (in order).
    /// Returns an error for the first missing section.
    pub fn require<E>(
        &self,
        required: &[(S, &str)],
        make_err: impl Fn(String) -> E,
    ) -> Result<(), E> {
        for (section, name) in required {
            if !self.seen.contains(section) {
                return Err(make_err((*name).to_string()));
            }
        }
        Ok(())
    }
}

/// An INTORG/INTEND marker bracketing a block of integer columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntegerMarker {
    Start,
    End,
}

/// Returns `Some(kind)` when `line` carries both a `'MARKER'` token and an
/// `INTORG`/`INTEND` token (quotes stripped, case-insensitive).
///
/// Both tokens are required. Keying off `'MARKER'` alone would silently discard
/// a COLUMNS line for a column legitimately *named* `MARKER`, losing its
/// coefficients — the same class of silent drop this module exists to prevent.
pub(crate) fn integer_marker_kind(line: &str) -> Option<IntegerMarker> {
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

/// Row type as defined in the ROWS section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowType {
    N,
    L,
    G,
    E,
}

impl RowType {
    /// `N` rows are free rows: the first is the objective, any further one is
    /// declared-but-ignored. Neither becomes a constraint row.
    pub(crate) fn is_constraint(self) -> bool {
        !matches!(self, RowType::N)
    }
}

/// Reads a ROWS line as `(row type, row name)`.
///
/// Free format demands exactly `FREE_ROWS_TOKENS` tokens: a fixed-column name
/// with embedded spaces (`" L  BR   1 1"`) splits into more, so the file fails
/// the free-format read here and is retried as fixed-column.
pub(crate) fn parse_row_decl(
    line: &str,
    tokens: &[&str],
    format: Format,
    line_num: usize,
) -> Result<(String, String), String> {
    let (type_str, name) = match format {
        Format::Free => {
            if tokens.len() != FREE_ROWS_TOKENS {
                return Err(format!(
                    "line {}: ROWS line must have exactly {} fields (type name), got {}",
                    line_num,
                    FREE_ROWS_TOKENS,
                    tokens.len()
                ));
            }
            (tokens[0].to_string(), tokens[1].to_string())
        }
        Format::Fixed => {
            let fixed = fixed_line(line, line_num, "ROWS", true)?;
            (
                fixed.field(FIELD_1)?.to_string(),
                fixed.field(FIELD_2)?.to_string(),
            )
        }
    };
    if name.is_empty() {
        return Err(format!("line {}: ROWS line missing row name", line_num));
    }
    Ok((type_str, name))
}

/// Reads a COLUMNS line as `(column name, (row, value) pairs)`.
pub(crate) fn parse_columns_entry(
    line: &str,
    tokens: &[&str],
    format: Format,
    line_num: usize,
) -> Result<(String, Vec<(String, f64)>), String> {
    match format {
        Format::Free => {
            if tokens.len() < FREE_COLUMNS_MIN_TOKENS {
                return Err(format!(
                    "line {}: COLUMNS line requires at least {} fields (col row value)",
                    line_num, FREE_COLUMNS_MIN_TOKENS
                ));
            }
            let pairs = pairs_from_tokens(&tokens[1..], line_num, "COLUMNS", None)?;
            Ok((tokens[0].to_string(), pairs))
        }
        Format::Fixed => {
            let fixed = fixed_line(line, line_num, "COLUMNS", false)?;
            let col_name = fixed.field(FIELD_2)?.to_string();
            if col_name.is_empty() {
                return Err(format!(
                    "line {}: COLUMNS line missing column name",
                    line_num
                ));
            }
            let pairs =
                pairs_from_fixed_fields(&fixed, None, [(FIELD_3, FIELD_4), (FIELD_5, FIELD_6)])?;
            if pairs.is_empty() {
                return Err(format!(
                    "line {}: COLUMNS line has no (row, value) pair",
                    line_num
                ));
            }
            Ok((col_name, pairs))
        }
    }
}

/// Lazily-built row-name membership index used to spot RHS/RANGES lines that
/// omit the vector name (see `parse_vector_entry`).
///
/// ROWS always precedes RHS/RANGES in a well-formed file (`SectionState`'s
/// duplicate-section check keeps ROWS from reappearing), so the set is frozen
/// by the time it is queried; building it once avoids an O(rows) rescan per
/// RHS/RANGES line.
#[derive(Default)]
pub(crate) struct RowNameIndex(Option<std::collections::HashSet<String>>);

impl RowNameIndex {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn contains(&mut self, rows: &[(String, RowType)], name: &str) -> bool {
        self.0
            .get_or_insert_with(|| rows.iter().map(|(n, _)| n.clone()).collect())
            .contains(name)
    }
}

/// `(vector_name, (row_name, value) pairs)` — see `parse_vector_entry`.
pub(crate) type VectorEntry = Result<(Option<String>, Vec<(String, f64)>), String>;

/// Reads an RHS/RANGES line as `(vector name, (row, value) pairs)`.
///
/// The vector-name field is commonly omitted when the section carries a single
/// anonymous vector — Netlib LP files decoded by `emps` write bare
/// `row val [row val ...]` RHS lines. Treating `tokens[0]` as a vector name
/// unconditionally swallows a row whenever the row name looks like one
/// (`blend`'s numeric row names "65", "66", ... are exactly that shape), so in
/// free format `tokens[0]` is checked against the declared rows: a real row
/// name means the vector name was omitted and pairing starts at index 0.
///
/// Fixed format cannot use that check: a declared row name is legal (if
/// unusual) as the vector name too (e.g. an RHS vector named `RHS` next to a
/// row also named `RHS`). Standard reading (vector name in field 2, pairs in
/// fields 3-4 and 5-6) is tried first; only when it reads no pair at all —
/// field 4 or 6 holding a value with no name, exactly what happens when a
/// writer put the first row name in field 2 and shifted both pairs one field
/// left — is field 2 reread as that row name (pairs then in fields 2-3, 4-5,
/// field 6 unused). A well-formed shorthand line always fails the standard
/// reading this way (its field 6 is never used, so the second pair's value
/// comes up empty), so the two readings never both succeed: there is no case
/// where this order of preference silently picks the wrong one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_vector_entry(
    line: &str,
    tokens: &[&str],
    format: Format,
    rows: &[(String, RowType)],
    row_index: &mut RowNameIndex,
    line_num: usize,
    section: &str,
    allow_nonfinite_for_row: Option<&str>,
) -> VectorEntry {
    match format {
        Format::Free => {
            let has_vector_name = !tokens.is_empty() && !row_index.contains(rows, tokens[0]);
            let (vector_name, pair_tokens) = if has_vector_name {
                (Some(tokens[0].to_string()), &tokens[1..])
            } else {
                (None, tokens)
            };
            let pairs = pairs_from_tokens(pair_tokens, line_num, section, allow_nonfinite_for_row)?;
            Ok((vector_name, pairs))
        }
        Format::Fixed => {
            let fixed = fixed_line(line, line_num, section, false)?;
            let vector_field = fixed.field(FIELD_2)?;
            let standard = pairs_from_fixed_fields(
                &fixed,
                allow_nonfinite_for_row,
                [(FIELD_3, FIELD_4), (FIELD_5, FIELD_6)],
            );

            if !vector_field.is_empty() && !matches!(standard, Ok(ref pairs) if !pairs.is_empty()) {
                if let Ok(pairs) = pairs_from_fixed_fields(
                    &fixed,
                    allow_nonfinite_for_row,
                    [(FIELD_2, FIELD_3), (FIELD_4, FIELD_5)],
                ) {
                    if !pairs.is_empty() {
                        let trailing = fixed.field(FIELD_6)?;
                        if !trailing.is_empty() {
                            return Err(format!(
                                "line {}: {} value '{}' has no matching name — field 2 is row \
                                 '{}', so the vector name is omitted and the (row, value) pairs \
                                 occupy fields 2-5; field 6 must be blank",
                                line_num, section, trailing, vector_field
                            ));
                        }
                        return Ok((None, pairs));
                    }
                }
                // Neither reading produced a pair; fall through to report the
                // standard reading's diagnostic, since it is the presumptive
                // layout.
            }

            let vector_name = (!vector_field.is_empty()).then(|| vector_field.to_string());
            let pairs = standard?;
            if pairs.is_empty() {
                return Err(format!(
                    "line {}: {} line has no (row, value) pair",
                    line_num, section
                ));
            }
            Ok((vector_name, pairs))
        }
    }
}

/// Reads a BOUNDS line as `(column name, value)`; the bound type is read by the
/// caller (its enum differs between MPS and QPS).
///
/// The bound-set-name field (standard: `TYPE BNDNAME COL [VALUE]`) is commonly
/// omitted (shorthand: `TYPE COL [VALUE]`). Free format disambiguates by token
/// count; fixed format tries standard reading (field 3 = column, field 4 =
/// value) and rereads field 2 as the column only when standard reading cannot
/// be well-formed — which is what keeps the two readings from both claiming
/// data.
///
/// Surplus input is rejected, but each format needs its own check. Free format
/// has no positional fields, so any token past the slot a type can use is
/// invalid. That alone is not enough: `parse_with_format_fallback` re-reads the
/// file as fixed format, where a surplus token on a grid-aligned line lands in
/// field 5/6 — undefined for BOUNDS, unlike RHS/RANGES which use them for a
/// second pair — so those are rejected too. Field 4 is not: real files pad it
/// with a redundant `0.0`/`1.0` for `FR`/`MI`/`BV`/`PL` (`leo1`/`leo2`).
pub(crate) fn parse_bounds_entry(
    line: &str,
    tokens: &[&str],
    format: Format,
    line_num: usize,
    value_required: bool,
) -> Result<(String, Option<f64>), String> {
    let (col_name, raw_value) = match format {
        Format::Free => {
            let min_standard_len = if value_required { 4 } else { 3 };
            let col_idx = if tokens.len() >= min_standard_len {
                2
            } else {
                1
            };
            if tokens.len() <= col_idx {
                return Err(format!(
                    "line {}: BOUNDS line missing column name",
                    line_num
                ));
            }
            let value_idx = col_idx + 1;
            if !value_required && tokens.len() > value_idx {
                return Err(format!(
                    "line {}: BOUNDS entry for col='{}' does not take a value, got trailing \
                     token '{}'",
                    line_num, tokens[col_idx], tokens[value_idx]
                ));
            }
            if value_required && tokens.len() > value_idx + 1 {
                return Err(format!(
                    "line {}: BOUNDS type {} takes exactly one value for col='{}'",
                    line_num, tokens[0], tokens[col_idx]
                ));
            }
            let raw = if value_required && tokens.len() > value_idx {
                Some(tokens[value_idx].to_string())
            } else {
                None
            };
            (tokens[col_idx].to_string(), raw)
        }
        Format::Fixed => {
            let fixed = fixed_line(line, line_num, "BOUNDS", true)?;
            let bndname_field = fixed.field(FIELD_2)?;
            let standard_col = fixed.field(FIELD_3)?;
            let standard_value = fixed.field(FIELD_4)?;
            let field_5 = fixed.field(FIELD_5)?;
            let field_6 = fixed.field(FIELD_6)?;
            let trailing = if !field_5.is_empty() {
                field_5
            } else {
                field_6
            };
            if !trailing.is_empty() {
                return Err(format!(
                    "line {}: BOUNDS line has unexpected content '{}' in field 5/6; BOUNDS \
                     defines only fields 1-4 (type, bound-set name, column, value)",
                    line_num, trailing
                ));
            }
            let standard_ok =
                !standard_col.is_empty() && (!value_required || !standard_value.is_empty());

            if standard_ok {
                (
                    standard_col.to_string(),
                    (!standard_value.is_empty()).then(|| standard_value.to_string()),
                )
            } else {
                let raw = if !standard_col.is_empty() {
                    standard_col
                } else {
                    standard_value
                };
                (
                    bndname_field.to_string(),
                    (!raw.is_empty()).then(|| raw.to_string()),
                )
            }
        }
    };

    if col_name.is_empty() {
        return Err(format!(
            "line {}: BOUNDS line missing column name",
            line_num
        ));
    }

    let value = match raw_value {
        Some(raw) => Some(parse_value(&raw, line_num, "BOUNDS", &col_name, None)?),
        None => None,
    };
    if value_required && value.is_none() {
        return Err(format!(
            "line {}: BOUNDS entry for col='{}' requires a value",
            line_num, col_name
        ));
    }
    Ok((col_name, value))
}

/// Reads a QUADOBJ line as `(col1, col2, value)`.
pub(crate) fn parse_quadobj_entry(
    line: &str,
    tokens: &[&str],
    format: Format,
    line_num: usize,
) -> Result<(String, String, f64), String> {
    let (col1, col2, raw) = match format {
        Format::Free => {
            if tokens.len() < FREE_QUADOBJ_TOKENS {
                return Err(format!(
                    "line {}: QUADOBJ line requires {} fields (col1 col2 value)",
                    line_num, FREE_QUADOBJ_TOKENS
                ));
            }
            (
                tokens[0].to_string(),
                tokens[1].to_string(),
                tokens[2].to_string(),
            )
        }
        Format::Fixed => {
            let fixed = fixed_line(line, line_num, "QUADOBJ", false)?;
            (
                fixed.field(FIELD_2)?.to_string(),
                fixed.field(FIELD_3)?.to_string(),
                fixed.field(FIELD_4)?.to_string(),
            )
        }
    };
    if col1.is_empty() || col2.is_empty() {
        return Err(format!(
            "line {}: QUADOBJ line missing a column name",
            line_num
        ));
    }
    let value = parse_value(&raw, line_num, "QUADOBJ", &col1, None)?;
    Ok((col1, col2, value))
}

/// Parse and validate one numeric field. Non-finite values are rejected unless
/// the entry names `allow_nonfinite_for_row` (the objective row, whose RHS is
/// the objective offset and is range-checked by the caller).
fn parse_value(
    raw: &str,
    line_num: usize,
    section: &str,
    name: &str,
    allow_nonfinite_for_row: Option<&str>,
) -> Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("line {}: Invalid {} value '{}'", line_num, section, raw))?;
    if allow_nonfinite_for_row != Some(name) && !value.is_finite() {
        return Err(format!(
            "line {}: Non-finite {} value '{}' for '{}'",
            line_num, section, value, name
        ));
    }
    Ok(value)
}

/// Pair up whitespace tokens as `name value name value ...`.
fn pairs_from_tokens(
    tokens: &[&str],
    line_num: usize,
    section: &str,
    allow_nonfinite_for_row: Option<&str>,
) -> Result<Vec<(String, f64)>, String> {
    if tokens.is_empty() || !tokens.len().is_multiple_of(2) {
        return Err(format!(
            "line {}: {} has a name without a matching value",
            line_num, section
        ));
    }
    let mut pairs = Vec::with_capacity(tokens.len() / 2);
    for pair in tokens.chunks_exact(2) {
        let value = parse_value(pair[1], line_num, section, pair[0], allow_nonfinite_for_row)?;
        pairs.push((pair[0].to_string(), value));
    }
    Ok(pairs)
}

/// Read two fixed-column `(name, value)` slots, given as `[(name, value); 2]`
/// field pairs — normally (F3, F4) and (F5, F6), or (F2, F3) and (F4, F5) when
/// the vector name is omitted and the pairs shift one field left (see
/// `parse_vector_entry`).
///
/// Both slots are always examined, even when the first is blank: skipping the
/// second slot's checks whenever its name is empty would leave a stray value
/// sitting in its columns unexamined.
fn pairs_from_fixed_fields(
    fixed: &FixedLine<'_>,
    allow_nonfinite_for_row: Option<&str>,
    slots: [((usize, usize), (usize, usize)); 2],
) -> Result<Vec<(String, f64)>, String> {
    let (line_num, section) = (fixed.line_num, fixed.section);
    let mut pairs = Vec::with_capacity(2);
    for (name_field, value_field) in slots {
        let name = fixed.field(name_field)?;
        let raw = fixed.field(value_field)?;
        if name.is_empty() {
            if !raw.is_empty() {
                return Err(format!(
                    "line {}: {} value '{}' has no matching name",
                    line_num, section, raw
                ));
            }
            continue;
        }
        if raw.is_empty() {
            return Err(format!(
                "line {}: {} name '{}' has no matching value",
                line_num, section, name
            ));
        }
        let value = parse_value(raw, line_num, section, name, allow_nonfinite_for_row)?;
        pairs.push((name.to_string(), value));
    }
    Ok(pairs)
}

/// Tracks `(vector_name, row_name)` pairs seen in a repeatable-vector section
/// (RHS or RANGES), and which vector's values are actually applied.
///
/// The standard allows a section to carry multiple named vectors, each free to
/// reuse the same row names. Only the FIRST vector encountered is applied
/// (GLPK/CPLEX convention: take the first vector, ignore the rest); a row
/// appearing ONLY in a later vector is therefore dropped. Later vectors are
/// still checked for duplicates so malformed input isn't silently accepted.
///
/// A line that omits its vector name is attributed to the vector already open
/// for the section. Naming a vector *after* the section has already taken
/// unnamed entries is rejected: the two readings (same vector, or a second
/// vector whose entries must then be discarded) differ in the values that reach
/// the model, and picking either silently would drop data.
#[derive(Default)]
pub(crate) struct VectorSectionState {
    seen: std::collections::HashSet<(String, String)>,
    first_vector: Option<String>,
}

/// Identity given to entries written before any vector name appears. It can
/// never collide with a real vector name, which is never empty when present.
const ANONYMOUS_VECTOR: &str = "";

impl VectorSectionState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Records `row_name -> value` into `target` (only when the entry belongs to
    /// the first vector this section opened). Returns an error when
    /// `(vector, row)` repeats, or when a named entry follows unnamed ones.
    pub(crate) fn record(
        &mut self,
        target: &mut std::collections::HashMap<String, f64>,
        section: &str,
        vector_name: Option<&str>,
        row_name: String,
        value: f64,
    ) -> Result<(), String> {
        let vector = match (self.first_vector.as_deref(), vector_name) {
            (Some(ANONYMOUS_VECTOR), Some(named)) => {
                return Err(format!(
                    "{section}: vector '{named}' is named after unnamed entries were already \
                     read; the section is ambiguous (one vector with the name omitted earlier, \
                     or a second vector whose entries must be discarded)"
                ))
            }
            (_, Some(named)) => named.to_string(),
            (Some(open), None) => open.to_string(),
            (None, None) => ANONYMOUS_VECTOR.to_string(),
        };

        if !self.seen.insert((vector.clone(), row_name.clone())) {
            return Err(format!(
                "{section}: duplicate entry for row '{row_name}' in vector '{vector}'"
            ));
        }
        let is_first_vector = self.first_vector.get_or_insert_with(|| vector.clone()) == &vector;
        if is_first_vector {
            target.insert(row_name, value);
        }
        Ok(())
    }
}
