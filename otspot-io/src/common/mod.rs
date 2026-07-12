//! Shared MPS/QPS parsing primitives (field extraction + free/fixed-format
//! pair parsing)。input validation (finite values, non-empty names) を集約する。

/// Parse an OBJSENSE value; returns `true` for MAX, `false` for MIN.
/// On failure returns an error message string (caller wraps it into its own error type).
pub(crate) fn parse_objsense_value(line: &str) -> Result<bool, String> {
    match line.trim().to_uppercase().as_str() {
        "MAX" => Ok(true),
        "MIN" => Ok(false),
        _ => Err(format!(
            "Invalid OBJSENSE value '{}'; expected MIN or MAX",
            line.trim()
        )),
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
    /// (caller should `break`). Returns `Err` on duplicate sections; `name_section`
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

/// Row type as defined in the ROWS section.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RowType {
    N,
    L,
    G,
    E,
}

/// Extract a fixed-width MPS field at byte offsets `start..end`, trimmed.
/// Standard positions: col_name `4..12`, row_name1 `14..22`, value1 `24..36`,
/// row_name2 `39..47`, value2 `49..61`.
pub(crate) fn mps_field(line: &str, start: usize, end: usize) -> &str {
    let len = line.len();
    if start >= len {
        return "";
    }
    let actual_end = end.min(len);
    if !line.is_char_boundary(start) || !line.is_char_boundary(actual_end) {
        return "";
    }
    line[start..actual_end].trim()
}

/// Returns `true` when col 15 (0-indexed: 14) is whitespace = fixed-width MPS.
pub(crate) fn is_fixed_width_format(line: &str) -> bool {
    line.chars().nth(14).is_some_and(|c| c.is_whitespace())
}

/// Lazily-built row-name membership index, used to disambiguate RHS/RANGES
/// vector-name-omitted (shorthand) lines from standard ones (see
/// `parse_vector_pairs`).
///
/// ROWS always precedes RHS/RANGES in a well-formed MPS/QPS file (a
/// requirement enforced by `SectionState`'s duplicate-section check — ROWS
/// cannot legally reappear once closed), so `rows` is frozen by the time
/// this is queried; building the set once and reusing it avoids an O(rows)
/// rescan per RHS/RANGES line.
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

/// `(vector_name, (row_name, value) pairs)` — see `parse_vector_pairs`.
pub(crate) type VectorPairsResult = Result<(Option<String>, Vec<(String, f64)>), String>;

/// Parses an RHS/RANGES-style free-format line into `(vector_name, pairs)`.
///
/// The vector-name field is commonly omitted when a section carries just one
/// (implicitly anonymous) vector — e.g. Netlib LP files emitted by `emps`
/// write bare `row val [row val ...]` RHS lines with no name at all. Blindly
/// treating `parts[0]` as the vector name silently swallows that row as a
/// bogus vector name whenever it looks like a value's row (`blend`'s numeric
/// row names "65", "66", ... are exactly this shape).
///
/// Disambiguation checks `parts[0]` against the rows declared in the
/// already-parsed ROWS section: a real row name means there is no vector
/// name and every token pairs up from index 0; otherwise `parts[0]` is the
/// vector name and pairing starts at index 1.
///
/// RHS/RANGES/BOUNDS are parsed purely by whitespace tokenization, so strict
/// fixed-column MPS files that rely on byte offsets (e.g. names containing
/// embedded spaces) are NOT supported — only whitespace-separated files.
pub(crate) fn parse_vector_pairs(
    parts: &[&str],
    rows: &[(String, RowType)],
    row_index: &mut RowNameIndex,
    line_num: usize,
    section: &str,
    allow_nonfinite_for_row: Option<&str>,
) -> VectorPairsResult {
    // A standard-form line whose vector name equals a declared row name is not
    // disambiguable (MPS keeps row and vector namespaces separate); it is read
    // as shorthand, yielding an odd token count that the parity check below
    // rejects with a hard error — intentional fail-safe over a silent mis-parse.
    let has_vector_name = !parts.is_empty() && !row_index.contains(rows, parts[0]);
    let (vector_name, pair_tokens): (Option<String>, &[&str]) = if has_vector_name {
        (Some(parts[0].to_string()), &parts[1..])
    } else {
        (None, parts)
    };

    if pair_tokens.len() % 2 != 0 {
        return Err(format!(
            "line {}: {} row name '{}' has no matching value",
            line_num,
            section,
            pair_tokens[pair_tokens.len() - 1]
        ));
    }

    let mut pairs = Vec::with_capacity(pair_tokens.len() / 2);
    let mut i = 0;
    while i < pair_tokens.len() {
        let name = pair_tokens[i].to_string();
        let raw = pair_tokens[i + 1];
        let value = raw
            .parse::<f64>()
            .map_err(|_| format!("line {}: Invalid {} value '{}'", line_num, section, raw))?;
        let is_exempt = allow_nonfinite_for_row == Some(name.as_str());
        if !is_exempt && !value.is_finite() {
            return Err(format!(
                "line {}: Non-finite {} value '{}'",
                line_num, section, value
            ));
        }
        pairs.push((name, value));
        i += 2;
    }
    Ok((vector_name, pairs))
}

/// Tracks `(vector_name, row_name)` pairs seen in a repeatable-vector MPS/QPS
/// section (RHS or RANGES), and which vector's values are actually applied.
///
/// The MPS/QPS standard allows a section to carry multiple named vectors,
/// each free to reuse the same row names. Only the FIRST vector encountered
/// is applied to the caller's value map (GLPK/CPLEX convention: take the
/// first vector, ignore the rest); a row appearing ONLY in a later vector is
/// therefore dropped (its value is never taken). Later vectors are still
/// checked for duplicates so malformed input isn't silently accepted.
///
/// A line that omits its vector name (the shorthand form recognized by
/// `parse_vector_pairs`) is attributed to whichever vector identity is
/// already open for the section — via `resolve_vector_name` — rather than a
/// fresh identity per line, so consecutive shorthand lines share one vector
/// and a shorthand entry still collides with an earlier *named* entry for
/// the same row.
#[derive(Default)]
pub(crate) struct VectorSectionState {
    seen: std::collections::HashSet<(String, String)>,
    first_vector: Option<String>,
}

impl VectorSectionState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Resolves the effective vector identity for a line: `Some(name)` is
    /// used verbatim; `None` (shorthand) resolves to whichever vector this
    /// section already opened, or `""` if none yet — `""` can never collide
    /// with a real vector name since `split_whitespace` never yields empty
    /// tokens.
    pub(crate) fn resolve_vector_name(&self, vector_name: Option<&str>) -> String {
        match vector_name {
            Some(name) => name.to_string(),
            None => self.first_vector.clone().unwrap_or_default(),
        }
    }

    /// Records `row_name -> value` from `vector_name` into `target` (only
    /// when `vector_name` is the first one seen by this state). Returns an
    /// error naming `section` when `(vector_name, row_name)` was already
    /// seen.
    pub(crate) fn record(
        &mut self,
        target: &mut std::collections::HashMap<String, f64>,
        section: &str,
        vector_name: &str,
        row_name: String,
        value: f64,
    ) -> Result<(), String> {
        if !self
            .seen
            .insert((vector_name.to_string(), row_name.clone()))
        {
            return Err(format!(
                "{section}: duplicate entry for row '{row_name}' in vector '{vector_name}'"
            ));
        }
        let is_first_vector = self
            .first_vector
            .get_or_insert_with(|| vector_name.to_string())
            .as_str()
            == vector_name;
        if is_first_vector {
            target.insert(row_name, value);
        }
        Ok(())
    }
}
