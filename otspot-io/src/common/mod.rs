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

/// Parse a free-format MPS section line into `(name, value)` pairs (skips
/// `parts[0]` = section name). Non-finite rejected except `allow_nonfinite_for_row`.
pub(crate) fn parse_mps_free_pairs(
    parts: &[&str],
    line_num: usize,
    section: &str,
    allow_nonfinite_for_row: Option<&str>,
) -> Result<Vec<(String, f64)>, String> {
    let mut pairs = Vec::new();
    let mut i = 1;
    while i + 1 < parts.len() {
        let name = parts[i].to_string();
        let raw = parts[i + 1];
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
    Ok(pairs)
}

/// Parse a fixed-format MPS section line into at most two `(name, value)` pairs
/// at standard positions (`14..22`/`24..36` and `39..47`/`49..61`). Non-finite
/// rejected except `allow_nonfinite_for_row`.
pub(crate) fn parse_mps_fixed_pairs(
    line: &str,
    line_num: usize,
    section: &str,
    allow_nonfinite_for_row: Option<&str>,
) -> Result<Vec<(String, f64)>, String> {
    let mut pairs = Vec::new();

    let name1 = mps_field(line, 14, 22).to_string();
    if !name1.is_empty() {
        let val_str1 = mps_field(line, 24, 36);
        if !val_str1.is_empty() {
            let value1 = val_str1.parse::<f64>().map_err(|_| {
                format!(
                    "line {}: Invalid {} value '{}'",
                    line_num, section, val_str1
                )
            })?;
            let is_exempt1 = allow_nonfinite_for_row == Some(name1.as_str());
            if !is_exempt1 && !value1.is_finite() {
                return Err(format!(
                    "line {}: Non-finite {} value '{}'",
                    line_num, section, value1
                ));
            }
            pairs.push((name1, value1));
        }
    }

    let name2 = mps_field(line, 39, 47).to_string();
    if !name2.is_empty() {
        let val_str2 = mps_field(line, 49, 61);
        if !val_str2.is_empty() {
            let value2 = val_str2.parse::<f64>().map_err(|_| {
                format!(
                    "line {}: Invalid {} value '{}'",
                    line_num, section, val_str2
                )
            })?;
            let is_exempt2 = allow_nonfinite_for_row == Some(name2.as_str());
            if !is_exempt2 && !value2.is_finite() {
                return Err(format!(
                    "line {}: Non-finite {} value '{}'",
                    line_num, section, value2
                ));
            }
            pairs.push((name2, value2));
        }
    }

    Ok(pairs)
}
