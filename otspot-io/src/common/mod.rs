//! Shared MPS/QPS parsing primitives.
//!
//! Both the MPS and QPS parsers share field-extraction helpers and the
//! free/fixed-format pair-parsing logic.  Centralising them here removes
//! the per-parser duplication and gives a single place to enforce
//! input-validation invariants (finite values, non-empty names).

/// Row type as defined in the ROWS section.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RowType {
    N,
    L,
    G,
    E,
}

/// Extract a fixed-width MPS field at byte offsets `start..end`, trimmed.
///
/// Standard MPS field positions (0-indexed):
/// - Field 2 (col_name / rhs_name): cols 4–11  → `mps_field(line, 4, 12)`
/// - Field 3 (row_name 1):          cols 14–21 → `mps_field(line, 14, 22)`
/// - Field 4 (value 1):             cols 24–35 → `mps_field(line, 24, 36)`
/// - Field 5 (row_name 2):          cols 39–46 → `mps_field(line, 39, 47)`
/// - Field 6 (value 2):             cols 49–60 → `mps_field(line, 49, 61)`
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

/// Returns `true` when column 15 (0-indexed: 14) is whitespace, indicating
/// fixed-width MPS format.  Short or empty lines return `false`.
pub(crate) fn is_fixed_width_format(line: &str) -> bool {
    line.chars().nth(14).is_some_and(|c| c.is_whitespace())
}

/// Parse a free-format MPS section line into `(name, value)` pairs.
///
/// Skips `parts[0]` (the RHS/RANGES section name) and collects adjacent
/// `(name, f64)` pairs from `parts[1..]`.
///
/// Non-finite values are rejected for all rows except the optional
/// `allow_nonfinite_for_row`.  Pass `None` to enforce finite for all rows.
///
/// # Errors
///
/// Returns an error string when a value cannot be parsed or is non-finite
/// for a constraint row.
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
        let value = raw.parse::<f64>().map_err(|_| {
            format!("line {}: Invalid {} value '{}'", line_num, section, raw)
        })?;
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

/// Parse a fixed-format MPS section line into at most two `(name, value)` pairs.
///
/// Extracts pairs from standard MPS field positions:
/// - Pair 1: name at cols 14–21, value at cols 24–35.
/// - Pair 2: name at cols 39–46, value at cols 49–60.
///
/// Non-finite values are rejected unless the row name matches `allow_nonfinite_for_row`.
/// Pass `None` to enforce finite for all rows.
///
/// # Errors
///
/// Returns an error string when a value cannot be parsed or is non-finite
/// for a constraint row.
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
                format!("line {}: Invalid {} value '{}'", line_num, section, val_str1)
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
                format!("line {}: Invalid {} value '{}'", line_num, section, val_str2)
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
