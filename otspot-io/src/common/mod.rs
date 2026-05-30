//! Shared MPS/QPS parsing primitives (field extraction + free/fixed-format
//! pair parsing)。input validation (finite values, non-empty names) を集約する。

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
