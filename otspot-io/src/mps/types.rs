/// Re-export from `crate::common` so that tests in `mps::mod` can access it
/// via `types::is_fixed_width_format`.
#[cfg(test)]
pub(super) use crate::common::is_fixed_width_format;

/// Default upper bound for integer variables that appear only inside an
/// INTORG/INTEND marker block and have no explicit BOUNDS entry.
///
/// Matches the classical OSL/CPLEX convention (also used by HiGHS): such
/// variables are treated as binary [0, 1].
pub(super) const INTEGER_DEFAULT_UPPER_BINARY: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IntegerMarker {
    Start,
    End,
}

/// Returns `Some(kind)` when `line` contains both a `'MARKER'` token and an
/// `INTORG`/`INTEND` token (quotes stripped, case-insensitive).
pub(super) fn integer_marker_kind(line: &str) -> Option<IntegerMarker> {
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
    if has_marker { kind } else { None }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum BoundType {
    LO,
    UP,
    FX,
    FR,
    MI,
    BV,
    PL,
    LI,
    UI,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Section {
    None,
    Name,
    ObjSense,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    EndData,
}

impl Section {
    pub(super) fn from_line(line: &str) -> Option<Self> {
        let upper = line.to_uppercase();
        if upper.starts_with("NAME") {
            Some(Section::Name)
        } else if upper.starts_with("OBJSENSE") {
            Some(Section::ObjSense)
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
