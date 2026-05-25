#[derive(Debug, Clone, Copy)]
pub(super) enum RowType {
    N,
    L,
    G,
    E,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Section {
    None,
    Name,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Quadobj,
    EndData,
}

impl Section {
    pub(super) fn from_line(line: &str) -> Option<Self> {
        let upper = line.to_uppercase();
        if upper.starts_with("NAME") {
            Some(Section::Name)
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
        } else if upper.starts_with("QUADOBJ") {
            Some(Section::Quadobj)
        } else if upper.starts_with("ENDATA") {
            Some(Section::EndData)
        } else {
            None
        }
    }
}

/// Extract a fixed-width MPS field at byte offsets `start..end`, trimmed.
///
/// Standard MPS field positions (0-indexed):
/// - Field 2 (col_name / rhs_name): cols 4–11  → `mps_field(line, 4, 12)`
/// - Field 3 (row_name 1):          cols 14–21 → `mps_field(line, 14, 22)`
/// - Field 4 (value 1):             cols 24–35 → `mps_field(line, 24, 36)`
/// - Field 5 (row_name 2):          cols 39–46 → `mps_field(line, 39, 47)`
/// - Field 6 (value 2):             cols 49–60 → `mps_field(line, 49, 61)`
pub(super) fn mps_field(line: &str, start: usize, end: usize) -> &str {
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
