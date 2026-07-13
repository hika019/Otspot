/// Default upper bound for integer variables that appear only inside an
/// INTORG/INTEND marker block and have no explicit BOUNDS entry.
///
/// Matches the classical OSL/CPLEX convention (also used by HiGHS): such
/// variables are treated as binary [0, 1].
pub(super) const INTEGER_DEFAULT_UPPER_BINARY: f64 = 1.0;

/// Marker detection lives in `crate::common` so the MPS and QPS parsers agree on
/// what a marker line is; re-exported here for `mps`'s own use.
pub(super) use crate::common::{integer_marker_kind, IntegerMarker};

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
