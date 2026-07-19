//! MPS format parser (LP / MILP).
//!
//! Parses NAME / ROWS / COLUMNS / RHS / RANGES / BOUNDS / ENDATA sections.
//! INTORG/INTEND markers and BV/LI/UI bound types identify integer variables.
//!
//! The free-format and fixed-column layouts are distinguished per file, not per
//! line: a file is read as free format first and re-read as fixed-column if that
//! fails (a malformed line, or a reference to an undeclared row/column).
//!
//! - [`parse_mps`] / [`parse_mps_file`]: returns an `LpProblem` (integrality dropped).
//! - [`parse_milp`] / [`parse_milp_file`]: returns a `MilpProblem` with integer vars.

mod parser;
mod types;

use std::path::Path;

use otspot_core::mip::MilpProblem;
use otspot_core::problem::LpProblem;

pub use otspot_core::error::MpsError;
pub use parser::{parse_milp_reader, parse_mps_reader};

use crate::common::{FileSource, TextSource};

/// Parse an MPS file from `path`, returning an LP relaxation.
///
/// Streams the file; a file that turns out to be fixed-column is streamed a
/// second time rather than buffered, so peak memory stays proportional to the
/// parsed model rather than the file (MPS files reach the GiB range).
///
/// # Errors
///
/// Returns [`MpsError`] for I/O failures or malformed content.
pub fn parse_mps_file(path: &Path) -> Result<LpProblem, MpsError> {
    parser::parse_lp_source(&FileSource(path.to_path_buf()))
}

/// Parse an MPS file from `path`, returning a `MilpProblem`.
///
/// Streams the file (see [`parse_mps_file`]). Integer variables identified via
/// INTORG/INTEND markers and BV/LI/UI bound types are preserved.
///
/// # Errors
///
/// Returns [`MpsError`] for I/O failures or malformed content.
pub fn parse_milp_file(path: &Path) -> Result<MilpProblem, MpsError> {
    parser::parse_milp_source(&FileSource(path.to_path_buf()))
}

/// Parse an MPS string, returning an LP relaxation. MILP files are accepted but
/// integrality is dropped; use [`parse_milp`] to retain integer variable info.
///
/// ```
/// use otspot_io::mps::parse_mps;
/// let mps = "NAME ex\nROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 1.0 c1 2.0\nRHS\n rhs c1 10.0\nENDATA\n";
/// let lp = parse_mps(mps).unwrap();
/// assert_eq!((lp.num_vars, lp.num_constraints), (1, 1));
/// ```
pub fn parse_mps(input: &str) -> Result<LpProblem, MpsError> {
    parser::parse_lp_source(&TextSource(input))
}

/// Parse an MPS string, returning a `MilpProblem`.
///
/// See [`parse_mps`] for format details. Integer variables are preserved via
/// INTORG/INTEND markers and BV/LI/UI bound types.
pub fn parse_milp(input: &str) -> Result<MilpProblem, MpsError> {
    parser::parse_milp_source(&TextSource(input))
}

#[cfg(test)]
mod tests {
    use super::*;
    use otspot_core::problem::ConstraintType;

    #[test]
    fn test_parse_minimal() {
        let mps = r"NAME          test
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  2.0
RHS
    rhs  c1  10.0
BOUNDS
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_vars, 1);
        assert_eq!(lp.num_constraints, 1);
        assert_eq!(lp.c, vec![1.0]);
        assert_eq!(lp.b, vec![10.0]);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le]);
        assert_eq!(lp.bounds, vec![(0.0, f64::INFINITY)]);
        assert_eq!(lp.name, Some("test".to_string()));
    }

    #[test]
    fn test_parse_equality() {
        let mps = r"NAME test2
ROWS
 N  obj
 E  eq1
COLUMNS
    x1  obj  2.0  eq1  1.0
RHS
    rhs  eq1  5.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 1);
        assert_eq!(lp.constraint_types, vec![ConstraintType::Eq]);
        assert_eq!(lp.b, vec![5.0]);
    }

    #[test]
    fn test_parse_ge_constraint() {
        let mps = r"NAME test3
ROWS
 N  obj
 G  ge1
COLUMNS
    x1  obj  1.0  ge1  1.0
RHS
    rhs  ge1  3.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.constraint_types, vec![ConstraintType::Ge]);
        assert_eq!(lp.b, vec![3.0]);
    }

    #[test]
    fn test_parse_mixed_constraints() {
        let mps = r"NAME mixed
ROWS
 N  obj
 L  c1
 G  c2
 E  c3
COLUMNS
    x1  obj  1.0  c1  1.0
    x1  c2  2.0  c3  3.0
RHS
    rhs  c1  10.0  c2  20.0
    rhs  c3  30.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 3);
        assert_eq!(
            lp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Ge, ConstraintType::Eq]
        );
        assert_eq!(lp.b, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_rhs_odd_trailing_token_has_name_without_value_error() {
        let mps = r"NAME odd_rhs
ROWS
 N obj
 L c1
COLUMNS
    x1 obj 1.0 c1 1.0
RHS
    rhs c1 10.0 c2
ENDATA
";
        let err = parse_mps(mps).unwrap_err();
        assert!(
            err.to_string()
                .contains("has a name without a matching value"),
            "odd RHS token must be rejected, got {err:?}"
        );
    }

    #[test]
    fn test_parse_bounds_lo_up() {
        let mps = r"NAME bounds1
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 LO BND  x1  2.0
 UP BND  x1  8.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(2.0, 8.0)]);
    }

    #[test]
    fn test_parse_bounds_fx() {
        let mps = r"NAME bounds2
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 FX BND  x1  5.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(5.0, 5.0)]);
    }

    #[test]
    fn test_parse_bounds_fr() {
        let mps = r"NAME bounds3
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 FR BND  x1
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(f64::NEG_INFINITY, f64::INFINITY)]);
    }

    #[test]
    fn test_parse_bounds_mi() {
        let mps = r"NAME bounds4
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 MI BND  x1
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.bounds, vec![(f64::NEG_INFINITY, f64::INFINITY)]);
    }

    /// Sentinel: `MI` takes no value; a trailing token past the column name
    /// (`MI BND x1 5.0`) must be a hard error, not a silently discarded value.
    /// Confirmed to fail without the check in `parse_bounds_entry`
    /// (`otspot-io/src/common/mod.rs`): the free-format branch used to derive
    /// `raw` purely from `value_required`, so it never looked at — let alone
    /// rejected — a token sitting past the column name for a non-value type.
    #[test]
    fn test_mps_bounds_mi_surplus_value_token_is_error() {
        let mps = r"NAME bounds_mi_surplus
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 MI BND  x1  5.0
ENDATA
";
        let err = parse_mps(mps).unwrap_err();
        assert!(
            err.to_string().contains("does not take a value"),
            "MI with a surplus token must be rejected, got {err:?}"
        );
    }

    /// Sentinel: `UP` takes exactly one value; a second trailing token past
    /// the value (`UP BND x1 5.0 10.0`) must be a hard error, not a silently
    /// discarded value. Confirmed to fail without the check in
    /// `parse_bounds_entry` (`otspot-io/src/common/mod.rs`): the free-format
    /// branch used to read only `tokens[value_idx]`, so any token past it was
    /// silently dropped.
    #[test]
    fn test_mps_bounds_up_surplus_value_token_is_error() {
        let mps = r"NAME bounds_up_surplus
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
BOUNDS
 UP BND  x1  5.0  10.0
ENDATA
";
        let err = parse_mps(mps).unwrap_err();
        assert!(
            err.to_string().contains("takes exactly one value"),
            "UP with a surplus value token must be rejected, got {err:?}"
        );
    }

    #[test]
    fn test_parse_ranges() {
        let mps = r"NAME ranges
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
RANGES
    rng  c1  5.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.b[0], 10.0);
        assert_eq!(lp.b[1], 5.0);
    }

    #[test]
    fn test_parse_multiple_rhs_entries() {
        let mps = r"NAME multi_rhs
ROWS
 N  obj
 L  c1
 L  c2
COLUMNS
    x1  obj  1.0  c1  1.0
    x2  obj  2.0  c2  1.0
RHS
    rhs  c1  10.0
    rhs  c2  20.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.b, vec![10.0, 20.0]);
    }

    #[test]
    fn test_parse_two_entries_per_line() {
        let mps = r"NAME two_per_line
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  2.0
RHS
    rhs  c1  10.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_vars, 1);
        let (rows, vals) = lp.a.get_column(0).unwrap();
        assert_eq!(rows, &[0]);
        assert_eq!(vals, &[2.0]);
    }

    #[test]
    fn test_parse_error_missing_endata() {
        let mps = r"NAME test
ROWS
 N  obj
COLUMNS
    x1  obj  1.0
";
        let result = parse_mps(mps);
        assert!(result.is_err());
        match result {
            Err(MpsError::MissingSection(s)) => assert_eq!(s, "ENDATA"),
            _ => panic!("Expected MissingSection error"),
        }
    }

    #[test]
    fn test_parse_error_invalid_row_type() {
        let mps = r"NAME test
ROWS
 N  obj
 X  bad
COLUMNS
    x1  obj  1.0
ENDATA
";
        let result = parse_mps(mps);
        assert!(result.is_err());
        match result {
            Err(MpsError::InvalidRowType('X')) => {}
            _ => panic!("Expected InvalidRowType error"),
        }
    }

    #[test]
    fn test_range_le_basic() {
        let mps = r"NAME range_le
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
RANGES
    rhs  c1  2.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(
            lp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Ge]
        );
        assert_eq!(lp.b[0], 10.0);
        assert_eq!(lp.b[1], 8.0);
    }

    #[test]
    fn test_range_ge_basic() {
        let mps = r"NAME range_ge
ROWS
 N  obj
 G  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  5.0
RANGES
    rhs  c1  3.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(
            lp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Ge]
        );
        assert_eq!(lp.b[0], 8.0);
        assert_eq!(lp.b[1], 5.0);
    }

    #[test]
    fn test_range_eq_positive() {
        let mps = r"NAME range_eq_pos
ROWS
 N  obj
 E  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  7.0
RANGES
    rhs  c1  2.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(
            lp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Ge]
        );
        assert_eq!(lp.b[0], 9.0);
        assert_eq!(lp.b[1], 7.0);
    }

    #[test]
    fn test_range_eq_negative() {
        let mps = r"NAME range_eq_neg
ROWS
 N  obj
 E  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  7.0
RANGES
    rhs  c1  -2.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(
            lp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Ge]
        );
        assert_eq!(lp.b[0], 7.0);
        assert_eq!(lp.b[1], 5.0);
    }

    #[test]
    fn test_range_solve_simple() {
        use otspot_core::problem::SolveStatus;
        use otspot_core::solve;

        let mps = r"NAME range_solve
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
    x2  obj  1.0  c1  1.0
RHS
    rhs  c1  7.0
RANGES
    rhs  c1  4.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_constraints, 2);
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal, "should reach Optimal");
        assert!(
            (result.objective - 3.0).abs() < 1e-6,
            "expected obj=3.0, got {}",
            result.objective
        );
    }

    #[test]
    fn test_integer_marker_kind_intorg_intend() {
        use types::{integer_marker_kind, IntegerMarker};
        assert_eq!(
            integer_marker_kind("    M1 'MARKER' 'INTORG'"),
            Some(IntegerMarker::Start)
        );
        assert_eq!(
            integer_marker_kind("    M2 'MARKER' 'INTEND'"),
            Some(IntegerMarker::End)
        );
        assert_eq!(
            integer_marker_kind("    m 'marker' intorg"),
            Some(IntegerMarker::Start)
        );
    }

    #[test]
    fn test_integer_marker_kind_non_marker() {
        use types::integer_marker_kind;
        assert_eq!(integer_marker_kind("    x1  obj  1.0  c1  2.0"), None);
        assert_eq!(integer_marker_kind("    INTORG  obj  1.0"), None);
    }

    /// Codex review R3 (common/mod.rs:386, "MARKER token over-matching"): a
    /// legitimate free-format COLUMNS row for a column named `MARKER`
    /// referencing a row named `INTORG` has the same three tokens as a
    /// directive but in the wrong slots -- the marker's own name is field 1,
    /// so `MARKER` never belongs in field 2 for a real directive. Scanning
    /// every token for `MARKER`/`INTORG` regardless of position (pre-fix)
    /// misreads this as `IntegerMarker::Start`.
    ///
    /// Sentinel: reverting to the whole-line token scan makes this FAIL with
    /// `Some(IntegerMarker::Start)` instead of `None`.
    #[test]
    fn test_integer_marker_kind_column_row_name_collision_is_not_a_marker() {
        use types::integer_marker_kind;
        assert_eq!(integer_marker_kind("MARKER INTORG 1"), None);
        assert_eq!(integer_marker_kind("    MARKER  INTORG  1.0"), None);
    }

    #[test]
    fn test_milp_marker_no_bounds_is_binary() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 1.0)]);
    }

    /// Codex review R3 (common/mod.rs:386): a column literally named `MARKER`
    /// with a coefficient on a row literally named `INTORG` must parse as an
    /// ordinary coefficient, not be swallowed as a directive.
    ///
    /// Sentinel: reverting `integer_marker_kind` to the whole-line token scan
    /// makes this FAIL -- the `MARKER INTORG 1.0` line is consumed as a
    /// (spurious) `INTORG` marker, so `MARKER`'s coefficient on row `INTORG`
    /// is silently dropped and `av[1]` comes back `0.0` instead of `1.0`.
    #[test]
    fn test_marker_column_row_name_collision_keeps_coefficient() {
        let mps = r"NAME marker_collision
ROWS
 N  obj
 L  c1
 L  INTORG
COLUMNS
    MARKER  obj  1.0
    MARKER  INTORG  1.0
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
    rhs  INTORG  5.0
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!((lp.num_vars, lp.num_constraints), (2, 2));
        // Column order: MARKER (0), x1 (1); row order: c1 (0), INTORG (1).
        let av = lp.a.mat_vec_mul(&[1.0, 0.0]).unwrap();
        assert_eq!(
            av[1], 1.0,
            "MARKER's coefficient on row INTORG must survive parsing, got {av:?}"
        );
    }

    #[test]
    fn test_milp_marker_with_up_bound() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
BOUNDS
 UP BND  x1  5.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 5.0)]);
    }

    #[test]
    fn test_milp_marker_with_lo_only() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
BOUNDS
 LO BND  x1  2.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(2.0, f64::INFINITY)]);
    }

    #[test]
    fn test_milp_ui_bound_marks_integer() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
BOUNDS
 UI BND  x1  7.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 7.0)]);
    }

    #[test]
    fn test_milp_li_bound_marks_integer() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.5
BOUNDS
 LI BND  x1  2.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(2.0, f64::INFINITY)]);
    }

    #[test]
    fn test_milp_bv_bound_marks_integer() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
BOUNDS
 BV BND  x1
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 1.0)]);
    }

    #[test]
    fn test_milp_mixed_integer_continuous() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
    x2  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds[0], (0.0, 1.0));
        assert_eq!(milp.lp.bounds[1], (0.0, f64::INFINITY));
    }

    #[test]
    fn test_parse_mps_returns_relaxation_dropping_integrality() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
    M2 'MARKER' 'INTEND'
RHS
    rhs  c1  10.5
ENDATA
";
        let lp = parse_mps(mps).unwrap();
        assert_eq!(lp.num_vars, 1);
        assert_eq!(lp.bounds, vec![(0.0, 1.0)]);
    }

    #[test]
    fn test_milp_pure_lp_has_no_integers() {
        let mps = r"NAME lp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert!(milp.integer_vars.is_empty());
        assert_eq!(milp.lp.bounds, vec![(0.0, f64::INFINITY)]);
    }

    #[test]
    fn test_milp_fixed_format_marker() {
        let mps = "NAME          milp\n\
ROWS\n\
 N  obj\n\
 L  c1\n\
COLUMNS\n    \
MARKER1                 'MARKER'                 'INTORG'\n    \
x1        c1        1.0   obj       -1.0\n    \
MARKER2                 'MARKER'                 'INTEND'\n\
RHS\n    \
rhs       c1        10.5\n\
ENDATA\n";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
    }

    #[test]
    fn test_milp_integer_vars_sorted() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    a   obj  1.0  c1  1.0
    M1 'MARKER' 'INTORG'
    b   obj  1.0  c1  1.0
    c   obj  1.0  c1  1.0
    M2 'MARKER' 'INTEND'
    d   obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![1, 2]);
    }

    #[test]
    fn test_milp_solve_bound_conventions() {
        use otspot_core::options::{MipConfig, SolverOptions};
        use otspot_core::problem::SolveStatus;

        let cases: &[(&str, &str, f64, f64)] = &[
            ("marker_no_bounds_binary", "", 10.5, -1.0),
            (
                "marker_up5_fractional",
                "BOUNDS\n UP BND  x1  5.0\n",
                3.5,
                -3.0,
            ),
            ("marker_lo2", "BOUNDS\n LO BND  x1  2.0\n", 10.5, -10.0),
        ];

        for (label, bounds_section, rhs, expected_obj) in cases {
            let mps = format!(
                "NAME milp\n\
ROWS\n N  obj\n L  c1\n\
COLUMNS\n    M1 'MARKER' 'INTORG'\n    x1  obj  -1.0  c1  1.0\n    M2 'MARKER' 'INTEND'\n\
RHS\n    rhs  c1  {rhs}\n\
{bounds_section}ENDATA\n"
            );
            let milp = parse_milp(&mps).unwrap();
            let opts = SolverOptions::default();
            let cfg = MipConfig::default();
            let res = otspot_core::mip::solve_milp(&milp, &opts, &cfg);
            assert_eq!(
                res.status,
                SolveStatus::Optimal,
                "[{label}] should be Optimal"
            );
            assert!(
                (res.objective - expected_obj).abs() < 1e-6,
                "[{label}] expected obj={expected_obj}, got {}",
                res.objective
            );
        }
    }

    #[test]
    fn test_milp_solve_ui_bound() {
        use otspot_core::options::{MipConfig, SolverOptions};
        use otspot_core::problem::SolveStatus;

        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  3.5
BOUNDS
 UI BND  x1  7.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        let res =
            otspot_core::mip::solve_milp(&milp, &SolverOptions::default(), &MipConfig::default());
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.objective - (-3.0)).abs() < 1e-6,
            "expected -3, got {}",
            res.objective
        );
        assert!((res.solution[0] - 3.0).abs() < 1e-6, "x1 should be 3");
    }

    #[test]
    fn test_milp_unclosed_intorg_errors() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  -1.0  c1  1.0
RHS
    rhs  c1  10.5
ENDATA
";
        let err = parse_milp(mps).unwrap_err();
        assert!(
            matches!(err, MpsError::UnclosedIntegerMarker),
            "unclosed INTORG must error, got {err:?}"
        );
        assert!(matches!(
            parse_mps(mps).unwrap_err(),
            MpsError::UnclosedIntegerMarker
        ));
    }

    #[test]
    fn test_columns_free_format_misclassified_as_fixed() {
        let pad = " ".repeat(22 - 4 - "x#1#1".len());
        let mps = format!(
            "NAME wide\n\
ROWS\n N  obj\n L  c\n\
COLUMNS\n\
    x#1#1{pad}obj   -1.0\n\
    x#1#1{pad}c     1.0\n\
RHS\n    rhs{rpad}c     3.5\n\
BOUNDS\n UI BND  x#1#1  7\n\
ENDATA\n",
            rpad = " ".repeat(22 - 4 - "rhs".len()),
        );
        let milp = parse_milp(&mps).expect("wide-padded free-format COLUMNS must parse");
        assert_eq!(milp.num_vars(), 1);
        assert_eq!(milp.integer_vars, vec![0]);
        assert_eq!(milp.lp.bounds, vec![(0.0, 7.0)]);
        let (rows, vals) = milp.lp.a.get_column(0).unwrap();
        assert_eq!(rows, &[0]);
        assert_eq!(vals, &[1.0]);
    }

    #[test]
    fn test_columns_wide_padding_solves() {
        use otspot_core::options::{MipConfig, SolverOptions};
        use otspot_core::problem::SolveStatus;
        let pad = " ".repeat(22 - 4 - "x#1#1".len());
        let mps = format!(
            "NAME wide\n\
ROWS\n N  obj\n L  c\n\
COLUMNS\n\
    x#1#1{pad}obj   -1.0\n\
    x#1#1{pad}c     1.0\n\
RHS\n    rhs{rpad}c     3.5\n\
BOUNDS\n UI BND  x#1#1  7\n\
ENDATA\n",
            rpad = " ".repeat(22 - 4 - "rhs".len()),
        );
        let milp = parse_milp(&mps).unwrap();
        let res =
            otspot_core::mip::solve_milp(&milp, &SolverOptions::default(), &MipConfig::default());
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!(
            (res.objective - (-3.0)).abs() < 1e-6,
            "expected -3, got {}",
            res.objective
        );
    }

    #[test]
    fn test_milp_closed_intorg_following_cols_continuous() {
        let mps = r"NAME milp
ROWS
 N  obj
 L  c1
COLUMNS
    M1 'MARKER' 'INTORG'
    x1  obj  1.0  c1  1.0
    M2 'MARKER' 'INTEND'
    x2  obj  1.0  c1  1.0
    x3  obj  1.0  c1  1.0
RHS
    rhs  c1  10.0
ENDATA
";
        let milp = parse_milp(mps).unwrap();
        assert_eq!(milp.integer_vars, vec![0]);
    }

    const STREAM_MPS: &str = "NAME          stream\n\
ROWS\n N  obj\n L  c1\n\
COLUMNS\n    x1  obj  3.0  c1  1.0\n    x2  obj  5.0  c1  2.0\n\
RHS\n    rhs  c1  10.0\n\
ENDATA\n";

    #[test]
    fn test_mps_reader_round_trip() {
        let expected = parse_mps(STREAM_MPS).unwrap();
        let got = parse_mps_reader(std::io::Cursor::new(STREAM_MPS.as_bytes())).unwrap();
        assert_eq!(got.num_vars, expected.num_vars);
        assert_eq!(got.num_constraints, expected.num_constraints);
        assert_eq!(got.c, expected.c);
        assert_eq!(got.b, expected.b);
        assert_eq!(got.bounds, expected.bounds);
    }

    #[test]
    fn test_milp_reader_round_trip() {
        let mps = "NAME          m\nROWS\n N  obj\n L  c1\n\
COLUMNS\n    M1 'MARKER' 'INTORG'\n    x1  obj  -1.0  c1  1.0\n    M2 'MARKER' 'INTEND'\n\
RHS\n    rhs  c1  10.5\nENDATA\n";
        let expected = parse_milp(mps).unwrap();
        let got = parse_milp_reader(std::io::Cursor::new(mps.as_bytes())).unwrap();
        assert_eq!(got.integer_vars, expected.integer_vars);
        assert_eq!(got.lp.bounds, expected.lp.bounds);
    }

    #[test]
    fn test_mps_reader_crlf_equivalence() {
        let lf = parse_mps_reader(std::io::Cursor::new(STREAM_MPS.as_bytes())).unwrap();
        let crlf_src = STREAM_MPS.replace('\n', "\r\n");
        let crlf = parse_mps_reader(std::io::Cursor::new(crlf_src.as_bytes())).unwrap();
        assert_eq!(crlf.num_vars, lf.num_vars);
        assert_eq!(crlf.num_constraints, lf.num_constraints);
        assert_eq!(crlf.c, lf.c);
        assert_eq!(crlf.b, lf.b);
        assert_eq!(crlf.bounds, lf.bounds);
    }

    #[test]
    fn test_mps_reader_fixture_afiro() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/netlib/afiro.mps");
        let content = std::fs::read_to_string(&path).unwrap();
        let expected = parse_mps(&content).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let got = parse_mps_reader(std::io::BufReader::new(file)).unwrap();
        assert_eq!(got.num_vars, expected.num_vars);
        assert_eq!(got.num_constraints, expected.num_constraints);
        assert_eq!(got.c, expected.c);
        assert_eq!(got.b, expected.b);
    }

    use std::io::{self, Read, Seek};

    /// A seekable reader that counts how many lines it has been asked for.
    struct LineCountingReader {
        inner: io::Cursor<Vec<u8>>,
        line_call_count: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl LineCountingReader {
        fn new(text: &str) -> (Self, std::rc::Rc<std::cell::Cell<usize>>) {
            let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
            (
                Self {
                    inner: io::Cursor::new(text.as_bytes().to_vec()),
                    line_call_count: counter.clone(),
                },
                counter,
            )
        }
    }

    impl Read for LineCountingReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.inner.read(buf)
        }
    }

    impl Seek for LineCountingReader {
        fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
            self.inner.seek(pos)
        }
    }

    impl io::BufRead for LineCountingReader {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.inner.fill_buf()
        }
        fn consume(&mut self, amt: usize) {
            self.inner.consume(amt)
        }
        fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
            let n = self.inner.read_line(buf)?;
            if n > 0 {
                self.line_call_count.set(self.line_call_count.get() + 1);
            }
            Ok(n)
        }
    }

    /// The reader entry points must **stream**: one line is held at a time and
    /// lines are pulled only as the parser consumes them. Draining the stream
    /// into a `Vec<String>` first would hold the whole input in memory, and MPS
    /// files reach the GiB range (`data/miplib_2017/square47.mps` is 1.4 GiB).
    ///
    /// Counting `read_line` calls on a *successful* parse cannot detect that: a
    /// buffering reader issues exactly the same number of calls. The
    /// discriminator is an input that fails on line 3 of many thousands — a
    /// streaming parser stops pulling lines there, while a buffering one has
    /// already pulled every line before parsing even began.
    ///
    /// **No-op failure guarantee**: restoring the `Vec<String>` buffering in
    /// `LineSource::from_reader` makes the count jump from single digits to the
    /// full line count — verified by temporarily reverting.
    #[test]
    fn test_mps_reader_pulls_lines_lazily_not_all_upfront() {
        use std::fmt::Write as _;

        // Fails on line 3 (a ROWS line with no row name), then thousands of lines.
        const PADDING_LINES: usize = 3000;
        // Two format readings (free, then fixed) each stop at line 3, so a
        // streaming parser pulls ~6 lines; allow generous slack.
        const MAX_LINES_A_STREAMING_PARSER_PULLS: usize = 50;

        let mut mps = String::from("NAME\nROWS\n Z\n");
        for i in 0..PADDING_LINES {
            writeln!(mps, "* filler line {i}").expect("write to String");
        }
        mps.push_str("ENDATA\n");
        let total_lines = mps.lines().count();
        assert!(total_lines > PADDING_LINES);

        let (reader, counter) = LineCountingReader::new(&mps);
        parse_mps_reader(reader).expect_err("a ROWS line with no row name must fail");
        assert!(
            counter.get() <= MAX_LINES_A_STREAMING_PARSER_PULLS,
            "parser must pull lines lazily and stop at the failing line 3; it pulled {} of \
             {total_lines} lines, which means the reader was drained up front",
            counter.get()
        );
    }

    // ── Sentinel tests: audit 141 parser strictness (A/B/C) ───────────────────

    /// A: COLUMNS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_mps_columns_malformed_too_few_fields_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1  obj\nRHS\n    rhs c1 1.0\nENDATA\n";
        assert!(parse_mps(mps).is_err(), "< 3 fields in COLUMNS must error");
    }

    /// A: RHS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_mps_rhs_malformed_too_few_fields_is_error() {
        let mps =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    c1\nENDATA\n";
        assert!(parse_mps(mps).is_err(), "< 3 fields in RHS must error");
    }

    /// A: RANGES line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_mps_ranges_malformed_too_few_fields_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nRANGES\n    c1\nENDATA\n";
        assert!(parse_mps(mps).is_err(), "< 3 fields in RANGES must error");
    }

    /// A: BOUNDS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_mps_bounds_malformed_too_few_fields_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nBOUNDS\n LO\nENDATA\n";
        assert!(parse_mps(mps).is_err(), "< 3 fields in BOUNDS must error");
    }

    /// Duplicate (col, row) entries in COLUMNS must accumulate (sum), not error.
    /// MPS spec allows repeated entries; CscMatrix::from_triplets merges them.
    #[test]
    fn test_parse_mps_accumulates_duplicate_objective_entries() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\n    x1 obj 2.0\nRHS\n    rhs c1 10.0\nENDATA\n";
        let lp = parse_mps(mps).expect("duplicate objective entries must parse OK");
        assert_eq!(lp.num_vars, 1);
        assert!(
            (lp.c[0] - 3.0).abs() < 1e-10,
            "1.0 + 2.0 = 3.0, got {}",
            lp.c[0]
        );
    }

    /// C: NaN coefficient in COLUMNS must be an error.
    #[test]
    fn test_mps_columns_nan_value_is_error() {
        let mps =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 NaN\nRHS\n    rhs c1 10.0\nENDATA\n";
        let err = parse_mps(mps);
        assert!(
            err.is_err(),
            "NaN coefficient in COLUMNS must error: {:?}",
            err
        );
    }

    /// C: Inf coefficient in COLUMNS must be an error.
    #[test]
    fn test_mps_columns_inf_value_is_error() {
        let mps =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 Inf\nRHS\n    rhs c1 10.0\nENDATA\n";
        let err = parse_mps(mps);
        assert!(
            err.is_err(),
            "Inf coefficient in COLUMNS must error: {:?}",
            err
        );
    }

    /// C: NaN in RHS value must be an error.
    #[test]
    fn test_mps_rhs_nan_value_is_error() {
        let mps =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 1.0\nRHS\n    rhs c1 NaN\nENDATA\n";
        let err = parse_mps(mps);
        assert!(err.is_err(), "NaN in RHS must error: {:?}", err);
    }

    /// C: NaN in BOUNDS value must be an error.
    #[test]
    fn test_mps_bounds_nan_value_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nBOUNDS\n UP BND x1 NaN\nENDATA\n";
        let err = parse_mps(mps);
        assert!(err.is_err(), "NaN in BOUNDS must error: {:?}", err);
    }

    /// N-row RHS with NaN must be a parse error (constraint-row symmetry).
    #[test]
    fn test_mps_rhs_n_row_nan_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 1.0\nRHS\n    rhs obj NaN\n    rhs c1 1.0\nENDATA\n";
        let result = parse_mps(mps);
        assert!(
            result.is_err(),
            "N-row RHS NaN must be rejected: {:?}",
            result
        );
    }

    /// N-row RHS with a finite value must propagate to LpProblem.obj_offset.
    #[test]
    fn test_mps_rhs_n_row_finite_propagates_to_obj_offset() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs obj 42.5\n    rhs c1 10.0\nENDATA\n";
        let lp = parse_mps(mps).expect("valid MPS with N-row RHS");
        assert!(
            (lp.obj_offset - 42.5).abs() < 1e-12,
            "obj_offset must equal N-row RHS 42.5, got {}",
            lp.obj_offset
        );
    }

    /// MPS OBJSENSE MAX with N-row RHS must sign-flip obj_offset (MAX→MIN negation).
    ///
    /// Sentinel: removing the `if self.maximize { -raw } else { raw }` sign-flip in
    /// `mps/parser.rs` causes `obj_offset == +10.0` instead of `-10.0` → FAIL.
    #[test]
    fn test_mps_objsense_max_obj_offset_sign_flip() {
        let mps = concat!(
            "NAME  MAX_OFFSET\n",
            "OBJSENSE\n",
            "    MAX\n",
            "ROWS\n",
            " N  obj\n",
            " L  c1\n",
            "COLUMNS\n",
            "    x1    obj    1.0    c1    1.0\n",
            "RHS\n",
            "    rhs   obj    10.0\n",
            "    rhs   c1    5.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect("valid MPS with OBJSENSE MAX + N-row RHS");
        assert!(
            (lp.obj_offset - (-10.0)).abs() < 1e-12,
            "OBJSENSE MAX with N-row RHS=10.0 must yield obj_offset=-10.0; got {}",
            lp.obj_offset,
        );
    }

    /// MPS N-row RHS (obj_offset) must appear in the solve result objective end-to-end.
    ///
    /// Problem: min x  s.t. x <= 5,  x >= 0,  N-row RHS = 10.0
    /// Optimal: x* = 0,  c^T x* = 0,  result.objective = 0 + 10.0 = 10.0.
    ///
    /// Sentinel: removing `result.objective += problem.obj_offset` from
    /// `lp::solve_lp_with` causes result.objective == 0.0 ≠ 10.0 → FAIL.
    #[test]
    fn test_mps_obj_offset_propagates_to_solve_result() {
        use otspot_core::lp::solve_lp_with;
        use otspot_core::problem::SolveStatus;

        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs obj 10.0\n    rhs c1 5.0\nENDATA\n";
        let lp = parse_mps(mps).expect("valid MPS with N-row RHS=10.0");
        let result = solve_lp_with(&lp, &Default::default());
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - 10.0).abs() < 1e-9,
            "expected objective 10.0 (c^Tx*=0 + offset 10.0), got {}",
            result.objective
        );
    }

    // ── Sentinel tests: input-validation audit ────────────────────────────────

    /// Fix-4: value-bearing BOUNDS type (LO) with missing value must error in MPS.
    /// Sentinel: reverting the value_required check → Ok instead of Err.
    #[test]
    fn test_sentinel_mps_bounds_lo_missing_value_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nBOUNDS\n LO BND x1\nENDATA\n";
        assert!(
            parse_mps(mps).is_err(),
            "LO bound without a value must error in MPS"
        );
    }

    /// Fix-4: value-bearing BOUNDS type (FX) with missing value must error in MPS.
    #[test]
    fn test_sentinel_mps_bounds_fx_missing_value_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nBOUNDS\n FX BND x1\nENDATA\n";
        assert!(
            parse_mps(mps).is_err(),
            "FX bound without a value must error in MPS"
        );
    }

    /// Fix-4: value-bearing BOUNDS type (UI) with missing value must error in MPS.
    #[test]
    fn test_sentinel_mps_bounds_ui_missing_value_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nBOUNDS\n UI BND x1\nENDATA\n";
        assert!(
            parse_milp(mps).is_err(),
            "UI bound without a value must error in MPS"
        );
    }

    /// Fix-5: odd trailing token in COLUMNS (row name with no value) must error in MPS.
    /// Sentinel: reverting the break→error → Ok instead of Err.
    #[test]
    fn test_sentinel_mps_columns_trailing_row_no_value_is_error() {
        let mps =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1\nRHS\n    rhs c1 5.0\nENDATA\n";
        assert!(
            parse_mps(mps).is_err(),
            "trailing row name without a value in COLUMNS must error in MPS"
        );
    }

    // ── Black-box parse tests ─────────────────────────────────────────────────

    /// TECHNIQUE: EQUIVALENCE PARTITIONING — one MPS file exercises all three row
    /// types (Le/Ge/Eq) simultaneously, plus the N (objective) row.
    ///
    /// Oracle (hand-derived from MPS text):
    ///   Rows: le1 (L→Le), ge1 (G→Ge), eq1 (E→Eq).
    ///   Columns: x1 → c[0]=1, x2 → c[1]=2.
    ///   A: le1: x1*2+x2*3=10; ge1: x1*1+x2*2=4; eq1: x1*1+x2*1=3.
    ///   b: [10, 4, 3]. Default bounds: (0, +inf).
    #[test]
    fn ep_mps_all_row_types() {
        let mps = "\
NAME          ep_all_row_types
ROWS
 N  obj
 L  le1
 G  ge1
 E  eq1
COLUMNS
    x1  obj  1.0  le1  2.0
    x1  ge1  1.0  eq1  1.0
    x2  obj  2.0  le1  3.0
    x2  ge1  2.0  eq1  1.0
RHS
    rhs  le1  10.0
    rhs  ge1  4.0
    rhs  eq1  3.0
ENDATA
";
        let lp = parse_mps(mps).expect("ep_mps_all_row_types: valid MPS must parse");
        assert_eq!(lp.num_vars, 2, "num_vars");
        assert_eq!(lp.num_constraints, 3, "num_constraints");
        assert_eq!(lp.c, vec![1.0, 2.0], "objective coefficients");
        assert_eq!(
            lp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Ge, ConstraintType::Eq],
            "constraint types: L→Le, G→Ge, E→Eq"
        );
        assert_eq!(lp.b, vec![10.0, 4.0, 3.0], "RHS values");
        assert_eq!(
            lp.bounds,
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            "default bounds (0, +inf)"
        );
    }

    /// TECHNIQUE: BOUNDARY VALUE ANALYSIS — bound types at the extremes:
    /// FX (fixed, lb==ub), FR (free, lb=-inf/ub=+inf), LO+UP (explicit range).
    ///
    /// Oracle (hand-derived):
    ///   x1: FX 3.0 → bounds (3.0, 3.0).
    ///   x2: FR    → bounds (-inf, +inf).
    ///   x3: LO 1.5 + UP 8.0 → bounds (1.5, 8.0).
    #[test]
    fn bva_mps_bounds_fx_fr_lo_up() {
        let mps = "\
NAME          bva_bounds
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
    x2  obj  1.0  c1  1.0
    x3  obj  1.0  c1  1.0
RHS
    rhs  c1  20.0
BOUNDS
 FX BND  x1  3.0
 FR BND  x2
 LO BND  x3  1.5
 UP BND  x3  8.0
ENDATA
";
        let lp = parse_mps(mps).expect("bva_mps_bounds: valid MPS must parse");
        assert_eq!(lp.num_vars, 3, "num_vars");
        assert_eq!(lp.bounds[0], (3.0, 3.0), "x1: FX 3.0 → (3,3)");
        assert_eq!(lp.bounds[1].0, f64::NEG_INFINITY, "x2: FR → lb = -inf");
        assert_eq!(lp.bounds[1].1, f64::INFINITY, "x2: FR → ub = +inf");
        assert_eq!(lp.bounds[2], (1.5, 8.0), "x3: LO 1.5 UP 8.0 → (1.5,8.0)");
    }

    /// TECHNIQUE: BOUNDARY VALUE ANALYSIS — RHS = 0 (zero boundary).
    ///
    /// Oracle: single Le constraint with RHS=0. b[0]=0.0 after parse.
    /// Non-trivial because a zero-valued token must not be skipped or misread.
    #[test]
    fn bva_mps_rhs_zero() {
        let mps = "\
NAME          bva_rhs_zero
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  0.0
ENDATA
";
        let lp = parse_mps(mps).expect("bva_mps_rhs_zero: valid MPS must parse");
        assert_eq!(lp.num_constraints, 1, "num_constraints");
        assert_eq!(lp.b, vec![0.0], "RHS = 0.0 must be preserved");
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le]);
    }

    /// TECHNIQUE: DECISION TABLE — 2-var × (Le/Ge/Eq) matrix-structure check.
    ///
    /// Oracle (hand-derived from MPS text):
    ///   A matrix (3×2), rows in order le1=0, ge1=1, eq1=2:
    ///   A[0,0]=2, A[0,1]=3 (le1)
    ///   A[1,0]=1, A[1,1]=2 (ge1)
    ///   A[2,0]=1, A[2,1]=1 (eq1)
    ///   Verified by get_column queries on the parsed CSC matrix.
    #[test]
    fn dt_mps_2var_le_ge_eq_matrix_structure() {
        let mps = "\
NAME          dt_2var_matrix
ROWS
 N  obj
 L  le1
 G  ge1
 E  eq1
COLUMNS
    x1  obj  1.0  le1  2.0
    x1  ge1  1.0  eq1  1.0
    x2  obj  2.0  le1  3.0
    x2  ge1  2.0  eq1  1.0
RHS
    rhs  le1  10.0  ge1  4.0
    rhs  eq1  3.0
ENDATA
";
        let lp = parse_mps(mps).expect("dt_mps_2var_matrix: valid MPS must parse");
        assert_eq!(lp.num_vars, 2);
        assert_eq!(lp.num_constraints, 3);

        // Column 0 (x1): rows 0,1,2 with values 2,1,1
        let (rows0, vals0) = lp.a.get_column(0).expect("col 0");
        let col0: std::collections::HashMap<usize, f64> =
            rows0.iter().copied().zip(vals0.iter().copied()).collect();
        assert!((col0[&0] - 2.0).abs() < 1e-12, "A[le1,x1]=2.0");
        assert!((col0[&1] - 1.0).abs() < 1e-12, "A[ge1,x1]=1.0");
        assert!((col0[&2] - 1.0).abs() < 1e-12, "A[eq1,x1]=1.0");

        // Column 1 (x2): rows 0,1,2 with values 3,2,1
        let (rows1, vals1) = lp.a.get_column(1).expect("col 1");
        let col1: std::collections::HashMap<usize, f64> =
            rows1.iter().copied().zip(vals1.iter().copied()).collect();
        assert!((col1[&0] - 3.0).abs() < 1e-12, "A[le1,x2]=3.0");
        assert!((col1[&1] - 2.0).abs() < 1e-12, "A[ge1,x2]=2.0");
        assert!((col1[&2] - 1.0).abs() < 1e-12, "A[eq1,x2]=1.0");
    }

    // -----------------------------------------------------------------------
    // PR #25 review horizontal expansion: RHS/RANGES duplicate-row detection.
    //
    // Unlike COLUMNS (which accumulates duplicate (row,col) entries by design,
    // see `test_parse_mps_accumulates_duplicate_objective_entries`), RHS and
    // RANGES hold exactly one scalar per row; a repeated row name is
    // ambiguous input that was previously silently resolved via last-write-wins.
    // -----------------------------------------------------------------------

    /// Sentinel: the same row name appearing twice in RHS (across lines or on
    /// one multi-pair line) must be a `ParseError`, not silently overwritten.
    ///
    /// **No-op failure guarantee**: reverting to plain `self.rhs.insert(name, value)`
    /// makes this parse succeed with `lp.b[0] == 20.0` (last-write-wins) instead of erroring.
    #[test]
    fn test_mps_duplicate_rhs_row_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\n    rhs c1 20.0\nENDATA\n";
        let err = parse_mps(mps).expect_err("duplicate RHS row must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("RHS") && msg.contains("duplicate"),
            "error should mention RHS duplicate, got: {msg}"
        );
    }

    /// Sentinel: the same row name appearing twice in RANGES must be a `ParseError`.
    ///
    /// **No-op failure guarantee**: reverting to plain `self.ranges.insert(name, value)`
    /// makes this parse succeed (silently keeping the last RANGES value) instead of erroring.
    #[test]
    fn test_mps_duplicate_ranges_row_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nRANGES\n    rng c1 2.0\n    rng c1 4.0\nENDATA\n";
        let err = parse_mps(mps).expect_err("duplicate RANGES row must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("RANGES") && msg.contains("duplicate"),
            "error should mention RANGES duplicate, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // vector-name-omitted (shorthand) RHS/RANGES/BOUNDS + multi-vector
    // RHS/RANGES support.
    // -----------------------------------------------------------------------

    /// Sentinel: a 2-pair-per-line RHS with the vector name omitted, using
    /// numeric row names ("1", "2") that also look like plausible values —
    /// this is the exact shape that historically broke: a naive parser skips
    /// `parts[0]` as if it were a vector name, then mispairs
    /// `(parts[1], parts[2])` = `("10.0", 2.0)`, both silently discarding row
    /// "1"'s value and fabricating a bogus entry for a nonexistent row "10.0".
    ///
    /// **No-op failure guarantee**: reverting `parse_rhs_line` to the old
    /// unconditional `parse_mps_free_pairs` (always skip `parts[0]`) makes
    /// `lp.b` become `[0.0, 2.0]` instead of `[10.0, 20.0]` (row "1"'s RHS
    /// silently lost) — verified by temporarily reverting during development.
    #[test]
    fn test_mps_rhs_shorthand_two_pairs_numeric_row_names() {
        let mps = "NAME\nROWS\n N obj\n L 1\n L 2\nCOLUMNS\n    x1 obj 1.0 1 1.0\n    x2 obj 1.0 2 1.0\nRHS\n    1  10.0  2  20.0\nENDATA\n";
        let lp = parse_mps(mps).expect("shorthand 2-pair RHS must parse");
        assert_eq!(lp.b, vec![10.0, 20.0], "both RHS values must survive");
    }

    /// Sentinel: `blend_shorthand.mps` is a real, live `emps`-decoded Netlib
    /// LP "blend" RHS section — numeric row names ("65".."72"), no vector
    /// name, 2 pairs packed per line. This is the historical trigger: an
    /// earlier attempt at multi-vector RHS support misread this shape and
    /// silently dropped all but the first row's value, corrupting `b` and
    /// the solved objective. Both the direct RHS values and the solved
    /// objective (matching Netlib's published optimum, also recorded in
    /// `data/baseline_objectives/netlib_lp.csv` as `blend,-3.0812149846e+01`)
    /// serve as independent oracles here.
    ///
    /// **No-op failure guarantee**: reverting the shorthand disambiguation
    /// makes only 2 of the 8 RHS rows survive (rows "65"/"66" from the first
    /// line, whichever vector identity gets established first) and the
    /// solved objective becomes wrong — verified by temporarily reverting.
    #[test]
    fn test_mps_blend_shorthand_rhs_matches_known_optimal() {
        use otspot_core::solve;

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/netlib/blend_shorthand.mps");
        let lp = parse_mps_file(&path).expect("blend_shorthand.mps must parse");
        assert_eq!(lp.num_constraints, 74, "43 E-rows + 31 L-rows, no N-row");

        // Independent oracle #1: the 8 RHS entries, hand-read off the fixture
        // file's `RHS` section, at their 0-indexed ROWS-declaration position
        // (rows "1".."74" declared in order, row "N" is 65th..72nd => index 64..71).
        let expected_rhs: [(usize, f64); 8] = [
            (64, 23.26),
            (65, 5.25),
            (66, 26.32),
            (67, 21.05),
            (68, 13.45),
            (69, 2.58),
            (70, 10.0),
            (71, 10.0),
        ];
        for &(idx, val) in &expected_rhs {
            assert!(
                (lp.b[idx] - val).abs() < 1e-9,
                "b[{idx}] must be {val}, got {}",
                lp.b[idx]
            );
        }
        let nonzero_count = lp.b.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            nonzero_count, 8,
            "exactly 8 RHS rows are nonzero in blend; got {nonzero_count} (silent data loss?)"
        );

        // Independent oracle #2: Netlib's published optimal objective.
        let result = solve(&lp);
        assert!(
            (result.objective - (-30.812149846)).abs() < 1e-6,
            "blend_shorthand objective must match Netlib optimal -30.812149846, got {}",
            result.objective
        );
    }

    /// Sentinel (multi-vector RHS, INLINE-P / PR #25 review finding): two
    /// distinct NAMED RHS vectors legitimately reusing the same row must
    /// parse successfully, applying only the first vector's value
    /// (GLPK/CPLEX "first vector wins" convention) — not be rejected as a
    /// row-only duplicate.
    ///
    /// **No-op failure guarantee**: reverting to a plain
    /// `if self.rhs.contains_key(&name) { error }` row-only duplicate check
    /// (ignoring vector identity) makes this a `ParseError` instead of
    /// `Ok` — verified by temporarily reverting.
    #[test]
    fn test_mps_rhs_multiple_named_vectors_first_wins() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    RHS1  c1  10.0\n    RHS2  c1  20.0\nENDATA\n";
        let lp = parse_mps(mps).expect("distinct named RHS vectors reusing a row must parse");
        assert_eq!(lp.b, vec![10.0], "first vector (RHS1) must win, not RHS2");
    }

    /// Sentinel (multi-vector RANGES): analogous to the RHS case above.
    #[test]
    fn test_mps_ranges_multiple_named_vectors_first_wins() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nRANGES\n    RNG1  c1  2.0\n    RNG2  c1  4.0\nENDATA\n";
        let lp = parse_mps(mps).expect("distinct named RANGES vectors reusing a row must parse");
        // RNG1=2.0 applied (b0-|2|..b0): Le row c1, b=10.0, range=2.0 → [8.0, 10.0].
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(lp.b[0], 10.0);
        assert_eq!(
            lp.b[1], 8.0,
            "RANGES value must come from RNG1 (2.0), not RNG2 (4.0)"
        );
    }

    /// PR #25 review (Codex): a row name that appears only in a *discarded*
    /// (non-first) RHS vector must still be checked against ROWS. `record`
    /// only writes a discarded vector's entries into `seen`, not into
    /// `target` (the map `validate_references` used to check alone), so an
    /// undeclared row confined to RHS2 used to slip past strict-reference
    /// validation entirely.
    ///
    /// Sentinel: reverting `validate_references` to check only
    /// `self.rhs.keys()` (dropping the `rhs_vectors.referenced_rows()` chain)
    /// makes this `Ok` instead of `Err` — verified by temporarily reverting.
    #[test]
    fn test_mps_rhs_second_vector_undefined_row_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    RHS1  c1  10.0\n    RHS2  ghost  20.0\nENDATA\n";
        let err =
            parse_mps(mps).expect_err("RHS2 (discarded) referencing an undeclared row must error");
        assert!(
            matches!(err, MpsError::UndefinedReference { ref kind, ref name } if kind == "row" && name == "ghost"),
            "expected UndefinedReference{{kind: row, name: ghost}}, got {err:?}"
        );
    }

    /// Analogous to the RHS case above, for RANGES: an undeclared row
    /// confined to the discarded second RANGES vector must still error.
    #[test]
    fn test_mps_ranges_second_vector_undefined_row_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nRANGES\n    RNG1  c1  2.0\n    RNG2  ghost  4.0\nENDATA\n";
        let err = parse_mps(mps)
            .expect_err("RANGES2 (discarded) referencing an undeclared row must error");
        assert!(
            matches!(err, MpsError::UndefinedReference { ref kind, ref name } if kind == "row" && name == "ghost"),
            "expected UndefinedReference{{kind: row, name: ghost}}, got {err:?}"
        );
    }

    /// A strict fixed-column MPS whose ROWS / COLUMNS / RHS all use a row name
    /// containing embedded spaces (`"BR   1 1"`, the shape real Netlib
    /// `forplan` uses), with a **single-token** RHS vector name (`RHS`).
    ///
    /// This is the case no per-line heuristic can get right. Tokenized, the
    /// RHS line reads `[RHS, LC123, 10., BR, 1, 1, 6.]`: strip the vector name
    /// and six tokens remain — an even count whose value slots (`10.`, `1`,
    /// `6.`) all parse as floats. Every local "does this look like free
    /// format?" test therefore says yes, and the line is read as the rows
    /// `LC123=10`, `BR=1`, `1=6` — inventing two rows and losing the 6.0 that
    /// belongs to `BR   1 1`. Only deciding the layout for the whole file, and
    /// hard-erroring on the undeclared names `BR` and `1`, rejects that read.
    ///
    /// Independent oracle: hand-solved LP. `min -x1-x2` s.t. `x1+x2 <= 10`
    /// (LC123), `x1 <= 6` (`BR   1 1`), `0 <= x2 <= 2` (BOUNDS), `x1 >= 0`.
    /// The optimum is `x1=6, x2=2` → objective `-8.0`, which discriminates each
    /// failure mode: dropping the `BR   1 1` COLUMNS coefficient frees `x1` to
    /// 8 (objective `-10`), dropping its RHS pins `x1` to 0 (objective `-2`).
    ///
    /// **No-op failure guarantee**: with the per-line format heuristics
    /// restored, the free-format read of the RHS line succeeds and `b` becomes
    /// `[10.0, 0.0]` — verified by temporarily reverting.
    #[test]
    fn test_mps_fixed_columns_embedded_space_row_name() {
        use otspot_core::solve;

        // Fixed columns: name 5-12, row 15-22, value 25-36, row 40-47, value 50-61.
        let mps = concat!(
            "NAME          FIXEDCOL\n",
            "ROWS\n",
            " N  obj\n",
            " L  LC123\n",
            " L  BR   1 1\n",
            "COLUMNS\n",
            "    X1        obj               -1.0   LC123              1.0\n",
            "    X1        BR   1 1           1.0\n",
            "    X2        obj               -1.0   LC123              1.0\n",
            "RHS\n",
            "    RHS       LC123              10.   BR   1 1            6.\n",
            "BOUNDS\n",
            " UP BND       X2                 2.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect("fixed-column MPS with embedded-space row name must parse");

        assert_eq!(lp.num_vars, 2, "X1, X2");
        assert_eq!(lp.num_constraints, 2, "LC123, 'BR   1 1'");
        assert_eq!(
            lp.b,
            vec![10.0, 6.0],
            "'BR   1 1' must keep its RHS 6.0, not lose it to an invented row"
        );

        let result = solve(&lp);
        assert!(
            (result.objective - (-8.0)).abs() < 1e-6,
            "hand-solved optimum is -8.0 (x1=6 capped by 'BR   1 1', x2=2 by BOUNDS); \
             -10 means the COLUMNS coefficient on 'BR   1 1' was dropped, \
             -2 means its RHS was dropped. got {}",
            result.objective
        );
    }

    /// The fixed-column retry must not become a second way to misread a file.
    ///
    /// A fixed-column name field holds 8 bytes. Silently truncating a name that
    /// overflows it lets a file whose free reading legitimately failed be
    /// re-read as fixed-column with every name clipped — and clipped names can
    /// make a broken file look whole. Here ROWS declares `ROWLONGNAME1` on the
    /// fixed grid while COLUMNS and RHS reference the typo `ROWLONGN`: clipping
    /// the declaration to 8 bytes yields exactly `ROWLONGN`, the typo resolves,
    /// and a plausible-but-wrong model is returned. Both readings must fail
    /// instead — the free one on the undeclared `ROWLONGN`, the fixed one
    /// because the name runs past its field.
    ///
    /// **No-op failure guarantee**: making the grid check accept an overflowing
    /// name (truncating instead) makes this parse succeed and solve — verified
    /// by temporarily reverting.
    #[test]
    fn test_mps_long_name_is_not_truncated_into_a_false_match() {
        // Column-aligned, so that a truncating reader really would clip
        // `ROWLONGNAME1` to `ROWLONGN` and match the typo.
        let typo = concat!(
            "NAME          FIXEDCOL\n",
            "ROWS\n",
            " N  obj\n",
            " L  ROWLONGNAME1\n",
            "COLUMNS\n",
            "    x1        obj                1.0\n",
            "    x1        ROWLONGN           1.0\n",
            "RHS\n",
            "    rhs       ROWLONGN           4.0\n",
            "ENDATA\n",
        );
        let err = parse_mps(typo)
            .expect_err("a reference to the undeclared 'ROWLONGN' must not resolve by truncation");
        assert!(
            format!("{err}").contains("ROWLONGN"),
            "the error must name the unresolved reference, got: {err}"
        );

        // The same guard from the other side: a file that genuinely needs the
        // fixed-column reader (its free reading dies on the embedded-space row
        // name) but whose other name overflows the field.
        let overflowing = concat!(
            "NAME          FIXEDCOL\n",
            "ROWS\n",
            " N  obj\n",
            " L  BR   1 1\n",
            " L  TOOLONGNAME\n",
            "COLUMNS\n",
            "    X1        BR   1 1           1.0\n",
            "RHS\n",
            "    RHS       BR   1 1            6.\n",
            "ENDATA\n",
        );
        assert!(
            parse_mps(overflowing).is_err(),
            "a name overflowing the fixed 8-byte field must error, not be truncated"
        );
    }

    /// Columns 62-80 of a fixed-column line are the comment field (62-72) and
    /// the sequence-number field (73-80). The MPS standard — and GLPK, lp_solve,
    /// IBM OSL — ignore everything past column 61, so their presence must not
    /// make a legal file unparseable.
    ///
    /// The guard that rejects an overflowing name must therefore look only at
    /// the data columns. Applying it to column 62 rejected the comment field:
    /// on a two-pair line the value ends at column 61 and the comment begins
    /// immediately at 62, so the line was read as "a name overflows its field"
    /// and the whole file failed (measured: `Parse error at line 6: COLUMNS has
    /// a name without a matching value`). One-pair lines happened to escape,
    /// which is why the corpus never caught it.
    ///
    /// Both line shapes are exercised here, together with an embedded-space row
    /// name so the file genuinely takes the fixed-column path.
    ///
    /// Independent oracle: hand-solved LP. `min -x1-x2` s.t. `x1+x2 <= 10`
    /// (LC123) and `x1 <= 6` (`BR   1 1`); optimum `-10.0` at `x1+x2 = 10`.
    ///
    /// **No-op failure guarantee**: applying the grid check past column 61 makes
    /// this a `ParseError` — verified by temporarily reverting.
    #[test]
    fn test_mps_fixed_columns_ignores_comment_and_sequence_fields() {
        use otspot_core::solve;

        // Data in columns 1-61; comments from column 62; sequence numbers from 73.
        let mps = concat!(
            "NAME          FIXEDCOL\n",
            "ROWS\n",
            " N  obj                                                      * objective\n",
            " L  LC123                                                               00000010\n",
            " L  BR   1 1                                                 * spaces!  00000020\n",
            "COLUMNS\n",
            "    X1        obj               -1.0   LC123              1.0* two pairs00000030\n",
            "    X1        BR   1 1           1.0                         * one pair 00000040\n",
            "    X2        obj               -1.0   LC123              1.0           00000050\n",
            "RHS\n",
            "    RHS       LC123              10.   BR   1 1            6.* rhs      00000060\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps)
            .expect("columns 62+ are the comment and sequence fields and must be ignored");

        assert_eq!(lp.num_vars, 2);
        assert_eq!(lp.num_constraints, 2);
        assert_eq!(
            lp.b,
            vec![10.0, 6.0],
            "both the two-pair RHS line and the embedded-space row name must survive"
        );
        let result = solve(&lp);
        assert!(
            (result.objective - (-10.0)).abs() < 1e-6,
            "hand-solved optimum is -10.0, got {}",
            result.objective
        );
    }

    /// The fixed-column grid is checked as a whole: every byte between the six
    /// fields must be blank, and the indicator field (columns 2-3) must be blank
    /// in the sections that do not use it. Checking only the bytes adjacent to
    /// each field left holes — a stray character in the middle of a gutter
    /// (column 38), or in the indicator field of a COLUMNS line, was silently
    /// ignored; and a value sitting in the second value field with no name
    /// beside it was skipped entirely.
    ///
    /// **No-op failure guarantee**: each case parses successfully under the
    /// previous per-field boundary checks — verified by temporarily reverting.
    #[test]
    fn test_mps_fixed_grid_rejects_content_outside_the_fields() {
        let file = |columns_line: &str| {
            format!(
                "NAME          FIXEDCOL\nROWS\n N  obj\n L  LC123\n L  BR   1 1\nCOLUMNS\n\
                 {columns_line}\n    X1        BR   1 1           1.0\nRHS\n    \
                 RHS       LC123              10.\nENDATA\n"
            )
        };

        // Stray character in the gutter between the fields at columns 37-39.
        let stray_gutter = file("    X1        obj               -1.0 X LC123              1.0");
        assert!(
            parse_mps(&stray_gutter).is_err(),
            "a stray character in the inter-field gutter must not be ignored"
        );

        // Content in the indicator field (columns 2-3), which COLUMNS does not use.
        let indicator = file(" ZZ X1        obj               -1.0");
        assert!(
            parse_mps(&indicator).is_err(),
            "content in the unused indicator field must not be ignored"
        );

        // A value in the second value field with no name beside it.
        let orphan_value = file("    X1        obj               -1.0                      9.9");
        assert!(
            parse_mps(&orphan_value).is_err(),
            "a value with no matching name must not be silently dropped"
        );

        // A multi-byte character straddling a field boundary must not yield an
        // empty name.
        let multibyte = file("    X1     \u{e9}  obj               -1.0");
        assert!(
            parse_mps(&multibyte).is_err(),
            "a multi-byte character across a field boundary must error, not read as empty"
        );
    }

    /// Two rows may not share a name: every reference to it would be ambiguous.
    /// The row-name index silently kept the last one, leaving the earlier row's
    /// matrix line empty while the model still built and solved.
    ///
    /// **No-op failure guarantee**: removing the `row_names` duplicate check
    /// makes this parse succeed — verified by temporarily reverting.
    #[test]
    fn test_mps_duplicate_row_name_is_error() {
        let mps = concat!(
            "NAME\n",
            "ROWS\n N obj\n L c1\n L c1\n",
            "COLUMNS\n x1 obj 1.0 c1 1.0\n",
            "RHS\n rhs c1 1.0\n",
            "ENDATA\n",
        );
        let err = parse_mps(mps).expect_err("a duplicate row name must error");
        assert!(
            format!("{err}").contains("duplicate row name"),
            "got: {err}"
        );
    }

    /// `OBJSENSE  MAX` written on the section header line is legal (HiGHS and
    /// SCIP accept it), and the spelled-out `MAXIMIZE` is too. Consuming the
    /// header and discarding the value silently minimized a problem that asked
    /// to be maximized — the same class of silent misparse as the rest of this
    /// module.
    ///
    /// **No-op failure guarantee**: ignoring the header's trailing value leaves
    /// `c = [1.0]` (minimize) instead of `[-1.0]`; rejecting `MAXIMIZE` makes
    /// the second parse a hard error — verified by temporarily reverting.
    #[test]
    fn test_mps_objsense_inline_and_spelled_out() {
        let body = "ROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 1.0 c1 1.0\nRHS\n rhs c1 10.0\nENDATA\n";

        let inline = format!("NAME\nOBJSENSE  MAX\n{body}");
        let lp = parse_mps(&inline).expect("OBJSENSE on the header line must be honoured");
        assert_eq!(
            lp.c,
            vec![-1.0],
            "MAX is normalized to MIN by negating c; +1.0 means OBJSENSE was dropped"
        );

        let spelled = format!("NAME\nOBJSENSE\n    MAXIMIZE\n{body}");
        let lp = parse_mps(&spelled).expect("the spelled-out MAXIMIZE must be accepted");
        assert_eq!(lp.c, vec![-1.0]);

        let minimize = format!("NAME\nOBJSENSE\n    MINIMIZE\n{body}");
        let lp = parse_mps(&minimize).expect("the spelled-out MINIMIZE must be accepted");
        assert_eq!(lp.c, vec![1.0]);
    }

    /// RANGES-expanded rows are appended in ROWS declaration order. Walking the
    /// `ranges` HashMap instead ordered them by hash, so the same input produced
    /// different constraint indices — and a different matrix — from run to run
    /// (Rust's HashMap seeds its hasher per process).
    ///
    /// Independent oracle: the IBM convention for a G row is
    /// `b <= Ax <= b + |r|`, so row `i` becomes `Le` with rhs `b_i + r_i` and a
    /// `Ge` row with rhs `b_i` is appended — in declaration order.
    ///
    /// **No-op failure guarantee**: iterating `self.ranges` (a HashMap) makes the
    /// appended order vary per run, so this fails on most runs — verified by
    /// temporarily reverting.
    #[test]
    fn test_mps_ranges_expansion_follows_declaration_order() {
        let mps = concat!(
            "NAME\n",
            "ROWS\n N obj\n G r1\n G r2\n G r3\n",
            "COLUMNS\n x1 obj 1.0 r1 1.0\n x1 r2 1.0 r3 1.0\n",
            "RHS\n rhs r1 1.0 r2 2.0\n rhs r3 3.0\n",
            "RANGES\n rng r1 10.0 r2 20.0\n rng r3 30.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect("RANGES must parse");
        assert_eq!(
            lp.b,
            vec![11.0, 22.0, 33.0, 1.0, 2.0, 3.0],
            "G rows become Le at b+|r| in declaration order, then the Ge rows at b \
             in declaration order"
        );
    }

    /// Naming a vector after the section has already taken unnamed entries is
    /// ambiguous — one vector whose name was omitted earlier, or a second vector
    /// whose entries must be discarded. The two readings disagree about which
    /// values reach the model, so the file is rejected rather than silently
    /// dropping the named entry (which is what used to happen: `c2` never
    /// reached `b`).
    ///
    /// **No-op failure guarantee**: attributing the named line to the anonymous
    /// vector's identity makes this parse succeed with `b = [10.0, 0.0]` —
    /// verified by temporarily reverting.
    #[test]
    fn test_mps_named_vector_after_unnamed_entries_is_error() {
        let mps = concat!(
            "NAME\n",
            "ROWS\n N obj\n L c1\n L c2\n",
            "COLUMNS\n x1 obj 1.0 c1 1.0\n x1 c2 1.0\n",
            "RHS\n c1 10.0\n RHS1 c2 20.0\n",
            "ENDATA\n",
        );
        assert!(
            parse_mps(mps).is_err(),
            "mixing an unnamed RHS entry with a later named vector must error, \
             not silently drop the named entry"
        );
    }

    /// When both readings fail, the diagnostic comes from whichever got further
    /// into the file. This file is fixed-column (its free reading dies on the
    /// embedded-space row name at ROWS) but is broken much later, at a
    /// non-numeric RHS value. Reporting the free-format complaint about ROWS
    /// would point at a line that is perfectly valid for what this file is.
    ///
    /// **No-op failure guarantee**: always returning the free-format error makes
    /// the message name the ROWS line instead of the RHS value — verified by
    /// temporarily reverting.
    #[test]
    fn test_mps_error_comes_from_the_reading_that_got_further() {
        let mps = concat!(
            "NAME          FIXEDCOL\n",
            "ROWS\n",
            " N  obj\n",
            " L  LC123\n",
            " L  BR   1 1\n",
            "COLUMNS\n",
            "    X1        obj               -1.0   LC123              1.0\n",
            "    X1        BR   1 1           1.0\n",
            "RHS\n",
            "    RHS       LC123        not_a_num\n",
            "ENDATA\n",
        );
        let err = parse_mps(mps).expect_err("the RHS value is not a number");
        let msg = format!("{err}");
        assert!(
            msg.contains("not_a_num"),
            "the reported error must be the fixed-column reading's (it got as far as RHS), \
             not the free-format reading's complaint about the ROWS line; got: {msg}"
        );
    }

    /// A second N row is a *free row*: declared, but neither the objective nor
    /// a constraint. It must occupy no constraint index and contribute no
    /// matrix coefficient.
    ///
    /// Giving it a `row_map` slot while skipping it in `constraint_types`
    /// desynchronised the two, so the row count exceeded the RHS count and the
    /// file failed to build at all. Two real MIPLIB files carry a second N row
    /// and were rejected outright: `data/miplib_2017/mad.mps` ("Dimension
    /// mismatch: b expected 52 but got 51" — 53 ROWS, 2 of them N, so 51 is the
    /// correct count) and `data/miplib_2017/neos-933966.mps` (12048 vs 12047).
    ///
    /// **No-op failure guarantee**: restoring the `row_map` insert for N rows
    /// makes this a `ParseError` ("Dimension mismatch: b expected 2 but got 1")
    /// — verified by temporarily reverting.
    #[test]
    fn test_mps_extra_free_row_is_not_a_constraint() {
        let mps = concat!(
            "NAME\n",
            "ROWS\n N obj\n N free2\n L c1\n",
            "COLUMNS\n    x1 obj 1.0 c1 1.0\n    x1 free2 5.0\n",
            "RHS\n    rhs c1 10.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect("a second N row (free row) must not break the build");
        assert_eq!(
            lp.num_constraints, 1,
            "only c1 is a constraint; obj and free2 are free rows"
        );
        assert_eq!(lp.b, vec![10.0]);
        assert_eq!(lp.c, vec![1.0], "objective comes from the FIRST N row only");
        assert_eq!(
            lp.a.values(),
            &[1.0],
            "only c1's coefficient enters A; the free row's 5.0 is ignored"
        );
    }

    /// A COLUMNS/RHS reference to a row that ROWS never declared must be a hard
    /// error, not a silent drop. Beyond being correct on its own, this is the
    /// signal the whole-file format decision relies on: a fixed-column file
    /// misread as free format produces exactly such phantom names.
    ///
    /// **No-op failure guarantee**: restoring the `unwrap_or(0.0)` /
    /// `if let Some(..)` silent-skip in `build_lp_problem` makes both parses
    /// succeed — verified by temporarily reverting.
    #[test]
    fn test_mps_undefined_row_reference_is_error() {
        let via_columns =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 typo_row 1.0\nRHS\n    rhs c1 1.0\nENDATA\n";
        assert!(
            parse_mps(via_columns).is_err(),
            "COLUMNS entry for an undeclared row must error"
        );

        let via_rhs =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs typo_row 1.0\nENDATA\n";
        assert!(
            parse_mps(via_rhs).is_err(),
            "RHS entry for an undeclared row must error"
        );
    }

    /// Sentinel: RANGES vector name still correctly detected when the same
    /// section also carries a genuine shorthand line (no vector name) —
    /// covers the `resolve_vector_name` attribution path for RANGES too.
    #[test]
    fn test_mps_ranges_shorthand_two_pairs_numeric_row_names() {
        let mps = "NAME\nROWS\n N obj\n E 1\n E 2\nCOLUMNS\n    x1 obj 1.0 1 1.0\n    x2 obj 1.0 2 1.0\nRHS\n    1 7.0 2 7.0\nRANGES\n    1  2.0  2  3.0\nENDATA\n";
        let lp = parse_mps(mps).expect("shorthand 2-pair RANGES must parse");
        assert_eq!(
            lp.num_constraints, 4,
            "2 base rows + 2 RANGES-expansion rows"
        );
        // E row (r>=0): [b, b+|r|]; stored as Le=b+|r| plus a Ge=b extra row.
        assert_eq!(lp.b[0], 9.0, "row 1: Le upper = 7.0+2.0");
        assert_eq!(lp.b[1], 10.0, "row 2: Le upper = 7.0+3.0");
        assert_eq!(lp.b[2], 7.0, "row 1: Ge lower = 7.0");
        assert_eq!(lp.b[3], 7.0, "row 2: Ge lower = 7.0");
    }

    /// Sentinel: BOUNDS bound-set-name omitted (shorthand `TYPE COL [VALUE]`)
    /// must parse for both value-taking (`UP`) and non-value-taking (`FR`)
    /// types, mirroring the MPS module's now-consistent RHS/RANGES handling.
    ///
    /// **No-op failure guarantee**: reverting `parse_bounds_line` to always
    /// assume a bound name is present (`parts[1]` discarded, `parts[2]` =
    /// column) makes this a `ParseError` (too few fields) instead of `Ok` —
    /// verified by temporarily reverting.
    #[test]
    fn test_mps_bounds_shorthand_no_bound_name() {
        let mps = "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\n    x2 obj 1.0\nRHS\nBOUNDS\n UP x1 42.0\n FR x2\nENDATA\n";
        let lp = parse_mps(mps).expect("BOUNDS shorthand (no bound name) must parse");
        assert_eq!(lp.bounds[0], (0.0, 42.0), "UP x1 42.0 (shorthand)");
        assert_eq!(
            lp.bounds[1],
            (f64::NEG_INFINITY, f64::INFINITY),
            "FR x2 (shorthand)"
        );
    }

    // -----------------------------------------------------------------------
    // Fixed-column shorthand (review finding, P2): the free-format shorthand
    // reader above (`test_mps_bounds_shorthand_no_bound_name`) checks declared
    // names to spot an omitted vector/bound-set name; the fixed-column reader
    // did not, and unconditionally read BOUNDS' column from field 3 and
    // RHS/RANGES' vector name from field 2. A fixed-column file whose BOUNDS
    // section omits the bound-set name therefore hard-errored with "BOUNDS
    // line missing column name" instead of parsing.
    // -----------------------------------------------------------------------

    /// Sentinel: a fixed-column BOUNDS line with the bound-set name omitted,
    /// where the column name lands in field 2 (the bound-set-name slot) and
    /// the value stays in field 4 (its usual slot), leaving field 3 blank.
    /// This is the exact shape a real `emps`-style writer produced (reported
    /// against `otspot_io::mps::parse_mps`, which previously returned
    /// `Err("line N: BOUNDS line missing column name")` for it).
    ///
    /// The row name `LIM 1` has an embedded space, forcing the whole file to
    /// the fixed-column reader (a free-format read of `ROWS` fails on it).
    ///
    /// **No-op failure guarantee**: reverting `parse_bounds_entry`'s Fixed arm
    /// to always read the column from field 3 makes this `Err("BOUNDS line
    /// missing column name")` instead of `Ok` — verified by temporarily
    /// reverting.
    #[test]
    fn test_mps_bounds_fixed_shorthand_value_stays_in_field4() {
        let mps = concat!(
            "NAME          BNDFIX\n",
            "ROWS\n",
            " N  COST\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    X 1       COST      1.0            LIM 1     1.0\n",
            "RHS\n",
            "    RHS       LIM 1     10.0\n",
            "BOUNDS\n",
            " UP X 1                 5.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps)
            .expect("fixed-column BOUNDS shorthand (col in field 2, value in field 4) must parse");
        assert_eq!(lp.bounds[0], (0.0, 5.0), "UP 'X 1' 5.0 (field-2 shorthand)");
    }

    /// Sentinel: a fixed-column BOUNDS line with the bound-set name omitted
    /// where the *whole* line shifts one field left instead — the column
    /// lands in field 2 and the value in field 3, leaving field 4 blank.
    /// Distinguishing this from the standard reading (bound-set name in
    /// field 2, column in field 3) requires checking field 2 against the
    /// declared columns: it must be a real column, and field 3 must not be
    /// one (it holds a number here, not a name).
    ///
    /// **No-op failure guarantee**: reverting the field-2/field-3 declared-name
    /// check makes this read field 3 (`"3.0"`) as the column name, which is
    /// undeclared, so `parse_mps` returns `Err` (`UndefinedReference`) instead
    /// of `Ok` — verified by temporarily reverting.
    #[test]
    fn test_mps_bounds_fixed_shorthand_uniform_shift() {
        let mps = concat!(
            "NAME          BNDFIX\n",
            "ROWS\n",
            " N  COST\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    X 1       COST      1.0            LIM 1     1.0\n",
            "    X2        COST      1.0            LIM 1     1.0\n",
            "RHS\n",
            "    RHS       LIM 1     10.0\n",
            "BOUNDS\n",
            " UP X 1                 5.0\n",
            " UP X2        3.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps)
            .expect("fixed-column BOUNDS shorthand (col in field 2, value in field 3) must parse");
        assert_eq!(
            lp.bounds[0],
            (0.0, 5.0),
            "UP 'X 1' 5.0 (field-2/field-4 shorthand)"
        );
        assert_eq!(
            lp.bounds[1],
            (0.0, 3.0),
            "UP X2 3.0 (field-2/field-3 shorthand)"
        );
    }

    /// Sentinel: a fixed-column BOUNDS line in the genuine standard form
    /// (bound-set name present in field 2, column in field 3) must still be
    /// read as standard, not misdetected as shorthand, even though the
    /// bound-set name is not itself a declared column (the common case).
    /// This guards the tie-break added for the shorthand fix: field 3 is
    /// non-empty and the standard reading is not implausible, so it wins.
    #[test]
    fn test_mps_bounds_fixed_standard_form_not_misread_as_shorthand() {
        let mps = concat!(
            "NAME          BNDFIX\n",
            "ROWS\n",
            " N  COST\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    X 1       COST      1.0            LIM 1     1.0\n",
            "RHS\n",
            "    RHS       LIM 1     10.0\n",
            "BOUNDS\n",
            " UP BND       X 1       5.0\n",
            "ENDATA\n",
        );
        let lp =
            parse_mps(mps).expect("standard fixed-column BOUNDS (with bound-set name) must parse");
        assert_eq!(lp.bounds[0], (0.0, 5.0), "UP BND 'X 1' 5.0 (standard form)");
    }

    /// Sentinel (review finding): a grid-aligned fixed-column BOUNDS line with
    /// a second numeric value in field 5 (`UP BND X 1 5.0 <field 5>10.0`) must
    /// be a hard error, not silently accepted with the field 5 content
    /// dropped. This is the case the free-format surplus-token check
    /// (`test_mps_bounds_up_surplus_value_token_is_error`) does *not* cover:
    /// that test's line is not column-aligned, so free format alone rejects
    /// it; a grid-aligned line with the same surplus instead re-parses
    /// successfully under the fixed-format fallback unless the fixed-format
    /// reading independently checks fields 5/6.
    ///
    /// **No-op failure guarantee**: reverting the field 5/6 check in
    /// `parse_bounds_entry` (`otspot-io/src/common/mod.rs`) makes this
    /// return `Ok` with `bounds[0] == (0.0, 5.0)` — the `10.0` silently
    /// dropped — instead of `Err`; verified by temporarily reverting.
    #[test]
    fn test_mps_bounds_fixed_grid_aligned_field5_surplus_value_is_error() {
        let mps = concat!(
            "NAME          BNDFIX\n",
            "ROWS\n",
            " N  COST\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    X 1       COST      1.0            LIM 1     1.0\n",
            "RHS\n",
            "    RHS       LIM 1     10.0\n",
            "BOUNDS\n",
            " UP BND       X 1       5.0            10.0\n",
            "ENDATA\n",
        );
        let err = parse_mps(mps).unwrap_err();
        assert!(
            err.to_string().contains("field 5"),
            "grid-aligned BOUNDS with a surplus value in field 5 must be rejected, got {err:?}"
        );
    }

    /// Sentinel (review finding): same as above but with non-numeric junk in
    /// field 5, confirming the check rejects *any* content there, not just
    /// content that happens to parse as a number.
    ///
    /// **No-op failure guarantee**: reverting the field 5/6 check makes this
    /// return `Ok` with `bounds[0] == (0.0, 5.0)` — the `JUNK` silently
    /// dropped — instead of `Err`; verified by temporarily reverting.
    #[test]
    fn test_mps_bounds_fixed_grid_aligned_field5_junk_is_error() {
        let mps = concat!(
            "NAME          BNDFIX\n",
            "ROWS\n",
            " N  COST\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    X 1       COST      1.0            LIM 1     1.0\n",
            "RHS\n",
            "    RHS       LIM 1     10.0\n",
            "BOUNDS\n",
            " UP BND       X 1       5.0            JUNK\n",
            "ENDATA\n",
        );
        let err = parse_mps(mps).unwrap_err();
        assert!(
            err.to_string().contains("field 5"),
            "grid-aligned BOUNDS with junk content in field 5 must be rejected, got {err:?}"
        );
    }

    /// Sentinel: a fixed-column RHS line with the vector name omitted, where
    /// the first row name lands in field 2 (the vector-name slot) instead of
    /// field 3, shifting both (row, value) pairs one field left: field 2/3
    /// carry the first pair, field 4/5 the second, field 6 unused.
    ///
    /// The row name `BR   1 1` has an embedded space (fills field 2's 8-byte
    /// width exactly), forcing the whole file to the fixed-column reader.
    ///
    /// **No-op failure guarantee**: reverting `parse_vector_entry`'s Fixed arm
    /// to always read the vector name from field 2 and pairs from
    /// (field 3, field 4) / (field 5, field 6) reads field 2 ("BR   1 1") as
    /// the vector name and field 3 ("6.") as a row name with no declared row
    /// of that name, so `parse_mps` returns `Err` (`UndefinedReference`)
    /// instead of `Ok` — verified by temporarily reverting.
    #[test]
    fn test_mps_rhs_fixed_shorthand_vector_name_omitted() {
        let mps = concat!(
            "NAME          RHSFIX\n",
            "ROWS\n",
            " N  obj\n",
            " L  BR   1 1\n",
            " L  R2\n",
            "COLUMNS\n",
            "    X1        obj       -1.0           BR   1 1  1.0\n",
            "    X1        R2        1.0\n",
            "RHS\n",
            "    BR   1 1  6.        R2             3.\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps)
            .expect("fixed-column RHS with the vector name omitted (field-2 shorthand) must parse");
        assert_eq!(lp.num_constraints, 2, "'BR   1 1' and 'R2'");
        assert_eq!(
            lp.b,
            vec![6.0, 3.0],
            "both RHS values must survive the field-2 shorthand read"
        );
    }

    // -----------------------------------------------------------------------
    // Redesign (review finding, P1): `parse_vector_entry` / `parse_bounds_entry`
    // used to tell standard reading from shorthand by checking whether field 2
    // named a declared row/column. That heuristic breaks when the vector name
    // or bound-set name legitimately collides with a declared row/column name
    // (e.g. an RHS vector conventionally named `RHS` next to a row also named
    // `RHS`) — a standard-form file was misread as shorthand and hard-errored.
    // The two are now told apart by which reading's field occupancy is
    // parseable, never by name membership.
    // -----------------------------------------------------------------------

    /// Sentinel (P1 regression): a fixed-column RHS line in genuine standard
    /// form — vector name `RHS` in field 2, pairs in fields 3-4 and 5-6 — must
    /// parse even though a declared row is also named `RHS` (field 2's content
    /// happens to equal that row's name). The row name `LIM 1` has an embedded
    /// space, forcing the whole file to the fixed-column reader.
    ///
    /// **No-op failure guarantee**: reverting to the name-membership check
    /// (field 2 counts as "omits vector name" whenever it matches a declared
    /// row) misreads this line as shorthand, since field 2's content ("RHS")
    /// is itself a declared row name: it reads field 2 as the first row name
    /// and field 3 ("LIM 1") as *its* value, which fails to parse as a number
    /// — `parse_mps` returns `Err("Invalid RHS value 'LIM 1'")` instead of
    /// `Ok` — verified by temporarily reverting.
    #[test]
    fn test_mps_rhs_fixed_vector_name_collides_with_declared_row_name() {
        let mps = concat!(
            "NAME          COLLIDE\n",
            "ROWS\n",
            " N  obj\n",
            " L  LIM 1\n",
            " L  RHS\n",
            "COLUMNS\n",
            "    X1        obj       1.0            LIM 1     1.0\n",
            "    X1        RHS       1.0\n",
            "RHS\n",
            "    RHS       LIM 1     10.0           RHS       4.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect(
            "standard fixed-column RHS must parse even when the vector name collides with a \
             declared row name",
        );
        assert_eq!(lp.num_constraints, 2, "'LIM 1' and 'RHS'");
        assert_eq!(
            lp.b,
            vec![10.0, 4.0],
            "RHS vector 'RHS': LIM 1 -> 10.0, row 'RHS' -> 4.0"
        );
    }

    /// Sentinel (P1 regression): a fixed-column RANGES line in genuine
    /// standard form — vector name `RNG` in field 2, row/value in fields 3-4 —
    /// must parse even though a declared row is also named `RNG`.
    ///
    /// **No-op failure guarantee**: reverting to the name-membership check
    /// misreads field 2 ("RNG") as the row name and field 3 ("LIM 1") as its
    /// value, so `parse_value` fails on the non-numeric row name — verified by
    /// temporarily reverting.
    #[test]
    fn test_mps_ranges_fixed_vector_name_collides_with_declared_row_name() {
        let mps = concat!(
            "NAME          COLLIDR\n",
            "ROWS\n",
            " N  obj\n",
            " L  LIM 1\n",
            " L  RNG\n",
            "COLUMNS\n",
            "    X1        obj       1.0            LIM 1     1.0\n",
            "    X1        RNG       1.0\n",
            "RHS\n",
            "    R         LIM 1     10.0           RNG       4.0\n",
            "RANGES\n",
            "    RNG       LIM 1     2.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect(
            "standard fixed-column RANGES must parse even when the vector name collides with a \
             declared row name",
        );
        // RANGES 'RNG' applies range 2.0 to row 'LIM 1' (an L row): b=10.0 ->
        // upper=10.0 (unchanged), extra Ge lower=10.0-2.0=8.0. Row 'RNG' (L,
        // rhs=4.0) carries no RANGES entry of its own.
        assert_eq!(
            lp.num_constraints, 3,
            "'LIM 1' + 'RNG' + LIM 1's Ge extra row"
        );
        assert_eq!(
            lp.b,
            vec![10.0, 4.0, 8.0],
            "LIM 1 Le=10.0, RNG Le=4.0, LIM 1 extra Ge=8.0"
        );
    }

    /// Sentinel (P1 regression): a fixed-column BOUNDS line in genuine
    /// standard form — bound-set name in field 2, column in field 3 — must
    /// parse even when the bound-set name collides with a declared column
    /// name (`BND` is both the bound-set name and a real column here).
    ///
    /// **No-op failure guarantee**: the prior name-membership tie-break
    /// happened to get this specific case right (field 3 was also a declared
    /// column, so standard reading still won), but relied on `ColNameIndex`
    /// membership rather than field occupancy; this sentinel guards the
    /// occupancy-only replacement continuing to read it correctly.
    #[test]
    fn test_mps_bounds_fixed_bound_set_name_collides_with_declared_column_name() {
        let mps = concat!(
            "NAME          COLLIDB\n",
            "ROWS\n",
            " N  obj\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    BND       obj       1.0            LIM 1     1.0\n",
            "    X1        obj       1.0            LIM 1     1.0\n",
            "RHS\n",
            "    R         LIM 1     10.0\n",
            "BOUNDS\n",
            " UP BND       X1        5.0\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect(
            "standard fixed-column BOUNDS must parse even when the bound-set name collides with \
             a declared column name",
        );
        assert_eq!(
            lp.bounds[0],
            (0.0, f64::INFINITY),
            "'BND' column itself is untouched by its own name appearing as a bound-set name"
        );
        assert_eq!(lp.bounds[1], (0.0, 5.0), "UP 'X1' 5.0 (standard form)");
    }

    /// Sentinel (P1 regression): same bound-set/column-name collision as
    /// above, but with a non-value-taking bound type (`FR`), which took a
    /// different code path in the prior name-membership tie-break
    /// (`value_required` was false, so field 4 was never consulted).
    #[test]
    fn test_mps_bounds_fixed_bound_set_name_collides_with_declared_column_name_non_value_type() {
        let mps = concat!(
            "NAME          COLLIDF\n",
            "ROWS\n",
            " N  obj\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    BND       obj       1.0            LIM 1     1.0\n",
            "    X1        obj       1.0            LIM 1     1.0\n",
            "RHS\n",
            "    R         LIM 1     10.0\n",
            "BOUNDS\n",
            " FR BND       X1\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect(
            "standard fixed-column FR BOUNDS must parse even when the bound-set name collides \
             with a declared column name",
        );
        assert_eq!(lp.bounds[0], (0.0, f64::INFINITY), "'BND' column untouched");
        assert_eq!(
            lp.bounds[1],
            (f64::NEG_INFINITY, f64::INFINITY),
            "FR 'X1' (standard form)"
        );
    }

    /// Sentinel: fixed-column RANGES with the vector name omitted (no test
    /// previously covered this — only the RHS analogue did). First row name
    /// lands in field 2, shifting both (row, value) pairs one field left:
    /// field 2/3 carry the first pair, field 4/5 the second, field 6 unused.
    ///
    /// **No-op failure guarantee**: reverting `parse_vector_entry`'s Fixed arm
    /// to always read the vector name from field 2 and pairs from
    /// (field 3, field 4) / (field 5, field 6) reads field 3 ("2.") as a row
    /// name with no declared row of that name, so `parse_mps` returns `Err`
    /// (`UndefinedReference`) instead of `Ok` — verified by temporarily
    /// reverting.
    #[test]
    fn test_mps_ranges_fixed_shorthand_vector_name_omitted() {
        let mps = concat!(
            "NAME          RNGFIX\n",
            "ROWS\n",
            " N  obj\n",
            " L  BR   1 1\n",
            " L  R2\n",
            "COLUMNS\n",
            "    X1        obj       -1.0           BR   1 1  1.0\n",
            "    X1        R2        1.0\n",
            "RHS\n",
            "    RHS       BR   1 1  10.0           R2        20.0\n",
            "RANGES\n",
            "    BR   1 1  2.        R2             4.\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps).expect(
            "fixed-column RANGES with the vector name omitted (field-2 shorthand) must parse",
        );
        // 'BR   1 1' (L, rhs=10.0, range=2.0): Le=10.0 (unchanged) + extra Ge=8.0.
        // 'R2' (L, rhs=20.0, range=4.0): Le=20.0 (unchanged) + extra Ge=16.0.
        assert_eq!(lp.num_constraints, 4);
        assert_eq!(lp.b, vec![10.0, 20.0, 8.0, 16.0]);
    }

    /// Sentinel: a fixed-column RANGES line where field 2 must be reread as a
    /// row name (standard reading finds no pair), but the shorthand reading
    /// then leaves stray content in field 6 — neither reading is valid, so
    /// this must hard-error rather than silently drop the extra value.
    #[test]
    fn test_mps_ranges_fixed_shorthand_trailing_field6_is_error() {
        let mps = concat!(
            "NAME          RNGTRL\n",
            "ROWS\n",
            " N  obj\n",
            " L  BR   1 1\n",
            " L  R2\n",
            "COLUMNS\n",
            "    X1        obj       -1.0           BR   1 1  1.0\n",
            "    X1        R2        1.0\n",
            "RHS\n",
            "    RHS       BR   1 1  10.0           R2        20.0\n",
            "RANGES\n",
            "    BR   1 1  2.        R2             4.        9.9\n",
            "ENDATA\n",
        );
        let err = parse_mps(mps).expect_err("stray content in field 6 must be a hard error");
        let message = format!("{}", err);
        assert!(
            message.contains("field 6 must be blank"),
            "unexpected error message: {}",
            message
        );
    }

    /// Sentinel: a fixed-column BOUNDS line for a non-value-taking type (`FR`)
    /// with the bound-set name omitted and the column shifted into field 2,
    /// leaving field 3 and field 4 both blank (there is no value to place).
    /// No test previously covered a non-value-taking fixed-column shorthand.
    ///
    /// **No-op failure guarantee**: reverting `parse_bounds_entry`'s Fixed arm
    /// to always read the column from field 3 reads an empty column name
    /// there, so `parse_mps` returns `Err("BOUNDS line missing column name")`
    /// instead of `Ok` — verified by temporarily reverting.
    #[test]
    fn test_mps_bounds_fixed_shorthand_non_value_type_no_bound_name() {
        let mps = concat!(
            "NAME          SHORTFR\n",
            "ROWS\n",
            " N  obj\n",
            " L  LIM 1\n",
            "COLUMNS\n",
            "    X 1       obj       1.0            LIM 1     1.0\n",
            "RHS\n",
            "    R         LIM 1     10.0\n",
            "BOUNDS\n",
            " FR X 1\n",
            "ENDATA\n",
        );
        let lp = parse_mps(mps)
            .expect("fixed-column FR shorthand (bound-set name omitted, no value) must parse");
        assert_eq!(
            lp.bounds[0],
            (f64::NEG_INFINITY, f64::INFINITY),
            "FR 'X 1' (shorthand, no bound-set name, no value)"
        );
    }

    /// Sentinel: an RHS line where neither the standard nor the shorthand
    /// reading is parseable — the standard reading is missing a value ('BR
    /// 1 1' has no matching value in field 4) and the shorthand reading
    /// cannot parse the row name as a number either. This must hard-error;
    /// the reported diagnostic is the standard reading's, since standard is
    /// the presumptive layout (mirroring the file-level Free/Fixed tie-break's
    /// convention).
    #[test]
    fn test_mps_rhs_fixed_neither_reading_parses_reports_standard_diagnostic() {
        let mps = concat!(
            "NAME          RHSBAD\n",
            "ROWS\n",
            " N  obj\n",
            " L  BR   1 1\n",
            " L  R2\n",
            "COLUMNS\n",
            "    X1        obj       -1.0           BR   1 1  1.0\n",
            "    X1        R2        1.0\n",
            "RHS\n",
            "    RHS1      BR   1 1\n",
            "ENDATA\n",
        );
        let err = parse_mps(mps).expect_err("a line valid under neither reading must hard-error");
        let message = format!("{}", err);
        assert!(
            message.contains("has no matching value"),
            "expected the standard reading's diagnostic (missing value), got: {}",
            message
        );
    }
}
