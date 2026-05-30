//! MPS format parser (LP / MILP).
//!
//! Parses NAME / ROWS / COLUMNS / RHS / RANGES / BOUNDS / ENDATA sections,
//! auto-detecting fixed-width and free-format layouts.  INTORG/INTEND markers
//! and BV/LI/UI bound types identify integer variables.
//!
//! - [`parse_mps`] / [`parse_mps_file`]: returns an `LpProblem` (integrality dropped).
//! - [`parse_milp`] / [`parse_milp_file`]: returns a `MilpProblem` with integer vars.

mod types;
mod parser;

use std::path::Path;

use otspot_core::mip::MilpProblem;
use otspot_core::problem::LpProblem;

pub use otspot_core::error::MpsError;
pub use parser::{parse_mps_reader, parse_milp_reader};

/// Parse an MPS file from `path`, returning an LP relaxation.
///
/// Uses streaming I/O — peak memory proportional to the longest line.
///
/// # Errors
///
/// Returns [`MpsError`] for I/O failures or malformed content.
pub fn parse_mps_file(path: &Path) -> Result<LpProblem, MpsError> {
    let file = std::fs::File::open(path)?;
    parse_mps_reader(std::io::BufReader::new(file))
}

/// Parse an MPS file from `path`, returning a `MilpProblem`.
///
/// Uses streaming I/O (`BufReader`). Integer variables identified via
/// INTORG/INTEND markers and BV/LI/UI bound types are preserved.
///
/// # Errors
///
/// Returns [`MpsError`] for I/O failures or malformed content.
pub fn parse_milp_file(path: &Path) -> Result<MilpProblem, MpsError> {
    let file = std::fs::File::open(path)?;
    parse_milp_reader(std::io::BufReader::new(file))
}

/// Parse an MPS string, returning an LP relaxation.
///
/// MILP files are accepted but integrality is dropped; use [`parse_milp`] to
/// retain integer variable information.
///
/// # Examples
///
/// ```
/// use otspot_io::mps::parse_mps;
///
/// let mps = r"NAME          example
/// ROWS
///  N  obj
///  L  c1
/// COLUMNS
///     x1  obj  1.0  c1  2.0
/// RHS
///     rhs  c1  10.0
/// ENDATA
/// ";
/// let lp = parse_mps(mps).unwrap();
/// assert_eq!(lp.num_vars, 1);
/// assert_eq!(lp.num_constraints, 1);
/// ```
pub fn parse_mps(input: &str) -> Result<LpProblem, MpsError> {
    parse_mps_reader(std::io::Cursor::new(input.as_bytes()))
}

/// Parse an MPS string, returning a `MilpProblem`.
///
/// # Examples
///
/// ```
/// use otspot_io::mps::parse_milp;
///
/// let mps = r"NAME          milp
/// ROWS
///  N  obj
///  L  c1
/// COLUMNS
///     MARKER1   'MARKER'   'INTORG'
///     x1  obj  -1.0  c1  1.0
///     MARKER2   'MARKER'   'INTEND'
/// RHS
///     rhs  c1  10.5
/// BOUNDS
///  UP BND  x1  7.0
/// ENDATA
/// ";
/// let milp = parse_milp(mps).unwrap();
/// assert_eq!(milp.integer_vars, vec![0]);
/// ```
pub fn parse_milp(input: &str) -> Result<MilpProblem, MpsError> {
    parse_milp_reader(std::io::Cursor::new(input.as_bytes()))
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
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
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
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
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
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
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
        assert_eq!(lp.constraint_types, vec![ConstraintType::Le, ConstraintType::Ge]);
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
    fn test_is_fixed_width_typical_fixed() {
        let fixed_line = "    x1          obj   1.0";
        assert!(types::is_fixed_width_format(fixed_line));
    }

    #[test]
    fn test_is_fixed_width_free_format() {
        let line = "    x1  obj  1.0";
        assert!(!types::is_fixed_width_format(line));
    }

    #[test]
    fn test_is_fixed_width_short_line() {
        assert!(!types::is_fixed_width_format(""));
        assert!(!types::is_fixed_width_format("    x1  c1 1"));
        assert!(!types::is_fixed_width_format("12345678901234"));
    }

    #[test]
    fn test_is_fixed_width_with_tab() {
        let line_with_tab = "    x1        \tobj  1.0";
        assert!(types::is_fixed_width_format(line_with_tab));
    }

    #[test]
    fn test_integer_marker_kind_intorg_intend() {
        use types::{IntegerMarker, integer_marker_kind};
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
            ("marker_up5_fractional", "BOUNDS\n UP BND  x1  5.0\n", 3.5, -3.0),
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
            assert_eq!(res.status, SolveStatus::Optimal, "[{label}] should be Optimal");
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
        let res = otspot_core::mip::solve_milp(
            &milp,
            &SolverOptions::default(),
            &MipConfig::default(),
        );
        assert_eq!(res.status, SolveStatus::Optimal);
        assert!((res.objective - (-3.0)).abs() < 1e-6, "expected -3, got {}", res.objective);
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
        assert!(matches!(parse_mps(mps).unwrap_err(), MpsError::UnclosedIntegerMarker));
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
        assert!((res.objective - (-3.0)).abs() < 1e-6, "expected -3, got {}", res.objective);
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

    use std::io::{self, Read};

    struct LineCountingReader<R: std::io::BufRead> {
        inner: R,
        pub line_call_count: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl<R: std::io::BufRead> Read for LineCountingReader<R> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.inner.read(buf)
        }
    }

    impl<R: std::io::BufRead> std::io::BufRead for LineCountingReader<R> {
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

    #[test]
    fn test_mps_reader_streaming_sentinel() {
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let reader = LineCountingReader {
            inner: std::io::Cursor::new(STREAM_MPS.as_bytes()),
            line_call_count: counter.clone(),
        };
        let lp = parse_mps_reader(reader).expect("parse must succeed");
        assert_eq!(lp.num_vars, 2);
        let expected_lines = STREAM_MPS.lines().count();
        assert!(
            counter.get() >= expected_lines,
            "streaming must call read_line at least {expected_lines} times, got {}",
            counter.get()
        );
    }

    // ── Sentinel tests: audit#141 parser strictness (A/B/C) ──────────────────

    /// A: COLUMNS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_mps_columns_malformed_too_few_fields_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1  obj\nRHS\n    rhs c1 1.0\nENDATA\n";
        assert!(parse_mps(mps).is_err(), "< 3 fields in COLUMNS must error");
    }

    /// A: RHS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_mps_rhs_malformed_too_few_fields_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    c1\nENDATA\n";
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
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 NaN\nRHS\n    rhs c1 10.0\nENDATA\n";
        let err = parse_mps(mps);
        assert!(err.is_err(), "NaN coefficient in COLUMNS must error: {:?}", err);
    }

    /// C: Inf coefficient in COLUMNS must be an error.
    #[test]
    fn test_mps_columns_inf_value_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 Inf\nRHS\n    rhs c1 10.0\nENDATA\n";
        let err = parse_mps(mps);
        assert!(err.is_err(), "Inf coefficient in COLUMNS must error: {:?}", err);
    }

    /// C: NaN in RHS value must be an error.
    #[test]
    fn test_mps_rhs_nan_value_is_error() {
        let mps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 1.0\nRHS\n    rhs c1 NaN\nENDATA\n";
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
}
