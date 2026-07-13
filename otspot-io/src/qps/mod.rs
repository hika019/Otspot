//! QPS format parser (quadratic programming).
//!
//! QPS = MPS + QUADOBJ section.  The `1/2` convention is used:
//! `min 1/2 x^T Q x + c^T x` — consistent with the Maros-Mészáros benchmark.

mod parser;
mod types;

use std::path::Path;

use otspot_core::qp::QpProblem;

pub use parser::parse_qps_reader;

/// Errors produced by the QPS parser.
#[non_exhaustive]
#[derive(Debug)]
pub enum QpsError {
    /// I/O error reading from the source.
    IoError(std::io::Error),
    /// Malformed content at the given line.
    ParseError { line: usize, message: String },
    /// A required section (ROWS / COLUMNS / ENDATA) is missing.
    MissingSection(String),
    /// A section appears more than once.
    DuplicateSection(String),
    /// An undefined column or row name was referenced.
    UndefinedReference { kind: String, name: String },
    /// The N-row RHS value (obj_offset) is NaN or infinite.
    InvalidObjectiveOffset(f64),
}

impl std::fmt::Display for QpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QpsError::IoError(e) => write!(f, "I/O error: {}", e),
            QpsError::ParseError { line, message } => {
                write!(f, "Parse error at line {}: {}", line, message)
            }
            QpsError::MissingSection(s) => write!(f, "Missing required section: {}", s),
            QpsError::DuplicateSection(s) => write!(f, "Duplicate section: {}", s),
            QpsError::UndefinedReference { kind, name } => {
                write!(f, "Undefined {} reference: {}", kind, name)
            }
            QpsError::InvalidObjectiveOffset(val) => {
                write!(f, "Invalid objective offset (NaN/Inf): {}", val)
            }
        }
    }
}

impl std::error::Error for QpsError {}

impl From<std::io::Error> for QpsError {
    fn from(err: std::io::Error) -> Self {
        QpsError::IoError(err)
    }
}

/// Parse a QPS file from `path`.
///
/// Uses streaming I/O — peak memory proportional to the longest line.
pub fn parse_qps(path: &Path) -> Result<QpProblem, QpsError> {
    let file = std::fs::File::open(path)?;
    parse_qps_reader(std::io::BufReader::new(file))
}

/// Parse a QPS string.
pub fn parse_qps_str(input: &str) -> Result<QpProblem, QpsError> {
    parse_qps_reader(std::io::Cursor::new(input.as_bytes()))
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::print_stderr)]
mod tests {
    use super::*;
    use otspot_core::problem::{ConstraintType, SolveStatus};
    use otspot_core::qp::solve_qp;

    #[test]
    fn test_parse_qps_simple() {
        let qps = r"NAME          TEST_QP
ROWS
 N  obj
 G  c1
COLUMNS
    x1    obj    -100.0    c1    1.0
    x2    obj    -1.0      c1    -0.05
RHS
    rhs   c1    -0.5
BOUNDS
 LO BND   x1    2.0
 UP BND   x1    50.0
 LO BND   x2    -50.0
 UP BND   x2    50.0
QUADOBJ
    x1    x1    100.0
    x2    x2    1.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 1);
    }

    #[test]
    fn test_parse_qps_no_quadobj() {
        let qps = r"NAME          LP_ONLY
ROWS
 N  obj
 L  c1
COLUMNS
    x1    obj    1.0    c1    1.0
    x2    obj    2.0    c1    1.0
RHS
    rhs   c1    10.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 1);
        assert!(prob.is_zero_q());
    }

    #[test]
    fn test_parse_qps_eq_constraint() {
        use otspot_core::problem::ConstraintType;
        let qps = r"NAME          EQ_TEST
ROWS
 N  obj
 E  eq1
COLUMNS
    x1    obj    2.0    eq1    1.0
    x2    obj    1.0    eq1    1.0
RHS
    rhs   eq1    5.0
QUADOBJ
    x1    x1    2.0
    x2    x2    2.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 1);
        assert_eq!(prob.constraint_types[0], ConstraintType::Eq);
    }

    #[test]
    fn test_solve_qps_basic() {
        let qps = r"NAME          BASIC
ROWS
 N  obj
 G  sum1
COLUMNS
    x    obj    0.0    sum1    1.0
    y    obj    0.0    sum1    1.0
RHS
    rhs   sum1    1.0
BOUNDS
 FR BND   x
 FR BND   y
QUADOBJ
    x    x    1.0
    y    y    1.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        let result = solve_qp(&prob);
        assert_eq!(result.status, SolveStatus::Optimal);
        let obj = result.objective;
        assert!((obj - 0.25).abs() < 2e-3, "expected obj≈0.25, got {}", obj);
    }

    #[test]
    fn test_quadobj_symmetry() {
        let qps_upper = r"NAME SYM
ROWS
 N  obj
COLUMNS
    x1  obj  0.0
    x2  obj  0.0
BOUNDS
 FR BND  x1
 FR BND  x2
QUADOBJ
    x1  x1  2.0
    x1  x2  1.0
    x2  x2  2.0
ENDATA
";
        let prob = parse_qps_str(qps_upper).unwrap();
        assert_eq!(prob.q.nrows(), 2);
        assert_eq!(prob.q.ncols(), 2);
        assert_eq!(prob.q.values().len(), 4);
    }

    #[test]
    fn test_parse_qps_obj_offset() {
        let qps = r"NAME          OBJ_OFFSET_TEST
ROWS
 N  obj
 L  c1
COLUMNS
    x1    obj    1.0    c1    1.0
RHS
    rhs   obj    -7.5
    rhs   c1    10.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert!((prob.obj_offset - (-7.5)).abs() < 1e-10);
    }

    #[test]
    fn test_e226_obj_offset() {
        let path = std::path::Path::new("data/lp_problems/e226.QPS");
        if !path.exists() {
            eprintln!("e226.QPS not found, skip");
            return;
        }
        let prob = parse_qps(path).unwrap();
        assert!((prob.obj_offset - (-7.113)).abs() < 1e-3);
    }

    #[test]
    fn test_obj_offset_nan_inf_guard() {
        let qps = "NAME          INF_TEST\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1    obj    1.0    c1    1.0\nRHS\n    rhs   obj    inf\n    rhs   c1    10.0\nENDATA\n";
        let result = parse_qps_str(qps);
        assert!(matches!(result, Err(QpsError::InvalidObjectiveOffset(_))));
    }

    /// QPS fixed-format (`force_fixed` path) での N 行 RHS NaN が `InvalidObjectiveOffset` を返すことを sentinel 化。
    /// `obj_row.as_deref()` を `None` に置換すると obj 行 NaN が ParseError 化 →
    /// `InvalidObjectiveOffset` でなくなる → assertion 失敗 (no-op fail 設計)。
    ///
    /// RHS 行: pos 4-11 空 + pos 14-21 "obj" で `force_fixed=true` → `parse_mps_fixed_pairs` 経由。
    #[test]
    fn test_qps_fixed_format_obj_row_nan_invalidates_offset() {
        // Fixed-format RHS line: pos 0-13 spaces, pos 14-16 "obj", pos 24-26 "NaN",
        // pos 39-40 "c1", pos 49-51 "1.0". mps_field(line,4,12)="" → force_fixed=true.
        let qps = concat!(
            "NAME          FIXFMT_NAN\n",
            "ROWS\n N  obj\n L  c1\n",
            "COLUMNS\n    x1  c1  1.0\n    x1  obj  1.0\n",
            "RHS\n",
            "              obj       NaN            c1        1.0\n",
            "ENDATA\n",
        );
        let result = parse_qps_str(qps);
        assert!(
            matches!(result, Err(QpsError::InvalidObjectiveOffset(_))),
            "fixed-format obj-row NaN must trigger InvalidObjectiveOffset: {:?}",
            result
        );
    }

    #[test]
    fn test_solve_with_obj_offset() {
        let qps = r"NAME          OFFSET_INTEG
ROWS
 N  obj
 G  sum1
COLUMNS
    x1    obj    1.0    sum1    1.0
    x2    obj    1.0    sum1    1.0
RHS
    rhs   obj    -7.0
    rhs   sum1    3.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert!((prob.obj_offset - (-7.0)).abs() < 1e-10);
        let result = solve_qp(&prob);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective - (-4.0)).abs() < 1e-3);
    }

    #[test]
    fn test_parse_bounds_3token_no_bname() {
        let qps = r"NAME  TEST
ROWS
 N  obj
COLUMNS
    x1  obj  1.0
    x2  obj  1.0
RHS
BOUNDS
 LO  x1  70000.
 UP  x2  100000.
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.bounds[0].0, 70000.0);
        assert_eq!(prob.bounds[1].1, 100000.0);
    }

    #[test]
    fn test_parse_bounds_3token_fr_bname() {
        let qps = r"NAME  TEST
ROWS
 N  obj
COLUMNS
    x  obj  1.0
    y  obj  1.0
RHS
BOUNDS
 FR BND  x
 MI BND  y
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.bounds[0].0, f64::NEG_INFINITY);
        assert_eq!(prob.bounds[0].1, f64::INFINITY);
        assert_eq!(prob.bounds[1].0, f64::NEG_INFINITY);
    }

    #[test]
    fn test_parse_bounds_4token_with_bname() {
        let qps = r"NAME  TEST
ROWS
 N  obj
COLUMNS
    x1  obj  1.0
RHS
BOUNDS
 LO BND  x1  2.0
 UP BND  x1  50.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.bounds[0].0, 2.0);
        assert_eq!(prob.bounds[0].1, 50.0);
    }

    #[test]
    fn test_parse_bounds_fr_with_numeric_var_name() {
        let qps = "NAME  DPKLO1_LIKE\nROWS\n N  obj\nCOLUMNS\n    1  obj  1.0\n    2  obj  1.0\n    3  obj  1.0\nRHS\nBOUNDS\n FR  BNDS  1\n FR  BNDS  2\n FR  BNDS  3\nENDATA\n";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.num_vars, 3);
        for j in 0..3 {
            assert_eq!(prob.bounds[j].0, f64::NEG_INFINITY);
            assert_eq!(prob.bounds[j].1, f64::INFINITY);
        }
    }

    #[test]
    fn test_parse_bounds_mi_with_numeric_var_name() {
        let qps = "NAME  TEST\nROWS\n N  obj\nCOLUMNS\n    1  obj  1.0\nRHS\nBOUNDS\n MI  BNDS  1\nENDATA\n";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.bounds[0].0, f64::NEG_INFINITY);
    }

    #[test]
    fn test_parse_bounds_bv_with_numeric_var_name() {
        let qps = "NAME  TEST\nROWS\n N  obj\nCOLUMNS\n    9  obj  1.0\nRHS\nBOUNDS\n BV  BNDS  9\nENDATA\n";
        let prob = parse_qps_str(qps).unwrap();
        assert_eq!(prob.bounds[0].0, 0.0);
        assert_eq!(prob.bounds[0].1, 1.0);
    }

    const STREAM_QPS: &str = "NAME          stream\n\
ROWS\n N  obj\n G  sum1\n\
COLUMNS\n    x  obj  0.0  sum1  1.0\n    y  obj  0.0  sum1  1.0\n\
RHS\n    rhs  sum1  1.0\n\
BOUNDS\n FR BND  x\n FR BND  y\n\
QUADOBJ\n    x  x  1.0\n    y  y  1.0\n\
ENDATA\n";

    #[test]
    fn test_qps_reader_round_trip() {
        let expected = parse_qps_str(STREAM_QPS).unwrap();
        let got = parse_qps_reader(std::io::Cursor::new(STREAM_QPS.as_bytes())).unwrap();
        assert_eq!(got.num_vars, expected.num_vars);
        assert_eq!(got.num_constraints, expected.num_constraints);
        assert_eq!(got.c, expected.c);
        assert_eq!(got.b, expected.b);
        assert_eq!(got.bounds, expected.bounds);
        assert_eq!(got.q.values().len(), expected.q.values().len());
    }

    #[test]
    fn test_qps_reader_crlf_equivalence() {
        let lf = parse_qps_reader(std::io::Cursor::new(STREAM_QPS.as_bytes())).unwrap();
        let crlf_src = STREAM_QPS.replace('\n', "\r\n");
        let crlf = parse_qps_reader(std::io::Cursor::new(crlf_src.as_bytes())).unwrap();
        assert_eq!(crlf.num_vars, lf.num_vars);
        assert_eq!(crlf.num_constraints, lf.num_constraints);
        assert_eq!(crlf.c, lf.c);
        assert_eq!(crlf.b, lf.b);
        assert_eq!(crlf.bounds, lf.bounds);
        assert_eq!(crlf.q.values(), lf.q.values());
    }

    #[test]
    fn test_qps_reader_fixture_tame() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/netlib/TAME.QPS");
        let content = std::fs::read_to_string(&path).unwrap();
        let expected = parse_qps_str(&content).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let got = parse_qps_reader(std::io::BufReader::new(file)).unwrap();
        assert_eq!(got.num_vars, expected.num_vars);
        assert_eq!(got.num_constraints, expected.num_constraints);
        assert_eq!(got.c, expected.c);
        assert_eq!(got.b, expected.b);
        assert_eq!(got.q.values(), expected.q.values());
        assert!(!got.q.values().is_empty());
    }

    /// Sentinel: OBJSENSE MAX with an N-row RHS must negate the offset.
    ///
    /// **No-op failure guarantee**: removing the `if self.maximize { -raw }` sign-flip
    /// leaves `obj_offset = 10.0` instead of `-10.0` → assertion fires.
    #[test]
    fn test_qps_objsense_max_obj_offset_sign_flip() {
        let qps = r"NAME  MAX_OFFSET
OBJSENSE
    MAX
ROWS
 N  obj
 L  c1
COLUMNS
    x1    obj    1.0    c1    1.0
RHS
    rhs   obj    10.0
    rhs   c1    5.0
ENDATA
";
        let prob = parse_qps_str(qps).unwrap();
        assert!(
            (prob.obj_offset - (-10.0)).abs() < 1e-12,
            "OBJSENSE MAX with N-row RHS=10.0 must yield obj_offset=-10.0; got {}",
            prob.obj_offset
        );
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
    fn test_qps_reader_streaming_sentinel() {
        let counter = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let reader = LineCountingReader {
            inner: std::io::Cursor::new(STREAM_QPS.as_bytes()),
            line_call_count: counter.clone(),
        };
        let prob = parse_qps_reader(reader).expect("parse must succeed");
        assert_eq!(prob.num_vars, 2);
        let expected_lines = STREAM_QPS.lines().count();
        assert!(
            counter.get() >= expected_lines,
            "streaming must call read_line at least {expected_lines} times, got {}",
            counter.get()
        );
    }

    // ── Sentinel tests: audit 141 parser strictness (A/B/C) ───────────────────

    fn minimal_qps_with_columns(col_section: &str) -> String {
        format!(
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n{}\nRHS\n    rhs c1 10.0\nENDATA\n",
            col_section
        )
    }

    /// A: COLUMNS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_qps_columns_malformed_too_few_fields_is_error() {
        let qps = minimal_qps_with_columns("    x1  obj");
        assert!(
            parse_qps_str(&qps).is_err(),
            "< 3 fields in COLUMNS must error"
        );
    }

    /// A: QUADOBJ line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_qps_quadobj_malformed_too_few_fields_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nQUADOBJ\n    x1\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "< 3 fields in QUADOBJ must error"
        );
    }

    /// A: BOUNDS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_qps_bounds_malformed_too_few_fields_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nBOUNDS\n LO\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "< 3 fields in BOUNDS must error"
        );
    }

    /// Duplicate (col, row) entries in COLUMNS must accumulate (sum), not error.
    /// QPS inherits MPS spec: repeated entries are summed via CscMatrix triplet merge.
    #[test]
    fn test_parse_qps_accumulates_duplicate_objective_entries() {
        let qps = "NAME          DUP_TEST\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  obj  1.0  c1  1.0\n    x1  obj  2.0\nRHS\n    rhs  c1  10.0\nENDATA\n";
        let prob = parse_qps_str(qps).expect("duplicate objective entries must parse OK");
        assert_eq!(prob.num_vars, 1);
        assert!(
            (prob.c[0] - 3.0).abs() < 1e-10,
            "1.0 + 2.0 = 3.0, got {}",
            prob.c[0]
        );
    }

    /// P2-1: NaN in constraint RHS (2-field shorthand) must error.
    #[test]
    fn test_qps_rhs_nan_constraint_row_is_error() {
        let qps =
            "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  c1  1.0\nRHS\n    c1  NaN\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "NaN in constraint RHS must error"
        );
    }

    /// P2-2: symmetric QUADOBJ entry (x2,x1) when (x1,x2) already present must error.
    #[test]
    fn test_qps_quadobj_symmetric_duplicate_is_error() {
        let qps = "NAME          SYM_DUP\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  obj  1.0  c1  1.0\n    x2  obj  1.0  c1  1.0\nRHS\n    rhs  c1  10.0\nQUADOBJ\n    x1  x2  1.0\n    x2  x1  2.0\nENDATA\n";
        let err = parse_qps_str(qps);
        assert!(err.is_err(), "(x1,x2) and (x2,x1) in QUADOBJ must error");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("Duplicate"),
            "error should mention 'Duplicate': {}",
            msg
        );
    }

    /// B: duplicate (col1, col2) pair in QUADOBJ must be an error.
    #[test]
    fn test_qps_quadobj_duplicate_entry_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nQUADOBJ\n    x1 x1 2.0\n    x1 x1 3.0\nENDATA\n";
        let err = parse_qps_str(qps);
        assert!(err.is_err(), "duplicate entry in QUADOBJ must error");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("Duplicate"),
            "error should mention 'Duplicate': {}",
            msg
        );
    }

    /// C: NaN coefficient in COLUMNS must be an error.
    #[test]
    fn test_qps_columns_nan_value_is_error() {
        let qps = minimal_qps_with_columns("    x1 c1 NaN");
        assert!(parse_qps_str(&qps).is_err(), "NaN in COLUMNS must error");
    }

    /// C: Inf coefficient in COLUMNS must be an error.
    #[test]
    fn test_qps_columns_inf_value_is_error() {
        let qps = minimal_qps_with_columns("    x1 c1 Inf");
        assert!(parse_qps_str(&qps).is_err(), "Inf in COLUMNS must error");
    }

    /// C: NaN in QUADOBJ must be an error.
    #[test]
    fn test_qps_quadobj_nan_value_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nQUADOBJ\n    x1 x1 NaN\nENDATA\n";
        assert!(parse_qps_str(qps).is_err(), "NaN in QUADOBJ must error");
    }

    /// NaN in a constraint-row RHS (3-field format) must error.
    #[test]
    fn test_qps_rhs_nan_constraint_row_3field_is_error() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  c1  1.0\nRHS\n    rhs  c1  NaN\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "NaN in constraint-row RHS (3-field) must error"
        );
    }

    /// Inf in a constraint-row RHS (3-field format) must error.
    #[test]
    fn test_qps_rhs_inf_constraint_row_3field_is_error() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  c1  1.0\nRHS\n    rhs  c1  Inf\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "Inf in constraint-row RHS (3-field) must error"
        );
    }

    /// NaN for an undefined (typo) row name in RHS must error, not be silently accepted.
    #[test]
    fn test_qps_rhs_nan_named_overwrite_is_error() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  c1  1.0\nRHS\n    rhs  typo_row  NaN  c1  1.0\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "NaN for unknown row in named RHS line must error (not silent accept)"
        );
    }

    // ── Sentinel tests: input-validation audit ────────────────────────────────

    /// Fix-3: BOUNDS entry referencing a column not in COLUMNS must error.
    /// Sentinel: reverting the UndefinedReference return to `continue` → Ok instead of Err.
    #[test]
    fn test_sentinel_qps_bounds_undefined_column_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nBOUNDS\n LO BND  ghost  1.0\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "BOUNDS referencing undefined column must error, not be silently ignored"
        );
    }

    /// Fix-3: QUADOBJ entry referencing a column not in COLUMNS must error.
    /// Sentinel: reverting to `continue` → Ok instead of Err.
    #[test]
    fn test_sentinel_qps_quadobj_undefined_column_is_error() {
        let qps =
            "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\nRHS\nQUADOBJ\n    x1 ghost 2.0\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "QUADOBJ referencing undefined column must error, not be silently ignored"
        );
    }

    /// Fix-4: value-bearing BOUNDS type (LO) with missing value must error.
    /// Sentinel: reverting to silent None default → Ok instead of Err.
    #[test]
    fn test_sentinel_qps_bounds_lo_missing_value_is_error() {
        let qps = "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\nRHS\nBOUNDS\n LO BND x1\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "LO bound without a value must error"
        );
    }

    /// Fix-4: value-bearing BOUNDS type (FX) with missing value must error.
    #[test]
    fn test_sentinel_qps_bounds_fx_missing_value_is_error() {
        let qps = "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\nRHS\nBOUNDS\n FX BND x1\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "FX bound without a value must error"
        );
    }

    /// Fix-5: odd trailing token in COLUMNS (row name with no value) must error.
    /// Sentinel: reverting trailing-token check → Ok instead of Err.
    #[test]
    fn test_sentinel_qps_columns_trailing_row_no_value_is_error() {
        let qps =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 c1 1.0 obj\nRHS\n    rhs c1 5.0\nENDATA\n";
        assert!(
            parse_qps_str(qps).is_err(),
            "trailing row name without a value in COLUMNS must error"
        );
    }

    // ── Black-box parse tests ─────────────────────────────────────────────────

    /// TECHNIQUE: EQUIVALENCE PARTITIONING — well-formed 2-var QPS with QUADOBJ.
    ///
    /// Oracle (hand-derived):
    ///   c = [-1, -2] (linear objective from COLUMNS).
    ///   b = [5.0], constraint_types = [Le], num_vars=2, num_constraints=1.
    ///   QUADOBJ: x1 x1 2.0 → Q[0,0]=2.0; x2 x2 4.0 → Q[1,1]=4.0.
    ///   Convention: min 1/2 x'Qx + c'x, so QUADOBJ stores Q directly.
    ///   Default bounds: (0, +inf) for both variables.
    #[test]
    fn ep_qps_2var_quadobj_structure() {
        let qps = "\
NAME      tiny2var
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  -1.0  c1  1.0
    x2  obj  -2.0  c1  1.0
RHS
    rhs  c1  5.0
QUADOBJ
    x1  x1  2.0
    x2  x2  4.0
ENDATA
";
        let qp = parse_qps_str(qps).expect("ep_qps_2var_quadobj: valid QPS must parse");
        assert_eq!(qp.num_vars, 2, "num_vars");
        assert_eq!(qp.num_constraints, 1, "num_constraints");
        assert_eq!(qp.c, vec![-1.0, -2.0], "linear objective c");
        assert_eq!(qp.b, vec![5.0], "RHS");
        assert_eq!(qp.constraint_types, vec![ConstraintType::Le], "L row → Le");
        // Q[0,0] = 2.0
        let (rows0, vals0) = qp.q.get_column(0).expect("Q col 0 must exist");
        let q00 = rows0
            .iter()
            .zip(vals0.iter())
            .find_map(|(&r, &v)| (r == 0).then_some(v))
            .expect("Q[0,0] must be present");
        assert!((q00 - 2.0).abs() < 1e-12, "Q[0,0]=2.0 got {q00}");
        // Q[1,1] = 4.0
        let (rows1, vals1) = qp.q.get_column(1).expect("Q col 1 must exist");
        let q11 = rows1
            .iter()
            .zip(vals1.iter())
            .find_map(|(&r, &v)| (r == 1).then_some(v))
            .expect("Q[1,1] must be present");
        assert!((q11 - 4.0).abs() < 1e-12, "Q[1,1]=4.0 got {q11}");
        // Default bounds
        assert_eq!(qp.bounds[0].0, 0.0, "x1 lb=0 (default)");
        assert_eq!(qp.bounds[1].0, 0.0, "x2 lb=0 (default)");
    }

    /// TECHNIQUE: DECISION TABLE — QPS G and E row types parsed correctly.
    ///
    /// Oracle (hand-derived from QPS parser source):
    ///   The QPS parser normalises G rows to Le form by negating: G becomes Le with
    ///   A-entries multiplied by -1 and RHS negated. E rows become Eq unchanged.
    ///   ge1 (G row, x1+x2>=2): stored as Le, b[0]=-2.0, A entries negated.
    ///   eq1 (E row, x1=1):     stored as Eq, b[1]=1.0, A entries unchanged.
    ///   constraint_types = [Le, Eq], b = [-2.0, 1.0].
    #[test]
    fn dt_qps_ge_eq_row_types() {
        let qps = "\
NAME      dt_ge_eq
ROWS
 N  obj
 G  ge1
 E  eq1
COLUMNS
    x1  obj  1.0  ge1  1.0
    x1  eq1  1.0
    x2  obj  2.0  ge1  1.0
RHS
    rhs  ge1  2.0
    rhs  eq1  1.0
ENDATA
";
        let qp = parse_qps_str(qps).expect("dt_qps_ge_eq: valid QPS must parse");
        assert_eq!(qp.num_vars, 2, "num_vars");
        assert_eq!(qp.num_constraints, 2, "num_constraints");
        // G row is normalised to Le (negated); E row stays Eq.
        assert_eq!(
            qp.constraint_types,
            vec![ConstraintType::Le, ConstraintType::Eq],
            "G→Le (negated), E→Eq"
        );
        // G row RHS 2.0 is negated to -2.0; E row RHS 1.0 is unchanged.
        assert_eq!(qp.b, vec![-2.0, 1.0], "b: G-row negated, E-row unchanged");
    }

    // -----------------------------------------------------------------------
    // PR #25 review horizontal expansion: RHS/RANGES duplicate-row detection.
    //
    // Unlike COLUMNS (which accumulates duplicate (row,col) entries by design,
    // see `test_parse_qps_accumulates_duplicate_objective_entries`), RHS and
    // RANGES hold exactly one scalar per row; a repeated row name is
    // ambiguous input that was previously silently resolved via last-write-wins.
    // -----------------------------------------------------------------------

    /// Sentinel: the same row name appearing twice in RHS (multi-pair line)
    /// must be a `ParseError`, not silently overwritten.
    ///
    /// **No-op failure guarantee**: reverting to plain `self.rhs.insert(name, value)`
    /// makes this parse succeed with `prob.b[0] == 20.0` (last-write-wins) instead of erroring.
    #[test]
    fn test_qps_duplicate_rhs_row_is_error() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  obj  1.0  c1  1.0\nRHS\n    rhs  c1  10.0\n    rhs  c1  20.0\nENDATA\n";
        let err = parse_qps_str(qps).expect_err("duplicate RHS row must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("RHS") && msg.contains("duplicate"),
            "error should mention RHS duplicate, got: {msg}"
        );
    }

    /// Sentinel: the 2-field shorthand RHS path must also reject a row that
    /// was already set via the named multi-pair path.
    #[test]
    fn test_qps_duplicate_rhs_row_shorthand_is_error() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  c1  1.0\nRHS\n    rhs  c1  10.0\n    c1  20.0\nENDATA\n";
        let err = parse_qps_str(qps).expect_err("duplicate RHS row (shorthand) must error");
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
    fn test_qps_duplicate_ranges_row_is_error() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  obj  1.0  c1  1.0\nRHS\n    rhs  c1  10.0\nRANGES\n    rng  c1  2.0\n    rng  c1  4.0\nENDATA\n";
        let err = parse_qps_str(qps).expect_err("duplicate RANGES row must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("RANGES") && msg.contains("duplicate"),
            "error should mention RANGES duplicate, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Task #9: vector-name-omitted (shorthand) RHS/RANGES/BOUNDS + multi-vector
    // RHS/RANGES support.
    // -----------------------------------------------------------------------

    /// Sentinel: a 2-pair-per-line RHS with the vector name omitted, using
    /// numeric row names ("1", "2") that also look like plausible values —
    /// the exact shape that historically broke `blend.QPS`: a naive parser
    /// skips `parts[0]` as if it were a vector name, then mispairs
    /// `(parts[1], parts[2])` = `("10.0", 2.0)`, silently discarding row
    /// "1"'s value and fabricating a bogus entry for a nonexistent row.
    ///
    /// **No-op failure guarantee**: reverting `parse_rhs_line` to the old
    /// `force_fixed`/`is_free` dispatch onto `parse_mps_free_pairs` (always
    /// skip `parts[0]`) makes `prob.b` become `[0.0, 20.0]` instead of
    /// `[10.0, 20.0]` (row "1"'s RHS silently lost) — verified by temporarily
    /// reverting during development.
    #[test]
    fn test_qps_rhs_shorthand_two_pairs_numeric_row_names() {
        let qps = "NAME\nROWS\n N  obj\n L  1\n L  2\nCOLUMNS\n    x1  obj  1.0  1  1.0\n    x2  obj  1.0  2  1.0\nRHS\n    1  10.0  2  20.0\nENDATA\n";
        let prob = parse_qps_str(qps).expect("shorthand 2-pair RHS must parse");
        assert_eq!(prob.b, vec![10.0, 20.0], "both RHS values must survive");
    }

    /// Sentinel: `blend_shorthand.mps` — a real, live `emps`-decoded Netlib
    /// LP "blend" — parsed through the QPS reader (as `data/lp_problems/*.QPS`
    /// netlib fixtures are in this codebase), matching the documented Netlib
    /// optimum. Its RHS section has numeric row names ("65".."72"), no vector
    /// name, 2 pairs packed per line: this is the historical trigger where an
    /// earlier multi-vector RHS attempt misread the shape and produced a
    /// wrong (previously reported as ~0) objective.
    ///
    /// **No-op failure guarantee**: reverting the shorthand disambiguation
    /// drops all but 2 of the 8 RHS rows and the solved objective becomes
    /// wrong — verified by temporarily reverting.
    #[test]
    fn test_qps_blend_shorthand_rhs_matches_known_optimal() {
        use otspot_core::qp::solve_qp;

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/netlib/blend_shorthand.mps");
        let prob = parse_qps(&path).expect("blend_shorthand.mps must parse via QPS reader");
        assert_eq!(prob.num_constraints, 74, "43 E-rows + 31 L-rows, no N-row");
        assert!(prob.is_zero_q(), "blend is a pure LP (no QUADOBJ section)");

        let nonzero_count = prob.b.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            nonzero_count, 8,
            "exactly 8 RHS rows are nonzero in blend; got {nonzero_count} (silent data loss?)"
        );

        let result = solve_qp(&prob);
        assert!(
            (result.objective - (-30.812149846)).abs() < 1e-6,
            "blend_shorthand objective must match Netlib optimal -30.812149846, got {}",
            result.objective
        );
    }

    /// Sentinel (INLINE-P / PR #25 review finding): two distinct NAMED RHS
    /// vectors legitimately reusing the same row must parse successfully,
    /// applying only the first vector's value (GLPK/CPLEX "first vector
    /// wins" convention) — not be rejected as a row-only duplicate.
    ///
    /// **No-op failure guarantee**: reverting the `(vector, row)`-keyed
    /// `VectorSectionState::record` to a plain
    /// `if self.rhs.contains_key(&name) { error }` row-only duplicate check
    /// makes this a `ParseError` instead of `Ok` — verified by temporarily
    /// reverting.
    #[test]
    fn test_qps_rhs_multiple_named_vectors_first_wins() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  obj  1.0  c1  1.0\nRHS\n    RHS1  c1  10.0\n    RHS2  c1  20.0\nENDATA\n";
        let prob = parse_qps_str(qps).expect("distinct named RHS vectors reusing a row must parse");
        assert_eq!(prob.b, vec![10.0], "first vector (RHS1) must win, not RHS2");
    }

    /// Sentinel (multi-vector RANGES): analogous to the RHS case above.
    ///
    /// Oracle: `c1` is an `L` row (b=10.0). RANGES expansion for `L` splits
    /// into `Le rhs=b` and `Ge rhs=b-|range|`, and the QPS builder then
    /// uniformly negates `Ge` rows to `Le` (see `dt_qps_ge_eq_row_types`) —
    /// so with RNG1's range=2.0 winning, the second row's rhs is
    /// `-(b-|range|) = -(10.0-2.0) = -8.0`.
    #[test]
    fn test_qps_ranges_multiple_named_vectors_first_wins() {
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  obj  1.0  c1  1.0\nRHS\n    rhs  c1  10.0\nRANGES\n    RNG1  c1  2.0\n    RNG2  c1  4.0\nENDATA\n";
        let prob =
            parse_qps_str(qps).expect("distinct named RANGES vectors reusing a row must parse");
        assert_eq!(prob.num_constraints, 2);
        assert_eq!(prob.b[0], 10.0);
        assert_eq!(
            prob.b[1], -8.0,
            "RANGES value must come from RNG1 (2.0), not RNG2 (4.0)"
        );
    }

    /// Sentinel: BOUNDS bound-set-name omitted for the non-value-taking
    /// 2-token shorthand (`TYPE COL`, e.g. `FR x1`) — previously rejected
    /// outright by the `parts.len() < 3` guard even though the value-taking
    /// 3-token shorthand (`TYPE COL VALUE`) already worked.
    ///
    /// **No-op failure guarantee**: reverting the `parts.len() < 2` guard
    /// and the `!value_taking` branch's length check makes this a
    /// `ParseError` (too few fields) instead of `Ok` — verified by
    /// temporarily reverting.
    #[test]
    fn test_qps_bounds_shorthand_non_value_taking_2token() {
        let qps =
            "NAME\nROWS\n N  obj\nCOLUMNS\n    x1  obj  1.0\n    x2  obj  1.0\nRHS\nBOUNDS\n FR  x1\n UP  x2  42.0\nENDATA\n";
        let prob = parse_qps_str(qps).expect("BOUNDS shorthand (no bound name) must parse");
        assert_eq!(
            prob.bounds[0],
            (f64::NEG_INFINITY, f64::INFINITY),
            "FR x1 (shorthand, 2-token)"
        );
        assert_eq!(
            prob.bounds[1],
            (0.0, 42.0),
            "UP x2 42.0 (shorthand, 3-token)"
        );
    }
}
