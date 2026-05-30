//! QPS format parser (quadratic programming).
//!
//! QPS = MPS + QUADOBJ section.  The `1/2` convention is used:
//! `min 1/2 x^T Q x + c^T x` — consistent with the Maros-Mészáros benchmark.

mod types;
mod parser;

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
    use otspot_core::problem::SolveStatus;
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
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/netlib/TAME.QPS");
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

    // ── Sentinel tests: audit#141 parser strictness (A/B/C) ──────────────────

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
        assert!(parse_qps_str(&qps).is_err(), "< 3 fields in COLUMNS must error");
    }

    /// A: QUADOBJ line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_qps_quadobj_malformed_too_few_fields_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nQUADOBJ\n    x1\nENDATA\n";
        assert!(parse_qps_str(qps).is_err(), "< 3 fields in QUADOBJ must error");
    }

    /// A: BOUNDS line with only 2 fields must be an error, not a silent skip.
    #[test]
    fn test_qps_bounds_malformed_too_few_fields_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 5.0\nBOUNDS\n LO\nENDATA\n";
        assert!(parse_qps_str(qps).is_err(), "< 3 fields in BOUNDS must error");
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
        let qps = "NAME\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  c1  1.0\nRHS\n    c1  NaN\nENDATA\n";
        assert!(parse_qps_str(qps).is_err(), "NaN in constraint RHS must error");
    }

    /// P2-2: symmetric QUADOBJ entry (x2,x1) when (x1,x2) already present must error.
    #[test]
    fn test_qps_quadobj_symmetric_duplicate_is_error() {
        let qps = "NAME          SYM_DUP\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x1  obj  1.0  c1  1.0\n    x2  obj  1.0  c1  1.0\nRHS\n    rhs  c1  10.0\nQUADOBJ\n    x1  x2  1.0\n    x2  x1  2.0\nENDATA\n";
        let err = parse_qps_str(qps);
        assert!(err.is_err(), "(x1,x2) and (x2,x1) in QUADOBJ must error");
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("Duplicate"), "error should mention 'Duplicate': {}", msg);
    }

    /// B: duplicate (col1, col2) pair in QUADOBJ must be an error.
    #[test]
    fn test_qps_quadobj_duplicate_entry_is_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nQUADOBJ\n    x1 x1 2.0\n    x1 x1 3.0\nENDATA\n";
        let err = parse_qps_str(qps);
        assert!(err.is_err(), "duplicate entry in QUADOBJ must error");
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("Duplicate"), "error should mention 'Duplicate': {}", msg);
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
}
