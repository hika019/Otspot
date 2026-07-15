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
/// Streams the file; a file that turns out to be fixed-column is streamed a
/// second time rather than buffered, so peak memory stays proportional to the
/// parsed model rather than the file.
pub fn parse_qps(path: &Path) -> Result<QpProblem, QpsError> {
    parser::parse_qps_source(&crate::common::FileSource(path.to_path_buf()))
}

/// Parse a QPS string.
pub fn parse_qps_str(input: &str) -> Result<QpProblem, QpsError> {
    parser::parse_qps_source(&crate::common::TextSource(input))
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

    /// 目的行 (N 行) の RHS が NaN のとき `InvalidObjectiveOffset` を返すことを sentinel 化。
    /// `parse_vector_line` の `allow_nonfinite_for_row` を `None` に置換すると
    /// obj 行 NaN が ParseError 化 → `InvalidObjectiveOffset` でなくなる → assertion 失敗。
    ///
    /// RHS 行はベクトル名を省略した shorthand (obj/c1 とも宣言済み行名)。
    #[test]
    fn test_qps_fixed_format_obj_row_nan_invalidates_offset() {
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

    /// A fixed-column QPS COLUMNS entry whose ROW name contains embedded spaces
    /// must keep its coefficient. The old QPS `is_free` heuristic tokenized
    /// `    X2        BR   1 1         1.0` into `[X2, BR, 1, 1, 1.0]`, whose
    /// value slots (`1`, `1.0`) both parse as floats, so it read the line as
    /// free format and dropped the 1.0 into invented rows — a silent loss of a
    /// matrix coefficient with `Ok` returned.
    ///
    /// Independent oracle: hand-solved LP. `min -x1-x2` s.t. `x1+x2 <= 10`
    /// (LC123), `x1 <= 6` (`BR   1 1`), `0 <= x2 <= 2` (BOUNDS). Optimum
    /// `x1=6, x2=2` → objective `-8.0`; losing the `BR   1 1` coefficient would
    /// free `x1` to 8 and give `-10.0`.
    ///
    /// **No-op failure guarantee**: with the per-line `is_free` heuristic
    /// restored, the coefficient is dropped and the objective becomes -10.0 —
    /// verified by temporarily reverting.
    #[test]
    fn test_qps_fixed_columns_embedded_space_row_name_in_columns() {
        use otspot_core::problem::LpProblem;
        use otspot_core::solve;

        let qps = concat!(
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
        let prob = parse_qps_str(qps).expect("fixed-column QPS with embedded-space row must parse");
        assert_eq!(prob.b, vec![10.0, 6.0], "'BR   1 1' must keep its RHS 6.0");

        let lp = LpProblem::new_general(
            prob.c.clone(),
            prob.a.clone(),
            prob.b.clone(),
            prob.constraint_types.clone(),
            prob.bounds.clone(),
            None,
        )
        .expect("LP construction");
        let result = solve(&lp);
        assert!(
            (result.objective - (-8.0)).abs() < 1e-6,
            "hand-solved optimum is -8.0; -10.0 means the COLUMNS coefficient on \
             'BR   1 1' was silently dropped. got {}",
            result.objective
        );
    }

    /// A free-format QPS separated by single spaces must read its row names from
    /// the tokens, not from fixed byte offsets. The old QPS ROWS parser preferred
    /// the fixed byte field at columns 5-12, so ` N obj` yielded the row name `bj` and ` L c1`
    /// yielded `1` — the objective row and every constraint silently misnamed,
    /// leaving `c` and `b` all-zero while still returning `Ok`.
    ///
    /// **No-op failure guarantee**: restoring the fixed-field-first ROWS read
    /// makes `c` and `b` both `[0.0]` — verified by temporarily reverting.
    #[test]
    fn test_qps_free_format_single_space_row_names() {
        let qps =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 1.0 c1 1.0\nRHS\n rhs c1 10.0\nENDATA\n";
        let prob = parse_qps_str(qps).expect("single-space free-format QPS must parse");
        assert_eq!(prob.num_vars, 1, "x1");
        assert_eq!(prob.num_constraints, 1, "c1");
        assert_eq!(prob.c, vec![1.0], "objective row must be 'obj', not 'bj'");
        assert_eq!(prob.b, vec![10.0], "constraint row must be 'c1', not '1'");
    }

    /// An integer-marker line is recognized only when it carries BOTH a
    /// `'MARKER'` token and an `INTORG`/`INTEND` token. Keying off `'MARKER'`
    /// alone would skip a COLUMNS line for a column legitimately *named*
    /// `MARKER`, silently dropping its coefficients.
    ///
    /// **No-op failure guarantee**: relaxing the rule to "any token equals
    /// MARKER" drops the column entirely (`num_vars` becomes 0) — verified by
    /// temporarily reverting.
    #[test]
    fn test_qps_column_named_marker_is_not_a_marker_line() {
        let qps = concat!(
            "NAME\nROWS\n N obj\n L c1\n",
            "COLUMNS\n",
            "    MARKER    obj  1.0  c1  1.0\n",
            "    MARKER2                 'MARKER'                 'INTORG'\n",
            "    x2        obj  1.0  c1  1.0\n",
            "    MARKER3                 'MARKER'                 'INTEND'\n",
            "RHS\n    rhs c1 10.0\n",
            "ENDATA\n",
        );
        let prob = parse_qps_str(qps).expect("a column named MARKER must parse");
        assert_eq!(
            prob.num_vars, 2,
            "the column named MARKER and x2 both survive; only the two real \
             'MARKER'/'INTORG'|'INTEND' lines are skipped"
        );
        assert_eq!(
            prob.c,
            vec![1.0, 1.0],
            "both columns keep their objective coefficient"
        );
    }

    /// The QPS twin of the MPS truncation/duplicate-name guards: a name that
    /// overflows the fixed 8-byte field must not be clipped into a false match,
    /// and two rows may not share a name.
    ///
    /// **No-op failure guarantee**: making the fixed-column grid check accept an
    /// overflowing name, or dropping the `row_names` duplicate check, makes
    /// these parse succeed — verified by temporarily reverting.
    #[test]
    fn test_qps_long_name_truncation_and_duplicate_rows_are_errors() {
        let typo = concat!(
            "NAME\nROWS\n N obj\n L ROWLONGNAME1\n",
            "COLUMNS\n x1 obj 1.0 ROWLONGN 1.0\n",
            "RHS\n rhs ROWLONGN 4.0\nENDATA\n",
        );
        assert!(
            parse_qps_str(typo).is_err(),
            "'ROWLONGN' must not resolve against the truncated 'ROWLONGNAME1'"
        );

        // The discriminating case: a genuinely fixed-column file (its free
        // reading dies on the embedded-space row name) whose other row name
        // overflows the 8-byte field. Truncating would clip it to `TOOLONGN`
        // and parse the file as if it were valid fixed-column MPS.
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
            parse_qps_str(overflowing).is_err(),
            "a name overflowing the fixed 8-byte field must error, not be truncated"
        );

        let dup = concat!(
            "NAME\nROWS\n N obj\n L c1\n L c1\n",
            "COLUMNS\n x1 obj 1.0 c1 1.0\n",
            "RHS\n rhs c1 1.0\nENDATA\n",
        );
        let err = parse_qps_str(dup).expect_err("a duplicate row name must error");
        assert!(
            format!("{err}").contains("duplicate row name"),
            "got: {err}"
        );
    }

    /// `OBJSENSE  MAX` on the header line, and the spelled-out `MAXIMIZE`, must
    /// both be honoured; dropping them silently minimizes a maximization.
    ///
    /// **No-op failure guarantee**: ignoring the header's trailing value leaves
    /// `c = [1.0]` instead of `[-1.0]` — verified by temporarily reverting.
    #[test]
    fn test_qps_objsense_inline_and_spelled_out() {
        let body = "ROWS\n N obj\n L c1\nCOLUMNS\n x1 obj 1.0 c1 1.0\nRHS\n rhs c1 10.0\nENDATA\n";

        let inline = format!("NAME\nOBJSENSE  MAX\n{body}");
        let prob = parse_qps_str(&inline).expect("OBJSENSE on the header line must be honoured");
        assert_eq!(prob.c, vec![-1.0], "MAX is normalized to MIN by negating c");

        let spelled = format!("NAME\nOBJSENSE\n    MAXIMIZE\n{body}");
        let prob = parse_qps_str(&spelled).expect("the spelled-out MAXIMIZE must be accepted");
        assert_eq!(prob.c, vec![-1.0]);

        let minimize = format!("NAME\nOBJSENSE\n    MINIMIZE\n{body}");
        let prob = parse_qps_str(&minimize).expect("the spelled-out MINIMIZE must be accepted");
        assert_eq!(prob.c, vec![1.0]);
    }

    /// A COLUMNS/RHS reference to a row that ROWS never declared must be a hard
    /// error, not a silent drop — both correct in itself and the signal the
    /// whole-file format decision relies on.
    ///
    /// **No-op failure guarantee**: restoring the `if let Some(indices) = ..`
    /// silent skip in `build_qp_problem` makes both parses succeed — verified
    /// by temporarily reverting.
    #[test]
    fn test_qps_undefined_row_reference_is_error() {
        let via_columns =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 typo_row 1.0\nRHS\n    rhs c1 1.0\nENDATA\n";
        assert!(
            parse_qps_str(via_columns).is_err(),
            "COLUMNS entry for an undeclared row must error"
        );

        let via_rhs =
            "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs typo_row 1.0\nENDATA\n";
        assert!(
            parse_qps_str(via_rhs).is_err(),
            "RHS entry for an undeclared row must error"
        );
    }

    /// Sentinel (P0 regression, `forplan.QPS`): strict fixed-column MPS/QPS
    /// RHS/RANGES with embedded spaces in the vector name (`"RHS 1"`, `"RNG 1"`)
    /// and a row name (`"BR   1 1"`) must parse via the whole-file fixed-column
    /// reader, not whitespace tokenization. Fixture mirrors
    /// `data/lp_problems/forplan.QPS`'s RHS/RANGES lines byte-for-byte (same row
    /// names, vector names, and field columns 4/12/14/22/24/36/39/47/49/61).
    ///
    /// Independent oracle #1: `prob.b`, hand-read off the fixture — LC123=10
    /// (Le), `BR   1 1`=6 (Le, unused by any column, must still parse to the
    /// right value), LTSYCT range-expanded to `[2, 7]` (RANGES `LTSYCT 5`) →
    /// Le rhs=7 then Le rhs=-2 (the G→Le sign flip for the lower bound).
    ///
    /// Independent oracle #2: hand-solved LP. `BR   1 1` has no COLUMNS entries
    /// (it exists purely to exercise the embedded-space RHS row name), so it is
    /// a slack `0<=6` row that never binds; feasibility is driven only by
    /// `x1+x2<=10` (LC123) and `2<=x2<=7` (LTSYCT range), default `x1,x2>=0`.
    /// `min -x1-x2` is optimized anywhere with `x1+x2=10`, `2<=x2<=7`
    /// (e.g. x1=6,x2=4), objective `-10.0`.
    ///
    /// **No-op failure guarantee**: reverting the fixed-column fallback in
    /// `common::parse_vector_pairs` (to whitespace-only tokenization) makes this
    /// fixture fail to parse (`ParseError`, "row name has no matching value").
    #[test]
    fn test_qps_forplan_style_rhs_ranges_fixed_columns() {
        use otspot_core::problem::LpProblem;
        use otspot_core::solve;

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/netlib/forplan_fixed_columns.qps");
        let prob = parse_qps(&path).expect("forplan-style fixed-column RHS/RANGES must parse");

        assert_eq!(prob.num_vars, 2, "X1, X2");
        assert_eq!(
            prob.b.len(),
            4,
            "LC123, BR   1 1, LTSYCT(range upper), LTSYCT(range lower)"
        );
        assert!(
            (prob.b[0] - 10.0).abs() < 1e-12,
            "LC123 rhs must be 10 (fixed-column field3), got {}",
            prob.b[0]
        );
        assert!(
            (prob.b[1] - 6.0).abs() < 1e-12,
            "'BR   1 1' (embedded-space row name) rhs must be 6 (fixed-column field5), got {}",
            prob.b[1]
        );
        assert!(
            (prob.b[2] - 7.0).abs() < 1e-12,
            "LTSYCT range-expanded upper bound must be 7 (rhs=2 + range=5), got {}",
            prob.b[2]
        );
        assert!(
            (prob.b[3] - (-2.0)).abs() < 1e-12,
            "LTSYCT range-expanded lower bound must be -2 (G-row sign flip of rhs=2), got {}",
            prob.b[3]
        );
        assert!(
            prob.constraint_types
                .iter()
                .all(|&ct| ct == ConstraintType::Le),
            "QPS RANGES/G-row expansion always yields Le rows, got {:?}",
            prob.constraint_types
        );

        let lp = LpProblem::new_general(
            prob.c.clone(),
            prob.a.clone(),
            prob.b.clone(),
            prob.constraint_types.clone(),
            prob.bounds.clone(),
            None,
        )
        .expect("fixture LP construction failed");
        let result = solve(&lp);
        assert!(
            (result.objective - (-10.0)).abs() < 1e-6,
            "hand-solved optimum is -10.0 (x1+x2=10 maximized under x1<=... /x2 in [2,7]), got {}",
            result.objective
        );
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

    /// The QPS reader entry point must **stream** — see the MPS twin of this
    /// test for the full rationale. An input that fails on line 3 of many
    /// thousands separates a lazy pull from a reader drained up front; counting
    /// `read_line` calls on a successful parse cannot.
    ///
    /// **No-op failure guarantee**: restoring the `Vec<String>` buffering in
    /// `LineSource::from_reader` makes the count jump from single digits to the
    /// full line count — verified by temporarily reverting.
    #[test]
    fn test_qps_reader_pulls_lines_lazily_not_all_upfront() {
        use std::fmt::Write as _;

        const PADDING_LINES: usize = 3000;
        const MAX_LINES_A_STREAMING_PARSER_PULLS: usize = 50;

        let mut qps = String::from("NAME\nROWS\n Z\n");
        for i in 0..PADDING_LINES {
            writeln!(qps, "* filler line {i}").expect("write to String");
        }
        qps.push_str("ENDATA\n");
        let total_lines = qps.lines().count();

        let (reader, counter) = LineCountingReader::new(&qps);
        parse_qps_reader(reader).expect_err("a ROWS line with no row name must fail");
        assert!(
            counter.get() <= MAX_LINES_A_STREAMING_PARSER_PULLS,
            "parser must pull lines lazily and stop at the failing line 3; it pulled {} of \
             {total_lines} lines, which means the reader was drained up front",
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

    #[test]
    fn test_qps_bounds_invalid_numeric_value_is_error() {
        let qps =
            "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\nBOUNDS\n LO BND x1 not_a_number\nENDATA\n";
        let err = parse_qps_str(qps).unwrap_err();
        assert!(
            err.to_string().contains("Invalid BOUNDS value"),
            "invalid BOUNDS value must not default silently, got {err:?}"
        );
    }

    #[test]
    fn test_qps_bounds_value_on_value_less_type_is_error() {
        let qps = "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\nBOUNDS\n FR BND x1 0.0\nENDATA\n";
        let err = parse_qps_str(qps).unwrap_err();
        assert!(
            err.to_string().contains("does not take a value"),
            "free-format FR with a value must be rejected, got {err:?}"
        );
    }

    /// Sentinel: `UP` takes exactly one value; a second trailing token past
    /// the value (`UP BND x1 5.0 10.0`) must be a hard error, not a silently
    /// discarded value. Confirmed to fail without the check in
    /// `parse_bounds_entry` (`otspot-io/src/common/mod.rs`): the free-format
    /// branch used to read only `tokens[value_idx]`, so any token past it was
    /// silently dropped.
    #[test]
    fn test_qps_bounds_up_surplus_value_token_is_error() {
        let qps =
            "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\nBOUNDS\n UP BND x1 5.0 10.0\nENDATA\n";
        let err = parse_qps_str(qps).unwrap_err();
        assert!(
            err.to_string().contains("takes exactly one value"),
            "UP with a surplus value token must be rejected, got {err:?}"
        );
    }

    /// Fixed-format BOUNDS: MPS field values may contain internal spaces (FORPLAN
    /// column "A   22 1", bound set "BND-1"). `split_whitespace` over-splits such a
    /// line into 5+ tokens, so the parser must fall back to fixed-column extraction
    /// (col 14..22, value 24..36) rather than rejecting it as "extra tokens".
    #[test]
    fn test_qps_bounds_fixed_format_spaced_names() {
        // Build lines with fields at exact MPS byte offsets so the spaced column
        // name "AB CD" round-trips through fixed-column extraction.
        fn put(line: &mut Vec<u8>, at: usize, s: &str) {
            if line.len() < at + s.len() {
                line.resize(at + s.len(), b' ');
            }
            line[at..at + s.len()].copy_from_slice(s.as_bytes());
        }
        // COLUMNS: col[4..], row1[14..], val1[24..].
        let mut col = vec![b' '; 4];
        put(&mut col, 4, "AB CD");
        put(&mut col, 14, "obj");
        put(&mut col, 24, "1.0");
        let mut col_bv = vec![b' '; 4];
        put(&mut col_bv, 4, "EF GH");
        put(&mut col_bv, 14, "obj");
        put(&mut col_bv, 24, "2.0");
        let mut col_bnd_spaced = vec![b' '; 4];
        put(&mut col_bnd_spaced, 4, "IJ");
        put(&mut col_bnd_spaced, 14, "obj");
        put(&mut col_bnd_spaced, 24, "3.0");
        // BOUNDS: type[1..], bndname[4..], col[14..], val[24..].
        let mut bnd = vec![b' '; 1];
        put(&mut bnd, 1, "UP");
        put(&mut bnd, 4, "BND");
        put(&mut bnd, 14, "AB CD");
        put(&mut bnd, 24, "2640.");
        let mut bnd_bv = vec![b' '; 1];
        put(&mut bnd_bv, 1, "BV");
        put(&mut bnd_bv, 4, "BND");
        put(&mut bnd_bv, 14, "EF GH");
        let mut bnd_name_spaced = vec![b' '; 1];
        put(&mut bnd_name_spaced, 1, "BV");
        put(&mut bnd_name_spaced, 4, "B ND");
        put(&mut bnd_name_spaced, 14, "IJ");
        let qps = format!(
            "NAME          FIXED\nROWS\n N  obj\nCOLUMNS\n{}\n{}\n{}\nRHS\nBOUNDS\n{}\n{}\n{}\nENDATA\n",
            String::from_utf8(col).unwrap(),
            String::from_utf8(col_bv).unwrap(),
            String::from_utf8(col_bnd_spaced).unwrap(),
            String::from_utf8(bnd).unwrap(),
            String::from_utf8(bnd_bv).unwrap(),
            String::from_utf8(bnd_name_spaced).unwrap(),
        );
        let prob = parse_qps_str(&qps).expect("fixed-format spaced BOUNDS must parse");
        assert_eq!(prob.num_vars, 3);
        assert_eq!(
            prob.bounds[0].1, 2640.0,
            "UP bound value must be read from fixed columns"
        );
        assert_eq!(
            prob.bounds[1],
            (0.0, 1.0),
            "BV bound without a value must keep the full fixed-format spaced name"
        );
        assert_eq!(
            prob.bounds[2],
            (0.0, 1.0),
            "spaced fixed-format bound-set names must not block no-value bounds"
        );
    }

    #[test]
    fn test_qps_bounds_free_extra_token_not_fixed_fallback() {
        let qps = "NAME\nROWS\n N obj\nCOLUMNS\n    x1 obj 1.0\nBOUNDS\n BV BND x1 extra\nENDATA\n";
        let err = parse_qps_str(qps).unwrap_err();
        assert!(
            err.to_string().contains("does not take a value"),
            "malformed free-format BV must not be parsed via fixed fallback, got {err:?}"
        );
    }

    #[test]
    fn test_qps_rhs_fixed_format_numeric_spaced_row_name() {
        fn put(line: &mut Vec<u8>, at: usize, s: &str) {
            if line.len() < at + s.len() {
                line.resize(at + s.len(), b' ');
            }
            line[at..at + s.len()].copy_from_slice(s.as_bytes());
        }
        let mut row = vec![b' '; 1];
        put(&mut row, 1, "L");
        put(&mut row, 4, "C 1");
        let mut col = vec![b' '; 4];
        put(&mut col, 4, "X1");
        put(&mut col, 14, "obj");
        put(&mut col, 24, "1.0");
        let mut rhs = vec![b' '; 4];
        put(&mut rhs, 4, "rhs");
        put(&mut rhs, 14, "C 1");
        put(&mut rhs, 24, "10.0");
        let qps = format!(
            "NAME          FIXRHS\nROWS\n N  obj\n{}\nCOLUMNS\n{}\nRHS\n{}\nENDATA\n",
            String::from_utf8(row).unwrap(),
            String::from_utf8(col).unwrap(),
            String::from_utf8(rhs).unwrap(),
        );
        let prob = parse_qps_str(&qps).expect("fixed RHS with numeric spaced row must parse");
        assert_eq!(prob.b, vec![10.0]);
    }

    #[test]
    fn test_qps_ranges_fixed_format_numeric_spaced_row_name() {
        fn put(line: &mut Vec<u8>, at: usize, s: &str) {
            if line.len() < at + s.len() {
                line.resize(at + s.len(), b' ');
            }
            line[at..at + s.len()].copy_from_slice(s.as_bytes());
        }
        let mut row = vec![b' '; 1];
        put(&mut row, 1, "L");
        put(&mut row, 4, "C 1");
        let mut col = vec![b' '; 4];
        put(&mut col, 4, "X1");
        put(&mut col, 14, "obj");
        put(&mut col, 24, "1.0");
        let mut rhs = vec![b' '; 4];
        put(&mut rhs, 4, "rhs");
        put(&mut rhs, 14, "C 1");
        put(&mut rhs, 24, "10.0");
        let mut ranges = vec![b' '; 4];
        put(&mut ranges, 4, "rng");
        put(&mut ranges, 14, "C 1");
        put(&mut ranges, 24, "5.0");
        let qps = format!(
            "NAME          FIXRNG\nROWS\n N  obj\n{}\nCOLUMNS\n{}\nRHS\n{}\nRANGES\n{}\nENDATA\n",
            String::from_utf8(row).unwrap(),
            String::from_utf8(col).unwrap(),
            String::from_utf8(rhs).unwrap(),
            String::from_utf8(ranges).unwrap(),
        );
        let prob = parse_qps_str(&qps).expect("fixed RANGES with numeric spaced row must parse");
        // The Le row "C 1" (b=10) with a range of 5 expands to an upper+lower pair,
        // proving the fixed-format RANGES record parsed instead of being rejected.
        assert_eq!(
            prob.b.len(),
            2,
            "range row should expand to upper+lower constraints"
        );
        assert!(
            (prob.b[0] - 10.0).abs() < 1e-12,
            "original RHS preserved, got {:?}",
            prob.b
        );
    }

    #[test]
    fn test_qps_rhs_fixed_format_second_pair_spaced_row_name() {
        fn put(line: &mut Vec<u8>, at: usize, s: &str) {
            if line.len() < at + s.len() {
                line.resize(at + s.len(), b' ');
            }
            line[at..at + s.len()].copy_from_slice(s.as_bytes());
        }
        let mut row0 = vec![b' '; 1];
        put(&mut row0, 1, "L");
        put(&mut row0, 4, "R0");
        let mut row1 = vec![b' '; 1];
        put(&mut row1, 1, "L");
        put(&mut row1, 4, "C 1");
        let mut col = vec![b' '; 4];
        put(&mut col, 4, "X1");
        put(&mut col, 14, "obj");
        put(&mut col, 24, "1.0");
        let mut rhs = vec![b' '; 4];
        put(&mut rhs, 4, "rhs");
        put(&mut rhs, 14, "R0");
        put(&mut rhs, 24, "1.0");
        put(&mut rhs, 39, "C 1");
        put(&mut rhs, 49, "10.0");
        let qps = format!(
            "NAME          FIX2ND\nROWS\n N  obj\n{}\n{}\nCOLUMNS\n{}\nRHS\n{}\nENDATA\n",
            String::from_utf8(row0).unwrap(),
            String::from_utf8(row1).unwrap(),
            String::from_utf8(col).unwrap(),
            String::from_utf8(rhs).unwrap(),
        );
        let prob = parse_qps_str(&qps).expect("fixed RHS second pair with spaced row must parse");
        assert_eq!(prob.b, vec![1.0, 10.0]);
    }

    #[test]
    fn test_qps_rhs_fixed_format_spaced_set_name_numeric_row() {
        fn put(line: &mut Vec<u8>, at: usize, s: &str) {
            if line.len() < at + s.len() {
                line.resize(at + s.len(), b' ');
            }
            line[at..at + s.len()].copy_from_slice(s.as_bytes());
        }
        let mut row = vec![b' '; 1];
        put(&mut row, 1, "L");
        put(&mut row, 4, "1");
        let mut col = vec![b' '; 4];
        put(&mut col, 4, "X1");
        put(&mut col, 14, "obj");
        put(&mut col, 24, "1.0");
        let mut rhs = vec![b' '; 4];
        put(&mut rhs, 4, "R HS");
        put(&mut rhs, 14, "1");
        put(&mut rhs, 24, "10.0");
        let qps = format!(
            "NAME          FIXSET\nROWS\n N  obj\n{}\nCOLUMNS\n{}\nRHS\n{}\nENDATA\n",
            String::from_utf8(row).unwrap(),
            String::from_utf8(col).unwrap(),
            String::from_utf8(rhs).unwrap(),
        );
        let prob = parse_qps_str(&qps).expect("fixed RHS with spaced set name must parse");
        assert_eq!(prob.b, vec![10.0]);
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

    #[test]
    fn test_qps_rhs_odd_trailing_token_has_name_without_value_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0 c2\nQUADOBJ\n    x1 x1 1.0\nENDATA\n";
        let err = parse_qps_str(qps).unwrap_err();
        assert!(
            err.to_string()
                .contains("has a name without a matching value"),
            "odd RHS token must be rejected, got {err:?}"
        );
    }

    #[test]
    fn test_qps_ranges_odd_trailing_token_has_name_without_value_error() {
        let qps = "NAME\nROWS\n N obj\n L c1\nCOLUMNS\n    x1 obj 1.0 c1 1.0\nRHS\n    rhs c1 10.0\nRANGES\n    rng c1 1.0 c2\nQUADOBJ\n    x1 x1 1.0\nENDATA\n";
        let err = parse_qps_str(qps).unwrap_err();
        assert!(
            err.to_string()
                .contains("has a name without a matching value"),
            "odd RANGES token must be rejected, got {err:?}"
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
    // vector-name-omitted (shorthand) RHS/RANGES/BOUNDS + multi-vector
    // RHS/RANGES support.
    // -----------------------------------------------------------------------

    /// Sentinel: a 2-pair-per-line RHS with the vector name omitted, using
    /// numeric row names ("1", "2") that also look like plausible values —
    /// the exact shape that historically broke `blend.QPS`: a naive parser
    /// skips `parts[0]` as if it were a vector name, then mispairs
    /// `(parts[1], parts[2])` = `("10.0", 2.0)`, silently discarding row
    /// "1"'s value and fabricating a bogus entry for a nonexistent row.
    ///
    /// **No-op failure guarantee**: dropping the shorthand disambiguation in
    /// `common::parse_vector_entry` (always treating the first token as a
    /// vector name) makes `prob.b` become `[0.0, 20.0]` instead of
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

    // -----------------------------------------------------------------------
    // Fixed-column shorthand (review finding, P2): the free-format shorthand
    // tests above check declared names to spot an omitted vector/bound-set
    // name; the fixed-column reader did not, and unconditionally read
    // BOUNDS' column from field 3 and RHS/RANGES' vector name from field 2.
    // A fixed-column file whose BOUNDS section omits the bound-set name
    // therefore hard-errored with "BOUNDS line missing column name" instead
    // of parsing. See the MPS-module twins of these tests for the byte-level
    // layout rationale.
    // -----------------------------------------------------------------------

    /// Sentinel: fixed-column BOUNDS shorthand, column in field 2 and value
    /// staying in field 4 (field 3 blank) — the shape reported against
    /// `otspot_io::mps::parse_mps` and shared by the QPS reader via
    /// `common::parse_bounds_entry`.
    ///
    /// **No-op failure guarantee**: reverting `parse_bounds_entry`'s Fixed arm
    /// to always read the column from field 3 makes this `Err("BOUNDS line
    /// missing column name")` instead of `Ok` — verified by temporarily
    /// reverting.
    #[test]
    fn test_qps_bounds_fixed_shorthand_value_stays_in_field4() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps)
            .expect("fixed-column BOUNDS shorthand (col in field 2, value in field 4) must parse");
        assert_eq!(
            prob.bounds[0],
            (0.0, 5.0),
            "UP 'X 1' 5.0 (field-2 shorthand)"
        );
    }

    /// Sentinel: fixed-column BOUNDS shorthand where the whole line shifts
    /// one field left — column in field 2, value in field 3, field 4 blank.
    ///
    /// **No-op failure guarantee**: reverting the field-2/field-3 declared-name
    /// check makes this read field 3 (`"3.0"`) as the column name, which is
    /// undeclared, so `parse_qps_str` returns `Err` (`UndefinedReference`)
    /// instead of `Ok` — verified by temporarily reverting.
    #[test]
    fn test_qps_bounds_fixed_shorthand_uniform_shift() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps)
            .expect("fixed-column BOUNDS shorthand (col in field 2, value in field 3) must parse");
        assert_eq!(
            prob.bounds[0],
            (0.0, 5.0),
            "UP 'X 1' 5.0 (field-2/field-4 shorthand)"
        );
        assert_eq!(
            prob.bounds[1],
            (0.0, 3.0),
            "UP X2 3.0 (field-2/field-3 shorthand)"
        );
    }

    /// Sentinel: a fixed-column BOUNDS line in the genuine standard form
    /// (bound-set name present in field 2, column in field 3) must still be
    /// read as standard, not misdetected as shorthand.
    #[test]
    fn test_qps_bounds_fixed_standard_form_not_misread_as_shorthand() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps)
            .expect("standard fixed-column BOUNDS (with bound-set name) must parse");
        assert_eq!(
            prob.bounds[0],
            (0.0, 5.0),
            "UP BND 'X 1' 5.0 (standard form)"
        );
    }

    /// Sentinel (review finding): a grid-aligned fixed-column BOUNDS line with
    /// a second numeric value in field 5 (`UP BND X 1 5.0 <field 5>10.0`) must
    /// be a hard error, not silently accepted with the field 5 content
    /// dropped. This is the case the free-format surplus-token check
    /// (`test_qps_bounds_up_surplus_value_token_is_error`) does *not* cover:
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
    fn test_qps_bounds_fixed_grid_aligned_field5_surplus_value_is_error() {
        let qps = concat!(
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
        let err = parse_qps_str(qps).unwrap_err();
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
    fn test_qps_bounds_fixed_grid_aligned_field5_junk_is_error() {
        let qps = concat!(
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
        let err = parse_qps_str(qps).unwrap_err();
        assert!(
            err.to_string().contains("field 5"),
            "grid-aligned BOUNDS with junk content in field 5 must be rejected, got {err:?}"
        );
    }

    /// Sentinel: fixed-column RHS with the vector name omitted, first row
    /// name landing in field 2 instead of field 3, shifting both (row,
    /// value) pairs one field left: fields 2/3 carry the first pair, 4/5
    /// the second, field 6 unused.
    ///
    /// **No-op failure guarantee**: reverting `parse_vector_entry`'s Fixed arm
    /// to always read the vector name from field 2 and pairs from
    /// (field 3, field 4) / (field 5, field 6) reads field 2 ("BR   1 1") as
    /// the vector name and field 3 ("6.") as a row name with no declared row
    /// of that name, so `parse_qps_str` returns `Err` (`UndefinedReference`)
    /// instead of `Ok` — verified by temporarily reverting.
    #[test]
    fn test_qps_rhs_fixed_shorthand_vector_name_omitted() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps)
            .expect("fixed-column RHS with the vector name omitted (field-2 shorthand) must parse");
        assert_eq!(prob.num_constraints, 2, "'BR   1 1' and 'R2'");
        assert_eq!(
            prob.b,
            vec![6.0, 3.0],
            "both RHS values must survive the field-2 shorthand read"
        );
    }

    // -----------------------------------------------------------------------
    // Redesign (review finding, P1): mirrors the MPS module's tests — see
    // `otspot_io::mps::tests` for the rationale. `parse_vector_entry` /
    // `parse_bounds_entry` are shared by both formats via `common`.
    // -----------------------------------------------------------------------

    /// Sentinel (P1 regression): a fixed-column RHS line in genuine standard
    /// form must parse even though a declared row is also named `RHS`.
    ///
    /// **No-op failure guarantee**: reverting to the name-membership check
    /// misreads field 2 ("RHS") as the first row name and field 3 ("LIM 1")
    /// as its value, which fails to parse as a number — `parse_qps_str`
    /// returns `Err("Invalid RHS value 'LIM 1'")` instead of `Ok` — verified
    /// by temporarily reverting.
    #[test]
    fn test_qps_rhs_fixed_vector_name_collides_with_declared_row_name() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps).expect(
            "standard fixed-column RHS must parse even when the vector name collides with a \
             declared row name",
        );
        assert_eq!(prob.num_constraints, 2, "'LIM 1' and 'RHS'");
        assert_eq!(
            prob.b,
            vec![10.0, 4.0],
            "RHS vector 'RHS': LIM 1 -> 10.0, row 'RHS' -> 4.0 (both L rows, no sign flip)"
        );
    }

    /// Sentinel (P1 regression): a fixed-column RANGES line in genuine
    /// standard form must parse even though a declared row is also named
    /// `RNG`.
    #[test]
    fn test_qps_ranges_fixed_vector_name_collides_with_declared_row_name() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps).expect(
            "standard fixed-column RANGES must parse even when the vector name collides with a \
             declared row name",
        );
        // RANGES 'RNG' applies range 2.0 to row 'LIM 1' (L, rhs=10.0): base Le
        // part keeps rhs=10.0; the split-off Ge part (lower=10.0-2.0=8.0) is
        // normalized to Le by negating both sides, giving rhs=-8.0. Row 'RNG'
        // (L, rhs=4.0) carries no RANGES entry, so it passes through as-is.
        assert_eq!(
            prob.num_constraints, 3,
            "'LIM 1' + 'RNG' + LIM 1's split-off row"
        );
        assert_eq!(prob.b, vec![10.0, 4.0, -8.0]);
    }

    /// Sentinel (P1 regression): a fixed-column BOUNDS line in genuine
    /// standard form must parse even when the bound-set name collides with a
    /// declared column name (`BND` is both the bound-set name and a real
    /// column here).
    #[test]
    fn test_qps_bounds_fixed_bound_set_name_collides_with_declared_column_name() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps).expect(
            "standard fixed-column BOUNDS must parse even when the bound-set name collides with \
             a declared column name",
        );
        assert_eq!(
            prob.bounds[0],
            (0.0, f64::INFINITY),
            "'BND' column itself is untouched by its own name appearing as a bound-set name"
        );
        assert_eq!(prob.bounds[1], (0.0, 5.0), "UP 'X1' 5.0 (standard form)");
    }

    /// Sentinel (P1 regression): same bound-set/column-name collision as
    /// above, but with a non-value-taking bound type (`FR`).
    #[test]
    fn test_qps_bounds_fixed_bound_set_name_collides_with_declared_column_name_non_value_type() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps).expect(
            "standard fixed-column FR BOUNDS must parse even when the bound-set name collides \
             with a declared column name",
        );
        assert_eq!(
            prob.bounds[0],
            (0.0, f64::INFINITY),
            "'BND' column untouched"
        );
        assert_eq!(
            prob.bounds[1],
            (f64::NEG_INFINITY, f64::INFINITY),
            "FR 'X1' (standard form)"
        );
    }

    /// Sentinel: fixed-column RANGES with the vector name omitted (no test
    /// previously covered this — only the RHS analogue did).
    #[test]
    fn test_qps_ranges_fixed_shorthand_vector_name_omitted() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps).expect(
            "fixed-column RANGES with the vector name omitted (field-2 shorthand) must parse",
        );
        // 'BR   1 1' (L, rhs=10.0, range=2.0): Le part rhs=10.0; split-off Ge
        // part (lower=8.0) normalized to Le gives rhs=-8.0.
        // 'R2' (L, rhs=20.0, range=4.0): Le part rhs=20.0; split-off part
        // (lower=16.0) normalized to Le gives rhs=-16.0.
        assert_eq!(prob.num_constraints, 4);
        assert_eq!(prob.b, vec![10.0, 20.0, -8.0, -16.0]);
    }

    /// Sentinel: a fixed-column RANGES line where field 2 must be reread as a
    /// row name, but the shorthand reading then leaves stray content in
    /// field 6 — neither reading is valid, so this must hard-error.
    #[test]
    fn test_qps_ranges_fixed_shorthand_trailing_field6_is_error() {
        let qps = concat!(
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
        let err = parse_qps_str(qps).expect_err("stray content in field 6 must be a hard error");
        let message = format!("{}", err);
        assert!(
            message.contains("field 6 must be blank"),
            "unexpected error message: {}",
            message
        );
    }

    /// Sentinel: a fixed-column BOUNDS line for a non-value-taking type
    /// (`FR`) with the bound-set name omitted and the column shifted into
    /// field 2, leaving field 3 and field 4 both blank.
    #[test]
    fn test_qps_bounds_fixed_shorthand_non_value_type_no_bound_name() {
        let qps = concat!(
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
        let prob = parse_qps_str(qps)
            .expect("fixed-column FR shorthand (bound-set name omitted, no value) must parse");
        assert_eq!(
            prob.bounds[0],
            (f64::NEG_INFINITY, f64::INFINITY),
            "FR 'X 1' (shorthand, no bound-set name, no value)"
        );
    }

    /// Sentinel: an RHS line where neither the standard nor the shorthand
    /// reading is parseable must hard-error with the standard reading's
    /// diagnostic (the presumptive layout).
    #[test]
    fn test_qps_rhs_fixed_neither_reading_parses_reports_standard_diagnostic() {
        let qps = concat!(
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
        let err =
            parse_qps_str(qps).expect_err("a line valid under neither reading must hard-error");
        let message = format!("{}", err);
        assert!(
            message.contains("has no matching value"),
            "expected the standard reading's diagnostic (missing value), got: {}",
            message
        );
    }
}
