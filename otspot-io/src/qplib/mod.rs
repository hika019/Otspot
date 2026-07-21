//! QPLIB format parser.
//!
//! QPLIB (<https://qplib.zib.de/>) is a token-stream format distinct from
//! MPS/QPS.  The parser uses a streaming tokenizer so that 200 MB+ files
//! are handled without OOM.

mod parser;
mod token_stream;

use std::io::BufRead;
use std::path::Path;

use otspot_core::mip::{MilpProblem, MiqpProblem};
use otspot_core::qp::QpProblem;

use token_stream::TokenStream;

/// Errors produced by the QPLIB parser.
#[non_exhaustive]
#[derive(Debug)]
pub enum QplibError {
    /// I/O error reading from the source.
    IoError(std::io::Error),
    /// Malformed content.
    ParseError(String),
    /// Problem type not supported (e.g. mixed-integer M/G/S variable types).
    UnsupportedType(String),
}

impl std::fmt::Display for QplibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QplibError::IoError(e) => write!(f, "I/O error: {}", e),
            QplibError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            QplibError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
        }
    }
}

impl std::error::Error for QplibError {}

impl From<std::io::Error> for QplibError {
    fn from(e: std::io::Error) -> Self {
        QplibError::IoError(e)
    }
}

/// Parsed result of a QPLIB file.
///
/// Continuous-variable problems return [`QplibProblem::Qp`]; problems with binary (`B`) or
/// integer (`I`) variables return [`QplibProblem::Milp`] (zero-Q) or [`QplibProblem::Miqp`] (non-zero Q).
#[derive(Debug)]
pub enum QplibProblem {
    /// Continuous-variable QP / QCQP / LP.
    Qp(QpProblem),
    /// Mixed-integer LP (linear objective, binary or integer variables).
    Milp(MilpProblem),
    /// Mixed-integer QP (quadratic objective, binary or integer variables).
    Miqp(MiqpProblem),
}

/// Parse a QPLIB file from `path`.
///
/// Uses a streaming tokenizer — O(1) memory regardless of file size.
pub fn parse_qplib(path: &Path) -> Result<QplibProblem, QplibError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    parser::parse_token_stream(TokenStream::from_reader(reader))
}

/// Parse a QPLIB string.
pub fn parse_qplib_str(input: &str) -> Result<QplibProblem, QplibError> {
    parser::parse_token_stream(TokenStream::from_str(input))
}

/// Parse from any `BufRead` source.
pub fn parse_qplib_reader<R: BufRead + 'static>(reader: R) -> Result<QplibProblem, QplibError> {
    parser::parse_token_stream(TokenStream::from_reader(reader))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unwrap_qp(r: QplibProblem) -> QpProblem {
        match r {
            QplibProblem::Qp(p) => p,
            other => panic!("expected Qp, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_qplib_simple() {
        let qplib = "\
SIMPLE_QP
QCL
minimize
2 # number of variables
1 # number of constraints
2 # number of quadratic terms in objective
1 1 1.0
2 2 1.0
0.0 # default linear obj coefficient
0 # number of non-default linear obj coefficients
0.0 # objective constant
2 # number of linear terms in all constraints
1 1 1.0
1 2 1.0
1.79769313486232E+308 # infinity
1.0 # default left-hand-side
0 # number of non-default left-hand-sides
1.0 # default right-hand-side
0 # number of non-default right-hand-sides
0.0 # default variable lower bound
0 # number of non-default variable lower bounds
1.79769313486232E+308 # default variable upper bound
0 # number of non-default variable upper bounds
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 1);
        assert_eq!(
            prob.constraint_types[0],
            otspot_core::problem::ConstraintType::Eq
        );
        assert_eq!(prob.q.nnz(), 2);
    }

    #[test]
    fn test_parse_qplib_unconstrained() {
        let qplib = "\
NO_CON
QCN
minimize
2 # vars
2 # qobj
1 1 2.0
2 2 2.0
0.0 # default b0
0 # non-default b0
0.0 # obj constant
1.79769313486232E+308 # infinity
-1.79769313486232E+308 # default var lb
0
1.79769313486232E+308 # default var ub
0
0.0
0
0.0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 0);
        assert_eq!(prob.q.nnz(), 2);
    }

    #[test]
    fn qplib_fractional_objective_count_is_rejected_in_official_n_layout() {
        let qplib = "\
FRACTIONAL_COUNT
QCN
minimize
1
0
0.0
0.5
0.0
0
1.79769313486232E+308
-1.79769313486232E+308
0
1.79769313486232E+308
0
";
        let err = parse_qplib_str(qplib)
            .expect_err("a fractional entry count must not be truncated to zero");
        assert!(
            err.to_string()
                .contains("number of non-default objective linear terms"),
            "{err}"
        );
    }

    #[test]
    fn test_parse_qcq_equality_diagonal_q() {
        let qplib = "\
QCQ_EQ_DIAG
QCQ
minimize
2 # n
1 # m
0 # nqobj
0.0 # default b0
2 # non-default b0
1 1.0
2 1.0
0.0 # q0
2 # n_con_quad_terms
1 1 1 2.0
1 2 2 4.0
0 # n_con_lin_terms
1.79769313486232E+308 # inf
5.0 # default lb_con
0 # non-default lb_con
5.0 # default ub_con
0 # non-default ub_con
0.0 # default lb_var
0 # non-default lb_var
1.0 # default ub_var
0 # non-default ub_var
0.0 # primal default
0
0.0 # dual default
0
0.0 # bound dual default
0
0 # non-default var names
0 # non-default con names
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.num_constraints, 1);
        assert_eq!(
            prob.constraint_types[0],
            otspot_core::problem::ConstraintType::Eq
        );
        assert_eq!(prob.b[0], 5.0);
        assert_eq!(prob.quadratic_constraints.len(), 1);
        assert_eq!(prob.quadratic_constraints[0].nnz(), 2);
        assert!(prob.quadratic_constraints.iter().any(|q| q.nnz() > 0));
    }

    #[test]
    fn test_parse_qcq_mixed_linear_and_quadratic() {
        let qplib = "\
QCQ_MIXED
QCQ
minimize
3 # n
2 # m
3 # nqobj: 1/2*(2x1^2+2x2^2+2x3^2)
1 1 2.0
2 2 2.0
3 3 2.0
0.0 # default b0
0 # non-default b0
0.0 # q0
2 # n_con_quad_terms: Q_2 has (1,1,1.0),(3,3,1.0)
2 1 1 1.0
2 3 3 1.0
3 # n_con_lin_terms: con1: x1+x2, con2: x2
1 1 1.0
1 2 1.0
2 2 1.0
1.79769313486232E+308 # inf
-1.79769313486232E+308 # default lb_con
2 # non-default lb_con
1 -1.79769313486232E+308
2 3.0
4.0 # default ub_con
1 # non-default ub_con
2 3.0
0.0 # default lb_var
0
1.79769313486232E+308 # default ub_var
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 3);
        assert_eq!(prob.num_constraints, 2);
        assert_eq!(
            prob.constraint_types[0],
            otspot_core::problem::ConstraintType::Le
        );
        assert_eq!(
            prob.constraint_types[1],
            otspot_core::problem::ConstraintType::Eq
        );
        assert_eq!(prob.b[0], 4.0);
        assert_eq!(prob.b[1], 3.0);
        assert_eq!(prob.quadratic_constraints.len(), 2);
        assert_eq!(prob.quadratic_constraints[0].nnz(), 0);
        assert_eq!(prob.quadratic_constraints[1].nnz(), 2);
        assert_eq!(prob.quadratic_constraints[1].n, 3);
    }

    #[test]
    fn test_parse_qcq_range_constraint_sign_flip() {
        let qplib = "\
QCQ_RANGE
QCQ
minimize
4 # n
3 # m
0 # nqobj
0.0 # default b0
0 # non-default b0
0.0 # q0
3 # n_con_quad_terms
2 1 1 2.0
3 1 1 1.0
3 2 1 0.5
1 # n_con_lin_terms
1 1 1.0
1.79769313486232E+308 # inf
-1.79769313486232E+308 # default lb_con
3 # non-default lb_con
1 -1.79769313486232E+308
2 1.0
3 -1.79769313486232E+308
5.0 # default ub_con
2 # non-default ub_con
2 3.0
3 10.0
0.0 # default lb_var
0
1.79769313486232E+308 # default ub_var
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 4);
        assert_eq!(prob.num_constraints, 4);
        for ct in &prob.constraint_types {
            assert_eq!(*ct, otspot_core::problem::ConstraintType::Le);
        }
        assert!((prob.b[0] - 5.0).abs() < 1e-12);
        assert!((prob.b[1] - 3.0).abs() < 1e-12);
        assert!((prob.b[2] - (-1.0)).abs() < 1e-12);
        assert!((prob.b[3] - 10.0).abs() < 1e-12);
        assert_eq!(prob.quadratic_constraints.len(), 4);
        assert_eq!(prob.quadratic_constraints[0].nnz(), 0);
        assert_eq!(prob.quadratic_constraints[1].nnz(), 1);
        assert_eq!(prob.quadratic_constraints[2].nnz(), 1);
        assert_eq!(prob.quadratic_constraints[3].nnz(), 3);
        let q_ub = &prob.quadratic_constraints[1];
        let q_lb = &prob.quadratic_constraints[2];
        assert!(q_ub.triplets.iter().all(|&(_, _, v)| v > 0.0));
        assert!(q_lb.triplets.iter().all(|&(_, _, v)| v < 0.0));
    }

    #[test]
    fn test_qcl_no_quadratic_constraints() {
        let qplib = "\
QCL_ROUND_TRIP
QCL
minimize
2
1
2
1 1 1.0
2 2 1.0
0.0
0
0.0
2
1 1 1.0
1 2 1.0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
1.79769313486232E+308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        assert!(
            prob.quadratic_constraints.is_empty(),
            "QCL must produce empty quadratic_constraints"
        );
    }

    #[test]
    fn test_parse_qplib_integer_to_milp() {
        let qplib = "\
INT_LP
QIL
minimize
2
1
0
0.0
0
0.0
0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
1.79769313486232E+308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let parsed = parse_qplib_str(qplib).unwrap();
        let milp = match parsed {
            QplibProblem::Milp(m) => m,
            other => panic!("expected Milp, got {:?}", other),
        };
        assert_eq!(milp.lp.num_vars, 2);
        assert_eq!(milp.integer_vars, vec![0, 1]);
        assert_eq!(milp.lp.num_constraints, 1);
    }

    #[test]
    fn test_parse_qplib_integer_to_miqp() {
        let qplib = "\
INT_QP
QIL
minimize
2
1
2
1 1 1.0
2 2 1.0
0.0
0
0.0
2
1 1 1.0
1 2 1.0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
1.79769313486232E+308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let parsed = parse_qplib_str(qplib).unwrap();
        let miqp = match parsed {
            QplibProblem::Miqp(m) => m,
            other => panic!("expected Miqp, got {:?}", other),
        };
        assert_eq!(miqp.qp.num_vars, 2);
        assert_eq!(miqp.integer_vars, vec![0, 1]);
        assert_eq!(miqp.qp.q.nnz(), 2);
    }

    #[test]
    fn test_parse_qplib_binary_to_milp() {
        let qplib = "\
BIN_LP
CBL
minimize
2
1
0
0.0
0
0.0
2
1 1 1.0
1 2 1.0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
0.0
0
0.0
0
0
0
";
        let parsed = parse_qplib_str(qplib).unwrap();
        let milp = match parsed {
            QplibProblem::Milp(m) => m,
            other => panic!("expected Milp, got {:?}", other),
        };
        assert_eq!(milp.lp.num_vars, 2);
        assert_eq!(milp.integer_vars, vec![0, 1]);
        for &(lb, ub) in &milp.lp.bounds {
            assert!((lb - 0.0).abs() < 1e-12);
            assert!((ub - 1.0).abs() < 1e-12);
        }
        assert_eq!(milp.lp.num_constraints, 1);
    }

    #[test]
    fn test_parse_qplib_binary_quad_to_miqp() {
        let qplib = "\
BIN_QP
CBL
minimize
2
1
2
1 1 2.0
2 2 2.0
0.0
0
0.0
2
1 1 1.0
1 2 1.0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
0.0
0
0.0
0
0
0
";
        let parsed = parse_qplib_str(qplib).unwrap();
        let miqp = match parsed {
            QplibProblem::Miqp(m) => m,
            other => panic!("expected Miqp, got {:?}", other),
        };
        assert_eq!(miqp.qp.num_vars, 2);
        assert_eq!(miqp.integer_vars, vec![0, 1]);
        for &(lb, ub) in &miqp.qp.bounds {
            assert!((lb - 0.0).abs() < 1e-12);
            assert!((ub - 1.0).abs() < 1e-12);
        }
        assert_eq!(miqp.qp.q.nnz(), 2);
    }

    /// memo 33 (P1): a binary/integer QPLIB with a *linear* objective
    /// (`nqobj = 0`) but non-empty **quadratic constraints** (`con_char = 'Q'`)
    /// must become a `Miqp` that keeps the quadratic constraints — not a `Milp`
    /// that silently drops them. Independent oracle: the file below has one
    /// binary variable, linear objective `min x1`, and the single quadratic
    /// constraint `x1^2 <= 0`; only `Miqp` can carry that constraint.
    #[test]
    fn test_parse_qplib_binary_linear_obj_quad_constraint_to_miqp() {
        let qplib = "\
LBQ_test
LBQ
minimize
1
1
0
1.0
0
0.0
1
1 1 1 2.0
0
1E+30
-1E+30
0
0.0
0
0.0
0
0.0
0
0.0
0
0
0
";
        let parsed = parse_qplib_str(qplib).unwrap();
        let miqp = match parsed {
            QplibProblem::Miqp(m) => m,
            other => panic!(
                "expected Miqp (quadratic constraints present), got {:?}",
                other
            ),
        };
        assert_eq!(miqp.qp.num_vars, 1);
        assert_eq!(miqp.integer_vars, vec![0]);
        assert_eq!(miqp.qp.q.nnz(), 0, "objective Q is zero (linear objective)");
        assert_eq!(
            miqp.qp.quadratic_constraints.len(),
            1,
            "the quadratic constraint must be preserved"
        );
        assert!(
            miqp.qp.quadratic_constraints[0].nnz() > 0,
            "constraint x1^2 must not be dropped"
        );
    }

    #[test]
    fn test_parse_qplib_mixed_integer_unsupported() {
        let qplib = "\
MIXED_QP
QML
minimize
2
1
0
0.0
0
0.0
0
1.79769313486232E+308
1.0
0
1.0
0
";
        assert!(matches!(
            parse_qplib_str(qplib),
            Err(QplibError::UnsupportedType(_))
        ));
    }

    fn data_path(rel: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(rel)
    }

    #[test]
    fn test_parse_qcq_file_1157_structure() {
        let path = data_path("data/qplib_unsupported/QPLIB_1157.qplib");
        if !path.exists() {
            return;
        }
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1157 parse"));
        assert_eq!(prob.num_vars, 40);
        assert_eq!(prob.num_constraints, 9);
        for i in 0..8 {
            assert_eq!(
                prob.constraint_types[i],
                otspot_core::problem::ConstraintType::Eq
            );
        }
        assert_eq!(
            prob.constraint_types[8],
            otspot_core::problem::ConstraintType::Le
        );
        let expected_b = [0.56_f64, -0.16, -0.4, -0.25, 0.45, 0.3, 0.99, 0.77, 16.22];
        for (i, &exp) in expected_b.iter().enumerate() {
            assert!((prob.b[i] - exp).abs() < 1e-10);
        }
        assert_eq!(prob.quadratic_constraints.len(), 9);
        for i in 0..8 {
            assert_eq!(prob.quadratic_constraints[i].nnz(), 0);
        }
        let qk = &prob.quadratic_constraints[8];
        assert_eq!(qk.n, 40);
        assert_eq!(qk.nnz(), 1516);
        let v00 = qk
            .triplets
            .iter()
            .find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v)
            .expect("Q_9 must have (0,0) entry");
        assert!((v00 - 0.38).abs() < 1e-10);
    }

    #[test]
    fn test_parse_qcq_file_1353_structure() {
        let path = data_path("data/qplib_unsupported/QPLIB_1353.qplib");
        if !path.exists() {
            return;
        }
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1353 parse"));
        assert_eq!(prob.num_vars, 50);
        assert_eq!(prob.num_constraints, 6);
        for i in 0..5 {
            assert_eq!(
                prob.constraint_types[i],
                otspot_core::problem::ConstraintType::Eq
            );
        }
        assert_eq!(
            prob.constraint_types[5],
            otspot_core::problem::ConstraintType::Le
        );
        let expected_b = [0.13_f64, -0.4, 0.1, -0.63, 0.57, 18.74];
        for (i, &exp) in expected_b.iter().enumerate() {
            assert!((prob.b[i] - exp).abs() < 1e-10);
        }
        assert_eq!(prob.quadratic_constraints.len(), 6);
        for i in 0..5 {
            assert_eq!(prob.quadratic_constraints[i].nnz(), 0);
        }
        let qk = &prob.quadratic_constraints[5];
        assert_eq!(qk.n, 50);
        assert_eq!(qk.nnz(), 2372);
        let v00 = qk
            .triplets
            .iter()
            .find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v)
            .expect("Q_6 must have (0,0) entry");
        assert!((v00 - 0.46).abs() < 1e-10);
    }

    #[test]
    fn test_parse_qcq_file_1055_all_le_dense_q() {
        let path = data_path("data/qplib_unsupported/QPLIB_1055.qplib");
        if !path.exists() {
            return;
        }
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1055 parse"));
        assert_eq!(prob.num_vars, 40);
        assert_eq!(prob.num_constraints, 20);
        for i in 0..20 {
            assert_eq!(
                prob.constraint_types[i],
                otspot_core::problem::ConstraintType::Le
            );
        }
        assert!((prob.b[0] - 71.197).abs() < 1e-10);
        assert!((prob.b[19] - 30.278).abs() < 1e-10);
        assert_eq!(prob.quadratic_constraints.len(), 20);
        for i in 0..20 {
            let qk = &prob.quadratic_constraints[i];
            assert_eq!(qk.n, 40);
            assert_eq!(qk.nnz(), 1600);
        }
        let qk0 = &prob.quadratic_constraints[0];
        let v00 = qk0
            .triplets
            .iter()
            .find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v)
            .expect("Q_1 must have (0,0) entry");
        assert!((v00 - 0.839).abs() < 1e-10);
    }

    #[test]
    fn test_parse_qcq_file_1493_structure() {
        let path = data_path("data/qplib_unsupported/QPLIB_1493.qplib");
        if !path.exists() {
            return;
        }
        let prob = unwrap_qp(parse_qplib(path.as_path()).expect("QPLIB_1493 parse"));

        assert_eq!(prob.num_vars, 40);
        assert_eq!(prob.num_constraints, 5);

        for i in 0..4 {
            assert_eq!(
                prob.constraint_types[i],
                otspot_core::problem::ConstraintType::Eq,
                "constraint {i} must be Eq"
            );
        }
        assert_eq!(
            prob.constraint_types[4],
            otspot_core::problem::ConstraintType::Le
        );

        let expected_b = [-0.17_f64, 0.51, -0.41, -0.15, 67.98];
        for (i, &exp) in expected_b.iter().enumerate() {
            assert!(
                (prob.b[i] - exp).abs() < 1e-10,
                "b[{i}]: expected {exp}, got {}",
                prob.b[i]
            );
        }

        assert_eq!(prob.quadratic_constraints.len(), 5);
        for i in 0..4 {
            assert_eq!(
                prob.quadratic_constraints[i].nnz(),
                0,
                "Q_k[{i}] must be empty"
            );
        }
        let qk = &prob.quadratic_constraints[4];
        assert_eq!(qk.n, 40);
        assert_eq!(qk.nnz(), 1547, "Q_5 nnz: 37 diag + 755 off-diag*2 = 1547");
        // Q_5[0,0] = 1.88 (file: 5 1 1 1.88)
        let v00 = qk
            .triplets
            .iter()
            .find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v)
            .expect("Q_5 must have (0,0) entry");
        assert!((v00 - 1.88).abs() < 1e-10, "Q_5[0,0] must be 1.88");
    }

    #[test]
    fn test_parse_all_qcq_unsupported_files() {
        let dir = data_path("data/qplib_unsupported");
        if !dir.exists() {
            return;
        }
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("qplib") {
                continue;
            }
            parse_qplib(path.as_path()).unwrap_or_else(|e| {
                panic!("parse failed for {}: {e}", path.display());
            });
            count += 1;
        }
        assert!(
            count > 0,
            "no .qplib files found in data/qplib_unsupported/"
        );
    }

    /// Regression: every tracked file in data/qplib/ parses without error.
    ///
    /// Binary/integer files (CBL etc.) now parse as `Milp`/`Miqp`.
    /// Mixed-integer types (M/G/S) still produce `UnsupportedType`.
    /// `ParseError` or `IoError` on any file is a regression.
    #[test]
    fn test_parse_existing_qplib_files_regression() {
        let dir = data_path("data/qplib");
        if !dir.exists() {
            return;
        }
        let mut count = 0;
        let mut count_mip = 0usize;
        let mut count_unsupported = 0usize;
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("qplib") {
                continue;
            }
            match parse_qplib(path.as_path()) {
                Ok(QplibProblem::Qp(_)) => {}
                Ok(QplibProblem::Milp(_)) | Ok(QplibProblem::Miqp(_)) => {
                    count_mip += 1;
                }
                Err(QplibError::UnsupportedType(_)) => {
                    count_unsupported += 1;
                }
                Err(e) => panic!(
                    "parse regression: {} failed with unexpected error: {e}",
                    path.display()
                ),
            }
            count += 1;
        }
        assert!(count > 0, "no .qplib files found in data/qplib/");
        assert!(
            count_mip > 0,
            "expected at least one binary/integer file to parse as Milp/Miqp (got 0 out of {count})"
        );
        let _ = count_unsupported;
    }

    /// Accumulation sentinel: parsing every file in data/qplib/ in sequence
    /// must return live allocations to ~baseline after each result is dropped.
    ///
    /// **No-op failure guarantee**: retaining results across iterations causes
    /// `live` to grow past `LIVE_RESIDUAL_LIMIT` → FAIL.
    #[test]
    fn test_parse_sweep_no_memory_accumulation() {
        let dir = data_path("data/qplib");
        if !dir.exists() {
            return;
        }
        const LIVE_RESIDUAL_LIMIT: isize = 4 * 1024 * 1024;

        let files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .expect("read_dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("qplib"))
            .collect();

        crate::peak_alloc::begin();
        for path in &files {
            let _ = parse_qplib(path.as_path());
            let live = crate::peak_alloc::current_bytes();
            assert!(
                live <= LIVE_RESIDUAL_LIMIT,
                "live allocations {live} B remain after dropping {} — parser is \
                 retaining results across files (accumulation bug); limit {LIVE_RESIDUAL_LIMIT} B",
                path.display()
            );
        }
    }

    /// Memory sentinel: QPLIB_8500 (25 MB, 1.2 M NZ) must parse without
    /// exceeding `QPLIB_8500_PARSE_PEAK_LIMIT` of concurrently live allocations.
    ///
    /// **No-op failure guarantee**: removing `drop(a_triplets)` raises peak to
    /// ~129 MB → assertion fires. Threshold calibrated: Vec path ~100.4 MB; limit 115 MB.
    #[test]
    fn test_memory_sentinel_no_double_hashmap_qplib8500() {
        let path = data_path("data/qplib/QPLIB_8500.qplib");
        if !path.exists() {
            return;
        }
        const QPLIB_8500_PARSE_PEAK_LIMIT: usize = 115 * 1024 * 1024;

        crate::peak_alloc::begin();
        parse_qplib(path.as_path()).expect("QPLIB_8500 must parse without OOM");
        let peak = crate::peak_alloc::peak_bytes();

        assert!(
            peak <= QPLIB_8500_PARSE_PEAK_LIMIT,
            "QPLIB_8500 parse peak allocation {:.1} MB exceeds {:.1} MB limit.\n\
             Vec+drop path expected ~100.4 MB; no-drop path is ~129 MB.\n\
             Check that drop(a_triplets) is present before from_triplets call.",
            peak as f64 / 1_048_576.0,
            QPLIB_8500_PARSE_PEAK_LIMIT as f64 / 1_048_576.0
        );
    }

    /// Memory probe: large DCL files (QPLIB_8547 = 144 MB, QPLIB_9008 = 210 MB).
    /// Generous 2 GB limit detects catastrophic O(n²) regressions.
    #[test]
    fn test_memory_probe_large_dcl_files() {
        const DCL_PROBE_LIMIT: usize = 2 * 1024 * 1024 * 1024;

        for name in &["QPLIB_8547", "QPLIB_9008"] {
            let path = data_path(&format!("data/qplib/{name}.qplib"));
            if !path.exists() {
                continue;
            }
            crate::peak_alloc::begin();
            let result = parse_qplib(path.as_path());
            let peak = crate::peak_alloc::peak_bytes();
            assert!(
                result.is_ok() || matches!(result, Err(QplibError::UnsupportedType(_))),
                "{name} parse must not fail with ParseError or IoError: {:?}",
                result.err()
            );
            assert!(
                peak <= DCL_PROBE_LIMIT,
                "{name} parse peak {:.1} MB exceeds {:.1} GB catastrophic-regression limit.\n\
                 Expected O(nnz) memory not O(n·m) or O(n²).",
                peak as f64 / 1_048_576.0,
                DCL_PROBE_LIMIT as f64 / 1_073_741_824.0
            );
        }
    }

    /// Memory sentinel: QPLIB_8683 (DCQ, n=200008, m=140000).
    ///
    /// Old `CscMatrix::from_triplets(..., n, n)` per filled slot → 224 GB OOM.
    /// `QcqpMatrix` (COO): ~7 MB total.
    ///
    /// **No-op failure guarantee**: reverting to `CscMatrix::from_triplets(..., n, n)`
    /// causes OOM before this assertion.
    #[test]
    fn test_memory_sentinel_qplib8683_qcqp() {
        let path = data_path("data/qplib/QPLIB_8683.qplib");
        if !path.exists() {
            return;
        }
        const QPLIB_8683_PEAK_LIMIT: usize = 300 * 1024 * 1024;

        crate::peak_alloc::begin();
        parse_qplib(path.as_path()).expect("QPLIB_8683 must parse without OOM");
        let peak = crate::peak_alloc::peak_bytes();

        assert!(
            peak <= QPLIB_8683_PEAK_LIMIT,
            "QPLIB_8683 parse peak {:.1} MB exceeds {:.1} MB limit.\n\
             QcqpMatrix (COO) path expected < 50 MB; CscMatrix per-slot causes 224 GB OOM.\n\
             Check that quadratic_constraints uses QcqpMatrix not CscMatrix::from_triplets.",
            peak as f64 / 1_048_576.0,
            QPLIB_8683_PEAK_LIMIT as f64 / 1_048_576.0
        );
    }

    // -----------------------------------------------------------------------
    // QCQP sparse-init sentinel
    // -----------------------------------------------------------------------

    fn make_synthetic_qcq_content(n: usize, m: usize) -> String {
        let mut s = String::with_capacity(256);
        s.push_str("SYNTHETIC_QCQP\nQCQ\nminimize\n");
        s.push_str(&format!("{n}\n{m}\n"));
        s.push_str("0\n");
        s.push_str("0.0\n");
        s.push_str("0\n");
        s.push_str("0.0\n");
        s.push_str("1\n");
        s.push_str("1 1 1 1.0\n");
        s.push_str("0\n");
        s.push_str("1.79769313486232E+308\n");
        s.push_str("0.0\n0\n");
        s.push_str("0.0\n0\n");
        s.push_str("0.0\n0\n");
        s.push_str("1.79769313486232E+308\n0\n");
        s.push_str("0.0\n0\n0.0\n0\n0.0\n0\n0\n0\n");
        s
    }

    /// QCQP COO-storage memory sentinel (synthetic, data-independent).
    ///
    /// Old `vec![CscMatrix::new(n, n); m_aug]`: O(m·n) col_ptr allocation.
    /// With N=50_000, M=200: 200 × 50_001 × 8 ≈ 80 MB → above `QCQP_DENSE_INIT_LIMIT`.
    ///
    /// **No-op failure guarantee**: reverting to `CscMatrix::new(n, n)` default raises
    /// peak above 20 MB → assertion fires.
    #[test]
    fn test_qcqp_sparse_init_memory_bounded() {
        const SYNTHETIC_N: usize = 50_000;
        const SYNTHETIC_M: usize = 200;
        const QCQP_DENSE_INIT_LIMIT: usize = 20 * 1024 * 1024;

        let content = make_synthetic_qcq_content(SYNTHETIC_N, SYNTHETIC_M);

        crate::peak_alloc::begin();
        let result = parse_qplib_str(&content).expect("synthetic QCQP must parse");
        let peak = crate::peak_alloc::peak_bytes();

        let prob = unwrap_qp(result);
        assert_eq!(prob.num_vars, SYNTHETIC_N);
        assert_eq!(prob.num_constraints, SYNTHETIC_M);
        assert_eq!(prob.quadratic_constraints.len(), SYNTHETIC_M);
        assert_eq!(
            prob.quadratic_constraints[0].nnz(),
            1,
            "constraint 0 must have nnz=1 (single diagonal entry)"
        );
        for i in 1..SYNTHETIC_M {
            assert_eq!(
                prob.quadratic_constraints[i].nnz(),
                0,
                "Q_k[{i}] must be empty for synthetic problem"
            );
        }

        assert!(
            peak <= QCQP_DENSE_INIT_LIMIT,
            "synthetic QCQP parse peak {:.1} MB exceeds {:.1} MB limit.\n\
             QcqpMatrix (COO) path expected < 1 MB; \
             CscMatrix::new(n,n) default path ≈ 80 MB.\n\
             Revert check: ensure quadratic_constraints uses QcqpMatrix not CscMatrix.",
            peak as f64 / 1_048_576.0,
            QCQP_DENSE_INIT_LIMIT as f64 / 1_048_576.0
        );
    }

    // -----------------------------------------------------------------------
    // Multi-pattern tests: large / small / sparse / dense
    // -----------------------------------------------------------------------

    /// Small LP (LCL): linear obj, 10 vars, 5 constraints, 20 A-matrix NZ.
    #[test]
    fn test_parse_small_lp_lcl() {
        let mut content = String::from("SMALL_LP\nLCL\nminimize\n10\n5\n0\n0.0\n0\n0.0\n20\n");
        for k in 1..=5usize {
            for i in (k..k + 4).filter(|&i| i <= 10) {
                content.push_str(&format!("{k} {i} 1.0\n"));
            }
        }
        content.push_str("1.0e308\n-1.0e308\n0\n100.0\n0\n0.0\n0\n1.0e308\n0\n");
        content.push_str("0.0\n0\n0.0\n0\n0.0\n0\n0\n0\n");
        let result = parse_qplib_str(&content);
        assert!(result.is_ok(), "small LP parse failed: {:?}", result.err());
        let prob = unwrap_qp(result.unwrap());
        assert_eq!(prob.num_vars, 10);
        assert_eq!(prob.num_constraints, 5);
        assert!(prob.a.nnz() > 0);
    }

    /// Dense Q matrix (QCB type): n=80, nqobj = 80*81/2 = 3240 lower-tri entries.
    /// Verifies Q is symmetric with correct nnz after symmetrization.
    #[test]
    fn test_parse_dense_q_box_constraints() {
        const N: usize = 80;
        const NQOBJ: usize = N * (N + 1) / 2;

        let mut content = String::from("DENSE_Q\nQCB\nminimize\n80\n");
        content.push_str(&format!("{NQOBJ}\n"));
        for i in 1..=N {
            for j in 1..=i {
                content.push_str(&format!("{i} {j} 1.0\n"));
            }
        }
        content.push_str("0.0\n0\n0.0\n");
        content.push_str("1.0e308\n");
        content.push_str("0.0\n0\n1.0\n0\n");
        content.push_str("0.0\n0\n0.0\n0\n0\n");

        let result = parse_qplib_str(&content);
        assert!(result.is_ok(), "dense Q parse failed: {:?}", result.err());
        let prob = unwrap_qp(result.unwrap());
        assert_eq!(prob.num_vars, N);
        let expected_nnz = N + (NQOBJ - N) * 2;
        assert_eq!(
            prob.q.nnz(),
            expected_nnz,
            "Q nnz: expected {expected_nnz} (full {N}×{N} symmetric)"
        );
    }

    /// Sparse A matrix: LCL type, n=500 vars, m=200 constraints, ~1000 NZ.
    /// Verifies sort-merge CSC construction correctness for sparse inputs.
    #[test]
    fn test_parse_sparse_a_matrix_correctness() {
        const N: usize = 500;
        const M: usize = 200;

        let mut content = format!("SPARSE_A\nLCL\nminimize\n{N}\n{M}\n0\n0.0\n0\n0.0\n");
        let mut nnz = 0usize;
        let mut entry_buf = String::new();
        for k in 1..=M {
            for offset in 0..5usize {
                let i = (k * 7 + offset * 13) % N + 1;
                entry_buf.push_str(&format!("{k} {i} 1.0\n"));
                nnz += 1;
            }
        }
        content.push_str(&format!("{nnz}\n"));
        content.push_str(&entry_buf);
        content.push_str("1.0e308\n-1.0e308\n0\n1000.0\n0\n0.0\n0\n1.0e308\n0\n");
        content.push_str("0.0\n0\n0.0\n0\n0.0\n0\n0\n0\n");

        let result = parse_qplib_str(&content);
        assert!(result.is_ok(), "sparse A parse failed: {:?}", result.err());
        let prob = unwrap_qp(result.unwrap());
        assert_eq!(prob.num_vars, N);
        assert!(prob.num_constraints >= M);
        assert!(prob.a.nnz() > 0, "A matrix should have non-zeros");
    }

    /// nqobj sanity bound: n=2 gives max nqobj=3; declaring nqobj=4 must return ParseError.
    #[test]
    fn test_sanity_bound_nqobj_too_large() {
        let content = "\
SANITY_NQOBJ
QCL
minimize
2
1
4
";
        let result = parse_qplib_str(content);
        assert!(
            result.is_err(),
            "expected ParseError for nqobj=4 > n*(n+1)/2=3"
        );
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(
            err_str.contains("nqobj") || err_str.contains("exceeds"),
            "error should mention nqobj bound, got: {}",
            err_str
        );
    }

    /// n_con_lin_terms sanity bound: n=2, m=1 gives max=2; declaring 3 must return ParseError.
    #[test]
    fn test_sanity_bound_n_con_lin_terms_too_large() {
        let content = "\
SANITY_NCON
LCL
minimize
2
1
0
0.0
0
0.0
3
";
        let result = parse_qplib_str(content);
        assert!(
            result.is_err(),
            "expected ParseError for n_con_lin_terms=3 > n*m=2"
        );
        let err_str = format!("{:?}", result.unwrap_err());
        assert!(
            err_str.contains("n_con_lin_terms") || err_str.contains("exceeds"),
            "error should mention n_con_lin_terms bound, got: {}",
            err_str
        );
    }

    /// Sentinel: objective constant q0 ≠ 0 must propagate to `prob.obj_offset`.
    ///
    /// **No-op failure guarantee**: reverting to `let _q0 = ts.read_f64()?;` (discarding q0)
    /// leaves `prob.obj_offset = 0.0` instead of 42.5 → assertion fires.
    #[test]
    fn test_qplib_objective_constant_propagates_to_obj_offset() {
        let qplib = "\
Q0_OFFSET
LCL
minimize
1
1
0
0.0
0
42.5
1
1 1 1.0
1.0e308
-1.0e308
0
1.0
0
0.0
0
1.0e308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 1);
        assert!(
            (prob.obj_offset - 42.5).abs() < 1e-12,
            "q0=42.5 must propagate to obj_offset; got {}",
            prob.obj_offset
        );
    }

    /// Sentinel: non-finite q0 must return a `ParseError`.
    ///
    /// **No-op failure guarantee**: if the NaN guard is removed, `parse_qplib_str`
    /// would succeed with `obj_offset = NaN` → `is_err()` fails → assertion fires.
    #[test]
    fn test_qplib_objective_constant_nan_is_error() {
        let qplib = "\
Q0_NAN
LCL
minimize
1
1
0
0.0
0
NaN
1
1 1 1.0
1.0e308
-1.0e308
0
1.0
0
0.0
0
1.0e308
0
";
        assert!(
            matches!(parse_qplib_str(qplib), Err(QplibError::ParseError(_))),
            "non-finite q0 must produce ParseError"
        );
    }

    /// Sentinel: `maximize` sense must negate q0, c, and Q before storing.
    ///
    /// **No-op failure guarantees**:
    /// - Removing `let q0_offset = if maximize { -q0 } else { q0 }` sign flip
    ///   leaves `obj_offset = +42.5` → assertion fires.
    /// - Removing `*v = -*v` in the `if maximize` block for `c` leaves
    ///   `c[0] = +3.0` instead of `-3.0` → assertion fires.
    /// - Changing `sign * v` to `v` in Q construction removes the sign flip
    ///   → `q.values()[0] = +1.0` instead of `-1.0` → assertion fires.
    #[test]
    fn test_qplib_maximize_negates_obj_offset() {
        // nqobj=1 (Q[1,1]=1.0), default linear c=3.0, q0=42.5.
        // After maximize parsing: obj_offset=-42.5, c[0]=-3.0, Q[0,0]=-1.0.
        let qplib = "\
Q0_MAX
LCL
maximize
1
1
1
1 1 1.0
3.0
0
42.5
1
1 1 1.0
1.0e308
-1.0e308
0
1.0
0
0.0
0
1.0e308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 1);
        assert!(
            (prob.obj_offset - (-42.5)).abs() < 1e-12,
            "maximize with q0=42.5 must store obj_offset=-42.5; got {}",
            prob.obj_offset
        );
        assert!(
            (prob.c[0] - (-3.0)).abs() < 1e-12,
            "maximize must negate c: source c=3.0 → stored c[0]=-3.0; got {}",
            prob.c[0]
        );
        assert_eq!(prob.q.nnz(), 1, "Q must have exactly 1 nonzero (diagonal)");
        assert!(
            (prob.q.values()[0] - (-1.0)).abs() < 1e-12,
            "maximize must negate Q: source Q[0,0]=1.0 → stored -1.0; got {}",
            prob.q.values()[0]
        );
    }

    /// Duplicate linear constraint entries must be accumulated (not double-counted).
    ///
    /// Sentinel: if sort-merge deduplication is broken, the final A matrix will
    /// have twice as many entries → nnz assertion fails.
    #[test]
    fn test_parse_duplicate_a_entries_accumulated() {
        let qplib = "\
DUP_TEST
LCL
minimize
2
1
0
0.0
0
0.0
2
1 1 2.0
1 1 2.0
1.0e308
-1.0e308
0
0.0
1
1 5.0
0.0
0
1.0e308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        assert_eq!(
            prob.a.nnz(),
            1,
            "duplicate entries must be merged to one NZ"
        );
        let col0 = prob.a.col_ptr()[0];
        assert!(
            (prob.a.values()[col0] - 4.0).abs() < 1e-12,
            "accumulated coeff should be 4.0, got {}",
            prob.a.values()[col0]
        );
    }

    // -----------------------------------------------------------------------
    // PR #25 review: OBJSENSE / objective-type / inf_val validation.
    // -----------------------------------------------------------------------

    /// Sentinel: an unrecognized OBJSENSE token must be a `ParseError`, not
    /// silently treated as `minimize`.
    ///
    /// **No-op failure guarantee**: reverting to
    /// `matches!(objsense.as_str(), "maximize" | "max")` (unknown => minimize)
    /// makes this parse succeed as a minimize problem instead of erroring.
    #[test]
    fn test_qplib_unknown_objsense_is_error() {
        let qplib = "\
BAD_OBJSENSE
LCL
sideways
1
1
0
0.0
0
0.0
1
1 1 1.0
1.0e308
-1.0e308
0
1.0
0
0.0
0
1.0e308
0
";
        let err = parse_qplib_str(qplib).expect_err("unknown OBJSENSE token must error");
        assert!(
            matches!(err, QplibError::ParseError(_)),
            "expected ParseError, got {:?}",
            err
        );
    }

    /// `minimize`/`min` and `maximize`/`max` are all accepted (case-insensitive
    /// via the existing `.to_lowercase()`), matching the pre-existing "max"
    /// abbreviation support.
    #[test]
    fn test_qplib_objsense_min_abbreviation_accepted() {
        let qplib = "\
MIN_ABBREV
LCL
min
1
1
0
0.0
0
0.0
1
1 1 1.0
1.0e308
-1.0e308
0
1.0
0
0.0
0
1.0e308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 1);
    }

    /// Sentinel: an unrecognized objective-type character (position 0 of the
    /// 3-char problem type) must be `UnsupportedType`, matching the existing
    /// var_char/con_char validation style.
    ///
    /// **No-op failure guarantee**: reverting to `let _obj_char = ...;`
    /// (discarding the char without validation) makes this parse succeed.
    #[test]
    fn test_qplib_unknown_objective_type_char_is_unsupported() {
        let qplib = "\
BAD_OBJ_TYPE
XCL
";
        assert!(matches!(
            parse_qplib_str(qplib),
            Err(QplibError::UnsupportedType(_))
        ));
    }

    /// All four QPLIB objective-type characters (L/D/C/Q per Furini et al.
    /// 2019 §3.3 PROBTYPE) must be accepted. L/C/Q are already exercised by
    /// other fixtures in this module (LCL/QCL/CBL); D (diagonal convex
    /// quadratic) is not, so it is covered explicitly here.
    #[test]
    fn test_qplib_diagonal_objective_type_d_accepted() {
        let qplib = "\
DIAG_OBJ
DCL
minimize
2
1
2
1 1 1.0
2 2 1.0
0.0
0
0.0
2
1 1 1.0
1 2 1.0
1.79769313486232E+308
1.0
0
1.0
0
0.0
0
1.79769313486232E+308
0
0.0
0
0.0
0
0.0
0
0
0
";
        let prob = unwrap_qp(parse_qplib_str(qplib).unwrap());
        assert_eq!(prob.num_vars, 2);
        assert_eq!(prob.q.nnz(), 2);
    }

    /// Sentinel: `inf_val <= 0` must be a `ParseError`. Without the guard,
    /// `is_pos_inf`/`is_neg_inf` divide-by-zero-scale semantics classify any
    /// finite bound (here `[-100, 100]`) as `(-inf, inf)`, silently dropping
    /// the constraint's bounds.
    ///
    /// **No-op failure guarantee**: removing the `inf_val <= 0.0` guard makes
    /// this parse succeed with `prob.num_constraints == 0` instead of erroring.
    #[test]
    fn test_qplib_non_positive_inf_val_is_error() {
        let qplib = "\
BAD_INF
LCL
minimize
1
1
0
0.0
0
0.0
1
1 1 1.0
0.0
-100.0
0
100.0
0
0.0
0
1.0e308
0
";
        let err = parse_qplib_str(qplib).expect_err("inf_val<=0 must produce ParseError");
        assert!(
            matches!(err, QplibError::ParseError(_)),
            "expected ParseError, got {:?}",
            err
        );
    }

    /// Sentinel: a negative `inf_val` must also be rejected (not just zero).
    #[test]
    fn test_qplib_negative_inf_val_is_error() {
        let qplib = "\
BAD_INF_NEG
LCL
minimize
1
1
0
0.0
0
0.0
1
1 1 1.0
-1.0e308
-100.0
0
100.0
0
0.0
0
1.0e308
0
";
        let err = parse_qplib_str(qplib).expect_err("negative inf_val must produce ParseError");
        assert!(
            matches!(err, QplibError::ParseError(_)),
            "expected ParseError, got {:?}",
            err
        );
    }
}
