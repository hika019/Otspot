//! QPLIB format parser.
//!
//! QPLIB (<https://qplib.zib.de/>) is a token-stream format distinct from
//! MPS/QPS.  The parser uses a streaming tokenizer so that 200 MB+ files
//! are handled without OOM.

mod token_stream;
mod parser;

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
/// Continuous-variable problems return [`Qp`]; problems with binary (`B`) or
/// integer (`I`) variables return [`Milp`] (zero-Q) or [`Miqp`] (non-zero Q).
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
        assert_eq!(prob.constraint_types[0], otspot_core::problem::ConstraintType::Eq);
        assert_eq!(prob.q.nnz(), 2);
    }

    #[test]
    fn test_parse_qplib_unconstrained() {
        let qplib = "\
NO_CON
QCN
minimize
2 # vars
0 # constraints
2 # qobj
1 1 2.0
2 2 2.0
0.0 # default b0
0 # non-default b0
0.0 # obj constant
0 # no linear constraints
1.79769313486232E+308 # infinity
0.0 # default lhs
0
0.0 # default rhs
0
-1.79769313486232E+308 # default var lb
0
1.79769313486232E+308 # default var ub
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
        assert_eq!(prob.num_constraints, 0);
        assert_eq!(prob.q.nnz(), 2);
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
        assert_eq!(prob.constraint_types[0], otspot_core::problem::ConstraintType::Eq);
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
        assert_eq!(prob.constraint_types[0], otspot_core::problem::ConstraintType::Le);
        assert_eq!(prob.constraint_types[1], otspot_core::problem::ConstraintType::Eq);
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
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
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
            assert_eq!(prob.constraint_types[i], otspot_core::problem::ConstraintType::Eq);
        }
        assert_eq!(prob.constraint_types[8], otspot_core::problem::ConstraintType::Le);
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
        let v00 = qk.triplets.iter().find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v).expect("Q_9 must have (0,0) entry");
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
            assert_eq!(prob.constraint_types[i], otspot_core::problem::ConstraintType::Eq);
        }
        assert_eq!(prob.constraint_types[5], otspot_core::problem::ConstraintType::Le);
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
        let v00 = qk.triplets.iter().find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v).expect("Q_6 must have (0,0) entry");
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
            assert_eq!(prob.constraint_types[i], otspot_core::problem::ConstraintType::Le);
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
        let v00 = qk0.triplets.iter().find(|&&(r, c, _)| r == 0 && c == 0)
            .map(|&(_, _, v)| v).expect("Q_1 must have (0,0) entry");
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
    }
}
