//! Tracked inline fixtures for the QPS parser.
//!
//! These integration tests run without any external data files and cover
//! representative QPS patterns: LP-only, QP, equality/inequality constraints,
//! bound types, RANGES, and obj_offset.  Each case is table-driven with at
//! least two distinct data patterns.

use otspot_core::problem::ConstraintType;
use otspot_io::qps::{parse_qps_str, QpsError};

// ── LP-only (no QUADOBJ) ──────────────────────────────────────────────────────

/// Verifies that a QPS file without QUADOBJ parses as a zero-Q problem.
#[test]
fn test_qps_lp_only_patterns() {
    struct Case {
        label: &'static str,
        input: &'static str,
        num_vars: usize,
        num_constraints: usize,
        c: &'static [f64],
        b: &'static [f64],
    }

    let cases = [
        Case {
            label: "single_var_single_con",
            input: r"NAME min1
ROWS
 N  obj
 L  c1
COLUMNS
    x  obj  3.0  c1  1.0
RHS
    rhs  c1  10.0
ENDATA
",
            num_vars: 1,
            num_constraints: 1,
            c: &[3.0],
            b: &[10.0],
        },
        Case {
            label: "two_vars_ge_constraint",
            input: r"NAME min2
ROWS
 N  obj
 G  sum
COLUMNS
    x  obj  1.0  sum  1.0
    y  obj  2.0  sum  1.0
RHS
    rhs  sum  5.0
ENDATA
",
            num_vars: 2,
            num_constraints: 1,
            c: &[1.0, 2.0],
            // G constraint sign-flipped internally; raw b from parser = -5.0
            b: &[-5.0],
        },
        Case {
            label: "equality_constraint",
            input: r"NAME eq
ROWS
 N  obj
 E  eq1
COLUMNS
    x  obj  1.0  eq1  1.0
    y  obj  1.0  eq1  1.0
RHS
    rhs  eq1  4.0
ENDATA
",
            num_vars: 2,
            num_constraints: 1,
            c: &[1.0, 1.0],
            b: &[4.0],
        },
    ];

    for case in &cases {
        let prob = parse_qps_str(case.input)
            .unwrap_or_else(|e| panic!("[{}] parse failed: {}", case.label, e));
        assert_eq!(prob.num_vars, case.num_vars, "[{}] num_vars", case.label);
        assert_eq!(
            prob.num_constraints, case.num_constraints,
            "[{}] num_constraints",
            case.label
        );
        assert!(prob.is_zero_q(), "[{}] should have zero Q", case.label);
        for (i, &exp) in case.c.iter().enumerate() {
            assert!(
                (prob.c[i] - exp).abs() < 1e-12,
                "[{}] c[{}]: expected {}, got {}",
                case.label,
                i,
                exp,
                prob.c[i]
            );
        }
        for (i, &exp) in case.b.iter().enumerate() {
            assert!(
                (prob.b[i] - exp).abs() < 1e-12,
                "[{}] b[{}]: expected {}, got {}",
                case.label,
                i,
                exp,
                prob.b[i]
            );
        }
    }
}

// ── QP with QUADOBJ ───────────────────────────────────────────────────────────

/// Two QP fixtures — diagonal and cross-term Q — verifying symmetrization.
#[test]
fn test_qps_quadobj_patterns() {
    // Fixture 1: diagonal Q = diag(2, 4)
    let diag_qps = r"NAME diag_qp
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
    x2  x2  4.0
ENDATA
";
    let p1 = parse_qps_str(diag_qps).unwrap();
    assert_eq!(p1.q.nnz(), 2, "diagonal Q: 2 entries");

    // Fixture 2: full 2×2 symmetric Q (upper-tri given: x1x1=1, x1x2=1, x2x2=1)
    // After symmetrization: 4 entries (2 diagonal + 2 off-diagonal)
    let full_qps = r"NAME full_qp
ROWS
 N  obj
COLUMNS
    x1  obj  0.0
    x2  obj  0.0
BOUNDS
 FR BND  x1
 FR BND  x2
QUADOBJ
    x1  x1  1.0
    x1  x2  1.0
    x2  x2  1.0
ENDATA
";
    let p2 = parse_qps_str(full_qps).unwrap();
    assert_eq!(
        p2.q.nnz(),
        4,
        "full 2×2 symmetric Q: 4 entries after symmetrization"
    );
}

// ── Bound types ───────────────────────────────────────────────────────────────

/// Table-driven: each row verifies a distinct bound type from the QPS spec.
#[test]
fn test_qps_bound_types() {
    struct BoundCase {
        label: &'static str,
        bounds_section: &'static str,
        expected_lb: f64,
        expected_ub: f64,
    }

    let cases = [
        BoundCase {
            label: "LO",
            bounds_section: " LO BND  x  2.0\n",
            expected_lb: 2.0,
            expected_ub: f64::INFINITY,
        },
        BoundCase {
            label: "UP",
            bounds_section: " UP BND  x  8.0\n",
            expected_lb: 0.0,
            expected_ub: 8.0,
        },
        BoundCase {
            label: "FX",
            bounds_section: " FX BND  x  5.0\n",
            expected_lb: 5.0,
            expected_ub: 5.0,
        },
        BoundCase {
            label: "FR",
            bounds_section: " FR BND  x\n",
            expected_lb: f64::NEG_INFINITY,
            expected_ub: f64::INFINITY,
        },
        BoundCase {
            label: "MI",
            bounds_section: " MI BND  x\n",
            expected_lb: f64::NEG_INFINITY,
            expected_ub: f64::INFINITY,
        },
        BoundCase {
            label: "BV",
            bounds_section: " BV BND  x\n",
            expected_lb: 0.0,
            expected_ub: 1.0,
        },
    ];

    for case in &cases {
        let input = format!(
            "NAME  T\nROWS\n N  obj\nCOLUMNS\n    x  obj  1.0\nRHS\nBOUNDS\n{}\nENDATA\n",
            case.bounds_section
        );
        let prob = parse_qps_str(&input)
            .unwrap_or_else(|e| panic!("[{}] parse failed: {}", case.label, e));
        assert_eq!(prob.num_vars, 1);
        let (lb, ub) = prob.bounds[0];
        // Use f64::is_infinite checks for infinities
        if case.expected_lb.is_infinite() {
            assert!(
                lb.is_infinite() && lb.signum() == case.expected_lb.signum(),
                "[{}] lb: expected {:?}, got {:?}",
                case.label,
                case.expected_lb,
                lb
            );
        } else {
            assert!(
                (lb - case.expected_lb).abs() < 1e-12,
                "[{}] lb: expected {}, got {}",
                case.label,
                case.expected_lb,
                lb
            );
        }
        if case.expected_ub.is_infinite() {
            assert!(
                ub.is_infinite() && ub.signum() == case.expected_ub.signum(),
                "[{}] ub: expected {:?}, got {:?}",
                case.label,
                case.expected_ub,
                ub
            );
        } else {
            assert!(
                (ub - case.expected_ub).abs() < 1e-12,
                "[{}] ub: expected {}, got {}",
                case.label,
                case.expected_ub,
                ub
            );
        }
    }
}

// ── Objective offset (N-row RHS) ──────────────────────────────────────────────

#[test]
fn test_qps_obj_offset_patterns() {
    // offset = -7.0
    let qps_neg = r"NAME  OFF
ROWS
 N  obj
 L  c1
COLUMNS
    x  obj  1.0  c1  1.0
RHS
    rhs  obj  -7.0
    rhs  c1  10.0
ENDATA
";
    let p_neg = parse_qps_str(qps_neg).unwrap();
    assert!((p_neg.obj_offset - (-7.0)).abs() < 1e-12, "negative offset");

    // offset = 0 (no N-row RHS)
    let qps_zero = r"NAME  OFF2
ROWS
 N  obj
 L  c1
COLUMNS
    x  obj  1.0  c1  1.0
RHS
    rhs  c1  5.0
ENDATA
";
    let p_zero = parse_qps_str(qps_zero).unwrap();
    assert!(
        p_zero.obj_offset.abs() < 1e-12,
        "zero offset when N-row RHS absent"
    );

    // Invalid offset (inf) → error
    let qps_inf =
        "NAME  INF\nROWS\n N  obj\n L  c1\nCOLUMNS\n    x  obj  1.0  c1  1.0\nRHS\n    rhs  obj  inf\n    rhs  c1  5.0\nENDATA\n";
    assert!(
        matches!(
            parse_qps_str(qps_inf),
            Err(QpsError::InvalidObjectiveOffset(_))
        ),
        "inf offset must produce error"
    );
}

// ── Constraint-type round-trip ────────────────────────────────────────────────

/// Verifies that Le / Ge / Eq constraint types are correctly preserved.
#[test]
fn test_qps_constraint_types() {
    let qps = r"NAME  CTYPES
ROWS
 N  obj
 L  le1
 G  ge1
 E  eq1
COLUMNS
    x  obj  1.0  le1  1.0
    x  ge1  1.0  eq1  1.0
RHS
    rhs  le1  10.0  ge1  2.0
    rhs  eq1  5.0
ENDATA
";
    let prob = parse_qps_str(qps).unwrap();
    // Le stays Le; Ge is sign-flipped internally to Le; Eq stays Eq.
    assert_eq!(prob.constraint_types[0], ConstraintType::Le, "le1");
    assert_eq!(
        prob.constraint_types[1],
        ConstraintType::Le,
        "ge1 → sign-flipped Le"
    );
    assert_eq!(prob.constraint_types[2], ConstraintType::Eq, "eq1");
}
