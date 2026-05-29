//! Sentinel tests for OBJSENSE support and QPS error-path hardening (#105).
//!
//! Covers:
//!   A.1 MPS/QPS OBJSENSE MAX — objective is negated to MIN form.
//!   B.1 Model::add_constraint() cross-model variable rejection.
//!   A.2 QPS DuplicateSection error.
//!   A.3 QPS unknown section reject.

use otspot::io::mps::{parse_mps, MpsError};
use otspot::io::qps::{parse_qps_str, QpsError};
use otspot::model::{Model, ModelError};
use otspot::constraint;

// ── A.1 MPS OBJSENSE MAX ─────────────────────────────────────────────────────

const MPS_MIN: &str = "\
NAME minimal_min
ROWS
 N COST
 L C1
COLUMNS
 X1 COST 1.0 C1 1.0
RHS
 RHS C1 5.0
BOUNDS
 UP BND X1 5.0
ENDATA
";

const MPS_MAX: &str = "\
NAME minimal_max
OBJSENSE
 MAX
ROWS
 N COST
 L C1
COLUMNS
 X1 COST 1.0 C1 1.0
RHS
 RHS C1 5.0
BOUNDS
 UP BND X1 5.0
ENDATA
";

/// MIN form: obj=1*x1, x1 ≤ 5 → optimal x1=0, obj=0.
#[test]
fn mps_objsense_min_parses_correctly() {
    let prob = parse_mps(MPS_MIN).expect("MIN parse should succeed");
    // MIN: c[0] = +1.0
    assert!(
        (prob.c[0] - 1.0).abs() < 1e-12,
        "MIN: c[0] should be +1.0, got {}",
        prob.c[0]
    );
}

/// MAX form: OBJSENSE MAX negates objective → c[0] becomes -1.0 (normalized to MIN).
/// Without the fix, c[0] stays +1.0 (silent wrong answer).
#[test]
fn mps_objsense_max_negates_objective() {
    let prob = parse_mps(MPS_MAX).expect("MAX parse should succeed");
    // MAX c^T x  ↔  MIN (-c)^T x: coefficient should be negated
    assert!(
        (prob.c[0] - (-1.0)).abs() < 1e-12,
        "MAX: c[0] should be -1.0 (negated for MIN form), got {}",
        prob.c[0]
    );
}

/// OBJSENSE MIN is accepted without error (symmetric to MAX).
#[test]
fn mps_objsense_min_explicit_accepted() {
    let mps = "\
NAME minimal_min_explicit
OBJSENSE
 MIN
ROWS
 N COST
 L C1
COLUMNS
 X1 COST 1.0 C1 1.0
RHS
 RHS C1 5.0
ENDATA
";
    let prob = parse_mps(mps).expect("explicit MIN parse should succeed");
    assert!(
        (prob.c[0] - 1.0).abs() < 1e-12,
        "explicit MIN: c[0] should be +1.0, got {}",
        prob.c[0]
    );
}

// ── A.1 QPS OBJSENSE MAX ─────────────────────────────────────────────────────

const QPS_MAX: &str = "\
NAME  qps_max
OBJSENSE
 MAX
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  5.0
BOUNDS
 UP BND  x1  5.0
ENDATA
";

/// QPS OBJSENSE MAX negates the linear objective coefficient.
#[test]
fn qps_objsense_max_negates_objective() {
    let prob = parse_qps_str(QPS_MAX).expect("QPS MAX parse should succeed");
    assert!(
        (prob.c[0] - (-1.0)).abs() < 1e-12,
        "QPS MAX: c[0] should be -1.0, got {}",
        prob.c[0]
    );
}

/// QPS OBJSENSE MIN leaves objective unchanged.
#[test]
fn qps_objsense_min_unchanged() {
    let qps = "\
NAME  qps_min
OBJSENSE
 MIN
ROWS
 N  obj
 L  c1
COLUMNS
    x1  obj  2.0  c1  1.0
RHS
    rhs  c1  5.0
ENDATA
";
    let prob = parse_qps_str(qps).expect("QPS MIN parse should succeed");
    assert!(
        (prob.c[0] - 2.0).abs() < 1e-12,
        "QPS MIN: c[0] should be 2.0, got {}",
        prob.c[0]
    );
}

// ── B.1 add_constraint cross-model variable rejection ────────────────────────

/// Adding a constraint that references a variable from a different model
/// must be rejected (recorded as InvalidInput) and cause solve() to fail.
#[test]
fn add_constraint_rejects_cross_model_variable() {
    let mut m1 = Model::new("m1");
    let x1 = m1.add_var("x1", 0.0, 10.0);

    let mut m2 = Model::new("m2");
    let x2 = m2.add_var("x2", 0.0, 10.0);
    m2.minimize(x2);

    // x1 belongs to m1; adding it to m2's constraint must be rejected.
    m2.add_constraint(constraint!(x1 <= 5.0));

    let result = m2.solve();
    assert!(
        matches!(result, Err(ModelError::InvalidInput(_))),
        "cross-model constraint must cause InvalidInput, got {:?}",
        result
    );
}

/// Constraints using variables from the same model are accepted.
#[test]
fn add_constraint_accepts_same_model_variable() {
    let mut m = Model::new("m");
    let x = m.add_var("x", 0.0, 10.0);
    m.minimize(x);
    m.add_constraint(constraint!(x <= 5.0));
    let result = m.solve();
    assert!(result.is_ok(), "same-model constraint must succeed, got {:?}", result);
    let r = result.unwrap();
    assert!(r.objective_value.abs() < 1e-6, "expected obj≈0, got {}", r.objective_value);
}

// ── A.2 QPS DuplicateSection error ───────────────────────────────────────────

/// Duplicate ROWS section must return QpsError::DuplicateSection.
#[test]
fn qps_duplicate_section_is_error() {
    let qps = "\
NAME  dup_test
ROWS
 N  obj
 L  c1
ROWS
 L  c2
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  5.0
ENDATA
";
    let result = parse_qps_str(qps);
    assert!(
        matches!(result, Err(QpsError::DuplicateSection(_))),
        "duplicate ROWS must yield DuplicateSection, got {:?}",
        result
    );
}

// ── A.3 QPS unknown section reject ───────────────────────────────────────────

/// An unrecognized section header must return a ParseError (not silently skip).
#[test]
fn qps_unknown_section_is_error() {
    let qps = "\
NAME  unknown_test
ROWS
 N  obj
 L  c1
UNKNOWN_SECTION
    x1  data  1.0
COLUMNS
    x1  obj  1.0  c1  1.0
RHS
    rhs  c1  5.0
ENDATA
";
    let result = parse_qps_str(qps);
    assert!(
        matches!(result, Err(QpsError::ParseError { .. })),
        "unknown section must yield ParseError, got {:?}",
        result
    );
}
