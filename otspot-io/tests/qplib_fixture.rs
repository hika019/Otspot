//! Tracked inline fixtures for the QPLIB parser.
//!
//! These integration tests run without any external data files and cover
//! representative QPLIB patterns: QCL / QCN / QCB / QCQ (with equality and
//! range constraints), minimize/maximize, integer and binary variables.

use otspot_core::problem::ConstraintType;
use otspot_core::qp::QpProblem;
use otspot_io::qplib::{parse_qplib_str, QplibProblem};

fn unwrap_qp(r: QplibProblem) -> QpProblem {
    match r {
        QplibProblem::Qp(p) => p,
        other => panic!("expected Qp, got {:?}", other),
    }
}

// ── QCL: linear constraints, quadratic objective ──────────────────────────────

/// Table-driven: 3 QCL problems with different constraint counts and Q densities.
#[test]
fn test_qplib_qcl_patterns() {
    struct Case {
        label: &'static str,
        input: &'static str,
        num_vars: usize,
        num_constraints: usize,
        q_nnz: usize,
    }

    let cases = [
        // 2 vars, 1 Eq constraint, diagonal Q
        Case {
            label: "qcl_2v_1eq",
            input: "\
QCL_2V_1EQ
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
",
            num_vars: 2,
            num_constraints: 1,
            q_nnz: 2,
        },
        // 3 vars, 2 Le constraints (lb=-inf), diagonal Q
        Case {
            label: "qcl_3v_2le",
            input: "\
QCL_3V_2LE
QCL
minimize
3
2
3
1 1 2.0
2 2 2.0
3 3 2.0
0.0
3
1 1.0
2 2.0
3 3.0
0.0
6
1 1 1.0
1 2 1.0
1 3 1.0
2 1 1.0
2 2 1.0
2 3 1.0
1.79769313486232E+308
-1.79769313486232E+308
0
6.0
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
",
            num_vars: 3,
            num_constraints: 2,
            q_nnz: 3, // diagonal: 3 entries
        },
        // 2 vars, unconstrained (QCN) with full 2×2 Q
        //
        // PR#25 review fix: same missing-`n_con_lin_terms`-field bug as
        // `test_qplib_maximize_sign_flip` (see comment there); fixed to the
        // same known-good QCN field layout.
        Case {
            label: "qcn_2v_unc",
            input: "\
QCN_2V
QCN
minimize
2
0
3
1 1 1.0
2 1 0.5
2 2 1.0
0.0
0
0.0
0
1.79769313486232E+308
0.0
0
0.0
0
-1.79769313486232E+308
0
1.79769313486232E+308
0
",
            num_vars: 2,
            num_constraints: 0,
            q_nnz: 4,
        },
    ];

    for case in &cases {
        let prob = unwrap_qp(
            parse_qplib_str(case.input)
                .unwrap_or_else(|e| panic!("[{}] parse failed: {}", case.label, e)),
        );
        assert_eq!(prob.num_vars, case.num_vars, "[{}] num_vars", case.label);
        assert_eq!(
            prob.num_constraints, case.num_constraints,
            "[{}] num_constraints",
            case.label
        );
        assert_eq!(prob.q.nnz(), case.q_nnz, "[{}] q_nnz", case.label);
    }
}

// ── QCB: box-constrained (no explicit constraint section) ─────────────────────

#[test]
fn test_qplib_qcb_box_constrained() {
    // QCB: no m field, no constraint linear terms, no lb_con/ub_con fields.
    let input = "\
QCB_TEST
QCB
minimize
2
2
1 1 2.0
2 2 2.0
1.0
2
1 2.0
2 3.0
0.0
1.79769313486232E+308
0.0
0
1.0
0
";
    let prob = unwrap_qp(parse_qplib_str(input).expect("QCB parse"));
    assert_eq!(prob.num_vars, 2);
    assert_eq!(prob.num_constraints, 0, "QCB has no linear constraints");
    assert_eq!(prob.q.nnz(), 2);
    // Default var lb=0, ub=1
    assert!((prob.bounds[0].0 - 0.0).abs() < 1e-12);
    assert!((prob.bounds[0].1 - 1.0).abs() < 1e-12);
    // c: default=1.0, non-defaults: x1→2.0, x2→3.0
    assert!((prob.c[0] - 2.0).abs() < 1e-12);
    assert!((prob.c[1] - 3.0).abs() < 1e-12);
}

// ── Equality vs. range constraints ───────────────────────────────────────────

/// Table-driven: lb=ub (Eq), lb only (no ub row), ub only, range (two Le rows).
#[test]
fn test_qplib_constraint_type_expansion() {
    const INF: &str = "1.79769313486232E+308";
    const NEG_INF: &str = "-1.79769313486232E+308";

    struct Case {
        label: &'static str,
        lb_default: &'static str,
        ub_default: &'static str,
        expected_num_constraints: usize,
        expected_types: &'static [ConstraintType],
    }

    let cases = [
        // lb=ub=5 → Eq
        Case {
            label: "equality",
            lb_default: "5.0",
            ub_default: "5.0",
            expected_num_constraints: 1,
            expected_types: &[ConstraintType::Eq],
        },
        // lb=-inf, ub=10 → single Le
        Case {
            label: "le_only",
            lb_default: NEG_INF,
            ub_default: "10.0",
            expected_num_constraints: 1,
            expected_types: &[ConstraintType::Le],
        },
        // lb=2, ub=8 → two Le rows (ub_row + lb_row)
        Case {
            label: "range",
            lb_default: "2.0",
            ub_default: "8.0",
            expected_num_constraints: 2,
            expected_types: &[ConstraintType::Le, ConstraintType::Le],
        },
    ];

    for case in &cases {
        let input = format!(
            "T\nQCL\nminimize\n2\n1\n0\n0.0\n0\n0.0\n2\n1 1 1.0\n1 2 1.0\n{INF}\n{lb}\n0\n{ub}\n0\n0.0\n0\n{INF}\n0\n0.0\n0\n0.0\n0\n0.0\n0\n0\n0\n",
            INF = INF,
            lb = case.lb_default,
            ub = case.ub_default,
        );
        let prob = unwrap_qp(
            parse_qplib_str(&input)
                .unwrap_or_else(|e| panic!("[{}] parse failed: {}", case.label, e)),
        );
        assert_eq!(
            prob.num_constraints, case.expected_num_constraints,
            "[{}] num_constraints",
            case.label
        );
        assert_eq!(
            prob.constraint_types, case.expected_types,
            "[{}] constraint_types",
            case.label
        );
    }
}

// ── maximize: sign-flip of c and Q ───────────────────────────────────────────

#[test]
fn test_qplib_maximize_sign_flip() {
    // minimize 1/2*x^2 + x  vs  maximize 1/2*x^2 + x (→ negate c and Q)
    //
    // PR#25 review fix: this fixture previously omitted the `n_con_lin_terms`
    // field (required for con_char 'N', same as 'L'/'Q'; see
    // `test_parse_qplib_unconstrained` in `src/qplib/mod.rs`), shifting every
    // subsequent field by one. The parser's old (unvalidated) `inf_val` read
    // silently absorbed the resulting misalignment — the shifted `inf_val`
    // ended up `0.0` — and produced an accidentally-degenerate but non-erroring
    // parse: with `inf_val` shifted to `0.0`, the `[0, 0]` raw default bounds
    // both overflow the `>= 0` infinity threshold and collapse back to
    // `(-inf, inf)`. The new `inf_val > 0`
    // validation (this PR) correctly rejects that. Fixed to the same
    // known-good field layout as `test_parse_qplib_unconstrained`.
    let min_input = "\
MAX_TEST
QCN
minimize
1
0
1
1 1 2.0
1.0
0
0.0
0
1.79769313486232E+308
0.0
0
0.0
0
-1.79769313486232E+308
0
1.79769313486232E+308
0
";
    let max_input = "\
MAX_TEST
QCN
maximize
1
0
1
1 1 2.0
1.0
0
0.0
0
1.79769313486232E+308
0.0
0
0.0
0
-1.79769313486232E+308
0
1.79769313486232E+308
0
";
    let p_min = unwrap_qp(parse_qplib_str(min_input).unwrap());
    let p_max = unwrap_qp(parse_qplib_str(max_input).unwrap());

    // maximize negates c and Q
    assert!((p_min.c[0] - (-p_max.c[0])).abs() < 1e-12, "c sign flip");
    // Q values should be negated as well
    let q_min_val = p_min.q.values()[0];
    let q_max_val = p_max.q.values()[0];
    assert!((q_min_val + q_max_val).abs() < 1e-12, "Q sign flip");
}

// ── Integer / binary variables ────────────────────────────────────────────────

#[test]
fn test_qplib_integer_and_binary() {
    // QIL: integer vars, linear obj → Milp
    let qil = "\
INT
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
    let milp = match parse_qplib_str(qil).unwrap() {
        QplibProblem::Milp(m) => m,
        other => panic!("expected Milp, got {:?}", other),
    };
    assert_eq!(milp.lp.num_vars, 2);
    assert_eq!(milp.integer_vars, vec![0, 1]);

    // CBL: binary vars, zero Q → Milp with bounds [0,1]
    let cbl = "\
BIN
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
    let milp2 = match parse_qplib_str(cbl).unwrap() {
        QplibProblem::Milp(m) => m,
        other => panic!("expected Milp, got {:?}", other),
    };
    for &(lb, ub) in &milp2.lp.bounds {
        assert!((lb - 0.0).abs() < 1e-12, "binary lb");
        assert!((ub - 1.0).abs() < 1e-12, "binary ub");
    }
}
