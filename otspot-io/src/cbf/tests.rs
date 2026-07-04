use super::*;
use otspot_core::conic::{solve_misocp, solve_socp, BbOptions, ConicOptions};
use otspot_core::problem::SolveStatus;

const TOL: f64 = 1e-6;

fn unwrap_socp(p: CbfProblem) -> otspot_core::conic::ConicProblem {
    match p {
        CbfProblem::Socp { problem, .. } => problem,
        CbfProblem::Misocp { .. } => panic!("expected Socp, got Misocp"),
    }
}

fn unwrap_misocp(p: CbfProblem) -> otspot_core::conic::MisocpProblem {
    match p {
        CbfProblem::Misocp { problem, .. } => problem,
        CbfProblem::Socp { .. } => panic!("expected Misocp, got Socp"),
    }
}

// ---------------------------------------------------------------------
// Cone-type correctness: parse -> solve_socp -> compare to hand optimum.
// ---------------------------------------------------------------------

#[test]
fn soc_cone_con_block_matches_hand_optimum() {
    // min x0 s.t. x1 = 1, (x0, x1) in Q  =>  x0* = 1.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
2 1
F 2

CON
3 2
L= 1
Q 2

OBJACOORD
1
0 1.0

ACOORD
3
0 1 1.0
1 0 1.0
2 1 1.0

BCOORD
1
0 -1.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!((res.objective - 1.0).abs() < TOL, "obj={}", res.objective);
    assert!((res.x[0] - 1.0).abs() < 1e-5);
    assert!((res.x[1] - 1.0).abs() < 1e-5);
}

#[test]
fn rotated_soc_var_block_matches_hand_optimum() {
    // (u, v, w) in QR, v = 1, w = 1, min u  =>  2*u*1 >= 1^2  =>  u* = 0.5.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
3 1
QR 3

CON
2 1
L= 2

OBJACOORD
1
0 1.0

ACOORD
2
0 1 1.0
1 2 1.0

BCOORD
2
0 -1.0
1 -1.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!((res.objective - 0.5).abs() < TOL, "obj={}", res.objective);
    assert!((res.x[1] - 1.0).abs() < 1e-5);
    assert!((res.x[2] - 1.0).abs() < 1e-5);
}

#[test]
fn variable_domain_soc_matches_hand_optimum() {
    // (x0, x1) in Q directly (VAR-side cone), x1 = 1, min x0  =>  x0* = 1.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
2 1
Q 2

CON
1 1
L= 1

OBJACOORD
1
0 1.0

ACOORD
1
0 1 1.0

BCOORD
1
0 -1.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!((res.objective - 1.0).abs() < TOL, "obj={}", res.objective);
}

#[test]
fn l_minus_con_block_upper_bound() {
    // min -x0 s.t. x0 - 5 <= 0  =>  x0* = 5, obj* = -5.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

CON
1 1
L- 1

OBJACOORD
1
0 -1.0

ACOORD
1
0 0 1.0

BCOORD
1
0 -5.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    assert_eq!(problem.cone.l, 1);
    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!(
        (res.objective - (-5.0)).abs() < TOL,
        "obj={}",
        res.objective
    );
    assert!((res.x[0] - 5.0).abs() < 1e-5);
}

#[test]
fn l_minus_var_block_matches_hand_optimum() {
    // x0 in L- (x0 <= 0), x0 + 3 >= 0 (L+ CON row)  =>  min x0 => x0* = -3.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
L- 1

CON
1 1
L+ 1

OBJACOORD
1
0 1.0

ACOORD
1
0 0 1.0

BCOORD
1
0 3.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!(
        (res.objective - (-3.0)).abs() < TOL,
        "obj={}",
        res.objective
    );
}

#[test]
fn l_zero_var_block_fixes_variable_via_equality() {
    // x0 in L= (x0 == 0): must become an equality row, not a cone row.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
L= 1

OBJACOORD
1
0 1.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    assert_eq!(problem.p(), 1, "L= must produce one equality row");
    assert_eq!(problem.m(), 0, "L= must not produce a cone row");
    assert_eq!(problem.a.nnz(), 1);
    assert!((problem.b[0] - 0.0).abs() < 1e-12);

    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!(res.objective.abs() < TOL);
    assert!(res.x[0].abs() < 1e-6);
}

// ---------------------------------------------------------------------
// Objective sense / constant handling.
// ---------------------------------------------------------------------

#[test]
fn maximize_negates_objective_and_true_objective_recovers_value() {
    // max x0 + 10 s.t. x0 - 5 <= 0  =>  x0* = 5, true objective = 15.
    let cbf = "\
VER
3

OBJSENSE
MAX

VAR
1 1
F 1

CON
1 1
L- 1

OBJACOORD
1
0 1.0

OBJBCOORD
10.0

ACOORD
1
0 0 1.0

BCOORD
1
0 -5.0
";
    let meta = parse_cbf_str(cbf).unwrap();
    assert!(meta.maximize());
    assert!((meta.obj_offset() - 10.0).abs() < 1e-12);

    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    assert!((problem.c[0] - (-1.0)).abs() < 1e-12, "MAX must negate c");

    let res = solve_socp(&problem, &ConicOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    let true_obj = meta.true_objective(res.objective);
    assert!((true_obj - 15.0).abs() < TOL, "true_obj={true_obj}");
}

// ---------------------------------------------------------------------
// INT section -> MisocpProblem.
// ---------------------------------------------------------------------

#[test]
fn int_section_builds_misocp_with_expected_integer_optimum() {
    // x0 integer, x0 >= 0 (L+), x0 <= 3.7 (L- CON row), min -x0 => x0* = 3.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
L+ 1

INT
1
0

CON
1 1
L- 1

OBJACOORD
1
0 -1.0

ACOORD
1
0 0 1.0

BCOORD
1
0 -3.7
";
    let problem = unwrap_misocp(parse_cbf_str(cbf).unwrap());
    assert_eq!(problem.integers, vec![0]);
    assert_eq!(problem.int_lb[0], 0.0);
    assert!(
        (problem.int_ub[0] - 3.7).abs() < 1e-12,
        "ub must be tightened from the single-variable L- CON row, got {}",
        problem.int_ub[0]
    );

    let res = solve_misocp(&problem, &ConicOptions::default(), &BbOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal, "res={res:?}");
    assert!(
        (res.objective - (-3.0)).abs() < TOL,
        "obj={}",
        res.objective
    );
    assert!((res.x[0] - 3.0).abs() < 1e-6);
}

#[test]
fn int_variable_with_no_finite_bound_is_unsupported_error() {
    // x0 integer, F domain (unbounded), no CON row bounds it at all.
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

INT
1
0

OBJACOORD
1
0 1.0
";
    match parse_cbf_str(cbf) {
        Err(CbfError::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn int_variable_bound_tightened_by_multiple_single_var_rows() {
    // x0 integer, F domain; x0 >= 2 (L+ row) and x0 <= 9 (L- row) => bounds [2, 9].
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

INT
1
0

CON
2 2
L+ 1
L- 1

OBJACOORD
1
0 1.0

ACOORD
2
0 0 1.0
1 0 1.0

BCOORD
2
0 -2.0
1 -9.0
";
    let problem = unwrap_misocp(parse_cbf_str(cbf).unwrap());
    assert!(
        (problem.int_lb[0] - 2.0).abs() < 1e-12,
        "lb={}",
        problem.int_lb[0]
    );
    assert!(
        (problem.int_ub[0] - 9.0).abs() < 1e-12,
        "ub={}",
        problem.int_ub[0]
    );
}

// ---------------------------------------------------------------------
// Duplicate-entry accumulation.
// ---------------------------------------------------------------------

#[test]
fn duplicate_objacoord_entries_accumulate() {
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

OBJACOORD
2
0 2.0
0 3.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    assert!((problem.c[0] - 5.0).abs() < 1e-12);
}

#[test]
fn duplicate_acoord_entries_accumulate() {
    let cbf = "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

CON
1 1
L+ 1

ACOORD
2
0 0 2.0
0 0 3.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    assert_eq!(
        problem.g.nnz(),
        1,
        "duplicate (row, col) must merge to one NZ"
    );
    assert!((problem.g.values()[0] - (-5.0)).abs() < 1e-12);
}

// ---------------------------------------------------------------------
// Comments / blank lines.
// ---------------------------------------------------------------------

#[test]
fn comment_and_blank_lines_are_ignored() {
    let cbf = "\
# a full CBF file with comments and blank lines
VER
3

# variables
OBJSENSE
MIN

VAR
1 1
F 1
# end of var

OBJACOORD
1
0 1.0
";
    let problem = unwrap_socp(parse_cbf_str(cbf).unwrap());
    assert_eq!(problem.n(), 1);
    assert!((problem.c[0] - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------
// Error branches.
// ---------------------------------------------------------------------

fn expect_parse_error(cbf: &str) {
    match parse_cbf_str(cbf) {
        Err(CbfError::ParseError(_)) => {}
        other => panic!("expected ParseError, got {other:?}"),
    }
}

fn expect_unsupported(cbf: &str) {
    match parse_cbf_str(cbf) {
        Err(CbfError::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn missing_ver_is_error() {
    expect_parse_error(
        "\
OBJSENSE
MIN

VAR
1 1
F 1
",
    );
}

#[test]
fn missing_objsense_is_error() {
    expect_parse_error(
        "\
VER
3

VAR
1 1
F 1
",
    );
}

#[test]
fn missing_var_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN
",
    );
}

#[test]
fn bad_objsense_token_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
FOO

VAR
1 1
F 1
",
    );
}

#[test]
fn unsupported_ver_is_error() {
    expect_unsupported(
        "\
VER
99

OBJSENSE
MIN

VAR
1 1
F 1
",
    );
}

#[test]
fn psdvar_section_is_unsupported_error() {
    expect_unsupported(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

PSDVAR
1
2
",
    );
}

#[test]
fn psdcon_section_is_unsupported_error() {
    expect_unsupported(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

PSDCON
1
2
",
    );
}

#[test]
fn objfcoord_section_is_unsupported_error() {
    expect_unsupported(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

OBJFCOORD
1
0 0 0 1.0
",
    );
}

#[test]
fn unsupported_cone_token_exp_is_error() {
    expect_unsupported(
        "\
VER
3

OBJSENSE
MIN

VAR
3 1
EXP 3
",
    );
}

#[test]
fn q_cone_size_too_small_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
Q 1
",
    );
}

#[test]
fn qr_cone_size_too_small_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN

VAR
2 1
QR 2
",
    );
}

#[test]
fn cone_block_size_sum_mismatch_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN

VAR
5 1
F 3
",
    );
}

#[test]
fn acoord_row_out_of_range_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

CON
1 1
L+ 1

ACOORD
1
5 0 1.0
",
    );
}

#[test]
fn objacoord_var_out_of_range_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

OBJACOORD
1
5 1.0
",
    );
}

#[test]
fn non_finite_float_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

CON
1 1
L+ 1

BCOORD
1
0 NaN
",
    );
}

#[test]
fn unknown_section_keyword_is_error() {
    expect_parse_error(
        "\
VER
3

OBJSENSE
MIN

VAR
1 1
F 1

FOO
",
    );
}
