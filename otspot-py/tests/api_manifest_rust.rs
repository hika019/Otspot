//! Rust-side half of the API parity guarantee: every symbol listed in
//! `api_manifest.json` is checked against the *real* `otspot_model` /
//! `otspot_core` public API (not the PyO3 bindings) via compile-time
//! references and live invocation. If a manifested method/type is renamed
//! or removed upstream, this file fails to compile; if its behavior
//! changes, the invocation tests below fail. See `tests/test_api_manifest.py`
//! for the Python-side half, and `otspot_model_api_snapshot.txt` +
//! `scripts/check_otspot_model_public_api.sh` for the `cargo public-api`
//! guard against *unmanifested* Rust-side additions.
//!
//! `#[non_exhaustive]` enums (`SolveStatus`, `ConstraintSense`,
//! `SolutionProof`, `SolveError`, `Tolerance`, `ModelError`) force a
//! wildcard match arm for any downstream crate, including this one — Rust's
//! own contract for that attribute. That means variant *removal/rename* is
//! still a compile error here (each arm names a real variant), but variant
//! *addition* upstream cannot be caught by the compiler; the rest of this
//! codebase accepts the same limitation (see the `// #[non_exhaustive]:
//! wildcard required` comments in otspot-model/src/model.rs).

use otspot_core::options::Tolerance;
use otspot_core::problem::SolveStatus;
use otspot_model::{
    Constraint, ConstraintSense, Expression, Model, ModelError, ModelResult, QuadExpr,
    SolutionProof, SolveError, VarKind, Variable,
};

const MANIFEST_JSON: &str = include_str!("../api_manifest.json");

fn manifest() -> serde_json::Value {
    serde_json::from_str(MANIFEST_JSON).expect("api_manifest.json must be valid JSON")
}

#[test]
fn manifest_has_expected_top_level_shape() {
    let v = manifest();
    for key in [
        "types",
        "model_error_exceptions",
        "methods",
        "variants",
        "python_only_variants",
        "out_of_scope",
    ] {
        assert!(
            v.get(key).is_some(),
            "manifest missing top-level key: {key}"
        );
    }
    assert_eq!(v["types"].as_array().unwrap().len(), 11);
    assert_eq!(
        v["model_error_exceptions"]["entries"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
}

/// Every `types` entry names a real `otspot_model`/`otspot_core` type.
/// Compile-time check: renaming/removing any of these breaks this file.
/// `ConstraintSense` is included even though its Python binding was removed
/// (see api_manifest.json's `out_of_scope`) — the *Rust* type is still real
/// and still referenced by `non_exhaustive_enum_variant_names_match_manifest`.
#[allow(dead_code, clippy::too_many_arguments)]
fn assert_types_exist(
    _m: &Model,
    _v: &Variable,
    _e: &Expression,
    _q: &QuadExpr,
    _c: &Constraint,
    _cs: &ConstraintSense,
    _vk: &VarKind,
    _mr: &ModelResult,
    _sp: &SolutionProof,
    _se: &SolveError,
    _ss: &SolveStatus,
    _tol: &Tolerance,
) {
}

/// Independently hand-computed oracle problems shared with the Python-side
/// behavior parity test (`tests/test_parity_lp_qp.py`). Each docstring is the
/// derivation, not a restatement of what the solver happens to return.
mod oracle {
    /// min x + 2y  s.t. -x <= 0, 2x + 3y <= 12, x + y >= 3, x in [0, inf), y in [0, 10].
    ///
    /// Hand solution: since c_x=1 < c_y=2, push y to 0 and satisfy x+y>=3
    /// with x=3 (tight); check 2*3+3*0=6<=12 (slack), and -3<=0 (slack). Any
    /// y>0 only raises the objective faster than it could relax the
    /// x-lower-bound, so (x,y)=(3,0) is optimal with objective 3.
    pub const LP_EXPECTED_OBJECTIVE: f64 = 3.0;
    pub const LP_EXPECTED_X: f64 = 3.0;
    pub const LP_EXPECTED_Y: f64 = 0.0;
    /// Constraint row 1 (2x+3y<=12) slack at (3,0): 12 - 6 = 6.
    pub const LP_EXPECTED_ROW1_SLACK: f64 = 6.0;
    /// Constraint row 2 (x+y>=3) is tight at (3,0): slack 0.
    pub const LP_EXPECTED_ROW2_SLACK: f64 = 0.0;

    /// min x^2 + y^2  s.t. x + y >= 2, x >= 0, y >= 0.
    ///
    /// Hand solution: the unconstrained minimizer of x^2+y^2 is the origin,
    /// which violates x+y>=2. The constrained minimum over the half-plane
    /// x+y>=2 is the perpendicular projection of the origin onto the line
    /// x+y=2, i.e. (1, 1) (elementary geometry: minimize distance-squared to
    /// a point subject to a linear equality is solved by the foot of the
    /// perpendicular). x=1,y=1 satisfies x,y>=0. Objective = 1+1 = 2. Unlike
    /// an earlier `x+y<=3` formulation (never binding at the unconstrained
    /// optimum (1,2) it admits), this constraint is load-bearing: removing
    /// it changes the optimum to (0,0).
    pub const QP_EXPECTED_OBJECTIVE: f64 = 2.0;
    pub const QP_EXPECTED_X: f64 = 1.0;
    pub const QP_EXPECTED_Y: f64 = 1.0;

    /// min 2b + z  s.t. b + z >= 2.5, b in {0,1}, z integer in [0, 10].
    ///
    /// Hand solution: b=0 forces z>=2.5, i.e. z>=3 (integer ceiling),
    /// objective 3. b=1 forces z>=1.5, i.e. z>=2, objective 2+2=4. The
    /// minimum over both branches is 3, at (b=0, z=3).
    pub const MILP_EXPECTED_OBJECTIVE: f64 = 3.0;
    pub const MILP_EXPECTED_B: f64 = 0.0;
    pub const MILP_EXPECTED_Z: f64 = 3.0;

    /// max x + y  s.t. x + y <= 8, x in [0, 10], y in [0, 10].
    ///
    /// Multiple optima (any point with x+y=8 and both coordinates in
    /// [0,10]), but the objective value is unique: 8.
    pub const MAX_EXPECTED_OBJECTIVE: f64 = 8.0;

    /// min x  s.t. 5.0 - x == 0, x in [0, 10].
    ///
    /// The equality constraint forces x=5 exactly, the only feasible point;
    /// objective there is 5.
    pub const EQ_EXPECTED_OBJECTIVE: f64 = 5.0;
    pub const EQ_EXPECTED_X: f64 = 5.0;

    /// IPM `Tolerance::Medium` (eps=1e-6) bounds KKT residuals, not raw
    /// variable-value error directly; 1e-4 is comfortably above the observed
    /// ~4e-5 gap on the QP oracle while still being a meaningful check.
    pub const TOL: f64 = 1e-4;
}

/// Builds and solves the LP oracle, invoking every manifested `Model`
/// setter plus `var_name`/`var_kind`, and checking `slack`/`dual_solution`/
/// `reduced_costs`/`bound_duals` against sign-convention-independent facts
/// (exact slack values, and complementary-slackness zero-duals on
/// non-binding rows/bounds) rather than the solver's internal sign choice.
#[test]
fn lp_oracle_matches_hand_solution_and_exercises_manifested_api() {
    let mut model = Model::new("lp_oracle");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);
    assert_eq!(model.var_name(x), "x");
    assert_eq!(model.var_name(y), "y");
    assert_eq!(model.var_kind(x), VarKind::Continuous);
    assert_eq!(model.var_kind(y), VarKind::Continuous);

    // Row 0: exercises Neg (redundant vs. x's own lb=0, never binding).
    model.add_constraint((-x).leq(0.0));
    // Row 1, row 2: the real problem.
    model.add_constraint((2.0 * x + 3.0 * y).leq(12.0));
    model.add_constraint((x + y).geq(3.0));
    model.minimize(QuadExpr::from(x + 2.0 * y));
    model.set_tolerance(Tolerance::Medium);
    model.set_presolve(true);
    model.set_threads(1);
    model.set_obj_offset(0.0);

    let result = model.solve().expect("LP oracle must solve");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert_eq!(result.proof, SolutionProof::GlobalOptimal);
    assert!(result.has_global_optimality_proof());
    assert!(
        (result.objective() - oracle::LP_EXPECTED_OBJECTIVE).abs() < oracle::TOL,
        "objective: expected {}, got {}",
        oracle::LP_EXPECTED_OBJECTIVE,
        result.objective()
    );
    assert!((result.value(x) - oracle::LP_EXPECTED_X).abs() < oracle::TOL);
    assert!((result[y] - oracle::LP_EXPECTED_Y).abs() < oracle::TOL);

    let slack = result.slack.expect("LP path must populate slack");
    assert_eq!(slack.len(), 3, "3 explicit add_constraint rows");
    assert!((slack[1] - oracle::LP_EXPECTED_ROW1_SLACK).abs() < oracle::TOL);
    assert!((slack[2] - oracle::LP_EXPECTED_ROW2_SLACK).abs() < oracle::TOL);

    let dual = result
        .dual_solution
        .expect("LP path must populate dual_solution");
    assert_eq!(dual.len(), 3);
    // Rows 0 and 1 are slack (not tight): by complementary slackness their
    // duals must be exactly 0, regardless of the solver's sign convention.
    assert!(dual[0].abs() < oracle::TOL, "row 0 (-x<=0) is slack");
    assert!(dual[1].abs() < oracle::TOL, "row 1 (2x+3y<=12) is slack");

    let rc = result
        .reduced_costs
        .expect("LP path must populate reduced_costs");
    assert_eq!(rc.len(), 2);
    // x sits strictly inside its bounds (3 in (0, inf)): reduced cost 0,
    // sign-convention independent.
    assert!(rc[0].abs() < oracle::TOL, "x is not at a bound");

    assert!(
        result.bound_duals.is_empty(),
        "LP path leaves bound_duals empty by design (see ModelResult doc comment); QP populates it"
    );
}

/// Builds and solves the QP oracle (quadratic objective via `pow2`), whose
/// constraint is load-bearing (nonzero multiplier at the optimum) — unlike
/// an `x+y<=3` formulation, dropping this constraint changes the optimum.
#[test]
fn qp_oracle_matches_hand_solution_and_exercises_manifested_api() {
    let mut model = Model::new("qp_oracle");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);

    model.add_constraint((x + y).geq(2.0));
    // `0.0 + ...` exercises Add<QuadExpr> for f64.
    let obj = 0.0 + x.pow2() + y.pow2();
    assert!(!obj.is_linear());
    model.minimize(obj);
    model.set_timeout(30.0);

    let result = model.solve().expect("QP oracle must solve");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.objective() - oracle::QP_EXPECTED_OBJECTIVE).abs() < oracle::TOL);
    assert!((result.value(x) - oracle::QP_EXPECTED_X).abs() < oracle::TOL);
    assert!((result.value(y) - oracle::QP_EXPECTED_Y).abs() < oracle::TOL);

    let dual = result
        .dual_solution
        .expect("QP path must populate dual_solution for a binding constraint");
    assert_eq!(dual.len(), 1);
    assert!(
        dual[0].abs() > 1e-3,
        "x+y>=2 must be load-bearing (nonzero multiplier), got {}",
        dual[0]
    );
    assert!(
        !result.bound_duals.is_empty(),
        "QP path must populate bound_duals, unlike LP"
    );
}

/// Exercises `add_int_var`/`add_binary_var`/`var_kind` on a hand-verified
/// MILP (unreachable by rounding a plain LP relaxation, since the b=1
/// branch is worse, not just fractional).
#[test]
fn milp_oracle_exercises_int_and_binary_vars() {
    let mut model = Model::new("milp_oracle");
    let b = model.add_binary_var("b");
    let z = model.add_int_var("z", 0.0, 10.0);
    assert_eq!(model.var_kind(b), VarKind::Binary);
    assert_eq!(model.var_kind(z), VarKind::Integer);

    model.add_constraint((b + z).geq(2.5));
    model.minimize(QuadExpr::from(2.0 * b + z));

    let result = model.solve().expect("MILP oracle must solve");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.objective() - oracle::MILP_EXPECTED_OBJECTIVE).abs() < oracle::TOL);
    assert!((result.value(b) - oracle::MILP_EXPECTED_B).abs() < oracle::TOL);
    assert!((result.value(z) - oracle::MILP_EXPECTED_Z).abs() < oracle::TOL);
}

/// Exercises `Model::maximize` and `Add<Variable> for f64` (`0.0 + x`).
#[test]
fn maximize_oracle_exercises_maximize_and_radd() {
    let mut model = Model::new("maximize_oracle");
    let x = model.add_var("x", 0.0, 10.0);
    let y = model.add_var("y", 0.0, 10.0);
    model.add_constraint((x + y).leq(8.0));
    model.maximize(0.0 + x + y);

    let result = model.solve().expect("maximize oracle must solve");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.objective() - oracle::MAX_EXPECTED_OBJECTIVE).abs() < oracle::TOL);
}

/// Exercises `Expression::eq_constraint` and `Sub<Variable> for f64` (`5.0 - x`).
#[test]
fn eq_constraint_oracle_exercises_rsub_and_eq_constraint() {
    let mut model = Model::new("eq_constraint_oracle");
    let x = model.add_var("x", 0.0, 10.0);
    model.add_constraint((5.0 - x).eq_constraint(0.0));
    model.minimize(QuadExpr::from(x));

    let result = model.solve().expect("eq_constraint oracle must solve");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.objective() - oracle::EQ_EXPECTED_OBJECTIVE).abs() < oracle::TOL);
    assert!((result.value(x) - oracle::EQ_EXPECTED_X).abs() < oracle::TOL);
}

/// `Model::solve` on a structurally infeasible LP returns
/// `ModelError::SolveError(SolveError::Infeasible)`, exercising the
/// `model_error_exceptions` mapping's `SolveError` branch.
#[test]
fn infeasible_lp_yields_solve_error_infeasible() {
    let mut model = Model::new("infeasible");
    let x = model.add_var("x", 0.0, 1.0);
    model.add_constraint(Expression::from(x).geq(5.0));
    model.minimize(QuadExpr::from(x));
    match model.solve() {
        Err(ModelError::SolveError(SolveError::Infeasible)) => {}
        other => panic!("expected SolveError(Infeasible), got {other:?}"),
    }
}

/// Same `model_error_exceptions` mapping, `SolveError::Unbounded` branch.
#[test]
fn unbounded_lp_yields_solve_error_unbounded() {
    let mut model = Model::new("unbounded");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    model.minimize(QuadExpr::from(-1.0 * x));
    match model.solve() {
        Err(ModelError::SolveError(SolveError::Unbounded)) => {}
        other => panic!("expected SolveError(Unbounded), got {other:?}"),
    }
}

/// `ModelError::Timeout` mapping: `set_timeout(0.0)` reliably (verified
/// empirically, not just "should" per the doc comment) yields Timeout even
/// for a trivial LP, since the deadline is already expired by the first
/// check.
#[test]
fn zero_timeout_yields_timeout_error() {
    let mut model = Model::new("timeout");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);
    model.add_constraint((2.0 * x + 3.0 * y).leq(12.0));
    model.add_constraint((x + y).geq(3.0));
    model.minimize(QuadExpr::from(x + 2.0 * y));
    model.set_timeout(0.0);
    match model.solve() {
        Err(ModelError::Timeout) => {}
        other => panic!("expected Timeout, got {other:?}"),
    }
}

/// `ModelError::NonConvex` mapping: an indefinite-Q MIQP (mirrors
/// otspot-model's own `miqp_nonconvex_q_returns_nonconvex_error` sentinel).
/// A *continuous* indefinite QP is not a reliable trigger here: IPM inertia
/// correction can converge it to a `LocallyOptimal` KKT point instead
/// (verified empirically) rather than reporting `NonConvex`.
#[test]
fn miqp_indefinite_q_yields_nonconvex_error() {
    let mut model = Model::new("nonconvex");
    let x = model.add_binary_var("x");
    let y = model.add_binary_var("y");
    model.minimize((-0.5) * (x * x) + 0.5 * (y * y));
    match model.solve() {
        Err(ModelError::NonConvex(_)) => {}
        other => panic!("expected NonConvex, got {other:?}"),
    }
}

/// `Model::solve` before `minimize`/`maximize` returns `ModelError::NoObjective`.
#[test]
fn missing_objective_yields_no_objective_error() {
    let mut model = Model::new("no_objective");
    model.add_var("x", 0.0, 1.0);
    match model.solve() {
        Err(ModelError::NoObjective) => {}
        other => panic!("expected NoObjective, got {other:?}"),
    }
}

/// Every `variants.VarKind` name is a real `VarKind` variant (exhaustive
/// match, no wildcard needed: `VarKind` is not `#[non_exhaustive]`, so this
/// is a full compile-time completeness guarantee in both directions).
#[test]
fn var_kind_variants_match_manifest_exhaustively() {
    fn name(k: VarKind) -> &'static str {
        match k {
            VarKind::Continuous => "Continuous",
            VarKind::Integer => "Integer",
            VarKind::Binary => "Binary",
        }
    }
    let want: Vec<String> = manifest()["variants"]["VarKind"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let got: Vec<&str> = vec![
        name(VarKind::Continuous),
        name(VarKind::Integer),
        name(VarKind::Binary),
    ];
    assert_eq!(got, want);
}

/// `#[non_exhaustive]` variant-name checks: compile error on removal/rename
/// (each name below must be a real variant), count-checked against the
/// manifest. See the module doc comment for why addition can't be caught.
/// `ConstraintSense`'s Rust variants are still checked here even though its
/// Python binding is out of scope (api_manifest.json has no `variants`
/// entry for it) — this only asserts the *Rust* enum's shape is unchanged.
#[test]
fn non_exhaustive_enum_variant_names_match_manifest() {
    let v = manifest();

    let _: [ConstraintSense; 3] = [
        ConstraintSense::Le,
        ConstraintSense::Ge,
        ConstraintSense::Eq,
    ];

    let sp_names = ["GlobalOptimal", "LocalOptimal", "FeasibleUnproven"];
    let _: [SolutionProof; 3] = [
        SolutionProof::GlobalOptimal,
        SolutionProof::LocalOptimal,
        SolutionProof::FeasibleUnproven,
    ];
    assert_eq!(sp_names.to_vec(), as_strs(&v["variants"]["SolutionProof"]));

    let se_names = [
        "Infeasible",
        "Unbounded",
        "MaxIterations",
        "Stalled",
        "NumericalError",
    ];
    let _: [SolveError; 5] = [
        SolveError::Infeasible,
        SolveError::Unbounded,
        SolveError::MaxIterations,
        SolveError::Stalled,
        SolveError::NumericalError,
    ];
    assert_eq!(se_names.to_vec(), as_strs(&v["variants"]["SolveError"]));

    let tol_names = ["High", "Medium", "Fast", "Custom"];
    let _: [Tolerance; 4] = [
        Tolerance::High,
        Tolerance::Medium,
        Tolerance::Fast,
        Tolerance::Custom(1e-7),
    ];
    assert_eq!(tol_names.to_vec(), as_strs(&v["variants"]["Tolerance"]));

    let ss_names = [
        "Optimal",
        "LocallyOptimal",
        "Infeasible",
        "Unbounded",
        "MaxIterations",
        "SuboptimalSolution",
        "Stalled",
        "FeasiblePoint",
        "Timeout",
        "NumericalError",
        "NonConvex",
        "NonconvexLocal",
        "NonconvexGlobal",
        "NotSupported",
    ];
    let _: [SolveStatus; 14] = [
        SolveStatus::Optimal,
        SolveStatus::LocallyOptimal,
        SolveStatus::Infeasible,
        SolveStatus::Unbounded,
        SolveStatus::MaxIterations,
        SolveStatus::SuboptimalSolution,
        SolveStatus::Stalled,
        SolveStatus::FeasiblePoint,
        SolveStatus::Timeout,
        SolveStatus::NumericalError,
        SolveStatus::NonConvex(String::new()),
        SolveStatus::NonconvexLocal,
        SolveStatus::NonconvexGlobal,
        SolveStatus::NotSupported(String::new()),
    ];
    assert_eq!(ss_names.to_vec(), as_strs(&v["variants"]["SolveStatus"]));
}

fn as_strs(v: &serde_json::Value) -> Vec<&str> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect()
}
