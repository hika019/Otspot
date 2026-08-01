//! Rust-side half of the API parity guarantee: every symbol listed in
//! `api_manifest.json` is checked against the *real* `otspot_model` /
//! `otspot_core` public API (not the PyO3 bindings) via compile-time
//! references and live invocation. If a manifested method/type is renamed
//! or removed upstream, this file fails to compile; if its behavior
//! changes, the invocation tests below fail. See `tests/test_api_manifest.py`
//! for the Python-side half.
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
        "out_of_scope",
    ] {
        assert!(
            v.get(key).is_some(),
            "manifest missing top-level key: {key}"
        );
    }
    assert_eq!(v["types"].as_array().unwrap().len(), 12);
    assert_eq!(
        v["model_error_exceptions"]["entries"]
            .as_array()
            .unwrap()
            .len(),
        7
    );
}

/// Every `types` entry names a real `otspot_model`/`otspot_core` type.
/// Compile-time check: renaming/removing any of these breaks this file.
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

/// The two independently hand-computed oracle problems shared with the
/// Python-side behavior parity test (`tests/test_parity_lp_qp.py`).
mod oracle {
    /// min x + 2y  s.t. 2x + 3y <= 12, x + y >= 3, x in [0, inf), y in [0, 10].
    ///
    /// Hand solution: since c_x=1 < c_y=2, push y to 0 and satisfy x+y>=3
    /// with x=3 (tight); check 2*3+3*0=6<=12 (slack). Any y>0 only raises
    /// the objective faster than it could relax the x-lower-bound, so
    /// (x,y)=(3,0) is optimal with objective 3.
    pub const LP_EXPECTED_OBJECTIVE: f64 = 3.0;
    pub const LP_EXPECTED_X: f64 = 3.0;
    pub const LP_EXPECTED_Y: f64 = 0.0;

    /// min x^2 + y^2 - 2x - 4y + 5  s.t. x + y <= 3, x >= 0, y >= 0.
    ///
    /// Hand solution: complete the square, x^2-2x+y^2-4y+5 = (x-1)^2+(y-2)^2.
    /// The unconstrained minimizer (1, 2) satisfies x+y=3<=3, x>=0, y>=0, so
    /// it is feasible; since the objective is strictly convex, a feasible
    /// unconstrained minimizer is automatically the constrained global
    /// minimizer too. Objective value at (1, 2) is exactly 0.
    pub const QP_EXPECTED_OBJECTIVE: f64 = 0.0;
    pub const QP_EXPECTED_X: f64 = 1.0;
    pub const QP_EXPECTED_Y: f64 = 2.0;

    /// IPM `Tolerance::Medium` (eps=1e-6) bounds KKT residuals, not raw
    /// variable-value error directly; 1e-4 is comfortably above the observed
    /// ~4e-5 gap on the QP oracle while still being a meaningful check.
    pub const TOL: f64 = 1e-4;
}

/// Builds and solves the LP oracle, invoking every `Model`/`Variable`/
/// `Expression`/`ModelResult` method the manifest lists for the linear path.
#[test]
fn lp_oracle_matches_hand_solution_and_exercises_manifested_api() {
    let mut model = Model::new("lp_oracle");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, 10.0);
    assert_eq!(model.var_name(x), "x");
    assert_eq!(model.var_name(y), "y");

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
}

/// Builds and solves the QP oracle (quadratic objective via `pow2`/`Mul`),
/// exercising the `QuadExpr` side of the manifest.
#[test]
fn qp_oracle_matches_hand_solution_and_exercises_manifested_api() {
    let mut model = Model::new("qp_oracle");
    let x = model.add_var("x", 0.0, f64::INFINITY);
    let y = model.add_var("y", 0.0, f64::INFINITY);

    model.add_constraint((x + y).leq(3.0));
    let obj = x.pow2() + y.pow2() - 2.0 * x - 4.0 * y + 5.0;
    assert!(!obj.is_linear());
    model.minimize(obj);
    model.set_timeout(30.0);

    let result = model.solve().expect("QP oracle must solve");
    assert_eq!(result.status, SolveStatus::Optimal);
    assert!((result.objective() - oracle::QP_EXPECTED_OBJECTIVE).abs() < oracle::TOL);
    assert!((result.value(x) - oracle::QP_EXPECTED_X).abs() < oracle::TOL);
    assert!((result.value(y) - oracle::QP_EXPECTED_Y).abs() < oracle::TOL);
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
#[test]
fn non_exhaustive_enum_variant_names_match_manifest() {
    let v = manifest();

    let cs_names = ["Le", "Ge", "Eq"];
    let _: [ConstraintSense; 3] = [
        ConstraintSense::Le,
        ConstraintSense::Ge,
        ConstraintSense::Eq,
    ];
    assert_eq!(
        cs_names.to_vec(),
        as_strs(&v["variants"]["ConstraintSense"])
    );

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
