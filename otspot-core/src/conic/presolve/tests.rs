//! Unit tests for the tier-1 conic presolve: reductions, hand-computed
//! (oracle) solutions and duals, certificate recovery, revert-fail
//! sentinels, dim/formula contracts.

use super::*;
use crate::conic::{solve_socp, ConeSpec, ConicOptions, ConicProblem, ConicResult};
use crate::problem::SolveStatus;
use otspot_num::sparse::CscMatrix;

fn csc(n_rows: usize, n_cols: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut v = Vec::new();
    for &(ri, ci, vi) in entries {
        r.push(ri);
        c.push(ci);
        v.push(vi);
    }
    CscMatrix::from_triplets(&r, &c, &v, n_rows, n_cols).expect("test csc must be valid")
}

fn prob(
    c: Vec<f64>,
    a: CscMatrix,
    b: Vec<f64>,
    g: CscMatrix,
    h: Vec<f64>,
    cone: ConeSpec,
) -> ConicProblem {
    ConicProblem {
        c,
        a,
        b,
        g,
        h,
        cone,
        var_bounds: Vec::new(),
    }
}

fn zip_rows(m: &CscMatrix, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
    let (rows, vals) = m.column(j);
    (0..rows.len()).map(move |t| (rows[t], vals[t]))
}

fn opts() -> ConicOptions {
    ConicOptions::default()
}

/// `min x0 + x3  s.t.  x1 = 3, x2 = 4 (equalities), ||(x1, x2)|| <= x0`
/// (rows: t = x0, u = x1, u = x2), `x3 >= 0` via a domain row, `x3` cost 1.
/// Oracle: fixing x1, x2 makes the cone `sqrt(9+16) = 5 <= x0`, `x3* = 0`
/// ⇒ x* = (5, 3, 4, 0), obj 5. The two identity rows fix x1/x2 (fixed-var
/// elimination distributes their values into the cone rows); the reduction
/// must engage (n0 = 4 > gate) and postsolve must rebuild x exactly.
#[test]
fn equality_fixing_recovers_hand_computed_optimum() {
    let a = csc(2, 4, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let b = vec![3.0, 4.0];
    // Cone layout is positional: G row 0 = orthant (x3 >= 0), rows 1..3 =
    // SOC block (t, u1, u2).
    let g = csc(
        4,
        4,
        &[(0, 3, -1.0), (1, 0, -1.0), (2, 1, -1.0), (3, 2, -1.0)],
    );
    let problem = prob(
        vec![1.0, 0.0, 0.0, 1.0],
        a,
        b,
        g,
        vec![0.0; 4],
        ConeSpec { l: 1, soc: vec![3] },
    );
    let red = ConicReduction::reduce(&problem).expect("presolve must engage");
    assert!(red.conflict().is_none());
    let reduced = red.reduced().expect("reduced problem");
    assert_eq!(
        red.stats().n_substituted,
        2,
        "identity rows x1=3, x2=4 pivot their columns out"
    );
    assert_eq!(reduced.n(), 2, "reduced to x0, x3");
    assert_eq!(reduced.cone.soc, vec![3], "cone block rows preserved");

    let res = solve_socp(&problem, &opts());
    assert_eq!(res.status, SolveStatus::Optimal, "{:?}", res.status);
    assert!((res.objective - 5.0).abs() <= 1e-6, "obj {}", res.objective);
    for (got, want) in res.x.iter().zip([5.0, 3.0, 4.0, 0.0]) {
        assert!((got - want).abs() <= 1e-5, "x {:?}", res.x);
    }
    // Stationarity convention (ipm.rs: rx = c + Aᵀy + Gᵀz): col x0 ⇒
    // 1 + (-1)·z_t = 0 ⇒ z_t = 1 (SOC leader is original G row 1); fixed
    // columns carry no stationarity pressure; the cone dual must map back
    // through the elimination chain.
    assert!((res.z[1] - 1.0).abs() <= 1e-4, "cone dual {:?}", res.z);
}

/// Dual-recovery chain, hand-driven (RED sentinel for the `Eliminated`
/// dual formula): `min x1 + 2·x2  s.t.  x1 = 1, x1 + x2 = 3`, plus an
/// inactive slack column `x3 >= 0` (cost 0) keeping the cone nondegenerate.
/// Oracle: x = (1, 2); stationarity c + Aᵀy + Gᵀz = 0 with z = 0 (rows 2,3
/// inactive): col x2 ⇒ 2 + y2 = 0 ⇒ y2 = -2; col x1 ⇒ 1 + y1 + y2 = 0 ⇒
/// y1 = 1. Feed the *reduced* solve (x3-side only, y1 = y2 = 0 there is
/// impossible — reduced keeps no columns for them, so both row duals must
/// come out of the LIFO replay) and assert the mapped duals exactly.
#[test]
fn substitution_dual_chain_matches_hand_computed() {
    let a = csc(2, 3, &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 1.0)]);
    let b = vec![1.0, 3.0];
    // Cone: one orthant row x3 >= 0 only; rows 0..0 soc: none.
    let g = csc(1, 3, &[(0, 2, -1.0)]);
    let problem = prob(
        vec![1.0, 2.0, 0.0],
        a,
        b,
        g,
        vec![0.0],
        ConeSpec { l: 1, soc: vec![] },
    );
    let red = ConicReduction::reduce(&problem).expect("substitution engages");
    let reduced = red.reduced().unwrap();
    // Both equality rows pivot out x0 then x1 (n 3 → gate at 2 stops after
    // one pivot, then the identity fix takes the other): reduced keeps the
    // slack column and one of x0/x2 or both — assert only that it shrank.
    assert!(reduced.n() < 3, "n {}", reduced.n());
    // Reduced space (gate stops at 2 live columns): x1 survives with row
    // `x1 = 2`, cost 2 ⇒ x1 = 2, row dual from stationarity 2 + y = 0 ⇒
    // y_red = -2; the slack column x2 = 0 with z = 0 (inactive).
    let _ = reduced.n();
    let res = ConicResult {
        status: SolveStatus::Optimal,
        objective: 3.0,
        x: vec![2.0, 0.0],
        y: vec![-2.0],
        z: vec![0.0],
        s: reduced.h.clone(),
        iterations: 0,
        residuals: (0.0, 0.0, 0.0),
        primal_ray: None,
        infeas_cert: None,
    };
    let mapped = red
        .postsolve(&problem, reduced, &res, 1e-8)
        .expect("postsolve maps");
    assert_eq!(mapped.x[0], 1.0, "x0 fixed by row 0");
    assert_eq!(mapped.x[1], 2.0, "x2 (index1) from x1+x2=3");
    // Original stationarity must hold with the *replayed* duals:
    // c + Aᵀy + Gᵀz = 0 and y = (1, -2) by hand.
    assert!(
        (mapped.y[0] - 1.0).abs() <= 1e-9 && (mapped.y[1] + 2.0).abs() <= 1e-9,
        "y {:?}",
        mapped.y
    );
    for j in 0..problem.c.len() {
        let mut r = problem.c[j];
        for (i, v) in zip_rows(&problem.a, j) {
            r += v * mapped.y[i];
        }
        for (k, v) in zip_rows(&problem.g, j) {
            r += v * mapped.z[k];
        }
        assert!(r.abs() <= 1e-8, "stationarity col {j}: {r}");
    }
}

/// Absorption: a duplicate domain row must shrink `m` without changing the
/// solve. `min -x1  s.t.  x0 >= 0 (orthant), x1 >= 0 twice (identical rows
/// 1,2), x1 <= 2 (orthant)`; optimum x1 = 2 with objective -2. The duplicate
/// drops as implied (`DroppedBound`, dual 0).
#[test]
fn duplicate_domain_row_absorbed_without_semantic_change() {
    let g = csc(
        4,
        2,
        &[
            (0, 0, -1.0), // x0 >= 0
            (1, 1, -1.0), // x1 >= 0
            (2, 1, -1.0), // x1 >= 0 (duplicate)
            (3, 1, 1.0),  // x1 <= 2
        ],
    );
    let problem = prob(
        vec![0.0, -1.0],
        csc(0, 2, &[]),
        vec![],
        g,
        vec![0.0, 0.0, 0.0, 2.0],
        ConeSpec { l: 4, soc: vec![] },
    );
    let red = ConicReduction::reduce(&problem).expect("duplicate absorbed");
    let reduced = red.reduced().unwrap();
    // All four identity rows are absorbed into the bound layer; the three
    // surviving bounds (x0 >= 0, x1 >= 0, x1 <= 2) re-materialize as rows,
    // the duplicate drops as implied.
    assert_eq!(reduced.m(), 3, "duplicate gone, surviving bounds kept");
    assert_eq!(red.stats().m_domain_absorbed, 4);
    let res = solve_socp(&problem, &opts());
    assert_eq!(res.status, SolveStatus::Optimal, "{:?}", res.status);
    assert!((res.objective + 2.0).abs() <= 1e-6, "obj {}", res.objective);
    // The dropped duplicate's dual is 0; the surviving bound rows carry the
    // pressure (upper bound active: z_row3 = 1 by stationarity on col x1:
    // -1 + z3 = 0).
    assert!(
        (res.z[2]).abs() <= 1e-6 && (res.z[3] - 1.0).abs() <= 1e-4,
        "z {:?}",
        res.z
    );
}

/// Bound conflict: `x1 >= 1` and `x1 <= 0` ⇒ Infeasible with an exact
/// Farkas pair verified independently in the original space.
#[test]
fn bound_conflict_emits_verified_farkas_pair() {
    let g = csc(
        4,
        2,
        &[
            (0, 0, -1.0), // x0 >= 0
            (1, 1, -1.0), // x1 >= 0
            (2, 1, 1.0),  // x1 <= 0
            (3, 1, -1.0), // x1 >= 1 (h = -1)
        ],
    );
    let problem = prob(
        vec![0.0, -1.0],
        csc(0, 2, &[]),
        vec![],
        g,
        vec![0.0, 0.0, 0.0, -1.0],
        ConeSpec { l: 4, soc: vec![] },
    );
    let res = solve_socp(&problem, &opts());
    assert_eq!(res.status, SolveStatus::Infeasible, "{:?}", res.status);
    assert_eq!(res.objective, f64::INFINITY);
    let (y, z) = res.infeas_cert.clone().expect("Farkas pair present");
    // Independent original-space check (ipm.rs convention): Gᵀz = 0 on
    // every column, z >= 0, and -(b·y + h·z) > 0.
    for j in 0..problem.c.len() {
        let mut r = 0.0;
        for (k, v) in zip_rows(&problem.g, j) {
            r += v * z[k];
        }
        assert!(r.abs() <= 1e-12, "Gᵀz col {j}: {r}");
    }
    assert!(z.iter().all(|v| *v >= 0.0), "z {z:?}");
    assert!(y.iter().all(|v| *v == 0.0), "y {y:?}");
    let val: f64 = problem.h.iter().zip(&z).map(|(h, v)| h * v).sum();
    assert!(val < -1e-6, "Farkas value {val}");
}

/// Conflicting equalities `x0 = 1`, `x0 = 2`: the tier declines (the exact
/// Farkas pair would need the substitution chain's dual replay), so the
/// original data reaches the IPM, whose own machinery must report the
/// infeasibility with its own verified certificate.
#[test]
fn conflicting_equalities_solve_infeasible_via_ipm() {
    let a = csc(2, 3, &[(0, 0, 1.0), (1, 0, 1.0)]);
    let g = csc(1, 3, &[(0, 2, -1.0)]);
    let problem = prob(
        vec![0.0, 0.0, 0.0],
        a,
        b2(),
        g,
        vec![0.0],
        ConeSpec { l: 1, soc: vec![] },
    );
    let res = solve_socp(&problem, &opts());
    assert_eq!(res.status, SolveStatus::Infeasible, "{:?}", res.status);
    let (y, z) = res.infeas_cert.clone().expect("cert");
    for j in 0..problem.c.len() {
        let mut r = 0.0;
        for (i, v) in zip_rows(&problem.a, j) {
            r += v * y[i];
        }
        for (k, v) in zip_rows(&problem.g, j) {
            r += v * z[k];
        }
        assert!(r.abs() <= 1e-6, "Aᵀy+Gᵀz col {j}: {r}");
    }
}

fn b2() -> Vec<f64> {
    vec![1.0, 2.0]
}

/// No-op contract: problems with nothing to reduce return `None` — the
/// pipeline stays bit-identical to the pre-presolve path.
#[test]
fn nothing_to_reduce_declines() {
    let g = csc(2, 2, &[(0, 0, -1.0), (1, 1, -1.0)]);
    let problem = prob(
        vec![1.0, 1.0],
        csc(0, 2, &[]),
        vec![],
        g,
        vec![0.0, 0.0],
        ConeSpec { l: 0, soc: vec![2] },
    );
    assert!(ConicReduction::reduce(&problem).is_none());
}

/// Unbounded ray postsolve: `min -x0  s.t.  x1 = 0, ||(x1)|| <= x0` ⇒ x0
/// unbounded above. The mapped ray must satisfy `A d = 0`, `-G d ∈ K`,
/// `c·d < 0` in the *original* space (asserted independently here;
/// `solve_socp`'s own gate falls back if it did not).
#[test]
fn unbounded_ray_maps_to_original_space() {
    let a = csc(1, 2, &[(0, 1, 1.0)]);
    let g = csc(2, 2, &[(0, 0, -1.0), (1, 1, -1.0)]);
    let problem = prob(
        vec![-1.0, 0.0],
        a,
        vec![0.0],
        g,
        vec![0.0, 0.0],
        ConeSpec { l: 0, soc: vec![2] },
    );
    let res = solve_socp(&problem, &opts());
    assert_eq!(res.status, SolveStatus::Unbounded, "{:?}", res.status);
    let ray = res.primal_ray.clone().expect("ray");
    assert!((ray[1]).abs() <= 1e-6, "equality forces d1 = 0: {ray:?}");
    assert!(ray[0] > 0.0, "improving direction: {ray:?}");
    let mut s = vec![0.0; problem.m()];
    for j in 0..problem.c.len() {
        for (k, v) in zip_rows(&problem.g, j) {
            s[k] += -v * ray[j];
        }
    }
    // -G d ∈ SOC: s0 >= |s1|.
    assert!(s[0] >= s[1].abs() - 1e-9, "-Gd = {s:?}");
    assert!(
        problem.c.iter().zip(&ray).map(|(a, b)| a * b).sum::<f64>() < 0.0,
        "c·d"
    );
}

/// `tightened_tol`: σ = 1 ⇒ unchanged; σ = 0.5 ⇒ 2·tol; σ clamped into
/// [`EPS_SIGMA_CLAMP`, 1]; floor at [`CONIC_PRESOLVE_EPS_FLOOR`].
#[test]
fn tightened_tol_formula() {
    assert_eq!(tightened_tol(1e-6, 1.0), 1e-6);
    assert!((tightened_tol(1e-6, 0.5) - 5e-7).abs() <= 1e-15);
    assert!((tightened_tol(1e-6, 5.0) - 1e-6).abs() <= 1e-18);
    assert!((tightened_tol(1e-30, 1e-9) - CONIC_PRESOLVE_EPS_FLOOR).abs() <= f64::EPSILON);
}

/// `kkt_dim` = `n + p + m` (report §1.2): the formula behind the
/// `cbf_dims_table.tsv` predictions — a wrong term silently breaks the
/// size contract asserted in the integration tests.
#[test]
fn kkt_dim_formula() {
    let g = csc(
        5,
        3,
        &[
            (0, 0, -1.0),
            (1, 1, -1.0),
            (2, 2, -1.0),
            (3, 1, -1.0),
            (4, 2, -1.0),
        ],
    );
    let a = csc(2, 3, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let problem = prob(
        vec![0.0; 3],
        a,
        vec![0.0; 2],
        g,
        vec![0.0; 5],
        ConeSpec {
            l: 0,
            soc: vec![3, 2],
        },
    );
    assert_eq!(kkt_dim(&problem), 3 + 2 + 5);
}
