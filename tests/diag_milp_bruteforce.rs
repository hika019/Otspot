//! Brute-force ground-truth correctness tests for MILP B&B (#14/#16).
//!
//! These tests do not hardcode optima: they enumerate every integer point of a
//! small bounded program to compute the *true* optimum, then assert `solve_milp`
//! returns it (status Optimal + matching objective + objective recomputable from
//! the reported solution). A 400-case randomized fuzz sweep (deterministic LCG,
//! Le/Ge/Eq, negative coeffs/bounds) guards against mis-pruning that silently
//! cuts the true optimum. Also covers maximize sign / obj_offset / mixed
//! int+continuous / fractional integer bounds.

use otspot::options::{MipConfig, SolverOptions};
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::{
    solve_milp_with_stats, solve_miqp_with_stats, CscMatrix, MilpProblem, MiqpProblem, Model,
    QpProblem,
};

const TOL: f64 = 1e-4;

fn opts() -> SolverOptions {
    let mut o = SolverOptions::default();
    o.timeout_secs = Some(20.0);
    o
}

struct Case {
    name: &'static str,
    c: Vec<f64>,
    // each constraint: (coeffs, ctype, rhs)
    cons: Vec<(Vec<f64>, ConstraintType, f64)>,
    // integer bounds (lo, hi) for every var (all integer here)
    bounds: Vec<(i64, i64)>,
}

fn build(case: &Case) -> MilpProblem {
    let n = case.c.len();
    let m = case.cons.len();
    let mut rows = vec![];
    let mut cols = vec![];
    let mut vals = vec![];
    let mut b = vec![];
    let mut ctypes = vec![];
    for (i, (coeffs, ct, rhs)) in case.cons.iter().enumerate() {
        for (j, &v) in coeffs.iter().enumerate() {
            if v != 0.0 {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        b.push(*rhs);
        ctypes.push(*ct);
    }
    let a = if m == 0 {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap()
    };
    let bounds: Vec<(f64, f64)> = case
        .bounds
        .iter()
        .map(|&(lo, hi)| (lo as f64, hi as f64))
        .collect();
    let lp = LpProblem::new_general(case.c.clone(), a, b, ctypes, bounds, None).unwrap();
    let integer_vars: Vec<usize> = (0..n).collect();
    MilpProblem::new(lp, integer_vars).unwrap()
}

/// Brute-force the true integer optimum (min c^T x). Returns None if infeasible.
fn brute_force(case: &Case) -> Option<f64> {
    let n = case.c.len();
    let mut idx: Vec<i64> = case.bounds.iter().map(|&(lo, _)| lo).collect();
    let mut best: Option<f64> = None;
    loop {
        // check feasibility
        let x: Vec<f64> = idx.iter().map(|&v| v as f64).collect();
        let mut feasible = true;
        for (coeffs, ct, rhs) in &case.cons {
            let lhs: f64 = coeffs.iter().zip(&x).map(|(a, b)| a * b).sum();
            let ok = match ct {
                ConstraintType::Le => lhs <= rhs + 1e-9,
                ConstraintType::Ge => lhs >= rhs - 1e-9,
                ConstraintType::Eq => (lhs - rhs).abs() <= 1e-9,
                _ => panic!("unexpected constraint type"),
            };
            if !ok {
                feasible = false;
                break;
            }
        }
        if feasible {
            let obj: f64 = case.c.iter().zip(&x).map(|(a, b)| a * b).sum();
            best = Some(best.map_or(obj, |b: f64| b.min(obj)));
        }
        // increment odometer
        let mut k = 0;
        loop {
            if k == n {
                return best;
            }
            idx[k] += 1;
            if idx[k] <= case.bounds[k].1 {
                break;
            }
            idx[k] = case.bounds[k].0;
            k += 1;
        }
    }
}

fn cases() -> Vec<Case> {
    use ConstraintType::*;
    vec![
        // Classic LP-rounding-fails: max 5x+4y s.t 6x+4y<=24, x+2y<=6 → int opt (4,0)=20,
        // LP opt (3,1.5)=21. minimize form: c = -(5,4).
        Case {
            name: "rounding_fails_max",
            c: vec![-5.0, -4.0],
            cons: vec![(vec![6.0, 4.0], Le, 24.0), (vec![1.0, 2.0], Le, 6.0)],
            bounds: vec![(0, 10), (0, 10)],
        },
        // Equality constraint: 2x+3y=12, maximize x+y → (6,0).
        Case {
            name: "equality_max_xy",
            c: vec![-1.0, -1.0],
            cons: vec![(vec![2.0, 3.0], Eq, 12.0)],
            bounds: vec![(0, 6), (0, 6)],
        },
        // Negative coeffs + negative bounds: min x+y s.t. x+y>=-4.
        Case {
            name: "neg_bounds_min",
            c: vec![1.0, 1.0],
            cons: vec![(vec![1.0, 1.0], Ge, -4.0)],
            bounds: vec![(-3, 3), (-3, 3)],
        },
        // 3-var knapsack-style, max value with weight cap.
        Case {
            name: "knapsack3",
            c: vec![-6.0, -10.0, -12.0],
            cons: vec![(vec![1.0, 2.0, 3.0], Le, 5.0)],
            bounds: vec![(0, 1), (0, 1), (0, 1)],
        },
        // Multi-level branching: tight feasible region.
        Case {
            name: "deep_branch",
            c: vec![-3.0, -2.0, -1.0],
            cons: vec![
                (vec![2.0, 1.0, 1.0], Le, 7.0),
                (vec![1.0, 3.0, 1.0], Le, 9.0),
                (vec![1.0, 1.0, 4.0], Le, 8.0),
            ],
            bounds: vec![(0, 5), (0, 5), (0, 5)],
        },
        // Mixed sign objective, both directions matter.
        Case {
            name: "mixed_sign_obj",
            c: vec![2.0, -3.0],
            cons: vec![(vec![1.0, 1.0], Le, 4.0), (vec![1.0, -1.0], Ge, -2.0)],
            bounds: vec![(0, 4), (0, 4)],
        },
        // Minimize with Ge constraints (covering, integer optimum > 0).
        Case {
            name: "covering_min",
            c: vec![3.0, 5.0],
            cons: vec![(vec![1.0, 2.0], Ge, 5.0), (vec![3.0, 1.0], Ge, 6.0)],
            bounds: vec![(0, 5), (0, 5)],
        },
        // Negative-only objective with negative lower bounds (push to corner).
        Case {
            name: "neg_corner",
            c: vec![1.0, 2.0],
            cons: vec![(vec![1.0, 1.0], Ge, -1.0)],
            bounds: vec![(-2, 2), (-2, 2)],
        },
    ]
}

#[test]
fn brute_force_matches_solver_all_cases() {
    for case in cases() {
        let truth = brute_force(&case);
        let problem = build(&case);
        let (r, stats) = solve_milp_with_stats(&problem, &opts(), &MipConfig::default());
        match truth {
            Some(opt) => {
                assert_eq!(
                    r.status,
                    SolveStatus::Optimal,
                    "case {} should be Optimal",
                    case.name
                );
                assert!(
                    (r.objective - opt).abs() < TOL,
                    "case {}: solver obj {} != brute-force {}",
                    case.name,
                    r.objective,
                    opt
                );
                // returned solution must be integral and feasible at the reported objective
                for (j, v) in r.solution.iter().enumerate() {
                    assert!(
                        (v - v.round()).abs() < 1e-5,
                        "case {}: var{} not integral: {}",
                        case.name,
                        j,
                        v
                    );
                }
                // recompute objective from rounded solution to detect any obj/solution mismatch
                let recomputed: f64 = case
                    .c
                    .iter()
                    .zip(&r.solution)
                    .map(|(a, b)| a * b.round())
                    .sum();
                assert!(
                    (recomputed - opt).abs() < TOL,
                    "case {}: recomputed obj {} from solution != truth {}",
                    case.name,
                    recomputed,
                    opt
                );
                println!(
                    "[{}] opt={} nodes={} pruned={} inc_updates={}",
                    case.name, opt, stats.nodes_processed, stats.pruned, stats.incumbent_updates
                );
            }
            None => {
                assert_eq!(
                    r.status,
                    SolveStatus::Infeasible,
                    "case {} should be Infeasible",
                    case.name
                );
            }
        }
    }
}

/// Deterministic LCG so the fuzz sweep is reproducible.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1) as u64;
        lo + (self.next() % span) as i64
    }
}

/// Randomized sweep: generate small bounded all-integer programs, brute-force the
/// true optimum, and assert solve_milp matches. Directly attacks "mis-pruning cuts
/// the true optimum" (silent-wrong). Many seeds, all pure-integer so brute force is exact.
#[test]
fn fuzz_sweep_solver_matches_brute_force() {
    use ConstraintType::*;
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let mut feasible = 0;
    let mut infeasible = 0;
    let cfg = MipConfig::default();
    for trial in 0..400 {
        let n = rng.range(2, 3) as usize;
        let m = rng.range(1, 3) as usize;
        let c: Vec<f64> = (0..n).map(|_| rng.range(-5, 5) as f64).collect();
        // bounds kept small so brute force stays cheap (<= 7^3 points).
        let bounds: Vec<(i64, i64)> = (0..n)
            .map(|_| {
                let lo = rng.range(-2, 1);
                let hi = lo + rng.range(1, 4);
                (lo, hi)
            })
            .collect();
        let cons: Vec<(Vec<f64>, ConstraintType, f64)> = (0..m)
            .map(|_| {
                let coeffs: Vec<f64> = (0..n).map(|_| rng.range(-3, 3) as f64).collect();
                let ct = match rng.range(0, 2) {
                    0 => Le,
                    1 => Ge,
                    _ => Eq,
                };
                let rhs = rng.range(-6, 6) as f64;
                (coeffs, ct, rhs)
            })
            .collect();
        let case = Case {
            name: "fuzz",
            c,
            cons,
            bounds,
        };
        let truth = brute_force(&case);
        let problem = build(&case);
        let (r, _stats) = solve_milp_with_stats(&problem, &opts(), &cfg);
        match truth {
            Some(opt) => {
                feasible += 1;
                assert_eq!(
                    r.status,
                    SolveStatus::Optimal,
                    "trial {trial}: brute-force feasible (opt={opt}) but solver status {:?}",
                    r.status
                );
                assert!(
                    (r.objective - opt).abs() < 1e-3,
                    "trial {trial}: solver obj {} != brute-force optimum {}",
                    r.objective,
                    opt
                );
                // The reported solution must reproduce the objective (no obj/solution drift).
                let recomputed: f64 = case
                    .c
                    .iter()
                    .zip(&r.solution)
                    .map(|(a, b)| a * b.round())
                    .sum();
                assert!(
                    (recomputed - opt).abs() < 1e-3,
                    "trial {trial}: solution recompute {} != optimum {}",
                    recomputed,
                    opt
                );
            }
            None => {
                infeasible += 1;
                assert_eq!(
                    r.status,
                    SolveStatus::Infeasible,
                    "trial {trial}: brute-force infeasible but solver status {:?}",
                    r.status
                );
            }
        }
    }
    println!("fuzz sweep: {feasible} feasible, {infeasible} infeasible — all matched brute force");
    assert!(
        feasible > 50,
        "sweep should hit many feasible cases, got {feasible}"
    );
    assert!(
        infeasible > 5,
        "sweep should hit some infeasible cases, got {infeasible}"
    );
}

// --- Model API direct checks (maximize sign / mixed / guards) ---

#[test]
fn model_maximize_negative_optimum() {
    // maximize -(x) i.e. minimize x with x>=2 integer in [0,5] → x=2, obj=-2.
    let mut m = Model::new("neg_max");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.add_constraint((1.0 * x).geq(2.0));
    m.maximize(-1.0 * x);
    let r = m.solve().unwrap();
    assert!(
        (r.objective() - (-2.0)).abs() < TOL,
        "obj={}",
        r.objective()
    );
    assert!((r[x] - 2.0).abs() < TOL, "x={}", r[x]);
}

#[test]
fn model_mixed_int_continuous_truth() {
    // maximize x + 2y: x int [0,5], y cont [0,2], x + y <= 3.7.
    // For fixed x, best y = min(2, 3.7-x): x=2,y=1.7 → 2+3.4=5.4 is the true optimum
    // (x=1,y=2→5; x=3,y=0.7→4.4). Verifies branching keeps continuous var free.
    let mut m = Model::new("mixed2");
    let x = m.add_int_var("x", 0.0, 5.0);
    let y = m.add_var("y", 0.0, 2.0);
    m.add_constraint((x + y).leq(3.7));
    m.maximize(x + 2.0 * y);
    let r = m.solve().unwrap();
    assert!(
        (r.objective() - 5.4).abs() < TOL,
        "obj={} (expected 5.4)",
        r.objective()
    );
    assert!(
        (r[x].round() - r[x]).abs() < TOL,
        "x must be integral: {}",
        r[x]
    );
    assert!((r[x] - 2.0).abs() < TOL, "x expected 2, got {}", r[x]);
}

#[test]
fn model_obj_offset_with_integer() {
    // minimize x with obj_offset 10, x>=3 integer → x=3, obj = 3 + 10 = 13.
    // Validates obj_offset is applied on the MIP path (set_obj_offset, not a bare constant).
    let mut m = Model::new("offset");
    let x = m.add_int_var("x", 0.0, 5.0);
    m.add_constraint((1.0 * x).geq(3.0));
    m.minimize(x);
    m.set_obj_offset(10.0);
    let r = m.solve().unwrap();
    assert!(
        (r.objective() - 13.0).abs() < TOL,
        "obj={} (expected 13)",
        r.objective()
    );

    // maximize with offset: max x (x<=4 int) + offset 100 → x=4, obj = 4 + 100 = 104.
    let mut m2 = Model::new("offset_max");
    let x2 = m2.add_int_var("x", 0.0, 4.0);
    m2.maximize(x2);
    m2.set_obj_offset(100.0);
    let r2 = m2.solve().unwrap();
    assert!(
        (r2.objective() - 104.0).abs() < TOL,
        "obj={} (expected 104)",
        r2.objective()
    );
}

#[test]
fn model_fractional_integer_bounds_infeasible() {
    // x integer in [2.3, 2.7] → no integer in range → infeasible.
    let mut m = Model::new("frac_bounds");
    let x = m.add_int_var("x", 2.3, 2.7);
    m.minimize(x);
    let err = m.solve().unwrap_err();
    println!("frac bounds err = {err:?}");
    assert!(
        format!("{err:?}").contains("Infeasible"),
        "expected infeasible, got {err:?}"
    );
}

// --- Brute-force ground truth for convex MIQP (diagonal PSD Q) -----------------
//
// Objective 1/2 x'Qx + c'x with diagonal Q (q_i >= 0) is separable, so the true
// integer optimum is exact to enumerate. A randomized sweep guards convex-MIQP
// B&B against mis-pruning the true optimum (the QP relaxation bound is ε-tight,
// so a small tolerance is allowed on the objective match).

/// A diagonal-Q MIQP test instance over all-integer variables.
struct MiqpCase {
    diag: Vec<f64>, // Q diagonal (q_i), must be >= 0 for convexity
    c: Vec<f64>,
    cons: Vec<(Vec<f64>, ConstraintType, f64)>,
    bounds: Vec<(i64, i64)>,
}

fn build_miqp(case: &MiqpCase) -> MiqpProblem {
    let n = case.diag.len();
    let m = case.cons.len();
    let qidx: Vec<usize> = (0..n).collect();
    let q = CscMatrix::from_triplets(&qidx, &qidx, &case.diag, n, n).unwrap();
    let (mut rows, mut cols, mut vals, mut b, mut ctypes) =
        (vec![], vec![], vec![], vec![], vec![]);
    for (i, (coeffs, ct, rhs)) in case.cons.iter().enumerate() {
        for (j, &v) in coeffs.iter().enumerate() {
            if v != 0.0 {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        b.push(*rhs);
        ctypes.push(*ct);
    }
    let a = if m == 0 {
        CscMatrix::new(0, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap()
    };
    let bounds: Vec<(f64, f64)> = case
        .bounds
        .iter()
        .map(|&(lo, hi)| (lo as f64, hi as f64))
        .collect();
    let qp = QpProblem::new(q, case.c.clone(), a, b, bounds, ctypes).unwrap();
    MiqpProblem::new(qp, (0..n).collect()).unwrap()
}

/// True integer optimum of 1/2 x'Qx + c'x (diagonal Q). None = infeasible.
fn brute_force_miqp(case: &MiqpCase) -> Option<f64> {
    let n = case.diag.len();
    let mut idx: Vec<i64> = case.bounds.iter().map(|&(lo, _)| lo).collect();
    let mut best: Option<f64> = None;
    loop {
        let x: Vec<f64> = idx.iter().map(|&v| v as f64).collect();
        let feasible = case.cons.iter().all(|(coeffs, ct, rhs)| {
            let lhs: f64 = coeffs.iter().zip(&x).map(|(a, b)| a * b).sum();
            match ct {
                ConstraintType::Le => lhs <= rhs + 1e-9,
                ConstraintType::Ge => lhs >= rhs - 1e-9,
                ConstraintType::Eq => (lhs - rhs).abs() <= 1e-9,
                _ => panic!("unexpected constraint type"),
            }
        });
        if feasible {
            let obj: f64 = (0..n)
                .map(|i| 0.5 * case.diag[i] * x[i] * x[i] + case.c[i] * x[i])
                .sum();
            best = Some(best.map_or(obj, |b: f64| b.min(obj)));
        }
        let mut k = 0;
        loop {
            if k == n {
                return best;
            }
            idx[k] += 1;
            if idx[k] <= case.bounds[k].1 {
                break;
            }
            idx[k] = case.bounds[k].0;
            k += 1;
        }
    }
}

#[test]
fn brute_force_matches_miqp_solver() {
    use ConstraintType::*;
    let cases = [
        // min (x-2)^2 + (y-2)^2 = x^2-4x+4 + y^2-4y+4, no constraint → (2,2), obj 0.
        MiqpCase {
            diag: vec![2.0, 2.0],
            c: vec![-4.0, -4.0],
            cons: vec![],
            bounds: vec![(0, 4), (0, 4)],
        },
        // separable with a covering constraint forcing off-corner integers.
        MiqpCase {
            diag: vec![2.0, 2.0],
            c: vec![0.0, 0.0],
            cons: vec![(vec![1.0, 1.0], Ge, 3.0)],
            bounds: vec![(0, 5), (0, 5)],
        },
        // mixed linear+quadratic, equality constraint.
        MiqpCase {
            diag: vec![1.0, 3.0],
            c: vec![-2.0, 1.0],
            cons: vec![(vec![1.0, 1.0], Eq, 4.0)],
            bounds: vec![(0, 4), (0, 4)],
        },
        // 3-var, two constraints.
        MiqpCase {
            diag: vec![2.0, 2.0, 2.0],
            c: vec![-1.0, -2.0, -3.0],
            cons: vec![
                (vec![1.0, 1.0, 1.0], Le, 4.0),
                (vec![1.0, 0.0, 1.0], Ge, 1.0),
            ],
            bounds: vec![(0, 3), (0, 3), (0, 3)],
        },
    ];
    for (idx, case) in cases.iter().enumerate() {
        // (array, not vec!, since we only iterate)
        let truth = brute_force_miqp(case);
        let problem = build_miqp(case);
        let (r, stats) = solve_miqp_with_stats(&problem, &opts(), &MipConfig::default());
        match truth {
            Some(opt) => {
                assert_eq!(
                    r.status,
                    SolveStatus::Optimal,
                    "case {idx} should be Optimal"
                );
                assert!(
                    (r.objective - opt).abs() < 1e-3,
                    "case {idx}: solver obj {} != brute-force {}",
                    r.objective,
                    opt
                );
                println!(
                    "[miqp {idx}] opt={opt} nodes={} pruned={} inc={}",
                    stats.nodes_processed, stats.pruned, stats.incumbent_updates
                );
            }
            None => assert_eq!(
                r.status,
                SolveStatus::Infeasible,
                "case {idx} should be Infeasible"
            ),
        }
    }
}

#[test]
fn fuzz_sweep_miqp_matches_brute_force() {
    use ConstraintType::*;
    let mut rng = Lcg(0x1234_5678_9ABC_DEF0);
    let mut feasible = 0;
    let cfg = MipConfig::default();
    for trial in 0..120 {
        let n = rng.range(2, 3) as usize;
        let m = rng.range(0, 2) as usize;
        // q_i in {1,2,3,4} → strictly convex, well-conditioned diagonal Q.
        let diag: Vec<f64> = (0..n).map(|_| rng.range(1, 4) as f64).collect();
        let c: Vec<f64> = (0..n).map(|_| rng.range(-4, 4) as f64).collect();
        let bounds: Vec<(i64, i64)> = (0..n)
            .map(|_| {
                let lo = rng.range(-1, 1);
                (lo, lo + rng.range(1, 3))
            })
            .collect();
        let cons: Vec<(Vec<f64>, ConstraintType, f64)> = (0..m)
            .map(|_| {
                let coeffs: Vec<f64> = (0..n).map(|_| rng.range(-2, 2) as f64).collect();
                let ct = match rng.range(0, 2) {
                    0 => Le,
                    1 => Ge,
                    _ => Eq,
                };
                (coeffs, ct, rng.range(-3, 4) as f64)
            })
            .collect();
        let case = MiqpCase {
            diag,
            c,
            cons,
            bounds,
        };
        let truth = brute_force_miqp(&case);
        let (r, _stats) = solve_miqp_with_stats(&build_miqp(&case), &opts(), &cfg);
        match truth {
            Some(opt) => {
                feasible += 1;
                assert_eq!(
                    r.status,
                    SolveStatus::Optimal,
                    "trial {trial}: brute-force feasible (opt={opt}) but solver {:?}",
                    r.status
                );
                assert!(
                    (r.objective - opt).abs() < 1e-3,
                    "trial {trial}: solver obj {} != brute-force optimum {}",
                    r.objective,
                    opt
                );
            }
            None => assert_eq!(
                r.status,
                SolveStatus::Infeasible,
                "trial {trial}: brute-force infeasible but solver {:?}",
                r.status
            ),
        }
    }
    println!("miqp fuzz: {feasible} feasible cases — all matched brute force");
    assert!(
        feasible > 30,
        "sweep should hit many feasible cases, got {feasible}"
    );
}
