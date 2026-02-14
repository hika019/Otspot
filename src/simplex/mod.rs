//! Primal Simplex algorithm for Linear Programming
//! Phase I + Phase II (Revised Simplex Method)

use crate::problem::{LpProblem, SolveStatus, SolverResult};

const EPS: f64 = 1e-8;

/// Solve an LP problem: min c^T x  s.t.  Ax <= b,  x >= 0
pub fn solve(problem: &LpProblem) -> SolverResult {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    // Edge case: no variables
    if n == 0 {
        for i in 0..m {
            if problem.b[i] < -EPS {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                };
            }
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
        };
    }

    // Edge case: no constraints
    if m == 0 {
        for j in 0..n {
            if problem.c[j] < -EPS {
                return SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                };
            }
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![0.0; n],
        };
    }

    // Standard form: Ax + s = b,  x >= 0, s >= 0
    // n_total = n (original) + m (slack)
    let n_total = n + m;

    // Build dense A matrix (m x n_total) — no sign flipping yet
    let mut a_dense = vec![vec![0.0; n_total]; m];
    for col in 0..n {
        let start = problem.a.col_ptr[col];
        let end = problem.a.col_ptr[col + 1];
        for idx in start..end {
            let row = problem.a.row_ind[idx];
            a_dense[row][col] = problem.a.values[idx];
        }
    }
    // Add slack identity
    for i in 0..m {
        a_dense[i][n + i] = 1.0;
    }
    let mut b_vec = problem.b.clone();

    // Determine which rows need artificial variables
    // If b_i < 0, negate the entire row (including slack) so b_i > 0.
    // After negation, the slack for that row has coefficient -1, so we need an artificial.
    let mut needs_artificial = vec![false; m];
    let mut num_artificial = 0;
    for i in 0..m {
        if b_vec[i] < -EPS {
            // Negate entire row
            for j in 0..n_total {
                a_dense[i][j] = -a_dense[i][j];
            }
            b_vec[i] = -b_vec[i];
            needs_artificial[i] = true;
            num_artificial += 1;
        }
    }

    // Cost vector for original problem: [c; 0...0]
    let mut c_std = vec![0.0; n_total];
    for j in 0..n {
        c_std[j] = problem.c[j];
    }

    if num_artificial == 0 {
        // All b >= 0, slacks form valid BFS → go directly to Phase II
        let mut basis: Vec<usize> = (n..n_total).collect();
        let mut a_work = a_dense;
        let mut b_work = b_vec;
        return run_phase2(&mut a_work, &mut b_work, &c_std, &mut basis, n, m, n_total);
    }

    // Phase I: add artificial variables, minimize their sum
    let n_ext = n_total + num_artificial;
    let mut a_ext = vec![vec![0.0; n_ext]; m];
    let mut b_ext = b_vec.clone();
    let mut basis: Vec<usize> = vec![0; m];

    let mut art_col = n_total;
    for i in 0..m {
        // Copy existing row
        for j in 0..n_total {
            a_ext[i][j] = a_dense[i][j];
        }
        if needs_artificial[i] {
            a_ext[i][art_col] = 1.0;
            basis[i] = art_col;
            art_col += 1;
        } else {
            basis[i] = n + i; // slack as basis
        }
    }

    // Phase I cost: minimize sum of artificials
    let mut c_phase1 = vec![0.0; n_ext];
    for j in n_total..n_ext {
        c_phase1[j] = 1.0;
    }

    match simplex_core(&mut a_ext, &mut b_ext, &c_phase1, &mut basis, m, n_ext) {
        SimplexOutcome::Optimal(obj) => {
            if obj > EPS {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                };
            }
            // Feasible. Remove artificials from basis if still present.
            for i in 0..m {
                if basis[i] >= n_total {
                    let mut pivoted = false;
                    for j in 0..n_total {
                        if a_ext[i][j].abs() > EPS {
                            pivot(&mut a_ext, &mut b_ext, &mut basis, i, j, m, n_ext);
                            pivoted = true;
                            break;
                        }
                    }
                    if !pivoted {
                        // Redundant constraint row, assign a slack
                        basis[i] = n + i;
                    }
                }
            }

            // Restrict tableau to n_total columns for Phase II
            let mut a_p2 = vec![vec![0.0; n_total]; m];
            for i in 0..m {
                for j in 0..n_total {
                    a_p2[i][j] = a_ext[i][j];
                }
            }
            let mut b_p2 = b_ext;

            return run_phase2(&mut a_p2, &mut b_p2, &c_std, &mut basis, n, m, n_total);
        }
        SimplexOutcome::Unbounded => {
            return SolverResult {
                status: SolveStatus::Infeasible,
                objective: 0.0,
                solution: vec![],
            };
        }
    }
}

fn run_phase2(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    c: &[f64],
    basis: &mut Vec<usize>,
    n_orig: usize,
    m: usize,
    n_total: usize,
) -> SolverResult {
    match simplex_core(a, b, c, basis, m, n_total) {
        SimplexOutcome::Optimal(obj) => {
            let mut solution = vec![0.0; n_orig];
            for i in 0..m {
                if basis[i] < n_orig {
                    solution[basis[i]] = b[i];
                }
            }
            SolverResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution,
            }
        }
        SimplexOutcome::Unbounded => SolverResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
        },
    }
}

enum SimplexOutcome {
    Optimal(f64),
    Unbounded,
}

/// Core simplex iterations on a full tableau.
/// a, b, basis are modified in-place so the caller can read the final state.
fn simplex_core(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    c: &[f64],
    basis: &mut Vec<usize>,
    m: usize,
    n: usize,
) -> SimplexOutcome {
    let max_iter = 50 * (m + n) + 100;

    for _iter in 0..max_iter {
        // Reduced costs
        let c_b: Vec<f64> = basis.iter().map(|&j| c[j]).collect();

        let mut entering = None;
        let mut min_rc = -EPS;

        for j in 0..n {
            if basis.contains(&j) {
                continue;
            }
            let mut rc = c[j];
            for i in 0..m {
                rc -= c_b[i] * a[i][j];
            }
            if rc < min_rc {
                min_rc = rc;
                entering = Some(j);
            }
        }

        let entering_col = match entering {
            None => {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * b[i]).sum();
                return SimplexOutcome::Optimal(obj);
            }
            Some(j) => j,
        };

        // Ratio test
        let mut leaving = None;
        let mut min_ratio = f64::INFINITY;

        for i in 0..m {
            let d = a[i][entering_col];
            if d > EPS {
                let ratio = b[i] / d;
                if ratio < min_ratio - EPS {
                    min_ratio = ratio;
                    leaving = Some(i);
                } else if (ratio - min_ratio).abs() < EPS {
                    // Bland's rule: prefer smaller basis index to avoid cycling
                    if let Some(prev) = leaving {
                        if basis[i] < basis[prev] {
                            leaving = Some(i);
                        }
                    }
                }
            }
        }

        let leaving_row = match leaving {
            None => return SimplexOutcome::Unbounded,
            Some(i) => i,
        };

        pivot(a, b, basis, leaving_row, entering_col, m, n);
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * b[i]).sum();
    SimplexOutcome::Optimal(obj)
}

fn pivot(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    basis: &mut Vec<usize>,
    leaving_row: usize,
    entering_col: usize,
    m: usize,
    n: usize,
) {
    let pivot_val = a[leaving_row][entering_col];
    let inv_pivot = 1.0 / pivot_val;

    for j in 0..n {
        a[leaving_row][j] *= inv_pivot;
    }
    b[leaving_row] *= inv_pivot;

    for i in 0..m {
        if i == leaving_row {
            continue;
        }
        let factor = a[i][entering_col];
        if factor.abs() > 1e-15 {
            for j in 0..n {
                a[i][j] -= factor * a[leaving_row][j];
            }
            b[i] -= factor * b[leaving_row];
        }
    }

    basis[leaving_row] = entering_col;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new(c, a, b).unwrap()
    }

    #[test]
    fn test_basic_2var() {
        // min -x1 - x2
        // s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3, x >= 0
        // Optimal: objective = -4
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-4.0)).abs() < EPS,
            "Expected objective -4.0, got {}",
            result.objective
        );
        let x1 = result.solution[0];
        let x2 = result.solution[1];
        assert!(x1 >= -EPS && x1 <= 3.0 + EPS, "x1={}", x1);
        assert!(x2 >= -EPS && x2 <= 3.0 + EPS, "x2={}", x2);
        assert!((x1 + x2 - 4.0).abs() < EPS);
    }

    #[test]
    fn test_basic_3var() {
        // min -2x1 - 3x2 - x3
        // s.t. x1 + x2 + x3 <= 10, 2x1 + x2 <= 14, x2 + x3 <= 8
        // x >= 0
        // Known optimal: x1=2, x2=8, x3=0, obj=-28
        let lp = make_lp(
            vec![-2.0, -3.0, -1.0],
            &[0, 0, 0, 1, 1, 2, 2],
            &[0, 1, 2, 0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            3,
            3,
            vec![10.0, 14.0, 8.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let x = &result.solution;
        assert!(x[0] >= -EPS);
        assert!(x[1] >= -EPS);
        assert!(x[2] >= -EPS);
        assert!(x[0] + x[1] + x[2] <= 10.0 + EPS);
        assert!(2.0 * x[0] + x[1] <= 14.0 + EPS);
        assert!(x[1] + x[2] <= 8.0 + EPS);
        assert!(
            (result.objective - (-28.0)).abs() < EPS,
            "Expected objective -28.0, got {}",
            result.objective
        );
    }

    #[test]
    fn test_unbounded() {
        // min -x1  s.t. x1 - x2 <= 1, x >= 0
        let lp = make_lp(
            vec![-1.0, 0.0],
            &[0, 0],
            &[0, 1],
            &[1.0, -1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_infeasible() {
        // min x1  s.t. x1 <= -1, x >= 0
        let lp = make_lp(vec![1.0], &[0], &[0], &[1.0], 1, 1, vec![-1.0]);
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Infeasible);
    }

    #[test]
    fn test_degenerate_zero_vars() {
        let a = CscMatrix::new(0, 0);
        let lp = LpProblem::new(vec![], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < EPS);
    }

    #[test]
    fn test_zero_constraints_unbounded() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![-1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_zero_constraints_optimal() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < EPS);
    }
}
