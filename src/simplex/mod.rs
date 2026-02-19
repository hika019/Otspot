//! Primal Simplex algorithm for Linear Programming
//! Phase I + Phase II (Revised Simplex Method)

use crate::basis::{BasisManager, LuBasis};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};

const EPS: f64 = 1e-8;

/// Solve an LP problem using Revised Simplex with LU factorization
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
        let mut x = vec![0.0; n];
        let mut obj = 0.0;
        for j in 0..n {
            if problem.c[j] < -EPS {
                // BUG-simplex-001修正: ubが有限なら最大値(ub)に設定、無限ならUnbounded
                let ub = problem.bounds[j].1;
                if ub.is_infinite() {
                    return SolverResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        solution: vec![],
                    };
                }
                x[j] = ub;
            }
            obj += problem.c[j] * x[j];
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: obj,
            solution: x,
        };
    }

    let sf = build_standard_form(problem);
    two_phase_simplex(&sf)
}

// --- Data structures ---

struct OrigVarInfo {
    offset: f64,
    new_vars: Vec<(usize, f64)>,
}

struct StandardForm {
    a: CscMatrix,
    b: Vec<f64>,
    c: Vec<f64>,
    m: usize,
    n_shifted: usize,
    n_total: usize,
    initial_basis: Vec<usize>,
    needs_artificial: Vec<bool>,
    num_artificial: usize,
    obj_offset: f64,
    n_orig: usize,
    orig_var_info: Vec<OrigVarInfo>,
}

enum SimplexOutcome {
    Optimal(f64),
    Unbounded,
}

// --- Standard form construction ---

fn build_standard_form(problem: &LpProblem) -> StandardForm {
    let n_orig = problem.num_vars;
    let m_orig = problem.num_constraints;

    // 1. Variable transformations
    let mut orig_var_info: Vec<OrigVarInfo> = Vec::with_capacity(n_orig);
    let mut n_shifted = 0usize;
    let mut obj_offset = 0.0f64;
    let mut new_c: Vec<f64> = Vec::new();

    for j in 0..n_orig {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            obj_offset += problem.c[j] * lb;
            orig_var_info.push(OrigVarInfo {
                offset: lb,
                new_vars: vec![(idx, 1.0)],
            });
        } else if ub.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            obj_offset += problem.c[j] * ub;
            orig_var_info.push(OrigVarInfo {
                offset: ub,
                new_vars: vec![(idx, -1.0)],
            });
        } else {
            let idx_plus = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            let idx_minus = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            orig_var_info.push(OrigVarInfo {
                offset: 0.0,
                new_vars: vec![(idx_plus, 1.0), (idx_minus, -1.0)],
            });
        }
    }

    // 2. Upper bound constraints
    let mut ub_constraints: Vec<(usize, f64)> = Vec::new();
    for j in 0..n_orig {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && ub.is_finite() {
            let effective_ub = ub - lb;
            let new_idx = orig_var_info[j].new_vars[0].0;
            ub_constraints.push((new_idx, effective_ub));
        }
    }
    let n_ub = ub_constraints.len();
    let m_ext = m_orig + n_ub;

    // 3. Compute adjusted b
    let mut b = problem.b.clone();
    for j in 0..n_orig {
        let offset = orig_var_info[j].offset;
        if offset.abs() > 1e-15 {
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    b[row] -= vals[k] * offset;
                }
            }
        }
    }
    for &(_, ub_val) in &ub_constraints {
        b.push(ub_val);
    }

    // 4. Constraint types
    let mut ctypes: Vec<ConstraintType> = problem.constraint_types.clone();
    for _ in 0..n_ub {
        ctypes.push(ConstraintType::Le);
    }

    // 5. Row negation and slack setup
    let mut row_negated = vec![false; m_ext];
    let mut slack_col_idx: Vec<Option<usize>> = Vec::with_capacity(m_ext);
    let mut n_slack = 0usize;
    let mut slack_coeff = vec![0.0f64; m_ext];

    for i in 0..m_ext {
        match ctypes[i] {
            ConstraintType::Le => {
                if b[i] < -EPS {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = -1.0;
                } else {
                    slack_coeff[i] = 1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Ge => {
                if b[i] < -EPS {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = 1.0; // -1 negated
                } else {
                    slack_coeff[i] = -1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Eq => {
                if b[i] < -EPS {
                    row_negated[i] = true;
                    b[i] = -b[i];
                }
                slack_col_idx.push(None);
            }
        }
    }

    let n_total = n_shifted + n_slack;

    // 6. Initial basis and artificial detection
    let mut initial_basis = vec![0usize; m_ext];
    let mut needs_artificial = vec![false; m_ext];
    let mut num_artificial = 0usize;

    for i in 0..m_ext {
        match slack_col_idx[i] {
            Some(s_idx) => {
                let col = n_shifted + s_idx;
                if slack_coeff[i] > 0.0 {
                    initial_basis[i] = col;
                } else {
                    needs_artificial[i] = true;
                    num_artificial += 1;
                    initial_basis[i] = col; // placeholder
                }
            }
            None => {
                needs_artificial[i] = true;
                num_artificial += 1;
            }
        }
    }

    // 7. Build CscMatrix from triplets
    let mut trip_rows = Vec::new();
    let mut trip_cols = Vec::new();
    let mut trip_vals = Vec::new();

    // Original variable columns (transformed)
    for j in 0..n_orig {
        if let Ok((a_rows, a_vals)) = problem.a.get_column(j) {
            for (k, &row) in a_rows.iter().enumerate() {
                let val = a_vals[k];
                let sign = if row_negated[row] { -1.0 } else { 1.0 };
                for &(new_col, coeff) in &orig_var_info[j].new_vars {
                    let actual_val = sign * val * coeff;
                    if actual_val.abs() > 1e-15 {
                        trip_rows.push(row);
                        trip_cols.push(new_col);
                        trip_vals.push(actual_val);
                    }
                }
            }
        }
    }

    // Upper bound constraint rows
    for (ub_idx, &(new_var_idx, _)) in ub_constraints.iter().enumerate() {
        let row = m_orig + ub_idx;
        trip_rows.push(row);
        trip_cols.push(new_var_idx);
        trip_vals.push(1.0);
    }

    // Slack/surplus columns
    for i in 0..m_ext {
        if let Some(s_idx) = slack_col_idx[i] {
            let col = n_shifted + s_idx;
            trip_rows.push(i);
            trip_cols.push(col);
            trip_vals.push(slack_coeff[i]);
        }
    }

    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_ext, n_total).unwrap();

    // Cost vector for Phase II
    let mut c_ext = vec![0.0; n_total];
    for k in 0..n_shifted {
        c_ext[k] = new_c[k];
    }

    StandardForm {
        a,
        b,
        c: c_ext,
        m: m_ext,
        n_shifted,
        n_total,
        initial_basis,
        needs_artificial,
        num_artificial,
        obj_offset,
        n_orig,
        orig_var_info,
    }
}

// --- Two-phase simplex ---

fn two_phase_simplex(sf: &StandardForm) -> SolverResult {
    let m = sf.m;

    if sf.num_artificial == 0 {
        // Direct Phase II
        let mut basis = sf.initial_basis.clone();
        let mut x_b = sf.b.clone();

        match revised_simplex_core(&sf.a, &mut x_b, &sf.c, &mut basis, m, sf.n_total, sf.n_total)
        {
            SimplexOutcome::Optimal(obj) => {
                let solution = extract_solution(sf, &basis, &x_b);
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: obj + sf.obj_offset,
                    solution,
                }
            }
            SimplexOutcome::Unbounded => SolverResult {
                status: SolveStatus::Unbounded,
                objective: f64::NEG_INFINITY,
                solution: vec![],
            },
        }
    } else {
        // Phase I: build extended matrix with artificials
        let n_ext = sf.n_total + sf.num_artificial;

        let mut trip_rows = Vec::new();
        let mut trip_cols = Vec::new();
        let mut trip_vals = Vec::new();

        // Copy existing matrix
        for j in 0..sf.a.ncols {
            if let Ok((r, v)) = sf.a.get_column(j) {
                for (k, &row) in r.iter().enumerate() {
                    trip_rows.push(row);
                    trip_cols.push(j);
                    trip_vals.push(v[k]);
                }
            }
        }

        // Add artificial columns
        let mut basis = sf.initial_basis.clone();
        let mut art_col = sf.n_total;
        for i in 0..m {
            if sf.needs_artificial[i] {
                trip_rows.push(i);
                trip_cols.push(art_col);
                trip_vals.push(1.0);
                basis[i] = art_col;
                art_col += 1;
            }
        }

        let a_ext =
            CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_ext).unwrap();

        // Phase I cost
        let mut c_phase1 = vec![0.0; n_ext];
        for j in sf.n_total..n_ext {
            c_phase1[j] = 1.0;
        }

        let mut x_b = sf.b.clone();

        match revised_simplex_core(&a_ext, &mut x_b, &c_phase1, &mut basis, m, n_ext, n_ext) {
            SimplexOutcome::Optimal(obj) => {
                if obj > EPS {
                    return SolverResult {
                        status: SolveStatus::Infeasible,
                        objective: 0.0,
                        solution: vec![],
                    };
                }

                // Phase II: restrict pricing to non-artificial columns
                let mut c_phase2 = vec![0.0; n_ext];
                for k in 0..sf.n_total {
                    c_phase2[k] = sf.c[k];
                }

                match revised_simplex_core(
                    &a_ext,
                    &mut x_b,
                    &c_phase2,
                    &mut basis,
                    m,
                    n_ext,
                    sf.n_total,
                ) {
                    SimplexOutcome::Optimal(obj2) => {
                        let solution = extract_solution(sf, &basis, &x_b);
                        SolverResult {
                            status: SolveStatus::Optimal,
                            objective: obj2 + sf.obj_offset,
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
            SimplexOutcome::Unbounded => SolverResult {
                status: SolveStatus::Infeasible,
                objective: 0.0,
                solution: vec![],
            },
        }
    }
}

fn extract_solution(sf: &StandardForm, basis: &[usize], x_b: &[f64]) -> Vec<f64> {
    let mut x_new = vec![0.0; sf.n_shifted];
    for i in 0..sf.m {
        if basis[i] < sf.n_shifted {
            x_new[basis[i]] = x_b[i];
        }
    }

    let mut solution = vec![0.0; sf.n_orig];
    for j in 0..sf.n_orig {
        let info = &sf.orig_var_info[j];
        solution[j] = info.offset;
        for &(new_idx, coeff) in &info.new_vars {
            solution[j] += coeff * x_new[new_idx];
        }
    }
    solution
}

// --- Core Revised Simplex ---

fn revised_simplex_core(
    a: &CscMatrix,
    x_b: &mut Vec<f64>,
    c: &[f64],
    basis: &mut Vec<usize>,
    m: usize,
    n_cols: usize,
    n_price: usize,
) -> SimplexOutcome {
    let mut basis_mgr = match LuBasis::new(a, basis) {
        Ok(bm) => bm,
        Err(_) => {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Optimal(obj);
        }
    };

    let max_iter = 100 * (m + n_cols) + 1000;

    let mut is_basic = vec![false; n_cols];
    for &b in basis.iter() {
        is_basic[b] = true;
    }

    for _iter in 0..max_iter {
        // 1. Dual variables: y = BTRAN(c_B)
        let c_b: Vec<f64> = (0..m).map(|i| c[basis[i]]).collect();
        let mut y_sv = SparseVec::from_dense(&c_b);
        basis_mgr.btran(&mut y_sv);
        let y = y_sv.to_dense();

        // 2. Pricing: find most negative reduced cost
        let mut entering = None;
        let mut min_rc = -EPS;

        for j in 0..n_price {
            if is_basic[j] {
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut ya = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                ya += y[row] * vals[k];
            }
            let rc = c[j] - ya;
            if rc < min_rc {
                min_rc = rc;
                entering = Some(j);
            }
        }

        let entering_col = match entering {
            None => {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Optimal(obj);
            }
            Some(j) => j,
        };

        // 3. FTRAN: pivot column d = B^{-1} * a_entering
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let mut d_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut d_sv);
        let d = d_sv.to_dense();

        // 4. Ratio test (Bland's rule for ties)
        let mut leaving = None;
        let mut min_ratio = f64::INFINITY;

        for i in 0..m {
            if d[i] > EPS {
                let ratio = x_b[i] / d[i];
                if ratio < min_ratio - EPS {
                    min_ratio = ratio;
                    leaving = Some(i);
                } else if (ratio - min_ratio).abs() < EPS {
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

        // 5. Update x_b
        let step = x_b[leaving_row] / d[leaving_row];
        for i in 0..m {
            x_b[i] -= d[i] * step;
        }
        x_b[leaving_row] = step;

        // Clamp near-zero to prevent drift
        for val in x_b.iter_mut() {
            if val.abs() < 1e-14 {
                *val = 0.0;
            }
        }

        // 6. Update basis tracking
        is_basic[basis[leaving_row]] = false;
        is_basic[entering_col] = true;

        // 7. Update basis manager
        basis_mgr.update(entering_col, leaving_row, &d_sv);
        basis[leaving_row] = entering_col;

        // 8. Refactor if needed
        basis_mgr.refactor_if_needed(a, basis);
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    SimplexOutcome::Optimal(obj)
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

    #[test]
    fn test_bug_simplex_001_finite_ub() {
        // BUG-simplex-001修正確認: m=0, maximize x with lb=0, ub=3
        // 修正前: Unbounded誤判定
        // 修正後: x=3, obj=3 (maximize) または obj=-3 (minimize として内部処理)
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new_general(
            vec![-1.0], // minimize -x (= maximize x)
            a,
            vec![],
            vec![],
            vec![(0.0, 3.0)], // lb=0, ub=3
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.solution[0] - 3.0).abs() < EPS,
            "Expected x=3, got {}",
            result.solution[0]
        );
        assert!(
            (result.objective - (-3.0)).abs() < EPS,
            "Expected obj=-3, got {}",
            result.objective
        );
    }
}
