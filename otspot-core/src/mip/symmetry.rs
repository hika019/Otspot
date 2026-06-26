//! Static structural symmetry breaking for MILP.
//!
//! Many MILP models contain interchangeable binary variables: columns identical
//! in the objective and every constraint. Permuting them maps any feasible point
//! to another with the same objective, so branch-and-bound can waste effort on
//! equivalent subtrees.
//!
//! We detect orbits of interchangeable binaries and add a static lex-leader
//! ordering `x_i >= x_{i+1}` within each orbit, forcing the canonical descending
//! assignment `1…1 0…0`.
//!
//! Correctness relies on exact grouping: binaries share an orbit only when their
//! objective coefficients are bit-for-bit equal and their constraint columns have
//! identical `(row, coefficient)` entries. Swapping such columns is an exact
//! automorphism, so sorting any feasible orbit assignment in descending order
//! preserves feasibility and objective value. At least one optimal representative
//! therefore remains after adding the ordering rows.

use super::problem::MilpProblem;
use crate::problem::{ConstraintType, LpProblem};
use crate::sparse::CscMatrix;
use crate::tolerances::ZERO_TOL;
use std::collections::HashMap;

/// Canonical key identifying a binary variable's column up to interchange.
///
/// `obj` is the objective coefficient and `col` the sorted list of non-zero
/// `(row, coefficient)` constraint entries, both stored as raw IEEE-754 bits so
/// that grouping requires *exact* equality (a relaxed/rounded comparison could
/// group near-identical-but-distinct columns and cut a genuine optimum).
#[derive(PartialEq, Eq, Hash)]
struct ColumnSignature {
    obj: u64,
    col: Vec<(usize, u64)>,
}

/// IEEE-754 bits of `v`, collapsing `-0.0` to `+0.0` so signed zero never
/// splits an otherwise-identical signature.
fn coef_bits(v: f64) -> u64 {
    (if v == 0.0 { 0.0 } else { v }).to_bits()
}

/// Build the interchange signature of binary variable `j` from `a` and `c`.
fn column_signature(a: &CscMatrix, c: &[f64], j: usize) -> ColumnSignature {
    let mut col = Vec::new();
    // CSC rows within a column are stored ascending, so the list is canonical.
    for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
        let v = a.values()[k];
        if v != 0.0 {
            col.push((a.row_ind()[k], coef_bits(v)));
        }
    }
    ColumnSignature {
        obj: coef_bits(c[j]),
        col,
    }
}

/// `true` when variable `j` is a `{0,1}` binary under `bounds`.
fn is_binary(j: usize, bounds: &[(f64, f64)]) -> bool {
    let (lb, ub) = bounds[j];
    (lb - 0.0).abs() < ZERO_TOL && (ub - 1.0).abs() < ZERO_TOL
}

/// Group interchangeable binary variables of `milp` into orbits.
///
/// Returns one ascending-index list per orbit of size `>= 2`, the orbits
/// themselves ordered by their smallest member for determinism. Variables fixed
/// by presolve (`lb == ub`) are not `{0,1}` binary and are therefore excluded.
fn binary_orbits(milp: &MilpProblem) -> Vec<Vec<usize>> {
    let lp = &milp.lp;
    let mut groups: HashMap<ColumnSignature, Vec<usize>> = HashMap::new();
    for &j in &milp.integer_vars {
        if j < lp.num_vars && is_binary(j, &lp.bounds) {
            groups
                .entry(column_signature(&lp.a, &lp.c, j))
                .or_default()
                .push(j);
        }
    }
    // `integer_vars` is sorted, so each group's indices are already ascending.
    let mut orbits: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() >= 2).collect();
    orbits.sort_unstable_by_key(|g| g[0]);
    orbits
}

/// Append lex-leader ordering rows `x_i - x_{i+1} >= 0` for every orbit.
///
/// Returns `milp` augmented with the symmetry-breaking constraints, or an
/// unchanged clone when no orbit of interchangeable binaries exists.
pub(crate) fn break_symmetry(milp: &MilpProblem) -> MilpProblem {
    let orbits = binary_orbits(milp);
    if orbits.is_empty() {
        return milp.clone();
    }

    let lp = &milp.lp;
    let m_old = lp.num_constraints;
    let n = lp.num_vars;

    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for col in 0..lp.a.ncols() {
        let (rs, vs) = lp.a.get_column(col).expect("valid column");
        for (&r, &v) in rs.iter().zip(vs) {
            trip_rows.push(r);
            trip_cols.push(col);
            trip_vals.push(v);
        }
    }

    let mut b = lp.b.clone();
    let mut ctypes = lp.constraint_types.clone();
    let mut row = m_old;
    for orbit in &orbits {
        for pair in orbit.windows(2) {
            // x_{pair[0]} - x_{pair[1]} >= 0  ⇔  x_{pair[0]} >= x_{pair[1]}
            trip_rows.push(row);
            trip_cols.push(pair[0]);
            trip_vals.push(1.0);
            trip_rows.push(row);
            trip_cols.push(pair[1]);
            trip_vals.push(-1.0);
            b.push(0.0);
            ctypes.push(ConstraintType::Ge);
            row += 1;
        }
    }

    let m_new = row;
    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_new, n)
        .expect("symmetry-augmented A is well-formed");
    let mut out_lp = LpProblem::new_general(
        lp.c.clone(),
        a,
        b,
        ctypes,
        lp.bounds.clone(),
        lp.name.clone(),
    )
    .expect("symmetry-augmented LP is valid");
    out_lp.obj_offset = lp.obj_offset;

    MilpProblem {
        lp: out_lp,
        integer_vars: milp.integer_vars.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// Two identical binary columns with equal objective form one orbit.
    #[test]
    fn detects_identical_binary_columns() {
        // 1 constraint: x0 + x1 <= 1, obj c = [1, 1], both binary.
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1]).unwrap();
        let orbits = binary_orbits(&milp);
        assert_eq!(orbits, vec![vec![0, 1]]);
    }

    /// Differing objective coefficients are NOT interchangeable.
    ///
    /// Sentinel: dropping the `obj` field from the signature would merge these
    /// columns and `x0 >= x1` could cut the true optimum.
    #[test]
    fn distinct_objective_is_not_symmetric() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 2.0],
            a,
            vec![1.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1]).unwrap();
        assert!(binary_orbits(&milp).is_empty());
    }

    /// Differing constraint coefficients are NOT interchangeable.
    ///
    /// Sentinel: ignoring the column entries would merge these and break a real
    /// solution (x0 with coef 1 vs x1 with coef 2 are not swappable).
    #[test]
    fn distinct_column_is_not_symmetric() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 2.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1]).unwrap();
        assert!(binary_orbits(&milp).is_empty());
    }

    /// Coefficients retained by CSC must participate in the signature.
    #[test]
    fn drop_tol_to_zero_tol_gap_coefficient_is_not_symmetric() {
        const GAP_COEF: f64 = crate::tolerances::DROP_TOL * 100.0;
        const { assert!(GAP_COEF > crate::tolerances::DROP_TOL) };
        const { assert!(GAP_COEF < ZERO_TOL) };

        let a =
            CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[1.0, 1.0, GAP_COEF], 2, 2).unwrap();
        assert_eq!(a.nnz(), 3);
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![1.0, 1.0],
            vec![ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1]).unwrap();

        assert!(binary_orbits(&milp).is_empty());
        let out = break_symmetry(&milp);
        assert_eq!(out.lp.num_constraints, 2);
    }

    /// Non-binary integer variables (and presolve-fixed binaries) are excluded.
    #[test]
    fn non_binary_integer_excluded() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![3.0],
            vec![ConstraintType::Le],
            vec![(0.0, 3.0), (0.0, 3.0)], // general integers, not {0,1}
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1]).unwrap();
        assert!(binary_orbits(&milp).is_empty());
    }

    /// `break_symmetry` adds exactly `(orbit_size - 1)` Ge rows of the form
    /// `x_i - x_{i+1} >= 0` and leaves the objective / bounds untouched.
    #[test]
    fn appends_lex_rows_for_orbit() {
        // 3 interchangeable binaries: x0 + x1 + x2 <= 2.
        let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[1.0, 1.0, 1.0], 1, 3).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0, 1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0); 3],
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1, 2]).unwrap();
        let out = break_symmetry(&milp);
        // 1 original + 2 lex rows (x0>=x1, x1>=x2).
        assert_eq!(out.lp.num_constraints, 3);
        assert_eq!(out.lp.constraint_types[1], ConstraintType::Ge);
        assert_eq!(out.lp.constraint_types[2], ConstraintType::Ge);
        assert_eq!(out.lp.b[1], 0.0);
        assert_eq!(out.lp.b[2], 0.0);
        // Row 1 = x0 - x1 >= 0.
        let (r1, v1) = column_entries(&out.lp.a, 1);
        assert_eq!(r1, vec![0, 1]);
        assert_eq!(v1, vec![1.0, -1.0]);
        // Objective and bounds preserved.
        assert_eq!(out.lp.c, vec![1.0, 1.0, 1.0]);
        assert_eq!(out.lp.bounds, vec![(0.0, 1.0); 3]);
    }

    /// No orbit → unchanged problem (clone).
    #[test]
    fn no_symmetry_returns_unchanged() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 2.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![2.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1]).unwrap();
        let out = break_symmetry(&milp);
        assert_eq!(out.lp.num_constraints, 1);
    }

    /// Extract the `(row, value)` entries of column `j` (row order ascending),
    /// reconstructing a single lex row for assertion.
    fn column_entries(a: &CscMatrix, target_row: usize) -> (Vec<usize>, Vec<f64>) {
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for c in 0..a.ncols() {
            let (rs, vs) = a.get_column(c).unwrap();
            for (&r, &v) in rs.iter().zip(vs) {
                if r == target_row {
                    cols.push(c);
                    vals.push(v);
                }
            }
        }
        (cols, vals)
    }
}
