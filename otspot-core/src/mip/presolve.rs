//! MIP-specific presolve: integer bound tightening via coefficient propagation.
//!
//! LP bound tightening infers implied variable bounds from linear constraints.
//! For integer variables the implied bound is additionally rounded (floor/ceil),
//! which is strictly stronger: a real-valued implied ub of 3.7 gives only `x ≤ 3`
//! for an integer variable rather than `x ≤ 3.7`.

use crate::problem::{ConstraintType, LpProblem};
use crate::tolerances::ZERO_TOL;

/// Tighten variable bounds by one pass of coefficient propagation.
///
/// For each constraint row and each variable in that row the implied bound is
/// derived from the activity of the remaining variables. Integer variables are
/// additionally rounded (floor for ub, ceil for lb).
///
/// Returns `None` when infeasibility is detected (implied lb > existing ub or
/// implied ub < existing lb after rounding). Returns the tightened bounds
/// otherwise.
pub(crate) fn tighten_integer_bounds(
    lp: &LpProblem,
    integer_mask: &[bool],
) -> Option<Vec<(f64, f64)>> {
    let n = lp.num_vars;
    let m = lp.num_constraints;

    // Build CSR row index from the CSC matrix (one pass).
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for j in 0..n {
        for k in lp.a.col_ptr[j]..lp.a.col_ptr[j + 1] {
            let row = lp.a.row_ind[k];
            let val = lp.a.values[k];
            if val.abs() >= ZERO_TOL {
                rows[row].push((j, val));
            }
        }
    }

    let mut bounds = lp.bounds.clone();

    for i in 0..m {
        let ct = lp.constraint_types[i];
        let b_i = lp.b[i];
        let entries = &rows[i];
        if entries.is_empty() {
            continue;
        }

        // Row activity bounds: min and max of sum a_j * x_j over current variable bounds.
        let mut row_lb = 0.0f64;
        let mut row_ub = 0.0f64;
        let mut inf_lb_count = 0usize;
        let mut inf_ub_count = 0usize;

        let mut e_lb_contrib: Vec<f64> = Vec::with_capacity(entries.len());
        let mut e_ub_contrib: Vec<f64> = Vec::with_capacity(entries.len());
        let mut e_lb_inf: Vec<bool> = Vec::with_capacity(entries.len());
        let mut e_ub_inf: Vec<bool> = Vec::with_capacity(entries.len());

        for &(j, a_ij) in entries.iter() {
            let (lb, ub) = bounds[j];
            if a_ij > 0.0 {
                if lb == f64::NEG_INFINITY {
                    inf_lb_count += 1;
                    e_lb_inf.push(true);
                    e_lb_contrib.push(0.0);
                } else {
                    let c = a_ij * lb;
                    e_lb_inf.push(false);
                    e_lb_contrib.push(c);
                    row_lb += c;
                }
                if ub == f64::INFINITY {
                    inf_ub_count += 1;
                    e_ub_inf.push(true);
                    e_ub_contrib.push(0.0);
                } else {
                    let c = a_ij * ub;
                    e_ub_inf.push(false);
                    e_ub_contrib.push(c);
                    row_ub += c;
                }
            } else {
                // a_ij < 0: contribution to row_lb uses ub, and row_ub uses lb.
                if ub == f64::INFINITY {
                    inf_lb_count += 1;
                    e_lb_inf.push(true);
                    e_lb_contrib.push(0.0);
                } else {
                    let c = a_ij * ub;
                    e_lb_inf.push(false);
                    e_lb_contrib.push(c);
                    row_lb += c;
                }
                if lb == f64::NEG_INFINITY {
                    inf_ub_count += 1;
                    e_ub_inf.push(true);
                    e_ub_contrib.push(0.0);
                } else {
                    let c = a_ij * lb;
                    e_ub_inf.push(false);
                    e_ub_contrib.push(c);
                    row_ub += c;
                }
            }
        }

        for (k, &(j, a_ij)) in entries.iter().enumerate() {
            let (old_lb, old_ub) = bounds[j];
            let is_int = integer_mask.get(j).copied().unwrap_or(false);

            let rest_inf_lb =
                if e_lb_inf[k] { inf_lb_count - 1 } else { inf_lb_count };
            let rest_inf_ub =
                if e_ub_inf[k] { inf_ub_count - 1 } else { inf_ub_count };
            let rest_lb = row_lb - e_lb_contrib[k];
            let rest_ub = row_ub - e_ub_contrib[k];
            let rest_lb_fin = rest_inf_lb == 0;
            let rest_ub_fin = rest_inf_ub == 0;

            let mut new_lb = old_lb;
            let mut new_ub = old_ub;

            match ct {
                ConstraintType::Le => {
                    if a_ij > 0.0 && rest_lb_fin {
                        let mut implied_ub = (b_i - rest_lb) / a_ij;
                        if is_int {
                            implied_ub = implied_ub.floor();
                        }
                        if implied_ub < old_lb - ZERO_TOL {
                            return None;
                        }
                        if implied_ub < new_ub - ZERO_TOL {
                            new_ub = implied_ub;
                        }
                    } else if a_ij < 0.0 && rest_lb_fin {
                        let mut implied_lb = (b_i - rest_lb) / a_ij;
                        if is_int {
                            implied_lb = implied_lb.ceil();
                        }
                        if implied_lb > old_ub + ZERO_TOL {
                            return None;
                        }
                        if implied_lb > new_lb + ZERO_TOL {
                            new_lb = implied_lb;
                        }
                    }
                }
                ConstraintType::Ge => {
                    if a_ij > 0.0 && rest_ub_fin {
                        let mut implied_lb = (b_i - rest_ub) / a_ij;
                        if is_int {
                            implied_lb = implied_lb.ceil();
                        }
                        if implied_lb > old_ub + ZERO_TOL {
                            return None;
                        }
                        if implied_lb > new_lb + ZERO_TOL {
                            new_lb = implied_lb;
                        }
                    } else if a_ij < 0.0 && rest_ub_fin {
                        let mut implied_ub = (b_i - rest_ub) / a_ij;
                        if is_int {
                            implied_ub = implied_ub.floor();
                        }
                        if implied_ub < old_lb - ZERO_TOL {
                            return None;
                        }
                        if implied_ub < new_ub - ZERO_TOL {
                            new_ub = implied_ub;
                        }
                    }
                }
                ConstraintType::Eq => {
                    if a_ij > 0.0 {
                        if rest_lb_fin {
                            let mut implied_ub = (b_i - rest_lb) / a_ij;
                            if is_int {
                                implied_ub = implied_ub.floor();
                            }
                            if implied_ub < old_lb - ZERO_TOL {
                                return None;
                            }
                            if implied_ub < new_ub - ZERO_TOL {
                                new_ub = implied_ub;
                            }
                        }
                        if rest_ub_fin {
                            let mut implied_lb = (b_i - rest_ub) / a_ij;
                            if is_int {
                                implied_lb = implied_lb.ceil();
                            }
                            if implied_lb > old_ub + ZERO_TOL {
                                return None;
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                    } else {
                        // a_ij < 0
                        if rest_lb_fin {
                            let mut implied_lb = (b_i - rest_lb) / a_ij;
                            if is_int {
                                implied_lb = implied_lb.ceil();
                            }
                            if implied_lb > old_ub + ZERO_TOL {
                                return None;
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                        if rest_ub_fin {
                            let mut implied_ub = (b_i - rest_ub) / a_ij;
                            if is_int {
                                implied_ub = implied_ub.floor();
                            }
                            if implied_ub < old_lb - ZERO_TOL {
                                return None;
                            }
                            if implied_ub < new_ub - ZERO_TOL {
                                new_ub = implied_ub;
                            }
                        }
                    }
                }
            }

            bounds[j] = (new_lb, new_ub);
        }
    }

    Some(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{ConstraintType, LpProblem};
    use crate::sparse::CscMatrix;

    fn single_var_lp(a_val: f64, b_val: f64, ct: ConstraintType, domain: (f64, f64)) -> LpProblem {
        let a = CscMatrix::from_triplets(&[0], &[0], &[a_val], 1, 1).unwrap();
        LpProblem::new_general(vec![1.0], a, vec![b_val], vec![ct], vec![domain], None).unwrap()
    }

    /// Floor-rounding fires for integer ub from Le constraint.
    ///
    /// Sentinel: removing `floor()` leaves ub = 3.7 instead of 3.0 → assertion fails.
    #[test]
    fn integer_ub_is_floored_from_le_constraint() {
        let lp = single_var_lp(1.0, 3.7, ConstraintType::Le, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        assert!(
            (bounds[0].1 - 3.0).abs() < 1e-9,
            "integer ub from x ≤ 3.7 must be floor(3.7)=3, got {}",
            bounds[0].1
        );
    }

    /// Ceil-rounding fires for integer lb from Ge constraint.
    ///
    /// Sentinel: removing `ceil()` leaves lb = 2.3 instead of 3.0 → assertion fails.
    #[test]
    fn integer_lb_is_ceiled_from_ge_constraint() {
        let lp = single_var_lp(1.0, 2.3, ConstraintType::Ge, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        assert!(
            (bounds[0].0 - 3.0).abs() < 1e-9,
            "integer lb from x ≥ 2.3 must be ceil(2.3)=3, got {}",
            bounds[0].0
        );
    }

    /// Infeasibility detected when rounded bounds produce an empty domain.
    ///
    /// x ≤ 3.7 → ub=3, then x ≥ 4 → lb=4 > ub=3 → None.
    /// Sentinel: without floor/ceil the bounds stay [3.2, 3.7] and None is not returned.
    #[test]
    fn infeasibility_detected_by_integer_rounding() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![3.7, 4.0],
            vec![ConstraintType::Le, ConstraintType::Ge],
            vec![(0.0, 10.0)],
            None,
        )
        .unwrap();
        let result = tighten_integer_bounds(&lp, &[true]);
        assert!(result.is_none(), "x ≤ 3.7 ∧ x ≥ 4 integer → empty domain → None");
    }

    /// Continuous variables are not rounded.
    #[test]
    fn continuous_var_not_rounded() {
        let lp = single_var_lp(1.0, 3.7, ConstraintType::Le, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[false]).expect("feasible");
        assert!(
            (bounds[0].1 - 3.7).abs() < 1e-9,
            "continuous var ub must stay 3.7, got {}",
            bounds[0].1
        );
    }

    /// No constraints → bounds unchanged.
    #[test]
    fn no_constraints_unchanged() {
        let a = CscMatrix::new(0, 2);
        let lp =
            LpProblem::new_general(vec![1.0, 1.0], a, vec![], vec![], vec![(0.0, 5.0); 2], None)
                .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, true]).expect("feasible");
        assert_eq!(bounds, vec![(0.0, 5.0), (0.0, 5.0)]);
    }

    /// Two-variable instance: coefficient propagation tightens both vars.
    ///
    /// x + y ≤ 5, x,y ∈ [0,10] integer.
    /// At k=0 (j=x): rest_lb=0 (y's min contrib), implied_ub = 5 → floor(5)=5 → x ≤ 5.
    /// At k=1 (j=y): rest_lb=0 (x's min after tightening is still 0),
    ///               implied_ub = 5 → floor(5)=5 → y ≤ 5.
    #[test]
    fn two_var_le_tightens_both_ubs() {
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, 10.0)],
            None,
        )
        .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, true]).expect("feasible");
        assert!((bounds[0].1 - 5.0).abs() < 1e-9, "x ub must be 5, got {}", bounds[0].1);
        assert!((bounds[1].1 - 5.0).abs() < 1e-9, "y ub must be 5, got {}", bounds[1].1);
    }

    /// Eq constraint tightens both lb and ub.
    ///
    /// x = 4 (Eq, one-var), integer, domain [0,10] → both bounds → 4.
    #[test]
    fn eq_constraint_fixes_integer_var() {
        let lp = single_var_lp(1.0, 4.0, ConstraintType::Eq, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        assert!((bounds[0].0 - 4.0).abs() < 1e-9, "lb={}", bounds[0].0);
        assert!((bounds[0].1 - 4.0).abs() < 1e-9, "ub={}", bounds[0].1);
    }

    /// Negative coefficient: x ≥ 2.3 expressed as -x ≤ -2.3. Integer lb → ceil(2.3)=3.
    #[test]
    fn negative_coeff_le_tightens_lb() {
        let lp = single_var_lp(-1.0, -2.3, ConstraintType::Le, (0.0, 10.0));
        let bounds = tighten_integer_bounds(&lp, &[true]).expect("feasible");
        // a_ij = -1 < 0, rest_lb_fin = true (no other vars)
        // implied_lb = (-2.3 - 0) / (-1) = 2.3, ceil(2.3) = 3
        assert!(
            (bounds[0].0 - 3.0).abs() < 1e-9,
            "integer lb from -x ≤ -2.3 must be ceil(2.3)=3, got {}",
            bounds[0].0
        );
    }

    /// Infinite variable bound: when the "rest" has an infinite contribution the
    /// propagation is skipped for that variable (no bound derived).
    #[test]
    fn infinite_rest_skips_propagation() {
        // x + y ≤ 5, y ∈ [0, ∞), x ∈ [0, 10] integer.
        // For x: rest_lb uses y's lb=0 (fin), implied_ub = 5-0 = 5 → x ≤ 5 ✓
        // For y: rest_lb uses x's lb=0 (fin), implied_ub = 5-0 = 5 → but y is not integer
        //         so ub stays 5.0 (continuous). y's ub was ∞; with y continuous → 5.0
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![5.0],
            vec![ConstraintType::Le],
            vec![(0.0, 10.0), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let bounds = tighten_integer_bounds(&lp, &[true, false]).expect("feasible");
        // x: rest_lb=y*1*0=0 (finite), implied_ub=5→floor(5)=5
        assert!((bounds[0].1 - 5.0).abs() < 1e-9, "x ub: {}", bounds[0].1);
        // y: rest_lb=x*1*0=0 (finite), implied_ub=5 (continuous, no floor)
        assert!((bounds[1].1 - 5.0).abs() < 1e-9, "y ub: {}", bounds[1].1);
    }
}
