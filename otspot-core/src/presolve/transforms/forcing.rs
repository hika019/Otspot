//! Step 2b: forcing row — when a row's activity range makes it tight,
//! every variable in the row is forced to its contributing bound.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

/// ULP coefficient of the per-member movement floor (twin of
/// `ROOT_PRESOLVE_ROUNDING_ULPS`).
const FORCING_MOVE_ULPS: f64 = 4.0;

/// Forcing margin for one side (`at_min` = force toward the min-activity
/// endpoints): `min(ZERO_TOL, min_j |a_j|·(ZERO_TOL + k·EPSILON·|endpoint_j|))`
/// — never more absolute slack than the historical scale-free `ZERO_TOL`,
/// and never more than the cheapest representable in-interval move of any
/// member (the endpoint being the bound the pin would write). The small-scale
/// half is the PR #89 P1 finding: an absolute `1e-12` is *5× an entire row*
/// at `1e-11` coefficient scale and false-pins members off the true optimum
/// (see `test_forcing_tiny_coefficient_scale_is_not_forcing`).
///
/// The margin deliberately has NO activity-scale term. PR #89 P1 shipped
/// `64·EPSILON·(‖rhs‖ + Σ|a_j|·max‖bounds_j‖) + movement`; on pilot87-class
/// rows whose activity terms reach 1e16..1e18 that term grows to O(10²..10⁵)
/// and truncated genuine feasible slivers (row misses of 13.8..11822 read as
/// "forcing", every member pinned, a later pass threw the false
/// `PresolveStatus::Infeasible` — first-bad `43b5494e`). The error scale of
/// evaluating the activity sum bounds what we may *conclude about* the
/// slack; it never bounds how much feasible slack may be *truncated*. When
/// rounding error swallows the slack, the sound conclusion is "forcing is
/// undecidable here — skip", not "pin everything". Dropping the term makes
/// the margin ≤ the pre-#89 `ZERO_TOL` cap structurally: genuinely forcing
/// rows at huge magnitudes lose the reduction (conservative, correct), and
/// no row can ever be pinned on a wider slack than before #89.
/// NOTE: the infeasibility test in [`step2b_forcing_row`] deliberately keeps
/// the plain `ZERO_TOL` gate — loosening it would trade false-Infeasible
/// claims for published points violating exactly-evaluable rows; its sound
/// scale-aware form needs an exact activity-evaluation analysis (separate
/// finding).
fn forcing_margin(active: &[(usize, f64)], bounds: &[(f64, f64)], at_min: bool) -> f64 {
    let mut quantum = f64::INFINITY;
    for &(j, a) in active {
        let (lb, ub) = bounds[j];
        if !(lb.is_finite() && ub.is_finite()) {
            // Rows with an infinite contributing bound fail the all-finite
            // guard below anyway; they must not set the movement gate.
            continue;
        }
        // The bound the pin would write for this member.
        let endpoint = if (a > 0.0) == at_min { lb } else { ub };
        quantum =
            quantum.min(a.abs() * (ZERO_TOL + FORCING_MOVE_ULPS * f64::EPSILON * endpoint.abs()));
    }
    let movement = if quantum.is_finite() { quantum } else { 0.0 };
    movement.min(ZERO_TOL)
}

pub(super) fn step2b_forcing_row(
    st: &mut PresolveState,
    deadline: Option<std::time::Instant>,
) -> Result<(), PresolveStatus> {
    let m = st.b.len();
    for i in 0..m {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(());
        }
        if st.removed_rows[i] {
            continue;
        }
        let active = st.active_row_entries(i);
        if active.len() < 2 {
            continue;
        }

        let (row_lb, row_ub, lb_fin, ub_fin) =
            crate::presolve::activity::activity_range(&active, &st.bounds, None);
        let rhs = st.b[i];
        let ct = st.constraint_types[i];

        // Detect infeasibility before checking forcing. (Plain ZERO_TOL on
        // purpose — see [`forcing_margin`]'s NOTE.)
        let infeasible = match ct {
            ConstraintType::Le => lb_fin && row_lb > rhs + ZERO_TOL,
            ConstraintType::Ge => ub_fin && row_ub < rhs - ZERO_TOL,
            ConstraintType::Eq => {
                (lb_fin && row_lb > rhs + ZERO_TOL) || (ub_fin && row_ub < rhs - ZERO_TOL)
            }
        };
        if infeasible {
            return Err(PresolveStatus::Infeasible);
        }

        // Forcing: the activity bound equals rhs, so all variables are at their
        // contributing bounds. The equality must hold within the pin-tolerance
        // margin of [`forcing_margin`] — at most the historical scale-free
        // `ZERO_TOL`, tightened to the cheapest member move at small
        // coefficient scale (PR #89 P1 class; the large-scale half of that PR's
        // formula is reverted here — see [`forcing_margin`]).
        // - Le: a_min >= b - margin (min already reaches rhs)
        // - Ge: a_max <= b + margin (max already reaches rhs)
        // - Eq: either side may force (infeasibility already excluded above
        //   bounds the miss distance of the chosen side by the margin).
        let lo_force = lb_fin && rhs - row_lb <= forcing_margin(&active, &st.bounds, true);
        let hi_force = ub_fin && row_ub - rhs <= forcing_margin(&active, &st.bounds, false);
        let forcing = match ct {
            ConstraintType::Le => lo_force,
            ConstraintType::Ge => hi_force,
            ConstraintType::Eq => lo_force || hi_force,
        };

        if !forcing {
            continue;
        }

        // Determine which bound direction forces: for Le (or Eq forced from below)
        // the min-activity achieves the rhs, so each variable is at its lb-contributing
        // bound. For Ge (or Eq forced from above) the max-activity achieves the rhs.
        // When both Eq extremes are within the margin, pin the CLOSER one
        // (mirrors the MILP root-presolve Eq rule): the farther extreme's
        // activity misses the rhs by more, dropping a wider slice of the
        // feasible window.
        let force_to_min = match ct {
            ConstraintType::Le => true,
            ConstraintType::Ge => false,
            ConstraintType::Eq => lo_force && !(hi_force && rhs - row_lb > row_ub - rhs),
        };

        // Check that all contributing bounds are finite.
        let all_finite = active.iter().all(|&(j, a_ij)| {
            let (lb_j, ub_j) = st.bounds[j];
            if force_to_min {
                if a_ij > 0.0 {
                    lb_j.is_finite()
                } else {
                    ub_j.is_finite()
                }
            } else {
                if a_ij > 0.0 {
                    ub_j.is_finite()
                } else {
                    lb_j.is_finite()
                }
            }
        });
        if !all_finite {
            continue;
        }

        // Collect (col, value, old_lb, old_ub) for each forced variable.
        let fixed_vars: Vec<(usize, f64, f64, f64)> = active
            .iter()
            .map(|&(j, a_ij)| {
                let (lb_j, ub_j) = st.bounds[j];
                let value = if force_to_min {
                    if a_ij > 0.0 {
                        lb_j
                    } else {
                        ub_j
                    }
                } else {
                    if a_ij > 0.0 {
                        ub_j
                    } else {
                        lb_j
                    }
                };
                (j, value, lb_j, ub_j)
            })
            .collect();

        // Fix each variable: substitute out of remaining rows and objective.
        for &(j, value, _, _) in &fixed_vars {
            if st.removed_cols[j] {
                continue;
            }
            let col_copy = st.col_entries[j].clone();
            for (row, val) in col_copy {
                if !st.removed_rows[row] && row != i {
                    st.b[row] -= val * value;
                }
            }
            st.obj_offset += st.c[j] * value;
            st.removed_cols[j] = true;
        }

        st.removed_rows[i] = true;
        st.postsolve_stack.push(PostsolveStep::ForcingRow {
            orig_row: i,
            fixed_vars,
            row_orig_entries: active,
        });
    }
    Ok(())
}
