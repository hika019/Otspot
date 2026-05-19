//! Step 5: bounds tightening from implied row activity.

use super::state::{PostsolveStep, PresolveState, PresolveStatus};
use crate::problem::ConstraintType;
use crate::tolerances::ZERO_TOL;

pub(super) fn step5_bounds_tightening(
    st: &mut PresolveState,
    new_fixed: &mut usize,
) -> Result<(), PresolveStatus> {
    // LP-side acceptance is unconditional: simplex relies on aggressive tightening
    // and QP-style dense / sanity caps caused timeouts on several test instances.
    let accept_implied_ub = |_implied: f64, _old_ub: f64| -> bool { true };
    let accept_implied_lb = |_implied: f64, _old_lb: f64| -> bool { true };

    let m = st.b.len();
    for i in 0..m {
        if st.removed_rows[i] {
            continue;
        }
        let ct = st.constraint_types[i];
        let entries = st.active_row_entries(i);
        if entries.is_empty() {
            continue;
        }

        let mut row_lb_sum = 0.0f64;
        let mut row_ub_sum = 0.0f64;
        let mut inf_lb_count = 0usize;
        let mut inf_ub_count = 0usize;
        let mut entry_lb_contrib = Vec::with_capacity(entries.len());
        let mut entry_ub_contrib = Vec::with_capacity(entries.len());
        let mut entry_lb_inf = Vec::with_capacity(entries.len());
        let mut entry_ub_inf = Vec::with_capacity(entries.len());

        for &(j, a_ij) in &entries {
            let (lb_j, ub_j) = st.bounds[j];
            if a_ij > 0.0 {
                if lb_j == f64::NEG_INFINITY {
                    inf_lb_count += 1;
                    entry_lb_inf.push(true);
                    entry_lb_contrib.push(0.0);
                } else {
                    entry_lb_inf.push(false);
                    let c = a_ij * lb_j;
                    entry_lb_contrib.push(c);
                    row_lb_sum += c;
                }
                if ub_j == f64::INFINITY {
                    inf_ub_count += 1;
                    entry_ub_inf.push(true);
                    entry_ub_contrib.push(0.0);
                } else {
                    entry_ub_inf.push(false);
                    let c = a_ij * ub_j;
                    entry_ub_contrib.push(c);
                    row_ub_sum += c;
                }
            } else if a_ij < 0.0 {
                if ub_j == f64::INFINITY {
                    inf_lb_count += 1;
                    entry_lb_inf.push(true);
                    entry_lb_contrib.push(0.0);
                } else {
                    entry_lb_inf.push(false);
                    let c = a_ij * ub_j;
                    entry_lb_contrib.push(c);
                    row_lb_sum += c;
                }
                if lb_j == f64::NEG_INFINITY {
                    inf_ub_count += 1;
                    entry_ub_inf.push(true);
                    entry_ub_contrib.push(0.0);
                } else {
                    entry_ub_inf.push(false);
                    let c = a_ij * lb_j;
                    entry_ub_contrib.push(c);
                    row_ub_sum += c;
                }
            } else {
                entry_lb_inf.push(false);
                entry_ub_inf.push(false);
                entry_lb_contrib.push(0.0);
                entry_ub_contrib.push(0.0);
            }
        }

        for (k, &(j, a_ij)) in entries.iter().enumerate() {
            if a_ij.abs() < ZERO_TOL {
                continue;
            }
            let (old_lb, old_ub) = st.bounds[j];

            let rest_inf_lb = if entry_lb_inf[k] { inf_lb_count - 1 } else { inf_lb_count };
            let rest_inf_ub = if entry_ub_inf[k] { inf_ub_count - 1 } else { inf_ub_count };
            let rest_lb = row_lb_sum - entry_lb_contrib[k];
            let rest_ub = row_ub_sum - entry_ub_contrib[k];
            let rest_lb_fin = rest_inf_lb == 0;
            let rest_ub_fin = rest_inf_ub == 0;

            let mut new_lb = old_lb;
            let mut new_ub = old_ub;

            match ct {
                ConstraintType::Le => {
                    if a_ij > 0.0 && rest_lb_fin {
                        let implied_ub = (st.b[i] - rest_lb) / a_ij;
                        if implied_ub < old_lb - ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_ub < new_ub - ZERO_TOL && accept_implied_ub(implied_ub, old_ub) {
                            new_ub = implied_ub;
                        }
                    } else if a_ij < 0.0 && rest_lb_fin {
                        let implied_lb = (st.b[i] - rest_lb) / a_ij;
                        if implied_lb > old_ub + ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_lb > new_lb + ZERO_TOL && accept_implied_lb(implied_lb, old_lb) {
                            new_lb = implied_lb;
                        }
                    }
                }
                ConstraintType::Ge => {
                    if a_ij > 0.0 && rest_ub_fin {
                        let implied_lb = (st.b[i] - rest_ub) / a_ij;
                        if implied_lb > old_ub + ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_lb > new_lb + ZERO_TOL && accept_implied_lb(implied_lb, old_lb) {
                            new_lb = implied_lb;
                        }
                    } else if a_ij < 0.0 && rest_ub_fin {
                        let implied_ub = (st.b[i] - rest_ub) / a_ij;
                        if implied_ub < old_lb - ZERO_TOL {
                            return Err(PresolveStatus::Infeasible);
                        }
                        if implied_ub < new_ub - ZERO_TOL && accept_implied_ub(implied_ub, old_ub) {
                            new_ub = implied_ub;
                        }
                    }
                }
                ConstraintType::Eq => {
                    if a_ij > 0.0 {
                        if rest_lb_fin {
                            let implied_ub = (st.b[i] - rest_lb) / a_ij;
                            if implied_ub < old_lb - ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_ub < new_ub - ZERO_TOL {
                                new_ub = implied_ub;
                            }
                        }
                        if rest_ub_fin {
                            let implied_lb = (st.b[i] - rest_ub) / a_ij;
                            if implied_lb > old_ub + ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                    } else {
                        if rest_lb_fin {
                            let implied_lb = (st.b[i] - rest_lb) / a_ij;
                            if implied_lb > old_ub + ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_lb > new_lb + ZERO_TOL {
                                new_lb = implied_lb;
                            }
                        }
                        if rest_ub_fin {
                            let implied_ub = (st.b[i] - rest_ub) / a_ij;
                            if implied_ub < old_lb - ZERO_TOL {
                                return Err(PresolveStatus::Infeasible);
                            }
                            if implied_ub < new_ub - ZERO_TOL {
                                new_ub = implied_ub;
                            }
                        }
                    }
                }
            }

            if (new_lb - old_lb).abs() > ZERO_TOL || (new_ub - old_ub).abs() > ZERO_TOL {
                st.postsolve_stack.push(PostsolveStep::BoundsTightened {
                    orig_col: j,
                    old_lb,
                    old_ub,
                });
                st.bounds[j] = (new_lb, new_ub);
                if (new_lb - new_ub).abs() < ZERO_TOL {
                    *new_fixed += 1;
                }
            }
        }
    }
    Ok(())
}
