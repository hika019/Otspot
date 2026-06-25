//! Single-node flow-cover (fixed-charge) cuts.
//!
//! Detects the single-node fixed-charge structure
//!   Σ_{j∈N} a_j x_j ≤ b,   0 ≤ x_j ≤ u_j y_j,   y_j ∈ {0,1}
//! where each flow variable `x_j` carries a variable-upper-bound (VUB) row
//! `x_j ≤ u_j y_j` (binary `y_j`). For a flow cover `C ⊆ N` with
//! `λ = Σ_{j∈C} ũ_j − b > 0` (effective capacity `ũ_j = a_j u_j`), the simple
//! flow-cover inequality
//!   Σ_{j∈C} [ a_j x_j + (ũ_j − λ)^+ (1 − y_j) ] ≤ b
//! is valid and is separated when violated by the LP solution.
//!
//! Validity (structural): split `C` into `C0` (y=0 ⇒ x=0) and `C1` (y=1).
//! The LHS equals `Σ_{C1} a_j x_j + Σ_{C0} (ũ_j−λ)^+`. Either some `j∈C0` has
//! `ũ_j ≥ λ` (then `Σ_{C0} min(ũ_j,λ) ≥ λ`, and the per-capacity bound gives
//! `≤ b`), or every `j∈C0` has `ũ_j < λ` (then `Σ_{C0}(ũ_j−λ)^+ = 0` and the
//! capacity row `Σ_{C1} a_j x_j ≤ b` closes it). A `debug_assert` re-verifies
//! validity by enumerating `y ∈ {0,1}^{|C|}` against the exact worst-case flow.

use crate::problem::{ConstraintType, LpProblem};
use crate::tolerances::{feas_rel_tol, ZERO_TOL};

use super::{finalize_cut, is_binary, row_lists, CutRow, MAX_CUTS_PER_ROUND};

/// A variable-upper-bound relation `x ≤ cap · y` (x ≥ 0 non-binary, y binary).
#[derive(Clone, Copy)]
struct Vub {
    y: usize,
    cap: f64,
}

/// VUB rows have exactly two structural nonzeros: the flow var and its binary.
const VUB_ROW_NNZ: usize = 2;
/// Minimum flow-cover cardinality worth separating.
const MIN_COVER_SIZE: usize = 2;
/// Cap on `|C|` for the debug-only exhaustive validity check (`2^|C|`).
const FLOW_VALIDITY_BRUTE_FORCE_LIMIT: usize = 20;

/// Detect VUB rows `c1·x − c2·y ≤ 0` ⇒ `x ≤ (c2/c1)·y`. Returns, per variable,
/// the tightest (smallest-cap) VUB found. Only non-binary, ≥0 variables qualify
/// as flow variables; the partner must be a binary with a negative coefficient.
fn detect_vubs(
    lp: &LpProblem,
    integer_mask: &[bool],
    rows: &[Vec<(usize, f64)>],
) -> Vec<Option<Vub>> {
    let mut vubs: Vec<Option<Vub>> = vec![None; lp.num_vars];
    for r in 0..lp.num_constraints {
        if lp.constraint_types[r] != ConstraintType::Le {
            continue;
        }
        if lp.b[r].abs() > ZERO_TOL {
            continue; // require rhs 0 for a clean x ≤ u·y relation
        }
        let row = &rows[r];
        if row.len() != VUB_ROW_NNZ {
            continue;
        }
        let (j0, v0) = row[0];
        let (j1, v1) = row[1];
        // Identify the positive-coeff flow var and the negative-coeff binary.
        let (xj, xc, yj, yc) = if v0 > ZERO_TOL && v1 < -ZERO_TOL {
            (j0, v0, j1, v1)
        } else if v1 > ZERO_TOL && v0 < -ZERO_TOL {
            (j1, v1, j0, v0)
        } else {
            continue;
        };
        if !is_binary(yj, integer_mask, &lp.bounds) {
            continue;
        }
        if is_binary(xj, integer_mask, &lp.bounds) {
            continue; // flow var must not itself be the binary
        }
        if lp.bounds[xj].0 < -ZERO_TOL {
            continue; // flow var must have lower bound 0
        }
        let cap = (-yc) / xc; // x ≤ (c2/c1) y
        if cap <= ZERO_TOL {
            continue;
        }
        let keep = match vubs[xj] {
            Some(prev) => cap < prev.cap,
            None => true,
        };
        if keep {
            vubs[xj] = Some(Vub { y: yj, cap });
        }
    }
    vubs
}

/// One member of a candidate flow cover.
#[derive(Clone, Copy)]
struct FlowItem {
    x: usize,
    y: usize,
    a: f64,      // capacity-row coefficient
    utilde: f64, // effective capacity ũ_j = a_j · u_j
}

/// Generate single-node flow-cover cuts (`g·x ≥ rhs`, Ge form).
pub(super) fn generate_flow_cover_cuts(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
) -> Vec<CutRow> {
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
    let vubs = detect_vubs(lp, integer_mask, &rows);
    let mut cuts = Vec::new();

    'row: for r in 0..lp.num_constraints {
        if cuts.len() >= MAX_CUTS_PER_ROUND {
            break;
        }
        if lp.constraint_types[r] != ConstraintType::Le {
            continue;
        }
        let b = lp.b[r];
        if b <= ZERO_TOL {
            continue;
        }
        let row = &rows[r];
        if row.len() < MIN_COVER_SIZE {
            continue;
        }
        // Every entry must be a positive-coefficient flow variable with a VUB.
        let mut items: Vec<FlowItem> = Vec::with_capacity(row.len());
        for &(j, a) in row {
            if a <= ZERO_TOL {
                continue 'row;
            }
            let Some(vub) = vubs[j] else {
                continue 'row;
            };
            items.push(FlowItem {
                x: j,
                y: vub.y,
                a,
                utilde: a * vub.cap,
            });
        }
        if items.len() < MIN_COVER_SIZE {
            continue;
        }

        if let Some(cut) = separate_flow_cover(lp, &items, b, x_star, frac_tol) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Build a flow cover by greedily including high-flow variables, minimise it,
/// then emit the simple flow-cover inequality if it cuts `x_star`.
fn separate_flow_cover(
    lp: &LpProblem,
    items: &[FlowItem],
    b: f64,
    x_star: &[f64],
    frac_tol: f64,
) -> Option<CutRow> {
    // Order by LP flow value descending: the variables the LP loads are the
    // ones whose cover membership yields violation.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&i, &j| {
        let xi = x_star.get(items[i].x).copied().unwrap_or(0.0);
        let xj = x_star.get(items[j].x).copied().unwrap_or(0.0);
        xj.total_cmp(&xi)
    });

    let mut cover: Vec<usize> = Vec::new();
    let mut sum_utilde = 0.0_f64;
    for &idx in &order {
        cover.push(idx);
        sum_utilde += items[idx].utilde;
        if sum_utilde > b {
            break;
        }
    }
    if sum_utilde <= b {
        return None; // not a cover
    }

    // Minimise: drop the smallest-ũ members while still a cover.
    cover.sort_by(|&i, &j| items[i].utilde.total_cmp(&items[j].utilde));
    let mut k = 0;
    while k < cover.len() {
        let cand = cover[k];
        if sum_utilde - items[cand].utilde > b {
            sum_utilde -= items[cand].utilde;
            cover.remove(k);
        } else {
            k += 1;
        }
    }
    if cover.len() < MIN_COVER_SIZE {
        return None;
    }

    let lambda = sum_utilde - b;
    debug_assert!(lambda > 0.0, "flow cover excess λ must be positive");

    // Le form: Σ_C [ a_j x_j + (ũ_j−λ)^+ (1 − y_j) ] ≤ b.
    // Ge form (g·x ≥ rhs): g[x_j] = −a_j, g[y_j] = +(ũ_j−λ)^+,
    // rhs = −b + Σ_C (ũ_j−λ)^+.
    let mut g = vec![0.0; lp.num_vars];
    let mut sum_coef = 0.0_f64;
    for &idx in &cover {
        let it = items[idx];
        let coef = (it.utilde - lambda).max(0.0);
        g[it.x] -= it.a;
        g[it.y] += coef;
        sum_coef += coef;
    }
    let rhs = -b + sum_coef;

    // Release-mode safety net: the (size-bounded) exhaustive check runs in
    // release too, and an invalid cut is dropped rather than shipped — so a
    // future regression in the structural argument can never remove an
    // integer-feasible point. A no-op while the proof holds.
    if !flow_cover_is_valid(items, &cover, b, lambda) {
        return None;
    }

    finalize_cut(g, rhs, x_star, frac_tol)
}

/// Exhaustively verify the simple flow-cover inequality over `y ∈ {0,1}^|C|`.
///
/// For fixed `y`, the cut's Le LHS is maximised by pushing flow to its feasible
/// limit: `Σ_C a_j x_j ≤ min(b, Σ_C ũ_j y_j)` (capacity ∧ VUB), giving worst-case
/// LHS `min(b, Σ ũ_j y_j) + Σ (ũ_j−λ)^+ (1−y_j)`, which must stay `≤ b`.
fn flow_cover_is_valid(items: &[FlowItem], cover: &[usize], b: f64, lambda: f64) -> bool {
    if cover.len() > FLOW_VALIDITY_BRUTE_FORCE_LIMIT {
        return true; // too large to enumerate; rely on the structural argument
    }
    let tol = 1e-7 * (1.0 + b.abs());
    let n = cover.len();
    for mask in 0u64..(1u64 << n) {
        let mut cap_used = 0.0_f64;
        let mut const_term = 0.0_f64;
        for (bit, &idx) in cover.iter().enumerate() {
            let it = items[idx];
            let coef = (it.utilde - lambda).max(0.0);
            if mask & (1 << bit) != 0 {
                cap_used += it.utilde; // y_j = 1
            } else {
                const_term += coef; // y_j = 0 ⇒ x_j = 0
            }
        }
        let worst_lhs = cap_used.min(b) + const_term;
        if worst_lhs > b + tol {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mip::problem::MilpProblem;
    use crate::sparse::CscMatrix;

    /// Fixed-charge single-node flow instance:
    ///   x1 + x2 ≤ 3,  x1 ≤ 2 y1,  x2 ≤ 2 y2,  x integer ∈ [0,2], y ∈ {0,1}.
    /// Objective min −(x1+x2) + (y1+y2)/4 keeps the LP root fractional. The flow
    /// cover C={1,2} (ũ=2 each, λ=1) yields x1+x2 − y1 − y2 ≤ 1.
    fn fixed_charge() -> (MilpProblem, Vec<bool>) {
        // vars: x1=0, x2=1, y1=2, y2=3.
        // rows: r0: x1+x2 ≤ 3; r1: x1 − 2y1 ≤ 0; r2: x2 − 2y2 ≤ 0.
        let rows = [0, 0, 1, 1, 2, 2];
        let cols = [0, 1, 0, 2, 1, 3];
        let vals = [1.0, 1.0, 1.0, -2.0, 1.0, -2.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 4).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, -1.0, 0.25, 0.25],
            a,
            vec![3.0, 0.0, 0.0],
            vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 2.0), (0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        // x1, x2 integer flow (brute-forceable); y1, y2 binary.
        let milp = MilpProblem::new(lp, vec![0, 1, 2, 3]).unwrap();
        let mask = crate::mip::integer_mask(4, &[0, 1, 2, 3]);
        (milp, mask)
    }

    fn lp_root(lp: &LpProblem) -> crate::problem::SolverResult {
        crate::mip::cuts::solve_cut_lp(lp, &crate::options::SolverOptions::default(), None)
    }

    /// **Sentinel — separation:** the flow-cover generator must emit a cut that
    /// the fractional LP optimum violates.
    #[test]
    fn flow_cover_separates_lp_optimum() {
        let (milp, mask) = fixed_charge();
        let root = lp_root(&milp.lp);
        assert_eq!(root.status, crate::problem::SolveStatus::Optimal);
        let x_star = &root.solution;
        let cuts = generate_flow_cover_cuts(&milp.lp, &mask, x_star);
        assert!(!cuts.is_empty(), "flow-cover cut must be generated");
        let any_violated = cuts.iter().any(|c| {
            let lhs: f64 = c.coeffs.iter().zip(x_star).map(|(&g, &x)| g * x).sum();
            lhs < c.rhs - 1e-9
        });
        assert!(
            any_violated,
            "a flow-cover cut must violate the LP optimum {x_star:?}"
        );
    }

    /// **Sentinel — validity:** no integer-feasible point is removed. The flow
    /// vars are integer so the box is brute-forceable.
    #[test]
    fn flow_cover_cut_valid_for_all_integer_points() {
        let (milp, mask) = fixed_charge();
        let x_star = lp_root(&milp.lp).solution;
        let cuts = generate_flow_cover_cuts(&milp.lp, &mask, &x_star);
        assert!(!cuts.is_empty());
        // Enumerate the integer box and check feasibility against original rows.
        for x1 in 0..=2 {
            for x2 in 0..=2 {
                for y1 in 0..=1 {
                    for y2 in 0..=1 {
                        let x = [x1 as f64, x2 as f64, y1 as f64, y2 as f64];
                        let ax = milp.lp.a.mat_vec_mul(&x).unwrap();
                        let feas = ax[0] <= 3.0 + 1e-9
                            && ax[1] <= 1e-9
                            && ax[2] <= 1e-9;
                        if !feas {
                            continue;
                        }
                        for (k, c) in cuts.iter().enumerate() {
                            let lhs: f64 =
                                c.coeffs.iter().zip(x.iter()).map(|(&g, &xi)| g * xi).sum();
                            assert!(
                                lhs >= c.rhs - 1e-9,
                                "flow cover cut {k} removes integer-feasible {x:?}: \
                                 lhs={lhs} < rhs={}",
                                c.rhs
                            );
                        }
                    }
                }
            }
        }
    }

    /// **No false cut:** a row with no cover (Σũ ≤ b) must not yield a cut.
    #[test]
    fn flow_cover_no_cut_without_cover() {
        // x1 + x2 ≤ 5, x1 ≤ 2 y1, x2 ≤ 2 y2: Σũ = 4 ≤ 5, no cover.
        let rows = [0, 0, 1, 1, 2, 2];
        let cols = [0, 1, 0, 2, 1, 3];
        let vals = [1.0, 1.0, 1.0, -2.0, 1.0, -2.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 4).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, -1.0, 0.25, 0.25],
            a,
            vec![5.0, 0.0, 0.0],
            vec![ConstraintType::Le, ConstraintType::Le, ConstraintType::Le],
            vec![(0.0, 2.0), (0.0, 2.0), (0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();
        let mask = crate::mip::integer_mask(4, &[0, 1, 2, 3]);
        let x_star = lp_root(&lp).solution;
        let cuts = generate_flow_cover_cuts(&lp, &mask, &x_star);
        assert!(cuts.is_empty(), "no cover ⇒ no flow-cover cut, got {}", cuts.len());
    }
}
