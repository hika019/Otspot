//! Lifted knapsack-cover cuts.
//!
//! From a 0-1 knapsack row `Σ_{j∈N} a_j x_j ≤ b` (positive-coefficient binaries)
//! we pick a minimal cover `C` (`Σ_{C} a_j > b`, minimal), giving the base cover
//! inequality `Σ_{C} x_j ≤ |C|−1`. Sequential up-lifting then assigns each
//! non-cover variable `k` a coefficient
//!   α_k = β_0 − max{ Σ_{j∈S} β_j x_j : Σ_{j∈S} a_j x_j ≤ b − a_k, x ∈ {0,1} },
//! where `S` is the cover plus previously-lifted variables (`β_0 = |C|−1`).
//! This exact sequential lifting is valid for any lifting order; the inner max
//! is solved exactly by a value-indexed 0-1 knapsack DP (handles real weights).
//!
//! A `debug_assert` re-verifies validity by enumerating the cut's binary support
//! against the original knapsack row.

use crate::problem::{ConstraintType, LpProblem};
use crate::tolerances::{feas_rel_tol, ZERO_TOL};

use super::{finalize_cut, is_binary, row_lists, CutRow, MAX_CUTS_PER_ROUND};

/// Minimum cover cardinality (a 1-element cover gives the trivial `x_j ≤ 0`).
const MIN_COVER_SIZE: usize = 2;
/// Cap on the cut-support size for the debug-only exhaustive validity check.
const KNAPSACK_VALIDITY_BRUTE_FORCE_LIMIT: usize = 18;

/// A knapsack item carried through lifting: weight `a` and current coefficient.
#[derive(Clone, Copy)]
struct KItem {
    weight: f64,
    coeff: i64,
}

/// Max `Σ value` over a 0-1 knapsack with capacity `cap`, via a value-indexed DP
/// (`minw[v]` = least weight achieving value exactly `v`). Returns `None` when
/// `cap < 0` (not even the empty selection fits).
fn knapsack_max_value(items: &[KItem], cap: f64) -> Option<i64> {
    if cap < -ZERO_TOL {
        return None;
    }
    let total: i64 = items.iter().map(|it| it.coeff.max(0)).sum();
    let total_u = total as usize;
    let mut minw = vec![f64::INFINITY; total_u + 1];
    minw[0] = 0.0;
    for it in items {
        if it.coeff <= 0 {
            continue;
        }
        let v = it.coeff as usize;
        for value in (v..=total_u).rev() {
            let prev = minw[value - v];
            if prev.is_finite() && prev + it.weight < minw[value] {
                minw[value] = prev + it.weight;
            }
        }
    }
    let mut best = 0i64;
    for value in 0..=total_u {
        if minw[value] <= cap + ZERO_TOL {
            best = value as i64;
        }
    }
    Some(best)
}

/// Generate lifted knapsack-cover cuts (`g·x ≥ rhs`, Ge form).
pub(super) fn generate_lifted_knapsack_cover_cuts(
    lp: &LpProblem,
    integer_mask: &[bool],
    x_star: &[f64],
) -> Vec<CutRow> {
    let frac_tol = feas_rel_tol();
    let rows = row_lists(&lp.a, lp.num_constraints);
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
        for &(j, v) in row {
            if v <= ZERO_TOL || !is_binary(j, integer_mask, &lp.bounds) {
                continue 'row;
            }
        }
        if let Some(cut) = separate_lifted_cover(lp, row, b, x_star, frac_tol) {
            cuts.push(cut);
        }
    }
    cuts
}

/// Pick a minimal cover violated by `x_star`, up-lift the non-cover variables,
/// and emit the lifted cover inequality if it cuts `x_star`.
fn separate_lifted_cover(
    lp: &LpProblem,
    row: &[(usize, f64)],
    b: f64,
    x_star: &[f64],
    frac_tol: f64,
) -> Option<CutRow> {
    // Greedy cover by LP value descending (small 1−x* first ⇒ most violated).
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&i, &j| {
        let xi = x_star.get(row[i].0).copied().unwrap_or(0.0);
        let xj = x_star.get(row[j].0).copied().unwrap_or(0.0);
        xj.total_cmp(&xi)
    });

    let mut in_cover = vec![false; row.len()];
    let mut sum_a = 0.0_f64;
    for &idx in &order {
        in_cover[idx] = true;
        sum_a += row[idx].1;
        if sum_a > b {
            break;
        }
    }
    if sum_a <= b {
        return None; // not a cover
    }

    // Minimise: drop members (smallest weight first) while still a cover.
    let mut by_weight: Vec<usize> = (0..row.len()).filter(|&i| in_cover[i]).collect();
    by_weight.sort_by(|&i, &j| row[i].1.total_cmp(&row[j].1));
    for &idx in &by_weight {
        if sum_a - row[idx].1 > b {
            sum_a -= row[idx].1;
            in_cover[idx] = false;
        }
    }

    let cover: Vec<usize> = (0..row.len()).filter(|&i| in_cover[i]).collect();
    if cover.len() < MIN_COVER_SIZE {
        return None;
    }
    let beta_0 = (cover.len() - 1) as i64;

    // Coefficients keyed by original variable index. Cover members get 1.
    let mut coeff: Vec<(usize, i64)> = cover.iter().map(|&i| (row[i].0, 1)).collect();
    let mut lift_set: Vec<KItem> = cover
        .iter()
        .map(|&i| KItem { weight: row[i].1, coeff: 1 })
        .collect();

    // Sequential up-lifting of non-cover variables, largest weight first.
    let mut non_cover: Vec<usize> = (0..row.len()).filter(|&i| !in_cover[i]).collect();
    non_cover.sort_by(|&i, &j| row[j].1.total_cmp(&row[i].1));
    for &idx in &non_cover {
        let (var, weight) = row[idx];
        let z = match knapsack_max_value(&lift_set, b - weight) {
            Some(z) => z,
            None => beta_0, // x_k=1 alone infeasible ⇒ clamp to β_0 (valid)
        };
        let alpha = (beta_0 - z).max(0);
        if alpha > 0 {
            coeff.push((var, alpha));
            lift_set.push(KItem { weight, coeff: alpha });
        }
    }

    // Ge form: Σ β_j x_j ≤ β_0 ⇒ −Σ β_j x_j ≥ −β_0.
    let mut g = vec![0.0; lp.num_vars];
    for &(var, c) in &coeff {
        g[var] -= c as f64;
    }
    let rhs = -(beta_0 as f64);

    // Release-mode safety net (size-bounded): drop the cut if the exhaustive
    // support check fails, so a future lifting-coefficient regression can never
    // ship a cut that removes an integer-feasible point. A no-op while the
    // exact-lifting proof holds.
    if !lifted_cover_is_valid(row, b, &coeff, beta_0) {
        return None;
    }

    finalize_cut(g, rhs, x_star, frac_tol)
}

/// Exhaustively verify the lifted inequality `Σ β_j x_j ≤ β_0` over `{0,1}` on the
/// cut support: every support assignment satisfying the knapsack row must satisfy
/// the cut. Variables outside the support are 0 (their worst case for the cut).
fn lifted_cover_is_valid(row: &[(usize, f64)], b: f64, coeff: &[(usize, i64)], beta_0: i64) -> bool {
    if coeff.len() > KNAPSACK_VALIDITY_BRUTE_FORCE_LIMIT {
        return true; // rely on the exact-lifting structural argument
    }
    // Weight of each support variable in the knapsack row.
    let weight_of = |var: usize| -> f64 {
        row.iter().find(|&&(j, _)| j == var).map_or(0.0, |&(_, w)| w)
    };
    let n = coeff.len();
    for mask in 0u64..(1u64 << n) {
        let mut weight = 0.0_f64;
        let mut lhs = 0i64;
        for (bit, &(var, c)) in coeff.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                weight += weight_of(var);
                lhs += c;
            }
        }
        if weight <= b + ZERO_TOL && lhs > beta_0 {
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

    /// Knapsack `4x1 + 3x2 + 3x3 + 3x4 ≤ 8`, all binary, max Σ x.
    /// Minimal cover {2,3,4} (9>8) ⇒ base x2+x3+x4 ≤ 2; up-lifting x1 (weight 4)
    /// yields α_1 = 1, giving the lifted cut x1+x2+x3+x4 ≤ 2.
    fn knap() -> (MilpProblem, Vec<bool>) {
        let rows = [0, 0, 0, 0];
        let cols = [0, 1, 2, 3];
        let vals = [4.0, 3.0, 3.0, 3.0];
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, 1, 4).unwrap();
        let lp = LpProblem::new_general(
            vec![-1.0, -1.0, -1.0, -1.0],
            a,
            vec![8.0],
            vec![ConstraintType::Le],
            vec![(0.0, 1.0); 4],
            None,
        )
        .unwrap();
        let milp = MilpProblem::new(lp, vec![0, 1, 2, 3]).unwrap();
        let mask = crate::mip::integer_mask(4, &[0, 1, 2, 3]);
        (milp, mask)
    }

    fn lp_root(lp: &LpProblem) -> crate::problem::SolverResult {
        crate::mip::cuts::solve_cut_lp(lp, &crate::options::SolverOptions::default(), None)
    }

    /// **Sentinel — lifting + separation:** a cut is generated, it cuts the LP
    /// optimum, and the non-cover variable x1 carries a lifted coefficient (the
    /// generator did more than the basic cover cut).
    #[test]
    fn lifted_cover_separates_and_lifts() {
        let (milp, mask) = knap();
        let root = lp_root(&milp.lp);
        assert_eq!(root.status, crate::problem::SolveStatus::Optimal);
        let x_star = &root.solution;
        let cuts = generate_lifted_knapsack_cover_cuts(&milp.lp, &mask, x_star);
        assert!(!cuts.is_empty(), "lifted cover cut must be generated");

        let violated = cuts.iter().any(|c| {
            let lhs: f64 = c.coeffs.iter().zip(x_star).map(|(&g, &x)| g * x).sum();
            lhs < c.rhs - 1e-9
        });
        assert!(violated, "a lifted cover cut must violate the LP optimum");

        // x1 (var 0) is the non-cover variable; its coefficient must be lifted.
        let lifted = cuts.iter().any(|c| c.coeffs[0].abs() > 0.5);
        assert!(lifted, "up-lifting must assign x1 a nonzero coefficient");
    }

    /// **Sentinel — validity:** no integer-feasible point is removed.
    #[test]
    fn lifted_cover_valid_for_all_integer_points() {
        let (milp, mask) = knap();
        let x_star = lp_root(&milp.lp).solution;
        let cuts = generate_lifted_knapsack_cover_cuts(&milp.lp, &mask, &x_star);
        assert!(!cuts.is_empty());
        for bits in 0u32..16 {
            let x: Vec<f64> = (0..4).map(|i| ((bits >> i) & 1) as f64).collect();
            let weight = 4.0 * x[0] + 3.0 * x[1] + 3.0 * x[2] + 3.0 * x[3];
            if weight > 8.0 + 1e-9 {
                continue;
            }
            for (k, c) in cuts.iter().enumerate() {
                let lhs: f64 = c.coeffs.iter().zip(&x).map(|(&g, &xi)| g * xi).sum();
                assert!(
                    lhs >= c.rhs - 1e-9,
                    "lifted cover cut {k} removes integer-feasible {x:?}: lhs={lhs} < rhs={}",
                    c.rhs
                );
            }
        }
    }

    /// The value-indexed knapsack DP matches a brute-force oracle.
    #[test]
    fn knapsack_dp_matches_bruteforce() {
        let items = vec![
            KItem { weight: 3.0, coeff: 1 },
            KItem { weight: 3.0, coeff: 1 },
            KItem { weight: 3.0, coeff: 1 },
            KItem { weight: 4.0, coeff: 2 },
        ];
        for cap_i in 0..=14 {
            let cap = cap_i as f64;
            let dp = knapsack_max_value(&items, cap).unwrap();
            let mut brute = 0i64;
            for mask in 0u32..(1 << items.len()) {
                let mut w = 0.0;
                let mut v = 0i64;
                for (bit, it) in items.iter().enumerate() {
                    if mask & (1 << bit) != 0 {
                        w += it.weight;
                        v += it.coeff;
                    }
                }
                if w <= cap + 1e-9 {
                    brute = brute.max(v);
                }
            }
            assert_eq!(dp, brute, "DP != brute force at cap={cap}");
        }
        assert_eq!(knapsack_max_value(&items, -1.0), None);
    }
}
