//! Postsolve stack, original-space certificate mapping, and the independent
//! re-verification gate for the tier-1 conic presolve.
//!
//! Contract from the research report §5 (a): every reduction must map the
//! primal `x`, duals `(y, z)`, the improving ray, and the Farkas certificate
//! back to the *original* problem and re-verify them there — the same-shape
//! gate `Equilibrator::unscale_result` (conic/mod.rs) applies to equilibration
//! (`verify_infeas_cert` / `verify_primal_ray` are reused verbatim for the
//! certificates). A mapping that cannot be re-verified makes the caller
//! discard the presolved result and re-solve the original problem, so a
//! postsolve bug can never fabricate a status or a certificate.
//!
//! Dual recovery is LIFO replay of the elimination stack, mirroring the LP
//! presolve (`presolve/postsolve.rs` `LinearSubstitution`): each eliminated
//! column `j` provides exactly one stationarity equation
//! `c_j + Σ_i A_ij y_i + Σ_k G_kj z_k = 0` in the coordinates current at the
//! elimination time, and every dual that equation references belongs to a row
//! eliminated *later* (replayed earlier) or survives into the reduced problem
//! (already mapped). Bound-layer rows dropped with a fixed variable recover
//! their (nonnegative) orthant duals from the same equation; dominated
//! duplicate bounds drop with dual `0`.

use crate::conic::cone::{self, Blocks};
use crate::conic::kkt;
use crate::conic::{ConicProblem, ConicResult};
use crate::problem::SolveStatus;
use crate::tolerances::ZERO_TOL;

/// Slack applied to `ConicOptions::tol` for the original-space re-verification
/// gate: LIFO back-substitution chains accumulate rounding of order
/// `chain_len * eps_mach` per reconstructed quantity, so the gate runs at
/// `tol * VERIFY_KKT_SLACK` (still `<= 1e-4` for the bench tol `1e-6`) —
/// loose enough for reconstruction noise, tight enough to reject a mis-mapped
/// certificate (which is off by `O(1)`, not by `1e-4`).
pub(crate) const VERIFY_KKT_SLACK: f64 = 100.0;

/// One absorbed domain-identity row: the original `G` row `k` (`orig` is
/// `None` only for rows materialized from input `var_bounds`, which have no
/// original identity to restore) with `g_coef·x_j + s_k = rhs` and the bound
/// direction `is_lower` (`g_coef < 0` means `x_j >= rhs/g_coef`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AbsorbedSnapshot {
    /// Workspace `G` row id (re-materialization writes the transformed row
    /// back into this slot, preserving dual identity).
    pub(crate) g_row: usize,
    /// Original problem row id, if any.
    pub(crate) orig_row: Option<usize>,
    pub(crate) g_coef: f64,
    pub(crate) rhs: f64,
    pub(crate) is_lower: bool,
}

impl AbsorbedSnapshot {
    /// Bound value `x_j (=|>|=) bound`.
    fn bound(&self) -> f64 {
        self.rhs / self.g_coef
    }
}

/// The two absorbed bounds currently held for one column (at most one per
/// direction; tighter always wins).
#[derive(Default, Clone)]
pub(super) struct BoundLayer {
    lower: Option<AbsorbedSnapshot>,
    upper: Option<AbsorbedSnapshot>,
}

/// Outcome of inserting an absorbed bound into a [`BoundLayer`].
pub(super) enum InsertOutcome {
    /// The row became the column's active bound in its direction; the
    /// `superseded` snapshot (if any) was dominated and drops with dual `0`.
    Absorbed {
        superseded: Option<AbsorbedSnapshot>,
    },
    /// An existing bound in this direction is at least as tight: the new row
    /// is implied and drops with dual `0` (never enters the layer).
    RedundantSelf,
    /// The insertion (or the existing opposite bound) proves `lb > ub`; the
    /// payload is the witness bound row on the opposite side.
    Conflict(AbsorbedSnapshot),
}

impl BoundLayer {
    /// Current `lb` / `ub` (infinite when the direction is unbounded).
    fn lb_ub(&self) -> (f64, f64) {
        (
            self.lower.map_or(f64::NEG_INFINITY, |s| s.bound()),
            self.upper.map_or(f64::INFINITY, |s| s.bound()),
        )
    }

    /// Insert candidate bound `val` (direction `is_lower`).
    pub(super) fn insert(
        &mut self,
        snap: AbsorbedSnapshot,
        val: f64,
        is_lower: bool,
    ) -> InsertOutcome {
        let (lb0, ub0) = self.lb_ub();
        let (lb, ub) = if is_lower {
            (lb0.max(val), ub0)
        } else {
            (lb0, ub0.min(val))
        };
        if lb > ub + ZERO_TOL {
            let opposite = if is_lower {
                self.upper.expect("conflicting ub exists")
            } else {
                self.lower.expect("conflicting lb exists")
            };
            return InsertOutcome::Conflict(opposite);
        }
        let existing = if is_lower { self.lower } else { self.upper };
        // The new bound wins only when strictly tighter than the incumbent
        // (equal and weaker bounds drop the newcomer as implied).
        let dominates = match existing {
            Some(old) => (is_lower && val > old.bound()) || (!is_lower && val < old.bound()),
            None => true,
        };
        if !dominates {
            return InsertOutcome::RedundantSelf;
        }
        if is_lower {
            self.lower = Some(snap);
        } else {
            self.upper = Some(snap);
        }
        InsertOutcome::Absorbed {
            superseded: existing,
        }
    }

    /// `Some(val)` when the layer fixes the column (`lb == ub` within scale
    /// tolerance), value clamped into the bounds.
    pub(super) fn fixed_value(&self) -> Option<f64> {
        let (lb, ub) = self.lb_ub();
        if lb.is_finite() && ub.is_finite() && (ub - lb).abs() <= ZERO_TOL * (1.0 + lb.abs()) {
            Some(((lb + ub) / 2.0).clamp(lb, ub))
        } else {
            None
        }
    }

    /// Removes and returns the active bound snapshots (used at elimination).
    pub(super) fn take(&mut self) -> Vec<AbsorbedSnapshot> {
        let mut out = Vec::new();
        if let Some(s) = self.lower.take() {
            out.push(s);
        }
        if let Some(s) = self.upper.take() {
            out.push(s);
        }
        out
    }

    /// The active bound snapshots without removing them.
    pub(super) fn peek(&self) -> Vec<AbsorbedSnapshot> {
        self.lower.into_iter().chain(self.upper).collect()
    }
}

/// One recorded reduction, replayed LIFO by [`postsolve_result`].
pub(super) enum ConicPostsolveStep {
    /// Column `col` eliminated. `piv_row` is `Some` for the equality-pivot
    /// path (the row is gone and its dual is recovered) and `None` for the
    /// fixed-variable path (`rhs` is the fixed value, `pivot = 1`).
    /// `others` are the pivot row's other entries at elimination time
    /// (`(original column, coefficient)`); `col_a` / `col_g` are the
    /// elimination-time column snapshots for dual recovery; `bounds` holds
    /// the absorbed bound rows that were dropped with the fixed variable
    /// (pivot-path bounds were re-materialized as active rows instead and
    /// ride the reduced problem's row map).
    Eliminated {
        col: usize,
        piv_row: Option<usize>,
        pivot: f64,
        rhs: f64,
        others: Vec<(usize, f64)>,
        col_a: Vec<(usize, f64)>,
        col_g: Vec<(usize, f64)>,
        c_orig: f64,
        bounds: Vec<AbsorbedSnapshot>,
    },
    /// A dominated duplicate bound row: no dual to recover (`z = 0`), but the
    /// stack keeps it for audit and future dominance-recovery extensions.
    DroppedBound { orig_row: usize },
    /// An equality row reduced to `0 = 0`: removed, dual fixed at `0` (free
    /// coordinate — see `Workspace::simplify_zero_rows`).
    RedundantEqRow { orig_row: usize },
}

/// Workspace → original index maps captured when the reduced problem is
/// finalized (`finalize` in `super::mod` semantics live there; the maps are
/// built there and consumed here).
pub(super) struct Maps {
    /// `red column -> original column`.
    pub(super) col: Vec<usize>,
    /// `red A row -> original A row`.
    pub(super) a: Vec<usize>,
    /// `red G row -> original G row` (`None` for rows materialized from input
    /// `var_bounds`, which have no original row).
    pub(super) g: Vec<Option<usize>>,
}

/// Replay mode.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Primal solution: eliminated column reconstructed with the constant
    /// (`x_j = (rhs − Σ a·x)/pivot`).
    Primal,
    /// Improving ray: same recurrence with the constant zeroed (recession
    /// directions ignore offsets; fixed columns get `d_j = 0`).
    Ray,
    /// Dual solution with cost terms (Optimal stationarity).
    Dual,
    /// Dual solution without cost terms (Farkas: homogeneous
    /// `A^T y + G^T z = 0`).
    Farkas,
}

/// Reconstructs the original-space arrays from a reduced-space solve.
/// Returns the mapped result, or `None` when the original-space
/// re-verification gate rejects it (caller must fall back to the
/// presolve-free solve).
pub(super) fn postsolve_result(
    problem: &ConicProblem,
    reduced: &ConicProblem,
    maps: &Maps,
    stack: &[ConicPostsolveStep],
    res: &ConicResult,
    tol_gate: f64,
) -> Option<ConicResult> {
    let n = problem.n();
    let mut x = vec![0.0; n];
    for (rj, &xv) in res.x.iter().enumerate() {
        x[maps.col[rj]] = xv;
    }
    replay_primal(stack, &mut x, Mode::Primal);

    let mut y = vec![0.0; problem.p()];
    for (ri, &yv) in res.y.iter().enumerate() {
        y[maps.a[ri]] = yv;
    }
    let mut z = vec![0.0; problem.m()];
    for (rk, &zv) in res.z.iter().enumerate() {
        if let Some(orig) = maps.g[rk] {
            z[orig] = zv;
        }
    }
    let needs_dual = matches!(
        res.status,
        SolveStatus::Optimal | SolveStatus::Infeasible | SolveStatus::Unbounded
    );
    if needs_dual {
        let mode = if res.status == SolveStatus::Infeasible {
            Mode::Farkas
        } else {
            Mode::Dual
        };
        replay_dual(stack, &mut y, &mut z, mode);
    }

    // Slack recomputed from the ORIGINAL data (never trusted from the
    // reduced space — same independence rule as `unscale_result`'s objective
    // recomputation and the LP postsolve's `b - Ax`).
    let gx = kkt::spmv(&problem.g, &x);
    let s: Vec<f64> = problem.h.iter().zip(&gx).map(|(h, g)| h - g).collect();
    let objective = problem.c.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();

    let mut out = ConicResult {
        objective,
        x,
        y,
        z,
        s,
        ..res.clone()
    };
    out.residuals = res.residuals;

    // Independent original-space re-verification, same-shape as the
    // equilibration gate. Failed verification => caller discards this result
    // and re-solves without presolve.
    let verified = match res.status {
        SolveStatus::Optimal => {
            verify_conic_point(problem, &out.x, &out.y, &out.z, &out.s, tol_gate)
        }
        SolveStatus::Infeasible => match &res.infeas_cert {
            Some(_) => {
                let cert = (&out.y, &out.z);
                crate::conic::equil::verify_infeas_cert(problem, cert.0, cert.1, tol_gate)
            }
            None => true,
        },
        SolveStatus::Unbounded => match &res.primal_ray {
            Some(d) => {
                let mut dfull = vec![0.0; n];
                for (rj, &dv) in d.iter().enumerate() {
                    dfull[maps.col[rj]] = dv;
                }
                replay_primal(stack, &mut dfull, Mode::Ray);
                if crate::conic::equil::verify_primal_ray(problem, &dfull, tol_gate) {
                    out.primal_ray = Some(dfull);
                    true
                } else {
                    false
                }
            }
            None => true,
        },
        _ => true,
    };
    if !verified {
        return None;
    }
    // Infeasible's `x`/`y`/`z` payload: carry the mapped Farkas pair so
    // callers see it in original space (also reported via `infeas_cert`).
    if res.status == SolveStatus::Infeasible && res.infeas_cert.is_some() {
        out.infeas_cert = Some((out.y.clone(), out.z.clone()));
    }
    let _ = reduced;
    Some(out)
}

/// LIFO replay of the elimination stack for the primal side (`x` or the ray).
fn replay_primal(stack: &[ConicPostsolveStep], full: &mut [f64], mode: Mode) {
    for step in stack.iter().rev() {
        if let ConicPostsolveStep::Eliminated {
            col,
            pivot,
            rhs,
            others,
            ..
        } = step
        {
            let mut sum = 0.0;
            for &(k, a) in others {
                sum += a * full[k];
            }
            let rhs_term = if mode == Mode::Ray { 0.0 } else { *rhs };
            full[*col] = (rhs_term - sum) / pivot;
        }
    }
}

/// LIFO replay of the elimination stack for the dual side (`y`/`z`).
/// `cost` contributes only in [`Mode::Dual`] (stationarity with the objective;
/// Farkas recovery is the homogeneous system, `c` terms dropped).
fn replay_dual(stack: &[ConicPostsolveStep], y: &mut [f64], z: &mut [f64], mode: Mode) {
    let cost_on = mode == Mode::Dual;
    for step in stack.iter().rev() {
        // Audit steps carry an explicit zero dual (their multiplier is a
        // free/implied coordinate — pinning it keeps the mapped certificate
        // the *same* solution the pre-presolve IPM could have reported).
        match step {
            ConicPostsolveStep::DroppedBound { orig_row } => z[*orig_row] = 0.0,
            ConicPostsolveStep::RedundantEqRow { orig_row } => y[*orig_row] = 0.0,
            ConicPostsolveStep::Eliminated { .. } => {}
        }
        if let ConicPostsolveStep::Eliminated {
            piv_row,
            pivot,
            col_a,
            col_g,
            c_orig,
            bounds,
            ..
        } = step
        {
            let mut sum = if cost_on { *c_orig } else { 0.0 };
            for &(r, a) in col_a {
                sum += a * y[r];
            }
            for &(k, g) in col_g {
                sum += g * z[k];
            }
            // `sum + Σ_bounds g_k z_k = 0` is the column's stationarity
            // equation. With a pivot row the row dual absorbs everything
            // (equality duals are sign-free); otherwise the dropped bound
            // rows must carry it with nonnegative orthant duals: the
            // `max`-split gives `z_ub = max(R, 0)`, `z_lb = max(−R, 0)` for
            // two-sided layers and the exact quotient for a single bound
            // (whose sign is then verified by the original-space gate).
            match piv_row {
                Some(r) => {
                    y[*r] = -sum / pivot;
                }
                None if bounds.is_empty() => {}
                None => {
                    // `need = Σ_bounds g_k z_k` split with `z >= 0`: the
                    // direction whose cone sign can carry `need` takes it all
                    // (`g_ub > 0` carries `need >= 0`, `g_lb < 0` carries
                    // `need < 0`); a single bound takes the exact quotient
                    // and the original-space gate verifies its sign.
                    let need = -sum;
                    if bounds.len() == 1 {
                        let b = &bounds[0];
                        if let Some(orig) = b.orig_row {
                            z[orig] = need / b.g_coef;
                        }
                    } else {
                        for b in bounds {
                            let takes = if need >= 0.0 { !b.is_lower } else { b.is_lower };
                            if let Some(orig) = b.orig_row {
                                z[orig] = if takes { need / b.g_coef } else { 0.0 };
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Independent original-space KKT re-verification for an `Optimal` postsolved
/// point. Mirrors `ipm::solve`'s acceptance formulas (same scale-invariant
/// normalizations `1+‖b‖`, `1+‖c‖`, `1+‖h‖`) computed from the *original*
/// `A`/`G`/`b`/`h`/`c` and the postsolved `x`/`y`/`z` plus the independently
/// recomputed slack `s = h − Gx`.
pub(super) fn verify_conic_point(
    problem: &ConicProblem,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    s: &[f64],
    tol: f64,
) -> bool {
    if !x.iter().all(|v| v.is_finite())
        || !y.iter().all(|v| v.is_finite())
        || !z.iter().all(|v| v.is_finite())
    {
        return false;
    }
    let blk = Blocks::new(&problem.cone);
    let norm2 = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
    let nb = 1.0 + norm2(&problem.b);
    let nc = 1.0 + norm2(&problem.c);
    let nh = 1.0 + norm2(&problem.h);

    let ax = kkt::spmv(&problem.a, x);
    let gx = kkt::spmv(&problem.g, x);
    let ry: Vec<f64> = problem.b.iter().zip(&ax).map(|(b, a)| a - b).collect();
    let rz: Vec<f64> = problem
        .h
        .iter()
        .zip(&gx)
        .zip(s)
        .map(|((h, g), si)| g + si - h)
        .collect();
    let aty = kkt::spmtv(&problem.a, y);
    let gtz = kkt::spmtv(&problem.g, z);
    let rx: Vec<f64> = (0..problem.n())
        .map(|i| problem.c[i] + aty[i] + gtz[i])
        .collect();
    let sz: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
    let cx: f64 = problem.c.iter().zip(x).map(|(a, b)| a * b).sum();

    let pres = (norm2(&ry) / nb).max(norm2(&rz) / nh);
    let dres = norm2(&rx) / nc;
    let gap = sz / (1.0 + cx.abs());
    if !(pres <= tol && dres <= tol && gap.abs() <= tol) {
        return false;
    }
    cone::in_cone(&blk, s, tol) && cone::in_cone(&blk, z, tol)
}
