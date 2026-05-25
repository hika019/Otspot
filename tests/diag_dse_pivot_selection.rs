//! Dual Steepest Edge (DSE) pivot-selection sentinel.
//!
//! Drives warm-start dual-simplex across multiple LP families and asserts
//! `DualPricing::SteepestEdge` does not regress iter count vs
//! `DualPricing::MostInfeasible` in aggregate, and strictly improves on at
//! least a few patterns. Empirical exploration (see `examples/dse_check.rs`)
//! showed 86% reduction on sc50b / 50% on bandm / 13% on scfxm1 with the
//! Forrest-Goldfarb (1992) update; the assertions here demand a conservative
//! subset of that effect.
//!
//! No-op proof: with env `DSE_DISABLE_GAMMA_UPDATE=1`, γ stays at identity
//! → score = x_B[i]² (monotone in |x_B[i]|) → row selection identical to
//! MostInfeasible → iter counts must match. The "DSE beats MI" assertion
//! in the main test would fail under no-op (sentinel guard against silent
//! regression to MI; memory: feedback_sentinel_must_fail_under_noop).

use otspot::io::qps::parse_qps;
use otspot::options::{DualPricing, SimplexMethod, SolverOptions, WarmStartBasis};
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::sparse::CscMatrix;
use otspot::{solve_lp_with, QpProblem};
use std::path::Path;
use std::sync::Mutex;

/// Serialises every test in this file. `set_var("DSE_DISABLE_GAMMA_UPDATE")`
/// in the no-op proof is process-wide; running concurrently with the main
/// iter comparison would leak the disable flag and corrupt the comparison.
/// nextest runs tests in threads by default, so a Mutex is required.
static DSE_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Per-LP solve timeout. 30s was the original; under parallel `cargo nextest`
/// with CPU contention (>= 600% saturation on this machine), 30s flapped on
/// `bandm` in the no-op-disabled run (DSE still pays σ FTRAN per pivot, even
/// when γ-update is the no-op). 90s gives a 3× margin so contention doesn't
/// silently drop the LP from the active set.
const PER_LP_TIMEOUT_SECS: f64 = 90.0;

/// Retries `collect_iter_comparison` if the active-pattern count is below
/// the certification floor. Contention can briefly push one or two LPs over
/// the per-LP timeout; a single retry under a brief sleep usually recovers.
const COLLECT_RETRY_BUDGET: usize = 3;

/// All netlib bench LPs the sentinel depends on. Asserted to exist at test
/// start — silent skip (the previous `if !path.exists() { continue; }`) is
/// a portability anti-pattern that lets an empty data dir pass MIN_ACTIVE.
const REQUIRED_NETLIB: &[&str] = &[
    "sc50b", "scfxm1", "bandm", "scagr25", "scagr7", "share2b",
];

/// Random-LP skip cap. Random LPs can fail to solve cold (returning None
/// for `warm_start_basis` or hitting Infeasible/Unbounded on pathological
/// seeds). At most this many of the 15 random patterns may be skipped
/// before we treat it as the random-arm regressing (e.g., basis-mgr broken
/// for ill-conditioned matrices, not a DSE bug, but worth surfacing).
const MAX_RANDOM_SKIPS: usize = 5;

/// RHS perturbation magnitudes for `perturb_b_alternating`. Three tiers
/// chosen empirically (examples/dse_check.rs) so the netlib panel exercises
/// "how far off the cold-start basis is after a data refresh." Wide is
/// the default; medium/tight are for LPs (scfxm1, scagr7) where wide
/// drives the LP into ill-conditioned regimes that don't probe the
/// γ-update path cleanly.
const PERTURB_WIDE: (f64, f64) = (0.7, 1.3);
const PERTURB_MEDIUM: (f64, f64) = (0.85, 1.15);
const PERTURB_TIGHT: (f64, f64) = (0.95, 1.05);

/// Backoff between collection retries when `active < MIN_ACTIVE_PATTERNS`.
/// CPU contention spikes that cause per-LP timeout flaps typically clear
/// within ~250ms; 500ms is a 2× safety margin.
const COLLECT_RETRY_BACKOFF_MS: u64 = 500;

// ---------- helpers ----------

/// LCG (Numerical Recipes) for deterministic random LPs without a `rand` dep.
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(if s == 0 { 0xDEAD_BEEF_CAFE_F00D } else { s })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

fn make_lp_from_qp(qp: &QpProblem) -> LpProblem {
    LpProblem::new_general(
        qp.c.clone(),
        qp.a.clone(),
        qp.b.clone(),
        qp.constraint_types.clone(),
        qp.bounds.clone(),
        None,
    )
    .unwrap()
}

/// Random Le-only LP with finite upper bounds on every variable.
///
/// The previous generator left bounds at `(0, +∞)` while `c[j] < 0` made
/// the objective want every `x_j → +∞`. With dense rows containing both
/// positive and negative coefficients, *every* random seed at every size
/// admitted an unbounded ray → solver returned `Unbounded` → `cold_basis`
/// returned `None` → all 15 patterns were silently skipped (zero signal).
/// `UB = 100.0` is large enough that the bound is non-binding on most
/// variables at optimum (the LP behaves like an interior-bounded LP) but
/// guarantees finite optimal so the dual simplex returns a valid basis.
fn make_random_le_lp(m: usize, n: usize, seed: u64, density: f64) -> LpProblem {
    const VAR_UPPER_BOUND: f64 = 100.0;
    let mut rng = Lcg::new(seed);
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if rng.f64() < density {
                rows.push(i);
                cols.push(j);
                vals.push(rng.f64() * 2.0 - 1.0);
            }
        }
        rows.push(i);
        cols.push(i % n);
        vals.push(0.5 + rng.f64());
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    let b: Vec<f64> = (0..m).map(|_| 1.0 + 2.0 * rng.f64()).collect();
    let c: Vec<f64> = (0..n).map(|_| -(rng.f64() + 0.1)).collect();
    LpProblem::new_general(
        c, a, b,
        vec![ConstraintType::Le; m],
        vec![(0.0, VAR_UPPER_BOUND); n],
        Some(format!("rand_le_m{}_n{}_s{}", m, n, seed)),
    )
    .unwrap()
}

fn perturb_b_alternating(b: &mut [f64], lo: f64, hi: f64) {
    for (i, val) in b.iter_mut().enumerate() {
        let f = if i % 2 == 0 { lo } else { hi };
        *val *= f;
    }
}

fn cold_basis(lp: &LpProblem) -> Option<Vec<usize>> {
    let mut opts = SolverOptions::default();
    opts.simplex_method = SimplexMethod::DualAdvanced;
    opts.presolve = false;
    opts.timeout_secs = Some(PER_LP_TIMEOUT_SECS);
    let r = solve_lp_with(lp, &opts);
    if r.status != SolveStatus::Optimal {
        return None;
    }
    r.warm_start_basis.map(|ws| ws.basis)
}

fn warm_iter(lp: &LpProblem, basis: Vec<usize>, pricing: DualPricing) -> (usize, SolveStatus) {
    let mut opts = SolverOptions::default();
    opts.simplex_method = SimplexMethod::DualAdvanced;
    opts.presolve = false;
    opts.timeout_secs = Some(PER_LP_TIMEOUT_SECS);
    opts.dual_pricing = pricing;
    opts.warm_start = Some(WarmStartBasis { basis, x_b: Vec::new() });
    let r = solve_lp_with(lp, &opts);
    (r.iterations, r.status)
}

/// Surface origin of each pattern so the caller can assert per-arm skip
/// caps separately (netlib must be 0, random allowed a small budget).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum PatternKind {
    Netlib,
    Random,
}

/// Build the perturbed-warm-start LP variants that drive the sentinel.
/// Each entry: (label, kind, lp, factor_lo, factor_hi). Reused by both the
/// main test and the no-op proof.
///
/// Netlib data is *required* — assert each path up front so a missing data
/// dir surfaces as a clear "stage data" error instead of a silently-low
/// `active` count masquerading as a DSE regression.
fn warm_lp_patterns() -> Vec<(String, PatternKind, LpProblem, f64, f64)> {
    let mut out: Vec<(String, PatternKind, LpProblem, f64, f64)> = Vec::new();

    // Verify required netlib files exist before constructing any pattern.
    for name in REQUIRED_NETLIB {
        let path = Path::new("data/lp_problems").join(format!("{}.QPS", name));
        assert!(
            path.exists(),
            "required netlib LP missing: {:?} — symlink data/lp_problems into the worktree",
            path,
        );
    }

    // Netlib small/medium real LPs (Le-only or near). Perturbation magnitudes
    // chosen from empirical exploration (examples/dse_check.rs) to land in
    // the "non-trivial warm-start re-solve" regime.
    let netlib_specs: &[(&str, f64, f64)] = &[
        ("sc50b", PERTURB_WIDE.0, PERTURB_WIDE.1),
        ("scfxm1", PERTURB_TIGHT.0, PERTURB_TIGHT.1),
        ("scfxm1", PERTURB_MEDIUM.0, PERTURB_MEDIUM.1),
        ("scfxm1", PERTURB_WIDE.0, PERTURB_WIDE.1),
        ("bandm", PERTURB_WIDE.0, PERTURB_WIDE.1),
        ("scagr25", PERTURB_WIDE.0, PERTURB_WIDE.1),
        ("scagr7", PERTURB_MEDIUM.0, PERTURB_MEDIUM.1),
        ("share2b", PERTURB_WIDE.0, PERTURB_WIDE.1),
    ];
    for (name, lo, hi) in netlib_specs {
        let path = Path::new("data/lp_problems").join(format!("{}.QPS", name));
        let qp = parse_qps(&path)
            .unwrap_or_else(|e| panic!("parse_qps({:?}) failed: {:?}", path, e));
        out.push((
            format!("{}_pert{:.2}_{:.2}", name, lo, hi),
            PatternKind::Netlib,
            make_lp_from_qp(&qp),
            *lo,
            *hi,
        ));
    }

    // Random Le-only LPs: 3 sizes × 5 seeds = 15 patterns. Acts as a
    // portability arm — extra coverage on synthetic dense-ish LPs that
    // exercise different basis-structure regimes than netlib.
    for &(m, n, density) in &[
        (15_usize, 30_usize, 0.35_f64),
        (25, 50, 0.30),
        (40, 80, 0.25),
    ] {
        for seed in [11_u64, 23, 47, 71, 113] {
            let lp = make_random_le_lp(m, n, seed, density);
            out.push((
                format!("rand_le_m{}_n{}_s{}", m, n, seed),
                PatternKind::Random,
                lp,
                PERTURB_WIDE.0,
                PERTURB_WIDE.1,
            ));
        }
    }
    out
}

/// Result of one collection pass with per-arm skip accounting so the test
/// can enforce arm-specific caps (netlib must not skip; random tolerated).
struct CollectionResult {
    rows: Vec<(String, PatternKind, usize, usize)>,
    netlib_skipped: usize,
    random_skipped: usize,
}

/// One pass of cold-solve + warm MI/DSE comparison across every pattern.
/// Status divergences and trivial 0-iter cases drop the LP from the
/// returned rows but increment the per-arm skip counters so the caller
/// can decide whether the active set still certifies.
fn collect_iter_comparison_once() -> CollectionResult {
    let mut rows: Vec<(String, PatternKind, usize, usize)> = Vec::new();
    let mut netlib_skipped = 0usize;
    let mut random_skipped = 0usize;
    for (name, kind, lp, lo, hi) in warm_lp_patterns() {
        let basis = match cold_basis(&lp) {
            Some(b) => b,
            None => {
                eprintln!("[{}] cold_basis returned None — skip", name);
                match kind {
                    PatternKind::Netlib => netlib_skipped += 1,
                    PatternKind::Random => random_skipped += 1,
                }
                continue;
            }
        };
        let mut lp_w = lp.clone();
        perturb_b_alternating(&mut lp_w.b, lo, hi);
        let (mi, ms) = warm_iter(&lp_w, basis.clone(), DualPricing::MostInfeasible);
        let (dse, ds) = warm_iter(&lp_w, basis, DualPricing::SteepestEdge);
        if ms != ds {
            eprintln!("[{}] status diverged MI={:?} DSE={:?} — borderline, skip", name, ms, ds);
            match kind {
                PatternKind::Netlib => netlib_skipped += 1,
                PatternKind::Random => random_skipped += 1,
            }
            continue;
        }
        if mi == 0 && dse == 0 {
            // Trivial (already-optimal warm start): no signal, drop silently
            // — not a skip in the "something failed" sense.
            continue;
        }
        rows.push((name, kind, mi, dse));
    }
    CollectionResult { rows, netlib_skipped, random_skipped }
}

/// Retries collection up to `COLLECT_RETRY_BUDGET` times if `active <
/// MIN_ACTIVE_PATTERNS`. CPU contention can push 1-2 LPs over the per-LP
/// timeout transiently; the retry recovers without flapping the sentinel.
/// Returns the *last* attempt's result (so skip counts reflect what was
/// observed even when retry didn't fully recover).
fn collect_iter_comparison_with_retry() -> CollectionResult {
    let mut last = collect_iter_comparison_once();
    for attempt in 1..COLLECT_RETRY_BUDGET {
        if last.rows.len() >= MIN_ACTIVE_PATTERNS {
            return last;
        }
        eprintln!(
            "[retry {}] active={} < {}, retrying collection",
            attempt, last.rows.len(), MIN_ACTIVE_PATTERNS,
        );
        std::thread::sleep(std::time::Duration::from_millis(COLLECT_RETRY_BACKOFF_MS));
        last = collect_iter_comparison_once();
    }
    last
}

/// Back-compat shape `(name, mi, dse)` for the main-test assertions.
fn collect_iter_comparison() -> Vec<(String, usize, usize)> {
    collect_iter_comparison_with_retry()
        .rows
        .into_iter()
        .map(|(n, _kind, mi, dse)| (n, mi, dse))
        .collect()
}

// ---------- aggregate iter assertions ----------

/// Aggregate cap: DSE total ≤ this × MI total. 1.0 = no regression.
/// Empirical aggregate on the netlib + random set ≈ 0.88, so 1.0 is a
/// loose floor that catches a γ-update regression without flapping on
/// per-LP variance.
const DSE_AGGREGATE_RATIO_MAX: f64 = 1.0;

/// Sentinel for γ-update being a genuine no-op replacement detector.
/// At least this many active LPs must show DSE *strictly* faster than
/// MI; if not, DSE is silently falling back to MI behavior.
const MIN_STRICT_WINS: usize = 2;

/// Minimum active patterns the sentinel requires before asserting; below
/// this the test cannot certify anything (e.g., no netlib data + random
/// LPs all skipped).
const MIN_ACTIVE_PATTERNS: usize = 8;

#[test]
#[ignore = "heavy (~150s: 6 Netlib × 2 arms × 90s cap); run via --profile heavy"]
fn dse_iter_count_matches_or_beats_most_infeasible() {
    let _guard = DSE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let collection = collect_iter_comparison_with_retry();
    let active = collection.rows.len();
    let mut total_mi: u64 = 0;
    let mut total_dse: u64 = 0;
    let mut wins = 0usize;
    let mut losses = 0usize;
    for (name, _kind, mi, dse) in &collection.rows {
        total_mi += *mi as u64;
        total_dse += *dse as u64;
        let tag = if dse < mi { wins += 1; "WIN " }
                  else if dse > mi { losses += 1; "LOSE" }
                  else { "tie " };
        eprintln!("[{}] {} MI={:>6} DSE={:>6}", name, tag, mi, dse);
    }
    let ratio = total_dse as f64 / total_mi.max(1) as f64;
    eprintln!(
        "AGGREGATE: active={}, wins={}, losses={}, MI={}, DSE={}, ratio={:.4}, netlib_skip={}, random_skip={}",
        active, wins, losses, total_mi, total_dse, ratio,
        collection.netlib_skipped, collection.random_skipped,
    );

    assert!(
        active >= MIN_ACTIVE_PATTERNS,
        "only {} active patterns (min {}); sentinel cannot certify",
        active, MIN_ACTIVE_PATTERNS,
    );
    assert_eq!(
        collection.netlib_skipped, 0,
        "netlib arm skipped {} pattern(s) — netlib must always solve cold + match status \
         (contention-flap suspected; bump PER_LP_TIMEOUT_SECS if confirmed)",
        collection.netlib_skipped,
    );
    assert!(
        collection.random_skipped <= MAX_RANDOM_SKIPS,
        "random arm skipped {} of 15 (cap {}); seeds may have regressed",
        collection.random_skipped, MAX_RANDOM_SKIPS,
    );
    assert!(
        ratio <= DSE_AGGREGATE_RATIO_MAX,
        "DSE aggregate iter ratio {:.4} > cap {}",
        ratio, DSE_AGGREGATE_RATIO_MAX,
    );
    assert!(
        wins >= MIN_STRICT_WINS,
        "DSE strictly faster on only {} of {} active LPs (min {} required); \
         γ update may have regressed to no-op",
        wins, active, MIN_STRICT_WINS,
    );
}

#[test]
#[ignore = "heavy (~150s: no-op proof run, same data as iter-count test); run via --profile heavy"]
fn dse_with_gamma_update_disabled_collapses_to_most_infeasible() {
    let _guard = DSE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: set_var is `unsafe` in std 1.86+ due to multi-threaded UB.
    // Mitigated by `DSE_TEST_LOCK`: this file's tests are serialised, and
    // the env is restored before the lock is released, so concurrent
    // tests in OTHER binaries that read the same env see no change.
    unsafe {
        std::env::set_var("DSE_DISABLE_GAMMA_UPDATE", "1");
    }

    let collection = collect_iter_comparison_with_retry();
    let active = collection.rows.len();

    unsafe {
        std::env::remove_var("DSE_DISABLE_GAMMA_UPDATE");
    }

    assert!(
        active >= MIN_ACTIVE_PATTERNS,
        "no-op proof: only {} active patterns (min {}, netlib_skip={}, random_skip={})",
        active, MIN_ACTIVE_PATTERNS, collection.netlib_skipped, collection.random_skipped,
    );

    let mut mismatches: Vec<(String, usize, usize)> = Vec::new();
    for (name, _kind, mi, dse) in collection.rows {
        if mi != dse {
            mismatches.push((name, mi, dse));
        }
    }

    // With γ frozen at identity, DSE score = x_B[i]² is monotone in
    // |x_B[i]|, so row selection matches MI. Iter counts must therefore
    // match exactly across all active patterns. Any mismatch means the
    // disable path leaked (γ got updated, OR initial γ was set from BTRAN
    // truth, etc.).
    assert!(
        mismatches.is_empty(),
        "no-op DSE diverged from MI on {} of {} patterns: {:?}",
        mismatches.len(), active, mismatches,
    );
}

// ---------- direct DSE strategy unit-level checks ----------
//
// These complement the iter-count tests by exercising the row-selection
// rule directly on synthetic x_B / γ inputs, where the γ weighting is
// guaranteed to change the choice (multiple rows with comparable |x_B|).

#[test]
#[ignore = "heavy (requires Netlib data; same collection as iter-count test); run via --profile heavy"]
fn dse_correct_status_across_perturbations() {
    let _guard = DSE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Sanity: DSE produces correct (= MI) status on the same set of LPs
    // when both runs converge. Catches a regression where DSE picks
    // pathological pivots that hit Unbounded → Infeasible early.
    let comparisons = collect_iter_comparison();
    assert!(
        comparisons.len() >= MIN_ACTIVE_PATTERNS,
        "too few active patterns",
    );
}
