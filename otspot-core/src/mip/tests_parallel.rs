//! Parallel MILP branch-and-bound tests.
//!
//! Two obligations are checked here, both from the user-facing contract of
//! `SolverOptions::threads`:
//!
//! 1. **The budget is a cap.** A solve never runs more concurrent relaxation
//!    solves, nor touches more distinct OS threads, than `threads`. Measured
//!    by instrumenting the `Relaxation` the search actually calls
//!    ([`ThreadWatch`]) rather than by trusting the driver.
//! 2. **The answer does not depend on the budget.** `threads = 1 / 2 / 4`
//!    return the same optimum, checked against an independent oracle
//!    (exhaustive enumeration of the binary assignments) rather than against
//!    the solver's own serial output.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::{integer_mask, solve_milp, solve_mip_dispatch, MilpProblem, Relaxation};
use crate::options::{MipConfig, SolverOptions};
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use otspot_num::sparse::CscMatrix;

/// How long the first relaxation solve of a [`ThreadWatch`] run waits for a
/// second worker before giving up. Only ever paid until the rendezvous is
/// first satisfied (see [`ThreadWatch::solve`]), so it bounds the whole
/// test's overhead, not each node's.
const RENDEZVOUS_TIMEOUT: Duration = Duration::from_millis(400);

/// A [`Relaxation`] that delegates everything to a real [`MilpProblem`] while
/// recording how much of the solver is running at once.
///
/// Every trait method forwards, so the search behaves exactly as it would on
/// the bare problem; the only addition is the concurrency bookkeeping around
/// `solve`, which is where a branch-and-bound worker spends essentially all
/// of its time.
///
/// `rendezvous` makes the parallelism observation deterministic instead of
/// timing-dependent: while the observed peak is still below the target, a
/// worker inside `solve` waits for its peers, so a second worker entering
/// `solve` is *forced* to overlap with the first rather than merely being
/// likely to. Set to 0 to disable (used for the serial expectation, where
/// waiting for a peer that cannot exist would only burn the timeout).
struct ThreadWatch {
    inner: MilpProblem,
    active: AtomicUsize,
    peak: AtomicUsize,
    threads_seen: Mutex<HashSet<ThreadId>>,
    rendezvous: usize,
}

impl ThreadWatch {
    fn new(inner: MilpProblem, rendezvous: usize) -> Self {
        Self {
            inner,
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            threads_seen: Mutex::new(HashSet::new()),
            rendezvous,
        }
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    fn distinct_threads(&self) -> usize {
        self.threads_seen.lock().expect("watch mutex").len()
    }
}

impl Relaxation for ThreadWatch {
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }
    fn root_bounds(&self) -> &[(f64, f64)] {
        self.inner.root_bounds()
    }
    fn integer_vars(&self) -> &[usize] {
        self.inner.integer_vars()
    }
    fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.threads_seen
            .lock()
            .expect("watch mutex")
            .insert(std::thread::current().id());
        if self.peak.load(Ordering::SeqCst) < self.rendezvous {
            let t0 = Instant::now();
            while self.peak.load(Ordering::SeqCst) < self.rendezvous
                && t0.elapsed() < RENDEZVOUS_TIMEOUT
            {
                std::thread::yield_now();
            }
        }
        let res = self.inner.solve(bounds, opts);
        self.active.fetch_sub(1, Ordering::SeqCst);
        res
    }
    fn skip_node_presolve(&self) -> bool {
        self.inner.skip_node_presolve()
    }
    fn can_skip_repeated_lp_scaling(&self) -> bool {
        self.inner.can_skip_repeated_lp_scaling()
    }
    fn propagation_data(&self) -> Option<(&CscMatrix, &[f64], &[ConstraintType])> {
        self.inner.propagation_data()
    }
    fn separate_tree_cuts(
        &self,
        bounds: &[(f64, f64)],
        res: &SolverResult,
        mask: &[bool],
        opts: &SolverOptions,
        depth: usize,
        node_index: usize,
        max_iters: u64,
    ) -> (Option<SolverResult>, u64, u64, bool) {
        self.inner
            .separate_tree_cuts(bounds, res, mask, opts, depth, node_index, max_iters)
    }
    fn run_rins(
        &self,
        x_lp: &[f64],
        x_inc: &[f64],
        cfg: &MipConfig,
        deadline: &Option<Instant>,
        iter_budget: u64,
        opts: &SolverOptions,
    ) -> (Option<SolverResult>, u64, u64) {
        self.inner
            .run_rins(x_lp, x_inc, cfg, deadline, iter_budget, opts)
    }
    fn run_rens(
        &self,
        x_lp: &[f64],
        cfg: &MipConfig,
        deadline: &Option<Instant>,
        iter_budget: u64,
        opts: &SolverOptions,
    ) -> (Option<SolverResult>, u64, u64) {
        self.inner.run_rens(x_lp, cfg, deadline, iter_budget, opts)
    }
    fn run_local_branching(
        &self,
        x_inc: &[f64],
        cfg: &MipConfig,
        deadline: &Option<Instant>,
        iter_budget: u64,
        opts: &SolverOptions,
    ) -> (Option<SolverResult>, u64, u64) {
        self.inner
            .run_local_branching(x_inc, cfg, deadline, iter_budget, opts)
    }
}

// ---------------------------------------------------------------------------
// Instance generation + independent oracle
// ---------------------------------------------------------------------------

/// Fixed-seed linear congruential generator (Knuth MMIX constants) so every
/// generated instance — and therefore every expected value — is identical on
/// every run and machine.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn in_range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

/// A multi-dimensional binary knapsack: `min c'x` s.t. `Ax <= b`, `x ∈ {0,1}^n`
/// with `c < 0` (so the optimum genuinely fills the knapsack) and `b` at half
/// the row sum (so roughly half the assignments are feasible and the LP
/// relaxation is fractional).
fn knapsack(seed: u64, n: usize, m: usize) -> (MilpProblem, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let c: Vec<f64> = (0..n).map(|_| -(rng.in_range(1, 20) as f64)).collect();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut dense = Vec::with_capacity(m);
    let mut b = Vec::with_capacity(m);
    for i in 0..m {
        let row: Vec<f64> = (0..n).map(|_| rng.in_range(1, 20) as f64).collect();
        for (j, &v) in row.iter().enumerate() {
            rows.push(i);
            cols.push(j);
            vals.push(v);
        }
        b.push((row.iter().sum::<f64>() / 2.0).floor());
        dense.push(row);
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).expect("triplets");
    let lp = LpProblem::new_general(
        c.clone(),
        a,
        b.clone(),
        vec![ConstraintType::Le; m],
        vec![(0.0, 1.0); n],
        None,
    )
    .expect("lp");
    let problem = MilpProblem::new(lp, (0..n).collect()).expect("milp");
    (problem, c, dense, b)
}

/// Independent oracle: enumerate every binary assignment and return the best
/// feasible objective. Never consults the solver.
fn brute_force_optimum(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> f64 {
    let n = c.len();
    assert!(n <= 20, "exhaustive enumeration is only for tiny instances");
    let mut best = f64::INFINITY;
    for bits in 0u32..(1u32 << n) {
        let x: Vec<f64> = (0..n).map(|j| f64::from((bits >> j) & 1)).collect();
        let feasible = a
            .iter()
            .zip(b)
            .all(|(row, &rhs)| row.iter().zip(&x).map(|(&v, &xj)| v * xj).sum::<f64>() <= rhs);
        if !feasible {
            continue;
        }
        let obj: f64 = c.iter().zip(&x).map(|(&cj, &xj)| cj * xj).sum();
        best = best.min(obj);
    }
    best
}

fn opts_with_threads(threads: usize) -> SolverOptions {
    SolverOptions {
        threads,
        timeout_secs: Some(60.0),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 1. The thread budget is a cap
// ---------------------------------------------------------------------------

/// The parallel search must never run more concurrent relaxation solves, nor
/// touch more distinct OS threads, than `SolverOptions::threads` — and it must
/// actually use more than one (a cap that holds because nothing is parallel
/// would be worthless).
///
/// Sentinel: sizing the pool from anything other than `options.threads` (e.g.
/// `available_parallelism`) breaks the `peak <= threads` assertion; reverting
/// `solve_mip_dispatch` to always call the serial driver breaks `peak >= 2`.
#[test]
fn parallel_search_stays_within_the_thread_budget() {
    for threads in [2usize, 3, 4] {
        let (problem, ..) = knapsack(0xB0BA_CAFE, 14, 3);
        let mask = integer_mask(problem.num_vars(), problem.integer_vars());
        let watch = ThreadWatch::new(problem, 2);
        let (res, stats) = solve_mip_dispatch(
            &watch,
            &opts_with_threads(threads),
            &MipConfig::default(),
            mask,
            None,
        );
        assert_eq!(res.status, SolveStatus::Optimal, "threads={threads}");
        assert!(
            stats.nodes_processed > 1,
            "threads={threads}: the instance must actually branch (nodes={})",
            stats.nodes_processed
        );
        assert!(
            watch.peak() <= threads,
            "threads={threads}: peak concurrent relaxation solves {} exceeds the budget",
            watch.peak()
        );
        assert!(
            watch.distinct_threads() <= threads,
            "threads={threads}: {} distinct threads ran relaxations, budget is {threads}",
            watch.distinct_threads()
        );
        assert!(
            watch.peak() >= 2,
            "threads={threads}: the search never ran two relaxations at once"
        );
    }
}

/// `threads = 1` (the default) must stay strictly serial: one thread, one
/// relaxation at a time. This is what keeps the default search deterministic.
///
/// Sentinel: lowering `solve_mip_dispatch`'s `threads >= 2` guard to
/// `threads >= 1` spawns a worker and fails the thread-identity assertion.
#[test]
fn threads_eq_1_runs_everything_on_the_calling_thread() {
    let (problem, ..) = knapsack(0xB0BA_CAFE, 12, 3);
    let mask = integer_mask(problem.num_vars(), problem.integer_vars());
    let watch = ThreadWatch::new(problem, 0);
    let caller = std::thread::current().id();
    let (res, _) = solve_mip_dispatch(
        &watch,
        &opts_with_threads(1),
        &MipConfig::default(),
        mask,
        None,
    );
    assert_eq!(res.status, SolveStatus::Optimal);
    assert_eq!(watch.peak(), 1, "threads=1 must never overlap two solves");
    assert_eq!(
        watch
            .threads_seen
            .lock()
            .expect("watch mutex")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![caller],
        "threads=1 must not spawn a worker"
    );
}

/// A deterministic iteration budget (`MipConfig::max_lp_iters`, set on every
/// heuristic sub-MIP) must keep the serial driver whatever `threads` says:
/// its whole purpose is a timing-independent stopping point.
///
/// Sentinel: dropping the `cfg.max_lp_iters.is_none()` term from
/// `solve_mip_dispatch` routes this to the worker pool and fails the
/// single-thread assertion.
#[test]
fn a_deterministic_iteration_budget_forces_the_serial_driver() {
    let (problem, ..) = knapsack(0xB0BA_CAFE, 12, 3);
    let mask = integer_mask(problem.num_vars(), problem.integer_vars());
    let watch = ThreadWatch::new(problem, 0);
    let cfg = MipConfig {
        max_lp_iters: Some(50_000),
        ..MipConfig::default()
    };
    let caller = std::thread::current().id();
    let _ = solve_mip_dispatch(&watch, &opts_with_threads(4), &cfg, mask, None);
    assert_eq!(
        watch
            .threads_seen
            .lock()
            .expect("watch mutex")
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![caller],
        "max_lp_iters must pin the search to the serial driver"
    );
}

// ---------------------------------------------------------------------------
// 2. The answer does not depend on the thread budget
// ---------------------------------------------------------------------------

/// Every thread count must return the *proved* optimum of a set of generated
/// multi-dimensional knapsacks, checked against exhaustive enumeration.
///
/// This is the mis-pruning detector: a lost incumbent update or a bound
/// compared against a stale-but-too-small upper bound shows up immediately as
/// an objective above the enumerated optimum.
#[test]
fn every_thread_count_finds_the_enumerated_optimum() {
    const SEEDS: [u64; 6] = [1, 7, 19, 101, 2027, 65_537];
    for seed in SEEDS {
        let (problem, c, a, b) = knapsack(seed, 12, 3);
        let expected = brute_force_optimum(&c, &a, &b);
        assert!(
            expected.is_finite(),
            "seed={seed}: oracle found no solution"
        );
        for threads in [1usize, 2, 4] {
            let res = solve_milp(&problem, &opts_with_threads(threads), &MipConfig::default());
            assert_eq!(
                res.status,
                SolveStatus::Optimal,
                "seed={seed} threads={threads}"
            );
            assert!(
                (res.objective - expected).abs() < 1e-6,
                "seed={seed} threads={threads}: obj={} but the enumerated optimum is {expected}",
                res.objective
            );
        }
    }
}

/// Hand-computable instance: `min -3x - 2y - z` s.t. `2x + 3y + 4z <= 5`,
/// `x, y, z ∈ {0,1}`. Feasible fillings are ∅, {x}, {y}, {z}, {x,y}, {x,z};
/// the best is `{x, y}` at `-5`. Every thread count must return exactly that
/// solution, not merely that objective.
#[test]
fn parallel_search_returns_the_hand_computed_solution() {
    let a = CscMatrix::from_triplets(&[0, 0, 0], &[0, 1, 2], &[2.0, 3.0, 4.0], 1, 3).expect("a");
    let lp = LpProblem::new_general(
        vec![-3.0, -2.0, -1.0],
        a,
        vec![5.0],
        vec![ConstraintType::Le],
        vec![(0.0, 1.0); 3],
        None,
    )
    .expect("lp");
    let problem = MilpProblem::new(lp, vec![0, 1, 2]).expect("milp");
    for threads in [1usize, 2, 4] {
        let res = solve_milp(&problem, &opts_with_threads(threads), &MipConfig::default());
        assert_eq!(res.status, SolveStatus::Optimal, "threads={threads}");
        assert!(
            (res.objective + 5.0).abs() < 1e-6,
            "threads={threads}: obj={}",
            res.objective
        );
        let x: Vec<f64> = res.solution.iter().map(|v| v.round()).collect();
        assert_eq!(x, vec![1.0, 1.0, 0.0], "threads={threads}");
    }
}

/// An infeasible instance must be *proved* infeasible by the parallel search
/// too — never reported as "no solution found", which is what an unsound
/// termination check (a worker declaring the pool exhausted while another
/// still holds nodes) would produce.
#[test]
fn parallel_search_proves_infeasibility() {
    // 3x >= 2 and 3x <= 2 with x integer: no integer point satisfies 3x = 2.
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[3.0, 3.0], 2, 1).expect("a");
    let lp = LpProblem::new_general(
        vec![1.0],
        a,
        vec![2.0, 2.0],
        vec![ConstraintType::Ge, ConstraintType::Le],
        vec![(0.0, 10.0)],
        None,
    )
    .expect("lp");
    let problem = MilpProblem::new(lp, vec![0]).expect("milp");
    for threads in [1usize, 2, 4] {
        let res = solve_milp(&problem, &opts_with_threads(threads), &MipConfig::default());
        assert_eq!(
            res.status,
            SolveStatus::Infeasible,
            "threads={threads}: {:?}",
            res.status
        );
    }
}

/// An exhausted deadline must never be dressed up as a proof. With a
/// zero-length budget the parallel search may report anything except
/// `Optimal`, and in particular must not emit a bound-gap certificate.
///
/// Scope: the budget here is *already spent* when the solve starts, so this
/// only covers the degenerate stop — the very first `check_stop_conditions`
/// fires before any worker owns a node. It says nothing about a deadline that
/// lands mid-search, which is where held nodes and bound accounting actually
/// interact; that is
/// `an_interrupted_parallel_search_never_overstates_its_lower_bound` (end to
/// end) and `mip::parallel::tests::a_stopped_worker_returns_every_node_it_
/// still_holds` (the conservation sentinel).
#[test]
fn parallel_search_does_not_claim_optimality_after_a_timeout() {
    let (problem, ..) = knapsack(31, 14, 4);
    let opts = SolverOptions {
        threads: 4,
        timeout_secs: Some(0.0),
        ..Default::default()
    };
    let res = solve_milp(&problem, &opts, &MipConfig::default());
    assert_ne!(
        res.status,
        SolveStatus::Optimal,
        "a zero budget cannot prove optimality"
    );
    assert!(
        res.bound_gap_cert.is_none(),
        "no certificate may be issued without a completed proof"
    );
}

/// `max_nodes` bounds the *search*, not each worker: four workers must not be
/// able to spend four times the configured budget.
///
/// Sentinel: comparing `cfg.max_nodes` against a worker's private
/// `stats.nodes_processed` instead of the shared counter lets the total run
/// up to `threads x max_nodes` and fails the upper bound below.
#[test]
fn max_nodes_bounds_the_whole_parallel_search() {
    const MAX_NODES: usize = 40;
    const THREADS: usize = 4;
    let (problem, ..) = knapsack(0xFEED_1234, 16, 4);
    let cfg = MipConfig {
        max_nodes: MAX_NODES,
        ..MipConfig::default()
    };
    let (_, stats) = super::solve_milp_with_stats(&problem, &opts_with_threads(THREADS), &cfg);
    // Each worker may be inside its own node when the shared counter crosses
    // the cap, so the overshoot is bounded by the number of workers — not by
    // a multiple of the budget.
    assert!(
        stats.nodes_processed <= MAX_NODES + THREADS,
        "nodes={} exceeds max_nodes={MAX_NODES} by more than the {THREADS} in-flight nodes",
        stats.nodes_processed
    );
}

/// An interrupted parallel search must not claim a bound it did not prove.
///
/// When the search stops, every worker still holds nodes — the dive stack it
/// was descending, plus the children of the node it just finished. Those are
/// pushed back to the shared pool before the worker joins (`run_worker`'s
/// `end_dive` + `drain_from` tail), which is what makes
/// `finalize_mip_result`'s remaining lower bound cover the *whole* unexplored
/// region. Drop them and the bound is computed over a strict subset of the
/// open nodes, so it comes out too high — and a too-high lower bound is
/// exactly what turns an unfinished search into a false `Optimal`.
///
/// Checked against the enumerated optimum rather than against the solver:
/// every `Optimal` must carry both the true objective and a certificate whose
/// lower bound really does bound it. `max_nodes` forces the interruption
/// deterministically (no wall-clock dependence), and the repetitions give the
/// worker interleaving many chances to leave nodes behind.
///
/// This is an end-to-end soundness check, **not** the sentinel for the
/// node-return tail: dropped dive-stack nodes are descendants, so their bounds
/// sit above the pool minimum and the reported bound usually survives losing
/// them (verified — deleting the tail leaves this test green). The sentinel is
/// `mip::parallel::tests::a_stopped_worker_returns_every_node_it_still_holds`,
/// which checks the conservation identity directly.
#[test]
fn an_interrupted_parallel_search_never_overstates_its_lower_bound() {
    const THREADS: usize = 4;
    const REPEATS: usize = 60;
    // Small enough that the search is always cut off deep inside worker dives,
    // large enough that several workers are holding nodes when it happens.
    const NODE_CAPS: [usize; 3] = [6, 17, 40];

    let mut optimal_seen = 0usize;
    for seed in [11u64, 3607, 90_113] {
        let (problem, c, a, b) = knapsack(seed, 14, 3);
        let expected = brute_force_optimum(&c, &a, &b);
        for cap in NODE_CAPS {
            let cfg = MipConfig {
                max_nodes: cap,
                ..MipConfig::default()
            };
            for repeat in 0..REPEATS {
                let res = solve_milp(&problem, &opts_with_threads(THREADS), &cfg);
                if res.status != SolveStatus::Optimal {
                    continue;
                }
                optimal_seen += 1;
                assert!(
                    (res.objective - expected).abs() < 1e-6,
                    "seed={seed} cap={cap} repeat={repeat}: claimed Optimal at {} \
                     but the enumerated optimum is {expected}",
                    res.objective
                );
                let cert = res
                    .bound_gap_cert
                    .as_ref()
                    .expect("an Optimal MILP result carries its bound-gap certificate");
                assert!(
                    cert.lower_bound() <= expected + 1e-6,
                    "seed={seed} cap={cap} repeat={repeat}: certificate lower bound {} \
                     exceeds the true optimum {expected} — unexplored nodes were dropped",
                    cert.lower_bound()
                );
            }
        }
    }
    // Every assertion above is guarded by `status == Optimal`, so a change
    // that made the capped search never claim optimality would leave this
    // test green while checking nothing. Pin that it does happen.
    assert!(
        optimal_seen > 0,
        "no run reached Optimal under the node caps {NODE_CAPS:?}: \
         the soundness assertions never executed"
    );
}

/// A panicking worker must surface as a panic, not as a hang.
///
/// The other workers park in `SharedPool::pop_blocking` waiting for nodes that
/// the dead worker will never publish, and `std::thread::scope` will not
/// return until they do — so without `run_worker`'s `StopOnExit` guard the
/// panic turns into a deadlock. Run on a helper thread with a bounded wait so
/// that a regression fails the test instead of wedging the suite.
///
/// Sentinel: removing the `StopOnExit` guard (or its `Drop`) makes this time
/// out on the channel rather than observing the panic.
#[test]
fn a_panicking_worker_ends_the_search_instead_of_hanging() {
    /// Solve calls to serve normally before panicking. Comfortably past the
    /// root so the other workers are already parked on the shared pool.
    const PANIC_AFTER: usize = 6;
    /// A hung `thread::scope` never returns; this is how long the test waits
    /// before calling it hung. Three orders of magnitude above the solve.
    const HANG_TIMEOUT: Duration = Duration::from_secs(30);

    struct PanickingMock {
        inner: MilpProblem,
        calls: AtomicUsize,
    }

    impl Relaxation for PanickingMock {
        fn num_vars(&self) -> usize {
            self.inner.num_vars()
        }
        fn root_bounds(&self) -> &[(f64, f64)] {
            self.inner.root_bounds()
        }
        fn integer_vars(&self) -> &[usize] {
            self.inner.integer_vars()
        }
        fn solve(&self, bounds: &[(f64, f64)], opts: &SolverOptions) -> SolverResult {
            if self.calls.fetch_add(1, Ordering::SeqCst) >= PANIC_AFTER {
                panic!("injected worker failure");
            }
            self.inner.solve(bounds, opts)
        }
        fn skip_node_presolve(&self) -> bool {
            self.inner.skip_node_presolve()
        }
        fn propagation_data(&self) -> Option<(&CscMatrix, &[f64], &[ConstraintType])> {
            self.inner.propagation_data()
        }
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let (problem, ..) = knapsack(0x5EED, 16, 3);
        let mask = integer_mask(problem.num_vars(), problem.integer_vars());
        let mock = PanickingMock {
            inner: problem,
            calls: AtomicUsize::new(0),
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_mip_dispatch(
                &mock,
                &opts_with_threads(4),
                &MipConfig::default(),
                mask,
                None,
            )
        }));
        let _ = tx.send(outcome.is_err());
    });

    let panicked = rx.recv_timeout(HANG_TIMEOUT).unwrap_or_else(|_| {
        panic!("the search did not return within {HANG_TIMEOUT:?}: a worker panic deadlocked its peers")
    });
    handle
        .join()
        .expect("the driver thread itself must not die");
    assert!(
        panicked,
        "a worker panic must propagate out of the parallel search"
    );
}

/// The reported statistics must describe the whole search, not one worker's
/// share: node counts add up and the root buckets are attributed exactly once.
#[test]
fn worker_statistics_reduce_to_one_coherent_search() {
    let (problem, ..) = knapsack(4242, 14, 3);
    let (serial, serial_stats) =
        super::solve_milp_with_stats(&problem, &opts_with_threads(1), &MipConfig::default());
    let (parallel, parallel_stats) =
        super::solve_milp_with_stats(&problem, &opts_with_threads(4), &MipConfig::default());
    assert_eq!(serial.status, parallel.status);
    assert!((serial.objective - parallel.objective).abs() < 1e-6);
    assert!(
        parallel_stats.nodes_processed >= 1,
        "the parallel search must report the nodes it processed"
    );
    assert_eq!(
        serial_stats.root_lp_bound, parallel_stats.root_lp_bound,
        "the root relaxation is the same problem regardless of the thread budget"
    );
    assert!(
        parallel_stats.relaxation_time_root_ms > 0.0,
        "exactly one worker owns the root timing bucket, and it must be recorded"
    );
    assert!(
        parallel_stats.approx_bounds_bytes_per_node == serial_stats.approx_bounds_bytes_per_node,
        "per-node footprint is a property of the problem, not of the reduction"
    );
}
