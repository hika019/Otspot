//! Parallel branch-and-bound search for MILP.
//!
//! `SolverOptions::threads` workers are spawned once per solve (via
//! [`std::thread::scope`], so they borrow the problem) and share one
//! best-bound [`SharedPool`]. Each runs the *same* [`super::run_node`] body as
//! the serial driver against its own [`super::SearchState`], so the two search
//! modes cannot drift apart in what a node means; they differ only in where
//! open nodes live and how per-worker results are reduced.
//!
//! Shared state: open nodes in [`SharedPool`]; the incumbent in
//! [`SharedIncumbent`]; pseudocosts delta-merged into a `Mutex` every
//! [`PSEUDOCOST_SYNC_INTERVAL`] nodes; the node budget in an `AtomicUsize`;
//! per-worker [`MipStats`] reduced by `merge_worker` at join. Conflict clauses
//! are worker-private — see [`run_worker`], which also documents the
//! scheduling, the warm-start locality it preserves, and why the answer does
//! not depend on the exploration order.
//!
//! `threads = 1` never reaches this module (see
//! [`super::solve_mip_dispatch`]): the serial driver stays bit-identical, and
//! with it every benchmark trajectory.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use super::branch::PseudocostState;
use super::conflict::ConflictStore;
use super::node::MipNode;
use super::queue::NodeQueue;
use super::{
    finalize_mip_result, prepare_search_inputs, run_node, MipState, MipStats, NodeLoop, Relaxation,
    SearchCtx, SearchOutcome, SearchState,
};
use crate::options::{MipBranching, MipConfig, SolverOptions};
use crate::problem::{SolveStatus, SolverResult};

/// Marker for a *partial* worker-spawn failure in [`solve_mip_parallel`].
///
/// The OS can refuse a worker thread (`RLIMIT_NPROC`, a pid cgroup cap, or no
/// memory for another stack) after some workers have already started, and
/// `MAX_THREADS` validation cannot foresee it — the limit depends on the
/// machine's state at solve time, not on the option. This is reported as a
/// failed solve (`SolveStatus::ResourceExhausted` — an OS resource shortfall,
/// not a numerical breakdown), never as a silently smaller one: the thread
/// budget the caller asked for could not be honoured.
struct WorkerSpawnFailed;

#[cfg(test)]
thread_local! {
    /// Test-only fault injection for the worker-spawn loop. When set to
    /// `Some(k)`, the `k`-th (0-based) and every later `spawn_scoped` is
    /// treated as an OS spawn failure so a sentinel can reproduce a *partial*
    /// spawn without exhausting real thread limits. `None` (default) spawns
    /// normally. `thread_local` so parallel tests cannot corrupt each other's
    /// plan (same rationale as `nonconvex::RELAX_STATUS_PLAN`).
    static SPAWN_FAIL_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Nodes a worker processes between shared-pseudocost synchronisations.
///
/// A sync costs two `PseudocostState` clones (four vectors of length
/// `n_integer_vars`), so doing it per node would add O(n_int) memory traffic
/// to every node — noticeable once `n_int` reaches the tens of thousands,
/// where a node relaxation is itself only a few hundred microseconds. 8 caps
/// that traffic at an eighth while still being well inside one dive
/// (`queue::MAX_DIVE_DEPTH` = 30), so a worker starting a dive branches on
/// observations that already include what the other workers found.
const PSEUDOCOST_SYNC_INTERVAL: usize = 8;

/// The search-wide best integer-feasible solution.
pub(crate) struct SharedIncumbent {
    inner: Mutex<Option<(f64, SolverResult)>>,
    /// `inner`'s objective, mirrored so a worker can check whether its local
    /// copy is stale without taking the lock. Meaningless until `present`.
    objective_bits: AtomicU64,
    /// Whether `inner` holds an incumbent, tracked separately rather than
    /// inferred from `objective_bits` being finite: a legitimate incumbent's
    /// objective can itself be `+inf` only if `is_finite_candidate()` already
    /// rejected it in [`Self::consider`], so in practice this is redundant
    /// with `objective_bits` today, but kept so `take_better_than` never has
    /// to special-case what "no incumbent yet" looks like in bit pattern.
    present: AtomicBool,
}

impl SharedIncumbent {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(None),
            objective_bits: AtomicU64::new(f64::INFINITY.to_bits()),
            present: AtomicBool::new(false),
        }
    }

    /// Adopt `res` when it strictly improves the shared incumbent. Returns
    /// whether the shared incumbent changed — the caller's cue to count an
    /// `incumbent_updates`, so concurrent workers that rediscover the same
    /// objective do not each count one.
    ///
    /// Guards `is_finite_candidate()` itself rather than trusting the
    /// caller: `MipState::consider` already checks this before delegating
    /// here, but the search-wide `initial_incumbent` seed calls this
    /// directly (see [`solve_mip_parallel`]), bypassing that guard entirely.
    pub(crate) fn consider(&self, res: &SolverResult) -> bool {
        if !res.is_finite_candidate() {
            return false;
        }
        let mut guard = lock(&self.inner);
        let better = match &*guard {
            None => true,
            Some((obj, _)) => res.objective < *obj,
        };
        if better {
            *guard = Some((res.objective, res.clone()));
            self.objective_bits
                .store(res.objective.to_bits(), Ordering::Release);
            self.present.store(true, Ordering::Release);
        }
        better
    }

    /// The shared incumbent when it beats `local_obj`, else `None`. Only
    /// takes the lock (and clones the solution) when the mirrored objective
    /// says there is something to gain.
    ///
    /// The test is phrased as "does the shared value improve on the local
    /// one", the same strict `<` `consider` itself applies, rather than as a
    /// negated `>=`. That matters for NaN: `>=` is false against it, so the
    /// `>=` phrasing would send every node of every worker through the lock
    /// and a full solution clone chasing a value that can never be adopted.
    pub(crate) fn take_better_than(&self, local_obj: Option<f64>) -> Option<(f64, SolverResult)> {
        if !self.present.load(Ordering::Acquire) {
            return None;
        }
        let shared_obj = f64::from_bits(self.objective_bits.load(Ordering::Acquire));
        let improves = match local_obj {
            None => true,
            Some(local) => shared_obj < local,
        };
        if !improves {
            return None;
        }
        lock(&self.inner).clone()
    }
}

/// Poisoning carries no meaning here: a panicking worker aborts the whole
/// `thread::scope`, so the only way to observe a poisoned lock would be from
/// the unwinding path that is already discarding the result.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct PoolInner {
    queue: NodeQueue,
    /// Workers blocked in [`SharedPool::pop_blocking`].
    idle: usize,
    stopped: bool,
}

/// The shared best-bound pool of open nodes, plus the search's termination
/// protocol.
pub(crate) struct SharedPool {
    inner: Mutex<PoolInner>,
    available: Condvar,
    workers: usize,
    nodes_processed: AtomicUsize,
}

impl SharedPool {
    fn new(workers: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner {
                queue: NodeQueue::new(),
                idle: 0,
                stopped: false,
            }),
            available: Condvar::new(),
            workers,
            nodes_processed: AtomicUsize::new(0),
        }
    }

    fn push(&self, node: MipNode) {
        lock(&self.inner).queue.push(node);
        self.available.notify_one();
    }

    /// Move everything `q` holds into the shared pool. `q` must not be diving
    /// (a dive stack is private to its worker for the duration of the dive).
    fn drain_from(&self, q: &mut NodeQueue) {
        debug_assert!(!q.is_diving(), "a dive stack is never published");
        if q.is_empty() {
            return;
        }
        let mut guard = lock(&self.inner);
        while let Some(node) = q.pop() {
            guard.queue.push(node);
        }
        drop(guard);
        self.available.notify_all();
    }

    /// The next node to process, blocking while other workers are still
    /// producing. Returns `None` once the search is over — either because a
    /// worker requested a [`SharedPool::stop`] or because every worker is
    /// simultaneously idle with the pool empty, which is exactly the
    /// condition "no node is open and none can appear".
    fn pop_blocking(&self) -> Option<MipNode> {
        let mut guard = lock(&self.inner);
        loop {
            if guard.stopped {
                return None;
            }
            if let Some(node) = guard.queue.pop() {
                return Some(node);
            }
            guard.idle += 1;
            if guard.idle == self.workers {
                guard.stopped = true;
                guard.idle -= 1;
                self.available.notify_all();
                return None;
            }
            guard = self
                .available
                .wait(guard)
                .unwrap_or_else(|e| e.into_inner());
            guard.idle -= 1;
        }
    }

    /// End the search for every worker (deadline, node budget, unbounded
    /// relaxation, ...). Nodes already in the pool are kept: the finalizer
    /// still needs their bounds.
    fn stop(&self) {
        lock(&self.inner).stopped = true;
        self.available.notify_all();
    }

    fn nodes_processed(&self) -> usize {
        self.nodes_processed.load(Ordering::Relaxed)
    }

    fn add_nodes_processed(&self, n: usize) {
        self.nodes_processed.fetch_add(n, Ordering::Relaxed);
    }

    fn into_queue(self) -> NodeQueue {
        self.inner
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
            .queue
    }
}

/// Ends the search when a worker leaves, however it leaves.
///
/// On every normal exit the pool is already stopped (a worker only leaves its
/// loop after `pop_blocking` returned `None`, which requires `stopped`, or
/// after requesting the stop itself), so this is a no-op there. It exists for
/// the abnormal exit: a panicking worker would otherwise never become idle
/// and never request a stop, leaving its peers blocked in `pop_blocking`
/// forever — and `std::thread::scope` waits for them, turning a panic that
/// should surface into a hang.
struct StopOnExit<'a>(&'a SharedPool);

impl Drop for StopOnExit<'_> {
    fn drop(&mut self) {
        self.0.stop();
    }
}

/// Run the parallel branch-and-bound search. Preconditions (both enforced by
/// [`super::solve_mip_dispatch`]): `threads >= 2` and `cfg.max_lp_iters` is
/// `None`.
pub(crate) fn solve_mip_parallel<R: Relaxation + Sync>(
    problem: &R,
    options: &SolverOptions,
    cfg: &MipConfig,
    mask: Vec<bool>,
    initial_incumbent: Option<SolverResult>,
    threads: usize,
) -> (SolverResult, MipStats) {
    debug_assert!(threads >= 2, "the serial driver handles threads < 2");
    debug_assert!(
        cfg.max_lp_iters.is_none(),
        "a deterministic iteration budget requires the serial driver"
    );
    let (mut shared_opts, integer_vars, j_to_k, root_bounds, mut stats) =
        match prepare_search_inputs(problem, options) {
            Ok(v) => v,
            Err(early_return) => return *early_return,
        };
    // The worker pool *is* this solve's parallelism: nothing a worker calls
    // may open a second level of it, or the pool size would stop being the
    // thread budget the caller asked for. (RINS / RENS / local-branching
    // sub-MIPs already force `threads = 1` on their own options; this covers
    // the node relaxation and in-tree separation solves too.)
    shared_opts.threads = 1;

    let ctx = SearchCtx {
        cfg,
        shared: &shared_opts,
        mask: &mask,
        integer_vars: &integer_vars,
        j_to_k: &j_to_k,
        deadline: shared_opts.deadline,
        root_bounds: &root_bounds,
        use_reliability: cfg.branching == MipBranching::Reliability,
    };

    let incumbent = Arc::new(SharedIncumbent::new());
    if let Some(inc) = initial_incumbent {
        if incumbent.consider(&inc) {
            stats.incumbent_updates += 1;
            stats.fp_incumbent_found = true;
        }
    }
    let pool = SharedPool::new(threads);
    pool.push(MipNode::root(
        problem.root_bounds().to_vec(),
        f64::NEG_INFINITY,
    ));
    let pseudocosts = Mutex::new(PseudocostState::new(integer_vars.len()));

    let joined = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for worker_index in 0..threads {
            let _ = worker_index; // read only by the test-only failure injector
            #[cfg(test)]
            let inject_failure =
                SPAWN_FAIL_AFTER.with(|at| at.get().is_some_and(|at| worker_index >= at));
            #[cfg(not(test))]
            let inject_failure = false;

            let handle = if inject_failure {
                Err(std::io::Error::other("injected MILP worker spawn failure"))
            } else {
                std::thread::Builder::new().spawn_scoped(scope, || {
                    run_worker(problem, &ctx, &pool, &incumbent, &pseudocosts)
                })
            };
            match handle {
                Ok(handle) => handles.push(handle),
                Err(_) => {
                    // Some workers may already be waiting in `pop_blocking`.
                    // Stop them *before* the scope joins them; unwinding or
                    // returning without this signal would wait forever because
                    // `idle` can never reach the requested worker count.
                    pool.stop();
                    for handle in handles {
                        handle
                            .join()
                            .expect("MILP branch-and-bound worker panicked");
                    }
                    return Err(WorkerSpawnFailed);
                }
            }
        }
        Ok(handles
            .into_iter()
            .map(|h| h.join().expect("MILP branch-and-bound worker panicked"))
            .collect::<Vec<_>>())
    });
    let joined = match joined {
        Ok(joined) => joined,
        Err(WorkerSpawnFailed) => {
            return (
                SolverResult {
                    status: SolveStatus::ResourceExhausted,
                    objective: f64::INFINITY,
                    solution: vec![],
                    ..Default::default()
                },
                stats,
            );
        }
    };

    let mut outcome = SearchOutcome::empty();
    for (worker_stats, worker_outcome) in &joined {
        stats.merge_worker(worker_stats);
        outcome.absorb(worker_outcome);
    }

    let mut state = MipState::shared(Arc::clone(&incumbent));
    state.sync_shared();
    finalize_mip_result(problem, cfg, pool.into_queue(), state, stats, outcome)
}

/// One worker: pull nodes (local dive stack first, then the shared pool) and
/// feed them to the shared node body until the search ends.
///
/// Scheduling mirrors the serial search's best-bound/dive alternation, split
/// so that each half lands where it belongs:
///
/// * **best-bound phase** — children go to the worker's local queue and are
///   drained into the shared pool immediately, so the next node any worker
///   picks up is the globally best-bound one, exactly as in the serial search;
/// * **dive phase** — the [`NodeQueue`] dive stack stays worker-private for
///   the whole dive, so a worker processes a parent and its children back to
///   back and the warm-start basis `child_branched` handed down is consumed on
///   the same core that produced it. Dives are entered by the same
///   `DIVE_FREQUENCY` rule as the serial search, counted per worker.
///
/// Why the answer does not depend on the order: a worker's local incumbent is
/// always a past value of the shared one, which decreases monotonically, so
/// `should_prune` can only be *too conservative* here — never prune the region
/// holding the optimum. And every node a worker still holds when the search
/// stops is pushed back to the shared pool before it joins, so
/// `finalize_mip_result`'s remaining lower bound covers the whole unexplored
/// region: an interrupted parallel search can no more claim a false `Optimal`
/// than a serial one.
fn run_worker<R: Relaxation>(
    problem: &R,
    ctx: &SearchCtx<'_>,
    pool: &SharedPool,
    incumbent: &Arc<SharedIncumbent>,
    pseudocosts: &Mutex<PseudocostState>,
) -> (MipStats, SearchOutcome) {
    let _stop_on_exit = StopOnExit(pool);
    let n_int = ctx.integer_vars.len();
    // Worker-private conflict store — a performance choice, not a soundness
    // one. A clause records the bound tightenings of a node whose *relaxation*
    // was infeasible, and that relaxation never saw a cutoff: it is a pure
    // feasibility fact about the box, so the clause is cutoff-independent.
    // Reduced-cost fixing does not weaken this — it only narrows the box
    // before the solve, so the clause still says "this box, as defined, has no
    // feasible relaxation point". `is_conflicted` then prunes only nodes whose
    // bounds *subsume* a stored clause, i.e. a subset of that empty box, which
    // is empty too. Sharing the store would therefore be correct and would
    // prune strictly more; it is skipped because `is_conflicted` runs on every
    // node against up to `MAX_CONFLICTS` clauses, and that scan is the one
    // per-node critical section long enough for a lock to matter. Revisiting
    // it (RwLock, or periodic exchange like the pseudocost sync) is deferred.
    let mut s = SearchState::new(
        MipStats::worker_seed(),
        n_int,
        MipState::shared(Arc::clone(incumbent)),
        ConflictStore::new(),
    );
    let mut pc_baseline = PseudocostState::new(n_int);
    let mut since_pc_sync = 0usize;

    loop {
        let node = match s.q.pop() {
            Some(node) => node,
            None => match pool.pop_blocking() {
                Some(node) => node,
                None => break,
            },
        };
        s.state.sync_shared();
        if ctx.use_reliability && since_pc_sync == 0 {
            let mut guard = lock(pseudocosts);
            guard.add_delta(&s.pc, &pc_baseline);
            s.pc.clone_from(&guard);
            pc_baseline.clone_from(&guard);
        }
        since_pc_sync = (since_pc_sync + 1) % PSEUDOCOST_SYNC_INTERVAL;

        let before = s.stats.nodes_processed;
        let control = run_node(problem, node, ctx, &mut s, pool.nodes_processed());
        pool.add_nodes_processed(s.stats.nodes_processed - before);

        if matches!(control, NodeLoop::Break) {
            pool.stop();
            break;
        }
        if !s.q.is_diving() {
            pool.drain_from(&mut s.q);
        }
    }

    // No final pseudocost publish: `solve_mip_parallel` drops the shared
    // accumulator as soon as the workers join, so anything written here has no
    // reader. The in-loop sync is what makes observations visible to peers.
    // Hand back every node still held locally so the finalizer's remaining
    // lower bound covers this worker's unexplored region too.
    s.q.end_dive();
    pool.drain_from(&mut s.q);
    let outcome = SearchOutcome::from(&s);
    (s.stats, outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(lb: f64) -> MipNode {
        MipNode::root(vec![(0.0, 1.0)], lb)
    }

    fn with_objective(objective: f64) -> SolverResult {
        SolverResult {
            objective,
            ..Default::default()
        }
    }

    /// Upper bound on a condvar handoff before a test calls it a lost wakeup.
    /// Four orders of magnitude above the microsecond one actually takes, so
    /// scheduler noise cannot reach it.
    const WAKE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

    /// A non-finite objective must never become the shared incumbent.
    ///
    /// `MipState::consider` now rejects `!is_finite_candidate()` outright
    /// (the `within_gap` false-Optimal fix), and `SharedIncumbent::consider`
    /// has to enforce the same rule independently: the `initial_incumbent`
    /// seed in `solve_mip_parallel` calls this directly, bypassing
    /// `MipState::consider` entirely, so a poisoned FP seed would otherwise
    /// reach every worker as if it were a genuine solution.
    ///
    /// Sentinel: reverting the `is_finite_candidate()` guard in `consider`
    /// makes the first `assert!(!...)` below fail (both non-finite
    /// objectives get adopted, and `take_better_than` then hands them out).
    #[test]
    fn non_finite_candidates_are_never_adopted() {
        for objective in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let inc = SharedIncumbent::new();
            assert!(
                !inc.consider(&with_objective(objective)),
                "a non-finite objective must never be reported as adopted"
            );
            assert!(
                inc.take_better_than(None).is_none(),
                "an empty incumbent must stay empty, not silently hold {objective}"
            );
        }

        // A poisoned candidate must not displace a genuine finite incumbent
        // either (the "adopt as the very first result" path is the one the
        // production `initial_incumbent` seed actually takes).
        let genuine = SharedIncumbent::new();
        assert!(genuine.consider(&with_objective(5.0)));
        assert!(
            !genuine.consider(&with_objective(f64::NEG_INFINITY)),
            "a non-finite candidate must not beat an existing finite incumbent"
        );
        assert_eq!(
            genuine.take_better_than(None).expect("still present").0,
            5.0,
            "the genuine incumbent must survive a rejected non-finite challenger"
        );
    }

    #[test]
    fn pool_pops_best_bound_first() {
        let pool = SharedPool::new(1);
        pool.push(node(3.0));
        pool.push(node(-1.0));
        pool.push(node(7.0));
        assert_eq!(pool.pop_blocking().unwrap().lower_bound, -1.0);
        assert_eq!(pool.pop_blocking().unwrap().lower_bound, 3.0);
        assert_eq!(pool.pop_blocking().unwrap().lower_bound, 7.0);
        assert!(
            pool.pop_blocking().is_none(),
            "an exhausted pool must end the search"
        );
    }

    /// Termination detection: with the pool empty and every worker idle the
    /// search is over, and *all* workers must observe it (not just the one
    /// that noticed).
    #[test]
    fn all_idle_workers_terminate_together() {
        let pool = SharedPool::new(4);
        let woke = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    assert!(pool.pop_blocking().is_none());
                    woke.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(woke.load(Ordering::SeqCst), 4, "every worker must return");
    }

    /// A worker blocked on an empty pool must wake when another worker
    /// publishes children, not spin or deadlock.
    #[test]
    fn blocked_worker_wakes_on_push() {
        let pool = SharedPool::new(2);
        let got = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                if let Some(n) = pool.pop_blocking() {
                    assert_eq!(n.lower_bound, 42.0);
                    got.store(true, Ordering::SeqCst);
                }
            });
            // The consumer is either already waiting or about to wait; either
            // way the push below must reach it.
            pool.push(node(42.0));
            // Bounded wait: a lost wakeup must fail the test, not hang the
            // suite. `WAKE_TIMEOUT` is orders of magnitude above the microsecond
            // a condvar handoff takes, so only a genuine miss reaches the stop
            // below (which then leaves `got` false and fails the assertion).
            let deadline = std::time::Instant::now() + WAKE_TIMEOUT;
            while !got.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                std::thread::yield_now();
            }
            pool.stop();
        });
        assert!(
            got.load(Ordering::SeqCst),
            "a worker blocked on an empty pool did not wake on push"
        );
    }

    #[test]
    fn stop_releases_waiting_workers() {
        let pool = SharedPool::new(3);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| assert!(pool.pop_blocking().is_none()));
            }
            pool.stop();
        });
    }

    /// A partial worker-spawn failure must (a) *not* deadlock — the workers
    /// already accepted by the OS get stopped before the scoped joins begin,
    /// so termination detection is reached even though the pool still records
    /// the full requested worker count — and (b) report the failure as
    /// [`SolveStatus::ResourceExhausted`] (an OS resource shortfall), not as a
    /// numerical breakdown. The one live worker below consumes an integral
    /// root and would otherwise block forever in `pop_blocking`.
    ///
    /// Two independent obligations are asserted: the bounded `recv_timeout`
    /// catches a *deadlock* (a hang, not a status), and the `assert_eq!`
    /// catches a *wrong status*.
    ///
    /// Sentinel: removing the `pool.stop()` in the spawn-error branch makes
    /// the receive time out (deadlock); returning `NumericalError` there makes
    /// the status assertion fail.
    #[test]
    fn partial_worker_spawn_failure_returns_resource_exhausted_instead_of_deadlocking() {
        struct IntegralLeaf {
            bounds: Vec<(f64, f64)>,
            integers: Vec<usize>,
        }

        impl Relaxation for IntegralLeaf {
            fn num_vars(&self) -> usize {
                1
            }
            fn root_bounds(&self) -> &[(f64, f64)] {
                &self.bounds
            }
            fn integer_vars(&self) -> &[usize] {
                &self.integers
            }
            fn solve(&self, _bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.0,
                    solution: vec![0.0],
                    ..Default::default()
                }
            }
        }

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            SPAWN_FAIL_AFTER.with(|at| at.set(Some(1)));
            let problem = IntegralLeaf {
                bounds: vec![(0.0, 1.0)],
                integers: vec![0],
            };
            let options = SolverOptions::default();
            let mask = vec![true];
            let (result, _) =
                solve_mip_parallel(&problem, &options, &MipConfig::default(), mask, None, 2);
            SPAWN_FAIL_AFTER.with(|at| at.set(None));
            tx.send(result.status).expect("test receiver alive");
        });

        // Obligation (a): no deadlock — a result arrives within the bound.
        let status = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("partial spawn failure deadlocked instead of returning");
        // Obligation (b): the failure is classified as an OS resource
        // shortfall, not a numerical breakdown.
        assert_eq!(
            status,
            SolveStatus::ResourceExhausted,
            "a requested worker budget that cannot be created must be an explicit \
             resource-exhaustion failure, not a numerical breakdown"
        );
    }

    #[test]
    fn stopped_pool_keeps_pushed_nodes_for_the_bound() {
        let pool = SharedPool::new(2);
        pool.stop();
        pool.push(node(-5.0));
        let q = pool.into_queue();
        assert_eq!(
            q.best_lower_bound(),
            Some(-5.0),
            "nodes pushed after the stop still bound the unexplored region"
        );
    }

    #[test]
    fn drain_from_moves_local_nodes_into_the_pool() {
        let pool = SharedPool::new(1);
        let mut local = NodeQueue::new();
        local.push(node(2.0));
        local.push(node(1.0));
        pool.drain_from(&mut local);
        assert!(local.is_empty(), "the local queue must be emptied");
        assert_eq!(pool.pop_blocking().unwrap().lower_bound, 1.0);
        assert_eq!(pool.pop_blocking().unwrap().lower_bound, 2.0);
    }

    #[test]
    fn shared_incumbent_keeps_only_strict_improvements() {
        let inc = SharedIncumbent::new();
        let first = with_objective(5.0);
        let worse = with_objective(7.0);
        let equal = with_objective(5.0);
        let better = with_objective(1.0);

        assert!(
            inc.consider(&first),
            "the first incumbent is an improvement"
        );
        assert!(!inc.consider(&worse));
        assert!(!inc.consider(&equal), "ties must not count as updates");
        assert!(inc.consider(&better));
        assert_eq!(inc.take_better_than(None).unwrap().0, 1.0);
    }

    #[test]
    fn take_better_than_skips_a_current_local_view() {
        let inc = SharedIncumbent::new();
        assert!(
            inc.take_better_than(None).is_none(),
            "no incumbent yet ⇒ nothing to adopt"
        );
        inc.consider(&with_objective(4.0));
        assert!(
            inc.take_better_than(Some(4.0)).is_none(),
            "not an improvement"
        );
        assert!(inc.take_better_than(Some(9.0)).is_some());
        assert!(inc.take_better_than(None).is_some());
    }

    /// Concurrent `consider` calls must serialise so that the minimum wins and
    /// **no objective is ever claimed as an improvement twice**.
    ///
    /// The second property is what `incumbent_updates` depends on: the caller
    /// counts one update per `true`, so two workers both being told they
    /// improved to the same value would double-count. It is a real race
    /// detector — every worker here proposes the whole descending range, so
    /// every value is offered `WORKERS` times and only a correctly serialised
    /// compare-and-set can hand out each one at most once.
    #[test]
    fn concurrent_consider_is_race_free() {
        const WORKERS: usize = 8;
        const PER_WORKER: usize = 200;
        const TOTAL: usize = WORKERS * PER_WORKER;
        let inc = SharedIncumbent::new();
        let winners = Mutex::new(Vec::<u64>::new());
        std::thread::scope(|scope| {
            for _ in 0..WORKERS {
                let inc = &inc;
                let winners = &winners;
                scope.spawn(move || {
                    // Every worker walks the same descending sequence, so each
                    // value races against WORKERS-1 identical proposals.
                    for i in 0..TOTAL {
                        let obj = (TOTAL - i) as f64;
                        if inc.consider(&with_objective(obj)) {
                            lock(winners).push(obj.to_bits());
                        }
                    }
                });
            }
        });
        let (obj, _) = inc.take_better_than(None).expect("an incumbent survives");
        assert_eq!(obj, 1.0, "the global minimum must win every race");

        let mut claimed = lock(&winners).clone();
        let before = claimed.len();
        claimed.sort_unstable();
        claimed.dedup();
        assert_eq!(
            claimed.len(),
            before,
            "an objective was claimed as an improvement more than once \
             ({before} claims, {} distinct) — incumbent_updates would double-count",
            claimed.len()
        );
        assert!(
            before >= 2,
            "the sequence must produce several genuine improvements, got {before}"
        );
    }

    /// Every node a worker still holds when the search stops goes back to the
    /// shared pool — checked as an exact conservation identity, not as a
    /// downstream symptom.
    ///
    /// The mock branches at *every* processed node (always Optimal, always
    /// fractional, no incumbent so nothing prunes), which makes the node
    /// bookkeeping closed-form. With `P` = nodes processed:
    ///
    /// * created   = 1 root + 2 per processed node = `1 + 2P`
    /// * consumed  = the `P` processed nodes + the one node popped by the
    ///   iteration that hit `max_nodes` (folded into `open_lb`, then dropped)
    /// * therefore the pool must end holding `1 + 2P - (P + 1)` = **`P`** nodes
    ///
    /// The stop fires while the worker is mid-dive (`DIVE_FREQUENCY_NO_
    /// INCUMBENT` = 2 starts one long before the cap), so a chunk of those `P`
    /// are sitting in its private dive stack at the moment it breaks.
    ///
    /// Sentinel: deleting `run_worker`'s `end_dive` + `drain_from` tail leaves
    /// the pool short by exactly the dive stack's contents and fails the
    /// equality.
    #[test]
    fn a_stopped_worker_returns_every_node_it_still_holds() {
        use super::super::queue::DIVE_FREQUENCY_NO_INCUMBENT;
        use super::super::{integer_mask, prepare_search_inputs, SearchCtx};
        use crate::options::MipBranching;
        use crate::problem::SolveStatus;
        use std::collections::HashMap;

        /// Wide enough that bisection never bottoms out within the cap.
        const ROOT_UB: f64 = 1_099_511_627_776.0; // 2^40
        const MAX_NODES: usize = 25;

        // The whole point of this test is that the worker is *mid-dive* when
        // the cap stops it, so that the tail has a private dive stack to hand
        // back. That premise is a coupling between two constants in different
        // modules, so it is machine-checked rather than left to a comment:
        // raise `DIVE_FREQUENCY_NO_INCUMBENT` past the cap (or lower the cap)
        // and this fails to compile instead of quietly going vacuous.
        const _: () = assert!(
            DIVE_FREQUENCY_NO_INCUMBENT < MAX_NODES,
            "no dive can start before the node cap stops the worker: \
             the test would no longer exercise the dive-stack hand-back"
        );

        struct AlwaysBranches {
            bounds: Vec<(f64, f64)>,
            ints: Vec<usize>,
        }

        impl Relaxation for AlwaysBranches {
            fn num_vars(&self) -> usize {
                1
            }
            fn root_bounds(&self) -> &[(f64, f64)] {
                &self.bounds
            }
            fn integer_vars(&self) -> &[usize] {
                &self.ints
            }
            fn solve(&self, bounds: &[(f64, f64)], _opts: &SolverOptions) -> SolverResult {
                // The box midpoint snapped to a half-integer. `floor` first is
                // what makes it *unconditionally* fractional: a plain midpoint
                // lands on a whole number for half the boxes, which the driver
                // would read as an integer-feasible leaf, adopt as an
                // incumbent, and then prune the rest of the tree against.
                let (lb, ub) = bounds[0];
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: 0.0,
                    solution: vec![((lb + ub) / 2.0).floor() + 0.5],
                    ..Default::default()
                }
            }
        }

        let problem = AlwaysBranches {
            bounds: vec![(0.0, ROOT_UB)],
            ints: vec![0],
        };
        let cfg = MipConfig {
            max_nodes: MAX_NODES,
            branching: MipBranching::MostFractional,
            cuts: false,
            tree_cuts: false,
            rins_enabled: false,
            rens_enabled: false,
            local_branching_enabled: false,
            symmetry: false,
            ..MipConfig::default()
        };
        let options = SolverOptions::default();
        let (shared_opts, integer_vars, j_to_k, root_bounds, _) =
            prepare_search_inputs(&problem, &options).expect("integer vars present");
        let mask = integer_mask(problem.num_vars(), problem.integer_vars());
        let j_to_k: HashMap<usize, usize> = j_to_k;
        let ctx = SearchCtx {
            cfg: &cfg,
            shared: &shared_opts,
            mask: &mask,
            integer_vars: &integer_vars,
            j_to_k: &j_to_k,
            deadline: shared_opts.deadline,
            root_bounds: &root_bounds,
            use_reliability: false,
        };

        // One worker, so the identity has no interleaving to average over.
        let pool = SharedPool::new(1);
        pool.push(MipNode::root(
            problem.root_bounds().to_vec(),
            f64::NEG_INFINITY,
        ));
        let incumbent = Arc::new(SharedIncumbent::new());
        let pseudocosts = Mutex::new(PseudocostState::new(integer_vars.len()));

        let (stats, outcome) = run_worker(&problem, &ctx, &pool, &incumbent, &pseudocosts);
        assert!(
            outcome.maxnodes_stop,
            "the node cap must be what stopped the worker"
        );
        assert_eq!(
            stats.pruned, 0,
            "nothing may prune: the identity assumes every processed node branched"
        );
        assert_eq!(stats.nodes_processed, MAX_NODES);

        let mut remaining = pool.into_queue();
        let mut held = 0usize;
        while remaining.pop().is_some() {
            held += 1;
        }
        assert_eq!(
            held, MAX_NODES,
            "the pool must end with one node per processed node \
             (1 + 2*{MAX_NODES} created, {MAX_NODES} + 1 consumed); \
             a shortfall means the worker's dive stack was dropped"
        );
    }

    #[test]
    fn pseudocost_delta_merge_is_order_independent() {
        let mut global = PseudocostState::new(2);
        let baseline = global.clone();

        let mut a = baseline.clone();
        a.record_up(0, 3.0);
        a.record_down(1, 1.0);
        let mut b = baseline.clone();
        b.record_up(0, 5.0);
        b.record_up(0, 1.0);

        global.add_delta(&b, &baseline);
        global.add_delta(&a, &baseline);

        // Independent oracle: one searcher making all four observations.
        let mut serial = PseudocostState::new(2);
        serial.record_up(0, 3.0);
        serial.record_down(1, 1.0);
        serial.record_up(0, 5.0);
        serial.record_up(0, 1.0);

        assert_eq!(global.up_sum, serial.up_sum);
        assert_eq!(global.up_count, serial.up_count);
        assert_eq!(global.down_sum, serial.down_sum);
        assert_eq!(global.down_count, serial.down_count);
    }
}
