//! Process-level enforcement of `SolverOptions::threads`.
//!
//! The in-crate tests (`otspot-core/src/mip/tests_parallel.rs`) instrument the
//! `Relaxation` the MILP search calls and prove the *search* never runs more
//! concurrent relaxations, nor touches more distinct threads, than the budget.
//! These tests close the remaining gap from the outside: they watch the
//! operating system's own thread table while a solve runs, so nothing the
//! solver (or a library beneath it) spawns can escape the observation.
//!
//! The two paths get different assertions because they bound different
//! things:
//!
//! * **MILP** owns its parallelism. `mip::parallel` spawns exactly `threads`
//!   workers with `std::thread::scope` and forces `threads = 1` on everything
//!   they call, so the *live thread count* is the thing to watch: the process
//!   may gain at most `threads` threads for the duration of the solve, and
//!   must give them all back.
//! * **QP** does not own its threads — faer runs on a rayon pool. Counting
//!   live threads there measures pool lifetime, not the budget, so these tests
//!   watch *concurrency* instead: how many of this process's threads are
//!   runnable at once while the solve is in flight.
//!
//! What bounds the QP path is **not** `Par::Rayon(threads)`. faer 0.24.4
//! silently widens that back to the global pool (`spindle`'s fallback), which
//! was measured here at 10 concurrent threads for a `threads = 2` solve on an
//! 8-core host. The bound comes from
//! `otspot_num::linalg::parallelism::with_solver_pool`, which confines the
//! solve to a dedicated pool of exactly `threads` workers and hands it the
//! matching `Par`; the same measurement then reads 3. Those pools are cached
//! per size for the process, so the *first* solve at a given budget does
//! create `threads` threads and later ones create none — which is why the QP
//! test measures concurrency after a warm-up rather than counting spawns.
//!
//! Multistart shares that cache (it used to build a pool per call, which is
//! now gone), so its `min(n_starts, threads)` workers are likewise created
//! once and then live for the rest of the process rather than per solve.
//!
//! Linux-only: `/proc/self/task` is the thread table being read.

#![cfg(target_os = "linux")]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use otspot::options::{MipConfig, SolverOptions};
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::{solve_milp, solve_qp_with, CscMatrix, MilpProblem, QpProblem};

/// Sampling period of the thread-table watcher. Short enough that a solve
/// lasting tens of milliseconds is sampled many times over.
const SAMPLE_PERIOD: Duration = Duration::from_millis(1);

/// How long a finished solve gets for its exited worker threads to disappear
/// from `/proc/self/task`. Generous relative to the observed reaping lag
/// (milliseconds) so contention cannot make the check flaky, yet finite so a
/// real thread leak still fails instead of hanging.
const LEAK_SETTLE_TIMEOUT: Duration = Duration::from_secs(5);

fn live_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("/proc/self/task is readable on Linux")
        .count()
}

/// Run `body` while sampling the process thread count; returns
/// `(baseline, peak, value)` where `baseline` already includes the sampler
/// thread itself.
fn watch_threads<T>(body: impl FnOnce() -> T) -> (usize, usize, T) {
    let stop = AtomicBool::new(false);
    let peak = AtomicUsize::new(0);
    let baseline = AtomicUsize::new(0);
    let value = std::thread::scope(|scope| {
        scope.spawn(|| {
            baseline.store(live_threads(), Ordering::SeqCst);
            while !stop.load(Ordering::SeqCst) {
                peak.fetch_max(live_threads(), Ordering::SeqCst);
                std::thread::sleep(SAMPLE_PERIOD);
            }
        });
        // Let the sampler publish its baseline (which counts itself) before
        // the measured work starts.
        while baseline.load(Ordering::SeqCst) == 0 {
            std::hint::spin_loop();
        }
        let value = body();
        stop.store(true, Ordering::SeqCst);
        value
    });
    (
        baseline.load(Ordering::SeqCst),
        peak.load(Ordering::SeqCst),
        value,
    )
}

/// A multi-dimensional binary knapsack big enough to keep a branch-and-bound
/// search busy for the sampling window.
fn knapsack(n: usize, m: usize) -> MilpProblem {
    let mut state = 0x5DEE_CE66_D123_4567u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % 20 + 1) as f64
    };
    let c: Vec<f64> = (0..n).map(|_| -next()).collect();
    let (mut rows, mut cols, mut vals, mut b) = (vec![], vec![], vec![], vec![]);
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            let v = next();
            sum += v;
            rows.push(i);
            cols.push(j);
            vals.push(v);
        }
        b.push((sum / 2.0).floor());
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).expect("triplets");
    let lp = LpProblem::new_general(
        c,
        a,
        b,
        vec![ConstraintType::Le; m],
        vec![(0.0, 1.0); n],
        None,
    )
    .expect("lp");
    MilpProblem::new(lp, (0..n).collect()).expect("milp")
}

/// Peak number of this process's threads in the kernel's running/runnable
/// state (`R`) while `body` runs. Unlike a live-thread count this tracks
/// *concurrency*, which is what a thread budget bounds when the threads
/// themselves belong to a shared pool.
fn watch_running_threads<T>(body: impl FnOnce() -> T) -> (usize, T) {
    let stop = AtomicBool::new(false);
    let peak = AtomicUsize::new(0);
    let started = AtomicBool::new(false);
    let value = std::thread::scope(|scope| {
        scope.spawn(|| {
            started.store(true, Ordering::SeqCst);
            while !stop.load(Ordering::SeqCst) {
                peak.fetch_max(running_threads(), Ordering::SeqCst);
                std::thread::sleep(SAMPLE_PERIOD);
            }
        });
        while !started.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        let value = body();
        stop.store(true, Ordering::SeqCst);
        value
    });
    (peak.load(Ordering::SeqCst), value)
}

fn running_threads() -> usize {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0;
    };
    tasks
        .filter_map(Result::ok)
        .filter(|task| {
            let Ok(stat) = std::fs::read_to_string(task.path().join("stat")) else {
                return false;
            };
            // `stat` is "pid (comm) state ..."; comm may contain spaces and
            // parentheses, so the state is the token after the last ')'.
            stat.rsplit_once(')')
                .and_then(|(_, rest)| rest.split_whitespace().next())
                .is_some_and(|state| state == "R")
        })
        .count()
}

/// A convex QP with a *dense* Hessian: `Q = D + 1 1'` (diagonally dominant, so
/// positive definite) over a box with a handful of dense rows. Density is the
/// point — faer only parallelizes a sparse factorization when its supernodes
/// are large, so a banded or diagonal `Q` would leave the parallel path
/// untaken and make the measurement below vacuous.
fn dense_qp(n: usize, m: usize) -> QpProblem {
    let (mut qr, mut qc, mut qv) = (
        Vec::with_capacity(n * n),
        Vec::with_capacity(n * n),
        Vec::with_capacity(n * n),
    );
    for j in 0..n {
        for i in 0..n {
            qr.push(i);
            qc.push(j);
            qv.push(if i == j { n as f64 + 1.0 } else { 1.0 });
        }
    }
    let q = CscMatrix::from_triplets(&qr, &qc, &qv, n, n).expect("q");
    let (mut ar, mut ac, mut av) = (vec![], vec![], vec![]);
    for i in 0..m {
        for j in 0..n {
            ar.push(i);
            ac.push(j);
            av.push(1.0 + ((i + j) % 5) as f64);
        }
    }
    let a = CscMatrix::from_triplets(&ar, &ac, &av, m, n).expect("a");
    QpProblem::new(
        q,
        (0..n).map(|j| -1.0 - (j % 7) as f64).collect(),
        a,
        vec![2.0 * n as f64; m],
        vec![(0.0, 10.0); n],
        vec![ConstraintType::Le; m],
    )
    .expect("qp")
}

fn milp_opts(threads: usize) -> SolverOptions {
    let mut o = SolverOptions::default();
    o.threads = threads;
    o.timeout_secs = Some(20.0);
    o
}

/// A parallel MILP solve may borrow at most `threads` operating-system
/// threads, and must return every one of them before it returns.
///
/// Sentinel: sizing the worker pool from anything but `options.threads` (for
/// instance `std::thread::available_parallelism`) pushes the observed peak
/// past `baseline + THREADS` on any machine with more cores than that.
#[test]
fn milp_solve_borrows_at_most_the_requested_threads() {
    const THREADS: usize = 4;
    let problem = knapsack(34, 5);
    // Warm-up: take first-use allocations and lazily initialised globals out
    // of the measured window.
    let _ = solve_milp(&problem, &milp_opts(1), &MipConfig::default());

    let (baseline, peak, res) =
        watch_threads(|| solve_milp(&problem, &milp_opts(THREADS), &MipConfig::default()));
    assert!(
        matches!(res.status, SolveStatus::Optimal | SolveStatus::Timeout),
        "unexpected status {:?}",
        res.status
    );
    assert!(
        peak <= baseline + THREADS,
        "peak {peak} threads exceeds the baseline {baseline} plus the {THREADS}-thread budget"
    );
    assert!(
        peak >= baseline + 2,
        "peak {peak} vs baseline {baseline}: the search never ran workers in parallel"
    );
    // No leak: the borrowed threads go away. Polled rather than asserted
    // outright, because `/proc/self/task` keeps an entry for a thread that has
    // already exited until the kernel reaps it, and under CPU contention that
    // lag outlives the solve — an exact equality here failed roughly one run
    // in three under an eight-way load while `std::thread::scope` was, by
    // construction, still joining every worker. A genuine leak never
    // converges, so the bounded wait keeps the property without the race.
    let target = baseline - 1;
    let deadline = std::time::Instant::now() + LEAK_SETTLE_TIMEOUT;
    while live_threads() > target && std::time::Instant::now() < deadline {
        std::thread::sleep(SAMPLE_PERIOD);
    }
    assert_eq!(
        live_threads(),
        target,
        "workers still live after the solve returned (baseline includes the sampler)"
    );
}

/// `threads = 1` must not create a worker at all: the default MILP search runs
/// entirely on the caller's thread, which is what keeps it deterministic.
#[test]
fn serial_milp_solve_creates_no_threads() {
    let problem = knapsack(28, 4);
    let _ = solve_milp(&problem, &milp_opts(1), &MipConfig::default());

    let (baseline, peak, _) =
        watch_threads(|| solve_milp(&problem, &milp_opts(1), &MipConfig::default()));
    assert_eq!(
        peak, baseline,
        "threads=1 must not add a thread (baseline {baseline}, peak {peak})"
    );
}

/// A QP solve must not run more of the process concurrently than its thread
/// budget allows.
///
/// The QP path does not own its threads — faer executes on a rayon pool, so
/// counting live threads says nothing here. What the budget bounds is how
/// many run *at once*, which is what this samples: threads of this process in
/// the kernel's runnable state while the solve is in flight.
///
/// Sentinel (measured, 8-core host): removing
/// `otspot_num::linalg::parallelism::with_solver_pool` from
/// `solve_ippmm_inner` — i.e. relying on `Par::Rayon(threads)` alone, which
/// spindle silently widens back to the global pool — takes this from 3
/// concurrent threads to **10** at `threads = 2`, failing the bound below.
#[test]
fn qp_solve_runs_no_more_than_the_thread_budget_at_once() {
    const THREADS: usize = 2;
    let problem = dense_qp(700, 40);
    let mut opts = SolverOptions::default();
    opts.threads = THREADS;
    opts.timeout_secs = Some(60.0);
    // Warm-up so lazy pool construction is not what the measurement catches.
    let _ = solve_qp_with(&problem, &opts);

    let (peak_running, res) = watch_running_threads(|| solve_qp_with(&problem, &opts));
    assert!(
        matches!(
            res.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution | SolveStatus::Timeout
        ),
        "unexpected status {:?}",
        res.status
    );
    // `THREADS` pool workers, plus two threads that are not solver work: the
    // sampler (runnable by construction — it is the one counting) and the
    // caller, which is parked inside `install` but is briefly runnable as it
    // enters and leaves. Still an order of magnitude below the 10 an
    // unconfined faer reached at the same budget, so the bound discriminates.
    assert!(
        peak_running <= THREADS + 2,
        "QP at threads={THREADS} had {peak_running} threads runnable at once"
    );
}
