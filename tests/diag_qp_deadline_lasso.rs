//! IPPMM / presolve deadline regression guard (synthetic LASSO).
//!
//! 元バグ (`fix/qp` audit): presolve hot loop と IPPMM Schur gondzio inner で
//! deadline check が欠落し、`for j in 0..n` × `for i in 0..m` が deadline を
//! 踏み越え続けた (IS_LASSO_300: wall=1025.6s vs timeout=1000s)。
//!
//! このテストはデータ依存を排除し、合成 LASSO を inline 生成して
//! `solve_qp_with` が `timeout_secs` を honor することを検証する。
//! 実データでの再現は別タスク (要 `data/` 配置) で取り扱う。

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// timer 開始 (test) と deadline 計算 (solver) のずれ、postprocess
/// (refine_primal_lsq + cleanup_lp + unscale) の post-cleanup overhead を吸収。
/// 過去 LP bench (dfl001) で postsolve 2-4s を観測しており、余裕を見て 5s。
/// この値を下回るたびに solver の post-cleanup が deadline 後にやり過ぎていない
/// か再確認すべし。
const SLACK_FOR_POSTPROCESS_SEC: f64 = 5.0;

/// Watchdog (mpsc + 別スレッド): solver が deadline 後も走り続け join しない
/// 退行を `RecvTimeoutError::Timeout` で検出する。
fn solve_with_watchdog(
    problem: QpProblem,
    timeout_secs: f64,
    watchdog: Duration,
    label: &str,
) -> (SolveStatus, f64) {
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new()
        .name(format!("{label}-solver"))
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(timeout_secs);
            let t0 = Instant::now();
            let r = solve_qp_with(&problem, &opts);
            let _ = tx.send((r.status, r.objective, t0.elapsed()));
        })
        .expect("spawn solver thread");

    match rx.recv_timeout(watchdog) {
        Ok((status, obj, elapsed)) => {
            let secs = elapsed.as_secs_f64();
            eprintln!(
                "[{label}] status={:?} obj={:.6e} wall={:.3}s (timeout={timeout_secs}s, watchdog={}s)",
                status, obj, secs, watchdog.as_secs_f64(),
            );
            let _ = handle.join();
            assert!(
                secs <= timeout_secs + SLACK_FOR_POSTPROCESS_SEC,
                "[{label}] wall={:.3}s > budget {}s + slack {}s — deadline path leaks",
                secs, timeout_secs, SLACK_FOR_POSTPROCESS_SEC
            );
            (status, secs)
        }
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "[{label}] solve_qp_with did not return within watchdog {}s (timeout={timeout_secs}s) — deadline path missing",
            watchdog.as_secs_f64(),
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("[{label}] solver thread panicked before reply")
        }
    }
}

/// LCG-based deterministic pseudo-random in (-1, 1). `std::rand` 非依存。
fn lcg_unit(seed: &mut u64) -> f64 {
    // Numerical Recipes constants
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
}

/// 合成 LASSO を QP に変換: min 0.5 ||A z - y||^2 + lambda ||z||_1.
///
/// 変数 z ∈ R^p を `x = z+ - z- (z+, z- >= 0)` の split 表現ではなく、
/// 補助変数 `t_i >= |z_i|` (epigraph) を導入して
///   min 0.5 z^T (A^T A) z - (A^T y)^T z + lambda * 1^T t
///   s.t.  z_i - t_i <= 0,  -z_i - t_i <= 0   (i=1..p)
///         z free, t >= 0
/// と定式化する。変数数 n=2p、制約数 m=2p、Q は左上 p×p に A^T A (dense)、
/// 右下 p×p に 0。Schur / Cholesky に十分な work を渡す size を狙う。
fn make_synthetic_lasso(p: usize, m_data: usize, lambda: f64, seed: u64) -> QpProblem {
    // 1. データ行列 A (m_data x p) と観測 y (m_data) を deterministic に生成
    let mut rng = seed;
    let mut a_data: Vec<Vec<f64>> = vec![vec![0.0; p]; m_data];
    for i in 0..m_data {
        for j in 0..p {
            a_data[i][j] = lcg_unit(&mut rng);
        }
    }
    let mut y = vec![0.0_f64; m_data];
    for i in 0..m_data {
        y[i] = lcg_unit(&mut rng);
    }

    // 2. Q_top = A^T A (dense p x p, symmetric PSD)
    let mut q_dense = vec![0.0_f64; p * p];
    for j in 0..p {
        for k in 0..p {
            let mut s = 0.0;
            for i in 0..m_data {
                s += a_data[i][j] * a_data[i][k];
            }
            q_dense[j * p + k] = s;
        }
    }

    // 3. c = [-A^T y ; lambda * 1_p]
    let n_var = 2 * p;
    let mut c = vec![0.0_f64; n_var];
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..m_data {
            s += a_data[i][j] * y[i];
        }
        c[j] = -s;
    }
    for j in 0..p {
        c[p + j] = lambda;
    }

    // 4. Q を CSC で構築 (左上 p×p block のみ、上三角 + 対角でなく full symmetric)
    let mut q_rows: Vec<usize> = Vec::with_capacity(p * p);
    let mut q_cols: Vec<usize> = Vec::with_capacity(p * p);
    let mut q_vals: Vec<f64> = Vec::with_capacity(p * p);
    for j in 0..p {
        for k in 0..p {
            let v = q_dense[j * p + k];
            if v.abs() > 0.0 {
                q_rows.push(k);
                q_cols.push(j);
                q_vals.push(v);
            }
        }
    }
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n_var, n_var)
        .expect("Q csc build");

    // 5. 制約行列 A_constr: [[ I_p, -I_p ], [-I_p, -I_p]] (m=2p, n=2p)
    let m_constr = 2 * p;
    let mut a_rows: Vec<usize> = Vec::with_capacity(4 * p);
    let mut a_cols: Vec<usize> = Vec::with_capacity(4 * p);
    let mut a_vals: Vec<f64> = Vec::with_capacity(4 * p);
    for i in 0..p {
        // row i: z_i - t_i <= 0
        a_rows.push(i);
        a_cols.push(i);
        a_vals.push(1.0);
        a_rows.push(i);
        a_cols.push(p + i);
        a_vals.push(-1.0);
        // row p+i: -z_i - t_i <= 0
        a_rows.push(p + i);
        a_cols.push(i);
        a_vals.push(-1.0);
        a_rows.push(p + i);
        a_cols.push(p + i);
        a_vals.push(-1.0);
    }
    let a_constr = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_constr, n_var)
        .expect("A csc build");
    let b = vec![0.0_f64; m_constr];
    let cts = vec![ConstraintType::Le; m_constr];

    // 6. bounds: z free, t >= 0
    let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n_var];
    for i in 0..p {
        bounds[p + i] = (0.0, f64::INFINITY);
    }

    QpProblem::new(q, c, a_constr, b, bounds, cts).expect("QpProblem::new")
}

/// 小規模合成 LASSO で deadline honor を smoke-test する。
/// 規模は solver が optimal で finite に終わる範囲、watchdog で infinite loop を catch。
#[test]
fn qp_synthetic_lasso_small_honors_deadline() {
    // p=30: n=60, m=60. dense Q (900 nnz)。
    let problem = make_synthetic_lasso(30, 30, 0.1, 0xC0FFEE);
    let budget = 5.0_f64;
    let watchdog = Duration::from_secs_f64(budget + SLACK_FOR_POSTPROCESS_SEC);
    let (status, _wall) = solve_with_watchdog(problem, budget, watchdog, "synth_lasso_small");
    assert!(
        matches!(
            status,
            SolveStatus::Optimal
                | SolveStatus::Timeout
                | SolveStatus::SuboptimalSolution
                | SolveStatus::NumericalError
                | SolveStatus::Infeasible
                | SolveStatus::LocallyOptimal
        ),
        "[synth_lasso_small] unexpected status {:?}",
        status
    );
}

/// 中規模合成 LASSO: p=150 → n=300, m=300, Q dense 22500 nnz。
/// IPPMM iteration あたり O(n^3) ≈ 2.7e7 ops、数十回 iter で wall ~0.5-2s。
/// budget=1s で solver が deadline check を踏み越えると watchdog (1+5=6s) で
/// 検出される。
#[test]
fn qp_synthetic_lasso_mid_honors_deadline() {
    let problem = make_synthetic_lasso(150, 120, 0.05, 0xBADCAFE);
    let budget = 1.0_f64;
    let watchdog = Duration::from_secs_f64(budget + SLACK_FOR_POSTPROCESS_SEC);
    let (status, _wall) = solve_with_watchdog(problem, budget, watchdog, "synth_lasso_mid");
    assert!(
        matches!(
            status,
            SolveStatus::Optimal
                | SolveStatus::Timeout
                | SolveStatus::SuboptimalSolution
                | SolveStatus::NumericalError
                | SolveStatus::Infeasible
                | SolveStatus::LocallyOptimal
        ),
        "[synth_lasso_mid] unexpected status {:?}",
        status
    );
}
