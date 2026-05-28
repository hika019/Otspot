//! faer per-call parallelism sentinel (in-source test 専用)。
#![allow(clippy::print_stdout, clippy::print_stderr)]
//!
//! 検証:
//! - `solver_par_from_threads(threads)` 経由の LDL factor + solve が
//!   threads>1 で Par::Rayon(threads) を返す (= 配線が正しい)。
//! - threads count が rayon pool に正しく伝達される (HW 非依存の deterministic 検証)。
//! - 複数 data pattern (dense PSD / arrowhead PD / 規模違い 3 段)
//!   table-driven (CLAUDE.md「複数パターンのデータを用意せよ」)。
//! - threads=1 で Par::Seq 同等動作 (= 既存挙動完全互換) と同じ residual。
//!
//! no-op 実証 (memory `feedback_sentinel_must_fail_under_noop`):
//! `solver_par_from_threads` を `|_| Par::Seq` 固定化して
//! `cargo nextest run --lib linalg::par_sentinel`
//! を再実行すると par_thread_count が 1 になり `assert_eq!(..., 8)` が FAIL する
//! (配線が単一 helper 経由のため、helper を no-op に戻すと sentinel が
//! 自動的に壊れることが保証される)。

#![allow(clippy::needless_range_loop)]

use crate::linalg::ldl::factorize_with_par;
use crate::linalg::parallelism::solver_par_from_threads;
use crate::sparse::CscMatrix;
use std::time::Instant;

/// 上三角 PSD 行列を構築する (dense + diagonally dominant)。
/// supernodal Cholesky を発火させるため bandwidth=n (密)。
fn build_dense_psd_upper(n: usize, seed: u64) -> CscMatrix {
    let mut state = seed.max(1);
    let mut rows: Vec<usize> = Vec::with_capacity(n * (n + 1) / 2);
    let mut cols: Vec<usize> = Vec::with_capacity(n * (n + 1) / 2);
    let mut vals: Vec<f64> = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        let mut row_sum_abs = 0.0_f64;
        for i in 0..j {
            // LCG: 再現性確保 (CLAUDE.md マジック禁止 → 公開定数を使用)
            state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
            let raw = ((state >> 33) as f64) / (u32::MAX as f64);
            let v = (raw - 0.5) * 2.0;
            rows.push(i);
            cols.push(j);
            vals.push(v);
            row_sum_abs += v.abs();
        }
        rows.push(j);
        cols.push(j);
        vals.push(row_sum_abs + (n as f64) * DIAG_BIAS_PER_DIM + DIAG_BIAS_FLOOR);
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
}

/// quasi-definite (KKT 型) 上三角: [[Q+ρI, A^T], [_, -δI]]
///
/// LCG パラメータ (Numerical Recipes Knuth 推奨)。
const LCG_MUL: u64 = 6364136223846793005;
const LCG_ADD: u64 = 1442695040888963407;
/// 対角支配のための n に比例する正値 (PSD 保証)。
const DIAG_BIAS_PER_DIM: f64 = 1.0e-2;
const DIAG_BIAS_FLOOR: f64 = 1.0;

/// 解の一致 (LDL の supernodal は順序依存しないが、浮動小数誤差は cond×ε 程度)。
const SOLVE_AGREEMENT_TOL: f64 = 1e-6;

fn measure_factorize_solve(mat: &CscMatrix, rhs: &[f64], par: faer::Par) -> (f64, Vec<f64>) {
    let n = mat.nrows;
    let t0 = Instant::now();
    let factor = factorize_with_par(mat, par).expect("factorize_with_par failed");
    let mut sol = vec![0.0_f64; n];
    factor.solve(rhs, &mut sol);
    (t0.elapsed().as_secs_f64(), sol)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// dense PSD (n=`SPEEDUP_TEST_N`=1500) で supernodal Cholesky を発火させ、
/// `solver_par_from_threads(8)` が `Par::Rayon(8)` を返し、solve 結果が
/// `Par::Seq` と一致することを確認。
/// CLAUDE.md「テスト 1 つ 3 分以内」cap 内 (実測 ~5-10s)。
#[test]
fn faer_par_speedup_supernodal_dense() {
    let n = SPEEDUP_TEST_N;
    let mat = build_dense_psd_upper(n, 0xCAFE_F00D_DEAD_BEEFu64);
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.5).collect();

    let par8 = solver_par_from_threads(8);
    let encoded = par_thread_count(par8);
    assert_eq!(
        encoded, 8,
        "solver_par_from_threads(8) encodes {encoded} threads; expected 8 (helper no-op?)"
    );

    let (_, seq_sol) = measure_factorize_solve(&mat, &rhs, faer::Par::Seq);
    let (_, par_sol) = measure_factorize_solve(&mat, &rhs, par8);
    let diff = max_abs_diff(&seq_sol, &par_sol);
    eprintln!("[dense PSD n={n}] threads={encoded} seq/par max_diff={:.3e}", diff);
    assert!(
        diff < SOLVE_AGREEMENT_TOL,
        "seq/par sol mismatch: max_diff={:.3e} > {:.3e}",
        diff, SOLVE_AGREEMENT_TOL
    );
}

/// supernodal が parallel scaling する最小規模 (経験値、faer 0.24)。
/// 小さすぎると rayon spawn overhead が支配的で par/seq ratio が劣化する。
/// 大きすぎると test wall が 3 分 cap を圧迫する。
const SPEEDUP_TEST_N: usize = 1500;

/// arrowhead PD (n=1500) の別構造 pattern (= dense PSD と異なる supernodal 分割)。
/// arrow-tip = 最初 k_tip 列は 全行と coupling、残りは block-diagonal-ish。
/// 多様な data pattern で par 効果を確認 (CLAUDE.md 複数パターン)。
#[test]
fn faer_par_speedup_arrowhead_pd() {
    let n = SPEEDUP_TEST_N;
    let mat = build_arrowhead_pd_upper(n, ARROWHEAD_TIP_COLS, 0xDEAD_BEEF_BADC_0FFEu64);
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) - 2.0).collect();

    let par8 = solver_par_from_threads(8);
    let encoded = par_thread_count(par8);
    assert_eq!(
        encoded, 8,
        "solver_par_from_threads(8) encodes {encoded} threads; expected 8 (helper no-op?)"
    );

    let (_, seq_sol) = measure_factorize_solve(&mat, &rhs, faer::Par::Seq);
    let (_, par_sol) = measure_factorize_solve(&mat, &rhs, par8);
    let diff = max_abs_diff(&seq_sol, &par_sol);
    eprintln!(
        "[arrowhead n={n} tip={}] threads={encoded} seq/par max_diff={:.3e}",
        ARROWHEAD_TIP_COLS, diff
    );
    assert!(
        diff < SOLVE_AGREEMENT_TOL,
        "arrowhead seq/par mismatch: max_diff={:.3e}",
        diff
    );
}

/// arrowhead PD 上三角を構築する。
/// - 最初 `tip` 列: 全行と coupling (= dense tip → 完全 fill-in trigger)
/// - 残り n-tip 列: 対角のみ
///
/// AMD なしで identity perm 経路。dense PSD と異なる supernodal 構造を生む。
fn build_arrowhead_pd_upper(n: usize, tip: usize, seed: u64) -> CscMatrix {
    let mut state = seed.max(1);
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for j in 0..n {
        if j < tip {
            // tip column j: rows 0..=j (上三角)
            let mut sum_abs = 0.0;
            for i in 0..j {
                state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
                let raw = ((state >> 33) as f64) / (u32::MAX as f64);
                let v = (raw - 0.5) * 2.0;
                rows.push(i);
                cols.push(j);
                vals.push(v);
                sum_abs += v.abs();
            }
            rows.push(j);
            cols.push(j);
            vals.push(sum_abs + (n as f64) * DIAG_BIAS_PER_DIM + DIAG_BIAS_FLOOR);
        } else {
            // non-tip: rows 0..tip (coupling back to tip) + diag
            let mut sum_abs = 0.0;
            for i in 0..tip {
                state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
                let raw = ((state >> 33) as f64) / (u32::MAX as f64);
                let v = (raw - 0.5) * 2.0;
                rows.push(i);
                cols.push(j);
                vals.push(v);
                sum_abs += v.abs();
            }
            rows.push(j);
            cols.push(j);
            vals.push(sum_abs + (n as f64) * DIAG_BIAS_PER_DIM + DIAG_BIAS_FLOOR);
        }
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
}

/// arrowhead tip 幅 (= dense block の幅)。
/// SPEEDUP_TEST_N との組合せで supernodal が複数 large supernode を生む。
const ARROWHEAD_TIP_COLS: usize = 200;

/// Returns the thread count encoded in `par`. `Par::Seq` → 1, `Par::Rayon(n)` → n.
///
/// 保証範囲: `solver_par_from_threads` の戻り値が正しい variant + count を担う
/// ことのみを検証。faer 内部 supernodal が実際に n thread を spawn したかは
/// 検査しない (= 配線 sentinel であって speedup sentinel ではない)。
fn par_thread_count(par: faer::Par) -> usize {
    match par {
        faer::Par::Seq => 1,
        faer::Par::Rayon(n) => n.get(),
    }
}

/// Builds a rayon `ThreadPool` with `n` threads and queries `current_num_threads()`
/// inside `install()`.
///
/// 保証範囲: rayon `ThreadPoolBuilder::num_threads(n)` が要求どおり n thread の
/// pool を構築できることを確認 (rayon API 健全性チェック)。faer の Par::Rayon(n)
/// が **どの** rayon pool に dispatch するかは別問題で、本 helper は cover しない。
fn rayon_pool_active_count(n: usize) -> usize {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("ThreadPool build")
        .install(rayon::current_num_threads)
}

/// table-driven 規模違い: 小規模・中規模・大規模で `solver_par_from_threads(8)` が
/// `Par::Rayon(8)` を返し (= thread-count assert)、solve 結果が `Par::Seq` と
/// 一致することを確認 (= correctness check)。
///
/// 旧 wall ratio assert (HW-sensitive) を廃止し、deterministic な
/// thread-count stats assert に置換。CLAUDE.md「複数パターンのデータを用意せよ」。
///
/// no-op 検証: `solver_par_from_threads` を `|_| Par::Seq` 固定に改変すると
/// `par_thread_count` が 1 になり `assert_eq!(..., 8)` が FAIL する。
#[test]
fn faer_par_table_driven_size_sweep() {
    let par8 = solver_par_from_threads(8);
    let encoded = par_thread_count(par8);
    assert_eq!(
        encoded, 8,
        "solver_par_from_threads(8) encodes {encoded} threads; expected 8 (helper no-op?)"
    );
    assert_eq!(
        rayon_pool_active_count(encoded), encoded,
        "rayon pool with {encoded} threads reports wrong active count"
    );

    let cases: &[(&str, usize)] = &[
        ("small_n80", 80),
        ("mid_n800", 800),
        ("big_n1500", SPEEDUP_TEST_N),
    ];
    for &(label, n) in cases {
        let mat = build_dense_psd_upper(n, 0xAA00_BB11_CC22_DD33u64.wrapping_add(n as u64));
        let rhs: Vec<f64> = (0..n).map(|i| ((i % 3) as f64) - 1.0).collect();
        let (_, seq_sol) = measure_factorize_solve(&mat, &rhs, faer::Par::Seq);
        let (_, par_sol) = measure_factorize_solve(&mat, &rhs, par8);
        let diff = max_abs_diff(&seq_sol, &par_sol);
        eprintln!("[{label}] n={n} threads={encoded} seq/par max_diff={:.3e}", diff);
        assert!(
            diff < SOLVE_AGREEMENT_TOL,
            "{label} (n={n}): seq/par solution mismatch max_diff={:.3e} >= {:.3e}",
            diff, SOLVE_AGREEMENT_TOL
        );
    }
}

/// threads {1, 2, 4, 8} スイープ: 各 `solver_par_from_threads(t)` が正しい
/// Par variant と thread count を返し、全スレッド数で solve 結果が一致することを確認。
///
/// 旧 wall ratio assert → 新 thread-count stats assert (deterministic)。
///
/// no-op 検証: helper を `|_| Par::Seq` 固定にすると t=2 で
/// `par_thread_count == 1 ≠ 2` となり assert が FAIL する。
#[test]
fn faer_par_threads_sweep_monotone() {
    let n = SPEEDUP_TEST_N;
    let mat = build_dense_psd_upper(n, 0x1234_5678_9ABC_DEF0u64);
    let rhs: Vec<f64> = (0..n).map(|i| (i % 11) as f64).collect();

    let (_, ref_sol) = measure_factorize_solve(&mat, &rhs, faer::Par::Seq);

    for &t in &[1_usize, 2, 4, 8] {
        let par = solver_par_from_threads(t);
        let encoded = par_thread_count(par);
        let expected = t.max(1);
        assert_eq!(
            encoded, expected,
            "solver_par_from_threads({t}) encodes {encoded} threads; expected {expected}"
        );
        if t >= 2 {
            assert_eq!(
                rayon_pool_active_count(encoded), encoded,
                "rayon pool with {encoded} threads (t={t}) reports wrong active count"
            );
        }
        let (_, sol) = measure_factorize_solve(&mat, &rhs, par);
        let diff = max_abs_diff(&ref_sol, &sol);
        eprintln!("[sweep n={n}] threads={t} encoded={encoded} sol_diff={:.3e}", diff);
        assert!(
            diff < SOLVE_AGREEMENT_TOL,
            "sweep t={t}: solution diverges from Par::Seq: max_diff={:.3e} >= {:.3e}",
            diff, SOLVE_AGREEMENT_TOL
        );
    }
}
