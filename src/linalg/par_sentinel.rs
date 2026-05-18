//! faer per-call parallelism sentinel (#31, in-source test 専用)。
//!
//! 検証:
//! - `solver_par_from_threads(threads)` 経由の LDL factor + solve が
//!   threads>1 で wall-clock 短縮を示す (= 単発 LP/QP solve で実効並列が効く)。
//! - 複数 data pattern (dense PSD / quasidefinite saddle-point / 規模違い)
//!   table-driven (CLAUDE.md「複数パターンのデータを用意せよ」)。
//! - threads=1 で Par::Seq 同等動作 (= 既存挙動完全互換) と同じ residual。
//!
//! no-op 実証 (memory `feedback_sentinel_must_fail_under_noop`):
//! `solver_par_from_threads` を `|_| Par::Seq` 固定化して
//! `cargo nextest run --lib linalg::par_sentinel`
//! を再実行すると wall_ratio が ~1.0 となり assert が FAIL する
//! (配線が単一 helper 経由のため、helper を no-op に戻すと sentinel が
//! 自動的に壊れることが保証される)。

#![allow(clippy::needless_range_loop)]

use crate::linalg::ldl::{
    factorize_quasidefinite_with_cached_perm_par, factorize_with_par,
};
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
/// A は dense ランダム (entries ~ [-0.05, 0.05]) で K = Q + δ^{-1}·A^T·A の
/// spectrum を保守的に保つ (δ=1e-2 と組合せで K が安定 PD)。
/// supernodal 経路に乗る規模 (n+m >= 800) で par scaling を sentinel する。
fn build_quasidefinite_upper(n: usize, m: usize, seed: u64) -> CscMatrix {
    let mut state = seed.max(1);
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for j in 0..n {
        let mut row_sum_abs = 0.0_f64;
        for i in 0..j {
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
        vals.push(row_sum_abs + (n as f64) * DIAG_BIAS_PER_DIM + DIAG_BIAS_FLOOR + QD_RHO);
    }
    for col in 0..m {
        let cglobal = n + col;
        for row in 0..n {
            state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
            let raw = ((state >> 33) as f64) / (u32::MAX as f64);
            let v = (raw - 0.5) * 2.0 * QD_CROSS_SCALE;
            rows.push(row);
            cols.push(cglobal);
            vals.push(v);
        }
        rows.push(cglobal);
        cols.push(cglobal);
        vals.push(-QD_DELTA);
    }
    let dim = n + m;
    CscMatrix::from_triplets(&rows, &cols, &vals, dim, dim).unwrap()
}

/// A 行列のスケール (δ^{-1}·A^T·A が小さく抑えられ、K が安定 PD)。
const QD_CROSS_SCALE: f64 = 0.05;

/// LCG パラメータ (Numerical Recipes Knuth 推奨)。
const LCG_MUL: u64 = 6364136223846793005;
const LCG_ADD: u64 = 1442695040888963407;
/// 対角支配のための n に比例する正値 (PSD 保証)。
const DIAG_BIAS_PER_DIM: f64 = 1.0e-2;
const DIAG_BIAS_FLOOR: f64 = 1.0;
/// quasi-definite regularization (LDL stable factor 用)。δ=1e-2 で
/// δ^{-1}·A^T·A の amplification を抑える (cond(K) ~1e4 で f64 内安全)。
const QD_RHO: f64 = 1.0e-2;
const QD_DELTA: f64 = 1.0e-2;

/// 解の一致 (LDL の supernodal は順序依存しないが、浮動小数誤差は cond×ε 程度)。
const SOLVE_AGREEMENT_TOL: f64 = 1e-6;

/// 速度 sentinel ceiling。par/seq 比がこの値未満なら sentinel PASS。
/// no-op (helper が Par::Seq 固定化) なら ratio ≈ 1.0 で FAIL する。
const SPEEDUP_RATIO_CEILING: f64 = 0.85;

/// rayon thread-pool init を計測外に逃すための warmup 回数。
const WARMUP_ITERS: usize = 1;
/// 各設定の best-of 回数 (CPU noise 緩和)。
const BEST_OF: usize = 3;

fn measure_factorize_solve(
    mat: &CscMatrix,
    rhs: &[f64],
    par: faer::Par,
) -> (f64, Vec<f64>) {
    let n = mat.nrows;
    let t0 = Instant::now();
    let factor = factorize_with_par(mat, par).expect("factorize_with_par failed");
    let mut sol = vec![0.0_f64; n];
    factor.solve(rhs, &mut sol);
    (t0.elapsed().as_secs_f64(), sol)
}

fn measure_qd_factorize_solve(
    mat: &CscMatrix,
    perm: &[usize],
    rhs: &[f64],
    par: faer::Par,
) -> (f64, Vec<f64>) {
    let n = mat.nrows;
    let t0 = Instant::now();
    let factor = factorize_quasidefinite_with_cached_perm_par(mat, perm, None, par)
        .expect("factorize_quasidefinite_with_cached_perm_par failed");
    let mut sol = vec![0.0_f64; n];
    factor.solve(rhs, &mut sol);
    (t0.elapsed().as_secs_f64(), sol)
}

fn best_wall(
    mat: &CscMatrix,
    rhs: &[f64],
    par: faer::Par,
) -> (f64, Vec<f64>) {
    for _ in 0..WARMUP_ITERS {
        let _ = measure_factorize_solve(mat, rhs, par);
    }
    let mut best = f64::INFINITY;
    let mut last_sol = Vec::new();
    for _ in 0..BEST_OF {
        let (w, sol) = measure_factorize_solve(mat, rhs, par);
        if w < best {
            best = w;
            last_sol = sol;
        }
    }
    (best, last_sol)
}

fn best_wall_qd(
    mat: &CscMatrix,
    perm: &[usize],
    rhs: &[f64],
    par: faer::Par,
) -> (f64, Vec<f64>) {
    for _ in 0..WARMUP_ITERS {
        let _ = measure_qd_factorize_solve(mat, perm, rhs, par);
    }
    let mut best = f64::INFINITY;
    let mut last_sol = Vec::new();
    for _ in 0..BEST_OF {
        let (w, sol) = measure_qd_factorize_solve(mat, perm, rhs, par);
        if w < best {
            best = w;
            last_sol = sol;
        }
    }
    (best, last_sol)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// dense PSD (n=2000) で supernodal Cholesky を発火させ、wall 短縮を assert。
/// n=2000 dense → factor ~150ms 級 (Apple M-series 単一スレッド)、
/// rayon spawn overhead を十分に上回り par scaling が観測可能。
/// CLAUDE.md「テスト 1 つ 3 分以内」cap 内 (実測 ~5-10s)。
#[test]
fn faer_par_speedup_supernodal_dense() {
    let n = SPEEDUP_TEST_N;
    let mat = build_dense_psd_upper(n, 0xCAFE_F00D_DEAD_BEEFu64);
    let rhs: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.5).collect();

    let (seq_best, seq_sol) = best_wall(&mat, &rhs, faer::Par::Seq);
    let (par_best, par_sol) = best_wall(&mat, &rhs, solver_par_from_threads(8));
    let ratio = par_best / seq_best;
    eprintln!(
        "[dense PSD n={n}] seq_best={:.4}s par_best={:.4}s ratio={:.3} (ceiling={:.2})",
        seq_best, par_best, ratio, SPEEDUP_RATIO_CEILING
    );

    let diff = max_abs_diff(&seq_sol, &par_sol);
    assert!(
        diff < SOLVE_AGREEMENT_TOL,
        "seq/par sol mismatch: max_diff={:.3e} > {:.3e}",
        diff, SOLVE_AGREEMENT_TOL
    );
    assert!(
        ratio < SPEEDUP_RATIO_CEILING,
        "faer per-call parallelism speedup not observed: ratio={:.3} >= {:.2}. \
         helper 経由が no-op 化している可能性 (= 配線壊れ)。",
        ratio, SPEEDUP_RATIO_CEILING
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

    let (seq_best, seq_sol) = best_wall(&mat, &rhs, faer::Par::Seq);
    let (par_best, par_sol) = best_wall(&mat, &rhs, solver_par_from_threads(8));
    let ratio = par_best / seq_best;
    eprintln!(
        "[arrowhead n={n} tip={}] seq_best={:.4}s par_best={:.4}s ratio={:.3}",
        ARROWHEAD_TIP_COLS, seq_best, par_best, ratio
    );

    let diff = max_abs_diff(&seq_sol, &par_sol);
    assert!(
        diff < SOLVE_AGREEMENT_TOL,
        "arrowhead seq/par mismatch: max_diff={:.3e}",
        diff
    );
    assert!(
        ratio < SPEEDUP_RATIO_CEILING,
        "arrowhead faer per-call parallelism speedup not observed: ratio={:.3} >= {:.2}",
        ratio, SPEEDUP_RATIO_CEILING
    );
}

/// arrowhead PD 上三角を構築する。
/// - 最初 `tip` 列: 全行と coupling (= dense tip → 完全 fill-in trigger)
/// - 残り n-tip 列: 対角のみ
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

/// table-driven 規模違い: 小規模では overhead 許容、中-大規模で明確な短縮。
///
/// CLAUDE.md「テストは可能な限り全分岐」「複数パターンのデータ」:
/// - 小規模 (n=80): rayon spawn overhead が支配的、helper が "強制 Rayon で
///   劇的に劣化" しないこと (= 2x 以下) を sentinel
/// - 中規模 (n=800): supernodal 境界、par が必ず実効
/// - 大規模 (n=1500): SPEEDUP_TEST_N、par scaling が支配的
#[test]
fn faer_par_table_driven_size_sweep() {
    let cases: &[(&str, usize, f64)] = &[
        ("small_n80", 80, 2.0),
        ("mid_n800", 800, SPEEDUP_RATIO_CEILING),
        ("big_n1500", SPEEDUP_TEST_N, SPEEDUP_RATIO_CEILING),
    ];
    for &(label, n, max_ratio) in cases {
        let mat = build_dense_psd_upper(n, 0xAA00_BB11_CC22_DD33u64.wrapping_add(n as u64));
        let rhs: Vec<f64> = (0..n).map(|i| ((i % 3) as f64) - 1.0).collect();
        let (seq_best, _) = best_wall(&mat, &rhs, faer::Par::Seq);
        let (par_best, _) = best_wall(&mat, &rhs, solver_par_from_threads(8));
        let ratio = par_best / seq_best;
        eprintln!(
            "[{label}] n={n} seq_best={:.4}s par_best={:.4}s ratio={:.3} max_ratio={:.2}",
            seq_best, par_best, ratio, max_ratio
        );
        assert!(
            ratio < max_ratio,
            "{label} (n={n}): par/seq ratio={:.3} >= max={:.2}",
            ratio, max_ratio
        );
    }
}

/// threads {1, 2, 4, 8} スイープ: threads=8 wall < threads=1 wall × ceiling。
/// helper が threads を捨てて常に Par::Seq を返していると ratio≈1.0 で FAIL する。
#[test]
fn faer_par_threads_sweep_monotone() {
    let n = SPEEDUP_TEST_N;
    let mat = build_dense_psd_upper(n, 0x1234_5678_9ABC_DEF0u64);
    let rhs: Vec<f64> = (0..n).map(|i| (i % 11) as f64).collect();

    // warmup: rayon thread pool init を計測外
    let _ = measure_factorize_solve(&mat, &rhs, solver_par_from_threads(8));

    let threads_list = [1_usize, 2, 4, 8];
    let mut walls = Vec::new();
    for &t in &threads_list {
        let (w, _) = best_wall(&mat, &rhs, solver_par_from_threads(t));
        walls.push((t, w));
    }
    for (t, w) in &walls {
        eprintln!("[sweep n={n}] threads={t} wall={:.4}s", w);
    }
    let w1 = walls[0].1;
    let w8 = walls[3].1;
    let ratio_8_vs_1 = w8 / w1;
    eprintln!("[sweep n={n}] w8/w1={:.3}", ratio_8_vs_1);
    assert!(
        ratio_8_vs_1 < SPEEDUP_RATIO_CEILING,
        "threads sweep: t=8 wall/t=1 = {:.3} >= {:.2} (per-call parallelism no-op)",
        ratio_8_vs_1, SPEEDUP_RATIO_CEILING
    );
}
