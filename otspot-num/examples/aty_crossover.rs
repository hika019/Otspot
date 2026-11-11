//! W2-1b crossover probe (throwaway, NOT committed): where does the column-split
//! parallel Aᵀy overtake the serial column-dot kernel, measured in the
//! PRODUCTION call shape — repeated per-trial calls made from inside
//! `with_solver_pool` (the solve path; install hits the SAME cached pool the
//! plan owns, no worker migration), plus the outside-pool caller for contrast.
//!
//! run: cargo run --release -p otspot-num --example aty_crossover -- <threads>

use otspot_num::linalg::parallelism::{solver_thread_pool, with_solver_pool};
use otspot_num::sparse::{mat_t_vec_mul_into, AtyParallelPlan};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn csc(
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_ind: Vec<usize>,
    values: Vec<f64>,
) -> otspot_num::sparse::CscMatrix {
    otspot_num::sparse::CscMatrix::from_raw_parts(nrows, ncols, col_ptr, row_ind, values)
}

/// Maros-style uniform sparse matrix: ~3 nnz per column.
fn uniform_matrix(seed: u64, nnz: usize) -> otspot_num::sparse::CscMatrix {
    let mut rng = Lcg(seed);
    let ncols = (nnz / 3).max(2);
    let nrows = ncols + ncols / 8 + 1;
    let mut col_ptr = Vec::with_capacity(ncols + 1);
    let mut row_ind = Vec::with_capacity(nnz + 8);
    let mut values = Vec::with_capacity(nnz + 8);
    col_ptr.push(0usize);
    let base = (nnz / ncols).max(1);
    for j in 0..ncols {
        let k = base + (rng.next() % 2) as usize;
        let mut rows: Vec<usize> = (0..k)
            .map(|_| (rng.next() % nrows as u64) as usize)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        for &r in &rows {
            row_ind.push(r);
            values.push(0.5 + ((j + r) % 7) as f64 * 0.25);
        }
        col_ptr.push(row_ind.len());
    }
    csc(nrows, ncols, col_ptr, row_ind, values)
}

/// Skewed: one hot column carries ~40% of nnz, rest ~3/col (stresses the
/// balanced column split).
fn skewed_matrix(seed: u64, nnz: usize) -> otspot_num::sparse::CscMatrix {
    let a_small = uniform_matrix(seed, nnz * 60 / 100);
    let hot_len = (nnz * 40 / 100).max(2);
    let nrows = a_small.nrows().max(hot_len);
    let mut col_ptr = vec![0usize];
    let mut row_ind: Vec<usize> = (0..hot_len).collect();
    let mut values: Vec<f64> = vec![0.75; hot_len];
    col_ptr.push(row_ind.len());
    for j in 0..a_small.ncols() {
        for k in a_small.col_ptr()[j]..a_small.col_ptr()[j + 1] {
            row_ind.push(a_small.row_ind()[k]);
            values.push(a_small.values()[k]);
        }
        col_ptr.push(row_ind.len());
    }
    csc(nrows, a_small.ncols() + 1, col_ptr, row_ind, values)
}

/// Gate-bypassing plan (production `AtyParallelPlan::new` refuses sub-gate
/// nnz; the probe must measure there too). Split + install mirror the private
/// production code; the final sanity block checks both agree at gated sizes.
struct ProbePlan {
    bounds: Vec<usize>,
    pool: Arc<rayon::ThreadPool>,
}

fn probe_split(col_ptr: &[usize], parts: usize) -> Vec<usize> {
    let items = col_ptr.len() - 1;
    let total = col_ptr[items];
    let mut bounds = Vec::with_capacity(parts + 1);
    bounds.push(0);
    let mut k = 0usize;
    for p in 1..parts {
        let target = ((total as u128 * p as u128) / parts as u128) as usize;
        while k < items && col_ptr[k + 1] < target {
            k += 1;
        }
        bounds.push(k);
    }
    bounds.push(items);
    bounds
}

fn probe_plan(a: &otspot_num::sparse::CscMatrix, threads: usize) -> ProbePlan {
    let pool = solver_thread_pool(threads).expect("pool");
    ProbePlan {
        bounds: probe_split(a.col_ptr(), threads.min(a.ncols())),
        pool,
    }
}

fn probe_run(a: &otspot_num::sparse::CscMatrix, y: &[f64], plan: &ProbePlan, out: &mut [f64]) {
    let cp = a.col_ptr();
    let ri = a.row_ind();
    let vv = a.values();
    plan.pool.install(|| {
        let mut segs: Vec<(usize, &mut [f64])> = Vec::with_capacity(plan.bounds.len() - 1);
        let mut rest = out;
        for w in plan.bounds.windows(2) {
            let (head, tail) = rest.split_at_mut(w[1] - w[0]);
            segs.push((w[0], head));
            rest = tail;
        }
        segs.into_par_iter().for_each(|(j0, seg)| {
            for (idx, slot) in seg.iter_mut().enumerate() {
                let j = j0 + idx;
                let mut sum = 0.0;
                for k in cp[j]..cp[j + 1] {
                    sum += vv[k] * y[ri[k]];
                }
                *slot = sum;
            }
        });
    });
}

fn best_of2(mut f: impl FnMut() -> f64) -> f64 {
    f().min(f())
}

fn best_of2_pair<T: FnMut() -> (f64, f64)>(mut f: T) -> (f64, f64) {
    let a = f();
    let b = f();
    (a.0.min(b.0), a.1.min(b.1))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    println!("# probe: production shape (same cached solver pool, install from pool worker)");
    println!(
        "shape\tnnz\tncols\titers\tserial_inpool_ns\tpar_inpool_ns\tpar_outpool_ns\tspeedup_inpool"
    );
    for &nnz in &[
        1_000usize, 2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 16_000, 20_000, 30_000, 50_000,
        100_000, 200_000, 630_000,
    ] {
        for (shape, a) in [
            ("uniform", uniform_matrix(0xABCD_1234 ^ nnz as u64, nnz)),
            ("skewed", skewed_matrix(0x9E37_79B9 ^ nnz as u64, nnz)),
        ] {
            let real_nnz = a.values().len();
            let ncols = a.ncols();
            let y: Vec<f64> = (0..a.nrows())
                .map(|i| ((i % 97) as f64 - 48.0) * 1e-3)
                .collect();
            let mut out_s = vec![0.0f64; ncols];
            let mut out_p = vec![0.0f64; ncols];
            // correctness first (probe plan must equal serial bitwise).
            let _ = mat_t_vec_mul_into(&a, &y, &mut out_s);
            let plan = probe_plan(&a, threads);
            probe_run(&a, &y, &plan, &mut out_p);
            assert!(
                out_s
                    .iter()
                    .zip(&out_p)
                    .all(|(x, z)| x.to_bits() == z.to_bits()),
                "probe par != serial at nnz={real_nnz} ({shape})"
            );
            let iters = (24_000_000usize / real_nnz).clamp(600, 20_000);
            // Production shape: install ONCE (solve entry), loop matvecs inside
            // (PG Cauchy trials). Per-call install would misplace the crossover.
            let (s, p_in) = best_of2_pair(|| {
                with_solver_pool(threads, |_| {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        let _ = mat_t_vec_mul_into(&a, &y, &mut out_s);
                    }
                    let s = t0.elapsed().as_nanos() as f64 / iters as f64;
                    let t1 = Instant::now();
                    for _ in 0..iters {
                        probe_run(&a, &y, &plan, &mut out_p);
                    }
                    let p = t1.elapsed().as_nanos() as f64 / iters as f64;
                    (s, p)
                })
            });
            let p_out = best_of2(|| {
                let t0 = Instant::now();
                for _ in 0..iters {
                    probe_run(&a, &y, &plan, &mut out_p);
                }
                t0.elapsed().as_nanos() as f64 / iters as f64
            });
            println!(
                "{shape}\t{real_nnz}\t{ncols}\t{iters}\t{s:.1}\t{p_in:.1}\t{p_out:.1}\t{}",
                s / p_in
            );
        }
    }
    // Production-API parity at a gated size (30k): the committed plan path
    // must agree bitwise with the serial kernel used for timing above.
    let a = uniform_matrix(0xF00D, 30_000);
    let y: Vec<f64> = (0..a.nrows()).map(|i| (i % 13) as f64 * 1e-2).collect();
    let mut s = vec![0.0; a.ncols()];
    let mut p = vec![0.0; a.ncols()];
    let _ = mat_t_vec_mul_into(&a, &y, &mut s);
    with_solver_pool(threads, |_| {
        let plan = AtyParallelPlan::new(&a, threads).expect("gate must fire at 30k");
        otspot_num::sparse::mat_t_vec_mul_into_parallel(&a, &y, &mut p, &plan)
            .expect("parallel aty");
    });
    assert!(s.iter().zip(&p).all(|(x, z)| x.to_bits() == z.to_bits()));
    println!("# production AtyParallelPlan parity at 30k: OK");
}
