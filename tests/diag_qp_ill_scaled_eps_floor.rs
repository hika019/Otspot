//! ill-scaled QP で `IPM_EPS_NOISE_FLOOR` が IPM 空転を断つ regression sentinel。
//!
//! 仕様 (`src/qp/ipm_core/mod.rs::IPM_EPS_NOISE_FLOOR`):
//!   floor = 100 × machine_eps ≈ 2.22e-14
//!   core.rs:    eps_scaled = max(eps_orig × σ_total,       floor)
//!   scaling.rs: ipm.eps    = max(user_eps  / amplification, floor)
//!
//! anti-pattern (修正前): 4×4 toy + `assert!(elapsed < 5.0)` のみは floor=0 でも常 PASS。
//! `assert!(iter ≤ N)` も無く、問題が toy のため sentinel が機能していなかった。
//!
//! 設計: 中規模 ill-scaled LASSO (p=500 → n_var=1000, σ_total ≈ 6e-11) を 2 seed で評価。
//!   floor active   → IPM 19-20 iter で Optimal
//!   floor = 0 (回帰) → IPM が達成不能 nr_d_rel 目標で 50-100 iter 空転
//! `iter ≤ ITER_BUDGET` で floor 機能を判定。orig-space 残差も併せて gate。
//! 実証: 一時 `IPM_EPS_NOISE_FLOOR = 0.0` 化 → 本 sentinel FAIL を確認済。

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, SolveStatus, SolverResult};
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;

/// LCG-based deterministic pseudo-random in (-1, 1)。`std::rand` 非依存。
fn lcg_unit(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*seed >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
}

/// 合成 LASSO + 列 D-scaling で σ_total ≪ 1 を再現。
///   min 0.5 z^T (A^T A) z - (A^T y)^T z + λ · 1^T t
///     s.t.  z_i - t_i <= 0,  -z_i - t_i <= 0,  t >= 0
/// 列 j に D[j]=10^{u·scale_max} (u ~ Uniform(0,1)) を適用し ill-scale 化。
/// Ruiz の e_min/d_min が極小化されて σ_total が落ちる。
fn lasso_ill_scaled(p: usize, m_data: usize, lambda: f64, scale_max: f64, seed: u64) -> QpProblem {
    let mut rng = seed;
    let mut a_data = vec![vec![0.0_f64; p]; m_data];
    for i in 0..m_data {
        for j in 0..p {
            a_data[i][j] = lcg_unit(&mut rng);
        }
    }
    let mut y = vec![0.0_f64; m_data];
    for i in 0..m_data {
        y[i] = lcg_unit(&mut rng);
    }

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

    let mut d_diag = vec![1.0_f64; n_var];
    let mut rng2 = seed.wrapping_mul(31).wrapping_add(7);
    for j in 0..n_var {
        let u = 0.5 * (lcg_unit(&mut rng2) + 1.0); // [0, 1]
        d_diag[j] = 10.0_f64.powf(u * scale_max);
    }

    // Q' = D Q D (左上 p×p block)
    let mut q_rows = Vec::with_capacity(p * p);
    let mut q_cols = Vec::with_capacity(p * p);
    let mut q_vals = Vec::with_capacity(p * p);
    for j in 0..p {
        for i in 0..p {
            let v = q_dense[j * p + i] * d_diag[i] * d_diag[j];
            if v.abs() > 0.0 {
                q_rows.push(i);
                q_cols.push(j);
                q_vals.push(v);
            }
        }
    }
    let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n_var, n_var).expect("Q build");

    for j in 0..n_var {
        c[j] *= d_diag[j];
    }

    // A' = A_constr · D, A_constr = [[ I_p, -I_p ], [ -I_p, -I_p ]]
    let m_constr = 2 * p;
    let mut a_rows = Vec::with_capacity(4 * p);
    let mut a_cols = Vec::with_capacity(4 * p);
    let mut a_vals = Vec::with_capacity(4 * p);
    for i in 0..p {
        a_rows.push(i);
        a_cols.push(i);
        a_vals.push(d_diag[i]);
        a_rows.push(i);
        a_cols.push(p + i);
        a_vals.push(-d_diag[p + i]);
        a_rows.push(p + i);
        a_cols.push(i);
        a_vals.push(-d_diag[i]);
        a_rows.push(p + i);
        a_cols.push(p + i);
        a_vals.push(-d_diag[p + i]);
    }
    let a_constr =
        CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_constr, n_var).expect("A build");
    let b = vec![0.0_f64; m_constr];
    let cts = vec![ConstraintType::Le; m_constr];

    let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n_var];
    for i in 0..p {
        bounds[p + i] = (0.0, f64::INFINITY);
    }

    QpProblem::new(q, c, a_constr, b, bounds, cts).expect("QpProblem::new")
}

/// IPM iter 上限。floor active 実測 (seed=0x1234/0x9abc) は 19/20 iter、+50% 余裕。
/// floor = 0 (回帰) で seed=0x1234 は 97 iter まで膨張 (5x) → 容易に超える。
const ITER_BUDGET: usize = 30;
/// orig-space primal residual 上限。post-processing が user_eps=1e-6 で gate する想定。
const PRES_REL_BUDGET: f64 = 1e-5;

fn assert_orig_primal_residual_ok(problem: &QpProblem, x: &[f64]) {
    let ax: Vec<f64> = (0..problem.num_constraints)
        .map(|i| {
            let mut s = 0.0;
            for col in 0..problem.num_vars {
                for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                    if problem.a.row_ind[k] == i {
                        s += problem.a.values[k] * x[col];
                    }
                }
            }
            s
        })
        .collect();
    let max_v = ax
        .iter()
        .zip(problem.b.iter())
        .enumerate()
        .map(|(i, (&axi, &bi))| match problem.constraint_types[i] {
            ConstraintType::Le => (axi - bi).max(0.0),
            ConstraintType::Ge => (bi - axi).max(0.0),
            ConstraintType::Eq => (axi - bi).abs(),
            _ => 0.0,
        })
        .fold(0.0_f64, f64::max);
    let denom = 1.0
        + ax.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(
            problem.b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
        );
    let pres_rel = max_v / denom;
    assert!(
        pres_rel < PRES_REL_BUDGET,
        "orig-space pres_rel {:.3e} > budget {:.0e} — post-processing が補えていない",
        pres_rel,
        PRES_REL_BUDGET
    );
}

fn solve_lasso_ill(seed: u64) -> SolverResult {
    // p=500 → n_var=1000 (中規模)、σ_total ≈ 6e-11 で floor 領域に確実に入る。
    let problem = lasso_ill_scaled(500, 500, 0.1, 14.0, seed);
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(20.0);
    let result = solve_qp_with(&problem, &opts);
    assert_orig_primal_residual_ok(&problem, &result.solution);
    result
}

/// seed=0x1234: floor active で 19 iter / floor=0 で 97 iter (5x)。
/// `iter ≤ ITER_BUDGET` で floor 退行を即時検出。
#[test]
fn ill_scaled_iter_budget_seed_0x1234() {
    let r = solve_lasso_ill(0x1234);
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "unexpected status {:?}",
        r.status
    );
    assert!(
        r.iterations <= ITER_BUDGET,
        "seed=0x1234: iter {} > budget {} — IPM_EPS_NOISE_FLOOR が機能していない疑い",
        r.iterations,
        ITER_BUDGET
    );
}

/// seed=0x9abc: floor active で 20 iter / floor=0 で 53 iter (2.7x)。
/// 別 seed でも同じ budget を満たすことで seed bias を排除。
#[test]
fn ill_scaled_iter_budget_seed_0x9abc() {
    let r = solve_lasso_ill(0x9abc);
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "unexpected status {:?}",
        r.status
    );
    assert!(
        r.iterations <= ITER_BUDGET,
        "seed=0x9abc: iter {} > budget {} — IPM_EPS_NOISE_FLOOR が機能していない疑い",
        r.iterations,
        ITER_BUDGET
    );
}

// ---------------------------------------------------------------------------
// scaling.rs:: EPS_FLOOR 経路を **単独で** sentinel する追加 test 群。
//
// 既存 2 test は `IPM_EPS_NOISE_FLOOR` (ipm_core/mod.rs) を 0 化した場合に FAIL する
// が、`scaling.rs::EPS_FLOOR` のみを 0 化した場合は `ipm_solver/core::eps_tighten`
// が `presolve_result` 経由で IPM eps を `IPM_EPS_NOISE_FLOOR` に floor し直す
// ため依然 PASS。scaling.rs 経路の回帰を isolate するには `presolve=false` で
// eps_tighten を no-op 化し、`scaling.rs::EPS_FLOOR` を唯一の guard にする必要がある。
//
// 実証 (本 file commit 時): scaling.rs:12 を `EPS_FLOOR = 0.0` に書換 →
// `no_presolve_*` 群が FAIL (iter budget 超 + Optimal 不達)。restore で PASS。
// ---------------------------------------------------------------------------

/// presolve=false で scaling.rs::EPS_FLOOR を唯一の floor とする solver 呼出。
/// `ipm_solver/core::eps_tighten` は `presolve_result.ruiz_scaler` を見て floor を
/// 印加するため presolve=false なら sigma_total=1.0 = no-op となり、`scaling.rs::EPS_FLOOR`
/// が唯一の noise floor guard となる。
fn solve_no_presolve(problem: &QpProblem) -> SolverResult {
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(20.0);
    solve_qp_with(problem, &opts)
}

/// scaling.rs path 用 IPM iter budget。floor active 実測 (seed 0xc3/0xd4/0xe5) は
/// 17-23 iter、floor=0 化で 60-100 iter まで膨張するため `40` で十分に検出可能。
const ITER_BUDGET_NO_PRESOLVE: usize = 40;

/// LASSO no-presolve seed=0xc3: scaling.rs floor 単独の sentinel (eps_tighten 介在無し)。
/// floor active で 17 iter / floor=0 で 82 iter (4.8x)。
#[test]
fn no_presolve_lasso_seed_0xc3() {
    let problem = lasso_ill_scaled(120, 100, 0.1, 14.0, 0xc3);
    let r = solve_no_presolve(&problem);
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "unexpected status {:?}",
        r.status
    );
    assert!(
        r.iterations <= ITER_BUDGET_NO_PRESOLVE,
        "LASSO no-presolve seed=0xc3: iter {} > budget {} — scaling.rs::EPS_FLOOR が機能していない疑い",
        r.iterations,
        ITER_BUDGET_NO_PRESOLVE
    );
    assert_orig_primal_residual_ok(&problem, &r.solution);
}

/// 別 seed=0xd4: seed bias 排除。
#[test]
fn no_presolve_lasso_seed_0xd4() {
    let problem = lasso_ill_scaled(120, 100, 0.1, 14.0, 0xd4);
    let r = solve_no_presolve(&problem);
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "unexpected status {:?}",
        r.status
    );
    assert!(
        r.iterations <= ITER_BUDGET_NO_PRESOLVE,
        "LASSO no-presolve seed=0xd4: iter {} > budget {} — scaling.rs::EPS_FLOOR が機能していない疑い",
        r.iterations,
        ITER_BUDGET_NO_PRESOLVE
    );
    assert_orig_primal_residual_ok(&problem, &r.solution);
}

/// 別 seed=0xe5 + 異なる λ (0.5): regularization 強度違いでも同 budget を満たすこと。
#[test]
fn no_presolve_lasso_seed_0xe5_lambda_0_5() {
    let problem = lasso_ill_scaled(120, 100, 0.5, 14.0, 0xe5);
    let r = solve_no_presolve(&problem);
    assert!(
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "unexpected status {:?}",
        r.status
    );
    assert!(
        r.iterations <= ITER_BUDGET_NO_PRESOLVE,
        "LASSO no-presolve seed=0xe5 λ=0.5: iter {} > budget {} — scaling.rs::EPS_FLOOR が機能していない疑い",
        r.iterations,
        ITER_BUDGET_NO_PRESOLVE
    );
    assert_orig_primal_residual_ok(&problem, &r.solution);
}
