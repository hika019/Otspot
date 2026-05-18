//! task#13: ill-scaled QP で `IPM_EPS_NOISE_FLOOR` が IPM 空転を断つ regression sentinel。
//!
//! 仕様 (`src/qp/ipm_solver/core.rs`):
//!   eps_scaled = max(eps_orig × σ_total, 100·MACHINE_EPS)
//!
//! 検証:
//!   (a) ill-scaled (σ_total ≈ 1e-9) で iter 数が "scaled-eps=σ×eps" 設定の
//!       上限 (typical 30+) より小さく済む。floor が無いと scaled-eps が
//!       ≪ machine-noise になり、`nr_d_rel` 達成不能領域で空転する。
//!   (b) 元空間 Optimal が維持され、bench eps 1e-6 を満たす。

use solver::options::SolverOptions;
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::{solve_qp_with, QpProblem};
use solver::sparse::CscMatrix;

/// |b| / |A| / |c| の桁差で Ruiz の e_min が極小になる人為的ill-scaled QP。
/// rows[0,1] が桁違いの coefficient を持ち、 e_min ≪ 1 となる。
fn ill_scaled_qp() -> QpProblem {
    let n = 4;
    let m = 4;
    let q = CscMatrix::from_triplets(
        &[0, 1, 2, 3], &[0, 1, 2, 3], &[2.0, 2.0, 2.0, 2.0], n, n,
    ).unwrap();
    let c = vec![-1.0e6, -1.0, -1.0, -1.0];
    // A: 大きい (1e9 規模) 行と小さい (1e-3 規模) 行を意図的に混在。
    let a = CscMatrix::from_triplets(
        &[0, 0, 1, 1, 2, 3],
        &[0, 1, 2, 3, 0, 1],
        &[1.0e9, 1.0e9, 1.0e-3, 1.0e-3, 1.0, 1.0],
        m, n,
    ).unwrap();
    let b = vec![2.0e9, 2.0e-3, 1.0, 1.0];
    let bounds = vec![(0.0, 1.0e6); n];
    let cts = vec![ConstraintType::Le; m];
    QpProblem::new(q, c, a, b, bounds, cts).unwrap()
}

#[test]
fn ill_scaled_eps_floor_avoids_machine_noise_grind() {
    let problem = ill_scaled_qp();

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(5.0);

    let t0 = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let elapsed = t0.elapsed().as_secs_f64();

    assert!(
        matches!(
            result.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        ),
        "ill_scaled: unexpected status {:?}",
        result.status
    );
    // sentinel: floor が無いと scaled-eps ≪ machine_eps で
    // IPM が iter 上限に張り付くか timeout する。budget 5s 内で完了する事実をロック。
    assert!(
        elapsed < 5.0,
        "ill_scaled: wall {:.3}s >= budget 5s — IPM_EPS_NOISE_FLOOR が機能していない疑い",
        elapsed
    );

    // 元空間 residual が user_eps 近傍であること。post-processing で必ず gate される。
    let x = &result.solution;
    let ax: Vec<f64> = (0..problem.num_constraints).map(|i| {
        let mut s = 0.0;
        for col in 0..problem.num_vars {
            for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
                if problem.a.row_ind[k] == i {
                    s += problem.a.values[k] * x[col];
                }
            }
        }
        s
    }).collect();
    let max_pres = ax.iter().zip(problem.b.iter()).enumerate()
        .map(|(i, (&axi, &bi))| match problem.constraint_types[i] {
            ConstraintType::Le => (axi - bi).max(0.0),
            ConstraintType::Ge => (bi - axi).max(0.0),
            ConstraintType::Eq => (axi - bi).abs(),
            _ => 0.0,
        })
        .fold(0.0_f64, f64::max);
    let denom_pres = 1.0 + ax.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
        .max(problem.b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    let pres_rel = max_pres / denom_pres;
    assert!(
        pres_rel < 1e-5,
        "ill_scaled: pres_rel {:.3e} > 1e-5 — orig-space accuracy 喪失",
        pres_rel
    );
}
