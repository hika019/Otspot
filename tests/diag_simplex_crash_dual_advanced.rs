//! Simplex crash basis → dual_advanced 配線の e2e sentinel。
//!
//! `solve_with` 経由で DualAdvanced 経路 (default) が Ge/Eq LP を解いた際、
//! `use_lp_crash_basis` トグルで:
//!   1. 解の正しさは不変 (objective drift < 1e-6)
//!   2. 退化 (Optimal → Timeout / Infeasible) しない
//!
//! crash の数値効果 (num_artificial 削減、iter 削減) は内部 unit test
//! (`src/simplex/dual_advanced/phase1.rs::tests::crash_*`) で Big-M Phase I を
//! 直接呼び出して検証している。default DualAdvanced 経路は primal-first 経由で
//! crash は primal.rs 既存配線で発動するため、ここでは「dual_advanced 入口に
//! 新規追加した crash 経路が退化を生まない」ことの regression sentinel に集中。
//!
//! no-op proof: `LP_CRASH_DUAL_ADV_DISABLE` 環境変数は env-var 全廃に伴い撤去済。
//! crash の on/off は `use_lp_crash_basis` option 経由のみ。

use otspot::options::{SimplexMethod, SolverOptions};
use otspot::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot::solve_with;
use otspot::sparse::CscMatrix;

/// LCG (Numerical Recipes) deterministic generator (rand dep 不要)。
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(if s == 0 { 0xDEAD_BEEF_CAFE_F00D } else { s })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f01(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Eq/Ge 混在の network-style LP。crash で artif 列を構造列で被覆できる。
fn build_eq_ge_network_lp(n_flow: usize, n_hub: usize, n_eq: usize, seed: u64) -> LpProblem {
    let mut g = Lcg::new(seed);
    let n = n_flow + n_hub;
    let m = n_flow;
    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();
    for i in 0..n_flow {
        a_rows.push(i);
        a_cols.push(i);
        a_vals.push(1.0); // singleton flow
    }
    for h in 0..n_hub {
        for i in 0..n_flow {
            let v = 0.01 + 0.02 * g.f01();
            a_rows.push(i);
            a_cols.push(n_flow + h);
            a_vals.push(v);
        }
    }
    let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();
    let b: Vec<f64> = (0..m).map(|_| 1.0 + 0.5 * g.f01()).collect();
    let c: Vec<f64> = (0..n).map(|_| g.f01()).collect();
    let mut ct = vec![ConstraintType::Eq; n_eq];
    ct.extend(std::iter::repeat_n(ConstraintType::Ge, m - n_eq));
    let bounds = vec![(0.0_f64, 10.0_f64); n];
    LpProblem::new_general(c, a, b, ct, bounds, None).unwrap()
}

/// 教科書 small Eq LP (objective 既知)。
fn build_textbook_eq_lp() -> (LpProblem, f64) {
    // min x1 + x2 + x3
    // s.t. x1 + x2      = 1
    //          x2 + x3 = 1
    //      x_i ≥ 0
    // 解: x1=1, x2=0, x3=1 or x1=0, x2=1, x3=0; obj=1 ※後者
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 1, 2], &[1.0, 1.0, 1.0, 1.0], 2, 3)
        .unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0, 1.0],
        a,
        vec![1.0, 1.0],
        vec![ConstraintType::Eq, ConstraintType::Eq],
        vec![(0.0, f64::INFINITY); 3],
        None,
    )
    .unwrap();
    (lp, 1.0)
}

/// crash on/off で objective drift しない (regression: crash 配線が解を壊さない)。
fn assert_crash_consistent(lp: &LpProblem, name: &str) {
    let mut opts_off = SolverOptions::default();
    opts_off.simplex_method = SimplexMethod::DualAdvanced;
    opts_off.use_lp_crash_basis = false;
    opts_off.timeout_secs = Some(60.0);
    let r_off = solve_with(lp, &opts_off);

    let mut opts_on = SolverOptions::default();
    opts_on.simplex_method = SimplexMethod::DualAdvanced;
    opts_on.use_lp_crash_basis = true;
    opts_on.timeout_secs = Some(60.0);
    let r_on = solve_with(lp, &opts_on);

    eprintln!(
        "[{}] off status={:?} obj={:.6e} iter={} | on status={:?} obj={:.6e} iter={}",
        name,
        r_off.status,
        r_off.objective,
        r_off.iterations,
        r_on.status,
        r_on.objective,
        r_on.iterations,
    );

    // 状態は一致 (退化禁止)
    assert_eq!(
        r_off.status, r_on.status,
        "[{}] crash 退化: off={:?} on={:?}",
        name, r_off.status, r_on.status
    );
    if r_off.status == SolveStatus::Optimal {
        let obj_diff = (r_on.objective - r_off.objective).abs() / (1.0 + r_off.objective.abs());
        assert!(
            obj_diff < 1e-6,
            "[{}] objective drift {:.3e}",
            name,
            obj_diff
        );
    }
}

#[test]
fn crash_dual_advanced_eq_ge_network_pattern_a() {
    let lp = build_eq_ge_network_lp(60, 3, 40, 0x1234_5678_9ABC_DEF0);
    assert_crash_consistent(&lp, "eq_ge_network_a");
}

#[test]
fn crash_dual_advanced_eq_ge_network_pattern_b() {
    let lp = build_eq_ge_network_lp(80, 4, 80, 0xC0FF_EE00_DEAD_BEEF);
    assert_crash_consistent(&lp, "eq_ge_network_b_all_eq");
}

#[test]
fn crash_dual_advanced_eq_ge_network_pattern_c() {
    let lp = build_eq_ge_network_lp(100, 2, 0, 0xA5A5_5A5A_3C3C_C3C3);
    assert_crash_consistent(&lp, "eq_ge_network_c_all_ge");
}

#[test]
fn crash_dual_advanced_textbook_eq() {
    let (lp, obj_expected) = build_textbook_eq_lp();
    assert_crash_consistent(&lp, "textbook_eq");
    let mut opts = SolverOptions::default();
    opts.simplex_method = SimplexMethod::DualAdvanced;
    opts.use_lp_crash_basis = true;
    let r = solve_with(&lp, &opts);
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        (r.objective - obj_expected).abs() < 1e-6,
        "expected obj={}, got {}",
        obj_expected,
        r.objective
    );
}

/// 複数 LCG seed で random Ge/Eq LP を生成し regression を集計検証
/// (CLAUDE.md「複数パターンのデータを用意せよ」)。
#[test]
fn crash_dual_advanced_multi_seed_regression() {
    let seeds: &[u64] = &[
        0xC0FF_EE00_DEAD_BEEF,
        0x1234_5678_9ABC_DEF0,
        0xF00D_BABE_FACE_CAFE,
        0xA5A5_5A5A_3C3C_C3C3,
        0x1111_2222_3333_4444,
    ];
    for &seed in seeds {
        let lp = build_eq_ge_network_lp(40, 2, 20, seed);
        assert_crash_consistent(&lp, &format!("seed_{:016x}", seed));
    }
}
