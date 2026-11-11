//! W2-1b: PG Cauchy loop の Aᵀy 列分担並列化の出力同一性契約。
//!
//! ゲート (threads>=2 かつ nnz >= ATY_PAR_MIN_NNZ) を発火させる行列で
//! `refine_dual_projected_gradient` を threads=1 / threads=2 の両方へ走らせ、
//! 得られる dual_solution が **ビット単位で一致**することを固定する。
//! 列分担は列内積の加算順序を変えない (並列 plan は逐次カーネルと同一の
//! 加算列を列ごとに実行する) のが設計の核心なので、trajectory が変わる
//! 実装 (atomic scatter・列内 2 分木還元など) に退化したらこのテストが
//! fail する。threads=1 側の経路そのものは p1_atv_golden の golden 定数が
//! 既に固定しており、ここでは「threads=2 がその逐次軌跡から 1 ビットも
//! 動かない」側だけを主張する。

use super::super::*;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot_num::sparse::{AtyParallelPlan, CscMatrix, ATY_PAR_MIN_NNZ};

/// テスト専用 LCG (決定的。再現性を rand 非依存で担保)。
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

/// nnz がゲートを明確に超える (列あたり ~120 非ゼロ × 200 列 ≒ 24k) 疎行列
/// と、PG が実際に反復して dual を動かす開始点の fixture。
fn fixture_gated_columns() -> (QpProblem, SolverResult) {
    let n = 200usize;
    let m = 400usize;
    let per_col = 120usize;
    let mut rng = Lcg(0x51A2_1B20_2611);
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_ind = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0usize);
    for _ in 0..n {
        let mut rows: Vec<usize> = (0..per_col)
            .map(|_| (rng.next_u64() % m as u64) as usize)
            .collect();
        rows.sort_unstable();
        rows.dedup();
        for &r in &rows {
            row_ind.push(r);
            values.push(rng.next_f64() * (1.0 + r as f64));
        }
        col_ptr.push(row_ind.len());
    }
    let a = CscMatrix::from_raw_parts(m, n, col_ptr, row_ind, values);
    assert!(
        a.values().len() >= ATY_PAR_MIN_NNZ,
        "fixture must clear the nnz gate (nnz={})",
        a.values().len()
    );
    assert!(
        AtyParallelPlan::new(&a, 2).is_some(),
        "fixture must actually fire the parallel plan at threads=2; \
         if the gate moved above this nnz, raise per_col"
    );
    // Q は対角 (Maros 群の実測格納形状と同じ。PG 冒頭の DD Qx を素通し)。
    let q = CscMatrix::identity(n);
    let mut b = Vec::with_capacity(m);
    let mut types = Vec::with_capacity(m);
    for i in 0..m {
        b.push(rng.next_f64() * 10.0);
        types.push(match i % 3 {
            0 => ConstraintType::Eq,
            1 => ConstraintType::Le,
            _ => ConstraintType::Ge,
        });
    }
    let problem = QpProblem::new(
        q,
        (0..n).map(|_| rng.next_f64()).collect(),
        a,
        b,
        vec![(0.0_f64, f64::INFINITY); n],
        types,
    )
    .unwrap();
    let result = SolverResult {
        status: SolveStatus::Optimal,
        solution: (0..n).map(|_| rng.next_f64().abs() + 0.1).collect(),
        dual_solution: (0..m).map(|_| rng.next_f64() * 3.0).collect(),
        bound_duals: (0..n).map(|_| rng.next_f64()).collect(),
        ..SolverResult::default()
    };
    (problem, result)
}

/// threads を変えた 2 回の PG 適用がビット一致であることを検証する。
/// threads=1 側で dual が実際に動くことも同時に要求し、末尾 KKT ガードの
/// 棄却で両者が「無変更のまま一致」する vacuous pass を防ぐ。
#[test]
fn w2_pg_threads2_duals_bitwise_equal_to_threads1() {
    let (problem, base) = fixture_gated_columns();
    let mut r1 = base.clone();
    refine_dual_projected_gradient(&problem, &mut r1, &[], None, 1);
    assert!(
        r1.dual_solution
            .iter()
            .zip(base.dual_solution.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "threads=1 PG must move the duals, otherwise the comparison below is vacuous"
    );
    for threads in [2usize, 4] {
        let mut rp = base.clone();
        refine_dual_projected_gradient(&problem, &mut rp, &[], None, threads);
        assert_eq!(rp.dual_solution.len(), r1.dual_solution.len());
        for (j, (p, s)) in rp
            .dual_solution
            .iter()
            .zip(r1.dual_solution.iter())
            .enumerate()
        {
            assert_eq!(
                p.to_bits(),
                s.to_bits(),
                "threads={threads} dual drift at row {j}: {p:?} vs {s:?}"
            );
        }
    }
    // rayon のスケジューリング差が混入しないことも固定 (同じ入力で 2 回)。
    let mut rp1 = base.clone();
    let mut rp2 = base.clone();
    refine_dual_projected_gradient(&problem, &mut rp1, &[], None, 2);
    refine_dual_projected_gradient(&problem, &mut rp2, &[], None, 2);
    for (j, (p, q)) in rp1
        .dual_solution
        .iter()
        .zip(rp2.dual_solution.iter())
        .enumerate()
    {
        assert_eq!(p.to_bits(), q.to_bits(), "repeat drift at row {j}");
    }
}
