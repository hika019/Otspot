//! 検証済み Unbounded の ray 証書ペイロードの sentinel (PR #71 issue 2)。
//!
//! ## 目的
//!
//! IPM が検証済み Unbounded を返すとき、`verify_unbounded_ray` は実際に回収
//! 方向 dx と検査対象 iterate を検証しているのに、従来 API は bool
//! (`stats.infeasibility_certified`、現在はこの file の sentinel と同時に
//! `unbounded_ray_certified` / `infeasible_farkas_certified` へ分離済み)
//! しか返さず、`objective = -inf` と空
//! solution だけで「証明済み」という主張を消費者が独立検証できなかった。
//! `SolverResult::ray_certificate` (direction + base_point) により、ソルバを
//! 信用せずに線形代数だけで非有界性を再検証できることをこの file が pin する。
//!
//! ## 台帳 (手計算オラクル、ソルバ非依存)
//!
//! インスタンス U1 (tests/qp_truly_unbounded_eps_consistency.rs と同一構成):
//! min (0, -delta)ᵀx s.t. x1 = 1 (Eq 1 行)、Q = 0、x 自由境界。手計算の ray
//! は dx = e2、基点は任意の (1, t)。A·dx = 0 (厳密)、Q·dx = 0、cᵀdx =
//! -delta < 0 なので x(t) = (1, t + s) 上で目的値は -delta·s → -∞。
//!
//! ## GREEN 基準
//!
//! - U1 (δ=5e-8、presolve off、eps=1e-8): `Unbounded` + `certified` +
//!   `ray_certificate` 存在。ペイロードに対する**テスト内独立再計算**
//!   (ソルバ呼出しなしの純 f64 演算) で A_eq·dx ≤ 検証器と同型の後方誤差境界、
//!   dxᵀQ·dx = 0、cᵀdx < 0、基点の等式残差 ≤ `BASE_EQ_TOL` を確認。
//! - QFORPLAN @1e-8 (STALLED 系・未受理): ペイロード空。
//! - Infeasible 最小インスタンス: Farkas は対象外スコープ (拡張系の内部行
//!   レイアウト依存 — `SolverResult::ray_certificate` の doc 参照) のため、
//!   判定がなんであれペイロード空。
//! - Optimal: ペイロード空 (certified 以外常に空の契約)。
//!
//! revert 確認: finalize_outcome の `ray_certificate` 引き継ぎを消すと
//! U1 のペイロード assert が fail する (revert 実走で確認すること)。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot::qp::ipm_solver::solve_ipm;
use otspot::qp::solve_qp_with;
use otspot::qp::QpProblem;
use otspot::sparse::CscMatrix;
use std::path::Path;

/// U1 の等式行残差が検証時点の fp 後方誤差として許容する上限。
///
/// 検証器は eps-free (componentwise 後方誤差境界) であり、受理時 iterate の
/// 等式残差は要求 eps (1e-8) と独立に machine-epsilon 規模 (‖A‖・‖x‖ 規模の
/// fp 丸め ≈ 1e-16 × 係数規模) に収まる。1e-12 はこの fp 床に対する 4 桁の
/// 安全係数で、要求 eps より 4 桁厳しく「受理時点の基点は実行可能」という
/// ペイロード契約を区別する水準。
const BASE_EQ_TOL: f64 = 1e-12;

/// 退行方向の独立後方誤差境界: 検証器 (`verify_unbounded_ray`) と同じ
/// componentwise 境界 `|A_row·dx| ≤ REL · ‖A_row‖∞ · ‖dx‖₁` の REL。
/// `otspot_core::qp::ipm_core::common::DETECT_REL_EPS` (1e-8) との手計算整合値
/// であり、非退行方向 (例 dx=(1,1): |A·dx| = 1 > 3.1e-3) を退ける検証力を持つ。
/// 受理時 dx は KKT 正則化由来の fp ノイズ (〜1e-14) を成分に含むため、
/// 厳密ゼロではなくこの境界が正味のオラクル期待値になる。
const RECESSION_BACKWARD_EPS: f64 = 1e-8;

/// ray 上の目的減少を確認するパラメータ step。
///
/// cᵀdx < 0 が検証されているので減少は任意 t > 0 で成立するが、1e-12 規模の
/// 減少を fp で判別可能にするため、減少量 ≥ 5e-8 × 1e9 = 50 ≫ 基点まわりの
/// 丸め誤差となる t を選ぶ。
const RAY_DECREASE_STEP: f64 = 1e9;

fn inf() -> f64 {
    f64::INFINITY
}

/// U1: min -delta*x2 s.t. x1 = 1 (Eq)、x1, x2 自由。
/// tests/qp_truly_unbounded_eps_consistency.rs の constructor と同一構成。
fn u1_truly_unbounded(delta: f64) -> QpProblem {
    QpProblem::new(
        CscMatrix::new(2, 2),
        vec![0.0, -delta],
        CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 2).unwrap(),
        vec![1.0],
        vec![(-inf(), inf()), (-inf(), inf())],
        vec![ConstraintType::Eq],
    )
    .unwrap()
}

fn solve_u1(delta: f64, eps: f64) -> SolverResult {
    let problem = u1_truly_unbounded(delta);
    let mut opts = SolverOptions::default();
    opts.ipm.eps = eps;
    opts.presolve = false;
    opts.timeout_secs = Some(4.0);
    solve_ipm(&problem, &opts)
}

/// Infeasible 最小インスタンス: x1 = 1 かつ x1 = 2 (Eq 2 行)、Q = 0、x 自由。
/// Farkas y = (1, -1): Aᵀy = 0、bᵀy = -1 < 0 (手計算)。
fn infeasible_two_eqs() -> QpProblem {
    QpProblem::new(
        CscMatrix::new(2, 2),
        vec![0.0, 0.0],
        CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1.0], 2, 2).unwrap(),
        vec![1.0, 2.0],
        vec![(-inf(), inf()), (-inf(), inf())],
        vec![ConstraintType::Eq, ConstraintType::Eq],
    )
    .unwrap()
}

/// 台帳 U1 の係数 (テスト自身のオラクルデータ。ソルバ内部値には依存しない)。
const U1_DELTA: f64 = 5e-8;
const U1_B_EQ: f64 = 1.0;

/// U1 等式行 A_eq = [1, 0] と dx の内積 (手計算)。
fn u1_eq_row_dot_dx(dx: &[f64]) -> f64 {
    1.0 * dx[0] + 0.0 * dx[1]
}

/// U1 目的係数 c = (0, -delta) と v の内積 (手計算)。
fn u1_c_dot(v: &[f64]) -> f64 {
    0.0 * v[0] + (-U1_DELTA) * v[1]
}

/// 検証済み Unbounded は再計算可能な ray 証書を運ぶ (sentinel 本体)。
///
/// finalize_outcome のペイロード引き継ぎを revert すると
/// `ray_certificate` が None になり本 test が fail する。
#[test]
fn u1_certified_unbounded_carries_recomputable_ray_payload() {
    let r = solve_u1(U1_DELTA, 1e-8);
    assert_eq!(
        r.status,
        SolveStatus::Unbounded,
        "delta=5e-8 eps=1e-8 must be certified Unbounded (precondition of \
         the payload contract), got {:?} obj={:e}",
        r.status,
        r.objective
    );
    assert!(
        r.stats.unbounded_ray_certified,
        "precondition: status Unbounded without cert must not carry payload"
    );
    // 公開 API 形状 (変更前からの契約): 目的値 -inf・solution 空。
    assert_eq!(r.objective, f64::NEG_INFINITY);
    assert!(r.solution.is_empty());

    let cert = r
        .ray_certificate
        .expect("certified Unbounded must expose the verified ray payload");
    assert_eq!(cert.direction.len(), 2);
    assert_eq!(cert.base_point.len(), 2);
    assert!(
        cert.direction.iter().all(|v| v.is_finite()),
        "direction must be finite, got {:?}",
        cert.direction
    );

    // ── 独立再計算 (ソルバ呼出しなし・手計算係数の純 f64 演算) ──────────
    // 1. 等式退行: 検証器と同じ後方誤差境界 |A_eq·dx| ≤
    //    RECESSION_BACKWARD_EPS · ‖A_eq‖∞ · ‖dx‖₁。受理時 Newton 方向は KKT
    //    正則化の fp ノイズ (実測 〜1e-14) を含み厳密ゼロにはならないため、
    //    オラクル期待値はこの境界 (非退行方向は 9 桁以上超過で fail する)。
    let dx_l1: f64 = cert.direction.iter().fold(0.0, |a, v| a + v.abs());
    let a_dx = u1_eq_row_dot_dx(&cert.direction);
    assert!(
        a_dx.abs() <= RECESSION_BACKWARD_EPS * 1.0 * dx_l1,
        "A_eq·dx {a_dx:e} exceeds backward-error bound \
         {RECESSION_BACKWARD_EPS:e} * 1 * {dx_l1:e} (direction {:?})",
        cert.direction
    );
    // 2. 平坦性: Q = 0 より dxᵀQ·dx = 0 (厳密ゼロの退化ケースとして確認)。
    let qdx = [0.0 * cert.direction[0], 0.0 * cert.direction[1]];
    assert_eq!(qdx[0] * cert.direction[0] + qdx[1] * cert.direction[1], 0.0);
    // 3. 目的減少: cᵀdx < 0 厳密。
    assert!(
        u1_c_dot(&cert.direction) < 0.0,
        "cᵀdx must be negative, c=(0,-{}) dx={:?}",
        U1_DELTA,
        cert.direction
    );
    // 4. 基点の実行可能性: 等式 x1 = 1 が BASE_EQ_TOL 以内。
    let base_eq_residual = (cert.base_point[0] - U1_B_EQ).abs();
    assert!(
        base_eq_residual <= BASE_EQ_TOL,
        "base point equality residual {base_eq_residual:e} exceeds fp \
         backward-error bound {BASE_EQ_TOL:e} (base_point {:?})",
        cert.base_point
    );
    // 5. ray 上の目的減少: φ は線形 (Q=0) なので cᵀ(base + t·dx) < cᵀbase。
    let stepped: Vec<f64> = cert
        .base_point
        .iter()
        .zip(cert.direction.iter())
        .map(|(b, d)| b + RAY_DECREASE_STEP * d)
        .collect();
    assert!(
        u1_c_dot(&stepped) < u1_c_dot(&cert.base_point),
        "objective must decrease along the certified ray (base {:?} dx {:?})",
        cert.base_point,
        cert.direction
    );
    eprintln!(
        "[U1 payload] direction={:?} base={:?} cdx={:e} base_eq_res={:e}",
        cert.direction,
        cert.base_point,
        u1_c_dot(&cert.direction),
        base_eq_residual
    );
}

/// QFORPLAN @1e-8: 検証が受理していないため ray ペイロードは空
/// (「certified でなければ常に空」の逆側 sentinel)。
#[test]
fn qforplan_unverified_verdict_carries_no_payload() {
    let path = Path::new("data/maros_meszaros/QFORPLAN.QPS");
    assert!(
        path.exists(),
        "data missing: {} — Maros-Meszaros suite 必須 (scripts/download_all_bench_data.sh)",
        path.display()
    );
    let problem = otspot::io::qps::parse_qps(path).expect("parse_qps QFORPLAN");
    let mut opts = SolverOptions::default();
    opts.ipm.eps = 1e-8;
    opts.tolerance = Some(otspot::options::Tolerance::Custom(1e-8));
    let r = solve_qp_with(&problem, &opts);
    // 現在の契約 (qp_qforplan_unbounded_verification.rs と同一前提): 偽
    // Unbounded / Infeasible を返さない (STALLED 系に落ちる)。
    assert!(
        !matches!(r.status, SolveStatus::Unbounded | SolveStatus::Infeasible),
        "QFORPLAN@1e-8 must not report a structural verdict, got {:?}",
        r.status
    );
    assert!(
        !r.stats.unbounded_ray_certified && !r.stats.infeasible_farkas_certified,
        "no ray/Farkas verification was accepted at eps=1e-8, so the cert \
         flag must stay false"
    );
    assert!(
        r.ray_certificate.is_none(),
        "unverified/stalled result must carry an empty payload, got {:?}",
        r.ray_certificate
    );
}

/// Infeasible 最小インスタンス: Farkas 証書はスコープ外 (ray のみ公開) の
/// ため、判定が Infeasible (certified) でもそれ以外でも ray ペイロードは空。
#[test]
fn certified_infeasible_carries_no_ray_payload() {
    let problem = infeasible_two_eqs();
    let mut opts = SolverOptions::default();
    opts.ipm.eps = 1e-8;
    opts.presolve = false;
    opts.timeout_secs = Some(4.0);
    let r = solve_ipm(&problem, &opts);
    assert!(
        r.ray_certificate.is_none(),
        "Farkas certificates are out of scope (extended-system internal row \
         layout — see SolverResult::ray_certificate docs); payload must be \
         empty for Infeasible-family results, got {:?}",
        r.ray_certificate
    );
    if r.status == SolveStatus::Infeasible {
        // 未証明 Infeasible は確定応答にならない既存契約との整合。
        assert!(
            r.stats.infeasible_farkas_certified,
            "an Infeasible final verdict must be Farkas-verified (status \
             {:?} obj={:e})",
            r.status, r.objective
        );
    }
    eprintln!(
        "[infeasible two-eq] status={:?} certified={} payload={:?}",
        r.status,
        r.stats.infeasible_farkas_certified,
        r.ray_certificate.is_some()
    );
}

/// Optimal (真に有限最適解) は ray ペイロードを運ばない。
#[test]
fn optimal_result_carries_no_payload() {
    // min (x-1)^2 相当: Q = 2 (1x1)、c = -2 → x* = 1、obj = -1 (+offset 0)。
    let problem = QpProblem::new(
        CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap(),
        vec![-2.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![(-inf(), inf())],
        vec![],
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(4.0);
    let r = solve_ipm(&problem, &opts);
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "precondition: hand-computed optimum x*=1 obj=-1 must converge, got {:?}",
        r.status
    );
    assert!(
        r.ray_certificate.is_none(),
        "payload is empty for every non-certified-Unbounded result, got {:?}",
        r.ray_certificate
    );
}
