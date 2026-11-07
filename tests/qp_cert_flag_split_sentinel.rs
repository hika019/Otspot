//! 証書フラグ分離の sentinel (PR #71 review 指摘 issuecomment-6682303623)。
//!
//! 旧 `stats.infeasibility_certified` は「検証済み Unbounded (ray 証書)」と
//! 「検証済み Infeasible (Farkas 証書)」の共有名で、`Unbounded + certified=true`
//! が「実行不可能が証明された」と誤読された。`unbounded_ray_certified` /
//! `infeasible_farkas_certified` へ分離した本実装では、
//! 各側が **自分の側のフラグのみ** を立て、反対側を絶対に立たせないことを
//! 下記のテストが pin する。
//! 共有フラグへ戻す (どちらか一方の検証結果を両方へ配る) revert では
//! `certified_unbounded_sets_ray_flag_only` と
//! `certified_infeasible_sets_farkas_flag_only` の一方が必ず FAIL する。
//!
//! ## オラクル (手計算、ソルバ非依存)
//!
//! - U1: `min -δ·x2 s.t. x1 = 1` (x 自由)。実行可能集合
//!   F = {(1, t) : t ∈ R} は**非空**なので Infeasible 主張はオラクル偽。
//!   dx = e2 は A·dx = 0、c'dx = -δ < 0 の recession ray なので非有界が真。
//!   (インスタンスは tests/qp_truly_unbounded_eps_consistency.rs と同一。)
//! - I1: `min -x s.t. 0·x ≤ -1, x ≥ 0`。制約行は 0 ≤ -1 で成立不能、つまり
//!   実行可能集合が**空**。よって Infeasible が真であり、実行可能 ray は
//!   存在しないので非有界の主張は成立し得ない。

use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus, SolverResult};
use otspot::qp::ipm_solver::solve_ipm;
use otspot::qp::QpProblem;
use otspot::sparse::CscMatrix;

fn inf() -> f64 {
    f64::INFINITY
}

/// U1: min -δ·x2 s.t. x1 = 1 (Eq)、x1, x2 自由。δ = eps の決定性構成は
/// `truly_unbounded_delta_eq_eps_ray_certified_deterministic_unbounded` (同ディレクトリ)
/// が検証済み Unbounded を pin 済みのパラメータを借用する。
fn solve_u1() -> SolverResult {
    let problem = QpProblem::new(
        CscMatrix::new(2, 2),
        vec![0.0, -1e-7],
        CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 2).unwrap(),
        vec![1.0],
        vec![(-inf(), inf()), (-inf(), inf())],
        vec![ConstraintType::Eq],
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.ipm.eps = 1e-7;
    opts.presolve = false;
    opts.timeout_secs = Some(4.0);
    solve_ipm(&problem, &opts)
}

/// I1: min -x s.t. 0·x ≤ -1, x ≥ 0。実行可能集合が空 (0 ≤ -1 が成立不能)。
/// presolve を落とし IPM の受理部 (Farkas 検証) を必ず通す構成
/// (`ipm_reports_infeasible_for_zero_row_with_negative_rhs_and_presolve_off`
/// と同一インスタンス)。
fn solve_i1() -> SolverResult {
    let problem = QpProblem::new_all_le(
        CscMatrix::new(1, 1),
        vec![-1.0],
        CscMatrix::new(1, 1),
        vec![-1.0],
        vec![(0.0, inf())],
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.presolve = false;
    opts.timeout_secs = Some(4.0);
    solve_ipm(&problem, &opts)
}

/// 検証済み Unbounded は ray フラグのみ立て、Farkas フラグは絶対に立たせない。
/// 旧共有名実装 (または証明結果を両フラグへ配る revert) では Farkas 側 assert
/// が FAIL する。
#[test]
fn certified_unbounded_sets_ray_flag_only() {
    for rep in 0..3 {
        let r = solve_u1();
        assert_eq!(
            r.status,
            SolveStatus::Unbounded,
            "U1 rep {rep}: fixture is hand-proved truly unbounded (ray e2), got {:?}",
            r.status
        );
        assert!(
            r.stats.unbounded_ray_certified,
            "U1 rep {rep}: certified Unbounded must carry the ray certificate"
        );
        assert!(
            !r.stats.infeasible_farkas_certified,
            "U1 rep {rep}: feasible set is nonempty (x=(1,t) for all t); a Farkas \
             (infeasibility) certificate must never accompany Unbounded"
        );
    }
}

/// 検証済み Infeasible は Farkas フラグのみ立て、ray フラグは絶対に立たせない。
/// 反対側フラグが立つ revert (共有名時代の実装) では ray 側 assert が FAIL する。
#[test]
fn certified_infeasible_sets_farkas_flag_only() {
    let r = solve_i1();
    assert_eq!(
        r.status,
        SolveStatus::Infeasible,
        "I1: empty feasible set (0 <= -1) must report Infeasible, got {:?}",
        r.status
    );
    assert!(
        r.stats.infeasible_farkas_certified,
        "I1: finalize_outcome only mints a definitive Infeasible through the Farkas \
         verification, so the Farkas flag must be set (got both flags false = \
         unproven verdict leaked through the gate)"
    );
    assert!(
        !r.stats.unbounded_ray_certified,
        "I1: no feasible point exists, so no recession ray exists; a ray certificate \
         must never accompany Infeasible"
    );
}

/// Optimal  solve はどちらの証書も主張しない (両フラグ false)。証書フラグが
/// 無関係な経路 (Default 再構築の取りこぼし・流用) で立つことのないことの
/// 下位保証。
#[test]
fn optimal_solve_carries_no_certificate_flags() {
    // min x^2 - 2x s.t. x >= 1 (Ge): 手計算解 x*=1、obj = -1 (実行可能域は
    // 非空、目的関数は下に有界で recession ray も Farkas 行も存在しない)。
    let problem = QpProblem::new(
        CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap(),
        vec![-2.0],
        CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap(),
        vec![1.0],
        vec![(-inf(), inf())],
        vec![ConstraintType::Ge],
    )
    .unwrap();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(4.0);
    let r = solve_ipm(&problem, &opts);
    assert_eq!(
        r.status,
        SolveStatus::Optimal,
        "optimal fixture: expected Optimal, got {:?}",
        r.status
    );
    assert!(
        !r.stats.unbounded_ray_certified && !r.stats.infeasible_farkas_certified,
        "Optimal must carry neither certificate, got ray={} farkas={}",
        r.stats.unbounded_ray_certified,
        r.stats.infeasible_farkas_certified
    );
}
