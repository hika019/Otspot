//! IP-PMM 単体テスト。
#![allow(clippy::print_stdout, clippy::print_stderr)]

use super::iter::{
    solve_ippmm_inner, CANCEL_AFTER_INFEAS_COMMIT, CANCEL_SIGNAL, INFEAS_COMMIT_COUNT,
};
use super::state::{warm_bound_margin, WARM_BOUND_REL_MARGIN};
use super::warm_start::apply_qp_warm_start;
use crate::options::{QpWarmStart, SolverOptions};
use crate::problem::{ConstraintType, SolveStatus};
use crate::qp::ipm_core::kkt::build_extended_constraints;
use crate::qp::problem::QpProblem;
use otspot_num::sparse::CscMatrix;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const EPS: f64 = 1e-4; // IP-PMM は標準 IPM より tolerance がゆるめでも通ることを確認

fn close(a: f64, b: f64, name: &str) {
    assert!(
        (a - b).abs() < EPS,
        "{}: expected {:.8}, got {:.8} (diff={:.2e})",
        name,
        b,
        a,
        (a - b).abs()
    );
}

fn default_opts() -> SolverOptions {
    SolverOptions {
        timeout_secs: Some(10.0),
        use_ruiz_scaling: false,
        ..Default::default()
    }
}

/// IPPMM-T1: 2変数基本 QP
/// min x^2 + y^2  (Q=2I, c=0)  s.t. x + y >= 1
/// 期待: x*=y*=0.5, obj=0.5
#[test]
fn test_ippmm_basic_2d() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T1: status");
    close(result.solution[0], 0.5, "IPPMM-T1: x[0]");
    close(result.solution[1], 0.5, "IPPMM-T1: x[1]");
    close(result.objective, 0.5, "IPPMM-T1: objective");
}

/// IPPMM-T2: 制約なし QP
/// min (x-3)^2 + (y-4)^2  → Q=2I, c=[-6,-8], 制約なし
/// 期待: x*=3, y*=4, obj=-25
#[test]
fn test_ippmm_unconstrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-6.0, -8.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T2: status");
    close(result.solution[0], 3.0, "IPPMM-T2: x[0]");
    close(result.solution[1], 4.0, "IPPMM-T2: x[1]");
    close(result.objective, -25.0, "IPPMM-T2: objective");
}

/// IPPMM-T3: 等式制約付き QP
/// min x^2 + y^2  s.t. x + y = 1  (2不等式で表現)
/// 期待: x*=y*=0.5, obj=0.5
#[test]
fn test_ippmm_equality_constraint() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[1.0, 1.0, -1.0, -1.0], 2, 2)
        .unwrap();
    let b = vec![1.0, -1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T3: status");
    close(result.solution[0], 0.5, "IPPMM-T3: x[0]");
    close(result.solution[1], 0.5, "IPPMM-T3: x[1]");
    close(result.objective, 0.5, "IPPMM-T3: objective");
}

/// IPPMM-T4: Box 制約付き QP
/// min (x-2)^2 + (y-2)^2  s.t. 0 <= x <= 1, 0 <= y <= 1
/// 期待: x*=y*=1, obj=-6
#[test]
fn test_ippmm_box_constrained() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-4.0, -4.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "IPPMM-T4: status");
    close(result.solution[0], 1.0, "IPPMM-T4: x[0]");
    close(result.solution[1], 1.0, "IPPMM-T4: x[1]");
    close(result.objective, -6.0, "IPPMM-T4: objective");
}

/// 境界超過許容幅 (revert sentinel 用)。eps=1e-6 運用の実行可能判定と同オーダー。
/// 境界超過がこれを越えたら fraction-to-boundary が realized 変位と不一致。
const SENTINEL_MAX_BOUND_OVERSHOOT: f64 = 1e-6;

/// T4 の反復数上限 (revert sentinel 用)。BASE extended 系および box 化実装は
/// 数十反復で収束。境界超過→床マスク→漂流パスは 3×10⁵ 反復まで張り付くため
/// この上限で検出できる (両側から根拠ある余裕)。
const SENTINEL_BOX_MAX_ITERS: usize = 500;

/// IPPMM-T4b (revert sentinel): 上境界収束で x が境界を絶対に超過しないこと。
///
/// box 化で fraction-to-boundary・μ 予測が realized 従属率 ds_t = −a_t·dx と
/// 一致しない実装 (extended Newton ds を α に使う) へ revert すると、
/// δ_d·dy_t < 0 の反復で s 減少を過小評価し x が ub を 0.17 超過、床マスクで
/// 検出不能なまま漂流する (実測: x=1.1718, 3.07×10⁵ iters, Timeout)。
#[test]
fn sentinel_box_no_boundary_overshoot() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![-4.0, -4.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "IPPMM-T4b: revert 時に漂流→Timeout"
    );
    assert!(
        result.iterations < SENTINEL_BOX_MAX_ITERS,
        "IPPMM-T4b: 反復数 {} が漂流を示唆",
        result.iterations
    );
    for (i, &xv) in result.solution.iter().enumerate() {
        assert!(
            (-SENTINEL_MAX_BOUND_OVERSHOOT..=1.0 + SENTINEL_MAX_BOUND_OVERSHOOT).contains(&xv),
            "IPPMM-T4b: x[{i}] = {xv} が境界 [0,1] を超過",
        );
    }
}

/// IPPMM-T4c (revert sentinel): 停留点初期化で境界行 dy が凍結しないこと。
///
/// min −x²+y² s.t. −1≤x,y≤1 の初期点 x=0 は正確な停留点 (鞍点)。箱化 solve が
/// 解の値から失敗を推論するヒューリスティク (dx 全ゼロ⇒失敗) へ revert すると
/// dy が凍結して 51 反復で Stalled になり、外層が鞍点を Optimal と誤認証する。
/// 正実装は収束し、不定 QP として LocallyOptimal に降格する (BASE と同型)。
#[test]
fn sentinel_stationary_point_not_frozen() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[-2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0; 2];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(-1.0_f64, 1.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let result = solve_ippmm_inner(&problem, &default_opts(), default_opts().ipm_eps());
    assert_eq!(
        result.status,
        SolveStatus::LocallyOptimal,
        "IPPMM-T4c: revert 時に Stalled (dy 凍結)"
    );
}

/// IPPMM-T5: タイムアウト動作確認
#[test]
fn test_ippmm_timeout() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(0.0001),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(
        result.status == SolveStatus::Timeout || result.status == SolveStatus::Optimal,
        "IPPMM-T5: expected Timeout or Optimal, got {:?}",
        result.status
    );
}

/// IPPMM-T-conv1: 等式制約収束確認
/// min x²+y² s.t. x+y=1 (ConstraintType::Eq)
/// QpProblem::new() を使用
/// 期待: 5秒以内にOptimal、x*=y*=0.5
#[test]
fn test_ippmm_eq_convergence_check() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Eq]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(
        start.elapsed().as_secs_f64() < 6.0,
        "Test exceeded 6 second wall-clock limit"
    );
    assert_eq!(result.status, SolveStatus::Optimal, "conv-eq: status");
    close(result.solution[0], 0.5, "conv-eq: x[0]");
    close(result.solution[1], 0.5, "conv-eq: x[1]");
}

/// IPPMM-T-conv2: 不等式制約収束確認
/// min x²+y² s.t. x+y>=1 (Le形式: -x-y <= -1、ConstraintType::Le)
#[test]
fn test_ippmm_le_convergence_check() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(
        start.elapsed().as_secs_f64() < 6.0,
        "Test exceeded 6 second wall-clock limit"
    );
    assert_eq!(result.status, SolveStatus::Optimal, "conv-le: status");
    close(result.solution[0], 0.5, "conv-le: x[0]");
    close(result.solution[1], 0.5, "conv-le: x[1]");
}

/// IPPMM-T-Ge1: Ge制約防御テスト
/// min x²+y² s.t. x+y≥1 (ConstraintType::Ge)
#[test]
fn test_ippmm_ge_defensive() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Ge]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert!(
        start.elapsed().as_secs_f64() < 6.0,
        "Test exceeded 6 second wall-clock limit"
    );
    assert_eq!(result.status, SolveStatus::Optimal, "ge-defensive: status");
    close(result.solution[0], 0.5, "ge-defensive: x[0]");
    close(result.solution[1], 0.5, "ge-defensive: x[1]");
}

/// IPPMM-T-F1: 空制約退化ケース
/// min 0.5*(x²+y²) - x - y (Q=I, c=[-1,-1], 制約なし)
#[test]
fn test_ippmm_empty_constraints() {
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let c = vec![-1.0, -1.0];
    let a = CscMatrix::new(0, 2);
    let b: Vec<f64> = vec![];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let problem = QpProblem::new(q, c, a, b, bounds, vec![]).unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "empty-constraints: status"
    );
    close(result.solution[0], 1.0, "empty-constraints: x[0]");
    close(result.solution[1], 1.0, "empty-constraints: x[1]");
}

/// IPPMM-T-F2: 複数等式制約退化ケース
/// min x²+y²+z² s.t. x+y=1 (Eq), y+z=1 (Eq)
#[test]
fn test_ippmm_multiple_equality_constraints() {
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0], 3, 3).unwrap();
    let c = vec![0.0, 0.0, 0.0];
    let a = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 1, 2], &[1.0, 1.0, 1.0, 1.0], 2, 3)
        .unwrap();
    let b = vec![1.0, 1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 3];
    let problem = QpProblem::new(
        q,
        c,
        a,
        b,
        bounds,
        vec![ConstraintType::Eq, ConstraintType::Eq],
    )
    .unwrap();

    let opts = SolverOptions {
        timeout_secs: Some(5.0),
        use_ruiz_scaling: false,
        ..Default::default()
    };
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert_eq!(result.status, SolveStatus::Optimal, "multi-eq: status");
    close(result.solution[0], 1.0 / 3.0, "multi-eq: x[0]");
    close(result.solution[1], 2.0 / 3.0, "multi-eq: x[1]");
    close(result.solution[2], 1.0 / 3.0, "multi-eq: x[2]");
}

/// warm_bound_margin が bound scale に追従することの直接 sentinel。
/// 旧 `WARM_BOUND_ABS_MARGIN = 1.0` 固定では小 bound 過大 / 大 bound 過小の両極。
/// no-op 書換 (REL=1e-6 は据置きで margin=1.0 fixed) では各 assertion が FAIL することを表で確認。
#[test]
fn test_warm_bound_margin_scale_tracking() {
    // (|bound|, expected = REL × max(|b|,1)) — multi-pattern data
    // 旧 fixed=1.0 の場合 expected との不一致を検出。
    let cases = [
        (0.0_f64, WARM_BOUND_REL_MARGIN),
        (0.25_f64, WARM_BOUND_REL_MARGIN),
        (-0.5_f64, WARM_BOUND_REL_MARGIN),
        (1.0_f64, WARM_BOUND_REL_MARGIN),
        (-1.0_f64, WARM_BOUND_REL_MARGIN),
        (10.0_f64, WARM_BOUND_REL_MARGIN * 10.0),
        (1e3_f64, WARM_BOUND_REL_MARGIN * 1e3),
        (-1e6_f64, WARM_BOUND_REL_MARGIN * 1e6),
        (1e8_f64, WARM_BOUND_REL_MARGIN * 1e8),
        (1e11_f64, WARM_BOUND_REL_MARGIN * 1e11),
    ];
    for (b, expected) in cases {
        let got = warm_bound_margin(b);
        let denom = expected.abs().max(WARM_BOUND_REL_MARGIN);
        assert!(
            ((got - expected) / denom).abs() < 1e-12,
            "warm_bound_margin({}) = {:.6e}, expected {:.6e} (旧 ABS=1.0 fixed retention 疑い)",
            b,
            got,
            expected,
        );
    }
    // |b|=1 で margin が old=1.0 から大きく縮小していることを明示。
    assert!(
        warm_bound_margin(1.0) < 1e-3,
        "|b|=1 で margin={:.3e} は old absolute 1.0 を残している",
        warm_bound_margin(1.0),
    );
    // |b|=1e8 で margin が old=1.0 から拡大していることを明示。
    assert!(
        warm_bound_margin(1e8) > 10.0,
        "|b|=1e8 で margin={:.3e} は old absolute 1.0 を残している",
        warm_bound_margin(1e8),
    );
}

/// apply_qp_warm_start: bound scale 別の strict-interior 補正 sentinel。
/// 旧 ABS=1.0 固定では small-bound で warm 値が過剰に押し込まれ、large-bound で
/// 不十分に押し込まれた。新 helper でいずれも適切な相対位置に収まることを確認。
///
/// no-op 書換 (margin を旧 1.0 fixed に戻す) では:
/// - small-bound case: x[0] が 0.5 → 1.0 に押し上げられ assertion FAIL
/// - large-bound case: x[0] が 1e8+50 → そのまま (1.0 < 50) で push 不足 assertion FAIL
#[test]
fn test_warm_start_half_finite_bound_scale_tracking() {
    // (lb, xj_warm, expected_x_after_correction, name)
    // multi-pattern: 各 |lb| scale × xj 位置パターンを cover。
    let cases: [(f64, f64, f64, &str); 5] = [
        // |lb|=1: 旧 margin=1.0 では 0.5 → 1.0 に過剰押込み。新 margin≈1e-6 で warm 尊重。
        (1.0, 1.5, 1.5, "|lb|=1 well-interior"),
        // |lb|=0: 旧 margin=1.0 では 0.1 → 1.0。新 margin=REL で 0.1 尊重。
        (0.0, 0.1, 0.1, "|lb|=0 small warm"),
        // |lb|=10: 旧 margin=1.0 では 10.1 → 11.0。新 margin=REL*10=1e-5 で 10.1 尊重。
        (10.0, 10.1, 10.1, "|lb|=10 close warm"),
        // |lb|=1e8: 旧 margin=1.0 では 1e8+0.5 → 1e8+1.0、足りない。新 margin=100 で push 必要。
        (1e8, 1e8 + 50.0, 1e8 + 100.0, "|lb|=1e8 needs scaled push"),
        // |lb|=1e6: 旧 margin=1.0, 新 margin=1.0 (たまたま一致) なので push 量同じ。境界 case。
        // ただし xj=0.999e6 (lb 未満) は旧 margin=1.0 → 1e6+1、新も同。
        (
            1e6,
            0.999e6,
            1e6 + 1.0,
            "|lb|=1e6 below bound floors at lb+margin",
        ),
    ];
    for (lb, xj, expected, name) in cases {
        let problem = build_single_var_lb_problem(lb);
        let x_out = apply_warm_and_extract_x(&problem, xj);
        assert!(
            (x_out - expected).abs() < 1e-9 * expected.abs().max(1.0),
            "{}: x={:.6e}, expected {:.6e} (lb={:.1e}, xj_warm={:.6e})",
            name,
            x_out,
            expected,
            lb,
            xj,
        );
    }
}

/// apply_qp_warm_start: 両端 finite box の range-scale margin sentinel。
/// 旧 `.min(WARM_BOUND_ABS_MARGIN=1.0)` cap では巨大 range で margin=1.0 に貼り付き
/// strict-interior が相対 1e-10 以下に縮退。新コードでは range×REL で常に相対 1e-6 を確保。
#[test]
fn test_warm_start_both_finite_bound_scale_tracking() {
    // (lb, ub, xj_warm, name, assertion-spec)
    // 巨大 range: lb=-1e10, ub=1e10, range=2e10。旧 margin=1.0、新 margin=2e4。
    // xj=ub-1.0 (旧では unchanged) は新では ub-2e4 に押し戻されるべき。
    {
        let problem = build_single_var_box_problem(-1e10, 1e10);
        let x_out = apply_warm_and_extract_x(&problem, 1e10 - 1.0);
        let new_margin_lower_bound = 1e3; // new margin=2e4、 旧 1.0 とは 4 桁差。
        assert!(
            x_out <= 1e10 - new_margin_lower_bound,
            "large-range box: x={:.6e} は ub-1.0 に貼り付き (旧 ABS=1.0 cap retention 疑い)",
            x_out,
        );
    }
    // 中間 range: [0, 1e6], range=1e6。旧 margin=min(1.0, 1.0)=1.0、新 margin=1.0。境界一致。
    {
        let problem = build_single_var_box_problem(0.0, 1e6);
        // 内部 well-interior: warm が範囲内なら維持。
        let x_out = apply_warm_and_extract_x(&problem, 5e5);
        assert!(
            (x_out - 5e5).abs() < 1e-6,
            "mid-range box well-interior 維持"
        );
    }
    // 小 range: [0, 1], range=1。旧 margin=min(1e-6, 1.0)=1e-6、新 margin=1e-6。一致。
    {
        let problem = build_single_var_box_problem(0.0, 1.0);
        let x_out = apply_warm_and_extract_x(&problem, 0.5);
        assert!((x_out - 0.5).abs() < 1e-9, "small box well-interior 維持");
    }
    // Collapsing box: range が 2×margin 未満 → midpoint 退避。
    // [0, 1e-10]: range=1e-10, margin=1e-16、range > 2×margin なので clamp 経路に入る。
    // [-1e-12, 1e-12]: range=2e-12, margin=2e-18、これも clamp。
    // 退避経路に入る極端: range = 0 (lb=ub) のとき range > 2*0=0 が false → midpoint。
    {
        let problem = build_single_var_box_problem(3.0, 3.0);
        let x_out = apply_warm_and_extract_x(&problem, 100.0);
        assert!(
            (x_out - 3.0).abs() < 1e-12,
            "collapsed box (lb=ub): midpoint=lb=ub"
        );
    }
}

fn build_single_var_lb_problem(lb: f64) -> QpProblem {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b: Vec<f64> = vec![];
    let bounds = vec![(lb, f64::INFINITY)];
    QpProblem::new(q, c, a, b, bounds, vec![]).unwrap()
}

fn build_single_var_box_problem(lb: f64, ub: f64) -> QpProblem {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::new(0, 1);
    let b: Vec<f64> = vec![];
    let bounds = vec![(lb, ub)];
    QpProblem::new(q, c, a, b, bounds, vec![]).unwrap()
}

/// DIAG-STALL: Reproducer for box-only non-diagonal Q stall via solve_qp_with (full pipeline).
/// min x^2+xy+y^2-6x-6y  s.t. [0,4]^2 → true opt (2,2), obj=-12
/// Q stored as FULL SYMMETRIC (API convention): (0,0)=2, (0,1)=1, (1,0)=1, (1,1)=2
#[test]
fn test_box_only_nondiag_q_stall_reproducer() {
    use crate::qp::solve_qp_with;
    // Q full-symmetric: both triangles
    let q = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0, 1.0, 1.0, 2.0], 2, 2)
        .unwrap();
    let c = vec![-6.0, -6.0];
    let a = CscMatrix::new(0, 2);
    let b = vec![];
    let bounds = vec![(0.0_f64, 4.0_f64); 2];
    let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
    // Test both with and without Ruiz scaling
    for use_ruiz in [false, true] {
        let opts = SolverOptions {
            use_ruiz_scaling: use_ruiz,
            timeout_secs: Some(10.0),
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        eprintln!(
            "ruiz={} STATUS={:?} x={:?} obj={:.6} iters={}",
            use_ruiz, result.status, result.solution, result.objective, result.iterations
        );
        if let Some((pf, df, mu)) = result.final_residuals {
            eprintln!("RESID pf={:.3e} df={:.3e} mu={:.3e}", pf, df, mu);
        }
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "ruiz={} expected Optimal, got {:?}",
            use_ruiz,
            result.status
        );
        assert!(
            (result.solution[0] - 2.0).abs() < 0.1,
            "ruiz={} x[0]={:.4}",
            use_ruiz,
            result.solution[0]
        );
        assert!(
            (result.solution[1] - 2.0).abs() < 0.1,
            "ruiz={} x[1]={:.4}",
            use_ruiz,
            result.solution[1]
        );
        assert!(
            (result.objective - (-12.0)).abs() < 0.5,
            "ruiz={} obj={:.4}",
            use_ruiz,
            result.objective
        );
    }
    // Also test via inner solver directly (no Ruiz)
    let opts_inner = default_opts();
    let q2 = CscMatrix::from_triplets(&[0, 0, 1, 1], &[0, 1, 0, 1], &[2.0, 1.0, 1.0, 2.0], 2, 2)
        .unwrap();
    let problem2 = QpProblem::new_all_le(
        q2,
        vec![-6.0, -6.0],
        CscMatrix::new(0, 2),
        vec![],
        vec![(0.0_f64, 4.0_f64); 2],
    )
    .unwrap();
    let r2 = solve_ippmm_inner(&problem2, &opts_inner, opts_inner.ipm_eps());
    eprintln!(
        "inner STATUS={:?} x={:?} obj={:.6}",
        r2.status, r2.solution, r2.objective
    );
    assert_eq!(
        r2.status,
        SolveStatus::Optimal,
        "inner: expected Optimal, got {:?}",
        r2.status
    );
}

/// PEU AND-gate sentinel: OR→AND change is load-bearing for box-only non-diagonal Q stall.
///
/// This test verifies that the AND gate (both primal AND dual must improve for fast δ decrease)
/// prevents the dual direction blow-up that causes the stall.
///
/// No-op check: if the AND gate is reverted to OR (either_improved → fast decrease), ruiz=true
/// will return SuboptimalSolution instead of Optimal, failing the assertion below.
/// That is, this test MUST fail when the fix is removed.
/// sentinel: PEU 停止判定の AND ゲートを OR へ逆戻りする張 — fast-decrease 単独停止へ戻すと SuboptimalSolution 判定に転じ fail
#[test]
#[allow(clippy::type_complexity)]
fn test_peu_and_gate_box_only_nondiag_stall_sentinel() {
    use crate::qp::solve_qp_with;
    // Multi-pattern: three different box-only non-diagonal Q problems with known optima.
    // Pattern 1: min x²+xy+y²-6x-6y on [0,4]²  → opt=(2,2), obj=-12
    // Pattern 2: min x²+xy+y²-8x-6y on [0,6]²
    //            KKT: 2x+y=8, x+2y=6 → 3y=4 → y=4/3, x=10/3. Feasible in [0,6]². opt=(10/3,4/3).
    //            obj = 100/9+40/9+16/9-80/3-8 = 156/9-312/9 = -156/9 = -52/3 ≈ -17.33
    // Pattern 3: min 2x²+xy+2y²-4x-4y on [0,3]² → KKT: 4x+y=4, x+4y=4 → 15x=12 → x=y=4/5=0.8
    //            obj = 2*(0.64)+0.64+2*(0.64)-4*0.8-4*0.8 = 2.56-6.4 = ... let me recompute.
    //            obj = 2(0.64)+0.8*0.8+2(0.64) - 4*0.8 - 4*0.8 = 1.28+0.64+1.28-3.2-3.2 = -3.2
    let cases: &[(
        Vec<(usize, usize, f64)>,
        Vec<f64>,
        Vec<(f64, f64)>,
        Vec<f64>,
        f64,
        &str,
    )] = &[
        // Q triplets (row,col,val), c, bounds, expected_x*, expected_obj, name
        (
            vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)],
            vec![-6.0, -6.0],
            vec![(0.0, 4.0), (0.0, 4.0)],
            vec![2.0, 2.0],
            -12.0,
            "min x²+xy+y²-6x-6y [0,4]²",
        ),
        (
            vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)],
            vec![-8.0, -6.0],
            vec![(0.0, 6.0), (0.0, 6.0)],
            // KKT: 2x+y=8, x+2y=6 → x=10/3, y=4/3; obj=156/9-104/3=-52/3≈-17.33
            vec![10.0 / 3.0, 4.0 / 3.0],
            -52.0 / 3.0,
            "min x²+xy+y²-8x-6y [0,6]²",
        ),
        (
            vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 4.0)],
            vec![-4.0, -4.0],
            vec![(0.0, 3.0), (0.0, 3.0)],
            vec![4.0 / 5.0, 4.0 / 5.0],
            -3.2,
            "min 2x²+xy+2y²-4x-4y [0,3]²",
        ),
    ];

    for (q_trips, c, bounds, x_star, obj_star, name) in cases {
        let rows: Vec<usize> = q_trips.iter().map(|&(r, _, _)| r).collect();
        let cols: Vec<usize> = q_trips.iter().map(|&(_, c, _)| c).collect();
        let vals: Vec<f64> = q_trips.iter().map(|&(_, _, v)| v).collect();
        let n = bounds.len();
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        let a = CscMatrix::new(0, n);
        let problem = QpProblem::new_all_le(q, c.clone(), a, vec![], bounds.clone()).unwrap();

        for use_ruiz in [false, true] {
            let opts = SolverOptions {
                use_ruiz_scaling: use_ruiz,
                timeout_secs: Some(10.0),
                ..Default::default()
            };
            let result = solve_qp_with(&problem, &opts);
            assert_eq!(
                result.status,
                SolveStatus::Optimal,
                "{} ruiz={}: expected Optimal, got {:?}",
                name,
                use_ruiz,
                result.status
            );
            for (i, &xi_star) in x_star.iter().enumerate() {
                assert!(
                    (result.solution[i] - xi_star).abs() < 0.1,
                    "{} ruiz={}: x[{}]={:.4} expected {:.4}",
                    name,
                    use_ruiz,
                    i,
                    result.solution[i],
                    xi_star
                );
            }
            assert!(
                (result.objective - obj_star).abs() < 0.5,
                "{} ruiz={}: obj={:.4} expected {:.4}",
                name,
                use_ruiz,
                result.objective,
                obj_star
            );
        }
    }
}

/// Multiple-pattern box-only non-diagonal QP with known optima.
/// Covers diagonal Q (should always work), near-diagonal, and strongly coupled cases.
#[test]
#[allow(clippy::type_complexity)]
fn test_box_only_nondiag_q_multi_pattern() {
    use crate::qp::solve_qp_with;

    // (Q as (rows,cols,vals), c, bounds, x_star, obj_star, name)
    let cases: &[(
        &[usize],
        &[usize],
        &[f64],
        &[f64],
        &[(f64, f64)],
        &[f64],
        f64,
        &str,
    )] = &[
        // Diagonal Q — always worked, regression check.
        (
            &[0, 1],
            &[0, 1],
            &[2.0, 2.0],
            &[-4.0, -4.0],
            &[(0.0, 5.0), (0.0, 5.0)],
            &[2.0, 2.0],
            -8.0,
            "diag Q baseline",
        ),
        // Off-diagonal: conditioned Q, box [0,4]²
        (
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[2.0, 1.0, 1.0, 2.0],
            &[-6.0, -6.0],
            &[(0.0, 4.0), (0.0, 4.0)],
            &[2.0, 2.0],
            -12.0,
            "nondiag Q [0,4]²",
        ),
        // Off-diagonal: asymmetric c, box [0,6]²; KKT: 2x+y=8, x+2y=6 → x=10/3, y=4/3
        (
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[2.0, 1.0, 1.0, 2.0],
            &[-8.0, -6.0],
            &[(0.0, 6.0), (0.0, 6.0)],
            &[10.0 / 3.0, 4.0 / 3.0],
            -52.0 / 3.0,
            "nondiag Q asymm-c",
        ),
        // Off-diagonal: tight box forcing boundary solution.
        // min x²+xy+y²-6x-6y on [0,1]² → unconstrained opt (2,2) outside box
        // KKT on box corner: boundary active. At x=(1,1): Qx+c=[-3,-3], dual z=[3,3]>0. ✓
        (
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &[2.0, 1.0, 1.0, 2.0],
            &[-6.0, -6.0],
            &[(0.0, 1.0), (0.0, 1.0)],
            &[1.0, 1.0],
            -9.0,
            "nondiag Q tight box [0,1]²",
        ),
    ];

    for &(rows, cols, vals, c, bounds, x_star, obj_star, name) in cases {
        let n = bounds.len();
        let q = CscMatrix::from_triplets(rows, cols, vals, n, n).unwrap();
        let a = CscMatrix::new(0, n);
        let problem = QpProblem::new_all_le(q, c.to_vec(), a, vec![], bounds.to_vec()).unwrap();

        for use_ruiz in [false, true] {
            let opts = SolverOptions {
                use_ruiz_scaling: use_ruiz,
                timeout_secs: Some(10.0),
                ..Default::default()
            };
            let result = solve_qp_with(&problem, &opts);
            assert_eq!(
                result.status,
                SolveStatus::Optimal,
                "{} ruiz={}: expected Optimal, got {:?}",
                name,
                use_ruiz,
                result.status
            );
            for (i, &xi) in x_star.iter().enumerate() {
                assert!(
                    (result.solution[i] - xi).abs() < 0.1,
                    "{} ruiz={}: x[{}]={:.4} expected {:.4}",
                    name,
                    use_ruiz,
                    i,
                    result.solution[i],
                    xi
                );
            }
            assert!(
                (result.objective - obj_star).abs() < 0.5,
                "{} ruiz={}: obj={:.4} expected {:.4}",
                name,
                use_ruiz,
                result.objective,
                obj_star
            );
        }
    }
}

/// 矛盾する bound (x>=1 かつ x<=0) の infeasible 1変数 QP。consecutive
/// fire が `MIN_CONSECUTIVE_INFEAS` に達し証明を確定する直前に cancel_flag が
/// 立った場合、Infeasible ではなく Timeout を返すこと。
///
/// Sentinel: `iter.rs` の受理直前 `timeout_ctx.should_stop()` チェックを
/// revert すると、cancel が立っても最終連続検出の commit がそのまま進み
/// status は Infeasible のまま返るため、このテストは FAIL する。
fn infeasible_1var_qp() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0], 1, 1).unwrap();
    let c = vec![0.0];
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[-1.0, 1.0], 2, 1).unwrap();
    let b = vec![-1.0, 0.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    QpProblem::new_all_le(q, c, a, b, bounds).unwrap()
}

#[test]
fn cancel_race_before_infeasible_certificate_prefers_timeout() {
    let problem = infeasible_1var_qp();

    INFEAS_COMMIT_COUNT.with(|c| c.set(0));
    CANCEL_AFTER_INFEAS_COMMIT.with(|c| c.set(Some(1)));
    CANCEL_SIGNAL.with(|flag| flag.store(false, Ordering::Relaxed));

    let opts = CANCEL_SIGNAL.with(|flag| SolverOptions {
        timeout_secs: Some(10.0),
        use_ruiz_scaling: false,
        cancel_flag: Some(Arc::clone(flag)),
        ..Default::default()
    });
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());

    CANCEL_AFTER_INFEAS_COMMIT.with(|c| c.set(None));

    assert_eq!(
        result.status,
        SolveStatus::Timeout,
        "cancel racing in right before the Infeasible certificate commit must yield Timeout, got {:?}",
        result.status
    );
}

/// Baseline/control for the sentinel above: without any cancel injection,
/// the same construction must still report the genuine Infeasible
/// certificate (confirms the fixture actually reaches the consecutive-fire
/// detector via `solve_ippmm_inner` directly, with no presolve involved).
#[test]
fn no_cancel_still_reports_infeasible_via_inner_solver() {
    let problem = infeasible_1var_qp();

    INFEAS_COMMIT_COUNT.with(|c| c.set(0));
    CANCEL_AFTER_INFEAS_COMMIT.with(|c| c.set(None));
    CANCEL_SIGNAL.with(|flag| flag.store(false, Ordering::Relaxed));

    let opts = default_opts();
    let result = solve_ippmm_inner(&problem, &opts, opts.ipm_eps());
    assert_eq!(
        result.status,
        SolveStatus::Infeasible,
        "baseline (no cancel): expected Infeasible, got {:?}",
        result.status
    );
}

fn apply_warm_and_extract_x(problem: &QpProblem, xj_warm: f64) -> f64 {
    let (a_ext, b_ext, m_ext, m_orig, _, is_eq_ext) = build_extended_constraints(problem);
    let ws = QpWarmStart {
        x: vec![xj_warm],
        y: vec![0.0_f64; m_orig],
        mu: 1e-3,
    };
    let mut x = vec![0.0_f64; 1];
    let mut y = vec![0.0_f64; m_ext];
    let mut s = vec![0.0_f64; m_ext];
    let mu = apply_qp_warm_start(
        &ws, problem, &a_ext, &b_ext, &is_eq_ext, m_orig, m_ext, &mut x, &mut y, &mut s,
    );
    assert!(mu.is_some(), "apply_qp_warm_start dropped (dim mismatch?)");
    x[0]
}

/// SENTINEL (box 化境界行へ δ 非適用): 狭幅 corner box の μ 停止 deadlock。
///
/// min ½‖x‖² − 1ᵀx s.t. x ∈ [0,3e-3]×[0,5e-3] (制約なし純 box)。box 幅
/// (3e-3,5e-3) < 初期 μ (~4e-3)・δ=DELTA_INIT=8 ≫ 全 σ_t (≤2.5e-3) の領域で
/// box 行へ dual damping を適用すると d_t = 1/(σ_t+δ) ≈ 1/δ に潰え、境界制約が
/// KKT 系から消える (box を無視する発散方向が出る) → fraction-to-boundary が
/// α≈0 で x を壁に固定 → μ が 1 桁も下がらず δ も減衰しない自己閉鎖 deadlock
/// (実測: 52 反復 Stalled、μ=4e-3 固定、x は最適点から 1.5e-3 手前で凍結)。
/// δ 非適用 (d_t = 1/σ_t) では同一問題が Optimal。
///
/// Revert-fail (確認済み): box 行の `bound_row_dual_scale(σ, 0.0)` を
/// `(σ, delta_used)` / `(σ, delta_d)` に戻すと本テストは Stalled で FAIL する。
#[test]
fn sentinel_narrow_box_corner_mu_deadlock() {
    use crate::qp::solve_qp_with;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let problem = QpProblem::new(
        q,
        vec![-1.0, -1.0],
        CscMatrix::new(0, 2),
        vec![],
        vec![(0.0_f64, 3.0e-3), (0.0, 5.0e-3)],
        vec![],
    )
    .unwrap();
    for eps in [1.0e-6, 1.0e-8, 1.0e-9] {
        let opts = SolverOptions {
            presolve: false,
            ipm: crate::options::IpmOptions {
                eps,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = solve_qp_with(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "eps={eps:E} box 幅 < μ の corner が δ 適用で μ 停止した疑い: {:?}",
            result.status
        );
        close(result.solution[0], 3.0e-3, &format!("eps={eps:E} x0 (ub0)"));
        close(result.solution[1], 5.0e-3, &format!("eps={eps:E} x1 (ub1)"));
        // bound_duals layout = [lb0, lb1, ub0, ub1]。非活性 lb 双対は 0、
        // 活性 ub 双対は定数オラクル c_j − x_j·Q_jj (= 1 − ub_j) に一致。
        close(result.bound_duals[0], 0.0, &format!("eps={eps:E} y_lb0"));
        close(result.bound_duals[1], 0.0, &format!("eps={eps:E} y_lb1"));
        close(
            result.bound_duals[2],
            1.0 - 3.0e-3,
            &format!("eps={eps:E} y_ub0"),
        );
        close(
            result.bound_duals[3],
            1.0 - 5.0e-3,
            &format!("eps={eps:E} y_ub1"),
        );
    }
}

/// 退化対 (lb==ub 固定変数) を含む raw 系 (presolve off) が Optimal に到達する
/// ことの回帰。box 化で固定対まで従属スラック化すると (revert: kkt.rs の
/// `bound_pair_is_degenerate` guard 撤去)、床張り付き 2 スラックの dx noise で
/// α が全局的に潰れ 51 反復 Stalled・双対 ~3e6 爆発 (revertfail/REVERT_B.log
/// 実測)。構造 sentinel は kkt.rs の
/// `sentinel_degenerate_bound_pair_kept_as_system_row`、本テストは同じ病理の
/// 求解挙動側を固定する。
#[test]
fn test_fixed_bound_pair_raw_solve_optimal() {
    use crate::qp::solve_qp_with;
    let q = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[0.001, 0.001, 0.001], 3, 3).unwrap();
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 3).unwrap();
    let problem = QpProblem::new(
        q,
        vec![-1.0, -1.0, 1.0],
        a,
        vec![10.0],
        vec![(0.0_f64, 3.0), (0.0, 5.0), (2.0, 2.0)],
        vec![ConstraintType::Le],
    )
    .unwrap();
    let opts = SolverOptions {
        presolve: false,
        ..Default::default()
    };
    let result = solve_qp_with(&problem, &opts);
    assert_eq!(result.status, SolveStatus::Optimal, "raw 固定対系が停留");
    close(result.solution[0], 3.0, "x0 (ub0 活性)");
    close(result.solution[1], 5.0, "x1 (ub1 活性)");
    close(result.solution[2], 2.0, "x2 (固定)");
}

/// SENTINEL (cold 政策 vs warm 用マージン): `strict_interior_shift` は cold 政策
/// (両端有限 `(range·COLD_BOUND_REL_MARGIN).min(COLD_BOUND_ABS_MARGIN)`、
/// 片側有限 `COLD_BOUND_ABS_MARGIN`) で押し込む。warm 専用 1e-6 相対
/// (`warm_bound_margin` / `range·WARM_BOUND_REL_MARGIN`) を cold 始点に適用すると
/// box 行スラックが初期から boundary に張り付き barrier 対角が発散する
/// (cold x0 への誤用が修正対象の病理そのもの)。
///
/// Revert-fail (実測 coldboyd_revert_fail.log): helper のマージン政策を warm 用
/// 1e-6 に戻すと (1) の押し込みが −4.999995 ≠ −4.0、(2) が 1e-4 ≠ 1.0 で FAIL。
#[test]
fn sentinel_strict_interior_shift_uses_cold_policy_not_warm_margin() {
    use super::init::strict_interior_shift;

    /// 押し込み量の実測は clamp 境界の f64 同一性なので余裕は機械_EPS 級で足りるが、
    /// 境界演算の丸め (加算・clamp) に備えて 1e-12。
    const SHIFT_TOL: f64 = 1e-12;

    /// (ラベル, bounds, 始点, cold 政策の期待終点)。
    type ShiftCase = (&'static str, Vec<(f64, f64)>, Vec<f64>, Vec<f64>);

    let cases: Vec<ShiftCase> = vec![
        // (1) 片側 lb=−5 上の始点: cold 絶対 1.0 → −4.0 (warm 1e-6·|−5| なら −4.999995)
        (
            "one-sided-lb",
            vec![(-5.0, f64::INFINITY)],
            vec![-5.0],
            vec![-4.0],
        ),
        // (2) 両端 [0,100]: range·1e-2 = 1.0 が cap 1.0 (warm 相対 1e-6 なら 1e-4)
        ("both-capped", vec![(0.0, 100.0)], vec![0.0], vec![1.0]),
        // (3) 両端 [0,1]: range·1e-2 = 0.01 (warm 相対なら 1e-6)
        ("both-rel", vec![(0.0, 1.0)], vec![0.0], vec![0.01]),
        // 不変条件: 内部始点・非有界・固定変数は動かさない
        (
            "unchanged",
            vec![(0.0, 10.0), (f64::NEG_INFINITY, f64::INFINITY), (2.0, 2.0)],
            vec![5.0, -1e9, 2.0],
            vec![5.0, -1e9, 2.0],
        ),
    ];
    for (label, bounds, x0, expect) in cases {
        let mut x = x0.clone();
        strict_interior_shift(&bounds, &mut x);
        for (i, (got, want)) in x.iter().zip(expect.iter()).enumerate() {
            assert!(
                (got - want).abs() <= SHIFT_TOL,
                "{label}[{i}]: shifted {got} != cold 政策の期待値 {want} \
                 (warm 用 1e-6 マージンの cold 誤用に逆戻りした疑い)"
            );
        }
    }
}

/// SENTINEL (cold-init 射影の境界行重み・cold 政策): 手計算オラクル x = 3/4 + m/4。
///
/// Fixture・導出は旧 `sentinel_cold_init_projection_bound_weight_oracle` から不変
/// (削除+再追加は cold 政策分離で期待 m が変わったため、assert 直接変更禁止規約準拠)。
/// min 0 (Q=0) s.t. x = 1 (等式)、x ∈ [0, 2]、raw 起点 x0 = 0 (lb 上)。
/// cold start は `strict_interior_shift` で x0 を cold 政策
/// m = (range·COLD_BOUND_REL_MARGIN).min(COLD_BOUND_ABS_MARGIN) = 0.02 (range = 2)
/// へ押し込んでから射影する。box 系射影 (σ=0、ρ=δ=1、境界行 lb/ub
/// 2 本とも box 化) は 2×2 系
///   [[1 + 2W, 1], [1, −1]] [dx, dy]ᵀ = [2 − 2m, 1 − m]ᵀ、W = 1/(0 + δ_d^bound)
/// (x 対角 = δ_p·1 + lb/ub 行の対角寄与 d_t。右辺 x 成分 =
///  Σ_t a_t(b_t − a_t·x0s) = (ub − x0s) − (x0s − lb) = 2 − 2m。系行は等式 1 本
/// で対角 −(0+δ) = −1、右辺 r_p = 1 − m)。2 式を足して dx = 3(1−m)/(2+2W)、
/// 終点は x = m + dx = 3/4 + m/4 (m ≪ 1 で射影後の clamp は非発動)。設計契約は
/// δ_d^bound = INIT_PROJECTION_REGULARIZATION = 1 (旧 extended 射影との
/// 代数同値 — `bound_row_dual_scale` / `materialize` の doc)。
///
/// 期待 m は named const 導出だと revert で期待値も動く自己参照になるため
/// literal 0.02 固定 + 出所 assert を併置する。
///
/// Revert-fail (実測 coldboyd_revert_fail.log): cold 政策を warm 用 1e-6 に戻すと
/// m = 2e-6 で終点 0.755 → 0.7500005、出所 assert も 0.02 ≠ 2e-6 で FAIL。
#[test]
fn sentinel_cold_init_projection_bound_weight_oracle_cold_policy() {
    use super::init::build_initial_point;
    use super::state::{COLD_BOUND_ABS_MARGIN, COLD_BOUND_REL_MARGIN};
    use crate::qp::ipm_core::kkt::{build_base_constraints, build_bound_box};
    use otspot_num::linalg::timeout::TimeoutCtx;

    /// strict-interior 押し込み量: cold 政策の literal 固定 (range = 2 で
    /// (2·1e-2).min(1.0) = 0.02)。導出禁止の理由は上記 doc。
    const INIT_SHIFT_MARGIN: f64 = 0.02;
    // 出所固定: named const の方が literal とズレたら本 test を直せ、という合図。
    assert_eq!(
        (2.0 * COLD_BOUND_REL_MARGIN).min(COLD_BOUND_ABS_MARGIN),
        INIT_SHIFT_MARGIN,
        "cold 政策 const が sentinel の期待 m と不一致 (政策変更時は literal も更新)"
    );
    /// 手計算オラクル: x = m + 3(1−m)/(2+2·1) = 3/4 + m/4 (導出は上の doc)。
    const ORACLE_COLD_INIT_X: f64 = 0.75 + 0.25 * INIT_SHIFT_MARGIN;
    /// 2×2 系は f64 因子化でこの桁まで一意 (条件数 ~6)。因子化残余の余裕。
    const ORACLE_COLD_INIT_TOL: f64 = 1e-9;

    let problem = QpProblem::new(
        CscMatrix::new(1, 1),
        vec![0.0],
        CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap(),
        vec![1.0],
        vec![(0.0_f64, 2.0)],
        vec![ConstraintType::Eq],
    )
    .unwrap();
    let (a_ext, b_ext, m_ext, m_orig, _n_lb, is_eq_ext) = build_extended_constraints(&problem);
    let (a_base, b_base, is_eq_base) = build_base_constraints(&problem);
    let bound = build_bound_box(&problem);
    assert_eq!(bound.count(), 2, "lb/ub 各 1 行 (box 化 2 行)");
    assert!(
        bound.eliminated.iter().all(|&e| e),
        "退化対でないので全行 box 化"
    );

    let opts = default_opts();
    let timeout_ctx = TimeoutCtx::new(None, opts.timeout_secs, None);
    let init = build_initial_point(
        &problem,
        &opts,
        &a_ext,
        &b_ext,
        &is_eq_ext,
        &a_base,
        &b_base,
        &is_eq_base,
        &bound,
        m_orig,
        m_ext,
        &timeout_ctx,
        faer::Par::Seq,
    );
    assert!(
        (init.x[0] - ORACLE_COLD_INIT_X).abs() <= ORACLE_COLD_INIT_TOL,
        "cold-init 射影が手計算オラクル 3/(2+2·1)=0.75 と不一致: x0=0 + dx = {} \
         (境界行重みが反復経路の δ=0 に逆戻りし、床 1/ε=4.5e15/行で歩幅が潰れた疑い)",
        init.x[0]
    );
}
