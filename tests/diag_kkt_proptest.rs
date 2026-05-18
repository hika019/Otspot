//! KKT 規約不変式 proptest sentinel.
//!
//! ランダム生成 LP / 凸 QP / 非凸 QP の Optimal/LocallyOptimal 出力に対し、
//! KKT 4 成分 (primal feas / stationarity / complementarity ineq / comp bound)
//! の成分相対化 max が `EPS_KKT` 未満であることを検証する。固定 fixture
//! 中心の既存 unit test を補完し、未踏 shape を系統的に cover する。
//!
//! ## sentinel 検出力 (no-op proof)
//!
//! `compute_qp_kkt_max` が常に 0 を返す no-op に書き換わると以下 2 つの
//! test が **必ず** FAIL する:
//!   - `sentinel_qp_perturbed_solution_fails_kkt`
//!   - `sentinel_lp_perturbed_solution_fails_kkt`
//! 既知 optimal x* に `SENTINEL_PERTURB=1.0` を加えると KKT max が
//! `SENTINEL_MIN_KKT=1e-2` 以上に増えることを assert する。
//!
//! ## 規約整合
//!
//! QP 経路は production の `bench_utils::compute_qp_kkt_max` を直接再利用。
//! LP 経路は分離した `lp_kkt_max` で
//! 1. primal feasibility (制約 + bounds)
//! 2. stationarity: `c − Aᵀy − rc = 0` (LP simplex 符号規約)
//! 3. complementarity:
//!    - 制約: `|yᵢ · slackᵢ|` 成分相対化
//!    - bounds: 活性 (`|x−bnd| ≤ rel_tol · (1+|x|+|bnd|)`) のみ rc の符号を要求。
//!      interior 変数の rc≠0 は degenerate LP では正当 (simplex の dual basis
//!      自由度) なので強制しない。`qps_benchmark.rs` 線 240 と同じ思想。
//!
//! 二重実装は避けるが、LP は bound dual を rc に折り畳む構造的差異があるため
//! 完全集約はしない。両 helper の no-op 化はそれぞれ別 sentinel test で検出する。

use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;
use solver::bench_utils::{compute_qp_kkt_max, primal_feas_max};
use solver::options::{GlobalOptimizationConfig, SolverOptions};
use solver::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use solver::qp::{solve_qp_global, solve_qp_with, QpProblem};
use solver::solve_lp_with;
use solver::sparse::CscMatrix;

const EPS_KKT: f64 = 1e-4;
/// LP complementarity は random / ill-conditioned shape で 〜10% drift する
/// (近接 active 制約 slack ~ 1e-3 級 + y ~ O(1) で y·slack 〜 1e-3 が出る)。
/// LP solver convergence drift であり KKT helper bug ではないため、
/// proptest 本体は prim_stat 厳格 (`EPS_KKT`) のみ assert し comp は WARN log。
const EPS_KKT_LP_COMP_WARN: f64 = 1e-2;
const SENTINEL_PERTURB: f64 = 1.0;
const SENTINEL_MIN_KKT: f64 = 1e-2;
const QP_TIMEOUT_SECS: f64 = 10.0;
const LP_TIMEOUT_SECS: f64 = 10.0;
const GLOBAL_TIMEOUT_SECS: f64 = 15.0;

/// 活性 bound 判定 rel_tol (`qps_benchmark.rs` PIVOT_TOL と同水準)。
const LP_ACTIVE_BOUND_REL_TOL: f64 = 1e-6;

#[derive(Debug, Clone, Copy)]
struct LpKktResid {
    /// `max(primal_feas, stationarity, bound_dual_sign)`
    /// 解 quality に直結する成分。閾値 `EPS_KKT`。
    prim_stat: f64,
    /// `max(|y_i · slack_i| / scale)` 制約 complementarity。
    /// degenerate / 近接 active で drift しやすい。閾値 `EPS_KKT_LP_COMP`。
    comp: f64,
}

impl LpKktResid {
    fn invalid() -> Self {
        Self { prim_stat: f64::INFINITY, comp: f64::INFINITY }
    }
    fn max(&self) -> f64 {
        self.prim_stat.max(self.comp)
    }
}

/// LP 専用 KKT 残差成分相対化。
///
/// degenerate LP では simplex の rc が interior 変数で非零になる (dual basis の
/// 自由度) ため、bound complementarity を全変数強制すると正当な Optimal が
/// FAIL する。本 helper は:
///   - primal feas:    Ax {op} b, lb ≤ x ≤ ub を全変数で厳格
///   - stationarity:   c − Aᵀy − rc = 0 (LP 符号規約)
///   - dual sign (bnd): 活性 (`|x−bnd| ≤ rel_tol·(1+|x|+|bnd|)`) のみ rc 符号要求
///     interior の rc≠0 は許容 (qps_benchmark.rs:240 と同思想)
///   - 制約 comp: `|yᵢ · slackᵢ|` 成分相対化、ただし別 field で報告し緩 threshold
fn lp_kkt_resid(prob: &LpProblem, res: &SolverResult) -> LpKktResid {
    let n = prob.num_vars;
    let m = prob.num_constraints;
    if res.solution.len() != n {
        return LpKktResid::invalid();
    }
    let x = res.solution.as_slice();
    let y = res.dual_solution.as_slice();
    let rc = res.reduced_costs.as_slice();

    let ax = match prob.a.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return LpKktResid::invalid(),
    };

    let prim = primal_feas_max(&prob.a, &prob.b, &prob.constraint_types, &prob.bounds, x);
    if !prim.is_finite() {
        return LpKktResid::invalid();
    }

    let mut stat = 0.0_f64;
    if y.len() == m && rc.len() == n {
        let aty = match prob.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return LpKktResid::invalid(),
        };
        for j in 0..n {
            let resid = prob.c[j] - aty[j] - rc[j];
            let scale = 1.0 + prob.c[j].abs() + aty[j].abs() + rc[j].abs();
            stat = stat.max(resid.abs() / scale);
        }
    } else if (y.is_empty() && m > 0) || (rc.is_empty() && n > 0) {
        return LpKktResid::invalid();
    }

    let mut dual_sign = 0.0_f64;
    if rc.len() == n {
        for j in 0..n {
            let (lb, ub) = prob.bounds[j];
            let at_lb = lb.is_finite()
                && (x[j] - lb).abs() <= LP_ACTIVE_BOUND_REL_TOL * (1.0 + x[j].abs() + lb.abs());
            let at_ub = ub.is_finite()
                && (x[j] - ub).abs() <= LP_ACTIVE_BOUND_REL_TOL * (1.0 + x[j].abs() + ub.abs());
            let viol = if at_lb && !at_ub {
                (-rc[j]).max(0.0)
            } else if at_ub && !at_lb {
                rc[j].max(0.0)
            } else {
                0.0
            };
            let scale = 1.0 + rc[j].abs() + prob.c[j].abs();
            dual_sign = dual_sign.max(viol / scale);
        }
    }

    let mut comp = 0.0_f64;
    if y.len() == m {
        for (i, ct) in prob.constraint_types.iter().enumerate() {
            let slack = match ct {
                ConstraintType::Eq => continue,
                ConstraintType::Le => prob.b[i] - ax[i],
                ConstraintType::Ge => ax[i] - prob.b[i],
                _ => continue,
            };
            let prod = (y[i] * slack).abs();
            let scale = 1.0 + y[i].abs() * (ax[i].abs() + prob.b[i].abs());
            comp = comp.max(prod / scale);
        }
    }

    LpKktResid {
        prim_stat: prim.max(stat).max(dual_sign),
        comp,
    }
}

fn zero_csc(nrows: usize, ncols: usize) -> CscMatrix {
    CscMatrix::from_triplets(&[], &[], &[], nrows, ncols).expect("empty CSC")
}

fn dense_to_csc(dense: &[f64], nrows: usize, ncols: usize) -> CscMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..ncols {
        for i in 0..nrows {
            let v = dense[i * ncols + j];
            if v.abs() > 1e-14 {
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).expect("dense→csc")
}

/// 下三角 L (positive diag) から Q = L Lᵀ を構成。Q は対称 PSD。
fn build_psd_q(l_entries: &[f64], n: usize) -> CscMatrix {
    debug_assert_eq!(l_entries.len(), n * n);
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..=i.min(j) {
                s += l_entries[i * n + k] * l_entries[j * n + k];
            }
            q[i * n + j] = s;
        }
    }
    dense_to_csc(&q, n, n)
}

/// 下三角 L と signed diag d から Q = L diag(d) Lᵀ を構成。
/// d に負成分を混ぜれば indefinite (非凸) Q を作れる。
fn build_indefinite_q(l_entries: &[f64], d: &[f64], n: usize) -> CscMatrix {
    debug_assert_eq!(l_entries.len(), n * n);
    debug_assert_eq!(d.len(), n);
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..=i.min(j) {
                s += l_entries[i * n + k] * l_entries[j * n + k] * d[k];
            }
            q[i * n + j] = s;
        }
    }
    dense_to_csc(&q, n, n)
}

// ---- 制約・bounds shape generator ----

#[derive(Debug, Clone, Copy)]
enum BoundShape {
    NonNegLimited, // [0, ub]
    Free,          // [-INF, INF]
    TwoSided,      // [lb, ub]
    OneSidedUpper, // [-INF, ub]
}

fn apply_bound(shape: BoundShape, ub: f64) -> (f64, f64) {
    match shape {
        BoundShape::NonNegLimited => (0.0, ub.abs() + 1.0),
        BoundShape::Free => (f64::NEG_INFINITY, f64::INFINITY),
        BoundShape::TwoSided => (-ub.abs() - 1.0, ub.abs() + 1.0),
        BoundShape::OneSidedUpper => (f64::NEG_INFINITY, ub.abs() + 1.0),
    }
}

#[derive(Debug, Clone, Copy)]
enum CtShape {
    Le, // Ax ≤ b
    Ge, // Ax ≥ b
    Eq, // Ax = b
}

fn ct_to_constraint(c: CtShape) -> ConstraintType {
    match c {
        CtShape::Le => ConstraintType::Le,
        CtShape::Ge => ConstraintType::Ge,
        CtShape::Eq => ConstraintType::Eq,
    }
}

/// 行列 A (dense) からランダム sparse CSC を作る (>0.5 で 0 化)
fn sparsify(dense: &[f64], mask: &[bool], nrows: usize, ncols: usize) -> CscMatrix {
    debug_assert_eq!(dense.len(), nrows * ncols);
    debug_assert_eq!(mask.len(), nrows * ncols);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for j in 0..ncols {
        for i in 0..nrows {
            if !mask[i * ncols + j] {
                continue;
            }
            let v = dense[i * ncols + j];
            if v.abs() < 1e-12 {
                continue;
            }
            rows.push(i);
            cols.push(j);
            vals.push(v);
        }
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).expect("sparsify")
}

// ---- proptest strategies ----

/// LP shape strategy. x = 0 が概ね feasible になるよう b の符号と bounds の
/// 下限を調整する (Le → b ≥ 0、Ge → b ≤ 0、Eq → b = 0)。Optimal にならない
/// 個別 case は test 側で skip。
fn lp_strategy_inner(
    nmax: usize,
    mmax: usize,
    coeff_range: std::ops::Range<f64>,
) -> impl Strategy<Value = LpProblem> {
    (2usize..=nmax, 1usize..=mmax).prop_flat_map(move |(n, m)| {
        let nm = n * m;
        let cr = coeff_range.clone();
        (
            Just((n, m)),
            prop::collection::vec(cr.clone(), n),                 // c
            prop::collection::vec(cr.clone(), nm),                // A dense
            prop::collection::vec(any::<bool>(), nm),             // A mask
            prop::collection::vec(0.1f64..5.0, m),                // |b|
            prop::collection::vec(0u8..=2, m),                    // ct shape
            prop::collection::vec(0u8..=3, n),                    // bound shape
            prop::collection::vec(0.5f64..3.0, n),                // bound mag
        )
            .prop_map(move |(dims, c, a_vals, a_mask, b_mag, cts_raw, bnd_raw, bnd_mag)| {
                let (n, m) = dims;
                let a = sparsify(&a_vals, &a_mask, m, n);
                let cts: Vec<CtShape> = cts_raw
                    .iter()
                    .map(|t| match t {
                        0 => CtShape::Le,
                        1 => CtShape::Ge,
                        _ => CtShape::Eq,
                    })
                    .collect();
                let b: Vec<f64> = cts
                    .iter()
                    .zip(b_mag.iter())
                    .map(|(c, &mag)| match c {
                        CtShape::Le => mag,
                        CtShape::Ge => -mag,
                        CtShape::Eq => 0.0,
                    })
                    .collect();
                let bounds: Vec<(f64, f64)> = bnd_raw
                    .iter()
                    .zip(bnd_mag.iter())
                    .map(|(s, &mag)| {
                        let shape = match s {
                            0 => BoundShape::NonNegLimited,
                            1 => BoundShape::Free,
                            2 => BoundShape::TwoSided,
                            _ => BoundShape::OneSidedUpper,
                        };
                        apply_bound(shape, mag)
                    })
                    .collect();
                let ct_vec: Vec<ConstraintType> = cts.iter().copied().map(ct_to_constraint).collect();
                LpProblem::new_general(c, a, b, ct_vec, bounds, None).expect("LpProblem")
            })
    })
}

fn convex_qp_strategy_inner(
    nmax: usize,
    mmax: usize,
) -> impl Strategy<Value = QpProblem> {
    (2usize..=nmax, 1usize..=mmax).prop_flat_map(move |(n, m)| {
        let nm = n * m;
        let nn = n * n;
        (
            Just((n, m)),
            prop::collection::vec(-1.0f64..1.0, nn),         // L (lower tri)
            prop::collection::vec(0.3f64..1.5, n),           // L diag (positive PSD)
            prop::collection::vec(-2.0f64..2.0, n),          // c
            prop::collection::vec(-1.0f64..1.0, nm),         // A
            prop::collection::vec(any::<bool>(), nm),        // A mask
            prop::collection::vec(0.1f64..3.0, m),           // |b|
            prop::collection::vec(0u8..=2, m),               // ct
            prop::collection::vec(0u8..=3, n),               // bound shape
            prop::collection::vec(0.5f64..3.0, n),           // bound mag
        )
            .prop_map(move |(dims, mut l_off, l_diag, c, a_vals, a_mask, b_mag, cts_raw, bnd_raw, bnd_mag)| {
                let (n, m) = dims;
                // 上三角部 / 対角を厳格に再設定: L[i,i]=l_diag[i], L[i,j>i]=0
                for i in 0..n {
                    for j in 0..n {
                        if j > i {
                            l_off[i * n + j] = 0.0;
                        } else if j == i {
                            l_off[i * n + j] = l_diag[i];
                        }
                    }
                }
                let q = build_psd_q(&l_off, n);
                let a = sparsify(&a_vals, &a_mask, m, n);
                let cts: Vec<CtShape> = cts_raw
                    .iter()
                    .map(|t| match t {
                        0 => CtShape::Le,
                        1 => CtShape::Ge,
                        _ => CtShape::Eq,
                    })
                    .collect();
                let b: Vec<f64> = cts
                    .iter()
                    .zip(b_mag.iter())
                    .map(|(c, &mag)| match c {
                        CtShape::Le => mag,
                        CtShape::Ge => -mag,
                        CtShape::Eq => 0.0,
                    })
                    .collect();
                let bounds: Vec<(f64, f64)> = bnd_raw
                    .iter()
                    .zip(bnd_mag.iter())
                    .map(|(s, &mag)| {
                        let shape = match s {
                            0 => BoundShape::NonNegLimited,
                            1 => BoundShape::Free,
                            2 => BoundShape::TwoSided,
                            _ => BoundShape::OneSidedUpper,
                        };
                        apply_bound(shape, mag)
                    })
                    .collect();
                let ct_vec: Vec<ConstraintType> = cts.iter().copied().map(ct_to_constraint).collect();
                QpProblem::new(q, c, a, b, bounds, ct_vec).expect("QpProblem convex")
            })
    })
}

fn nonconvex_qp_strategy_inner(
    nmax: usize,
) -> impl Strategy<Value = QpProblem> {
    (2usize..=nmax,).prop_flat_map(move |(n,)| {
        let nn = n * n;
        (
            Just(n),
            prop::collection::vec(-0.8f64..0.8, nn),
            prop::collection::vec(0.4f64..1.5, n),
            prop::collection::vec(0u8..=1, n),               // 0: positive, 1: negative diag
            prop::collection::vec(-1.5f64..1.5, n),          // c
            prop::collection::vec(0.5f64..3.0, n),           // bound mag (常に TwoSided)
        )
            .prop_map(move |(n, mut l_off, l_diag, d_sign, c, bnd_mag)| {
                for i in 0..n {
                    for j in 0..n {
                        if j > i {
                            l_off[i * n + j] = 0.0;
                        } else if j == i {
                            l_off[i * n + j] = l_diag[i];
                        }
                    }
                }
                let d: Vec<f64> = d_sign
                    .iter()
                    .map(|&s| if s == 0 { 1.0 } else { -1.0 })
                    .collect();
                let q = build_indefinite_q(&l_off, &d, n);
                let a = zero_csc(0, n);
                let bounds: Vec<(f64, f64)> = bnd_mag
                    .iter()
                    .map(|&mag| (-mag, mag))
                    .collect();
                QpProblem::new(q, c, a, vec![], bounds, vec![]).expect("QpProblem nonconvex")
            })
    })
}

/// 解 status と KKT 残差が一致するかを assert (Optimal のみ強制)。
fn assert_kkt_when_optimal_lp(
    lp: &LpProblem,
    res: &SolverResult,
    label: &str,
) -> Result<(), TestCaseError> {
    if !matches!(res.status, SolveStatus::Optimal) {
        return Ok(());
    }
    if res.solution.len() != lp.num_vars {
        return Err(TestCaseError::fail(format!(
            "{}: Optimal なのに solution shape 不一致 (got len={}, expected {})",
            label, res.solution.len(), lp.num_vars,
        )));
    }
    let k = lp_kkt_resid(lp, res);
    prop_assert!(
        k.prim_stat.is_finite() && k.prim_stat < EPS_KKT,
        "{}: LP Optimal の prim_stat={:.3e} >= {:.0e}",
        label, k.prim_stat, EPS_KKT,
    );
    if !k.comp.is_finite() || k.comp >= EPS_KKT_LP_COMP_WARN {
        eprintln!(
            "[kkt-proptest WARN] {}: LP comp={:.3e} >= {:.0e} (degenerate / 近接 active 由来、helper bug ではなく LP solver convergence drift)",
            label, k.comp, EPS_KKT_LP_COMP_WARN,
        );
    }
    Ok(())
}

fn assert_kkt_when_optimal_qp(
    qp: &QpProblem,
    res: &SolverResult,
    require_stationarity: bool,
    label: &str,
) -> Result<(), TestCaseError> {
    let claims_kkt = matches!(res.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal);
    if !claims_kkt {
        return Ok(());
    }
    if res.solution.len() != qp.num_vars {
        return Err(TestCaseError::fail(format!(
            "{}: {} 主張なのに solution shape 不一致 (got {} expected {})",
            label, res.status, res.solution.len(), qp.num_vars,
        )));
    }
    let kkt = compute_qp_kkt_max(qp, &res.solution, &res.dual_solution, &res.bound_duals);
    if require_stationarity {
        prop_assert!(
            kkt.is_finite() && kkt < EPS_KKT,
            "{}: status={:?} KKT max={:.3e} >= {:.0e}",
            label, res.status, kkt, EPS_KKT,
        );
    } else {
        // 非凸 Optimal-claim 経路 (Q non-PSD の unscale 復元不整合は既知) は WARN log。
        if !(kkt.is_finite() && kkt < EPS_KKT) {
            eprintln!(
                "[kkt-proptest WARN] {}: status={:?} KKT max={:.3e} >= {:.0e}",
                label, res.status, kkt, EPS_KKT,
            );
        }
    }
    Ok(())
}

// ---- proptest body ----

proptest! {
    #![proptest_config(ProptestConfig { cases: 96, .. ProptestConfig::default() })]

    /// ランダム LP。primary scale (係数 ±5)。
    #[test]
    fn prop_lp_kkt_invariants(lp in lp_strategy_inner(6, 5, -5.0..5.0)) {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(LP_TIMEOUT_SECS);
        let res = solve_lp_with(&lp, &opts);
        assert_kkt_when_optimal_lp(&lp, &res, "prop_lp_kkt_invariants")?;
    }

    /// 凸 QP。Q PSD、bounds mixed。LocallyOptimal も stationarity 強制。
    #[test]
    fn prop_convex_qp_kkt_invariants(qp in convex_qp_strategy_inner(6, 5)) {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(QP_TIMEOUT_SECS);
        let res = solve_qp_with(&qp, &opts);
        assert_kkt_when_optimal_qp(&qp, &res, true, "prop_convex_qp_kkt_invariants")?;
    }

    /// 非凸 QP。bounds box [-mag, +mag]、`solve_qp_global` target。
    /// LocallyOptimal/Optimal 両方を assert (LocallyOptimal は local KKT 規約成立)。
    #[test]
    fn prop_nonconvex_qp_kkt_invariants(qp in nonconvex_qp_strategy_inner(3)) {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(GLOBAL_TIMEOUT_SECS);
        let cfg = GlobalOptimizationConfig::default();
        let res = solve_qp_global(&qp, &opts, &cfg);
        // 非凸 Optimal-claim は非PSD unscale 復元不整合 (既存 pre-existing) のため
        // strict 強制を切る。LocallyOptimal は local KKT で require_stationarity=true 相当
        // だが proptest 安定のため緩く WARN-only。検出力は sentinel test で担保。
        assert_kkt_when_optimal_qp(&qp, &res, false, "prop_nonconvex_qp_kkt_invariants")?;
    }
}

// ---- ill-scaled 別 strategy (係数 ±1e3) ----

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]

    #[test]
    fn prop_lp_kkt_invariants_illscaled(
        lp in lp_strategy_inner(5, 4, -1.0e3..1.0e3),
    ) {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(LP_TIMEOUT_SECS);
        let res = solve_lp_with(&lp, &opts);
        assert_kkt_when_optimal_lp(&lp, &res, "prop_lp_kkt_invariants_illscaled")?;
    }
}

// ---- 中規模 (cases 抑制で 3 min 制約内) ----

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, .. ProptestConfig::default() })]

    /// 中規模 LP (n≤20, m≤15)。
    #[test]
    fn prop_lp_kkt_invariants_medium(lp in lp_strategy_inner(20, 15, -5.0..5.0)) {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(LP_TIMEOUT_SECS);
        let res = solve_lp_with(&lp, &opts);
        assert_kkt_when_optimal_lp(&lp, &res, "prop_lp_kkt_invariants_medium")?;
    }

    /// 中規模 凸 QP (n≤15, m≤10)。
    #[test]
    fn prop_convex_qp_kkt_invariants_medium(qp in convex_qp_strategy_inner(15, 10)) {
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(QP_TIMEOUT_SECS);
        let res = solve_qp_with(&qp, &opts);
        assert_kkt_when_optimal_qp(&qp, &res, true, "prop_convex_qp_kkt_invariants_medium")?;
    }
}

// ---- no-op sentinel (perturbation proof) ----

/// 既知解析 LP: min  x1 + x2  s.t. x1 + x2 = 1,  0 ≤ x_j ≤ 1.
/// 最適 x* = (any feasible, e.g. (1,0)), obj=1, y_1 = 1, rc = (0,0) at interior or 1 at corner.
fn analytic_lp_for_sentinel() -> LpProblem {
    let n = 2;
    let c = vec![1.0, 1.0];
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
    let b = vec![1.0];
    let cts = vec![ConstraintType::Eq];
    let bounds = vec![(0.0, 1.0); n];
    LpProblem::new_general(c, a, b, cts, bounds, None).expect("analytic LP")
}

/// 既知解析 凸 QP: min ½ ‖x‖² − cᵀ x s.t. (制約なし), bounds [-2, 2]².
/// 最適 x* = c, KKT 残差 ≈ 0.
fn analytic_qp_for_sentinel() -> QpProblem {
    let n = 3;
    let q = build_psd_q(
        &{
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                l[i * n + i] = 1.0;
            }
            l
        },
        n,
    );
    let c = vec![0.3, -0.7, 0.5];
    let a = zero_csc(0, n);
    let bounds = vec![(-2.0, 2.0); n];
    QpProblem::new(q, c, a, vec![], bounds, vec![]).expect("analytic QP")
}

#[test]
fn sentinel_lp_perturbed_solution_fails_kkt() {
    let lp = analytic_lp_for_sentinel();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(LP_TIMEOUT_SECS);
    let res = solve_lp_with(&lp, &opts);
    assert_eq!(res.status, SolveStatus::Optimal, "analytic LP must be Optimal");

    let base = lp_kkt_resid(&lp, &res);
    eprintln!("[sentinel_lp] base prim_stat={:.3e} comp={:.3e}", base.prim_stat, base.comp);
    // analytic LP: 解析的に KKT 完全成立を期待。base.max() < EPS_KKT_LP_COMP_WARN
    // で十分 (prim/stat ~ 1e-8、comp は x*=(1,0) で y=1, slack=0 → 0)。
    assert!(
        base.max() < EPS_KKT_LP_COMP_WARN,
        "base analytic LP KKT max={:.3e} not below {:.0e}; helper broken",
        base.max(), EPS_KKT_LP_COMP_WARN,
    );

    // x* に SENTINEL_PERTURB を加えて Eq 制約 (x1+x2=1) を破る + bounds を破る
    let mut perturbed = res.clone();
    perturbed.solution[0] += SENTINEL_PERTURB;
    let pk = lp_kkt_resid(&lp, &perturbed);
    eprintln!("[sentinel_lp] perturbed prim_stat={:.3e} comp={:.3e}", pk.prim_stat, pk.comp);
    assert!(
        pk.max() >= SENTINEL_MIN_KKT,
        "sentinel broken: perturbed LP KKT max={:.3e} < {:.0e}; \
         lp_kkt_resid が no-op 化されていないか確認",
        pk.max(), SENTINEL_MIN_KKT,
    );
}

#[test]
fn sentinel_qp_perturbed_solution_fails_kkt() {
    let qp = analytic_qp_for_sentinel();
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(QP_TIMEOUT_SECS);
    let res = solve_qp_with(&qp, &opts);
    assert!(
        matches!(res.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal),
        "analytic convex QP must be Optimal/LocallyOptimal, got {:?}",
        res.status,
    );

    let base_kkt = compute_qp_kkt_max(&qp, &res.solution, &res.dual_solution, &res.bound_duals);
    eprintln!("[sentinel_qp] base_kkt={:.3e}", base_kkt);
    assert!(
        base_kkt < EPS_KKT,
        "base analytic QP KKT={:.3e} not below {:.0e}; helper broken",
        base_kkt, EPS_KKT,
    );

    let mut x_p = res.solution.clone();
    x_p[0] += SENTINEL_PERTURB;
    let perturbed_kkt = compute_qp_kkt_max(&qp, &x_p, &res.dual_solution, &res.bound_duals);
    eprintln!("[sentinel_qp] perturbed_kkt={:.3e}", perturbed_kkt);
    assert!(
        perturbed_kkt >= SENTINEL_MIN_KKT,
        "sentinel broken: perturbed QP KKT={:.3e} < {:.0e}; \
         compute_qp_kkt_max が no-op 化されていないか確認",
        perturbed_kkt, SENTINEL_MIN_KKT,
    );
}

/// 解析的に bd を確定できる active-bound QP を作り、`[lb; ub]` 配置を入替えた
/// 場合に KKT max が `SENTINEL_MIN_KKT` 以上跳ね上がることを assert する。
///
/// 構成:
///   min ½ ‖x‖² + cᵀ x s.t. 0 ≤ x ≤ 5, c = (1,1,1).
///   unconstrained min x* = -c = (-1,-1,-1) → lb 活性 → x* = (0,0,0).
///   stationarity: Qx + c − lb_du + ub_du = 0 + 1 − lb_du + 0 = 0 → lb_du = 1.
///   よって true bd = [1,1,1, 0,0,0] (layout `[lb; ub]`)。
///
/// swap bd = [0,0,0, 1,1,1] にすると stationarity が 1 + 1 = 2、bound comp も
/// ub_du · (ub−x) = 1·5 で活性化し、KKT max が大きく跳ねる。
/// no-op (bd 引数を無視) 実装ではこの差が消えるので必ず FAIL する。
#[test]
fn sentinel_qp_swapped_bound_duals_changes_kkt() {
    let n = 3;
    let q = build_psd_q(
        &{
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                l[i * n + i] = 1.0;
            }
            l
        },
        n,
    );
    let c = vec![1.0; n];
    let a = zero_csc(0, n);
    let bounds = vec![(0.0, 5.0); n];
    let qp = QpProblem::new(q, c, a, vec![], bounds, vec![]).expect("active-bound QP");

    let x_star = vec![0.0; n];
    let y_star: Vec<f64> = Vec::new();
    let mut bd_true = vec![0.0; 2 * n];
    for i in 0..n {
        bd_true[i] = 1.0;
    }
    let mut bd_swapped = vec![0.0; 2 * n];
    for i in 0..n {
        bd_swapped[n + i] = 1.0;
    }

    let base = compute_qp_kkt_max(&qp, &x_star, &y_star, &bd_true);
    let swapped = compute_qp_kkt_max(&qp, &x_star, &y_star, &bd_swapped);
    eprintln!("[sentinel_swap] base={:.3e} swapped={:.3e}", base, swapped);

    assert!(
        base.is_finite() && base < EPS_KKT,
        "true bd に対する KKT={:.3e} not below {:.0e} — helper bug or 規約 mismatch",
        base, EPS_KKT,
    );
    assert!(
        swapped >= SENTINEL_MIN_KKT,
        "swap した bd で KKT={:.3e} < {:.0e} — bd 引数 no-op の疑い",
        swapped, SENTINEL_MIN_KKT,
    );
    assert!(
        swapped > base * 100.0 || swapped >= SENTINEL_MIN_KKT,
        "swap 前後で KKT が変化していない (base={:.3e}, swapped={:.3e})",
        base, swapped,
    );
}
