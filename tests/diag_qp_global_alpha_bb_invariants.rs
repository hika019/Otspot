//! α-BB underestimator 不変式 sentinel。
//!
//! ## 目的
//! Phase 4 α-BB lower bound が「box 上で L(x) ≤ f(x)」という有効性条件を満たすこと、
//! および corner 上で L(x) = f(x) であることを **独立実装** で検証する。
//! sentinel が tautology (= 同じ実装を呼び合うだけ) に陥らないよう、Gershgorin α 計算と
//! L(x) 評価は src を呼ばず本ファイル内で完結する。
//!
//! ## 複数 data pattern
//! convex / light non-convex / strong non-convex / bilinear (zero diag) /
//! rank-deficient narrow / 高次元 concave (n=8) の 6 fixture × 多 seed sample。
//!
//! ## no-op 実証
//! `eval_L` の `2α·I` 項を恒等化 (= α=0 強制) すると corner 一致は保たれるが、
//! interior 沈み込みが消えるため `underestimator_strictly_below_objective_in_interior`
//! が FAIL (内部点で L=f になり strict-below assertion 違反)。
//! `feedback_sentinel_must_fail_under_noop` 準拠。

use solver::problem::ConstraintType;
use solver::qp::global::bound_alpha_bb::gershgorin_alpha as gershgorin_alpha_src;
use solver::qp::QpProblem;
use solver::sparse::CscMatrix;

/// L(x) と f(x) の同値判定許容 (mat_vec_mul + ULP).
const EQUAL_TOL: f64 = 1e-9;

/// 線形合同法 (LCG). seed-deterministic、test resource は std のみで完結。
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // upper 53 bits → [0, 1)
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn sample_in(&mut self, l: f64, u: f64) -> f64 {
        l + self.next_f64() * (u - l)
    }
}

/// 独立実装の Gershgorin α (= max_j (R_j − Q[j,j]) / 2)。
/// src/qp/global/bound_alpha_bb.rs と独立計算で交差検証する。
fn gershgorin_alpha_local(q: &CscMatrix) -> f64 {
    let n = q.nrows;
    let mut diag = vec![0.0_f64; n];
    let mut row_sum = vec![0.0_f64; n];
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            let v = q.values[k];
            if row == col {
                diag[col] = v;
            } else if row < col {
                let a = v.abs();
                row_sum[row] += a;
                row_sum[col] += a;
            }
        }
    }
    let mut delta = 0.0_f64;
    for j in 0..n {
        let lower = diag[j] - row_sum[j];
        if lower < 0.0 {
            delta = delta.max(-lower);
        }
    }
    0.5 * delta
}

/// f(x) = 0.5 x'Q x + c'x + obj_offset (Q は full-symmetric storage 規約に従う)。
fn eval_f(p: &QpProblem, x: &[f64]) -> f64 {
    let qx = p.q.mat_vec_mul(x).unwrap();
    let xqx: f64 = x.iter().zip(qx.iter()).map(|(a, b)| a * b).sum();
    let cx: f64 = x.iter().zip(p.c.iter()).map(|(a, b)| a * b).sum();
    0.5 * xqx + cx + p.obj_offset
}

/// α-BB underestimator L(x) を formula 直接で評価。
/// Source of truth: Maranas & Floudas (1995) "Finding all solutions of nonlinearly
/// constrained systems of equations" eq. (12)、
/// L(x) = f(x) + α Σ (x_i − l_i)(x_i − u_i) = 0.5 x'(Q + 2α I) x + (c − α(l+u))' x
///        + obj_offset + α Σ l_i u_i。
/// src 側 `build_convex_relaxation` を呼ばずに同 formula を独立実装して交差検証する。
fn eval_l(p: &QpProblem, x: &[f64], alpha: f64) -> f64 {
    let qx = p.q.mat_vec_mul(x).unwrap();
    let xqx: f64 = x.iter().zip(qx.iter()).map(|(a, b)| a * b).sum();
    let two_alpha_xx: f64 = 2.0 * alpha * x.iter().map(|v| v * v).sum::<f64>();
    let mut cmod_x: f64 = 0.0;
    let mut offset_extra: f64 = 0.0;
    for i in 0..p.num_vars {
        let (l, u) = p.bounds[i];
        cmod_x += (p.c[i] - alpha * (l + u)) * x[i];
        offset_extra += alpha * l * u;
    }
    0.5 * (xqx + two_alpha_xx) + cmod_x + p.obj_offset + offset_extra
}

// ---------------- fixtures ----------------

fn diag_q(diag: &[f64]) -> CscMatrix {
    let n = diag.len();
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).collect();
    CscMatrix::from_triplets(&rows, &cols, diag, n, n).unwrap()
}

/// strict 凸 (Q PSD)。α=0 が期待値。L=f が恒等で成立する境界 case。
fn convex_diag_2d() -> QpProblem {
    let q = diag_q(&[2.0, 2.0]);
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new(q, vec![-1.0, -1.0], a, vec![], vec![(0.0, 1.0); 2], vec![]).unwrap()
}

/// 1 negative eigenvalue (diag(-1, 3)). α=0.5 程度。
fn light_nonconvex_2d() -> QpProblem {
    let q = diag_q(&[-1.0, 3.0]);
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2], vec![]).unwrap()
}

/// 全成分 negative (diag(-2, -3, -1))。α=1.5 程度。
fn strong_nonconvex_3d() -> QpProblem {
    let q = diag_q(&[-2.0, -3.0, -1.0]);
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 3).unwrap();
    QpProblem::new(q, vec![0.0; 3], a, vec![], vec![(-1.0, 1.0); 3], vec![]).unwrap()
}

/// Zero diag indefinite Q = [[0,1],[1,0]] (純粋 bilinear)。raw Gershgorin 経路必須。
fn bilinear_zero_diag_2d() -> QpProblem {
    let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new(q, vec![0.0, 0.0], a, vec![], vec![(-1.0, 1.0); 2], vec![]).unwrap()
}

/// rank-deficient Q = [[-1,-1],[-1,-1]] (eigvals 0, -2)、box [0.4, 0.6]² (狭い)。
/// α は 1 (off-diag |1|, diag -1 → R_j-Q[j,j]=1-(-1)=2 → α=1)。
fn rank_deficient_narrow_2d() -> QpProblem {
    let q = CscMatrix::from_triplets(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[-1.0, -1.0, -1.0, -1.0],
        2,
        2,
    )
    .unwrap();
    let a = CscMatrix::from_triplets(&[], &[], &[], 0, 2).unwrap();
    QpProblem::new(q, vec![0.0, 0.0], a, vec![], vec![(0.4, 0.6); 2], vec![]).unwrap()
}

/// 8D concave (BB スケール検証用)。constraint Σ x_i ≤ 3, box [0,1]^8.
fn concave_8d_sumcap() -> QpProblem {
    let n = 8;
    let q = diag_q(&vec![-2.0_f64; n]);
    let a = CscMatrix::from_triplets(
        &vec![0_usize; n],
        &(0..n).collect::<Vec<_>>(),
        &vec![1.0_f64; n],
        1,
        n,
    )
    .unwrap();
    QpProblem::new(
        q,
        vec![0.0; n],
        a,
        vec![3.0],
        vec![(0.0, 1.0); n],
        vec![ConstraintType::Le],
    )
    .unwrap()
}

struct Fixture {
    label: &'static str,
    problem: QpProblem,
    /// 期待 α (Gershgorin 計算結果). convex case は 0、それ以外は > 0。
    expects_positive_alpha: bool,
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            label: "convex_diag_2d",
            problem: convex_diag_2d(),
            expects_positive_alpha: false,
        },
        Fixture {
            label: "light_nonconvex_2d",
            problem: light_nonconvex_2d(),
            expects_positive_alpha: true,
        },
        Fixture {
            label: "strong_nonconvex_3d",
            problem: strong_nonconvex_3d(),
            expects_positive_alpha: true,
        },
        Fixture {
            label: "bilinear_zero_diag_2d",
            problem: bilinear_zero_diag_2d(),
            expects_positive_alpha: true,
        },
        Fixture {
            label: "rank_deficient_narrow_2d",
            problem: rank_deficient_narrow_2d(),
            expects_positive_alpha: true,
        },
        Fixture {
            label: "concave_8d_sumcap",
            problem: concave_8d_sumcap(),
            expects_positive_alpha: true,
        },
    ]
}

// ---------------- tests ----------------

/// src `bound_alpha_bb::gershgorin_alpha` と本ファイル独立実装 `gershgorin_alpha_local`
/// が **全 fixture で bit-level に近い精度で一致** すること。
/// `gershgorin_alpha_local` 単体だと src 側が壊れても sentinel が気付けない
/// (Medium 4: 「invariants は src 非接続で tautology 化しうる」reviewer 指摘)。
/// 本テストで両者を直接照会し、src 側 regression を検出可能にする。
#[test]
fn gershgorin_alpha_src_matches_local_implementation() {
    for fx in fixtures() {
        let alpha_src = gershgorin_alpha_src(&fx.problem.q);
        let alpha_local = gershgorin_alpha_local(&fx.problem.q);
        assert!(
            (alpha_src - alpha_local).abs() < EQUAL_TOL,
            "{}: src α={alpha_src} vs local α={alpha_local} diverge by {:.3e}",
            fx.label,
            (alpha_src - alpha_local).abs(),
        );
    }
}

/// Gershgorin α が convex / non-convex で正しく分岐すること。
#[test]
fn gershgorin_alpha_sign_matches_convexity() {
    for fx in fixtures() {
        let a = gershgorin_alpha_local(&fx.problem.q);
        if fx.expects_positive_alpha {
            assert!(
                a > 0.0,
                "{}: non-convex fixture should yield positive α, got {a}",
                fx.label
            );
        } else {
            assert!(
                a == 0.0,
                "{}: convex fixture should yield α=0, got {a}",
                fx.label
            );
        }
    }
}

/// 全 fixture × 多 seed sample で `L(x) ≤ f(x)` (= 有効 lower bound condition)。
/// box 内部からの一様 sample で **statistical** に網羅。corner も別途 explicit に check。
///
/// **chain 完結**: local α だけでなく src `gershgorin_alpha` でも同じ assertion を回す。
/// src α が誤値 (例: 0.5× 過小評価) を返した場合は src α 経由の L(x) が一部 sample で
/// f(x) を超え、本 test 自身が FAIL する。これにより src α regression が sample test
/// レベルで直接検出される (= invariants の chain が src 側で切れない)。
#[test]
fn underestimator_dominates_objective_on_uniform_samples() {
    const N_SAMPLES_PER_SEED: usize = 30;
    const SEEDS: [u64; 3] = [1, 7, 42];
    for fx in fixtures() {
        let alpha_local = gershgorin_alpha_local(&fx.problem.q);
        let alpha_src = gershgorin_alpha_src(&fx.problem.q);
        for (alpha_label, alpha) in [("local", alpha_local), ("src", alpha_src)] {
            for seed in SEEDS {
                let mut rng = Lcg::new(seed);
                for _ in 0..N_SAMPLES_PER_SEED {
                    let x: Vec<f64> = fx
                        .problem
                        .bounds
                        .iter()
                        .map(|&(l, u)| rng.sample_in(l, u))
                        .collect();
                    let f = eval_f(&fx.problem, &x);
                    let l = eval_l(&fx.problem, &x, alpha);
                    let slack = f - l;
                    assert!(
                        slack >= -EQUAL_TOL,
                        "{} α[{alpha_label}] seed={seed}: L({x:?})={l:.6e} exceeded f={f:.6e} by {:.3e} (α={alpha})",
                        fx.label,
                        -slack,
                    );
                }
            }
        }
    }
}

/// すべての box corner で `L(x) = f(x)`. (x_i − l_i)(x_i − u_i)=0 ∀ i の境界条件証明。
/// n ≤ 3 fixture のみ (2^n = 8 corner 以内)。
#[test]
fn underestimator_equals_objective_at_corners() {
    for fx in fixtures() {
        let n = fx.problem.num_vars;
        if n > 3 {
            continue;
        }
        let alpha = gershgorin_alpha_local(&fx.problem.q);
        let n_corners = 1usize << n;
        for mask in 0..n_corners {
            let x: Vec<f64> = (0..n)
                .map(|i| {
                    let (l, u) = fx.problem.bounds[i];
                    if (mask >> i) & 1 == 1 { u } else { l }
                })
                .collect();
            let f = eval_f(&fx.problem, &x);
            let l = eval_l(&fx.problem, &x, alpha);
            assert!(
                (f - l).abs() < EQUAL_TOL,
                "{} corner {x:?}: L={l:.6e} vs f={f:.6e} (diff {:.3e}, α={alpha})",
                fx.label,
                (f - l).abs(),
            );
        }
    }
}

/// box 内部 (中心 + 中心近傍) で non-convex fixture は `L(x) < f(x)` (strict)。
/// no-op proof: α を 0 に置換 (= underestimator が原関数恒等になる) すると本 assertion FAIL。
#[test]
fn underestimator_strictly_below_objective_in_interior() {
    for fx in fixtures() {
        if !fx.expects_positive_alpha {
            continue;
        }
        let alpha = gershgorin_alpha_local(&fx.problem.q);
        // mid-box の x で評価 (interior 点)
        let x_mid: Vec<f64> = fx
            .problem
            .bounds
            .iter()
            .map(|&(l, u)| 0.5 * (l + u))
            .collect();
        let f_mid = eval_f(&fx.problem, &x_mid);
        let l_mid = eval_l(&fx.problem, &x_mid, alpha);
        // L(midpoint) - f(midpoint) = α Σ (mid-l)(mid-u) = -α Σ (width/2)² < 0
        assert!(
            l_mid < f_mid - EQUAL_TOL,
            "{}: interior midpoint should have L < f strictly. L={l_mid:.6e}, f={f_mid:.6e}, α={alpha}",
            fx.label,
        );
    }
}

/// α=0 (= 凸化恒等) を擬似的に当てたとき、L=f になり sentinel が FAIL する経路を
/// 直接確認する。`feedback_sentinel_must_fail_under_noop` の機械実証。
#[test]
fn noop_alpha_zero_collapses_underestimator_to_objective() {
    // strong_nonconvex_3d の midpoint で本来 L < f だが、α=0 強制で L = f を観測。
    let fx = strong_nonconvex_3d();
    let alpha_real = gershgorin_alpha_local(&fx.q);
    assert!(alpha_real > 0.0, "guard: fixture must be non-convex");
    let x_mid: Vec<f64> = fx.bounds.iter().map(|&(l, u)| 0.5 * (l + u)).collect();
    let f_mid = eval_f(&fx, &x_mid);
    let l_real = eval_l(&fx, &x_mid, alpha_real);
    let l_noop = eval_l(&fx, &x_mid, 0.0);
    assert!(
        l_real < f_mid - EQUAL_TOL,
        "with real α, L must strictly underestimate. L={l_real}, f={f_mid}"
    );
    assert!(
        (l_noop - f_mid).abs() < EQUAL_TOL,
        "with α=0 (no-op), L must collapse to f. L_noop={l_noop}, f={f_mid}"
    );
}
