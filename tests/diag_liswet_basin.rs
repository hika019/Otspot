//! LISWET9/12 regression guard: ill-conditioned QPs against the exact
//! certified baselines. Old premise ("never claim Optimal; f64 insufficient
//! to certify") was invalidated by this PR: the obj constant convention
//! (92ce6cc2) and duality-gap constant exclusion (2938b3c5) made LISWET9/12
//! genuinely certifiable, and a CI-fast-runner Optimal verdict of 1736.927
//! was 3e-7 off the exact certified 1736.927426116 — i.e. correct. The
//! sentinel now pins: Optimal must match the certified baseline within band,
//! a non-Optimal completion must be an allowed non-claiming status, and a
//! returned x must be feasible either way. False Optimal stays forbidden.
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop)]

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::ConstraintType;
use otspot::qp::solve_qp_with;
use otspot::{QpProblem, QpWarmStart};

use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver};

#[path = "helpers/clarabel_utils.rs"]
mod clarabel_helper;
use clarabel_helper::build_clarabel;

const STRICT_TOL: f64 = 1e-12;
const STRICT_MAX_ITER: u32 = 100_000;

fn solve_clarabel_tol(prob: &QpProblem, tol: f64, max_iter: u32) -> (Vec<f64>, Vec<f64>, String) {
    let (p, q, a, b, cones) = build_clarabel(prob);
    let mut s = DefaultSettings::default();
    s.verbose = false;
    s.tol_gap_abs = tol;
    s.tol_gap_rel = tol;
    s.tol_feas = tol;
    s.max_iter = max_iter;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, s).expect("clarabel setup");
    solver.solve();
    (
        solver.solution.x.clone(),
        solver.solution.z.clone(),
        format!("{:?}", solver.info.status),
    )
}

/// clarabel z (cone 順) を元制約順の y にマップ。Ge は A_clar=-A 変換で y=z。
fn map_clarabel_y(prob: &QpProblem, z: &[f64]) -> Vec<f64> {
    let m = prob.num_constraints;
    let mut row_ord: Vec<(usize, ConstraintType)> =
        (0..m).map(|i| (i, prob.constraint_types[i])).collect();
    row_ord.sort_by_key(|&(_, ct)| match ct {
        ConstraintType::Eq => 0,
        _ => 1,
    });
    let mut row_pos = vec![0_usize; m];
    for (new_row, &(orig_row, _)) in row_ord.iter().enumerate() {
        row_pos[orig_row] = new_row;
    }
    (0..m)
        .map(|i| {
            if row_pos[i] < z.len() {
                z[row_pos[i]]
            } else {
                0.0
            }
        })
        .collect()
}

fn obj_internal(prob: &QpProblem, x: &[f64]) -> f64 {
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    0.5 * qx.iter().zip(x.iter()).map(|(&q, &x)| q * x).sum::<f64>()
        + prob
            .c
            .iter()
            .zip(x.iter())
            .map(|(&c, &x)| c * x)
            .sum::<f64>()
}

/// Double-double Ax (cancellation-safe) → max Ge violation (−min slack).
fn max_viol_dd(prob: &QpProblem, x: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    let m = prob.num_constraints;
    let mut ax = vec![TwoFloat::from(0.0); m];
    for col in 0..prob.num_vars {
        let xc = x[col];
        for k in prob.a.col_ptr()[col]..prob.a.col_ptr()[col + 1] {
            let r = prob.a.row_ind()[k];
            ax[r] += TwoFloat::new_mul(prob.a.values()[k], xc);
        }
    }
    let mut mv = 0.0_f64;
    for i in 0..m {
        let axi = f64::from(ax[i]);
        let viol = match prob.constraint_types[i] {
            ConstraintType::Ge => (prob.b[i] - axi).max(0.0),
            ConstraintType::Le => (axi - prob.b[i]).max(0.0),
            ConstraintType::Eq => (axi - prob.b[i]).abs(),
            _ => 0.0,
        };
        if viol > mv {
            mv = viol;
        }
    }
    mv
}

/// row slack (Ge: ax-b, Le: b-ax, Eq: -(|ax-b|))。負 = 違反。
fn row_slacks(prob: &QpProblem, x: &[f64]) -> Vec<f64> {
    let ax = prob.a.mat_vec_mul(x).expect("Ax");
    (0..prob.num_constraints)
        .map(|i| match prob.constraint_types[i] {
            ConstraintType::Ge => ax[i] - prob.b[i],
            ConstraintType::Le => prob.b[i] - ax[i],
            ConstraintType::Eq => -(ax[i] - prob.b[i]).abs(),
            _ => 0.0,
        })
        .collect()
}

/// y ← A·(Aᵀ·v) (banded matvec, O(nnz))。
fn aat_mul(prob: &QpProblem, v: &[f64]) -> Vec<f64> {
    let n = prob.num_vars;
    let m = prob.num_constraints;
    // w = Aᵀ v  (length n)
    let mut w = vec![0.0_f64; n];
    for col in 0..n {
        let mut acc = 0.0;
        for k in prob.a.col_ptr()[col]..prob.a.col_ptr()[col + 1] {
            acc += prob.a.values()[k] * v[prob.a.row_ind()[k]];
        }
        w[col] = acc;
    }
    // y = A w  (length m)
    let mut y = vec![0.0_f64; m];
    for col in 0..n {
        let wc = w[col];
        for k in prob.a.col_ptr()[col]..prob.a.col_ptr()[col + 1] {
            y[prob.a.row_ind()[k]] += prob.a.values()[k] * wc;
        }
    }
    y
}

/// power iteration で AAᵀ の λmax, λmin を推定し cond を返す。
fn estimate_cond_aat(prob: &QpProblem) -> (f64, f64, f64) {
    let m = prob.num_constraints;
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    // λmax
    let mut v: Vec<f64> = (0..m)
        .map(|i| ((i * 2654435761) % 1000) as f64 / 500.0 - 1.0)
        .collect();
    let mut nv = norm(&v);
    for x in v.iter_mut() {
        *x /= nv;
    }
    let mut lmax = 0.0;
    for _ in 0..200 {
        let w = aat_mul(prob, &v);
        lmax = v.iter().zip(&w).map(|(&a, &b)| a * b).sum();
        nv = norm(&w);
        if nv == 0.0 {
            break;
        }
        for (vi, &wi) in v.iter_mut().zip(&w) {
            *vi = wi / nv;
        }
    }
    // λmin via shifted power iteration on (lmax·I - AAᵀ)
    let mut u: Vec<f64> = (0..m)
        .map(|i| ((i * 40503 + 7) % 997) as f64 / 498.0 - 1.0)
        .collect();
    nv = norm(&u);
    for x in u.iter_mut() {
        *x /= nv;
    }
    let mut shifted_max = 0.0;
    for _ in 0..400 {
        let aatu = aat_mul(prob, &u);
        let w: Vec<f64> = u
            .iter()
            .zip(&aatu)
            .map(|(&ui, &ai)| lmax * ui - ai)
            .collect();
        shifted_max = u.iter().zip(&w).map(|(&a, &b)| a * b).sum();
        nv = norm(&w);
        if nv == 0.0 {
            break;
        }
        for (ui, &wi) in u.iter_mut().zip(&w) {
            *ui = wi / nv;
        }
    }
    let lmin = lmax - shifted_max;
    (lmax, lmin, lmax / lmin.max(1e-300))
}

/// Exact certified baselines from the CSV `exact_enclosure_v1` claims,
/// machine-verified by the LISWET rational proofs (see
/// data/baseline_objectives/proofs/liswet/manifest.json) — an independent
/// oracle, not a snapshot of this solver's own output.
/// 1.736927426116e3 / 1.963251260895e3.
const LISWET9_CERT_OBJ: f64 = 1.963_251_260_895e3;
const LISWET12_CERT_OBJ: f64 = 1.736_927_426_116e3;
/// Relative band for an Optimal claim. The genuine Optimal verdict measured
/// in CI sat 3e-7 from the baseline; 1e-5 keeps >30x headroom while still
/// rejecting the pre-convention values (~1.45 away, see unit tests).
const OPT_OBJ_REL_BAND: f64 = 1e-5;
/// Max doubled-double constraint violation for any returned x, regardless of
/// status (inherited from the old sentinel's 1e-5 threshold).
const DD_MAX: f64 = 1e-5;

/// The load-bearing decision, pure so it can be pinned by synthetic inputs.
/// `dd_viol` is None when no x was returned.
fn sentinel_check(
    claims_optimal: bool,
    objective: f64,
    cert_obj: f64,
    dd_viol: Option<f64>,
) -> Result<(), String> {
    if let Some(v) = dd_viol {
        if v > DD_MAX {
            return Err(format!("returned x is infeasible (DD maxviol={v:.3e})"));
        }
    }
    if claims_optimal {
        let rel = (objective - cert_obj).abs() / cert_obj.abs().max(1.0);
        if rel > OPT_OBJ_REL_BAND {
            return Err(format!(
                "false Optimal: obj={objective:.6e} is rel {rel:.3e} off the certified \
                 baseline {cert_obj:.6e} (band {OPT_OBJ_REL_BAND:.0e})"
            ));
        }
    }
    Ok(())
}

/// Committed budget stays at the historical 60s: a fast runner certifies
/// within it (Optimal branch green), a slow one completes non-deterministically
/// (Timeout branch green) — both branches green by construction.
/// LISWET_BUDGET_SECS only exists to exercise the Optimal branch on local
/// hardware (natural completion measured 163s; the bench repro uses 165).
fn budget_secs() -> f64 {
    match std::env::var("LISWET_BUDGET_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(s) if s > 0.0 => {
            eprintln!("LISWET_BUDGET_SECS override: {s}s");
            s
        }
        _ => 60.0,
    }
}

struct Outcome {
    claims_optimal: bool,
    objective: f64,
    dd_ours: Option<f64>,
    status_debug: String,
    dd_ref: f64,
}

fn measure(name: &str) -> Outcome {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    assert!(path.exists(), "{:?} not found", path);
    let prob = parse_qps(&path).expect("parse");

    let (x_ref, z_ref, st_strict) = solve_clarabel_tol(&prob, STRICT_TOL, STRICT_MAX_ITER);
    let (x_def, _z_def, st_def) = solve_clarabel_tol(&prob, 1e-8, 5_000);
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(budget_secs());
    let res = solve_qp_with(&prob, &opts);
    let x = &res.solution;

    // warm-start from clarabel (x, y) → basin path-dependence の測定。
    let y_ws = map_clarabel_y(&prob, &z_ref);
    let mu_ws = {
        let ax = prob.a.mat_vec_mul(&x_ref).unwrap();
        let mut acc = 0.0;
        let mut cnt = 0usize;
        for i in 0..prob.num_constraints {
            let slack = match prob.constraint_types[i] {
                ConstraintType::Ge => ax[i] - prob.b[i],
                ConstraintType::Le => prob.b[i] - ax[i],
                _ => 0.0,
            };
            acc += (slack * y_ws[i]).abs();
            cnt += 1;
        }
        if cnt > 0 {
            (acc / cnt as f64).max(1e-8)
        } else {
            1e-8
        }
    };
    let mut opts_ws = SolverOptions::default();
    opts_ws.timeout_secs = Some(budget_secs());
    opts_ws.warm_start_qp = Some(QpWarmStart::new(x_ref.clone(), y_ws, mu_ws));
    let res_ws = solve_qp_with(&prob, &opts_ws);
    let obj_ws = obj_internal(&prob, &res_ws.solution);

    let obj_ref = obj_internal(&prob, &x_ref);
    let obj_def = obj_internal(&prob, &x_def);
    let obj_ours = obj_internal(&prob, x);
    let rel = (obj_ref - obj_ours).abs() / obj_ref.abs().max(obj_ours.abs()).max(1.0);

    // max constraint violation (min over rows of slack, negated when negative)
    let max_viol = |xv: &[f64]| -> f64 {
        row_slacks(&prob, xv)
            .iter()
            .map(|&v| (-v).max(0.0))
            .fold(0.0_f64, f64::max)
    };
    eprintln!(
        "\n===== {} (n={} m={}) =====",
        name, prob.num_vars, prob.num_constraints
    );
    eprintln!("clarabel: strict(1e-12)={} obj={:.6e} maxviol_f64={:.3e} maxviol_dd={:.3e} | default(1e-8)={} obj={:.6e}",
        st_strict, obj_ref, max_viol(&x_ref), max_viol_dd(&prob, &x_ref), st_def, obj_def);
    eprintln!(
        "ours: maxviol_f64={:.3e} maxviol_dd={:.3e}",
        max_viol(x),
        max_viol_dd(&prob, x)
    );
    eprintln!("warm-start-from-clarabel: status={:?} obj={:.6e} maxviol_dd={:.3e} (basin path-dependence test)",
        res_ws.status, obj_ws, max_viol_dd(&prob, &res_ws.solution));
    let y_inf = res
        .dual_solution
        .iter()
        .fold(0.0_f64, |a, &v| a.max(v.abs()));
    // λmax(AAᵀ) は power-iter で確実、λmin はクラスタで収束遅く下限のみ → cond は λmax/λmin_est で
    // 下限扱い (真の cond は biharmonic 解析より ~n⁴ で更に大)。
    let (lmax, lmin_est, cond_lb) = estimate_cond_aat(&prob);
    eprintln!(
        "ours |y|inf={:.3e} | AAᵀ λmax={:.3e} (analytic≈16) λmin_powiter≳{:.3e} cond_lb≳{:.2e}",
        y_inf, lmax, lmin_est, cond_lb
    );

    let sl_ref = row_slacks(&prob, &x_ref);
    let sl_ours = row_slacks(&prob, x);
    const ACT: f64 = 1e-6;
    let act_ref: Vec<bool> = sl_ref.iter().map(|&v| v.abs() <= ACT).collect();
    let act_ours: Vec<bool> = sl_ours.iter().map(|&v| v.abs() <= ACT).collect();
    let n_aref = act_ref.iter().filter(|&&b| b).count();
    let n_aours = act_ours.iter().filter(|&&b| b).count();
    let both = act_ref
        .iter()
        .zip(&act_ours)
        .filter(|&(&r, &o)| r && o)
        .count();
    let jac = both as f64 / (n_aref + n_aours - both).max(1) as f64;

    // 違反集計 (slack < -ACT)
    let viol_ours: Vec<(usize, f64)> = sl_ours
        .iter()
        .enumerate()
        .filter(|(_, &v)| v < -ACT)
        .map(|(i, &v)| (i, v))
        .collect();
    let viol_ref: usize = sl_ref.iter().filter(|&&v| v < -ACT).count();
    let max_viol_ours = viol_ours.iter().map(|&(_, v)| -v).fold(0.0_f64, f64::max);

    // x 距離
    let (mut max_dx, mut sum_dx, mut argmax) = (0.0_f64, 0.0_f64, 0usize);
    for (j, (&a, &b)) in x_ref.iter().zip(x.iter()).enumerate() {
        let d = (a - b).abs();
        sum_dx += d;
        if d > max_dx {
            max_dx = d;
            argmax = j;
        }
    }
    let mean_dx = sum_dx / x_ref.len() as f64;
    let xref_inf = x_ref.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let xours_inf = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));

    eprintln!("status_ours={:?} iters={}", res.status, res.iterations);
    eprintln!(
        "obj: ref={:.6e} ours={:.6e} rel_diff={:.3e} (ours<ref ⇒ infeasible-lower)",
        obj_ref, obj_ours, rel
    );
    eprintln!(
        "active: ref={} ours={} both={} jaccard={:.4}",
        n_aref, n_aours, both, jac
    );
    eprintln!(
        "viol(slack<-{:.0e}): ref={} ours={} max_viol_ours={:.3e}",
        ACT,
        viol_ref,
        viol_ours.len(),
        max_viol_ours
    );
    eprintln!(
        "|x_ref - x_ours|: max={:.3e}@{} mean={:.3e} | |x_ref|inf={:.3e} |x_ours|inf={:.3e}",
        max_dx, argmax, mean_dx, xref_inf, xours_inf
    );
    // top-5 violated rows detail
    let mut vsorted = viol_ours.clone();
    vsorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ax_ours = prob.a.mat_vec_mul(x).expect("Ax");
    eprintln!("top violated rows (ours): row | slack | ax | b | type");
    for &(i, v) in vsorted.iter().take(5) {
        eprintln!(
            "  {:5} | {:+.3e} | {:+.3e} | {:+.3e} | {:?}",
            i, v, ax_ours[i], prob.b[i], prob.constraint_types[i]
        );
    }

    // 旧センチネルはここで (1) never Optimal (2) feasible (3) clarabel×10 を
    // 主張していた。(1) は 92ce6cc2 (obj 定数規約・HiGHS 準拠) と 2938b3c5
    // (duality gap 定数除外) で前提が陳腐化 — LISWET9/12 は認証可能になり、
    // CI 高速ランナーでの Optimal (obj_full=1736.927、認証値と rel 3e-7) は
    // 正解だった。(3) の参照 clarabel x_ref は解測定で DD viol 1.32e-6 かつ
    // obj が真の最小より ~1500 低い infeasible-lower と判明、比較は診断のみ
    // に降格 (feasibility 非劣位の主張だけが load-bearing だった)。
    // load-bearing 化は sentinel_check 側で: 誤 Optimal (band 外) と非可行 x は
    // 従来以上に明確に赤、非決定完了 (Timeout 等) は両分岐緑で決定的。
    let ours_mv = max_viol_dd(&prob, x);
    let cl_mv = max_viol_dd(&prob, &x_ref);
    eprintln!(
        "clarabel-compare (diagnostic only; x_ref measured infeasible-lower): ours DD={ours_mv:.3e} vs clarabel DD={cl_mv:.3e} (old assert: ours <= 10x clarabel)"
    );
    Outcome {
        claims_optimal: res.status == otspot::problem::SolveStatus::Optimal,
        objective: res.objective,
        dd_ours: if x.is_empty() { None } else { Some(ours_mv) },
        status_debug: format!("{:?}", res.status),
        dd_ref: cl_mv,
    }
}

/// Full sentinel decision as a `Result` whose `Err` carries the evidence
/// (status / objective / DD values). The heavy test bodies keep the load-
/// bearing `assert!` so the verdict is asserted where nextest reports it
/// (and where the observation-only audit can see it), not hidden in a
/// helper. Problem-name attribution lives at the assert sites.
fn sentinel_verdict(cert_obj: f64, out: &Outcome) -> Result<(), String> {
    // Non-claiming completion statuses that are honest for this convex,
    // bounded, feasible problem under a budget. Infeasible/Unbounded/
    // NumericalError etc. would be wrong-direction claims for this class and
    // stay red; LocallyOptimal is excluded (convex path — a "local" claim
    // would be a status misuse).
    const ALLOWED: [&str; 5] = [
        "Timeout",
        "Stalled",
        "MaxIterations",
        "SuboptimalSolution",
        "FeasiblePoint",
    ];
    if !out.claims_optimal && !ALLOWED.contains(&out.status_debug.as_str()) {
        return Err(format!(
            "completion {} is neither an in-band Optimal nor an allowed \
             non-claiming status {ALLOWED:?}",
            out.status_debug
        ));
    }
    sentinel_check(out.claims_optimal, out.objective, cert_obj, out.dd_ours)
}

/// Post-assert diagnostic only (the assert in the test bodies is the
/// load-bearing part).
fn report_green(name: &str, out: &Outcome) {
    eprintln!(
        "{name}: SENTINEL GREEN status={} objective={:.6e} dd={:?} (clarabel dd {:.3e} is diagnostic only)",
        out.status_debug, out.objective, out.dd_ours, out.dd_ref
    );
}

/// ~180s under CPU contention. The committed 60s budget keeps both branches
/// (in-band Optimal / Timeout) green deterministically; module docs record why
/// the old never-Optimal premise retired (92ce6cc2 / 2938b3c5).
#[test]
#[ignore = "heavy: 無負荷でも clarabel 復習 + warm-start 込みで数分 (main solve は 60s budget、local 自然完走 163s → Timeout 分岐で緑)。Optimal 分岐の実測は LISWET_BUDGET_SECS=165。never-Optimal 前提は 92ce6cc2/2938b3c5 で退役、センチネルは認証 baseline 突合へ刷新済み"]
fn diag_liswet9_basin_sentinel() {
    let out = measure("LISWET9");
    let verdict = sentinel_verdict(LISWET9_CERT_OBJ, &out);
    assert!(
        verdict.is_ok(),
        "LISWET9 sentinel violated: {}",
        verdict.unwrap_err()
    );
    report_green("LISWET9", &out);
}

#[test]
#[ignore = "heavy: 同上 (LISWET9 と同一設計)。CI 着弾はこの問題の Optimal 誤発火 — 認証値と rel 3e-7 で一致する正解 Optimal を never-Optimal が罰していた。刷新後は Optimal=band 突合 / 非決定完了=許容集合 で両分岐緑が決定的"]
fn diag_liswet12_basin_sentinel() {
    let out = measure("LISWET12");
    let verdict = sentinel_verdict(LISWET12_CERT_OBJ, &out);
    assert!(
        verdict.is_ok(),
        "LISWET12 sentinel violated: {}",
        verdict.unwrap_err()
    );
    report_green("LISWET12", &out);
}

// ---- pure sentinel logic pinned by synthetic inputs (no data, no solver) ----

#[test]
fn sentinel_accepts_true_optimal() {
    // CI 実測の正解 Optimal (認証値に rel 3e-7) を再現した合成入力。
    let obj = LISWET12_CERT_OBJ * (1.0 + 3e-7);
    sentinel_check(true, obj, LISWET12_CERT_OBJ, Some(1e-9))
        .expect("in-band Optimal must be green");
}

#[test]
fn sentinel_rejects_pre_convention_optimal() {
    // 定数規約変更前の値 (~-788 系) を Optimal と主張する合成入力。band 外
    // (rel ≈1.45) で必ず赤 — 実測失敗 run の正確な値ではなく、帯を 5 桁
    // 外れることが load-bearing なので値の置き換えでも同じ分岐に入る。
    let err = sentinel_check(true, -788.852898, LISWET12_CERT_OBJ, Some(1e-9))
        .expect_err("pre-convention obj claimed as Optimal must be red");
    assert!(err.contains("false Optimal"), "{err}");
}

#[test]
fn sentinel_band_edges_absolute_probes() {
    // band 1e-5 の絶対リテラル probe: rel 9e-6 で緑、1.1e-5 で赤。
    sentinel_check(
        true,
        LISWET12_CERT_OBJ * (1.0 + 9e-6),
        LISWET12_CERT_OBJ,
        Some(0.0),
    )
    .expect("rel 9e-6 inside band");
    let err = sentinel_check(
        true,
        LISWET12_CERT_OBJ * (1.0 + 1.1e-5),
        LISWET12_CERT_OBJ,
        Some(0.0),
    )
    .expect_err("rel 1.1e-5 outside band");
    assert!(err.contains("false Optimal"), "{err}");
}

#[test]
fn sentinel_nonoptimal_completion_ignores_objective() {
    // 非決定完了の診断 iterate は目的値に無意味な値を取り得る (1e99 相当)。
    // 「主張しない」ことだけが load-bearing なので band 判定は走らない。
    sentinel_check(false, 1e99, LISWET12_CERT_OBJ, Some(5e-6))
        .expect("non-claiming completion must not be band-checked");
}

#[test]
fn sentinel_rejects_infeasible_solution_any_status() {
    let err = sentinel_check(true, LISWET12_CERT_OBJ, LISWET12_CERT_OBJ, Some(2e-5))
        .expect_err("DD viol 2e-5 must be red");
    assert!(err.contains("infeasible"), "{err}");
    let err = sentinel_check(false, 0.0, LISWET12_CERT_OBJ, Some(2e-5))
        .expect_err("infeasible x is forbidden for non-claiming completions too");
    assert!(err.contains("infeasible"), "{err}");
}
