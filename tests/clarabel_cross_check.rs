//! 外部ソルバ Clarabel との cross-check テスト。
//!
//! 目的: 本ソルバの solve_qp が外部 reference と一致するか確認。
//! 一致しない問題 (LISWET9 等) は parser/transform のバグの可能性。

use otspot::io::qps::parse_qps;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::options::SolverOptions;
use otspot::qp::solve_qp_with;
use otspot::QpProblem;
use clarabel::solver::{DefaultSolver, DefaultSettings, IPSolver};

#[path = "helpers/clarabel_utils.rs"]
mod clarabel_helper;
use clarabel_helper::{build_clarabel, compute_internal_obj, solve_clarabel};

/// より厳しい tol + iter で本当に「真の最適」が本ソルバに近いか確認する。
fn solve_clarabel_strict(prob: &otspot::QpProblem) -> Option<(f64, Vec<f64>, String)> {
    let (p, q, a, b, cones) = build_clarabel(prob);
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.tol_gap_abs = 1e-12;
    settings.tol_gap_rel = 1e-12;
    settings.tol_feas = 1e-12;
    settings.max_iter = 100_000;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).ok()?;
    solver.solve();
    Some((solver.info.cost_primal, solver.solution.x.clone(), format!("{:?} iters={}", solver.info.status, solver.info.iterations)))
}

#[test]
fn test_simple_2var_qp_matches_clarabel() {
    // min 0.5(x1^2 + x2^2)  s.t. x1 + x2 >= 1
    // 解: x1 = x2 = 0.5, obj = 0.25
    use otspot::sparse::CscMatrix;
    let q = CscMatrix::from_triplets(&[0,1], &[0,1], &[1.0, 1.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    // 本ソルバの new_all_le は Ax <= b。Ge 制約は -Ax <= -b で表現
    // x1 + x2 >= 1  →  -x1 - x2 <= -1
    let a = CscMatrix::from_triplets(&[0,0], &[0,1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let our = solve_qp_with(&prob, &opts);
    let our_obj = our.objective;

    let cl = solve_clarabel(&prob).expect("Clarabel solved");
    let cl_obj = cl.0;

    println!("simple 2var: ours obj={:.6e}, clarabel obj={:.6e}", our_obj, cl_obj);
    assert!((our_obj - cl_obj).abs() < 1e-4, "obj differs: ours={}, cl={}", our_obj, cl_obj);
    // 期待値 0.25
    assert!((our_obj - 0.25).abs() < 1e-4);
}

/// 深掘り: 同じ x で同じ obj が出るかの sanity check.
/// 出る → Q, c は parser で同じに読まれている。
/// 出ない → Q もしくは c が違う。
///
/// さらに feasibility cross check:
/// 本ソルバの x が Clarabel 用 b で feasible (Ax compared to b 本ソルバの constraint_types)
/// Clarabel の x が同じく feasible
fn deep_check(name: &str, path: &std::path::Path) -> bool {
    assert!(path.exists(), "{:?} ({}) not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path, name);
    let prob = parse_qps(path).expect("parse failed");

    println!("\n=== {} ===", name);
    println!("obj_offset={:.6e}, n={}, m={}", prob.obj_offset, prob.num_vars, prob.num_constraints);

    let n_eq = prob.constraint_types.iter().filter(|&&ct| ct == ConstraintType::Eq).count();
    let n_le = prob.constraint_types.iter().filter(|&&ct| ct == ConstraintType::Le).count();
    let n_ge = prob.constraint_types.iter().filter(|&&ct| ct == ConstraintType::Ge).count();
    let n_lb = prob.bounds.iter().filter(|&&(lb, _): &&(f64, f64)| lb.is_finite()).count();
    let n_ub = prob.bounds.iter().filter(|&&(_, ub): &&(f64, f64)| ub.is_finite()).count();
    println!("constraint mix: Eq={} Le={} Ge={}", n_eq, n_le, n_ge);
    println!("bounds: n_lb={}, n_ub={}", n_lb, n_ub);

    // Clarabel
    let cl = solve_clarabel(&prob);
    if cl.is_none() {
        println!("Clarabel failed to solve");
        return false;
    }
    let (cl_obj_clar, cl_x) = cl.unwrap();
    let cl_internal_obj = compute_internal_obj(&prob, &cl_x);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    let our = solve_qp_with(&prob, &opts);
    let our_internal_obj = compute_internal_obj(&prob, &our.solution);

    // Clarabel x で 本ソルバの式 → Q, c の解釈チェック
    println!("Clarabel: cost_primal={:.6e}, internal_obj(via our Q,c)={:.6e}", cl_obj_clar, cl_internal_obj);
    println!("Ours: objective={:.6e}, internal_obj={:.6e}", our.objective, our_internal_obj);

    // feasibility check: 本ソルバの x で 本ソルバの constraints
    let check_feasibility = |x: &[f64], label: &str| {
        if !x.is_empty() && prob.num_constraints > 0 {
            let ax = prob.a.mat_vec_mul(x).unwrap();
            let mut max_v = 0.0_f64;
            for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
                let v = match prob.constraint_types[i] {
                    ConstraintType::Eq => (ax_i - b_i).abs(),
                    ConstraintType::Ge => (b_i - ax_i).max(0.0),
                    _ => (ax_i - b_i).max(0.0),
                };
                max_v = max_v.max(v);
            }
            // bound check
            let mut bound_v = 0.0_f64;
            for (xi, &(lb, ub)) in x.iter().zip(prob.bounds.iter()) {
                if lb.is_finite() { bound_v = bound_v.max((lb - xi).max(0.0)); }
                if ub.is_finite() { bound_v = bound_v.max((xi - ub).max(0.0)); }
            }
            println!("{} feas: max_constr_violation={:.3e}, max_bound_violation={:.3e}",
                label, max_v, bound_v);
        }
    };
    check_feasibility(&cl_x, "Clarabel x");
    check_feasibility(&our.solution, "Ours x");

    // |x| 比較
    let cl_x_inf: f64 = cl_x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let our_x_inf: f64 = our.solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    println!("|x|_inf: ours={:.3e}, cl={:.3e}", our_x_inf, cl_x_inf);

    let diff = (our_internal_obj - cl_internal_obj).abs();
    let rel = diff / our_internal_obj.abs().max(cl_internal_obj.abs()).max(1.0);
    println!("internal obj diff={:.3e}, rel={:.3e}", diff, rel);

    let ok = rel < 1e-3;
    println!("MATCH: {}", ok);
    ok
}

/// Double-double (cancellation-safe) max Ge/Le/Eq 違反。LISWET の 2 階差分行は
/// f64 で x_i−2x_{i+1}+x_{i+2} に桁落ちするため DD で評価する。
fn max_violation_dd(prob: &QpProblem, x: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    let m = prob.num_constraints;
    let mut ax = vec![TwoFloat::from(0.0); m];
    for col in 0..prob.num_vars {
        let xc = x[col];
        for k in prob.a.col_ptr()[col]..prob.a.col_ptr()[col + 1] {
            let r = prob.a.row_ind()[k];
            ax[r] = ax[r] + TwoFloat::new_mul(prob.a.values()[k], xc);
        }
    }
    let mut mv = 0.0_f64;
    for i in 0..m {
        let axi = f64::from(ax[i]);
        let v = match prob.constraint_types[i] {
            ConstraintType::Ge => (prob.b[i] - axi).max(0.0),
            ConstraintType::Le => (axi - prob.b[i]).max(0.0),
            ConstraintType::Eq => (axi - prob.b[i]).abs(),
            _ => 0.0,
        };
        if v > mv { mv = v; }
    }
    mv
}

/// LISWET9/12 honest-behavior sentinel (旧 `test_liswet9_matches_clarabel` を置換)。
///
/// これらは「凸数列の錐」への射影で、制約正規行列が離散 biharmonic 作用素
/// (cond ≈ n⁴ ≈ 1e15, 最適 dual |y|∞ ≈ 1e5–1e6)。f64 内点法は誰も optimum を
/// tight に出せない: Clarabel ですら tol=1e-12 で AlmostSolved 止まり
/// (max constraint violation ≈ 2e-6, cf. `diag_liswet_basin.rs`)。よって
/// **clarabel の obj は certified optimum でなく、それに pin した assert は不健全**
/// (旧 test の「rel<1e-3 vs clarabel = solver bug」framing は誤り)。
///
/// 本 sentinel が固定する事実 = otspot の honest 契約:
///   (1) 劣最適点を **false `Optimal` として返さない** (status は Suboptimal/Timeout)。
///   (2) 返却点が **feasible** (DD max violation ≤ FEAS_TOL)。
/// load-bearing: solver が false Optimal を返す / feasibility が桁で退化すると FAIL。
/// (Clarabel との maxviol competitive 比較は `diag_liswet_basin.rs` の観測に残す。)
#[test]
fn liswet_family_honest_no_false_optimal() {
    // 観測値: ours maxviol_dd は ~1e-7〜1.6e-6 (timeout/初期点で変動)。
    // FEAS_TOL=1e-5 は実測 worst の ~6×、「feasible / 桁退化していない」を principled に判定。
    const FEAS_TOL: f64 = 1e-5;
    for name in ["LISWET9", "LISWET12"] {
        let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
        assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh", path);
        let prob = parse_qps(&path).expect("parse");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(10.0);
        let res = solve_qp_with(&prob, &opts);

        assert!(!res.solution.is_empty(), "{}: 空解 (solver が解を返さない)", name);
        // (1) false Optimal を返さない。f64 では tight な optimum に到達不能なので
        //     Optimal を主張したら誤判定 (honest 契約違反)。
        assert_ne!(
            res.status, SolveStatus::Optimal,
            "{}: f64 で certify 不能な ill-cond QP を Optimal と誤主張 (status={:?})",
            name, res.status
        );
        // (2) 返却点は feasible。
        let mv = max_violation_dd(&prob, &res.solution);
        eprintln!("{}: status={:?} maxviol_dd={:.3e}", name, res.status, mv);
        assert!(
            mv <= FEAS_TOL,
            "{}: 返却点が infeasible (DD maxviol={:.3e} > {:.0e})",
            name, mv, FEAS_TOL
        );
    }
}

/// LISWET9 / YAO で Clarabel を厳しく走らせて真の最適を確認
#[test]
#[ignore = "diag (~12s; Clarabel strict tol=1e-12 / max_iter=100k 出力のみ、assertion なし)"]
fn test_liswet9_yao_strict_clarabel() {
    for name in &["LISWET9", "YAO"] {
        let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
        assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path);
        let prob = parse_qps(&path).expect("parse");
        println!("\n=== {} (strict Clarabel) ===", name);
        let cl = solve_clarabel_strict(&prob);
        if let Some((cost, x, status)) = cl {
            let internal = compute_internal_obj(&prob, &x);
            let xinf: f64 = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            println!("  Clarabel strict: status={}, cost={:.6e}, internal={:.6e}, |x|_inf={:.3e}",
                status, cost, internal, xinf);
        } else {
            println!("  Clarabel strict failed");
        }
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(60.0);
        let our = solve_qp_with(&prob, &opts);
        let our_internal = compute_internal_obj(&prob, &our.solution);
        let our_xinf: f64 = our.solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        println!("  Ours: objective={:.6e}, internal={:.6e}, |x|_inf={:.3e}",
            our.objective, our_internal, our_xinf);
    }
}

/// 横展開: 様々な問題で internal obj が一致するか
#[test]
#[ignore = "diag (~44s; 35 Maros 問題で Clarabel cross-check、SUMMARY 出力のみ assertion なし)"]
fn test_multi_problems_match_clarabel() {
    let problems = [
        "HS21", "HS35", "HS35MOD", "HS268", "HS76",
        "AUG2D", "AUG3D", "AUG2DC", "AUG3DC",
        "QADLITTL", "QSC205", "QSCAGR7", "QSCFXM1",
        "DUALC1", "DUALC2", "DUALC5", "DUALC8",
        "GENHS28", "ZECEVIC2", "S268", "TAME",
        "STCQP1", "STCQP2", "STADAT1",
        "PRIMAL1", "PRIMAL2", "PRIMAL3", "PRIMAL4",
        "QPCBOEI1", "QPCSTAIR",
        "LISWET1", "LISWET9",
        "YAO",
        "QSHIP04L", "QSHARE1B",
    ];
    let mut results = Vec::new();
    for name in &problems {
        let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
        assert!(path.exists(), "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行", path);
        let ok = std::panic::catch_unwind(|| deep_check(name, &path)).unwrap_or(false);
        results.push((name.to_string(), if ok { "MATCH".to_string() } else { "MISMATCH".to_string() }));
    }
    println!("\n========= SUMMARY =========");
    for (n, r) in &results {
        println!("{:20} {}", n, r);
    }
    let mismatches: Vec<&String> = results.iter().filter(|(_, r)| r == "MISMATCH").map(|(n, _)| n).collect();
    println!("\nTotal mismatches: {}", mismatches.len());
    for n in &mismatches { println!("  - {}", n); }
}
