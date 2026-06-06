//! 外部ソルバ Clarabel との cross-check テスト。
//!
//! 目的: 本ソルバの solve_qp が外部 reference と一致するか確認。
//! 一致しない問題 (LISWET9 等) は parser/transform のバグの可能性。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::qp::solve_qp_with;
use otspot::QpProblem;

#[path = "helpers/clarabel_utils.rs"]
mod clarabel_helper;
use clarabel_helper::solve_clarabel;

#[test]
fn test_simple_2var_qp_matches_clarabel() {
    // min 0.5(x1^2 + x2^2)  s.t. x1 + x2 >= 1
    // 解: x1 = x2 = 0.5, obj = 0.25
    use otspot::sparse::CscMatrix;
    let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let c = vec![0.0, 0.0];
    // 本ソルバの new_all_le は Ax <= b。Ge 制約は -Ax <= -b で表現
    // x1 + x2 >= 1  →  -x1 - x2 <= -1
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
    let b = vec![-1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
    let prob = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    let our = solve_qp_with(&prob, &opts);
    let our_obj = our.objective;

    let cl = solve_clarabel(&prob).expect("Clarabel solved");
    let cl_obj = cl.0;

    println!(
        "simple 2var: ours obj={:.6e}, clarabel obj={:.6e}",
        our_obj, cl_obj
    );
    assert!(
        (our_obj - cl_obj).abs() < 1e-4,
        "obj differs: ours={}, cl={}",
        our_obj,
        cl_obj
    );
    // 期待値 0.25
    assert!((our_obj - 0.25).abs() < 1e-4);
}

/// Double-double (cancellation-safe) max Ge/Le/Eq 違反。LISWET の 2 階差分行は
/// f64 で x_i−2x_{i+1}+x_{i+2} に桁落ちするため DD で評価する。
fn max_violation_dd(prob: &QpProblem, x: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    let m = prob.num_constraints;
    let mut ax = vec![TwoFloat::from(0.0); m];
    for (col, &xc) in x.iter().enumerate().take(prob.num_vars) {
        for k in prob.a.col_ptr()[col]..prob.a.col_ptr()[col + 1] {
            let r = prob.a.row_ind()[k];
            ax[r] += TwoFloat::new_mul(prob.a.values()[k], xc);
        }
    }
    let mut mv = 0.0_f64;
    for (i, ax_i) in ax.iter().enumerate() {
        let axi = f64::from(ax_i);
        let v = match prob.constraint_types[i] {
            ConstraintType::Ge => (prob.b[i] - axi).max(0.0),
            ConstraintType::Le => (axi - prob.b[i]).max(0.0),
            ConstraintType::Eq => (axi - prob.b[i]).abs(),
            _ => 0.0,
        };
        if v > mv {
            mv = v;
        }
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
        assert!(
            path.exists(),
            "{:?} not found — bench data 未配置。scripts/maros_meszaros_download.sh",
            path
        );
        let prob = parse_qps(&path).expect("parse");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(10.0);
        let res = solve_qp_with(&prob, &opts);

        assert!(
            !res.solution.is_empty(),
            "{}: 空解 (solver が解を返さない)",
            name
        );
        // (1) false Optimal を返さない。f64 では tight な optimum に到達不能なので
        //     Optimal を主張したら誤判定 (honest 契約違反)。
        assert_ne!(
            res.status,
            SolveStatus::Optimal,
            "{}: f64 で certify 不能な ill-cond QP を Optimal と誤主張 (status={:?})",
            name,
            res.status
        );
        // (2) 返却点は feasible。
        let mv = max_violation_dd(&prob, &res.solution);
        eprintln!("{}: status={:?} maxviol_dd={:.3e}", name, res.status, mv);
        assert!(
            mv <= FEAS_TOL,
            "{}: 返却点が infeasible (DD maxviol={:.3e} > {:.0e})",
            name,
            mv,
            FEAS_TOL
        );
    }
}
