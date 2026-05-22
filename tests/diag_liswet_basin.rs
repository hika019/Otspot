//! LISWET9/12 wrong-basin 真因切り分け diag。
//! ours vs Clarabel strict で per-row primal violation 局在 / active-set / x 距離を出力。
//! (assertion なし、観測専用)

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::ConstraintType;
use otspot::qp::solve_qp_with;
use otspot::{QpProblem, QpWarmStart};

use clarabel::algebra::CscMatrix as ClCsc;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SupportedConeT};

const STRICT_TOL: f64 = 1e-12;
const STRICT_MAX_ITER: u32 = 100_000;

fn build_clarabel(
    prob: &QpProblem,
) -> (ClCsc<f64>, Vec<f64>, ClCsc<f64>, Vec<f64>, Vec<SupportedConeT<f64>>) {
    let n = prob.num_vars;
    let m = prob.num_constraints;
    let n_lb = prob.bounds.iter().filter(|&&(lb, _): &&(f64, f64)| lb.is_finite()).count();
    let n_ub = prob.bounds.iter().filter(|&&(_, ub): &&(f64, f64)| ub.is_finite()).count();
    let mut row_ord: Vec<(usize, ConstraintType)> =
        (0..m).map(|i| (i, prob.constraint_types[i])).collect();
    row_ord.sort_by_key(|&(_, ct)| match ct { ConstraintType::Eq => 0, _ => 1 });
    let n_eq = row_ord.iter().filter(|&&(_, ct)| ct == ConstraintType::Eq).count();
    let n_le_ge = m - n_eq;
    let mut row_pos = vec![0_usize; m];
    for (new_row, &(orig_row, _)) in row_ord.iter().enumerate() {
        row_pos[orig_row] = new_row;
    }
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let total_rows = m + n_lb + n_ub;
    let mut b_clar = vec![0.0_f64; total_rows];
    for j in 0..n {
        for ptr in prob.a.col_ptr[j]..prob.a.col_ptr[j + 1] {
            let orig_row = prob.a.row_ind[ptr];
            let val = prob.a.values[ptr];
            let new_row = row_pos[orig_row];
            match prob.constraint_types[orig_row] {
                ConstraintType::Ge => triplets.push((new_row, j, -val)),
                _ => triplets.push((new_row, j, val)),
            }
        }
    }
    for (orig_row, ct) in prob.constraint_types.iter().enumerate() {
        let new_row = row_pos[orig_row];
        match ct {
            ConstraintType::Ge => b_clar[new_row] = -prob.b[orig_row],
            _ => b_clar[new_row] = prob.b[orig_row],
        }
    }
    let mut bound_row = m;
    for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() { triplets.push((bound_row, j, -1.0)); b_clar[bound_row] = -lb; bound_row += 1; }
    }
    for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
        if ub.is_finite() { triplets.push((bound_row, j, 1.0)); b_clar[bound_row] = ub; bound_row += 1; }
    }
    triplets.sort_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &triplets { col_ptr[c + 1] += 1; }
    for j in 0..n { col_ptr[j + 1] += col_ptr[j]; }
    let mut row_ind = vec![0_usize; triplets.len()];
    let mut values = vec![0.0_f64; triplets.len()];
    let mut cursor = col_ptr.clone();
    for &(r, c, v) in &triplets { let pos = cursor[c]; row_ind[pos] = r; values[pos] = v; cursor[c] += 1; }
    let a_clar = ClCsc::new(total_rows, n, col_ptr, row_ind, values);

    let mut p_triplets: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..n {
        for ptr in prob.q.col_ptr[j]..prob.q.col_ptr[j + 1] {
            let i = prob.q.row_ind[ptr];
            if i <= j { p_triplets.push((i, j, prob.q.values[ptr])); }
        }
    }
    p_triplets.sort_by_key(|&(r, c, _)| (c, r));
    let mut p_col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &p_triplets { p_col_ptr[c + 1] += 1; }
    for j in 0..n { p_col_ptr[j + 1] += p_col_ptr[j]; }
    let mut p_row_ind = vec![0_usize; p_triplets.len()];
    let mut p_values = vec![0.0_f64; p_triplets.len()];
    let mut p_cursor = p_col_ptr.clone();
    for &(r, c, v) in &p_triplets { let pos = p_cursor[c]; p_row_ind[pos] = r; p_values[pos] = v; p_cursor[c] += 1; }
    let p_clar = ClCsc::new(n, n, p_col_ptr, p_row_ind, p_values);

    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    if n_eq > 0 { cones.push(SupportedConeT::ZeroConeT(n_eq)); }
    if n_le_ge + n_lb + n_ub > 0 { cones.push(SupportedConeT::NonnegativeConeT(n_le_ge + n_lb + n_ub)); }
    (p_clar, prob.c.clone(), a_clar, b_clar, cones)
}

fn solve_clarabel_tol(prob: &QpProblem, tol: f64, max_iter: u32) -> (Vec<f64>, Vec<f64>, String) {
    let (p, q, a, b, cones) = build_clarabel(prob);
    let mut s = DefaultSettings::default();
    s.verbose = false;
    s.tol_gap_abs = tol; s.tol_gap_rel = tol; s.tol_feas = tol;
    s.max_iter = max_iter;
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, s).expect("clarabel setup");
    solver.solve();
    (solver.solution.x.clone(), solver.solution.z.clone(), format!("{:?}", solver.info.status))
}

/// clarabel z (cone 順) を元制約順の y にマップ。Ge は A_clar=-A 変換で y=z。
fn map_clarabel_y(prob: &QpProblem, z: &[f64]) -> Vec<f64> {
    let m = prob.num_constraints;
    let mut row_ord: Vec<(usize, ConstraintType)> =
        (0..m).map(|i| (i, prob.constraint_types[i])).collect();
    row_ord.sort_by_key(|&(_, ct)| match ct { ConstraintType::Eq => 0, _ => 1 });
    let mut row_pos = vec![0_usize; m];
    for (new_row, &(orig_row, _)) in row_ord.iter().enumerate() { row_pos[orig_row] = new_row; }
    (0..m).map(|i| if row_pos[i] < z.len() { z[row_pos[i]] } else { 0.0 }).collect()
}

fn obj_internal(prob: &QpProblem, x: &[f64]) -> f64 {
    let qx = prob.q.mat_vec_mul(x).expect("Qx");
    0.5 * qx.iter().zip(x.iter()).map(|(&q, &x)| q * x).sum::<f64>()
        + prob.c.iter().zip(x.iter()).map(|(&c, &x)| c * x).sum::<f64>()
}

/// Double-double Ax (cancellation-safe) → max Ge violation (−min slack).
fn max_viol_dd(prob: &QpProblem, x: &[f64]) -> f64 {
    use twofloat::TwoFloat;
    let m = prob.num_constraints;
    let mut ax = vec![TwoFloat::from(0.0); m];
    for col in 0..prob.num_vars {
        let xc = x[col];
        for k in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
            let r = prob.a.row_ind[k];
            ax[r] = ax[r] + TwoFloat::new_mul(prob.a.values[k], xc);
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
        if viol > mv { mv = viol; }
    }
    mv
}

/// row slack (Ge: ax-b, Le: b-ax, Eq: -(|ax-b|))。負 = 違反。
fn row_slacks(prob: &QpProblem, x: &[f64]) -> Vec<f64> {
    let ax = prob.a.mat_vec_mul(x).expect("Ax");
    (0..prob.num_constraints).map(|i| match prob.constraint_types[i] {
        ConstraintType::Ge => ax[i] - prob.b[i],
        ConstraintType::Le => prob.b[i] - ax[i],
        ConstraintType::Eq => -(ax[i] - prob.b[i]).abs(),
        _ => 0.0,
    }).collect()
}

/// y ← A·(Aᵀ·v) (banded matvec, O(nnz))。
fn aat_mul(prob: &QpProblem, v: &[f64]) -> Vec<f64> {
    let n = prob.num_vars;
    let m = prob.num_constraints;
    // w = Aᵀ v  (length n)
    let mut w = vec![0.0_f64; n];
    for col in 0..n {
        let mut acc = 0.0;
        for k in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
            acc += prob.a.values[k] * v[prob.a.row_ind[k]];
        }
        w[col] = acc;
    }
    // y = A w  (length m)
    let mut y = vec![0.0_f64; m];
    for col in 0..n {
        let wc = w[col];
        for k in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
            y[prob.a.row_ind[k]] += prob.a.values[k] * wc;
        }
    }
    y
}

/// power iteration で AAᵀ の λmax, λmin を推定し cond を返す。
fn estimate_cond_aat(prob: &QpProblem) -> (f64, f64, f64) {
    let m = prob.num_constraints;
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    // λmax
    let mut v: Vec<f64> = (0..m).map(|i| ((i * 2654435761) % 1000) as f64 / 500.0 - 1.0).collect();
    let mut nv = norm(&v); for x in v.iter_mut() { *x /= nv; }
    let mut lmax = 0.0;
    for _ in 0..200 {
        let w = aat_mul(prob, &v);
        lmax = v.iter().zip(&w).map(|(&a, &b)| a * b).sum();
        nv = norm(&w); if nv == 0.0 { break; }
        for (vi, &wi) in v.iter_mut().zip(&w) { *vi = wi / nv; }
    }
    // λmin via shifted power iteration on (lmax·I - AAᵀ)
    let mut u: Vec<f64> = (0..m).map(|i| ((i * 40503 + 7) % 997) as f64 / 498.0 - 1.0).collect();
    nv = norm(&u); for x in u.iter_mut() { *x /= nv; }
    let mut shifted_max = 0.0;
    for _ in 0..400 {
        let aatu = aat_mul(prob, &u);
        let w: Vec<f64> = u.iter().zip(&aatu).map(|(&ui, &ai)| lmax * ui - ai).collect();
        shifted_max = u.iter().zip(&w).map(|(&a, &b)| a * b).sum();
        nv = norm(&w); if nv == 0.0 { break; }
        for (ui, &wi) in u.iter_mut().zip(&w) { *ui = wi / nv; }
    }
    let lmin = lmax - shifted_max;
    (lmax, lmin, lmax / lmin.max(1e-300))
}

fn diagnose(name: &str) {
    let path = std::path::PathBuf::from(format!("data/maros_meszaros/{}.QPS", name));
    assert!(path.exists(), "{:?} not found", path);
    let prob = parse_qps(&path).expect("parse");

    let (x_ref, z_ref, st_strict) = solve_clarabel_tol(&prob, STRICT_TOL, STRICT_MAX_ITER);
    let (x_def, _z_def, st_def) = solve_clarabel_tol(&prob, 1e-8, 5_000);
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(60.0);
    let res = solve_qp_with(&prob, &opts);
    let x = &res.solution;

    // warm-start from clarabel (x, y) → basin path-dependence の測定。
    let y_ws = map_clarabel_y(&prob, &z_ref);
    let mu_ws = {
        let ax = prob.a.mat_vec_mul(&x_ref).unwrap();
        let mut acc = 0.0; let mut cnt = 0usize;
        for i in 0..prob.num_constraints {
            let slack = match prob.constraint_types[i] {
                ConstraintType::Ge => ax[i] - prob.b[i],
                ConstraintType::Le => prob.b[i] - ax[i],
                _ => 0.0,
            };
            acc += (slack * y_ws[i]).abs(); cnt += 1;
        }
        if cnt > 0 { (acc / cnt as f64).max(1e-8) } else { 1e-8 }
    };
    let mut opts_ws = SolverOptions::default();
    opts_ws.timeout_secs = Some(60.0);
    opts_ws.warm_start_qp = Some(QpWarmStart { x: x_ref.clone(), y: y_ws, mu: mu_ws });
    let res_ws = solve_qp_with(&prob, &opts_ws);
    let obj_ws = obj_internal(&prob, &res_ws.solution);

    let obj_ref = obj_internal(&prob, &x_ref);
    let obj_def = obj_internal(&prob, &x_def);
    let obj_ours = obj_internal(&prob, x);
    let rel = (obj_ref - obj_ours).abs() / obj_ref.abs().max(obj_ours.abs()).max(1.0);

    // max constraint violation (min over rows of slack, negated when negative)
    let max_viol = |xv: &[f64]| -> f64 {
        row_slacks(&prob, xv).iter().map(|&v| (-v).max(0.0)).fold(0.0_f64, f64::max)
    };
    eprintln!("\n===== {} (n={} m={}) =====", name, prob.num_vars, prob.num_constraints);
    eprintln!("clarabel: strict(1e-12)={} obj={:.6e} maxviol_f64={:.3e} maxviol_dd={:.3e} | default(1e-8)={} obj={:.6e}",
        st_strict, obj_ref, max_viol(&x_ref), max_viol_dd(&prob, &x_ref), st_def, obj_def);
    eprintln!("ours: maxviol_f64={:.3e} maxviol_dd={:.3e}", max_viol(x), max_viol_dd(&prob, x));
    eprintln!("warm-start-from-clarabel: status={:?} obj={:.6e} maxviol_dd={:.3e} (basin path-dependence test)",
        res_ws.status, obj_ws, max_viol_dd(&prob, &res_ws.solution));
    let y_inf = res.dual_solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    // λmax(AAᵀ) は power-iter で確実、λmin はクラスタで収束遅く下限のみ → cond は λmax/λmin_est で
    // 下限扱い (真の cond は biharmonic 解析より ~n⁴ で更に大)。
    let (lmax, lmin_est, cond_lb) = estimate_cond_aat(&prob);
    eprintln!("ours |y|inf={:.3e} | AAᵀ λmax={:.3e} (analytic≈16) λmin_powiter≳{:.3e} cond_lb≳{:.2e}",
        y_inf, lmax, lmin_est, cond_lb);

    let sl_ref = row_slacks(&prob, &x_ref);
    let sl_ours = row_slacks(&prob, x);
    const ACT: f64 = 1e-6;
    let act_ref: Vec<bool> = sl_ref.iter().map(|&v| v.abs() <= ACT).collect();
    let act_ours: Vec<bool> = sl_ours.iter().map(|&v| v.abs() <= ACT).collect();
    let n_aref = act_ref.iter().filter(|&&b| b).count();
    let n_aours = act_ours.iter().filter(|&&b| b).count();
    let both = act_ref.iter().zip(&act_ours).filter(|&(&r, &o)| r && o).count();
    let jac = both as f64 / (n_aref + n_aours - both).max(1) as f64;

    // 違反集計 (slack < -ACT)
    let viol_ours: Vec<(usize, f64)> = sl_ours.iter().enumerate()
        .filter(|(_, &v)| v < -ACT).map(|(i, &v)| (i, v)).collect();
    let viol_ref: usize = sl_ref.iter().filter(|&&v| v < -ACT).count();
    let max_viol_ours = viol_ours.iter().map(|&(_, v)| -v).fold(0.0_f64, f64::max);

    // x 距離
    let (mut max_dx, mut sum_dx, mut argmax) = (0.0_f64, 0.0_f64, 0usize);
    for (j, (&a, &b)) in x_ref.iter().zip(x.iter()).enumerate() {
        let d = (a - b).abs(); sum_dx += d;
        if d > max_dx { max_dx = d; argmax = j; }
    }
    let mean_dx = sum_dx / x_ref.len() as f64;
    let xref_inf = x_ref.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let xours_inf = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));

    eprintln!("status_ours={:?} iters={}", res.status, res.iterations);
    eprintln!("obj: ref={:.6e} ours={:.6e} rel_diff={:.3e} (ours<ref ⇒ infeasible-lower)", obj_ref, obj_ours, rel);
    eprintln!("active: ref={} ours={} both={} jaccard={:.4}", n_aref, n_aours, both, jac);
    eprintln!("viol(slack<-{:.0e}): ref={} ours={} max_viol_ours={:.3e}", ACT, viol_ref, viol_ours.len(), max_viol_ours);
    eprintln!("|x_ref - x_ours|: max={:.3e}@{} mean={:.3e} | |x_ref|inf={:.3e} |x_ours|inf={:.3e}",
        max_dx, argmax, mean_dx, xref_inf, xours_inf);
    // top-5 violated rows detail
    let mut vsorted = viol_ours.clone();
    vsorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ax_ours = prob.a.mat_vec_mul(x).expect("Ax");
    eprintln!("top violated rows (ours): row | slack | ax | b | type");
    for &(i, v) in vsorted.iter().take(5) {
        eprintln!("  {:5} | {:+.3e} | {:+.3e} | {:+.3e} | {:?}", i, v, ax_ours[i], prob.b[i], prob.constraint_types[i]);
    }

    // load-bearing facts (案A): (1) false Optimal を返さない、(2) feasible、
    // (3) clarabel と competitive (= feasibility で劣らない)。obj は未確定 baseline に pin しない。
    let ours_mv = max_viol_dd(&prob, x);
    let cl_mv = max_viol_dd(&prob, &x_ref);
    assert_ne!(res.status, otspot::problem::SolveStatus::Optimal,
        "{}: f64 で certify 不能な ill-cond QP を Optimal と誤主張", name);
    assert!(ours_mv <= 1e-5, "{}: ours infeasible (DD maxviol={:.3e})", name, ours_mv);
    assert!(ours_mv <= cl_mv * 10.0 + 1e-12,
        "{}: ours maxviol {:.3e} が clarabel {:.3e} の 10× 超 = competitive 退化", name, ours_mv, cl_mv);
}

#[test]
#[ignore = "diag: LISWET wrong-basin root-cause (clarabel strict + 60s solve)"]
fn diag_liswet_basin_9_12() {
    diagnose("LISWET9");
    diagnose("LISWET12");
}
