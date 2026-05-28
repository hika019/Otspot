//! Mehrotra IPM (IP-PMM)。
//! 1 層 retry で eps を直線厳格化、status 変換は API 境界の 1 箇所に集約、KKT は元空間判定。

pub mod outcome;
pub mod kkt;
pub mod core;
pub mod attempt;

pub use attempt::solve_ipm;

#[cfg(test)]
#[allow(clippy::field_reassign_with_default, clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::io::qps::parse_qps;
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use std::path::Path;

    // test_ipm_hs21_cmp_full_solver moved to otspot-io/tests/bug_repro.rs (#28 dedup).

    #[test]
    fn test_v2_hs21() {
        let path = Path::new("data/maros_meszaros/HS21.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse HS21");
        let opts = SolverOptions::default();
        let r = solve_ipm(&prob, &opts);
        assert_eq!(r.status, SolveStatus::Optimal);
    }

    /// QADLITTL の DFEAS_FAIL を z, y, r_j 単位で診断。
    #[test]
    fn test_ipm_qadlittl_diagnose() {
        let path = Path::new("data/maros_meszaros/QADLITTL.QPS");
        if !path.exists() {
            eprintln!("QADLITTL.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse QADLITTL");
        let opts = SolverOptions::default();
        let r = solve_ipm(&prob, &opts);
        eprintln!("=== QADLITTL v2 result ===");
        eprintln!("  status={:?} obj={:.6e} iters={}", r.status, r.objective, r.iterations);
        eprintln!("  n={} m={} n_lb={} n_ub={}", prob.num_vars, prob.num_constraints,
                 prob.bounds.iter().filter(|(lb,_)| lb.is_finite()).count(),
                 prob.bounds.iter().filter(|(_,ub)| ub.is_finite()).count());

        // 元空間 r_j = (Q*x + c + A^T*y - z_lb + z_ub)_j を計算
        let qx = prob.q.mat_vec_mul(&r.solution).unwrap();
        let aty = prob.a.transpose().mat_vec_mul(&r.dual_solution).unwrap();
        let n_lb = prob.bounds.iter().filter(|(lb,_)| lb.is_finite()).count();
        let mut bound_contrib = vec![0.0_f64; prob.num_vars];
        let mut idx = 0;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && idx < r.bound_duals.len() {
                bound_contrib[j] -= r.bound_duals[idx];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && idx < r.bound_duals.len() {
                bound_contrib[j] += r.bound_duals[idx];
                idx += 1;
            }
        }
        // top-10 大きな |r_j| 成分を出力
        let mut residuals: Vec<(usize, f64, f64, f64, f64, f64)> = (0..prob.num_vars).map(|j| {
            let r_j = qx[j] + prob.c[j] + aty[j] + bound_contrib[j];
            (j, r_j, qx[j], aty[j], bound_contrib[j], r.solution[j])
        }).collect();
        residuals.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        eprintln!("Top-10 worst residuals:");
        eprintln!("  j     | r_j       | qx_j      | aty_j     | bnd_j     | x_j       | lb,ub");
        for &(j, r_j, qxj, atyj, bndj, xj) in residuals.iter().take(10) {
            let (lb, ub) = prob.bounds[j];
            eprintln!("  {:4} | {:+.3e} | {:+.3e} | {:+.3e} | {:+.3e} | {:+.3e} | ({:+.3e}, {:+.3e})",
                     j, r_j, qxj, atyj, bndj, xj, lb, ub);
        }
        // z_lb, z_ub 値の分布を確認
        let max_z_lb = r.bound_duals[..n_lb].iter().fold(0.0_f64, |a, &v| a.max(v));
        let nonzero_z_lb = r.bound_duals[..n_lb].iter().filter(|&&v| v.abs() > 1e-12).count();
        eprintln!("z_lb: max={:.3e}, nonzero count={}/{}", max_z_lb, nonzero_z_lb, n_lb);
    }

    /// Q-prefix LP の Q 規模比較 (Q≒0 検出基準確認用)。
    #[test]
    fn test_q_magnitude_catastrophic() {
        let problems = [
            "QADLITTL", "QBORE3D", "QCAPRI", "QETAMACR", "QFFFFF80",
            "QPCBOEI1", "QSEBA", "QSHELL", "QSCRS8",
            "QSHARE2B", "QSC205", "QGROW7", "QPCSTAIR", "QSCAGR25", "QSHIP12L",
        ];
        eprintln!("{:12} {:>5} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "name", "n", "nnz_Q", "||Q||_F", "max|Q|", "||c||", "max|c|", "Q/(1+c)");
        for p in problems.iter() {
            let path_str = format!("data/maros_meszaros/{}.QPS", p);
            let path = Path::new(&path_str);
            if !path.exists() {
                eprintln!("{:12} not found", p);
                continue;
            }
            let prob = parse_qps(path).expect(p);
            let n = prob.num_vars;
            let nnz_q = prob.q.values.len();
            let q_max: f64 = prob.q.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let q_fro: f64 = prob.q.values.iter().map(|v| v*v).sum::<f64>().sqrt();
            let c_norm: f64 = prob.c.iter().map(|v| v*v).sum::<f64>().sqrt();
            let c_max: f64 = prob.c.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let ratio = q_fro / (1.0 + c_norm);
            eprintln!("{:12} {:5} {:6} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e}",
                p, n, nnz_q, q_fro, q_max, c_norm, c_max, ratio);
        }
    }

    /// BD-T4 (rank-deficient Q + EmptyCol) の詰まり調査。
    #[test]
    fn test_ipm_bd_t4_diagnose() {
        use crate::sparse::CscMatrix;
        use crate::qp::problem::QpProblem;
        let n = 3usize;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[0.001, 0.001], n, n).unwrap();
        let c = vec![-1.0, -1.0, 1.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, n).unwrap();
        let b = vec![4.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY), (0.0_f64, 3.0_f64)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(5.0);
        let r = solve_ipm(&problem, &opts);
        eprintln!("BD-T4 v2: status={:?} obj={:.5e} iters={}", r.status, r.objective, r.iterations);
        eprintln!("  x={:?}", r.solution);
        eprintln!("  y={:?}", r.dual_solution);
        eprintln!("  z={:?}", r.bound_duals);
        let view = super::outcome::ProblemView {
            q: &problem.q, a: &problem.a, c: &problem.c, b: &problem.b,
            bounds: &problem.bounds, constraint_types: &problem.constraint_types,
            eliminated_cols: &[],
        };
        let kkt = super::kkt::kkt_residual_rel(&view, &r.solution, &r.dual_solution, &r.bound_duals);
        let pres = super::kkt::primal_residual_rel(&view, &r.solution);
        let bv = super::kkt::bound_violation(&problem.bounds, &r.solution);
        eprintln!("  kkt={:.3e} pres={:.3e} bv={:.3e}", kkt, pres, bv);
    }

    #[test]
    fn test_osqp_eval_catastrophic() {
        let problems = [
            "QADLITTL", "QBORE3D", "QCAPRI", "QETAMACR", "QFFFFF80",
            "QPCBOEI1", "QSEBA", "QSHELL", "QSCRS8",
        ];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { continue; }
            let prob = parse_qps(path).expect(name);
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let r = solve_ipm(&prob, &opts);
            let view = super::outcome::ProblemView {
                q: &prob.q, a: &prob.a, c: &prob.c, b: &prob.b,
                bounds: &prob.bounds, constraint_types: &prob.constraint_types, eliminated_cols: &[],
            };
            let kkt = super::kkt::kkt_residual_rel(&view, &r.solution, &r.dual_solution, &r.bound_duals);
            let pres = super::kkt::primal_residual_rel(&view, &r.solution);
            let bv = super::kkt::bound_violation(&prob.bounds, &r.solution);
            let prospect_1000s = kkt < 1e-3 && pres < 1e-3 && bv < 1e-6;
            eprintln!("{:10} status={:?} kkt={:.3e} pres={:.3e} prospect_1000s={}",
                name, r.status, kkt, pres, prospect_1000s);
        }
    }

    #[test]
    fn test_osqp_eval_marginal() {
        let problems = [
            "PRIMALC5", "QSCAGR25", "QSCAGR7", "QSHIP12L", "QSHIP12S",
            "QBANDM", "QSHARE1B",
            "QADLITTL", "QSCRS8",
        ];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { continue; }
            let prob = parse_qps(path).expect(name);
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let r = solve_ipm(&prob, &opts);
            let view = super::outcome::ProblemView {
                q: &prob.q, a: &prob.a, c: &prob.c, b: &prob.b,
                bounds: &prob.bounds, constraint_types: &prob.constraint_types, eliminated_cols: &[],
            };
            let kkt = super::kkt::kkt_residual_rel(&view, &r.solution, &r.dual_solution, &r.bound_duals);
            let pres = super::kkt::primal_residual_rel(&view, &r.solution);
            let bv = super::kkt::bound_violation(&prob.bounds, &r.solution);
            let pass = kkt < 1e-6 && pres < 1e-6 && bv < 1e-6;
            eprintln!("{:10} status={:?} kkt={:.3e} pres={:.3e} bv={:.3e} PASS_eval={}",
                name, r.status, kkt, pres, bv, pass);
        }
    }

    /// Marginal 問題で x の bound 距離分布 (snap_tol 決定用)。
    #[test]
    fn test_marginal_x_bound_gap() {
        let problems = ["PRIMALC5", "QSCAGR25", "QSCAGR7", "QSHIP12L", "QSHIP12S"];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { continue; }
            let prob = parse_qps(path).expect(name);
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let r = solve_ipm(&prob, &opts);
            let n = prob.num_vars;
            let mut gaps: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
            for j in 0..n {
                let (lb, ub) = prob.bounds[j];
                let d_lb = if lb.is_finite() { (r.solution[j] - lb).abs() } else { f64::INFINITY };
                let d_ub = if ub.is_finite() { (ub - r.solution[j]).abs() } else { f64::INFINITY };
                let min_dist = d_lb.min(d_ub);
                gaps.push((j, r.solution[j], lb, ub, min_dist));
            }
            gaps.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
            let buckets = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0];
            let counts: Vec<usize> = buckets.iter().map(|&t| {
                gaps.iter().filter(|(_, _, lb, ub, d)| (lb.is_finite() || ub.is_finite()) && *d < t).count()
            }).collect();
            eprintln!("{:10} n={} status={:?}", name, n, r.status);
            eprintln!("  min_dist 分布: <1e-12:{} <1e-10:{} <1e-8:{} <1e-6:{} <1e-4:{} <1e-2:{} <1.0:{}",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]);
            let snap_candidates: Vec<_> = gaps.iter().filter(|(_, _, lb, ub, d)|
                (lb.is_finite() || ub.is_finite()) && *d > 1e-12 && *d < 1e-3
            ).take(5).collect();
            for (j, x, lb, ub, dist) in snap_candidates {
                eprintln!("  j={} x={:.3e} lb={:.3e} ub={:.3e} min_dist={:.3e}", j, x, lb, ub, dist);
            }
        }
    }

    /// QPILOTNO row 14 / 130 の A 行ペア一致確認。
    #[test]
    fn test_ipm_qpilotno_row_pair_check() {
        let path = Path::new("data/maros_meszaros/QPILOTNO.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse QPILOTNO");
        let extract_row = |row_target: usize| -> Vec<(usize, f64)> {
            let mut entries = Vec::new();
            for col in 0..prob.num_vars {
                let cs = prob.a.col_ptr[col];
                let ce = prob.a.col_ptr[col + 1];
                for k in cs..ce {
                    if prob.a.row_ind[k] == row_target {
                        entries.push((col, prob.a.values[k]));
                    }
                }
            }
            entries
        };
        let r14 = extract_row(14);
        let r130 = extract_row(130);
        eprintln!("Row 14: nnz={} b={}", r14.len(), prob.b[14]);
        for (c, v) in &r14 { eprintln!("  col[{}] = {:+.5e}", c, v); }
        eprintln!("Row 130: nnz={} b={}", r130.len(), prob.b[130]);
        for (c, v) in &r130 { eprintln!("  col[{}] = {:+.5e}", c, v); }
        if r14.len() == r130.len() {
            let same_cols = r14.iter().zip(r130.iter()).all(|(a, b)| a.0 == b.0);
            if same_cols {
                let val_pattern: Vec<(f64, f64, f64)> = r14.iter().zip(r130.iter())
                    .map(|((_, v1), (_, v2))| (*v1, *v2, v1 + v2)).collect();
                eprintln!("Same column support. Sum (v1+v2): {:?}", &val_pattern[..5.min(val_pattern.len())]);
                let max_sum = val_pattern.iter().map(|(_,_,s)| s.abs()).fold(0.0_f64, f64::max);
                eprintln!("max |v1+v2| = {:.3e}  (0 → 行 14 = -行 130)", max_sum);
            }
        }
    }

    /// QFORPLAN の dual residual 集中 col 特定 (NO_PRESOLVE=1 / QFORPLAN_LONG=1)。
    #[test]
    fn test_ipm_qforplan_dual_residual_diagnose() {
        let path = Path::new("data/maros_meszaros/QFORPLAN.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse QFORPLAN");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(if std::env::var("QFORPLAN_LONG").ok().as_deref() == Some("1") { 60.0 } else { 10.0 });
        if std::env::var("NO_PRESOLVE").ok().as_deref() == Some("1") {
            opts.presolve = false;
        }
        let r = solve_ipm(&prob, &opts);
        eprintln!("QFORPLAN status={:?} obj={:.5e} presolve={}", r.status, r.objective, opts.presolve);
        use twofloat::TwoFloat;
        let n = prob.num_vars;
        let zero_dd = TwoFloat::from(0.0);
        let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let xv = r.solution[col];
            for k in prob.q.col_ptr[col]..prob.q.col_ptr[col + 1] {
                qx_dd[prob.q.row_ind[k]] += TwoFloat::new_mul(prob.q.values[k], xv);
            }
        }
        let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
        let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
                let row = prob.a.row_ind[k];
                aty_dd[col] += TwoFloat::new_mul(prob.a.values[k], r.dual_solution[row]);
            }
        }
        let aty: Vec<f64> = aty_dd.iter().map(|&v| f64::from(v)).collect();
        let n_lb = prob.bounds.iter().filter(|(lb,_)| lb.is_finite()).count();
        let mut bnd = vec![0.0_f64; n];
        let mut idx = 0;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && idx < r.bound_duals.len() {
                bnd[j] -= r.bound_duals[idx];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && idx < r.bound_duals.len() {
                bnd[j] += r.bound_duals[idx];
                idx += 1;
            }
        }
        let mut entries: Vec<(usize, f64, f64, f64, f64, f64, f64, f64)> = (0..n).map(|j| {
            let res = qx[j] + prob.c[j] + aty[j] + bnd[j];
            let scale = 1.0 + qx[j].abs() + prob.c[j].abs() + aty[j].abs() + bnd[j].abs();
            let rel = res.abs() / scale;
            (j, res, qx[j], aty[j], bnd[j], prob.c[j], rel, r.solution[j])
        }).collect();
        entries.sort_by(|a, b| b.6.partial_cmp(&a.6).unwrap());
        eprintln!("Top dual residual cols (componentwise rel):");
        let _ = n_lb;
        for (j, res, qx_j, aty_j, bnd_j, c_j, rel, x_j) in entries.iter().take(8) {
            let (lb, ub) = prob.bounds[*j];
            let nrows = prob.a.col_ptr[j+1] - prob.a.col_ptr[*j];
            eprintln!("  j={:5} rel={:.3e} r={:+.3e} qx={:+.3e} aty={:+.3e} bnd={:+.3e} c={:+.3e} x={:+.3e} lb={:+.2e} ub={:+.2e} nrows={}",
                j, rel, res, qx_j, aty_j, bnd_j, c_j, x_j, lb, ub, nrows);
        }
    }

    /// QPILOTNO primal violation 集中行特定 (NO_PRESOLVE=1 で切り分け)。
    #[test]
    fn test_ipm_qpilotno_primal_violation_diagnose() {
        let path = Path::new("data/maros_meszaros/QPILOTNO.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse QPILOTNO");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(30.0);
        if std::env::var("NO_PRESOLVE").ok().as_deref() == Some("1") {
            opts.presolve = false;
        }
        let r = solve_ipm(&prob, &opts);
        eprintln!("QPILOTNO status={:?} obj={:.5e} presolve={}", r.status, r.objective, opts.presolve);
        let ax = prob.a.mat_vec_mul(&r.solution).unwrap();
        let mut violations: Vec<(usize, f64, f64, f64, f64)> = (0..prob.num_constraints).map(|i| {
            let v = match prob.constraint_types[i] {
                crate::problem::ConstraintType::Eq => (ax[i] - prob.b[i]).abs(),
                crate::problem::ConstraintType::Ge => (prob.b[i] - ax[i]).max(0.0),
                crate::problem::ConstraintType::Le => (ax[i] - prob.b[i]).max(0.0),
            };
            let scale = 1.0 + ax[i].abs() + prob.b[i].abs();
            (i, v, v / scale, ax[i], prob.b[i])
        }).collect();
        violations.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        eprintln!("Top 10 primal violations (componentwise rel):");
        for (i, v, rel, ax_i, b_i) in violations.iter().take(10) {
            eprintln!("  i={:5} viol={:.3e} rel={:.3e} ax={:+.3e} b={:+.3e} type={:?}",
                i, v, rel, ax_i, b_i, prob.constraint_types[*i]);
        }
    }

    /// DPKLO1 が hang せず timeout/optimal で返ること。
    #[test]
    fn test_ipm_dpklo1() {
        let path = Path::new("data/maros_meszaros/DPKLO1.QPS");
        if !path.exists() {
            eprintln!("DPKLO1.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse DPKLO1");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(5.0);
        let r = solve_ipm(&prob, &opts);
        eprintln!("DPKLO1 v2: status={:?} obj={} iters={}", r.status, r.objective, r.iterations);
        assert!(matches!(r.status, SolveStatus::Optimal | SolveStatus::Timeout));
    }
}
