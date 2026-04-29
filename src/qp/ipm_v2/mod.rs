//! IP-PMM v2 - クリーン設計の Mehrotra Interior Point Method
//!
//! 既存 `ipm/ippmm.rs` を温存しつつ、設計書 (`docs/solver_overview_design.md`) の
//! 原則に厳密に従って新規実装する:
//!
//! 1. **retry 1 層**: 時間内で eps 厳格化を直線的に進める。多層 retry を排除。
//! 2. **status 変換 1 箇所**: 内部は `IpmOutcome` struct で残差・解を持ち、
//!    API 境界 (`solve_qp_v2`) で `SolverResult` (外部 status) に変換する。
//! 3. **元空間 KKT 直接判定**: scaled 空間判定で偽 Optimal を出さない。
//! 4. **大規模対応**: supernode-aware LDL を `linalg::ldl` 経由で直接利用。
//!
//! 既存 `solve_qp_with` の Concurrent solver 経路で v2 を選択肢として追加する。
//! v2 が品質・性能で旧 ippmm を上回ったら旧版を削除する段階移行を行う。
//!
//! # アーキテクチャ
//!
//! ```text
//! solve_qp_v2(prob, opts) -> SolverResult
//!     ├── presolve(prob, deadline) -> reduced
//!     ├── deadline = compute_deadline(opts)
//!     ├── for attempt in 0.. while now() < deadline:
//!     │       eps_attempt = opts.eps / 10^attempt   # 直線的に厳格化
//!     │       outcome = single_attempt(reduced, eps_attempt, deadline_attempt)
//!     │       if outcome.kkt_satisfied(eps_orig): break  # 元空間判定
//!     ├── postsolve(reduced_outcome) -> orig_solution
//!     └── finalize(outcome) -> SolverResult  # 外部 status に変換
//! ```
//!
//! # 各モジュール
//!
//! - `outcome`: 内部 `IpmOutcome` struct (status mutation の対象を 1 箇所に集約)
//! - `attempt`: 1 回の Mehrotra IPM 呼出 (Ruiz scale + iterate + unscale + KKT verify)
//! - `kkt`: 元空間 KKT 残差計算 (bench compute_dfeas_orig と同形)
//! - `core`: Mehrotra predictor-corrector の純粋実装

pub mod outcome;
pub mod kkt;
pub mod core;
pub mod attempt;

pub use attempt::solve_qp_v2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::qps::parse_qps;
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use std::path::Path;

    /// HS21 で v1 と v2 を直接比較する診断テスト。
    #[test]
    #[ignore]
    fn test_v2_hs21_cmp_v1() {
        let path = Path::new("data/maros_meszaros/HS21.QPS");
        if !path.exists() {
            eprintln!("HS21.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse HS21");
        let opts = SolverOptions::default();
        let v1 = crate::qp::solve_qp_with(&prob, &opts);
        let v2 = solve_qp_v2(&prob, &opts);
        eprintln!("=== v1 ===");
        eprintln!("  status={:?} obj={} iters={}", v1.status, v1.objective, v1.iterations);
        eprintln!("  x={:?}", v1.solution);
        eprintln!("  y={:?}", v1.dual_solution);
        eprintln!("  z={:?}", v1.bound_duals);
        eprintln!("=== v2 ===");
        eprintln!("  status={:?} obj={} iters={}", v2.status, v2.objective, v2.iterations);
        eprintln!("  x={:?}", v2.solution);
        eprintln!("  y={:?}", v2.dual_solution);
        eprintln!("  z={:?}", v2.bound_duals);
        let view = super::outcome::ProblemView {
            q: &prob.q, a: &prob.a, c: &prob.c, b: &prob.b,
            bounds: &prob.bounds, constraint_types: &prob.constraint_types,
        };
        let v1_kkt = super::kkt::kkt_residual_rel(&view, &v1.solution, &v1.dual_solution, &v1.bound_duals);
        let v2_kkt = super::kkt::kkt_residual_rel(&view, &v2.solution, &v2.dual_solution, &v2.bound_duals);
        eprintln!("v1 KKT_rel={:.3e}", v1_kkt);
        eprintln!("v2 KKT_rel={:.3e}", v2_kkt);
    }

    /// HS21 で v2 が PASS することを確認 (smoke test)。次セッションで debug 完了後 #[ignore] 外す。
    #[test]
    #[ignore]
    fn test_v2_hs21() {
        let path = Path::new("data/maros_meszaros/HS21.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse HS21");
        let opts = SolverOptions::default();
        let r = solve_qp_v2(&prob, &opts);
        assert_eq!(r.status, SolveStatus::Optimal, "HS21 v2 should be Optimal");
    }

    /// QADLITTL (n=97, m=56) で残 DFEAS_FAIL の構造を診断する。
    /// 現状: dfr=1.0e0 なのに obj は正解値と一致 → x は正しい、z が完全に外れている。
    /// このテストで z, y, r_j の詳細を出力し、どの bound で大きな寄与が出てるか把握する。
    #[test]
    #[ignore]
    fn test_v2_qadlittl_diagnose() {
        let path = Path::new("data/maros_meszaros/QADLITTL.QPS");
        if !path.exists() {
            eprintln!("QADLITTL.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse QADLITTL");
        let opts = SolverOptions::default();
        let r = solve_qp_v2(&prob, &opts);
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

    /// Catastrophic 9件 (Q-prefix LP 由来) と比較対象の Q 規模を出力する診断。
    /// task: Q≒0 検出基準を相対値に変更すべきか確認する。
    #[test]
    #[ignore]
    fn test_q_magnitude_catastrophic() {
        let problems = [
            // Catastrophic 9件
            "QADLITTL", "QBORE3D", "QCAPRI", "QETAMACR", "QFFFFF80",
            "QPCBOEI1", "QSEBA", "QSHELL", "QSCRS8",
            // 比較: 通常 PASS する Q-prefix 問題
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

    /// Marginal 問題で x が bound からどの程度離れているか診断する。
    /// snap_tol を決めるための事実確認。
    #[test]
    #[ignore]
    fn test_marginal_x_bound_gap() {
        let problems = ["PRIMALC5", "QSCAGR25", "QSCAGR7", "QSHIP12L", "QSHIP12S"];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { continue; }
            let prob = parse_qps(path).expect(name);
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let r = solve_qp_v2(&prob, &opts);
            let n = prob.num_vars;
            // 各変数の bound 距離を計算
            let mut gaps: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
            for j in 0..n {
                let (lb, ub) = prob.bounds[j];
                let d_lb = if lb.is_finite() { (r.solution[j] - lb).abs() } else { f64::INFINITY };
                let d_ub = if ub.is_finite() { (ub - r.solution[j]).abs() } else { f64::INFINITY };
                let min_dist = d_lb.min(d_ub);
                gaps.push((j, r.solution[j], lb, ub, min_dist));
            }
            // min_dist で sort
            gaps.sort_by(|a, b| a.4.partial_cmp(&b.4).unwrap());
            // < 1e-3 の変数数を集計、ヒストグラム
            let buckets = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0];
            let counts: Vec<usize> = buckets.iter().map(|&t| {
                gaps.iter().filter(|(_, _, lb, ub, d)| (lb.is_finite() || ub.is_finite()) && *d < t).count()
            }).collect();
            eprintln!("{:10} n={} status={:?}", name, n, r.status);
            eprintln!("  min_dist 分布: <1e-12:{} <1e-10:{} <1e-8:{} <1e-6:{} <1e-4:{} <1e-2:{} <1.0:{}",
                counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]);
            // 1e-6〜1e-3 範囲の変数を出力 (snap で動かしうる)
            let snap_candidates: Vec<_> = gaps.iter().filter(|(_, _, lb, ub, d)|
                (lb.is_finite() || ub.is_finite()) && *d > 1e-12 && *d < 1e-3
            ).take(5).collect();
            for (j, x, lb, ub, dist) in snap_candidates {
                eprintln!("  j={} x={:.3e} lb={:.3e} ub={:.3e} min_dist={:.3e}", j, x, lb, ub, dist);
            }
        }
    }

    /// LP の reduced_costs/dual そのままで KKT 残差を計算する (post-processing なし)。
    #[test]
    #[ignore]
    fn test_lp_raw_kkt() {
        let problems = ["QADLITTL", "QSCRS8", "QSHELL", "QBORE3D", "QSHARE2B"];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { continue; }
            let prob = parse_qps(path).expect(name);
            let mut q_zero = prob.q.clone();
            for v in q_zero.values.iter_mut() { *v = 0.0; }
            let prob_lp = crate::qp::problem::QpProblem::new(
                q_zero, prob.c.clone(), prob.a.clone(), prob.b.clone(),
                prob.bounds.clone(), prob.constraint_types.clone(),
            ).unwrap();
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let lp_result = crate::qp::solve_qp_with(&prob_lp, &opts);
            if !matches!(lp_result.status, SolveStatus::Optimal) { continue; }

            let n = prob.num_vars;
            let mut x = lp_result.solution.clone();
            for (xi, &(lb, ub)) in x.iter_mut().zip(prob.bounds.iter()) {
                if lb.is_finite() { *xi = xi.max(lb); }
                if ub.is_finite() { *xi = xi.min(ub); }
            }
            let n_lb = prob.bounds.iter().filter(|(lb, _)| lb.is_finite()).count();
            let n_ub = prob.bounds.iter().filter(|(_, ub)| ub.is_finite()).count();
            let mut z = vec![0.0_f64; n_lb + n_ub];
            if lp_result.reduced_costs.len() == n {
                let mut lb_idx = 0;
                let mut ub_idx = 0;
                for (j, &(lb, ub)) in prob.bounds.iter().enumerate() {
                    let rc = lp_result.reduced_costs[j];
                    if lb.is_finite() {
                        if rc > 0.0 { z[lb_idx] = rc; }
                        lb_idx += 1;
                    }
                    if ub.is_finite() {
                        if rc < 0.0 { z[n_lb + ub_idx] = -rc; }
                        ub_idx += 1;
                    }
                }
            }
            let y = lp_result.dual_solution.clone();
            let view = super::outcome::ProblemView {
                q: &prob.q, a: &prob.a, c: &prob.c, b: &prob.b,
                bounds: &prob.bounds, constraint_types: &prob.constraint_types,
            };
            let kkt_raw = super::kkt::kkt_residual_rel(&view, &x, &y, &z);
            // y のみ refine
            let mut tmp = crate::problem::SolverResult {
                status: SolveStatus::Optimal,
                solution: x.clone(),
                dual_solution: y.clone(),
                bound_duals: z.clone(),
                ..Default::default()
            };
            if prob.num_constraints > 0 {
                crate::qp::refine_dual_lsq(&prob, &mut tmp);
            }
            let kkt_yref = super::kkt::kkt_residual_rel(&view, &tmp.solution, &tmp.dual_solution, &tmp.bound_duals);
            eprintln!("{:10} kkt_raw(LP_z+LP_y)={:.3e} kkt_y_refined(LP_z)={:.3e} y_changed={}",
                name, kkt_raw, kkt_yref, tmp.dual_solution != y);
        }
    }

    /// LP 試行の outcome (post-processing 済み) の KKT 残差を直接確認する。
    #[test]
    #[ignore]
    fn test_lp_postprocess_kkt() {
        let problems = ["QADLITTL", "QSCRS8", "QSHELL", "QBORE3D", "QPCBOEI1", "QSHARE2B"];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { eprintln!("{} skipped", name); continue; }
            let prob = parse_qps(path).expect(name);
            let mut q_zero = prob.q.clone();
            for v in q_zero.values.iter_mut() { *v = 0.0; }
            let prob_lp = crate::qp::problem::QpProblem::new(
                q_zero, prob.c.clone(), prob.a.clone(), prob.b.clone(),
                prob.bounds.clone(), prob.constraint_types.clone(),
            ).unwrap();
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let lp_result = crate::qp::solve_qp_with(&prob_lp, &opts);
            if !matches!(lp_result.status, SolveStatus::Optimal) {
                eprintln!("{:10} LP status={:?} (skip)", name, lp_result.status);
                continue;
            }
            let outcome = super::core::run_lp_postprocess(
                &prob,
                lp_result.solution,
                lp_result.dual_solution,
                lp_result.reduced_costs,
            );
            eprintln!("{:10} LP_kkt={:.3e} pres={:.3e} bv={:.3e} obj={:.5e} satisfies_1e-6={}",
                name, outcome.kkt_residual_rel, outcome.primal_residual_rel,
                outcome.bound_violation, outcome.objective, outcome.satisfies_eps(1e-6));
        }
    }

    /// LP-dispatch を統合した solve_qp_v2 が LP-dominant 問題を救えるか確認。
    #[test]
    #[ignore]
    fn test_v2_lp_dispatch_integrated() {
        let problems = ["QADLITTL", "QSCRS8", "QSHELL", "QBORE3D", "QPCBOEI1", "QSHARE2B"];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { eprintln!("{} skipped", name); continue; }
            let prob = parse_qps(path).expect(name);
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let r = solve_qp_v2(&prob, &opts);
            // 元 QP の objective として 0.5 x'Qx + c'x を計算 (検証用)
            let qx = prob.q.mat_vec_mul(&r.solution).unwrap_or(vec![0.0; prob.num_vars]);
            let qp_obj: f64 = 0.5 * qx.iter().zip(r.solution.iter()).map(|(a,b)| a*b).sum::<f64>()
                + prob.c.iter().zip(r.solution.iter()).map(|(a,b)| a*b).sum::<f64>();
            eprintln!("{:10} status={:?} obj={:.5e} iters={} qp_obj_calc={:.5e}",
                name, r.status, r.objective, r.iterations, qp_obj);
        }
    }

    /// LP-dominant な Catastrophic 問題 (QADLITTL/QSCRS8/QSHELL) を LP solver で解く実験。
    /// QP として失敗するが、Q を無視して LP として解けば PASS するか検証する。
    #[test]
    #[ignore]
    fn test_lp_dispatch_catastrophic() {
        // 注意: Q を無視するので得られる x は QP の最適とは異なる。
        // ただし LP-dominant な問題では QP 最適 ≈ LP 最適のはず。
        let problems = [
            ("QADLITTL", 4.803189e5),
            ("QSCRS8",   9.045e2),
            ("QSHELL",   1.572e12),
            ("QBORE3D",  3.110e3),
        ];
        for (name, _expected_obj) in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { eprintln!("{} skipped", name); continue; }
            let prob = parse_qps(path).expect(name);

            // Q を 0 にした「LP 化」した問題を作って solve_qp_with に流す。
            // Q.values を全 0 にして再構築。
            let mut q_zero = prob.q.clone();
            for v in q_zero.values.iter_mut() { *v = 0.0; }
            let prob_lp = crate::qp::problem::QpProblem::new(
                q_zero,
                prob.c.clone(),
                prob.a.clone(),
                prob.b.clone(),
                prob.bounds.clone(),
                prob.constraint_types.clone(),
            ).unwrap();

            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let r = crate::qp::solve_qp_with(&prob_lp, &opts);
            // 元 QP の objective として 0.5 x'Qx + c'x を計算
            let qx = prob.q.mat_vec_mul(&r.solution).unwrap_or(vec![0.0; prob.num_vars]);
            let qp_obj: f64 = 0.5 * qx.iter().zip(r.solution.iter()).map(|(a,b)| a*b).sum::<f64>()
                + prob.c.iter().zip(r.solution.iter()).map(|(a,b)| a*b).sum::<f64>();
            eprintln!("{:10} status={:?} LP_obj={:.5e} QP_obj={:.5e} iters={}",
                name, r.status, r.objective, qp_obj, r.iterations);
        }
    }

    /// DPKLO1 で parser bug 修正と v2 が両立することを確認 (timeout/optimal ok)。
    #[test]
    #[ignore]
    fn test_v2_dpklo1() {
        let path = Path::new("data/maros_meszaros/DPKLO1.QPS");
        if !path.exists() {
            eprintln!("DPKLO1.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse DPKLO1");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(5.0);
        let r = solve_qp_v2(&prob, &opts);
        eprintln!("DPKLO1 v2: status={:?} obj={} iters={}", r.status, r.objective, r.iterations);
        // DPKLO1 が timeout/optimal いずれかで返ってくることを確認 (v2 が hang しない)
        assert!(matches!(r.status, SolveStatus::Optimal | SolveStatus::Timeout));
    }
}
