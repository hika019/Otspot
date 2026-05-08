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

    /// HS21 で v2 が PASS することを確認 (smoke test)。
    #[test]
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

    /// BD-T4 (rank-deficient Q + EmptyCol) で v2 が何で詰まるか調査。
    #[test]
    fn test_v2_bd_t4_diagnose() {
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
        opts.qp_solver = crate::options::QpSolverChoice::IpPmm;
        let r = solve_qp_v2(&problem, &opts);
        eprintln!("BD-T4 v2: status={:?} obj={:.5e} iters={}", r.status, r.objective, r.iterations);
        eprintln!("  x={:?}", r.solution);
        eprintln!("  y={:?}", r.dual_solution);
        eprintln!("  z={:?}", r.bound_duals);
        let view = super::outcome::ProblemView {
            q: &problem.q, a: &problem.a, c: &problem.c, b: &problem.b,
            bounds: &problem.bounds, constraint_types: &problem.constraint_types,
        };
        let kkt = super::kkt::kkt_residual_rel(&view, &r.solution, &r.dual_solution, &r.bound_duals);
        let pres = super::kkt::primal_residual_rel(&view, &r.solution);
        let bv = super::kkt::bound_violation(&problem.bounds, &r.solution);
        eprintln!("  kkt={:.3e} pres={:.3e} bv={:.3e}", kkt, pres, bv);
    }

    /// 1000s で救える見込みのある問題を特定する。
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
            let r = solve_qp_v2(&prob, &opts);
            let view = super::outcome::ProblemView {
                q: &prob.q, a: &prob.a, c: &prob.c, b: &prob.b,
                bounds: &prob.bounds, constraint_types: &prob.constraint_types,
            };
            let kkt = super::kkt::kkt_residual_rel(&view, &r.solution, &r.dual_solution, &r.bound_duals);
            let pres = super::kkt::primal_residual_rel(&view, &r.solution);
            let bv = super::kkt::bound_violation(&prob.bounds, &r.solution);
            // 1000s で救える見込み: KKT < 1e-3 (1000s で更に IPM 進めば 1e-6 達成)
            let prospect_1000s = kkt < 1e-3 && pres < 1e-3 && bv < 1e-6;
            eprintln!("{:10} status={:?} kkt={:.3e} pres={:.3e} prospect_1000s={}",
                name, r.status, kkt, pres, prospect_1000s);
        }
    }

    #[test]
    fn test_osqp_eval_marginal() {
        let problems = [
            // Marginal 5件
            "PRIMALC5", "QSCAGR25", "QSCAGR7", "QSHIP12L", "QSHIP12S",
            // Mid sample
            "QBANDM", "QSHARE1B",
            // Catastrophic sample
            "QADLITTL", "QSCRS8",
        ];
        for name in problems {
            let path_str = format!("data/maros_meszaros/{}.QPS", name);
            let path = Path::new(&path_str);
            if !path.exists() { continue; }
            let prob = parse_qps(path).expect(name);
            let mut opts = SolverOptions::default();
            opts.timeout_secs = Some(60.0);
            let r = solve_qp_v2(&prob, &opts);
            let view = super::outcome::ProblemView {
                q: &prob.q, a: &prob.a, c: &prob.c, b: &prob.b,
                bounds: &prob.bounds, constraint_types: &prob.constraint_types,
            };
            let kkt = super::kkt::kkt_residual_rel(&view, &r.solution, &r.dual_solution, &r.bound_duals);
            let pres = super::kkt::primal_residual_rel(&view, &r.solution);
            let bv = super::kkt::bound_violation(&prob.bounds, &r.solution);
            let pass = kkt < 1e-6 && pres < 1e-6 && bv < 1e-6;
            eprintln!("{:10} status={:?} kkt={:.3e} pres={:.3e} bv={:.3e} PASS_eval={}",
                name, r.status, kkt, pres, bv, pass);
        }
    }

    /// Marginal 問題で x が bound からどの程度離れているか診断する。
    /// snap_tol を決めるための事実確認。
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

    /// QPILOTNO row 14 と row 130 の関係 (A 行ペア確認 + presolve 経路追跡)。
    #[test]
    fn test_v2_qpilotno_row_pair_check() {
        let path = Path::new("data/maros_meszaros/QPILOTNO.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse QPILOTNO");
        // 行 14 と 130 の非ゼロ要素を抽出
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
        // pairwise 一致確認
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

    /// QFORPLAN の dual residual 集中 col を特定する診断。
    /// componentwise pfeas 化以後、col 15 等で Aty != 0 が残る現象を観察。
    /// NO_PRESOLVE=1 で presolve 無効化、QFORPLAN_LONG=1 で 60s 動作。
    #[test]
    fn test_v2_qforplan_dual_residual_diagnose() {
        let path = Path::new("data/maros_meszaros/QFORPLAN.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse QFORPLAN");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(if std::env::var("QFORPLAN_LONG").ok().as_deref() == Some("1") { 60.0 } else { 10.0 });
        if std::env::var("NO_PRESOLVE").ok().as_deref() == Some("1") {
            opts.presolve = false;
        }
        let r = solve_qp_v2(&prob, &opts);
        eprintln!("QFORPLAN status={:?} obj={:.5e} presolve={}", r.status, r.objective, opts.presolve);
        // dual residual 上位 10 col を出力
        use twofloat::TwoFloat;
        let n = prob.num_vars;
        let zero_dd = TwoFloat::from(0.0);
        let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let xv = r.solution[col];
            for k in prob.q.col_ptr[col]..prob.q.col_ptr[col + 1] {
                qx_dd[prob.q.row_ind[k]] = qx_dd[prob.q.row_ind[k]] + TwoFloat::new_mul(prob.q.values[k], xv);
            }
        }
        let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
        let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            for k in prob.a.col_ptr[col]..prob.a.col_ptr[col + 1] {
                let row = prob.a.row_ind[k];
                aty_dd[col] = aty_dd[col] + TwoFloat::new_mul(prob.a.values[k], r.dual_solution[row]);
            }
        }
        let aty: Vec<f64> = aty_dd.iter().map(|&v| f64::from(v)).collect();
        // bound contrib
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
            // 列 j に touching する row 集合のサイズ
            let nrows = prob.a.col_ptr[j+1] - prob.a.col_ptr[*j];
            eprintln!("  j={:5} rel={:.3e} r={:+.3e} qx={:+.3e} aty={:+.3e} bnd={:+.3e} c={:+.3e} x={:+.3e} lb={:+.2e} ub={:+.2e} nrows={}",
                j, rel, res, qx_j, aty_j, bnd_j, c_j, x_j, lb, ub, nrows);
        }
    }

    /// QPILOTNO の primal violation 集中行を特定する診断 (componentwise pfeas 化以後)。
    /// 環境変数 NO_PRESOLVE=1 で presolve 無効化して bug 切り分け。
    #[test]
    fn test_v2_qpilotno_primal_violation_diagnose() {
        let path = Path::new("data/maros_meszaros/QPILOTNO.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse QPILOTNO");
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(30.0);
        if std::env::var("NO_PRESOLVE").ok().as_deref() == Some("1") {
            opts.presolve = false;
        }
        let r = solve_qp_v2(&prob, &opts);
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

    /// DPKLO1 で parser bug 修正と v2 が両立することを確認 (timeout/optimal ok)。
    #[test]
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
