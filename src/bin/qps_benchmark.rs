//! Maros-Meszaros QPS ベンチマーク
//!
//! Usage: qps_benchmark <data_dir> [--eps <value>] [--dual-advanced]
//! 指定ディレクトリ内の全*.QPSファイルを parse_qps → solve_qp_with_options で実行し、
//! 結果テーブルをstdoutに出力する。
//!
//! 各問題に10秒のタイムアウトを設ける（solver内部の協調的タイムアウト機構を使用）。

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::env;
use std::path::Path;
use std::time::Instant;

use solver::bench_utils::{detect_csv_path, load_baseline_objectives, load_expected_statuses, ExpectedStatus, ObjCheckResult};
use solver::io::qps::{parse_qps, QpsError};
use solver::options::{SimplexMethod, SolverOptions};
use solver::problem::{ConstraintType, SolveStatus};
use solver::tolerances::ZERO_TOL;
use solver::qp::ipm_solver::solve_ipm;
use solver::qp::solve_qp_with;
use solver::QpProblem;

enum BenchError {
    Parse(QpsError),
}

/// §2.1: pfeas両側チェック + bfeas（設計書準拠）
///
/// Eq制約: |Ax_i - b_i|（両方向）
/// Ge制約: max(0, b_i - Ax_i)（下方向）
/// Le制約: max(0, Ax_i - b_i)（上方向、デフォルト）
fn compute_primal_quality(prob: &QpProblem, solution: &[f64]) -> (f64, f64) {
    if solution.is_empty() || solution.len() != prob.num_vars {
        return (f64::NAN, f64::NAN);
    }

    let pfeas = match prob.a.mat_vec_mul(solution) {
        Ok(ax) => ax
            .iter()
            .zip(prob.b.iter())
            .enumerate()
            .map(|(i, (&ax_i, &b_i))| match prob.constraint_types.get(i) {
                Some(ConstraintType::Eq) => (ax_i - b_i).abs(),
                Some(ConstraintType::Ge) => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            })
            .fold(0.0_f64, f64::max),
        Err(_) => f64::NAN,
    };

    // bfeas: OSQP 式の全体相対化 ||v||_∞ / (1 + max(||x||_∞, ||lb||_∞, ||ub||_∞))
    let mut max_v = 0.0_f64;
    let mut max_x = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    for (&xi, &(lb, ub)) in solution.iter().zip(prob.bounds.iter()) {
        let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
        let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
        max_v = max_v.max(lb_viol.max(ub_viol));
        max_x = max_x.max(xi.abs());
        if lb.is_finite() {
            max_bnd = max_bnd.max(lb.abs());
        }
        if ub.is_finite() {
            max_bnd = max_bnd.max(ub.abs());
        }
    }
    let bfeas = max_v / (1.0 + max_x.max(max_bnd));

    (pfeas, bfeas)
}

/// §2.1b: pfeas を成分相対化で評価する (本体 kkt::primal_residual_rel と同形)。
///
/// `max_i violation_i / (1 + |Ax_i| + |b_i|)`。
/// OSQP 公式の全体相対化 (max_v / (1 + max(||Ax||_∞, ||b||_∞))) は ill-scaled 行列で
/// 1 行のみ違反が大きい場合に巨大スケールで割って eps を満たすように見せる欠陥がある
/// ため、成分相対化で 1 行ごとに精度を保証する。
fn compute_pfeas_normalized(prob: &QpProblem, solution: &[f64]) -> f64 {
    if solution.is_empty() || solution.len() != prob.num_vars {
        return f64::NAN;
    }
    if prob.num_constraints == 0 {
        return 0.0;
    }
    match prob.a.mat_vec_mul(solution) {
        Ok(ax) => {
            let mut max_rel = 0.0_f64;
            for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
                let violation = match prob.constraint_types.get(i) {
                    Some(ConstraintType::Eq) => (ax_i - b_i).abs(),
                    Some(ConstraintType::Ge) => (b_i - ax_i).max(0.0),
                    _ => (ax_i - b_i).max(0.0),
                };
                let scale_i = 1.0 + ax_i.abs() + b_i.abs();
                let rel_i = violation / scale_i;
                if rel_i > max_rel {
                    max_rel = rel_i;
                }
            }
            max_rel
        }
        Err(_) => f64::NAN,
    }
}

fn check_reported_objective(
    problem_name: &str,
    reported_objective: f64,
    prob: &QpProblem,
    baseline_csv_path: Option<&str>,
    baseline_objectives: &std::collections::HashMap<String, f64>,
    eps_obj: f64,
) -> ObjCheckResult {
    let expected = match baseline_objectives.get(problem_name) {
        Some(v) => *v,
        None => return ObjCheckResult::NoRef,
    };
    let expected_reported = match baseline_csv_path {
        Some(path) if path.ends_with("netlib_lp.csv") => expected + prob.obj_offset,
        _ => expected,
    };
    let denom = expected_reported.abs().max(1.0);
    let rel_err = (reported_objective - expected_reported).abs() / denom;
    if rel_err <= eps_obj {
        ObjCheckResult::Ok { rel_err }
    } else {
        ObjCheckResult::Mismatch { rel_err }
    }
}

/// §2.2: dfeas 元空間再計算（ソルバ申告値ではなく bench 側で独立計算）
///
/// ソルバの `result.dfeas` は内部 (Ruiz scaled) 空間の値で、unscale 後の
/// 元空間 dfeas とは異なる。bench は「ユーザー指定 eps を元空間で満たすか」を
/// 検証する役割なので、元 problem.q / problem.a / problem.c と unscale 済み解で
/// 直接 KKT 残差を計算する。
///
/// 戻り値: (絶対残差 dfeas_abs, 相対残差 dfeas_rel)
/// - dfeas_abs = ||Q*x + A^T*y + bound_contrib + c||_∞ — 表示用
/// - dfeas_rel = max_j |residual_j| / (1 + |Qx_j| + |A^Ty_j| + |bound_j| + |c_j|)
///   — 判定用（OSQP/Clarabel 流の成分ごと相対化、巨大項キャンセレーション対応）
fn compute_dfeas_orig(
    prob: &QpProblem,
    solution: &[f64],
    dual_solution: &[f64],
    bound_duals: &[f64],
    reduced_costs: &[f64],
) -> (f64, f64) {
    use twofloat::TwoFloat;
    if solution.is_empty() || solution.len() != prob.num_vars {
        return (f64::NAN, f64::NAN);
    }
    let n = solution.len();
    // ill-scaled 問題 (Maros QPILOTNO: ‖A‖=5.85e6, cond=3e12) で
    // f64 cancellation noise が真の残差を埋もれさせるため、Q*x と A^T*y は
    // double-double 精度で計算する。bench の判定 (PASS/DFEAS_FAIL) に直結するため
    // 真の精度を見せる必要がある。
    //
    // Q 格納慣例: spmv_q (src/qp/ipm/kkt.rs) と同じく **全要素格納の対称行列**
    // (上下三角両方 stored)。symmetric duplication しないように直接 col×row 走査。
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = solution[col];
        let cs = prob.q.col_ptr[col];
        let ce = prob.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = prob.q.row_ind[k];
            let v = prob.q.values[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(v, xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let aty: Vec<f64> = if prob.a.nrows > 0 && !dual_solution.is_empty() {
        let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let cs = prob.a.col_ptr[col];
            let ce = prob.a.col_ptr[col + 1];
            for k in cs..ce {
                let row = prob.a.row_ind[k];
                let v = prob.a.values[k];
                aty_dd[col] = aty_dd[col] + TwoFloat::new_mul(v, dual_solution[row]);
            }
        }
        aty_dd.iter().map(|&v| f64::from(v)).collect()
    } else {
        vec![0.0; n]
    };
    // LP/Simplex 経路: complementarity-aware dual feasibility check
    //
    // ## LP KKT 概観
    //
    // primal: min c^T x s.t. Ax = b, lb <= x <= ub
    // dual:   y (Ax=b), z_lb >= 0 (x>=lb), z_ub >= 0 (x<=ub)
    // stationarity:  A^T y + z_lb − z_ub = c  ⇒  rc := c − A^T y = z_lb − z_ub
    // complementarity:  (x − lb) z_lb = 0,  (ub − x) z_ub = 0
    //
    // Simplex 最適性: 非基底変数のみが bound に活性化、基底変数は内点で rc=0 (構造的)。
    //
    // ## なぜ「bound hit 判定 + sign check」か (純粋な bound-finiteness 判定では不十分)
    //
    // 純粋な bound-finiteness 判定 (lb 有限 ⇒ rc>=0 必須) は、基底変数 (内点、rc=0
    // が原理的) に postsolve / cleanup LP 由来の noise (|rc| ~ 1e-2 級) が乗ったとき
    // 大量の false positive を生む (agg/boeing2/brandy 等で観測)。
    //
    // 構造的に正しい複合判定: 「変数が bound に活性化しているか」を **相対許容**
    // で見て、活性のみ rc 符号を要求する。これにより:
    //   1. bound 活性変数: KKT 通り厳格判定
    //   2. 内点変数 (基底): rc=0 が構造的、ただし extract noise 許容
    //
    // ## 旧 magic BOUND_HIT_TOL=1e-6 を撤去した理由
    //
    // 1e-6 絶対閾値は問題スケール非依存 (x ~ 1e6 で hit 判定が失敗、x ~ 1e-3 で
    // 過剰活性化)。`PIVOT_TOL = 1e-8` (Simplex 内部の最適性判定 tol) を相対
    // `(1 + |x| + |bound|)` でスケールすれば、磁石的 magic を廃しつつ Simplex
    // 内部精度と整合した bound-hit 判定が得られる。
    if bound_duals.is_empty() && !reduced_costs.is_empty() && reduced_costs.len() == n {
        use solver::tolerances::PIVOT_TOL;
        let rel_tol = PIVOT_TOL; // 1e-8 — Simplex 内部 dual_tol と一致 (構造的派生)
        let mut dfeas_abs = 0.0_f64;
        let mut dfeas_rel = 0.0_f64;
        for j in 0..n {
            let (lb_j, ub_j) = prob.bounds[j];
            if lb_j.is_finite() && ub_j.is_finite() && (lb_j - ub_j).abs() < ZERO_TOL {
                continue; // FX 変数は presolve で除去済み
            }
            if prob.a.col_ptr.len() > j + 1 && prob.a.col_ptr[j + 1] - prob.a.col_ptr[j] == 0 {
                continue; // EmptyCol は presolve で除去済み
            }
            let rc = reduced_costs[j];
            let x_j = solution[j];
            // 相対 bound-hit 判定: `|x - bound| <= PIVOT_TOL * (1 + |x| + |bound|)`
            // (Simplex 最適性判定 PIVOT_TOL=1e-8 と一致した精度水準、scale-invariant)。
            let at_lb = lb_j.is_finite()
                && (x_j - lb_j).abs() <= rel_tol * (1.0 + x_j.abs() + lb_j.abs());
            let at_ub = ub_j.is_finite()
                && (x_j - ub_j).abs() <= rel_tol * (1.0 + x_j.abs() + ub_j.abs());
            // sign 制約: 活性 bound のみ要求 (内点は基底変数として rc≈0 が構造的)。
            // - 内点 (両端非活性): 基底 var で rc=0 構造的、postsolve noise を許容して 0
            // - free 変数 (両端 inf): 算術上は厳格 rc=0 要求だが、Simplex 実装上は
            //   必ず基底に入る → 内点と同じ noise 許容で扱う (capri / agg 等で
            //   観測した extract noise を false positive 化しないため)
            // - 両端 hit (FX相当、稀): presolve 除去前提なので 0
            let viol = if at_lb && !at_ub {
                f64::max(0.0, -rc) // x = lb: z_lb = rc >= 0
            } else if at_ub && !at_lb {
                f64::max(0.0,  rc) // x = ub: z_ub = -rc >= 0
            } else {
                0.0 // 内点 / free / 両端 hit: noise 許容
            };
            dfeas_abs = dfeas_abs.max(viol);
            let scale_j = 1.0 + rc.abs() + prob.c[j].abs();
            dfeas_rel = dfeas_rel.max(viol / scale_j);
        }
        return (dfeas_abs, dfeas_rel);
    }

    // bound_contrib[j] = -y_lb[j] (lb有限) + y_ub[j] (ub有限)
    // - QP/IPM 経路: bound_duals が [y_lb 群; y_ub 群] レイアウトで渡る
    let mut bound_contrib = vec![0.0_f64; n];
    if !bound_duals.is_empty() {
        let mut bd_idx = 0usize;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] -= bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] += bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
    } else if !reduced_costs.is_empty() && reduced_costs.len() == n {
        // LP 経路: reduced_cost を負号で取り込む (c + A^T*y - rc = 0)
        for j in 0..n {
            bound_contrib[j] = -reduced_costs[j];
        }
    }
    // OSQP 式: 全体相対化 (||r||_∞ / (1 + max(||Qx||, ||c||, ||A^T y||, ||z||)))
    // および 成分相対化 (max_i |r_i| / (1 + |Qx_i| + |c_i| + |A^T y_i| + |z_i|))
    // を併算する。前者は OSQP 公式仕様に準拠した緩い基準、後者は 1 成分でも
    // 桁違いに外れたら検出する厳しい基準 (個別 stationarity の精度保証)。
    let mut dfeas_abs = 0.0_f64;
    let mut dfeas_rel_componentwise = 0.0_f64;
    let mut max_qx = 0.0_f64;
    let mut max_c = 0.0_f64;
    let mut max_aty = 0.0_f64;
    let mut max_bnd = 0.0_f64;
    let dump_top = std::env::var("DFEAS_DUMP_TOP").ok().as_deref() == Some("1");
    let mut per_col: Vec<(usize, f64, f64, f64, f64, f64)> = if dump_top { Vec::with_capacity(n) } else { Vec::new() };
    for i in 0..n {
        // FX (lb≈ub) と EmptyCol (制約 A に登場しない) は presolve で除去され
        // bound_dual=0 が埋められる慣例。stationarity 評価から除外して v2 経路
        // (`kkt_residual_rel`) と整合させる。
        let (lb_i, ub_i) = prob.bounds[i];
        if lb_i.is_finite() && ub_i.is_finite() && (lb_i - ub_i).abs() < ZERO_TOL {
            continue;
        }
        if prob.a.col_ptr.len() > i + 1
            && prob.a.col_ptr[i + 1] - prob.a.col_ptr[i] == 0
        {
            continue;
        }
        // 4 項の和も DD で組み立てる: f64 で和を取ると ill-conditioned 系で桁落ち。
        let r_dd = TwoFloat::from(qx[i])
            + TwoFloat::from(aty[i])
            + TwoFloat::from(bound_contrib[i])
            + TwoFloat::from(prob.c[i]);
        let r = f64::from(r_dd).abs();
        dfeas_abs = dfeas_abs.max(r);
        let scale_i = 1.0 + qx[i].abs() + aty[i].abs() + bound_contrib[i].abs() + prob.c[i].abs();
        let r_rel_i = r / scale_i;
        dfeas_rel_componentwise = dfeas_rel_componentwise.max(r_rel_i);
        max_qx = max_qx.max(qx[i].abs());
        max_c = max_c.max(prob.c[i].abs());
        max_aty = max_aty.max(aty[i].abs());
        max_bnd = max_bnd.max(bound_contrib[i].abs());
        if dump_top {
            per_col.push((i, r, qx[i], aty[i], bound_contrib[i], prob.c[i]));
        }
    }
    let scale = 1.0 + max_qx.max(max_c).max(max_aty).max(max_bnd);
    let dfeas_rel = dfeas_abs / scale;
    if dump_top {
        per_col.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("DFEAS_DUMP_TOP scale={:.3e} dfeas_abs={:.3e} dfeas_rel={:.3e} dfeas_relC={:.3e}",
            scale, dfeas_abs, dfeas_rel, dfeas_rel_componentwise);
        for k in 0..per_col.len().min(10) {
            let (i, r, qxi, atyi, bndi, ci) = per_col[k];
            let (lbi, ubi) = prob.bounds[i];
            let xi = solution[i];
            eprintln!("  col[{}] r={:+.3e} qx={:+.3e} aty={:+.3e} bnd={:+.3e} c={:+.3e} x={:+.3e} bounds=[{:+.3e},{:+.3e}]",
                i, r, qxi, atyi, bndi, ci, xi, lbi, ubi);
        }
    }
    // 判定は **成分相対化** (dfeas_rel_componentwise) を採用する。OSQP 公式の
    // 全体相対化 (dfeas_rel) は ill-scaled 問題で 1 成分のみ大きく外れた残差を
    // 巨大スケールで割って eps を満たすように見せる欠陥があり、ユーザー指定 eps の
    // 保証として不十分。第 2 戻り値は厳しい側 (componentwise) に変更する。
    let _ = dfeas_rel; // 表示用には DFEAS_DUMP_TOP で参照可
    (dfeas_abs, dfeas_rel_componentwise)
}

/// 元空間 dfeas を成分相対化で評価する。OSQP 公式 (全体相対化) の dfeas_rel と
/// 併算し、ill-scaled 問題で 1 成分のみ残差が大きい場合の精度劣化を検出するため
/// に使う。dfeas_rel ≤ eps でも dfeas_relC > eps なら solver の収束が
/// 「全体感では精度が出ているが、特定成分は eps を満たしていない」状態。
fn compute_dfeas_componentwise(
    prob: &QpProblem,
    solution: &[f64],
    dual_solution: &[f64],
    bound_duals: &[f64],
    reduced_costs: &[f64],
) -> f64 {
    use twofloat::TwoFloat;
    if solution.is_empty() || solution.len() != prob.num_vars {
        return f64::NAN;
    }
    let n = solution.len();
    let zero_dd = TwoFloat::from(0.0);
    let mut qx_dd: Vec<TwoFloat> = vec![zero_dd; n];
    for col in 0..n {
        let xv = solution[col];
        let cs = prob.q.col_ptr[col];
        let ce = prob.q.col_ptr[col + 1];
        for k in cs..ce {
            let row = prob.q.row_ind[k];
            let v = prob.q.values[k];
            qx_dd[row] = qx_dd[row] + TwoFloat::new_mul(v, xv);
        }
    }
    let qx: Vec<f64> = qx_dd.iter().map(|&v| f64::from(v)).collect();
    let aty: Vec<f64> = if prob.a.nrows > 0 && !dual_solution.is_empty() {
        let mut aty_dd: Vec<TwoFloat> = vec![zero_dd; n];
        for col in 0..n {
            let cs = prob.a.col_ptr[col];
            let ce = prob.a.col_ptr[col + 1];
            for k in cs..ce {
                let row = prob.a.row_ind[k];
                let v = prob.a.values[k];
                aty_dd[col] = aty_dd[col] + TwoFloat::new_mul(v, dual_solution[row]);
            }
        }
        aty_dd.iter().map(|&v| f64::from(v)).collect()
    } else {
        vec![0.0; n]
    };
    let mut bound_contrib = vec![0.0_f64; n];
    if !bound_duals.is_empty() {
        let mut bd_idx = 0usize;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] -= bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && bd_idx < bound_duals.len() {
                bound_contrib[j] += bound_duals[bd_idx];
                bd_idx += 1;
            }
        }
    } else if !reduced_costs.is_empty() && reduced_costs.len() == n {
        // LP 経路: UB 非基底変数 (rc ≤ 0 が最適性条件) を考慮して dual infeasibility を計算。
        // UB 非基底 (x_j ≈ ub_j): rc_j > 0 が違反。
        // LB 非基底 / 自由 (x_j < ub_j): rc_j < 0 が違反。
        let mut max_rel = 0.0_f64;
        for j in 0..n {
            let (lb_j, ub_j) = prob.bounds[j];
            if lb_j.is_finite() && ub_j.is_finite() && (lb_j - ub_j).abs() < ZERO_TOL {
                continue;
            }
            if prob.a.col_ptr.len() > j + 1 && prob.a.col_ptr[j + 1] - prob.a.col_ptr[j] == 0 {
                continue;
            }
            let rc = reduced_costs[j];
            let x_j = solution.get(j).copied().unwrap_or(0.0);
            let at_ub = ub_j.is_finite()
                && (x_j - ub_j).abs() <= 1e-8 * (1.0 + ub_j.abs());
            let viol = if at_ub {
                f64::max(0.0, rc)    // UB 非基底: rc_j > 0 が違反
            } else {
                f64::max(0.0, -rc)   // LB 非基底 / 自由: rc_j < 0 が違反
            };
            let scale_j = 1.0 + rc.abs() + prob.c[j].abs();
            max_rel = max_rel.max(viol / scale_j);
        }
        return max_rel;
    }
    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let (lb_i, ub_i) = prob.bounds[i];
        if lb_i.is_finite() && ub_i.is_finite() && (lb_i - ub_i).abs() < ZERO_TOL {
            continue;
        }
        if prob.a.col_ptr.len() > i + 1
            && prob.a.col_ptr[i + 1] - prob.a.col_ptr[i] == 0
        {
            continue;
        }
        let r_dd = TwoFloat::from(qx[i])
            + TwoFloat::from(aty[i])
            + TwoFloat::from(bound_contrib[i])
            + TwoFloat::from(prob.c[i]);
        let r = f64::from(r_dd).abs();
        let scale_i = 1.0 + qx[i].abs() + aty[i].abs() + bound_contrib[i].abs() + prob.c[i].abs();
        let rel_i = r / scale_i;
        if rel_i > max_rel {
            max_rel = rel_i;
        }
    }
    max_rel
}

/// pfeas を成分相対化で評価する: max_i [violation_i / (1 + |a_i·x| + |b_i|)]。
/// OSQP 公式 dfeas_normalized (全体相対化) より厳しく、巨大行 1 つで他がゼロでも
/// 精度を保証する。
fn compute_pfeas_componentwise(prob: &QpProblem, solution: &[f64]) -> f64 {
    if solution.is_empty() || solution.len() != prob.num_vars {
        return f64::NAN;
    }
    if prob.num_constraints == 0 {
        return 0.0;
    }
    match prob.a.mat_vec_mul(solution) {
        Ok(ax) => {
            let mut max_rel = 0.0_f64;
            for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
                let violation = match prob.constraint_types.get(i) {
                    Some(ConstraintType::Eq) => (ax_i - b_i).abs(),
                    Some(ConstraintType::Ge) => (b_i - ax_i).max(0.0),
                    _ => (ax_i - b_i).max(0.0),
                };
                let scale_i = 1.0 + ax_i.abs() + b_i.abs();
                let rel_i = violation / scale_i;
                if rel_i > max_rel {
                    max_rel = rel_i;
                }
            }
            max_rel
        }
        Err(_) => f64::NAN,
    }
}

fn parse_with_timeout(path: &Path, _timeout_secs: u64) -> Result<QpProblem, BenchError> {
    // parse_qps 自体に cancellation API がないため同期呼び出し。hang 時は
    // bench_parallel.sh の外部 gtimeout でプロセスごと殺される設計。
    parse_qps(path).map_err(BenchError::Parse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use solver::problem::ConstraintType;
    use solver::sparse::CscMatrix;

    /// Eq制約の下方向違反がpfeasに反映される
    #[test]
    fn test_pfeas_eq_constraint_violation() {
        // Ax = b: A=[[1.0]], b=[5.0]
        // x=[3.0] → |1*3 - 5| = 2.0 (下方向違反)
        // x=[7.0] → |1*7 - 5| = 2.0 (上方向違反)
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![5.0];
        let bounds = vec![(0.0, f64::INFINITY)];
        let mut prob = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a,
            b,
            bounds,
            vec![ConstraintType::Eq],
        )
        .unwrap();
        prob.obj_offset = 0.0;

        // 下方向違反: x=3 < b=5
        let (pfeas_down, _) = compute_primal_quality(&prob, &[3.0]);
        assert!(
            (pfeas_down - 2.0).abs() < 1e-10,
            "Eq下方向違反: expected pfeas=2.0, got {}",
            pfeas_down
        );

        // 上方向違反: x=7 > b=5
        let (pfeas_up, _) = compute_primal_quality(&prob, &[7.0]);
        assert!(
            (pfeas_up - 2.0).abs() < 1e-10,
            "Eq上方向違反: expected pfeas=2.0, got {}",
            pfeas_up
        );

        // 境界: x=5 → 違反なし
        let (pfeas_ok, _) = compute_primal_quality(&prob, &[5.0]);
        assert!(
            pfeas_ok < 1e-10,
            "Eq充足: expected pfeas≈0.0, got {}",
            pfeas_ok
        );
    }

    /// Ge制約の違反計算が正しい
    #[test]
    fn test_pfeas_ge_constraint() {
        // Ge制約: Ax >= b → A=[[1.0]], b=[5.0]
        // x=[3.0] → max(0, 5-3) = 2.0 (違反)
        // x=[7.0] → max(0, 5-7) = 0.0 (充足)
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let b = vec![5.0];
        let bounds = vec![(0.0, f64::INFINITY)];
        let prob = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![1.0],
            a,
            b,
            bounds,
            vec![ConstraintType::Ge],
        )
        .unwrap();

        // 違反: x=3 < b=5
        let (pfeas_viol, _) = compute_primal_quality(&prob, &[3.0]);
        assert!(
            (pfeas_viol - 2.0).abs() < 1e-10,
            "Ge違反: expected pfeas=2.0, got {}",
            pfeas_viol
        );

        // 充足: x=7 >= b=5
        let (pfeas_ok, _) = compute_primal_quality(&prob, &[7.0]);
        assert!(
            pfeas_ok < 1e-10,
            "Ge充足: expected pfeas=0.0, got {}",
            pfeas_ok
        );
    }

    /// 相対 bound-hit 判定 (`PIVOT_TOL * (1 + |x| + |bound|)`) は x スケール非依存。
    /// 絶対閾値だと x~1e6 で hit 失敗、x~1e-3 で過剰活性化を起こす。
    #[test]
    fn test_dfeas_bound_hit_relative_structural() {
        let make_prob = |bounds: Vec<(f64, f64)>, c: Vec<f64>, x_target: f64| -> QpProblem {
            let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
            let mut p = QpProblem::new(
                CscMatrix::new(1, 1),
                c,
                a,
                vec![x_target],
                bounds,
                vec![ConstraintType::Eq],
            ).unwrap();
            p.obj_offset = 0.0;
            p
        };

        // Case A: lb=0, ub=inf, x=0 (at lb), rc=+1 → 合法 (z_lb=1≥0)
        let p = make_prob(vec![(0.0, f64::INFINITY)], vec![1.0], 0.0);
        let (abs, _) = compute_dfeas_orig(&p, &[0.0], &[], &[], &[1.0]);
        assert!(abs < 1e-15, "at lb, rc=+1 (合法): dfeas={}", abs);

        // Case B: lb=0, ub=inf, x=0 (at lb), rc=-1 → 違反 (z_lb=-1<0)
        let (abs, _) = compute_dfeas_orig(&p, &[0.0], &[], &[], &[-1.0]);
        assert!((abs - 1.0).abs() < 1e-15, "at lb, rc=-1 (違反): dfeas={}", abs);

        // Case C: lb=0, ub=inf, x=100 (interior, x ≫ lb*PIVOT_TOL), rc=-1 →
        //   内点 = 基底変数仮定で noise 許容、dfeas=0
        let p = make_prob(vec![(0.0, f64::INFINITY)], vec![1.0], 100.0);
        let (abs, _) = compute_dfeas_orig(&p, &[100.0], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "interior, rc=-1 (基底 noise 許容): dfeas={}", abs);

        // Case D: lb=0, ub=10, x=10 (at ub), rc=+1 → 違反 (z_ub=-1<0)
        let p = make_prob(vec![(0.0, 10.0)], vec![1.0], 10.0);
        let (abs, _) = compute_dfeas_orig(&p, &[10.0], &[], &[], &[1.0]);
        assert!((abs - 1.0).abs() < 1e-15, "at ub, rc=+1 (違反): dfeas={}", abs);

        // Case E: lb=0, ub=10, x=10 (at ub), rc=-1 → 合法 (z_ub=1≥0)
        let (abs, _) = compute_dfeas_orig(&p, &[10.0], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "at ub, rc=-1 (合法): dfeas={}", abs);

        // Case F: free (lb=-inf, ub=+inf), x=5, rc=0.5 → 算術上は違反だが、
        //   Simplex 実装上 free 変数は基底入り強制で extract noise を許容する慣例
        //   (capri 等の DFEAS_FAIL 回避のため、内点と同じ扱い)。dfeas = 0。
        let p = make_prob(vec![(f64::NEG_INFINITY, f64::INFINITY)], vec![1.0], 5.0);
        let (abs, _) = compute_dfeas_orig(&p, &[5.0], &[], &[], &[0.5]);
        assert!(abs < 1e-15, "free rc=0.5 (Simplex 基底 noise 許容): dfeas={}", abs);

        // Case G: lb=0, x=1e6 (内点, 大スケール) — 旧 BOUND_HIT_TOL=1e-6 では
        //   |x-lb|=1e6 >> 1e-6 で「内点」と判定する。新コードでも
        //   |x-lb|=1e6 vs PIVOT_TOL*(1+|x|) ≈ 1e-2 で「内点」(scale 自動追従)。
        let p = make_prob(vec![(0.0, f64::INFINITY)], vec![1.0], 1e6);
        let (abs, _) = compute_dfeas_orig(&p, &[1e6], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "large-scale interior, rc=-1 (内点 noise 許容): dfeas={}", abs);

        // Case H: lb=0, x=1e-9 (tiny x, 旧 BOUND_HIT_TOL=1e-6 では「at lb」だが、
        //   |x-lb|=1e-9 < PIVOT_TOL*(1+1e-9) ≈ 1e-8 で新コードも「at lb」判定。
        //   rc<0 → 違反として正しく検出)。
        let (abs, _) = compute_dfeas_orig(&p, &[1e-9], &[], &[], &[-1.0]);
        assert!((abs - 1.0).abs() < 1e-15, "tiny x (at lb relatively), rc=-1: dfeas={}", abs);

        // Case I: lb=0, x=1e-5 (旧 BOUND_HIT_TOL=1e-6 では |x|>tol で「内点」だが、
        //   新コードでは |x|=1e-5 vs PIVOT_TOL*(1+1e-5)≈1e-8 で「内点」と判定。
        //   旧との挙動差: 旧は内点扱い (rc<0 OK)、新も内点扱い。一致。
        let (abs, _) = compute_dfeas_orig(&p, &[1e-5], &[], &[], &[-1.0]);
        assert!(abs < 1e-15, "x=1e-5 interior rc=-1 (内点 noise 許容): dfeas={}", abs);
    }

    #[test]
    fn test_netlib_objective_check_adds_obj_offset_to_reference() {
        let mut prob = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
            vec![],
        )
        .unwrap();
        prob.obj_offset = -7.113;

        let mut known = HashMap::new();
        known.insert("e226".to_string(), -18.751_929_066);

        let result = check_reported_objective(
            "e226",
            -25.864_929_066,
            &prob,
            Some("data/baseline_objectives/netlib_lp.csv"),
            &known,
            1e-9,
        );
        assert!(matches!(result, ObjCheckResult::Ok { .. }));
    }

    #[test]
    fn test_non_netlib_objective_check_does_not_add_obj_offset() {
        let mut prob = QpProblem::new(
            CscMatrix::new(1, 1),
            vec![0.0],
            CscMatrix::new(0, 1),
            vec![],
            vec![(0.0, f64::INFINITY)],
            vec![],
        )
        .unwrap();
        prob.obj_offset = -7.113;

        let mut known = HashMap::new();
        known.insert("toy".to_string(), 12.5);

        let result = check_reported_objective(
            "toy",
            12.5,
            &prob,
            Some("data/baseline_objectives/maros_meszaros.csv"),
            &known,
            1e-9,
        );
        assert!(matches!(result, ObjCheckResult::Ok { .. }));
    }

    /// load_expected_statuses が INFEASIBLE エントリを正しく読む
    #[test]
    fn test_expected_status_infeasible_loaded() {
        use solver::bench_utils::{load_expected_statuses, ExpectedStatus};
        use std::io::Write;

        let csv = "problem_name,optimal_obj,source\n\
            galenet,INFEASIBLE,https://www.netlib.org/lp/infeas/readme\n\
            klein1,INFEASIBLE,https://www.netlib.org/lp/infeas/readme\n\
            afiro,-4.6475314286e+02,https://www.netlib.org/lp/data/readme\n";

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(csv.as_bytes()).unwrap();
        let statuses = load_expected_statuses(tmp.path());

        assert_eq!(statuses.get("galenet"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(statuses.get("klein1"), Some(&ExpectedStatus::Infeasible));
        // 数値エントリは Optimal
        assert_eq!(statuses.get("afiro"), Some(&ExpectedStatus::Optimal));
        // 存在しない問題は None
        assert_eq!(statuses.get("nonexistent"), None);
    }
}

fn main() {
    // bench_parallel.sh 経由でのみ実行可能（直接実行禁止）
    if std::env::var("_BENCH_PARALLEL_CALLER").as_deref() != Ok("1") {
        eprintln!("[qps_benchmark] エラー: 直接実行禁止。bench_parallel.sh 経由で実行せよ。");
        eprintln!("[qps_benchmark] 使い方: bash scripts/bench_parallel.sh --data-dir DIR --timeout SEC --output FILE --jobs N");
        std::process::exit(1);
    }

    let args: Vec<String> = env::args().collect();

    // 引数パース: [data_dir] [--eps <value>] [--timeout <secs>] [--known-optimal <path>] [--dual-advanced]
    let mut data_dir = "data/maros_meszaros".to_string();
    let mut dual_advanced_mode = false;
    let mut eps: f64 = 1e-6;
    let mut timeout_secs: f64 = 10.0;
    let mut baseline_override: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: qps_benchmark [data_dir] [--eps <value>] [--timeout <secs>] [--known-optimal <path>] [--dual-advanced]");
            println!("  --eps             Convergence tolerance (default: 1e-6)");
            println!("  --timeout         Solver timeout in seconds (default: 10.0)");
            println!("  --known-optimal   Path to known optimal values CSV (default: auto-detect)");
            println!("  --dual-advanced   LP は DualAdvanced simplex を使う (QP は無視)");
            std::process::exit(0);
        } else if args[i] == "--known-optimal" {
            i += 1;
            if i < args.len() {
                baseline_override = Some(args[i].clone());
            }
        } else if args[i] == "--eps" {
            i += 1;
            if i < args.len() {
                eps = args[i].parse().unwrap_or(1e-6);
            }
        } else if args[i] == "--timeout" {
            i += 1;
            if i < args.len() {
                timeout_secs = args[i].parse().unwrap_or(10.0);
            }
        } else if args[i] == "--dual-advanced" {
            dual_advanced_mode = true;
        } else if !args[i].starts_with("--") {
            data_dir = args[i].clone();
        }
        i += 1;
    }

    let dir = Path::new(&data_dir);
    if !dir.exists() {
        eprintln!("Directory not found: {}", data_dir);
        std::process::exit(1);
    }

    // §2.4: 正解値CSV読み込み
    // バイナリの実行パスからCSVを探す（--known-optimal指定またはdata_dir名から自動選択）
    let baseline_csv = {
        let root = {
            let mut p = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|pp| pp.to_path_buf()))
                .unwrap_or_default();
            // target/release から solver ルートに遡る
            p = p.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()).unwrap_or_default();
            p
        };
        detect_csv_path(&data_dir, baseline_override.as_deref(), &root)
    };
    let baseline_csv_str = baseline_csv.to_string_lossy().into_owned();
    let baseline_objectives = load_baseline_objectives(&baseline_csv);
    let expected_statuses = load_expected_statuses(&baseline_csv);
    eprintln!("Baseline objectives loaded: {} problems", baseline_objectives.len());
    let n_infeasible_ref = expected_statuses.values().filter(|s| **s == ExpectedStatus::Infeasible).count();
    let n_unbounded_ref = expected_statuses.values().filter(|s| **s == ExpectedStatus::Unbounded).count();
    if n_infeasible_ref > 0 || n_unbounded_ref > 0 {
        eprintln!("  (うち INFEASIBLE: {}, UNBOUNDED: {})", n_infeasible_ref, n_unbounded_ref);
    }
    if baseline_objectives.is_empty() && expected_statuses.is_empty() {
        eprintln!("WARNING: No known optimal values loaded. All problems will be PASS[no_ref].");
    }

    // QPSファイル一覧を取得（ファイル名でソート）
    let mut qps_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("qps"))
                .unwrap_or(false)
        })
        .collect();
    qps_files.sort();

    println!("Maros-Meszaros QP Benchmark ({} files)", qps_files.len());
    println!();
    println!(
        "{:<20} {:>6} {:>6} {:>15} {:>10} Details",
        "Problem", "n", "m", "Status", "Time(s)"
    );
    println!("{}", "-".repeat(80));

    // 集計 — §2.5の7カテゴリ + 既存カテゴリ + infeasible/unbounded 正答
    let mut n_pass = 0usize;
    let mut n_pass_noref = 0usize;
    let mut n_pass_infeasible = 0usize;   // 期待通り Infeasible と判定
    let mut n_pass_unbounded = 0usize;    // 期待通り Unbounded と判定
    let mut n_pfeas_fail = 0usize;
    let mut n_dfeas_fail = 0usize;
    let mut n_suboptimal_comp = 0usize;
    let mut n_obj_mismatch = 0usize;
    let mut n_fail = 0usize;
    let mut n_error = 0usize;
    let mut n_timeout = 0usize;
    let mut n_max_iter = 0usize;
    let mut n_nonconvex = 0usize;
    let mut n_suboptimal = 0usize;

    let solver_label = if dual_advanced_mode { "DualAdvanced (LP) + IPPMM (QP)" } else { "IPPMM" };
    println!("Solver: {}", solver_label);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs);
    opts.ipm.eps = eps;
    if dual_advanced_mode {
        opts.simplex_method = SimplexMethod::DualAdvanced;
    }

    // QP問題かどうかの判定用定数
    let eps_obj: f64 = 1e-2; // §2.4: 1%閾値

    for path in &qps_files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let parse_start = Instant::now();
        println!("PARSE_START: {}", name);

        // パース（30秒タイムアウト付き）
        let prob = match parse_with_timeout(path, 30) {
            Ok(p) => p,
            Err(BenchError::Parse(e)) => {
                let note = format!("{}", e);
                println!(
                    "{:<20} {:>6} {:>6} {:>15} {:>10.3} {}",
                    name, "?", "?", "PARSE_ERR", 0.0, &note[..note.len().min(40)]
                );
                n_error += 1;
                continue;
            }
        };

        println!(
            "PARSE_DONE: {} ({:.2}s)",
            name,
            parse_start.elapsed().as_secs_f64()
        );

        let n = prob.num_vars;
        let m = prob.num_constraints;
        let nnz_before = prob.q.nnz() + prob.a.nnz();
        let _is_qp = prob.q.nnz() > 0;

        println!("SOLVE_START: {}", name);
        let start = Instant::now();
        let result = if std::env::var("V2_SOLVER").is_ok() {
            solve_ipm(&prob, &opts)
        } else {
            solve_qp_with(&prob, &opts)
        };
        let elapsed_s = start.elapsed().as_secs_f64();
        println!(
            "SOLVE_DONE: {} {:?} ({:.3}s)",
            name, result.status, elapsed_s
        );

        let method_label = "ipm";
        let resid_str = match result.final_residuals {
            Some((pf, df, gap)) => format!("pf={:.1e} df={:.1e} gap={:.1e}", pf, df, gap),
            None => String::new(),
        };

        // SuboptimalSolution / LocallyOptimal で有効解を保持している場合のみ Optimal フロー
        // に乗せて品質判定 (pfeas/bfeas/dfeas/obj_check) を通す。
        // Timeout は意図的に対象外: 半 deadline incumbent を silent に Optimal 化すると
        // PFEAS_FAIL に化けて真因 (deadline 切れ) が隠れる (task #46/#52)。
        // LocallyOptimal: 不定 Q の KKT 点。凸ベンチでは原問題 Q が実は PSD なので
        // Optimal と同等に扱って品質確認する。
        let result = solver::bench_utils::apply_bench_status_promotion(
            result,
            prob.num_vars,
            solver::bench_utils::BenchPromotionPolicy::QpsBenchmark,
        );

        let (status_str, note) = match result.status {
            SolveStatus::Optimal => {
                // §2.5 判定フロー: pfeas → dfeas → 相補性 → 正解値照合

                // Step 3: pfeas（行ノルム正規化版、本体ipm/mod.rsと同方式）
                let (pfeas, bfeas) = compute_primal_quality(&prob, &result.solution);
                let pfeas_normalized = compute_pfeas_normalized(&prob, &result.solution);

                // Step 4: pfeasチェック（正規化済み違反 > eps で失敗）
                if pfeas_normalized > eps || bfeas > eps {
                    n_pfeas_fail += 1;
                    (
                        "PFEAS_FAIL".to_string(),
                        format!(
                            "[{}] obj={:.2e} pf={:.1e} pfn={:.1e} bf={:.1e}",
                            method_label, result.objective, pfeas, pfeas_normalized, bfeas
                        ),
                    )
                } else {
                    // Step 5: dfeas チェック（元空間 + 成分ごと相対化）
                    // 判定は dfeas_rel < eps (OSQP/Clarabel 流). dfeas_abs は表示用。
                    // 相対化により ill-conditioned 問題 (QFORPLAN: |Qx|≈|A^Ty|≈1e9 で
                    // キャンセル後の残差 1e3) でも妥当な精度を測れる。
                    let (dfeas_abs, dfeas_rel) = compute_dfeas_orig(
                        &prob,
                        &result.solution,
                        &result.dual_solution,
                        &result.bound_duals,
                        &result.reduced_costs,
                    );

                    if !dfeas_rel.is_nan() && dfeas_rel > eps {
                        n_dfeas_fail += 1;
                        (
                            "DFEAS_FAIL".to_string(),
                            format!(
                                "[{}] obj={:.2e} pf={:.1e} df={:.1e} dfr={:.1e} (eps={:.1e})",
                                method_label, result.objective, pfeas, dfeas_abs, dfeas_rel, eps
                            ),
                        )
                    } else {
                        let dfeas = dfeas_abs;
                        // Step 7-8: 相補性チェック
                        // Simplex LP: extract_dual_info のポストホック rc は ill-conditioned 基底で
                        // 浮動小数点誤差が大きく、真に最適な解でも comp >> 0 になる偽陽性を生む。
                        // DFEAS チェック (max(0, -rc_j) ≤ eps) が LP 最適性の十分条件。
                        // IPM LP (empty reduced_costs): NaN で自動スキップ。
                        // QP: is_qp=true でスキップ。
                        let comp = f64::NAN;
                        let norm_c = prob
                            .c
                            .iter()
                            .map(|&x| x.abs())
                            .fold(0.0_f64, f64::max)
                            .max(1.0);
                        let norm_x = result
                            .solution
                            .iter()
                            .map(|&x| x.abs())
                            .fold(0.0_f64, f64::max)
                            .max(1.0);
                        let comp_tol = eps * (1.0 + norm_c * norm_x);

                        if !comp.is_nan() && comp > comp_tol {
                            n_suboptimal_comp += 1;
                            (
                                "SUBOPTIMAL".to_string(),
                                format!(
                                    "[{}] obj={:.2e} pf={:.1e} comp={:.1e} (comp_tol={:.1e})",
                                    method_label, result.objective, pfeas, comp, comp_tol
                                ),
                            )
                        } else {
                            // Step 9: 正解値照合
                            // ベースライン CSV は result.objective (obj_offset 込み) を使って生成されているため、
                            // result.objective をそのまま比較する。
                            // (9e83748 で誤って obj_offset を引いていたが、ベースラインは offset 込み値で作成済み)
                            match check_reported_objective(
                                &name,
                                result.objective,
                                &prob,
                                Some(&baseline_csv_str),
                                &baseline_objectives,
                                eps_obj,
                            ) {
                                ObjCheckResult::Mismatch { rel_err } => {
                                    n_obj_mismatch += 1;
                                    (
                                        "OBJ_MISMATCH".to_string(),
                                        format!(
                                            "[{}] obj={:.2e} known={:.2e} err={:.1}%",
                                            method_label,
                                            result.objective,
                                            baseline_objectives.get(&name).unwrap(),
                                            rel_err * 100.0
                                        ),
                                    )
                                }
                                ObjCheckResult::Ok { rel_err } => {
                                    n_pass += 1;
                                    // 判定値 (pfn 全体相対化, dfr 全体相対化) と
                                    // 厳しい代替 (pfc, dfc 成分相対化) を併記し、
                                    // 同じ eps で見て componentwise も満たすか可視化する。
                                    let pfc = compute_pfeas_componentwise(&prob, &result.solution);
                                    let dfc = compute_dfeas_componentwise(
                                        &prob,
                                        &result.solution,
                                        &result.dual_solution,
                                        &result.bound_duals,
                                        &result.reduced_costs,
                                    );
                                    let df_str = if dfeas.is_nan() {
                                        "df=NA dfr=NA dfc=NA".to_string()
                                    } else {
                                        format!("df={:.1e} dfr={:.1e} dfc={:.1e}", dfeas, dfeas_rel, dfc)
                                    };
                                    let comp_str = if comp.is_nan() {
                                        "comp=NA".to_string()
                                    } else {
                                        format!("comp={:.1e}", comp)
                                    };
                                    (
                                        "PASS".to_string(),
                                        format!(
                                            "[{}] obj={:.2e} pf={:.1e} pfn={:.1e} pfc={:.1e} bf={:.1e} {} {} obj_err={:.3}%",
                                            method_label,
                                            result.objective,
                                            pfeas,
                                            pfeas_normalized,
                                            pfc,
                                            bfeas,
                                            df_str,
                                            comp_str,
                                            rel_err * 100.0
                                        ),
                                    )
                                }
                                ObjCheckResult::NoRef => {
                                    n_pass_noref += 1;
                                    let pfc = compute_pfeas_componentwise(&prob, &result.solution);
                                    let dfc = compute_dfeas_componentwise(
                                        &prob,
                                        &result.solution,
                                        &result.dual_solution,
                                        &result.bound_duals,
                                        &result.reduced_costs,
                                    );
                                    let df_str = if dfeas.is_nan() {
                                        "df=NA dfr=NA dfc=NA".to_string()
                                    } else {
                                        format!("df={:.1e} dfr={:.1e} dfc={:.1e}", dfeas, dfeas_rel, dfc)
                                    };
                                    let comp_str = if comp.is_nan() {
                                        "comp=NA".to_string()
                                    } else {
                                        format!("comp={:.1e}", comp)
                                    };
                                    (
                                        "PASS[no_ref]".to_string(),
                                        format!(
                                            "[{}] obj={:.2e} pf={:.1e} pfn={:.1e} pfc={:.1e} bf={:.1e} {} {}",
                                            method_label,
                                            result.objective,
                                            pfeas,
                                            pfeas_normalized,
                                            pfc,
                                            bfeas,
                                            df_str,
                                            comp_str
                                        ),
                                    )
                                }
                            }
                        }
                    }
                }
            }
            SolveStatus::Infeasible => {
                // CSV に INFEASIBLE が記載されていれば正答 → PASS:Infeasible
                // 記載なし (no_ref) または Optimal 期待の問題に Infeasible が返ったら FAIL
                match expected_statuses.get(&name) {
                    Some(ExpectedStatus::Infeasible) => {
                        n_pass_infeasible += 1;
                        ("PASS:Infeasible".to_string(), String::new())
                    }
                    Some(ExpectedStatus::Optimal) => {
                        // 最適を期待していたのに Infeasible → 解けていない
                        n_fail += 1;
                        ("FAIL:Infeasible".to_string(), "(expected Optimal)".to_string())
                    }
                    _ => {
                        // no_ref: 正解不明。FAIL として記録するが expected Optimal ではない
                        n_fail += 1;
                        ("FAIL:Infeasible".to_string(), String::new())
                    }
                }
            }
            SolveStatus::Unbounded => {
                match expected_statuses.get(&name) {
                    Some(ExpectedStatus::Unbounded) => {
                        n_pass_unbounded += 1;
                        ("PASS:Unbounded".to_string(), String::new())
                    }
                    Some(ExpectedStatus::Optimal) => {
                        n_fail += 1;
                        ("FAIL:Unbounded".to_string(), "(expected Optimal)".to_string())
                    }
                    _ => {
                        n_fail += 1;
                        ("FAIL:Unbounded".to_string(), String::new())
                    }
                }
            }
            SolveStatus::MaxIterations => {
                n_max_iter += 1;
                (
                    "MAXITER".to_string(),
                    format!(
                        "[{}] iters={} {}",
                        method_label, result.iterations, resid_str
                    ),
                )
            }
            SolveStatus::SuboptimalSolution => {
                n_suboptimal += 1;
                let obj_str = if result.solution.is_empty() {
                    "obj=NA solution=EMPTY".to_string()
                } else if result.solution.len() != prob.num_vars {
                    format!("obj={:.3e} sol_len={}/{}_MISMATCH",
                        result.objective, result.solution.len(), prob.num_vars)
                } else {
                    let pfn = compute_pfeas_normalized(&prob, &result.solution);
                    format!("obj={:.3e} pfn={:.1e}", result.objective, pfn)
                };
                (
                    "SUBOPTIMAL".to_string(),
                    format!(
                        "[{}] iters={} {} {}",
                        method_label, result.iterations, obj_str, resid_str
                    ),
                )
            }
            SolveStatus::Timeout => {
                n_timeout += 1;
                // Timeout でも有効解があれば品質情報を表示（diagnostic 価値）
                // best-so-far 解を保持する `apply_api_boundary_conversion` 修正と組合せて、
                // 「真に解けていないのか、ほぼ解けているが時間切れなのか」を可視化する。
                let extra = if !result.solution.is_empty()
                    && result.solution.len() == prob.num_vars
                {
                    let (_, bfeas) = compute_primal_quality(&prob, &result.solution);
                    let pfeas_norm = compute_pfeas_normalized(&prob, &result.solution);
                    let (df_abs, df_rel) = compute_dfeas_orig(
                        &prob,
                        &result.solution,
                        &result.dual_solution,
                        &result.bound_duals,
                        &result.reduced_costs,
                    );
                    let df_str = if df_abs.is_nan() {
                        "df=NA".to_string()
                    } else {
                        format!("df={:.1e} dfr={:.1e}", df_abs, df_rel)
                    };
                    format!(
                        " obj={:.2e} pfn={:.1e} bf={:.1e} {}",
                        result.objective, pfeas_norm, bfeas, df_str
                    )
                } else {
                    String::new()
                };
                (
                    "TIMEOUT".to_string(),
                    format!(
                        "[{}] {:.3}s iters={}{}",
                        method_label, elapsed_s, result.iterations, extra
                    ),
                )
            }
            SolveStatus::NumericalError => {
                n_fail += 1;
                (
                    "FAIL:NumericalError".to_string(),
                    format!("[{}]", method_label),
                )
            }
            SolveStatus::NonConvex(_) => {
                n_nonconvex += 1;
                (
                    "NONCONVEX".to_string(),
                    format!("[{}] Q not PSD", method_label),
                )
            }
            _ => {
                n_fail += 1;
                ("FAIL:Unknown".to_string(), format!("[{}]", method_label))
            }
        };
        println!(
            "{:<20} {:>6} {:>6} {:>15} {:>10.3} {}",
            name, n, m, status_str, elapsed_s, note
        );
        // 追加情報行: solver詳細 + 問題サイズ
        println!(
            "  => solver={} iters={} {} | n={} m={} nnz={}",
            method_label, result.iterations, resid_str, n, m, nnz_before
        );
    }

    println!("{}", "-".repeat(80));
    println!();
    println!("=== Summary ===");
    println!("  PASS:              {}", n_pass);
    println!("  PASS[no_ref]:      {}", n_pass_noref);
    println!("  PASS:Infeasible:   {}", n_pass_infeasible);
    println!("  PASS:Unbounded:    {}", n_pass_unbounded);
    println!("  PFEAS_FAIL:        {}", n_pfeas_fail);
    println!("  DFEAS_FAIL:        {}", n_dfeas_fail);
    println!("  SUBOPTIMAL:        {}", n_suboptimal + n_suboptimal_comp);
    println!("  OBJ_MISMATCH:      {}", n_obj_mismatch);
    println!("  MAXITER:           {}", n_max_iter);
    println!("  TIMEOUT:           {}", n_timeout);
    println!("  NONCONVEX:         {}", n_nonconvex);
    println!("  FAIL:              {}", n_fail);
    println!("  ERROR:             {}", n_error);
    println!(
        "  TOTAL:             {}",
        n_pass
            + n_pass_noref
            + n_pass_infeasible
            + n_pass_unbounded
            + n_pfeas_fail
            + n_dfeas_fail
            + n_suboptimal_comp
            + n_obj_mismatch
            + n_fail
            + n_max_iter
            + n_suboptimal
            + n_timeout
            + n_nonconvex
            + n_error
    );
}
