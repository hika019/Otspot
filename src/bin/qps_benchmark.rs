//! Maros-Meszaros QPS ベンチマーク
//!
//! Usage: qps_benchmark <data_dir> [--solver ipm|ippmm_new|concurrent|dualadvanced] [--eps <value>]
//! 指定ディレクトリ内の全*.QPSファイルを parse_qps → solve_qp_with_options で実行し、
//! 結果テーブルをstdoutに出力する。
//!
//! 各問題に10秒のタイムアウトを設ける（solver内部の協調的タイムアウト機構を使用）。

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::env;
use std::path::Path;
use std::time::{Duration, Instant};

use solver::bench_utils::{check_baseline_objective, detect_csv_path, load_baseline_objectives, ObjCheckResult};
use solver::io::qps::{parse_qps, QpsError};
use solver::options::{QpSolverChoice, SimplexMethod, SolverOptions};
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::solve_qp_with;
use solver::QpProblem;

enum BenchError {
    Parse(QpsError),
    ParseTimeout,
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

    let bfeas = solution
        .iter()
        .zip(prob.bounds.iter())
        .map(|(&xi, &(lb, ub))| {
            let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
            let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
            lb_viol.max(ub_viol)
        })
        .fold(0.0_f64, f64::max);

    (pfeas, bfeas)
}

/// §2.1b: 行ノルム正規化pfeas（本体ipm/mod.rsと同方式）
///
/// max_k [ violation_k / (1 + ||a_k||_∞ + |b_k|) ]
/// 本体の post_verify_solution と同一の正規化式を使用。
fn compute_pfeas_normalized(prob: &QpProblem, solution: &[f64]) -> f64 {
    if solution.is_empty() || solution.len() != prob.num_vars {
        return f64::NAN;
    }
    if prob.num_constraints == 0 {
        return 0.0;
    }
    match prob.a.mat_vec_mul(solution) {
        Ok(ax) => {
            let row_norms = prob.a.row_infinity_norms();
            ax.iter()
                .zip(prob.b.iter())
                .enumerate()
                .zip(row_norms.iter())
                .map(|((i, (&ax_i, &b_i)), &rn)| {
                    let violation = match prob.constraint_types.get(i) {
                        Some(ConstraintType::Eq) => (ax_i - b_i).abs(),
                        Some(ConstraintType::Ge) => (b_i - ax_i).max(0.0),
                        _ => (ax_i - b_i).max(0.0),
                    };
                    violation / (1.0 + rn + b_i.abs())
                })
                .fold(0.0_f64, f64::max)
        }
        Err(_) => f64::NAN,
    }
}

/// §2.2: dfeas 元空間再計算（ソルバ申告値ではなく bench 側で独立計算）
///
/// ソルバの `result.dfeas` は内部 (Ruiz scaled) 空間の値で、unscale 後の
/// 元空間 dfeas とは異なる。bench は「ユーザー指定 eps を元空間で満たすか」を
/// 検証する役割なので、元 problem.q / problem.a / problem.c と unscale 済み解で
/// 直接 KKT 残差 dfeas = ||Q*x + A^T*y + bound_contrib + c||_∞ を計算する。
///
/// この修正前は scaled dfeas を eps と比較しており、UBH1 のように内部 OK でも
/// 元空間で乖離が大きいケースで「pfeas/dfeas 通過なのに obj 55% 乖離」という
/// 矛盾した PASS 判定が起きていた。
fn compute_dfeas_orig(
    prob: &QpProblem,
    solution: &[f64],
    dual_solution: &[f64],
    bound_duals: &[f64],
) -> f64 {
    if solution.is_empty() || solution.len() != prob.num_vars {
        return f64::NAN;
    }
    let n = solution.len();
    let qx = match prob.q.mat_vec_mul(solution) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let aty: Vec<f64> = if prob.a.nrows > 0 && !dual_solution.is_empty() {
        match prob.a.transpose().mat_vec_mul(dual_solution) {
            Ok(v) => v,
            Err(_) => return f64::NAN,
        }
    } else {
        vec![0.0; n]
    };
    // bound_contrib[j] = -y_lb[j] (lb有限) + y_ub[j] (ub有限)
    // bound_duals レイアウト: [lb 有限の y_lb...; ub 有限の y_ub...]
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
    }
    (0..n)
        .map(|i| (qx[i] + aty[i] + bound_contrib[i] + prob.c[i]).abs())
        .fold(0.0_f64, f64::max)
}

/// 旧: ソルバ申告 dfeas を返す（後方互換のため残置、現状未使用）。
#[allow(dead_code)]
fn get_dfeas(result: &solver::problem::SolverResult) -> f64 {
    result.dfeas.unwrap_or(f64::NAN)
}

/// §2.3: 相補性チェック（LP限定。QPスキップ）
///
/// 双対相補性: max_i |(x_i - lb_i)*max(rc_i,0) + (ub_i - x_i)*max(-rc_i,0)|
fn compute_complementarity(
    solution: &[f64],
    reduced_costs: &[f64],
    bounds: &[(f64, f64)],
) -> f64 {
    if solution.is_empty() || reduced_costs.is_empty() {
        return f64::NAN;
    }
    let n = solution.len().min(reduced_costs.len());
    (0..n)
        .map(|i| {
            let (lb, ub) = if i < bounds.len() { bounds[i] } else { (0.0, f64::INFINITY) };
            let rc = reduced_costs[i];
            // 双対相補性: 下限側 + 上限側（上限無限の場合はスキップ）
            let lower_comp = (solution[i] - lb) * rc.max(0.0);
            let upper_comp = if ub.is_finite() { (ub - solution[i]) * (-rc).max(0.0) } else { 0.0 };
            lower_comp + upper_comp
        })
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
}

fn parse_with_timeout(path: &Path, timeout_secs: u64) -> Result<QpProblem, BenchError> {
    let path = path.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = parse_qps(&path);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Ok(prob)) => Ok(prob),
        Ok(Err(e)) => Err(BenchError::Parse(e)),
        Err(_) => Err(BenchError::ParseTimeout),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // 引数パース: [data_dir] [--solver ipm|ippmm_new|concurrent] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]
    let mut data_dir = "data/maros_meszaros".to_string();
    let mut solver_choice = QpSolverChoice::Concurrent;
    let mut dual_advanced_mode = false;
    let mut eps: f64 = 1e-6;
    let mut timeout_secs: f64 = 10.0;
    let mut baseline_override: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: qps_benchmark [data_dir] [--solver ipm|ippmm_new|concurrent|dualadvanced] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]");
            println!("  --solver        Solver to use (default: concurrent/auto)");
            println!("  --eps           Convergence tolerance (default: 1e-6)");
            println!("  --timeout       Solver timeout in seconds (default: 10.0)");
            println!("  --known-optimal Path to known optimal values CSV (default: auto-detect)");
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
        } else if args[i] == "--solver" {
            i += 1;
            if i < args.len() {
                match args[i].as_str() {
                    "ipm" => solver_choice = QpSolverChoice::Ipm,
                    "ippmm_new" => solver_choice = QpSolverChoice::IpPmmNew,
                    "concurrent" => solver_choice = QpSolverChoice::Concurrent,
                    "dualadvanced" => {
                        dual_advanced_mode = true;
                        solver_choice = QpSolverChoice::Concurrent; // QP問題のフォールバック
                    }
                    other => {
                        eprintln!("Unknown solver: {}. Use ipm|ippmm_new|concurrent|dualadvanced", other);
                        std::process::exit(1);
                    }
                };
            }
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
    let baseline_objectives = {
        let root = {
            let mut p = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|pp| pp.to_path_buf()))
                .unwrap_or_default();
            // target/release から solver ルートに遡る
            p = p.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()).unwrap_or_default();
            p
        };
        let csv = detect_csv_path(&data_dir, baseline_override.as_deref(), &root);
        load_baseline_objectives(&csv)
    };
    eprintln!("Baseline objectives loaded: {} problems", baseline_objectives.len());
    if baseline_objectives.is_empty() {
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

    // 集計 — §2.5の7カテゴリ + 既存カテゴリ
    let mut n_pass = 0usize;
    let mut n_pass_noref = 0usize;
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

    let solver_label = if dual_advanced_mode {
        "DualAdvanced"
    } else {
        match solver_choice {
            QpSolverChoice::Concurrent => "Concurrent",
            QpSolverChoice::Ipm => "IPM",
            QpSolverChoice::IpPmmNew => "IP-PMM-New",
            _ => "Unknown",
        }
    };
    println!("Solver: {}", solver_label);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs);
    opts.qp_solver = solver_choice;
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
            Err(BenchError::ParseTimeout) => {
                println!(
                    "{:<20} {:>6} {:>6} {:>15} {:>10.3} ",
                    name, "?", "?", "PARSE_TIMEOUT", 0.0
                );
                n_timeout += 1;
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
        let is_qp = prob.q.nnz() > 0;

        println!("SOLVE_START: {}", name);
        let start = Instant::now();
        let result = solve_qp_with(&prob, &opts);
        let elapsed_s = start.elapsed().as_secs_f64();
        println!(
            "SOLVE_DONE: {} {:?} ({:.3}s)",
            name, result.status, elapsed_s
        );

        let method_label = match result.solver_used {
            Some(QpSolverChoice::Ipm) => "ipm",
            Some(QpSolverChoice::Concurrent) => "concurrent",
            Some(QpSolverChoice::IpPmmNew) => "ippmm_new",
            Some(_) => "other",
            None => "-",
        };
        let resid_str = match result.final_residuals {
            Some((pf, df, gap)) => format!("pf={:.1e} df={:.1e} gap={:.1e}", pf, df, gap),
            None => String::new(),
        };

        // Timeout だが有効解 (best-so-far) を保持している場合、Optimal フローに乗せて
        // 品質判定 (pfeas/bfeas/dfeas/obj_check) を通す。
        // PASS 判定が出れば bench 上 PASS としてカウントし、TIMEOUT として誤分類しない。
        // 品質判定で fail した場合は PFEAS_FAIL/DFEAS_FAIL/OBJ_MISMATCH 等の正確な分類になる。
        // CONT-300 等が「内部はほぼ Optimal なのに TIMEOUT 表示」される hostile 状態の解消。
        let result = if matches!(result.status, SolveStatus::Timeout)
            && !result.solution.is_empty()
            && result.solution.len() == prob.num_vars
        {
            solver::problem::SolverResult { status: SolveStatus::Optimal, ..result }
        } else {
            result
        };

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
                    // Step 5: dfeas チェック（元空間で独立再計算 — ソルバ申告値は scaled）
                    let dfeas = compute_dfeas_orig(
                        &prob,
                        &result.solution,
                        &result.dual_solution,
                        &result.bound_duals,
                    );
                    let norm_c = prob
                        .c
                        .iter()
                        .map(|&x| x.abs())
                        .fold(0.0_f64, f64::max)
                        .max(1.0);
                    let dfeas_tol = eps * (1.0 + norm_c);

                    if !dfeas.is_nan() && dfeas > dfeas_tol {
                        n_dfeas_fail += 1;
                        (
                            "DFEAS_FAIL".to_string(),
                            format!(
                                "[{}] obj={:.2e} pf={:.1e} df={:.1e} (dfeas_tol={:.1e})",
                                method_label, result.objective, pfeas, dfeas, dfeas_tol
                            ),
                        )
                    } else {
                        // Step 7-8: 相補性チェック（LP限定、QPスキップ）
                        let comp = if !is_qp {
                            compute_complementarity(&result.solution, &result.reduced_costs, &prob.bounds)
                        } else {
                            f64::NAN // QPでは相補性スキップ
                        };
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
                            match check_baseline_objective(
                                &name,
                                result.objective,
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
                                    let df_str = if dfeas.is_nan() {
                                        "df=NA".to_string()
                                    } else {
                                        format!("df={:.1e}", dfeas)
                                    };
                                    let comp_str = if comp.is_nan() {
                                        "comp=NA".to_string()
                                    } else {
                                        format!("comp={:.1e}", comp)
                                    };
                                    (
                                        "PASS".to_string(),
                                        format!(
                                            "[{}] obj={:.2e} pf={:.1e} bf={:.1e} {} {} obj_err={:.3}%",
                                            method_label,
                                            result.objective,
                                            pfeas,
                                            bfeas,
                                            df_str,
                                            comp_str,
                                            rel_err * 100.0
                                        ),
                                    )
                                }
                                ObjCheckResult::NoRef => {
                                    n_pass_noref += 1;
                                    let df_str = if dfeas.is_nan() {
                                        "df=NA".to_string()
                                    } else {
                                        format!("df={:.1e}", dfeas)
                                    };
                                    let comp_str = if comp.is_nan() {
                                        "comp=NA".to_string()
                                    } else {
                                        format!("comp={:.1e}", comp)
                                    };
                                    (
                                        "PASS[no_ref]".to_string(),
                                        format!(
                                            "[{}] obj={:.2e} pf={:.1e} bf={:.1e} {} {}",
                                            method_label,
                                            result.objective,
                                            pfeas,
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
                n_fail += 1;
                ("FAIL:Infeasible".to_string(), String::new())
            }
            SolveStatus::Unbounded => {
                n_fail += 1;
                ("FAIL:Unbounded".to_string(), String::new())
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
                (
                    "SUBOPTIMAL".to_string(),
                    format!(
                        "[{}] iters={} {}",
                        method_label, result.iterations, resid_str
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
                    let df = compute_dfeas_orig(
                        &prob,
                        &result.solution,
                        &result.dual_solution,
                        &result.bound_duals,
                    );
                    let df_str = if df.is_nan() {
                        "df=NA".to_string()
                    } else {
                        format!("df={:.1e}", df)
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
    println!("  PASS:           {}", n_pass);
    println!("  PASS[no_ref]:   {}", n_pass_noref);
    println!("  PFEAS_FAIL:     {}", n_pfeas_fail);
    println!("  DFEAS_FAIL:     {}", n_dfeas_fail);
    println!("  SUBOPTIMAL:     {}", n_suboptimal + n_suboptimal_comp);
    println!("  OBJ_MISMATCH:   {}", n_obj_mismatch);
    println!("  MAXITER:        {}", n_max_iter);
    println!("  TIMEOUT:        {}", n_timeout);
    println!("  NONCONVEX:      {}", n_nonconvex);
    println!("  FAIL:           {}", n_fail);
    println!("  ERROR:          {}", n_error);
    println!(
        "  TOTAL:          {}",
        n_pass
            + n_pass_noref
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
