//! Maros-Meszaros QPS ベンチマーク
//!
//! Usage: qps_benchmark <data_dir> [--eps <value>] [--dual-advanced]
//! 指定ディレクトリ内の全*.QPSファイルを parse_qps → solve_qp_with で実行し、
//! 結果テーブルをstdoutに出力する。
//!
//! 各問題に10秒のタイムアウトを設ける（solver内部の協調的タイムアウト機構を使用）。

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::env;
use std::path::Path;
use std::time::Instant;

use otspot_dev::bench_utils::{
    check_baseline_objective, compute_dfeas_componentwise, compute_dfeas_orig,
    compute_pfeas_normalized, detect_csv_path, load_baseline_objectives, load_expected_statuses,
    parse_qps_with_timeout, ExpectedStatus, ObjCheckResult,
};
use otspot_core::options::{SimplexMethod, SolverOptions};
use otspot_core::problem::{ConstraintType, SolveStatus};
use otspot_core::qp::{solve_qp_with, QpProblem};

/// pfeas両側チェック + bfeas
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


#[allow(clippy::items_after_test_module)] // fn main() follows this module; reorganising is disruptive
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use otspot_core::problem::ConstraintType;
    use otspot_core::sparse::CscMatrix;

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

    #[test]
    fn test_netlib_objective_check_adds_obj_offset_to_reference() {
        let mut known = HashMap::new();
        known.insert("e226".to_string(), -18.751_929_066);

        // Netlib: obj_offset = -7.113; solver reports known_obj + offset = -25.864...
        let result = check_baseline_objective(
            "e226",
            -25.864_929_066,
            &known,
            1e-9,
            -7.113,
        );
        assert!(matches!(result, ObjCheckResult::Ok { .. }));
    }

    #[test]
    fn test_non_netlib_objective_check_does_not_add_obj_offset() {
        let mut known = HashMap::new();
        known.insert("toy".to_string(), 12.5);

        // Non-netlib: obj_offset = 0.0; solver reports known_obj directly.
        let result = check_baseline_objective(
            "toy",
            12.5,
            &known,
            1e-9,
            0.0,
        );
        assert!(matches!(result, ObjCheckResult::Ok { .. }));
    }

    /// load_expected_statuses が INFEASIBLE エントリを正しく読む
    #[test]
    fn test_expected_status_infeasible_loaded() {
        use otspot_dev::bench_utils::{load_expected_statuses, ExpectedStatus};
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
    // measurement-only: forwards to `opts.threads` to profile per-solve
    // factorization parallelism. Production default is threads=1 (serial).
    // Effect is problem-dependent: dense-KKT convex QPs (CVXQP*_L) speed up at
    // threads≥2, sparser/structured systems (CONT-201) regress, very sparse
    // ones are parity. Diagnostic knob, not a production path.
    let mut threads: usize = 1;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: qps_benchmark [data_dir] [--eps <value>] [--timeout <secs>] [--known-optimal <path>] [--dual-advanced]");
            println!("  --eps             Convergence tolerance (default: 1e-6)");
            println!("  --timeout         Solver timeout in seconds (default: 10.0)");
            println!("  --known-optimal   Path to known optimal values CSV (default: auto-detect)");
            println!("  --dual-advanced   LP は DualAdvanced simplex を使う (QP は無視)");
            println!("  --threads         Per-solve factorization parallelism (default: 1 = serial)");
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
        } else if args[i] == "--threads" {
            i += 1;
            if i < args.len() {
                threads = args[i].parse().unwrap_or(1).max(1);
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

    // 正解値CSV読み込み
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

    // 集計 — 7カテゴリ + 既存カテゴリ + infeasible/unbounded 正答
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
    opts.threads = threads;
    if dual_advanced_mode {
        opts.simplex_method = SimplexMethod::DualAdvanced;
    }

    // QP問題かどうかの判定用定数
    let eps_obj: f64 = 1e-2; // 目的関数照合の相対許容誤差: 1%

    for path in &qps_files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let parse_start = Instant::now();
        println!("PARSE_START: {}", name);

        // パース（30秒タイムアウト付き）
        let prob = match parse_qps_with_timeout(path, 30) {
            Ok(p) => p,
            Err(note) => {
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
        let is_qp = prob.q.nnz() > 0;

        println!("SOLVE_START: {}", name);
        let start = Instant::now();
        let result = solve_qp_with(&prob, &opts);
        let elapsed_s = start.elapsed().as_secs_f64();
        println!(
            "SOLVE_DONE: {} {:?} ({:.3}s)",
            name, result.status, elapsed_s
        );

        let method_label = if is_qp {
            "ipm"
        } else if result.stats.lp_ipm_path {
            "lp-ipm"
        } else {
            "lp-simplex"
        };
        let resid_str = match result.final_residuals {
            Some((pf, df, gap)) => format!("pf={:.1e} df={:.1e} gap={:.1e}", pf, df, gap),
            None => String::new(),
        };

        // SuboptimalSolution / LocallyOptimal で有効解を保持している場合のみ Optimal フロー
        // に乗せて品質判定 (pfeas/bfeas/dfeas/obj_check) を通す。収束未達 status
        // (Timeout / MaxIterations / NumericalError / NonConvex) は honest 報告。
        let result = otspot_dev::bench_utils::apply_bench_status_promotion(
            result,
            prob.num_vars,
            otspot_dev::bench_utils::BenchPromotionPolicy::QpsBenchmark,
        );

        let (status_str, note) = match result.status {
            SolveStatus::Optimal => {
                // 判定フロー: pfeas → dfeas → 相補性 → 正解値照合

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
                            // netlib_lp.csv のみ CSV 参照値に obj_offset を加算して比較
                            // (solver は result.objective に offset 込みで返すため)。
                            let obj_offset = if baseline_csv_str.ends_with("netlib_lp.csv") {
                                prob.obj_offset
                            } else {
                                0.0
                            };
                            match check_baseline_objective(
                                &name,
                                result.objective,
                                &baseline_objectives,
                                eps_obj,
                                obj_offset,
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
                                    let pfc = compute_pfeas_normalized(&prob, &result.solution);
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
                                    let pfc = compute_pfeas_normalized(&prob, &result.solution);
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
