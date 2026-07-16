//! QPLIBベンチマーク
//!
//! Usage: `bench_qplib [data_dir] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]`
//! 指定ディレクトリ内の全 *.qplib ファイルを parse_qplib → solve_qp_with で実行し、
//! 結果テーブルを stdout に出力する。
//!
//! Maros-Meszaros（QPS形式）とは独立した集計。
//! 対応問題タイプ: *C*（連続変数）かつ L/B/N/Q（線形/境界/無制約/二次制約）。
//! Q（QCQP）は quadratic_constraints にロードして解く。非対応
//! （整数変数 B/I 等、および L/B/N/Q 以外の制約型）は SKIP として記録する。

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::env;
use std::path::Path;
use std::time::Instant;

use otspot_core::options::{GlobalOptimizationConfig, SolverOptions};
use otspot_core::presolve::{run_qp_presolve_phase1, run_qp_presolve_phase2};
use otspot_core::problem::SolveStatus;
use otspot_core::qp::{solve_qp_global, solve_qp_with};
use otspot_dev::bench_utils::{
    check_baseline_objective, compute_gap_to_global, compute_qp_kkt_max, is_kkt_violation,
    kkt_gated_label, load_qplib_baselines, parse_qplib_outcome, qcqp_pfeas_max, ExpectedStatus,
    ObjCheckResult, ParseQplibOutcome,
};

/// QP 元空間 KKT 残差の PASS 閾値 (Ruiz 振幅 100 級まで許容、`diag_nonconvex_kkt::EPS_KKT` 整合)。
const KKT_FAIL_EPS: f64 = 1e-4;

/// QPLIB UnsupportedType の category (parse error message 由来).
///
/// 推論ではなく emit 元の error message prefix を引用判定:
/// "Variable type" → integer (otspot-io/src/qplib/parser.rs:57 の M/G/S reject と
/// bench_utils::parse_qplib_outcome の B/I→MIP メッセージの両方)、
/// "Constraint type" → qcqp (otspot-io/src/qplib/parser.rs:67)、それ以外 → other。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnsupportedCategory {
    /// 変数型が C(continuous) 以外 (B/I/G/S/M)
    Integer,
    /// 制約型が L/B/N/Q 以外 (Q = QCQP は quadratic_constraints として解くため対象外)
    Qcqp,
    /// 上記外
    Other,
}

fn classify_unsupported(msg: &str) -> UnsupportedCategory {
    if msg.starts_with("Variable type") {
        UnsupportedCategory::Integer
    } else if msg.starts_with("Constraint type") {
        UnsupportedCategory::Qcqp
    } else {
        UnsupportedCategory::Other
    }
}

#[allow(clippy::items_after_test_module)] // fn main() and helpers follow; reorganising is disruptive
#[cfg(test)]
mod unsupported_classify_tests {
    use super::{classify_unsupported, UnsupportedCategory};

    #[test]
    fn variable_type_message_is_integer_category() {
        // otspot-io/src/qplib/parser.rs:57 fmt (M/G/S mixed-integer reject)
        let m = "Variable type 'M' not supported (C/B/I supported; M/G/S mixed-integer unsupported). Type=QML";
        assert_eq!(classify_unsupported(m), UnsupportedCategory::Integer);
    }

    #[test]
    fn constraint_type_message_is_qcqp_category() {
        // otspot-io/src/qplib/parser.rs:67 fmt ('Q' は QCQP としてロードされるため
        // このメッセージを出さない; L/B/N/Q 以外の制約型、例 'C'=convex quad が出す)
        let m = "Constraint type 'C' not supported (only L/B/N/Q supported). Type=QCC";
        assert_eq!(classify_unsupported(m), UnsupportedCategory::Qcqp);
    }

    #[test]
    fn binary_var_message_is_integer_category() {
        // bench_utils::parse_qplib_outcome fmt (B/I は parser を通り Milp/Miqp に
        // route されるため、この bench 向けメッセージが emit 元)
        let m = "Variable type 'B'/'I' (binary/integer): MIP problem";
        assert_eq!(classify_unsupported(m), UnsupportedCategory::Integer);
    }

    #[test]
    fn unknown_message_is_other() {
        let m = "Some other unsupported feature";
        assert_eq!(classify_unsupported(m), UnsupportedCategory::Other);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // 引数パース: [data_dir] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]
    //             [--global] [--gap-tol <f64>] [--max-nodes <usize>]
    let mut data_dir = "data/qplib".to_string();
    let mut eps: f64 = 1e-6;
    let mut timeout_secs: f64 = 10.0;
    let mut baseline_override: Option<String> = None;
    let mut use_global: bool = false;
    let mut global_gap_tol: f64 = 1e-3;
    let mut global_max_nodes: usize = 10_000;

    // known flagリスト（値を持つフラグ）
    const KNOWN_FLAGS_WITH_VALUE: &[&str] = &[
        "--eps",
        "--timeout",
        "--known-optimal",
        "--gap-tol",
        "--max-nodes",
    ];

    let mut i = 1usize;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: bench_qplib [data_dir] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]");
            println!("  --eps           Convergence tolerance (default: 1e-6)");
            println!("  --timeout       Solver timeout in seconds (default: 10.0)");
            println!("  --known-optimal Path to known optimal values CSV (default: auto-detect)");
            println!(
                "  --global        Use solve_qp_global (spatial B&B) instead of single-shot IPM"
            );
            println!("  --gap-tol       Global optimality gap tolerance (default: 1e-3)");
            println!("  --max-nodes     B&B node limit (default: 10000)");
            std::process::exit(0);
        } else if args[i] == "--global" {
            use_global = true;
            i += 1;
        } else if KNOWN_FLAGS_WITH_VALUE.contains(&args[i].as_str()) {
            // known flag: 次引数が値 → i+=2で消費
            i += 1;
            if i < args.len() {
                match args[i - 1].as_str() {
                    "--eps" => {
                        eps = args[i].parse().unwrap_or(1e-6);
                    }
                    "--timeout" => {
                        timeout_secs = args[i].parse().unwrap_or(10.0);
                    }
                    "--known-optimal" => {
                        baseline_override = Some(args[i].clone());
                    }
                    "--gap-tol" => {
                        global_gap_tol = args[i].parse().unwrap_or(1e-3);
                    }
                    "--max-nodes" => {
                        global_max_nodes = args[i].parse().unwrap_or(10_000);
                    }
                    _ => {}
                }
            }
            i += 1;
        } else if args[i].starts_with("--") {
            // 未知フラグ: 値は消費しない → i+=1のみ
            eprintln!("Warning: unknown flag '{}', ignoring", args[i]);
            i += 1;
        } else {
            // positional引数: 最初の非フラグ引数がdata_dir
            data_dir = args[i].clone();
            i += 1;
        }
    }

    let dir = Path::new(&data_dir);
    if !dir.exists() {
        eprintln!("Directory not found: {}", data_dir);
        std::process::exit(1);
    }

    // 正解値CSV読み込み
    let (baseline_objectives, expected_statuses) = {
        let root = {
            let p = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|pp| pp.to_path_buf()))
                .unwrap_or_default();
            // target/release から solver ルートに遡る
            p.parent()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
                .unwrap_or_default()
        };
        // qplib_qcqp.csv covers the CCQ/DCQ/QCQ instances mixed into this same
        // data dir alongside the DCL/QCL ones `detect_csv_path` resolves;
        // `load_qplib_baselines` merges it in unconditionally, so QCQP-route
        // regressions gate instead of falling through as CHECKED[no_ref]
        // (PR #25 review).
        load_qplib_baselines(&data_dir, baseline_override.as_deref(), &root)
    };
    eprintln!(
        "Baseline objectives loaded: {} problems",
        baseline_objectives.len()
    );
    eprintln!(
        "Expected statuses loaded: {} problems",
        expected_statuses.len()
    );
    if baseline_objectives.is_empty() && expected_statuses.is_empty() {
        eprintln!(
            "WARNING: No known optimal values loaded. Optimal-feasible problems will be CHECKED[no_ref], not PASS."
        );
    }

    // .qplib ファイル一覧（ファイル名でソート）
    let mut qplib_files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("qplib"))
                .unwrap_or(false)
        })
        .collect();
    qplib_files.sort();

    println!("QPLIB Benchmark ({} files)", qplib_files.len());
    if use_global {
        println!(
            "Mode: GLOBAL (solve_qp_global / spatial B&B, gap_tol={:.0e}, max_nodes={})",
            global_gap_tol, global_max_nodes
        );
    } else {
        println!("Mode: LOCAL (solve_qp_with / single-shot IPM)");
    }
    println!();

    println!("Solver: IPPMM");

    println!(
        "{:<24} {:>6} {:>6} {:>15} {:>10} Note",
        "Problem", "n", "m", "Status", "Time(s)"
    );
    println!("{}", "-".repeat(80));

    let mut n_pass = 0usize;
    let mut n_pass_noref = 0usize;
    let mut n_pass_infeasible = 0usize;
    let mut n_pass_unbounded = 0usize;
    let mut n_obj_mismatch = 0usize;
    let mut n_fail = 0usize;
    let mut n_error = 0usize;
    let mut n_timeout = 0usize;
    let mut n_max_iter = 0usize;
    let mut n_suboptimal = 0usize;
    let mut n_skip = 0usize;
    // SKIP 内訳 (CLAUDE.md L46: 推論排除、parse 由来 message を category 化).
    // - integer: var_char != 'C' (B=binary, I=integer, G=general-int, S=semi-cont, M=mixed)
    // - qcqp:    con_char != L/B/N/Q (Q は quadratic_constraints として解くため SKIP しない)
    // - other:   上記外の Unsupported message (現状 parser 経路では発生しないが保険)
    let mut n_skip_integer = 0usize;
    let mut n_skip_qcqp = 0usize;
    let mut n_skip_other = 0usize;
    let mut n_nonconvex = 0usize;
    let mut n_nonconvex_local = 0usize;
    let mut n_nonconvex_global = 0usize;
    // solve_qp_with/solve_qp_global が out-of-scope と判断し正当に declined したケース
    // (例: 非凸QCQPで有限境界を要求するが変数が (-inf, inf))。
    // n_fail に混ぜない: これは solver の誤りではなく正しい scope 判定なので、
    // n_skip (parse 時点の SKIP) とも区別する専用 bucket とする。
    let mut n_not_supported = 0usize;
    // Phase 1A: status=Optimal だが元空間 KKT 残差 >= KKT_FAIL_EPS の解。
    // 非凸 QP で false-positive Optimal を obj-only judge が見逃す穴を埋める。
    let mut n_kkt_fail = 0usize;

    let eps_obj: f64 = 1e-2; // 目的関数照合の相対許容誤差: 1%

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs);
    opts.ipm.eps = eps;

    for path in &qplib_files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let parse_start = Instant::now();
        println!("PARSE_START: {}", name);

        let prob = match parse_qplib_outcome(path) {
            ParseQplibOutcome::Qp(p) => *p,
            ParseQplibOutcome::Unsupported(msg) => {
                // parse_qplib::UnsupportedType message に基づき category 分類
                // (emit 元は UnsupportedCategory の doc コメント参照)
                let category = classify_unsupported(&msg);
                let status_label = match category {
                    UnsupportedCategory::Integer => "SKIP:integer",
                    UnsupportedCategory::Qcqp => "SKIP:qcqp",
                    UnsupportedCategory::Other => "SKIP:other",
                };
                match category {
                    UnsupportedCategory::Integer => n_skip_integer += 1,
                    UnsupportedCategory::Qcqp => n_skip_qcqp += 1,
                    UnsupportedCategory::Other => n_skip_other += 1,
                }
                let note = msg.chars().take(40).collect::<String>();
                println!(
                    "{:<24} {:>6} {:>6} {:>15} {:>10.3} {}",
                    name, "-", "-", status_label, 0.0, note
                );
                n_skip += 1;
                continue;
            }
            ParseQplibOutcome::ParseError(note) => {
                println!(
                    "{:<24} {:>6} {:>6} {:>15} {:>10.3} {}",
                    name,
                    "?",
                    "?",
                    "PARSE_ERR",
                    0.0,
                    &note[..note.len().min(40)]
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

        // presolve削減量を取得（ベンチ計装のみ）
        // 大規模問題はpresolveに長時間かかるためスキップ（タイムアウト計測の精度を確保）
        const PRESOLVE_INSTR_MAX: usize = 50_000;
        let (n_after, m_after, nnz_after) =
            if opts.presolve && n <= PRESOLVE_INSTR_MAX && m <= PRESOLVE_INSTR_MAX {
                let phase1 = run_qp_presolve_phase1(&prob, &opts);
                let presolve_result = run_qp_presolve_phase2(phase1, &opts);
                let rn = presolve_result.reduced.num_vars;
                let rm = presolve_result.reduced.num_constraints;
                let rnnz = presolve_result.reduced.q.nnz() + presolve_result.reduced.a.nnz();
                (rn, rm, rnnz)
            } else {
                (n, m, nnz_before)
            };

        println!("SOLVE_START: {}", name);
        let start = Instant::now();
        let result = if use_global {
            let mut cfg = GlobalOptimizationConfig::default();
            cfg.gap_tol = global_gap_tol;
            cfg.max_nodes = global_max_nodes;
            solve_qp_global(&prob, &opts, &cfg)
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

        // SuboptimalSolution / LocallyOptimal で有効解 + 有限 obj を持つ result を Optimal に
        // 格上げし baseline obj 照合に流す。Timeout / MaxIterations / NumericalError /
        // NonConvex は格上げ対象外 (収束未達 status を honest に報告する)。
        let result = otspot_dev::bench_utils::apply_bench_status_promotion(
            result,
            prob.num_vars,
            otspot_dev::bench_utils::BenchPromotionPolicy::BenchQplib,
        );

        // Optimal/LocallyOptimal/Nonconvex*/Suboptimal で有効解を持つ result に KKT 残差を計測し、
        // 違反したら kkt_gated_label でその arm の verdict を KKT_FAIL に降格する
        // (Optimal だけでなく NONCONVEX_LOCAL/NONCONVEX_GLOBAL/SUBOPTIMAL も対象:
        // obj や status がそれらしくても制約違反解は false positive)。元空間 gap は
        // baseline_objectives の最適値が数値であれば算出 (Infeasible/Unbounded sentinel
        // 文字列は除外済)。QCQP (quadratic_constraints 非空) は compute_qp_kkt_max が
        // 二次制約を読まないため、qcqp_pfeas_max (二次制約の primal feasibility) を
        // 同じスカラーに畳み込む。stationarity は対象外 (primal feasibility のみ)。
        let kkt_max = if matches!(
            result.status,
            SolveStatus::Optimal
                | SolveStatus::LocallyOptimal
                | SolveStatus::NonconvexLocal
                | SolveStatus::NonconvexGlobal
                | SolveStatus::SuboptimalSolution
        ) && result.solution.len() == prob.num_vars
        {
            let stationarity_max = compute_qp_kkt_max(
                &prob,
                &result.solution,
                &result.dual_solution,
                &result.bound_duals,
            );
            let qc_pfeas_max = qcqp_pfeas_max(&prob, &result.solution);
            Some(stationarity_max.max(qc_pfeas_max))
        } else {
            None
        };
        let gap_to_global = baseline_objectives
            .get(&name)
            .and_then(|&gr| compute_gap_to_global(result.objective, gr));

        let kkt_str = match kkt_max {
            Some(v) => format!("kkt_max={:.2e}", v),
            None => "kkt_max=—".to_string(),
        };
        let gap_str = match gap_to_global {
            Some(v) => format!("gap_to_global={:.3e}", v),
            None => "gap_to_global=—".to_string(),
        };

        let (status_str, note) = match result.status {
            SolveStatus::Optimal => {
                if result.objective.is_finite() {
                    // KKT_FAIL: status=Optimal だが元空間で KKT 違反 → PASS 判定の優先打ち消し。
                    if is_kkt_violation(kkt_max, KKT_FAIL_EPS) {
                        n_kkt_fail += 1;
                        (
                            "KKT_FAIL".to_string(),
                            format!(
                                "[{}] obj={:.6e} {} {}",
                                method_label, result.objective, kkt_str, gap_str
                            ),
                        )
                    } else {
                        match check_baseline_objective(
                            &name,
                            result.objective,
                            &baseline_objectives,
                            eps_obj,
                            0.0,
                        ) {
                            ObjCheckResult::Ok { rel_err } => {
                                n_pass += 1;
                                (
                                    "PASS".to_string(),
                                    format!(
                                        "[{}] obj={:.6e} obj_err={:.3}% {} {}",
                                        method_label,
                                        result.objective,
                                        rel_err * 100.0,
                                        kkt_str,
                                        gap_str,
                                    ),
                                )
                            }
                            ObjCheckResult::Mismatch { rel_err } => {
                                n_obj_mismatch += 1;
                                (
                                    "OBJ_MISMATCH".to_string(),
                                    format!(
                                        "[{}] obj={:.6e} known={:.6e} err={:.1}% {} {}",
                                        method_label,
                                        result.objective,
                                        baseline_objectives.get(&name).unwrap(),
                                        rel_err * 100.0,
                                        kkt_str,
                                        gap_str,
                                    ),
                                )
                            }
                            ObjCheckResult::NoRef => {
                                n_pass_noref += 1;
                                (
                                    "CHECKED[no_ref]".to_string(),
                                    format!(
                                        "[{}] obj={:.6e} {} {}",
                                        method_label, result.objective, kkt_str, gap_str
                                    ),
                                )
                            }
                        }
                    }
                } else {
                    n_fail += 1;
                    (
                        "FAIL:NumericalError".to_string(),
                        format!("[{}] obj={}", method_label, result.objective),
                    )
                }
            }
            SolveStatus::Infeasible => {
                // CSV に INFEASIBLE が記載されていれば正答 → PASS:Infeasible
                match expected_statuses.get(&name) {
                    Some(ExpectedStatus::Infeasible) => {
                        n_pass_infeasible += 1;
                        ("PASS:Infeasible".to_string(), String::new())
                    }
                    _ => {
                        n_fail += 1;
                        ("FAIL:Infeasible".to_string(), String::new())
                    }
                }
            }
            SolveStatus::Unbounded => {
                // CSV に UNBOUNDED が記載されていれば正答 → PASS:Unbounded
                match expected_statuses.get(&name) {
                    Some(ExpectedStatus::Unbounded) => {
                        n_pass_unbounded += 1;
                        ("PASS:Unbounded".to_string(), String::new())
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
                let label = kkt_gated_label("SUBOPTIMAL", kkt_max, KKT_FAIL_EPS);
                if label == "KKT_FAIL" {
                    n_kkt_fail += 1;
                } else {
                    n_suboptimal += 1;
                }
                let extra = if result.solution.is_empty() {
                    "obj=NA solution=EMPTY".to_string()
                } else if result.solution.len() != prob.num_vars {
                    format!(
                        "obj={:.3e} sol_len={}/{}_MISMATCH",
                        result.objective,
                        result.solution.len(),
                        prob.num_vars
                    )
                } else {
                    let x_inf = result.solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
                    format!("obj={:.6e} x_inf={:.2e}", result.objective, x_inf)
                };
                (
                    label.to_string(),
                    format!(
                        "[{}] iters={} {} {} {} {}",
                        method_label, result.iterations, extra, resid_str, kkt_str, gap_str
                    ),
                )
            }
            SolveStatus::Timeout => {
                n_timeout += 1;
                (
                    "TIMEOUT".to_string(),
                    format!(
                        "[{}] {:.3}s iters={}",
                        method_label, elapsed_s, result.iterations
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
            // BB driver から global path 経由で来た場合の caller 視点表示。
            // 単発 IPM 経路 (apply_bench_status_promotion 後) では現状出ない (Optimal 化 or
            // LocallyOptimal で別 arm を通る)。global path 統合時にここに乗る。
            SolveStatus::NonconvexLocal => {
                let label = kkt_gated_label("NONCONVEX_LOCAL", kkt_max, KKT_FAIL_EPS);
                if label == "KKT_FAIL" {
                    n_kkt_fail += 1;
                } else {
                    n_nonconvex_local += 1;
                }
                (
                    label.to_string(),
                    format!(
                        "[{}] obj={:.6e} {} {}",
                        method_label, result.objective, kkt_str, gap_str
                    ),
                )
            }
            SolveStatus::NonconvexGlobal => {
                let label = kkt_gated_label("NONCONVEX_GLOBAL", kkt_max, KKT_FAIL_EPS);
                if label == "KKT_FAIL" {
                    n_kkt_fail += 1;
                } else {
                    n_nonconvex_global += 1;
                }
                (
                    label.to_string(),
                    format!(
                        "[{}] obj={:.6e} {} {}",
                        method_label, result.objective, kkt_str, gap_str
                    ),
                )
            }
            SolveStatus::NotSupported(ref msg) => {
                n_not_supported += 1;
                (
                    "NOT_SUPPORTED".to_string(),
                    format!("[{}] {}", method_label, msg),
                )
            }
            _ => {
                n_fail += 1;
                ("FAIL:Unknown".to_string(), format!("[{}]", method_label))
            }
        };
        println!(
            "{:<24} {:>6} {:>6} {:>15} {:>10.3} {}",
            name, n, m, status_str, elapsed_s, note
        );
        // 追加情報行: solver詳細 + presolve削減量
        let presolve_info = if n_after != n || m_after != m || nnz_after != nnz_before {
            format!(
                "n={}→{} m={}→{} nnz={}→{}",
                n, n_after, m, m_after, nnz_before, nnz_after
            )
        } else {
            format!("n={} m={} nnz={} (no reduction)", n, m, nnz_before)
        };
        println!(
            "  => solver={} iters={} {} | presolve: {}",
            method_label, result.iterations, resid_str, presolve_info
        );
    }

    println!("{}", "-".repeat(80));
    println!();
    println!("=== Summary ===");
    println!("  PASS:              {}", n_pass);
    println!("  CHECKED[no_ref]:   {}", n_pass_noref);
    println!("  PASS:Infeasible:   {}", n_pass_infeasible);
    println!("  PASS:Unbounded:    {}", n_pass_unbounded);
    println!("  TIMEOUT:           {}", n_timeout);
    println!("  FAIL:              {}", n_fail);
    println!("  OBJ_MISMATCH:      {}", n_obj_mismatch);
    println!("  KKT_FAIL:          {}", n_kkt_fail);
    println!("  NONCONVEX:         {}", n_nonconvex);
    println!("  NONCONVEX_LOCAL:   {}", n_nonconvex_local);
    println!("  NONCONVEX_GLOBAL:  {}", n_nonconvex_global);
    println!("  SUBOPTIMAL:        {}", n_suboptimal);
    println!("  MAXITER:           {}", n_max_iter);
    println!("  ERROR:             {}", n_error);
    println!("  SKIP:              {}", n_skip);
    // SKIP 内訳を fact 出力 (parse error message 由来、推論排除).
    println!("    SKIP:integer:    {}", n_skip_integer);
    println!("    SKIP:qcqp:       {}", n_skip_qcqp);
    println!("    SKIP:other:      {}", n_skip_other);
    // solve 時点の out-of-scope declined (SolveStatus::NotSupported)。SKIP (parse時点)
    // とは別 bucket: parser は通過したが solver が対応外と正しく判定したケース。
    println!("  NOT_SUPPORTED:     {}", n_not_supported);
    println!(
        "  TOTAL:             {}",
        n_pass
            + n_pass_noref
            + n_pass_infeasible
            + n_pass_unbounded
            + n_timeout
            + n_fail
            + n_obj_mismatch
            + n_kkt_fail
            + n_nonconvex
            + n_nonconvex_local
            + n_nonconvex_global
            + n_suboptimal
            + n_max_iter
            + n_error
            + n_skip
            + n_not_supported
    );
}
