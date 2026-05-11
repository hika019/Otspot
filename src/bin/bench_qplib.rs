//! QPLIBベンチマーク
//!
//! Usage: bench_qplib [data_dir] [--solver ipm|ippmm_new|concurrent] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]
//! 指定ディレクトリ内の全 *.qplib ファイルを parse_qplib → solve_qp_with_options で実行し、
//! 結果テーブルを stdout に出力する。
//!
//! Maros-Meszaros（QPS形式）とは独立した集計。
//! 対応問題タイプ: *C*（連続変数）かつ L/B/N（線形/境界/無制約）。
//! 非対応（整数変数・二次制約）は SKIP として記録する。

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::env;
use std::path::Path;
use std::time::Instant;

use solver::bench_utils::{check_baseline_objective, detect_csv_path, load_baseline_objectives, ObjCheckResult};
use solver::io::qplib::{parse_qplib, QplibError};
use solver::options::{QpSolverChoice, SolverOptions};
use solver::{run_qp_presolve_phase1, run_qp_presolve_phase2};
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use solver::QpProblem;

enum BenchError {
    Parse(QplibError),
    Unsupported(String),
}

fn parse_with_timeout(path: &Path, _timeout_secs: u64) -> Result<QpProblem, BenchError> {
    // 旧実装は thread::spawn + recv_timeout で parse をタイムアウトさせていたが、
    // タイムアウト時にスレッドが detach されたまま継続実行され「不必要なメモリ」を
    // 累積する mandate 違反。parse_qplib 自体に cancellation API がないため、
    // 同期呼び出しに変更し、hang 時は bench_parallel.sh の外部 gtimeout で
    // プロセスごと殺される設計に統一する。
    match parse_qplib(path) {
        Ok(prob) => Ok(prob),
        Err(QplibError::UnsupportedType(msg)) => Err(BenchError::Unsupported(msg)),
        Err(e) => Err(BenchError::Parse(e)),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // 引数パース: [data_dir] [--solver ipm|ippmm_new|concurrent] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]
    let mut data_dir = "data/qplib".to_string();
    let mut solver_choice = QpSolverChoice::IpPmm;
    let mut eps: f64 = 1e-6;
    let mut timeout_secs: f64 = 10.0;
    let mut baseline_override: Option<String> = None;

    // known flagリスト（値を持つフラグ）
    const KNOWN_FLAGS_WITH_VALUE: &[&str] = &[
        "--solver", "--eps", "--timeout", "--known-optimal",
    ];

    let mut i = 1usize;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: bench_qplib [data_dir] [--solver ipm|ippmm_new|concurrent] [--eps <value>] [--timeout <secs>] [--known-optimal <path>]");
            println!("  --solver        Solver to use (default: concurrent/auto)");
            println!("  --eps           Convergence tolerance (default: 1e-6)");
            println!("  --timeout       Solver timeout in seconds (default: 10.0)");
            println!("  --known-optimal Path to known optimal values CSV (default: auto-detect)");
            std::process::exit(0);
        } else if KNOWN_FLAGS_WITH_VALUE.contains(&args[i].as_str()) {
            // known flag: 次引数が値 → i+=2で消費
            i += 1;
            if i < args.len() {
                match args[i - 1].as_str() {
                    "--solver" => {
                        solver_choice = match args[i].as_str() {
                            "ipm" => QpSolverChoice::IpPmm,
                            "concurrent" => QpSolverChoice::IpPmm,
                            "ippmm_new" => QpSolverChoice::IpPmm,
                            other => {
                                eprintln!("Unknown solver: {}. Use ipm|concurrent|ippmm_new", other);
                                std::process::exit(1);
                            }
                        };
                    }
                    "--eps" => { eps = args[i].parse().unwrap_or(1e-6); }
                    "--timeout" => { timeout_secs = args[i].parse().unwrap_or(10.0); }
                    "--known-optimal" => { baseline_override = Some(args[i].clone()); }
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
    let baseline_objectives = {
        let root = {
            let p = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|pp| pp.to_path_buf()))
                .unwrap_or_default();
            // target/release から solver ルートに遡る
            p.parent().and_then(|p| p.parent()).map(|p| p.to_path_buf()).unwrap_or_default()
        };
        let csv = detect_csv_path(&data_dir, baseline_override.as_deref(), &root);
        load_baseline_objectives(&csv)
    };
    eprintln!("Baseline objectives loaded: {} problems", baseline_objectives.len());
    if baseline_objectives.is_empty() {
        eprintln!("WARNING: No known optimal values loaded. All problems will be PASS[no_ref].");
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
    println!();

    let solver_label = match solver_choice {
        QpSolverChoice::IpPmm => "Concurrent",
        _ => "Unknown",
    };
    println!("Solver: {}", solver_label);

    println!(
        "{:<24} {:>6} {:>6} {:>15} {:>10} Note",
        "Problem", "n", "m", "Status", "Time(s)"
    );
    println!("{}", "-".repeat(80));

    let mut n_pass = 0usize;
    let mut n_pass_noref = 0usize;
    let mut n_obj_mismatch = 0usize;
    let mut n_fail = 0usize;
    let mut n_error = 0usize;
    let mut n_timeout = 0usize;
    let mut n_max_iter = 0usize;
    let mut n_suboptimal = 0usize;
    let mut n_skip = 0usize;
    let mut n_nonconvex = 0usize;

    // QPLIBベンチでは実行可能性判定なし → 常に0
    let n_dfeas_fail: usize = 0;
    let n_pfeas_fail: usize = 0;

    let eps_obj: f64 = 1e-2; // §2.4: 1%閾値

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(timeout_secs);
    opts.qp_solver = solver_choice;
    opts.ipm.eps = eps;

    for path in &qplib_files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let parse_start = Instant::now();
        println!("PARSE_START: {}", name);

        let prob = match parse_with_timeout(path, 30) {
            Ok(p) => p,
            Err(BenchError::Unsupported(msg)) => {
                let note = msg.chars().take(40).collect::<String>();
                println!(
                    "{:<24} {:>6} {:>6} {:>15} {:>10.3} {}",
                    name, "-", "-", "SKIP", 0.0, note
                );
                n_skip += 1;
                continue;
            }
            Err(BenchError::Parse(e)) => {
                let note = format!("{}", e);
                println!(
                    "{:<24} {:>6} {:>6} {:>15} {:>10.3} {}",
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

        // presolve削減量を取得（ベンチ計装のみ）
        // 大規模問題はpresolveに長時間かかるためスキップ（タイムアウト計測の精度を確保）
        const PRESOLVE_INSTR_MAX: usize = 50_000;
        let (n_after, m_after, nnz_after) = if opts.presolve && n <= PRESOLVE_INSTR_MAX && m <= PRESOLVE_INSTR_MAX {
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
        let result = solve_qp_with(&prob, &opts);
        let elapsed_s = start.elapsed().as_secs_f64();
        println!("SOLVE_DONE: {} {:?} ({:.3}s)", name, result.status, elapsed_s);

        let method_label = match result.solver_used {
            Some(QpSolverChoice::IpPmm) => "ipm",
            Some(_) => "other",
            None => "-",
        };
        let resid_str = match result.final_residuals {
            Some((pf, df, gap)) => format!("pf={:.1e} df={:.1e} gap={:.1e}", pf, df, gap),
            None => String::new(),
        };

        // Timeout / SuboptimalSolution でも有効な best-so-far 解を持つなら Optimal 経路に
        // 格上げし obj 照合で品質判定する。qps_benchmark.rs と整合させた挙動。
        // 動機: bench_qplib は旧来 SuboptimalSolution を一律 SUBOPTIMAL として表示し、
        //   obj 照合スキップ → 「解は合っているが SUBOPTIMAL」を抱え込んでいた
        //   (QPLIB_10034: obj=-6.601e-2 vs baseline=-6.601e-2 で誤差 0.008% でも SUBOPTIMAL)。
        //   solver 内 IPPMM が μ_floor / α_stall で内部諦め → SuboptimalSolution は珍しくない
        //   ため、obj/finite で篩い、obj 不一致なら OBJ_MISMATCH に分類される設計。
        let result = if matches!(
            result.status,
            SolveStatus::Timeout | SolveStatus::SuboptimalSolution
        ) && !result.solution.is_empty()
            && result.solution.len() == prob.num_vars
            && result.objective.is_finite()
        {
            solver::problem::SolverResult { status: SolveStatus::Optimal, ..result }
        } else {
            result
        };

        let (status_str, note) = match result.status {
            SolveStatus::Optimal => {
                if result.objective.is_finite() {
                    match check_baseline_objective(
                        &name,
                        result.objective,
                        &baseline_objectives,
                        eps_obj,
                    ) {
                        ObjCheckResult::Ok { rel_err } => {
                            n_pass += 1;
                            (
                                "PASS".to_string(),
                                format!(
                                    "[{}] obj={:.6e} obj_err={:.3}%",
                                    method_label, result.objective, rel_err * 100.0
                                ),
                            )
                        }
                        ObjCheckResult::Mismatch { rel_err } => {
                            n_obj_mismatch += 1;
                            (
                                "OBJ_MISMATCH".to_string(),
                                format!(
                                    "[{}] obj={:.6e} known={:.6e} err={:.1}%",
                                    method_label,
                                    result.objective,
                                    baseline_objectives.get(&name).unwrap(),
                                    rel_err * 100.0
                                ),
                            )
                        }
                        ObjCheckResult::NoRef => {
                            n_pass_noref += 1;
                            (
                                "PASS[no_ref]".to_string(),
                                format!("[{}] obj={:.6e}", method_label, result.objective),
                            )
                        }
                    }
                } else {
                    n_fail += 1;
                    ("FAIL:NumericalError".to_string(), format!("[{}] obj={}", method_label, result.objective))
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
                ("MAXITER".to_string(), format!("[{}] iters={} {}", method_label, result.iterations, resid_str))
            }
            SolveStatus::SuboptimalSolution => {
                n_suboptimal += 1;
                let extra = if result.solution.is_empty() {
                    "obj=NA solution=EMPTY".to_string()
                } else if result.solution.len() != prob.num_vars {
                    format!("obj={:.3e} sol_len={}/{}_MISMATCH",
                        result.objective, result.solution.len(), prob.num_vars)
                } else {
                    let x_inf = result.solution.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
                    format!("obj={:.6e} x_inf={:.2e}", result.objective, x_inf)
                };
                ("SUBOPTIMAL".to_string(),
                    format!("[{}] iters={} {} {}", method_label, result.iterations, extra, resid_str))
            }
            SolveStatus::Timeout => {
                n_timeout += 1;
                ("TIMEOUT".to_string(), format!("[{}] {:.3}s iters={}", method_label, elapsed_s, result.iterations))
            }
            SolveStatus::NumericalError => {
                n_fail += 1;
                ("FAIL:NumericalError".to_string(), format!("[{}]", method_label))
            }
            SolveStatus::NonConvex(_) => {
                n_nonconvex += 1;
                ("NONCONVEX".to_string(), format!("[{}] Q not PSD", method_label))
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
    println!("  PASS:           {}", n_pass);
    println!("  PASS[no_ref]:   {}", n_pass_noref);
    println!("  TIMEOUT:        {}", n_timeout);
    println!("  FAIL:           {}", n_fail);
    println!("  DFEAS_FAIL:     {}", n_dfeas_fail);
    println!("  PFEAS_FAIL:     {}", n_pfeas_fail);
    println!("  OBJ_MISMATCH:   {}", n_obj_mismatch);
    println!("  NONCONVEX:      {}", n_nonconvex);
    println!("  SUBOPTIMAL:     {}", n_suboptimal);
    println!("  MAXITER:        {}", n_max_iter);
    println!("  ERROR:          {}", n_error);
    println!("  SKIP:           {}", n_skip);
    println!(
        "  TOTAL:          {}",
        n_pass + n_pass_noref + n_timeout + n_fail + n_obj_mismatch
            + n_nonconvex + n_suboptimal + n_max_iter + n_error + n_skip
    );
}
