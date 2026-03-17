//! QPLIBベンチマーク
//!
//! Usage: bench_qplib [data_dir]
//! 指定ディレクトリ内の全 *.qplib ファイルを parse_qplib → solve_qp_with_options で実行し、
//! 結果テーブルを stdout に出力する。
//!
//! Maros-Meszaros（QPS形式）とは独立した集計。
//! 対応問題タイプ: *C*（連続変数）かつ L/B/N（線形/境界/無制約）。
//! 非対応（整数変数・二次制約）は SKIP として記録する。

use std::env;
use std::path::Path;
use std::time::{Duration, Instant};

use solver::io::qplib::{parse_qplib, QplibError};
use solver::options::{QpSolverChoice, SolverOptions};
use solver::presolve::{run_qp_presolve_phase1, run_qp_presolve_phase2};
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use solver::QpProblem;

enum BenchError {
    Parse(QplibError),
    ParseTimeout,
    Unsupported(String),
}

fn parse_with_timeout(path: &Path, timeout_secs: u64) -> Result<QpProblem, BenchError> {
    let path = path.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = parse_qplib(&path);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Ok(prob)) => Ok(prob),
        Ok(Err(QplibError::UnsupportedType(msg))) => Err(BenchError::Unsupported(msg)),
        Ok(Err(e)) => Err(BenchError::Parse(e)),
        Err(_) => Err(BenchError::ParseTimeout),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // 引数パース: [data_dir] [--solver as|ipm|ipm-schur] [--eps <value>] [--timeout <secs>]
    let mut data_dir = "data/qplib".to_string();
    let mut solver_choice = QpSolverChoice::Concurrent;
    let mut eps: f64 = 1e-6;
    let mut timeout_secs: f64 = 10.0;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: bench_qplib [data_dir] [--solver ipm|ipm-schur] [--eps <value>] [--timeout <secs>]");
            println!("  --solver   Solver to use (default: concurrent/auto)");
            println!("  --eps      Convergence tolerance (default: 1e-6)");
            println!("  --timeout  Solver timeout in seconds (default: 10.0)");
            std::process::exit(0);
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
                solver_choice = match args[i].as_str() {
                    "ipm" => QpSolverChoice::Ipm,
                    "ipm-schur" => QpSolverChoice::IpmSchur,
                    "concurrent" => QpSolverChoice::Concurrent,
                    "ippmm_new" => QpSolverChoice::IpPmmNew,
                    other => {
                        eprintln!("Unknown solver: {}. Use ipm|ipm-schur|concurrent|ippmm_new", other);
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
        QpSolverChoice::Concurrent => "Concurrent",
        QpSolverChoice::Ipm => "IPM",
        QpSolverChoice::IpmSchur => "IPM-Schur",
        QpSolverChoice::IpPmmNew => "IP-PMM-New",
        _ => "Unknown",
    };
    println!("Solver: {}", solver_label);

    println!(
        "{:<24} {:>6} {:>6} {:>12} {:>10} Note",
        "Problem", "n", "m", "Status", "Time(s)"
    );
    println!("{}", "-".repeat(80));

    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_error = 0usize;
    let mut n_timeout = 0usize;
    let mut n_max_iter = 0usize;
    let mut n_suboptimal = 0usize;
    let mut n_skip = 0usize;
    let mut n_nonconvex = 0usize;

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
                    "{:<24} {:>6} {:>6} {:>12} {:>10.3} {}",
                    name, "-", "-", "SKIP", 0.0, note
                );
                n_skip += 1;
                continue;
            }
            Err(BenchError::Parse(e)) => {
                let note = format!("{}", e);
                println!(
                    "{:<24} {:>6} {:>6} {:>12} {:>10.3} {}",
                    name, "?", "?", "PARSE_ERR", 0.0, &note[..note.len().min(40)]
                );
                n_error += 1;
                continue;
            }
            Err(BenchError::ParseTimeout) => {
                println!(
                    "{:<24} {:>6} {:>6} {:>12} {:>10.3} ",
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
            Some(QpSolverChoice::Ipm) => "ipm",
            Some(QpSolverChoice::IpmSchur) => "ipm-schur",
            Some(QpSolverChoice::Concurrent) => "concurrent",
            Some(QpSolverChoice::IpPmmNew) => "ippmm_new",
            Some(_) => "other",
            None => "-",
        };
        let resid_str = match result.final_residuals {
            Some((pf, df, gap)) => format!("pf={:.1e} df={:.1e} gap={:.1e}", pf, df, gap),
            None => String::new(),
        };
        let (status_str, note) = match result.status {
            SolveStatus::Optimal => {
                if result.objective.is_finite() {
                    n_pass += 1;
                    ("PASS".to_string(), format!("[{}] obj={:.6e}", method_label, result.objective))
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
                ("SUBOPTIMAL".to_string(), format!("[{}] iters={} {}", method_label, result.iterations, resid_str))
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
            "{:<24} {:>6} {:>6} {:>12} {:>10.3} {}",
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
    println!("  PASS:      {}", n_pass);
    println!("  FAIL:      {}", n_fail);
    println!("  MAXITER:   {}", n_max_iter);
    println!("  SUBOPTIMAL: {}", n_suboptimal);
    println!("  TIMEOUT:   {}", n_timeout);
    println!("  NONCONVEX: {}", n_nonconvex);
    println!("  ERROR:     {}", n_error);
    println!("  SKIP:      {}", n_skip);
    println!(
        "  TOTAL:     {}",
        n_pass + n_fail + n_max_iter + n_suboptimal + n_timeout + n_nonconvex + n_error + n_skip
    );
}
