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

    // 引数パース: [data_dir] [--solver as|ipm|ipm-schur] [--eps <value>]
    let mut data_dir = "data/qplib".to_string();
    let mut solver_choice = QpSolverChoice::Concurrent;
    let mut eps: f64 = 1e-8;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            println!("Usage: bench_qplib [data_dir] [--solver as|ipm|ipm-schur] [--eps <value>]");
            println!("  --solver  Solver to use (default: concurrent/auto)");
            println!("  --eps     Convergence tolerance (default: 1e-8)");
            std::process::exit(0);
        } else if args[i] == "--eps" {
            i += 1;
            if i < args.len() {
                eps = args[i].parse().unwrap_or(1e-8);
            }
        } else if args[i] == "--solver" {
            i += 1;
            if i < args.len() {
                solver_choice = match args[i].as_str() {
                    "as" => QpSolverChoice::ActiveSet,
                    "ipm" => QpSolverChoice::Ipm,
                    "ipm-schur" => QpSolverChoice::IpmSchur,
                    other => {
                        eprintln!("Unknown solver: {}. Use as|ipm|ipm-schur", other);
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
        QpSolverChoice::ActiveSet => "AS",
        QpSolverChoice::Ipm => "IPM",
        QpSolverChoice::IpmSchur => "IPM-Schur",
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
    let mut n_skip = 0usize;

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
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

        println!("SOLVE_START: {}", name);
        let start = Instant::now();
        let result = solve_qp_with(&prob, &opts);
        let elapsed_s = start.elapsed().as_secs_f64();
        println!("SOLVE_DONE: {} {:?} ({:.3}s)", name, result.status, elapsed_s);

        let (status_str, note) = match result.status {
            SolveStatus::Optimal => {
                if result.objective.is_finite() {
                    n_pass += 1;
                    ("PASS".to_string(), format!("obj={:.6e}", result.objective))
                } else {
                    n_fail += 1;
                    ("FAIL:NumericalError".to_string(), format!("obj={}", result.objective))
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
                ("MAXITER".to_string(), format!("iters={}", result.iterations))
            }
            SolveStatus::Timeout => {
                n_timeout += 1;
                ("TIMEOUT".to_string(), format!("{:.3}s", elapsed_s))
            }
            SolveStatus::NumericalError => {
                n_fail += 1;
                ("FAIL:NumericalError".to_string(), String::new())
            }
        };
        println!(
            "{:<24} {:>6} {:>6} {:>12} {:>10.3} {}",
            name, n, m, status_str, elapsed_s, note
        );
    }

    println!("{}", "-".repeat(80));
    println!();
    println!("=== Summary ===");
    println!("  PASS:    {}", n_pass);
    println!("  FAIL:    {}", n_fail);
    println!("  MAXITER: {}", n_max_iter);
    println!("  TIMEOUT: {}", n_timeout);
    println!("  ERROR:   {}", n_error);
    println!("  SKIP:    {}", n_skip);
    println!(
        "  TOTAL:   {}",
        n_pass + n_fail + n_max_iter + n_timeout + n_error + n_skip
    );
}
