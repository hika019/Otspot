//! Maros-Meszaros QPS ベンチマーク
//!
//! Usage: qps_benchmark <data_dir>
//! 指定ディレクトリ内の全*.QPSファイルを parse_qps → solve_qp_with_options で実行し、
//! 結果テーブルをstdoutに出力する。
//!
//! 各問題に10秒のタイムアウトを設ける（solver内部の協調的タイムアウト機構を使用）。

use std::env;
use std::path::Path;
use std::time::{Duration, Instant};

use solver::io::qps::{parse_qps, QpsError};
use solver::options::{QpSolverChoice, SolverOptions};
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with_options;
use solver::QpProblem;

enum BenchError {
    Parse(QpsError),
    ParseTimeout,
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

fn main() {
    let args: Vec<String> = env::args().collect();

    // 引数パース: [data_dir] [--solver as|admm|ipm]
    let mut data_dir = "data/maros_meszaros".to_string();
    let mut solver_choice = QpSolverChoice::Auto;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--solver" {
            i += 1;
            if i < args.len() {
                solver_choice = match args[i].as_str() {
                    "as" => QpSolverChoice::ActiveSet,
                    "admm" => QpSolverChoice::Admm,
                    "ipm" => QpSolverChoice::Ipm,
                    other => {
                        eprintln!("Unknown solver: {}. Use as|admm|ipm", other);
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
        "{:<20} {:>6} {:>6} {:>12} {:>10} Error",
        "Problem", "n", "m", "Status", "Time(s)"
    );
    println!("{}", "-".repeat(75));

    // 集計
    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_error = 0usize;
    let mut n_timeout = 0usize;
    let mut n_max_iter = 0usize;

    let solver_label = match solver_choice {
        QpSolverChoice::Auto => "Auto",
        QpSolverChoice::ActiveSet => "AS",
        QpSolverChoice::Admm => "ADMM",
        QpSolverChoice::Ipm => "IPM",
    };
    println!("Solver: {}", solver_label);

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    opts.qp_solver = solver_choice;

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
                    "{:<20} {:>6} {:>6} {:>12} {:>10.3} {}",
                    name, "?", "?", "PARSE_ERR", 0.0, &note[..note.len().min(40)]
                );
                n_error += 1;
                continue;
            }
            Err(BenchError::ParseTimeout) => {
                println!(
                    "{:<20} {:>6} {:>6} {:>12} {:>10.3} ",
                    name, "?", "?", "PARSE_TIMEOUT", 0.0
                );
                n_timeout += 1;
                continue;
            }
        };

        println!("PARSE_DONE: {} ({:.2}s)", name, parse_start.elapsed().as_secs_f64());

        let n = prob.num_vars;
        let m = prob.num_constraints;

        println!("SOLVE_START: {}", name);
        let start = Instant::now();
        let result = solve_qp_with_options(&prob, &opts);
        let elapsed_s = start.elapsed().as_secs_f64();
        println!("SOLVE_DONE: {} {:?} ({:.3}s)", name, result.status, elapsed_s);

        let (status_str, note) = match result.status {
            SolveStatus::Optimal => {
                n_pass += 1;
                ("PASS".to_string(), format!("obj={:.6e}", result.objective))
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
                    format!("iters={}", result.iterations),
                )
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
            "{:<20} {:>6} {:>6} {:>12} {:>10.3} {}",
            name, n, m, status_str, elapsed_s, note
        );
    }

    println!("{}", "-".repeat(75));
    println!();
    println!("=== Summary ===");
    println!("  PASS:    {}", n_pass);
    println!("  FAIL:    {}", n_fail);
    println!("  MAXITER: {}", n_max_iter);
    println!("  TIMEOUT: {}", n_timeout);
    println!("  ERROR:   {}", n_error);
    println!("  TOTAL:   {}", n_pass + n_fail + n_max_iter + n_timeout + n_error);
}
