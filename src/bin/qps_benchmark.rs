//! Maros-Meszaros QPS ベンチマーク
//!
//! Usage: qps_benchmark <data_dir>
//! 指定ディレクトリ内の全*.QPSファイルを parse_qps → solve_qp で実行し、
//! 結果テーブルをstdoutに出力する。
//!
//! 各問題に10秒のウォールクロックタイムアウトを設ける。

use std::env;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use solver::io::qps::parse_qps;
use solver::qp::solve_qp;
use solver::problem::SolveStatus;

fn main() {
    let args: Vec<String> = env::args().collect();
    let data_dir = if args.len() >= 2 {
        args[1].clone()
    } else {
        "data/maros_meszaros".to_string()
    };

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
        "{:<20} {:>6} {:>6} {:>12} {:>10} {}",
        "Problem", "n", "m", "Status", "Time(s)", "Error"
    );
    println!("{}", "-".repeat(75));

    // 集計
    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_error = 0usize;
    let mut n_timeout = 0usize;
    let mut n_max_iter = 0usize;

    // 結果行を収集して最後にもう一度出力
    struct Row {
        name: String,
        n: usize,
        m: usize,
        status: String,
        elapsed: f64,
        note: String,
    }
    let mut rows: Vec<Row> = Vec::new();

    for path in &qps_files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        // パース
        let prob = match parse_qps(path) {
            Ok(p) => p,
            Err(e) => {
                let note = format!("{}", e);
                println!(
                    "{:<20} {:>6} {:>6} {:>12} {:>10.3} {}",
                    name, "?", "?", "PARSE_ERR", 0.0, &note[..note.len().min(40)]
                );
                n_error += 1;
                rows.push(Row {
                    name,
                    n: 0,
                    m: 0,
                    status: "PARSE_ERR".to_string(),
                    elapsed: 0.0,
                    note,
                });
                continue;
            }
        };

        let n = prob.num_vars;
        let m = prob.num_constraints;

        // solve in a thread with 10s timeout
        let prob_clone = prob.clone();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let start = Instant::now();
            let result = solve_qp(&prob_clone);
            let elapsed = start.elapsed();
            let _ = tx.send((result, elapsed));
        });

        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok((result, elapsed)) => {
                let elapsed_s = elapsed.as_secs_f64();
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
                };
                println!(
                    "{:<20} {:>6} {:>6} {:>12} {:>10.3} {}",
                    name, n, m, status_str, elapsed_s, note
                );
                rows.push(Row {
                    name,
                    n,
                    m,
                    status: status_str,
                    elapsed: elapsed_s,
                    note,
                });
            }
            Err(_) => {
                n_timeout += 1;
                println!(
                    "{:<20} {:>6} {:>6} {:>12} {:>10.3} {}",
                    name, n, m, "TIMEOUT", 10.0, "(10s wall clock exceeded)"
                );
                rows.push(Row {
                    name,
                    n,
                    m,
                    status: "TIMEOUT".to_string(),
                    elapsed: 10.0,
                    note: "10s wall clock exceeded".to_string(),
                });
            }
        }
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
