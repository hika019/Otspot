//! KKT残差独立検証バイナリ (cmd_230)
//!
//! ソルバー内部の判定を信頼せず、各PASS問題についてKKT条件を外部から再計算する。
//!
//! Usage:
//!   cargo run --release --features parallel --bin verify_solutions
//!   cargo run --release --features parallel --bin verify_solutions -- --qplib
//!   cargo run --release --features parallel --bin verify_solutions -- --solver ipm
//!
//! KKT条件（min 1/2 x^T Q x + c^T x  s.t. Ax <= b, lb <= x <= ub）:
//!   primal_feas : max(0, max_i(A*x - b)_i)        (Ax <= b 違反)
//!   bound_feas  : max(max_i(lb_i - x_i, 0), max_i(x_i - ub_i, 0))
//!   stat_resid  : ||Q*x + c - A^T*y||_inf           (相補性なし、縮小勾配)
//!   comp_slack  : max_i |y_i * (A_i*x - b_i)|       (制約の相補スラック)
//!   dual_feas   : y_i の最小値（<0 なら双対非実行可能）

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::env;
use std::path::Path;
use std::time::{Duration, Instant};

use solver::io::qps::parse_qps;
use solver::io::qplib::{parse_qplib, QplibError};
use solver::options::{QpSolverChoice, SolverOptions};
use solver::problem::{ConstraintType, SolveStatus};
use solver::qp::solve_qp_with;
use solver::QpProblem;

const EPS: f64 = 1e-6;

#[derive(Debug)]
struct KktResiduals {
    primal_feas: f64,  // max violation of Ax <= b
    bound_feas: f64,   // max violation of lb <= x <= ub
    stat_resid: f64,   // ||Qx + c - A^T y||_inf (reduced gradient norm)
    comp_slack: f64,   // max_i |y_i * (A_i*x - b_i)|
    min_dual_y: f64,   // min y_i (should be >= 0 for <= constraints)
}

/// KKT残差を独立計算する
fn compute_kkt_residuals(prob: &QpProblem, x: &[f64], y: &[f64]) -> KktResiduals {
    let n = prob.num_vars;
    let m = prob.num_constraints;

    // 1. Ax を計算
    let ax = prob.a.mat_vec_mul(x).unwrap_or_else(|_| vec![f64::NAN; m]);

    // primal_feas: Ge対応 (Ax >= b は (b - Ax).max(0), Eq は |Ax - b|, Le は (Ax - b).max(0))
    let primal_feas = ax
        .iter()
        .zip(prob.b.iter())
        .zip(prob.constraint_types.iter())
        .map(|((&axi, &bi), ct)| match ct {
            ConstraintType::Eq => (axi - bi).abs(),
            ConstraintType::Ge => (bi - axi).max(0.0),
            _ => (axi - bi).max(0.0),
        })
        .fold(0.0_f64, f64::max);

    // bound_feas: max violation of lb <= x <= ub
    let bound_feas = x
        .iter()
        .zip(prob.bounds.iter())
        .map(|(&xi, &(lb, ub))| {
            let lb_viol = if lb.is_finite() { (lb - xi).max(0.0) } else { 0.0 };
            let ub_viol = if ub.is_finite() { (xi - ub).max(0.0) } else { 0.0 };
            lb_viol.max(ub_viol)
        })
        .fold(0.0_f64, f64::max);

    // Q*x を計算
    let qx = prob.q.mat_vec_mul(x).unwrap_or_else(|_| vec![f64::NAN; n]);

    // A^T * y を計算（y が空の場合は 0 ベクトル）
    let aty = if y.is_empty() {
        vec![0.0_f64; n]
    } else {
        prob.a.transpose().mat_vec_mul(y).unwrap_or_else(|_| vec![f64::NAN; n])
    };

    // stat_resid: ||Qx + c - A^T y||_inf  (縮小勾配ノルム)
    let stat_resid = qx
        .iter()
        .zip(prob.c.iter())
        .zip(aty.iter())
        .map(|((&qxi, &ci), &atyi)| (qxi + ci - atyi).abs())
        .fold(0.0_f64, f64::max);

    // comp_slack: max_i |y_i * (A_i*x - b_i)|
    let comp_slack = if y.is_empty() || ax.len() != m {
        0.0
    } else {
        ax.iter()
            .zip(prob.b.iter())
            .zip(y.iter())
            .map(|((&axi, &bi), &yi)| (yi * (axi - bi)).abs())
            .fold(0.0_f64, f64::max)
    };

    // dual_feas: min(y) — should be >= 0 for <= constraints
    let min_dual_y = if y.is_empty() {
        0.0
    } else {
        y.iter().cloned().fold(f64::INFINITY, f64::min)
    };

    KktResiduals {
        primal_feas,
        bound_feas,
        stat_resid,
        comp_slack,
        min_dual_y,
    }
}

/// 1問を検証し、VIOLATIONならtrue、OKならfalseを返す
fn check_violation(r: &KktResiduals) -> bool {
    r.primal_feas > EPS
        || r.bound_feas > EPS
        || r.stat_resid > EPS
        || r.comp_slack > EPS
        || r.min_dual_y < -EPS
}

enum ParseResult {
    Ok(Box<QpProblem>),
    ParseErr(String),
    Timeout,
    Unsupported(String),
}

fn parse_qps_with_timeout(path: &Path, timeout_secs: u64) -> ParseResult {
    let path = path.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = parse_qps(&path);
        let _ = tx.send(r);
    });
    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Ok(p)) => ParseResult::Ok(Box::new(p)),
        Ok(Err(e)) => ParseResult::ParseErr(format!("{}", e)),
        Err(_) => ParseResult::Timeout,
    }
}

fn parse_qplib_with_timeout(path: &Path, timeout_secs: u64) -> ParseResult {
    let path = path.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = parse_qplib(&path);
        let _ = tx.send(r);
    });
    match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Ok(p)) => ParseResult::Ok(Box::new(p)),
        Ok(Err(QplibError::UnsupportedType(msg))) => ParseResult::Unsupported(msg),
        Ok(Err(e)) => ParseResult::ParseErr(format!("{:?}", e)),
        Err(_) => ParseResult::Timeout,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut use_qplib = false;
    let mut solver_choice = QpSolverChoice::Concurrent;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--qplib" => use_qplib = true,
            "--solver" => {
                i += 1;
                if i < args.len() {
                    solver_choice = match args[i].as_str() {
                        "ipm" => QpSolverChoice::Ipm,
                        "ippmm_new" => QpSolverChoice::IpPmmNew,
                        "concurrent" => QpSolverChoice::Concurrent,
                        other => {
                            eprintln!("Unknown solver: {}", other);
                            std::process::exit(1);
                        }
                    };
                }
            }
            _ => {}
        }
        i += 1;
    }

    let data_dir = if use_qplib { "data/qplib" } else { "data/maros_meszaros" };
    let dir = Path::new(data_dir);
    if !dir.exists() {
        eprintln!("Directory not found: {}", data_dir);
        std::process::exit(1);
    }

    let ext = if use_qplib { "qplib" } else { "qps" };
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case(ext))
                .unwrap_or(false)
        })
        .collect();
    files.sort();

    let solver_label = match solver_choice {
        QpSolverChoice::Concurrent => "Concurrent",
        QpSolverChoice::Ipm => "IPM",
        QpSolverChoice::IpPmmNew => "IP-PMM-New",
        _ => "Unknown",
    };

    let dataset = if use_qplib { "QPLIB" } else { "Maros-Meszaros" };
    println!("=== KKT Solution Verification: {} ({} files, solver={}) ===", dataset, files.len(), solver_label);
    println!("EPS = {:.0e}", EPS);
    println!();
    println!(
        "{:<20} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12}  KKT",
        "Problem", "Status", "pfeas", "bfeas", "stat_resid", "comp_slk", "min_dual_y"
    );
    println!("{}", "-".repeat(110));

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(10.0);
    opts.qp_solver = solver_choice;

    let mut n_pass = 0usize;
    let mut n_fail = 0usize;
    let mut n_skip = 0usize;
    let mut violations: Vec<(String, String, f64)> = vec![];

    for path in &files {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();

        let prob = if use_qplib {
            match parse_qplib_with_timeout(path, 30) {
                ParseResult::Ok(p) => *p,
                ParseResult::Unsupported(msg) => {
                    println!("{:<20} {:>8}  (skipped: {})", name, "SKIP", &msg[..msg.len().min(40)]);
                    n_skip += 1;
                    continue;
                }
                ParseResult::ParseErr(e) => {
                    println!("{:<20} {:>8}  (parse error: {})", name, "ERR", &e[..e.len().min(40)]);
                    n_skip += 1;
                    continue;
                }
                ParseResult::Timeout => {
                    println!("{:<20} {:>8}  (parse timeout)", name, "ERR");
                    n_skip += 1;
                    continue;
                }
            }
        } else {
            match parse_qps_with_timeout(path, 30) {
                ParseResult::Ok(p) => *p,
                ParseResult::ParseErr(e) => {
                    println!("{:<20} {:>8}  (parse error: {})", name, "ERR", &e[..e.len().min(40)]);
                    n_skip += 1;
                    continue;
                }
                ParseResult::Timeout => {
                    println!("{:<20} {:>8}  (parse timeout)", name, "ERR");
                    n_skip += 1;
                    continue;
                }
                ParseResult::Unsupported(_) => unreachable!(),
            }
        };

        let start = Instant::now();
        let result = solve_qp_with(&prob, &opts);
        let elapsed = start.elapsed().as_secs_f64();

        match result.status {
            SolveStatus::Optimal => {
                let r = compute_kkt_residuals(&prob, &result.solution, &result.dual_solution);
                let is_violation = check_violation(&r);
                let kkt_label = if is_violation { "VIOLATION" } else { "OK" };

                println!(
                    "{:<20} {:>8} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e}  {}",
                    name, "PASS",
                    r.primal_feas, r.bound_feas, r.stat_resid, r.comp_slack, r.min_dual_y,
                    kkt_label
                );

                if is_violation {
                    n_fail += 1;
                    // どの条件が違反しているか記録
                    let mut kinds = vec![];
                    if r.primal_feas > EPS { kinds.push(format!("pfeas={:.2e}", r.primal_feas)); }
                    if r.bound_feas > EPS { kinds.push(format!("bfeas={:.2e}", r.bound_feas)); }
                    if r.stat_resid > EPS { kinds.push(format!("stat={:.2e}", r.stat_resid)); }
                    if r.comp_slack > EPS { kinds.push(format!("comp={:.2e}", r.comp_slack)); }
                    if r.min_dual_y < -EPS { kinds.push(format!("min_y={:.2e}", r.min_dual_y)); }
                    let max_viol = r.primal_feas
                        .max(r.bound_feas)
                        .max(r.stat_resid)
                        .max(r.comp_slack)
                        .max((-r.min_dual_y).max(0.0));
                    violations.push((name.clone(), kinds.join(", "), max_viol));
                } else {
                    n_pass += 1;
                }
            }
            other_status => {
                println!(
                    "{:<20} {:>8}  ({:.3}s)",
                    name,
                    format!("{}", other_status).to_uppercase(),
                    elapsed
                );
                n_skip += 1;
            }
        }
    }

    println!("{}", "=".repeat(110));
    println!();
    println!("=== Summary ===");
    println!("  Total files       : {}", files.len());
    println!("  PASS (KKT OK)     : {}", n_pass);
    println!("  PASS (KKT VIOLATION): {}", n_fail);
    println!("  Non-PASS (skip)   : {}", n_skip);
    println!();

    if violations.is_empty() {
        println!("All PASS solutions satisfy KKT conditions within eps={:.0e}.", EPS);
    } else {
        println!("KKT VIOLATIONS ({} problems):", violations.len());
        for (name, kinds, max_viol) in &violations {
            println!("  {:<20}  max_viol={:.3e}  [{}]", name, max_viol, kinds);
        }
    }
}
