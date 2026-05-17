//! Screening test: run solve() on all available Netlib LP problems and report failures.
//!
//! 目的: LP simplex 経路 (solve()) のバグ網羅検出。
//! 13 既存 LP テストでは検出できない問題 (RANGES / FR / OBJSENSE MAX / 負境界など) を露出させる。
//!
//! 実行: `cargo test --release --test lp_coverage_screen -- --nocapture --test-threads=1`
//!
//! 各問題は 20s でタイムアウト。テスト全体で 90 問 * 平均数秒 で 3 分以内を目標。
//!
//! # ベースラインとobj_offsetの扱い
//! ベースライン CSV の値は Netlib 公式値 (https://www.netlib.org/lp/data/readme, MINOS 5.3 計算)。
//! Netlib 値は純粋な c^T x（N-row RHS を目的関数定数として含まない）。
//! このソルバーは N-row RHS を problem.obj_offset として保存し、報告する目的関数値に加算する。
//! 比較時は exp_adjusted = netlib_ref + problem.obj_offset で補正する。
//! 例: e226 → netlib_ref=-18.751929, obj_offset=-7.113, solver_reported=-25.864929, exp_adj=-25.864929 → PASS.

use solver::io::qps::parse_qps;
use solver::options::SolverOptions;
use solver::problem::SolveStatus;
use solver::qp::solve_qp_with;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

const PER_PROBLEM_TIMEOUT_SEC: f64 = 20.0;
const PROBLEMS_DIR: &str = "data/lp_problems";
const BASELINE_CSV: &str = "data/baseline_objectives/netlib_lp.csv";
/// 相対誤差許容: 0.1%（Netlib 公式値との比較基準）
const REL_TOL: f64 = 1e-3;

fn load_baseline() -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let content = match fs::read_to_string(BASELINE_CSV) {
        Ok(c) => c,
        Err(e) => panic!("Failed to read baseline {}: {}", BASELINE_CSV, e),
    };
    for line in content.lines() {
        if line.starts_with('#') || line.is_empty() || line.starts_with("problem_name") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() >= 2 {
            if let Ok(v) = cols[1].parse::<f64>() {
                map.insert(cols[0].to_string(), v);
            }
        }
    }
    map
}

#[derive(Debug)]
#[allow(dead_code)]
enum Verdict {
    Optimal,
    ObjMismatch { got: f64, expected: f64, rel_err: f64 },
    BadStatus { status: SolveStatus, expected_optimal: f64 },
    Timeout,
    Slow { secs: f64 },
}

/// 90 LP問題全体のスクリーニング。実行時間が長い (約5-6分) ため `#[ignore]` で隔離。
/// 手動実行: `cargo test --release --test lp_coverage_screen -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "heavy (~5-6 min: 90 LP screen、要 data/lp_problems/)、cargo test --release で個別実行"]
fn lp_coverage_screen_all() {
    let dir = Path::new(PROBLEMS_DIR);
    if !dir.exists() {
        eprintln!("SKIP: {} not found", PROBLEMS_DIR);
        return;
    }
    let baseline = load_baseline();

    let mut entries: Vec<_> = fs::read_dir(dir)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().map(|s| s == "QPS" || s == "qps").unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut bugs: Vec<(String, Verdict, f64)> = Vec::new();
    let mut pass = 0;
    let mut total_time = 0.0;

    for entry in &entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();

        // タイムアウト保護: 大規模問題は20s以内で screening を終わらせる
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(PER_PROBLEM_TIMEOUT_SEC);

        let problem = match parse_qps(&path) {
            Ok(p) => p,
            Err(e) => {
                bugs.push((
                    name.clone(),
                    Verdict::BadStatus {
                        status: SolveStatus::NumericalError,
                        expected_optimal: 0.0,
                    },
                    0.0,
                ));
                eprintln!("[parse_fail] {}: {:?}", name, e);
                continue;
            }
        };

        let start = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve_qp_with(&problem, &opts)
        }));
        let elapsed = start.elapsed().as_secs_f64();
        total_time += elapsed;

        match result {
            Err(_) => {
                bugs.push((
                    name.clone(),
                    Verdict::BadStatus {
                        status: SolveStatus::NumericalError,
                        expected_optimal: *baseline.get(&name).unwrap_or(&0.0),
                    },
                    elapsed,
                ));
                eprintln!("[PANIC] {}: panicked during solve", name);
                continue;
            }
            Ok(r) => {
                let expected = baseline.get(&name).copied();
                match (r.status, expected) {
                    (SolveStatus::Optimal, Some(exp)) => {
                        // ベースライン CSV は Netlib 公式値 (pure c^T x, N-row RHS 除外)。
                        // ソルバーは problem.obj_offset (N-row RHS) を目的関数値に加算して報告する。
                        // 比較のため: exp_adjusted = netlib_ref + problem.obj_offset
                        let exp_adjusted = exp + problem.obj_offset;
                        let denom = exp_adjusted.abs().max(1.0);
                        let rel_err = (r.objective - exp_adjusted).abs() / denom;
                        if rel_err > REL_TOL {
                            bugs.push((
                                name.clone(),
                                Verdict::ObjMismatch {
                                    got: r.objective,
                                    expected: exp_adjusted,
                                    rel_err,
                                },
                                elapsed,
                            ));
                            if problem.obj_offset != 0.0 {
                                eprintln!(
                                    "[OBJ_MISMATCH] {}: got={:.6e} netlib_ref={:.6e} obj_offset={:.6e} exp_adj={:.6e} rel={:.2e} time={:.2}s",
                                    name, r.objective, exp, problem.obj_offset, exp_adjusted, rel_err, elapsed
                                );
                            } else {
                                eprintln!(
                                    "[OBJ_MISMATCH] {}: got={:.6e} expected={:.6e} rel={:.2e} time={:.2}s",
                                    name, r.objective, exp_adjusted, rel_err, elapsed
                                );
                            }
                        } else {
                            pass += 1;
                            // 小規模問題 (<200 vars) で 60s 以上は遅すぎ
                            if problem.num_vars < 200 && elapsed > 30.0 {
                                bugs.push((
                                    name.clone(),
                                    Verdict::Slow { secs: elapsed },
                                    elapsed,
                                ));
                                eprintln!("[SLOW] {}: small problem took {:.2}s", name, elapsed);
                            } else if problem.obj_offset != 0.0 {
                                eprintln!(
                                    "[OK] {}: obj={:.6e} (netlib_ref={:.6e} + obj_offset={:.6e}) time={:.2}s",
                                    name, r.objective, exp, problem.obj_offset, elapsed
                                );
                            } else {
                                eprintln!(
                                    "[OK] {}: obj={:.6e} time={:.2}s",
                                    name, r.objective, elapsed
                                );
                            }
                        }
                    }
                    (SolveStatus::Optimal, None) => {
                        // baseline 不在: KKT verify はせず PASS[no_ref]
                        pass += 1;
                        eprintln!(
                            "[OK_NO_REF] {}: obj={:.6e} time={:.2}s",
                            name, r.objective, elapsed
                        );
                    }
                    (SolveStatus::Timeout, _) => {
                        bugs.push((name.clone(), Verdict::Timeout, elapsed));
                        eprintln!("[TIMEOUT] {}: time={:.2}s", name, elapsed);
                    }
                    (status, exp) => {
                        let s_dbg = format!("{:?}", status);
                        bugs.push((
                            name.clone(),
                            Verdict::BadStatus {
                                status,
                                expected_optimal: exp.unwrap_or(0.0),
                            },
                            elapsed,
                        ));
                        eprintln!(
                            "[BAD_STATUS] {}: status={} obj={:.6e} time={:.2}s",
                            name, s_dbg, r.objective, elapsed
                        );
                    }
                }
            }
        }
    }

    eprintln!("\n=== LP COVERAGE SCREEN SUMMARY ===");
    eprintln!("Total problems: {}", entries.len());
    eprintln!("PASS: {}", pass);
    eprintln!("BUGS: {}", bugs.len());
    eprintln!("Total wall time: {:.2}s", total_time);
    eprintln!("\n=== BUG LIST ===");
    for (name, verdict, time) in &bugs {
        eprintln!("  {} [{:.2}s]: {:?}", name, time, verdict);
    }

    // テスト本体は失敗させない (screening 用)。CI で fail させたければ assert!(bugs.is_empty()).
}
