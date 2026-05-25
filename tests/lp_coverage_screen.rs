//! Screening tests: run solve() on Netlib LP problems and report failures.
//!
//! Purpose: LP simplex regression detection across diverse LP features
//! (RANGES / FR / OBJSENSE MAX / negative bounds, EQ/LE/GE constraints, degenerate bases).
//!
//! # ベースラインとobj_offsetの扱い
//! ベースライン CSV の値は Netlib 公式値 (MINOS 5.3)。c^T x のみ (N-row RHS 除外)。
//! ソルバーは N-row RHS を problem.obj_offset として加算して報告するため、比較は
//! exp_adjusted = netlib_ref + problem.obj_offset で補正する。

use otspot::options::SolverOptions;
use otspot_dev::screening::{is_bug, load_baseline, screen_single, DEFAULT_REL_TOL, DEFAULT_TIMEOUT_SEC};
use std::path::Path;

const PROBLEMS_DIR: &str = "data/lp_problems";
const FIXTURE_DIR: &str = "tests/lp_problems";
const BASELINE_CSV: &str = "data/baseline_objectives/netlib_lp.csv";

/// 90 LP問題全体のスクリーニング。実行時間が長い (約5-6分) ため `#[ignore]` で隔離。
/// 手動実行: `cargo nextest run --release --test lp_coverage_screen -- --ignored`
#[test]
#[ignore = "heavy (~5-6 min: 90 LP screen、要 data/lp_problems/)、cargo nextest で個別実行"]
fn lp_coverage_screen_all() {
    let dir = Path::new(PROBLEMS_DIR);
    assert!(dir.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行", PROBLEMS_DIR);
    let baseline = load_baseline(BASELINE_CSV);

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|s| s == "QPS" || s == "qps").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.path());

    let mut bugs = 0usize;
    let mut pass = 0usize;
    let mut total_time = 0.0f64;
    let mut bug_entries: Vec<(String, String, f64)> = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(DEFAULT_TIMEOUT_SEC);

        let screen = screen_single(&path, &name, &opts, &baseline, DEFAULT_REL_TOL);
        total_time += screen.elapsed_secs;
        if is_bug(&screen.verdict) {
            bugs += 1;
            bug_entries.push((screen.name.clone(), format!("{:?}", screen.verdict), screen.elapsed_secs));
        } else {
            pass += 1;
        }
    }

    eprintln!("\n=== LP COVERAGE SCREEN SUMMARY ===");
    eprintln!("Total: {}  PASS: {}  BUGS: {}  wall: {:.2}s", entries.len(), pass, bugs, total_time);
    if !bug_entries.is_empty() {
        eprintln!("=== BUG LIST ===");
        for (name, verdict, secs) in &bug_entries {
            eprintln!("  {} [{:.2}s]: {}", name, secs, verdict);
        }
    }
    // Screening mode: do not assert — caller inspects output.
}

/// Sample screening: curated small/medium Netlib LPs for fast regression detection.
///
/// Always tests afiro from `tests/lp_problems/afiro.QPS` (committed fixture) so the
/// test never silently skips. Additional problems from `data/lp_problems/` are tested
/// if the directory exists (requires `scripts/netlib_lp_download.sh`).
///
/// Covers: bounds (bandm/capri), RANGES (cycle/agg), OBJSENSE (blend),
/// degenerate (etamacro), medium scale (boeing2/brandy).
/// Expected runtime: well under 3 min.
#[test]
fn lp_coverage_screen_sample() {
    let baseline = load_baseline(BASELINE_CSV);

    // afiro is a committed fixture — always present, always tested.
    let afiro_path = Path::new(FIXTURE_DIR).join("afiro.QPS");
    assert!(afiro_path.exists(),
        "tests/lp_problems/afiro.QPS missing — committed fixture must always be present");

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(DEFAULT_TIMEOUT_SEC);

    let afiro_screen = screen_single(&afiro_path, "afiro", &opts, &baseline, DEFAULT_REL_TOL);
    assert!(!is_bug(&afiro_screen.verdict),
        "afiro (committed fixture) failed: {:?}", afiro_screen.verdict);

    // Additional problems from data/lp_problems/ if available (optional).
    let data_dir = Path::new(PROBLEMS_DIR);
    if !data_dir.exists() {
        eprintln!("SKIP optional sample: {} absent (only afiro tested)", PROBLEMS_DIR);
        return;
    }

    // Selected for diversity: RANGES, OBJSENSE, bounds, degenerate, medium scale.
    const SAMPLE: &[&str] = &[
        "adlittle", "agg", "bandm", "blend",
        "boeing2", "brandy", "capri", "cycle", "etamacro",
    ];

    let mut bugs: Vec<(String, String, f64)> = Vec::new();
    let mut found = 0;

    for &name in SAMPLE {
        let path = data_dir.join(format!("{}.QPS", name));
        if !path.exists() {
            continue;
        }
        found += 1;

        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(DEFAULT_TIMEOUT_SEC);

        let screen = screen_single(&path, name, &opts, &baseline, DEFAULT_REL_TOL);
        if is_bug(&screen.verdict) {
            bugs.push((screen.name, format!("{:?}", screen.verdict), screen.elapsed_secs));
        }
    }

    if found == 0 {
        eprintln!("SKIP optional sample problems: no QPS files found in {}", PROBLEMS_DIR);
    }

    assert!(bugs.is_empty(),
        "lp_coverage_screen_sample: {} failure(s): {:?}", bugs.len(), bugs);
}
