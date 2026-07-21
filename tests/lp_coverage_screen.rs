//! Screening tests: run solve() on Netlib LP problems and report failures.
//!
//! Purpose: LP simplex regression detection across diverse LP features
//! (RANGES / FR / OBJSENSE MAX / negative bounds, EQ/LE/GE constraints, degenerate bases).
//!
//! # ベースラインとobj_offsetの扱い
//! ベースライン CSV の値は Netlib 公式値 (MINOS 5.3)。c^T x のみ (N-row RHS 除外)。
//! ソルバーは N-row RHS を problem.obj_offset として加算して報告するため、比較は
//! exp_adjusted = netlib_ref + problem.obj_offset で補正する。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::SolveStatus;
use otspot::qp::solve_qp_with;
use otspot_dev::bench_utils::load_baseline_objectives;
use otspot_dev::screening::{is_bug, screen_single};
use std::path::Path;

const DEFAULT_TIMEOUT_SEC: f64 = 20.0;
const DEFAULT_REL_TOL: f64 = 1e-3;

const PROBLEMS_DIR: &str = "data/lp_problems";
const FIXTURE_DIR: &str = "tests/lp_problems";
const BASELINE_CSV: &str = "data/baseline_objectives/netlib_lp.csv";

/// 90 LP問題全体のスクリーニング。実行時間が長い (約5-6分) ため `#[ignore]` で隔離。
/// 手動実行: `cargo nextest run --release --test lp_coverage_screen -- --ignored`
#[test]
#[ignore = "heavy (~5-6 min: 90 LP screen、要 data/lp_problems/)、cargo nextest で個別実行"]
fn lp_coverage_screen_all() {
    let dir = Path::new(PROBLEMS_DIR);
    assert!(
        dir.exists(),
        "{} not found — bench data 未配置。scripts/netlib_lp_download.sh を実行",
        PROBLEMS_DIR
    );
    let baseline = load_baseline_objectives(Path::new(BASELINE_CSV))
        .expect("baseline CSV missing — run scripts/netlib_lp_download.sh");

    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("read_dir failed")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|s| s == "QPS" || s == "qps")
                .unwrap_or(false)
        })
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
            bug_entries.push((
                screen.name.clone(),
                format!("{:?}", screen.verdict),
                screen.elapsed_secs,
            ));
        } else {
            pass += 1;
        }
    }

    eprintln!("\n=== LP COVERAGE SCREEN SUMMARY ===");
    eprintln!(
        "Total: {}  PASS: {}  BUGS: {}  wall: {:.2}s",
        entries.len(),
        pass,
        bugs,
        total_time
    );
    if !bug_entries.is_empty() {
        eprintln!("=== BUG LIST ===");
        for (name, verdict, secs) in &bug_entries {
            eprintln!("  {} [{:.2}s]: {}", name, secs, verdict);
        }
    }
    // Screening mode: do not assert — caller inspects output.
}

// ── Replacement tests (supersede timing-sensitive lp_coverage_screen_sample) ─

/// Fast screening: same diversity coverage as lp_coverage_screen_sample but
/// excludes `cycle`, which is covered separately (and runs by default) in
/// `lp_coverage_screen_cycle`.
#[test]
fn lp_coverage_screen_sample_fast() {
    let baseline = load_baseline_objectives(Path::new(BASELINE_CSV)).expect("baseline CSV missing");

    let afiro_path = Path::new(FIXTURE_DIR).join("afiro.QPS");
    assert!(afiro_path.exists(), "tests/lp_problems/afiro.QPS missing");

    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(DEFAULT_TIMEOUT_SEC);

    let afiro_screen = screen_single(&afiro_path, "afiro", &opts, &baseline, DEFAULT_REL_TOL);
    assert!(
        !is_bug(&afiro_screen.verdict),
        "afiro (committed fixture) failed: {:?}",
        afiro_screen.verdict
    );

    let data_dir = Path::new(PROBLEMS_DIR);
    if !data_dir.exists() {
        eprintln!("SKIP: {} absent", PROBLEMS_DIR);
        return;
    }

    // cycle excluded: covered separately by lp_coverage_screen_cycle.
    const SAMPLE_FAST: &[&str] = &[
        "adlittle", "agg", "bandm", "blend", "boeing2", "brandy", "capri", "etamacro",
    ];

    let mut bugs: Vec<(String, String, f64)> = Vec::new();
    let mut found = 0;
    for &name in SAMPLE_FAST {
        let path = data_dir.join(format!("{}.QPS", name));
        if !path.exists() {
            continue;
        }
        found += 1;
        let mut o = SolverOptions::default();
        o.timeout_secs = Some(DEFAULT_TIMEOUT_SEC);
        let screen = screen_single(&path, name, &o, &baseline, DEFAULT_REL_TOL);
        if is_bug(&screen.verdict) {
            bugs.push((
                screen.name,
                format!("{:?}", screen.verdict),
                screen.elapsed_secs,
            ));
        }
    }
    if found == 0 {
        eprintln!("SKIP fast sample: no QPS files found");
    }
    assert!(
        bugs.is_empty(),
        "lp_coverage_screen_sample_fast: {} failure(s): {:?}",
        bugs.len(),
        bugs
    );
}

/// `cycle` LP convergence sentinel (`Optimal` pinned). Previously gated
/// behind two claims that no longer hold: an assumed ~19-20s runtime, and an
/// "Optimal 未証明" concern once noted as open issue 31 — no such GitHub issue
/// was ever created (`gh issue view 31` 404s; this repo has zero GitHub issues).
/// Measured PASS across 5 independent runs (2026-07-09, local + heavy CI):
/// 4.25-9.59s, rel_err ~1e-12 — well under the 30s internal timeout below
/// and the default profile's per-test budget. Renamed from
/// `lp_coverage_screen_cycle_tier2` (dropped the `_tier2` suffix) now that it
/// runs by default rather than as an ignored tier-2 test.
#[test]
fn lp_coverage_screen_cycle() {
    let baseline = load_baseline_objectives(Path::new(BASELINE_CSV)).expect("baseline CSV missing");
    let data_dir = Path::new(PROBLEMS_DIR);
    if !data_dir.exists() {
        eprintln!("SKIP: {} absent", PROBLEMS_DIR);
        return;
    }
    let path = data_dir.join("cycle.QPS");
    if !path.exists() {
        eprintln!("SKIP: cycle.QPS absent");
        return;
    }
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(30.0);
    let problem = parse_qps(&path).expect("parse cycle.QPS");
    let t0 = std::time::Instant::now();
    let result = solve_qp_with(&problem, &opts);
    let elapsed = t0.elapsed().as_secs_f64();
    let expected = baseline
        .get("cycle")
        .copied()
        .expect("cycle baseline missing");
    let rel_err = (result.objective - expected).abs() / expected.abs().max(1.0);
    eprintln!(
        "cycle: status={:?} obj={:.10e} expected={:.10e} rel_err={:.2e} {:.2}s",
        result.status, result.objective, expected, rel_err, elapsed
    );
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "cycle expected Optimal, got {:?}",
        result.status
    );
    assert!(
        rel_err < DEFAULT_REL_TOL,
        "cycle obj rel_err {:.2e} >= {:.0e}",
        rel_err,
        DEFAULT_REL_TOL
    );
}
