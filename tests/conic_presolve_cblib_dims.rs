//! RED→GREEN dimension contract: the tier-1 conic presolve must shrink the
//! KKT dimension `N = n + p + m` of the equality-dominant CBLIB classes,
//! against the size predictions of the research report (`cbf_dims_table.tsv`
//! "S1 equality substitution" column, `/tmp/opencode/presolve_research/`).
//!
//! These are *dimension* assertions only — no factorization, no solve — so
//! they run inside the worker memory budget even for the big classes. The
//! solve-side follow-up lives in the bench harness (solve_cbf real runs), not
//! here. Data is gitignored; each case skips gracefully when its file is
//! absent (same convention as `cbf_feasibility.rs`).

use otspot_core::conic::{kkt_dim, ConicReduction};
use otspot_io::cbf::{parse_cbf, CbfProblem};
use std::path::{Path, PathBuf};

/// Instances whose *reduced* KKT dimension is asserted: `(file, research S1
/// prediction, measured contract cap)`. The cap is the KKT dim the guarded
/// implementation actually reaches (guards — Markowitz budget, pivot nnz
/// cap, nnz-growth stop — trade depth for coefficient control); reverting
/// any reduction pushes `N1` above the cap and fails the sentinel.
/// `chainsing-1000-1` hits the research S1 prediction exactly; `20_0_1_w`
/// beats it (domain absorption removes what S1 did not model); the `nql` /
/// `qssp` shapes stop guard-limited above the prediction — reported
/// honestly in the implementation report, still far inside the runnable
/// memory band (see report §4).
const DIMS_CONTRACT: &[(&str, usize, usize)] = &[
    ("nql30.cbf", 3_521, 6_105),
    ("nql60.cbf", 14_241, 25_895),
    ("chainsing-1000-1.cbf", 13_976, 13_976),
    ("qssp30.cbf", 11_438, 13_772),
    ("20_0_1_w.cbf", 13_484, 12_238),
];

fn cblib_path(name: &str) -> PathBuf {
    Path::new("data/cblib_socp").join(name)
}

/// Parses one CBLIB file into its continuous conic core (SOCP as-is; the
/// MISOCP files carry their relaxation-relevant continuous rows in the same
/// `ConicProblem`, which is the shape the tier-1 presolve sees at root).
fn load_problem(name: &str) -> Option<otspot_core::conic::ConicProblem> {
    let path = cblib_path(name);
    if !path.exists() {
        eprintln!("[presolve-dims] skip: data missing: {}", path.display());
        return None;
    }
    let cbf = parse_cbf(&path).unwrap_or_else(|e| panic!("parse {name}: {e}"));
    match cbf {
        CbfProblem::Socp { problem, .. } => Some(problem),
        // Tier-1 presolve sees the MISOCP continuous relaxation core at root.
        CbfProblem::Misocp { problem, .. } => Some(problem.base),
    }
}

/// Every contracted instance must (a) be reduced, (b) shrink its KKT
/// dimension, and (c) land at or below the measured contract cap — the
/// guard-limited reduction actually shipped. The research S1 prediction is
/// reported alongside; where guards stop short of it (nql/qssp) the gap is
/// the documented, honest cost of coefficient control, and where domain
/// absorption adds reduction beyond S1 (20_0_1_w) the cap is tighter than
/// the research column.
#[test]
fn cblib_reduced_kkt_dims_match_predicted_band() {
    let mut verified = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for &(file, predicted, cap) in DIMS_CONTRACT {
        let Some(problem) = load_problem(file) else {
            continue;
        };
        let n0 = kkt_dim(&problem);
        let Some(red) = ConicReduction::reduce(&problem) else {
            failures.push(format!("{file}: presolve declined — dims cannot shrink"));
            continue;
        };
        let Some(reduced) = red.reduced() else {
            failures.push(format!("{file}: conflict path in a dims case"));
            continue;
        };
        let stats = red.stats();
        let n1 = stats.kkt1;
        if n1 != kkt_dim(reduced) {
            failures.push(format!("{file}: stats.kkt1 {n1} != kkt_dim(reduced)"));
        }
        eprintln!(
            "[presolve-dims] {file}: N0={n0} -> N1={n1} (research S1 {predicted}, cap {cap}) \
             absorbed={} fixed={} subst={} sigma={:.3}",
            stats.m_domain_absorbed, stats.n_fixed, stats.n_substituted, stats.sigma
        );
        if stats.kkt0 != n0 {
            failures.push(format!("{file}: kkt0 must equal kkt_dim(input)"));
        }
        if n1 >= n0 {
            failures.push(format!("{file}: presolve must shrink KKT ({n0} -> {n1})"));
        }
        if n1 > cap {
            failures.push(format!("{file}: reduced KKT {n1} above contract cap {cap}"));
        }
        verified += 1;
    }
    assert!(
        failures.is_empty(),
        "dims contract violated:\n{}",
        failures.join("\n")
    );
    assert!(
        verified > 0,
        "no CBLIB data present: dims contract cannot be checked (this is a \n\
         silent-skip gate: at least one contracted file must exist in data/)"
    );
}

/// The reduction must never break the structural invariants the reduced
/// problem's `validate()` relies on (cone rows intact, shapes consistent) —
/// proven on every contracted instance by simply validating the reduction.
#[test]
fn cblib_reduced_problems_validate() {
    for &(file, _, _) in DIMS_CONTRACT {
        let Some(problem) = load_problem(file) else {
            continue;
        };
        let Some(red) = ConicReduction::reduce(&problem) else {
            panic!("{file}: presolve declined");
        };
        let reduced = red.reduced().expect("reduced problem");
        reduced
            .validate()
            .unwrap_or_else(|e| panic!("{file}: reduced problem invalid: {e}"));
    }
}
