//! Integration: parse real CBLIB (`.cbf`) files and validate the resulting
//! `ConicProblem`/`MisocpProblem`.
//!
//! Data is gitignored and downloaded out-of-band; the test skips gracefully
//! when the directory is absent or empty. Files using cone types outside this
//! bridge's scope (e.g. the exponential cone `EXP`) are expected to fail with
//! `CbfError::Unsupported` — that is not a parser bug, so it is counted rather
//! than treated as a failure. Any other error (`ParseError`/`IoError`) is a
//! real regression and fails the test loudly.

use otspot_io::cbf::{parse_cbf, CbfError, CbfProblem};
use std::path::Path;

/// Resolve the `data/` directory relative to this crate (nextest and `cargo
/// test` both run with CWD = crate root, so a bare `data/...` silently misses
/// the workspace-root symlink/directory).
fn cblib_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/cblib_socp")
}

#[test]
fn cblib_socp_files_parse_and_validate() {
    let dir = cblib_dir();
    if !dir.exists() {
        eprintln!("[cbf-cblib] skip: data missing: {}", dir.display());
        return;
    }
    let mut count_total = 0usize;
    let mut count_socp = 0usize;
    let mut count_misocp = 0usize;
    let mut count_unsupported = 0usize;
    for entry in std::fs::read_dir(&dir).expect("read_dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cbf") {
            continue;
        }
        count_total += 1;
        match parse_cbf(&path) {
            Ok(CbfProblem::Socp { problem, .. }) => {
                problem
                    .validate()
                    .unwrap_or_else(|e| panic!("validate failed for {}: {e}", path.display()));
                count_socp += 1;
            }
            Ok(CbfProblem::Misocp { problem, .. }) => {
                problem
                    .base
                    .validate()
                    .unwrap_or_else(|e| panic!("validate failed for {}: {e}", path.display()));
                count_misocp += 1;
            }
            Err(CbfError::Unsupported(msg)) => {
                eprintln!(
                    "[cbf-cblib] {} uses an unsupported feature: {msg}",
                    path.display()
                );
                count_unsupported += 1;
            }
            Err(e) => panic!("parse failed for {}: {e}", path.display()),
        }
    }
    assert!(count_total > 0, "no .cbf files found in {}", dir.display());
    assert!(
        count_socp + count_misocp > 0,
        "no .cbf file in {} parsed as a supported SOCP/MISOCP (all {count_total} were Unsupported)",
        dir.display()
    );
    eprintln!(
        "[cbf-cblib] {count_total} files: {count_socp} SOCP, {count_misocp} MISOCP, \
         {count_unsupported} unsupported (e.g. EXP cone)"
    );
}
