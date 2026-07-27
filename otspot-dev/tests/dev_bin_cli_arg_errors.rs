//! NEW (Phase 1d/P3-D): a CLI flag given as the very last argument, with no
//! value following it, must exit cleanly (status code 2) rather than panic
//! (which would show up as a process abort / non-zero-but-uncontrolled exit
//! with a "panicked at" message on stderr).
//!
//! Sentinel: reverting either binary's flag parsing from `args.get(i)` back
//! to direct `args[i]` indexing makes the process panic instead of printing
//! a clean `error: ... requires a value` message, failing the `!contains
//! "panicked"` assertion below (the exit code alone would not distinguish a
//! panic from a controlled `ExitCode::from(2)` / `std::process::exit(2)`,
//! since both are non-zero).

use std::process::Command;

#[test]
fn lp_solve_diag_missing_flag_value_exits_cleanly() {
    let output = Command::new(env!("CARGO_BIN_EXE_lp_solve_diag"))
        .args(["dummy.mps", "--timeout"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "missing --timeout value must be reported as an error, not succeed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "missing --timeout value must not panic; stderr:\n{stderr}"
    );
}

#[test]
fn mip_speed_bench_missing_flag_value_exits_cleanly() {
    let output = Command::new(env!("CARGO_BIN_EXE_mip_speed_bench"))
        .args(["--timeout"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "missing --timeout value must be reported as an error, not succeed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "missing --timeout value must not panic; stderr:\n{stderr}"
    );
}

#[test]
fn mip_speed_bench_missing_out_value_exits_cleanly() {
    let output = Command::new(env!("CARGO_BIN_EXE_mip_speed_bench"))
        .args(["--out"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "missing --out value must be reported as an error, not succeed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "missing --out value must not panic; stderr:\n{stderr}"
    );
}
