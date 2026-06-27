use std::fs;
use std::process::Command;

#[test]
fn milp_solve_reports_generic_mip_stats_as_key_value_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny_binary.mps");
    fs::write(
        &path,
        r"NAME tiny_binary
ROWS
 N  obj
 L  cap
COLUMNS
    x1  obj  -1.0  cap  1.0
RHS
    rhs  cap  1.0
BOUNDS
 BV BND  x1
ENDATA
",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_milp_solve"))
        .arg(&path)
        .arg("--timeout")
        .arg("2")
        .arg("--eps")
        .arg("1e-6")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("status: Optimal"), "stdout:\n{stdout}");
    for key in [
        "rens_calls",
        "rens_improvements",
        "rins_calls",
        "rins_improvements",
        "local_branching_calls",
        "local_branching_improvements",
        "tree_cut_rounds",
        "conflict_clauses_learned",
        "conflict_pruned",
        "propagation_pruned",
        "rc_vars_fixed",
    ] {
        let expected = format!("{key}:");
        let line = stdout
            .lines()
            .find(|line| line.starts_with(&expected))
            .unwrap_or_else(|| panic!("missing {key} in stdout:\n{stdout}"));
        let _value: i64 = line[expected.len()..]
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("could not parse integer for {key}: {line}"));
    }
}
