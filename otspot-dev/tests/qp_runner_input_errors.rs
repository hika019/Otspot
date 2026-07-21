use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qp_runner"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn malformed_input_exits_nonzero_on_stderr() {
    let output = run(&[], "bad header\n");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("qp_runner:"));
}

#[test]
fn invalid_cli_options_exit_nonzero() {
    for args in [&["--unknown"][..], &["--eps"][..], &["--eps", "NaN"][..]] {
        let output = run(args, "");
        assert!(!output.status.success(), "args={args:?}");
        assert!(output.stdout.is_empty(), "args={args:?}");
        assert!(!output.stderr.is_empty(), "args={args:?}");
    }
}

#[test]
fn valid_minimal_qp_exits_zero_with_exactly_one_stdout_record() {
    let output = run(&[], "1 0\n0\n0\n1\n1\n0 0 2\n0\n");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.ends_with('\n'));
    assert_eq!(stdout.lines().count(), 1, "stdout={stdout:?}");
    let fields: Vec<_> = stdout.split_whitespace().collect();
    assert_eq!(fields.len(), 3, "stdout={stdout:?}");
    assert_eq!(fields[0], "Optimal", "stdout={stdout:?}");
    assert!(fields[1].parse::<f64>().unwrap().abs() < 1e-6);
    assert!(fields[2].parse::<usize>().is_ok());
}

#[test]
fn nonfinite_matrix_coefficients_are_input_errors() {
    for input in [
        "1 0\n0\n0\n1\n1\n0 0 NaN\n0\n",
        "1 1\n0\n0\n1\n0\n1\n0 0 inf\n0\n",
    ] {
        let output = run(&[], input);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("qp_runner:"));
    }
}
