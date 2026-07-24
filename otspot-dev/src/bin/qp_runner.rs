//! Maros-Meszaros QP runner CLI (bench scripts 用)。stdin から text format で QP
//! 問題を読み取り、`solve_qp` を実行して `STATUS objective iterations` を出力する。
//!
//! Input: `n m_ub` / `c` / `lb` (`-1e300` = -inf) / `ub` / `nnz_Q` + Q triplets
//! (upper triangular, 0-indexed) / `nnz_A` + A triplets / `b`。
//! 正常に読み取ってsolveを実行した場合は、solver status（数値失敗等を含む）をstdoutへ
//! `STATUS objective iterations` の厳密1行で出力しexit 0。CLI・stdin I/O・入力形式の
//! errorはstdoutを空のままstderrへ理由を出しexit 2。

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use otspot_core::sparse::CscMatrix;
use otspot_core::{solve_qp_with, QpProblem, SolveStatus, SolverOptions};
use std::io::{self, BufRead};

const INF_THRESHOLD: f64 = 1e200;

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("qp_runner: {message}");
    std::process::exit(2)
}

fn parse_floats(s: &str) -> Result<Vec<f64>, String> {
    s.split_whitespace()
        .map(|t| {
            let v: f64 = t.parse().map_err(|_| format!("invalid float '{t}'"))?;
            Ok(if v > INF_THRESHOLD {
                f64::INFINITY
            } else if v < -INF_THRESHOLD {
                f64::NEG_INFINITY
            } else {
                v
            })
        })
        .collect()
}

fn parse_usize(s: &str) -> Result<usize, String> {
    s.trim()
        .parse()
        .map_err(|_| format!("invalid non-negative integer '{}'", s.trim()))
}

fn status_label(status: &SolveStatus) -> &'static str {
    match status {
        SolveStatus::Optimal => "Optimal",
        SolveStatus::LocallyOptimal => "LocallyOptimal",
        SolveStatus::Infeasible => "Infeasible",
        SolveStatus::Unbounded => "Unbounded",
        SolveStatus::MaxIterations => "MaxIterations",
        SolveStatus::SuboptimalSolution => "SuboptimalSolution",
        SolveStatus::Stalled => "Stalled",
        SolveStatus::FeasiblePoint => "FeasiblePoint",
        SolveStatus::Timeout => "Timeout",
        SolveStatus::NumericalError => "NumericalError",
        SolveStatus::NonConvex(_) => "NonConvex",
        SolveStatus::NonconvexLocal => "NonconvexLocal",
        SolveStatus::NonconvexGlobal => "NonconvexGlobal",
        SolveStatus::NotSupported(_) => "NotSupported",
        _ => panic!("qp_runner must explicitly label every SolveStatus variant"),
    }
}

fn main() {
    // Parse --eps VALUE from argv (default: 1e-6)
    let args: Vec<String> = std::env::args().collect();
    let mut eps: f64 = 1e-6;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--eps" {
            let val = args
                .get(i + 1)
                .unwrap_or_else(|| fail("--eps requires a value"));
            let value: f64 = val.parse().unwrap_or_else(|_| fail("invalid --eps value"));
            if !value.is_finite() || value <= 0.0 {
                fail("--eps must be finite and positive");
            }
            eps = value;
            i += 2;
        } else {
            fail(format!("unknown option '{}'", args[i]));
        }
    }

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    macro_rules! next_line {
        () => {
            match lines.next() {
                Some(Ok(l)) => l,
                Some(Err(e)) => fail(format!("stdin read failed: {e}")),
                None => fail("unexpected end of input"),
            }
        };
    }

    // Line 1: n m_ub
    let header = next_line!();
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 2 {
        fail("header must contain n and m_ub");
    }
    let (Ok(n), Ok(m_ub)) = (parse_usize(parts[0]), parse_usize(parts[1])) else {
        fail("invalid header dimensions");
    };

    // Line 2: c
    let c_line = next_line!();
    let Ok(c) = parse_floats(&c_line) else {
        fail("invalid objective vector");
    };
    if c.len() != n {
        fail("objective vector length mismatch");
    }

    // Line 3: lb
    let lb_line = next_line!();
    let Ok(lb) = parse_floats(&lb_line) else {
        fail("invalid lower-bound vector");
    };
    if lb.len() != n {
        fail("lower-bound vector length mismatch");
    }

    // Line 4: ub
    let ub_line = next_line!();
    let Ok(ub) = parse_floats(&ub_line) else {
        fail("invalid upper-bound vector");
    };
    if ub.len() != n {
        fail("upper-bound vector length mismatch");
    }

    let bounds: Vec<(f64, f64)> = lb.iter().zip(ub.iter()).map(|(&l, &u)| (l, u)).collect();

    // Line 5: nnz_Q
    let nnz_q_line = next_line!();
    let Ok(nnz_q) = parse_usize(&nnz_q_line) else {
        fail("invalid input");
    };

    // Q entries (upper triangular stored; expand to full symmetric)
    let mut q_rows: Vec<usize> = Vec::new();
    let mut q_cols: Vec<usize> = Vec::new();
    let mut q_vals: Vec<f64> = Vec::new();
    for _ in 0..nnz_q {
        let entry = next_line!();
        let parts: Vec<&str> = entry.split_whitespace().collect();
        if parts.len() < 3 {
            fail("invalid input");
        }
        let (Ok(r), Ok(c_idx), Ok(v)) = (
            parse_usize(parts[0]),
            parse_usize(parts[1]),
            parts[2].parse::<f64>(),
        ) else {
            fail("invalid input");
        };
        q_rows.push(r);
        q_cols.push(c_idx);
        q_vals.push(v);
        // Add symmetric entry if off-diagonal
        if r != c_idx {
            q_rows.push(c_idx);
            q_cols.push(r);
            q_vals.push(v);
        }
    }

    let q = match CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n) {
        Ok(m) => m,
        Err(_) => {
            fail("invalid input");
        }
    };

    // A: constraint matrix
    let nnz_a_line = next_line!();
    let Ok(nnz_a) = parse_usize(&nnz_a_line) else {
        fail("invalid input");
    };

    let mut a_rows: Vec<usize> = Vec::new();
    let mut a_cols: Vec<usize> = Vec::new();
    let mut a_vals: Vec<f64> = Vec::new();
    for _ in 0..nnz_a {
        let entry = next_line!();
        let parts: Vec<&str> = entry.split_whitespace().collect();
        if parts.len() < 3 {
            fail("invalid input");
        }
        let (Ok(r), Ok(c_idx), Ok(v)) = (
            parse_usize(parts[0]),
            parse_usize(parts[1]),
            parts[2].parse::<f64>(),
        ) else {
            fail("invalid input");
        };
        a_rows.push(r);
        a_cols.push(c_idx);
        a_vals.push(v);
    }

    let a = if m_ub > 0 {
        match CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_ub, n) {
            Ok(m) => m,
            Err(_) => {
                fail("invalid input");
            }
        }
    } else {
        CscMatrix::new(0, n)
    };

    // b vector
    let b = if m_ub > 0 {
        let b_line = next_line!();
        let Ok(b_vals) = parse_floats(&b_line) else {
            fail("invalid input");
        };
        if b_vals.len() != m_ub {
            fail("invalid input");
        }
        b_vals
    } else {
        vec![]
    };

    // Build and solve
    let problem = match QpProblem::new_all_le(q, c, a, b, bounds) {
        Ok(p) => p,
        Err(_) => {
            fail("invalid input");
        }
    };

    let mut options = SolverOptions::default();
    options.ipm.eps = eps;
    let result = solve_qp_with(&problem, &options);

    let status_str = status_label(&result.status);

    println!(
        "{} {:.10e} {}",
        status_str, result.objective, result.iterations
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_numbers_are_not_defaulted() {
        assert!(parse_floats("1 nope 3").is_err());
        assert!(parse_usize("nope").is_err());
        assert!(parse_usize("-1").is_err());
    }

    #[test]
    fn legal_numbers_and_infinity_markers_remain_compatible() {
        assert_eq!(parse_usize("2").unwrap(), 2);
        assert_eq!(
            parse_floats("1 -1e300 1e300").unwrap(),
            vec![1.0, f64::NEG_INFINITY, f64::INFINITY]
        );
    }

    #[test]
    fn every_solver_status_has_an_explicit_stable_label() {
        let cases = [
            (SolveStatus::Optimal, "Optimal"),
            (SolveStatus::LocallyOptimal, "LocallyOptimal"),
            (SolveStatus::Infeasible, "Infeasible"),
            (SolveStatus::Unbounded, "Unbounded"),
            (SolveStatus::MaxIterations, "MaxIterations"),
            (SolveStatus::SuboptimalSolution, "SuboptimalSolution"),
            (SolveStatus::Stalled, "Stalled"),
            (SolveStatus::FeasiblePoint, "FeasiblePoint"),
            (SolveStatus::Timeout, "Timeout"),
            (SolveStatus::NumericalError, "NumericalError"),
            (SolveStatus::NonConvex("detail".to_string()), "NonConvex"),
            (SolveStatus::NonconvexLocal, "NonconvexLocal"),
            (SolveStatus::NonconvexGlobal, "NonconvexGlobal"),
            (
                SolveStatus::NotSupported("detail".to_string()),
                "NotSupported",
            ),
        ];

        for (status, expected) in cases {
            assert_eq!(status_label(&status), expected);
        }
    }
}
