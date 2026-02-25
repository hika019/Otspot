//! Maros-Meszaros QP Runner CLI
//!
//! Reads QP problem data from stdin (text format) and runs solve_qp.
//! Used by scripts/run_maros_all.py for batch testing.
//!
//! ## Input format (stdin):
//! Line 1: n m_ub            <- num variables, num inequality constraints
//! Line 2: c[0] c[1] ... c[n-1]
//! Line 3: lb[0] lb[1] ... lb[n-1]   (use -1e300 for -inf)
//! Line 4: ub[0] ub[1] ... ub[n-1]   (use 1e300 for +inf)
//! Line 5: nnz_Q
//! Lines 6..6+nnz_Q: row col val      (upper triangular, 0-indexed)
//! Line 6+nnz_Q: nnz_A
//! Lines ...: row col val             (0-indexed)
//! Last line: b[0] b[1] ... b[m_ub-1]
//!
//! ## Output format (stdout):
//! STATUS objective iterations
//! (STATUS = Optimal | Infeasible | Unbounded | MaxIterations | Error)

use std::io::{self, BufRead};
use solver::{solve_qp, QpProblem, SolveStatus};
use solver::sparse::CscMatrix;

const INF_THRESHOLD: f64 = 1e200;

fn parse_floats(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .map(|t| {
            let v: f64 = t.parse().unwrap_or(0.0);
            if v > INF_THRESHOLD { f64::INFINITY }
            else if v < -INF_THRESHOLD { f64::NEG_INFINITY }
            else { v }
        })
        .collect()
}

fn parse_usize(s: &str) -> usize {
    s.trim().parse().unwrap_or(0)
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().filter_map(|l| l.ok());

    macro_rules! next_line {
        () => {
            match lines.next() {
                Some(l) => l,
                None => {
                    println!("Error 0.0 0");
                    return;
                }
            }
        }
    }

    // Line 1: n m_ub
    let header = next_line!();
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 2 {
        println!("Error 0.0 0");
        return;
    }
    let n: usize = parts[0].parse().unwrap_or(0);
    let m_ub: usize = parts[1].parse().unwrap_or(0);

    // Line 2: c
    let c_line = next_line!();
    let c = parse_floats(&c_line);
    if c.len() != n { println!("Error 0.0 0"); return; }

    // Line 3: lb
    let lb_line = next_line!();
    let lb = parse_floats(&lb_line);
    if lb.len() != n { println!("Error 0.0 0"); return; }

    // Line 4: ub
    let ub_line = next_line!();
    let ub = parse_floats(&ub_line);
    if ub.len() != n { println!("Error 0.0 0"); return; }

    let bounds: Vec<(f64, f64)> = lb.iter().zip(ub.iter()).map(|(&l, &u)| (l, u)).collect();

    // Line 5: nnz_Q
    let nnz_q_line = next_line!();
    let nnz_q = parse_usize(&nnz_q_line);

    // Q entries (upper triangular stored; expand to full symmetric)
    let mut q_rows: Vec<usize> = Vec::new();
    let mut q_cols: Vec<usize> = Vec::new();
    let mut q_vals: Vec<f64> = Vec::new();
    for _ in 0..nnz_q {
        let entry = next_line!();
        let parts: Vec<&str> = entry.split_whitespace().collect();
        if parts.len() < 3 { println!("Error 0.0 0"); return; }
        let r: usize = parts[0].parse().unwrap_or(0);
        let c_idx: usize = parts[1].parse().unwrap_or(0);
        let v: f64 = parts[2].parse().unwrap_or(0.0);
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
            println!("Error 0.0 0");
            return;
        }
    };

    // A: constraint matrix
    let nnz_a_line = next_line!();
    let nnz_a = parse_usize(&nnz_a_line);

    let mut a_rows: Vec<usize> = Vec::new();
    let mut a_cols: Vec<usize> = Vec::new();
    let mut a_vals: Vec<f64> = Vec::new();
    for _ in 0..nnz_a {
        let entry = next_line!();
        let parts: Vec<&str> = entry.split_whitespace().collect();
        if parts.len() < 3 { println!("Error 0.0 0"); return; }
        let r: usize = parts[0].parse().unwrap_or(0);
        let c_idx: usize = parts[1].parse().unwrap_or(0);
        let v: f64 = parts[2].parse().unwrap_or(0.0);
        a_rows.push(r);
        a_cols.push(c_idx);
        a_vals.push(v);
    }

    let a = if m_ub > 0 {
        match CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_ub, n) {
            Ok(m) => m,
            Err(_) => { println!("Error 0.0 0"); return; }
        }
    } else {
        CscMatrix::new(0, n)
    };

    // b vector
    let b = if m_ub > 0 {
        let b_line = next_line!();
        let b_vals = parse_floats(&b_line);
        if b_vals.len() != m_ub { println!("Error 0.0 0"); return; }
        b_vals
    } else {
        vec![]
    };

    // Build and solve
    let problem = match QpProblem::new(q, c, a, b, bounds) {
        Ok(p) => p,
        Err(_) => { println!("Error 0.0 0"); return; }
    };

    let result = solve_qp(&problem);

    let status_str = match result.status {
        SolveStatus::Optimal => "Optimal",
        SolveStatus::Infeasible => "Infeasible",
        SolveStatus::Unbounded => "Unbounded",
        SolveStatus::MaxIterations => "MaxIterations",
        SolveStatus::Timeout => "Timeout",
        SolveStatus::NumericalError => "NumericalError",
    };

    println!("{} {:.10e} {}", status_str, result.objective, result.iterations);
}
// temp check - remove later
