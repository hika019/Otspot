//! Clarabel で LISWET9 等を解いて、本ソルバの出力と比較する。
//!
//! 目的: 本ソルバの reference obj (data/baseline_objectives/) は self-reference のため、
//! 真の最適 obj が外部ソルバで何か確認する。
//!
//! 使い方: `cargo run --release --example clarabel_compare -- <path/to/file.QPS>`

use clarabel::algebra::CscMatrix as ClCsc;
use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use otspot::io::qps::parse_qps;
use otspot::problem::ConstraintType;
use otspot::QpProblem;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path/to/file.QPS>", args[0]);
        std::process::exit(2);
    }
    let path = std::path::PathBuf::from(&args[1]);
    let prob_box = parse_qps(&path).expect("parse failed");
    let prob: &QpProblem = &prob_box;

    let n = prob.num_vars;
    let m = prob.num_constraints;
    let n_lb = prob
        .bounds
        .iter()
        .filter(|&&(lb, _): &&(f64, f64)| lb.is_finite())
        .count();
    let n_ub = prob
        .bounds
        .iter()
        .filter(|&&(_, ub): &&(f64, f64)| ub.is_finite())
        .count();

    println!("Problem: {}", path.display());
    println!("  n={}, m={}, n_lb={}, n_ub={}", n, m, n_lb, n_ub);
    println!(
        "  Q nnz={}, max|Q|={:.3e}",
        prob.q.values().len(),
        prob.q
            .values()
            .iter()
            .fold(0.0_f64, |a, &v: &f64| a.max(v.abs()))
    );

    // Clarabel 形式に変換: min 0.5 x^T P x + q^T x  s.t. A x + s = b, s ∈ K
    //   Le (Ax ≤ b):    A_clar = A,  b_clar = b,  s ≥ 0 (Nonnegative)
    //   Ge (Ax ≥ b):    A_clar = -A, b_clar = -b, s ≥ 0 (Nonnegative)
    //   Eq (Ax = b):    A_clar = A,  b_clar = b,  s = 0 (Zero)
    //   bound lb ≤ x:   行追加 -x + s = -lb, s ≥ 0  → A_clar 1 行 (-1 at j), b -lb
    //   bound x ≤ ub:   行追加  x + s = ub,  s ≥ 0  → A_clar 1 行 (+1 at j), b ub

    // 三角配置: まず Eq, 次に Le/Ge (Nonnegative cone)
    let mut row_ord: Vec<(usize, ConstraintType)> =
        (0..m).map(|i| (i, prob.constraint_types[i])).collect();
    row_ord.sort_by_key(|&(_, ct)| match ct {
        ConstraintType::Eq => 0,
        _ => 1,
    });
    let n_eq = row_ord
        .iter()
        .filter(|&&(_, ct)| ct == ConstraintType::Eq)
        .count();
    let n_le_ge = m - n_eq;

    // build A_clar (CSC by triplets)
    // collect triplets
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut b_clar: Vec<f64> = vec![0.0; m + n_lb + n_ub];

    let row_pos: Vec<usize> = {
        let mut rp = vec![0_usize; m];
        for (new_row, &(orig_row, _)) in row_ord.iter().enumerate() {
            rp[orig_row] = new_row;
        }
        rp
    };

    for j in 0..n {
        for ptr in prob.a.col_ptr()[j]..prob.a.col_ptr()[j + 1] {
            let orig_row = prob.a.row_ind()[ptr];
            let val = prob.a.values()[ptr];
            let new_row = row_pos[orig_row];
            let ct = prob.constraint_types[orig_row];
            match ct {
                ConstraintType::Ge => {
                    triplets.push((new_row, j, -val));
                }
                _ => {
                    triplets.push((new_row, j, val));
                }
            }
        }
    }
    for (orig_row, ct) in prob.constraint_types.iter().enumerate() {
        let new_row = row_pos[orig_row];
        match ct {
            ConstraintType::Ge => {
                b_clar[new_row] = -prob.b[orig_row];
            }
            _ => {
                b_clar[new_row] = prob.b[orig_row];
            }
        }
    }

    // bound rows: lb (negated), then ub
    let mut bound_row = m;
    for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() {
            triplets.push((bound_row, j, -1.0));
            b_clar[bound_row] = -lb;
            bound_row += 1;
        }
    }
    for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
        if ub.is_finite() {
            triplets.push((bound_row, j, 1.0));
            b_clar[bound_row] = ub;
            bound_row += 1;
        }
    }

    // sort triplets by (col, row) to build CSC
    triplets.sort_by_key(|&(r, c, _)| (c, r));
    let total_rows = m + n_lb + n_ub;
    let mut col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &triplets {
        col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    let mut row_ind = vec![0_usize; triplets.len()];
    let mut values = vec![0.0_f64; triplets.len()];
    let mut col_cursor = col_ptr.clone();
    for &(r, c, v) in &triplets {
        let pos = col_cursor[c];
        row_ind[pos] = r;
        values[pos] = v;
        col_cursor[c] += 1;
    }
    let a_clar = ClCsc::new(total_rows, n, col_ptr, row_ind, values);

    // P (Hessian) は upper triangular にして渡す
    // 本ソルバの Q は CSC 全体形式。Clarabel は upper-triangular CSC を期待する。
    let mut p_triplets: Vec<(usize, usize, f64)> = Vec::new();
    for j in 0..n {
        for ptr in prob.q.col_ptr()[j]..prob.q.col_ptr()[j + 1] {
            let i = prob.q.row_ind()[ptr];
            let v = prob.q.values()[ptr];
            if i <= j {
                p_triplets.push((i, j, v));
            }
        }
    }
    p_triplets.sort_by_key(|&(r, c, _)| (c, r));
    let mut p_col_ptr = vec![0_usize; n + 1];
    for &(_, c, _) in &p_triplets {
        p_col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        p_col_ptr[j + 1] += p_col_ptr[j];
    }
    let mut p_row_ind = vec![0_usize; p_triplets.len()];
    let mut p_values = vec![0.0_f64; p_triplets.len()];
    let mut p_cursor = p_col_ptr.clone();
    for &(r, c, v) in &p_triplets {
        let pos = p_cursor[c];
        p_row_ind[pos] = r;
        p_values[pos] = v;
        p_cursor[c] += 1;
    }
    let p_clar = ClCsc::new(n, n, p_col_ptr, p_row_ind, p_values);

    // cones
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();
    if n_eq > 0 {
        cones.push(SupportedConeT::ZeroConeT(n_eq));
    }
    if n_le_ge + n_lb + n_ub > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(n_le_ge + n_lb + n_ub));
    }

    let settings = DefaultSettings {
        verbose: false,
        tol_gap_abs: 1e-9,
        tol_gap_rel: 1e-9,
        tol_feas: 1e-9,
        max_iter: 2000,
        ..Default::default()
    };

    println!("\nSolving with Clarabel (eps=1e-9)...");
    let mut solver = DefaultSolver::new(&p_clar, &prob.c, &a_clar, &b_clar, &cones, settings)
        .expect("Clarabel new");
    solver.solve();

    let info = &solver.info;
    let solution = &solver.solution;
    println!("\nClarabel result:");
    println!("  status: {:?}", info.status);
    println!("  iters: {}", info.iterations);
    println!("  obj_val: {:.10e}", info.cost_primal);
    println!("  solve_time: {:.3}s", info.solve_time);

    if matches!(
        info.status,
        SolverStatus::Solved | SolverStatus::AlmostSolved
    ) {
        // primal feas
        let x = &solution.x;
        let ax = prob.a.mat_vec_mul(x).expect("Ax");
        let mut max_pf = 0.0_f64;
        let mut max_ax = 0.0_f64;
        let mut max_b = 0.0_f64;
        for (i, (&ax_i, &b_i)) in ax.iter().zip(prob.b.iter()).enumerate() {
            let v = match prob.constraint_types[i] {
                ConstraintType::Eq => (ax_i - b_i).abs(),
                ConstraintType::Ge => (b_i - ax_i).max(0.0),
                _ => (ax_i - b_i).max(0.0),
            };
            max_pf = max_pf.max(v);
            max_ax = max_ax.max(ax_i.abs());
            max_b = max_b.max(b_i.abs());
        }
        println!(
            "  pf_abs={:.3e} pf_rel={:.3e} (max_ax={:.3e} max_b={:.3e})",
            max_pf,
            max_pf / (1.0 + max_ax.max(max_b)),
            max_ax,
            max_b
        );
        println!(
            "  |x|_inf={:.3e}",
            x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
        );
    }
}
