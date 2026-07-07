//! Converts a `.qplib` or `.cbf` problem into a plain whitespace-delimited
//! token stream that external solvers (HiGHS/SCIP via Python) can consume
//! without re-implementing the QPLIB/CBF parsers.
//!
//! Neither HiGHS (`highspy`) nor the SCIP build bundled in the `pyscipopt`
//! PyPI wheel (SCIP 10.0.2, no `reader_cbf.c`/`reader_qplib.c` linked in)
//! can read these formats natively, so this tool reuses otspot's own
//! (already-tested) parsers as the single source of truth for the
//! conversion, instead of duplicating QPLIB/CBF parsing logic in Python.
//!
//! Run: `cargo run --release --example dump_problem -- <in.qplib|in.cbf> <out.txt>`
//!
//! Output format (all tokens whitespace-separated, read positionally):
//!
//! QP dump (from `QplibProblem::{Qp,Milp,Miqp}`; MILP represented as QP with
//! an all-zero `Q`):
//! ```text
//! QP
//! n m
//! obj_offset
//! lb_0 ub_0 ... lb_{n-1} ub_{n-1}
//! c_0 ... c_{n-1}
//! q_nnz
//! row col val   (q_nnz lines, 0-indexed, full symmetric storage)
//! a_nnz
//! row col val   (a_nnz lines, 0-indexed)
//! b_0 ... b_{m-1}
//! ctype_0 ... ctype_{m-1}     (0=Le, 1=Ge, 2=Eq)
//! has_quadratic_constraints   (0 or 1)
//! [if 1: for k in 0..m: nnz_k \n row col val (nnz_k lines)]
//! num_integer_vars
//! idx_0 ... idx_{num_integer_vars-1}
//! ```
//!
//! CONIC dump (from `CbfProblem::{Socp,Misocp}`, standard form
//! `min c^T x s.t. Ax=b, Gx+s=h, s in K`):
//! ```text
//! CONIC
//! n
//! p m l
//! num_soc_blocks
//! soc_dim_0 ... soc_dim_{k-1}
//! maximize (0/1)
//! obj_offset
//! c_0 ... c_{n-1}
//! a_nnz
//! row col val   (a_nnz lines)
//! b_0 ... b_{p-1}
//! g_nnz
//! row col val   (g_nnz lines)
//! h_0 ... h_{m-1}
//! num_integers
//! idx lb ub   (num_integers lines)
//! ```

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use otspot::io::cbf::{parse_cbf, CbfProblem};
use otspot::io::qplib::{parse_qplib, QplibProblem};
use otspot::problem::ConstraintType;
use otspot::qp::QcqpMatrix;
use otspot::sparse::CscMatrix;
use otspot::ConicProblem;

fn write_csc_triplets(w: &mut impl Write, m: &CscMatrix) {
    writeln!(w, "{}", m.nnz()).unwrap();
    let col_ptr = m.col_ptr();
    let row_ind = m.row_ind();
    let values = m.values();
    for j in 0..m.ncols() {
        for k in col_ptr[j]..col_ptr[j + 1] {
            writeln!(w, "{} {} {:.17e}", row_ind[k], j, values[k]).unwrap();
        }
    }
}

fn ctype_code(ct: ConstraintType) -> i32 {
    match ct {
        ConstraintType::Le => 0,
        ConstraintType::Ge => 1,
        ConstraintType::Eq => 2,
        _ => unreachable!("ConstraintType has only Le/Ge/Eq variants"),
    }
}

#[allow(clippy::too_many_arguments)]
fn dump_qp(
    w: &mut impl Write,
    q: &CscMatrix,
    c: &[f64],
    a: &CscMatrix,
    b: &[f64],
    bounds: &[(f64, f64)],
    ctypes: &[ConstraintType],
    quadratic_constraints: &[QcqpMatrix],
    obj_offset: f64,
    integer_vars: &[usize],
) {
    writeln!(w, "QP").unwrap();
    writeln!(w, "{} {}", c.len(), b.len()).unwrap();
    writeln!(w, "{:.17e}", obj_offset).unwrap();
    for &(lb, ub) in bounds {
        writeln!(w, "{:.17e} {:.17e}", lb, ub).unwrap();
    }
    for &v in c {
        writeln!(w, "{:.17e}", v).unwrap();
    }
    write_csc_triplets(w, q);
    write_csc_triplets(w, a);
    for &v in b {
        writeln!(w, "{:.17e}", v).unwrap();
    }
    for &ct in ctypes {
        writeln!(w, "{}", ctype_code(ct)).unwrap();
    }
    if quadratic_constraints.is_empty() {
        writeln!(w, "0").unwrap();
    } else {
        writeln!(w, "1").unwrap();
        for qk in quadratic_constraints {
            writeln!(w, "{}", qk.nnz()).unwrap();
            for &(r, c2, v) in &qk.triplets {
                writeln!(w, "{} {} {:.17e}", r, c2, v).unwrap();
            }
        }
    }
    writeln!(w, "{}", integer_vars.len()).unwrap();
    for &idx in integer_vars {
        writeln!(w, "{}", idx).unwrap();
    }
}

fn dump_conic(
    w: &mut impl Write,
    problem: &ConicProblem,
    maximize: bool,
    obj_offset: f64,
    integers: &[usize],
    int_lb: &[f64],
    int_ub: &[f64],
) {
    writeln!(w, "CONIC").unwrap();
    writeln!(w, "{}", problem.n()).unwrap();
    writeln!(w, "{} {} {}", problem.p(), problem.m(), problem.cone.l).unwrap();
    writeln!(w, "{}", problem.cone.soc.len()).unwrap();
    for &d in &problem.cone.soc {
        write!(w, "{} ", d).unwrap();
    }
    writeln!(w).unwrap();
    writeln!(w, "{}", i32::from(maximize)).unwrap();
    writeln!(w, "{:.17e}", obj_offset).unwrap();
    for &v in &problem.c {
        writeln!(w, "{:.17e}", v).unwrap();
    }
    write_csc_triplets(w, &problem.a);
    for &v in &problem.b {
        writeln!(w, "{:.17e}", v).unwrap();
    }
    write_csc_triplets(w, &problem.g);
    for &v in &problem.h {
        writeln!(w, "{:.17e}", v).unwrap();
    }
    writeln!(w, "{}", integers.len()).unwrap();
    for i in 0..integers.len() {
        writeln!(w, "{} {:.17e} {:.17e}", integers[i], int_lb[i], int_ub[i]).unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: dump_problem <in.qplib|in.cbf> <out.txt>");
        std::process::exit(2);
    }
    let in_path = Path::new(&args[0]);
    let out_path = Path::new(&args[1]);
    let ext = in_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let file = File::create(out_path)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_path.display()));
    let mut w = BufWriter::new(file);

    match ext.as_str() {
        "qplib" => match parse_qplib(in_path) {
            Ok(QplibProblem::Qp(p)) => dump_qp(
                &mut w,
                &p.q,
                &p.c,
                &p.a,
                &p.b,
                &p.bounds,
                &p.constraint_types,
                &p.quadratic_constraints,
                p.obj_offset,
                &[],
            ),
            Ok(QplibProblem::Milp(m)) => {
                let n = m.lp.num_vars;
                let zero_q = CscMatrix::new(n, n);
                dump_qp(
                    &mut w,
                    &zero_q,
                    &m.lp.c,
                    &m.lp.a,
                    &m.lp.b,
                    &m.lp.bounds,
                    &m.lp.constraint_types,
                    &[],
                    m.lp.obj_offset,
                    &m.integer_vars,
                );
            }
            Ok(QplibProblem::Miqp(m)) => dump_qp(
                &mut w,
                &m.qp.q,
                &m.qp.c,
                &m.qp.a,
                &m.qp.b,
                &m.qp.bounds,
                &m.qp.constraint_types,
                &m.qp.quadratic_constraints,
                m.qp.obj_offset,
                &m.integer_vars,
            ),
            Err(e) => {
                eprintln!("qplib parse error for {}: {e}", in_path.display());
                std::process::exit(1);
            }
        },
        "cbf" => match parse_cbf(in_path) {
            Ok(cbf) => {
                let maximize = cbf.maximize();
                let obj_offset = cbf.obj_offset();
                match cbf {
                    CbfProblem::Socp { problem, .. } => {
                        dump_conic(&mut w, &problem, maximize, obj_offset, &[], &[], &[])
                    }
                    CbfProblem::Misocp { problem, .. } => dump_conic(
                        &mut w,
                        &problem.base,
                        maximize,
                        obj_offset,
                        &problem.integers,
                        &problem.int_lb,
                        &problem.int_ub,
                    ),
                }
            }
            Err(e) => {
                eprintln!("cbf parse error for {}: {e}", in_path.display());
                std::process::exit(1);
            }
        },
        other => {
            eprintln!("unsupported extension: {other} (expected .qplib or .cbf)");
            std::process::exit(2);
        }
    }

    w.flush().expect("flush output");
}
