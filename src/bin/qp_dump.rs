//! QPS/QPLIB → JSON dump utility（baseline校正用）
//!
//! 外部 solver (PIQP/Clarabel/OSQP) に食わせるため、問題行列を JSON で吐き出す。
//! 形式: {
//!   "name": str, "n": int, "m": int, "obj_offset": f64,
//!   "Q": {"col_ptr": [...], "row_ind": [...], "values": [...]},
//!   "c": [...], "A": {...}, "b": [...],
//!   "constraint_types": [0=Le, 1=Ge, 2=Eq],
//!   "bounds_lb": [...], "bounds_ub": [...]  (Inf/-Inf は大きな値で表現)
//! }

use otspot::io::{qplib, qps};
use otspot::problem::ConstraintType;
use otspot::qp::QpProblem;
use otspot::sparse::CscMatrix;
use std::env;
use std::fmt::Write;
use std::path::Path;
use std::process::ExitCode;

fn usage(prog: &str) -> ExitCode {
    eprintln!("Usage: {} <input.qps|input.qplib> <output.json>", prog);
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return usage(&args[0]);
    }
    let in_path = Path::new(&args[1]);
    let out_path = Path::new(&args[2]);

    let ext = in_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let problem: QpProblem = match ext.as_str() {
        "qps" | "mps" => match qps::parse_qps(in_path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("QPS parse error: {}", e);
                return ExitCode::FAILURE;
            }
        },
        "qplib" => match qplib::parse_qplib(in_path) {
            Ok(qplib::QplibProblem::Qp(p)) => p,
            Ok(qplib::QplibProblem::Miqp(m)) => {
                eprintln!("warning: MIQP — dumping QP relaxation (integrality constraints dropped)");
                m.qp
            }
            Ok(qplib::QplibProblem::Milp(_)) => {
                eprintln!("QPLIB: MILP (binary/integer linear) — cannot dump as QP JSON");
                return ExitCode::FAILURE;
            }
            Err(e) => {
                eprintln!("QPLIB parse error: {}", e);
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!("Unknown extension: {}. Expected .qps / .mps / .qplib", ext);
            return ExitCode::from(2);
        }
    };

    let name = in_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let mut out = String::new();
    out.push_str("{\n");
    writeln!(out, "  \"name\": \"{}\",", escape(name)).unwrap();
    writeln!(out, "  \"n\": {},", problem.num_vars).unwrap();
    writeln!(out, "  \"m\": {},", problem.num_constraints).unwrap();
    writeln!(out, "  \"obj_offset\": {},", f64_json(problem.obj_offset)).unwrap();

    dump_csc(&mut out, "Q", &problem.q);
    out.push_str(",\n");
    dump_vec_f64(&mut out, "c", &problem.c);
    out.push_str(",\n");
    dump_csc(&mut out, "A", &problem.a);
    out.push_str(",\n");
    dump_vec_f64(&mut out, "b", &problem.b);
    out.push_str(",\n");

    let ctype: Vec<u8> = problem
        .constraint_types
        .iter()
        .map(|c| match c {
            ConstraintType::Le => 0u8,
            ConstraintType::Ge => 1u8,
            ConstraintType::Eq => 2u8,
            _ => 255u8,
        })
        .collect();
    writeln!(
        out,
        "  \"constraint_types\": [{}],",
        ctype
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();

    let lb: Vec<f64> = problem.bounds.iter().map(|(l, _)| *l).collect();
    let ub: Vec<f64> = problem.bounds.iter().map(|(_, u)| *u).collect();
    dump_vec_f64(&mut out, "bounds_lb", &lb);
    out.push_str(",\n");
    dump_vec_f64(&mut out, "bounds_ub", &ub);
    out.push('\n');
    out.push_str("}\n");

    if let Err(e) = std::fs::write(out_path, out) {
        eprintln!("Write error: {}", e);
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn dump_csc(out: &mut String, name: &str, m: &CscMatrix) {
    writeln!(out, "  \"{}\": {{", name).unwrap();
    writeln!(out, "    \"nrows\": {},", m.nrows()).unwrap();
    writeln!(out, "    \"ncols\": {},", m.ncols()).unwrap();
    writeln!(
        out,
        "    \"col_ptr\": [{}],",
        m.col_ptr()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();
    writeln!(
        out,
        "    \"row_ind\": [{}],",
        m.row_ind()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();
    writeln!(
        out,
        "    \"values\": [{}]",
        m.values()
            .iter()
            .map(|v| f64_json(*v))
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();
    out.push_str("  }");
}

fn dump_vec_f64(out: &mut String, name: &str, v: &[f64]) {
    writeln!(
        out,
        "  \"{}\": [{}]",
        name,
        v.iter()
            .map(|x| f64_json(*x))
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();
    // trailing newline removed for caller to add comma
    out.truncate(out.trim_end_matches('\n').len());
}

fn f64_json(x: f64) -> String {
    if x.is_nan() {
        "\"NaN\"".to_string()
    } else if x.is_infinite() {
        if x > 0.0 {
            "\"Infinity\"".to_string()
        } else {
            "\"-Infinity\"".to_string()
        }
    } else {
        format!("{:.17e}", x)
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
