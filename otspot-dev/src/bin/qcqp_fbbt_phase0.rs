//! Phase-0 free measurements for the QCQP root-stage FBBT (plan §2 a–g,
//! research doc §5): per-problem width distributions of the bound-requirement
//! set `R` under the RAW declared box and under the derived root box, the
//! `width > 1e6` counts, and which NOT_SUPPORTED-class problems FBBT
//! finitizes (routes forward). No solves run — parse, classify, propagate.
//!
//! Usage:
//! `qcqp_fbbt_phase0 [--out F.csv] [--vars DIR] DIR [DIR ...]`
//! (defaults: `--out qcqp_fbbt_phase0.csv`, dirs = `data/qplib`).
//! `--vars DIR` additionally dumps one `R`-variable row per problem
//! (`<id>.vars.csv`: raw/derived endpoints) for the branch-history × width join.

use otspot_core::conic::{
    classify_bound_requirements, derive_root_box, effective_bound_requirement, GlobalOptions,
};
use otspot_core::qp::qcqp_route::build_nonconvex_raw;
use otspot_io::qplib::{parse_qplib, QplibProblem};
use std::io::Write;
use std::path::{Path, PathBuf};

/// QPLIB ships "unbounded but defaulted" bounds at ±1e10-ish and the OBBT
/// filter question (research §5 P0) is whether any accepted-class `R` variable
/// carries a width past `1e6` — the threshold from the research doc's own
/// wording ("±1e6 を超える実幅").
const WIDE_WIDTH: f64 = 1e6;

fn width(l: f64, u: f64) -> Option<f64> {
    if l.is_finite() && u.is_finite() {
        Some(u - l)
    } else {
        None
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

struct RowStats {
    id: String,
    n: usize,
    nquad: usize,
    r_raw: usize,
    r_eff: usize,
    raw_bad: usize,
    derived_bad: usize,
    raw_wide: usize,
    derived_wide: usize,
    raw_med: f64,
    raw_p90: f64,
    raw_max: f64,
    der_med: f64,
    der_p90: f64,
    der_max: f64,
    gap_raw: f64,
    gap_der: f64,
    rounds: usize,
    tightened: usize,
    contradiction_rows: usize,
}

fn measure(path: &Path, vars_dir: Option<&Path>) -> Result<RowStats, String> {
    let id = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .trim_end_matches(".qplib")
        .to_string();
    let qp = match parse_qplib(path) {
        Ok(QplibProblem::Qp(p)) => p,
        Ok(_) => return Err("not a continuous QP/QCQP".to_string()),
        Err(e) => return Err(format!("parse: {e}")),
    };
    let nc = build_nonconvex_raw(&qp).map_err(|e| format!("build: {e}"))?;
    let raw_req = classify_bound_requirements(nc.n, nc.p0.as_ref(), &nc.quad)
        .map_err(|e| format!("classify: {e}"))?;
    let feas_tol = GlobalOptions::default().feas_tol;
    let box_ = derive_root_box(&nc, feas_tol);
    // The route's acceptance set (plan §3.2 line 89) is judged on the
    // EFFECTIVE requirement over the RAW bounds; `req.required` recomputed on
    // the derived box answers "would the gate pass after FBBT".
    let eff = effective_bound_requirement(nc.n, nc.p0.as_ref(), &nc.quad, &nc.lb, &nc.ub)
        .map_err(|e| format!("effective: {e}"))?;
    let mut raw_ws: Vec<f64> = Vec::new();
    let mut der_ws: Vec<f64> = Vec::new();
    let mut raw_bad = 0usize;
    let mut derived_bad = 0usize;
    let mut gap_raw = 0.0f64;
    let mut gap_der = 0.0f64;
    for j in 0..nc.n {
        if !eff.required[j] {
            continue;
        }
        match width(nc.lb[j], nc.ub[j]) {
            Some(w) => {
                raw_ws.push(w);
                gap_raw += w * w / 4.0;
            }
            None => raw_bad += 1,
        }
        match width(box_.lb[j], box_.ub[j]) {
            Some(w) => {
                der_ws.push(w);
                gap_der += w * w / 4.0;
            }
            None => derived_bad += 1,
        }
    }
    raw_ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    der_ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Per-variable dump (branch-history × width join, research doc §5 P0-2):
    // one row per `R` variable with raw and derived endpoints, so a run's
    // BRANCH records (variable index) can be priced against the box widths
    // the search actually saw.
    if let Some(dir) = vars_dir {
        let _ = std::fs::create_dir_all(dir);
        let mut lines = String::from("idx,raw_lb,raw_ub,der_lb,der_ub\n");
        for j in 0..nc.n {
            if !eff.required[j] {
                continue;
            }
            lines.push_str(&format!(
                "{},{},{},{},{}\n",
                j, nc.lb[j], nc.ub[j], box_.lb[j], box_.ub[j]
            ));
        }
        let file = dir.join(format!("{id}.vars.csv"));
        let _ = std::fs::write(file, lines);
    }
    Ok(RowStats {
        id,
        n: nc.n,
        nquad: nc.quad.len(),
        r_raw: raw_req.required.iter().filter(|b| **b).count(),
        r_eff: eff.required.iter().filter(|b| **b).count(),
        raw_bad,
        derived_bad,
        raw_wide: raw_ws.iter().filter(|w| **w > WIDE_WIDTH).count(),
        derived_wide: der_ws.iter().filter(|w| **w > WIDE_WIDTH).count(),
        raw_med: quantile(&raw_ws, 0.5),
        raw_p90: quantile(&raw_ws, 0.9),
        raw_max: raw_ws.last().copied().unwrap_or(f64::NAN),
        der_med: quantile(&der_ws, 0.5),
        der_p90: quantile(&der_ws, 0.9),
        der_max: der_ws.last().copied().unwrap_or(f64::NAN),
        gap_raw,
        gap_der,
        rounds: box_.rounds,
        tightened: box_.tightened,
        contradiction_rows: box_.contradiction_rows,
    })
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut out = PathBuf::from("qcqp_fbbt_phase0.csv");
    let mut vars_dir: Option<PathBuf> = None;
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--out" => {
                out = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--vars" => {
                vars_dir = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            a => {
                dirs.push(PathBuf::from(a));
                i += 1;
            }
        }
    }
    if dirs.is_empty() {
        dirs.push(PathBuf::from("data/qplib"));
    }
    let mut w = csv_writer(&out);
    let mut skipped = 0usize;
    let mut rows = 0usize;
    for dir in &dirs {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "qplib"))
            .collect();
        files.sort();
        for path in files {
            match measure(&path, vars_dir.as_deref()) {
                Ok(r) => {
                    let line = format!(
                        "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{}\n",
                        r.id, r.n, r.nquad, r.r_raw, r.r_eff, r.raw_bad, r.derived_bad,
                        r.raw_wide, r.derived_wide, r.raw_med, r.raw_p90, r.raw_max,
                        r.der_med, r.der_p90, r.der_max, r.gap_raw, r.gap_der,
                        r.rounds, r.tightened, r.contradiction_rows
                    );
                    w.write_all(line.as_bytes()).expect("row");
                    rows += 1;
                }
                Err(e) => {
                    eprintln!("[skip] {}: {e}", path.display());
                    skipped += 1;
                }
            }
        }
    }
    w.flush().expect("flush");
    println!("{rows} rows -> {} ({skipped} skipped)", out.display());
}

fn csv_writer(path: &Path) -> impl Write {
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display())),
    );
    f.write_all(
        b"id,n,nquad,r_raw,r_eff,raw_bad,derived_bad,raw_wide1e6,derived_wide1e6,\
          raw_med,raw_p90,raw_max,der_med,der_p90,der_max,gap_raw,gap_der,rounds,tightened,contradictions\n",
    )
    .expect("header");
    f
}
