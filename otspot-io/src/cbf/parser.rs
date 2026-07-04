use std::collections::HashSet;

use super::cone::{read_cone_blocks, ConeBlock};
use super::token_stream::TokenStream;
use super::CbfError;

/// Oldest CBF version this parser accepts.
const MIN_CBF_VERSION: u32 = 1;
/// Newest CBF version this parser accepts.
const MAX_CBF_VERSION: u32 = 3;

const UNSUPPORTED_SECTIONS: &[&str] = &[
    "PSDVAR",
    "PSDCON",
    "OBJFCOORD",
    "FCOORD",
    "HCOORD",
    "DCOORD",
    "POWCONES",
    "POW*CONES",
];

/// Parsed CBF file content, prior to conversion into a solver-facing problem.
pub(super) struct RawCbf {
    pub(super) maximize: bool,
    pub(super) n: usize,
    pub(super) var_blocks: Vec<ConeBlock>,
    pub(super) integers: Vec<usize>,
    pub(super) m: usize,
    pub(super) con_blocks: Vec<ConeBlock>,
    /// Sparse objective linear coefficients: `(var, coeff)`.
    pub(super) obj_a: Vec<(usize, f64)>,
    /// Objective constant term (`OBJBCOORD`), in the file's stated sense.
    pub(super) obj_b: f64,
    /// Sparse constraint matrix entries: `(row, var, coeff)`.
    pub(super) a_coord: Vec<(usize, usize, f64)>,
    /// Sparse constraint constant entries: `(row, value)`.
    pub(super) b_coord: Vec<(usize, f64)>,
}

pub(super) fn parse_token_stream(mut ts: TokenStream) -> Result<RawCbf, CbfError> {
    let mut version: Option<u32> = None;
    let mut maximize: Option<bool> = None;
    let mut n = 0usize;
    let mut var_blocks: Vec<ConeBlock> = Vec::new();
    let mut var_seen = false;
    let mut integers: Vec<usize> = Vec::new();
    let mut m = 0usize;
    let mut con_blocks: Vec<ConeBlock> = Vec::new();
    let mut obj_a: Vec<(usize, f64)> = Vec::new();
    let mut obj_a_seen: HashSet<usize> = HashSet::new();
    let mut obj_b = 0.0f64;
    let mut a_coord: Vec<(usize, usize, f64)> = Vec::new();
    let mut a_seen: HashSet<(usize, usize)> = HashSet::new();
    let mut b_coord: Vec<(usize, f64)> = Vec::new();
    let mut b_seen: HashSet<usize> = HashSet::new();

    while let Some(kw) = ts.next_token() {
        match kw.as_str() {
            "VER" => {
                let raw = ts.read_usize()?;
                let v = u32::try_from(raw)
                    .map_err(|_| CbfError::ParseError(format!("VER {raw} out of range")))?;
                if !(MIN_CBF_VERSION..=MAX_CBF_VERSION).contains(&v) {
                    return Err(CbfError::Unsupported(format!(
                        "CBF VER {v} is not supported (supported: {MIN_CBF_VERSION}-{MAX_CBF_VERSION})"
                    )));
                }
                version = Some(v);
            }
            "OBJSENSE" => {
                let s = ts.read_string()?;
                maximize = Some(match s.as_str() {
                    "MIN" => false,
                    "MAX" => true,
                    other => {
                        return Err(CbfError::ParseError(format!(
                            "OBJSENSE: expected MIN or MAX, got '{other}'"
                        )))
                    }
                });
            }
            "VAR" => {
                let (total, blocks) = read_cone_blocks(&mut ts)?;
                n = total;
                var_blocks = blocks;
                var_seen = true;
            }
            "INT" => {
                let k = ts.read_usize()?;
                for _ in 0..k {
                    integers.push(ts.read_index_0based(n, "INT")?);
                }
            }
            "CON" => {
                let (total, blocks) = read_cone_blocks(&mut ts)?;
                m = total;
                con_blocks = blocks;
            }
            "OBJACOORD" => {
                let k = ts.read_usize()?;
                for _ in 0..k {
                    let v = ts.read_index_0based(n, "OBJACOORD")?;
                    let val = ts.read_f64()?;
                    if !obj_a_seen.insert(v) {
                        return Err(CbfError::ParseError(format!(
                            "OBJACOORD: duplicate entry for variable {v}"
                        )));
                    }
                    obj_a.push((v, val));
                }
            }
            "OBJBCOORD" => {
                obj_b = ts.read_f64()?;
            }
            "ACOORD" => {
                let k = ts.read_usize()?;
                for _ in 0..k {
                    let row = ts.read_index_0based(m, "ACOORD row")?;
                    let col = ts.read_index_0based(n, "ACOORD var")?;
                    let val = ts.read_f64()?;
                    if !a_seen.insert((row, col)) {
                        return Err(CbfError::ParseError(format!(
                            "ACOORD: duplicate entry at (row {row}, var {col})"
                        )));
                    }
                    a_coord.push((row, col, val));
                }
            }
            "BCOORD" => {
                let k = ts.read_usize()?;
                for _ in 0..k {
                    let row = ts.read_index_0based(m, "BCOORD row")?;
                    let val = ts.read_f64()?;
                    if !b_seen.insert(row) {
                        return Err(CbfError::ParseError(format!(
                            "BCOORD: duplicate entry for row {row}"
                        )));
                    }
                    b_coord.push((row, val));
                }
            }
            // The spec permits readers to interpret CHANGE (incremental
            // problem updates) as end-of-file.
            "CHANGE" => break,
            kw if UNSUPPORTED_SECTIONS.contains(&kw) => {
                return Err(CbfError::Unsupported(format!(
                    "CBF section '{kw}' is not supported"
                )));
            }
            other => {
                return Err(CbfError::ParseError(format!(
                    "unknown CBF section keyword '{other}'"
                )))
            }
        }
    }
    if let Some(e) = ts.take_io_err() {
        return Err(CbfError::IoError(e));
    }

    let _version = version.ok_or_else(|| CbfError::ParseError("missing VER section".into()))?;
    let maximize =
        maximize.ok_or_else(|| CbfError::ParseError("missing OBJSENSE section".into()))?;
    if !var_seen {
        return Err(CbfError::ParseError("missing VAR section".into()));
    }

    Ok(RawCbf {
        maximize,
        n,
        var_blocks,
        integers,
        m,
        con_blocks,
        obj_a,
        obj_b,
        a_coord,
        b_coord,
    })
}
