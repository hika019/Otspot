//! Converts parsed CBF content into [`ConicProblem`]/[`MisocpProblem`].
//!
//! Sign conventions follow [`otspot_core::conic::ConicProblem`]: `G x + s = h`,
//! `s in K`. For a CBF affine expression `A x + b`:
//! - `L+` (`A x + b >= 0`): `G = -A`, `h = b`  (rows land in the orthant block).
//! - `L-` (`A x + b <= 0`): `G = A`,  `h = -b` (rows land in the orthant block).
//! - `L=` (`A x + b == 0`): appended to the equality system as `A x = -b`.
//! - `Q`  (`A x + b in Q`): `G = -A`, `h = b`  (rows land in an SOC block).
//! - `QR` (`A x + b in Q_rotated`): same as `Q`, then the block's rows are
//!   linearly transformed into a standard SOC via
//!   `((u+v)/sqrt(2), (u-v)/sqrt(2), w)`.
//!
//! Variable-domain cones (from the `VAR` section) are handled identically by
//! treating the affine expression as the identity map `x_block` (`A = I`, `b = 0`).

use std::collections::BTreeMap;

use otspot_core::conic::{ConeSpec, ConicProblem, MisocpProblem};
use otspot_core::sparse::CscMatrix;

use super::cone::{ConeBlock, ConeKind};
use super::parser::RawCbf;
use super::CbfError;

/// `1/sqrt(2)`, the rotated-second-order-cone-to-standard-cone scale factor.
use std::f64::consts::FRAC_1_SQRT_2;

/// A parsed CBF instance: either a continuous SOCP or a mixed-integer SOCP,
/// plus the metadata needed to recover the original objective value and sense.
#[derive(Debug)]
pub enum CbfProblem {
    /// No `INT` entries: a continuous second-order cone program.
    Socp {
        /// The conic problem, in `min c^T x` standard form.
        problem: ConicProblem,
        /// `true` if the CBF file declared `OBJSENSE MAX`.
        maximize: bool,
        /// The CBF `OBJBCOORD` constant, in the file's original sense.
        obj_offset: f64,
    },
    /// One or more `INT` entries: a mixed-integer second-order cone program.
    Misocp {
        /// The mixed-integer conic problem.
        problem: MisocpProblem,
        /// `true` if the CBF file declared `OBJSENSE MAX`.
        maximize: bool,
        /// The CBF `OBJBCOORD` constant, in the file's original sense.
        obj_offset: f64,
    },
}

impl CbfProblem {
    /// `true` if the CBF file declared `OBJSENSE MAX`.
    pub fn maximize(&self) -> bool {
        match self {
            CbfProblem::Socp { maximize, .. } | CbfProblem::Misocp { maximize, .. } => *maximize,
        }
    }

    /// The CBF objective constant term (`OBJBCOORD`), in the file's original sense.
    pub fn obj_offset(&self) -> f64 {
        match self {
            CbfProblem::Socp { obj_offset, .. } | CbfProblem::Misocp { obj_offset, .. } => {
                *obj_offset
            }
        }
    }

    /// Recovers the true objective value (in the CBF file's original sense
    /// and units) from a solver-reported `c^T x` value.
    ///
    /// `solve_socp`/`solve_misocp` always minimise; when the file declared
    /// `OBJSENSE MAX` the internal objective coefficients were negated, so
    /// the reported value must be negated back before adding the constant.
    pub fn true_objective(&self, conic_objective: f64) -> f64 {
        let signed = if self.maximize() {
            -conic_objective
        } else {
            conic_objective
        };
        signed + self.obj_offset()
    }
}

/// One row of the (yet-unassembled) `G`/`A` matrices: sparse `(col, value)`
/// entries plus the corresponding `h`/`b` scalar.
struct RawRow {
    entries: Vec<(usize, f64)>,
    rhs: f64,
}

/// `wa * a + wb * b`, merging entries at shared columns.
fn combine_rows(a: &RawRow, wa: f64, b: &RawRow, wb: f64) -> RawRow {
    let mut merged: BTreeMap<usize, f64> = BTreeMap::new();
    for &(col, val) in &a.entries {
        *merged.entry(col).or_insert(0.0) += wa * val;
    }
    for &(col, val) in &b.entries {
        *merged.entry(col).or_insert(0.0) += wb * val;
    }
    RawRow {
        entries: merged.into_iter().filter(|&(_, v)| v != 0.0).collect(),
        rhs: wa * a.rhs + wb * b.rhs,
    }
}

/// Applies the rotated-SOC-to-standard-SOC transform to a block's rows:
/// `((u+v)/sqrt(2), (u-v)/sqrt(2), w)`, where `u = rows[0]`, `v = rows[1]`,
/// `w = rows[2..]` (possibly empty).
fn rotate_soc_block(rows: Vec<RawRow>) -> Vec<RawRow> {
    let mut it = rows.into_iter();
    let u = it.next().expect("QR block has size >= 2");
    let v = it.next().expect("QR block has size >= 2");
    let mut out = vec![
        combine_rows(&u, FRAC_1_SQRT_2, &v, FRAC_1_SQRT_2),
        combine_rows(&u, FRAC_1_SQRT_2, &v, -FRAC_1_SQRT_2),
    ];
    out.extend(it);
    out
}

/// Builder accumulating the equality system, orthant rows, and SOC blocks
/// that eventually become a [`ConicProblem`]'s `A`/`G`/`cone`.
#[derive(Default)]
struct ConicBuilder {
    eq_rows: Vec<RawRow>,
    orthant_rows: Vec<RawRow>,
    soc_blocks: Vec<Vec<RawRow>>,
}

impl ConicBuilder {
    /// Routes one affine expression `A_row x + b_row` (given as `row`) into
    /// the equality system, orthant block, or a fresh SOC block, per `kind`.
    /// `Free` rows are dropped (they impose no restriction).
    fn push_row(&mut self, kind: ConeKind, row: RawRow) {
        match kind {
            ConeKind::Free => {}
            ConeKind::Lzero => self.eq_rows.push(RawRow {
                entries: row.entries,
                rhs: -row.rhs,
            }),
            ConeKind::Lpos => self.orthant_rows.push(RawRow {
                entries: row.entries.into_iter().map(|(c, v)| (c, -v)).collect(),
                rhs: row.rhs,
            }),
            ConeKind::Lneg => self.orthant_rows.push(RawRow {
                entries: row.entries,
                rhs: -row.rhs,
            }),
            ConeKind::Soc | ConeKind::SocRotated => unreachable!("push_block handles cone rows"),
        }
    }

    /// Routes a whole `Q`/`QR` block (`kind` in `{Soc, SocRotated}`) of rows,
    /// each given as an affine expression `A_row x + b_row`.
    fn push_block(&mut self, kind: ConeKind, rows: Vec<RawRow>) {
        let negated: Vec<RawRow> = rows
            .into_iter()
            .map(|row| RawRow {
                entries: row.entries.into_iter().map(|(c, v)| (c, -v)).collect(),
                rhs: row.rhs,
            })
            .collect();
        let final_rows = match kind {
            ConeKind::Soc => negated,
            ConeKind::SocRotated => rotate_soc_block(negated),
            _ => unreachable!("push_block only handles Soc/SocRotated"),
        };
        self.soc_blocks.push(final_rows);
    }

    /// Assembles the accumulated rows into a complete [`ConicProblem`].
    fn finish(self, n: usize, c: Vec<f64>) -> Result<ConicProblem, CbfError> {
        let mut a_rows = Vec::new();
        let mut a_cols = Vec::new();
        let mut a_vals = Vec::new();
        let mut b_eq = Vec::with_capacity(self.eq_rows.len());
        for row in &self.eq_rows {
            let r = b_eq.len();
            for &(c, v) in &row.entries {
                a_rows.push(r);
                a_cols.push(c);
                a_vals.push(v);
            }
            b_eq.push(row.rhs);
        }
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, b_eq.len(), n)
            .map_err(|e| CbfError::ParseError(format!("equality matrix: {e}")))?;

        let mut g_rows = Vec::new();
        let mut g_cols = Vec::new();
        let mut g_vals = Vec::new();
        let mut h = Vec::new();
        for row in &self.orthant_rows {
            let r = h.len();
            for &(c, v) in &row.entries {
                g_rows.push(r);
                g_cols.push(c);
                g_vals.push(v);
            }
            h.push(row.rhs);
        }
        let l = h.len();
        let mut soc = Vec::with_capacity(self.soc_blocks.len());
        for block in &self.soc_blocks {
            for row in block {
                let r = h.len();
                for &(c, v) in &row.entries {
                    g_rows.push(r);
                    g_cols.push(c);
                    g_vals.push(v);
                }
                h.push(row.rhs);
            }
            soc.push(block.len());
        }
        let g = CscMatrix::from_triplets(&g_rows, &g_cols, &g_vals, h.len(), n)
            .map_err(|e| CbfError::ParseError(format!("conic matrix: {e}")))?;

        Ok(ConicProblem {
            c,
            a,
            b: b_eq,
            g,
            h,
            cone: ConeSpec { l, soc },
        })
    }
}

fn expand_blocks(blocks: &[ConeBlock]) -> Vec<(ConeKind, usize, usize)> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut cursor = 0usize;
    for b in blocks {
        out.push((b.kind, cursor, b.size));
        cursor += b.size;
    }
    out
}

/// Per-variable `(lb, ub)` implied directly by its `VAR` cone domain.
/// `Soc`/`SocRotated` couple variables together rather than bound them
/// individually, so they contribute no box information here.
fn variable_domain_bounds(n: usize, var_blocks: &[ConeBlock]) -> Vec<(f64, f64)> {
    let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
    for &(kind, start, size) in &expand_blocks(var_blocks) {
        let bound = match kind {
            ConeKind::Lpos => (0.0, f64::INFINITY),
            ConeKind::Lneg => (f64::NEG_INFINITY, 0.0),
            ConeKind::Lzero => (0.0, 0.0),
            ConeKind::Free | ConeKind::Soc | ConeKind::SocRotated => continue,
        };
        for b in &mut bounds[start..start + size] {
            *b = bound;
        }
    }
    bounds
}

/// Tightens `bounds` using `CON` rows that constrain exactly one variable
/// (the common encoding for e.g. `x <= 1` on a binary variable). Rows tying
/// together multiple variables are not a simple box constraint and are left
/// alone; general bound propagation is out of scope for this bridge.
fn tighten_single_variable_bounds(
    con_blocks: &[ConeBlock],
    a_by_row: &[Vec<(usize, f64)>],
    b_by_row: &[f64],
    bounds: &mut [(f64, f64)],
) {
    for &(kind, start, size) in &expand_blocks(con_blocks) {
        if !matches!(kind, ConeKind::Lpos | ConeKind::Lneg | ConeKind::Lzero) {
            continue;
        }
        for row in start..start + size {
            let &[(col, val)] = a_by_row[row].as_slice() else {
                continue;
            };
            if val == 0.0 {
                continue;
            }
            // Row expresses `val * x_col + b_by_row[row] {>=,<=,==} 0`.
            let boundary = -b_by_row[row] / val;
            let (row_lb, row_ub) = match kind {
                ConeKind::Lzero => (boundary, boundary),
                ConeKind::Lpos if val > 0.0 => (boundary, f64::INFINITY),
                ConeKind::Lpos => (f64::NEG_INFINITY, boundary),
                ConeKind::Lneg if val > 0.0 => (f64::NEG_INFINITY, boundary),
                ConeKind::Lneg => (boundary, f64::INFINITY),
                ConeKind::Free | ConeKind::Soc | ConeKind::SocRotated => unreachable!(),
            };
            let (lb, ub) = bounds[col];
            bounds[col] = (lb.max(row_lb), ub.min(row_ub));
        }
    }
}

pub(super) fn build(raw: RawCbf) -> Result<CbfProblem, CbfError> {
    let n = raw.n;
    let m = raw.m;

    let mut c = vec![0.0f64; n];
    for &(v, val) in &raw.obj_a {
        c[v] += val;
    }
    if raw.maximize {
        for v in c.iter_mut() {
            *v = -*v;
        }
    }

    let mut a_by_row: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for &(row, col, val) in &raw.a_coord {
        a_by_row[row].push((col, val));
    }
    let mut b_by_row = vec![0.0f64; m];
    for &(row, val) in &raw.b_coord {
        b_by_row[row] += val;
    }

    let mut builder = ConicBuilder::default();

    for &(kind, start, size) in &expand_blocks(&raw.con_blocks) {
        match kind {
            ConeKind::Soc | ConeKind::SocRotated => {
                let rows = (start..start + size)
                    .map(|r| RawRow {
                        entries: a_by_row[r].clone(),
                        rhs: b_by_row[r],
                    })
                    .collect();
                builder.push_block(kind, rows);
            }
            _ => {
                for r in start..start + size {
                    builder.push_row(
                        kind,
                        RawRow {
                            entries: a_by_row[r].clone(),
                            rhs: b_by_row[r],
                        },
                    );
                }
            }
        }
    }

    for &(kind, start, size) in &expand_blocks(&raw.var_blocks) {
        match kind {
            ConeKind::Soc | ConeKind::SocRotated => {
                let rows = (start..start + size)
                    .map(|v| RawRow {
                        entries: vec![(v, 1.0)],
                        rhs: 0.0,
                    })
                    .collect();
                builder.push_block(kind, rows);
            }
            _ => {
                for v in start..start + size {
                    builder.push_row(
                        kind,
                        RawRow {
                            entries: vec![(v, 1.0)],
                            rhs: 0.0,
                        },
                    );
                }
            }
        }
    }

    let problem = builder.finish(n, c)?;
    problem
        .validate()
        .map_err(|e| CbfError::ParseError(format!("bridge produced invalid ConicProblem: {e}")))?;

    if raw.integers.is_empty() {
        return Ok(CbfProblem::Socp {
            problem,
            maximize: raw.maximize,
            obj_offset: raw.obj_b,
        });
    }

    let mut bounds = variable_domain_bounds(n, &raw.var_blocks);
    tighten_single_variable_bounds(&raw.con_blocks, &a_by_row, &b_by_row, &mut bounds);

    let mut int_lb = Vec::with_capacity(raw.integers.len());
    let mut int_ub = Vec::with_capacity(raw.integers.len());
    for &v in &raw.integers {
        let (lb, ub) = bounds[v];
        if !lb.is_finite() || !ub.is_finite() {
            return Err(CbfError::Unsupported(format!(
                "integer variable {v} has no finite bound derivable from its VAR domain \
                 or a single-variable CON row (lb={lb}, ub={ub}); branch-and-bound requires \
                 finite bounds and general multi-row bound tightening is not supported"
            )));
        }
        int_lb.push(lb);
        int_ub.push(ub);
    }

    Ok(CbfProblem::Misocp {
        problem: MisocpProblem {
            base: problem,
            integers: raw.integers,
            int_lb,
            int_ub,
        },
        maximize: raw.maximize,
        obj_offset: raw.obj_b,
    })
}
