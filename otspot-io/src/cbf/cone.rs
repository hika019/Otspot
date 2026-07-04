//! CBF cone-domain tokens shared by the `VAR` and `CON` sections.

use super::token_stream::TokenStream;
use super::CbfError;

/// Minimum dimension of a (non-rotated) second-order cone `Q`.
const SOC_MIN_DIM: usize = 2;
/// Minimum dimension of a rotated second-order cone `QR`.
const ROTATED_SOC_MIN_DIM: usize = 3;

/// The cone assigned to a contiguous block of variables or constraint rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConeKind {
    /// `F`: unrestricted.
    Free,
    /// `L+`: nonnegative orthant.
    Lpos,
    /// `L-`: nonpositive orthant.
    Lneg,
    /// `L=`: fixed at zero.
    Lzero,
    /// `Q`: standard second-order cone.
    Soc,
    /// `QR`: rotated second-order cone.
    SocRotated,
}

/// A contiguous block of `size` variables/rows assigned to `kind`.
#[derive(Debug, Clone, Copy)]
pub(super) struct ConeBlock {
    pub(super) kind: ConeKind,
    pub(super) size: usize,
}

fn parse_cone_token(tok: &str, size: usize) -> Result<ConeKind, CbfError> {
    match tok {
        "F" => Ok(ConeKind::Free),
        "L+" => Ok(ConeKind::Lpos),
        "L-" => Ok(ConeKind::Lneg),
        "L=" => Ok(ConeKind::Lzero),
        "Q" => {
            if size < SOC_MIN_DIM {
                return Err(CbfError::ParseError(format!(
                    "Q cone size {size} < {SOC_MIN_DIM}"
                )));
            }
            Ok(ConeKind::Soc)
        }
        "QR" => {
            if size < ROTATED_SOC_MIN_DIM {
                return Err(CbfError::ParseError(format!(
                    "QR cone size {size} < {ROTATED_SOC_MIN_DIM}"
                )));
            }
            Ok(ConeKind::SocRotated)
        }
        other => Err(CbfError::Unsupported(format!(
            "cone type '{other}' is not supported (only F, L+, L-, L=, Q, QR)"
        ))),
    }
}

/// Reads a `VAR`/`CON`-style header: `<total> <num_blocks>` followed by
/// `<num_blocks>` lines of `<cone_token> <block_size>`.
pub(super) fn read_cone_blocks(ts: &mut TokenStream) -> Result<(usize, Vec<ConeBlock>), CbfError> {
    let total = ts.read_usize()?;
    let num_blocks = ts.read_usize()?;
    let mut blocks = Vec::with_capacity(num_blocks);
    let mut sum = 0usize;
    for _ in 0..num_blocks {
        let tok = ts.read_string()?;
        let size = ts.read_usize()?;
        let kind = parse_cone_token(&tok, size)?;
        sum += size;
        blocks.push(ConeBlock { kind, size });
    }
    if sum != total {
        return Err(CbfError::ParseError(format!(
            "cone block sizes sum to {sum}, expected {total}"
        )));
    }
    Ok((total, blocks))
}
