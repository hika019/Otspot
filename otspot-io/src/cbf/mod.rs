//! CBF (Conic Benchmark Format) parser, bridging to [`otspot_core::conic`].
//!
//! CBF (<https://cblib.zib.de>) describes second-order-cone and mixed-integer
//! second-order-cone programs as a sequence of sections. This parser accepts
//! `VER` 1-3 and the sections `OBJSENSE`, `VAR`, `INT`, `CON`, `OBJACOORD`,
//! `OBJBCOORD`, `ACOORD`, `BCOORD`, with cone types `F`, `L+`, `L-`, `L=`,
//! `Q`, `QR`. Semidefinite sections (`PSDVAR`, `PSDCON`, `OBJFCOORD`,
//! `FCOORD`, `HCOORD`, `DCOORD`) and parametric/exponential cones
//! (`POWCONES`, `POW*CONES`, `EXP`) are explicitly rejected rather than
//! silently ignored.

mod bridge;
mod cone;
mod parser;
mod token_stream;

use std::io::BufRead;
use std::path::Path;

use token_stream::TokenStream;

pub use bridge::CbfProblem;

/// Errors produced by the CBF parser.
#[non_exhaustive]
#[derive(Debug)]
pub enum CbfError {
    /// I/O error reading from the source.
    IoError(std::io::Error),
    /// Malformed content (bad token, out-of-range index, size mismatch, ...).
    ParseError(String),
    /// A recognized but unsupported section or cone type (e.g. `PSDVAR`,
    /// `EXP`, an unrecognized `VER`).
    Unsupported(String),
}

impl std::fmt::Display for CbfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CbfError::IoError(e) => write!(f, "I/O error: {e}"),
            CbfError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            CbfError::Unsupported(msg) => write!(f, "Unsupported: {msg}"),
        }
    }
}

impl std::error::Error for CbfError {}

impl From<std::io::Error> for CbfError {
    fn from(e: std::io::Error) -> Self {
        CbfError::IoError(e)
    }
}

/// Parses a CBF file from `path`.
pub fn parse_cbf(path: &Path) -> Result<CbfProblem, CbfError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    bridge::build(parser::parse_token_stream(TokenStream::from_reader(
        reader,
    ))?)
}

/// Parses a CBF file's contents from a string.
pub fn parse_cbf_str(input: &str) -> Result<CbfProblem, CbfError> {
    bridge::build(parser::parse_token_stream(TokenStream::from_str(input))?)
}

/// Parses CBF content from any `BufRead` source.
pub fn parse_cbf_reader<R: BufRead + 'static>(reader: R) -> Result<CbfProblem, CbfError> {
    bridge::build(parser::parse_token_stream(TokenStream::from_reader(
        reader,
    ))?)
}

#[cfg(test)]
mod tests;
