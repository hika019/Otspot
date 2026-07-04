use std::collections::VecDeque;
use std::io::BufRead;

use super::CbfError;

/// Whitespace-splitting token stream for CBF files.
///
/// CBF comments are full-line only (first byte `#`); blank lines are
/// separators between information items and carry no tokens.
///
/// Two backends mirror the QPLIB tokenizer: `Mem` (pre-tokenized `&str`,
/// used by tests and `parse_cbf_str`) and `Stream` (line-at-a-time from a
/// `BufRead`, O(1) memory for large files).
pub(super) struct TokenStream {
    inner: TsInner,
}

enum TsInner {
    Mem {
        tokens: Vec<String>,
        pos: usize,
    },
    Stream {
        reader: Box<dyn BufRead>,
        pending: VecDeque<String>,
        line_buf: String,
        io_err: Option<std::io::Error>,
    },
}

const COMMENT_PREFIX: char = '#';

fn tokenize_line(line: &str, out: &mut Vec<String>) {
    if line.trim_start().starts_with(COMMENT_PREFIX) {
        return;
    }
    for token in line.split_whitespace() {
        out.push(token.to_string());
    }
}

impl TokenStream {
    pub(super) fn from_str(input: &str) -> Self {
        let mut tokens = Vec::new();
        for line in input.lines() {
            tokenize_line(line, &mut tokens);
        }
        TokenStream {
            inner: TsInner::Mem { tokens, pos: 0 },
        }
    }

    pub(super) fn from_reader<R: BufRead + 'static>(reader: R) -> Self {
        TokenStream {
            inner: TsInner::Stream {
                reader: Box::new(reader),
                pending: VecDeque::new(),
                line_buf: String::new(),
                io_err: None,
            },
        }
    }

    pub(super) fn next_token(&mut self) -> Option<String> {
        match &mut self.inner {
            TsInner::Mem { tokens, pos } => {
                if *pos < tokens.len() {
                    let t = tokens[*pos].clone();
                    *pos += 1;
                    Some(t)
                } else {
                    None
                }
            }
            TsInner::Stream {
                reader,
                pending,
                line_buf,
                io_err,
            } => loop {
                if io_err.is_some() {
                    return None;
                }
                if let Some(tok) = pending.pop_front() {
                    return Some(tok);
                }
                line_buf.clear();
                match reader.read_line(line_buf) {
                    Ok(0) => return None,
                    Err(e) => {
                        *io_err = Some(e);
                        return None;
                    }
                    Ok(_) => {
                        let mut buf = Vec::new();
                        tokenize_line(line_buf, &mut buf);
                        pending.extend(buf);
                    }
                }
            },
        }
    }

    pub(super) fn take_io_err(&mut self) -> Option<std::io::Error> {
        match &mut self.inner {
            TsInner::Stream { io_err, .. } => io_err.take(),
            TsInner::Mem { .. } => None,
        }
    }

    fn eof_error(&mut self, expected: &str) -> CbfError {
        self.take_io_err()
            .map(CbfError::IoError)
            .unwrap_or_else(|| {
                CbfError::ParseError(format!("unexpected end of file (expected {expected})"))
            })
    }

    pub(super) fn read_string(&mut self) -> Result<String, CbfError> {
        self.next_token().ok_or_else(|| self.eof_error("token"))
    }

    pub(super) fn read_usize(&mut self) -> Result<usize, CbfError> {
        let t = self.read_string()?;
        t.parse::<usize>()
            .map_err(|_| CbfError::ParseError(format!("expected non-negative integer, got '{t}'")))
    }

    pub(super) fn read_f64(&mut self) -> Result<f64, CbfError> {
        let t = self.read_string()?;
        let v = t
            .parse::<f64>()
            .map_err(|_| CbfError::ParseError(format!("expected float, got '{t}'")))?;
        if !v.is_finite() {
            return Err(CbfError::ParseError(format!(
                "expected finite float, got '{t}'"
            )));
        }
        Ok(v)
    }

    /// Reads a 0-based index and validates it against `bound` (exclusive).
    pub(super) fn read_index_0based(
        &mut self,
        bound: usize,
        context: &str,
    ) -> Result<usize, CbfError> {
        let idx = self.read_usize()?;
        if idx >= bound {
            return Err(CbfError::ParseError(format!(
                "{context}: index {idx} out of range (expected 0..{bound})"
            )));
        }
        Ok(idx)
    }
}
