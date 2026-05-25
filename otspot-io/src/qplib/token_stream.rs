use std::collections::VecDeque;
use std::io::BufRead;

use super::QplibError;

/// Whitespace/comment-stripping token stream.
///
/// Two backends:
/// - `Mem`: all tokens pre-loaded from a `&str` (used by `parse_qplib_str` / tests).
/// - `Stream`: reads one line at a time from a `BufRead`; O(1) memory regardless
///   of file size (used by `parse_qplib` to avoid OOM on 200 MB+ files).
pub(super) struct TokenStream {
    inner: TsInner,
}

enum TsInner {
    Mem { tokens: Vec<String>, pos: usize },
    Stream {
        reader: Box<dyn BufRead>,
        pending: VecDeque<String>,
        line_buf: String,
        /// Sticky I/O error: set on the first `read_line` failure.
        io_err: Option<std::io::Error>,
    },
}

impl TokenStream {
    pub(super) fn from_str(input: &str) -> Self {
        let mut tokens = Vec::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('%') || trimmed.starts_with('!') {
                continue;
            }
            let effective = if let Some(idx) = line.find('#') { &line[..idx] } else { line };
            for token in effective.split_whitespace() {
                tokens.push(token.to_string());
            }
        }
        TokenStream { inner: TsInner::Mem { tokens, pos: 0 } }
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

    /// Returns the next token, or `None` at EOF (or after a sticky I/O error).
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
            TsInner::Stream { reader, pending, line_buf, io_err } => loop {
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
                        let trimmed = line_buf.trim();
                        if trimmed.starts_with('%') || trimmed.starts_with('!') {
                            continue;
                        }
                        let effective = if let Some(idx) = line_buf.find('#') {
                            &line_buf[..idx]
                        } else {
                            line_buf.as_str()
                        };
                        for token in effective.split_whitespace() {
                            pending.push_back(token.to_string());
                        }
                    }
                }
            },
        }
    }

    /// Takes a sticky I/O error, or returns `None` (no error).
    pub(super) fn take_io_err(&mut self) -> Option<std::io::Error> {
        match &mut self.inner {
            TsInner::Stream { io_err, .. } => io_err.take(),
            TsInner::Mem { .. } => None,
        }
    }

    pub(super) fn read_string(&mut self) -> Result<String, QplibError> {
        match self.next_token() {
            Some(t) => Ok(t),
            None => Err(self.take_io_err().map(QplibError::IoError).unwrap_or_else(|| {
                QplibError::ParseError("unexpected end of file (expected string)".to_string())
            })),
        }
    }

    pub(super) fn read_usize(&mut self) -> Result<usize, QplibError> {
        let t = match self.next_token() {
            Some(t) => t,
            None => {
                return Err(self.take_io_err().map(QplibError::IoError).unwrap_or_else(|| {
                    QplibError::ParseError(
                        "unexpected end of file (expected integer)".to_string(),
                    )
                }))
            }
        };
        if let Ok(u) = t.parse::<usize>() {
            Ok(u)
        } else if let Ok(f) = t.parse::<f64>() {
            Ok(f as usize)
        } else {
            Err(QplibError::ParseError(format!("expected integer, got '{}'", t)))
        }
    }

    pub(super) fn read_f64(&mut self) -> Result<f64, QplibError> {
        let t = match self.next_token() {
            Some(t) => t,
            None => {
                return Err(self.take_io_err().map(QplibError::IoError).unwrap_or_else(|| {
                    QplibError::ParseError(
                        "unexpected end of file (expected float)".to_string(),
                    )
                }))
            }
        };
        t.parse::<f64>()
            .map_err(|_| QplibError::ParseError(format!("expected float, got '{}'", t)))
    }

    /// Reads a 1-based index, validates range, and returns the 0-based equivalent.
    pub(super) fn read_index_1based(
        &mut self,
        max_val: usize,
        context: &str,
    ) -> Result<usize, QplibError> {
        let raw = self.read_usize()?;
        if raw == 0 || raw > max_val {
            return Err(QplibError::ParseError(format!(
                "{}: index {} out of range (expected 1..={})",
                context, raw, max_val
            )));
        }
        Ok(raw - 1)
    }
}
