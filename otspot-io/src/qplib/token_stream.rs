use std::collections::VecDeque;
use std::io::{BufRead, Cursor};

use super::QplibError;

/// Whitespace/comment-stripping token stream.
///
/// Both string and reader inputs use the same line-at-a-time backend, so record
/// comments and errors have identical semantics without loading large files.
pub(super) struct TokenStream {
    inner: TsInner,
    token_ordinal: usize,
}

enum TsInner {
    Stream {
        reader: Box<dyn BufRead>,
        pending: VecDeque<String>,
        line_buf: String,
        /// Sticky I/O error: set on the first `read_line` failure.
        io_err: Option<std::io::Error>,
        line_number: usize,
    },
}

impl TokenStream {
    pub(super) fn from_str(input: &str) -> Self {
        Self::from_reader(Cursor::new(input.as_bytes().to_vec()))
    }

    pub(super) fn from_reader<R: BufRead + 'static>(reader: R) -> Self {
        TokenStream {
            inner: TsInner::Stream {
                reader: Box::new(reader),
                pending: VecDeque::new(),
                line_buf: String::new(),
                io_err: None,
                line_number: 0,
            },
            token_ordinal: 0,
        }
    }

    /// Returns the next token, or `None` at EOF (or after a sticky I/O error).
    pub(super) fn next_token(&mut self) -> Option<String> {
        let token = match &mut self.inner {
            TsInner::Stream {
                reader,
                pending,
                line_buf,
                io_err,
                line_number,
            } => loop {
                if io_err.is_some() {
                    break None;
                }
                if let Some(tok) = pending.pop_front() {
                    break Some(tok);
                }
                line_buf.clear();
                match reader.read_line(line_buf) {
                    Ok(0) => break None,
                    Err(e) => {
                        *io_err = Some(e);
                        break None;
                    }
                    Ok(_) => {
                        *line_number += 1;
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
        };
        if token.is_some() {
            self.token_ordinal += 1;
        }
        token
    }

    /// Discards the unused words on the current physical record. QPLIB permits
    /// plain-text annotations after the required fields of a record.
    pub(super) fn finish_record(&mut self) {
        let TsInner::Stream { pending, .. } = &mut self.inner;
        pending.clear();
    }

    pub(super) fn line_number(&self) -> usize {
        let TsInner::Stream { line_number, .. } = &self.inner;
        *line_number
    }

    /// Takes a sticky I/O error, or returns `None` (no error).
    pub(super) fn take_io_err(&mut self) -> Option<std::io::Error> {
        match &mut self.inner {
            TsInner::Stream { io_err, .. } => io_err.take(),
        }
    }

    pub(super) fn read_string(&mut self) -> Result<String, QplibError> {
        match self.next_token() {
            Some(t) => Ok(t),
            None => Err(self
                .take_io_err()
                .map(QplibError::IoError)
                .unwrap_or_else(|| {
                    QplibError::ParseError("unexpected end of file (expected string)".to_string())
                })),
        }
    }

    pub(super) fn read_usize(&mut self, context: &str) -> Result<usize, QplibError> {
        let t = match self.next_token() {
            Some(t) => t,
            None => {
                return Err(self
                    .take_io_err()
                    .map(QplibError::IoError)
                    .unwrap_or_else(|| {
                        QplibError::ParseError(format!(
                            "line {}, token {} ({}): unexpected end of file (expected integer)",
                            self.line_number(),
                            self.token_ordinal + 1,
                            context
                        ))
                    }))
            }
        };
        // QPLIB dimensions, entry counts, and indices are integer fields.
        // Parsing through f64 and casting would silently truncate fractions and
        // saturate negative, NaN, or overflowing values before the parser can
        // validate the surrounding structure.
        t.parse::<usize>().map_err(|_| {
            QplibError::ParseError(format!(
                "line {}, token {} ({}): expected integer, got '{}'",
                self.line_number(),
                self.token_ordinal,
                context,
                t
            ))
        })
    }

    pub(super) fn read_f64(&mut self) -> Result<f64, QplibError> {
        let t = match self.next_token() {
            Some(t) => t,
            None => {
                return Err(self
                    .take_io_err()
                    .map(QplibError::IoError)
                    .unwrap_or_else(|| {
                        QplibError::ParseError(
                            "unexpected end of file (expected float)".to_string(),
                        )
                    }))
            }
        };
        let normalized = t.replace(['D', 'd'], "E");
        normalized.parse::<f64>().map_err(|_| {
            QplibError::ParseError(format!(
                "line {}: expected real number, got '{}'",
                self.line_number(),
                t
            ))
        })
    }

    /// Reads a 1-based index, validates range, and returns the 0-based equivalent.
    pub(super) fn read_index_1based(
        &mut self,
        max_val: usize,
        context: &str,
    ) -> Result<usize, QplibError> {
        let raw = self.read_usize(context)?;
        if raw == 0 || raw > max_val {
            return Err(QplibError::ParseError(format!(
                "line {}, token {} ({}): index {} out of range (expected 1..={})",
                self.line_number(),
                self.token_ordinal,
                context,
                raw,
                max_val
            )));
        }
        Ok(raw - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::TokenStream;
    use std::io::Cursor;

    #[test]
    fn read_usize_rejects_non_decimal_integer_tokens() {
        for token in ["1.5", "-1", "NaN", "1e100"] {
            let mut ts = TokenStream::from_str(token);
            let err = ts
                .read_usize("test count")
                .expect_err("non-decimal integer token must not be coerced to usize");
            assert!(
                err.to_string()
                    .contains("token 1 (test count): expected integer"),
                "unexpected error for {token:?}: {err}"
            );
        }
    }

    #[test]
    fn read_usize_rejects_plain_decimal_overflow() {
        let overflow = (usize::MAX as u128 + 1).to_string();
        let mut ts = TokenStream::from_str(&overflow);
        let err = ts
            .read_usize("number of entries")
            .expect_err("usize::MAX + 1 must be rejected");
        assert!(
            err.to_string()
                .contains("token 1 (number of entries): expected integer"),
            "{err}"
        );
    }

    #[test]
    fn read_usize_accepts_explicit_plus_sign() {
        let mut ts = TokenStream::from_str("+1");
        assert_eq!(ts.read_usize("number of entries").unwrap(), 1);
    }

    #[test]
    fn finish_record_discards_plain_text_but_not_the_next_line() {
        let mut ts = TokenStream::from_str("2 required words are a comment\n3 next\n");
        assert_eq!(ts.read_usize("first").unwrap(), 2);
        ts.finish_record();
        assert_eq!(ts.read_usize("second").unwrap(), 3);
        assert_eq!(ts.line_number(), 2);
    }

    #[test]
    fn read_f64_accepts_fortran_exponents() {
        let mut ts = TokenStream::from_str("1.25D+2\n-2.5d-1\n");
        assert_eq!(ts.read_f64().unwrap(), 125.0);
        ts.finish_record();
        assert_eq!(ts.read_f64().unwrap(), -0.25);
    }

    #[test]
    fn streamed_integer_error_reports_ordinal_and_context() {
        let mut ts = TokenStream::from_reader(Cursor::new(b"name\n1.5\n"));
        assert_eq!(ts.read_string().unwrap(), "name");
        let err = ts
            .read_usize("number of variables")
            .expect_err("a streamed fractional integer must be rejected");
        assert!(
            err.to_string()
                .contains("token 2 (number of variables): expected integer"),
            "{err}"
        );
    }

    #[test]
    fn read_index_rejects_fractional_token() {
        let mut ts = TokenStream::from_str("1.5");
        let err = ts
            .read_index_1based(2, "test index")
            .expect_err("a fractional index must not be truncated");
        assert!(err.to_string().contains("expected integer"), "{err}");
    }
}
