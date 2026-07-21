use otspot_core::mip::{MilpProblem, MiqpProblem};
use otspot_core::problem::{ConstraintType, LpProblem};
use otspot_core::qp::{QcqpMatrix, QpProblem};
use otspot_core::sparse::CscMatrix;
use std::collections::HashSet;

use super::token_stream::TokenStream;
use super::{QplibError, QplibProblem};

/// Relative tolerance for the QPLIB declared-infinity marker.
///
/// QPLIB files include an explicit `inf_val` field.  Any bound `x` satisfying
/// `|x| >= QPLIB_INF_REL_TOL * inf_val` is treated as ±∞.  The 1% margin
/// absorbs rounding during file generation without falsely classifying
/// finite bounds as infinite.
///
/// Source: QPLIB format (Furini et al., *Math. Prog. Computation* 2019, §3) defines the
/// `inf_val` marker concept; the 1% margin (0.99) is an implementation convention,
/// not explicitly specified in the paper.
const QPLIB_INF_REL_TOL: f64 = 0.99;

fn require_finite(value: f64, context: &str, line: usize) -> Result<(), QplibError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(QplibError::ParseError(format!(
            "line {line} ({context}): value must be finite"
        )))
    }
}

fn require_finite_or_infinite_marker(value: f64, context: &str) -> Result<(), QplibError> {
    if value.is_nan() {
        Err(QplibError::ParseError(format!(
            "{context}: NaN is not a valid QPLIB real value"
        )))
    } else {
        Ok(())
    }
}

fn consume_optional_tail(
    ts: &mut TokenStream,
    n: usize,
    m: usize,
    has_constraint_sections: bool,
) -> Result<(), QplibError> {
    let Some(default_primal_token) = ts.next_token() else {
        return Err(ts
            .take_io_err()
            .map(QplibError::IoError)
            .unwrap_or_else(|| {
                QplibError::ParseError(
                    "unexpected end of file (required initial-primal section is missing)"
                        .to_string(),
                )
            }));
    };
    let default_primal =
        parse_tail_real(&default_primal_token, "initial primal", ts.line_number())?;
    let _ = default_primal;
    ts.finish_record();
    consume_indexed_values(ts, n, "initial primal")?;

    // Table 8 note [2] omits constraint-indexed records for **N and **B.
    if has_constraint_sections {
        let value = ts.read_f64()?;
        require_finite(value, "initial constraint dual default", ts.line_number())?;
        ts.finish_record();
        consume_indexed_values(ts, m, "initial constraint dual")?;
    }

    let bound_default = ts.read_f64()?;
    require_finite(
        bound_default,
        "initial bound dual default",
        ts.line_number(),
    )?;
    ts.finish_record();
    consume_indexed_values(ts, n, "initial bound dual")?;
    consume_indexed_names(ts, n, "variable name")?;
    if has_constraint_sections {
        consume_indexed_names(ts, m, "constraint name")?;
    }

    match ts.next_token() {
        Some(token) => Err(QplibError::ParseError(format!(
            "line {}: unexpected record '{}' after QPLIB optional fields",
            ts.line_number(),
            token
        ))),
        None => ts
            .take_io_err()
            .map(QplibError::IoError)
            .map_or(Ok(()), Err),
    }
}

fn parse_tail_real(token: &str, context: &str, line: usize) -> Result<f64, QplibError> {
    let normalized = token.replace(['D', 'd'], "E");
    let value = normalized.parse::<f64>().map_err(|_| {
        QplibError::ParseError(format!(
            "line {line} ({context}): expected real number, got '{token}'"
        ))
    })?;
    require_finite(value, context, line)?;
    Ok(value)
}

fn consume_indexed_values(
    ts: &mut TokenStream,
    dimension: usize,
    context: &str,
) -> Result<usize, QplibError> {
    let count = ts.read_usize(&format!("number of non-default {context} values"))?;
    ts.finish_record();
    if count > dimension {
        return Err(QplibError::ParseError(format!(
            "number of non-default {context} values {count} exceeds dimension {dimension}"
        )));
    }
    let mut seen = HashSet::with_capacity(count);
    for _ in 0..count {
        let index = ts.read_index_1based(dimension, context)?;
        let value = ts.read_f64()?;
        require_finite(value, context, ts.line_number())?;
        ts.finish_record();
        if !seen.insert(index) {
            return Err(QplibError::ParseError(format!(
                "duplicate {context} index {}",
                index + 1
            )));
        }
    }
    Ok(count)
}

fn consume_indexed_names(
    ts: &mut TokenStream,
    dimension: usize,
    context: &str,
) -> Result<(), QplibError> {
    let count = ts.read_usize(&format!("number of {context}s"))?;
    ts.finish_record();
    consume_indexed_names_with_count(ts, dimension, context, count)
}

fn consume_indexed_names_with_count(
    ts: &mut TokenStream,
    dimension: usize,
    context: &str,
    count: usize,
) -> Result<(), QplibError> {
    if count > dimension {
        return Err(QplibError::ParseError(format!(
            "number of {context}s {count} exceeds dimension {dimension}"
        )));
    }
    let mut seen = HashSet::with_capacity(count);
    for _ in 0..count {
        let index = ts.read_index_1based(dimension, context)?;
        let name = ts.read_string()?;
        ts.finish_record();
        if name.is_empty() || !seen.insert(index) {
            return Err(QplibError::ParseError(format!(
                "invalid or duplicate {context} index {}",
                index + 1
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tail_tests {
    use super::*;
    use std::io::{self, BufRead, Read};

    #[test]
    fn required_records_ignore_plain_line_end_comments() {
        let input = "plain_comment_fixture descriptive title\nlcb problem type\nminimize objective sense\n1 variables\n0 quadratic terms\n0 objective default\n0 objective exceptions\n0 objective constant\n1e20 infinity marker\n0 lower default\n0 lower exceptions\n1 upper default\n0 upper exceptions\n0 initial primal\n0 primal count\n0 bound dual\n0 bound dual count\n0 variable names\n";
        assert!(parse_token_stream(TokenStream::from_str(input)).is_ok());
    }

    #[test]
    fn full_parser_accepts_official_n_and_b_omissions() {
        let n_problem =
            "official_n\nqcn\nmin\n1\n1\n1 1 2\n0\n0\n0\n1e20\n0\n0\n1\n0\n0\n0\n0\n0\n0\n";
        assert!(parse_token_stream(TokenStream::from_str(n_problem)).is_ok());

        let b_problem =
            "official_b\nqcb\nmin\n1\n1\n1 1 2\n0\n0\n0\n1e20\n0\n0\n1\n0\n0\n0\n0\n0\n0\n";
        assert!(parse_token_stream(TokenStream::from_str(b_problem)).is_ok());
    }

    #[test]
    fn full_parser_requires_constraint_tail_sections_for_l_and_q_even_when_m_is_zero() {
        let l_problem = "zero_l\nQCL\nmin\n1\n0\n0\n0\n0\n0\n0\n1e20\n0\n0\n0\n0\n0\n0\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n";
        let q_problem = "zero_q\nQCQ\nmin\n1\n0\n0\n0\n0\n0\n0\n0\n1e20\n0\n0\n0\n0\n0\n0\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n";
        parse_token_stream(TokenStream::from_str(l_problem)).unwrap();
        parse_token_stream(TokenStream::from_str(q_problem)).unwrap();

        let missing_constraint_names = l_problem.strip_suffix("0\n").unwrap();
        assert!(
            parse_token_stream(TokenStream::from_str(missing_constraint_names)).is_err(),
            "**L with m=0 must still contain the constraint-name count record"
        );
    }

    #[test]
    fn full_parser_accepts_official_n_with_zero_q_and_integer_default() {
        let official =
            "official_zero_q\nQCN\nmin\n1\n0\n1\n0\n0\n1e20\n0\n0\n1\n0\n0\n0\n0\n0\n0\n";
        assert!(parse_token_stream(TokenStream::from_str(official)).is_ok());
    }

    #[test]
    fn full_parser_rejects_nonstandard_n_extra_zero_sections() {
        let nonstandard = "extra_zero_n\nQCN\nmin\n1\n0\n1\n1 1 2\n0\n0\n0\n0\n1e20\n0\n0\n0\n0\n0\n0\n1\n0\n0\n0\n0\n0\n0\n0\n0\n0\n";
        assert!(parse_token_stream(TokenStream::from_str(nonstandard)).is_err());
    }

    #[test]
    fn full_parser_rejects_missing_or_truncated_required_tail() {
        let prefix = "missing_tail\nLCB\nmin\n1\n0\n0\n0\n0\n1e20\n0\n0\n1\n0\n";
        assert!(parse_token_stream(TokenStream::from_str(prefix)).is_err());
        let truncated = format!("{prefix}0\n0\n");
        assert!(parse_token_stream(TokenStream::from_str(&truncated)).is_err());
    }

    #[test]
    fn full_parser_string_and_reader_have_identical_record_semantics() {
        let input = "parity words\nLCB type\nmin sense\n1 vars\n0 q terms\n0 c\n0 c count\n0 q0\n1e20 inf\n0 lb\n0 lb count\n1 ub\n0 ub count\n0 x\n0 x count\n0 z\n0 z count\n0 names\n";
        assert!(parse_token_stream(TokenStream::from_str(input)).is_ok());
        assert!(parse_token_stream(TokenStream::from_reader(io::Cursor::new(
            input.as_bytes().to_vec()
        )))
        .is_ok());
    }

    #[test]
    fn optional_tail_accepts_comments_names_and_fortran_exponents() {
        let input = "0D0 initial point\n1 count\n1 2.5d0 primal comment\n0 constraint dual\n0 count\n0 bound dual\n0 count\n1 names\n1 x_one plain text\n1 constraint names\n1 row_one more text\n";
        assert!(consume_optional_tail(&mut TokenStream::from_str(input), 1, 1, true).is_ok());
    }

    #[test]
    fn optional_tail_accepts_official_m_zero_omission() {
        let official = "0\n0\n0\n0\n0\n";
        assert!(consume_optional_tail(&mut TokenStream::from_str(official), 2, 0, false).is_ok());
    }

    #[test]
    fn optional_tail_rejects_truncation_counts_indices_duplicates_names_and_garbage() {
        for invalid in [
            "0\n1\n1\n",
            "0\n2\n",
            "0\n1\n2 1\n",
            "0\n2\n1 1\n1 2\n",
            "0\n0\n0\n0\n1\n1\n",
            "0\n0\n0\n0\n0\ngarbage\n",
        ] {
            assert!(
                consume_optional_tail(&mut TokenStream::from_str(invalid), 1, 0, false).is_err(),
                "malformed tail accepted: {invalid:?}"
            );
        }
    }

    #[test]
    fn optional_tail_rejects_nonfinite_values() {
        for value in ["NaN", "inf", "1D9999"] {
            let input = format!("{value}\n0\n0\n0\n0\n");
            assert!(
                consume_optional_tail(&mut TokenStream::from_str(&input), 1, 0, false).is_err()
            );
        }
    }

    struct FailsAfterFirstLine {
        first: io::Cursor<Vec<u8>>,
        failed: bool,
    }

    struct FailsInsteadOfEof {
        input: io::Cursor<Vec<u8>>,
    }

    impl Read for FailsInsteadOfEof {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            unreachable!()
        }
    }

    impl BufRead for FailsInsteadOfEof {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.input.fill_buf()
        }

        fn consume(&mut self, amt: usize) {
            self.input.consume(amt);
        }

        fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
            let read = self.input.read_line(buf)?;
            if read == 0 {
                Err(io::Error::other("late full-parser read failed"))
            } else {
                Ok(read)
            }
        }
    }

    impl Read for FailsAfterFirstLine {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            unreachable!()
        }
    }

    impl BufRead for FailsAfterFirstLine {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.first.fill_buf()
        }

        fn consume(&mut self, amt: usize) {
            self.first.consume(amt);
        }

        fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
            if self.failed {
                Err(io::Error::other("tail read failed"))
            } else {
                self.failed = true;
                self.first.read_line(buf)
            }
        }
    }

    #[test]
    fn optional_tail_propagates_sticky_late_io_error() {
        let reader = FailsAfterFirstLine {
            first: io::Cursor::new(b"0\n".to_vec()),
            failed: false,
        };
        assert!(matches!(
            consume_optional_tail(&mut TokenStream::from_reader(reader), 1, 0, false),
            Err(QplibError::IoError(_))
        ));
    }

    #[test]
    fn full_parser_propagates_io_error_after_the_last_required_record() {
        let input = b"late_io\nLCB\nmin\n1\n0\n0\n0\n0\n1e20\n0\n0\n1\n0\n0\n0\n0\n0\n0\n".to_vec();
        assert!(matches!(
            parse_token_stream(TokenStream::from_reader(FailsInsteadOfEof {
                input: io::Cursor::new(input)
            })),
            Err(QplibError::IoError(_))
        ));
    }
}

pub(super) fn parse_token_stream(mut ts: TokenStream) -> Result<QplibProblem, QplibError> {
    // Problem name (skip)
    let _name = ts.read_string()?;
    ts.finish_record();

    // Problem type: 3 chars (Objective, Variables, Constraints)
    let prob_type = ts.read_string()?.to_ascii_uppercase();
    ts.finish_record();
    if prob_type.len() != 3 {
        return Err(QplibError::ParseError(format!(
            "Problem type must be 3 characters, got '{}'",
            prob_type
        )));
    }
    let type_bytes = prob_type.as_bytes();
    let obj_char = type_bytes[0] as char;
    let var_char = type_bytes[1] as char;
    let con_char = type_bytes[2] as char;

    // QPLIB objective type: L (linear), D (diagonal convex/concave quadratic),
    // C (convex/concave quadratic), Q (indefinite quadratic). Source:
    // qplib.zib.de PROBTYPE definition (Furini et al. 2019, §3.3).
    match obj_char {
        'L' | 'D' | 'C' | 'Q' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Objective type '{}' not supported (only L/D/C/Q supported). Type={}",
                c, prob_type
            )));
        }
    }

    let var_binary = var_char == 'B';
    let var_integer = var_char == 'I';
    match var_char {
        'C' | 'B' | 'I' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Variable type '{}' not supported (C/B/I supported; M/G/S mixed-integer unsupported). Type={}",
                c, prob_type
            )));
        }
    }

    match con_char {
        'L' | 'B' | 'N' | 'Q' => {}
        c => {
            return Err(QplibError::UnsupportedType(format!(
                "Constraint type '{}' not supported (only L/B/N/Q supported). Type={}",
                c, prob_type
            )));
        }
    }

    let objsense = ts.read_string()?.to_lowercase();
    ts.finish_record();
    let maximize = match objsense.as_str() {
        "minimize" | "min" => false,
        "maximize" | "max" => true,
        other => {
            return Err(QplibError::ParseError(format!(
                "OBJSENSE must be 'minimize'/'min' or 'maximize'/'max', got '{}'",
                other
            )));
        }
    };

    let n = ts.read_usize("number of variables")?;
    ts.finish_record();
    let m = match con_char {
        'L' | 'Q' => {
            let value = ts.read_usize("number of constraints")?;
            ts.finish_record();
            value
        }
        _ => 0, // 'B': no m field
    };

    // Objective quadratic terms (lower-triangular, symmetrized)
    let nqobj = ts.read_usize("number of objective quadratic terms")?;
    ts.finish_record();
    if nqobj > n.saturating_mul(n.saturating_add(1)) / 2 {
        return Err(QplibError::ParseError(format!(
            "nqobj {} exceeds n*(n+1)/2={} (n={})",
            nqobj,
            n.saturating_mul(n.saturating_add(1)) / 2,
            n
        )));
    }

    let mut q_triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(nqobj * 2);
    for _ in 0..nqobj {
        let i = ts.read_index_1based(n, "Q row")?;
        let j = ts.read_index_1based(n, "Q col")?;
        let v = ts.read_f64()?;
        require_finite(v, "objective quadratic value", ts.line_number())?;
        ts.finish_record();
        q_triplets.push((i, j, v));
        if i != j {
            q_triplets.push((j, i, v));
        }
    }

    // Objective linear terms
    let default_b0 = ts.read_f64()?;
    require_finite(
        default_b0,
        "default objective linear value",
        ts.line_number(),
    )?;
    ts.finish_record();
    let mut c = vec![default_b0; n];
    let n_nondefault_b0 = ts.read_usize("number of non-default objective linear terms")?;
    ts.finish_record();
    for _ in 0..n_nondefault_b0 {
        let i = ts.read_index_1based(n, "linear obj index")?;
        let v = ts.read_f64()?;
        require_finite(v, "objective linear value", ts.line_number())?;
        ts.finish_record();
        c[i] = v;
    }

    let q0 = ts.read_f64()?;
    ts.finish_record();
    if !q0.is_finite() {
        return Err(QplibError::ParseError(format!(
            "objective constant q0 is not finite: {}",
            q0
        )));
    }

    // Constraint quadratic terms (QCQ only)
    let mut con_q_triplets: Vec<Vec<(usize, usize, f64)>> = if con_char == 'Q' {
        vec![vec![]; m]
    } else {
        vec![]
    };
    if con_char == 'Q' {
        let n_con_quad_terms = ts.read_usize("number of constraint quadratic terms")?;
        ts.finish_record();
        for _ in 0..n_con_quad_terms {
            let k = ts.read_index_1based(m, "constraint quad index")?;
            let i = ts.read_index_1based(n, "constraint quad row")?;
            let j = ts.read_index_1based(n, "constraint quad col")?;
            let v = ts.read_f64()?;
            require_finite(v, "constraint quadratic value", ts.line_number())?;
            ts.finish_record();
            con_q_triplets[k].push((i, j, v));
            if i != j {
                con_q_triplets[k].push((j, i, v));
            }
        }
    }

    // Constraint linear terms (L/N/Q types)
    let mut a_triplets: Vec<(usize, usize, f64)> = Vec::new();
    if matches!(con_char, 'L' | 'Q') {
        let n_con_lin_terms = ts.read_usize("number of constraint linear terms")?;
        ts.finish_record();
        if n_con_lin_terms > n.saturating_mul(m) {
            return Err(QplibError::ParseError(format!(
                "n_con_lin_terms {} exceeds n*m={}",
                n_con_lin_terms,
                n.saturating_mul(m)
            )));
        }
        a_triplets = Vec::with_capacity(n_con_lin_terms);
        for _ in 0..n_con_lin_terms {
            let k = ts.read_index_1based(m, "constraint index")?;
            let i = ts.read_index_1based(n, "variable index")?;
            let v = ts.read_f64()?;
            require_finite(v, "constraint linear value", ts.line_number())?;
            ts.finish_record();
            a_triplets.push((k, i, v));
        }
    }
    // Infinity value. Must be strictly positive (rejects zero, negative, and
    // NaN, all of which corrupt the `is_pos_inf`/`is_neg_inf` scale below).
    // `+inf` itself is a legitimate declared-infinity sentinel: many QPLIB
    // files (and this module's own fixtures) write DBL_MAX with fewer than
    // 17 significant digits (e.g. `1.79769313486232E+308`), which overflows
    // to `f64::INFINITY` on parse; that must continue to parse successfully.
    let inf_val = ts.read_f64()?;
    ts.finish_record();
    if inf_val.is_nan() || inf_val <= 0.0 {
        return Err(QplibError::ParseError(format!(
            "inf_val must be positive, got {}",
            inf_val
        )));
    }
    let is_pos_inf = |x: f64| x >= inf_val * QPLIB_INF_REL_TOL;
    let is_neg_inf = |x: f64| x <= -inf_val * QPLIB_INF_REL_TOL;

    // Constraint bounds (L/N/Q types)
    let mut lb_con = vec![f64::NEG_INFINITY; m];
    let mut ub_con = vec![f64::INFINITY; m];
    if matches!(con_char, 'L' | 'Q') {
        let lb_con_default = ts.read_f64()?;
        require_finite_or_infinite_marker(lb_con_default, "constraint lower bound")?;
        ts.finish_record();
        let n_nondefault_lb_con = ts.read_usize("number of non-default constraint lower bounds")?;
        ts.finish_record();
        lb_con = vec![lb_con_default; m];
        for _ in 0..n_nondefault_lb_con {
            let k = ts.read_index_1based(m, "lb_con index")?;
            let v = ts.read_f64()?;
            require_finite_or_infinite_marker(v, "constraint lower bound")?;
            ts.finish_record();
            lb_con[k] = v;
        }

        let ub_con_default = ts.read_f64()?;
        require_finite_or_infinite_marker(ub_con_default, "constraint upper bound")?;
        ts.finish_record();
        let n_nondefault_ub_con = ts.read_usize("number of non-default constraint upper bounds")?;
        ts.finish_record();
        ub_con = vec![ub_con_default; m];
        for _ in 0..n_nondefault_ub_con {
            let k = ts.read_index_1based(m, "ub_con index")?;
            let v = ts.read_f64()?;
            require_finite_or_infinite_marker(v, "constraint upper bound")?;
            ts.finish_record();
            ub_con[k] = v;
        }
    }
    // Variable bounds
    // Binary ('B'): implicit [0,1]; Continuous/Integer: explicit in file.
    let (lb_var, ub_var) = if var_binary {
        (vec![0.0_f64; n], vec![1.0_f64; n])
    } else {
        let lb_var_default = ts.read_f64()?;
        require_finite_or_infinite_marker(lb_var_default, "variable lower bound")?;
        ts.finish_record();
        let n_nondefault_lb_var = ts.read_usize("number of non-default variable lower bounds")?;
        ts.finish_record();
        let mut lb_var = vec![lb_var_default; n];
        for _ in 0..n_nondefault_lb_var {
            let i = ts.read_index_1based(n, "lb_var index")?;
            let v = ts.read_f64()?;
            require_finite_or_infinite_marker(v, "variable lower bound")?;
            ts.finish_record();
            lb_var[i] = v;
        }
        let ub_var_default = ts.read_f64()?;
        require_finite_or_infinite_marker(ub_var_default, "variable upper bound")?;
        ts.finish_record();
        let n_nondefault_ub_var = ts.read_usize("number of non-default variable upper bounds")?;
        ts.finish_record();
        let mut ub_var = vec![ub_var_default; n];
        for _ in 0..n_nondefault_ub_var {
            let i = ts.read_index_1based(n, "ub_var index")?;
            let v = ts.read_f64()?;
            require_finite_or_infinite_marker(v, "variable upper bound")?;
            ts.finish_record();
            ub_var[i] = v;
        }
        (lb_var, ub_var)
    };

    consume_optional_tail(&mut ts, n, m, matches!(con_char, 'L' | 'Q'))?;

    // ── Build QpProblem ───────────────────────────────────────────────────────

    let sign = if maximize { -1.0 } else { 1.0 };

    let q = {
        let q_rows: Vec<usize> = q_triplets.iter().map(|&(r, _, _)| r).collect();
        let q_cols: Vec<usize> = q_triplets.iter().map(|&(_, c, _)| c).collect();
        let q_vals: Vec<f64> = q_triplets.iter().map(|&(_, _, v)| sign * v).collect();
        drop(q_triplets);
        if q_rows.is_empty() {
            CscMatrix::new(n, n)
        } else {
            CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n)
                .map_err(|e| QplibError::ParseError(format!("Q matrix error: {}", e)))?
        }
    };

    if maximize {
        for v in &mut c {
            *v = -*v;
        }
    }

    // Expand lb_con[k] <= a[k]^T x <= ub_con[k] → Ax <= b rows.
    let mut aug_ub_row: Vec<Option<usize>> = vec![None; m];
    let mut aug_lb_row: Vec<Option<usize>> = vec![None; m];
    let mut b_vec: Vec<f64> = Vec::new();
    let mut constraint_types: Vec<ConstraintType> = Vec::new();

    for k in 0..m {
        let lb = lb_con[k];
        let ub = ub_con[k];
        if !is_pos_inf(ub) && !is_neg_inf(lb) && (lb - ub).abs() < 1e-15 {
            // Equality constraint: store as single Eq row.
            aug_ub_row[k] = Some(b_vec.len());
            b_vec.push(ub);
            constraint_types.push(ConstraintType::Eq);
        } else {
            if !is_pos_inf(ub) {
                aug_ub_row[k] = Some(b_vec.len());
                b_vec.push(ub);
                constraint_types.push(ConstraintType::Le);
            }
            if !is_neg_inf(lb) {
                aug_lb_row[k] = Some(b_vec.len());
                b_vec.push(-lb);
                constraint_types.push(ConstraintType::Le);
            }
        }
    }

    let m_aug = b_vec.len();

    let a_mat = {
        let cap = a_triplets.len();
        let mut a_rows: Vec<usize> = Vec::with_capacity(cap);
        let mut a_cols: Vec<usize> = Vec::with_capacity(cap);
        let mut a_vals: Vec<f64> = Vec::with_capacity(cap);

        for &(con_idx, var_idx, val) in &a_triplets {
            if let Some(aug_row) = aug_ub_row[con_idx] {
                a_rows.push(aug_row);
                a_cols.push(var_idx);
                a_vals.push(val);
            }
            if let Some(aug_row) = aug_lb_row[con_idx] {
                a_rows.push(aug_row);
                a_cols.push(var_idx);
                a_vals.push(-val);
            }
        }
        drop(a_triplets);

        if a_rows.is_empty() {
            CscMatrix::new(m_aug, n)
        } else {
            CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m_aug, n)
                .map_err(|e| QplibError::ParseError(format!("A matrix error: {}", e)))?
        }
    };

    let bounds: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let lb = if is_neg_inf(lb_var[i]) {
                f64::NEG_INFINITY
            } else {
                lb_var[i]
            };
            let ub = if is_pos_inf(ub_var[i]) {
                f64::INFINITY
            } else {
                ub_var[i]
            };
            (lb, ub)
        })
        .collect();

    // Quadratic constraint matrices (QCQP only).
    // Stored as QcqpMatrix (COO triplets) to avoid O(n) col_ptr per constraint.
    let quadratic_constraints = if con_char == 'Q' {
        let mut qc: Vec<QcqpMatrix> = vec![QcqpMatrix::new(n); m_aug];
        for k in 0..m {
            let trips = &con_q_triplets[k];
            if trips.is_empty() {
                continue;
            }
            if let Some(aug_row) = aug_ub_row[k] {
                qc[aug_row].triplets = trips.clone();
            }
            if let Some(aug_row) = aug_lb_row[k] {
                qc[aug_row].triplets = trips.iter().map(|&(r, c, v)| (r, c, -v)).collect();
            }
        }
        qc
    } else {
        vec![]
    };

    let q0_offset = if maximize { -q0 } else { q0 };

    let mut prob = QpProblem::new(q, c, a_mat, b_vec, bounds, constraint_types)
        .map_err(|e| QplibError::ParseError(e.to_string()))?;
    prob.quadratic_constraints = quadratic_constraints;
    prob.obj_offset = q0_offset;

    if var_binary || var_integer {
        let integer_vars: Vec<usize> = (0..n).collect();
        // A zero objective Q with quadratic *constraints* is still a QCQP:
        // route it to Miqp (which carries `quadratic_constraints`) rather than
        // Milp, which would silently drop them.
        let has_quad_constraints = prob.quadratic_constraints.iter().any(|qc| qc.nnz() > 0);
        if prob.q.nnz() == 0 && !has_quad_constraints {
            let mut lp = LpProblem::new_general(
                prob.c,
                prob.a,
                prob.b,
                prob.constraint_types,
                prob.bounds,
                None,
            )
            .map_err(|e: otspot_core::error::SolverError| QplibError::ParseError(e.to_string()))?;
            lp.obj_offset = q0_offset;
            let milp = MilpProblem::new(lp, integer_vars).map_err(
                |e: otspot_core::mip::MipProblemError| QplibError::ParseError(e.to_string()),
            )?;
            Ok(QplibProblem::Milp(milp))
        } else {
            let miqp = MiqpProblem::new(prob, integer_vars).map_err(
                |e: otspot_core::mip::MipProblemError| QplibError::ParseError(e.to_string()),
            )?;
            Ok(QplibProblem::Miqp(miqp))
        }
    } else {
        Ok(QplibProblem::Qp(prob))
    }
}
