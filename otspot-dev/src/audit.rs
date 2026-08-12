//! Workspace-wide code quality audit tests.
//!
//! Two automated gates that catch AI blind spots:
//!
//! - `no_eprintln_in_production`: production source must not contain direct print
//!   macros (`println!`, `eprintln!`, `print!`, `eprint!`) outside test contexts.
//! - `no_observation_only_tests`: every `#[test]` / `#[tokio::test]` function must
//!   contain at least one assertion-like expression.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use walkdir::WalkDir;

    // ── helpers ────────────────────────────────────────────────────────────────

    fn workspace_root() -> PathBuf {
        // CARGO_MANIFEST_DIR = otspot-dev/
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    /// Walks `line` for `{`/`}` outside `//`-comments and string literals,
    /// updating `*depth` and popping every stack in `stacks` immediately
    /// after each individual `}` — in true left-to-right character order,
    /// rather than by applying a line's net `opens - closes` in one atomic
    /// step and popping once against the line-final depth.
    ///
    /// Scanners that track skip/context zones with a stack of entry-floor
    /// depths (`scan_dead_params`, `scan_underscore_sig_params`) only ever
    /// check zone membership using the depth *as of the end of a line*, so
    /// per-character precision within a line is not needed for that check —
    /// but the atomic-net shortcut still corrupts the zone *stack itself*:
    /// a line that closes a zone and then opens an unrelated block (e.g.
    /// `} fn wrapper() {`) nets to a depth equal to what it was before the
    /// line, so the pop that should happen when depth transiently touches
    /// the zone's floor never fires, leaving a stale entry that keeps
    /// misclassifying every following line as "still in the zone" until
    /// some later, unrelated `}` happens to trigger the pop instead.
    /// Popping immediately after each `}` (this function) cannot miss that
    /// transient floor-touch, because depth is checked at every point it
    /// actually decreases, not just once per line after any later same-line
    /// re-opens have already raised it back up.
    ///
    /// Returns `(opens, closes, in_str_out)` for the line, mirroring the
    /// `(opens, closes, in_str_out)` shape other scanners in this module use.
    fn advance_depth_pop_stacks(
        line: &str,
        in_str_in: bool,
        depth: &mut i32,
        stacks: &mut [&mut Vec<i32>],
    ) -> (i32, i32, bool) {
        let mut opens = 0i32;
        let mut closes = 0i32;
        let mut in_str = in_str_in;
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            let prev = if i > 0 { chars[i - 1] } else { '\0' };
            if !in_str && ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                break;
            }
            if ch == '"' && prev != '\\' {
                in_str = !in_str;
            }
            if !in_str {
                if ch == '{' {
                    opens += 1;
                    *depth += 1;
                } else if ch == '}' {
                    closes += 1;
                    *depth -= 1;
                    for stack in stacks.iter_mut() {
                        while stack.last().is_some_and(|&d| *depth <= d) {
                            stack.pop();
                        }
                    }
                }
            }
            i += 1;
        }
        (opens, closes, in_str)
    }

    /// Scans `content` for direct print macro calls that are not inside a
    /// test context (i.e., outside `#[cfg(test)]` blocks, `mod tests`/`mod test`
    /// blocks, and `#[test]`/`#[tokio::test]` function bodies).
    ///
    /// Position-aware: each line's braces are walked left-to-right and `depth`
    /// is updated one brace at a time (rather than by applying a line's net
    /// `opens - closes` atomically, or by checking skip-zone membership
    /// against a single per-line depth value). A print macro's skip-zone
    /// membership is decided using the depth *at the macro's own character
    /// position*. This structurally rules out both failure modes that a
    /// line-level depth check is prone to:
    /// - False positive: a same-line balanced pair, e.g.
    ///   `if b { '1' } else { '0' }` (net 0), can no longer spuriously pop an
    ///   enclosing skip zone via an intermediate closes-only depth, because
    ///   depth only ever decreases one `}` at a time and the pop check runs
    ///   after each individual close.
    /// - False negative: a line that closes a skip zone and then opens an
    ///   unrelated block, e.g. `} println!("x"); if true {`, is no longer
    ///   treated as "inside the zone" for its entire length just because the
    ///   zone-close and the new block happen to net to the same line-final
    ///   depth — the `println!` is checked at the depth that holds strictly
    ///   between the `}` and the following `{`.
    ///
    /// Returns `(1-based line number, macro name)` for each violation.
    fn scan_production_prints(content: &str) -> Vec<(usize, String)> {
        const PRINT_MACROS: &[&str] = &["eprintln!(", "println!(", "eprint!(", "print!("];

        let mut violations = Vec::new();
        let mut depth: i32 = 0;
        let mut in_str_state = false; // tracks multi-line string literals

        // Stack of brace depths at which we entered a skip zone.
        // We are "in skip" while `depth > *skip_at_depth.last()`.
        let mut skip_stack: Vec<i32> = Vec::new();

        // Pending flags: set when an attribute is seen; cleared once we open a block.
        let mut pending_cfg_test = false;
        let mut pending_test_attr = false; // #[test] / #[tokio::test]
                                           // Stack of depths at which #[allow(clippy::print_stderr/stdout)] was seen.
                                           // Print macros at depth >= allow_stack.last() are exempted.
        let mut allow_print_stack: Vec<i32> = Vec::new();

        let macro_chars: Vec<Vec<char>> =
            PRINT_MACROS.iter().map(|m| m.chars().collect()).collect();

        for (line_idx, raw_line) in content.lines().enumerate() {
            let line_no = line_idx + 1;
            let trimmed = raw_line.trim();

            // Fast-path skip for pure line comments. Only valid when we are
            // NOT already inside a multi-line string entering this line: a
            // string opened on a previous line can close mid-line here (e.g.
            // the tail of `"start\n// end of string";`), and text that merely
            // *looks* like a `//` comment at the trimmed start is then really
            // string content followed by real code. Early-continuing in that
            // case would skip the char walk below entirely, so the closing
            // quote's toggle is never applied and `in_str_state` gets stuck
            // `true` for the rest of the file.
            if !in_str_state && trimmed.starts_with("//") {
                continue;
            }

            // Detect attributes that trigger skip zones. None of these carry
            // braces themselves, so evaluating them against the depth carried
            // in from the previous line (before this line's brace walk) is
            // equivalent to evaluating them after, and keeps the trigger
            // scoped to whatever block follows.
            if trimmed.contains("#[cfg(test)]") {
                pending_cfg_test = true;
            }
            if trimmed.contains("#[test]") || trimmed.contains("#[tokio::test]") {
                pending_test_attr = true;
            }
            // #[allow(clippy::print_stderr/stdout)] exempts prints at deeper depth.
            if trimmed.contains("#[allow(clippy::print_stderr")
                || trimmed.contains("#[allow(clippy::print_stdout")
            {
                allow_print_stack.push(depth);
            }

            // `mod tests {` or `mod test {` (including pub variants) — the
            // opening brace must be on this same physical line to be
            // attributed to this trigger (mirrors the cfg(test)/test-attr
            // triggers below, which only consume the *first* `{` seen).
            let is_mod_tests_start = trimmed.starts_with("mod tests")
                || trimmed.starts_with("mod test ")
                || trimmed.starts_with("pub mod tests")
                || trimmed.starts_with("pub(crate) mod tests");

            // Consumed by the first `{` encountered while scanning this line.
            let mut pending_enter = pending_cfg_test || pending_test_attr || is_mod_tests_start;

            let chars: Vec<char> = raw_line.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                let ch = chars[i];
                let prev = if i > 0 { chars[i - 1] } else { '\0' };
                let mut step = 1usize;

                if !in_str_state && ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                    break; // rest of line is a comment
                }
                if ch == '"' && prev != '\\' {
                    in_str_state = !in_str_state;
                }

                if !in_str_state {
                    if ch == '{' {
                        depth += 1;
                        if pending_enter {
                            // depth already includes this brace; the zone's
                            // floor is the depth just before it.
                            skip_stack.push(depth - 1);
                            pending_enter = false;
                            pending_cfg_test = false;
                            pending_test_attr = false;
                        }
                    } else if ch == '}' {
                        depth -= 1;
                        while skip_stack.last().is_some_and(|&d| depth <= d) {
                            skip_stack.pop();
                        }
                        while allow_print_stack.last().is_some_and(|&d| depth < d) {
                            allow_print_stack.pop();
                        }
                    }
                }

                // Macro detection intentionally does NOT gate on
                // `in_str_state`: a wrong string-state guess would otherwise
                // be a single point of failure permanently disabling print
                // detection for the rest of the file (e.g. a raw string's
                // internal `"` desyncing the naive toggle), unlike the old
                // plain-substring-search implementation, which never
                // consulted string state. Trade-off: a possible false
                // positive for print-lookalike text genuinely inside a
                // string, same trade-off the old implementation made.
                if ch != '{' && ch != '}' {
                    for (macro_idx, mchars) in macro_chars.iter().enumerate() {
                        let end = i + mchars.len();
                        if end <= chars.len() && chars[i..end] == mchars[..] {
                            let in_skip = skip_stack.last().is_some_and(|&d| depth > d);
                            let print_is_allowed =
                                allow_print_stack.last().is_some_and(|&d| depth >= d);
                            if !in_skip && !print_is_allowed {
                                let macro_name = PRINT_MACROS[macro_idx];
                                violations.push((
                                    line_no,
                                    macro_name.trim_end_matches('(').to_string() + "!",
                                ));
                            }
                            // Skip past the whole matched token so a
                            // shorter macro name that happens to be a
                            // substring at a later offset (e.g. "println!("
                            // starting one char into "eprintln!(") is not
                            // rechecked and double-reported.
                            step = mchars.len();
                            break;
                        }
                    }
                }
                i += step;
            }
        }

        violations
    }

    /// Returns `(1-based test-attribute line, function name)` for `#[test]`
    /// functions whose body contains no assertion-like expression.
    fn scan_observation_only_tests(content: &str) -> Vec<(usize, String)> {
        const ASSERTION_PATTERNS: &[&str] = &[
            "assert!(",
            "assert_eq!(",
            "assert_ne!(",
            "assert_matches!(",
            "assert_relative_eq!(",
            "assert_abs_diff_eq!(",
            "panic!(",
            "unreachable!(",
            "#[should_panic",
            // Assertion-helper function calls (Rust convention: assert_*, check_*)
            "assert_",
            "check_",
            // proptest macros provide their own assertion framework
            "proptest!",
            "prop_assert",
            // .expect() is an assertion (panics on Err)
            ".expect(",
            // Common assertion-wrapping test helpers in this workspace
            "solve_and_check(",
            "diagnose(",
            "solve_with_watchdog(",
        ];

        let mut violations = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let n = lines.len();
        let mut i = 0;

        while i < n {
            let trimmed = lines[i].trim();

            // Skip comment lines (including doc comments containing "#[test]")
            if trimmed.starts_with("//") {
                i += 1;
                continue;
            }

            let is_test_attr = trimmed.contains("#[test]") || trimmed.contains("#[tokio::test]");
            // Match both `#[should_panic]` and `#[should_panic(expected = "...")]`.
            let has_should_panic = trimmed.contains("#[should_panic");

            if !is_test_attr && !has_should_panic {
                i += 1;
                continue;
            }

            let attr_line_no = i + 1;
            let mut should_panic_seen = has_should_panic;

            // Scan forward up to ~10 lines for additional attrs + `fn` signature
            let mut fn_line_idx = None;
            let window = (i + 10).min(n) - (i + 1);
            for (j, t) in lines.iter().enumerate().skip(i + 1).take(window) {
                let t = t.trim();
                // Skip comment lines so `// #[should_panic(...)]` is not mistaken
                // for the real attribute.
                if t.starts_with("//") {
                    continue;
                }
                if t.contains("#[should_panic") {
                    should_panic_seen = true;
                }
                if t.starts_with("fn ") || t.starts_with("async fn ") || t.starts_with("pub fn ") {
                    fn_line_idx = Some(j);
                    break;
                }
            }

            let fn_idx = match fn_line_idx {
                Some(idx) => idx,
                None => {
                    i += 1;
                    continue;
                }
            };

            // Extract function name
            let fn_name =
                extract_fn_name(lines[fn_idx]).unwrap_or_else(|| format!("<line_{}>", fn_idx + 1));

            // should_panic functions are always valid
            if should_panic_seen {
                i = fn_idx + 1;
                continue;
            }

            // Extract the body using brace matching. Braces are walked
            // left-to-right and `depth` is checked immediately after each
            // individual `}`, rather than by applying a line's net
            // `opens - closes` atomically and checking `depth <= 0` once at
            // line-end: the atomic form cannot observe the function's true
            // closing brace when that same line re-opens an unrelated block
            // afterward (e.g. `} fn helper() {`, net 0) — it would silently
            // absorb the unrelated code (and any later `#[test]` function
            // inside it) into this function's "body" until some later,
            // unrelated `}` happens to bring the net back to zero.
            let mut depth = 0i32;
            let mut body = String::new();
            let mut started = false;
            let mut end_idx = fn_idx;

            let mut body_in_str = false;
            'body_scan: for (k, line) in lines.iter().enumerate().skip(fn_idx) {
                let chars: Vec<char> = line.chars().collect();
                let mut ci = 0;
                let mut closed_here = false;
                while ci < chars.len() {
                    let ch = chars[ci];
                    let prev = if ci > 0 { chars[ci - 1] } else { '\0' };
                    if !body_in_str && ch == '/' && ci + 1 < chars.len() && chars[ci + 1] == '/' {
                        break;
                    }
                    if ch == '"' && prev != '\\' {
                        body_in_str = !body_in_str;
                    }
                    if !body_in_str {
                        if ch == '{' {
                            depth += 1;
                            started = true;
                        } else if ch == '}' {
                            depth -= 1;
                            if started && depth <= 0 {
                                closed_here = true;
                                break;
                            }
                        }
                    }
                    ci += 1;
                }
                if started {
                    body.push_str(line);
                    body.push('\n');
                }
                if closed_here {
                    end_idx = k;
                    break 'body_scan;
                }
            }

            // Check body for assertions
            let has_assertion = ASSERTION_PATTERNS.iter().any(|pat| body.contains(pat))
                || body.lines().any(|l| {
                    // Result-returning tests: lines ending with `?` or containing `Err(`
                    let t = l.trim();
                    t.ends_with('?') || t.ends_with("?;") || t.contains("Err(")
                });

            if !has_assertion {
                violations.push((attr_line_no, fn_name));
            }

            i = end_idx + 1;
        }

        violations
    }

    fn extract_fn_name(line: &str) -> Option<String> {
        // matches: `fn foo(`, `async fn foo(`, `pub fn foo(`
        let after_fn = line.find("fn ")?.checked_add(3)?;
        let rest = &line[after_fn..];
        let paren = rest.find('(')?;
        let name = rest[..paren].trim().to_string();
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    // ── Layer 1: nextest audit — production eprintln ────────────────────────

    /// Verifies that production source files (`otspot-core/src/`, `otspot-io/src/`,
    /// `otspot-model/src/`, `src/`) contain no direct `println!`, `eprintln!`,
    /// `print!`, or `eprint!` calls outside test contexts.
    ///
    /// Excludes: `#[cfg(test)]` blocks, `mod tests` blocks, `#[test]` function
    /// bodies, and `src/bin/` paths.
    #[test]
    fn no_eprintln_in_production() {
        let root = workspace_root();
        let prod_dirs = [
            root.join("otspot-core/src"),
            root.join("otspot-io/src"),
            root.join("otspot-model/src"),
            root.join("src"),
        ];

        let mut all_violations: Vec<(PathBuf, usize, String)> = Vec::new();

        for dir in &prod_dirs {
            for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path().to_path_buf();
                if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                    continue;
                }
                // Skip bin/ directories
                if path.components().any(|c| c.as_os_str() == "bin") {
                    continue;
                }
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let file_violations = scan_production_prints(&content);
                for (line_no, macro_name) in file_violations {
                    all_violations.push((path.clone(), line_no, macro_name));
                }
            }
        }

        if !all_violations.is_empty() {
            let mut msg = String::from(
                "production print macros detected (remove or move to test context):\n",
            );
            for (path, line_no, mac) in &all_violations {
                msg.push_str(&format!("  {}:{}: {}\n", path.display(), line_no, mac));
            }
            panic!("{msg}");
        }
    }

    /// Sentinel: a line with balanced same-line braces inside a
    /// `#[cfg(test)] mod tests` block (e.g. `if b { '1' } else { '0' }`) must
    /// not spuriously exit the skip zone. Before the fix, `scan_production_prints`
    /// applied `closes` and `opens` as two separate steps and popped the
    /// skip-stack against the intermediate (closes-only) depth, so a matched
    /// `{...} {...}` pair on one line dropped depth below the zone's entry
    /// floor even though the line's net effect on nesting is zero — causing a
    /// `println!` on the *next* line to be misreported as a production print.
    #[test]
    fn scan_production_prints_ignores_balanced_braces_before_println_in_test_mod() {
        let content = r#"
#[cfg(test)]
mod tests {
    #[test]
    fn probe() {
        let encoded: String = v.iter().map(|&b| if b { '1' } else { '0' }).collect();
        println!("RESULT={encoded}");
    }
}
"#;
        let violations = scan_production_prints(content);
        assert!(
            violations.is_empty(),
            "BUG: balanced same-line braces before println! inside mod tests \
             falsely exited the skip zone; violations: {:?}",
            violations
        );
    }

    /// Sentinel: a print macro that appears on the *same physical line* as the
    /// closing brace of a skip zone — immediately after that brace — must
    /// still be detected. Before the position-aware fix, `scan_production_prints`
    /// judged skip-zone membership using a single depth value per line (either
    /// the line's post-net depth, or the aggregate opens/closes count). A line
    /// like `} println!("LEAK"); if true {` nets to zero (one close, one open),
    /// so the line-final depth equals the pre-line depth and the skip zone
    /// appeared to still be active for the *entire* line, hiding the
    /// `println!` that actually sits strictly between the zone-closing `}`
    /// and the following unrelated `{`.
    #[test]
    fn scan_production_prints_detects_print_right_after_skip_zone_closes_same_line() {
        let content = r#"
#[cfg(test)]
mod tests {
    fn helper() {}
} println!("LEAK"); if true {
    let x = 1;
}

fn prod2() {
    println!("AFTER");
}
"#;
        let violations = scan_production_prints(content);
        let lines: Vec<usize> = violations.iter().map(|(l, _)| *l).collect();
        assert!(
            lines.contains(&5),
            "BUG: println! immediately after a same-line skip-zone close was \
             missed; violations: {:?}",
            violations
        );
        assert!(
            lines.contains(&10),
            "println! in prod2 must still be detected; violations: {:?}",
            violations
        );
        assert_eq!(
            violations.len(),
            2,
            "expected exactly the two production prints (line 5 and line 10); violations: {:?}",
            violations
        );
    }

    /// Sentinel (P1 fix, reviewer-reported): a multi-line string whose
    /// closing line's trimmed text happens to start with `//` (because the
    /// string's own content looks like a comment, e.g. the tail of
    /// `"start\n// end of string";`) must not disable print detection for
    /// the rest of the file. Before the fix, `if trimmed.starts_with("//") {
    /// continue; }` ran unconditionally, skipping that line's entire char
    /// walk — so the closing quote's `in_str_state` toggle never happened,
    /// leaving `in_str_state` stuck `true`. Because macro detection was
    /// gated on `!in_str_state`, every print macro for the rest of the file
    /// (not just on the desynced line) then went undetected — a strictly
    /// worse outcome than the old plain-substring-search implementation,
    /// which never consulted `in_str_state` for macro detection at all.
    #[test]
    fn scan_production_prints_survives_multiline_string_closing_on_comment_lookalike_line() {
        // Built with an escaped (non-raw) literal, not `r#"..."#`: a raw
        // string reproducing this exact quote pattern inline in *this*
        // source file would itself confuse `scan_observation_only_tests`'s
        // (unrelated, pre-existing, not fixed here) raw-string handling when
        // it self-scans audit.rs, corrupting extraction of *this test's own*
        // body. Every inner `"` here is escaped so this file's own scanners
        // see a single balanced string.
        let content = "\nfn prod() {\n    let s = \"start\n// end of string\"; \
                        println!(\"REAL_ON_COMMENT_LOOKALIKE_LINE\");\n}\nfn prod2() {\n    \
                        println!(\"AFTER_DESYNC_CHECK\");\n}\n";
        let violations = scan_production_prints(content);
        let lines: Vec<usize> = violations.iter().map(|(l, _)| *l).collect();
        assert!(
            lines.contains(&4),
            "BUG: print! right after a string closes on a comment-lookalike \
             line was missed; violations: {:?}",
            violations
        );
        assert!(
            lines.contains(&7),
            "BUG: print detection was permanently disabled for the rest of \
             the file after the string-closing desync; violations: {:?}",
            violations
        );
    }

    // ── Layer B: nextest audit — observation-only tests ─────────────────────

    /// Verifies that every `#[test]` / `#[tokio::test]` function in the workspace
    /// contains at least one assertion-like expression (`assert!`, `assert_eq!`,
    /// `panic!`, `?`, etc.).
    ///
    /// Covers: `tests/*.rs` (integration), and inline `#[cfg(test)] mod tests`
    /// blocks inside `src/**/*.rs` for all crates.
    #[test]
    fn no_observation_only_tests() {
        let root = workspace_root();

        // Integration test files (root crate)
        let integration_dirs = [root.join("tests")];

        // Source dirs with potential inline tests
        let src_dirs = [
            root.join("src"),
            root.join("otspot-core/src"),
            root.join("otspot-io/src"),
            root.join("otspot-model/src"),
            root.join("otspot-dev/src"),
        ];

        let mut all_violations: Vec<(PathBuf, usize, String)> = Vec::new();

        for dir in integration_dirs.iter().chain(src_dirs.iter()) {
            for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path().to_path_buf();
                if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                    continue;
                }
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                // Only scan files that actually contain a test attribute
                if !content.contains("#[test]") && !content.contains("#[tokio::test]") {
                    continue;
                }
                let file_violations = scan_observation_only_tests(&content);
                for (line_no, fn_name) in file_violations {
                    all_violations.push((path.clone(), line_no, fn_name));
                }
            }
        }

        if !all_violations.is_empty() {
            let mut msg =
                String::from("observation-only tests detected (add assertions or delete):\n");
            for (path, line_no, name) in &all_violations {
                msg.push_str(&format!("  {}:{}: fn {}\n", path.display(), line_no, name));
            }
            panic!("{msg}");
        }
    }

    // ── Layer C: dead-param detector ────────────────────────────────────────

    /// Returns `true` when `ident` appears as a standalone word in `text`
    /// at a location that is NOT a `let _ = ident;` discard statement.
    fn ident_used_elsewhere(text: &str, ident: &str) -> bool {
        // The dead-discard prefix is exactly "let _ = " (8 bytes).
        const DEAD_PREFIX: &str = "let _ = ";
        let mut search = text;
        while let Some(pos) = search.find(ident) {
            let before = if pos == 0 {
                b' '
            } else {
                search.as_bytes()[pos - 1]
            };
            let after_pos = pos + ident.len();
            let after = search.as_bytes().get(after_pos).copied().unwrap_or(b' ');
            let is_word = !before.is_ascii_alphanumeric()
                && before != b'_'
                && !after.is_ascii_alphanumeric()
                && after != b'_';
            if is_word {
                // True dead-discard: exactly "let _ = <ident>" immediately before.
                let is_dead_let = pos >= DEAD_PREFIX.len()
                    && &search[pos - DEAD_PREFIX.len()..pos] == DEAD_PREFIX;
                if !is_dead_let {
                    return true;
                }
            }
            search = &search[pos + 1..];
        }
        false
    }

    /// Returns `(1-based line number, parameter name)` for every `let _ = name;`
    /// where `name` is an explicit named parameter of the immediately enclosing
    /// function AND `name` is not used anywhere else in that function.
    ///
    /// Skips test-context blocks (`mod tests`, `#[cfg(test)]`, `#[test]` bodies).
    fn scan_dead_params(content: &str) -> Vec<(usize, String)> {
        let mut violations = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Track test/cfg(test) skip zones identically to scan_production_prints.
        let mut depth: i32 = 0;
        let mut in_str_state = false;
        let mut skip_stack: Vec<i32> = Vec::new();
        let mut pending_cfg_test = false;
        let mut pending_test_attr = false;

        // `entering_in_str` for every line, recorded as the main loop below
        // walks forward — so the backward "find the enclosing fn" search
        // further down can look up whether an *earlier* line was itself
        // entering a multi-line string, instead of re-deriving it with a
        // naive `starts_with("//")` that has no string awareness at all.
        let mut entering_in_str_by_line: Vec<bool> = Vec::with_capacity(lines.len());

        for (idx, &line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            let entering_in_str = in_str_state;
            entering_in_str_by_line.push(entering_in_str);

            let (opens, _closes, in_str_next) =
                advance_depth_pop_stacks(line, in_str_state, &mut depth, &mut [&mut skip_stack]);
            in_str_state = in_str_next;

            if trimmed.contains("#[cfg(test)]") {
                pending_cfg_test = true;
            }
            if trimmed.contains("#[test]") || trimmed.contains("#[tokio::test]") {
                pending_test_attr = true;
            }
            let is_mod_tests = (trimmed.starts_with("mod tests")
                || trimmed.starts_with("mod test ")
                || trimmed.starts_with("pub mod tests")
                || trimmed.starts_with("pub(crate) mod tests"))
                && opens > 0;
            if (pending_cfg_test || pending_test_attr || is_mod_tests) && opens > 0 {
                let entry_floor = depth - opens;
                skip_stack.push(entry_floor);
                pending_cfg_test = false;
                pending_test_attr = false;
            }
            let in_skip = skip_stack.last().is_some_and(|&d| depth > d);
            if in_skip {
                continue;
            }
            if opens > 0 {
                pending_cfg_test = false;
                pending_test_attr = false;
            }

            // Only a genuine comment when not entering this line already
            // inside a multi-line string (see scan_production_prints for why
            // trimmed-prefix "//" is not reliable otherwise).
            if !entering_in_str && trimmed.starts_with("//") {
                continue;
            }

            // Match `let _ = ident;`
            let ident = {
                let Some(rest) = trimmed.strip_prefix("let _ = ") else {
                    continue;
                };
                let candidate = rest.trim_end_matches(';').trim();
                if candidate.is_empty()
                    || !candidate
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_')
                {
                    continue;
                }
                candidate.to_string()
            };

            // Find the nearest enclosing `fn` by searching backward from idx.
            // Simple backward scan for `fn ` (handles nested blocks correctly because
            // we want the entire function body, not just the immediately enclosing block).
            let fn_line = {
                let mut found = None;
                for back in (0..idx).rev() {
                    if idx - back > 200 {
                        break;
                    }
                    let bl = lines[back].trim();
                    // Same "genuine comment" rule as the check above: a line
                    // that trims to a `//`-prefix is only a real comment when
                    // it was not itself entered already inside a multi-line
                    // string (the tail of a string like `"start\n// end";`
                    // trims to something starting with `//` but is code, not
                    // a comment).
                    let back_is_comment = !entering_in_str_by_line[back] && bl.starts_with("//");
                    if !back_is_comment && lines[back].contains("fn ") {
                        found = Some(back);
                        break;
                    }
                }
                match found {
                    Some(l) => l,
                    None => continue,
                }
            };

            // Find the function's opening `{` by scanning forward from fn_line.
            let body_line = {
                let mut found = None;
                let end = (fn_line + 40).min(lines.len().saturating_sub(1));
                for (fwd, l) in lines
                    .iter()
                    .enumerate()
                    .skip(fn_line)
                    .take(end.saturating_sub(fn_line) + 1)
                {
                    if l.contains('{') {
                        found = Some(fwd);
                        break;
                    }
                }
                match found {
                    Some(l) => l,
                    None => continue,
                }
            };

            // Collect signature text (fn_line..=body_line).
            let sig: String = lines[fn_line..=body_line].join("\n");

            // Extract parameter region between first `(` and `{`.
            let paren_start = match sig.find('(') {
                Some(i) => i,
                None => continue,
            };
            let brace_end = sig.find('{').unwrap_or(sig.len());
            let param_region = &sig[paren_start..brace_end.min(sig.len())];

            let needle = format!("{}:", ident);
            let is_param = {
                let mut found = false;
                let mut sf = 0;
                while let Some(pos) = param_region[sf..].find(&needle) {
                    let abs = sf + pos;
                    let before = if abs == 0 {
                        b'('
                    } else {
                        *param_region.as_bytes().get(abs - 1).unwrap_or(&b'(')
                    };
                    if matches!(before, b'(' | b' ' | b'\t' | b'\n' | b',') {
                        found = true;
                        break;
                    }
                    sf = abs + 1;
                }
                found
            };
            if !is_param {
                continue;
            }

            // Collect function body text (after the opening `{`, before closing `}`)
            // to check if ident is used anywhere other than `let _ = ident;`.
            let fn_body: String = {
                let mut body = String::new();
                let mut d: i32 = 0;
                let mut after_open = false;
                'outer: for (fwd, &l) in lines.iter().enumerate().skip(body_line) {
                    if !after_open {
                        // Emit only what follows the opening `{` on this line.
                        if let Some(brace_pos) = l.find('{') {
                            body.push_str(&l[brace_pos + 1..]);
                            body.push('\n');
                            after_open = true;
                            d = 1;
                            // Account for braces after the first `{` on this line.
                            let rest = &l[brace_pos + 1..];
                            d += rest.chars().filter(|&c| c == '{').count() as i32;
                            d -= rest.chars().filter(|&c| c == '}').count() as i32;
                            if d <= 0 {
                                break 'outer;
                            }
                        }
                    } else {
                        let opens_f = l.chars().filter(|&c| c == '{').count() as i32;
                        let closes_f = l.chars().filter(|&c| c == '}').count() as i32;
                        d += opens_f - closes_f;
                        if d <= 0 {
                            break 'outer;
                        }
                        body.push_str(l);
                        body.push('\n');
                    }
                    if fwd - body_line > 500 {
                        break;
                    }
                }
                body
            };

            // Only report if ident is NOT used anywhere else in the function body.
            if !ident_used_elsewhere(&fn_body, &ident) {
                violations.push((idx + 1, ident));
            }
        }

        violations
    }

    /// Verifies that production source files contain no `let _ = name;` where
    /// `name` is an explicit named parameter of the enclosing function.
    ///
    /// This pattern silently discards a caller-supplied value, making the
    /// parameter dead code that misleads both callers and readers.
    #[test]
    fn no_dead_params_in_production() {
        let root = workspace_root();
        let prod_dirs = [
            root.join("otspot-core/src"),
            root.join("otspot-io/src"),
            root.join("otspot-model/src"),
            root.join("src"),
        ];

        let mut all_violations: Vec<(std::path::PathBuf, usize, String)> = Vec::new();

        for dir in &prod_dirs {
            for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path().to_path_buf();
                if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                    continue;
                }
                if path.components().any(|c| c.as_os_str() == "bin") {
                    continue;
                }
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                // Only scan files that contain the pattern (fast skip).
                if !content.contains("let _ = ") {
                    continue;
                }
                for (line_no, param) in scan_dead_params(&content) {
                    all_violations.push((path.clone(), line_no, param));
                }
            }
        }

        if !all_violations.is_empty() {
            let mut msg = String::from(
                "dead function parameters detected (remove from signature or use the value):\n",
            );
            for (path, line_no, param) in &all_violations {
                msg.push_str(&format!(
                    "  {}:{}: `let _ = {};`\n",
                    path.display(),
                    line_no,
                    param
                ));
            }
            panic!("{msg}");
        }
    }

    // ── Layer C extension: underscore-prefixed signature parameters ─────────

    /// Returns `(1-based line number, parameter name)` for every function
    /// parameter of the form `_name: T` (non-trivially underscore-prefixed).
    ///
    /// A bare `_: T` (anonymous) is allowed; only named underscore parameters
    /// like `_target_pf: f64` are reported.  Methods (functions with a `self`
    /// parameter) inside `trait` definitions or `impl Trait for Type` blocks are
    /// skipped: those signatures are part of a required interface and may
    /// legitimately silence warnings with `_name`.  Inherent-impl methods
    /// (`impl Foo`) are checked.  Skips test-context blocks.
    fn scan_underscore_sig_params(content: &str) -> Vec<(usize, String)> {
        let mut violations = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        let mut depth: i32 = 0;
        let mut in_str_state = false;
        let mut skip_stack: Vec<i32> = Vec::new();
        // Entry floors of trait impl blocks (`impl Trait for Type`) or trait
        // definitions (`trait Foo { ... }`).  Methods with `self` in either
        // context may legitimately use `_name` to silence "unused" warnings
        // while matching a required interface.
        let mut trait_impl_stack: Vec<i32> = Vec::new();
        let mut pending_cfg_test = false;
        let mut pending_test_attr = false;
        // Deferred trait/impl context: set when we see `impl Trait for T` or
        // `trait Foo` on a line whose opening brace is on a subsequent line.
        let mut pending_trait_context = false;
        // Deferred inherent/trait impl header: set when `impl` is seen without
        // `for` on the same line and without an opening brace (rustfmt may wrap
        // `impl<T>\n    SomeTrait for Foo<T>\n{`). Cleared when a continuation
        // line reveals `for` (→ trait impl, upgrades to pending_trait_context) or
        // when the opening brace arrives without `for` (→ inherent impl).
        let mut pending_impl_header = false;

        // Accumulator for multi-line `fn` signature (fn ... {).
        let mut in_fn_sig = false;
        let mut fn_sig_buf = String::new();
        let mut fn_start_line: usize = 0;

        for (idx, &line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            let entering_in_str = in_str_state;

            let (opens, closes, in_str_next) = advance_depth_pop_stacks(
                line,
                in_str_state,
                &mut depth,
                &mut [&mut skip_stack, &mut trait_impl_stack],
            );
            in_str_state = in_str_next;

            // Track trait impl blocks (`impl Trait for Type`) and trait
            // definitions (`trait Foo { ... }`) — both need the exemption.
            // Handles same-line brace (`impl Trait for Foo {`), next-line brace
            // (`impl Trait for Foo\n{`), and rustfmt-wrapped headers where `impl`
            // and `for` appear on separate lines (`impl<T>\n    SomeTrait for Foo<T>\n{`).
            let is_impl_line = trimmed.starts_with("impl ") || trimmed.starts_with("impl<");
            let is_trait_impl_line = is_impl_line && trimmed.contains(" for ");
            let is_trait_def_line = trimmed.starts_with("trait ")
                || trimmed.starts_with("pub trait ")
                || trimmed.starts_with("pub(crate) trait ");
            if is_trait_impl_line || is_trait_def_line {
                // Fully-determined trait context on this line.
                pending_impl_header = false;
                if opens > 0 {
                    // Single-line completion (`trait Marker {}`, `impl Trait for T {}`)
                    // has no interior — skip push to avoid polluting the stack.
                    if opens != closes {
                        let entry_floor = depth - opens;
                        trait_impl_stack.push(entry_floor);
                    }
                    pending_trait_context = false;
                } else {
                    pending_trait_context = true;
                }
            } else if is_impl_line && opens == 0 {
                // `impl<T>` or `impl Foo` without `for` on the same line and no
                // opening brace yet — rustfmt may place `SomeTrait for Foo<T>`
                // on the next line.
                pending_impl_header = true;
            } else if pending_trait_context && opens > 0 {
                // Deferred: the opening brace arrived (may be on a `where` clause
                // continuation line or a standalone `{`).
                if opens != closes {
                    let entry_floor = depth - opens;
                    trait_impl_stack.push(entry_floor);
                }
                pending_trait_context = false;
                pending_impl_header = false;
            } else if pending_impl_header {
                // Still accumulating a multi-line impl header (no `{` yet).
                if trimmed.contains(" for ") {
                    // Continuation line contains `for` → confirmed trait impl.
                    pending_impl_header = false;
                    if opens > 0 {
                        if opens != closes {
                            let entry_floor = depth - opens;
                            trait_impl_stack.push(entry_floor);
                        }
                    } else {
                        // Opening brace on a later line (e.g. after `where` clause).
                        pending_trait_context = true;
                    }
                } else if opens > 0 {
                    // Opening brace arrived without ever seeing `for` → inherent impl.
                    pending_impl_header = false;
                }
                // Where-clause and type-bound lines are left intact so that
                // `impl<T>\n    SomeTrait for Foo<T>\nwhere\n    T: X,\n{` works.
            } else if opens == 0 && pending_trait_context {
                // Clear pending context only on unambiguous new declaration keywords.
                // Where-clause continuation lines (`where`, `T: Bound,`) are left
                // intact so multiline `impl Trait for T where T: X,\n{\n` works.
                let clears = trimmed.starts_with("fn ")
                    || trimmed.starts_with("pub fn ")
                    || trimmed.starts_with("async fn ")
                    || trimmed.starts_with("unsafe fn ")
                    || trimmed.starts_with("const fn ")
                    || trimmed.starts_with("extern fn ")
                    || trimmed.starts_with("pub(crate) fn ")
                    || trimmed.starts_with("pub(super) fn ")
                    || trimmed.starts_with("struct ")
                    || trimmed.starts_with("pub struct ")
                    || trimmed.starts_with("enum ")
                    || trimmed.starts_with("pub enum ")
                    || trimmed.starts_with("type ")
                    || trimmed.starts_with("impl ")
                    || trimmed.starts_with("mod ")
                    || trimmed.starts_with("use ")
                    || trimmed.starts_with("let ");
                if clears {
                    pending_trait_context = false;
                }
            }

            if trimmed.contains("#[cfg(test)]") {
                pending_cfg_test = true;
            }
            if trimmed.contains("#[test]") || trimmed.contains("#[tokio::test]") {
                pending_test_attr = true;
            }
            let is_mod_tests = (trimmed.starts_with("mod tests")
                || trimmed.starts_with("mod test ")
                || trimmed.starts_with("pub mod tests")
                || trimmed.starts_with("pub(crate) mod tests"))
                && opens > 0;
            if (pending_cfg_test || pending_test_attr || is_mod_tests) && opens > 0 {
                let entry_floor = depth - opens;
                skip_stack.push(entry_floor);
                pending_cfg_test = false;
                pending_test_attr = false;
                in_fn_sig = false;
                fn_sig_buf.clear();
            }
            let in_skip = skip_stack.last().is_some_and(|&d| depth > d);

            if in_skip {
                in_fn_sig = false;
                fn_sig_buf.clear();
                continue;
            }
            if opens > 0 {
                pending_cfg_test = false;
                pending_test_attr = false;
            }
            // Only a genuine comment when not entering this line already
            // inside a multi-line string (see scan_production_prints for why
            // trimmed-prefix "//" is not reliable otherwise).
            if !entering_in_str && trimmed.starts_with("//") {
                continue;
            }

            // Detect start of a function signature.
            if !in_fn_sig && line.contains("fn ") && line.contains('(') {
                in_fn_sig = true;
                fn_sig_buf.clear();
                fn_start_line = idx + 1; // 1-based
            }

            if in_fn_sig {
                fn_sig_buf.push_str(line);
                fn_sig_buf.push('\n');

                // Bodyless method: `fn required(&self, _unused: f64);` — only
                // legal inside a trait definition.  Terminate the accumulator
                // without reporting violations (the `_name` is there to satisfy
                // a required interface, same exemption as trait-impl methods).
                if line.contains(';') && !line.contains('{') {
                    in_fn_sig = false;
                    fn_sig_buf.clear();
                    continue;
                }

                // Once we see `{` the signature is complete.
                if line.contains('{') {
                    let sig = &fn_sig_buf;
                    if let Some(paren_start) = sig.find('(') {
                        let brace_end = sig.find('{').unwrap_or(sig.len());
                        let param_region = &sig[paren_start..brace_end.min(sig.len())];
                        // Skip methods in trait impls: trait implementations must match
                        // the trait's signature and may intentionally silence warnings
                        // with `_name`. Inherent methods are still checked.
                        let in_trait_impl = trait_impl_stack.last().is_some_and(|&d| depth > d);
                        if in_trait_impl
                            && (param_region.contains("&self")
                                || param_region.contains("&mut self")
                                || param_region.contains("self:")
                                || param_region.contains("mut self"))
                        {
                            in_fn_sig = false;
                            fn_sig_buf.clear();
                            continue;
                        }
                        // Detect `_name:` where name has at least one char after `_`.
                        let mut search = param_region;
                        while let Some(pos) = search.find("_") {
                            let rest = &search[pos..];
                            // Check if this `_` starts an identifier: `_` followed by alphanum/`_`
                            let after_underscore = &rest[1..];
                            let name_end = after_underscore
                                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                                .unwrap_or(after_underscore.len());
                            let name = &after_underscore[..name_end];
                            // Must have non-empty name (not bare `_`) and end with `:`
                            let after_name = &rest[1 + name_end..];
                            let is_param = !name.is_empty()
                                && after_name.starts_with(':')
                                && !after_name.starts_with("::");
                            // Check that the `_` is preceded by a param delimiter, not part of another ident.
                            let before_pos = param_region.len() - search.len() + pos;
                            let before = if before_pos == 0 {
                                b'('
                            } else {
                                *param_region.as_bytes().get(before_pos - 1).unwrap_or(&b'(')
                            };
                            let starts_ident = matches!(before, b'(' | b',' | b' ' | b'\t' | b'\n');
                            if is_param && starts_ident {
                                let full_name = format!("_{}", name);
                                violations.push((fn_start_line, full_name));
                            }
                            search = &search[pos + 1..];
                            if search.is_empty() {
                                break;
                            }
                        }
                    }
                    in_fn_sig = false;
                    fn_sig_buf.clear();
                }
            }
        }
        violations
    }

    /// Verifies that production source files contain no `_name: T` function
    /// parameters (underscore-prefixed named parameters that are never used).
    /// These mislead callers into thinking the parameter matters.
    #[test]
    fn no_underscore_sig_params_in_production() {
        let root = workspace_root();
        let prod_dirs = [
            root.join("otspot-core/src"),
            root.join("otspot-io/src"),
            root.join("otspot-model/src"),
            root.join("src"),
        ];

        let mut all_violations: Vec<(std::path::PathBuf, usize, String)> = Vec::new();

        for dir in &prod_dirs {
            for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path().to_path_buf();
                if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                    continue;
                }
                if path.components().any(|c| c.as_os_str() == "bin") {
                    continue;
                }
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                if !content.contains("fn ") {
                    continue;
                }
                for (line_no, param) in scan_underscore_sig_params(&content) {
                    all_violations.push((path.clone(), line_no, param));
                }
            }
        }

        if !all_violations.is_empty() {
            let mut msg = String::from(
                "underscore-prefixed named function parameters detected \
                 (remove from signature or use the value):\n",
            );
            for (path, line_no, param) in &all_violations {
                msg.push_str(&format!(
                    "  {}:{}: param `{}`\n",
                    path.display(),
                    line_no,
                    param
                ));
            }
            panic!("{msg}");
        }
    }

    /// Sentinel: `_name: T` in a function signature must be detected.
    #[test]
    fn scan_underscore_sig_params_detects_violation() {
        let content = r#"
pub(crate) fn collect_cluster_rows(
    problem: &str,
    candidate_cols: &[usize],
    _target_pf: f64,
) -> Option<usize> {
    None
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"_target_pf"),
            "_target_pf should be detected as underscore-sig param; violations: {:?}",
            violations
        );
    }

    /// Sentinel: bare `_: T` (anonymous) must NOT be flagged.
    #[test]
    fn scan_underscore_sig_params_ignores_anonymous() {
        let content = r#"
fn example(_: usize, x: f64) -> f64 {
    x
}
"#;
        let violations = scan_underscore_sig_params(content);
        assert!(
            violations.is_empty(),
            "bare `_: T` must not be flagged; violations: {:?}",
            violations
        );
    }

    /// Sentinel: `_name: T` in an *inherent* method (`impl Foo`) must still be detected.
    ///
    /// **No-op failure guarantee**: if inherent methods were also skipped (old behaviour
    /// "skip all methods with self"), `_unused_param` would not appear in violations
    /// → `assert!(names.contains(...))` fires.
    #[test]
    fn scan_underscore_sig_params_detects_inherent_method_unused_param() {
        let content = r#"
struct Foo;
impl Foo {
    fn do_something(&self, _unused_param: f64) -> f64 {
        42.0
    }
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"_unused_param"),
            "inherent-method `_unused_param` must be detected; violations: {:?}",
            violations
        );
    }

    /// Sentinel: `_name: T` in a *trait-impl* method or trait *default method* must NOT be flagged.
    ///
    /// **No-op failure guarantee**: if either exemption is removed, `_unused_param`
    /// would appear in violations → `assert!(!names.contains(...))` fires.
    #[test]
    fn scan_underscore_sig_params_skips_trait_impl_and_trait_default_method() {
        let content = r#"
trait MyTrait {
    fn default_method(&self, _unused_param: f64) -> f64 {
        0.0
    }
}
struct Bar;
impl MyTrait for Bar {
    fn default_method(&self, _unused_param: f64) -> f64 {
        1.0
    }
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            !names.contains(&"_unused_param"),
            "trait-impl and trait-default `_unused_param` must NOT be detected; violations: {:?}",
            violations
        );
    }

    /// Codex P2: multiline trait/impl header (opening brace on a separate line)
    /// must still exempt `_param` names inside the block.
    #[test]
    fn scan_underscore_sig_params_skips_multiline_trait_header() {
        let content = r#"
trait MyTrait
{
    fn required(&self, _unused: f64);
}
struct Foo;
impl MyTrait for Foo
{
    fn required(&self, _unused: f64) {}
}
impl<T> MyTrait for Vec<T>
where
    T: Clone,
{
    fn required(&self, _unused: f64) {}
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            !names.contains(&"_unused"),
            "multiline trait/impl header must exempt `_unused`; violations: {:?}",
            violations
        );
    }

    /// Sentinel (P2-1): rustfmt-wrapped `impl<T>\n    SomeTrait for Foo<T>\n{` must
    /// be recognized as a trait impl block, exempting `_param` inside from flagging.
    ///
    /// **No-op failure guarantee**: removing the `pending_impl_header` accumulation
    /// causes `_param` to be flagged as a violation → `assert!(!names.contains(...))`
    /// fires.
    #[test]
    fn scan_underscore_sig_params_recognizes_rustfmt_wrapped_impl_for() {
        let content = r#"
trait MyTrait {
    fn required(&self, _param: i32) -> i32;
}
struct Foo;
impl<T>
    MyTrait for Foo
{
    fn required(&self, _param: i32) -> i32 {
        _param
    }
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            !names.contains(&"_param"),
            "rustfmt-wrapped `impl<T>\\n    Trait for Foo\\n{{` must exempt `_param`; \
             violations: {:?}",
            violations
        );
    }

    /// Sentinel (P2-2): a bodyless trait method `fn required(&self, _unused: f64);`
    /// must terminate the signature accumulator at `;`, so the immediately following
    /// inherent impl block is judged independently and does not inherit trait context.
    ///
    /// **No-op failure guarantee**: removing the `;` terminator causes the accumulator
    /// to bleed into the next `{` (the inherent impl's brace), making `_unused` appear
    /// as a violation in inherent-impl context → `assert!(!names.contains(...))` fires.
    #[test]
    fn scan_underscore_sig_params_terminates_bodyless_trait_sig_at_semicolon() {
        let content = r#"
trait MyTrait {
    fn required(&self, _unused: f64);
}
struct Y;
impl Y {
    fn other(&self, x: i32) -> i32 {
        x
    }
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            !names.contains(&"_unused"),
            "bodyless trait method `_unused` must not bleed into inherent-impl context; \
             violations: {:?}",
            violations
        );
        assert!(
            violations.is_empty(),
            "no violations expected; violations: {:?}",
            violations
        );
    }

    /// Sentinel (P2-3): a single-line `trait Marker {}` must NOT leave a stale entry
    /// on the trait-impl stack that causes a subsequent inherent-impl method to be
    /// incorrectly exempted.
    ///
    /// **No-op failure guarantee**: removing the `opens != closes` guard in the
    /// `is_trait_def_line` branch causes `entry_floor = depth - opens` to push a
    /// floor of `-1` at top-level depth, making every later `depth > -1` check true.
    /// `_unused` in the inherent `impl Foo` then appears exempt → `assert!(names.contains(...))`
    /// fires.
    #[test]
    fn scan_underscore_sig_params_does_not_leak_context_from_single_line_trait() {
        let content = r#"
trait Marker {}
impl Foo {
    fn method(&self, _unused: f64) -> f64 {
        _unused
    }
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"_unused"),
            "inherent-method `_unused` after single-line `trait Marker {{}}` must be detected; \
             violations: {:?}",
            violations
        );
    }

    /// Sentinel (P2-4): a compact one-line `impl Marker for Foo {}` must NOT leave a
    /// stale entry on the trait-impl stack that exempts a subsequent inherent-impl method.
    ///
    /// **No-op failure guarantee**: removing the `opens != closes` guard in the
    /// `is_trait_impl_line` branch causes `entry_floor = depth - 1 = -1` to be pushed,
    /// making `in_trait_impl` true for all later code at depth ≥ 0.  `_unused` in the
    /// inherent `impl Foo` then appears exempt → `assert!(names.contains(...))` fires.
    #[test]
    fn scan_underscore_sig_params_does_not_leak_context_from_compact_impl_trait_for() {
        let content = r#"
trait Marker {}
struct Foo;
impl Marker for Foo {}
impl Foo {
    fn method(&self, _unused: f64) -> f64 {
        _unused
    }
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"_unused"),
            "inherent-method `_unused` after compact `impl Marker for Foo {{}}` must be detected; \
             violations: {:?}",
            violations
        );
    }

    /// Sentinel: an underscore-sig-param violation immediately after a
    /// same-line skip-zone close (`} fn wrapper() {`, net brace effect 0)
    /// must still be detected. Same root cause as the `scan_dead_params` and
    /// `scan_production_prints` sentinels of the same name: a stale
    /// `skip_stack` entry, left behind because a same-line re-open masked
    /// the line-final depth from ever touching the zone's floor, misclassifies
    /// the real `fn prod(_unused: i32) {}` as still inside the exited
    /// `mod tests` block.
    #[test]
    fn scan_underscore_sig_params_detects_violation_right_after_skip_zone_closes_same_line() {
        let content = r#"
#[cfg(test)]
mod tests {
    fn helper() {}
} fn wrapper() {
    fn prod(_unused: i32) {}
}
"#;
        let violations = scan_underscore_sig_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"_unused"),
            "BUG: _unused right after a same-line skip-zone close was missed; \
             violations: {:?}",
            violations
        );
    }

    // ── Unit tests for scanner helpers ──────────────────────────────────────

    /// Sentinel: `let _ = name;` where name is a function parameter must be detected.
    #[test]
    fn scan_dead_params_detects_dead_param() {
        let content = r#"
fn example(s_mat: &str, n: usize, x: f64) {
    let result = x + 1.0;
    let _ = s_mat;
    let _ = n;
    result
}
"#;
        let violations = scan_dead_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"s_mat"),
            "s_mat should be detected as dead param; violations: {:?}",
            violations
        );
        assert!(
            names.contains(&"n"),
            "n should be detected as dead param; violations: {:?}",
            violations
        );
    }

    /// Sentinel: `let _ = local_var;` where the name is NOT a parameter must NOT be flagged.
    #[test]
    fn scan_dead_params_ignores_non_params() {
        let content = r#"
fn compute(x: f64) -> f64 {
    let tmp = x * 2.0;
    let _ = tmp;
    x
}
"#;
        let violations = scan_dead_params(content);
        assert!(
            violations.is_empty(),
            "local variable `tmp` must not be flagged; violations: {:?}",
            violations
        );
    }

    /// Sentinel: a dead param immediately after a same-line skip-zone close
    /// (`} fn wrapper() {`, net brace effect 0) must still be detected. Same
    /// root cause as `scan_production_prints`'s false-negative: applying a
    /// line's net `opens - closes` atomically and popping `skip_stack` once
    /// against the line-final depth cannot observe a transient floor-touch
    /// that a later same-line re-open masks, leaving a stale skip-zone entry
    /// that misclassifies subsequent lines (here, the real `let _ = unused;`)
    /// as still being inside the already-exited `mod tests` block.
    #[test]
    fn scan_dead_params_detects_dead_param_right_after_skip_zone_closes_same_line() {
        let content = r#"
#[cfg(test)]
mod tests {
    fn helper() {}
} fn wrapper() {
    fn prod(unused: i32) {
        let _ = unused;
    }
}
"#;
        let violations = scan_dead_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"unused"),
            "BUG: dead param right after a same-line skip-zone close was missed; \
             violations: {:?}",
            violations
        );
    }

    /// Sentinel (reviewer-reported, live): the backward "find the nearest
    /// enclosing `fn`" search used a naive `bl.starts_with("//")` check with
    /// no `in_str`/`entering_in_str` awareness at all — unlike the line-comment
    /// check a few lines above it in this same function, which was already
    /// fixed for exactly this reason. A multi-line string whose closing line
    /// happens to trim to something starting with `//` (the tail of
    /// `"start\n// end"; fn prod(unused: i32) {`) makes the backward scan
    /// misclassify the real `fn prod(...)` line as a comment, skip past it,
    /// and instead find the outer `fn wrapper()` — so the dead-param check
    /// never sees `prod`'s own signature and the violation goes undetected.
    /// Built with an escaped (non-raw) literal for the same self-scan reason
    /// as `scan_production_prints_survives_multiline_string_closing_on_comment_lookalike_line`.
    #[test]
    fn scan_dead_params_finds_fn_across_multiline_string_closing_on_comment_lookalike_line() {
        let content =
            "\nfn wrapper() {\n    let s = \"start\n// end\"; fn prod(unused: i32) {\n        \
                        let _ = unused;\n    }\n}\n";
        let violations = scan_dead_params(content);
        let names: Vec<&str> = violations.iter().map(|(_, n)| n.as_str()).collect();
        assert!(
            names.contains(&"unused"),
            "BUG: dead param `unused` in `fn prod` was missed because the backward \
             fn-search misclassified the real `fn prod(...)` line as a comment \
             (multi-line string closing on a comment-lookalike line) and instead \
             attributed the `let _ = unused;` to the outer `fn wrapper()`; \
             violations: {:?}",
            violations
        );
    }

    /// Sentinel: a `// #[should_panic(...)]` comment above a `#[test]` function
    /// must NOT suppress the observation-only violation.  Before the fix the
    /// look-ahead treated it as the real attribute and silently skipped the body.
    #[test]
    fn scan_observation_only_tests_ignores_commented_should_panic() {
        let content = r#"#[test]
// #[should_panic(expected = "boom")]
fn fake_should_panic() {
    let _x = 1;
}
"#;
        let violations = scan_observation_only_tests(content);
        assert!(
            violations
                .iter()
                .any(|(_, name)| name == "fake_should_panic"),
            "commented #[should_panic] must not suppress observation-only detection; \
             violations: {:?}",
            violations
        );
    }

    /// Sentinel: an assertion-less `#[test]` fn whose closing brace shares a
    /// line with a later, unrelated re-open (`} fn helper() {`, net brace
    /// effect 0) must still be flagged. Before the fix, extracting the body
    /// via a per-line atomic `depth += opens - closes` (checked once at
    /// line-end) could not observe the true zero-crossing on that line —
    /// it kept absorbing lines (including the unrelated helper's own
    /// `assert!(true);`) into `no_assertion_test`'s "body" until some later,
    /// unrelated `}` brought the net back to zero, making the genuinely
    /// assertion-less test wrongly appear to contain an assertion.
    #[test]
    fn scan_observation_only_tests_detects_violation_when_closing_brace_shares_line_with_reopen() {
        let content = r#"
#[test]
fn no_assertion_test() {
    let x = 1;
} fn helper() {
    assert!(true);
}
"#;
        let violations = scan_observation_only_tests(content);
        assert!(
            violations
                .iter()
                .any(|(_, name)| name == "no_assertion_test"),
            "BUG: assertion-less test whose closing brace shares a line with a \
             later re-open was missed; violations: {:?}",
            violations
        );
    }
}
