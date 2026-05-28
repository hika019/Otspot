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

    /// Counts `{` and `}` on a line, ignoring `//`-comments and string literals.
    ///
    /// `in_str_in` carries multi-line string state from the previous line.
    /// Returns `(opens, closes, in_str_out)`.
    fn count_braces(line: &str, in_str_in: bool) -> (i32, i32, bool) {
        let mut opens = 0i32;
        let mut closes = 0i32;
        let mut in_str = in_str_in;
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            let prev = if i > 0 { chars[i - 1] } else { '\0' };
            // Check for line comment only when not in a string
            if !in_str && ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                break; // rest of line is a comment
            }
            if ch == '"' && prev != '\\' {
                in_str = !in_str;
            }
            if !in_str {
                if ch == '{' { opens += 1; }
                if ch == '}' { closes += 1; }
            }
            i += 1;
        }
        (opens, closes, in_str)
    }

    /// Scans `content` for direct print macro calls that are not inside a
    /// test context (i.e., outside `#[cfg(test)]` blocks, `mod tests`/`mod test`
    /// blocks, and `#[test]`/`#[tokio::test]` function bodies).
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
        // Print macros at depth > allow_stack.last() are exempted.
        let mut allow_print_stack: Vec<i32> = Vec::new();

        for (line_idx, raw_line) in content.lines().enumerate() {
            let line_no = line_idx + 1;
            let trimmed = raw_line.trim();

            // Skip pure line comments
            if trimmed.starts_with("//") {
                continue;
            }

            let (opens, closes, in_str_next) = count_braces(raw_line, in_str_state);
            in_str_state = in_str_next;

            // Apply closes first, then check for skip-zone exit
            depth -= closes;
            while skip_stack.last().is_some_and(|&d| depth <= d) {
                skip_stack.pop();
            }
            // Pop allow_print exemptions when depth drops below their entry depth.
            while allow_print_stack.last().is_some_and(|&d| depth < d) {
                allow_print_stack.pop();
            }

            // Apply opens
            depth += opens;

            // Detect attributes that trigger skip zones
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

            // `mod tests {` or `mod test {` on the same line (including pub variants)
            let is_mod_tests = (trimmed.starts_with("mod tests")
                || trimmed.starts_with("mod test ")
                || trimmed.starts_with("pub mod tests")
                || trimmed.starts_with("pub(crate) mod tests"))
                && opens > 0;

            // Enter skip zone if:
            // - we saw a cfg(test)/test attr on a previous line and now opened a block, OR
            // - this line is `mod tests {`
            if (pending_cfg_test || pending_test_attr || is_mod_tests) && opens > 0 {
                // depth already includes this line's opens; entry depth = depth - opens + (opens-1)
                // i.e., we want to skip while depth > (depth - opens).
                let entry_floor = depth - opens;
                skip_stack.push(entry_floor);
                pending_cfg_test = false;
                pending_test_attr = false;
            }

            let in_skip = skip_stack.last().is_some_and(|&d| depth > d);
            if in_skip {
                continue;
            }

            // Reset pending flags if we passed a function or mod opening without
            // triggering skip (e.g., attribute + non-opening line sequence)
            if opens > 0 {
                pending_cfg_test = false;
                pending_test_attr = false;
            }

            // Check for print macros (strip comment before checking)
            // Skip if inside an #[allow(clippy::print_stderr/stdout)] scope.
            let print_is_allowed = allow_print_stack.last().is_some_and(|&d| depth >= d);
            if !print_is_allowed {
                let code_part = raw_line.split("//").next().unwrap_or(raw_line);
                for &macro_name in PRINT_MACROS {
                    if code_part.contains(macro_name) {
                        violations.push((line_no, macro_name.trim_end_matches('(').to_string() + "!"));
                        break;
                    }
                }
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
            let fn_name = extract_fn_name(lines[fn_idx]).unwrap_or_else(|| format!("<line_{}>", fn_idx + 1));

            // should_panic functions are always valid
            if should_panic_seen {
                i = fn_idx + 1;
                continue;
            }

            // Extract the body using brace matching
            let mut depth = 0i32;
            let mut body = String::new();
            let mut started = false;
            let mut end_idx = fn_idx;

            let mut body_in_str = false;
            for (k, line) in lines.iter().enumerate().skip(fn_idx) {
                let (opens, closes, in_str_next) = count_braces(line, body_in_str);
                body_in_str = in_str_next;
                depth += opens - closes;
                if opens > 0 {
                    started = true;
                }
                if started {
                    body.push_str(line);
                    body.push('\n');
                }
                if started && depth <= 0 {
                    end_idx = k;
                    break;
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
            for entry in WalkDir::new(dir)
                .into_iter()
                .filter_map(|e| e.ok())
            {
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
            for entry in WalkDir::new(dir)
                .into_iter()
                .filter_map(|e| e.ok())
            {
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
            let mut msg = String::from(
                "observation-only tests detected (add assertions or delete):\n",
            );
            for (path, line_no, name) in &all_violations {
                msg.push_str(&format!("  {}:{}: fn {}\n", path.display(), line_no, name));
            }
            panic!("{msg}");
        }
    }

    // ── Unit tests for scanner helpers ──────────────────────────────────────

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
            violations.iter().any(|(_, name)| name == "fake_should_panic"),
            "commented #[should_panic] must not suppress observation-only detection; \
             violations: {:?}",
            violations
        );
    }
}
