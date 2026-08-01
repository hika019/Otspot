//! Second stage of the `cargo public-api` guardrail (see
//! `scripts/check_otspot_model_public_api.sh` for the first stage: a diff
//! between the live public API and each checked-in snapshot below, run in
//! CI's `public-api` job since it needs the nightly toolchain +
//! `cargo-public-api`).
//!
//! This test needs neither: it only reads the already-generated snapshot
//! files and `api_manifest.json`, all checked in, and asserts every
//! meaningful identifier in them is mentioned somewhere in the manifest (a
//! method/type/variant entry, or an `out_of_scope` note). That catches the
//! case the hand-written `api_manifest_rust.rs` compile checks cannot: a
//! *new* pub item added upstream that nobody updated the manifest for.
//!
//! Two snapshots: `otspot_model_api_snapshot.txt` (the full `otspot-model`
//! public API, the crate otspot-py binds directly) and
//! `otspot_core_status_tolerance_snapshot.txt` (otspot-core's public API
//! filtered to `SolveStatus`/`Tolerance` lines — otspot-model's own snapshot
//! only shows *references* to these otspot_core types as field/parameter
//! types, not their variants, since they are defined in a different crate).
//!
//! The match is a case-insensitive whole-word search over the manifest's
//! full JSON text, not a structured lookup — approximate, but proportionate:
//! it catches "this identifier is mentioned nowhere," which is exactly the
//! silent-drift failure mode this guardrail exists for.

const MODEL_SNAPSHOT: &str = include_str!("../otspot_model_api_snapshot.txt");
const CORE_STATUS_TOLERANCE_SNAPSHOT: &str =
    include_str!("../otspot_core_status_tolerance_snapshot.txt");
const MANIFEST_JSON: &str = include_str!("../api_manifest.json");

/// Trait-plumbing method names that appear in the snapshot purely because
/// their parent type derives/implements a standard trait (Debug's `fmt`,
/// PartialEq's `eq`/`ne`, etc.) — the parent *type* is what matters for API
/// coverage, and it is already checked separately; these leaf names would
/// otherwise be false-positive "uncovered" hits since they don't appear as
/// English words anywhere in the manifest prose.
const STANDARD_TRAIT_METHOD_NAMES: &[&str] = &["fmt", "eq", "ne", "clone", "hash", "index"];

/// Extracts the identifier a snapshot line is "about": the last `::`-
/// separated path segment before any `(` (for `pub fn ...path::name(...)`),
/// before a field-declaration `:` (for `pub path::Type::field: TypeExpr`),
/// or at the end of the line (for `pub struct path::Name`, `pub enum
/// path::Name`, `pub path::Enum::Variant`).
fn line_subject(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.is_empty()
        || line.starts_with("pub mod ")
        || line.starts_with("impl ")
        || line.starts_with("pub type ")
    {
        // `pub type ...::Output = f64` (associated types from trait impls
        // like `Index`) is structural: the trait impl itself is what
        // matters for coverage, already caught via the `impl ...` line
        // pattern this function is meant to skip anyway.
        return None;
    }
    if let Some(colon_pos) = field_declaration_colon(line) {
        let path = line[..colon_pos].trim_end();
        return path.rsplit("::").next().filter(|s| !s.is_empty());
    }
    let before_paren = line.split('(').next().unwrap_or(line);
    let path = before_paren
        .trim_end()
        .rsplit(' ')
        .next()
        .unwrap_or(before_paren);
    path.rsplit("::").next().filter(|s| !s.is_empty())
}

/// Finds the byte offset of a lone `:` (field-declaration separator, e.g.
/// `pub Type::field: Vec<f64>`), skipping over `::` path separators.
fn field_declaration_colon(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b':' {
            if bytes.get(i + 1) == Some(&b':') {
                i += 2;
                continue;
            }
            return Some(i);
        }
        i += 1;
    }
    None
}

fn word_present_case_insensitive(haystack_lower: &str, word: &str) -> bool {
    let word_lower = word.to_lowercase();
    let bytes = haystack_lower.as_bytes();
    let wbytes = word_lower.as_bytes();
    if wbytes.is_empty() {
        return true;
    }
    let is_word_byte = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut start = 0;
    while let Some(rel) = haystack_lower[start..].find(&word_lower) {
        let idx = start + rel;
        let before_ok = idx == 0 || !is_word_byte(bytes[idx - 1]);
        let end = idx + wbytes.len();
        let after_ok = end >= bytes.len() || !is_word_byte(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        start = idx + 1;
    }
    false
}

fn uncovered_identifiers<'a>(snapshot: &'a str, manifest_lower: &str) -> Vec<&'a str> {
    let mut uncovered: Vec<&str> = Vec::new();
    for line in snapshot.lines() {
        let Some(subject) = line_subject(line) else {
            continue;
        };
        if STANDARD_TRAIT_METHOD_NAMES.contains(&subject) {
            continue;
        }
        if !word_present_case_insensitive(manifest_lower, subject) {
            uncovered.push(subject);
        }
    }
    uncovered.sort_unstable();
    uncovered.dedup();
    uncovered
}

#[test]
fn public_api_snapshot_is_covered_by_manifest_or_out_of_scope() {
    let manifest_lower = MANIFEST_JSON.to_lowercase();

    let model_uncovered = uncovered_identifiers(MODEL_SNAPSHOT, &manifest_lower);
    assert!(
        model_uncovered.is_empty(),
        "otspot_model_api_snapshot.txt has identifiers not mentioned anywhere in \
         api_manifest.json (add a methods/types/variants entry or an out_of_scope \
         note): {model_uncovered:?}"
    );

    let core_uncovered = uncovered_identifiers(CORE_STATUS_TOLERANCE_SNAPSHOT, &manifest_lower);
    assert!(
        core_uncovered.is_empty(),
        "otspot_core_status_tolerance_snapshot.txt has identifiers not mentioned anywhere in \
         api_manifest.json (add a methods/types/variants entry or an out_of_scope \
         note): {core_uncovered:?}"
    );
}

/// Sanity check on the extractor itself: known lines must yield the subject
/// a human would expect. If this fails, the coverage test above is silently
/// checking the wrong thing.
#[test]
fn line_subject_extracts_expected_names() {
    assert_eq!(
        line_subject("pub fn otspot_model::Model::var_kind(&self, otspot_model::variable::Variable) -> otspot_model::variable::VarKind"),
        Some("var_kind")
    );
    assert_eq!(
        line_subject("pub struct otspot_model::expression::Expression"),
        Some("Expression")
    );
    assert_eq!(
        line_subject("#[non_exhaustive] pub enum otspot_model::constraint::ConstraintSense"),
        Some("ConstraintSense")
    );
    assert_eq!(
        line_subject("pub otspot_model::constraint::ConstraintSense::Eq"),
        Some("Eq")
    );
    assert_eq!(line_subject("pub mod otspot_model::constraint"), None);
    assert_eq!(
        line_subject("impl core::clone::Clone for otspot_model::constraint::Constraint"),
        None
    );
    assert_eq!(
        line_subject("pub type otspot_model::ModelResult::Output = f64"),
        None
    );
    assert_eq!(
        line_subject("pub otspot_model::ModelResult::bound_duals: alloc::vec::Vec<f64>"),
        Some("bound_duals")
    );
}
