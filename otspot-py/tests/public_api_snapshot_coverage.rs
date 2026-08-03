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
//! Matching is *qualified* (`Owner::leaf` / `Owner.leaf`), not a bare leaf-
//! name word search: an earlier version matched on the leaf name alone,
//! so an (injected, for verification) `Variable::value` snapshot entry
//! false-passed as "covered" purely because `ModelResult::value` is a real,
//! manifested entry containing the word "value" — the two are unrelated
//! symbols that happen to share a name. Enum variants are checked
//! structurally against the manifest's parsed `variants` map (no text
//! search at all, so no collision is possible there); types are checked
//! against the parsed `types` list. Only generic Rust operator-trait method
//! names (`add`/`sub`/`mul`/`neg`/`from`, see `GENERIC_METHOD_NAMES`) fall
//! back to a leaf-only search, since the manifest legitimately describes
//! these once per class via prose ("impl core::ops::{Add,Sub,Neg,Mul} for
//! Variable") rather than spelling out every concrete `Owner::add` pair --
//! collisions among *these* specific names are expected and harmless (they
//! are Rust's own generic operator vocabulary, not meaningfully
//! owner-specific the way "value" is).

use std::collections::{HashMap, HashSet};

const MODEL_SNAPSHOT: &str = include_str!("../otspot_model_api_snapshot.txt");
const CORE_STATUS_TOLERANCE_SNAPSHOT: &str =
    include_str!("../otspot_core_status_tolerance_snapshot.txt");
const MANIFEST_JSON: &str = include_str!("../api_manifest.json");

/// Pure derive/trait-impl boilerplate (`Debug`'s `fmt`, `PartialEq`'s
/// `eq`/`ne`, `Clone`'s `clone`, `Hash`'s `hash`, `Index`'s `index`):
/// present on essentially every type that derives the trait, never
/// individually mentioned in the manifest (nor should it be -- the *type*
/// deriving it is what matters for API coverage, already checked
/// separately). Unconditionally treated as covered, no search at all.
const UNCONDITIONAL_SKIP_NAMES: &[&str] = &["fmt", "eq", "ne", "clone", "hash", "index"];

/// Generic Rust operator-trait method names: legitimately described once in
/// manifest prose per class rather than spelled out per concrete `Owner::leaf`
/// pair (e.g. "impl core::ops::{Add,Sub,Neg,Mul} for Variable"), so these use
/// a leaf-only search rather than qualified `Owner::leaf` matching -- still a
/// real check (the word must appear *somewhere*), just not owner-qualified.
const GENERIC_METHOD_NAMES: &[&str] = &["add", "sub", "mul", "neg", "from"];

/// Rust built-in primitive type names: legitimate associated-type owners
/// (`f64::Output`, see `classify_pub_type_alias`) that never appear as a
/// `pub struct`/`pub enum`/`pub trait` declaration in any snapshot, so the
/// `known_types` collection below would otherwise misclassify them as a
/// module path segment and treat e.g. `f64::Output = Expression` as a real
/// top-level `Output` alias.
const RUST_PRIMITIVE_TYPES: &[&str] = &[
    "bool", "char", "str", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16",
    "u32", "u64", "u128", "usize",
];

#[derive(Debug, PartialEq)]
enum Subject<'a> {
    /// `pub struct Owner` / `pub enum Owner` / `pub trait Owner`.
    Type(&'a str),
    /// `pub Owner::Variant` (enum variant, no `fn`).
    Variant { owner: &'a str, variant: &'a str },
    /// `pub fn Owner::leaf(...)` or a field `pub Owner::leaf: Type`.
    Member { owner: &'a str, leaf: &'a str },
}

/// Extracts the declared name from a `pub struct `/`pub enum `/`pub trait `
/// line (attribute prefixes like `#[non_exhaustive]` are fine since this
/// looks for `.contains`, not `.starts_with`). Shared by `classify_line` and
/// the `known_types` pre-pass `classify_pub_type_alias` needs to tell a
/// genuine top-level `pub type` alias apart from associated-type boilerplate.
fn type_name_from_declaration(line: &str) -> Option<&str> {
    if line.contains("pub struct ") || line.contains("pub enum ") || line.contains("pub trait ") {
        last_path_segment(line.rsplit(' ').next()?)
    } else {
        None
    }
}

/// Classifies a `pub type <path> = <target>` line. Associated types from
/// trait impls (`Add::Output`, `Index::Output`, ...) render as
/// `pub type <OwnerType>::<AssocName> = <Concrete>`, where `<OwnerType>` is
/// always a real, already-declared type (a `pub struct`/`pub enum` in this
/// same snapshot, per `known_types`) or a Rust primitive (`f64::Output`).
/// A genuine top-level alias (`pub type Foo = Bar;` in source) instead
/// renders with the *module path* as the owner segment, e.g.
/// `pub type otspot_model::TempCalibrationAlias = f64` -- confirmed
/// empirically by injecting `pub type TempCalibrationAlias = f64;` into
/// `otspot-model/src/lib.rs` and regenerating the live snapshot: the
/// associated-type lines all kept their `Owner::Output` shape, while the
/// injected alias appeared as `pub type otspot_model::TempCalibrationAlias
/// = f64` -- `otspot_model` (the crate name) is never a declared type, so
/// it falls through to `Some(Subject::Type("TempCalibrationAlias"))` here,
/// while every real associated type's owner matches `known_types` or
/// `RUST_PRIMITIVE_TYPES` and stays skipped.
fn classify_pub_type_alias<'a>(line: &'a str, known_types: &HashSet<&str>) -> Option<Subject<'a>> {
    let line = line.trim();
    let path = line.strip_prefix("pub type ")?.split('=').next()?.trim();
    let (owner, leaf) = owner_and_leaf(path)?;
    if known_types.contains(owner) || RUST_PRIMITIVE_TYPES.contains(&owner) {
        return None;
    }
    Some(Subject::Type(leaf))
}

fn classify_line(line: &str) -> Option<Subject<'_>> {
    let line = line.trim();
    if line.is_empty()
        || line.starts_with("pub mod ")
        || line.starts_with("impl ")
        || line.starts_with("pub type ")
    {
        // `pub type ...::Output = f64` (associated types from trait impls
        // like `Index`) is structural: the trait impl itself is what
        // matters for coverage, already caught via the `impl ...` skip.
        // Genuine top-level aliases are handled separately by
        // `classify_pub_type_alias`, which needs the whole-snapshot
        // `known_types` set this single-line function doesn't have.
        return None;
    }
    if let Some(name) = type_name_from_declaration(line) {
        return Some(Subject::Type(name));
    }
    if line.starts_with("pub fn ") {
        let before_paren = line.split('(').next().unwrap_or(line);
        let path = before_paren
            .trim_end()
            .rsplit(' ')
            .next()
            .unwrap_or(before_paren);
        let (owner, leaf) = owner_and_leaf(path)?;
        return Some(Subject::Member { owner, leaf });
    }
    if let Some(colon_pos) = field_declaration_colon(line) {
        let path = line[..colon_pos].trim_end();
        let (owner, leaf) = owner_and_leaf(path)?;
        return Some(Subject::Member { owner, leaf });
    }
    // Bare variant path: `pub path::Enum::Variant` or `...Variant(Payload)`.
    let before_paren = line.split('(').next().unwrap_or(line);
    let path = before_paren
        .trim_end()
        .rsplit(' ')
        .next()
        .unwrap_or(before_paren);
    let (owner, variant) = owner_and_leaf(path)?;
    Some(Subject::Variant { owner, variant })
}

fn last_path_segment(path: &str) -> Option<&str> {
    path.rsplit("::").next().filter(|s| !s.is_empty())
}

/// Splits a `::`-separated path into (second-to-last segment, last segment)
/// -- i.e. (owning type, member name) for `path::to::Owner::leaf`.
fn owner_and_leaf(path: &str) -> Option<(&str, &str)> {
    let mut segs: Vec<&str> = path.split("::").filter(|s| !s.is_empty()).collect();
    let leaf = segs.pop()?;
    let owner = segs.pop().unwrap_or("");
    Some((owner, leaf))
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
    substr_present_case_insensitive(haystack_lower, word)
}

/// Case-insensitive word-boundary search for `needle` in `haystack_lower`
/// (already lowercased). Used both for bare leaf-name search (generic
/// operator methods, types) and for qualified `Owner::leaf`/`Owner.leaf`
/// search (regular methods/fields) -- the boundary check is what prevents
/// e.g. `Owner::leaf` from matching inside a longer identifier.
fn substr_present_case_insensitive(haystack_lower: &str, needle: &str) -> bool {
    let needle_lower = needle.to_lowercase();
    let bytes = haystack_lower.as_bytes();
    let nbytes = needle_lower.as_bytes();
    if nbytes.is_empty() {
        return true;
    }
    let is_word_byte = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut start = 0;
    while let Some(rel) = haystack_lower[start..].find(&needle_lower) {
        let idx = start + rel;
        let before_ok = idx == 0 || !is_word_byte(bytes[idx - 1]);
        let end = idx + nbytes.len();
        let after_ok = end >= bytes.len() || !is_word_byte(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        start = idx + 1;
    }
    false
}

struct ManifestIndex {
    lower_text: String,
    type_names: HashSet<String>,
    variants_by_owner: HashMap<String, HashSet<String>>,
}

impl ManifestIndex {
    fn parse(json: &str) -> Self {
        let v: serde_json::Value =
            serde_json::from_str(json).expect("api_manifest.json must be valid JSON");

        let mut type_names = HashSet::new();
        for entry in v["types"].as_array().expect("types must be an array") {
            let rust = entry["rust"]
                .as_str()
                .expect("types[].rust must be a string");
            if let Some(name) = last_path_segment(rust) {
                type_names.insert(name.to_string());
            }
        }

        let mut variants_by_owner = HashMap::new();
        let variants_obj = v["variants"]
            .as_object()
            .expect("variants must be an object");
        for (owner, arr) in variants_obj {
            let set: HashSet<String> = arr
                .as_array()
                .expect("variants[owner] must be an array")
                .iter()
                .map(|s| {
                    s.as_str()
                        .expect("variant name must be a string")
                        .to_string()
                })
                .collect();
            variants_by_owner.insert(owner.clone(), set);
        }
        // `python_only_variants` (e.g. SolveStatus::Unknown) are Python-side
        // additions, not real Rust variants -- never appear in the snapshot,
        // so they are irrelevant here and intentionally not merged in.

        ManifestIndex {
            lower_text: json.to_lowercase(),
            type_names,
            variants_by_owner,
        }
    }

    /// An owner type this manifest never binds at all (not in `types[]`,
    /// e.g. `ConstraintSense`, `SolverOptions`) only needs to be *mentioned*
    /// (by an `out_of_scope` note) once, not have every individual
    /// variant/member spelled out -- unlike a *bound* type (`Variable`,
    /// `ModelResult`, ...), where each member must be individually
    /// accounted for (this is what keeps the qualified match strict enough
    /// to catch the `Variable::value` collision below, while not forcing
    /// exhaustive enumeration of e.g. `SolverOptions`'s entire API).
    fn owner_is_out_of_scope_type(&self, owner: &str) -> bool {
        !self.type_names.contains(owner) && word_present_case_insensitive(&self.lower_text, owner)
    }

    fn covers(&self, subject: &Subject<'_>) -> bool {
        match subject {
            Subject::Type(name) => {
                self.type_names.contains(*name)
                    || word_present_case_insensitive(&self.lower_text, name)
            }
            Subject::Variant { owner, variant } => {
                if let Some(vs) = self.variants_by_owner.get(*owner) {
                    return vs.contains(*variant);
                }
                self.owner_is_out_of_scope_type(owner)
            }
            Subject::Member { owner, leaf } => {
                if UNCONDITIONAL_SKIP_NAMES.contains(leaf) {
                    return true;
                }
                if GENERIC_METHOD_NAMES.contains(leaf) {
                    return word_present_case_insensitive(&self.lower_text, leaf);
                }
                let qualified_colons = format!("{owner}::{leaf}");
                let qualified_dot = format!("{owner}.{leaf}");
                if substr_present_case_insensitive(&self.lower_text, &qualified_colons)
                    || substr_present_case_insensitive(&self.lower_text, &qualified_dot)
                {
                    return true;
                }
                self.owner_is_out_of_scope_type(owner)
            }
        }
    }
}

fn describe(subject: &Subject<'_>) -> String {
    match subject {
        Subject::Type(name) => format!("type {name}"),
        Subject::Variant { owner, variant } => format!("variant {owner}::{variant}"),
        Subject::Member { owner, leaf } => format!("member {owner}::{leaf}"),
    }
}

fn uncovered_identifiers(snapshot: &str, index: &ManifestIndex) -> Vec<String> {
    let known_types: HashSet<&str> = snapshot
        .lines()
        .filter_map(|l| type_name_from_declaration(l.trim()))
        .collect();

    let mut uncovered = Vec::new();
    for line in snapshot.lines() {
        let subject = classify_pub_type_alias(line, &known_types).or_else(|| classify_line(line));
        let Some(subject) = subject else {
            continue;
        };
        if !index.covers(&subject) {
            uncovered.push(describe(&subject));
        }
    }
    uncovered.sort_unstable();
    uncovered.dedup();
    uncovered
}

#[test]
fn public_api_snapshot_is_covered_by_manifest_or_out_of_scope() {
    let index = ManifestIndex::parse(MANIFEST_JSON);

    let model_uncovered = uncovered_identifiers(MODEL_SNAPSHOT, &index);
    assert!(
        model_uncovered.is_empty(),
        "otspot_model_api_snapshot.txt has identifiers not mentioned anywhere in \
         api_manifest.json (add a methods/types/variants entry or an out_of_scope \
         note): {model_uncovered:?}"
    );

    let core_uncovered = uncovered_identifiers(CORE_STATUS_TOLERANCE_SNAPSHOT, &index);
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
fn classify_line_extracts_expected_subjects() {
    assert_eq!(
        classify_line("pub fn otspot_model::Model::var_kind(&self, otspot_model::variable::Variable) -> otspot_model::variable::VarKind"),
        Some(Subject::Member { owner: "Model", leaf: "var_kind" })
    );
    assert_eq!(
        classify_line("pub struct otspot_model::expression::Expression"),
        Some(Subject::Type("Expression"))
    );
    assert_eq!(
        classify_line("#[non_exhaustive] pub enum otspot_model::constraint::ConstraintSense"),
        Some(Subject::Type("ConstraintSense"))
    );
    assert_eq!(
        classify_line("pub otspot_model::constraint::ConstraintSense::Eq"),
        Some(Subject::Variant {
            owner: "ConstraintSense",
            variant: "Eq"
        })
    );
    assert_eq!(classify_line("pub mod otspot_model::constraint"), None);
    assert_eq!(
        classify_line("impl core::clone::Clone for otspot_model::constraint::Constraint"),
        None
    );
    assert_eq!(
        classify_line("pub type otspot_model::ModelResult::Output = f64"),
        None
    );
    assert_eq!(
        classify_line("pub otspot_model::ModelResult::bound_duals: alloc::vec::Vec<f64>"),
        Some(Subject::Member {
            owner: "ModelResult",
            leaf: "bound_duals"
        })
    );
}

/// The exact regression this rewrite fixes: a `Variable::value` member must
/// NOT be considered covered just because `ModelResult::value` is a real,
/// manifested entry sharing the leaf name "value" (verified against the
/// live manifest by injecting this exact synthetic line into a copy of the
/// snapshot before this fix; reverting `ManifestIndex::covers`'s qualified
/// match back to a bare leaf-name search makes this test FAIL).
#[test]
fn qualified_matching_rejects_leaf_name_collision() {
    let index = ManifestIndex::parse(MANIFEST_JSON);
    let real = Subject::Member {
        owner: "ModelResult",
        leaf: "value",
    };
    assert!(index.covers(&real), "ModelResult::value must be covered");

    let collision = Subject::Member {
        owner: "Variable",
        leaf: "value",
    };
    assert!(
        !index.covers(&collision),
        "Variable::value is not a real manifested member and must not be \
         covered merely because ModelResult::value shares the leaf name \"value\""
    );
}

/// `classify_pub_type_alias` must tell a genuine top-level type alias apart
/// from associated-type boilerplate using only the owner segment's identity
/// (a known declared type / a Rust primitive) -- not by, say, path length or
/// module-name casing, which would misclassify a same-crate alias placed
/// inside a submodule. Exact line shapes and the crate-name-as-owner case
/// (`otspot_model::TempCalibrationAlias`) were confirmed empirically by
/// injecting a real alias into `otspot-model/src/lib.rs` and regenerating
/// the live snapshot (see `classify_pub_type_alias`'s doc comment).
#[test]
fn classify_pub_type_alias_distinguishes_real_alias_from_associated_type() {
    let known_types: HashSet<&str> = ["Variable", "ModelResult", "Expression", "QuadExpr"]
        .into_iter()
        .collect();

    // Associated types: owner is a known declared type or a primitive.
    assert_eq!(
        classify_pub_type_alias(
            "pub type otspot_model::ModelResult::Output = f64",
            &known_types
        ),
        None
    );
    assert_eq!(
        classify_pub_type_alias(
            "pub type f64::Output = otspot_model::expression::Expression",
            &known_types
        ),
        None
    );
    assert_eq!(
        classify_pub_type_alias(
            "pub type otspot_model::variable::Variable::Output = otspot_model::quad_expr::QuadExpr",
            &known_types
        ),
        None
    );

    // Genuine top-level alias: owner is the crate name, not a declared type.
    assert_eq!(
        classify_pub_type_alias(
            "pub type otspot_model::TempCalibrationAlias = f64",
            &known_types
        ),
        Some(Subject::Type("TempCalibrationAlias"))
    );
    // Same, nested one level deeper inside a submodule rather than the crate root.
    assert_eq!(
        classify_pub_type_alias(
            "pub type otspot_model::expression::ExprAlias = otspot_model::expression::Expression",
            &known_types
        ),
        Some(Subject::Type("ExprAlias"))
    );

    // Not a `pub type` line at all.
    assert_eq!(
        classify_pub_type_alias("pub struct otspot_model::variable::Variable", &known_types),
        None
    );
}

/// End-to-end: an unmanifested top-level alias injected into a copy of the
/// real snapshot must be reported as uncovered by `uncovered_identifiers`,
/// not silently discarded the way the unconditional `pub type` skip used to
/// discard it. Sentinel: reverting `uncovered_identifiers` back to calling
/// only `classify_line` (dropping the `classify_pub_type_alias` pre-pass)
/// makes this assert an empty `Vec` instead -- confirmed by reverting and
/// re-running.
#[test]
fn uncovered_identifiers_flags_unmanifested_top_level_alias() {
    let index = ManifestIndex::parse(MANIFEST_JSON);
    let injected =
        format!("{MODEL_SNAPSHOT}\npub type otspot_model::TotallyUnmanifestedAlias = f64\n");

    let uncovered = uncovered_identifiers(&injected, &index);

    assert!(
        uncovered.contains(&"type TotallyUnmanifestedAlias".to_string()),
        "an injected top-level alias absent from api_manifest.json must be \
         flagged as uncovered, got: {uncovered:?}"
    );
}
