// Internal QPS parser — a source-duplicate of otspot-io's canonical parser,
// retained only for otspot-core's own qp::ipm_solver diagnostic tests, which
// access crate-internal IPM machinery and pub(crate) CscMatrix fields and so
// cannot move to otspot-io integration tests without exposing those internals.
// Compiled under cfg(test) only (see lib.rs); not part of the production library.
// Canonical, published parsers and their 95 tests live in otspot-io.

pub mod qps;
