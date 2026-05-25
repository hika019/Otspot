// Internal parser modules — pub(crate) so only test code in otspot-core can use them.
//
// These parsers are source-duplicates of the parsers in otspot-io.  Full dedup
// requires moving the ~12 diagnostic tests in qp::ipm_solver (which access
// pub(crate) CscMatrix fields) to integration tests in otspot-io.  Until that
// refactor is done, this module remains as a bridge.
//
// Canonical parser tests: otspot-io (95 tests).
// Remaining blocker: qp::ipm_solver tests use `crate::io::qps::parse_qps` and
// `pub(crate)` CscMatrix fields — cannot be moved without further API exposure.

pub mod mps;
pub mod qps;
pub mod qplib;
