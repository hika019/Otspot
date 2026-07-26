//! Compatibility re-exports for error types.
//!
//! `MpsError` is physically owned by `crate::mps_error` (an otspot-core I/O
//! error type); `SolverError` is physically owned by `otspot_num`. Both are
//! re-exported here solely as a public API surface for downstream crates
//! (`otspot-io`, `otspot-model`, `otspot-dev`). Code inside `otspot-core`
//! must depend on `crate::mps_error::MpsError` / `otspot_num::SolverError`
//! directly rather than routing through this module's re-export or the
//! crate-root alias of it.

pub use crate::mps_error::MpsError;
pub use otspot_num::SolverError;
