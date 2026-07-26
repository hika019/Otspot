//! Compatibility re-exports for the numerical sparse layer.
//!
//! This module exists solely as a public API surface for downstream crates
//! (`otspot-io`, `otspot-model`, `otspot-dev`). Code inside `otspot-core`
//! must depend on `otspot_num::sparse` directly rather than `crate::sparse`.

pub use otspot_num::sparse::{CscMatrix, SparseVec};
