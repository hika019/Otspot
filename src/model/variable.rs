//! Variable type for the modeling API

/// A lightweight handle to a decision variable.
///
/// `Variable` is `Copy`, so it can be used in expressions multiple times
/// without being consumed: `x + y` does not move `x`.
///
/// Internally, a `Variable` is just an index into the `Model`'s variable list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable {
    pub(crate) index: usize,
}

/// Metadata about a variable stored in the `Model`.
pub(crate) struct VariableDefinition {
    #[allow(dead_code)]  // stored for future display/debugging use
    pub name: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
}
