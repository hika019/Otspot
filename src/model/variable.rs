//! Variable type for the modeling API

/// Integrality requirement for a decision variable.
///
/// `Continuous` is the default (matches pre-MIP behaviour). `Integer` / `Binary`
/// route the model through the MILP/MIQP branch-and-bound solver. `Binary` is an
/// `Integer` variable additionally fixed to the `[0, 1]` box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarKind {
    /// Real-valued variable (no integrality requirement).
    Continuous,
    /// Variable constrained to integer values within its bounds.
    Integer,
    /// Integer variable additionally restricted to `{0, 1}`.
    Binary,
}

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
    /// Integrality requirement. `Continuous` for `add_var`.
    pub kind: VarKind,
}
