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
/// Internally, a `Variable` carries its index and the ID of the model that
/// created it. The model ID is used by checked variants such as
/// [`crate::Model::try_var_name`] and [`crate::ModelResult::try_value`] to detect
/// cross-model variable misuse at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable {
    pub(crate) index: usize,
    pub(crate) model_id: u64,
}

/// Metadata about a variable stored in the `Model`.
pub(crate) struct VariableDefinition {
    pub name: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Integrality requirement. `Continuous` for `add_var`.
    pub kind: VarKind,
}
