//! Quadratic expression type for QP objectives.
//!
//! `QuadExpr` extends [`Expression`] with quadratic terms, enabling ergonomic
//! construction of QP objectives via operator overloading:
//!
//! ```rust,no_run
//! use otspot_model::Model;
//!
//! let mut model = Model::new("qp");
//! let x = model.add_var("x", 1.0, f64::INFINITY);
//! let y = model.add_var("y", 0.0, f64::INFINITY);
//! model.minimize(x * x + 2.0 * x * y);  // min x² + 2xy
//! ```

use std::collections::HashMap;
use std::ops::{Add, Mul, Neg, Sub};

use super::expression::Expression;
use super::variable::Variable;
use otspot_core::sparse::CscMatrix;

/// A quadratic (or linear) expression for use as a QP objective.
///
/// Stores quadratic terms as `(va, vb) → coefficient` where the pair is in
/// canonical order: `(va.model_id, va.index) ≤ (vb.model_id, vb.index)`.
///
/// # Q-matrix convention
/// When converted to `CscMatrix` via [`quad_to_csc`], the "1/2 xᵀQx" convention is used:
/// - Diagonal `c · xi²` → `Q[i][i] = 2c`  (so `1/2 · Q[i][i] · xi² = c · xi²`)
/// - Cross `c · xi · xj` (i≠j) → `Q[i][j] = Q[j][i] = c`  (symmetric fill, both sides)
#[derive(Debug, Clone, Default)]
pub struct QuadExpr {
    /// Quadratic terms in canonical-pair key order.
    pub(crate) quad: HashMap<(Variable, Variable), f64>,
    /// Linear and constant parts.
    pub(crate) linear: Expression,
}

impl QuadExpr {
    /// Returns `true` if this expression contains no quadratic terms.
    pub fn is_linear(&self) -> bool {
        self.quad.is_empty()
    }

    fn merge_add(&mut self, rhs: QuadExpr) {
        for (pair, c) in rhs.quad {
            insert_quad_term(&mut self.quad, pair, c);
        }
        let old = std::mem::take(&mut self.linear);
        self.linear = old + rhs.linear;
    }
}

/// Accumulate a quadratic term into the map, skipping zero deltas and removing
/// entries that cancel to exactly zero.  This is the **single insertion
/// chokepoint** for all quad-term construction — routing every write through
/// here prevents zero-coefficient entries from leaking into `QuadExpr::quad`
/// and causing spurious `is_linear() == false`.
fn insert_quad_term(
    quad: &mut HashMap<(Variable, Variable), f64>,
    key: (Variable, Variable),
    delta: f64,
) {
    if delta == 0.0 {
        return; // zero contribution — skip to avoid polluting the map
    }
    let entry = quad.entry(key).or_insert(0.0);
    *entry += delta;
    if *entry == 0.0 {
        quad.remove(&key);
    }
}

/// Returns the canonical pair `(a, b)` with `(a.model_id, a.index) ≤ (b.model_id, b.index)`.
fn canon(a: Variable, b: Variable) -> (Variable, Variable) {
    if (a.model_id, a.index) <= (b.model_id, b.index) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Convert quadratic-term map to a symmetric `CscMatrix` using the 1/2 xᵀQx convention.
///
/// - Diagonal entry `(i, i) → c`: emits `Q[i][i] = 2c`.
/// - Off-diagonal `(i, j) → c` (i ≠ j): emits both `Q[i][j] = c` and `Q[j][i] = c`.
///
/// Returns an error if any variable index is out of range for a matrix of size `n×n`.
pub(crate) fn quad_to_csc(
    terms: &HashMap<(Variable, Variable), f64>,
    n: usize,
) -> Result<CscMatrix, String> {
    if terms.is_empty() {
        return Ok(CscMatrix::new(n, n));
    }

    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    for (&(va, vb), &c) in terms {
        let (i, j) = (va.index, vb.index);
        if !c.is_finite() {
            return Err(format!(
                "non-finite quad coefficient at ({i}, {j}): {c}"
            ));
        }
        if i >= n || j >= n {
            return Err(format!(
                "quad term ({i}, {j}) out of range for {n} variables"
            ));
        }
        if i == j {
            // Diagonal: 1/2 · Q[i][i] · xi² = c · xi²  ⟹  Q[i][i] = 2c
            rows.push(i);
            cols.push(j);
            vals.push(2.0 * c);
        } else {
            // Off-diagonal symmetric: 1/2 · (Q[i][j] + Q[j][i]) · xi·xj = c · xi·xj  ⟹  Q[i][j] = Q[j][i] = c
            rows.push(i);
            cols.push(j);
            vals.push(c);
            rows.push(j);
            cols.push(i);
            vals.push(c);
        }
    }

    CscMatrix::from_triplets(&rows, &cols, &vals, n, n)
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Variable extension: pow2
// ---------------------------------------------------------------------------

impl Variable {
    /// Returns `x²` as a [`QuadExpr`].
    pub fn pow2(self) -> QuadExpr {
        self * self
    }
}

// ---------------------------------------------------------------------------
// From conversions
// ---------------------------------------------------------------------------

impl From<Variable> for QuadExpr {
    fn from(v: Variable) -> Self {
        QuadExpr { quad: HashMap::new(), linear: Expression::from(v) }
    }
}

impl From<Expression> for QuadExpr {
    fn from(e: Expression) -> Self {
        QuadExpr { quad: HashMap::new(), linear: e }
    }
}

impl From<f64> for QuadExpr {
    fn from(c: f64) -> Self {
        QuadExpr { quad: HashMap::new(), linear: Expression::from(c) }
    }
}

impl From<i32> for QuadExpr {
    fn from(c: i32) -> Self {
        QuadExpr { quad: HashMap::new(), linear: Expression::from(c) }
    }
}

// ---------------------------------------------------------------------------
// Variable * Variable → QuadExpr
// ---------------------------------------------------------------------------

impl Mul<Variable> for Variable {
    type Output = QuadExpr;
    fn mul(self, rhs: Variable) -> QuadExpr {
        let mut quad = HashMap::new();
        insert_quad_term(&mut quad, canon(self, rhs), 1.0);
        QuadExpr { quad, linear: Expression::default() }
    }
}

// ---------------------------------------------------------------------------
// Expression * Variable  /  Variable * Expression → QuadExpr
// ---------------------------------------------------------------------------

impl Mul<Variable> for Expression {
    type Output = QuadExpr;
    fn mul(self, var: Variable) -> QuadExpr {
        let mut quad = HashMap::new();
        let mut linear = Expression::default();
        for (&v, &c) in &self.coefficients {
            // Route through the single chokepoint so zero coefficients (e.g.
            // from `(x - x) * y`) never enter the map.
            insert_quad_term(&mut quad, canon(v, var), c);
        }
        if self.constant != 0.0 {
            *linear.coefficients.entry(var).or_insert(0.0) += self.constant;
        }
        QuadExpr { quad, linear }
    }
}

impl Mul<Expression> for Variable {
    type Output = QuadExpr;
    fn mul(self, rhs: Expression) -> QuadExpr {
        rhs * self
    }
}

// ---------------------------------------------------------------------------
// f64 * QuadExpr  /  QuadExpr * f64
// ---------------------------------------------------------------------------

impl Mul<f64> for QuadExpr {
    type Output = QuadExpr;
    fn mul(mut self, rhs: f64) -> QuadExpr {
        for v in self.quad.values_mut() {
            *v *= rhs;
        }
        // Prune entries zeroed by multiplication (e.g. 0.0 * x*x → is_linear = true).
        self.quad.retain(|_, c| *c != 0.0);
        self.linear = rhs * self.linear;
        self
    }
}

impl Mul<QuadExpr> for f64 {
    type Output = QuadExpr;
    fn mul(self, rhs: QuadExpr) -> QuadExpr {
        rhs * self
    }
}

// ---------------------------------------------------------------------------
// Neg
// ---------------------------------------------------------------------------

impl Neg for QuadExpr {
    type Output = QuadExpr;
    fn neg(mut self) -> QuadExpr {
        for v in self.quad.values_mut() {
            *v = -*v;
        }
        self.linear = -self.linear;
        self
    }
}

// ---------------------------------------------------------------------------
// QuadExpr ± QuadExpr
// ---------------------------------------------------------------------------

impl Add for QuadExpr {
    type Output = QuadExpr;
    fn add(mut self, rhs: QuadExpr) -> QuadExpr {
        self.merge_add(rhs);
        self
    }
}

impl Sub for QuadExpr {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        self + (-rhs)
    }
}

// ---------------------------------------------------------------------------
// QuadExpr ± Expression
// ---------------------------------------------------------------------------

impl Add<Expression> for QuadExpr {
    type Output = QuadExpr;
    fn add(self, rhs: Expression) -> QuadExpr {
        self + QuadExpr::from(rhs)
    }
}

impl Add<QuadExpr> for Expression {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        rhs + self
    }
}

impl Sub<Expression> for QuadExpr {
    type Output = QuadExpr;
    fn sub(self, rhs: Expression) -> QuadExpr {
        self + (-rhs)
    }
}

impl Sub<QuadExpr> for Expression {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        QuadExpr::from(self) + (-rhs)
    }
}

// ---------------------------------------------------------------------------
// QuadExpr ± Variable
// ---------------------------------------------------------------------------

impl Add<Variable> for QuadExpr {
    type Output = QuadExpr;
    fn add(self, rhs: Variable) -> QuadExpr {
        self + Expression::from(rhs)
    }
}

impl Add<QuadExpr> for Variable {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        rhs + self
    }
}

impl Sub<Variable> for QuadExpr {
    type Output = QuadExpr;
    fn sub(self, rhs: Variable) -> QuadExpr {
        self + (-Expression::from(rhs))
    }
}

impl Sub<QuadExpr> for Variable {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        QuadExpr::from(Expression::from(self)) + (-rhs)
    }
}

// ---------------------------------------------------------------------------
// QuadExpr ± f64
// ---------------------------------------------------------------------------

impl Add<f64> for QuadExpr {
    type Output = QuadExpr;
    fn add(self, rhs: f64) -> QuadExpr {
        self + Expression::from(rhs)
    }
}

impl Add<QuadExpr> for f64 {
    type Output = QuadExpr;
    fn add(self, rhs: QuadExpr) -> QuadExpr {
        rhs + self
    }
}

impl Sub<f64> for QuadExpr {
    type Output = QuadExpr;
    fn sub(self, rhs: f64) -> QuadExpr {
        self + (-rhs)
    }
}

impl Sub<QuadExpr> for f64 {
    type Output = QuadExpr;
    fn sub(self, rhs: QuadExpr) -> QuadExpr {
        self + (-rhs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;

    const TOL: f64 = 1e-5;

    fn assert_close(a: f64, b: f64, label: &str) {
        assert!((a - b).abs() < TOL, "{label}: expected {b}, got {a}");
    }

    /// Extract Q[row][col] from a CscMatrix (returns 0.0 if absent).
    fn q_entry(q: &CscMatrix, row: usize, col: usize) -> f64 {
        let col_start = q.col_ptr()[col];
        let col_end = q.col_ptr()[col + 1];
        for k in col_start..col_end {
            if q.row_ind()[k] == row {
                return q.values()[k];
            }
        }
        0.0
    }

    // --- quad_to_csc unit tests ---

    #[test]
    fn test_quad_to_csc_diagonal() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        // c · x² with c = 3.0  →  Q[0][0] = 6.0
        let mut terms = HashMap::new();
        terms.insert((x, x), 3.0);
        let q = quad_to_csc(&terms, 1).unwrap();
        assert_eq!(q_entry(&q, 0, 0), 6.0, "diagonal: Q[0][0] should be 2*c");
    }

    #[test]
    fn test_quad_to_csc_cross_symmetric() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        // c · x·y with c = 5.0  →  Q[0][1] = Q[1][0] = 5.0
        let mut terms = HashMap::new();
        terms.insert(canon(x, y), 5.0);
        let q = quad_to_csc(&terms, 2).unwrap();
        assert_eq!(q_entry(&q, 0, 1), 5.0, "cross: Q[0][1] must equal c");
        assert_eq!(q_entry(&q, 1, 0), 5.0, "cross: Q[1][0] must equal c (symmetry)");
    }

    /// Sentinel: verify that `quad_to_csc` fills both Q[i][j] and Q[j][i].
    ///
    /// A broken upper-triangle-only implementation would produce nnz=1 for a
    /// cross term (only Q[0][1], missing Q[1][0]).  The correct implementation
    /// produces nnz=2.  We also confirm Q[1][0] == 0 in the broken Q, proving
    /// the missing entry would cause a wrong (asymmetric) matrix.
    #[test]
    fn test_symmetry_sentinel_quad_to_csc_fills_both_sides() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);

        // DSL: x * y  →  quad[(x,y)] = 1.0  →  must emit Q[0][1] = Q[1][0] = 1.0
        let mut terms = HashMap::new();
        terms.insert(canon(x, y), 5.0);
        let correct = quad_to_csc(&terms, 2).unwrap();

        // Both sides must be present and equal (symmetric fill):
        assert_eq!(q_entry(&correct, 0, 1), 5.0, "sentinel: Q[0][1] must be 5.0");
        assert_eq!(
            q_entry(&correct, 1, 0),
            5.0,
            "sentinel: Q[1][0] must be 5.0 — missing this entry is the classic bug"
        );
        // nnz = 2: one entry per triangle
        assert_eq!(correct.nnz(), 2, "sentinel: cross term must emit exactly 2 triplets");

        // No-op proof: a broken upper-triangle-only matrix has Q[1][0] == 0.
        let broken = CscMatrix::from_triplets(&[0], &[1], &[5.0], 2, 2).unwrap();
        assert_eq!(broken.nnz(), 1, "broken: only 1 triplet (missing lower side)");
        assert_eq!(
            q_entry(&broken, 1, 0),
            0.0,
            "broken: Q[1][0] is 0 — this is the missing-symmetry bug"
        );
        assert_ne!(
            q_entry(&broken, 0, 1),
            q_entry(&broken, 1, 0),
            "broken: Q is not symmetric (upper ≠ lower), confirming the bug exists"
        );
    }

    // --- Operator DSL tests ---

    #[test]
    fn test_var_times_var_is_quadratic() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let q = x * x;
        assert!(!q.is_linear());
        let q2 = x * y;
        assert!(!q2.is_linear());
    }

    #[test]
    fn test_pow2_equals_var_times_var() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let q1 = x * x;
        let q2 = x.pow2();
        // Both should produce the same quad entry
        assert_eq!(q1.quad.len(), 1);
        assert_eq!(q2.quad.len(), 1);
        let c1: f64 = q1.quad.values().copied().sum();
        let c2: f64 = q2.quad.values().copied().sum();
        assert!((c1 - c2).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_mul_quad_expr() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        // 3.0 * x * x: coefficient should be 3.0 in quad map
        let q = 3.0 * (x * x);
        assert_eq!(q.quad.len(), 1);
        let c: f64 = q.quad.values().copied().sum();
        assert!((c - 3.0).abs() < 1e-12, "scalar mul: coefficient should be 3.0, got {c}");
    }

    #[test]
    fn test_expression_times_var() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        // (2.0 * x) * y  →  QuadExpr with quad[(x,y)] = 2.0
        let expr = 2.0 * x;
        let q = expr * y;
        assert!(!q.is_linear());
        // Extract coefficient for the x-y pair
        let c: f64 = q.quad.values().copied().sum();
        assert!((c - 2.0).abs() < 1e-12, "expr*var: coefficient should be 2.0, got {c}");
    }

    #[test]
    fn test_add_quadexprs() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        // x*x + y*y should have two diagonal entries
        let q = x * x + y * y;
        assert_eq!(q.quad.len(), 2);
    }

    #[test]
    fn test_neg_quad_expr() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let q = -(x * x);
        let c: f64 = q.quad.values().copied().sum();
        assert!((c + 1.0).abs() < 1e-12, "neg: coefficient should be -1.0, got {c}");
    }

    #[test]
    fn test_mixed_quad_linear() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        // 2*x*x + 3*x*y + y   (quadratic + linear mixed)
        let q = 2.0 * x * x + 3.0 * x * y + y;
        assert!(!q.is_linear());
        // Should have two quad entries: (x,x) and (x,y)
        assert_eq!(q.quad.len(), 2);
        // Linear part should have y with coefficient 1.0
        let lin_y = q.linear.coefficient(y);
        assert!((lin_y - 1.0).abs() < 1e-12, "linear y coeff should be 1.0, got {lin_y}");
    }

    // --- Model solve tests (through Model API) ---

    #[test]
    fn test_minimize_x_squared_with_lb() {
        // min x²  s.t. x ≥ 1  →  x* = 1, obj* = 1
        let mut model = Model::new("min_x2");
        let x = model.add_var("x", 1.0, f64::INFINITY);
        model.minimize(x * x);
        let result = model.solve().unwrap();
        assert_close(result[x], 1.0, "min x²: x*");
        assert_close(result.objective_value, 1.0, "min x²: obj*");
    }

    #[test]
    fn test_minimize_x_squared_plus_y_squared() {
        // min x² + y²  s.t. x + y = 2, x,y ≥ 0  →  x* = y* = 1, obj* = 2
        let mut model = Model::new("min_x2_y2");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(2.0));
        model.minimize(x * x + y * y);
        let result = model.solve().unwrap();
        assert_close(result[x], 1.0, "min x²+y²: x*");
        assert_close(result[y], 1.0, "min x²+y²: y*");
        assert_close(result.objective_value, 2.0, "min x²+y²: obj*");
    }

    #[test]
    fn test_minimize_pow2_api() {
        // Same as above via x.pow2() + y.pow2()
        let mut model = Model::new("pow2");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(2.0));
        model.minimize(x.pow2() + y.pow2());
        let result = model.solve().unwrap();
        assert_close(result.objective_value, 2.0, "pow2 API: obj*");
    }

    #[test]
    fn test_maximize_concave_qp() {
        // max -x² + 4x  s.t. x ≥ 0  (concave → unique interior max at x=2, obj=4)
        // Sign-flip check: Q must be negated for maximize.
        let mut model = Model::new("max_concave");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.maximize(-(x * x) + 4.0 * x);
        let result = model.solve().unwrap();
        assert_close(result[x], 2.0, "max -x²+4x: x*");
        assert_close(result.objective_value, 4.0, "max -x²+4x: obj*");
    }

    #[test]
    fn test_minimize_cross_term_q_symmetry() {
        // min x² + x·y + y²  s.t. x + y = 2, x,y ≥ 0
        // → x* = y* = 1, obj* = 1 + 1 + 1 = 3
        //
        // Symmetry proof: if Q[0][1] were set but Q[1][0] omitted (upper-triangle only),
        // the effective objective would be x² + y² + ½·x·y, giving obj* = 2.5 ≠ 3.
        // This test therefore FAILS under a broken (upper-triangle-only) implementation.
        let mut model = Model::new("cross_sym");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).eq_constraint(2.0));
        model.minimize(x * x + x * y + y * y);
        let result = model.solve().unwrap();
        let tol = 1e-3;
        assert!(
            (result[x] - 1.0).abs() < tol,
            "cross_sym: x* ≈ 1, got {}",
            result[x]
        );
        assert!(
            (result[y] - 1.0).abs() < tol,
            "cross_sym: y* ≈ 1, got {}",
            result[y]
        );
        assert!(
            (result.objective_value - 3.0).abs() < tol,
            "cross_sym: obj* ≈ 3 (symmetric Q fill required), got {}",
            result.objective_value
        );
    }

    #[test]
    fn test_mixed_quad_linear_solve() {
        // min x² - 4x  s.t. x ≥ 0  →  x* = 2, obj* = 4 - 8 = -4
        // Written as: minimize(x*x + (-4.0) * x)
        let mut model = Model::new("quad_linear");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(x * x + (-4.0) * x);
        let result = model.solve().unwrap();
        assert_close(result[x], 2.0, "quad+linear: x*");
        assert_close(result.objective_value, -4.0, "quad+linear: obj*");
    }

    #[test]
    fn test_scalar_multiple_quad_solve() {
        // min 2·x² - 8·x  s.t. x ≥ 0  →  x* = 2, obj* = 8 - 16 = -8
        let mut model = Model::new("2x2_8x");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        model.minimize(2.0 * x * x + (-8.0) * x);
        let result = model.solve().unwrap();
        assert_close(result[x], 2.0, "2x²-8x: x*");
        assert_close(result.objective_value, -8.0, "2x²-8x: obj*");
    }

    #[test]
    fn test_dsl_qp_solves_correctly() {
        // Verify DSL minimize(x*x + y*y) gives correct answer
        // min x² + y²  s.t. x+y=3, x,y≥0  →  x=y=1.5, obj=4.5
        let mut m = Model::new("dsl");
        let x = m.add_var("x", 0.0, f64::INFINITY);
        let y = m.add_var("y", 0.0, f64::INFINITY);
        m.add_constraint((x + y).eq_constraint(3.0));
        m.minimize(x * x + y * y);
        let r = m.solve().unwrap();

        let tol = 1e-3;
        assert!((r[x] - 1.5).abs() < tol, "DSL x={} expected 1.5", r[x]);
        assert!((r[y] - 1.5).abs() < tol, "DSL y={} expected 1.5", r[y]);
        assert!((r.objective_value - 4.5).abs() < tol, "DSL obj={} expected 4.5", r.objective_value);
    }

    #[test]
    fn test_linear_objective_still_works_after_quad_change() {
        // minimize(x) should work normally via Into<QuadExpr> (pure-linear path)
        let mut model = Model::new("lin");
        let x = model.add_var("x", 2.0, 10.0);
        model.minimize(x);
        let result = model.solve().unwrap();
        assert_close(result[x], 2.0, "linear min x: x*");
    }

    #[test]
    fn test_from_expression_into_quad_expr() {
        // model.minimize(2.0 * x + y) — Expression → QuadExpr via From
        let mut model = Model::new("lin_expr");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, 10.0);
        model.add_constraint((x + y).geq(3.0));
        model.minimize(2.0 * x + y);  // Expression into QuadExpr (no quad terms)
        let result = model.solve().unwrap();
        assert_close(result[x], 0.0, "linear via QuadExpr: x*");
        assert_close(result[y], 3.0, "linear via QuadExpr: y*");
    }

    // --- P3.1: ゼロ quad entry の pruning ---

    #[test]
    fn test_cancelled_quad_term_is_linear() {
        // x*y - x*y should cancel to zero quad terms → is_linear() == true
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let q = x * y - x * y;
        assert!(q.is_linear(), "x*y - x*y should cancel to is_linear() == true");
    }

    #[test]
    fn test_zero_scalar_mul_is_linear() {
        // 0.0 * (x * x) should prune the quad entry → is_linear() == true
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let q = 0.0 * (x * x);
        assert!(q.is_linear(), "0.0 * x*x should prune to is_linear() == true");
    }

    #[test]
    fn test_cancelled_quad_routes_to_lp() {
        // x*y - x*y (pure linear 0) minimized: routes to LP, not QP
        // With only constant, any feasible x,y is optimal with obj=0.
        let mut model = Model::new("cancel_route");
        let x = model.add_var("x", 2.0, 2.0);
        let y = model.add_var("y", 3.0, 3.0);
        model.minimize(x * y - x * y + 1.0);  // = 1.0 (constant only, LP path)
        let result = model.solve().unwrap();
        assert!((result.objective_value - 1.0).abs() < TOL,
            "cancelled quad routes to LP: obj should be 1.0, got {}", result.objective_value);
    }

    // --- P3.3: coverage gap — NaN 係数 / indefinite QP ---

    #[test]
    fn test_nan_quad_coefficient_gives_error() {
        // NaN coefficient in quadratic term should produce an error at solve time.
        let mut model = Model::new("nan_q");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let q_expr = f64::NAN * (x * x);
        model.minimize(q_expr);
        let result = model.solve();
        assert!(
            result.is_err(),
            "NaN quad coefficient should produce an error, got Ok"
        );
    }

    #[test]
    fn test_indefinite_qp_no_silent_optimal() {
        use crate::SolutionProof;
        // min x·y  s.t. x+y≥1, x,y≥0 — indefinite (non-convex) QP.
        // Must NOT silently claim GlobalOptimal.
        let mut model = Model::new("indef");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        model.add_constraint((x + y).geq(1.0));
        model.minimize(x * y);
        let result = model.solve();
        match result {
            Ok(r) => {
                assert_ne!(
                    r.proof,
                    SolutionProof::GlobalOptimal,
                    "indefinite QP must not claim global optimality"
                );
            }
            Err(_) => {
                // Error (e.g. NonConvex) is also acceptable for indefinite QP
            }
        }
    }

    // ---------------------------------------------------------------------------
    // P2-c: zero-coefficient prune via single chokepoint (insert_quad_term)
    // ---------------------------------------------------------------------------

    // `(x - x) * y` — Expression has coef[x]=0, must not enter quad map.
    #[test]
    fn test_zero_coef_expr_times_var_is_linear() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let q = (x - x) * y;
        assert!(q.is_linear(), "(x-x)*y must be is_linear(); quad.len()={}", q.quad.len());
    }

    // `(x + x - 2*x) * y` — three-way cancellation.
    #[test]
    fn test_multi_cancel_expr_times_var_is_linear() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let y = model.add_var("y", 0.0, f64::INFINITY);
        let q = (x + x + ((-2.0) * x)) * y;
        assert!(q.is_linear(), "(x+x-2x)*y must be is_linear(); quad.len()={}", q.quad.len());
    }

    // x*x - x*x: merge cancellation still works (existing test extended).
    #[test]
    fn test_quad_sub_self_is_linear() {
        let mut model = Model::new("m");
        let x = model.add_var("x", 0.0, f64::INFINITY);
        let q = x * x - x * x;
        assert!(q.is_linear(), "x*x - x*x must cancel to is_linear()");
        assert_eq!(q.quad.len(), 0, "quad map must be empty after cancellation");
    }

    // ---------------------------------------------------------------------------
    // P2-d: cross-model variable validation in apply_objective
    // ---------------------------------------------------------------------------

    // Diagonal term from another model must be rejected.
    #[test]
    fn test_p2d_cross_model_diagonal_rejected() {
        use crate::ModelError;
        let mut m1 = Model::new("m1");
        let x1 = m1.add_var("x", 0.0, f64::INFINITY);

        let mut m2 = Model::new("m2");
        // x1 has m1's model_id; minimizing it in m2 must error.
        m2.minimize(x1 * x1);
        let result = m2.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-d: cross-model diagonal must give InvalidInput, got {result:?}"
        );
    }

    // Cross term mixing variables from two models must be rejected.
    #[test]
    fn test_p2d_cross_model_mixed_term_rejected() {
        use crate::ModelError;
        let mut m1 = Model::new("m1");
        let x1 = m1.add_var("x", 0.0, f64::INFINITY);

        let mut m2 = Model::new("m2");
        let y2 = m2.add_var("y", 0.0, f64::INFINITY);

        // x1 belongs to m1, y2 belongs to m2; the cross term is invalid for both.
        m1.minimize(x1 * y2);
        let result = m1.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-d: cross-model cross-term must give InvalidInput, got {result:?}"
        );
    }

    // Sanity: same-model variable works correctly (no false positive).
    #[test]
    fn test_p2d_same_model_accepted() {
        let mut model = Model::new("sanity");
        let x = model.add_var("x", 1.0, f64::INFINITY);
        model.minimize(x * x);
        let result = model.solve();
        assert!(result.is_ok(), "P2-d: same-model quad must be accepted, got {result:?}");
    }

    // maximize path: cross-model rejection via maximize (not just minimize).
    #[test]
    fn test_p2d_cross_model_maximize_rejected() {
        use crate::ModelError;
        let mut m1 = Model::new("m1");
        let x1 = m1.add_var("x", 0.0, 5.0);

        let mut m2 = Model::new("m2");
        m2.maximize(x1 * x1);
        let result = m2.solve();
        assert!(
            matches!(result, Err(ModelError::InvalidInput(_))),
            "P2-d: cross-model maximize must give InvalidInput, got {result:?}"
        );
    }
}
