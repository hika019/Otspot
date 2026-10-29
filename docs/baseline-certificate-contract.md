# Baseline reference certificates

The baseline generator must establish an objective reference independently of
the solver under test. Certification targets relative error at most `1e-7`,
one tenth of the benchmark's default `1e-6` tolerance. The number of accepted
problems is a coverage result, not a correctness criterion.

## Model

The shared QPS parser defines one canonical model with binary64 coefficients,
including coefficient accumulation, selected named vectors, bounds, and the
objective constant. The checker interprets these binary64 values as exact
rational numbers. Every candidate generator, including the LISWET specialization,
must refer to this same model. Decimal-token and binary64 certificates must not
be mixed as certificates of an identical model.

## Proof

For a minimization problem, the checker establishes an interval `[L, U]`:

1. Prove that the symmetric Hessian is positive semidefinite. A tolerance on
   negative eigenvalues is not a proof. An exhausted proof budget is unknown.
2. Check every original row and variable bound for a rational primal point.
   Its exactly recomputed objective is an upper bound `U`.
3. Independently recompute reduced costs from a dual witness and linearization
   point, check multiplier and bound signs, and evaluate a rational weak-duality
   lower bound `L`.
4. Require `L <= U`. Exact KKT equality is a special case, not a requirement.
5. For the serialized CSV reference `r`, require
   `max(abs(r - L), abs(r - U)) / (1 + abs(r)) <= 1e-7`, accounting for both the
   decimal representation and its binary64 interpretation by benchmark readers.

The primal point and dual linearization point may differ. Solver status,
small residuals, active-set guesses, multiplier-weighted violations, and
producer-supplied reduced costs are not substitutes for these checks.

## Candidate search and provenance

Candidate generation may use numerical solvers, polishing, or structured exact
recovery. Residual and sensitivity scores only order unverified candidates.
Verified enclosures take priority, and successful termination depends on their
certified error. A failed recovery attempt cannot make an unverified value valid.

Each retained certification claim must be traceable to the input hash, checker
version, bounds, and reference checked. Revalidation compares every row with its
previous value. Failure or unknown status must not silently preserve an old
certification label; neither implies that the previous numerical value is wrong.
