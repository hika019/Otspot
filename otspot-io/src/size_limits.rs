//! Shared upper bounds on parser-declared sizes (variable/constraint counts,
//! sparse-entry counts, ...) that would otherwise gate a `Vec` allocation
//! before the file's actual content has been read.
//!
//! Every CBF/QPLIB header count that sizes a `Vec` must be run through
//! [`check_declared_size`] immediately after it is read, before it is used
//! for anything else. Skipping this lets a few-byte file declare an
//! astronomical count and:
//! - abort the process outright (`Vec::with_capacity`/`vec![v; n]` call
//!   `handle_alloc_error` -> `abort()` on a request the allocator can't
//!   satisfy, which no `Result`/`catch_unwind` can intercept), or
//! - pass a bare `try_reserve` check (Linux overcommit lets the *virtual*
//!   reservation succeed) and then get OOM-killed once an eager fill
//!   (`.resize()`/`vec![v; n]`) actually pages the buffer in. Verified during
//!   this fix: a VAR block declaring 500,000,000 variables reserves without
//!   error but is killed by the cgroup OOM killer while the `c`/`bounds`
//!   buffers are filled (4-8 GB of real pages touched in one call).
//!
//! Capping the declared count up front avoids both failure modes without
//! relying on `try_reserve` to catch a request that was never legitimate.

/// Maximum accepted "one entry per variable/constraint" dimension: CBF `VAR`
/// total, CBF `CON` total, QPLIB "number of variables", QPLIB "number of
/// constraints".
///
/// Set from the largest verified real problem across this repo's test
/// corpora, with a ~10x margin:
/// - CBLIB `db-joint-soerensen.cbf`: 1,478,669 variables / 1,978,142 rows.
/// - QPLIB_9008: 1,009,306 variables / 989,604 constraints.
///
/// At this cap, the largest single dimension-sized buffer any parser fills
/// eagerly (QPLIB's `Vec<QcqpMatrix>`, 32 bytes/element, sized to at most
/// `2 * m`) tops out around 1.3 GB -- comfortably below the multi-GB range
/// where the overcommit-then-OOM-kill failure mode above was observed.
pub(crate) const MAX_DECLARED_DIMENSION: usize = 20_000_000;

/// Maximum accepted count of explicitly-listed sparse entries that reserve
/// `Vec` capacity before the loop reading them has validated any real file
/// content: QPLIB "number of quadratic terms in objective" (`nqobj`) and
/// "number of linear terms in all constraints" (`n_con_lin_terms`).
///
/// Largest verified real value: QPLIB_9008's 9,634,086 constraint linear
/// terms. 100,000,000 gives >10x margin. These counts only ever gate a
/// capacity *reservation* (never an eager fill -- the entries are pushed one
/// at a time from real tokens), so the margin can be looser than
/// [`MAX_DECLARED_DIMENSION`] without approaching the eager-fill OOM range.
pub(crate) const MAX_DECLARED_TERM_COUNT: usize = 100_000_000;

/// Validates a parser-declared count against `max` before it is used to size
/// any allocation. `context` names the field, for the error message.
pub(crate) fn check_declared_size(
    value: usize,
    max: usize,
    context: &str,
) -> Result<usize, String> {
    if value > max {
        Err(format!(
            "{context} {value} exceeds the maximum accepted value {max}"
        ))
    } else {
        Ok(value)
    }
}

/// Allocates a `Vec<T>` of `len` clones of `value` without letting the
/// allocator abort the process on an unsatisfiable request: `try_reserve`
/// converts allocator failure into `Err`, and the buffer is only ever
/// resized to `len` -- never larger -- so a caller that has already bounded
/// `len` via [`check_declared_size`] never eagerly fills more than that
/// bound's worth of memory.
pub(crate) fn try_vec_filled<T: Clone>(
    len: usize,
    value: T,
    context: &str,
) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|e| format!("cannot allocate {context} of length {len}: {e}"))?;
    out.resize(len, value);
    Ok(out)
}

/// Reserves capacity for a `Vec<T>` without filling it, converting allocator
/// failure into `Err` instead of aborting the process. Intended for buffers
/// that are then filled incrementally by a loop gated on real file content
/// (so the reservation itself, not an eager fill, is the only risk).
pub(crate) fn try_vec_with_capacity<T>(cap: usize, context: &str) -> Result<Vec<T>, String> {
    let mut out = Vec::new();
    out.try_reserve_exact(cap)
        .map_err(|e| format!("cannot allocate {context} of capacity {cap}: {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_declared_size_accepts_boundary_and_rejects_above() {
        assert_eq!(check_declared_size(10, 10, "x").unwrap(), 10);
        assert!(check_declared_size(11, 10, "x").is_err());
    }

    #[test]
    fn try_vec_filled_rejects_huge_length_gracefully() {
        let err = try_vec_filled(usize::MAX / 4, 0.0f64, "probe").unwrap_err();
        assert!(err.contains("probe"));
    }

    #[test]
    fn try_vec_with_capacity_rejects_huge_capacity_gracefully() {
        let err = try_vec_with_capacity::<f64>(usize::MAX / 4, "probe").unwrap_err();
        assert!(err.contains("probe"));
    }
}
