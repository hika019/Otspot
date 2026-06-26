use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

#[derive(Clone, Copy)]
struct TraceConfig {
    enabled: bool,
    every: usize,
    max_lines: usize,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn trace_config() -> &'static TraceConfig {
    static CONFIG: OnceLock<TraceConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let enabled = std::env::var("OTSPOT_SIMPLEX_TRACE")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        TraceConfig {
            enabled,
            every: env_usize("OTSPOT_SIMPLEX_TRACE_EVERY", 1000),
            max_lines: env_usize("OTSPOT_SIMPLEX_TRACE_MAX_LINES", 2000),
        }
    })
}

#[inline]
fn basis_hash(basis: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    basis.len().hash(&mut hasher);
    basis.hash(&mut hasher);
    hasher.finish()
}

/// Lightweight iteration tracer enabled by env vars:
/// - OTSPOT_SIMPLEX_TRACE=1
/// - OTSPOT_SIMPLEX_TRACE_EVERY=1000
/// - OTSPOT_SIMPLEX_TRACE_MAX_LINES=2000
pub(super) struct IterTrace {
    tag: &'static str,
    cfg: TraceConfig,
    lines: usize,
    repeats: usize,
    seen_basis: HashMap<u64, usize>,
    last_obj: Option<f64>,
    no_obj_progress: usize,
    detail_lines: usize,
    pivots: usize,
    degenerate_pivots: usize,
    flips: usize,
}

// Env-gated diagnostic (OTSPOT_SIMPLEX_TRACE, default-off): the eprintln here are
// the intended output of this tracer, not stray production prints. Allow clippy's
// crate-wide print_stderr deny (and the audit no_eprintln gate keys on the same token).
#[allow(clippy::print_stderr)]
impl IterTrace {
    pub(super) fn new(tag: &'static str) -> Option<Self> {
        let cfg = *trace_config();
        if !cfg.enabled {
            return None;
        }
        Some(Self {
            tag,
            cfg,
            lines: 0,
            repeats: 0,
            seen_basis: HashMap::new(),
            last_obj: None,
            no_obj_progress: 0,
            detail_lines: 0,
            pivots: 0,
            degenerate_pivots: 0,
            flips: 0,
        })
    }

    /// Record one basis-changing pivot and its primal step length. A step at or
    /// below `tol` is a *degenerate* pivot (the basic solution does not move).
    /// The degenerate fraction distinguishes a degenerate stall (most pivots
    /// near-zero step) from genuinely slow pricing (large steps, few iters).
    pub(super) fn note_pivot(&mut self, step: f64, tol: f64) {
        self.pivots = self.pivots.saturating_add(1);
        if step <= tol {
            self.degenerate_pivots = self.degenerate_pivots.saturating_add(1);
        }
    }

    /// Record one bound-flip iteration (no basis change).
    pub(super) fn note_flip(&mut self) {
        self.flips = self.flips.saturating_add(1);
    }

    pub(super) fn log(&mut self, iter: usize, obj: f64, basis: &[usize], bland_mode: bool) {
        if self.lines >= self.cfg.max_lines {
            return;
        }

        let h = basis_hash(basis);
        let mut repeat_from: Option<usize> = None;
        if let Some(prev) = self.seen_basis.insert(h, iter) {
            self.repeats = self.repeats.saturating_add(1);
            repeat_from = Some(prev);
        }

        let mut improved = false;
        if let Some(prev) = self.last_obj {
            let eps = prev.abs().max(1.0) * 1e-12;
            improved = prev - obj > eps;
            if improved {
                self.no_obj_progress = 0;
            } else {
                self.no_obj_progress = self.no_obj_progress.saturating_add(1);
            }
        }
        self.last_obj = Some(obj);

        let force_line = repeat_from.is_some();
        if !force_line && !iter.is_multiple_of(self.cfg.every) {
            return;
        }

        self.lines = self.lines.saturating_add(1);
        if let Some(prev) = repeat_from {
            eprintln!(
                "[simplex-trace:{}] iter={} obj={:.9e} bland={} no_obj_prog={} repeat_basis_from={} period={}",
                self.tag,
                iter,
                obj,
                bland_mode,
                self.no_obj_progress,
                prev,
                iter.saturating_sub(prev)
            );
        } else {
            eprintln!(
                "[simplex-trace:{}] iter={} obj={:.9e} bland={} no_obj_prog={} improved={}",
                self.tag, iter, obj, bland_mode, self.no_obj_progress, improved
            );
        }
    }

    pub(super) fn log_ratio_test(
        &mut self,
        candidates: &[usize],
        ratios: &[f64],
        selected: Option<usize>,
        is_bland: bool,
    ) {
        if self.detail_lines >= self.cfg.max_lines {
            return;
        }
        self.detail_lines = self.detail_lines.saturating_add(1);
        let selected_text = selected
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string());
        eprintln!(
            "[simplex-trace:{}] ratio_test(candidates={:?}, ratios={:?}, selected={}, is_bland={})",
            self.tag, candidates, ratios, selected_text, is_bland
        );
    }

    pub(super) fn log_lex_perturbation(&mut self, delta: f64, effect: f64) {
        if self.detail_lines >= self.cfg.max_lines {
            return;
        }
        self.detail_lines = self.detail_lines.saturating_add(1);
        eprintln!(
            "[simplex-trace:{}] lex perturbation applied: delta={:.9e}, effect={:.9e}",
            self.tag, delta, effect
        );
    }

    pub(super) fn log_stall_bail(&mut self, iter: usize, obj: f64, trigger: usize) {
        if self.detail_lines >= self.cfg.max_lines {
            return;
        }
        self.detail_lines = self.detail_lines.saturating_add(1);
        eprintln!(
            "[simplex-trace:{}] cleanup stall bail: iter={} obj={:.9e} no_progress_trigger={}",
            self.tag, iter, obj, trigger
        );
    }
}

#[allow(clippy::print_stderr)] // env-gated diagnostic summary (see impl above)
impl Drop for IterTrace {
    fn drop(&mut self) {
        if !self.cfg.enabled {
            return;
        }
        let degen_frac = if self.pivots > 0 {
            self.degenerate_pivots as f64 / self.pivots as f64
        } else {
            0.0
        };
        eprintln!(
            "[simplex-trace:{}:summary] lines={} detail_lines={} repeats={} unique_basis={} \
             pivots={} degenerate_pivots={} degen_frac={:.4} flips={}",
            self.tag,
            self.lines,
            self.detail_lines,
            self.repeats,
            self.seen_basis.len(),
            self.pivots,
            self.degenerate_pivots,
            degen_frac,
            self.flips,
        );
    }
}
