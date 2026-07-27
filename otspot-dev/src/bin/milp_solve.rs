//! Solve a MILP from an MPS file with otspot's branch-and-bound MILP solver.
//!
//! Reads integer variables from INTORG/INTEND markers and BV/LI/UI bounds
//! (see `io::mps::parse_milp`). Prints a small key:value report parseable by the
//! MILP-vs-HiGHS comparison harness.
//!
//! Usage:
//!   `cargo run --release --bin milp_solve -- <file.mps> [--timeout <secs>] [--eps <tol>] [--cuts|--no-cuts] [--cut-rounds N] [--symmetry|--no-symmetry]`

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use otspot_core::options::{MipConfig, SolverOptions, Tolerance};
use otspot_core::{solve_milp_with_stats, MipStats};
use otspot_io::mps::parse_milp_file;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let cli = match parse_args(std::env::args().skip(1)) {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(cli.timeout_secs);
    opts.tolerance = Some(Tolerance::Custom(cli.eps));
    let cfg = mip_config_from_cli(&cli);
    let path = cli.path;

    let milp = match parse_milp_file(Path::new(&path)) {
        Ok(m) => m,
        Err(e) => {
            println!("file: {path}");
            println!("status: PARSE_ERROR");
            println!("error: {e}");
            return ExitCode::from(1);
        }
    };

    let n_vars = milp.num_vars();
    let n_cons = milp.lp.num_constraints;
    let n_int = milp.integer_vars.len();

    let profile = otspot_core::diag::lp_scale_profile_enabled();
    if profile {
        otspot_core::diag::reset_lp_scale_profile();
        otspot_core::diag::reset_simplex_fallback_profile();
    }

    let start = Instant::now();
    let (res, stats) = solve_milp_with_stats(&milp, &opts, &cfg);
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("file: {path}");
    println!("n_vars: {n_vars}");
    println!("n_cons: {n_cons}");
    println!("n_int: {n_int}");
    println!("symmetry: {}", cfg.symmetry);
    println!("status: {:?}", res.status);
    if res.objective.is_finite() {
        println!("objective: {:.9}", res.objective);
    } else {
        println!("objective: {}", res.objective);
    }
    println!("wall_ms: {wall_ms:.3}");
    println!("root_lp_bound: {}", stats.root_lp_bound);
    println!("nodes: {}", stats.nodes_processed);
    println!("incumbent_updates: {}", stats.incumbent_updates);
    println!("fp_incumbent_found: {}", stats.fp_incumbent_found);
    println!("max_depth: {}", stats.max_depth_seen);
    println!("pruned: {}", stats.pruned);
    println!("propagation_pruned: {}", stats.propagation_pruned);
    println!(
        "conflict_clauses_learned: {}",
        stats.conflict_clauses_learned
    );
    println!("conflict_pruned: {}", stats.conflict_pruned);
    println!("rc_vars_fixed: {}", stats.rc_vars_fixed);
    println!("rens_calls: {}", stats.rens_calls);
    println!("rens_improvements: {}", stats.rens_improvements);
    println!("rins_calls: {}", stats.rins_calls);
    println!("rins_improvements: {}", stats.rins_improvements);
    println!("local_branching_calls: {}", stats.local_branching_calls);
    println!(
        "local_branching_improvements: {}",
        stats.local_branching_improvements
    );
    println!("tree_cut_rounds: {}", stats.tree_cut_rounds);
    println!("lp_iters_total: {}", stats.lp_iters_total);
    println!("strong_branch_iters: {}", stats.strong_branch_iters);
    println!("rins_iters: {}", stats.rins_iters);
    println!("rens_iters: {}", stats.rens_iters);
    println!("local_branching_iters: {}", stats.local_branching_iters);
    println!("tree_cut_iters: {}", stats.tree_cut_iters);
    println!("lp_presolve_us: {}", stats.lp_presolve_us_total);
    println!("lp_solve_us: {}", stats.lp_solve_us_total);
    println!("lp_postsolve_us: {}", stats.lp_postsolve_us_total);
    // Node-level time attribution (Phase 0: measurement only, unconditional —
    // not gated behind OTSPOT_LP_SOLVE_PROFILE like the finer-grained
    // scale/fallback counters below). `node_propagation_us` predates Phase 0
    // but is printed here too because the attribution buckets below are only
    // complete (i.e. sum to ~wall_us) when it is included.
    let wall_us = wall_ms * 1000.0;
    println!("node_propagation_us: {}", stats.node_propagation_us);
    println!("tree_cut_us: {}", stats.tree_cut_us);
    println!("rins_us: {}", stats.rins_us);
    println!("rens_us: {}", stats.rens_us);
    println!("local_branching_us: {}", stats.local_branching_us);
    println!("branch_select_us: {}", stats.branch_select_us);
    println!("conflict_us: {}", stats.conflict_us);
    println!("node_loop_other_us: {}", stats.node_loop_other_us);
    println!("sub_mip_nodes_total: {}", stats.sub_mip_nodes_total);
    println!("wall_us: {wall_us:.0}");
    println!(
        "lp_solve_us_pct_wall: {:.2}",
        pct_of_wall(stats.lp_solve_us_total, wall_us)
    );
    println!(
        "node_propagation_us_pct_wall: {:.2}",
        pct_of_wall(stats.node_propagation_us, wall_us)
    );
    println!(
        "tree_cut_us_pct_wall: {:.2}",
        pct_of_wall(stats.tree_cut_us, wall_us)
    );
    println!(
        "rins_us_pct_wall: {:.2}",
        pct_of_wall(stats.rins_us, wall_us)
    );
    println!(
        "rens_us_pct_wall: {:.2}",
        pct_of_wall(stats.rens_us, wall_us)
    );
    println!(
        "local_branching_us_pct_wall: {:.2}",
        pct_of_wall(stats.local_branching_us, wall_us)
    );
    println!(
        "branch_select_us_pct_wall: {:.2}",
        pct_of_wall(stats.branch_select_us, wall_us)
    );
    println!(
        "conflict_us_pct_wall: {:.2}",
        pct_of_wall(stats.conflict_us, wall_us)
    );
    println!(
        "node_loop_other_us_pct_wall: {:.2}",
        pct_of_wall(stats.node_loop_other_us, wall_us)
    );
    let attribution_covered_us = loop_attribution_covered_us(&stats);
    println!("attribution_covered_us: {attribution_covered_us}");
    println!(
        "attribution_covered_pct_wall: {:.2}",
        pct_of_wall(attribution_covered_us, wall_us)
    );
    // Root-inclusive variant (P2-4): folds in root-level fp_us/root_cut_us/
    // root_probing_us/root_symmetry_us (Codex review), which the loop-only
    // attribution_covered_us above omits by design (it only covers the B&B
    // loop itself). See tests/mip_bnb_attribution.rs's
    // attribution_covers_loop_wall_clock vs.
    // attribution_covers_wall_clock_including_root_overhead for why both
    // views matter: once heuristics/separation are throttled the loop can
    // become fast enough that root-level presolve/FP/cut time is no longer
    // a negligible fraction of total wall clock.
    let attribution_covered_us_root_inclusive =
        root_inclusive_attribution_covered_us(attribution_covered_us, &stats);
    println!("attribution_covered_us_root_inclusive: {attribution_covered_us_root_inclusive}");
    println!(
        "attribution_covered_pct_wall_root_inclusive: {:.2}",
        pct_of_wall(attribution_covered_us_root_inclusive, wall_us)
    );
    if profile {
        println!("lp_solve_us_root: {}", stats.lp_solve_us_root);
        println!("lp_solve_us_desc: {}", stats.lp_solve_us_desc);
        println!("lp_scale_us_root: {}", stats.lp_scale_us_root);
        println!("lp_scale_us_desc: {}", stats.lp_scale_us_desc);
        println!("lp_scale_calls_root: {}", stats.lp_scale_calls_root);
        println!("lp_scale_calls_desc: {}", stats.lp_scale_calls_desc);
        println!("fp_us: {}", stats.fp_us);
        println!("root_cut_us: {}", stats.root_cut_us);
        println!("root_probing_us: {}", stats.root_probing_us);
        println!("root_symmetry_us: {}", stats.root_symmetry_us);
        println!("strong_branch_calls: {}", stats.strong_branch_calls);
        println!(
            "strong_branch_candidates: {}",
            stats.strong_branch_candidates
        );
        println!("strong_branch_lp_solves: {}", stats.strong_branch_lp_solves);
        println!("strong_branch_us: {}", stats.strong_branch_us);
        println!(
            "fallback_ub_violation_out_of_scope: {}",
            stats.fallback_ub_violation_out_of_scope
        );
        println!(
            "fallback_phase1_bound_violation: {}",
            stats.fallback_phase1_bound_violation
        );
        println!(
            "fallback_crash_infeasible: {}",
            stats.fallback_crash_infeasible
        );
    }
    if stats.nodes_processed > 0 {
        let n = stats.nodes_processed as f64;
        println!(
            "per_node_us: presolve={:.1} solve={:.1} postsolve={:.1}",
            stats.lp_presolve_us_total as f64 / n,
            stats.lp_solve_us_total as f64 / n,
            stats.lp_postsolve_us_total as f64 / n,
        );
    }

    ExitCode::SUCCESS
}

/// Percentage of `wall_us` that `part_us` accounts for; `0.0` when `wall_us`
/// is non-positive (e.g. a sub-millisecond solve rounding to 0).
fn pct_of_wall(part_us: u64, wall_us: f64) -> f64 {
    if wall_us > 0.0 {
        part_us as f64 / wall_us * 100.0
    } else {
        0.0
    }
}

/// Wall-clock microseconds covered by named B&B-loop attribution buckets
/// (Codex review, P2: `lp_presolve_us_total`/`lp_postsolve_us_total` were
/// previously omitted, undercounting whenever LP presolve/postsolve took
/// non-negligible per-node time).
fn loop_attribution_covered_us(stats: &MipStats) -> u64 {
    stats
        .lp_presolve_us_total
        .saturating_add(stats.lp_solve_us_total)
        .saturating_add(stats.lp_postsolve_us_total)
        .saturating_add(stats.node_propagation_us)
        .saturating_add(stats.tree_cut_us)
        .saturating_add(stats.rins_us)
        .saturating_add(stats.rens_us)
        .saturating_add(stats.local_branching_us)
        .saturating_add(stats.branch_select_us)
        .saturating_add(stats.conflict_us)
        .saturating_add(stats.node_loop_other_us)
}

/// [`loop_attribution_covered_us`] plus root-level setup time (feasibility
/// pump, root cuts, root bound-tightening probing, static symmetry
/// breaking — the latter two added per Codex review, P2), which the
/// loop-only figure omits by design.
fn root_inclusive_attribution_covered_us(loop_covered_us: u64, stats: &MipStats) -> u64 {
    loop_covered_us
        .saturating_add(stats.fp_us)
        .saturating_add(stats.root_cut_us)
        .saturating_add(stats.root_probing_us)
        .saturating_add(stats.root_symmetry_us)
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    path: String,
    timeout_secs: f64,
    eps: f64,
    cuts: bool,
    cut_rounds: usize,
    symmetry: Option<bool>,
    /// Ablation: force in-tree cut separation off regardless of `cuts`/`--cuts`.
    no_tree_cuts: bool,
    /// Ablation: force the RINS heuristic off.
    no_rins: bool,
    /// Ablation: force the RENS heuristic off.
    no_rens: bool,
    /// Ablation: force the local-branching heuristic off.
    no_local_branching: bool,
    /// Ablation: override `MipConfig::max_nodes` (node-count budget cap).
    max_nodes: Option<usize>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliArgs, String> {
    let mut path: Option<String> = None;
    let mut timeout_secs = 100.0_f64;
    let mut eps = 1e-6_f64;
    let mut cuts = true;
    let mut cut_rounds = 0usize;
    let mut symmetry: Option<bool> = None;
    let mut no_tree_cuts = false;
    let mut no_rins = false;
    let mut no_rens = false;
    let mut no_local_branching = false;
    let mut max_nodes: Option<usize> = None;
    let args: Vec<String> = args.into_iter().collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--timeout" => {
                i += 1;
                let value = args.get(i).ok_or("error: --timeout requires a value")?;
                timeout_secs = value
                    .parse()
                    .map_err(|_| format!("error: invalid --timeout value: {value}"))?;
            }
            "--eps" => {
                i += 1;
                let value = args.get(i).ok_or("error: --eps requires a value")?;
                eps = value
                    .parse()
                    .map_err(|_| format!("error: invalid --eps value: {value}"))?;
            }
            "--cuts" => cuts = true,
            "--no-cuts" => cuts = false,
            "--cut-rounds" => {
                i += 1;
                let value = args.get(i).ok_or("error: --cut-rounds requires a value")?;
                cut_rounds = value
                    .parse()
                    .map_err(|_| format!("error: invalid --cut-rounds value: {value}"))?;
                cuts = true;
            }
            "--symmetry" => symmetry = Some(true),
            "--no-symmetry" => symmetry = Some(false),
            "--no-tree-cuts" => no_tree_cuts = true,
            "--no-rins" => no_rins = true,
            "--no-rens" => no_rens = true,
            "--no-local-branching" => no_local_branching = true,
            "--max-nodes" => {
                i += 1;
                let value = args.get(i).ok_or("error: --max-nodes requires a value")?;
                max_nodes = Some(
                    value
                        .parse()
                        .map_err(|_| format!("error: invalid --max-nodes value: {value}"))?,
                );
            }
            other => path = Some(other.to_string()),
        }
        i += 1;
    }

    let path = path.ok_or_else(|| {
        "usage: milp_solve <file.mps> [--timeout <secs>] [--eps <tol>] [--cuts|--no-cuts] \
         [--cut-rounds N] [--symmetry|--no-symmetry] [--no-tree-cuts] [--no-rins] [--no-rens] \
         [--no-local-branching] [--max-nodes N]"
            .to_string()
    })?;
    Ok(CliArgs {
        path,
        timeout_secs,
        eps,
        cuts,
        cut_rounds,
        symmetry,
        no_tree_cuts,
        no_rins,
        no_rens,
        no_local_branching,
        max_nodes,
    })
}

fn mip_config_from_cli(cli: &CliArgs) -> MipConfig {
    let mut cfg = MipConfig::default();
    cfg.gap_tol = cli.eps;
    cfg.integer_feas_tol = cli.eps;
    configure_cuts(&mut cfg, cli.cuts, cli.cut_rounds);
    if let Some(symmetry) = cli.symmetry {
        cfg.symmetry = symmetry;
    }
    if cli.no_tree_cuts {
        cfg.tree_cuts = false;
    }
    if cli.no_rins {
        cfg.rins_enabled = false;
    }
    if cli.no_rens {
        cfg.rens_enabled = false;
    }
    if cli.no_local_branching {
        cfg.local_branching_enabled = false;
    }
    if let Some(max_nodes) = cli.max_nodes {
        cfg.max_nodes = max_nodes;
    }
    cfg
}

fn configure_cuts(cfg: &mut MipConfig, cuts: bool, cut_rounds: usize) {
    cfg.cuts = cuts;
    cfg.tree_cuts = cuts;
    cfg.max_cut_rounds = cut_rounds;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_cuts_disables_root_and_tree_cuts() {
        let mut cfg = MipConfig::default();
        cfg.cuts = true;
        cfg.tree_cuts = true;
        cfg.max_cut_rounds = 7;

        configure_cuts(&mut cfg, false, 0);

        assert!(!cfg.cuts);
        assert!(!cfg.tree_cuts);
        assert_eq!(cfg.max_cut_rounds, 0);
    }

    /// Codex review (P2): `lp_presolve_us_total`/`lp_postsolve_us_total` must
    /// be part of the loop-attribution coverage sum, not just
    /// `lp_solve_us_total` — omitting them undercounts wall clock whenever
    /// per-node LP presolve/postsolve takes non-negligible time.
    ///
    /// Sentinel: removing either ``
    /// or `.saturating_add(stats.lp_postsolve_us_total)` from
    /// `loop_attribution_covered_us` makes this FAIL.
    #[test]
    fn loop_attribution_includes_lp_presolve_and_postsolve() {
        let mut stats = MipStats::default();
        stats.lp_presolve_us_total = 100;
        stats.lp_solve_us_total = 200;
        stats.lp_postsolve_us_total = 300;
        stats.node_propagation_us = 10;
        stats.tree_cut_us = 20;
        stats.rins_us = 30;
        stats.rens_us = 40;
        stats.local_branching_us = 50;
        stats.branch_select_us = 60;
        stats.conflict_us = 70;
        stats.node_loop_other_us = 80;
        assert_eq!(loop_attribution_covered_us(&stats), 960);
    }

    /// Codex review (P2): `root_probing_us`/`root_symmetry_us` must be part
    /// of the root-inclusive coverage sum, not just `fp_us`/`root_cut_us`.
    ///
    /// Sentinel: removing either `.saturating_add(stats.root_probing_us)` or
    /// `.saturating_add(stats.root_symmetry_us)` from
    /// `root_inclusive_attribution_covered_us` makes this FAIL.
    #[test]
    fn root_inclusive_attribution_includes_probing_and_symmetry() {
        let mut stats = MipStats::default();
        stats.fp_us = 1;
        stats.root_cut_us = 2;
        stats.root_probing_us = 3;
        stats.root_symmetry_us = 4;
        assert_eq!(root_inclusive_attribution_covered_us(1000, &stats), 1010);
    }

    #[test]
    fn no_cuts_cli_option_disables_root_and_tree_cuts() {
        let cli = parse_args(["tiny.mps".to_string(), "--no-cuts".to_string()]).unwrap();
        let cfg = mip_config_from_cli(&cli);

        assert!(!cfg.cuts);
        assert!(!cfg.tree_cuts);
    }

    #[test]
    fn symmetry_defaults_to_config_default_when_unset() {
        let cli = parse_args(["tiny.mps".to_string()]).unwrap();
        assert_eq!(cli.symmetry, None);
        let cfg = mip_config_from_cli(&cli);
        assert_eq!(cfg.symmetry, MipConfig::default().symmetry);
    }

    #[test]
    fn symmetry_cli_overrides_default_both_ways() {
        let off = parse_args(["tiny.mps".to_string(), "--no-symmetry".to_string()]).unwrap();
        assert_eq!(off.symmetry, Some(false));
        assert!(!mip_config_from_cli(&off).symmetry);

        let on = parse_args([
            "tiny.mps".to_string(),
            "--no-symmetry".to_string(),
            "--symmetry".to_string(),
        ])
        .unwrap();
        assert_eq!(on.symmetry, Some(true));
        assert!(mip_config_from_cli(&on).symmetry);
    }

    #[test]
    fn cut_rounds_cli_option_reenables_root_and_tree_cuts() {
        let cli = parse_args([
            "tiny.mps".to_string(),
            "--no-cuts".to_string(),
            "--cut-rounds".to_string(),
            "3".to_string(),
        ])
        .unwrap();
        let cfg = mip_config_from_cli(&cli);

        assert!(cfg.cuts);
        assert!(cfg.tree_cuts);
        assert_eq!(cfg.max_cut_rounds, 3);
    }

    // -----------------------------------------------------------------
    // Phase 0 ablation flags: --no-tree-cuts / --no-rins / --no-rens /
    // --no-local-branching / --max-nodes
    // -----------------------------------------------------------------

    #[test]
    fn no_tree_cuts_disables_tree_cuts_only_leaving_root_cuts_on() {
        let cli = parse_args(["tiny.mps".to_string(), "--no-tree-cuts".to_string()]).unwrap();
        let cfg = mip_config_from_cli(&cli);

        assert!(cfg.cuts, "root cutting-plane pass must stay enabled");
        assert!(
            !cfg.tree_cuts,
            "--no-tree-cuts must disable in-tree separation"
        );
    }

    #[test]
    fn no_rins_disables_rins_only() {
        let cli = parse_args(["tiny.mps".to_string(), "--no-rins".to_string()]).unwrap();
        let cfg = mip_config_from_cli(&cli);

        assert!(!cfg.rins_enabled);
        assert!(cfg.rens_enabled);
        assert!(cfg.local_branching_enabled);
    }

    #[test]
    fn no_rens_disables_rens_only() {
        let cli = parse_args(["tiny.mps".to_string(), "--no-rens".to_string()]).unwrap();
        let cfg = mip_config_from_cli(&cli);

        assert!(cfg.rins_enabled);
        assert!(!cfg.rens_enabled);
        assert!(cfg.local_branching_enabled);
    }

    #[test]
    fn no_local_branching_disables_local_branching_only() {
        let cli = parse_args(["tiny.mps".to_string(), "--no-local-branching".to_string()]).unwrap();
        let cfg = mip_config_from_cli(&cli);

        assert!(cfg.rins_enabled);
        assert!(cfg.rens_enabled);
        assert!(!cfg.local_branching_enabled);
    }

    #[test]
    fn all_ablation_flags_combine_independently() {
        let cli = parse_args([
            "tiny.mps".to_string(),
            "--no-tree-cuts".to_string(),
            "--no-rins".to_string(),
            "--no-rens".to_string(),
            "--no-local-branching".to_string(),
        ])
        .unwrap();
        let cfg = mip_config_from_cli(&cli);

        assert!(!cfg.tree_cuts);
        assert!(!cfg.rins_enabled);
        assert!(!cfg.rens_enabled);
        assert!(!cfg.local_branching_enabled);
        assert!(
            cfg.cuts,
            "root cuts flag is independent of the ablation flags"
        );
    }

    #[test]
    fn max_nodes_cli_overrides_default_node_budget() {
        let default_cli = parse_args(["tiny.mps".to_string()]).unwrap();
        assert_eq!(default_cli.max_nodes, None);
        assert_eq!(
            mip_config_from_cli(&default_cli).max_nodes,
            MipConfig::default().max_nodes,
            "no --max-nodes ⇒ MipConfig's own default budget"
        );

        let cli = parse_args([
            "tiny.mps".to_string(),
            "--max-nodes".to_string(),
            "42".to_string(),
        ])
        .unwrap();
        assert_eq!(cli.max_nodes, Some(42));
        assert_eq!(mip_config_from_cli(&cli).max_nodes, 42);
    }

    /// NEW (P3-6): a non-numeric flag value returns a `Result::Err` (a
    /// process exit code from `main`'s `Err(message) => ... ExitCode::from(2)`
    /// path), not a panic.
    ///
    /// Sentinel: reverting any of these `.map_err(...)?` conversions back to
    /// `.expect(...)` makes `parse_args` panic instead of returning `Err`,
    /// failing this test (a panicked test shows as FAILED, not as an `Err`
    /// result to match against).
    #[test]
    fn invalid_numeric_flag_values_return_err_not_panic() {
        for flag in ["--timeout", "--eps", "--cut-rounds", "--max-nodes"] {
            let result = parse_args([
                "tiny.mps".to_string(),
                flag.to_string(),
                "not-a-number".to_string(),
            ]);
            assert!(
                result.is_err(),
                "{flag} with a non-numeric value must return Err, not panic; got {result:?}"
            );
        }
    }
}
