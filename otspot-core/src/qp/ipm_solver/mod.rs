//! Mehrotra IPM (IP-PMM)。
//! 1 層 retry で eps を直線厳格化、status 変換は API 境界の 1 箇所に集約、KKT は元空間判定。

pub mod outcome;
pub mod kkt;
pub mod core;
pub mod attempt;

pub use attempt::solve_ipm;

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::print_stderr)]
mod tests {
    use super::*;
    use crate::io::qps::parse_qps;
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use std::path::Path;

    // test_ipm_hs21_cmp_full_solver moved to otspot-io/tests/bug_repro.rs (#28 dedup).

    #[test]
    fn test_v2_hs21() {
        let path = Path::new("data/maros_meszaros/HS21.QPS");
        if !path.exists() { return; }
        let prob = parse_qps(path).expect("parse HS21");
        let opts = SolverOptions::default();
        let r = solve_ipm(&prob, &opts);
        assert_eq!(r.status, SolveStatus::Optimal);
    }

    /// DPKLO1 が hang せず timeout/optimal で返ること。
    #[test]
    fn test_ipm_dpklo1() {
        let path = Path::new("data/maros_meszaros/DPKLO1.QPS");
        if !path.exists() {
            eprintln!("DPKLO1.QPS not found, skipping");
            return;
        }
        let prob = parse_qps(path).expect("parse DPKLO1");
        let opts = SolverOptions { timeout_secs: Some(5.0), ..Default::default() };
        let r = solve_ipm(&prob, &opts);
        eprintln!("DPKLO1 v2: status={:?} obj={} iters={}", r.status, r.objective, r.iterations);
        assert!(matches!(r.status, SolveStatus::Optimal | SolveStatus::Timeout));
    }
}
