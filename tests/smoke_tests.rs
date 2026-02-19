/// Smoke tests: verify that key modules exist and are accessible
/// (対策6: cmd_094消失事故の再発防止)
#[cfg(test)]
mod smoke_tests {
    #[test]
    fn scaling_module_exists() {
        use solver::presolve::RuizScaler;
        let _ = std::mem::size_of::<RuizScaler>();
    }

    #[test]
    fn pricing_module_exists() {
        // SteepestEdgePricing は pub(crate) （内部実装）のため直接アクセス不可。
        // simplex::solve を通じて pricing が動作することを確認。
        use solver::problem::LpProblem;
        use solver::sparse::CscMatrix;
        use solver::simplex;
        let a = CscMatrix::new(0, 1);
        let prob = LpProblem::new_general(
            vec![1.0], a, vec![], vec![], vec![(0.0, 1.0)], None
        ).unwrap();
        let _ = simplex::solve(&prob);
    }
}
