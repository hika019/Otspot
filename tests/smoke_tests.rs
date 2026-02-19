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
        use solver::simplex::pricing::SteepestEdgePricing;
        let _ = std::mem::size_of::<SteepestEdgePricing>();
    }
}
