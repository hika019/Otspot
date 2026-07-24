//! Compatibility re-exports for the numerical linear-algebra layer.

pub use otspot_num::linalg::*;

#[cfg(test)]
mod tests {
    use super::kkt_solver::{KktConfig, BYTES_PER_L_ENTRY, DEFAULT_MEMORY_BUDGET_BYTES};
    use crate::options::IpmOptions;

    #[test]
    fn ipm_options_control_kkt_memory_budget() {
        let defaults = IpmOptions::default();
        assert_eq!(
            defaults.effective_max_l_nnz(),
            DEFAULT_MEMORY_BUDGET_BYTES / BYTES_PER_L_ENTRY
        );
        let small = IpmOptions {
            kkt_memory_budget_bytes: Some(1600),
            ..Default::default()
        };
        assert_eq!(small.effective_max_l_nnz(), 100);
    }

    #[test]
    fn kkt_config_default_matches_ipm_options_default() {
        let options = IpmOptions::default();
        let config = KktConfig {
            dd_ldl: options.dd_ldl,
            minres_ir: options.effective_minres_ir(),
            max_l_nnz: options.effective_max_l_nnz(),
        };
        let defaults = KktConfig::default();
        assert_eq!(config.dd_ldl, defaults.dd_ldl);
        assert_eq!(config.minres_ir, defaults.minres_ir);
        assert_eq!(config.max_l_nnz, defaults.max_l_nnz);
    }
}
