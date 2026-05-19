//! IP-PMM tuning constants and per-call state.

/// 論文 §5.1 推奨初期値。
pub(super) const RHO_INIT: f64 = 8.0;
pub(super) const DELTA_INIT: f64 = 8.0;

/// warm start safe guard.
/// μ floor: x·y=0 / s=0 を渡された場合に central path から外れないため。
pub(super) const WARM_MU_MIN: f64 = 1e-8;
/// 両端有限 box では range × WARM_BOUND_REL_MARGIN を interior 余白にとる
/// (cold init の 1% 余白より tighter、warm 値を最大限尊重する)。
pub(super) const WARM_BOUND_REL_MARGIN: f64 = 1e-6;
/// 半側有限 bound では cold init と同等の絶対 1.0 余白。
pub(super) const WARM_BOUND_ABS_MARGIN: f64 = 1.0;
/// 不等式行 s, y の boundary 上で σ=s/y が発散するため両側を floor。
pub(super) const WARM_SY_MIN: f64 = 1e-8;

/// 5% 以上の残差減少を改善とみなす (Gondzio2021 MATLAB)。
pub(super) const PMM_IMPROVE_THRESHOLD: f64 = 0.95;
pub(super) const PMM_SLOW_RATE: f64 = 2.0 / 3.0;

/// μ が実質 0 と判定する境界 (機械精度直上)。
pub(super) const MU_ZERO_THRESHOLD: f64 = 1e-15;

pub(super) const LDL_REG_RETRY_MAX: usize = 10;
pub(super) const LDL_REG_GROWTH: f64 = 10.0;
pub(super) const LDL_REG_CEILING: f64 = 1.0;
pub(super) const LDL_FALLBACK_DELTA_MIN: f64 = 1e-2;

/// tight eps で正常な小 alpha を stall 扱いしないため eps スケールで閾値を緩める。
pub(super) fn alpha_stall_eps_for(eps: f64) -> f64 {
    (eps * 1e-2).max(1e-14)
}
pub(super) const ALPHA_STALL_N: usize = 5;
pub(super) const ALPHA_DEADLOCK_N: usize = 20;

/// alpha > 0 でも residual が改善しない病理 (n=250k 級) 用の停滞窓。
/// 50 iter は典型収束速度 0.5^50 ≈ 9e-16 を踏まえた観測窓、REL_DEC=1e-3 は数値飽和判定。
pub(super) const RESIDUAL_STALL_WINDOW: usize = 50;
pub(super) const RESIDUAL_STALL_REL_DEC: f64 = 1e-3;

pub(super) struct PmmState {
    pub(super) x_ref: Vec<f64>,
    pub(super) y_ref: Vec<f64>,
    pub(super) rho: f64,
    pub(super) delta: f64,
    pub(super) prev_nr_p: f64,
    pub(super) prev_nr_d: f64,
}
