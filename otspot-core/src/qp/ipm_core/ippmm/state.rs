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
/// 半側有限 / 単側 bound の strict-interior 余白。
/// 絶対固定だと |b|≫1 で相対 0、|b|≪1 で warm を過剰に押し込む両極が出るため
/// `max(|b|, 1.0)` で scale 追従させる (floor=1 で原点付近の margin=0 退化を回避)。
pub(super) fn warm_bound_margin(bound: f64) -> f64 {
    WARM_BOUND_REL_MARGIN * bound.abs().max(1.0)
}
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

/// 残差小・gap 大の偽 Optimal を弾く duality gap 上限。
pub(super) const DUALITY_GAP_TOL: f64 = 1e-3;

/// reg_limit 下限と一段引下げ倍率。
pub(super) const REG_LIMIT_MIN: f64 = 1e-14;
pub(super) const REG_LIMIT_STEP: f64 = 1e-3;
/// initial_reg_limit のデフォルト値 (QP / LP)。
pub(super) const REG_LIMIT_INIT_QP: f64 = 5e-8;
pub(super) const REG_LIMIT_INIT_LP: f64 = 5e-10;

/// σ=s/y が非有限 (NaN/Inf) のときの fallback 上限。旧 `1/options.ipm.delta_min`
/// (delta_min=1e-8 固定) と数値的に同じ 1e8 を維持しつつ、削除された
/// `delta_min` オプション (正則化 floor と無関係な数値安全弁) から独立させた。
pub(super) const SIGMA_MAX_FALLBACK: f64 = 1e8;
/// prox 項が dual residual を支配と判定する比率。
pub(super) const PROX_DOMINATE_RATIO: f64 = 0.5;

/// pf-stagnation 検出窓 + 停滞判定比率。
pub(super) const PF_HISTORY_LEN: usize = 5;
pub(super) const PF_STUCK_RATIO: f64 = 0.95;

/// finite-but-huge 方向 (LDL blow-up) を弾く閾値。
pub(super) const DIRECTION_BLOWUP_THRESHOLD: f64 = 1e30;

/// false-positive 緩衝のための連続 infeasible 検出回数。
pub(super) const MIN_CONSECUTIVE_INFEAS: usize = 3;

/// infeasibility 検出器を不信任にする best_score / eps 比。
///
/// best-so-far がこの倍率以内まで収束している iterate を持つ場合、Newton 方向の
/// Farkas-like 近似 (PMM floor 由来の false-positive がある) より iterate 側を
/// 信用し、Infeasible ではなく Stalled + best-so-far で返す。10 は「eps に 1 桁
/// 以内まで迫った iterate は infeasible な問題では現れない」という経験則。
pub(super) const INFEAS_DETECTOR_DISTRUST_SCORE: f64 = 10.0;

/// fraction-to-boundary を補う trust-region cap (alpha·|dv|_inf ≤ cap·max(|v|_inf,1))。
pub(super) const STEP_REL_CAP: f64 = 1e3;

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

/// rank-deficient Q + c≈0 の適応 reg trigger: ||c||_inf がこの値未満なら c≈0 とみなす。
pub(super) const ADAPTIVE_REG_C_MAX_THRESH: f64 = 1e-6;

/// Gondzio corrector trigger: alpha がこの値未満のときのみ追加補正を適用する。
pub(super) const GONDZIO_ALPHA_TRIGGER: f64 = 0.999;

/// pf-stagnation trigger: 直近 `PF_HISTORY_LEN` 反復で primal residual (`nr_p`) が
/// 実質改善せず (`ratio > PF_STUCK_RATIO`) かつ未収束 (`nr_p > eps_orig`) なら
/// reg_limit floor を下げるべきと判定する。
///
/// 旧実装は `nr_p > eps_orig * 100` (「target から桁違いに離れている」場合のみ)
/// を追加要求していた。だが reg_limit の初期floor (`REG_LIMIT_INIT_QP` = 5e-8) は
/// eps_orig に依存しない絶対定数のため、tight eps (例 1e-8) では floor 由来の
/// 残差が `[eps_orig, 100·eps_orig)` に恒久的に張り付き、このゲートが永久に閉じる。
///
/// 実測 (LISWET7 @ eps=1e-8, commit a735fea6, `OTSPOT_IPM_TRACE=1`):
/// iter 12 から nr_p=6.216e-8 で frozen (floor=5e-8 一致)。100·eps_orig=1e-6 を
/// 満たさず iter 40 まで trigger せず、その間 μ が 1e-59 まで無意味に underflow
/// し数値ノイズで残差が発散、iter 62 で `residual_stall` (window=50) が発火し
/// Stalled のまま終了した。「stuck (5反復で ratio>0.95) かつ未収束 (>eps_orig)」で
/// 判定十分であり、追加の桁数マージンは進捗を止める副作用しか持たない。
pub(super) fn pf_stuck_should_lower_reg_limit(nr_p: f64, pf_oldest: f64, eps_orig: f64) -> bool {
    if pf_oldest <= 0.0 || nr_p <= eps_orig {
        return false;
    }
    (nr_p / pf_oldest) > PF_STUCK_RATIO
}

pub(super) struct PmmState {
    pub(super) x_ref: Vec<f64>,
    pub(super) y_ref: Vec<f64>,
    pub(super) rho: f64,
    pub(super) delta: f64,
    pub(super) prev_nr_p: f64,
    pub(super) prev_nr_d: f64,
}

#[cfg(test)]
mod pf_stuck_tests {
    use super::*;

    /// Sentinel: LISWET7 @ eps=1e-8 で実測した凍結値 (nr_p=6.216e-8、5反復前も
    /// bit-identical の同値) は「未収束 (>eps_orig) かつ停滞 (ratio=1.0>0.95)」
    /// なので reg_limit を下げるべき。
    ///
    /// 旧実装は `nr_p > eps_orig * 100 (=1e-6)` を追加要求しており、6.216e-8 は
    /// これを満たさず false を返していた (このテストは revert で fail する)。
    #[test]
    fn pf_stuck_fires_for_liswet7_observed_floor_residual() {
        let nr_p = 6.216e-8;
        let pf_oldest = 6.216e-8;
        let eps_orig = 1e-8;
        assert!(
            pf_stuck_should_lower_reg_limit(nr_p, pf_oldest, eps_orig),
            "stuck-above-target residual (nr_p={nr_p:.3e} > eps_orig={eps_orig:.3e}, \
             frozen over PF_HISTORY_LEN iters) must trigger reg_limit lowering"
        );
    }

    /// nr_p が既に eps_orig 未満: 追加で reg_limit を下げる必要はない。
    #[test]
    fn pf_stuck_does_not_fire_once_already_converged() {
        assert!(!pf_stuck_should_lower_reg_limit(0.5e-8, 0.5e-8, 1e-8));
    }

    /// ratio = 5e-8/6e-7 ≈ 0.083 << PF_STUCK_RATIO(0.95): 実質前進中で stuck でない。
    #[test]
    fn pf_stuck_does_not_fire_while_still_making_progress() {
        assert!(!pf_stuck_should_lower_reg_limit(5e-8, 6e-7, 1e-8));
    }

    /// pf_history 未充足 (oldest=0.0 は「5反復分のデータがまだ無い」を表す番兵)。
    #[test]
    fn pf_stuck_requires_nonzero_history() {
        assert!(!pf_stuck_should_lower_reg_limit(6e-8, 0.0, 1e-8));
    }

    /// 旧実装 (nr_p > eps_orig*100) を模した回帰確認: LISWET7 の凍結値は
    /// 100倍マージンを満たさないため旧ゲートは閉じたまま (新実装との対比)。
    #[test]
    fn old_hundred_x_margin_gate_would_have_stayed_closed_for_liswet7() {
        let nr_p = 6.216e-8_f64;
        let eps_orig = 1e-8_f64;
        let old_far_from_target_ratio = 1e2_f64;
        assert!(
            nr_p <= eps_orig * old_far_from_target_ratio,
            "documents why the old gate never opened for the LISWET7 floor residual"
        );
        assert!(
            nr_p > eps_orig,
            "yet the residual is still genuinely unconverged"
        );
    }
}
