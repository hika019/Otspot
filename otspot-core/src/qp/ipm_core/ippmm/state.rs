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

/// KKT 行列正則化 (`rho_matrix`/`delta_matrix`) の下限。
///
/// **値も適用範囲も実測較正であり、原理的導出ではない。** 値は 2e700955 が
/// 撤去した `DEFAULT_IPM_DELTA_MIN` と同じ 1e-8 で、長期運用で suite 全体に
/// 対して検証されていた床をそのまま戻している (新規較正ではない)。
///
/// 機構として実測で閉じているのはここまで: `pmm.rho`/`pmm.delta` は PMM の
/// 不動点を決める proximal 係数と KKT 行列の正則化を兼ねる。後者は残差が 0 なら
/// 方向も 0 なので不動点を動かさず、Newton 方向の条件数だけを決める。2e700955 が
/// 後者を前者の floor (`REG_LIMIT_INIT_LP` = 5e-10、適応引下げ後 5e-13) に
/// 追随させた結果、ken-13 で ‖dy‖ が 1e4 → 1e17 へ発散し α が 1e-15 へ潰れて
/// Stalled になった。`iter.rs` の該当 2 行だけを revert すると 137s / Stalled /
/// rel_err 2.06e-8 が再現する。LDL 健全性プローブは全 iter で rel_resid ≤ 1e-8 /
/// retry 0 回なので、因子化誤差ではなく系そのものの発散である。
///
/// **PASS/FAIL を分けている量は特定できていない。** ken-13 での 4-arm 実測:
///
/// | `rho_matrix` | `delta_matrix` | 積 ρ·δ | 結果 |
/// |---|---|---|---|
/// | `max(ρ,1e-8)` | `max(δ,1e-8)` | ~1e-16 | PASS 5.2s |
/// | `ρ` (床なし) | `max(δ,1e-8)` | ~5e-21 | PASS 4.5s |
/// | `max(ρ,1e-8)` | `δ` (床なし) | ~5e-21 | PASS 10.4s |
/// | `ρ` | `δ` | ~2.5e-25 | FAIL 137s |
///
/// 分けているのは積 `ρ·δ` ではない (中央 2 arm は積が等しく共に PASS)。
/// 「ρ か δ の少なくとも一方が 1e-8 級」が観測に整合する唯一の記述だが、
/// これは説明ではなく要約である。以前ここに書かれていた
/// `ρ·δ ≳ ε_machine` (order で `√ε_machine`) という条件数由来の導出は、
/// 上表の中央 2 arm を誤って FAIL と予測するため撤去した。
///
/// 値の感度も単調でない (LP heavy sentinel。dfl001 / ken-13 以外は不変):
/// `1e-14` → dfl001 PASS / ken-13 FAIL、`1e-9` → dfl001 FAIL、`3e-9` → 双方 PASS、
/// `1e-8` → 双方 PASS、`1.49e-8` (=√ε_machine) → dfl001 FAIL。obj は全点で truth
/// と一致し、揺れるのは Optimal 証明の可否だけ。実測安全窓は `[3e-9, 1e-8]`
/// (幅 3.3 倍・上側は閉じている) で、採用値はその上端に接している。
/// **この定数を動かす変更は LP heavy sentinel (dfl001 / ken-13) の再ベンチ必須。**
pub(super) const KKT_PIVOT_FLOOR: f64 = 1e-8;

/// 行列正則化の floor を Gershgorin の λ_min(Q) 下界でゲートする。
///
/// `pmm.rho`/`pmm.delta` は 2 つの役割を兼ねている:
///  1. proximal 係数 — `r_d_pmm = r_d − ρ(x−x_ref)`, `r_p_pmm = r_p − δ(y−y_ref)`。
///     PMM の不動点を決めるので `reg_limit` (→ `REG_LIMIT_MIN`) まで下げてよい。
///  2. 行列正則化 — Newton 方向の条件数だけを決め、不動点には影響しない。
///
/// ゲートの根拠は 2 つの較正点の実測であって導出ではない:
///  * LP (ken-13、Q ≡ 0) は床が要る (`KKT_PIVOT_FLOOR` の 4-arm 表を参照)。
///  * LISWET7 (Q は単位行列、eps=1e-8) は床があると `dy_i·δ = r_p_i` の恒等式で
///    nr_p が 9.8e-8 に凍結する (`liswet7_pfn_breaks_delta_matrix_floor_at_eps_1e8`)。
///
/// 両者を分ける観測可能量として「Q が (1,1) ブロック `Q+ρI` に供給するピボット
/// 質量」を採り、供給分だけ床を減額する。(1,1) については Q が ρ に対して加算的
/// なのでこの形に意味がある。
///
/// **既知の限界 (設計上の再検討は別 task)**:
///  * (2,2) ブロックは `−(Σ+δI)` で Q は構造的に入らない。それでも δ 側の床まで
///    Q 質量でクレジットしているのは、ρ 側だけに適用すると LISWET7 の
///    `delta_matrix` に 1e-8 が復活し上記 sentinel が FAIL するため。導出ではなく
///    較正上の妥協である。
///  * 比較対象は Ruiz scaling 後の Q なのでゲートはスケール依存。BOYD1 は
///    `λ_min(Q_raw) = 6.0` が `λ_min(Q_scaled) = 1.03e-7` まで縮み、僅差で床が
///    消える。Maros-Meszaros 138 問の走査では `λ_min ≤ 0` が 106 問 (床は満額)、
///    `≥ 1e-8` が 32 問 (床は消失)、中間帯 `0 < λ_min < 1e-8` は 0 問 —
///    実質 2 値ゲートで、減額の連続性は unit test でしか行使されていない。
///  * 床が消える側の問題群 (BOYD1 / HS118 / HS21 / QPCBOEI1 / QPCBOEI2 / KSIP) に
///    無条件 1e-8 を掛ける arm との A/B では live regression は観測されていない
///    (BOYD1 のみ iters 40 vs 43、他はビット一致)。
pub(super) fn matrix_reg_floor_for(q_lambda_min_lower: f64) -> f64 {
    // `f64::max` は NaN 側を捨てるので、下界が NaN / 負 (indefinite) なら質量 0 =
    // floor 全量。ただし indefinite 時に実際に使われる ρ は
    // `factorize.rs` の `rho_matrix.max(inertia_correction)` (和ではなく max) なので、
    // `inertia_correction > KKT_PIVOT_FLOOR` の問題ではここで返す満額の床は届かない。
    let q_pivot = q_lambda_min_lower.max(0.0);
    (KKT_PIVOT_FLOOR - q_pivot).max(REG_LIMIT_MIN)
}

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
mod matrix_reg_floor_tests {
    use super::*;

    /// 2e700955 以前の `DEFAULT_IPM_DELTA_MIN` と同値であること。値の変更は
    /// LP heavy sentinel (dfl001 / ken-13) の再ベンチ無しには許されない —
    /// 感度は単調でなく実測安全窓は `[3e-9, 1e-8]` (doc の表を参照)。
    #[test]
    fn kkt_pivot_floor_matches_the_long_validated_value() {
        assert_eq!(KKT_PIVOT_FLOOR, 1e-8);
    }

    /// 採用値は実測安全窓の内側にあること。窓の上端に接している事実も pin する
    /// (上端を跨ぐ 1.49e-8 は dfl001 を FAIL させると doc に記録がある)。
    #[test]
    fn kkt_pivot_floor_sits_inside_the_measured_safe_window() {
        assert!((3e-9..=1e-8).contains(&KKT_PIVOT_FLOOR));
    }

    /// LP (Q ≡ 0 → Gershgorin 下界 0): (1,1) ピボットを ρ が単独で担うので
    /// floor は `KKT_PIVOT_FLOOR` 全量。実測 (ken-13): この床を外すと
    /// ‖dy‖ が 1e17 まで発散し α が 1e-15 に潰れて Stalled。
    #[test]
    fn zero_hessian_gets_full_pivot_floor() {
        assert_eq!(matrix_reg_floor_for(0.0), KKT_PIVOT_FLOOR);
    }

    /// Q が自前で床以上の質量を持つ (LISWET7 の Q = I → 下界 1.0): floor 不要。
    /// 実測: ここで `KKT_PIVOT_FLOOR` を課すと nr_p が δ·|dy| で 9.8e-8 に凍結する。
    #[test]
    fn positive_definite_hessian_needs_no_floor() {
        assert_eq!(matrix_reg_floor_for(1.0), REG_LIMIT_MIN);
        assert_eq!(matrix_reg_floor_for(KKT_PIVOT_FLOOR), REG_LIMIT_MIN);
    }

    /// Q が部分的にしか供給しないなら不足分だけを補う (連続な切り分け)。
    #[test]
    fn partial_hessian_mass_is_credited_against_the_floor() {
        let half = KKT_PIVOT_FLOOR / 2.0;
        assert_eq!(matrix_reg_floor_for(half), KKT_PIVOT_FLOOR - half);
    }

    /// indefinite Q (下界が負) は質量 0 として扱う: 負値を引いて床を
    /// 押し上げてはならない (`inertia_correction` が別途 PSD 化を担う)。
    #[test]
    fn indefinite_hessian_lower_bound_is_clamped_to_zero() {
        assert_eq!(matrix_reg_floor_for(-5.0), KKT_PIVOT_FLOOR);
        assert_eq!(matrix_reg_floor_for(f64::NAN), KKT_PIVOT_FLOOR);
        assert_eq!(matrix_reg_floor_for(f64::NEG_INFINITY), KKT_PIVOT_FLOOR);
    }

    /// 返り値は常に proximal 側の絶対下限以上 (0 に潰れない)。
    #[test]
    fn floor_never_drops_below_reg_limit_min() {
        for q in [0.0, 1e-30, 1e30, f64::INFINITY] {
            assert!(matrix_reg_floor_for(q) >= REG_LIMIT_MIN, "q={q}");
        }
    }
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
