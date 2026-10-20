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

/// inner 停止判定の 4 指標 (Ruiz scaled 空間)。
///
/// 各指標の定義:
/// - `nr_p_rel` / `nr_d_rel`: per-row / per-column の `1+|·|` 相対残差 (bench と同形)。
/// - `mu`: 不等式行の `sᵀy / m_ineq`。
/// - `gap`: [`duality_gap`] (`prove_optimal` の gap と同じ定義 + 評価誤差の見積り)。
pub(super) struct ConvergenceMeasures {
    pub(super) nr_p_rel: f64,
    pub(super) nr_d_rel: f64,
    pub(super) mu: f64,
    pub(super) gap: DualityGap,
}

impl ConvergenceMeasures {
    /// 4 指標すべてを lane の `eps` で判定する。gap だけは評価誤差 `gap.noise_rel` を
    /// 下限に持つ。
    ///
    /// gap は残差と μ からは従わない独立条件である: 恒等的に
    /// `p − d = sᵀy + r_pᵀy − xᵀr_d` なので、per-row 相対残差が eps 未満でも
    /// 行数と ‖y‖ が大きければ gap は eps より桁で大きく残る (LISWET7: 残差 < eps、
    /// gap 8.5e-7、真値比 obj 誤差 23%)。gap だけを eps と無関係な定数で比べると、
    /// その lane が狙う精度より粗い点を収束と判定してしまう。
    /// 「残差小・gap 大」の偽収束 (rank-deficient Q + c=0、UBH1 で gap 91%) も
    /// 同じ条件で弾かれる。
    ///
    /// 残差は per-component の `1+|·|` 相対なので丸め誤差は分母に吸収されるが、
    /// gap は打ち消し合う内積の差を `max(|p|,|d|,1)` で割るため、その評価誤差より
    /// 小さい値は観測できない。eps が `IPM_EPS_NOISE_FLOOR` まで締まった lane
    /// (BOYD1: Ruiz 増幅 1.1e6 で eps=2.2e-14) では gap が 2.7e-12 の誤差床に張り付き、
    /// 残差・μ が 1e-15 未満でも収束と判定できなくなる。
    pub(super) fn converged(&self, eps: f64) -> bool {
        self.nr_p_rel < eps
            && self.nr_d_rel < eps
            && self.mu < eps
            && self.gap.rel.abs() < eps.max(self.gap.noise_rel)
    }
}

/// 相対 duality gap とその浮動小数点評価誤差の見積り。
#[derive(Clone, Copy)]
pub(super) struct DualityGap {
    /// `(p − d) / max(|p|, |d|, 1)` (符号付き)。
    pub(super) rel: f64,
    /// `rel` の丸め誤差の見積り `√N·ε·(½Σ|x_i(Qx)_i| + Σ|c_i x_i| + Σ|b_i y_i|) / max(|p|,|d|,1)`。
    ///
    /// N 項の和の丸め誤差は確率的評価 (Higham & Mary 2019) で `√N·ε·Σ|項|` 程度に収まる。
    /// N は最長の内積の長さ `max(n, m_ext)`。実測の誤差床は見積りの内側にある
    /// (BOYD1: 床 2.7e-12、見積り 7.7e-12。BOYD2: 床 9.4e-12、見積り 3.7e-11)。
    pub(super) noise_rel: f64,
}

/// 符号規約 `r_d = −(Qx + c + A_extᵀy)` の下で `p = ½xᵀQx + cᵀx`、
/// `d = −½xᵀQx − b_extᵀy`。分母は `prove_optimal` に渡す元空間 gap
/// (`compute_duality_gap_rel`) と同じ。
pub(super) fn duality_gap(
    qx: &[f64],
    x: &[f64],
    c: &[f64],
    b_ext: &[f64],
    y: &[f64],
) -> DualityGap {
    let qx_dot_x: f64 = qx.iter().zip(x).map(|(&a, &b)| a * b).sum();
    let c_dot_x: f64 = c.iter().zip(x).map(|(&a, &b)| a * b).sum();
    let b_dot_y: f64 = b_ext.iter().zip(y).map(|(&a, &b)| a * b).sum();
    let p_obj = 0.5 * qx_dot_x + c_dot_x;
    let d_obj = -0.5 * qx_dot_x - b_dot_y;
    let denom = p_obj.abs().max(d_obj.abs()).max(1.0);

    let abs_sum =
        |u: &[f64], v: &[f64]| -> f64 { u.iter().zip(v).map(|(&a, &b)| (a * b).abs()).sum() };
    let magnitude = 0.5 * abs_sum(qx, x) + abs_sum(c, x) + abs_sum(b_ext, y);
    let terms = x.len().max(y.len()) as f64;
    DualityGap {
        rel: (p_obj - d_obj) / denom,
        noise_rel: terms.sqrt() * f64::EPSILON * magnitude / denom,
    }
}

/// reg_limit 下限と一段引下げ倍率。
pub(super) const REG_LIMIT_MIN: f64 = 1e-14;
pub(super) const REG_LIMIT_STEP: f64 = 1e-3;
/// initial_reg_limit のデフォルト値 (QP / LP)。
pub(super) const REG_LIMIT_INIT_QP: f64 = 5e-8;
pub(super) const REG_LIMIT_INIT_LP: f64 = 5e-10;

/// LP 経路の KKT 行列正則化 (`rho_matrix` / `delta_matrix`) の下限。
///
/// **値も適用範囲も実測較正であり、原理的導出ではない。** 値は 2e700955 が
/// 撤去した `DEFAULT_IPM_DELTA_MIN` と同じ 1e-8 で、長期運用で suite 全体に
/// 対して検証されていた床をそのまま戻している (新規較正ではない)。
///
/// 床が要る理由 (ken-13 = LP): 2e700955 が ρ/δ を `REG_LIMIT_INIT_LP`
/// (5e-10、適応引下げ後 5e-13) まで追随させた結果、‖dy‖ が 1e4 → 1e17 へ発散し
/// α が 1e-15 へ潰れて Stalled になった (137s / rel_err 2.06e-8)。LDL 健全性
/// プローブは全 iter で rel_resid ≤ 1e-8 / retry 0 回なので、因子化誤差ではなく
/// (1,1) が `ρI`、等式行の (2,2) が `−δI` だけになった系そのものの発散である。
///
/// 値の感度は単調でない (LP heavy sentinel。dfl001 / ken-13 以外は不変):
/// `1e-14` → dfl001 PASS / ken-13 FAIL、`1e-9` → dfl001 FAIL、`3e-9` → 双方 PASS、
/// `1e-8` → 双方 PASS、`1.49e-8` (=√ε_machine) → dfl001 FAIL。obj は全点で truth
/// と一致し、揺れるのは Optimal 証明の可否だけ。実測安全窓は `[3e-9, 1e-8]`
/// (幅 3.3 倍・上側は閉じている) で、採用値はその上端に接している。
/// **この定数を動かす変更は LP heavy sentinel (dfl001 / ken-13) の再ベンチ必須。**
pub(super) const KKT_PIVOT_FLOOR: f64 = 1e-8;

/// KKT 行列正則化の床は **LP 経路 (`Q ≡ 0`) にだけ** 掛ける。
///
/// 床の根拠は「Q が無いので (1,1) ブロックが `ρI` そのものになり、ρ が
/// `REG_LIMIT_MIN` まで落ちると系が発散する」という LP 固有の事情
/// (`KKT_PIVOT_FLOOR` の doc)。Q があれば (1,1) は `Q+ρI` なので同じ議論は立たない。
///
/// 1babe8bc はこのゲートを Gershgorin の `λ_min(Q)` 下界で行っていたが、
/// 下界は Q の対角に 1 つでもゼロ行があれば `≤ 0` になるため QP へも LP 相当の
/// 満額の床が掛かっていた (Maros-Meszaros 138 問中 106 問)。その結果 QP 側で
/// 床が行列にだけ入り、PMM 残差 (`r_d − ρ_prox(x−x_ref)`) と食い違って残差が
/// `ρ_matrix·‖dx‖` に張り付いた。実測 (QGFRDXPN): `nr_d` が
/// `1e-8 × ‖dx‖_∞ = 1e-8 × 11.83 = 1.183e-7` で 50 反復凍結 → residual_stall →
/// STALLED / pfn=1.3e-5。QSHELL も 1babe8bc では STALLED だが、そちらはこの
/// 張り付きではなく「床で軌跡が変わり 3 attempt が bit 同一 IterationLimit に
/// なる → 延長 attempt が予算を食い切る」という別経路 (`attempt.rs` の
/// `charged_iterations`)。どちらもゲートを `Q ≡ 0` に絞れば 8565fe5c の軌跡に戻る。
///
/// **なぜ QP 側で「行列と PMM 残差の両方に床を掛けて整合させる」を採らないか**
/// (すべて実測、obj は truth と一致し Optimal 証明の可否だけが揺れる):
/// ρ・δ 双方の残差側にも掛ける → dfl001 258s Stalled。δ の残差側だけ掛ける
/// → dfl001 274s Stalled。δ の残差側に掛け ρ の床を全廃 → dfl001 は通るが
/// QPILOTNO が `ipm_reg_retries=98` で 13.7s STALLED。
///
/// LISWET7 (Q = I) は元々 Gershgorin ゲートで床が消えていたので本変更で不変
/// (`liswet7_pfn_breaks_delta_matrix_floor_at_eps_1e8`)。
pub(super) fn matrix_reg_floor_for_lp(is_lp: bool) -> f64 {
    if is_lp {
        KKT_PIVOT_FLOOR
    } else {
        REG_LIMIT_MIN
    }
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

    /// LP (Q ≡ 0): (1,1) ブロックが `ρI` そのものになるので床を全量掛ける。
    /// 実測 (ken-13): この床を外すと ‖dy‖ が 1e17 まで発散し α が 1e-15 に
    /// 潰れて Stalled。
    #[test]
    fn lp_gets_the_full_pivot_floor() {
        assert_eq!(matrix_reg_floor_for_lp(true), KKT_PIVOT_FLOOR);
    }

    /// QP (Q ≠ 0) には床を掛けない。掛けると行列側と PMM 残差側の ρ が食い違い、
    /// 残差が `ρ_matrix·‖dx‖` に張り付く (QGFRDXPN 実測 1.183e-7 / 50 反復凍結)。
    #[test]
    fn qp_gets_no_pivot_floor() {
        assert_eq!(matrix_reg_floor_for_lp(false), REG_LIMIT_MIN);
    }

    /// 返り値は常に `reg_limit` の絶対下限以上 (0 に潰れない)。
    #[test]
    fn floor_never_drops_below_reg_limit_min() {
        for is_lp in [true, false] {
            assert!(
                matrix_reg_floor_for_lp(is_lp) >= REG_LIMIT_MIN,
                "is_lp={is_lp}"
            );
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

#[cfg(test)]
mod convergence_tests {
    use super::*;

    /// `min ½x²  s.t. −x ≤ −1` (内部形 `A_ext x + s = b_ext`, `A_ext=[−1]`, `b_ext=[−1]`) の
    /// 点 `x = y = 1−δ`, `s = σ` における 4 指標の手計算値。
    ///
    /// - `qx = 1−δ`, `aty = −(1−δ)` → `r_d = −(qx + c + aty) = 0`。
    /// - `r_p = b − ax − s = −1 + (1−δ) − σ = −(δ+σ)`、
    ///   `nr_p_rel = (δ+σ) / (1 + |ax| + |b|) = (δ+σ) / (3−δ)`。
    /// - `μ = σ(1−δ)`。
    /// - `p = ½(1−δ)²`, `d = −½(1−δ)² + (1−δ)` → `p − d = −δ(1−δ)`、分母 `max(|p|,|d|,1) = 1`。
    /// - 誤差見積り: N=1、`½|x·qx| + |c·x| + |b·y| = ½(1−δ)² + (1−δ)` → `ε(½(1−δ)² + (1−δ))`。
    struct ShiftedPoint {
        x: f64,
        y: f64,
        nr_p_rel: f64,
        mu: f64,
        rel_gap: f64,
        noise_rel: f64,
    }

    fn shifted_point(delta: f64, sigma: f64) -> ShiftedPoint {
        let v = 1.0 - delta;
        ShiftedPoint {
            x: v,
            y: v,
            nr_p_rel: (delta + sigma) / (3.0 - delta),
            mu: sigma * v,
            rel_gap: -delta * v,
            noise_rel: f64::EPSILON * (0.5 * v * v + v),
        }
    }

    fn gap_of(pt: &ShiftedPoint) -> DualityGap {
        duality_gap(&[pt.x], &[pt.x], &[0.0], &[-1.0], &[pt.y])
    }

    fn measures_of(pt: &ShiftedPoint) -> ConvergenceMeasures {
        ConvergenceMeasures {
            nr_p_rel: pt.nr_p_rel,
            nr_d_rel: 0.0,
            mu: pt.mu,
            gap: gap_of(pt),
        }
    }

    #[test]
    fn duality_gap_matches_hand_computed_value() {
        for (delta, sigma) in [(1e-6, 1e-12), (1e-3, 0.0), (0.25, 1e-4)] {
            let pt = shifted_point(delta, sigma);
            let got = gap_of(&pt);
            assert!(
                (got.rel - pt.rel_gap).abs() <= 1e-15,
                "delta={delta:e}: got {:e}, expected {:e}",
                got.rel,
                pt.rel_gap
            );
            assert!(
                (got.noise_rel - pt.noise_rel).abs() <= 1e-30,
                "delta={delta:e}: noise got {:e}, expected {:e}",
                got.noise_rel,
                pt.noise_rel
            );
        }
    }

    /// 分母は `max(|p|, |d|, 1)`: `|p|` が 1 を超えると `|p|` で割る。
    /// `min x  s.t. −x ≤ −1` (LP) で `x = 10`, `y = 1`: `p = 10`, `d = 1`, gap = 9/10、
    /// 誤差見積り `ε(|1·10| + |−1·1|)/10 = 1.1ε`。
    #[test]
    fn duality_gap_divides_by_the_larger_objective() {
        let got = duality_gap(&[0.0], &[10.0], &[1.0], &[-1.0], &[1.0]);
        assert_eq!(got.rel, 0.9);
        assert!((got.noise_rel - 1.1 * f64::EPSILON).abs() <= 1e-30);
    }

    /// 誤差見積りは最長の内積の長さ N の平方根に比例する。
    /// n=4 (x=1, c=1, Q=0)、m_ext=9 (b=0): p = 4, d = 0、`Σ|c_i x_i| = 4` → `√9·ε·4/4 = 3ε`。
    #[test]
    fn duality_gap_noise_scales_with_sqrt_of_longest_sum() {
        let got = duality_gap(&[0.0; 4], &[1.0; 4], &[1.0; 4], &[0.0; 9], &[1.0; 9]);
        assert_eq!(got.rel, 1.0);
        assert!((got.noise_rel - 3.0 * f64::EPSILON).abs() <= 1e-30);
    }

    /// LISWET7 の吸引点の構造: 残差と μ は eps 未満、gap だけが eps を超える。
    /// δ = 1e-6, eps = 5e-7: `nr_p_rel ≈ 3.3e-7`, `μ ≈ 1e-12`, `|gap| ≈ 1e-6`。
    /// gap を eps と無関係な定数 1e-3 と比べる旧判定はこの点を収束と判定していた。
    #[test]
    fn gap_above_eps_is_not_converged_even_with_small_residuals() {
        let eps = 5e-7;
        let pt = shifted_point(1e-6, 1e-12);
        let m = measures_of(&pt);
        assert!(m.nr_p_rel < eps && m.nr_d_rel < eps && m.mu < eps);
        assert!(m.gap.rel.abs() > eps && m.gap.rel.abs() < 1e-3);
        assert!(m.gap.noise_rel < eps);
        assert!(!m.converged(eps));
    }

    /// 同じ点でも lane の eps が gap より緩ければ収束と判定する。
    #[test]
    fn gap_below_eps_with_small_residuals_is_converged() {
        let pt = shifted_point(1e-6, 1e-12);
        assert!(measures_of(&pt).converged(1e-5));
        let tight = shifted_point(1e-9, 1e-15);
        assert!(measures_of(&tight).converged(5e-9));
    }

    /// eps が gap の評価誤差より小さい lane (BOYD1 実測: eps=2.2e-14、gap 2.7e-12 で
    /// 張り付き、見積り 7.7e-12): 評価誤差未満の gap は収束と判定する。
    #[test]
    fn gap_within_evaluation_noise_is_converged_when_eps_is_below_noise() {
        let eps = 2.2e-14;
        let at_floor = ConvergenceMeasures {
            nr_p_rel: 1e-15,
            nr_d_rel: 5e-15,
            mu: 1e-21,
            gap: DualityGap {
                rel: -2.7e-12,
                noise_rel: 7.7e-12,
            },
        };
        assert!(at_floor.converged(eps));
        let above_floor = ConvergenceMeasures {
            gap: DualityGap {
                rel: -1e-11,
                noise_rel: 7.7e-12,
            },
            ..at_floor
        };
        assert!(!above_floor.converged(eps));
    }

    /// rank-deficient Q + c=0 の偽収束 (UBH1: 残差小・gap 91%) は引き続き弾く。
    #[test]
    fn large_gap_with_tiny_residuals_is_not_converged() {
        let m = ConvergenceMeasures {
            nr_p_rel: 1e-12,
            nr_d_rel: 1e-12,
            mu: 1e-30,
            gap: DualityGap {
                rel: 0.91,
                noise_rel: 1e-14,
            },
        };
        assert!(!m.converged(1e-6));
        let negative = ConvergenceMeasures {
            gap: DualityGap {
                rel: -0.91,
                noise_rel: 1e-14,
            },
            ..m
        };
        assert!(!negative.converged(1e-6));
    }

    /// 各指標が単独で eps を超えれば収束としない (全分岐)。
    #[test]
    fn each_measure_alone_blocks_convergence() {
        let eps = 1e-6;
        let gap = |rel: f64| DualityGap {
            rel,
            noise_rel: 1e-14,
        };
        let ok = || ConvergenceMeasures {
            nr_p_rel: 1e-9,
            nr_d_rel: 1e-9,
            mu: 1e-9,
            gap: gap(1e-9),
        };
        assert!(ok().converged(eps));
        let over = 2e-6;
        assert!(!ConvergenceMeasures {
            nr_p_rel: over,
            ..ok()
        }
        .converged(eps));
        assert!(!ConvergenceMeasures {
            nr_d_rel: over,
            ..ok()
        }
        .converged(eps));
        assert!(!ConvergenceMeasures { mu: over, ..ok() }.converged(eps));
        assert!(!ConvergenceMeasures {
            gap: gap(over),
            ..ok()
        }
        .converged(eps));
        assert!(!ConvergenceMeasures {
            gap: gap(-over),
            ..ok()
        }
        .converged(eps));
        assert!(!ConvergenceMeasures {
            gap: gap(f64::NAN),
            ..ok()
        }
        .converged(eps));
    }
}
