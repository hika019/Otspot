//! Predictor-Corrector-Gondzio 共通ループ部品。

use super::common::fraction_to_boundary_masked;
use super::{ALPHA_IMPROVE_THRESHOLD, BETA_GONDZIO, GAMMA_L, GAMMA_U, TAU};
use crate::tolerances::any_nonfinite;
use otspot_num::linalg::kkt_solver::KktFactor;
use otspot_num::sparse::CscMatrix;

/// f64 IR の反復上限。3 を超えると LDL 精度限界で利得が出ないため。
pub(crate) const IR_MAX_ITERS: usize = 3;

/// f64 IR の残差収束判定: resid_inf ≤ rhs_inf × この係数 で早期終了。
pub(crate) const IR_RESID_SKIP_REL: f64 = 1e-13;

/// μ が実質 0 と判定する境界: これ未満のとき centering sigma を 0 に落とす。
pub(crate) const CENTERING_MU_FLOOR: f64 = 1e-15;

/// aug_mat は fac に factorize された対称上三角 CSC である必要がある。
pub(crate) fn solve_with_iterative_refinement(
    fac: &KktFactor,
    aug_mat: &CscMatrix,
    rhs: &[f64],
    sol: &mut [f64],
    max_iters: usize,
    deadline: Option<std::time::Instant>,
) -> bool {
    // aug_dim = augmented KKT 次元 = n_vars + m_ext (呼び出し側で sol/rhs を
    // total = n + m_ext で確保するため sol.len() がそのまま拡大系の次元)。
    let aug_dim = sol.len();
    debug_assert_eq!(rhs.len(), aug_dim);
    debug_assert_eq!(aug_mat.nrows(), aug_dim);
    debug_assert_eq!(aug_mat.ncols(), aug_dim);

    // Primary solve: zero sol on MINRES error to prevent NaN; IPM continues with
    // degraded steps until residual stall or max_iter. 戻り値 false = primary solve
    // 失敗 (sol はゼロ化済み)。IR 段の中断は primary 解が有効なので true を返す。
    if fac.solve_with_deadline(rhs, sol, deadline).is_err() {
        for v in sol.iter_mut() {
            *v = 0.0;
        }
        return false;
    }

    if max_iters == 0 {
        return true;
    }
    // 反復 backend は自分で tol まで収束するため IR を被せると deadline 浪費。
    if fac.is_iterative() {
        return true;
    }
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return true;
    }

    // 大型 (拡大系次元 aug_dim = n+m_ext > 100k) は IR overhead が deadline を
    // 圧迫するため skip。aug_dim は既に m_ext を含むので別途加算不要。
    const IR_SKIP_LARGE_THRESHOLD: usize = 100_000;
    if aug_dim > IR_SKIP_LARGE_THRESHOLD {
        return true;
    }

    let rhs_inf = rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
    let resid_skip_threshold = rhs_inf * IR_RESID_SKIP_REL;

    let mut kx = vec![0.0_f64; aug_dim];
    let mut residual = vec![0.0_f64; aug_dim];
    let mut correction = vec![0.0_f64; aug_dim];

    for _ir_iter in 0..max_iters {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return true;
        }
        for v in kx.iter_mut() {
            *v = 0.0;
        }
        for col in 0..aug_mat.ncols() {
            for ptr in aug_mat.col_ptr()[col]..aug_mat.col_ptr()[col + 1] {
                let row = aug_mat.row_ind()[ptr];
                let val = aug_mat.values()[ptr];
                kx[row] += val * sol[col];
                if row != col {
                    kx[col] += val * sol[row];
                }
            }
        }
        for i in 0..aug_dim {
            residual[i] = rhs[i] - kx[i];
        }

        let mut resid_inf = 0.0_f64;
        for i in 0..aug_dim {
            resid_inf = resid_inf.max(residual[i].abs());
        }
        if resid_inf <= resid_skip_threshold {
            return true;
        }

        for v in correction.iter_mut() {
            *v = 0.0;
        }
        // IR solve error: correction stays zero; the partial correction would corrupt sol.
        if fac
            .solve_with_deadline(&residual, &mut correction, deadline)
            .is_err()
        {
            break;
        }

        // Backtrack guard: NaN/Inf protection
        let any_bad = any_nonfinite(&correction);
        if any_bad {
            return true;
        }

        // correction が sol を超える = LDL 精度限界の虚偽補正 → IR を打ち切る。
        let correction_inf = correction.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let sol_inf = sol.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        if correction_inf > sol_inf {
            return true;
        }

        for i in 0..aug_dim {
            sol[i] += correction[i];
        }
    }
    true
}

pub(crate) struct PredictorResult {
    pub dy_pred: Vec<f64>,
    pub ds_pred: Vec<f64>,
    pub sigma_center: f64,
}

/// Σ = diag(s_i / y_i)。等式行は 0、nan/inf は sigma_max でクランプ。
pub(crate) fn compute_sigma_vec(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    sigma_max: f64,
) -> Vec<f64> {
    s.iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| {
            if is_eq_ext[i] {
                0.0
            } else {
                let v = si / yi;
                if v.is_finite() {
                    v
                } else {
                    sigma_max
                }
            }
        })
        .collect()
}

/// Schur 経由で KKT step を解く中核: S·dx = r_d + AᵀD⁻¹r_p、続いて
/// dy = D⁻¹(A·dx − r_p)。active constraints で |D⁻¹| 増幅による cancellation が
/// df を発散させるため、dy の back-substitution は TwoFloat (DD ≈106 bit) で計算する。
/// Solve 失敗時は Err を返し、呼び出し側が
/// 追加で保持する行 (box 化の境界行 dy など) も含めて一括でゼロ化できるようにする
/// (失敗経路で半分だけ埋まった方向ベクトルを消費させない)。
/// 返り値の Ok は「Solve が完了し dx_out/dy_out に有効な値が入った」こと。
fn try_solve_kkt_via_schur(
    s_fac: &KktFactor,
    d_inv: &[f64],
    a_ext: &CscMatrix,
    r_d: &[f64],
    r_p_mod: &[f64],
    dx_out: &mut [f64],
    dy_out: &mut [f64],
    deadline: Option<std::time::Instant>,
) -> Result<(), ()> {
    use super::kkt::spmtv;
    use twofloat::TwoFloat;

    let n = r_d.len();
    let m_ext = r_p_mod.len();

    let mut d_inv_rp = vec![0.0_f64; m_ext];
    for i in 0..m_ext {
        d_inv_rp[i] = d_inv[i] * r_p_mod[i];
    }
    let mut at_d_inv_rp = vec![0.0_f64; n];
    spmtv(a_ext, &d_inv_rp, &mut at_d_inv_rp);
    let rhs_s: Vec<f64> = r_d
        .iter()
        .zip(at_d_inv_rp.iter())
        .map(|(&r, &v)| r + v)
        .collect();

    if s_fac.solve_with_deadline(&rhs_s, dx_out, deadline).is_err() {
        return Err(());
    }

    let zero_dd = TwoFloat::from(0.0);
    let mut a_dx_dd: Vec<TwoFloat> = vec![zero_dd; m_ext];
    for col in 0..n {
        let cs = a_ext.col_ptr()[col];
        let ce = a_ext.col_ptr()[col + 1];
        let dx_col = dx_out[col];
        for k in cs..ce {
            let row = a_ext.row_ind()[k];
            let v = a_ext.values()[k];
            a_dx_dd[row] += TwoFloat::new_mul(v, dx_col);
        }
    }
    for i in 0..m_ext {
        let diff_dd = a_dx_dd[i] - TwoFloat::from(r_p_mod[i]);
        let scaled = diff_dd * TwoFloat::from(d_inv[i]);
        dy_out[i] = f64::from(scaled);
    }
    Ok(())
}

/// box 化線形求解のバックエンド。
///
/// どちらのバックエンドでも係数行列は box 化済み (box 化境界行を持たない)。
/// augmented は n+m_sys 準正定系を直接 LDL、Schur は境界対角 D_b を Q 対角へ
/// 畳み込んだ n×n SPD 系。`a_real`/d_inv は `build_boxed_system_matrix` の
/// 系行列 A_sys = [実制約行 | 保持境界行] に対応する。box 化境界行の dy は
/// 両バックエンド共通の後退代入 `dy_t = d_t (a_t·dx − rhs_t)` で復元する
/// (正確 Schur の第2段)。
pub(crate) enum BoxedBackend<'a> {
    Aug {
        fac: &'a KktFactor,
        mat: &'a CscMatrix,
    },
    Schur {
        s_fac: &'a KktFactor,
        a_real: &'a CscMatrix,
        d_inv_real: &'a [f64],
    },
}

/// box 化 KKT への Newton 求解口 (extended レイアウトを跨いだ入出力)。
///
/// 入力の `r_p_mod_full`・出力の `dy_out`/`ds` は extended レイアウト
/// ([実制約 | 全境界行]) を保ち、系行 ([実制約 | 保持境界行]) への
/// gather/scatter と box 化境界行の再結合をこの層が担う。これにより予測子・
/// 修正子・Gondzio・証書検証は m_ext-layout のまま無変更で動く
/// (受理ゲートの独立検証器は box 化を知らない)。
pub(crate) struct BoxedSolve<'a> {
    pub(crate) backend: BoxedBackend<'a>,
    pub(crate) bound: &'a super::kkt::BoundBox,
    /// box 化境界行ごとの後退代入係数 d_t = bound_row_dual_scale(σ_t, δ_used)。
    /// extended 境界 index t で添え字 (保持行 slot は使われない)。
    /// 因子化が実際に採用した δ_used から算出され、KKT 対角の D_b と bit 一致する
    /// (`build_boxed_schur_system` / `AugmentedKktCache::materialize` と共有関数)。
    pub(crate) d_b: &'a [f64],
    pub(crate) m_real: usize,
    /// 系行 k → extended 行 index の対応 (`BoundBox::system_ext_rows`)。
    /// 先頭 m_real 件は恒等で、その後に保持境界行 (lb==ub 退化対) が続く。
    pub(crate) sys_ext_rows: &'a [usize],
}

impl BoxedSolve<'_> {
    /// box 化系を解いて (dx, dy_full) を得る。Solve 失敗時は全成分ゼロ
    /// (部分埋まりの方向を上位に渡さない)。
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn solve(
        &self,
        r_dual: &[f64],
        r_p_mod_full: &[f64],
        dx_out: &mut [f64],
        dy_out: &mut [f64],
        deadline: Option<std::time::Instant>,
    ) {
        let n = r_dual.len();
        let m_real = self.m_real;
        let n_bound = self.bound.count();
        debug_assert_eq!(r_p_mod_full.len(), m_real + n_bound);
        debug_assert_eq!(dy_out.len(), m_real + n_bound);
        debug_assert_eq!(self.d_b.len(), n_bound);
        debug_assert!(self.sys_ext_rows.len() >= m_real);
        debug_assert_eq!(
            self.sys_ext_rows[..m_real],
            *self.backend_real_identity(m_real)
        );

        // box 化境界行の Schur 前向き代入: x 右辺へ Σ_t a_t d_t rhs_t を加算
        // (a_t = sign_t e_{var_t} なので scatter は O(n_bound))。保持行は
        // scatter せず、その r_p が系行右辺としてそのまま求解に入る。
        let mut rhs_x = r_dual.to_vec();
        for t in 0..n_bound {
            if !self.bound.eliminated[t] {
                continue;
            }
            let j = self.bound.var[t];
            rhs_x[j] += self.bound.sign[t] * self.d_b[t] * r_p_mod_full[m_real + t];
        }

        // 系行順への右辺 gather (実制約行は恒等、保持境界行は extended 位置から)。
        let mut rhs_sys: Vec<f64> = self
            .sys_ext_rows
            .iter()
            .map(|&ext| r_p_mod_full[ext])
            .collect();

        let solve_ok = match &self.backend {
            BoxedBackend::Aug { fac, mat } => {
                let mut rhs = Vec::with_capacity(n + rhs_sys.len());
                rhs.extend_from_slice(&rhs_x);
                rhs.append(&mut rhs_sys);
                let mut sol = vec![0.0_f64; n + self.sys_ext_rows.len()];
                let ok = solve_with_iterative_refinement(
                    fac,
                    mat,
                    &rhs,
                    &mut sol,
                    IR_MAX_ITERS,
                    deadline,
                );
                dx_out.copy_from_slice(&sol[..n]);
                for (k, &ext) in self.sys_ext_rows.iter().enumerate() {
                    dy_out[ext] = sol[n + k];
                }
                ok
            }
            BoxedBackend::Schur {
                s_fac,
                a_real,
                d_inv_real,
            } => {
                let mut dy_sys = vec![0.0_f64; self.sys_ext_rows.len()];
                let ok = try_solve_kkt_via_schur(
                    s_fac,
                    d_inv_real,
                    a_real,
                    &rhs_x,
                    &rhs_sys,
                    &mut dx_out[..n],
                    &mut dy_sys,
                    deadline,
                )
                .is_ok();
                for (k, &ext) in self.sys_ext_rows.iter().enumerate() {
                    dy_out[ext] = dy_sys[k];
                }
                ok
            }
        };
        if !solve_ok {
            // 契約: 失敗時は全成分ゼロ (部分埋まりの方向を上位に渡さない)。
            dx_out[..n].fill(0.0);
            dy_out[..m_real + n_bound].fill(0.0);
        }

        // box 化境界行 dy の復元 (正確 Schur 第2段): dy_t = d_t (a_t·dx − rhs_t)。
        // 失敗判定はバックエンドの実際の成否のみを使う。成敗を解の値から推論
        // してはならない: 停留点 (定数 iterate) では dx = 0 が正解で、右辺が
        // 非ゼロでも dx ゼロは正当 (境界行 scatter が lb/ub で相殺しうる)。
        // dx 非ゼロヒューリスティクはかかる正当解を失敗と誤認し dy を凍結するため禁止。
        for t in 0..n_bound {
            if !self.bound.eliminated[t] {
                continue;
            }
            let j = self.bound.var[t];
            dy_out[m_real + t] = if solve_ok {
                self.d_b[t] * (self.bound.sign[t] * dx_out[j] - r_p_mod_full[m_real + t])
            } else {
                0.0
            };
        }
    }

    /// 系行対応の先頭 m_real 件は恒等 ([0..m_real)) でなければならない。
    fn backend_real_identity(&self, m_real: usize) -> Vec<usize> {
        (0..m_real).collect()
    }
}

/// ds 生成 (box 化)。実制約行は従来通り相補性線形化 ds_i = r_c_i/y_i − σ_i dy_i。
///
/// 境界行は realized 従属率 ds_t = −a_t·dx を使う (設計 qp.md §1: 束縛同一性
/// s_t(x) = b_t − a_t x の線形化)。extended 系の Newton ds_t = r_c_t/y_t − σ_t dy_t
/// = −a_t dx + δ_d·dy_t は「独立スラックが along-Newton で動く」系の正確な導関数で、
/// δ_d·dy_t 項はその意味では本物 (naive な従属形がこれを α に使うのは逆方向の
/// 誤り: box 化では realized 変位が −a_t dx そのものなので、δ_d·dy_t < 0 のとき
/// s の減少を過小評価し fraction-to-boundary が超過 α を通す)。超過した x は
/// fill_slacks の床で s<0 がマスクされ (r_p 境界行は構造的 0)、実行不可能が
/// 検出不能になったまま boundary 上を漂流する (IPPMM-T4 で実測: x>ub+0.17 で
/// score≈1e-13 に張り付き Timeout)。α・μ_aff・Gondzio の予測と realized 経路を
/// 一致させることが収束の前提条件。
///
/// 例外: lb==ub 退化対の保持境界行 (independent slack, 系行) は従属率が成立せず、
/// 実制約行と同じ相補線形化を使う (`bound_pair_is_degenerate` の doc)。
#[allow(clippy::too_many_arguments)]
fn compute_ds_boxed(
    m_real: usize,
    bound: &super::kkt::BoundBox,
    is_eq_ext: &[bool],
    r_c: &[f64],
    y: &[f64],
    sigma_vec: &[f64],
    dy: &[f64],
    dx: &[f64],
    ds_out: &mut [f64],
) {
    let m_ext = ds_out.len();
    debug_assert_eq!(m_ext, m_real + bound.count());
    for i in 0..m_real {
        if is_eq_ext[i] {
            ds_out[i] = 0.0;
        } else {
            ds_out[i] = r_c[i] / y[i] - sigma_vec[i] * dy[i];
        }
    }
    for t in 0..bound.count() {
        let i = m_real + t;
        if bound.eliminated[t] {
            ds_out[i] = -bound.sign[t] * dx[bound.var[t]];
        } else if is_eq_ext[i] {
            ds_out[i] = 0.0;
        } else {
            // 保持境界行 (lb==ub 退化対): 独立スラックなので extended 系と同一の
            // 相補線形化。従属形 −a·dx は従属スラックが存在しない行に適用すると
            // α が床張り付き slacks で潰れる (bound_pair_is_degenerate の doc)。
            ds_out[i] = r_c[i] / y[i] - sigma_vec[i] * dy[i];
        }
    }
}

/// box 化予測子ステップ。旧 `predictor_step_schur`/`predictor_step` の置換
/// (box 化 KKT での設計実装。設計文書 qp.md §4.1)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn predictor_boxed(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    solver: &BoxedSolve<'_>,
    mu: f64,
    deadline: Option<std::time::Instant>,
) -> PredictorResult {
    let m_ext = s.len();
    let r_c_pred: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .enumerate()
        .map(|(i, (&si, &yi))| if is_eq_ext[i] { 0.0 } else { -si * yi })
        .collect();

    let r_p_mod_pred: Vec<f64> = r_primal
        .iter()
        .zip(r_c_pred.iter())
        .zip(y.iter())
        .enumerate()
        .map(
            |(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] {
                    rpi
                } else {
                    rpi - rci / yi
                }
            },
        )
        .collect();

    let mut dx = vec![0.0_f64; r_dual.len()];
    let mut dy_pred = vec![0.0_f64; m_ext];
    solver.solve(r_dual, &r_p_mod_pred, &mut dx, &mut dy_pred, deadline);

    let mut ds_pred = vec![0.0_f64; m_ext];
    compute_ds_boxed(
        solver.m_real,
        solver.bound,
        is_eq_ext,
        &r_c_pred,
        y,
        sigma_vec,
        &dy_pred,
        &dx,
        &mut ds_pred,
    );

    let alpha_s_pred = fraction_to_boundary_masked(s, &ds_pred, TAU, is_eq_ext);
    let alpha_y_pred = fraction_to_boundary_masked(y, &dy_pred, TAU, is_eq_ext);
    let alpha_pred = alpha_s_pred.min(alpha_y_pred);

    let mu_aff: f64 = if m_ineq > 0 {
        s.iter()
            .zip(y.iter())
            .zip(ds_pred.iter())
            .zip(dy_pred.iter())
            .enumerate()
            .filter(|&(i, _)| !is_eq_ext[i])
            .map(|(_, (((&si, &yi), &dsi), &dyi))| {
                (si + alpha_pred * dsi) * (yi + alpha_pred * dyi)
            })
            .sum::<f64>()
            / m_ineq as f64
    } else {
        0.0
    };

    let sigma_center = if mu > CENTERING_MU_FLOOR {
        (mu_aff / mu).powi(3).min(1.0)
    } else {
        0.0
    };

    PredictorResult {
        dy_pred,
        ds_pred,
        sigma_center,
    }
}

/// box 化修正子ステップ。戻り値 r_c_corr は続く Gondzio correctors に渡す。
#[allow(clippy::too_many_arguments)]
pub(crate) fn corrector_boxed(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    pred: &PredictorResult,
    mu: f64,
    r_dual: &[f64],
    r_primal: &[f64],
    sigma_vec: &[f64],
    solver: &BoxedSolve<'_>,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
    deadline: Option<std::time::Instant>,
) -> (f64, Vec<f64>) {
    let r_c_corr: Vec<f64> = s
        .iter()
        .zip(y.iter())
        .zip(pred.ds_pred.iter())
        .zip(pred.dy_pred.iter())
        .enumerate()
        .map(|(i, (((&si, &yi), &dsi), &dyi))| {
            if is_eq_ext[i] {
                0.0
            } else {
                pred.sigma_center * mu - si * yi - dsi * dyi
            }
        })
        .collect();

    let r_p_mod_corr: Vec<f64> = r_primal
        .iter()
        .zip(r_c_corr.iter())
        .zip(y.iter())
        .enumerate()
        .map(
            |(i, ((&rpi, &rci), &yi))| {
                if is_eq_ext[i] {
                    rpi
                } else {
                    rpi - rci / yi
                }
            },
        )
        .collect();

    solver.solve(r_dual, &r_p_mod_corr, dx, dy, deadline);

    compute_ds_boxed(
        solver.m_real,
        solver.bound,
        is_eq_ext,
        &r_c_corr,
        y,
        sigma_vec,
        dy,
        dx,
        ds,
    );

    let alpha_s = fraction_to_boundary_masked(s, ds, TAU, is_eq_ext);
    let alpha_y = fraction_to_boundary_masked(y, dy, TAU, is_eq_ext);
    let alpha = alpha_s.min(alpha_y);

    (alpha, r_c_corr)
}

/// box 化 Gondzio centering correctors。停止基準は加算形 (audit 166 P1-A の
/// sentinel は Schur 変種 `gondzio_correctors_schur` 経由で維持)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn gondzio_correctors_boxed(
    s: &[f64],
    y: &[f64],
    is_eq_ext: &[bool],
    m_ineq: usize,
    r_dual: &[f64],
    r_primal: &[f64],
    r_c_corr: &[f64],
    sigma_vec: &[f64],
    solver: &BoxedSolve<'_>,
    max_correctors: usize,
    alpha_init: f64,
    dx: &mut [f64],
    dy: &mut [f64],
    ds: &mut [f64],
    deadline: Option<std::time::Instant>,
) -> f64 {
    let m_ext = s.len();
    let mut alpha_prev = alpha_init;
    for _k in 0..max_correctors {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        let alpha_target = (alpha_prev + BETA_GONDZIO * (1.0 - alpha_prev)).min(1.0);
        let mu_target: f64 = if m_ineq > 0 {
            s.iter()
                .zip(y.iter())
                .zip(ds.iter().zip(dy.iter()))
                .enumerate()
                .filter(|&(i, _)| !is_eq_ext[i])
                .map(|(_, ((&si, &yi), (&dsi, &dyi)))| {
                    (si + alpha_target * dsi) * (yi + alpha_target * dyi)
                })
                .sum::<f64>()
                / m_ineq as f64
        } else {
            0.0
        };
        let mu_target = mu_target.max(0.0);

        let target_lo = GAMMA_L * mu_target;
        let target_hi = GAMMA_U * mu_target;

        let mut r_c_gondzio = vec![0.0_f64; m_ext];
        for i in 0..m_ext {
            if is_eq_ext[i] {
                r_c_gondzio[i] = 0.0;
                continue;
            }
            let si_new = s[i] + alpha_prev * ds[i];
            let yi_new = y[i] + alpha_prev * dy[i];
            let v_i = si_new * yi_new;
            let v_target = if v_i < target_lo {
                target_lo - v_i
            } else if v_i > target_hi {
                target_hi - v_i
            } else {
                0.0
            };
            r_c_gondzio[i] = r_c_corr[i] + v_target;
        }

        let r_p_mod_gondzio: Vec<f64> = r_primal
            .iter()
            .zip(r_c_gondzio.iter())
            .zip(y.iter())
            .enumerate()
            .map(
                |(i, ((&rpi, &rci), &yi))| {
                    if is_eq_ext[i] {
                        rpi
                    } else {
                        rpi - rci / yi
                    }
                },
            )
            .collect();

        let mut dx_new = vec![0.0_f64; r_dual.len()];
        let mut dy_new = vec![0.0_f64; m_ext];
        solver.solve(r_dual, &r_p_mod_gondzio, &mut dx_new, &mut dy_new, deadline);

        let mut ds_new = vec![0.0_f64; m_ext];
        compute_ds_boxed(
            solver.m_real,
            solver.bound,
            is_eq_ext,
            &r_c_gondzio,
            y,
            sigma_vec,
            &dy_new,
            &dx_new,
            &mut ds_new,
        );

        let alpha_s_new = fraction_to_boundary_masked(s, &ds_new, TAU, is_eq_ext);
        let alpha_y_new = fraction_to_boundary_masked(y, &dy_new, TAU, is_eq_ext);
        let alpha_new = alpha_s_new.min(alpha_y_new);

        if alpha_new < alpha_prev + ALPHA_IMPROVE_THRESHOLD {
            break;
        }

        dx.copy_from_slice(&dx_new);
        dy.copy_from_slice(&dy_new);
        ds.copy_from_slice(&ds_new);
        alpha_prev = alpha_new;
    }
    alpha_prev
}

/// 等式行は s=0 維持で y のみ更新、不等式行は s,y 両方を 1e-12 floor で更新。
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_variables(
    x: &mut [f64],
    s: &mut [f64],
    y: &mut [f64],
    dx: &[f64],
    ds: &[f64],
    dy: &[f64],
    alpha: f64,
    is_eq_ext: &[bool],
) {
    for i in 0..x.len() {
        x[i] += alpha * dx[i];
    }
    let m_ext = s.len();
    for i in 0..m_ext {
        if is_eq_ext[i] {
            y[i] += alpha * dy[i];
        } else {
            s[i] += alpha * ds[i];
            y[i] += alpha * dy[i];
            if s[i] <= 0.0 {
                s[i] = 1e-12;
            }
            if y[i] <= 0.0 {
                y[i] = 1e-12;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::print_stderr)]
mod tests {
    use super::{
        compute_sigma_vec, gondzio_correctors_boxed, try_solve_kkt_via_schur, update_variables,
        BoxedBackend, BoxedSolve,
    };
    use crate::qp::ipm_core::kkt::{build_augmented_system, build_schur_system};
    use crate::qp::ipm_core::ALPHA_IMPROVE_THRESHOLD;
    use otspot_num::linalg::amd::amd_with_deadline;
    use otspot_num::sparse::CscMatrix;

    /// 多制約・等式・不等式混在で σ 幅広い range の下、Schur が augmented と数値一致すること。
    #[test]
    fn test_schur_matches_augmented_realistic() {
        let n = 4;
        let m_ext = 6;

        let q = CscMatrix::from_triplets(&[0, 1, 2, 3], &[0, 1, 2, 3], &[2.0, 4.0, 0.5, 1.0], n, n)
            .unwrap();

        // A_ext: 6×4 にいくつかの非ゼロ
        // 行 0: x0 + x1 (eq)
        // 行 1: x2 + x3 (eq)
        // 行 2: x0 (lb)
        // 行 3: x1 (lb)
        // 行 4: -x2 (ub)
        // 行 5: x0 + x3 (mixed)
        let rows = vec![0, 0, 1, 1, 2, 3, 4, 5, 5];
        let cols = vec![0, 1, 2, 3, 0, 1, 2, 0, 3];
        let vals = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0];
        let a_ext = CscMatrix::from_triplets(&rows, &cols, &vals, m_ext, n).unwrap();

        // sigma: equality は 0、inequality は様々な値 (LISWET 風 dynamic range)
        let sigma_vec = vec![0.0, 0.0, 1e-3, 1e1, 1e3, 5e-2];
        let rho_p = 0.05_f64;
        let delta_d = 0.02_f64;

        let aug_mat = build_augmented_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let aug_perm =
            amd_with_deadline(aug_mat.nrows(), aug_mat.col_ptr(), aug_mat.row_ind(), None);
        let aug_fac = otspot_num::linalg::kkt_solver::KktFactor::Direct(
            otspot_num::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
                &aug_mat,
                &aug_perm,
                None,
                None,
                faer::Par::Seq,
            )
            .unwrap(),
        );

        let (s_mat, d_inv) = build_schur_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let s_perm = amd_with_deadline(s_mat.nrows(), s_mat.col_ptr(), s_mat.row_ind(), None);
        let s_fac = otspot_num::linalg::kkt_solver::KktFactor::Direct(
            otspot_num::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
                &s_mat,
                &s_perm,
                None,
                None,
                faer::Par::Seq,
            )
            .unwrap(),
        );

        let r_d = vec![0.5, -1.0, 0.2, 0.8];
        let r_p_mod = vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6];

        let mut rhs_aug = vec![0.0_f64; n + m_ext];
        let mut sol_aug = vec![0.0_f64; n + m_ext];
        rhs_aug[..n].copy_from_slice(&r_d);
        rhs_aug[n..].copy_from_slice(&r_p_mod);
        aug_fac.solve(&rhs_aug, &mut sol_aug).unwrap();
        let dx_aug = sol_aug[..n].to_vec();
        let dy_aug = sol_aug[n..].to_vec();

        let mut dx_schur = vec![0.0_f64; n];
        let mut dy_schur = vec![0.0_f64; m_ext];
        try_solve_kkt_via_schur(
            &s_fac,
            &d_inv,
            &a_ext,
            &r_d,
            &r_p_mod,
            &mut dx_schur,
            &mut dy_schur,
            None,
        )
        .unwrap();

        eprintln!("dx_aug   = {:?}", dx_aug);
        eprintln!("dx_schur = {:?}", dx_schur);
        eprintln!("dy_aug   = {:?}", dy_aug);
        eprintln!("dy_schur = {:?}", dy_schur);
        for i in 0..n {
            let diff = (dx_aug[i] - dx_schur[i]).abs();
            let scale = dx_aug[i].abs().max(dx_schur[i].abs()).max(1e-12);
            assert!(
                diff / scale < 1e-6,
                "dx[{}]: aug={}, schur={}, rel_diff={}",
                i,
                dx_aug[i],
                dx_schur[i],
                diff / scale
            );
        }
        for i in 0..m_ext {
            let diff = (dy_aug[i] - dy_schur[i]).abs();
            let scale = dy_aug[i].abs().max(dy_schur[i].abs()).max(1e-12);
            assert!(
                diff / scale < 1e-6,
                "dy[{}]: aug={}, schur={}, rel_diff={}",
                i,
                dy_aug[i],
                dy_schur[i],
                diff / scale
            );
        }
    }

    /// 2 変数 1 制約で Schur と augmented LDL が同じ (dx, dy) を出すこと。
    #[test]
    fn test_schur_matches_augmented() {
        let n = 2;
        let m_ext = 1;

        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 4.0], n, n).unwrap();

        let a_ext = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], m_ext, n).unwrap();

        let sigma_vec = vec![0.5_f64];
        let rho_p = 0.1_f64;
        let delta_d = 0.05_f64;

        let aug_mat = build_augmented_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let perm: Vec<usize> = (0..aug_mat.nrows()).collect();
        let aug_fac = otspot_num::linalg::kkt_solver::KktFactor::Direct(
            otspot_num::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
                &aug_mat,
                &perm,
                None,
                None,
                faer::Par::Seq,
            )
            .unwrap(),
        );

        let (s_mat, d_inv) = build_schur_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let s_perm: Vec<usize> =
            amd_with_deadline(s_mat.nrows(), s_mat.col_ptr(), s_mat.row_ind(), None);
        let s_fac = otspot_num::linalg::kkt_solver::KktFactor::Direct(
            otspot_num::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
                &s_mat,
                &s_perm,
                None,
                None,
                faer::Par::Seq,
            )
            .unwrap(),
        );

        let r_d = vec![1.0, 2.0];
        let r_p_mod = vec![3.0];

        let mut rhs_aug = vec![0.0_f64; n + m_ext];
        let mut sol_aug = vec![0.0_f64; n + m_ext];
        rhs_aug[..n].copy_from_slice(&r_d);
        rhs_aug[n..].copy_from_slice(&r_p_mod);
        aug_fac.solve(&rhs_aug, &mut sol_aug).unwrap();
        let dx_aug = sol_aug[..n].to_vec();
        let dy_aug = sol_aug[n..].to_vec();

        let mut dx_schur = vec![0.0_f64; n];
        let mut dy_schur = vec![0.0_f64; m_ext];
        try_solve_kkt_via_schur(
            &s_fac,
            &d_inv,
            &a_ext,
            &r_d,
            &r_p_mod,
            &mut dx_schur,
            &mut dy_schur,
            None,
        )
        .unwrap();

        eprintln!("dx_aug = {:?}", dx_aug);
        eprintln!("dx_schur = {:?}", dx_schur);
        eprintln!("dy_aug = {:?}", dy_aug);
        eprintln!("dy_schur = {:?}", dy_schur);
        for i in 0..n {
            assert!(
                (dx_aug[i] - dx_schur[i]).abs() < 1e-9,
                "dx[{}]: aug={}, schur={}",
                i,
                dx_aug[i],
                dx_schur[i]
            );
        }
        for i in 0..m_ext {
            assert!(
                (dy_aug[i] - dy_schur[i]).abs() < 1e-9,
                "dy[{}]: aug={}, schur={}",
                i,
                dy_aug[i],
                dy_schur[i]
            );
        }
    }

    /// 削除した `schur_solve_honors_expired_deadline` の named successor
    /// (helper の契約変更をその場の assert 編集ではなく、削除 + 新規追加で
    /// 表現している)。
    ///
    /// 旧契約が変わった理由: 旧 `solve_kkt_via_schur` は戻り値なしで、期限切れ
    /// deadline 時に helper 内部で dx/dy をゼロ化する契約だった。境界行 box 化で
    /// 求解経路が `BoxedSolve::solve` に統合され、helper (`try_solve_kkt_via_schur`)
    /// は失敗を Err で返し、バッファのゼロ化は呼び出し側 (BoxedSolve) の責務に
    /// 移った。ゼロ化契約自体は同ファイルの
    /// `boxed_solve_failure_zeroes_bound_rows` が実装と同時に固定するため、本
    /// successor は「期限切れ → Err 伝播」と「生存 deadline では非自明な方向が
    /// 出る (no-op proof)」の 2 点だけを固定する。
    #[test]
    fn try_schur_solve_reports_expired_deadline_failure() {
        let s_mat = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let s_fac = otspot_num::linalg::kkt_solver::KktFactor::Iterative(
            otspot_num::linalg::kkt_solver::PreconditionedMinres::new(s_mat),
        );
        let a_ext = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let d_inv = vec![1.0];
        let r_d = vec![1.0];
        let r_p_mod = vec![0.0];

        // no-op proof: live deadline yields non-zero direction on this system.
        let mut dx_live = vec![0.0];
        let mut dy_live = vec![0.0];
        try_solve_kkt_via_schur(
            &s_fac,
            &d_inv,
            &a_ext,
            &r_d,
            &r_p_mod,
            &mut dx_live,
            &mut dy_live,
            Some(std::time::Instant::now() + std::time::Duration::from_secs(1)),
        )
        .expect("live deadline must solve");
        assert!(
            dx_live[0].abs() > 1e-12,
            "live deadline should solve: dx={dx_live:?}"
        );

        // sentinel: expired deadline must signal failure to the caller. If the
        // deadline were dropped (`None`) the inner solve would succeed on this
        // tiny 1x1 system and the call would return Ok.
        let mut dx_expired = vec![3.0];
        let mut dy_expired = vec![-2.0];
        let failed = try_solve_kkt_via_schur(
            &s_fac,
            &d_inv,
            &a_ext,
            &r_d,
            &r_p_mod,
            &mut dx_expired,
            &mut dy_expired,
            Some(std::time::Instant::now() - std::time::Duration::from_millis(1)),
        )
        .is_err();
        assert!(
            failed,
            "expired deadline must short-circuit Schur solve: dx={dx_expired:?} dy={dy_expired:?}",
        );
    }

    /// box 化求解の deadline 伝播: Schur backend が失敗したら dx/dy は実行・境界行
    /// ともに完全ゼロ (部分埋まりの方向を上位へ渡さない契約)。境界行 dy は
    /// dx 復元後に埋まるため、ゼロ化が尾部まで届くことを sentinel する。
    #[test]
    fn boxed_solve_failure_zeroes_bound_rows() {
        let s_mat = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let s_fac = otspot_num::linalg::kkt_solver::KktFactor::Iterative(
            otspot_num::linalg::kkt_solver::PreconditionedMinres::new(s_mat),
        );
        let a_real = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let d_inv_real = vec![1.0];
        // x0 ≤ 5 の ub 行 1 本。
        let bound = crate::qp::ipm_core::kkt::BoundBox {
            var: vec![0],
            sign: vec![1.0],
            rhs: vec![5.0],
            eliminated: vec![true],
        };
        let d_b = vec![1.0];
        let sys_ext_rows = vec![0usize];
        let solver = BoxedSolve {
            backend: BoxedBackend::Schur {
                s_fac: &s_fac,
                a_real: &a_real,
                d_inv_real: &d_inv_real,
            },
            bound: &bound,
            d_b: &d_b,
            m_real: 1,
            sys_ext_rows: &sys_ext_rows,
        };
        let mut dx = vec![7.0];
        let mut dy = vec![9.0, -9.0];
        solver.solve(
            &[1.0],
            &[0.0, 0.0],
            &mut dx,
            &mut dy,
            Some(std::time::Instant::now() - std::time::Duration::from_millis(1)),
        );
        assert!(
            dx.iter().all(|v| *v == 0.0) && dy.iter().all(|v| *v == 0.0),
            "failed solve must zero real and bound dy rows: dx={dx:?} dy={dy:?}"
        );
    }

    #[test]
    fn test_compute_sigma_vec_eq_row_is_zero() {
        let s = vec![2.0, 4.0];
        let y = vec![1.0, 2.0];
        let is_eq_ext = vec![true, false];
        let sigma_max = 1e6_f64;
        let result = compute_sigma_vec(&s, &y, &is_eq_ext, sigma_max);
        assert_eq!(result[0], 0.0);
        assert!((result[1] - 2.0).abs() < 1e-12, "got {}", result[1]);
    }

    #[test]
    fn test_update_variables_alpha_one() {
        let mut x = vec![1.0, 2.0];
        let mut s = vec![0.5, 0.5];
        let mut y = vec![1.0, 1.0];
        let dx = vec![0.1, 0.2];
        let ds = vec![0.3, -0.6];
        let dy = vec![0.1, 0.1];
        let is_eq_ext = vec![false, false];
        update_variables(&mut x, &mut s, &mut y, &dx, &ds, &dy, 1.0, &is_eq_ext);
        assert!((x[0] - 1.1).abs() < 1e-12);
        assert!((x[1] - 2.2).abs() < 1e-12);
        assert!((s[0] - 0.8).abs() < 1e-12);
        assert_eq!(s[1], 1e-12);
        assert!((y[0] - 1.1).abs() < 1e-12);
        assert!((y[1] - 1.1).abs() < 1e-12);
    }

    /// Regression sentinel for audit 166 P1-A: the Gondzio corrector loop must use the
    /// additive stop condition (`alpha_new < alpha_prev + threshold`), not the multiplicative
    /// drift form (`alpha_new <= alpha_prev * threshold`). Box 化後は box 化 corrector が
    /// 唯一の生産経路なので sentinel もそこへ移設 (旧 Schur 変種の削除に伴い削除+新規)。
    ///
    /// With all-equality constraints, `fraction_to_boundary_masked` skips every row, so
    /// `alpha_new = 1.0` regardless of the correction direction.
    /// Setting `alpha_init = 0.9999`:
    ///   - additive (correct): `1.0 < 0.9999 + 0.001` → true → early stop → returns 0.9999
    ///   - multiplicative (drift): `1.0 <= 0.9999 * 0.001` → false → loop runs to completion → returns 1.0
    #[test]
    fn test_gondzio_boxed_stop_additive_sentinel() {
        let n = 2;
        let m_ext = 1;

        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        // Single equality constraint: x0 + x1 = 1
        let a_ext = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], m_ext, n).unwrap();

        let is_eq_ext = vec![true]; // all equality
        let sigma_vec = vec![0.0_f64];
        let rho_p = 0.1_f64;
        let delta_d = 0.05_f64;

        let (s_mat, d_inv) = build_schur_system(&q, &a_ext, &sigma_vec, rho_p, delta_d);
        let s_perm = amd_with_deadline(s_mat.nrows(), s_mat.col_ptr(), s_mat.row_ind(), None);
        let s_fac = otspot_num::linalg::kkt_solver::KktFactor::Direct(
            otspot_num::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
                &s_mat,
                &s_perm,
                None,
                None,
                faer::Par::Seq,
            )
            .unwrap(),
        );

        // alpha_init just below 1.0: improvement = 1.0 - 0.9999 = 0.0001 < ALPHA_IMPROVE_THRESHOLD
        // so the additive check fires on the first iteration and the loop returns alpha_init.
        let alpha_init = 0.9999_f64;
        let s = vec![0.0_f64; m_ext];
        let y = vec![1.0_f64; m_ext];
        let r_dual = vec![0.0_f64; n];
        let r_primal = vec![0.0_f64; m_ext];
        let r_c_corr = vec![0.0_f64; m_ext];
        let sigma_vec_loop = vec![0.0_f64; m_ext];
        let mut dx = vec![0.0_f64; n];
        let mut dy = vec![0.0_f64; m_ext];
        let mut ds = vec![0.0_f64; m_ext];

        // 境界行なしの box 化求解 (m_real = m_ext)。
        let bound = crate::qp::ipm_core::kkt::BoundBox {
            var: vec![],
            sign: vec![],
            rhs: vec![],
            eliminated: vec![],
        };
        let d_b: Vec<f64> = vec![];
        let sys_ext_rows: Vec<usize> = vec![0];
        let solver = BoxedSolve {
            backend: BoxedBackend::Schur {
                s_fac: &s_fac,
                a_real: &a_ext,
                d_inv_real: &d_inv,
            },
            bound: &bound,
            d_b: &d_b,
            m_real: m_ext,
            sys_ext_rows: &sys_ext_rows,
        };

        let result = gondzio_correctors_boxed(
            &s,
            &y,
            &is_eq_ext,
            0,
            &r_dual,
            &r_primal,
            &r_c_corr,
            &sigma_vec_loop,
            &solver,
            /*max_correctors=*/ 5,
            alpha_init,
            &mut dx,
            &mut dy,
            &mut ds,
            None,
        );

        // With the fix (additive): loop breaks on first iteration → returns alpha_init.
        // Under drift (multiplicative): 1.0 <= 0.9999 * 0.001 = false → loop continues
        // → alpha_prev updated to 1.0 → returns 1.0.
        assert!(
            (result - alpha_init).abs() < 1e-12,
            "expected early stop (additive criterion), got result={} (drift would return 1.0); \
             ALPHA_IMPROVE_THRESHOLD={}",
            result,
            ALPHA_IMPROVE_THRESHOLD,
        );
    }
}
