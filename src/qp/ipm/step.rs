//! IPM ステップ計算（Mehrotra predictor-corrector）
//!
//! - メインループ (`solve_qp_ipm_inner`)
//! - 制約なし QP (`solve_unconstrained`)
//! - fraction-to-boundary
//! - ユーティリティ

use crate::linalg::ldl;
use crate::linalg::ruiz::RuizScaler;
use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveStatus, SolverResult};
use crate::qp::problem::QpProblem;
use super::kkt::{
    build_augmented_system, build_extended_constraints,
    collect_part1_diag_indices, collect_part3_diag_indices,
    collect_q_diag_base, update_augmented_values,
    norm_inf, spmtv, spmv, spmv_q, KktCache,
};
use crate::linalg::amd::amd_with_deadline;
use crate::linalg::ldl::LdlFactorizationAmd;
use super::init::compute_initial_point;
use super::common::{check_infeasible_or_unbounded, solve_unconstrained, timeout_result, numerical_error_result};
use super::solver_loop::{compute_sigma_vec, predictor_step, corrector_step, gondzio_correctors, update_variables};
use super::kkt::collapse_extended_dual;



// ---------------------------------------------------------------------------
// IPM 内部ソルバー
// ---------------------------------------------------------------------------

/// IPM内部ソルバー（Ruizスケーリング適用済みproblemを受け取る）
///
/// augmented KKT system + LDLT（DirectLDL一本化）
pub(crate) fn solve_qp_ipm_inner(
    problem: &QpProblem,
    options: &SolverOptions,
    scaler: Option<&RuizScaler>,
    orig_problem: Option<&QpProblem>,
    eps_orig: f64,
) -> SolverResult {
    let n = problem.num_vars;
    let timeout_ctx = TimeoutCtx::from_options(options);

    // T1: 処理前タイムアウトチェック
    if timeout_ctx.should_stop() {
        return timeout_result(n);
    }

    // 制約なし特殊ケース
    if problem.num_constraints == 0
        && problem.bounds.iter().all(|&(lb, ub)| lb.is_infinite() && ub.is_infinite())
    {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 拡張制約行列を構築（6-tuple: is_eq_ext追加）
    let (a_ext, b_ext, m_ext, m_orig, _n_lb, is_eq_ext) = build_extended_constraints(problem);

    if m_ext == 0 {
        return solve_unconstrained(problem, &timeout_ctx);
    }

    // 等式行数と不等式行数
    let eq_count = is_eq_ext.iter().filter(|&&v| v).count();
    let m_ineq = m_ext - eq_count;

    // 初期点（eq行: s=0, y=0）
    let (mut x, mut s, mut y) = compute_initial_point(n, &b_ext, &is_eq_ext);

    // 作業バッファ
    let mut ax = vec![0.0f64; m_ext];
    let mut aty = vec![0.0f64; n];
    let mut qx = vec![0.0f64; n];
    let mut r_d = vec![0.0f64; n];
    let mut r_p = vec![0.0f64; m_ext];

    let mut dx = vec![0.0f64; n];
    let mut dy = vec![0.0f64; m_ext];
    let mut ds = vec![0.0f64; m_ext];
    // AMD permutation キャッシュ（augmented system のスパースパターンは反復間で不変）
    let mut amd_perm_cache: Option<Vec<usize>> = None;
    // KKT 差分更新キャッシュ（方式 D: values 直接更新 + refactorize_numeric 再利用）
    let mut kkt_cache: Option<KktCache> = None;
    let mut fac_cache: Option<LdlFactorizationAmd> = None;

    let mut status = SolveStatus::Timeout;
    let mut final_iter = options.ipm.max_iter;
    let mut final_residuals: Option<(f64, f64, f64)> = None;
    // 双対ギャップ（scaled）。毎反復更新し、return 時 SolverResult.duality_gap_rel に populate。
    // ippmm.rs:340-350 と同一定義で、ipm 側の rel_gap ゲート非対称性を解消する。
    let mut final_rel_gap: f64 = f64::INFINITY;

    // 収束判定に併用する相対ギャップ閾値。ippmm.rs 内部判定と同一。
    const DUALITY_GAP_TOL: f64 = 1e-3;

    // C-1: mu非依存proximal正則化（rho_ipmフロア）
    let mut rho_ipm = 1e-4_f64;

    for iter in 0..options.ipm.max_iter {
        // T3: 反復先頭タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // 残差計算
        spmv(&a_ext, &x, &mut ax);
        spmtv(&a_ext, &y, &mut aty);
        spmv_q(&problem.q, &x, &mut qx);

        for i in 0..n {
            r_d[i] = -(qx[i] + problem.c[i] + aty[i]);
        }
        for i in 0..m_ext {
            r_p[i] = b_ext[i] - ax[i] - s[i];
        }

        // μ = s^T y / m_ineq（相補性ギャップ、等式行除外）
        let mu: f64 = if m_ineq > 0 {
            s.iter().zip(y.iter()).zip(is_eq_ext.iter())
                .filter(|&(_, &eq)| !eq)
                .map(|((&si, &yi), _)| si * yi)
                .sum::<f64>() / m_ineq as f64
        } else {
            0.0
        };

        // 最終残差を更新（収束・MaxIterations・Timeout いずれの場合も最後の値を保持）
        final_residuals = Some((norm_inf(&r_p), norm_inf(&r_d), mu));

        // 双対ギャップ（scaled）を算出。ippmm.rs:340-350 と同一定義。
        // 符号規約: r_d = -(Qx + c + A^T y) → dual = -0.5 x^T Q x - Σ b_ext·y。
        // UBH1 (||x||≈1459, c=0, Q rank-deficient) では r_stat=2e-6・mu=1e-30 でも
        // duality_gap = 9.49（obj 91% 誤差）になる既知事例があり、rel_gap ゲートで棄却する。
        let qx_dot_x: f64 = qx.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        let c_dot_x: f64 = problem.c.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
        let p_obj_s = 0.5 * qx_dot_x + c_dot_x;
        let mut d_lin: f64 = 0.0;
        for i in 0..m_ext {
            d_lin -= b_ext[i] * y[i];
        }
        let d_obj_s = -0.5 * qx_dot_x + d_lin;
        let gap_abs = p_obj_s - d_obj_s;
        let gap_denom = p_obj_s.abs().max(d_obj_s.abs()).max(1.0);
        let rel_gap = gap_abs / gap_denom;
        final_rel_gap = rel_gap;

        // 収束判定: 混合許容誤差 eps_abs + eps_rel * norm (Gurobi方式)
        // prim: ||r_p|| < eps * (1 + norm_b), dual: ||r_d|| < eps * (1 + norm_c)
        let norm_c = norm_inf(&problem.c).max(1.0);
        let norm_b = norm_inf(&b_ext).max(1.0);
        let eps = options.ipm_eps();

        if norm_inf(&r_d) < eps * (1.0 + norm_c)
            && norm_inf(&r_p) < eps * (1.0 + norm_b)
            && mu < eps
            && rel_gap.abs() < DUALITY_GAP_TOL
        {
            status = SolveStatus::Optimal;
            final_iter = iter;
            break;
        }

        // μ が正則化下限以下まで収縮し、残差も eps 水準以下なら SuboptimalSolution
        // （delta_min バイアスにより完全収束不能だが実用精度に達している状態を検出）
        // 閾値 = max(eps*(1+norm), delta_min*10): 正則化限界(~delta_min)の10倍をフロアとする
        // delta_min*100(旧)はpv_retry時にeps連動を阻害しDTOC3退行を引き起こした
        let thr_d = (eps * (1.0 + norm_c)).max(options.ipm.delta_min * 10.0);
        let thr_p = (eps * (1.0 + norm_b)).max(options.ipm.delta_min * 10.0);
        if mu < options.ipm.delta_min * 1e-2
            && norm_inf(&r_d) < thr_d
            && norm_inf(&r_p) < thr_p
            && rel_gap.abs() < DUALITY_GAP_TOL
        {
            // ── Method C: 原空間pfeasチェック（Clarabel方式）──
            if let (Some(sc), Some(orig)) = (scaler, orig_problem) {
                let m_orig_check = orig.b.len();
                let pfeas_orig = if m_orig_check == 0 {
                    0.0
                } else {
                    let n_orig = orig.num_vars;
                    let mut ax_orig = vec![0.0_f64; m_orig_check];
                    for (j, (&dj, &xj)) in sc.d[..n_orig].iter().zip(x[..n_orig].iter()).enumerate() {
                        let dj_xj = dj * xj;
                        for ptr in orig.a.col_ptr[j]..orig.a.col_ptr[j + 1] {
                            let row = orig.a.row_ind[ptr];
                            if row < m_orig_check {
                                ax_orig[row] += orig.a.values[ptr] * dj_xj;
                            }
                        }
                    }
                    ax_orig
                        .iter()
                        .zip(orig.b.iter())
                        .zip(orig.constraint_types.iter())
                        .map(|((&axi, &bi), ct)| match ct {
                            ConstraintType::Eq => (axi - bi).abs(),
                            ConstraintType::Ge => (bi - axi).max(0.0),
                            _ => (axi - bi).max(0.0),
                        })
                        .fold(0.0_f64, f64::max)
                };
                let norm_b_orig = norm_inf(&orig.b).max(1.0);
                if pfeas_orig < eps_orig * (1.0 + norm_b_orig)
                    && norm_inf(&r_d) < eps_orig * (1.0 + norm_c)
                    && mu < eps_orig
                    && rel_gap.abs() < DUALITY_GAP_TOL
                {
                    status = SolveStatus::Optimal;
                    final_iter = iter;
                    break;
                }
            }
            // Method Cで昇格できなかった場合 or scaler=None → SuboptimalSolution
            status = SolveStatus::SuboptimalSolution;
            final_iter = iter;
            break;
        }

        // δ を μ に追従して縮小（IP-PMM）
        let delta_p = options.ipm.delta_min.max(options.ipm.delta_p_init * mu);
        let delta_d = options.ipm.delta_min.max(options.ipm.delta_d_init * mu);
        // C-1: rho_ipmフロアによるmu非依存proximal正則化
        let effective_delta_p = delta_p.max(rho_ipm);

        // Σ = diag(s_i / y_i)（等式行は0）
        // y→0 のとき si/yi→Inf になる場合がある。sigma_max = 1/delta_min でクリップ
        let sigma_max = 1.0 / options.ipm.delta_min.max(1e-15);
        let sigma_vec = compute_sigma_vec(&s, &y, &is_eq_ext, sigma_max);

        // ===== LDLパス: augmented system + factorize_quasidefinite_with_deadline =====

        // T2: 因子化前タイムアウトチェック
        if timeout_ctx.should_stop() {
            status = SolveStatus::Timeout;
            final_iter = iter;
            break;
        }

        // augmented KKT行列構築 + factorize（delta_p/delta_d リトライ最大10回, 上限1e0）
        // 方式 D: 初回は full 構築 + symbolic/numeric 全因子化、2反復目以降は values 差分更新
        //         + refactorize_numeric（symbolic 再利用）で O(nnz log nnz) → O(n + m_ext) に削減。
        // AMD permutation はスパースパターン不変なので初回のみ計算してキャッシュ
        let mut delta_p_retry = effective_delta_p;
        let mut delta_d_retry = delta_d;
        let mut retry_timeout = false;
        'retry: for _retry in 0..10 {
            if timeout_ctx.should_stop() {
                status = SolveStatus::Timeout;
                final_iter = iter;
                retry_timeout = true;
                break 'retry;
            }
            if let Some(cache) = kkt_cache.as_mut() {
                // 2反復目以降: values のみ O(n + m_ext) で更新（高速パス）
                update_augmented_values(cache, &sigma_vec, delta_p_retry, delta_d_retry);
                // Bug-T1修正: refactorize_numeric_threaded は事実上同期実行であり
                // 大規模行列の再因子化中は deadline チェック不可（157s超過の主因）。
                // factorize_quasidefinite_with_cached_perm_threaded（真のスレッド版）に統一する。
                // symbolic 再計算コストは増えるが deadline 安全性が保証される。
                let perm = amd_perm_cache.as_ref().unwrap();
                match ldl::factorize_quasidefinite_with_cached_perm_threaded(
                    &cache.mat, perm, timeout_ctx.deadline
                ) {
                    Ok(f) => {
                        fac_cache = Some(f);
                        break 'retry;
                    }
                    Err(ldl::LdlError::DeadlineExceeded) => {
                        status = SolveStatus::Timeout;
                        final_iter = iter;
                        retry_timeout = true;
                        break 'retry;
                    }
                    Err(_) => {
                        // SingularOrIndefinite → delta_p 増加してリトライ
                        if delta_p_retry >= 1e0 {
                            // GQ-01: 高速パス全リトライ失敗時 fac_cache を無効化
                            // → M-02 チェックで NumericalError を返す（stale cache で solve しない）
                            fac_cache = None;
                            break 'retry;
                        }
                        delta_p_retry = (delta_p_retry * 10.0).min(1e0);
                        delta_d_retry = (delta_d_retry * 10.0).min(1e0);
                        continue;
                    }
                }
            } else {
                // 初回: KKT 行列を full 構築し、インデックスを収集
                let aug_mat = build_augmented_system(
                    &problem.q, &a_ext, &sigma_vec, delta_p_retry, delta_d_retry,
                );
                // 初回のみ AMD permutation を計算してキャッシュ
                if amd_perm_cache.is_none() {
                    // 第1防御: full AMD（primal/dual交互消去でfill-in最小化）
                    amd_perm_cache = Some(amd_with_deadline(aug_mat.nrows, &aug_mat.col_ptr, &aug_mat.row_ind, timeout_ctx.deadline));
                }
                let perm = amd_perm_cache.as_ref().unwrap();
                // Part 1/3 のインデックスを収集して KktCache を構築
                let part1_idx = collect_part1_diag_indices(&aug_mat, n);
                let part3_idx = collect_part3_diag_indices(&aug_mat, n, m_ext);
                let q_diag_base = collect_q_diag_base(&problem.q, n);
                kkt_cache = Some(KktCache {
                    mat: aug_mat,
                    part1_diag_idx: part1_idx,
                    q_diag_base,
                    part3_diag_idx: part3_idx,
                    part1_updated_idx: (0..n).collect(),
                });
                // 初回は symbolic + numeric の全因子化
                match ldl::factorize_quasidefinite_with_cached_perm_threaded(
                    &kkt_cache.as_ref().unwrap().mat, perm, timeout_ctx.deadline
                ) {
                    Ok(f) => { fac_cache = Some(f); break 'retry; }
                    Err(ldl::LdlError::DeadlineExceeded) => {
                        status = SolveStatus::Timeout;
                        final_iter = iter;
                        retry_timeout = true;
                        break 'retry;
                    }
                    Err(_) => {
                        kkt_cache = None; // 初回失敗時はキャッシュをリセット
                        if delta_p_retry >= 1e0 { break 'retry; }
                        delta_p_retry = (delta_p_retry * 10.0).min(1e0);
                        delta_d_retry = (delta_d_retry * 10.0).min(1e0);
                    }
                }
            }
        }
        // retry ループ後: Timeout が発生した場合は外ループを抜ける
        if retry_timeout {
            break;
        }
        // 第3防御: Identity fallback — 全リトライ失敗時に identity perm + 大きな delta で再試行
        // amd_perm_cache を無効化し、次の反復で block AMD が再計算されるようにする
        if fac_cache.is_none() {
            amd_perm_cache = None;
            drop(kkt_cache.take()); // 旧キャッシュを解放し、次行で再構築
            let delta_fallback = 1e-2_f64.max(delta_p_retry).max(delta_d_retry);
            let aug_mat_fb = build_augmented_system(
                &problem.q, &a_ext, &sigma_vec, delta_fallback, delta_fallback,
            );
            let identity_perm: Vec<usize> = (0..aug_mat_fb.nrows).collect();
            let part1_idx = collect_part1_diag_indices(&aug_mat_fb, n);
            let part3_idx = collect_part3_diag_indices(&aug_mat_fb, n, m_ext);
            let q_diag_base = collect_q_diag_base(&problem.q, n);
            kkt_cache = Some(KktCache {
                mat: aug_mat_fb,
                part1_diag_idx: part1_idx,
                q_diag_base,
                part3_diag_idx: part3_idx,
                part1_updated_idx: (0..n).collect(),
            });
            match ldl::factorize_quasidefinite_with_cached_perm_threaded(
                &kkt_cache.as_ref().unwrap().mat, &identity_perm, timeout_ctx.deadline
            ) {
                Ok(f) => { fac_cache = Some(f); }
                Err(ldl::LdlError::DeadlineExceeded) => {
                    status = SolveStatus::Timeout;
                    final_iter = iter;
                    break;
                }
                Err(_) => {
                    // identity fallback も失敗 → fac_cache は None のまま → M-02
                }
            }
            // C1バグ修正: identity perm で因子化したため kkt_cache と
            // amd_perm_cache は整合しない（amd_perm_cache=None, kkt_cache=Some の状態）。
            // kkt_cache を None にリセットし、次反復で AMD 再計算＋フル初期化させる。
            kkt_cache = None;
        }

        // M-02: fac_cache が None なら全リトライ失敗 → NumericalError
        if fac_cache.is_none() {
            return numerical_error_result(n);
        }
        let fac = fac_cache.as_ref().unwrap();
        let aug_mat_for_ir = &kkt_cache.as_ref().expect("kkt_cache must be Some when fac is").mat;

        // --- Predictor ---
        let pred = predictor_step(
            &s, &y, &is_eq_ext, m_ineq,
            &r_d, &r_p,  // r_dual=r_d, r_primal=r_p (IPM)
            &sigma_vec, fac, aug_mat_for_ir, n, m_ext, mu,
        );

        // --- Corrector ---
        let (alpha, r_c_corr) = corrector_step(
            &s, &y, &is_eq_ext,
            &pred, mu,
            &r_d, &r_p,  // r_dual=r_d, r_primal=r_p (IPM)
            &sigma_vec, fac, aug_mat_for_ir, n, m_ext,
            &mut dx, &mut dy, &mut ds,
        );

        // --- Gondzio multiple centrality correctors ---
        let mut alpha = alpha;
        if alpha < 0.999 {
            alpha = gondzio_correctors(
                &s, &y, &is_eq_ext, m_ineq,
                &r_d, &r_p,  // r_dual=r_d, r_primal=r_p (IPM)
                &r_c_corr, &sigma_vec, fac, aug_mat_for_ir, n, m_ext,
                options.ipm.max_correctors, alpha,
                &mut dx, &mut dy, &mut ds,
            );
        }

        // Infeasibility / Unboundedness 検出（augmented パス）
        if let Some(infeas_status) = check_infeasible_or_unbounded(
            &dx, &dy, problem, &a_ext, m_orig, m_ext, iter, delta_p_retry,
        ) {
            status = infeas_status;
            final_iter = iter;
            break;
        }

        // 変数更新
        update_variables(&mut x, &mut s, &mut y, &dx, &ds, &dy, alpha, &is_eq_ext);

        // C-1: rho_ipm減衰（RHO_IPM_DECAY=0.9, RHO_IPM_MIN=1e-9）
        rho_ipm = (rho_ipm * 0.9_f64).max(1e-9_f64);
    }

    // 目的関数値
    spmv_q(&problem.q, &x, &mut qx);
    let objective = 0.5
        * qx.iter().zip(x.iter()).map(|(&qi, &xi)| qi * xi).sum::<f64>()
        + problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum::<f64>();

    let dual_solution = collapse_extended_dual(&y, m_orig, &problem.constraint_types);
    let bound_duals = y[m_orig..m_ext].to_vec();

    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution,
        bound_duals,

        iterations: final_iter,
        final_residuals,
        pfeas: final_residuals.map(|(pf, _, _)| pf),
        dfeas: final_residuals.map(|(_, df, _)| df),
        gap: final_residuals.map(|(_, _, g)| g),
        // scaling.rs の rel_gap ゲート (1e-1) を発火させるため populate。
        // ippmm.rs:858 と対称。未計算 (INFINITY) は None のまま（従来挙動互換）。
        duality_gap_rel: if final_rel_gap.is_finite() { Some(final_rel_gap) } else { None },
        ..Default::default()
    }
}


// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::common::check_infeasible_or_unbounded;
    use crate::problem::SolveStatus;
    use crate::qp::problem::QpProblem;
    use crate::CscMatrix;

    /// STEP-T1: iter < MIN_ITER(=5) の場合 None が返ること
    #[test]
    fn test_iter_guard() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // MIN_DIR_NORM を超える大きさだが iter ガードが先
        let dy: Vec<f64> = vec![];
        // iter=4 < MIN_ITER=5 → None
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 4, 0.0),
            None,
            "STEP-T1: iter < MIN_ITER は None であること"
        );
    }

    /// STEP-T2: ||Δx||_inf <= MIN_DIR_NORM(=1e-3) の場合 None が返ること
    #[test]
    fn test_min_dir_norm_guard() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap();
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![5e-4]; // ||dx||_inf = 5e-4 <= MIN_DIR_NORM = 1e-3
        let dy: Vec<f64> = vec![];
        // 収束時偽陽性防止: dx が小さすぎる → None
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            None,
            "STEP-T2: ||dx||_inf <= MIN_DIR_NORM は None であること"
        );
    }

    /// STEP-T3: Farkas dual ray 条件を満たすベクトルで Infeasible 判定を確認
    ///
    /// A_orig = 0 (1x2 ゼロ行列), b = [-1], dy_orig = [2.0]
    /// ① ||A^T * dy_orig|| = 0 < ε ✓
    /// ② b · dy_orig = -2 < -ε ✓
    /// → Infeasible
    #[test]
    fn test_primal_infeasible_farkas() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
        let c = vec![1.0, 0.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap(); // 1x2 ゼロ行列
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 1, 2).unwrap();
        let dx = vec![1e-10, 1e-10]; // 非常に小さい → dual チェックはスキップ
        let dy = vec![2.0]; // norm = 2.0 > MIN_DIR_NORM
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 1, 1, 10, 0.0),
            Some(SolveStatus::Infeasible),
            "STEP-T3: Farkas ray 条件 → Infeasible であること"
        );
    }

    /// STEP-T4: LP (Q=0) で c·Δx < 0 条件の Unbounded 判定を確認
    ///
    /// n=1, m_orig=0: c=[-1], dx=[1.0] → c·dx/norm_dx = -1 < -ε → Unbounded
    #[test]
    fn test_dual_infeasible_lp() {
        let q = CscMatrix::from_triplets(&[], &[], &[], 1, 1).unwrap(); // Q=0 (LP)
        let c = vec![-1.0];
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap(); // 制約なし
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let dx = vec![1.0]; // c·dx = -1 < -ε, m_ext=0 なので dual guard は無効
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            Some(SolveStatus::Unbounded),
            "STEP-T4: LP dual infeasibility → Unbounded であること"
        );
    }

    /// STEP-T5: QP c=0 のとき Unbounded を返さないことを確認（QPLIB_9002バグ回帰防止）
    ///
    /// Q = diag([0, 2]) (1エントリのみ → is_lp=false)
    /// c = [0, 0], dx = [1.0, 0.0]
    /// 条件1: ||Q*dx||/norm_dx = 0 < EPS_INF → 通過
    /// 条件2: c^T*dx / norm_dx = 0 → NOT < -EPS_INF → 不成立
    /// → cond_obj = false → None (Unbounded不判定)
    #[test]
    fn test_qp_c_zero_not_unbounded() {
        // Q に (1,1)=2.0 のエントリのみ → is_lp=false (Q.values=[2.0]≠0)
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0]; // c=0
        let a = CscMatrix::new(0, 2);
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::new(0, 2);
        let dx = vec![1.0, 0.0]; // Q*dx = [0,0], norm_qdx=0 < EPS_INF, c_dx=0 ≥ -EPS_INF
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            None,
            "STEP-T5: QP c=0 → Unbounded不判定（QPLIB_9002回帰防止）"
        );
    }

    /// STEP-T6: QP c≠0 の真のUnbounded問題で正しく Unbounded を返すことを確認
    ///
    /// Q = diag([0, 2]) (is_lp=false), c = [-1, 0], dx = [1.0, 0.0]
    /// 条件1: ||Q*dx||/norm_dx = 0 < EPS_INF → 通過
    /// 条件2: c^T*dx / norm_dx = -1 < -EPS_INF → 通過
    /// → cond_obj = true, m_orig=0 → Unbounded
    #[test]
    fn test_qp_c_nonzero_true_unbounded() {
        // Q に (1,1)=2.0 のエントリのみ → is_lp=false
        let q = CscMatrix::from_triplets(&[1], &[1], &[2.0], 2, 2).unwrap();
        let c = vec![-1.0, 0.0]; // c≠0、x[0]方向に目的減少
        let a = CscMatrix::new(0, 2);
        let b: Vec<f64> = vec![];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new_all_le(q, c, a, b, bounds).unwrap();
        let a_ext = CscMatrix::new(0, 2);
        let dx = vec![1.0, 0.0]; // Q*dx=[0,0] → 条件1通過; c_dx=-1 → 条件2通過
        let dy: Vec<f64> = vec![];
        assert_eq!(
            check_infeasible_or_unbounded(&dx, &dy, &problem, &a_ext, 0, 0, 10, 0.0),
            Some(SolveStatus::Unbounded),
            "STEP-T6: QP c≠0 真Unbounded → Unbounded判定"
        );
    }
}

