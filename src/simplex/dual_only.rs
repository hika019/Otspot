//! Pure Dual Simplex (single source of truth for LP).
//!
//! 設計:
//! - `build_standard_form` で LP を「min c·x s.t. A·x = b, x ≥ 0, b ≥ 0」へ整形
//! - 人工変数 (Eq/Ge with b>0 など slack 不能行) を **明示列で追加**
//! - 人工列に Big-M cost、cost 摂動で初期 dual 実行可能性を確保
//! - `dual_simplex_core` (既存の DSE/Bland 実装) を Phase I で 1 回呼び出し
//! - 摂動を解いて Phase II として再呼び出し
//! - 人工変数残量 > tol で Infeasible 判定
//!
//! Primal Simplex への一切のフォールバックなし。

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::PIVOT_TOL;
use crate::basis::{BasisManager, LuBasis};
use super::{StandardForm, SimplexOutcome, build_standard_form, extract_solution};

/// 人工列に与える Big-M cost。
/// 1e6 で Netlib の |c| (≈ 1e4-1e5) を支配しつつ、f64 round-off (~1e-15 相対) で
/// 余裕。1e10 級は単純な加減算で精度が崩れるため避ける。
const BIG_M: f64 = 1e6;

/// 人工変数残量の判定閾値。これを超えると infeasibility と判定。
const ARTIFICIAL_FEAS_TOL: f64 = 1e-6;

/// LP を pure dual simplex で解く (Phase I / Phase II 統合)。
pub fn solve(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let sf = build_standard_form(problem);

    // 拡張: standard form の上に「人工変数の明示列」を被せる
    let ext = augment_with_artificials(&sf);

    let dbg = std::env::var("DUAL_ONLY_TRACE").ok().as_deref() == Some("1");
    if dbg {
        eprintln!("[DUAL_ONLY] solve: m={} n_standard={} n_ext={} n_artificial={}",
            ext.m, ext.n_standard, ext.n_ext, ext.n_ext - ext.n_standard);
    }

    // Phase I: Big-M 込みコストで dual simplex
    let phase1_result = run_dual_with_perturbation(&ext, options, /*phase=*/1);

    if dbg {
        let st = match &phase1_result.outcome {
            SimplexOutcome::Optimal(o, _) => format!("Optimal({})", o),
            SimplexOutcome::Unbounded => "Unbounded".to_string(),
            SimplexOutcome::Timeout(_) => "Timeout".to_string(),
            SimplexOutcome::SingularBasis => "SingularBasis".to_string(),
        };
        let n_art_in_basis = phase1_result.basis.iter().filter(|&&b| b >= ext.n_standard).count();
        eprintln!("[DUAL_ONLY] phase1 outcome={} n_art_in_basis={}", st, n_art_in_basis);
    }

    match phase1_result.outcome {
        SimplexOutcome::Optimal(obj, _y) => {
            // 人工変数残量チェック
            let art_max = artificial_max(&ext, &phase1_result.basis, &phase1_result.x_b);
            if dbg {
                eprintln!("[DUAL_ONLY] art_max={:.3e} threshold={:.3e}", art_max, ARTIFICIAL_FEAS_TOL);
            }
            if art_max > ARTIFICIAL_FEAS_TOL {
                return result_infeasible(&sf);
            }
            // Phase II: 摂動なしで再実行 (Big-M も外す → 元の c のみ)
            let phase2_result = run_dual_pure(&ext, options, phase1_result.basis, phase1_result.x_b);
            match phase2_result.outcome {
                SimplexOutcome::Optimal(_obj2, y2) => {
                    result_optimal(&sf, problem, &phase2_result.basis, &phase2_result.x_b, &y2, &ext)
                }
                SimplexOutcome::Unbounded => result_infeasible(&sf), // dual unbounded = primal infeasible
                SimplexOutcome::Timeout(_) => {
                    result_timeout(&sf, problem, &phase2_result.basis, &phase2_result.x_b, &ext)
                }
                SimplexOutcome::SingularBasis => SolverResult { status: SolveStatus::NumericalError, ..Default::default() },
            }
            .also(|_| { let _ = obj; }) // silence unused
        }
        SimplexOutcome::Unbounded => result_infeasible(&sf),
        SimplexOutcome::Timeout(_) => {
            result_timeout(&sf, problem, &phase1_result.basis, &phase1_result.x_b, &ext)
        }
        SimplexOutcome::SingularBasis => SolverResult { status: SolveStatus::NumericalError, ..Default::default() },
    }
}

// =====================================================================
// 拡張系 = standard_form + 明示人工列
// =====================================================================

struct Extended {
    /// 拡張制約行列 (m × n_ext)
    a: CscMatrix,
    /// 拡張 RHS
    b: Vec<f64>,
    /// 拡張コスト (元 + 0 slack + BIG_M artificial)
    c: Vec<f64>,
    /// 変数下限 (全 0、standard form 慣例)
    lb: Vec<f64>,
    /// 変数上限 (slack/structural=+inf、artificial=0 FX)
    ub: Vec<f64>,
    /// 初期基底 (m 行 ⇒ 列 index)
    initial_basis: Vec<usize>,
    /// 元 LP の standard form 部分の列数 (= n_total)。これより大きい列は artificial
    n_standard: usize,
    /// 全列数
    n_ext: usize,
    /// 行数
    m: usize,
}

fn augment_with_artificials(sf: &StandardForm) -> Extended {
    let m = sf_m(sf);
    let n_standard = sf_n_total(sf);

    // sf.needs_artificial に基づいて artificial 列を追加
    let mut n_artificial = 0usize;
    let mut art_col_of_row: Vec<Option<usize>> = vec![None; m];
    for i in 0..m {
        if sf_needs_artificial(sf, i) {
            art_col_of_row[i] = Some(n_standard + n_artificial);
            n_artificial += 1;
        }
    }
    let n_ext = n_standard + n_artificial;

    // 既存 sf.a (CSC) を triplet に展開 → artificial 列を append
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    let a_orig = sf_a(sf);
    for col in 0..a_orig.ncols {
        let cs = a_orig.col_ptr[col];
        let ce = a_orig.col_ptr[col + 1];
        for k in cs..ce {
            trip_rows.push(a_orig.row_ind[k]);
            trip_cols.push(col);
            trip_vals.push(a_orig.values[k]);
        }
    }
    // 人工列を **-e_i 係数**で追加する。
    // これにより basis 行列 B が art 行で -1 を持ち、x_B = B^{-1} b = -b < 0 となる。
    // dual_simplex_core は x_B < 0 を起点に反復するため、人工列が pivot 駆動される。
    for i in 0..m {
        if let Some(art_col) = art_col_of_row[i] {
            trip_rows.push(i);
            trip_cols.push(art_col);
            trip_vals.push(-1.0);
        }
    }
    let a_ext = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_ext)
        .expect("augment: triplet construction");

    // コスト: 標準形部分は sf.c、artificial は +BIG_M
    // B^T y = c_B で art 行は B=-1 なので y_art = -BIG_M。
    // 構造変数 c_bar_j = c_j - sum y_i A_ij = c_j + BIG_M * sum_{art rows} A_ij。
    // sum >= 0 なら c_bar very positive (dual feasible)、sum < 0 なら摂動で吸収。
    let mut c = vec![0.0_f64; n_ext];
    let c_sf = sf_c(sf);
    c[..c_sf.len()].copy_from_slice(c_sf);
    for i in 0..m {
        if let Some(art_col) = art_col_of_row[i] {
            c[art_col] = BIG_M;
        }
    }

    // 初期基底:
    //   needs_artificial の行 → artificial 列
    //   それ以外 → sf.initial_basis (slack)
    let mut initial_basis = vec![0_usize; m];
    let sf_basis = sf_initial_basis(sf);
    for i in 0..m {
        initial_basis[i] = if let Some(art_col) = art_col_of_row[i] {
            art_col
        } else {
            sf_basis[i]
        };
    }

    // bounds: standard form は全変数 lb=0, slack/structural ub=+inf。
    // 人工変数は FX [0, 0]: dual_iter_blp が UB=0 違反を検出して pivot 駆動。
    let lb = vec![0.0_f64; n_ext];
    let mut ub = vec![f64::INFINITY; n_ext];
    for i in 0..m {
        if let Some(art_col) = art_col_of_row[i] {
            ub[art_col] = 0.0;
        }
    }

    // b に lexicographic 摂動 (textbook anti-cycling)。
    // b_perturbed[i] = b[i] + epsilon * (i+1) で degeneracy を解消。
    // epsilon は 1e-10 程度で十分小さく、最適解への影響は無視可能。
    let mut b_vec = sf_b(sf).to_vec();
    let lex_eps = 1e-10_f64;
    for (i, bi) in b_vec.iter_mut().enumerate() {
        *bi += lex_eps * ((i + 1) as f64);
    }

    Extended {
        a: a_ext,
        b: b_vec,
        c,
        lb,
        ub,
        initial_basis,
        n_standard,
        n_ext,
        m,
    }
}

// =====================================================================
// dual simplex 呼び出しラッパ
// =====================================================================

struct PhaseResult {
    outcome: SimplexOutcome,
    basis: Vec<usize>,
    x_b: Vec<f64>,
}

/// Phase I: cost に dual-feasibility 摂動を適用して dual simplex
fn run_dual_with_perturbation(ext: &Extended, options: &SolverOptions, _phase: u8) -> PhaseResult {
    let mut basis = ext.initial_basis.clone();
    let mut x_b = ext.b.clone();
    let mut basis_mgr = match LuBasis::new(&ext.a, &basis, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => return PhaseResult { outcome: SimplexOutcome::SingularBasis, basis, x_b },
    };
    basis_mgr.ftran_dense(&mut x_b);

    let mut y: Vec<f64> = basis.iter().map(|&b| ext.c[b]).collect();
    basis_mgr.btran_dense(&mut y);

    let mut c_perturbed = ext.c.clone();
    let n_ext = ext.n_ext;
    let mut is_basic = vec![false; n_ext];
    for &b in &basis { is_basic[b] = true; }
    // graded 摂動: 列 j に DUAL_FEAS_BASE + j * DUAL_FEAS_STEP の摂動を最低保証。
    // 列ごとに異なる shift で degeneracy を破壊し、Bland 規則の cycling を防ぐ。
    // BASE は十分大きく取り Phase I 進捗速度を確保、STEP は tie-break 用の微少差。
    const DUAL_FEAS_BASE: f64 = 1.0;
    const DUAL_FEAS_STEP: f64 = 1e-7;
    for j in 0..n_ext {
        if is_basic[j] { continue; }
        let cs = ext.a.col_ptr[j];
        let ce = ext.a.col_ptr[j + 1];
        let aty: f64 = (cs..ce).map(|k| ext.a.values[k] * y[ext.a.row_ind[k]]).sum();
        let c_bar = c_perturbed[j] - aty;
        let target = DUAL_FEAS_BASE + (j as f64) * DUAL_FEAS_STEP;
        if c_bar < target {
            c_perturbed[j] += target - c_bar;
        }
    }

    let outcome = dual_iter_blp(&ext.a, &c_perturbed, &ext.lb, &ext.ub, &mut basis, &mut x_b, basis_mgr, &is_basic, ext.n_ext, ext.m, options);
    PhaseResult { outcome, basis, x_b }
}

/// Phase II: 摂動なしの cost で dual simplex を再実行
fn run_dual_pure(
    ext: &Extended,
    options: &SolverOptions,
    init_basis: Vec<usize>,
    init_x_b: Vec<f64>,
) -> PhaseResult {
    let mut basis = init_basis;
    let mut x_b = init_x_b;
    let basis_mgr = match LuBasis::new(&ext.a, &basis, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => return PhaseResult { outcome: SimplexOutcome::SingularBasis, basis, x_b },
    };

    let mut c2 = ext.c.clone();
    for j in ext.n_standard..ext.n_ext { c2[j] = 0.0; }

    let n_ext = ext.n_ext;
    let mut is_basic = vec![false; n_ext];
    for &b in &basis { is_basic[b] = true; }

    let outcome = dual_iter_blp(&ext.a, &c2, &ext.lb, &ext.ub, &mut basis, &mut x_b, basis_mgr, &is_basic, ext.n_ext, ext.m, options);
    PhaseResult { outcome, basis, x_b }
}

// =====================================================================
// Pure dual simplex iteration (textbook)
// =====================================================================

/// 教科書 dual simplex iteration (BLP 対応):
///   1. Leaving: x_B[p] < lb_basic[p] (下違反) OR x_B[p] > ub_basic[p] (上違反)
///   2. BTRAN: rho = e_p^T B^{-1}
///   3. trow[j] = rho^T A_j (alpha_p,j)
///   4. Ratio test:
///      - 下違反: alpha < -tol を候補、min c_bar / |alpha|
///      - 上違反: alpha > +tol を候補、min c_bar / alpha
///   5. FTRAN, pivot, x_b 更新、basis 更新、c_bar 再計算
///
/// `ext_lb`/`ext_ub` を渡すと FX 人工変数 (lb=ub=0) の UB 違反検出が有効化される。
fn dual_iter_blp(
    a: &CscMatrix,
    c: &[f64],
    ext_lb: &[f64],
    ext_ub: &[f64],
    basis: &mut Vec<usize>,
    x_b: &mut Vec<f64>,
    mut basis_mgr: LuBasis,
    is_basic_init: &[bool],
    n: usize,
    m: usize,
    options: &SolverOptions,
) -> SimplexOutcome {
    let mut is_basic = is_basic_init.to_vec();

    // c_bar 初期計算
    let mut y: Vec<f64> = basis.iter().map(|&b| c[b]).collect();
    basis_mgr.btran_dense(&mut y);
    let mut c_bar = vec![0.0_f64; n];
    for j in 0..n {
        if is_basic[j] { continue; }
        let cs = a.col_ptr[j]; let ce = a.col_ptr[j + 1];
        let aty: f64 = (cs..ce).map(|k| a.values[k] * y[a.row_ind[k]]).sum();
        c_bar[j] = c[j] - aty;
    }

    // Devex (approximate Dual Steepest Edge) 重み γ_i ≈ ||B^{-T} e_i||^2。
    // 初期 basis ≒ I のため γ_i = 1 から開始。pivot 毎に Harris-Devex 更新で
    // 真の DSE に近似的に追随、degenerate cycling を実用速度で回避。
    let mut gamma = vec![1.0_f64; m];

    const FEAS_TOL: f64 = 1e-7;
    // pivot 候補の最小絶対値。これを下回ると eta 蓄積で数値破綻するため除外。
    const PIVOT_NEG_TOL: f64 = 1e-6;
    let max_iter = 1_000_000usize;
    let deadline = options.deadline;

    let dbg_iter = std::env::var("DUAL_ONLY_TRACE").ok().as_deref() == Some("1");
    let mut iter_count = 0usize;
    for _iter in 0..max_iter {
        iter_count = _iter;
        if dbg_iter && _iter % 200 == 0 && _iter > 0 {
            let xb_min = x_b.iter().cloned().fold(f64::INFINITY, f64::min);
            let xb_max = x_b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let n_basic_art = basis.iter().filter(|&&b| ext_ub[b] == 0.0 && ext_lb[b] == 0.0).count();
            let gmax = gamma.iter().cloned().fold(0.0_f64, f64::max);
            eprintln!("[dual_iter] iter {} xb=[{:.3e},{:.3e}] n_art_basic={} gmax={:.2e}",
                _iter, xb_min, xb_max, n_basic_art, gmax);
        }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                if dbg_iter {
                    eprintln!("[dual_iter] Timeout after {} iters", iter_count);
                }
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
        }

        // 1. Leaving row: Dual Steepest Edge (Devex 近似)。
        //    max_i violation_i^2 / γ_i を最大化する i を選ぶ。
        //    最も「下降幅が大きい」 leaving を選ぶことで degenerate cycling を実用回避。
        let mut leaving_row: Option<usize> = None;
        let mut leaving_direction = 0i8;
        let mut best_score = 0.0_f64;
        for i in 0..m {
            let j = basis[i];
            let lb_j = ext_lb[j];
            let ub_j = ext_ub[j];
            let lo_viol = if lb_j.is_finite() { (lb_j - x_b[i]).max(0.0) } else { 0.0 };
            let hi_viol = if ub_j.is_finite() { (x_b[i] - ub_j).max(0.0) } else { 0.0 };
            let (viol, dir) = if lo_viol >= hi_viol { (lo_viol, 1) } else { (hi_viol, -1) };
            if viol <= FEAS_TOL { continue; }
            let g = gamma[i].max(1e-12);
            let score = viol * viol / g;
            if score > best_score {
                best_score = score;
                leaving_row = Some(i);
                leaving_direction = dir;
            }
        }
        let p = match leaving_row {
            None => {
                if dbg_iter {
                    let mut max_viol_art = 0.0_f64;
                    let mut max_xb_art = 0.0_f64;
                    let mut sample_row = 0_usize;
                    let mut sample_xb = 0.0_f64;
                    let mut sample_ub = 0.0_f64;
                    let mut sample_gamma = 0.0_f64;
                    for i in 0..m {
                        if basis[i] >= 138 {
                            if x_b[i].abs() > sample_xb.abs() {
                                sample_row = i;
                                sample_xb = x_b[i];
                                sample_ub = ext_ub[basis[i]];
                                sample_gamma = gamma[i];
                            }
                            max_xb_art = max_xb_art.max(x_b[i]);
                            let v = (x_b[i] - ext_ub[basis[i]]).max(0.0);
                            max_viol_art = max_viol_art.max(v);
                        }
                    }
                    eprintln!("[dual_iter] Optimal {} iters, max_xb_art={:.3e} max_viol={:.3e}",
                        iter_count, max_xb_art, max_viol_art);
                    eprintln!("[dual_iter] worst sample: row={} x_b={:.3e} ub={:.3e} gamma={:.3e}",
                        sample_row, sample_xb, sample_ub, sample_gamma);
                }
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Optimal(obj, y);
            }
            Some(p) => p,
        };

        // 2. BTRAN: rho = e_p^T B^{-1}
        let mut rho = vec![0.0_f64; m];
        rho[p] = 1.0;
        basis_mgr.btran_dense(&mut rho);

        // 3+4. Ratio test
        // direction = +1 (LB違反): alpha < -tol を候補, ratio = c_bar / (-alpha)
        // direction = -1 (UB違反): alpha > +tol を候補, ratio = c_bar / alpha
        let mut entering_col: Option<usize> = None;
        let mut min_ratio = f64::INFINITY;
        let mut pivot_element = 0.0_f64;
        // min-ratio + Bland 規則 tie-break (textbook standard)
        for j in 0..n {
            if is_basic[j] { continue; }
            let cs = a.col_ptr[j]; let ce = a.col_ptr[j + 1];
            let alpha: f64 = (cs..ce).map(|k| rho[a.row_ind[k]] * a.values[k]).sum();
            let (valid, denom) = if leaving_direction > 0 {
                (alpha < -PIVOT_NEG_TOL, -alpha)
            } else {
                (alpha > PIVOT_NEG_TOL, alpha)
            };
            if !valid { continue; }
            let ratio = c_bar[j] / denom;
            if ratio < min_ratio - PIVOT_TOL {
                min_ratio = ratio;
                entering_col = Some(j);
                pivot_element = alpha;
            } else if (ratio - min_ratio).abs() <= PIVOT_TOL {
                if let Some(prev) = entering_col {
                    if j < prev {
                        entering_col = Some(j);
                        pivot_element = alpha;
                    }
                }
            }
        }
        let q = match entering_col {
            None => {
                if dbg_iter {
                    eprintln!("[dual_iter] Unbounded after {} iters, leaving p={} x_b[p]={:.3e}", iter_count, p, x_b[p]);
                }
                return SimplexOutcome::Unbounded;
            },
            Some(q) => q,
        };

        // 5. FTRAN: alpha_col = B^{-1} A_q
        let (q_rows, q_vals) = a.get_column(q).unwrap();
        let mut alpha_sv = SparseVec {
            indices: q_rows.to_vec(),
            values: q_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut alpha_sv);
        let mut alpha_dense = vec![0.0_f64; m];
        alpha_sv.to_dense_into(&mut alpha_dense);

        // step: 入基変数 q の新基底値
        //   target = lb_basic[p] (下違反) or ub_basic[p] (上違反)
        //   step = (x_B[p] - target) / pivot_element
        // standard form では lb=0、artificial の ub=0 のみ FX。
        let target = if leaving_direction > 0 { ext_lb[basis[p]] } else { ext_ub[basis[p]] };
        let step = (x_b[p] - target) / pivot_element;
        if dbg_iter && iter_count < 5 {
            eprintln!("[dual_iter] iter {} leaving p={} dir={} x_b[p]={:.3e} target={:.3e} pivot={:.3e} step={:.3e} entering q={}",
                iter_count, p, leaving_direction, x_b[p], target, pivot_element, step, q);
        }
        for i in 0..m {
            x_b[i] -= alpha_dense[i] * step;
        }
        x_b[p] = step; // 入基 q の新値

        let leaving_col = basis[p];
        is_basic[leaving_col] = false;
        is_basic[q] = true;
        basis[p] = q;
        basis_mgr.update(q, p, &alpha_sv);

        // Devex 重み更新 (Harris 1973) with overflow cap:
        //   γ_p_new = max(γ_p_old / pivot^2, max_{i!=p} (α_iq/pivot)^2 * γ_p_old)
        //   γ_i_new = max(γ_i_old, (α_iq/pivot)^2 * γ_p_old)  for i != p
        // GAMMA_MAX で発散を防ぐ (発散すると score=0 で leaving 候補から外れるバグ)。
        const GAMMA_MAX: f64 = 1e8;
        let gp_old = gamma[p].min(GAMMA_MAX);
        let inv_pivot_sq = 1.0 / (pivot_element * pivot_element);
        let mut max_w = (gp_old * inv_pivot_sq).min(GAMMA_MAX);
        for i in 0..m {
            if i == p { continue; }
            let r = alpha_dense[i] / pivot_element;
            let new_w = (r * r * gp_old).max(gamma[i]).min(GAMMA_MAX);
            gamma[i] = new_w;
            if new_w > max_w { max_w = new_w; }
        }
        gamma[p] = max_w.min(GAMMA_MAX);

        // 周期的リセット: 数値誤差累積で gamma が真値から大きく離れる前にリセット。
        // 1000 iter ごとに 1.0 へ戻す (reference Devex)。
        if iter_count > 0 && iter_count % 1000 == 0 {
            for g in gamma.iter_mut() { *g = 1.0; }
        }

        // c_bar 再計算 (簡略: 全再計算)
        for i in 0..m { y[i] = c[basis[i]]; }
        basis_mgr.btran_dense(&mut y);
        for j in 0..n {
            if is_basic[j] {
                c_bar[j] = 0.0;
                continue;
            }
            let cs = a.col_ptr[j]; let ce = a.col_ptr[j + 1];
            let aty: f64 = (cs..ce).map(|k| a.values[k] * y[a.row_ind[k]]).sum();
            c_bar[j] = c[j] - aty;
        }
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    SimplexOutcome::Timeout(obj)
}

// =====================================================================
// 後処理
// =====================================================================

fn artificial_max(ext: &Extended, basis: &[usize], x_b: &[f64]) -> f64 {
    let mut max_v = 0.0_f64;
    for (i, &b) in basis.iter().enumerate() {
        if b >= ext.n_standard {
            max_v = max_v.max(x_b[i].abs());
        }
    }
    max_v
}

fn result_optimal(
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    y: &[f64],
    ext: &Extended,
) -> SolverResult {
    let col_scale = vec![1.0_f64; ext.n_standard];
    let solution = extract_solution(sf, basis, x_b, &col_scale);
    let obj: f64 = problem.c.iter().zip(solution.iter()).map(|(&c, &x)| c * x).sum::<f64>()
        + sf_obj_offset(sf);

    // 元 LP の dual 復元: A^T y_orig = c_orig - reduced_costs (元空間)
    // ここでは拡張行列 ext.a の y を渡す。bench 側で行数 = problem.num_constraints
    // を期待するため、必要数だけ取り出す。
    let m_orig = problem.num_constraints;
    let dual_solution: Vec<f64> = if y.len() >= m_orig {
        y[..m_orig].to_vec()
    } else { vec![] };

    // reduced_costs: 元変数 (j < problem.num_vars) のみ。c - A^T y を元 A から再計算
    let n_orig = problem.num_vars;
    let mut reduced_costs = vec![0.0_f64; n_orig];
    for j in 0..n_orig {
        let cs = problem.a.col_ptr[j];
        let ce = problem.a.col_ptr[j + 1];
        let aty: f64 = (cs..ce)
            .map(|k| {
                let r = problem.a.row_ind[k];
                if r < dual_solution.len() { dual_solution[r] * problem.a.values[k] } else { 0.0 }
            }).sum();
        reduced_costs[j] = problem.c[j] - aty;
    }

    // slack: b_i - A_i·x (制約 type 依存だが、bench は raw 値を期待することが多い)
    let mut slack = vec![0.0_f64; m_orig];
    for i in 0..m_orig {
        slack[i] = problem.b[i];
    }
    for j in 0..n_orig {
        let cs = problem.a.col_ptr[j];
        let ce = problem.a.col_ptr[j + 1];
        for k in cs..ce {
            let r = problem.a.row_ind[k];
            if r < m_orig {
                slack[r] -= problem.a.values[k] * solution[j];
            }
        }
    }

    SolverResult {
        status: SolveStatus::Optimal,
        objective: obj,
        solution,
        dual_solution,
        reduced_costs,
        slack,
        ..Default::default()
    }
}

fn result_infeasible(_sf: &StandardForm) -> SolverResult {
    SolverResult { status: SolveStatus::Infeasible, ..Default::default() }
}

fn result_timeout(
    sf: &StandardForm,
    problem: &LpProblem,
    basis: &[usize],
    x_b: &[f64],
    ext: &Extended,
) -> SolverResult {
    let col_scale = vec![1.0_f64; ext.n_standard];
    let solution = extract_solution(sf, basis, x_b, &col_scale);
    let obj: f64 = problem.c.iter().zip(solution.iter()).map(|(&c, &x)| c * x).sum::<f64>()
        + sf_obj_offset(sf);
    let _ = PIVOT_TOL;
    SolverResult {
        status: SolveStatus::Timeout,
        objective: obj,
        solution,
        ..Default::default()
    }
}

// =====================================================================
// StandardForm の private フィールドアクセサ
// (super:: 経由でしか触れないので関数経由で取得)
// =====================================================================

fn sf_m(sf: &StandardForm) -> usize { sf_field_m(sf) }
fn sf_n_total(sf: &StandardForm) -> usize { sf_field_n_total(sf) }
fn sf_a(sf: &StandardForm) -> &CscMatrix { sf_field_a(sf) }
fn sf_b(sf: &StandardForm) -> &[f64] { sf_field_b(sf) }
fn sf_c(sf: &StandardForm) -> &[f64] { sf_field_c(sf) }
fn sf_initial_basis(sf: &StandardForm) -> &[usize] { sf_field_initial_basis(sf) }
fn sf_needs_artificial(sf: &StandardForm, i: usize) -> bool { sf_field_needs_artificial(sf, i) }
fn sf_obj_offset(sf: &StandardForm) -> f64 { sf_field_obj_offset(sf) }

// 以下は mod.rs 側で StandardForm に accessor を追加する必要がある。
// 一時的に extern 風 declaration (super:: 経由)。
use super::sf_field_m;
use super::sf_field_n_total;
use super::sf_field_a;
use super::sf_field_b;
use super::sf_field_c;
use super::sf_field_initial_basis;
use super::sf_field_needs_artificial;
use super::sf_field_obj_offset;

// =====================================================================
// 補助 trait
// =====================================================================

trait AlsoExt {
    fn also<F: FnOnce(&Self)>(self, f: F) -> Self where Self: Sized;
}
impl<T> AlsoExt for T {
    fn also<F: FnOnce(&Self)>(self, f: F) -> Self where Self: Sized {
        f(&self);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ConstraintType;
    use crate::sparse::CscMatrix;

    fn make_lp(c: Vec<f64>, a_triplets: Vec<(usize, usize, f64)>, b: Vec<f64>,
               cts: Vec<ConstraintType>, bounds: Vec<(f64, f64)>, m: usize, n: usize) -> LpProblem {
        let rows: Vec<usize> = a_triplets.iter().map(|&(r,_,_)| r).collect();
        let cols: Vec<usize> = a_triplets.iter().map(|&(_,c,_)| c).collect();
        let vals: Vec<f64> = a_triplets.iter().map(|&(_,_,v)| v).collect();
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        LpProblem {
            c,
            a,
            b,
            num_vars: n,
            num_constraints: m,
            constraint_types: cts,
            bounds,
            name: None,
        }
    }

    #[test]
    fn dual_only_tiny_le() {
        // min x1 + 2*x2 s.t. x1+x2 <= 3, x1,x2 >= 0
        // 解: x1=0, x2=0, obj=0 (最小化なので原点で最小)
        let lp = make_lp(
            vec![1.0, 2.0],
            vec![(0,0,1.0), (0,1,1.0)],
            vec![3.0],
            vec![ConstraintType::Le],
            vec![(0.0, f64::INFINITY); 2],
            1, 2,
        );
        let opts = SolverOptions::default();
        let r = solve(&lp, &opts);
        assert_eq!(r.status, SolveStatus::Optimal, "tiny Le: status");
        assert!((r.objective - 0.0).abs() < 1e-6, "tiny Le: obj={}", r.objective);
    }

    #[test]
    fn dual_only_tiny_ge() {
        // min x1 + 2*x2 s.t. x1+x2 >= 1, x1,x2 >= 0
        // 解: x1=1, x2=0, obj=1
        let lp = make_lp(
            vec![1.0, 2.0],
            vec![(0,0,1.0), (0,1,1.0)],
            vec![1.0],
            vec![ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2],
            1, 2,
        );
        let opts = SolverOptions::default();
        let r = solve(&lp, &opts);
        assert_eq!(r.status, SolveStatus::Optimal, "tiny Ge: status");
        assert!((r.objective - 1.0).abs() < 1e-6, "tiny Ge: obj={}", r.objective);
    }

    #[test]
    fn dual_only_tiny_eq() {
        // min x1 + 2*x2 s.t. x1+x2 = 2, x1,x2 >= 0
        // 解: x1=2, x2=0, obj=2
        let lp = make_lp(
            vec![1.0, 2.0],
            vec![(0,0,1.0), (0,1,1.0)],
            vec![2.0],
            vec![ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 2],
            1, 2,
        );
        let opts = SolverOptions::default();
        let r = solve(&lp, &opts);
        assert_eq!(r.status, SolveStatus::Optimal, "tiny Eq: status");
        assert!((r.objective - 2.0).abs() < 1e-6, "tiny Eq: obj={}", r.objective);
    }

    #[test]
    fn dual_only_infeasible() {
        // min x1 s.t. x1 = 1, x1 = 2 → infeasible
        let lp = make_lp(
            vec![1.0],
            vec![(0,0,1.0), (1,0,1.0)],
            vec![1.0, 2.0],
            vec![ConstraintType::Eq, ConstraintType::Eq],
            vec![(0.0, f64::INFINITY); 1],
            2, 1,
        );
        let opts = SolverOptions::default();
        let r = solve(&lp, &opts);
        assert_eq!(r.status, SolveStatus::Infeasible, "infeas detection");
    }
}
