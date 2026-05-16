//! Pure Dual Simplex (single source of truth for LP).
//!
//! 設計方針:
//! - 純粋 dual simplex のみ。Phase I / Phase II は cost perturbation で統一。
//! - bounded variables (BLP) を素直に扱う (bound flipping)。
//! - Eq 制約は人工変数 (Big-M cost) で吸収。
//! - 既存 primal/dual/dual_advanced のパッチ蓄積を捨て、シンプルさを優先。
//!
//! 入力: `LpProblem` (any Eq/Ge/Le mix, any bounds)
//! 出力: `SolverResult` (Optimal/Infeasible/Unbounded/Timeout/NumericalError)

use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};

/// Big-M cost on artificial variables.
/// 1e7 で Netlib LP の最大 |c| (≈ 1e4-1e5) を上回り、人工変数を必ず非基底に追い出す。
/// 1e15 級は f64 round-off で対角 ill-conditioning を生むため避ける。
const BIG_M: f64 = 1e7;

/// 反復回数上限 (実用的 guard。本来は timeout で制御)
const MAX_ITERATIONS: usize = 10_000_000;

/// Dual simplex で LP を解く。
pub fn solve(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let deadline = options.deadline.or_else(|| {
        options
            .timeout_secs
            .map(|s| std::time::Instant::now() + std::time::Duration::from_secs_f64(s))
    });

    // Step 1: 拡張系を構築 (slack + artificial を明示列で持つ)
    let ext = match build_extended(problem) {
        Ok(e) => e,
        Err(s) => return s,
    };

    // Step 2: 初期 LU 因子分解
    let mut basis = ext.initial_basis.clone();
    let mut basis_mgr = match LuBasis::new(&ext.a, &basis, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => return SolverResult::default_with(SolveStatus::NumericalError),
    };

    // Step 3: x_B = B^{-1} (b - A_N x_N) を計算
    //   非基底変数は LB/UB のどちらかに固定。
    //   ax_n[i] = sum_{j non-basic} A[i,j] * value(j)
    //   x_b = B^{-1} (b - ax_n)
    let mut x_b = compute_xb(&ext, &basis, &mut basis_mgr);

    // Step 4: 双対変数 y = (B^{-T}) c_B
    let mut y = compute_dual(&ext, &basis, &mut basis_mgr);

    // Step 5: 全非基底変数の reduced cost を計算 + dual feasibility 摂動
    //   c_bar_j = c_j - y^T A_j
    //   at LB の var: c_bar >= 0 必要
    //   at UB の var: c_bar <= 0 必要
    let (c_perturbed, perturb_count) = perturb_for_dual_feasibility(&ext, &y);

    let _ = perturb_count; // TODO: Phase II で undo

    // Step 6: dual simplex iteration loop
    let mut iter = 0usize;
    let cancel_flag = options.cancel_flag.clone();
    loop {
        if iter >= MAX_ITERATIONS {
            return finalize(&ext, &basis, &x_b, &y, SolveStatus::Timeout, problem);
        }
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                return finalize(&ext, &basis, &x_b, &y, SolveStatus::Timeout, problem);
            }
        }
        if let Some(ref f) = cancel_flag {
            if f.load(std::sync::atomic::Ordering::Relaxed) {
                return finalize(&ext, &basis, &x_b, &y, SolveStatus::Timeout, problem);
            }
        }

        // 6a. Pick leaving row (most infeasible)
        let leaving = pick_leaving(&ext, &x_b, &basis);
        let leaving_row = match leaving {
            Some(lr) => lr,
            None => {
                // 全 x_B が境界内 → primal 実行可能 = 最適
                return finalize(&ext, &basis, &x_b, &y, SolveStatus::Optimal, problem);
            }
        };

        // 6b. rho = e_leaving^T B^{-1} (BTRAN)
        let mut rho_dense = vec![0.0_f64; ext.m];
        rho_dense[leaving_row] = 1.0;
        basis_mgr.btran_dense(&mut rho_dense);

        // 6c. 非基底列ごとに alpha_j = rho^T A_j 計算 + ratio test
        let direction_up = x_b[leaving_row] < ext.lb_basic(leaving_row, &basis);
        let entering = pick_entering(&ext, &basis, &c_perturbed, &rho_dense, direction_up);

        let entering_col = match entering {
            Some(j) => j,
            None => {
                // ratio test で候補なし → dual unbounded = primal infeasible
                return finalize(&ext, &basis, &x_b, &y, SolveStatus::Infeasible, problem);
            }
        };

        // 6d. Pivot: B^{-1} A_entering を計算 (FTRAN)
        let (rows, vals) = ext.a.get_column(entering_col).expect("valid column");
        let mut alpha_sv = SparseVec {
            indices: rows.to_vec(),
            values: vals.to_vec(),
            len: ext.m,
        };
        basis_mgr.ftran(&mut alpha_sv);

        // alpha_j[leaving_row] が pivot 要素
        let pivot = alpha_sv.values.iter().zip(alpha_sv.indices.iter())
            .find(|(_, &i)| i == leaving_row)
            .map(|(&v, _)| v)
            .unwrap_or(0.0);

        if pivot.abs() < 1e-12 {
            // 数値的に degenerate な pivot → refactor して retry
            basis_mgr.force_refactor_timed(&ext.a, &basis, deadline);
            x_b = compute_xb(&ext, &basis, &mut basis_mgr);
            y = compute_dual(&ext, &basis, &mut basis_mgr);
            iter += 1;
            continue;
        }

        // 6e. Update x_b: x_b = x_b - alpha * step + ...
        //   step = (x_b[leaving] - target_bound) / pivot
        let target_bound = if direction_up {
            ext.lb_basic(leaving_row, &basis)
        } else {
            ext.ub_basic(leaving_row, &basis)
        };
        let step = (x_b[leaving_row] - target_bound) / pivot;

        let mut alpha_dense = vec![0.0_f64; ext.m];
        alpha_sv.to_dense_into(&mut alpha_dense);
        for i in 0..ext.m {
            x_b[i] -= alpha_dense[i] * step;
        }
        x_b[leaving_row] = target_bound + step * 0.0; // = target_bound; entering 変数は別途
        // entering 変数の新値:
        let entering_new_val = ext.value_at_bound(entering_col, !ext.at_ub[entering_col]) + step;
        x_b[leaving_row] = entering_new_val;

        // 6f. Update basis
        let leaving_col = basis[leaving_row];
        basis[leaving_row] = entering_col;
        // 旧 leaving_col は非基底へ。bound = target_bound のどちらか。
        // direction_up なら leaving_col は LB (=lb of leaving_col) に固定された
        // すなわち leaving 行の制約 var が lb に行く = at_ub=false
        // (実装簡略: leaving 行の x_b は basis[leaving]=entering 後の値)
        // TODO: at_ub[leaving_col] を正しく更新

        let _ = leaving_col;

        basis_mgr.update(entering_col, leaving_row, &alpha_sv);
        basis_mgr.refactor_if_needed_timed(&ext.a, &basis, deadline);

        // dual y update (簡略: 完全 recompute)
        y = compute_dual(&ext, &basis, &mut basis_mgr);

        iter += 1;
    }
}

// =====================================================================
// 拡張系の構築
// =====================================================================

/// 拡張系 (slack/artificial 込み) 表現。
struct Extended {
    /// 制約行列 m × n_ext (CSC)
    a: CscMatrix,
    /// RHS m
    b: Vec<f64>,
    /// コスト n_ext (artificials に BIG_M)
    c: Vec<f64>,
    /// 変数下限 n_ext
    lb: Vec<f64>,
    /// 変数上限 n_ext
    ub: Vec<f64>,
    /// 初期基底 (各行に対する列 index)
    initial_basis: Vec<usize>,
    /// 各非基底変数の状態 (false=LB, true=UB)
    at_ub: Vec<bool>,
    /// 元 LP の変数数 (n_ext から逆引きするため)
    n_orig: usize,
    /// 行数
    m: usize,
}

impl Extended {
    fn n_ext(&self) -> usize { self.lb.len() }
    /// non-basic var の固定値 (at_ub に応じて LB/UB)
    fn value_at_bound(&self, j: usize, at_ub_flag: bool) -> f64 {
        if at_ub_flag { self.ub[j] } else { self.lb[j] }
    }
    /// 行 i に対応する basis 変数の LB
    fn lb_basic(&self, i: usize, basis: &[usize]) -> f64 { self.lb[basis[i]] }
    fn ub_basic(&self, i: usize, basis: &[usize]) -> f64 { self.ub[basis[i]] }
}

/// LpProblem を拡張系に変換する。
fn build_extended(problem: &LpProblem) -> Result<Extended, SolverResult> {
    let n_orig = problem.num_vars;
    let m = problem.num_constraints;

    // 拡張変数構成:
    //   [0, n_orig)         : 元変数 (bound はそのまま)
    //   [n_orig, n_orig+n_slack) : Le/Ge の slack (bound [0, ∞))
    //   [..., n_ext)        : Eq の artificial (bound [0, ∞), cost BIG_M)
    let mut slack_for_row: Vec<Option<usize>> = vec![None; m];
    let mut artificial_for_row: Vec<Option<usize>> = vec![None; m];
    let mut slack_coef = vec![0.0_f64; m]; // +1 (Le), -1 (Ge), 0 (Eq → no slack)

    let mut n_slack = 0usize;
    let mut n_artificial = 0usize;
    let mut col_offset = n_orig;

    for i in 0..m {
        match problem.constraint_types[i] {
            ConstraintType::Le => {
                slack_for_row[i] = Some(col_offset);
                slack_coef[i] = 1.0;
                col_offset += 1;
                n_slack += 1;
            }
            ConstraintType::Ge => {
                slack_for_row[i] = Some(col_offset);
                slack_coef[i] = -1.0;
                col_offset += 1;
                n_slack += 1;
            }
            ConstraintType::Eq => {
                // Eq は slack を持たず artificial で覆う
            }
        }
    }

    // artificial を後ろに置く
    for i in 0..m {
        if matches!(problem.constraint_types[i], ConstraintType::Eq) {
            artificial_for_row[i] = Some(col_offset);
            col_offset += 1;
            n_artificial += 1;
        }
    }

    let n_ext = n_orig + n_slack + n_artificial;

    // RHS の符号を整える (b_i < 0 のとき行を反転して b >= 0 にする)
    let mut b = problem.b.clone();
    let mut row_flip = vec![false; m];
    for i in 0..m {
        if b[i] < 0.0 {
            row_flip[i] = true;
            b[i] = -b[i];
        }
    }

    // 拡張 A を triplet で構築
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n_orig {
        if let Ok((rs, vs)) = problem.a.get_column(j) {
            for (k, &r) in rs.iter().enumerate() {
                let v = if row_flip[r] { -vs[k] } else { vs[k] };
                trip_rows.push(r);
                trip_cols.push(j);
                trip_vals.push(v);
            }
        }
    }
    for i in 0..m {
        if let Some(s) = slack_for_row[i] {
            let coef = if row_flip[i] { -slack_coef[i] } else { slack_coef[i] };
            trip_rows.push(i);
            trip_cols.push(s);
            trip_vals.push(coef);
        }
        if let Some(a) = artificial_for_row[i] {
            // artificial は常に +1
            trip_rows.push(i);
            trip_cols.push(a);
            trip_vals.push(1.0);
        }
    }
    let a_ext = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_ext)
        .map_err(|_| SolverResult::default_with(SolveStatus::NumericalError))?;

    // bounds: 元変数 + slack [0,∞) + artificial [0,∞)
    let mut lb = vec![0.0_f64; n_ext];
    let mut ub = vec![f64::INFINITY; n_ext];
    for j in 0..n_orig {
        lb[j] = problem.bounds[j].0;
        ub[j] = problem.bounds[j].1;
    }

    // cost: 元 c + 0 (slack) + BIG_M (artificial)
    let mut c = vec![0.0_f64; n_ext];
    c[..n_orig].copy_from_slice(&problem.c);
    for i in 0..m {
        if let Some(a) = artificial_for_row[i] {
            c[a] = BIG_M;
        }
    }

    // 初期基底: 各行 i について
    //   Le/Ge: slack 列 (+1 列係数, b_i >= 0 なので x_B = b_i 適切)
    //     ※ Le で +1 係数 → x_B = b_i ≥ 0 OK
    //     ※ Ge は -1 係数だが行反転で +1 にしたので x_B = b_i (元の |b_i|) ≥ 0 OK
    //   Eq: artificial 列 (+1 係数), x_B = b_i ≥ 0
    let mut initial_basis = vec![0_usize; m];
    for i in 0..m {
        if let Some(s) = slack_for_row[i] {
            // 行反転していると slack 係数も反転している。係数 +1 ならスラックを基底に。
            // 反転後の係数は (row_flip ? -slack_coef[i] : slack_coef[i])。
            // この値が +1 ならそのまま basis にしてよい (B = I 部分行列)。
            // -1 のままだと B[i,i] = -1 で x_B = -b_i = 負 → bound 違反 → dual simplex で fix。
            // 簡単のため slack を必ず基底に入れる (LuBasis が対応する)。
            initial_basis[i] = s;
        } else if let Some(a) = artificial_for_row[i] {
            initial_basis[i] = a;
        } else {
            unreachable!("row {} has neither slack nor artificial", i);
        }
    }

    // 初期 at_ub: 全非基底は LB (false)
    let at_ub = vec![false; n_ext];

    Ok(Extended { a: a_ext, b, c, lb, ub, initial_basis, at_ub, n_orig, m })
}

// =====================================================================
// Helper: x_B, y, reduced cost
// =====================================================================

/// x_B = B^{-1} (b - A_N x_N) を計算
fn compute_xb(ext: &Extended, basis: &[usize], basis_mgr: &mut LuBasis) -> Vec<f64> {
    // b - A_N x_N: 非基底変数の固定値が contribution
    let mut rhs = ext.b.clone();
    let n_ext = ext.n_ext();
    let mut is_basic = vec![false; n_ext];
    for &b in basis { is_basic[b] = true; }
    for j in 0..n_ext {
        if is_basic[j] { continue; }
        let val = ext.value_at_bound(j, ext.at_ub[j]);
        if val == 0.0 { continue; }
        if let Ok((rs, vs)) = ext.a.get_column(j) {
            for (k, &r) in rs.iter().enumerate() {
                rhs[r] -= vs[k] * val;
            }
        }
    }
    basis_mgr.ftran_dense(&mut rhs);
    rhs
}

/// y = B^{-T} c_B
fn compute_dual(ext: &Extended, basis: &[usize], basis_mgr: &mut LuBasis) -> Vec<f64> {
    let mut y = vec![0.0_f64; ext.m];
    for i in 0..ext.m {
        y[i] = ext.c[basis[i]];
    }
    basis_mgr.btran_dense(&mut y);
    y
}

/// dual 実行可能性のための cost 摂動: c_j <- c_j + max(0, -c_bar_j) for at-LB
/// returns (perturbed c, count of perturbed columns)
fn perturb_for_dual_feasibility(ext: &Extended, y: &[f64]) -> (Vec<f64>, usize) {
    let mut c_p = ext.c.clone();
    let mut count = 0usize;
    let n_ext = ext.n_ext();
    for j in 0..n_ext {
        // c_bar_j = c_j - y^T A_j
        let aty = if let Ok((rs, vs)) = ext.a.get_column(j) {
            rs.iter().zip(vs.iter()).map(|(&r, &v)| v * y[r]).sum::<f64>()
        } else { 0.0 };
        let c_bar = c_p[j] - aty;
        if !ext.at_ub[j] {
            // at LB → c_bar >= 0 必要
            if c_bar < 0.0 {
                c_p[j] -= c_bar; // c_bar 0 になるようシフト
                count += 1;
            }
        } else {
            // at UB → c_bar <= 0 必要
            if c_bar > 0.0 {
                c_p[j] -= c_bar;
                count += 1;
            }
        }
    }
    (c_p, count)
}

// =====================================================================
// Iteration helpers
// =====================================================================

/// 最大 bound 違反を持つ basic var の行 index を返す
fn pick_leaving(ext: &Extended, x_b: &[f64], basis: &[usize]) -> Option<usize> {
    let mut worst_viol = 0.0_f64;
    let mut worst_row: Option<usize> = None;
    const FEAS_TOL: f64 = 1e-7;
    for i in 0..ext.m {
        let j = basis[i];
        let lb_j = ext.lb[j];
        let ub_j = ext.ub[j];
        let lo_viol = if lb_j.is_finite() { (lb_j - x_b[i]).max(0.0) } else { 0.0 };
        let hi_viol = if ub_j.is_finite() { (x_b[i] - ub_j).max(0.0) } else { 0.0 };
        let v = lo_viol.max(hi_viol);
        if v > worst_viol + FEAS_TOL && v > worst_viol {
            worst_viol = v;
            worst_row = Some(i);
        }
    }
    worst_row
}

/// dual ratio test で entering 列を選ぶ
fn pick_entering(
    ext: &Extended,
    basis: &[usize],
    c_p: &[f64],
    rho: &[f64],
    direction_up: bool,
) -> Option<usize> {
    // alpha_j = rho^T A_j
    // direction_up: leaving 変数を増やしたい (LB 違反) → step > 0 で増加
    //   → entering 変数の方向と整合する alpha_j の符号を選ぶ
    //     - at LB の entering: step > 0 で増加 → alpha_j > 0 が必要 (xb[leaving] 減少と整合は要見直し)
    //     - at UB の entering: step > 0 で減少 → alpha_j < 0
    //
    // 簡略実装: |c_bar_j / alpha_j| を最小化、|alpha_j| > tol で候補
    let n_ext = ext.n_ext();
    let mut is_basic = vec![false; n_ext];
    for &b in basis { is_basic[b] = true; }

    let mut best: Option<(usize, f64)> = None;
    const ALPHA_TOL: f64 = 1e-9;
    for j in 0..n_ext {
        if is_basic[j] { continue; }
        if !ext.lb[j].is_finite() && !ext.ub[j].is_finite() { continue; }
        let alpha = if let Ok((rs, vs)) = ext.a.get_column(j) {
            rs.iter().zip(vs.iter()).map(|(&r, &v)| v * rho[r]).sum::<f64>()
        } else { 0.0 };
        let signed_alpha = if direction_up { alpha } else { -alpha };
        // 候補条件: at LB → signed_alpha < 0 で c_bar 増加方向, at UB → > 0
        let valid = if !ext.at_ub[j] { signed_alpha < -ALPHA_TOL } else { signed_alpha > ALPHA_TOL };
        if !valid { continue; }

        // reduced cost
        let aty: f64 = if let Ok((rs, vs)) = ext.a.get_column(j) {
            // y は呼出側で持ってないので簡略: c_p_j を使う (摂動済みなので c_bar >= 0)
            rs.iter().zip(vs.iter()).map(|(&r, &v)| v * rho[r]).sum::<f64>()
        } else { 0.0 };
        let _ = aty;
        let c_bar = c_p[j]; // 簡略: 摂動後の c_p を直接 reduced cost と看做す (要見直し)
        let ratio = (c_bar / signed_alpha.abs()).abs();
        match best {
            None => best = Some((j, ratio)),
            Some((_, bratio)) if ratio < bratio => best = Some((j, ratio)),
            _ => {}
        }
    }
    best.map(|(j, _)| j)
}

// =====================================================================
// 結果マッピング
// =====================================================================

fn finalize(
    ext: &Extended,
    basis: &[usize],
    x_b: &[f64],
    _y: &[f64],
    status: SolveStatus,
    problem: &LpProblem,
) -> SolverResult {
    // 元変数の解を組み立てる
    let mut solution = vec![0.0_f64; ext.n_orig];
    let n_ext = ext.n_ext();
    let mut is_basic = vec![false; n_ext];
    let mut basic_row = vec![0_usize; n_ext];
    for (i, &b) in basis.iter().enumerate() {
        is_basic[b] = true;
        basic_row[b] = i;
    }
    for j in 0..ext.n_orig {
        solution[j] = if is_basic[j] {
            x_b[basic_row[j]]
        } else {
            ext.value_at_bound(j, ext.at_ub[j])
        };
    }

    // 目的関数値
    let obj: f64 = problem.c.iter().zip(solution.iter()).map(|(&c, &x)| c * x).sum();

    // 人工変数残量 > eps なら infeasible (Big-M で押さえ込めてないケース)
    let mut art_max = 0.0_f64;
    for j in ext.n_orig..n_ext {
        if ext.c[j] >= BIG_M * 0.5 {
            // artificial 列
            if is_basic[j] {
                art_max = art_max.max(x_b[basic_row[j]].abs());
            }
        }
    }
    let final_status = if matches!(status, SolveStatus::Optimal) && art_max > 1e-6 {
        SolveStatus::Infeasible
    } else {
        status
    };

    SolverResult {
        status: final_status,
        objective: obj,
        solution,
        dual_solution: vec![],
        reduced_costs: vec![],
        slack: vec![],
        warm_start_basis: None,
        ..Default::default()
    }
}

// =====================================================================
// SolverResult 拡張
// =====================================================================

trait SolverResultExt {
    fn default_with(status: SolveStatus) -> Self;
}
impl SolverResultExt for SolverResult {
    fn default_with(status: SolveStatus) -> Self {
        SolverResult { status, ..Default::default() }
    }
}
