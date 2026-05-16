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

    // Phase I: Big-M 込みコストで dual simplex
    let phase1_result = run_dual_with_perturbation(&ext, options, /*phase=*/1);

    match phase1_result.outcome {
        SimplexOutcome::Optimal(obj, _y) => {
            // 人工変数残量チェック
            let art_max = artificial_max(&ext, &phase1_result.basis, &phase1_result.x_b);
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

    Extended {
        a: a_ext,
        b: sf_b(sf).to_vec(),
        c,
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
    const DUAL_FEAS_MARGIN: f64 = 1e-3;
    for j in 0..n_ext {
        if is_basic[j] { continue; }
        let cs = ext.a.col_ptr[j];
        let ce = ext.a.col_ptr[j + 1];
        let aty: f64 = (cs..ce).map(|k| ext.a.values[k] * y[ext.a.row_ind[k]]).sum();
        let c_bar = c_perturbed[j] - aty;
        if c_bar < DUAL_FEAS_MARGIN {
            c_perturbed[j] += DUAL_FEAS_MARGIN - c_bar;
        }
    }

    let outcome = dual_iter(&ext.a, &c_perturbed, &mut basis, &mut x_b, basis_mgr, &is_basic, ext.n_ext, ext.m, options);
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

    let outcome = dual_iter(&ext.a, &c2, &mut basis, &mut x_b, basis_mgr, &is_basic, ext.n_ext, ext.m, options);
    PhaseResult { outcome, basis, x_b }
}

// =====================================================================
// Pure dual simplex iteration (textbook)
// =====================================================================

/// 教科書 dual simplex iteration:
///   1. Leaving: x_B[p] < -tol で最も infeasible な行
///   2. BTRAN: rho = e_p^T B^{-1}
///   3. trow[j] = rho^T A_j (alpha_p,j)
///   4. Ratio test: j non-basic with trow[j] < -tol を候補とし、min |c_bar[j] / trow[j]| を選ぶ
///   5. FTRAN, pivot, x_b 更新、basis 更新、c_bar 再計算
fn dual_iter(
    a: &CscMatrix,
    c: &[f64],
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

    const FEAS_TOL: f64 = 1e-7;
    const PIVOT_NEG_TOL: f64 = 1e-9;
    let max_iter = 1_000_000usize;
    let deadline = options.deadline;

    for _iter in 0..max_iter {
        if let Some(dl) = deadline {
            if std::time::Instant::now() >= dl {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Timeout(obj);
            }
        }

        // 1. Leaving row (most negative x_B)
        let mut leaving_row: Option<usize> = None;
        let mut worst = -FEAS_TOL;
        for i in 0..m {
            if x_b[i] < worst {
                worst = x_b[i];
                leaving_row = Some(i);
            }
        }
        let p = match leaving_row {
            None => {
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
        let mut entering_col: Option<usize> = None;
        let mut min_ratio = f64::INFINITY;
        let mut pivot_element = 0.0_f64;
        for j in 0..n {
            if is_basic[j] { continue; }
            let cs = a.col_ptr[j]; let ce = a.col_ptr[j + 1];
            let alpha: f64 = (cs..ce).map(|k| rho[a.row_ind[k]] * a.values[k]).sum();
            if alpha < -PIVOT_NEG_TOL {
                let ratio = c_bar[j] / (-alpha);
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
        }
        let q = match entering_col {
            None => return SimplexOutcome::Unbounded, // 双対非有界 = 主実行不可
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

        // step = x_B[p] / pivot_element (両方負 → step > 0)
        let step = x_b[p] / pivot_element;
        for i in 0..m {
            x_b[i] -= alpha_dense[i] * step;
        }
        x_b[p] = step; // 入基 q の新値

        let leaving_col = basis[p];
        is_basic[leaving_col] = false;
        is_basic[q] = true;
        basis[p] = q;
        basis_mgr.update(q, p, &alpha_sv);

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
    _y: &[f64],
    ext: &Extended,
) -> SolverResult {
    // 標準形 → 元変数空間に戻す
    // 拡張部分 (artificial) は元空間に影響しないので、最初の n_standard 列のみで extract_solution を呼ぶ
    // x_b は m 個、basis は m 個。basis[i] が n_standard 以上なら artificial が basis に残っている (≈0)
    // ⇒ そのまま渡しても extract_solution 側は元変数のみ参照するので問題なし
    let col_scale = vec![1.0_f64; ext.n_standard]; // Ruiz scaling なし
    let solution = extract_solution(sf, basis, x_b, &col_scale);
    let obj: f64 = problem.c.iter().zip(solution.iter()).map(|(&c, &x)| c * x).sum::<f64>()
        + sf_obj_offset(sf);
    SolverResult {
        status: SolveStatus::Optimal,
        objective: obj,
        solution,
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
