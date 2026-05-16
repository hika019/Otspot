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
use crate::sparse::CscMatrix;
use crate::tolerances::PIVOT_TOL;
use super::{StandardForm, SimplexOutcome, build_standard_form, extract_solution};
use super::dual::dual_simplex_core;

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
    use crate::basis::{BasisManager, LuBasis};
    let mut basis = ext.initial_basis.clone();

    // 初期 x_B = B^{-1} b を計算 (basis 構造は identity / -identity 混在)
    let mut basis_mgr = match LuBasis::new(&ext.a, &basis, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => return PhaseResult {
            outcome: SimplexOutcome::SingularBasis,
            basis,
            x_b: ext.b.clone(),
        },
    };
    let mut x_b = ext.b.clone();
    basis_mgr.ftran_dense(&mut x_b);

    // 初期 y = B^{-T} c_B
    let mut y_init: Vec<f64> = basis.iter().map(|&b| ext.c[b]).collect();
    basis_mgr.btran_dense(&mut y_init);

    // c_bar_j = c_j - y^T A_j を計算し、c_perturbed で dual 実行可能化
    let mut c_perturbed = ext.c.clone();
    let n_ext = ext.n_ext;
    let mut is_basic = vec![false; n_ext];
    for &b in &basis { is_basic[b] = true; }
    // DUAL_FEAS_MARGIN: 摂動でちょうど c_bar=0 にすると、後続 pivot で
    // ratio=0 となり進歩しなくなる。小さい正の余裕を残すことで dual simplex の
    // entering 候補を有効化する。
    const DUAL_FEAS_MARGIN: f64 = 1e-3;
    for j in 0..n_ext {
        if is_basic[j] { continue; }
        // a_j^T y を計算
        let aty: f64 = {
            let cs = ext.a.col_ptr[j];
            let ce = ext.a.col_ptr[j + 1];
            (cs..ce).map(|k| ext.a.values[k] * y_init[ext.a.row_ind[k]]).sum()
        };
        let c_bar = c_perturbed[j] - aty;
        if c_bar < DUAL_FEAS_MARGIN {
            c_perturbed[j] += DUAL_FEAS_MARGIN - c_bar; // c_bar = MARGIN に
        }
    }

    let outcome = dual_simplex_core(
        &ext.a, &mut x_b, &c_perturbed, &mut basis, ext.m, ext.n_standard, options,
    );
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

    // Phase II では Big-M を外し、人工変数の cost を 0 にする
    let mut c2 = ext.c.clone();
    for j in ext.n_standard..ext.n_ext {
        c2[j] = 0.0;
    }

    let outcome = dual_simplex_core(
        &ext.a, &mut x_b, &c2, &mut basis, ext.m, ext.n_standard, options,
    );
    PhaseResult { outcome, basis, x_b }
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
