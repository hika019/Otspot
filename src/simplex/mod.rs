//! 線形計画法（LP）のための主シンプレックス法モジュール
//!
//! Phase I + Phase II の改訂シンプレックス法（Revised Simplex Method）を実装する。
//! LU分解を用いた基底行列の効率的な操作により、数値的安定性と計算速度を両立する。
//!
//! # アルゴリズム概要
//!
//! 1. **Phase I**: 人工変数を導入し、初期実行可能基底解を探索する
//! 2. **Phase II**: 実行可能解から最適解に向けて目的関数を改善する
//!
//! 改訂シンプレックス法では完全な単体表ではなく基底行列のLU分解を保持するため、
//! 大規模疎行列に対して高い計算効率を発揮する。

use crate::basis::{BasisManager, LuBasis};
use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus, SolverResult};
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::*;

/// LU分解を用いた改訂シンプレックス法でLPを解く（後方互換 API）
///
/// デフォルトの [`SolverOptions`] を使用して [`solve_with`] を呼び出す。
///
/// # 引数
///
/// * `problem` - 解くべき線形計画問題
///
/// # 戻り値
///
/// [`SolverResult`] — 求解ステータス（最適・非有界・実行不可）と目的関数値・解ベクトル
pub fn solve(problem: &LpProblem) -> SolverResult {
    solve_with(problem, &SolverOptions::default())
}

/// カスタム設定でLPを解く
///
/// 与えられた線形計画問題を標準形に変換し、2相シンプレックス法で最適解を求める。
/// 変数変換（有界変数・符号制約）、スラック変数の追加、人工変数の導入を自動的に行う。
///
/// # 引数
///
/// * `problem` - 解くべき線形計画問題
/// * `options` - ソルバー動作設定（許容誤差・反復上限・eta 保持数など）
///
/// # 戻り値
///
/// [`SolverResult`] — 求解ステータス（最適・非有界・実行不可）と目的関数値・解ベクトル
pub fn solve_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    // Edge case: no variables
    if n == 0 {
        for i in 0..m {
            if problem.b[i] < -options.primal_tol {
                return SolverResult {
                    status: SolveStatus::Infeasible,
                    objective: 0.0,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                };
            }
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![],
            dual_solution: vec![0.0; m],
            reduced_costs: vec![],
            slack: problem.b.clone(),
        };
    }

    // Edge case: no constraints
    if m == 0 {
        for j in 0..n {
            if problem.c[j] < -options.primal_tol {
                return SolverResult {
                    status: SolveStatus::Unbounded,
                    objective: f64::NEG_INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                };
            }
        }
        return SolverResult {
            status: SolveStatus::Optimal,
            objective: 0.0,
            solution: vec![0.0; n],
            dual_solution: vec![],
            reduced_costs: problem.c.clone(),
            slack: vec![],
        };
    }

    let sf = build_standard_form(problem);
    two_phase_simplex(&sf, problem, options)
}

// --- Data structures ---

/// 元の変数の変換情報を保持する構造体
///
/// 各元変数がどのように標準形の変数群に対応しているかを記録する。
/// 下限・上限制約によって変数分割や符号反転が生じた際に使用する。
struct OrigVarInfo {
    /// 変数変換のオフセット（下限 lb または上限 ub の値）
    offset: f64,
    /// 新変数インデックスと係数のペアのリスト
    ///
    /// 通常は1要素（下限付き変数）または2要素（無制限変数を正部・負部に分割）
    new_vars: Vec<(usize, f64)>,
}

/// 線形計画問題の標準形表現
///
/// 元のLPを改訂シンプレックス法に適した形式に変換した結果を保持する。
/// スラック変数の追加、変数変換（下限シフト・符号反転・分割）、
/// 人工変数の要否判定を含む完全な変換済みデータを格納する。
struct StandardForm {
    /// 制約行列（疎CSC形式）
    a: CscMatrix,
    /// 制約右辺ベクトル（変換済み）
    b: Vec<f64>,
    /// 目的関数係数ベクトル（Phase II用）
    c: Vec<f64>,
    /// 制約数（上限制約行も含む）
    m: usize,
    /// 元変数を変換した新変数の数
    n_shifted: usize,
    /// 全変数数（新変数 + スラック変数）
    n_total: usize,
    /// 初期基底変数のインデックスリスト（行ごと）
    initial_basis: Vec<usize>,
    /// 各制約行が人工変数を必要とするかのフラグ
    needs_artificial: Vec<bool>,
    /// 人工変数の総数
    num_artificial: usize,
    /// 変数変換による目的関数のオフセット
    obj_offset: f64,
    /// 元の変数数
    n_orig: usize,
    /// 元変数ごとの変換情報
    orig_var_info: Vec<OrigVarInfo>,
    /// 各制約行が符号反転されているか（双対変数の符号調整に使用）
    row_negated: Vec<bool>,
}

/// シンプレックス法コア関数の実行結果
enum SimplexOutcome {
    /// 最適解が得られた。値は最適目的関数値と最適時点の双対変数ベクトル
    Optimal(f64, Vec<f64>),
    /// 問題が非有界（unbounded）であった
    Unbounded,
    /// 反復回数上限に到達した。値は打ち切り時点の目的関数値
    MaxIterations(f64),
}

// --- Standard form construction ---

/// LPを改訂シンプレックス法用の標準形に変換する
///
/// 以下の7ステップで変換を行う:
///
/// 1. 変数変換（下限シフト・上限反転・無制限変数の正負分割）
/// 2. 上限制約の追加
/// 3. 調整済み右辺ベクトルの計算
/// 4. 制約タイプの整理
/// 5. 行の符号調整とスラック変数の設定
/// 6. 初期基底と人工変数の決定
/// 7. CSC疎行列の構築
fn build_standard_form(problem: &LpProblem) -> StandardForm {
    let n_orig = problem.num_vars;
    let m_orig = problem.num_constraints;

    // 1. Variable transformations
    let mut orig_var_info: Vec<OrigVarInfo> = Vec::with_capacity(n_orig);
    let mut n_shifted = 0usize;
    let mut obj_offset = 0.0f64;
    let mut new_c: Vec<f64> = Vec::new();

    for j in 0..n_orig {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            obj_offset += problem.c[j] * lb;
            orig_var_info.push(OrigVarInfo {
                offset: lb,
                new_vars: vec![(idx, 1.0)],
            });
        } else if ub.is_finite() {
            let idx = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            obj_offset += problem.c[j] * ub;
            orig_var_info.push(OrigVarInfo {
                offset: ub,
                new_vars: vec![(idx, -1.0)],
            });
        } else {
            let idx_plus = n_shifted;
            n_shifted += 1;
            new_c.push(problem.c[j]);
            let idx_minus = n_shifted;
            n_shifted += 1;
            new_c.push(-problem.c[j]);
            orig_var_info.push(OrigVarInfo {
                offset: 0.0,
                new_vars: vec![(idx_plus, 1.0), (idx_minus, -1.0)],
            });
        }
    }

    // 2. Upper bound constraints
    let mut ub_constraints: Vec<(usize, f64)> = Vec::new();
    for j in 0..n_orig {
        let (lb, ub) = problem.bounds[j];
        if lb.is_finite() && ub.is_finite() {
            let effective_ub = ub - lb;
            let new_idx = orig_var_info[j].new_vars[0].0;
            ub_constraints.push((new_idx, effective_ub));
        }
    }
    let n_ub = ub_constraints.len();
    let m_ext = m_orig + n_ub;

    // 3. Compute adjusted b
    let mut b = problem.b.clone();
    for j in 0..n_orig {
        let offset = orig_var_info[j].offset;
        if offset.abs() > DROP_TOL {
            if let Ok((rows, vals)) = problem.a.get_column(j) {
                for (k, &row) in rows.iter().enumerate() {
                    b[row] -= vals[k] * offset;
                }
            }
        }
    }
    for &(_, ub_val) in &ub_constraints {
        b.push(ub_val);
    }

    // 4. Constraint types
    let mut ctypes: Vec<ConstraintType> = problem.constraint_types.clone();
    for _ in 0..n_ub {
        ctypes.push(ConstraintType::Le);
    }

    // 5. Row negation and slack setup
    let mut row_negated = vec![false; m_ext];
    let mut slack_col_idx: Vec<Option<usize>> = Vec::with_capacity(m_ext);
    let mut n_slack = 0usize;
    let mut slack_coeff = vec![0.0f64; m_ext];

    for i in 0..m_ext {
        match ctypes[i] {
            ConstraintType::Le => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = -1.0;
                } else {
                    slack_coeff[i] = 1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Ge => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                    slack_coeff[i] = 1.0; // -1 negated
                } else {
                    slack_coeff[i] = -1.0;
                }
                slack_col_idx.push(Some(n_slack));
                n_slack += 1;
            }
            ConstraintType::Eq => {
                if b[i] < -PIVOT_TOL {
                    row_negated[i] = true;
                    b[i] = -b[i];
                }
                slack_col_idx.push(None);
            }
        }
    }

    let n_total = n_shifted + n_slack;

    // 6. Initial basis and artificial detection
    let mut initial_basis = vec![0usize; m_ext];
    let mut needs_artificial = vec![false; m_ext];
    let mut num_artificial = 0usize;

    for i in 0..m_ext {
        match slack_col_idx[i] {
            Some(s_idx) => {
                let col = n_shifted + s_idx;
                if slack_coeff[i] > 0.0 {
                    // Le: slack >= 0 → no artificial needed
                    initial_basis[i] = col;
                } else if b[i].abs() <= PIVOT_TOL {
                    // Ge(b≈0): surplus at 0 is primal-feasible → no artificial needed.
                    // Using an artificial here causes it to remain in the Phase II basis
                    // (Phase I terminates immediately since obj=0), which allows the solver
                    // to drift and violate this constraint. Use surplus directly.
                    initial_basis[i] = col;
                } else {
                    needs_artificial[i] = true;
                    num_artificial += 1;
                    initial_basis[i] = col; // placeholder
                }
            }
            None => {
                needs_artificial[i] = true;
                num_artificial += 1;
            }
        }
    }

    // 7. Build CscMatrix from triplets
    let mut trip_rows = Vec::new();
    let mut trip_cols = Vec::new();
    let mut trip_vals = Vec::new();

    // Original variable columns (transformed)
    for j in 0..n_orig {
        if let Ok((a_rows, a_vals)) = problem.a.get_column(j) {
            for (k, &row) in a_rows.iter().enumerate() {
                let val = a_vals[k];
                let sign = if row_negated[row] { -1.0 } else { 1.0 };
                for &(new_col, coeff) in &orig_var_info[j].new_vars {
                    let actual_val = sign * val * coeff;
                    if actual_val.abs() > DROP_TOL {
                        trip_rows.push(row);
                        trip_cols.push(new_col);
                        trip_vals.push(actual_val);
                    }
                }
            }
        }
    }

    // Upper bound constraint rows
    for (ub_idx, &(new_var_idx, _)) in ub_constraints.iter().enumerate() {
        let row = m_orig + ub_idx;
        trip_rows.push(row);
        trip_cols.push(new_var_idx);
        trip_vals.push(1.0);
    }

    // Slack/surplus columns
    for i in 0..m_ext {
        if let Some(s_idx) = slack_col_idx[i] {
            let col = n_shifted + s_idx;
            trip_rows.push(i);
            trip_cols.push(col);
            trip_vals.push(slack_coeff[i]);
        }
    }

    let a = CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m_ext, n_total).unwrap();

    // Cost vector for Phase II
    let mut c_ext = vec![0.0; n_total];
    for k in 0..n_shifted {
        c_ext[k] = new_c[k];
    }

    StandardForm {
        a,
        b,
        c: c_ext,
        m: m_ext,
        n_shifted,
        n_total,
        initial_basis,
        needs_artificial,
        num_artificial,
        obj_offset,
        n_orig,
        orig_var_info,
        row_negated,
    }
}

// --- Two-phase simplex ---

/// 双対変数・被縮小費用・スラックを元問題の情報から計算する
///
/// # 引数
/// * `sf` - 標準形（行符号反転情報を含む）
/// * `problem` - 元のLP問題（制約行列・右辺ベクトル・目的係数）
/// * `y_std` - 標準形の双対変数ベクトル（長さ: sf.m）
/// * `solution` - 元問題の解ベクトル（長さ: problem.num_vars）
///
/// # 戻り値
/// `(dual_solution, reduced_costs, slack)` のタプル
fn extract_dual_info(
    sf: &StandardForm,
    problem: &LpProblem,
    y_std: &[f64],
    solution: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m_orig = problem.num_constraints;
    let n_orig = problem.num_vars;

    // 双対変数: y_std を行符号反転で調整して元制約に対応
    let mut dual_solution = vec![0.0; m_orig];
    for i in 0..m_orig {
        let sign = if sf.row_negated[i] { -1.0 } else { 1.0 };
        dual_solution[i] = sign * y_std[i];
    }

    // スラック: b - Ax（元問題の解から直接計算）
    let mut slack = problem.b.clone();
    for j in 0..n_orig {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                slack[row] -= vals[k] * solution[j];
            }
        }
    }

    // 被縮小費用: c_j - lambda^T A_j（元問題の変数に対して）
    let mut reduced_costs = problem.c.clone();
    for j in 0..n_orig {
        if let Ok((rows, vals)) = problem.a.get_column(j) {
            for (k, &row) in rows.iter().enumerate() {
                if row < m_orig {
                    reduced_costs[j] -= dual_solution[row] * vals[k];
                }
            }
        }
    }

    (dual_solution, reduced_costs, slack)
}

/// 2相シンプレックス法で標準形LPを解く
///
/// - 人工変数が不要な場合: Phase II に直接進む
/// - 人工変数が必要な場合: Phase I で実行可能基底を求めてから Phase II を実行
///
/// Phase I の目的関数は人工変数の和の最小化。
/// 最小値がゼロより大きければ元問題は実行不可能（Infeasible）。
fn two_phase_simplex(sf: &StandardForm, problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    let m = sf.m;

    if sf.num_artificial == 0 {
        // Direct Phase II
        let mut basis = sf.initial_basis.clone();
        let mut x_b = sf.b.clone();

        match revised_simplex_core(&sf.a, &mut x_b, &sf.c, &mut basis, m, sf.n_total, sf.n_total, options)
        {
            SimplexOutcome::Optimal(obj, y) => {
                let solution = extract_solution(sf, &basis, &x_b);
                let (dual_solution, reduced_costs, slack) =
                    extract_dual_info(sf, problem, &y, &solution);
                SolverResult {
                    status: SolveStatus::Optimal,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution,
                    reduced_costs,
                    slack,
                }
            }
            SimplexOutcome::Unbounded => SolverResult {
                status: SolveStatus::Unbounded,
                objective: f64::NEG_INFINITY,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
            },
            SimplexOutcome::MaxIterations(obj) => {
                let solution = extract_solution(sf, &basis, &x_b);
                SolverResult {
                    status: SolveStatus::MaxIterations,
                    objective: obj + sf.obj_offset,
                    solution,
                    dual_solution: vec![],
                    reduced_costs: vec![],
                    slack: vec![],
                }
            }
        }
    } else {
        // Phase I: build extended matrix with artificials
        let n_ext = sf.n_total + sf.num_artificial;

        let mut trip_rows = Vec::new();
        let mut trip_cols = Vec::new();
        let mut trip_vals = Vec::new();

        // Copy existing matrix
        for j in 0..sf.a.ncols {
            if let Ok((r, v)) = sf.a.get_column(j) {
                for (k, &row) in r.iter().enumerate() {
                    trip_rows.push(row);
                    trip_cols.push(j);
                    trip_vals.push(v[k]);
                }
            }
        }

        // Add artificial columns
        let mut basis = sf.initial_basis.clone();
        let mut art_col = sf.n_total;
        for i in 0..m {
            if sf.needs_artificial[i] {
                trip_rows.push(i);
                trip_cols.push(art_col);
                trip_vals.push(1.0);
                basis[i] = art_col;
                art_col += 1;
            }
        }

        let a_ext =
            CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, m, n_ext).unwrap();

        // Phase I cost
        let mut c_phase1 = vec![0.0; n_ext];
        for j in sf.n_total..n_ext {
            c_phase1[j] = 1.0;
        }

        let mut x_b = sf.b.clone();

        match revised_simplex_core(&a_ext, &mut x_b, &c_phase1, &mut basis, m, n_ext, n_ext, options) {
            SimplexOutcome::Optimal(obj, _) => {
                if obj > PIVOT_TOL {
                    return SolverResult {
                        status: SolveStatus::Infeasible,
                        objective: 0.0,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                    };
                }

                // Phase II: restrict pricing to non-artificial columns
                let mut c_phase2 = vec![0.0; n_ext];
                for k in 0..sf.n_total {
                    c_phase2[k] = sf.c[k];
                }

                match revised_simplex_core(
                    &a_ext,
                    &mut x_b,
                    &c_phase2,
                    &mut basis,
                    m,
                    n_ext,
                    sf.n_total,
                    options,
                ) {
                    SimplexOutcome::Optimal(obj2, y) => {
                        let solution = extract_solution(sf, &basis, &x_b);
                        let (dual_solution, reduced_costs, slack) =
                            extract_dual_info(sf, problem, &y, &solution);
                        SolverResult {
                            status: SolveStatus::Optimal,
                            objective: obj2 + sf.obj_offset,
                            solution,
                            dual_solution,
                            reduced_costs,
                            slack,
                        }
                    }
                    SimplexOutcome::Unbounded => SolverResult {
                        status: SolveStatus::Unbounded,
                        objective: f64::NEG_INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        reduced_costs: vec![],
                        slack: vec![],
                    },
                    SimplexOutcome::MaxIterations(obj2) => {
                        let solution = extract_solution(sf, &basis, &x_b);
                        SolverResult {
                            status: SolveStatus::MaxIterations,
                            objective: obj2 + sf.obj_offset,
                            solution,
                            dual_solution: vec![],
                            reduced_costs: vec![],
                            slack: vec![],
                        }
                    }
                }
            }
            SimplexOutcome::Unbounded => SolverResult {
                status: SolveStatus::Infeasible,
                objective: 0.0,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
            },
            SimplexOutcome::MaxIterations(_) => SolverResult {
                status: SolveStatus::MaxIterations,
                objective: 0.0,
                solution: vec![],
                dual_solution: vec![],
                reduced_costs: vec![],
                slack: vec![],
            },
        }
    }
}

/// 最適基底解から元の変数への解ベクトルを復元する
///
/// 標準形の最適解を、変数変換（オフセット・係数）を逆適用して
/// 元問題の変数値に変換する。
fn extract_solution(sf: &StandardForm, basis: &[usize], x_b: &[f64]) -> Vec<f64> {
    let mut x_new = vec![0.0; sf.n_shifted];
    for i in 0..sf.m {
        if basis[i] < sf.n_shifted {
            x_new[basis[i]] = x_b[i];
        }
    }

    let mut solution = vec![0.0; sf.n_orig];
    for j in 0..sf.n_orig {
        let info = &sf.orig_var_info[j];
        solution[j] = info.offset;
        for &(new_idx, coeff) in &info.new_vars {
            solution[j] += coeff * x_new[new_idx];
        }
    }
    solution
}

// --- Core Revised Simplex ---

/// 改訂シンプレックス法のコアアルゴリズム
///
/// LU分解を用いて基底行列を管理し、以下の手順を繰り返す:
///
/// 1. **BTRAN**: 双対変数 `y = B^{-T} c_B` を計算
/// 2. **価格付け（Pricing）**: 最も負の被縮小費用を持つ入基変数を選択
/// 3. **FTRAN**: ピボット列 `d = B^{-1} a_j` を計算
/// 4. **比率テスト**: Bland則を用いて離基変数を選択（退化サイクルの防止）
/// 5. **基底更新**: `x_B` の更新と基底行列のrank-1更新
///
/// # 引数
///
/// * `a` - 制約行列（CSC疎行列形式）
/// * `x_b` - 現在の基底解ベクトル（更新される）
/// * `c` - 目的関数係数ベクトル
/// * `basis` - 基底変数インデックスリスト（更新される）
/// * `m` - 制約数
/// * `n_cols` - 全列数
/// * `n_price` - 価格付けの対象列数（人工変数を除く場合に `n_total` を指定）
/// * `options` - ソルバー設定（反復上限・eta 保持数・クランプ閾値を含む）
///
/// # 戻り値
///
/// [`SimplexOutcome::Optimal`] — 最適目的関数値、または [`SimplexOutcome::Unbounded`]
fn revised_simplex_core(
    a: &CscMatrix,
    x_b: &mut Vec<f64>,
    c: &[f64],
    basis: &mut Vec<usize>,
    m: usize,
    n_cols: usize,
    n_price: usize,
    options: &SolverOptions,
) -> SimplexOutcome {
    let max_iter = options.max_iterations.unwrap_or(100 * (m + n_cols) + 1000);
    let mut basis_mgr = match LuBasis::new(a, basis, options.max_etas) {
        Ok(bm) => bm,
        Err(_) => {
            let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
            return SimplexOutcome::Optimal(obj, vec![0.0; m]);
        }
    };

    let mut is_basic = vec![false; n_cols];
    for &b in basis.iter() {
        is_basic[b] = true;
    }

    // Pre-allocate reusable buffers to avoid heap allocation inside the iteration loop
    let mut c_b = vec![0.0f64; m];
    let mut y_dense = vec![0.0f64; m];
    let mut d_dense = vec![0.0f64; m];

    for _iter in 0..max_iter {
        // 1. Dual variables: y = BTRAN(c_B)
        for i in 0..m {
            c_b[i] = c[basis[i]];
        }
        let mut y_sv = SparseVec::from_dense(&c_b);
        basis_mgr.btran(&mut y_sv);
        y_sv.to_dense_into(&mut y_dense);
        let y = &y_dense;

        // 2. Pricing: find most negative reduced cost
        let mut entering = None;
        let mut min_rc = -PIVOT_TOL;

        for j in 0..n_price {
            if is_basic[j] {
                continue;
            }
            let (rows, vals) = a.get_column(j).unwrap();
            let mut ya = 0.0;
            for (k, &row) in rows.iter().enumerate() {
                ya += y[row] * vals[k];
            }
            let rc = c[j] - ya;
            if rc < min_rc {
                min_rc = rc;
                entering = Some(j);
            }
        }

        let entering_col = match entering {
            None => {
                let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
                return SimplexOutcome::Optimal(obj, y_dense.clone());
            }
            Some(j) => j,
        };

        // 3. FTRAN: pivot column d = B^{-1} * a_entering
        let (col_rows, col_vals) = a.get_column(entering_col).unwrap();
        let mut d_sv = SparseVec {
            indices: col_rows.to_vec(),
            values: col_vals.to_vec(),
            len: m,
        };
        basis_mgr.ftran(&mut d_sv);
        d_sv.to_dense_into(&mut d_dense);
        let d = &d_dense;

        // 4. Ratio test (Bland's rule for ties)
        let mut leaving = None;
        let mut min_ratio = f64::INFINITY;

        for i in 0..m {
            if d[i] > PIVOT_TOL {
                let ratio = x_b[i] / d[i];
                if ratio < min_ratio - PIVOT_TOL {
                    min_ratio = ratio;
                    leaving = Some(i);
                } else if (ratio - min_ratio).abs() < PIVOT_TOL {
                    if let Some(prev) = leaving {
                        if basis[i] < basis[prev] {
                            leaving = Some(i);
                        }
                    }
                }
            }
        }

        let leaving_row = match leaving {
            None => return SimplexOutcome::Unbounded,
            Some(i) => i,
        };

        // 5. Update x_b
        let step = x_b[leaving_row] / d[leaving_row];
        for i in 0..m {
            x_b[i] -= d[i] * step;
        }
        x_b[leaving_row] = step;

        // Clamp near-zero to prevent drift
        for val in x_b.iter_mut() {
            if val.abs() < options.clamp_tol {
                *val = 0.0;
            }
        }

        // 6. Update basis tracking
        is_basic[basis[leaving_row]] = false;
        is_basic[entering_col] = true;

        // 7. Update basis manager
        basis_mgr.update(entering_col, leaving_row, &d_sv);
        basis[leaving_row] = entering_col;

        // 8. Refactor if needed
        basis_mgr.refactor_if_needed(a, basis);
    }

    let obj: f64 = (0..m).map(|i| c[basis[i]] * x_b[i]).sum();
    SimplexOutcome::MaxIterations(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    fn make_lp(
        c: Vec<f64>,
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        nrows: usize,
        ncols: usize,
        b: Vec<f64>,
    ) -> LpProblem {
        let a = CscMatrix::from_triplets(rows, cols, vals, nrows, ncols).unwrap();
        LpProblem::new(c, a, b).unwrap()
    }

    #[test]
    fn test_basic_2var() {
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-4.0)).abs() < PIVOT_TOL,
            "Expected objective -4.0, got {}",
            result.objective
        );
        let x1 = result.solution[0];
        let x2 = result.solution[1];
        assert!(x1 >= -PIVOT_TOL && x1 <= 3.0 + PIVOT_TOL, "x1={}", x1);
        assert!(x2 >= -PIVOT_TOL && x2 <= 3.0 + PIVOT_TOL, "x2={}", x2);
        assert!((x1 + x2 - 4.0).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_basic_3var() {
        let lp = make_lp(
            vec![-2.0, -3.0, -1.0],
            &[0, 0, 0, 1, 1, 2, 2],
            &[0, 1, 2, 0, 1, 1, 2],
            &[1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            3,
            3,
            vec![10.0, 14.0, 8.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        let x = &result.solution;
        assert!(x[0] >= -PIVOT_TOL);
        assert!(x[1] >= -PIVOT_TOL);
        assert!(x[2] >= -PIVOT_TOL);
        assert!(x[0] + x[1] + x[2] <= 10.0 + PIVOT_TOL);
        assert!(2.0 * x[0] + x[1] <= 14.0 + PIVOT_TOL);
        assert!(x[1] + x[2] <= 8.0 + PIVOT_TOL);
        assert!(
            (result.objective - (-28.0)).abs() < PIVOT_TOL,
            "Expected objective -28.0, got {}",
            result.objective
        );
    }

    #[test]
    fn test_unbounded() {
        let lp = make_lp(
            vec![-1.0, 0.0],
            &[0, 0],
            &[0, 1],
            &[1.0, -1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_infeasible() {
        let lp = make_lp(vec![1.0], &[0], &[0], &[1.0], 1, 1, vec![-1.0]);
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Infeasible);
    }

    #[test]
    fn test_degenerate_zero_vars() {
        let a = CscMatrix::new(0, 0);
        let lp = LpProblem::new(vec![], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_zero_constraints_unbounded() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![-1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Unbounded);
    }

    #[test]
    fn test_zero_constraints_optimal() {
        let a = CscMatrix::new(0, 1);
        let lp = LpProblem::new(vec![1.0], a, vec![]).unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!((result.objective).abs() < PIVOT_TOL);
    }

    #[test]
    fn test_max_iter_returns_max_iterations() {
        // min -x1 - x2  s.t.  x1 + x2 <= 4, x1 <= 3, x2 <= 3  (optimal: -4.0)
        // max_iter=1 なら反復1回で打ち切られ MaxIterations が返る
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let sf = build_standard_form(&lp);
        let m = sf.m;
        let mut basis = sf.initial_basis.clone();
        let mut x_b = sf.b.clone();
        let opts = SolverOptions {
            max_iterations: Some(1),
            ..SolverOptions::default()
        };
        let outcome = revised_simplex_core(
            &sf.a,
            &mut x_b,
            &sf.c,
            &mut basis,
            m,
            sf.n_total,
            sf.n_total,
            &opts,
        );
        assert!(
            matches!(outcome, SimplexOutcome::MaxIterations(_)),
            "Expected MaxIterations, got something else"
        );
    }

    #[test]
    fn test_solve_with_custom_max_iterations() {
        // max_iterations=1 で MaxIterations ステータスが返ること
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let opts = SolverOptions {
            max_iterations: Some(1),
            ..SolverOptions::default()
        };
        let result = solve_with(&lp, &opts);
        assert_eq!(
            result.status,
            SolveStatus::MaxIterations,
            "Expected MaxIterations with max_iterations=1"
        );
    }

    #[test]
    fn test_solve_with_default_options() {
        // SolverOptions::default() で solve() と同じ結果が返ること
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result_default = solve(&lp);
        let result_with = solve_with(&lp, &SolverOptions::default());
        assert_eq!(result_default.status, result_with.status);
        assert!(
            (result_default.objective - result_with.objective).abs() < PIVOT_TOL,
            "solve() and solve_with(default) should return same objective"
        );
    }

    /// 基本LP（Le制約のみ）の双対解・スラック・被縮小費用を検証する
    ///
    /// 問題: min -x1 - 2*x2
    ///   s.t. x1 + x2 <= 4
    ///        x1 <= 3
    ///        x2 <= 3
    ///        x1, x2 >= 0
    ///
    /// 最適解: x1=1, x2=3, obj=-7
    /// 双対変数: y=[-1, 0, -1] (Le制約のshadow price、最小化LPでは<=0)
    /// スラック: s=[0, 2, 0]
    /// 被縮小費用: rc=[0, 0] (基底変数なのでゼロ)
    #[test]
    fn test_dual_solution_basic_le_constraints() {
        let lp = make_lp(
            vec![-1.0, -2.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![4.0, 3.0, 3.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - (-7.0)).abs() < PIVOT_TOL,
            "Expected obj=-7.0, got {}",
            result.objective
        );

        // 双対変数の検証
        assert_eq!(result.dual_solution.len(), 3, "dual_solution should have 3 elements");
        assert!(
            (result.dual_solution[0] - (-1.0)).abs() < PIVOT_TOL,
            "y[0] should be -1.0, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.dual_solution[1].abs() < PIVOT_TOL,
            "y[1] should be 0.0 (non-binding), got {}",
            result.dual_solution[1]
        );
        assert!(
            (result.dual_solution[2] - (-1.0)).abs() < PIVOT_TOL,
            "y[2] should be -1.0, got {}",
            result.dual_solution[2]
        );

        // スラック変数の検証
        assert_eq!(result.slack.len(), 3, "slack should have 3 elements");
        assert!(
            result.slack[0].abs() < PIVOT_TOL,
            "slack[0] should be 0 (binding), got {}",
            result.slack[0]
        );
        assert!(
            (result.slack[1] - 2.0).abs() < PIVOT_TOL,
            "slack[1] should be 2.0 (non-binding), got {}",
            result.slack[1]
        );
        assert!(
            result.slack[2].abs() < PIVOT_TOL,
            "slack[2] should be 0 (binding), got {}",
            result.slack[2]
        );

        // 被縮小費用の検証（基底変数なのでゼロ）
        assert_eq!(result.reduced_costs.len(), 2, "reduced_costs should have 2 elements");
        assert!(
            result.reduced_costs[0].abs() < PIVOT_TOL,
            "rc[0] should be 0 (basic), got {}",
            result.reduced_costs[0]
        );
        assert!(
            result.reduced_costs[1].abs() < PIVOT_TOL,
            "rc[1] should be 0 (basic), got {}",
            result.reduced_costs[1]
        );
    }

    #[test]
    fn test_large_coefficient_lp() {
        // 係数に 1e12 と 1e-12 を混合した問題 → Optimal or 適切なステータス（オーバーフローしない）
        // min -1e12 * x1 + 1e-12 * x2, s.t. x1 + x2 <= 1, x1,x2 >= 0
        // 最適解: x1=1, x2=0, obj=-1e12
        let lp = make_lp(
            vec![-1e12, 1e-12],
            &[0, 0],
            &[0, 1],
            &[1.0, 1.0],
            1,
            2,
            vec![1.0],
        );
        let result = solve(&lp);
        assert!(
            result.status == SolveStatus::Optimal || result.status == SolveStatus::MaxIterations,
            "Expected Optimal or MaxIterations, got {:?}",
            result.status
        );
        assert!(!result.objective.is_nan(), "Objective should not be NaN");
        assert!(result.objective.is_finite(), "Objective should be finite for bounded LP");

        // 全係数 0.0 の目的関数 → Optimal, objective=0.0
        // min 0*x1 + 0*x2, s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
        let lp_zero = make_lp(
            vec![0.0, 0.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![2.0, 1.0, 1.0],
        );
        let result_zero = solve(&lp_zero);
        assert_eq!(result_zero.status, SolveStatus::Optimal, "Expected Optimal for zero-objective LP");
        assert!(
            result_zero.objective.abs() < PIVOT_TOL,
            "Expected objective=0.0, got {}",
            result_zero.objective
        );
    }

    #[test]
    fn test_highly_degenerate_lp() {
        // 高度退化 LP: 3制約が (1,1) で交わる → 基底解が退化
        // min -x1 - x2
        // s.t. x1 + x2 <= 2, x1 <= 1, x2 <= 1
        // 最適解: x1=1, x2=1, obj=-2（サイクリングせずに到達すること）
        let lp = make_lp(
            vec![-1.0, -1.0],
            &[0, 0, 1, 2],
            &[0, 1, 0, 1],
            &[1.0, 1.0, 1.0, 1.0],
            3,
            2,
            vec![2.0, 1.0, 1.0],
        );
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal, "Expected Optimal for degenerate LP");
        assert!(
            (result.objective - (-2.0)).abs() < PIVOT_TOL,
            "Expected objective=-2.0, got {}",
            result.objective
        );
        let x1 = result.solution[0];
        let x2 = result.solution[1];
        assert!((x1 - 1.0).abs() < PIVOT_TOL, "Expected x1=1.0, got {}", x1);
        assert!((x2 - 1.0).abs() < PIVOT_TOL, "Expected x2=1.0, got {}", x2);
    }

    /// 等式制約付きLPの双対解・スラック・被縮小費用を検証する
    ///
    /// 問題: min x1 + 2*x2
    ///   s.t. x1 + x2 = 6  (Eq)
    ///        x2 <= 5       (Le)
    ///        x1, x2 >= 0
    ///
    /// 最適解: x1=6, x2=0, obj=6
    /// 双対変数: y=[1, 0] (Eq制約shadow price=1、x2<=5は非binding)
    /// スラック: s=[0, 5] (等式制約は0、x2<=5は5余裕)
    /// 被縮小費用: rc=[0, 1] (x1は基底でrc=0、x2は非基底でrc=1)
    #[test]
    fn test_dual_solution_equality_constraint() {
        use crate::problem::ConstraintType;
        let a = CscMatrix::from_triplets(
            &[0, 0, 1],
            &[0, 1, 1],
            &[1.0, 1.0, 1.0],
            2,
            2,
        )
        .unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 2.0],
            a,
            vec![6.0, 5.0],
            vec![ConstraintType::Eq, ConstraintType::Le],
            vec![(0.0, f64::INFINITY), (0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        let result = solve(&lp);
        assert_eq!(result.status, SolveStatus::Optimal);
        assert!(
            (result.objective - 6.0).abs() < PIVOT_TOL,
            "Expected obj=6.0, got {}",
            result.objective
        );

        // 双対変数の検証
        assert_eq!(result.dual_solution.len(), 2, "dual_solution should have 2 elements");
        assert!(
            (result.dual_solution[0] - 1.0).abs() < PIVOT_TOL,
            "y[0] (Eq constraint shadow price) should be 1.0, got {}",
            result.dual_solution[0]
        );
        assert!(
            result.dual_solution[1].abs() < PIVOT_TOL,
            "y[1] (Le constraint, non-binding) should be 0.0, got {}",
            result.dual_solution[1]
        );

        // スラック変数の検証
        assert_eq!(result.slack.len(), 2, "slack should have 2 elements");
        assert!(
            result.slack[0].abs() < PIVOT_TOL,
            "slack[0] (Eq constraint) should be 0, got {}",
            result.slack[0]
        );
        assert!(
            (result.slack[1] - 5.0).abs() < PIVOT_TOL,
            "slack[1] (x2<=5, non-binding) should be 5.0, got {}",
            result.slack[1]
        );

        // 被縮小費用の検証
        assert_eq!(result.reduced_costs.len(), 2, "reduced_costs should have 2 elements");
        assert!(
            result.reduced_costs[0].abs() < PIVOT_TOL,
            "rc[0] (x1, basic) should be 0.0, got {}",
            result.reduced_costs[0]
        );
        assert!(
            (result.reduced_costs[1] - 1.0).abs() < PIVOT_TOL,
            "rc[1] (x2, non-basic) should be 1.0, got {}",
            result.reduced_costs[1]
        );
    }
}
