//! IPM→ASクロスオーバーソルバー
//!
//! IPMで近似解を得た後、活性制約を同定し、
//! Active Set法にwarm-startして高精度解を得る。
//!
//! Gurobi/CPLEXが採用する確立技術（Megiddo 1991, Ye 1992）。
//! IPMのテーリング問題（制約面への着地が遅い）を解消する。

use crate::linalg::timeout::TimeoutCtx;
use crate::options::SolverOptions;
use crate::problem::{SolveStatus, SolverResult};
use crate::qp::problem::{QpProblem, QpWarmStart};
use crate::qp::QpSolver;
use crate::qp::{ipm, solver};

// ---------------------------------------------------------------------------
// Phase A: Active Constraint Identification
// ---------------------------------------------------------------------------

/// 変数→AS augmented制約インデックスのルックアップテーブルを構築する
///
/// ASの `augment_bounds_to_constraints` は各変数jに対してub→lbの順で
/// 有限な境界を行に追加する（m_origから開始）。
/// 本関数はその同じ順序でインデックスを構築する。
///
/// # 引数
/// - `bounds`: 変数境界 (lb, ub) のスライス
/// - `m_orig`: オリジナル制約数（AS augmented行のオフセット）
///
/// # 戻り値
/// Vec<(ub_as_idx, lb_as_idx)> — Noneは対応する境界が無限大であることを示す
pub(crate) fn build_bound_index_map(
    bounds: &[(f64, f64)],
    m_orig: usize,
) -> Vec<(Option<usize>, Option<usize>)> {
    let mut map = Vec::with_capacity(bounds.len());
    let mut as_row = m_orig;
    for &(lb, ub) in bounds {
        let ub_idx = if ub.is_finite() {
            let idx = as_row;
            as_row += 1;
            Some(idx)
        } else {
            None
        };
        let lb_idx = if lb.is_finite() {
            let idx = as_row;
            as_row += 1;
            Some(idx)
        } else {
            None
        };
        map.push((ub_idx, lb_idx));
    }
    map
}

/// IPM解から活性制約インデックスを同定する
///
/// unscaled primal解 x と original problemのスラックを再計算して
/// |slack| < eps_id の制約を活性と判定する。
/// 返り値はASのaugmented constraint indexing体系でのインデックス。
///
/// Ruizスケーリングの影響を回避するため、sではなくxから直接スラックを再計算する。
///
/// # 引数
/// - `x`: unscaled primal解 (n次元)
/// - `problem`: 元のQpProblem (unscaled)
/// - `eps_id`: 同定tolerance (推奨: sqrt(solver_eps))
///
/// # 戻り値
/// - `active_indices`: ASのaugmented indexingでの活性制約インデックス
/// - `count`: 同定された活性制約数
pub(crate) fn identify_active_set(
    x: &[f64],
    problem: &QpProblem,
    eps_id: f64,
) -> (Vec<usize>, usize) {
    let m = problem.num_constraints;

    let mut active_indices = Vec::new();
    let mut count = 0;

    // (1) オリジナル制約のスラック: s_i = b_i - (A*x)_i
    let mut ax = vec![0.0; m];
    for (col, &x_col) in x.iter().enumerate() {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            ax[problem.a.row_ind[k]] += problem.a.values[k] * x_col;
        }
    }
    for (i, (&b_i, &ax_i)) in problem.b.iter().zip(ax.iter()).enumerate() {
        let slack = b_i - ax_i;
        if slack.abs() < eps_id {
            active_indices.push(i);
            count += 1;
        }
    }

    // (2) 変数境界のスラック
    let bound_map = build_bound_index_map(&problem.bounds, m);
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        // 上界: slack = ub - x[j]
        if ub.is_finite() && (ub - x[j]).abs() < eps_id {
            if let Some(idx) = bound_map[j].0 {
                active_indices.push(idx);
                count += 1;
            }
        }
        // 下界: slack = x[j] - lb
        if lb.is_finite() && (x[j] - lb).abs() < eps_id {
            if let Some(idx) = bound_map[j].1 {
                active_indices.push(idx);
                count += 1;
            }
        }
    }

    (active_indices, count)
}

// ---------------------------------------------------------------------------
// Phase B: IpmCrossoverSolver
// ---------------------------------------------------------------------------

/// IPM→ASクロスオーバーソルバー
///
/// `QpSolver` trait を実装する。内部で [`solve_ipm_crossover`] を呼ぶ。
pub struct IpmCrossoverSolver;

impl QpSolver for IpmCrossoverSolver {
    fn solve(&self, problem: &QpProblem, options: &SolverOptions) -> SolverResult {
        solve_ipm_crossover(problem, options)
    }

    fn name(&self) -> &'static str {
        "IpmCrossover"
    }
}

/// IPM→ASクロスオーバーで QP を解く
///
/// IPMで近似解を得た後、活性制約を同定し、
/// Active Set法にwarm-startして高精度解を得る。
///
/// # フォールバック動作
/// - IPM が Infeasible/Unbounded → IPM結果をそのまま返す
/// - IPM後にタイムアウト → IPM結果をそのまま返す
/// - 活性制約が0個（内部点最適）→ IPM結果をそのまま返す
/// - AS が失敗 → IPM結果にフォールバック
pub fn solve_ipm_crossover(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    // deadline を1回だけ計算して IPM・AS 両方に渡す（二重カウント防止）
    let mut effective_opts;
    let opts = if let (Some(secs), true) = (options.timeout_secs, options.deadline.is_none()) {
        effective_opts = options.clone();
        effective_opts.deadline = Some(
            std::time::Instant::now() + std::time::Duration::from_secs_f64(secs),
        );
        &effective_opts
    } else {
        options
    };

    let timeout_ctx = TimeoutCtx::from_options(opts);

    // ─── Step 1: IPM実行 ───
    let ipm_result = ipm::solve_qp_ipm(problem, opts);

    // IPM が Infeasible/Unbounded → そのまま返す（crossover不要）
    match ipm_result.status {
        SolveStatus::Infeasible | SolveStatus::Unbounded => return ipm_result,
        _ => {}
    }

    // x が空（初期化すら失敗）→ そのまま返す
    if ipm_result.solution.is_empty() {
        return ipm_result;
    }

    // タイムアウトチェック（IPM後の残り時間を確認）
    if timeout_ctx.should_stop() {
        return ipm_result;
    }

    // ─── Step 2: Active Constraint Identification ───
    let eps_id = opts.ipm.eps.sqrt();
    let (active_indices, _id_count) =
        identify_active_set(&ipm_result.solution, problem, eps_id);

    // 活性制約が0個 → 内部点最適解 → ASに渡す意味がない
    if active_indices.is_empty() {
        return ipm_result;
    }

    // ─── Step 3: AS Warm-Start 構築 ───
    let warm_start = QpWarmStart {
        initial_active_set: active_indices,
        initial_point: Some(ipm_result.solution.clone()),
    };

    // ─── Step 4: AS実行 ───
    // Concurrent Solver内ではAS内部並列は不要
    let mut as_opts = opts.clone();
    as_opts.parallel_runs = 1;

    let as_result = solver::qp_solve_impl(problem, Some(&warm_start), &as_opts);

    // AS が Optimal → AS結果を採用（bound_duals等が揃う）
    // AS が失敗 → IPM結果にフォールバック
    match as_result.status {
        SolveStatus::Optimal => as_result,
        _ => ipm_result,
    }
}

// ---------------------------------------------------------------------------
// テスト (C1 + C2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use crate::qp::problem::QpProblem;
    use crate::qp::solver as qp_solver;
    use crate::sparse::CscMatrix;

    const EPS: f64 = 1e-4;

    fn assert_close(a: f64, b: f64, eps: f64, name: &str) {
        assert!(
            (a - b).abs() < eps,
            "{}: expected {:.8}, got {:.8} (diff={:.2e})",
            name, b, a, (a - b).abs()
        );
    }

    /// C1-T1: identify_active_set_basic
    /// 2変数QP: min x^2+y^2  s.t. x+y>=1  (A=[[-1,-1]], b=[-1])
    /// 最適解 x=[0.5, 0.5] で制約0のスラック = -1 - (-0.5 - 0.5) = 0 → 活性
    #[test]
    fn test_identify_active_set_basic() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let x = vec![0.5, 0.5];
        let eps_id = 1e-6;
        let (active, count) = identify_active_set(&x, &problem, eps_id);

        assert_eq!(count, 1, "C1-T1: 1つの活性制約");
        assert_eq!(active, vec![0], "C1-T1: 制約0が活性");
    }

    /// C1-T2: identify_active_set_bounds
    /// 境界あり問題で上界が活性な場合のインデックス変換
    /// bounds=[(0,1),(0,1)], m=0, x=[1.0, 0.5]
    /// x[0]=1.0=ub → ub(x[0])のAS index=0が活性
    #[test]
    fn test_identify_active_set_bounds() {
        let n = 2;
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], n, n).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, n);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        // x[0]=1.0=ub → active; x[1]=0.5 → not active
        let x = vec![1.0, 0.5];
        let eps_id = 1e-6;
        let (active, count) = identify_active_set(&x, &problem, eps_id);

        // bound_map for m=0: j=0 → (Some(0), Some(1)), j=1 → (Some(2), Some(3))
        // ub(x[0]) = 1.0 - 1.0 = 0 < eps_id → active at index 0
        assert_eq!(count, 1, "C1-T2: 1つの活性境界");
        assert!(active.contains(&0), "C1-T2: ub(x[0])のAS index=0が活性");
    }

    /// C1-T3: identify_active_set_empty
    /// 内部点解（制約も境界も活性でない）の場合に空が返るか
    #[test]
    fn test_identify_active_set_empty() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        // x=[0.3,0.3]: slack = -1 - (-0.3-0.3) = -0.4 ≠ 0 → not active
        let x = vec![0.3, 0.3];
        let eps_id = 1e-6;
        let (active, count) = identify_active_set(&x, &problem, eps_id);

        assert_eq!(count, 0, "C1-T3: 活性制約なし");
        assert!(active.is_empty(), "C1-T3: 空のactive set");
    }

    /// C1-T4: test_bound_index_map
    /// bounds=[(0.0,1.0),(NEG_INF,2.0),(3.0,INF)] with m_orig=2
    /// j=0: ub=1.0→Some(2), lb=0.0→Some(3)
    /// j=1: ub=2.0→Some(4), lb=-inf→None
    /// j=2: ub=+inf→None, lb=3.0→Some(5)
    #[test]
    fn test_bound_index_map() {
        let bounds = vec![
            (0.0_f64, 1.0_f64),
            (f64::NEG_INFINITY, 2.0_f64),
            (3.0_f64, f64::INFINITY),
        ];
        let m_orig = 2;
        let map = build_bound_index_map(&bounds, m_orig);

        assert_eq!(map.len(), 3, "T4: map length");
        assert_eq!(map[0].0, Some(2), "T4: j=0 ub_idx=m_orig=2");
        assert_eq!(map[0].1, Some(3), "T4: j=0 lb_idx=3");
        assert_eq!(map[1].0, Some(4), "T4: j=1 ub_idx=4");
        assert_eq!(map[1].1, None, "T4: j=1 lb=-inf→None");
        assert_eq!(map[2].0, None, "T4: j=2 ub=+inf→None");
        assert_eq!(map[2].1, Some(5), "T4: j=2 lb_idx=5");
    }

    /// C2-T1: test_crossover_basic
    /// 2変数QP: min x^2+y^2  s.t. x+y>=1
    /// crossoverがOptimalを返し、解が正しいか
    #[test]
    fn test_crossover_basic() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_ipm_crossover(&problem, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Optimal, "C2-T1: status should be Optimal");
        assert_close(result.solution[0], 0.5, EPS, "C2-T1: x[0]");
        assert_close(result.solution[1], 0.5, EPS, "C2-T1: x[1]");
        assert_close(result.objective, 0.5, EPS, "C2-T1: objective");
    }

    /// C2-T2: test_crossover_box_constrained
    /// Box制約付きQP: min (x-2)^2+(y-2)^2  s.t. 0<=x,y<=1
    /// 最適: x=y=1 (上界が活性). crossoverはbound_dualsを返すか
    #[test]
    fn test_crossover_box_constrained() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![-4.0, -4.0];
        let a = CscMatrix::new(0, 2);
        let b = vec![];
        let bounds = vec![(0.0_f64, 1.0_f64); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let result = solve_ipm_crossover(&problem, &SolverOptions::default());
        assert_eq!(result.status, SolveStatus::Optimal, "C2-T2: status should be Optimal");
        assert_close(result.solution[0], 1.0, EPS, "C2-T2: x[0] at upper bound");
        assert_close(result.solution[1], 1.0, EPS, "C2-T2: x[1] at upper bound");
        assert_close(result.objective, -6.0, EPS, "C2-T2: objective");
        // AS経由なのでbound_dualsが設定される
        assert!(!result.bound_duals.is_empty(), "C2-T2: bound_duals should be set");
    }

    /// C2-T3: test_crossover_vs_as_consistency
    /// AS単体とcrossoverが同一問題で同じ解を返すか
    #[test]
    fn test_crossover_vs_as_consistency() {
        let q = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, 2.0], 2, 2).unwrap();
        let c = vec![0.0, 0.0];
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[-1.0, -1.0], 1, 2).unwrap();
        let b = vec![-1.0];
        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); 2];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let opts = SolverOptions::default();

        let as_result = qp_solver::qp_solve_impl(&problem, None, &opts);
        let crossover_result = solve_ipm_crossover(&problem, &opts);

        assert_eq!(as_result.status, SolveStatus::Optimal, "C2-T3: AS status");
        assert_eq!(crossover_result.status, SolveStatus::Optimal, "C2-T3: crossover status");

        assert_close(crossover_result.solution[0], as_result.solution[0], EPS, "C2-T3: x[0]");
        assert_close(crossover_result.solution[1], as_result.solution[1], EPS, "C2-T3: x[1]");
        assert_close(crossover_result.objective, as_result.objective, EPS, "C2-T3: objective");
    }
}
