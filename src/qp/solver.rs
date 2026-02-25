//! Active Set法 QPソルバー実装
//!
//! Phase I（初期実行可能点探索）と Phase II（Active Setメインループ）を実装する。
//! NC1修正済み KktSolver を使用する。

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, LpProblem, SolveStatus};
use crate::qp::active_set::WorkingSet;
use crate::qp::kkt::{self, KktSolver};
use crate::qp::problem::{QpProblem, QpResult, QpWarmStart};
use crate::sparse::CscMatrix;
use crate::tolerances::*;
use crate::backend::{LpBackend, SimplexBackend};
use crate::qp::kkt::extract_active_rows;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

/// タイムアウト + キャンセルを一元管理するヘルパー
struct TimeoutContext {
    deadline: Option<Instant>,
    cancel: Arc<AtomicBool>,
}

impl TimeoutContext {
    /// SolverOptions からコンテキストを構築する（最初の1回のみ呼ぶ）
    fn from_options(options: &SolverOptions) -> Self {
        let deadline = options.deadline.or_else(|| {
            options
                .timeout_secs
                .map(|s| Instant::now() + Duration::from_secs_f64(s))
        });
        let cancel = options
            .cancel_flag
            .clone()
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        Self { deadline, cancel }
    }

    #[inline]
    fn should_stop(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
            || self
                .deadline
                .is_some_and(|d| Instant::now() >= d)
    }
}

/// 変数境界を明示的な不等式制約行に変換して A 行列に追加する
///
/// 各有限境界を追加制約として展開する:
/// - x[j] <= ub[j] → 行インデックス m + k: [0,...,+1,...,0] <= ub[j]
/// - x[j] >= lb[j] → 行インデックス m + k: [0,...,-1,...,0] <= -lb[j]
///
/// 境界が無限大の変数はスキップする。
/// 返値は (augmented_A, augmented_b)。境界なしの場合は元の A, b をクローンして返す。
fn augment_bounds_to_constraints(
    a: &CscMatrix,
    b: &[f64],
    bounds: &[(f64, f64)],
) -> (CscMatrix, Vec<f64>) {
    let m = b.len();
    let n = bounds.len();

    // 追加する境界制約: (var_j, coeff, rhs)
    let mut extra: Vec<(usize, f64, f64)> = Vec::new();
    for (j, &(lb, ub)) in bounds.iter().enumerate() {
        if ub.is_finite() {
            extra.push((j, 1.0, ub));   // x[j] <= ub
        }
        if lb.is_finite() {
            extra.push((j, -1.0, -lb)); // -x[j] <= -lb  (x[j] >= lb)
        }
    }

    if extra.is_empty() {
        return (a.clone(), b.to_vec());
    }

    let new_m = m + extra.len();
    let mut new_b = b.to_vec();

    // COO 形式で新 A 行列を構築
    let mut rows_coo: Vec<usize> = Vec::new();
    let mut cols_coo: Vec<usize> = Vec::new();
    let mut vals_coo: Vec<f64> = Vec::new();

    // 元の A 要素をコピー
    for col in 0..n {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            rows_coo.push(a.row_ind[k]);
            cols_coo.push(col);
            vals_coo.push(a.values[k]);
        }
    }

    // 境界制約行を追加
    for (idx, &(j, coeff, rhs)) in extra.iter().enumerate() {
        rows_coo.push(m + idx);
        cols_coo.push(j);
        vals_coo.push(coeff);
        new_b.push(rhs);
    }

    let new_a = CscMatrix::from_triplets(&rows_coo, &cols_coo, &vals_coo, new_m, n)
        .expect("augment_bounds_to_constraints: CSC construction failed");

    (new_a, new_b)
}


/// QP求解の実装コア（Active Set法）
pub(crate) fn qp_solve_impl(
    problem: &QpProblem,
    warm_start: Option<&QpWarmStart>,
    options: &SolverOptions,
) -> QpResult {
    let n = problem.num_vars;

    // deadline を一度だけ計算してオプションに設定（Simplex 内でも使用）
    let mut opts_with_deadline;
    let effective_opts: &SolverOptions = if let (Some(secs), true) =
        (options.timeout_secs, options.deadline.is_none())
    {
        opts_with_deadline = options.clone();
        opts_with_deadline.deadline =
            Some(Instant::now() + Duration::from_secs_f64(secs));
        &opts_with_deadline
    } else {
        options
    };

    // TimeoutContext: deadline + cancel_flag を一元管理
    let timeout = TimeoutContext::from_options(effective_opts);

    // Q=0 の退化ケース（LP問題）: LP solverに委譲
    if problem.is_zero_q() {
        return solve_as_lp(problem, effective_opts);
    }

    // 並列Active Set: Phase Iで実行可能点を取得後、N本並列でactive_set_loopを実行する。
    // WS0/WS1/WS3 の初期ワーキングセット多様化で有意な並列化を実現する。
    //
    // 注意: warm_start.initial_pointをそのままactive_set_loopに渡すと実行不可能点から
    // スタートする可能性があるため、並列ブロックでは必ずPhase Iで実行可能点を取得する。
    #[cfg(feature = "parallel")]
    {
        if warm_start.is_none() && effective_opts.parallel_runs > 1 {
            use rayon::prelude::*;

            // Phase Iで実行可能点を1回取得
            eprintln!("DBG: parallel Phase1 start, deadline={:?}", effective_opts.deadline);
            let phase1_start = std::time::Instant::now();
            let feasible_x = match find_initial_feasible_point(problem, effective_opts) {
                Phase1Result::Feasible(x) => { eprintln!("DBG: Phase1 Feasible ({:.2}s)", phase1_start.elapsed().as_secs_f64()); x },
                Phase1Result::Infeasible => { eprintln!("DBG: Phase1 Infeasible"); return QpResult::infeasible() },
                Phase1Result::MaxIterations => {
                    eprintln!("DBG: Phase1 MaxIterations");
                    return QpResult::max_iterations(vec![], f64::INFINITY, vec![], 0)
                }
                Phase1Result::Timeout => { eprintln!("DBG: Phase1 Timeout ({:.2}s)", phase1_start.elapsed().as_secs_f64()); return QpResult {
                    status: SolveStatus::Timeout,
                    objective: f64::INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    bound_duals: vec![],
                    active_set: vec![],
                    iterations: 0,
                }},
            };

            // cancel フラグ（他ワーカーが Optimal を見つけたら残りを止める）
            let cancel = Arc::new(AtomicBool::new(false));
            let run_count = effective_opts.parallel_runs;
            eprintln!("DBG: launching {} parallel workers, deadline_remaining={:.2}s", run_count,
                effective_opts.deadline.map_or(-1.0, |d| (d - std::time::Instant::now()).as_secs_f64()));

            // 初期ワーキングセット多様化: WS0（空集合）/ WS1（境界アクティブ）/ WS3（ハッシュ乱択）
            let initial_working_sets = build_initial_working_sets(problem, &feasible_x, run_count);

            let result = initial_working_sets
                .into_par_iter()
                .find_map_any(|ws_indices| {
                    if cancel.load(Ordering::Relaxed) {
                        return None;
                    }
                    // 各ワーカー用の cancel_flag を共有
                    let mut worker_opts = effective_opts.clone();
                    worker_opts.cancel_flag = Some(cancel.clone());
                    let worker_timeout = TimeoutContext::from_options(&worker_opts);
                    eprintln!("DBG: worker start, ws_len={}, should_stop={}, deadline_remaining={:.3}s",
                        ws_indices.len(), worker_timeout.should_stop(),
                        worker_opts.deadline.map_or(-1.0, |d| (d - std::time::Instant::now()).as_secs_f64()));

                    let r = active_set_loop(
                        problem,
                        feasible_x.clone(),
                        WorkingSet::from_indices(ws_indices),
                        &worker_opts,
                        &worker_timeout,
                    );
                    eprintln!("DBG: worker done, status={:?}, iters={}", r.status, r.iterations);
                    if r.status == SolveStatus::Optimal {
                        cancel.store(true, Ordering::Relaxed);
                        Some(r)
                    } else {
                        None
                    }
                });

            if let Some(r) = result {
                return r;
            }
            // 全並列試行が失敗（Timeout / MaxIterations）→ 同じfeasible_xで直列実行
            return active_set_loop(
                problem,
                feasible_x,
                WorkingSet::from_indices(vec![]),
                effective_opts,
                &timeout,
            );
        }
    }

    // Phase I: 初期実行可能点の取得
    // Phase1Result::MaxIterations は数値困難（refactor_failed 等）による早期打切りで
    // 偽陽性の Infeasible を防ぐため QpResult::max_iterations() を返す。
    let initial_x = if let Some(ws) = warm_start {
        if let Some(ref x0) = ws.initial_point {
            if x0.len() == n {
                x0.clone()
            } else {
                match find_initial_feasible_point(problem, effective_opts) {
                    Phase1Result::Feasible(x) => x,
                    Phase1Result::Infeasible => return QpResult::infeasible(),
                    Phase1Result::MaxIterations => {
                        return QpResult::max_iterations(vec![], f64::INFINITY, vec![], 0)
                    }
                    Phase1Result::Timeout => return QpResult {
                        status: SolveStatus::Timeout,
                        objective: f64::INFINITY,
                        solution: vec![],
                        dual_solution: vec![],
                        bound_duals: vec![],
                        active_set: vec![],
                        iterations: 0,
                    },
                }
            }
        } else {
            match find_initial_feasible_point(problem, effective_opts) {
                Phase1Result::Feasible(x) => x,
                Phase1Result::Infeasible => return QpResult::infeasible(),
                Phase1Result::MaxIterations => {
                    return QpResult::max_iterations(vec![], f64::INFINITY, vec![], 0)
                }
                Phase1Result::Timeout => return QpResult {
                    status: SolveStatus::Timeout,
                    objective: f64::INFINITY,
                    solution: vec![],
                    dual_solution: vec![],
                    bound_duals: vec![],
                    active_set: vec![],
                    iterations: 0,
                },
            }
        }
    } else {
        match find_initial_feasible_point(problem, effective_opts) {
            Phase1Result::Feasible(x) => x,
            Phase1Result::Infeasible => return QpResult::infeasible(),
            Phase1Result::MaxIterations => {
                return QpResult::max_iterations(vec![], f64::INFINITY, vec![], 0)
            }
            Phase1Result::Timeout => return QpResult {
                status: SolveStatus::Timeout,
                objective: f64::INFINITY,
                solution: vec![],
                dual_solution: vec![],
                bound_duals: vec![],
                active_set: vec![],
                iterations: 0,
            },
        }
    };

    // Phase II: Active Set メインループ
    // 初期working setは空から始める（等式制約の2不等式エンコード時の線形従属を防ぐため）
    // warm-startの場合は提供されたactive_setを使用するが、線形独立性が保証された集合を前提とする
    let initial_active = if let Some(ws) = warm_start {
        WorkingSet::from_indices(ws.initial_active_set.clone())
    } else {
        WorkingSet::from_indices(vec![])
    };

    active_set_loop(problem, initial_x, initial_active, effective_opts, &timeout)
}

/// 並列Active Set用の初期ワーキングセットを生成する（WS0 / WS1 / WS3）
///
/// - WS0: 空集合（標準的な Active Set の初期状態）
/// - WS1: feasible_x で実際に活性(binding)な全制約（オリジナル + 境界制約）
/// - WS3以降: WS1 からのハッシュベース疑似ランダムサブセット
///
/// # 安全性
/// WS1/WS3 は feasible_x でバインドしている制約のみを含むため、
/// Active Set 初期化として数値的に安全。
#[cfg(feature = "parallel")]
fn build_initial_working_sets(
    problem: &QpProblem,
    feasible_x: &[f64],
    count: usize,
) -> Vec<Vec<usize>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let m = problem.num_constraints;
    let n = feasible_x.len();
    let tol = 1e-6;

    // feasible_x で活性な制約インデックスを収集
    let mut active: Vec<usize> = vec![];

    // オリジナル制約: a_i^T x ≈ b_i なら活性
    for i in 0..m {
        let ai_x: f64 = (0..n)
            .map(|j| get_a_element(&problem.a, i, j) * feasible_x.get(j).copied().unwrap_or(0.0))
            .sum();
        if (ai_x - problem.b[i]).abs() < tol {
            active.push(i);
        }
    }

    // 境界制約 (augment_bounds_to_constraints と同じ順序で aug インデックスを計算)
    // ub: x[j] <= ub → row m + idx (coeff +1.0)
    // lb: -x[j] <= -lb → row m + idx (coeff -1.0)
    let mut aug_idx = m;
    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        let xj = feasible_x.get(j).copied().unwrap_or(0.0);
        if ub.is_finite() {
            if (xj - ub).abs() < tol {
                active.push(aug_idx);
            }
            aug_idx += 1;
        }
        if lb.is_finite() {
            if (xj - lb).abs() < tol {
                active.push(aug_idx);
            }
            aug_idx += 1;
        }
    }

    let mut sets = Vec::with_capacity(count);

    // WS0: 空集合
    sets.push(vec![]);

    // WS1: 全活性制約
    if count >= 2 {
        sets.push(active.clone());
    }

    // WS3以降: 活性制約のランダムサブセット
    for i in 2..count {
        if active.is_empty() {
            sets.push(vec![]);
            continue;
        }
        let k = (active.len() / 2 + 1).min(active.len());
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let seed = hasher.finish();
        let mut ws: Vec<usize> = (0..k)
            .map(|j| {
                let mut h = DefaultHasher::new();
                (seed, j).hash(&mut h);
                active[(h.finish() as usize) % active.len()]
            })
            .collect();
        ws.sort_unstable();
        ws.dedup();
        sets.push(ws);
    }

    sets
}

/// LP ソルバーに委譲してQP結果に変換（Q=0 ケース）
fn solve_as_lp(problem: &QpProblem, options: &SolverOptions) -> QpResult {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    let ct = vec![ConstraintType::Le; m];
    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        ct,
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return QpResult::infeasible(),
    };

    let result = SimplexBackend.solve(&lp, options);
    match result.status {
        SolveStatus::Optimal => {
            let x = result.solution.clone();
            let obj = problem.c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
            // active_set: 活性制約インデックス
            let active: Vec<usize> = (0..m)
                .filter(|&i| {
                    let ax_i: f64 = (0..n)
                        .map(|j| get_a_element(&problem.a, i, j) * x[j])
                        .sum();
                    (ax_i - problem.b[i]).abs() < PIVOT_TOL
                })
                .collect();
            QpResult {
                status: SolveStatus::Optimal,
                objective: obj,
                solution: x,
                dual_solution: result.dual_solution,
                bound_duals: vec![],
                active_set: active,
                iterations: 0,
            }
        }
        SolveStatus::Infeasible => QpResult::infeasible(),
        SolveStatus::Unbounded => QpResult {
            status: SolveStatus::Unbounded,
            objective: f64::NEG_INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
        },
        SolveStatus::MaxIterations => QpResult::max_iterations(vec![], f64::INFINITY, vec![], 0),
        SolveStatus::Timeout => QpResult {
            status: SolveStatus::Timeout,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
        },
        SolveStatus::NumericalError => QpResult {
            status: SolveStatus::NumericalError,
            objective: f64::INFINITY,
            solution: vec![],
            dual_solution: vec![],
            bound_duals: vec![],
            active_set: vec![],
            iterations: 0,
        },
    }
}

/// Phase I LP の結果を表す列挙型
///
/// `SolveStatus::MaxIterations` は数値困難（refactor_failed 等）による早期打切りで、
/// 問題が実行不可能であることを意味しない。この場合は `QpResult::infeasible()` ではなく
/// `QpResult::max_iterations()` を返して偽陽性の Infeasible を防ぐ。
enum Phase1Result {
    /// 初期実行可能点が見つかった
    Feasible(Vec<f64>),
    /// 問題は確実に実行不可能（LP が Infeasible を返した）
    Infeasible,
    /// 数値困難で打ち切り（LP が MaxIterations を返した）; 実行可能性は不明
    MaxIterations,
    /// タイムアウト（Phase I LP が timeout_secs を超過した）
    Timeout,
}

/// Phase I: LP を使って初期実行可能点を求める
///
/// QPS パーサーは等式制約 (Eq) を 2 行の Le (Ax<=b, -Ax<=-b) に展開する。
/// この展開形を全 Le として Phase I LP を解くと、連続ペア行が退化を引き起こし
/// 基底が数値的に特異化 → refactor_failed → MaxIterations となる。
///
/// 対策: 連続ペア行 (b[i+1]=-b[i], A[i+1]=-A[i]) を検出して Eq 制約に再構成し、
/// 半分のサイズ・退化なしの Phase I LP を作る。
///
/// 戻り値: `Phase1Result` で3状態を区別する。MaxIterations は偽陽性 Infeasible を防ぐために
/// 呼び出し元が `QpResult::max_iterations()` を返すべきことを意味する。
fn find_initial_feasible_point(
    problem: &QpProblem,
    options: &SolverOptions,
) -> Phase1Result {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    // 制約なしの場合: 初期点は bounds の lower bound（または 0）
    if m == 0 {
        let x: Vec<f64> = problem
            .bounds
            .iter()
            .map(|&(lb, _ub)| if lb.is_finite() { lb } else { 0.0 })
            .collect();
        return Phase1Result::Feasible(x);
    }

    // 行ごとのスパースエントリを構築（CSC→行アクセスのため）
    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![vec![]; m];
    for j in 0..n {
        let start = problem.a.col_ptr[j];
        let end = problem.a.col_ptr[j + 1];
        for k in start..end {
            let row = problem.a.row_ind[k];
            row_entries[row].push((j, problem.a.values[k]));
        }
    }

    // 連続ペア行 (i, i+1) を検出: b[i+1] ≈ -b[i] かつ A[i+1] ≈ -A[i]
    // QPS パーサーが Eq→2Le に展開する際、常に連続ペアを生成する。
    let mut is_eq_first = vec![false; m];   // row i: Eq 制約の代表行
    let mut is_eq_second = vec![false; m];  // row i+1: スキップ（Eq に統合済み）
    let mut i = 0;
    while i + 1 < m {
        let b_i = problem.b[i];
        let b_j = problem.b[i + 1];
        let tol = 1e-10 * (1.0 + b_i.abs());
        if (b_i + b_j).abs() < tol && rows_are_negation(&row_entries[i], &row_entries[i + 1]) {
            is_eq_first[i] = true;
            is_eq_second[i + 1] = true;
            i += 2;
        } else {
            i += 1;
        }
    }

    // 選択行インデックス（Eq 第2行をスキップ）と制約タイプを構築
    let selected: Vec<usize> = (0..m).filter(|&r| !is_eq_second[r]).collect();
    let new_m = selected.len();
    let new_b: Vec<f64> = selected.iter().map(|&r| problem.b[r]).collect();
    let new_ct: Vec<ConstraintType> = selected
        .iter()
        .map(|&r| if is_eq_first[r] { ConstraintType::Eq } else { ConstraintType::Le })
        .collect();

    // 新 A 行列を CSC 形式で構築（選択行のみ）
    let row_remap: Vec<usize> = {
        let mut remap = vec![usize::MAX; m];
        for (new_r, &orig_r) in selected.iter().enumerate() {
            remap[orig_r] = new_r;
        }
        remap
    };
    let mut trip_rows: Vec<usize> = Vec::new();
    let mut trip_cols: Vec<usize> = Vec::new();
    let mut trip_vals: Vec<f64> = Vec::new();
    for j in 0..n {
        let start = problem.a.col_ptr[j];
        let end = problem.a.col_ptr[j + 1];
        for k in start..end {
            let orig_row = problem.a.row_ind[k];
            let new_row = row_remap[orig_row];
            if new_row != usize::MAX {
                trip_rows.push(new_row);
                trip_cols.push(j);
                trip_vals.push(problem.a.values[k]);
            }
        }
    }
    let new_a = match CscMatrix::from_triplets(&trip_rows, &trip_cols, &trip_vals, new_m, n) {
        Ok(a) => a,
        Err(_) => return Phase1Result::MaxIterations,
    };

    // CSC構築完了後、deadline到達なら即座にTimeoutを返す（防御的チェック）
    if options.deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return Phase1Result::Timeout;
    }

    let lp = match LpProblem::new_general(
        vec![0.0f64; n],
        new_a,
        new_b,
        new_ct,
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return Phase1Result::MaxIterations,
    };

    // Phase I LP: まず presolve 無効で試行
    // LP presolve は等式制約のある QP で初期実行可能点を誤った点に誘導することがある（DUALC2確認済み）。
    // 等式制約の大規模系（QBORE3D, QBRANDY 等）は presolve 無効で Infeasible になるため、
    // その場合のみ presolve 有効でフォールバック再試行する。
    let mut phase1_opts = options.clone();
    phase1_opts.presolve = false;

    // MaxIterations が返った場合: refactor_failed など数値困難による早期打切り。
    // 問題が実行不可能であることを意味しない → Phase1Result::MaxIterations で返す（偽陽性防止）。
    let mut had_max_iterations = false;

    let result = SimplexBackend.solve(&lp, &phase1_opts);
    if result.status == SolveStatus::Optimal {
        return Phase1Result::Feasible(result.solution);
    }
    if result.status == SolveStatus::Timeout {
        return Phase1Result::Timeout;
    }
    if result.status == SolveStatus::MaxIterations {
        had_max_iterations = true;
    }

    // presolve 無効で失敗 → presolve 有効でフォールバック再試行
    if options.presolve {
        let result2 = SimplexBackend.solve(&lp, options);
        if result2.status == SolveStatus::Optimal {
            return Phase1Result::Feasible(result2.solution);
        }
        if result2.status == SolveStatus::Timeout {
            return Phase1Result::Timeout;
        }
        if result2.status == SolveStatus::MaxIterations {
            had_max_iterations = true;
        }
    }

    if had_max_iterations {
        // 数値困難で実行可能点を見つけられなかった。実行不可能と断定しない。
        Phase1Result::MaxIterations
    } else {
        Phase1Result::Infeasible
    }
}

/// 2 行のスパースエントリが互いに符号反転関係かを判定する
/// （同じ非零位置で val_i ≈ -val_j）
fn rows_are_negation(
    row_i: &[(usize, f64)],
    row_j: &[(usize, f64)],
) -> bool {
    if row_i.len() != row_j.len() {
        return false;
    }
    for ((ci, vi), (cj, vj)) in row_i.iter().zip(row_j.iter()) {
        if ci != cj {
            return false;
        }
        let tol = 1e-10 * (1.0 + vi.abs());
        if (vi + vj).abs() > tol {
            return false;
        }
    }
    true
}


/// Active Set メインループ
fn active_set_loop(
    problem: &QpProblem,
    mut x: Vec<f64>,
    mut working_set: WorkingSet,
    options: &SolverOptions,
    timeout: &TimeoutContext,
) -> QpResult {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    let max_iter = options.max_iterations.unwrap_or(100 * (n + m) + 1000);

    // NC-BOUND1修正: 変数境界を明示的な制約行に変換する。
    // これにより compute_step_size がブロッキング境界を working_set に追加できる。
    let (aug_a, aug_b) = augment_bounds_to_constraints(&problem.a, &problem.b, &problem.bounds);

    for iter in 0..max_iter {
        if iter < 3 { eprintln!("DBG: active_set_loop iter={}, n={}, ws_len={}, should_stop={}", iter, n, working_set.len(), timeout.should_stop()); }
        // タイムアウト / キャンセルチェック
        if timeout.should_stop() {
            let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
            return QpResult {
                status: SolveStatus::Timeout,
                objective: obj,
                solution: x,
                dual_solution: vec![0.0; m],
                bound_duals: vec![0.0; aug_b.len() - m],
                active_set: working_set.indices().to_vec(),
                iterations: iter,
            };
        }

        // 勾配 grad = Qx + c を計算
        let grad = kkt::compute_gradient(&problem.q, &x, &problem.c);

        // KKTシステムを構築して解く (aug_a を使用)
        let a_active = match extract_active_rows(&aug_a, working_set.indices()) {
            Ok(a) => a,
            Err(_) => {
                let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                return QpResult::max_iterations(x, obj, working_set.indices().to_vec(), iter);
            }
        };

        let (d, lambda) = if working_set.is_empty() {
            // 活性制約なし: 制約なし最適化方向
            if timeout.should_stop() {
                let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                return QpResult {
                    status: SolveStatus::Timeout,
                    objective: obj,
                    solution: x,
                    dual_solution: vec![0.0; m],
                    bound_duals: vec![0.0; aug_b.len() - m],
                    active_set: working_set.indices().to_vec(),
                    iterations: iter,
                };
            }
            match solve_unconstrained_direction(&problem.q, &grad) {
                Ok(d) => (d, vec![]),
                Err(_) => {
                    // Q が特異: 停留点として扱う
                    (vec![0.0; n], vec![])
                }
            }
        } else {
            if timeout.should_stop() {
                let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                return QpResult {
                    status: SolveStatus::Timeout,
                    objective: obj,
                    solution: x,
                    dual_solution: vec![0.0; m],
                    bound_duals: vec![0.0; aug_b.len() - m],
                    active_set: working_set.indices().to_vec(),
                    iterations: iter,
                };
            }
            let kkt_solver = match KktSolver::new(&problem.q, &a_active) {
                Ok(s) => s,
                Err(_) => {
                    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                    return QpResult::max_iterations(x, obj, working_set.indices().to_vec(), iter);
                }
            };
            match kkt_solver.solve(&grad) {
                Ok(result) => result,
                Err(_) => {
                    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                    return QpResult::max_iterations(x, obj, working_set.indices().to_vec(), iter);
                }
            }
        };

        let d_norm: f64 = d.iter().map(|&di| di * di).sum::<f64>().sqrt();

        if d_norm < PIVOT_TOL {
            // d ≈ 0: KKT条件確認
            if lambda.is_empty() {
                // 制約なし最適: 勾配が小さければ最適
                if grad.iter().map(|&g| g * g).sum::<f64>().sqrt() < PIVOT_TOL {
                    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                    return QpResult {
                        status: SolveStatus::Optimal,
                        objective: obj,
                        solution: x,
                        dual_solution: vec![0.0; m],
                        bound_duals: vec![0.0; aug_b.len() - m],
                        active_set: working_set.indices().to_vec(),
                        iterations: iter + 1,
                    };
                }
            }

            // 最小のラグランジュ乗数を確認
            let min_lambda_val = lambda.iter().cloned().fold(f64::INFINITY, f64::min);
            if min_lambda_val >= -PIVOT_TOL {
                // KKT条件満足: 最適解
                let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                // full_dual: aug_b長（元の制約m + 境界制約数）の双対値ベクトル
                let mut full_dual = vec![0.0; aug_b.len()];
                for (k, &ci) in working_set.indices().iter().enumerate() {
                    full_dual[ci] = lambda[k];
                }
                // dual_solution[0..m]: 元の制約の双対値（公開API契約: 長さm）
                // bound_duals[m..]: 変数境界の双対値
                let (orig_dual, bounds_dual) = full_dual.split_at(m);
                return QpResult {
                    status: SolveStatus::Optimal,
                    objective: obj,
                    solution: x,
                    dual_solution: orig_dual.to_vec(),
                    bound_duals: bounds_dual.to_vec(),
                    active_set: working_set.indices().to_vec(),
                    iterations: iter + 1,
                };
            }

            // 最小λを持つ制約を除去（Bland則: 複数ある場合は最小インデックスを選択）
            let min_lambda_idx = lambda
                .iter()
                .enumerate()
                .filter(|(_, &lv)| lv < -PIVOT_TOL)
                .min_by(|a, b| {
                    // Bland則: 活性集合内の制約インデックスが小さい方を選ぶ
                    let idx_a = working_set.get(a.0).unwrap_or(usize::MAX);
                    let idx_b = working_set.get(b.0).unwrap_or(usize::MAX);
                    idx_a.cmp(&idx_b)
                })
                .map(|(i, _)| i);

            if let Some(k) = min_lambda_idx {
                if let Some(constraint_idx) = working_set.get(k) {
                    working_set.remove(constraint_idx);
                }
            }
        } else {
            // d ≠ 0: ステップ幅計算 (aug_a, aug_b を使用)
            let alpha = compute_step_size(&aug_a, &aug_b, &x, &d, &working_set, timeout);

            // タイムアウト発生時は即座に返す
            if alpha.timed_out {
                let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
                return QpResult {
                    status: SolveStatus::Timeout,
                    objective: obj,
                    solution: x,
                    dual_solution: vec![0.0; m],
                    bound_duals: vec![0.0; aug_b.len() - m],
                    active_set: working_set.indices().to_vec(),
                    iterations: iter,
                };
            }

            // x を更新
            for i in 0..n {
                x[i] += alpha.step * d[i];
            }

            // α < 1: ブロッキング制約を活性集合に追加
            if alpha.step < 1.0 - ZERO_TOL {
                if let Some(blocking) = alpha.blocking_constraint {
                    working_set.add(blocking);
                }
            }
        }
    }

    // 反復上限超過
    let obj = kkt::compute_objective(&problem.q, &x, &problem.c);
    QpResult::max_iterations(x, obj, working_set.indices().to_vec(), max_iter)
}

/// 制約なしの探索方向: Q * d = -grad を解く（対角Q高速パス）
fn solve_unconstrained_direction(
    q: &CscMatrix,
    grad: &[f64],
) -> Result<Vec<f64>, ()> {
    let n = grad.len();
    let mut d = vec![0.0f64; n];

    // 対角行列の場合: d[i] = -grad[i] / q[i][i]
    let mut is_diag = true;
    for col in 0..n {
        let start = q.col_ptr[col];
        let end = q.col_ptr[col + 1];
        for k in start..end {
            if q.row_ind[k] != col {
                is_diag = false;
                break;
            }
        }
        if !is_diag {
            break;
        }
    }

    if is_diag {
        for i in 0..n {
            let q_ii = get_diagonal(q, i);
            if q_ii.abs() < 1e-12 {
                return Err(()); // 特異
            }
            d[i] = -grad[i] / q_ii;
        }
        return Ok(d);
    }

    // 一般PSDの場合: LU分解で解く
    // 一時的にQをKKT行列として使用（活性制約なし）
    let a_empty = CscMatrix::new(0, n);
    match KktSolver::new(q, &a_empty) {
        Ok(solver) => match solver.solve(grad) {
            Ok((d_result, _)) => Ok(d_result),
            Err(_) => Err(()),
        },
        Err(_) => Err(()),
    }
}

/// ステップ幅計算の結果
struct StepResult {
    step: f64,
    blocking_constraint: Option<usize>,
    timed_out: bool,
}

/// ステップ幅 α* を計算する（ライン探索）
///
/// 非活性制約（境界制約を含む）が活性化しないよう最大ステップ幅を計算する。
/// aug_a / aug_b は変数境界を含む拡張制約行列を指定する（NC-BOUND1修正）。
fn compute_step_size(
    aug_a: &CscMatrix,
    aug_b: &[f64],
    x: &[f64],
    d: &[f64],
    working_set: &WorkingSet,
    timeout: &TimeoutContext,
) -> StepResult {
    let mut alpha_crit = 1.0f64;
    let mut blocking: Option<usize> = None;

    for (i, &b_i) in aug_b.iter().enumerate() {
        // 1000行ごとにtimeoutチェック
        if i % 1000 == 0 && timeout.should_stop() {
            return StepResult {
                step: alpha_crit.max(0.0),
                blocking_constraint: blocking,
                timed_out: true,
            };
        }

        // 活性制約はスキップ
        if working_set.contains(i) {
            continue;
        }

        // a_i^T d を計算
        let ai_d = dot_row_a(aug_a, i, d);
        if ai_d <= ZERO_TOL {
            continue; // この制約はブロックしない
        }

        // a_i^T x を計算
        let ai_x = dot_row_a(aug_a, i, x);
        let slack = b_i - ai_x;

        // α ≤ slack / (a_i^T d)
        let alpha_i = slack / ai_d;
        if alpha_i < alpha_crit {
            alpha_crit = alpha_i;
            blocking = Some(i); // Bland則: 最小インデックスを採用
        } else if (alpha_i - alpha_crit).abs() < ZERO_TOL {
            // タイブレーク: 最小インデックスを採用（Bland則）
            if let Some(prev) = blocking {
                if i < prev {
                    blocking = Some(i);
                }
            }
        }
    }

    StepResult {
        step: alpha_crit.max(0.0),
        blocking_constraint: blocking,
        timed_out: false,
    }
}

/// 行列 A の第 row 行と x のドット積を計算する
fn dot_row_a(a: &CscMatrix, row: usize, x: &[f64]) -> f64 {
    let mut result = 0.0f64;
    for (col, &xj) in x.iter().enumerate().take(a.ncols) {
        let start = a.col_ptr[col];
        let end = a.col_ptr[col + 1];
        for k in start..end {
            if a.row_ind[k] == row {
                result += a.values[k] * xj;
                break;
            }
        }
    }
    result
}

/// 行列 A の (row, col) 要素を返す
fn get_a_element(a: &CscMatrix, row: usize, col: usize) -> f64 {
    let start = a.col_ptr[col];
    let end = a.col_ptr[col + 1];
    for k in start..end {
        if a.row_ind[k] == row {
            return a.values[k];
        }
    }
    0.0
}

/// 対角要素 Q[i,i] を返す
fn get_diagonal(q: &CscMatrix, i: usize) -> f64 {
    let start = q.col_ptr[i];
    let end = q.col_ptr[i + 1];
    for k in start..end {
        if q.row_ind[k] == i {
            return q.values[k];
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use crate::options::SolverOptions;
    use crate::problem::SolveStatus;
    use crate::qp::{solve_qp_with_options, QpProblem};
    use crate::sparse::CscMatrix;

    /// compute_step_size 内 timeout 動作確認
    ///
    /// n=500, m=100 の合成 QP 問題を timeout_secs=0.001 (1ms) で解かせ、
    /// SolveStatus::Timeout が返ることを確認する。
    /// Phase I LP (n=500, m=100) の処理時間が 1ms を超えるため、
    /// active_set_loop または compute_step_size の timeout チェックで停止する。
    #[test]
    fn test_compute_step_size_timeout() {
        let n = 500usize;
        let m = 100usize;

        // Q = 2*I (n×n 対角、正定値)
        let q_rows: Vec<usize> = (0..n).collect();
        let q_cols: Vec<usize> = (0..n).collect();
        let q_vals: Vec<f64> = vec![2.0; n];
        let q = CscMatrix::from_triplets(&q_rows, &q_cols, &q_vals, n, n).unwrap();

        // c = ones (無制約最小点を x = -0.5 に設定)
        let c = vec![1.0; n];

        // A: 行 i に変数 i*5 の係数 1.0 を配置 (m=100 行, n=500 列)
        let a_rows: Vec<usize> = (0..m).collect();
        let a_cols: Vec<usize> = (0..m).map(|i| i * 5).collect();
        let a_vals: Vec<f64> = vec![1.0; m];
        let a = CscMatrix::from_triplets(&a_rows, &a_cols, &a_vals, m, n).unwrap();

        // b = 1.0: x[i*5] <= 1.0 (x=-0.5 は実行可能)
        let b = vec![1.0; m];

        let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); n];
        let problem = QpProblem::new(q, c, a, b, bounds).unwrap();

        let mut opts = SolverOptions::default();
        opts.timeout_secs = Some(0.001); // 1ms タイムアウト

        let result = solve_qp_with_options(&problem, &opts);
        assert_eq!(
            result.status,
            SolveStatus::Timeout,
            "test_compute_step_size_timeout: expected Timeout, got {:?}",
            result.status
        );
    }
}
