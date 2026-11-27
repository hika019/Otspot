//! LP 求解入口的独立連結成分分割 (independent component splitting)。
//!
//! 変数–制約の所属関係を作る変数ハイパーグラフが非連結のとき、LP を成分ごとの
//! 副 LP に分解し、逐次求解して結果を結合する。実行可能集合は成分の直積なので
//! status / 目的値の結合は数学的に厳密 ([`join_components`] の結合規則表参照)。
//! 副問題は既存入口 (presolve を含むフルパイプライン) を再帰で通り、副問題は
//! 連結なので次の検出で必ず単一成分となり再帰は自然に停止する。
//!
//! コスト: 検出 O(nnz + m + n)、副問題再構築 O(nnz)。単一成分のときも検出の
//! O(nnz) は発生し、それ以外は従来経路と同一 (分割 overhead は発火時にも
//! 再構築 O(nnz) ×分割前後の分行けにかかる)。利得が薄い「最大成分がほぼ n」
//! 型の自明な不連結も常に分割する (発火条件は成分数 >=2 のみ)。
//!
//! 打ち切り: 各副 solve にはグローバル deadline (options.deadline /
//! timeout_secs から実体化) をそのまま渡す。成分間に残り時間を再配分せず、
//! 副 solve 内で deadline に到達すれば Timeout になる。求解は初版は逐次
//! (rayon は意図的に使わない)。
//!
//! 復元精度: 副 A 行列は `CscMatrix::from_triplets` で再構築するため
//! |v| <= DROP_TOL (1e-15) の構造的非ゼロが落ちる。パーサ由来入力は同じ
//! 経路で構築されるため実データでは差異不出。

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveStats, SolveStatus, SolverResult, TimingBreakdown};
use otspot_num::sparse::CscMatrix;

/// 1 成分分 (列と、その行 support が属する行)。
struct Component {
    cols: Vec<usize>,
    rows: Vec<usize>,
}

/// 副問題。行を持たない成分は箱制約だけの自明 LP なので閉形式で解く
/// ([`solve_box_component`])。0 行 LP を simplex に渡す経路を作らない。
enum SubLp {
    Box {
        cols: Vec<usize>,
    },
    Lp {
        problem: LpProblem,
        cols: Vec<usize>,
        rows: Vec<usize>,
    },
}

impl SubLp {
    fn cols(&self) -> &[usize] {
        match self {
            SubLp::Box { cols } | SubLp::Lp { cols, .. } => cols,
        }
    }
    fn rows(&self) -> &[usize] {
        match self {
            SubLp::Box { .. } => &[],
            SubLp::Lp { rows, .. } => rows,
        }
    }
    fn num_vars(&self) -> usize {
        self.cols().len()
    }
}

/// 分割済み入口。分割不发火 (単一成分・ガード発動・副構築失敗) は
/// `simplex::solve_with` をそのまま通すので従来挙動と完全一致。
/// `honor_user_no_presolve` は `solve_lp_with` (LP 直入口) 専用フラグ:
/// sensitivity 規約が `presolve=false` を標準形 basis 取得の合図に使うため、
/// 直入口では presolve=false の要求があれば分割しない。転送入口
/// (`solve_lp_forwarded_from_qp`) は false を渡す: 内部パイプライン
/// (`qp::lp_dispatch`) は自前の presolve 後や backend 呼び出しで常に
/// `presolve` を落とするため、転送経路でフラグを見ると全 QPS LP が
/// 誤って非発火になる (lp_dispatch の no_presolve 変換は basis 契約でなく
/// 段分工順の内部表現。実測 fome21 で bail=presolve_off を確認済)。
pub(crate) fn solve_with_split(
    problem: &LpProblem,
    options: &SolverOptions,
    honor_user_no_presolve: bool,
) -> SolverResult {
    let materialized = options.materialize_deadline();
    let options = materialized.as_ref().unwrap_or(options);
    let Some(components) = split_plan(problem, options, honor_user_no_presolve) else {
        return crate::simplex::solve_with(problem, options);
    };
    let mut col_local = vec![usize::MAX; problem.num_vars];
    let mut row_local = vec![usize::MAX; problem.num_constraints];
    let mut subs: Vec<SubLp> = Vec::with_capacity(components.len());
    for comp in &components {
        let Some(sub) = build_sub_lp(problem, comp, &mut col_local, &mut row_local) else {
            // 副問題構築に失敗したら全体求解へ退避 (分割は最適化であり意味を変えない)。
            return crate::simplex::solve_with(problem, options);
        };
        subs.push(sub);
    }
    let k = subs.len();
    let mut results: Vec<SolverResult> = Vec::with_capacity(k);
    for sub in &subs {
        // 残時間ゼロなら以降の副 solve は開始しない (実クロック比較のみで mint)。
        // 未完成成分があるまま Optimal を主張してはいけないため、ここで
        // 集まった結果を join せず Timeout を返す。
        if options.external_stop_requested() {
            return no_claim_result(SolveStatus::Timeout, f64::INFINITY, k, &results);
        }
        let result = match sub {
            SubLp::Box { cols } => solve_box_component(problem, cols),
            SubLp::Lp {
                problem: sub_lp, ..
            } => solve_with_split(sub_lp, options, false),
        };
        let decided_infeasible = result.status == SolveStatus::Infeasible;
        results.push(result);
        // Infeasible は他成分の結果を待たず全体 status を確定させる (直積の空性)。
        if decided_infeasible {
            break;
        }
    }
    join_components(problem, k, &subs, results)
}

/// QP 転送経路 (`qp::lp_dispatch::solve_as_lp`) 用の分割入口。
///
/// 単一成分・ガード発動・副構築失敗は `None` を返し、呼び出し側は通常の
/// 非分割経路 (従来挙動と完全一致) へ進む。成分は `recurse` (呼び出し側の
/// `solve_as_lp` そのもの) で解くため、成分ごとに同関数の分割検出・
/// presolve・LP/IPM ルート選択が再適用される (副問題は連結なので検出は
/// 2 回目以降不发火で、再帰は自然に停止する)。
///
/// `obj_offset` 規約: 副問題の offset は 0 (`LpProblem::new_general` は
/// offset を受け付けない) で成分目的値は raw。結合後に元 `QpProblem.obj_offset`
/// を [`crate::qp::lp_dispatch::add_qp_obj_offset`] で 1 回だけ加算する
/// (lp.rs 直入口 = `solve_with_split` の内側で split し外側で 1 回加算、と
/// 同一規約)。ルートスタンプは転送経路の `LpForwardedFromQp` に揃える
/// (結合結果は新規構築なので `join_stats` が route を引き継がないため)。
pub(crate) fn solve_as_lp_via_split(
    problem: &crate::qp::QpProblem,
    options: &SolverOptions,
    lp: &LpProblem,
    recurse: fn(&crate::qp::QpProblem, &SolverOptions) -> SolverResult,
) -> Option<SolverResult> {
    let components = split_plan(lp, options, true)?;
    let mut col_local = vec![usize::MAX; lp.num_vars];
    let mut row_local = vec![usize::MAX; lp.num_constraints];
    let mut subs: Vec<SubLp> = Vec::with_capacity(components.len());
    for comp in &components {
        // 構築失敗は全体求解 (非分割経路) へ退避 — 分割は最適化であり意味を変えない。
        subs.push(build_sub_lp(lp, comp, &mut col_local, &mut row_local)?);
    }
    let k = subs.len();
    let mut results: Vec<SolverResult> = Vec::with_capacity(k);
    for sub in &subs {
        if options.external_stop_requested() {
            return Some(no_claim_result(
                SolveStatus::Timeout,
                f64::INFINITY,
                k,
                &results,
            ));
        }
        let result = match sub {
            SubLp::Box { cols } => solve_box_component(lp, cols),
            SubLp::Lp {
                problem: sub_lp, ..
            } => {
                let sub_qp = sub_lp_as_zero_q(sub_lp)?;
                recurse(&sub_qp, options)
            }
        };
        let decided_infeasible = result.status == SolveStatus::Infeasible;
        results.push(result);
        if decided_infeasible {
            break;
        }
    }
    let mut joined = join_components(lp, k, &subs, results);
    crate::qp::lp_dispatch::add_qp_obj_offset(&mut joined, problem.obj_offset);
    joined.stats.route = crate::problem::SolveRoute::LpForwardedFromQp;
    Some(joined)
}

/// 副 LpProblem を Q=0 の QpProblem へ変換する (`lp_dispatch` の Q 変換と同型)。
/// 変換失敗 (次元不整合 — 構築済み副問題では起きない想定) は None で全体退避。
fn sub_lp_as_zero_q(sub_lp: &LpProblem) -> Option<crate::qp::QpProblem> {
    let q = CscMatrix::new(sub_lp.num_vars, sub_lp.num_vars);
    crate::qp::QpProblem::new(
        q,
        sub_lp.c.clone(),
        (*sub_lp.a).clone(),
        sub_lp.b.clone(),
        sub_lp.bounds.clone(),
        sub_lp.constraint_types.clone(),
    )
    .ok()
}

/// 変数ハイパーグラフの連結成分を union-find で検出する (O(nnz + m + n))。
/// 分割すべき成分 >=2 があれば Some、単一成分・ガード発動は None。
fn split_plan(
    problem: &LpProblem,
    options: &SolverOptions,
    honor_user_no_presolve: bool,
) -> Option<Vec<Component>> {
    // warm start / 初期 basis 系ガード: 行列表現 (列の再ランク・行の副問題割当) が
    // 変わるため、標準形 basis を渡す・要求するオプションでは分割しない。
    if options.warm_start.is_some() || options.warm_start_lp.is_some() {
        return None;
    }
    if options.recover_warm_start_basis || options.max_iters.is_some() {
        return None;
    }
    // presolve=false の意味は入口で異なる。直入口 (`solve_lp_with`) では
    // sensitivity が標準形 basis を結果から受け取るための文書化済み規約
    // (`sensitivity` モジュール doc / `compute_sensitivity` の basis 欠落条件)
    // なので分割しない。転送入口は内部パイプライン (`qp::lp_dispatch`) が
    // 段分工順上常に presolve を落とすため、ここではフラグを見ない
    // (`honor_user_no_presolve` の doc 参照)。
    if honor_user_no_presolve && !options.presolve {
        return None;
    }
    let n = problem.num_vars;
    if n == 0 {
        return None;
    }
    let mut uf = UnionFind::new(n);
    // 各行の最初の support 列 (usize::MAX = support 空)。同一行の support 同士を union。
    let mut row_anchor = vec![usize::MAX; problem.num_constraints];
    let a = problem.a.as_ref();
    for j in 0..n {
        for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
            let i = a.row_ind()[k];
            if row_anchor[i] == usize::MAX {
                row_anchor[i] = j;
            } else {
                uf.union(j, row_anchor[i]);
            }
        }
    }
    // support 空の行 (自明 0<=b / 矛盾行) が 1 行でもあれば分割しない: 行を成分に
    // 割り当てられず副問題から脱落し、矛盾行を見逃す誤 Optimal を招くため。
    // 既存 presolve / simplex 経路に委ねる。
    if row_anchor.contains(&usize::MAX) {
        return None;
    }
    let mut components: Vec<Component> = Vec::new();
    let mut comp_of_root = vec![usize::MAX; n];
    for j in 0..n {
        let root = uf.find(j);
        if comp_of_root[root] == usize::MAX {
            comp_of_root[root] = components.len();
            components.push(Component {
                cols: Vec::new(),
                rows: Vec::new(),
            });
        }
        components[comp_of_root[root]].cols.push(j);
    }
    if components.len() < 2 {
        return None;
    }
    for i in 0..problem.num_constraints {
        let root = uf.find(row_anchor[i]);
        components[comp_of_root[root]].rows.push(i);
    }
    Some(components)
}

/// 成分から副 LpProblem を構築する。行を持たない成分は箱成分として構築を省略。
/// scratch (col_local / row_local) は呼び出し側で再利用するため、この関数内で
/// 該当成分分だけ設定し最後に MAX へ戻す (全成分合計 O(nnz + m + n))。
fn build_sub_lp(
    problem: &LpProblem,
    comp: &Component,
    col_local: &mut [usize],
    row_local: &mut [usize],
) -> Option<SubLp> {
    if comp.rows.is_empty() {
        return Some(SubLp::Box {
            cols: comp.cols.clone(),
        });
    }
    for (local, &global) in comp.cols.iter().enumerate() {
        col_local[global] = local;
    }
    for (local, &global) in comp.rows.iter().enumerate() {
        row_local[global] = local;
    }
    let a = problem.a.as_ref();
    let mut t_rows = Vec::with_capacity(comp.cols.len());
    let mut t_cols = Vec::with_capacity(comp.cols.len());
    let mut t_vals = Vec::with_capacity(comp.cols.len());
    for &j in &comp.cols {
        let jl = col_local[j];
        for k in a.col_ptr()[j]..a.col_ptr()[j + 1] {
            t_rows.push(row_local[a.row_ind()[k]]);
            t_cols.push(jl);
            t_vals.push(a.values()[k]);
        }
    }
    // 割当済み scratch を戻す (構築成否の両経路で実行する)。
    let built = build_sub_problem(problem, comp, &t_rows, &t_cols, &t_vals);
    for &j in &comp.cols {
        col_local[j] = usize::MAX;
    }
    for &i in &comp.rows {
        row_local[i] = usize::MAX;
    }
    Some(SubLp::Lp {
        problem: built?,
        cols: comp.cols.clone(),
        rows: comp.rows.clone(),
    })
}

fn build_sub_problem(
    problem: &LpProblem,
    comp: &Component,
    t_rows: &[usize],
    t_cols: &[usize],
    t_vals: &[f64],
) -> Option<LpProblem> {
    let a_sub =
        CscMatrix::from_triplets(t_rows, t_cols, t_vals, comp.rows.len(), comp.cols.len()).ok()?;
    let c_sub: Vec<f64> = comp.cols.iter().map(|&j| problem.c[j]).collect();
    let b_sub: Vec<f64> = comp.rows.iter().map(|&i| problem.b[i]).collect();
    let types_sub: Vec<_> = comp
        .rows
        .iter()
        .map(|&i| problem.constraint_types[i])
        .collect();
    let bounds_sub: Vec<(f64, f64)> = comp.cols.iter().map(|&j| problem.bounds[j]).collect();
    let name = problem
        .name
        .as_ref()
        .map(|base| format!("{base}#comp{}", comp.cols[0]));
    LpProblem::new_general(c_sub, a_sub, b_sub, types_sub, bounds_sub, name).ok()
}

/// 箱成分 (行なし) の閉形式求解: min cᵀx, x∈[lb,ub] は座標ごとに
/// `c>0 → x=lb`, `c<0 → x=ub`, `c=0 → 適当な箱内点`。行がないので
/// dual / slack は空、reduced cost は c そのもの (m=0 simplex と一致)。
fn solve_box_component(problem: &LpProblem, cols: &[usize]) -> SolverResult {
    let mut x = Vec::with_capacity(cols.len());
    let mut rc = Vec::with_capacity(cols.len());
    let mut objective = 0.0;
    for &j in cols {
        let c = problem.c[j];
        let (lb, ub) = problem.bounds[j];
        if c > 0.0 {
            if !lb.is_finite() {
                return box_unbounded_result();
            }
            x.push(lb);
            objective += c * lb;
        } else if c < 0.0 {
            if !ub.is_finite() {
                return box_unbounded_result();
            }
            x.push(ub);
            objective += c * ub;
        } else {
            let pick = if lb.is_finite() {
                lb
            } else if ub.is_finite() {
                ub
            } else {
                0.0
            };
            x.push(pick);
        }
        rc.push(c);
    }
    SolverResult {
        status: SolveStatus::Optimal,
        objective,
        solution: x,
        reduced_costs: rc,
        ..Default::default()
    }
}

fn box_unbounded_result() -> SolverResult {
    SolverResult {
        status: SolveStatus::Unbounded,
        objective: f64::NEG_INFINITY,
        ..Default::default()
    }
}

/// 成分 status の結合規則 (表はモジュール doc 兼 報告用):
///
/// | 成分の status 構成               | 全体 status                        |
/// |--------------------------------|------------------------------------|
/// | どれか Infeasible               | Infeasible (obj=+inf, 解ベクトル空) |
/// | どれか Unbounded かつ他は全て Optimal/Unbounded | Unbounded (obj=-inf) |
/// | 上記 Unbounded 条件を満たさない Unbounded + 非厳密成分 | 非厳密成分の最大 rank (無ければ Stalled) |
/// | 非厳密 status が 1 つ以上        | rank 最大 (同順位は先頭成分)        |
/// | 全て Optimal                    | Optimal, obj=Σ, x/dual/slack/rc を対応で散布 |
/// | Optimal + SuboptimalSolution の混成のみ | SuboptimalSolution (公開入口の incumbent 検証が元問題で再検証) |
///
/// rank (高いほど前面に出す): 型系 (LocallyOptimal/NonConvex/NonconvexLocal/
/// NonconvexGlobal/NotSupported) > NumericalError > ResourceExhausted >
/// Timeout > Stalled > MaxIterations > FeasiblePoint。Optimal 主張は全成分
/// Optimal のときのみ。ray / Farkas 証書ペイロードと warm start basis は
/// 副問題空間の座標なので結合結果では常に破棄 (flag も立たせない)。
fn join_components(
    problem: &LpProblem,
    k: usize,
    subs: &[SubLp],
    mut results: Vec<SolverResult>,
) -> SolverResult {
    for (sub, r) in subs.iter().zip(results.iter_mut()) {
        // 点・目的値が主張に合わない Optimal は SuboptimalSolution へ正規化
        // (公開入口の incumbent ゲートと同じ向きに保守化する)。
        if r.status == SolveStatus::Optimal
            && (!r.is_finite_candidate() || r.solution.len() != sub.num_vars())
        {
            r.status = SolveStatus::SuboptimalSolution;
        }
    }
    if results.iter().any(|r| r.status == SolveStatus::Infeasible) {
        return no_claim_result(SolveStatus::Infeasible, f64::INFINITY, k, &results);
    }
    if results.iter().any(|r| r.status == SolveStatus::Unbounded) {
        let feasible_all = results
            .iter()
            .all(|r| matches!(r.status, SolveStatus::Optimal | SolveStatus::Unbounded));
        if feasible_all {
            return no_claim_result(SolveStatus::Unbounded, f64::NEG_INFINITY, k, &results);
        }
        if has_unclean(&results) {
            return worst_unclean_result(k, &results);
        }
        return no_claim_result(SolveStatus::Stalled, f64::INFINITY, k, &results);
    }
    if has_unclean(&results) {
        return worst_unclean_result(k, &results);
    }
    if results.iter().all(|r| r.status == SolveStatus::Optimal) {
        return point_join(problem, k, subs, &results, SolveStatus::Optimal);
    }
    if results.iter().all(|r| {
        matches!(
            r.status,
            SolveStatus::Optimal | SolveStatus::SuboptimalSolution
        )
    }) {
        return point_join(problem, k, subs, &results, SolveStatus::SuboptimalSolution);
    }
    worst_unclean_result(k, &results)
}

fn point_join(
    problem: &LpProblem,
    k: usize,
    subs: &[SubLp],
    results: &[SolverResult],
    status: SolveStatus,
) -> SolverResult {
    let mut x = vec![0.0; problem.num_vars];
    let mut rc = vec![0.0; problem.num_vars];
    let mut duals = vec![0.0; problem.num_constraints];
    let mut slack = vec![0.0; problem.num_constraints];
    let mut point_ok = true;
    let mut rc_ok = true;
    let mut dual_ok = true;
    let mut slack_ok = true;
    let mut objective = 0.0;
    for (sub, r) in subs.iter().zip(results) {
        objective += r.objective;
        point_ok &= scatter_cols(&mut x, sub.cols(), &r.solution);
        rc_ok &= scatter_cols(&mut rc, sub.cols(), &r.reduced_costs);
        dual_ok &= scatter_cols(&mut duals, sub.rows(), &r.dual_solution);
        slack_ok &= scatter_cols(&mut slack, sub.rows(), &r.slack);
    }
    point_ok &= objective.is_finite();
    if !point_ok {
        return no_claim_result(SolveStatus::Stalled, f64::INFINITY, k, results);
    }
    SolverResult {
        status,
        objective,
        solution: x,
        dual_solution: if dual_ok { duals } else { vec![] },
        reduced_costs: if rc_ok { rc } else { vec![] },
        slack: if slack_ok { slack } else { vec![] },
        iterations: results.iter().map(|r| r.iterations).sum(),
        timing_breakdown: Some(summed_timing(results)),
        stats: join_stats(k, results),
        ..Default::default()
    }
}

/// 解ベクトルを主張しない status の結合結果 (iterations / timing は診断値として合計)。
fn no_claim_result(
    status: SolveStatus,
    objective: f64,
    k: usize,
    results: &[SolverResult],
) -> SolverResult {
    SolverResult {
        status,
        objective,
        iterations: results.iter().map(|r| r.iterations).sum(),
        timing_breakdown: Some(summed_timing(results)),
        stats: join_stats(k, results),
        ..Default::default()
    }
}

fn has_unclean(results: &[SolverResult]) -> bool {
    results
        .iter()
        .any(|r| hard_unclean_rank(&r.status).is_some())
}

fn worst_unclean_result(k: usize, results: &[SolverResult]) -> SolverResult {
    let mut best_rank = 0u8;
    let mut best_status = SolveStatus::Stalled;
    for r in results {
        if let Some(rank) = hard_unclean_rank(&r.status) {
            if rank > best_rank {
                best_rank = rank;
                best_status = r.status.clone();
            }
        }
    }
    no_claim_result(best_status, f64::INFINITY, k, results)
}

/// 非厳密 status の優先度。None = 厳密 (Optimal / Unbounded / Infeasible /
/// SuboptimalSolution — 個別に扱う)。
fn hard_unclean_rank(status: &SolveStatus) -> Option<u8> {
    match status {
        SolveStatus::LocallyOptimal
        | SolveStatus::NonConvex(_)
        | SolveStatus::NonconvexLocal
        | SolveStatus::NonconvexGlobal
        | SolveStatus::NotSupported(_) => Some(7),
        SolveStatus::NumericalError => Some(6),
        SolveStatus::ResourceExhausted => Some(5),
        SolveStatus::Timeout => Some(4),
        SolveStatus::Stalled => Some(3),
        SolveStatus::MaxIterations => Some(2),
        SolveStatus::FeasiblePoint => Some(1),
        _ => None,
    }
}

fn summed_timing(results: &[SolverResult]) -> TimingBreakdown {
    let mut total = TimingBreakdown::default();
    for r in results {
        if let Some(t) = &r.timing_breakdown {
            total.presolve_us += t.presolve_us;
            total.solve_us += t.solve_us;
            total.postsolve_us += t.postsolve_us;
        }
    }
    total
}

fn join_stats(k: usize, results: &[SolverResult]) -> SolveStats {
    SolveStats {
        lp_split_components: k,
        bounded_eq_ub_path: results.iter().any(|r| r.stats.bounded_eq_ub_path),
        ..Default::default()
    }
}

/// local ベクトルを global 列 (行) 対応へ散布。長さ不一致なら false (その
/// global ベクトルは主張しない)。
fn scatter_cols(dest: &mut [f64], global: &[usize], local: &[f64]) -> bool {
    if local.len() != global.len() {
        return false;
    }
    for (&g, &v) in global.iter().zip(local) {
        dest[g] = v;
    }
    true
}

/// 可変数用 union-find (経路圧縮)。同一制約行の support 変数を結ぶ。
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }
    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent[rx] = ry;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INF: f64 = f64::INFINITY;

    fn box_problem(c: Vec<f64>, bounds: Vec<(f64, f64)>) -> LpProblem {
        let n = c.len();
        let a = CscMatrix::new(0, n);
        LpProblem::new_general(c, a, vec![], vec![], bounds, None).unwrap()
    }

    #[test]
    fn plan_none_for_single_component() {
        // x0+x1 >= 1 の 1 行: 全変数が同一成分。
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0, 1.0],
            a,
            vec![1.0],
            vec![crate::problem::ConstraintType::Ge],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        assert!(split_plan(&lp, &SolverOptions::default(), true).is_none());
    }

    #[test]
    fn plan_two_components_and_row_assignment() {
        let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        let lp = LpProblem::new_general(
            vec![2.0, 3.0],
            a,
            vec![1.0, 2.0],
            vec![
                crate::problem::ConstraintType::Ge,
                crate::problem::ConstraintType::Ge,
            ],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let plan = split_plan(&lp, &SolverOptions::default(), true).unwrap();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].cols, vec![0]);
        assert_eq!(plan[0].rows, vec![0]);
        assert_eq!(plan[1].cols, vec![1]);
        assert_eq!(plan[1].rows, vec![1]);
    }

    #[test]
    fn plan_skips_empty_support_rows_and_warm_start() {
        // 行 0 は support 空 (0 <= 1)。分割不能 (矛盾行版は誤 Optimal 防止)。
        let a = CscMatrix::from_triplets(&[1], &[0], &[1.0], 2, 1).unwrap();
        let lp = LpProblem::new_general(
            vec![1.0],
            a,
            vec![1.0, 1.0],
            vec![
                crate::problem::ConstraintType::Le,
                crate::problem::ConstraintType::Ge,
            ],
            vec![(0.0, f64::INFINITY)],
            None,
        )
        .unwrap();
        assert!(split_plan(&lp, &SolverOptions::default(), true).is_none());

        let a2 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
        let lp2 = LpProblem::new_general(
            vec![1.0, 1.0],
            a2,
            vec![1.0, 1.0],
            vec![
                crate::problem::ConstraintType::Ge,
                crate::problem::ConstraintType::Ge,
            ],
            vec![(0.0, f64::INFINITY); 2],
            None,
        )
        .unwrap();
        let opts = SolverOptions {
            warm_start: Some(crate::options::WarmStartBasis {
                basis: vec![],
                x_b: vec![],
            }),
            ..SolverOptions::default()
        };
        assert!(split_plan(&lp2, &opts, true).is_none());
        let opts2 = SolverOptions {
            warm_start_lp: Some(crate::options::LpWarmStart {
                basis: vec![],
                x_orig: None,
                y_orig: None,
            }),
            ..SolverOptions::default()
        };
        assert!(split_plan(&lp2, &opts2, true).is_none());
        let opts3 = SolverOptions {
            recover_warm_start_basis: true,
            ..SolverOptions::default()
        };
        assert!(split_plan(&lp2, &opts3, true).is_none());
        let opts4 = SolverOptions {
            presolve: false,
            ..SolverOptions::default()
        };
        assert!(split_plan(&lp2, &opts4, true).is_none());
    }

    #[test]
    fn box_component_closed_form_matches_hand_computed() {
        // min x0 - x1, x0 in [0,inf), x1 in [0,5] → x*=[0,5], obj=-5, rc=[1,-1]
        let lp = box_problem(vec![1.0, -1.0], vec![(0.0, INF), (0.0, 5.0)]);
        let r = solve_box_component(&lp, &[0, 1]);
        assert_eq!(r.status, SolveStatus::Optimal);
        assert_eq!(r.objective, -5.0);
        assert_eq!(r.solution, vec![0.0, 5.0]);
        assert_eq!(r.reduced_costs, vec![1.0, -1.0]);

        // c<0 かつ ub=+inf の座標は閉形式で即 Unbounded。
        let unbounded_lp = box_problem(vec![-1.0], vec![(0.0, INF)]);
        let unbounded = solve_box_component(&unbounded_lp, &[0]);
        assert_eq!(unbounded.status, SolveStatus::Unbounded);
    }

    #[test]
    fn join_table_rules() {
        let lp = box_problem(vec![1.0, 1.0], vec![(0.0, 1.0); 2]);
        let subs = [SubLp::Box { cols: vec![0] }, SubLp::Box { cols: vec![1] }];
        // Optimal / SuboptimalSolution は「点を持つ」主張なので solution を付ける
        // (無点の Optimal は join の正規化で SuboptimalSolution に落ちる)。
        let mk = |status: SolveStatus, objective: f64| SolverResult {
            status,
            objective,
            ..Default::default()
        };
        let mk_point = |status: SolveStatus, objective: f64| SolverResult {
            status,
            objective,
            solution: vec![0.0],
            ..Default::default()
        };
        let joined = join_components(
            &lp,
            2,
            &subs,
            vec![
                mk(SolveStatus::Infeasible, f64::INFINITY),
                mk(SolveStatus::Timeout, f64::INFINITY),
            ],
        );
        assert_eq!(joined.status, SolveStatus::Infeasible);

        let joined = join_components(
            &lp,
            2,
            &subs,
            vec![
                mk(SolveStatus::Unbounded, f64::NEG_INFINITY),
                mk_point(SolveStatus::Optimal, 3.0),
            ],
        );
        assert_eq!(joined.status, SolveStatus::Unbounded);

        // Unbounded + Timeout: 全体の可行集合を証明できないので Timeout を前面に。
        let joined = join_components(
            &lp,
            2,
            &subs,
            vec![
                mk(SolveStatus::Unbounded, f64::NEG_INFINITY),
                mk(SolveStatus::Timeout, f64::INFINITY),
            ],
        );
        assert_eq!(joined.status, SolveStatus::Timeout);

        // Unbounded + SuboptimalSolution: Stalled (Optimal を主張しない)。
        let joined = join_components(
            &lp,
            2,
            &subs,
            vec![
                mk(SolveStatus::Unbounded, f64::NEG_INFINITY),
                mk_point(SolveStatus::SuboptimalSolution, 1.0),
            ],
        );
        assert_eq!(joined.status, SolveStatus::Stalled);
        assert_eq!(joined.stats.lp_split_components, 2);
    }
}
