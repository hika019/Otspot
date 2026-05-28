//! McCormick envelope lower bound (Phase 5 非凸 QP 大域最適化)。
//!
//! ## 原理
//! 各 bilinear 項 `x_i · x_j` を新変数 `w_{ij}` に置換し、box `[l, u]` 上の convex
//! 包絡 (McCormick 1976) を 4 本の線形不等式で表現する:
//!
//!   w ≥ l_j x_i + l_i x_j − l_i l_j
//!   w ≥ u_j x_i + u_i x_j − u_i u_j
//!   w ≤ u_j x_i + l_i x_j − l_i u_j
//!   w ≤ l_j x_i + u_i x_j − u_i l_j
//!
//! 対角項 `x_i²` (= bilinear i==j) では UB が 1 本に縮約され、LB は端点接線 2 本に
//! なる (凸関数の secant 上界 + 接線下界)。
//!
//! 元 QP `0.5 x'Q x + c'x` は lifted 空間 `(x, w)` 上で **線形** となり、box +
//! 元 `Ax {=,≤,≥} b` + McCormick 制約を満たす LP の最小値が原問題の有効 lower bound。
//!
//! ## α-BB との関係
//! α-BB は box 全体に対し 1 つの保守的 α を取って `Q + 2α·I` PSD 化するため、
//! bilinear-rich (off-diag 主体) な問題で α が過剰に大きくなり下界が緩む。McCormick
//! は項ごとに凸包絡を張るため bilinear に対し格段に tight。
//!
//! 対角 only / 凸 Q では McCormick 寄与はわずか (= α-BB と同等)。caller は両者の
//! `max` を取ることで loss なく統合する。
//!
//! ## semi-infinite box
//! l_i / u_i に ±∞ があると w_{ij} bound が定義できないため `None` を返し、caller
//! は α-BB / interval / `-∞` に fall back する。
//!
//! ## Q storage 規約
//! `QpProblem::q` は full-symmetric storage を前提とする (`crate::qp::global::bound`
//! のコメント参照)。本実装は full / triangle どちらでも `mat_vec_mul` と整合する
//! `0.5 * v` 累積で w 係数を算出するため storage 規約に強く依存しない。

use std::collections::BTreeMap;
use std::time::Instant;

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolverResult};
use crate::qp::problem::QpProblem;
use crate::sparse::CscMatrix;

use super::bound::{all_bounds_finite, is_feasible_result};

/// 値がゼロかどうかの数値的判定許容。`QpProblem::is_zero_q` と一致させる。
const Q_ZERO_TOL: f64 = 1e-12;

/// 1 つの off-diag bilinear 項に対する McCormick 不等式数 (LB×2 + UB×2)。
const MCCORMICK_INEQ_PER_OFF_DIAG: usize = 4;

/// 対角 (x²) 項に対する McCormick 不等式数 (端点接線 LB×2 + secant UB×1)。
const MCCORMICK_INEQ_PER_DIAG: usize = 3;

/// McCormick lower bound on the given box.
///
/// `Some(lb)`: lifted LP が feasible、`lb` は元問題の有効 lower bound。
/// `None`: semi-infinite box / Q が実質ゼロ (= McCormick 寄与なし) / LP infeasible
/// / unbounded で lb を得られない場合。caller は他の lb 経路に倒す。
pub(crate) fn mccormick_lower_bound(
    problem: &QpProblem,
    node_bounds: &[(f64, f64)],
    base_opts: &SolverOptions,
    deadline: Option<Instant>,
) -> Option<f64> {
    solve_lifted_lp(problem, node_bounds, base_opts, deadline, /* include_envelope = */ true)
}

/// 共通 driver: lifted LP を構築 (`include_envelope=false` で McCormick 不等式を省略) し
/// `solve_qp_with` 経由で min を返す。`false` 経路は in-source unit test の no-op teeth
/// proof 専用 (envelope 不等式を機械的に剥がす単一切替点を 1 つに集約)。
fn solve_lifted_lp(
    problem: &QpProblem,
    node_bounds: &[(f64, f64)],
    base_opts: &SolverOptions,
    deadline: Option<Instant>,
    include_envelope: bool,
) -> Option<f64> {
    if !all_bounds_finite(node_bounds) {
        return None;
    }
    let pairs = collect_bilinear_pairs(&problem.q);
    if pairs.is_empty() {
        return None;
    }
    let lifted = build_mccormick_lp(problem, node_bounds, &pairs, include_envelope)?;
    let mut opts = base_opts.clone();
    opts.multistart = None;
    opts.global_optimization = None;
    // sub-solve hygiene: 元問題向け warm hint は lifted 空間で dim も active set も
    // 異なるため、3 種すべて消去する (warm_start_qp 単独消去では不十分)。
    opts.warm_start = None;
    opts.warm_start_qp = None;
    opts.warm_start_lp = None;
    opts.deadline = deadline;
    let res: SolverResult = crate::qp::solve_qp_with(&lifted, &opts);
    if !is_feasible_result(&res.status) {
        return None;
    }
    Some(res.objective)
}

/// 1 unordered pair (i ≤ j) と objective 係数 `0.5 · (Q[i,j] + Q[j,i])` 相当。
#[derive(Clone, Copy)]
struct BilinearTerm {
    i: usize,
    j: usize,
    /// w_{ij} に乗じる係数。`0.5 x'Q x` を lifted 表現 `Σ coef · w` に書き直したときの値。
    coef: f64,
}

/// Q の nnz 全 entry を unordered (i, j) でグルーピングし、各 pair に `0.5·v` を集約。
/// full-symmetric storage では off-diag pair (i<j) が `(i,j)`/`(j,i)` の 2 entry に
/// 分かれて格納されるが、ここでまとめることで storage 規約に依存しない係数を得る。
fn collect_bilinear_pairs(q: &CscMatrix) -> Vec<BilinearTerm> {
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for col in 0..q.ncols {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            let v = q.values[k];
            if v.abs() < Q_ZERO_TOL {
                continue;
            }
            let key = if row <= col { (row, col) } else { (col, row) };
            *acc.entry(key).or_insert(0.0) += 0.5 * v;
        }
    }
    acc.into_iter()
        .filter(|(_, c)| c.abs() >= Q_ZERO_TOL)
        .map(|((i, j), coef)| BilinearTerm { i, j, coef })
        .collect()
}

/// box 上で w_{ij} = x_i · x_j が取りうる値域 (LP の w 変数 box bound に用いる)。
fn w_interval(bounds: &[(f64, f64)], i: usize, j: usize) -> (f64, f64) {
    let (li, ui) = bounds[i];
    if i == j {
        let lo = if li <= 0.0 && ui >= 0.0 {
            0.0
        } else {
            (li * li).min(ui * ui)
        };
        let hi = (li * li).max(ui * ui);
        return (lo, hi);
    }
    let (lj, uj) = bounds[j];
    let candidates = [li * lj, li * uj, ui * lj, ui * uj];
    let mut lo = candidates[0];
    let mut hi = candidates[0];
    for &v in &candidates[1..] {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// 行 push helper (3 列構成: x_var + w_var)。`coef_x == 0` の場合は entry を省略し
/// 後段 LP solver が空 col を扱わずに済むよう保つ。
fn push_row_3(
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<f64>,
    row: usize,
    var_x: usize,
    var_w: usize,
    coef_x: f64,
    coef_w: f64,
) {
    if coef_x != 0.0 {
        rows.push(row);
        cols.push(var_x);
        vals.push(coef_x);
    }
    rows.push(row);
    cols.push(var_w);
    vals.push(coef_w);
}

/// 行 push helper (4 列構成: x_i + x_j + w_var)。
fn push_row_4(
    rows: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<f64>,
    row: usize,
    var_xi: usize,
    var_xj: usize,
    var_w: usize,
    coef_xi: f64,
    coef_xj: f64,
    coef_w: f64,
) {
    if coef_xi != 0.0 {
        rows.push(row);
        cols.push(var_xi);
        vals.push(coef_xi);
    }
    if coef_xj != 0.0 {
        rows.push(row);
        cols.push(var_xj);
        vals.push(coef_xj);
    }
    rows.push(row);
    cols.push(var_w);
    vals.push(coef_w);
}

/// lifted LP 構築。変数順: `[x_0, …, x_{n-1}, w_0, …, w_{p-1}]`。
/// Q = 0 (実質 LP) で組み立て、`solve_qp_with` 内の `is_zero_q` 判定で LP 経路に流れる。
///
/// `include_envelope=false` の場合は McCormick 不等式行を省略し box+linear だけ残す。
/// これは "no-op proof teeth" 用の差分 = envelope を剥がすと lb は w 変数 box の min
/// に退化し strict に緩む、を機械的に観測するための専用経路。production では常に true。
fn build_mccormick_lp(
    problem: &QpProblem,
    node_bounds: &[(f64, f64)],
    pairs: &[BilinearTerm],
    include_envelope: bool,
) -> Option<QpProblem> {
    let n = problem.num_vars;
    let p = pairs.len();
    let total_vars = n + p;

    let q = CscMatrix::from_triplets(&[], &[], &[], total_vars, total_vars).ok()?;

    let mut c = vec![0.0_f64; total_vars];
    c[..n].copy_from_slice(&problem.c);
    for (k, term) in pairs.iter().enumerate() {
        c[n + k] = term.coef;
    }

    let mut bounds = Vec::with_capacity(total_vars);
    bounds.extend_from_slice(node_bounds);
    for term in pairs {
        bounds.push(w_interval(node_bounds, term.i, term.j));
    }

    // 既存 A の re-emit + McCormick 行追加 (envelope OFF なら 0 行)
    let extra_rows: usize = if include_envelope {
        pairs
            .iter()
            .map(|t| {
                if t.i == t.j {
                    MCCORMICK_INEQ_PER_DIAG
                } else {
                    MCCORMICK_INEQ_PER_OFF_DIAG
                }
            })
            .sum()
    } else {
        0
    };
    let total_rows = problem.num_constraints + extra_rows;

    let nnz_estimate = problem.a.values.len() + extra_rows * 3;
    let mut rows: Vec<usize> = Vec::with_capacity(nnz_estimate);
    let mut cols: Vec<usize> = Vec::with_capacity(nnz_estimate);
    let mut vals: Vec<f64> = Vec::with_capacity(nnz_estimate);
    let mut b: Vec<f64> = Vec::with_capacity(total_rows);
    let mut types: Vec<ConstraintType> = Vec::with_capacity(total_rows);

    for col in 0..n {
        for k in problem.a.col_ptr[col]..problem.a.col_ptr[col + 1] {
            rows.push(problem.a.row_ind[k]);
            cols.push(col);
            vals.push(problem.a.values[k]);
        }
    }
    b.extend_from_slice(&problem.b);
    types.extend_from_slice(&problem.constraint_types);

    let mut row = problem.num_constraints;
    if !include_envelope {
        debug_assert_eq!(row, total_rows);
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, total_rows, total_vars).ok()?;
        let mut lifted = QpProblem::new(q, c, a, b, bounds, types).ok()?;
        lifted.obj_offset = problem.obj_offset;
        return Some(lifted);
    }
    for (k, term) in pairs.iter().enumerate() {
        let (li, ui) = node_bounds[term.i];
        let (lj, uj) = node_bounds[term.j];
        let w_col = n + k;

        if term.i == term.j {
            // x² LB tangent at l: w − 2l x ≥ −l²
            push_row_3(&mut rows, &mut cols, &mut vals, row, term.i, w_col, -2.0 * li, 1.0);
            b.push(-li * li);
            types.push(ConstraintType::Ge);
            row += 1;
            // x² LB tangent at u: w − 2u x ≥ −u²
            push_row_3(&mut rows, &mut cols, &mut vals, row, term.i, w_col, -2.0 * ui, 1.0);
            b.push(-ui * ui);
            types.push(ConstraintType::Ge);
            row += 1;
            // x² UB secant: w − (l+u) x ≤ −l u
            push_row_3(&mut rows, &mut cols, &mut vals, row, term.i, w_col, -(li + ui), 1.0);
            b.push(-li * ui);
            types.push(ConstraintType::Le);
            row += 1;
        } else {
            // LB: w − l_j x_i − l_i x_j ≥ −l_i l_j
            push_row_4(&mut rows, &mut cols, &mut vals, row, term.i, term.j, w_col, -lj, -li, 1.0);
            b.push(-li * lj);
            types.push(ConstraintType::Ge);
            row += 1;
            // LB: w − u_j x_i − u_i x_j ≥ −u_i u_j
            push_row_4(&mut rows, &mut cols, &mut vals, row, term.i, term.j, w_col, -uj, -ui, 1.0);
            b.push(-ui * uj);
            types.push(ConstraintType::Ge);
            row += 1;
            // UB: w − u_j x_i − l_i x_j ≤ −l_i u_j
            push_row_4(&mut rows, &mut cols, &mut vals, row, term.i, term.j, w_col, -uj, -li, 1.0);
            b.push(-li * uj);
            types.push(ConstraintType::Le);
            row += 1;
            // UB: w − l_j x_i − u_i x_j ≤ −u_i l_j
            push_row_4(&mut rows, &mut cols, &mut vals, row, term.i, term.j, w_col, -lj, -ui, 1.0);
            b.push(-ui * lj);
            types.push(ConstraintType::Le);
            row += 1;
        }
    }
    debug_assert_eq!(row, total_rows);

    let a = CscMatrix::from_triplets(&rows, &cols, &vals, total_rows, total_vars).ok()?;

    let mut lifted = QpProblem::new(q, c, a, b, bounds, types).ok()?;
    lifted.obj_offset = problem.obj_offset;
    Some(lifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SolverOptions;
    use crate::problem::ConstraintType;

    fn diag_q(diag: &[f64]) -> CscMatrix {
        let n = diag.len();
        let rows: Vec<usize> = (0..n).collect();
        let cols: Vec<usize> = (0..n).collect();
        CscMatrix::from_triplets(&rows, &cols, diag, n, n).unwrap()
    }

    fn build_problem(q: CscMatrix, c: Vec<f64>, bounds: Vec<(f64, f64)>) -> QpProblem {
        let n = c.len();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, n).unwrap();
        QpProblem::new(q, c, a, vec![], bounds, vec![]).unwrap()
    }

    #[test]
    fn collect_bilinear_pairs_full_symmetric_aggregates_off_diag() {
        // Q = [[2, 1], [1, 0]] full-symmetric storage の 3 entry → unordered pair
        // (0,0)=0.5·2=1.0, (0,1)=0.5·(1+1)=1.0, (1,1) は値ゼロで skip。
        let q = CscMatrix::from_triplets(&[0, 1, 0], &[0, 0, 1], &[2.0, 1.0, 1.0], 2, 2).unwrap();
        let pairs = collect_bilinear_pairs(&q);
        let mut got: Vec<(usize, usize, f64)> = pairs.iter().map(|t| (t.i, t.j, t.coef)).collect();
        got.sort_by_key(|(i, j, _)| (*i, *j));
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, 0);
        assert_eq!(got[0].1, 0);
        assert!((got[0].2 - 1.0).abs() < 1e-12);
        assert_eq!(got[1].0, 0);
        assert_eq!(got[1].1, 1);
        assert!((got[1].2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn collect_bilinear_pairs_upper_only_storage() {
        // Q = [[0, 1], [0, 0]] (lower 半 entry なし) → (0,1) coef = 0.5 (= storage 規約に
        // 依存しない accumulation の挙動を確認、α-BB は full-sym 前提だが McCormick は
        // accumulate するだけなので "半 storage" でもクラッシュしない)
        let q = CscMatrix::from_triplets(&[0], &[1], &[1.0], 2, 2).unwrap();
        let pairs = collect_bilinear_pairs(&q);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].i, 0);
        assert_eq!(pairs[0].j, 1);
        assert!((pairs[0].coef - 0.5).abs() < 1e-12);
    }

    #[test]
    fn w_interval_diag_zero_crossing() {
        let (lo, hi) = w_interval(&[(-2.0, 3.0)], 0, 0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 9.0);
    }

    #[test]
    fn w_interval_off_diag_sign_change() {
        let (lo, hi) = w_interval(&[(-1.0, 2.0), (-3.0, 1.0)], 0, 1);
        // candidates: -1*-3=3, -1*1=-1, 2*-3=-6, 2*1=2 → [-6, 3]
        assert_eq!(lo, -6.0);
        assert_eq!(hi, 3.0);
    }

    #[test]
    fn mccormick_lb_none_for_infinite_box() {
        let q = diag_q(&[-2.0]);
        let p = build_problem(q, vec![0.0], vec![(f64::NEG_INFINITY, 1.0)]);
        let opts = SolverOptions::default();
        assert!(mccormick_lower_bound(&p, &p.bounds, &opts, None).is_none());
    }

    #[test]
    fn mccormick_lb_none_for_zero_q() {
        // Q ≡ 0 → pairs is empty → McCormick adds nothing (caller should rely on interval/α-BB)
        let q = CscMatrix::from_triplets(&[], &[], &[], 2, 2).unwrap();
        let p = build_problem(q, vec![1.0, -1.0], vec![(-1.0, 1.0); 2]);
        let opts = SolverOptions::default();
        assert!(mccormick_lower_bound(&p, &p.bounds, &opts, None).is_none());
    }

    #[test]
    fn mccormick_lb_recovers_concave_1d_global() {
        // f = −x² on [−2, 2], global = −4. McCormick of x² on [−2,2] gives convex hull
        // [tangent at ±2, secant top] → min of −w = −max w = −4. lb = −4 (tight)。
        let q = diag_q(&[-2.0]);
        let p = build_problem(q, vec![0.0], vec![(-2.0, 2.0)]);
        let opts = SolverOptions::default();
        let lb = mccormick_lower_bound(&p, &p.bounds, &opts, None).expect("McCormick must solve");
        assert!(lb.is_finite());
        assert!(
            (lb - (-4.0)).abs() < 1e-4,
            "expected tight lb ≈ -4, got {lb}"
        );
    }

    #[test]
    fn mccormick_lb_tighter_than_alpha_bb_on_asymmetric_bilinear() {
        // f(x,y) = -xy on [-2,1] × [-1,2]。 global = -2 at (1,2) と (-2,-1).
        // 解析:
        //   α-BB: Gershgorin で α=0.5、L 凸化 → lb = −2.125 (緩い)
        //   McCormick: LP relaxation で max xy = 2 (corner で達成) → lb = −2 (tight)
        // 両者の差 0.125 が strict 比較で検出可能 (no-op で lb=-∞ なら両 assertion FAIL)。
        use crate::qp::global::bound_alpha_bb::{alpha_bb_lower_bound, gershgorin_alpha};
        let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[-1.0, -1.0], 2, 2).unwrap();
        let p = build_problem(q, vec![0.0, 0.0], vec![(-2.0, 1.0), (-1.0, 2.0)]);
        let opts = SolverOptions::default();
        let alpha = gershgorin_alpha(&p.q);
        let lb_alpha = alpha_bb_lower_bound(&p, &p.bounds, alpha, &opts, None).expect("α-BB");
        let lb_mc = mccormick_lower_bound(&p, &p.bounds, &opts, None).expect("McCormick");
        // 両 lb とも valid (≤ −2)、ただし McCormick は厳密に tight、α-BB は緩い。
        assert!(lb_mc <= -2.0 + 1e-4, "McCormick lb {lb_mc} must underestimate global -2");
        assert!(lb_alpha <= -2.0 + 1e-4, "α-BB lb {lb_alpha} must underestimate global -2");
        // McCormick が strict に tight ( > α-BB lb )
        assert!(
            lb_mc > lb_alpha + 1e-3,
            "McCormick should beat α-BB on pure bilinear asym box: mc={lb_mc}, alpha={lb_alpha}"
        );
        // McCormick は -2 に近い (-2.0 ± 1e-4)、α-BB は明確に劣る
        assert!(
            lb_mc >= -2.0 - 1e-4,
            "McCormick should reach tight -2, got {lb_mc}"
        );
    }

    #[test]
    fn mccormick_lb_underestimates_objective_at_corners() {
        // f(x,y) = xy on [-1,1]^2 . 4 corner で f = ±1, McCormick lb ≤ −1.
        let q = CscMatrix::from_triplets(&[0, 1], &[1, 0], &[1.0, 1.0], 2, 2).unwrap();
        let p = build_problem(q, vec![0.0, 0.0], vec![(-1.0, 1.0); 2]);
        let opts = SolverOptions::default();
        let lb = mccormick_lower_bound(&p, &p.bounds, &opts, None).expect("McCormick");
        assert!(lb <= -1.0 + 1e-6, "lb {lb} must underestimate global min -1");
    }

    #[test]
    fn mccormick_lb_with_linear_eq_constraint() {
        // x+y=1, min -x²-y² with x,y in [0,1]: global = -1 at corner.
        let q = diag_q(&[-2.0, -2.0]);
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let p = QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![1.0],
            vec![(0.0, 1.0); 2],
            vec![ConstraintType::Eq],
        )
        .unwrap();
        let opts = SolverOptions::default();
        let lb = mccormick_lower_bound(&p, &p.bounds, &opts, None).expect("McCormick");
        assert!(lb.is_finite() && lb <= -1.0 + 1e-4, "lb {lb} should be ≤ -1");
    }

    #[test]
    fn mccormick_lb_tightens_as_box_shrinks() {
        // 同じ問題で box 幅を縮めると lb は上がる (= 枝刈効果の数値根拠)。
        let q = diag_q(&[-2.0]);
        let p = build_problem(q, vec![0.0], vec![(-2.0, 2.0)]);
        let opts = SolverOptions::default();
        let lb_wide = mccormick_lower_bound(&p, &[(-2.0, 2.0)], &opts, None).expect("wide");
        let lb_narrow = mccormick_lower_bound(&p, &[(0.0, 1.0)], &opts, None).expect("narrow");
        assert!(
            lb_narrow >= lb_wide - 1e-6,
            "narrow lb {lb_narrow} should not be worse than wide {lb_wide}"
        );
    }

    /// no-op teeth: McCormick 不等式を機械的に剥がす (`include_envelope=false`) と lb は
    /// w 変数 box bound しか効かなくなり厳格に緩む。同 fixture で full vs no-op を比較し
    /// strict 差を観測することで、後段 BB driver sentinel
    /// (`mccormick_reduces_or_matches_bb_node_count_on_bilinear_rich_set`) の teeth を
    /// 数学的に裏付ける (= envelope を消すと当該 sentinel は確実に FAIL する)。
    #[test]
    fn mccormick_lb_strictly_better_than_envelope_removed() {
        // f(x) = -x² + x on [0,1]. global = 0 at x=0 or x=1 (両端で f=0)。
        // 解析:
        //   envelope OFF: w_box=[0,1], c_w=-1, c_x=1 → min -w+x = -1+0 = -1 (緩い)
        //   envelope ON : tangent + secant が xy=x² の hull を作り、w=0 を強制 → lb=0 (tight)
        // → 厳格な差 1.0 を観測 (envelope なしでは絶対に達成できない)
        let q = diag_q(&[-2.0]);
        let p = build_problem(q, vec![1.0], vec![(0.0, 1.0)]);
        let opts = SolverOptions::default();
        let lb_full = solve_lifted_lp(&p, &p.bounds, &opts, None, true).expect("full");
        let lb_noop = solve_lifted_lp(&p, &p.bounds, &opts, None, false).expect("noop");
        assert!(lb_noop.is_finite(), "noop must remain finite, got {lb_noop}");
        assert!(
            lb_full > lb_noop + 0.5,
            "McCormick envelope must add real tightness: full={lb_full}, noop={lb_noop}"
        );
        // sanity: 両者とも global f*=0 を underestimate
        assert!(lb_full <= 0.0 + 1e-4, "full lb {lb_full} must underestimate 0");
        assert!(lb_noop <= 0.0 + 1e-4, "noop lb {lb_noop} must underestimate 0");
    }

    /// Ge constraint type: x+y ≥ 1, min -x²-y² on [0,1]². global = -1 at (1,0)/(0,1)/(1,1)。
    /// 既存 Eq テストでカバーされない Ge 経路の build_mccormick_lp 行 emit 整合性を確認。
    #[test]
    fn mccormick_lb_with_linear_ge_constraint() {
        let q = diag_q(&[-2.0, -2.0]);
        let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
        let p = QpProblem::new(
            q,
            vec![0.0, 0.0],
            a,
            vec![1.0],
            vec![(0.0, 1.0); 2],
            vec![ConstraintType::Ge],
        )
        .unwrap();
        let opts = SolverOptions::default();
        let lb = mccormick_lower_bound(&p, &p.bounds, &opts, None).expect("Ge McCormick");
        assert!(
            lb.is_finite() && lb <= -2.0 + 1e-4,
            "Ge lb {lb} must underestimate global ≥ -2"
        );
    }

    /// coef=0 path: box の片端が原点に張り付くと McCormick 行係数が 0 になる
    /// (例: li=0 のとき diag LB tangent at l は w − 2·0·x ≥ 0 = `w ≥ 0`、x 列係数なし)。
    /// push_row_3/4 の `coef_x != 0.0` skip が正しく分岐し triplet が壊れないこと、
    /// 結果 lb が一貫していることを確認する。
    #[test]
    fn mccormick_lb_with_zero_coef_box_edges() {
        // f = -x² on [0, 2]: global = -4 at x=2. li=0 で diag LB tangent at l は
        // x 列係数ゼロ (= entry skip)、x² の McCormick envelope は w≥0, w≥4x-4, w≤2x。
        let q = diag_q(&[-2.0]);
        let p = build_problem(q, vec![0.0], vec![(0.0, 2.0)]);
        let opts = SolverOptions::default();
        let lb = mccormick_lower_bound(&p, &p.bounds, &opts, None).expect("zero-coef path");
        // LP IPM tol ~1e-3 を加味、tight 上限 -4 + 1e-3
        assert!(
            lb.is_finite() && (lb - (-4.0)).abs() < 1e-3,
            "zero-coef edge lb {lb} must reach tight global -4 (±1e-3)"
        );
    }

    /// Q_ZERO_TOL 境界: 全 entry が threshold 未満なら pairs 空 → None。
    /// 一方 threshold 直上の entry は採用される (= 境界の側性が明確)。
    #[test]
    fn mccormick_lb_q_zero_tol_boundary() {
        // 1) 全 entry < Q_ZERO_TOL: pairs 空 → None
        let half = Q_ZERO_TOL * 0.5;
        let q_below = CscMatrix::from_triplets(&[0], &[0], &[half], 1, 1).unwrap();
        let p_below = build_problem(q_below, vec![0.0], vec![(-1.0, 1.0)]);
        let opts = SolverOptions::default();
        assert!(
            mccormick_lower_bound(&p_below, &p_below.bounds, &opts, None).is_none(),
            "Q values below Q_ZERO_TOL must produce empty pairs → None"
        );
        // 2) threshold の 10 倍 (= 明確に live) なら pairs 非空 → Some
        let live = Q_ZERO_TOL * 10.0;
        let q_above = CscMatrix::from_triplets(&[0], &[0], &[live], 1, 1).unwrap();
        let p_above = build_problem(q_above, vec![0.0], vec![(-1.0, 1.0)]);
        let lb = mccormick_lower_bound(&p_above, &p_above.bounds, &opts, None)
            .expect("above-threshold Q must yield lb");
        assert!(lb.is_finite(), "above-threshold lb must be finite, got {lb}");
    }

    /// 大規模 fixture (n=8): dense bilinear + box bound で BB 経路に乗らず純 lb 評価。
    /// 多数 pair (n=8 → 28 off-diag + 8 diag = 36 pairs × 3-4 ineq = 100+ 行) の
    /// 行構築・LP solve が壊れないこと、underestimator 性質を維持することを確認。
    #[test]
    fn mccormick_lb_holds_on_n8_dense_bilinear() {
        const N: usize = 8;
        // Q: full-symmetric, off-diag = 1, diag = -1 (light nonconvex)
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for c in 0..N {
            for r in 0..N {
                rows.push(r);
                cols.push(c);
                vals.push(if r == c { -1.0 } else { 1.0 });
            }
        }
        let q = CscMatrix::from_triplets(&rows, &cols, &vals, N, N).unwrap();
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, N).unwrap();
        let p = QpProblem::new(q, vec![0.0; N], a, vec![], vec![(-1.0, 1.0); N], vec![])
            .unwrap();
        let opts = SolverOptions::default();
        let lb = mccormick_lower_bound(&p, &p.bounds, &opts, None)
            .expect("n=8 dense bilinear must solve");
        assert!(lb.is_finite(), "n=8 lb must be finite, got {lb}");
        // x=1 ベクトル等の corner 評価で valid underestimator を確認
        let qx = p.q.mat_vec_mul(&[1.0; N]).unwrap();
        let xqx: f64 = [1.0; N].iter().zip(qx.iter()).map(|(a, b)| a * b).sum();
        let f_corner = 0.5 * xqx;
        assert!(
            lb <= f_corner + 1e-5,
            "n=8 lb {lb} must underestimate corner f={f_corner}"
        );
    }
}
