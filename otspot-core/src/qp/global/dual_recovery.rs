//! incumbent の主解 `x` を固定したまま元問題の KKT 乗数 `(y, z)` を復元する。
//!
//! B&B の node 解が持つ乗数は「分枝で加えた人工 bound を含む sub-box」のもので、
//! 元問題の乗数ではない (人工 bound の乗数が z に残り、元問題では非活性な bound に
//! 正の乗数が付く)。原問題で解き直す `polish_incumbent_duals` は非凸では別の local
//! optimum へ滑って棄却されるため、`x` を動かさない復元経路が要る。
//!
//! active set 上の stationarity
//! `Q x + c + Aᵀ y + (−z_lb + z_ub) = 0`
//! を、符号制約 (`Le: y ≥ 0`, `Ge: y ≤ 0`, `Eq: 自由`, `z ≥ 0`) と非活性成分の
//! 0 固定 (= complementarity) つきで解く。残差 `s⁺ + s⁻` の最小化 LP なので、
//! `x` が元問題の KKT 点でなければ残差が残り、呼び出し側の検証 (`local_kkt_within`)
//! が採用を拒む — この関数は「主張を作る」のではなく「主張の根拠を探す」。

use crate::options::SolverOptions;
use crate::problem::{ConstraintType, SolveStatus};
use crate::qp::kkt_resid::dd_impl;
use crate::qp::problem::QpProblem;
use crate::tolerances::FX_TOL;
use otspot_num::sparse::CscMatrix;

/// 復元した乗数。`y` は元問題の制約数、`bound_duals` は `[lb 有限列, ub 有限列]` 順
/// (`kkt_resid::bound_contrib` の layout)。
pub(crate) struct RecoveredDuals {
    pub(crate) y: Vec<f64>,
    pub(crate) bound_duals: Vec<f64>,
}

/// active 判定の許容幅。
///
/// 下流の受理条件 (`local_kkt_within` の `kkt_tol`) と同じ幅を使う。ここを狭くすると、
/// 境界から `user_eps`〜`kkt_tol` の距離にある近似 incumbent で「最終ゲートは通せる
/// のに、その乗数を復元候補から外したせいで復元できない」状態になる。逆に広く取り
/// すぎても、非活性な bound に付いた乗数は復元後の complementarity 検証が弾くため、
/// この閾値は安全側にしか効かない。
fn active_tol(user_eps: f64, scale: f64) -> f64 {
    let kkt_tol = (user_eps * super::POLISH_KKT_ACCEPT_FACTOR).min(super::POLISH_KKT_ABS_CAP);
    (kkt_tol * (1.0 + scale.abs())).max(FX_TOL)
}

/// LP の列が元問題のどの乗数に対応するか。
enum Multiplier {
    /// 制約 `i` の双対。`sign` を掛けて `y[i]` になる (Le: +1, Ge: −1, Eq: +1 で自由変数)。
    Row { i: usize, sign: f64 },
    /// 変数 `j` の下界乗数 (`z_lb`)。
    Lower { j: usize },
    /// 変数 `j` の上界乗数 (`z_ub`)。
    Upper { j: usize },
}

/// `x` を固定したまま元問題の乗数を復元する。復元 LP が解けなければ `None`。
pub(crate) fn recover_duals_at_fixed_x(
    problem: &QpProblem,
    x: &[f64],
    base_opts: &SolverOptions,
    user_eps: f64,
) -> Option<RecoveredDuals> {
    let n = problem.num_vars;
    let m = problem.num_constraints;
    if n == 0 || x.len() != n || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // 勾配と行 activity は DD で積算する: 検証側 (`local_kkt_within` → `kkt_residual_rel`)
    // が DD なので、素の f64 で組むと打ち消しの大きい行で右辺と active set が
    // 食い違い、復元できるはずの乗数を取り逃す。
    let qx = dd_impl::qx(&problem.q, x);
    let g: Vec<f64> = (0..n)
        .map(|j| f64::from(qx[j] + twofloat::TwoFloat::from(problem.c[j])))
        .collect();
    if g.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let ax: Vec<f64> = dd_impl::ax(&problem.a, x)
        .into_iter()
        .map(f64::from)
        .collect();
    let mut multipliers: Vec<Multiplier> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut bounds_lp: Vec<(f64, f64)> = Vec::new();

    // A の行ごとの成分 (CSC を 1 度だけ走査して行方向に集める)。
    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for j in 0..problem.a.ncols() {
        let (row_ind, values) = problem.a.column(j);
        for (k, &i) in row_ind.iter().enumerate() {
            if i < m {
                row_entries[i].push((j, values[k]));
            }
        }
    }

    for i in 0..m {
        let tol = active_tol(user_eps, problem.b[i]);
        let (sign, free) = match problem.constraint_types[i] {
            ConstraintType::Eq => (1.0, true),
            ConstraintType::Le => {
                if ax[i] < problem.b[i] - tol {
                    continue;
                }
                (1.0, false)
            }
            ConstraintType::Ge => {
                if ax[i] > problem.b[i] + tol {
                    continue;
                }
                (-1.0, false)
            }
        };
        let col = multipliers.len();
        for &(j, v) in &row_entries[i] {
            rows.push(j);
            cols.push(col);
            vals.push(sign * v);
        }
        bounds_lp.push(if free {
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            (0.0, f64::INFINITY)
        });
        multipliers.push(Multiplier::Row { i, sign });
    }

    for (j, &(lb, ub)) in problem.bounds.iter().enumerate() {
        if lb.is_finite() && x[j] <= lb + active_tol(user_eps, lb) {
            let col = multipliers.len();
            rows.push(j);
            cols.push(col);
            vals.push(-1.0);
            bounds_lp.push((0.0, f64::INFINITY));
            multipliers.push(Multiplier::Lower { j });
        }
        if ub.is_finite() && x[j] >= ub - active_tol(user_eps, ub) {
            let col = multipliers.len();
            rows.push(j);
            cols.push(col);
            vals.push(1.0);
            bounds_lp.push((0.0, f64::INFINITY));
            multipliers.push(Multiplier::Upper { j });
        }
    }

    // 残差変数 s⁺ − s⁻ (両方 ≥ 0)。目的はその総和の最小化。
    let n_mult = multipliers.len();
    let mut c_lp = vec![0.0_f64; n_mult];
    for j in 0..n {
        for (offset, coefficient) in [(0_usize, 1.0_f64), (1, -1.0)] {
            rows.push(j);
            cols.push(n_mult + 2 * j + offset);
            vals.push(coefficient);
        }
        bounds_lp.push((0.0, f64::INFINITY));
        bounds_lp.push((0.0, f64::INFINITY));
        c_lp.push(1.0);
        c_lp.push(1.0);
    }
    let ncols = n_mult + 2 * n;

    let a_lp = CscMatrix::from_triplets(&rows, &cols, &vals, n, ncols).ok()?;
    let lp = QpProblem::new(
        CscMatrix::new(ncols, ncols),
        c_lp,
        a_lp,
        g.iter().map(|v| -v).collect(),
        bounds_lp,
        vec![ConstraintType::Eq; n],
    )
    .ok()?;

    let mut opts = base_opts.clone();
    opts.multistart = None;
    opts.global_optimization = None;
    opts.warm_start = None;
    opts.warm_start_qp = None;
    opts.warm_start_lp = None;
    // 残差 (s⁺ + s⁻) の最小化なので、目的値 0 に達していれば status が
    // `SuboptimalSolution` でも乗数としては使える (LP 側の `SuboptimalSolution` は
    // 実行可能性が検証済み)。採否は呼び出し側の `local_kkt_within` が決めるため、
    // ここで status だけを理由に捨てると復元できる incumbent を落とす。
    let res = crate::qp::solve_qp_with(&lp, &opts);
    if !matches!(
        res.status,
        SolveStatus::Optimal | SolveStatus::SuboptimalSolution
    ) || res.solution.len() != ncols
    {
        return None;
    }

    let n_lb = problem
        .bounds
        .iter()
        .filter(|&&(lb, _)| lb.is_finite())
        .count();
    let lb_slot: Vec<Option<usize>> = slot_index(problem.bounds.iter().map(|&(lb, _)| lb));
    let ub_slot: Vec<Option<usize>> = slot_index(problem.bounds.iter().map(|&(_, ub)| ub));

    let mut y = vec![0.0_f64; m];
    let mut bound_duals = vec![0.0_f64; n_lb + ub_slot.iter().flatten().count()];
    for (col, mult) in multipliers.iter().enumerate() {
        let value = res.solution[col];
        if !value.is_finite() {
            return None;
        }
        match *mult {
            Multiplier::Row { i, sign } => y[i] = sign * value,
            Multiplier::Lower { j } => bound_duals[lb_slot[j]?] = value,
            Multiplier::Upper { j } => bound_duals[n_lb + ub_slot[j]?] = value,
        }
    }
    Some(RecoveredDuals { y, bound_duals })
}

/// 有限な bound を持つ列に、出現順の slot 番号を割り当てる。
fn slot_index(bounds: impl Iterator<Item = f64>) -> Vec<Option<usize>> {
    let mut next = 0_usize;
    bounds
        .map(|bnd| {
            if bnd.is_finite() {
                let slot = next;
                next += 1;
                Some(slot)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::qp::kkt_resid::bound_contrib;

    fn opts() -> SolverOptions {
        let mut o = SolverOptions::default();
        o.timeout_secs = Some(10.0);
        o
    }

    /// stationarity 残差 `max_j |g_j + (Aᵀy)_j + bound_contrib_j|`。
    fn stationarity_max(problem: &QpProblem, x: &[f64], rec: &RecoveredDuals) -> f64 {
        let n = problem.num_vars;
        let qx = dd_impl::qx(&problem.q, x);
        let aty = dd_impl::aty(&problem.a, &rec.y, n);
        let contrib = bound_contrib(&problem.bounds, &rec.bound_duals);
        (0..n)
            .map(|j| {
                f64::from(qx[j] + aty[j] + twofloat::TwoFloat::from(problem.c[j] + contrib[j]))
                    .abs()
            })
            .fold(0.0_f64, f64::max)
    }

    /// min 0.5x² − x s.t. x ≤ 0.5 (Le), x ∈ [−10, 10]。
    /// x* = 0.5 で g = −0.5、Le 行が active なので y = 0.5 (≥ 0) が唯一の乗数。
    #[test]
    fn recovers_row_multiplier_on_active_le_row() {
        let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let p = QpProblem::new(
            q,
            vec![-1.0],
            a,
            vec![0.5],
            vec![(-10.0, 10.0)],
            vec![ConstraintType::Le],
        )
        .unwrap();
        let rec = recover_duals_at_fixed_x(&p, &[0.5], &opts(), 1e-6).expect("recovered");
        assert!((rec.y[0] - 0.5).abs() < 1e-9, "y = {:?}", rec.y);
        assert!(rec.bound_duals.iter().all(|z| z.abs() < 1e-9));
        assert!(stationarity_max(&p, &[0.5], &rec) < 1e-9);
    }

    /// min −x s.t. x ∈ [0, 1] (制約行なし)。x* = 1 で上界が active、z_ub = 1。
    #[test]
    fn recovers_upper_bound_multiplier() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let p = QpProblem::new(q, vec![-1.0], a, vec![], vec![(0.0, 1.0)], vec![]).unwrap();
        let rec = recover_duals_at_fixed_x(&p, &[1.0], &opts(), 1e-6).expect("recovered");
        // layout: [lb slot, ub slot]
        assert!(rec.bound_duals[0].abs() < 1e-9, "z = {:?}", rec.bound_duals);
        assert!(
            (rec.bound_duals[1] - 1.0).abs() < 1e-9,
            "z = {:?}",
            rec.bound_duals
        );
        assert!(stationarity_max(&p, &[1.0], &rec) < 1e-9);
    }

    /// 境界から `user_eps` より遠く `kkt_tol` より近い近似 incumbent でも、その bound
    /// の乗数を復元できる (Codex P2: active 判定が下流の受理閾値より狭いと、最終
    /// ゲートは通せるのに復元候補から外れて `FeasiblePoint` へ落ちていた)。
    ///
    /// min −x, x ∈ [0, 1], x = 1 − 1e−5 (user_eps=1e-6 の 10 倍離れている)。
    /// 手計算オラクル: g = −1、上界 active なら z_ub = 1。
    ///
    /// ## Sentinel (no-op-fail)
    /// `active_tol` を `user_eps` 基準へ戻すと上界が active と判定されず、
    /// z_ub = 0・残差 1 が残って FAIL する。
    #[test]
    fn recovers_multiplier_for_approximately_active_bound() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let p = QpProblem::new(q, vec![-1.0], a, vec![], vec![(0.0, 1.0)], vec![]).unwrap();
        let x = [1.0 - 1e-5];
        let rec = recover_duals_at_fixed_x(&p, &x, &opts(), 1e-6).expect("recovered");
        assert!(
            (rec.bound_duals[1] - 1.0).abs() < 1e-9,
            "z_ub を復元すべき: {:?}",
            rec.bound_duals
        );
        assert!(stationarity_max(&p, &x, &rec) < 1e-9);
    }

    /// KKT 点でない内点は復元不能: active set が空なので残差 |g| がそのまま残る。
    /// 「乗数を捏造しない」ことの teeth。
    #[test]
    fn interior_non_stationary_point_leaves_residual() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[], &[], &[], 0, 1).unwrap();
        let p = QpProblem::new(q, vec![-1.0], a, vec![], vec![(0.0, 10.0)], vec![]).unwrap();
        let rec = recover_duals_at_fixed_x(&p, &[5.0], &opts(), 1e-6).expect("solved");
        assert!(rec.bound_duals.iter().all(|z| z.abs() < 1e-9));
        assert!(
            (stationarity_max(&p, &[5.0], &rec) - 1.0).abs() < 1e-9,
            "残差 |g| = 1 が残るべき"
        );
    }

    /// 符号制約が効いていること: Ge 行の乗数は y ≤ 0 側にしか出せない。
    /// min x s.t. x ≥ 1 (Ge), x ∈ [−10, 10] → x* = 1, g = 1, y = −1。
    #[test]
    fn recovers_ge_row_multiplier_with_negative_sign() {
        let q = CscMatrix::new(1, 1);
        let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
        let p = QpProblem::new(
            q,
            vec![1.0],
            a,
            vec![1.0],
            vec![(-10.0, 10.0)],
            vec![ConstraintType::Ge],
        )
        .unwrap();
        let rec = recover_duals_at_fixed_x(&p, &[1.0], &opts(), 1e-6).expect("recovered");
        assert!((rec.y[0] + 1.0).abs() < 1e-9, "y = {:?}", rec.y);
        assert!(stationarity_max(&p, &[1.0], &rec) < 1e-9);
    }
}
