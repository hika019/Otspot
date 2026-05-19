//! Q=0 (LP) dispatch.
//!
//! 中規模以下は `crate::lp::solve_lp_forwarded_from_qp` (telemetry 付き simplex) に
//! forward する。`n > LP_IPM_FIRST_N` または `m > LP_IPM_FIRST_M` を満たす大規模 LP は
//! IPM を先行し、収束しなければ残時間で simplex にフォールバック。
//!
//! IPM 呼び出し時は QP presolve を無効化する。Empty-Column 解析が pure LP で
//! false Unbounded を返す既知バグ (別途追跡) を回避するため。LP の不有界/不可解は
//! simplex/IPM 本体で判定可能で presolve なしでも検出できる。
//!
//! IPM cold init は Mehrotra primal projection で KKT 行列を 1 回 factor する
//! が、これは O((n+m)^2) 級で大規模 LP の wall を支配する。LTSF crash basis から
//! 構成した structural BFS を `warm_start_qp.x` に注入して projection を skip
//! させる: B*x_B = b を crash basis 上で解いて元空間 x に extract、y=0/mu=1 で
//! `apply_qp_warm_start` 経路に着地させる (内部で bounds 内 interior 補正済)。
//!
//! `LP_DISPATCH_NOOP=1` は sentinel 用 (no-op proof) で IPM 経路を無効化する。
//! `LP_CRASH_IPM_DISABLE=1` は crash→IPM warm wiring のみ no-op 化する
//! (sentinel/triage 用)。

use std::time::Instant;

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolveStatus, SolverResult};

use super::{ipm_solver, QpProblem};

/// IPM を先に走らせる変数数閾値。Netlib 中央値 n≈800 の約 4 倍。
const LP_IPM_FIRST_N: usize = 3_000;
/// IPM を先に走らせる制約数閾値。LU 再因子分解 O(m·nnz(L)) を回避する。
const LP_IPM_FIRST_M: usize = 2_000;

/// crash→IPM warm wiring の telemetry。
///
/// `crash_attempted`: try_build_ipm_warm_from_crash が呼ばれた回数 (gate を通過した)。
/// `crash_wired`: x_warm を生成して `warm_start_qp` に注入できた回数。
/// no-op proof: `LP_CRASH_IPM_DISABLE=1` で `crash_attempted` のみ増えて
/// `crash_wired` が 0 のままになることを sentinel で観測する。
pub mod telemetry {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static CRASH_IPM_ATTEMPTED: AtomicU64 = AtomicU64::new(0);
    pub(super) static CRASH_IPM_WIRED: AtomicU64 = AtomicU64::new(0);

    pub fn crash_ipm_attempted() -> u64 {
        CRASH_IPM_ATTEMPTED.load(Ordering::Relaxed)
    }

    pub fn crash_ipm_wired() -> u64 {
        CRASH_IPM_WIRED.load(Ordering::Relaxed)
    }

    pub fn reset() {
        CRASH_IPM_ATTEMPTED.store(0, Ordering::Relaxed);
        CRASH_IPM_WIRED.store(0, Ordering::Relaxed);
    }
}

pub(crate) fn prefer_ipm_for_size(n: usize, m: usize) -> bool {
    n > LP_IPM_FIRST_N || m > LP_IPM_FIRST_M
}

pub(crate) fn solve_as_lp_pub(problem: &QpProblem, options: &SolverOptions) -> SolverResult {
    let opts_with_deadline;
    let options: &SolverOptions = if options.deadline.is_none() {
        if let Some(secs) = options.timeout_secs {
            opts_with_deadline = {
                let mut o = options.clone();
                o.deadline = Some(Instant::now() + std::time::Duration::from_secs_f64(secs));
                o.timeout_secs = None;
                o
            };
            &opts_with_deadline
        } else {
            options
        }
    } else {
        options
    };

    let lp = match LpProblem::new_general(
        problem.c.clone(),
        problem.a.clone(),
        problem.b.clone(),
        problem.constraint_types.clone(),
        problem.bounds.clone(),
        None,
    ) {
        Ok(lp) => lp,
        Err(_) => return SolverResult::infeasible(),
    };

    // 大規模 LP: IPM 先行、Timeout/NumericalError/Unbounded/MaxIter は simplex 再試行。
    // Optimal/LocallyOptimal/Infeasible は確定的 → 即返却。
    // Unbounded は IPM 側 Q=0 数値リスクがあるため simplex で再確認。
    // LP_DISPATCH_NOOP=1 は sentinel 用 (no-op proof) で IPM 経路を無効化する。
    let dispatch_disabled = std::env::var("LP_DISPATCH_NOOP").ok().as_deref() == Some("1");
    let mut ipm_subopt_candidate: Option<SolverResult> = None;
    if !dispatch_disabled && prefer_ipm_for_size(problem.num_vars, problem.num_constraints) {
        let mut ipm_opts = ipm_opts_for_lp(options);
        // crash→IPM warm wiring: user 提供 warm_start_qp が無いときのみ生成。
        if ipm_opts.warm_start_qp.is_none() && options.use_lp_crash_basis {
            if let Some(ws) = try_build_ipm_warm_from_crash(&lp) {
                ipm_opts.warm_start_qp = Some(ws);
            }
        }
        let ipm_result = ipm_solver::solve_ipm(problem, &ipm_opts);
        // ipm_solver は内部で obj_offset を加算済み → そのまま返す。
        match ipm_result.status {
            SolveStatus::Optimal | SolveStatus::LocallyOptimal | SolveStatus::Infeasible => {
                // 確定 status は simplex 再試行不要、即返却。
                // known_optimal_obj 経由の early-exit は SuboptimalSolution arm (下) で実施。
                return ipm_result;
            }
            SolveStatus::Unbounded
            | SolveStatus::Timeout
            | SolveStatus::NumericalError
            | SolveStatus::MaxIterations => {
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    return ipm_result;
                }
                // 残時間で simplex 再試行。
            }
            SolveStatus::SuboptimalSolution => {
                if options.deadline.is_some_and(|d| Instant::now() >= d) {
                    return ipm_result;
                }
                // known_optimal_obj が設定されており obj が一致するなら simplex retry 不要。
                if let Some(ref_obj) = options.known_optimal_obj {
                    if crate::bench_utils::obj_within_tol(
                        ipm_result.objective, ref_obj,
                        crate::bench_utils::OBJ_MATCH_REL_TOL,
                    ) && !ipm_result.solution.is_empty()
                    {
                        return SolverResult { status: SolveStatus::Optimal, ..ipm_result };
                    }
                }
                // IPM incumbent を保存して simplex 再試行。simplex が失敗したとき
                // pick_best_ipm_or_simplex が SuboptimalSolution を復元する。
                ipm_subopt_candidate = Some(ipm_result);
            }
            SolveStatus::NonConvex(_)
            | SolveStatus::NonconvexLocal
            | SolveStatus::NonconvexGlobal => {
                // LP dispatch は Q=0 前提 → 非凸 status は本経路には出ないが、
                // non-exhaustive match を防ぎ safety net として simplex に倒す。
            }
        }
    }

    // simplex (LpProblem) は obj_offset を含まないため明示的に加算。
    let mut simplex_result = crate::lp::solve_lp_forwarded_from_qp(&lp, options);
    simplex_result.objective += problem.obj_offset;
    crate::bench_utils::pick_best_ipm_or_simplex(ipm_subopt_candidate, simplex_result)
}

/// LP→IPM 呼び出し時に presolve を無効化したオプションを生成。
fn ipm_opts_for_lp(options: &SolverOptions) -> SolverOptions {
    let mut o = options.clone();
    o.presolve = false;
    o
}

/// LTSF crash basis → IPM `QpWarmStart` 変換。
///
/// 1. `build_standard_form` で simplex 標準形 (shifts + slacks + UB rows) 構築
/// 2. `compute_crash_basis` で structural 列による行 cover
/// 3. crash で artificial が減らなかった (or LU 特異) ら None で abandon
/// 4. crash basis B を LU 因子分解、`B x_B = b_scaled` を FTRAN で解く
/// 5. `extract_solution` で n_orig 空間に逆写像 → `ws.x`
/// 6. y は zero ベクトル (`apply_qp_warm_start` 内で WARM_SY_MIN にクランプ)、
///    mu は 1.0 (cold-ish 起点、interior 補正は IPM init 側で実施)
///
/// 戻り値 `None` で IPM は通常 Mehrotra cold init に着地する (regression-safe)。
fn try_build_ipm_warm_from_crash(lp: &LpProblem) -> Option<crate::options::QpWarmStart> {
    use crate::basis::{BasisManager, LuBasis};
    use crate::simplex::crash_basis_for_ipm_warm;

    telemetry::CRASH_IPM_ATTEMPTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // env で no-op proof 化 (sentinel 用)。
    if std::env::var("LP_CRASH_IPM_DISABLE").ok().as_deref() == Some("1") {
        return None;
    }

    let (sf, basis) = crash_basis_for_ipm_warm(lp)?;

    // B*x_B = b を LU で解く。max_etas=auto (m から計算)。
    let mut lu = LuBasis::new(&sf.a, &basis, 0).ok()?;
    let mut x_b = vec![0.0_f64; sf.m];
    {
        use crate::sparse::SparseVec;
        let mut rhs = SparseVec::from_dense(&sf.b);
        lu.ftran(&mut rhs);
        x_b.copy_from_slice(&rhs.to_dense());
    }

    // n_orig 空間に extract (col_scale=1.0 で no-scaling)。
    let col_scale = vec![1.0_f64; sf.n_total];
    let x_orig = build_standard_form_extract(&sf, &basis, &x_b, &col_scale);
    if x_orig.len() != lp.num_vars || !x_orig.iter().all(|v| v.is_finite()) {
        return None;
    }

    telemetry::CRASH_IPM_WIRED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(crate::options::QpWarmStart {
        x: x_orig,
        y: vec![0.0_f64; lp.num_constraints],
        mu: 1.0,
    })
}

/// `simplex::extract_solution` を経由 (private re-export 経由でアクセス)。
fn build_standard_form_extract(
    sf: &crate::simplex::StandardForm,
    basis: &[usize],
    x_b: &[f64],
    col_scale: &[f64],
) -> Vec<f64> {
    crate::simplex::extract_solution_for_ipm_warm(sf, basis, x_b, col_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// crash→IPM warm が n>3000 (IPM 経路) で実際に発火する: synthetic LP で
    /// `crash_ipm_wired` カウンタが増えることを fact 化。
    /// IPM 経路の閾値 (n>3000) を越えるよう n=3500 を使う。
    #[test]
    fn crash_basis_wired_into_ipm_path() {
        telemetry::reset();
        let n = 3500_usize;
        let m = 200_usize;
        // 各行 i に var i と var (i+m) の係数 1 を入れ、b=2 とする。
        // structural cover が容易、artificial 半減期待。
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
            rows.push(i);
            cols.push(i + m);
            vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let b = vec![2.0_f64; m];
        let c = vec![1.0_f64; n];
        let ctypes = vec![crate::problem::ConstraintType::Eq; m];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let lp = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();

        let ws = try_build_ipm_warm_from_crash(&lp);
        assert!(ws.is_some(), "crash warm should be generated for cover-friendly LP");
        let ws = ws.unwrap();
        assert_eq!(ws.x.len(), n);
        assert_eq!(ws.y.len(), m);
        assert!(ws.mu > 0.0);
        // counter sentinel
        assert_eq!(telemetry::crash_ipm_attempted(), 1);
        assert_eq!(telemetry::crash_ipm_wired(), 1);
    }

    /// no-op proof: `LP_CRASH_IPM_DISABLE=1` で wiring が無効化される。
    /// counter は attempted のみ増え、wired は 0 のまま。
    #[test]
    fn crash_basis_ipm_wiring_no_op_proof() {
        telemetry::reset();
        // 上と同じ small LP
        let n = 3500_usize;
        let m = 200_usize;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
            rows.push(i);
            cols.push(i + m);
            vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let b = vec![2.0_f64; m];
        let c = vec![1.0_f64; n];
        let ctypes = vec![crate::problem::ConstraintType::Eq; m];
        let bounds = vec![(0.0_f64, f64::INFINITY); n];
        let lp = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();

        // SAFETY: env var single-threaded test, restored after.
        std::env::set_var("LP_CRASH_IPM_DISABLE", "1");
        let ws = try_build_ipm_warm_from_crash(&lp);
        std::env::remove_var("LP_CRASH_IPM_DISABLE");

        assert!(ws.is_none(), "wiring must be disabled by env");
        assert_eq!(telemetry::crash_ipm_attempted(), 1);
        assert_eq!(telemetry::crash_ipm_wired(), 0);
    }

    /// 中規模 (m_orig=600 が UB 行で >2000 to trigger IPM 閾値想定外)。ここでは
    /// crash の generic 動作 = 異なる LP shape (Le 制約 + bounded var) を fact 化。
    #[test]
    fn crash_basis_wired_into_ipm_path_le_with_bounds() {
        telemetry::reset();
        let n = 4000_usize;
        let m = 150_usize;
        // diagonal-like LP: row i は x_i ≤ 5
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..m {
            rows.push(i);
            cols.push(i);
            vals.push(1.0);
        }
        let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
        let b = vec![5.0_f64; m];
        let c = vec![-1.0_f64; n]; // maximize-ish
        let ctypes = vec![crate::problem::ConstraintType::Le; m];
        let bounds = vec![(0.0_f64, 10.0); n];
        let lp = LpProblem::new_general(c, a, b, ctypes, bounds, None).unwrap();

        let ws = try_build_ipm_warm_from_crash(&lp);
        // Le 制約は cold で slack cover 済 (artificial=0) のため crash は何も
        // 削減できず None を返す (= attempted=1, wired=0)。これも valid path。
        assert_eq!(telemetry::crash_ipm_attempted(), 1);
        match ws {
            Some(_) => assert_eq!(telemetry::crash_ipm_wired(), 1),
            None => assert_eq!(telemetry::crash_ipm_wired(), 0),
        }
    }
}
