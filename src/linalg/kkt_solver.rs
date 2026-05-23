//! Common KKT-solver trait abstracting `K · u = rhs` for IPM/IPPMM Newton steps.
//! Implementations expose `factorize` + `solve` (with deadline awareness) and report
//! failures in distinct categories so the dispatcher can fall back from direct to
//! iterative methods on memory pressure or singularity.

use crate::sparse::CscMatrix;
use std::time::Instant;

/// Default per-factorisation memory budget (4 GiB). Overridable via
/// `KKT_MEMORY_BUDGET_BYTES`. Applied uniformly via `L_nnz × BYTES_PER_L_ENTRY`
/// rather than via problem-size heuristics.
const DEFAULT_MEMORY_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// LDL の L 値 1 entry あたりのバイト数。f64 = 8B + row index usize = 8B (上限見積り)。
const BYTES_PER_L_ENTRY: usize = 16;

/// 現在の memory budget (バイト) を返す。env > default の優先順位。
pub fn memory_budget_bytes() -> usize {
    std::env::var("KKT_MEMORY_BUDGET_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MEMORY_BUDGET_BYTES)
}

/// memory budget を L_nnz 上限に変換する。
pub fn max_l_nnz_from_budget() -> usize {
    memory_budget_bytes() / BYTES_PER_L_ENTRY
}

/// Failure modes from `KktSolver::solve` / `refactor`.
#[non_exhaustive]
#[derive(Debug)]
pub enum KktError {
    DeadlineExceeded,
    /// Singular or indefinite K; the caller may retry after regularisation.
    SingularOrIndefinite,
    /// Symbolic estimation predicts the factor would exceed the memory budget.
    WouldExceedMemory,
    DidNotConverge,
}

impl std::fmt::Display for KktError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KktError::DeadlineExceeded => write!(f, "KKT solver: deadline exceeded"),
            KktError::SingularOrIndefinite => write!(f, "KKT solver: singular or indefinite"),
            KktError::WouldExceedMemory => write!(f, "KKT solver: would exceed memory budget"),
            KktError::DidNotConverge => write!(f, "KKT solver: did not converge"),
        }
    }
}

impl std::error::Error for KktError {}

/// Solver abstraction for the symmetric saddle-point KKT system `K · u = rhs`.
/// `solve` is `&self` so predictor/corrector can share one factorisation; `refactor`
/// is `&mut self` because it replaces the cached factorisation.
pub trait KktSolver: Send {
    /// 1 つの右辺に対して `K · u = rhs` を解いて `sol` に書き込む。
    ///
    /// `deadline` は反復法のときだけ意味を持つ (直接法は事前因子化済みなので
    /// solve は速い)。実装は反復中に `Instant::now() >= deadline` を確認して
    /// 早期終了することを推奨。
    fn solve(
        &self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), KktError>;

    /// 行列 K を更新して再因子化する (Newton step ごとに K の値が変わる前提)。
    ///
    /// 直接法: AMD + symbolic + numeric を実行。
    /// 反復法: K の参照を更新し、必要なら前処理を再構築。
    fn refactor(
        &mut self,
        k: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), KktError>;

    /// 行列の次元 (= n + m for KKT 鞍点系)。
    fn dim(&self) -> usize;
}

/// `KktSolver` adapter around `factorize_quasidefinite_with_amd_budget`.
/// Constructing with `max_l_nnz = Some(_)` causes `refactor` to return
/// `WouldExceedMemory` when symbolic factorisation exceeds the budget.
///
/// `par` は per-call parallelism。default は `Par::Seq` (= 既存挙動)。
pub struct DirectLdl {
    factor: Option<crate::linalg::ldl::LdlFactorizationAmd>,
    n: usize,
    max_l_nnz: Option<usize>,
    par: faer::Par,
}

impl DirectLdl {
    pub fn new(n: usize) -> Self {
        Self { factor: None, n, max_l_nnz: None, par: faer::Par::Seq }
    }

    pub fn with_budget(n: usize, max_l_nnz: usize) -> Self {
        Self { factor: None, n, max_l_nnz: Some(max_l_nnz), par: faer::Par::Seq }
    }

    /// per-call parallelism を指定する。
    pub fn with_par(mut self, par: faer::Par) -> Self {
        self.par = par;
        self
    }

    pub fn from_matrix(k: &CscMatrix, deadline: Option<Instant>) -> Result<Self, KktError> {
        let mut s = Self::new(k.nrows);
        s.refactor(k, deadline)?;
        Ok(s)
    }

    pub fn l_nnz(&self) -> usize {
        self.factor.as_ref().map_or(0, |f| f.nnz_l())
    }
}

impl KktSolver for DirectLdl {
    fn solve(
        &self,
        rhs: &[f64],
        sol: &mut [f64],
        _deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        let factor = self.factor.as_ref().ok_or(KktError::SingularOrIndefinite)?;
        factor.solve(rhs, sol);
        Ok(())
    }

    fn refactor(
        &mut self,
        k: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        if k.nrows != self.n || k.ncols != self.n {
            return Err(KktError::SingularOrIndefinite);
        }
        match crate::linalg::ldl::factorize_quasidefinite_with_amd_budget_par(
            k, deadline, self.max_l_nnz, self.par,
        ) {
            Ok(f) => {
                self.factor = Some(f);
                Ok(())
            }
            Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => {
                self.factor = None;
                Err(KktError::DeadlineExceeded)
            }
            Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
                self.factor = None;
                Err(KktError::SingularOrIndefinite)
            }
            Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
                self.factor = None;
                Err(KktError::WouldExceedMemory)
            }
        }
    }

    fn dim(&self) -> usize {
        self.n
    }
}

/// Diagonally-preconditioned MINRES, used as an iterative alternative when an LDL
/// factorisation would exceed the memory budget. `BlockDiag` exploits the saddle-point
/// block structure and is typically much more effective than plain `Jacobi`.
pub struct PreconditionedMinres {
    k: CscMatrix,
    m_inv_diag: Vec<f64>,
    kind: PreconditionerKind,
    max_iter: usize,
    tol: f64,
    /// Iterative-refinement rounds: each round reapplies MINRES to the residual,
    /// driving the effective relative tolerance toward `tol^(ir_steps + 1)`. Default 0.
    ir_steps: usize,
}

/// Clamp preconditioner diagonal entries below this to avoid blowing up M^{-1}.
const MIN_DIAG: f64 = 1e-12;

#[derive(Debug, Clone, Copy)]
pub enum PreconditionerKind {
    Jacobi,
    /// Block-diagonal preconditioner for a saddle-point K = [top × top; bottom × bottom].
    BlockDiag { n_top: usize },
}

/// Tie the inexact-Newton forcing term η to the user-specified eps so the
/// inner-solve precision tracks the outer tolerance.
pub(crate) const IPM_OUTER_VS_INNER_RATIO: f64 = 0.1;
/// Floor on η — below f64 working precision the MINRES tolerance is meaningless.
pub(crate) const IPM_INEXACT_ETA_FLOOR: f64 = 1e-13;

pub fn inexact_eta_for_eps(eps: f64) -> f64 {
    (eps * IPM_OUTER_VS_INNER_RATIO).max(IPM_INEXACT_ETA_FLOOR)
}

/// Backward-compatible default η (equivalent to `inexact_eta_for_eps(1e-6)`),
/// used by callers (mostly tests) that don't have an eps in hand.
pub(crate) const MINRES_INEXACT_NEWTON_ETA: f64 = 1e-7;

/// Default IR rounds for inexact MINRES. 0 because the auto-Schur path
/// makes the saddle-point conditioning manageable in practice.
const MINRES_INEXACT_NEWTON_IR_STEPS: usize = 0;

/// Default convergence tolerance for non-inexact MINRES constructors.
/// Tighter than `MINRES_INEXACT_NEWTON_ETA` because there is no outer IPM
/// relaxation — the system must be solved accurately each call.
const MINRES_DEFAULT_TOL: f64 = 1e-9;

/// Max iterations = `MINRES_MAX_ITER_MULTIPLIER × n`, giving O(n) budget that
/// scales with problem size.
const MINRES_MAX_ITER_MULTIPLIER: usize = 2;

/// Resolve η from the `MINRES_ETA` env (constrained to `(0, 1]`), else default.
fn minres_eta_runtime() -> f64 {
    std::env::var("MINRES_ETA")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
        .unwrap_or(MINRES_INEXACT_NEWTON_ETA)
}

/// Resolve IR rounds from the `MINRES_IR` env (constrained to `0..=10`), else default.
fn minres_ir_runtime(default: usize) -> usize {
    std::env::var("MINRES_IR")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v <= 10)
        .unwrap_or(default)
}

impl PreconditionedMinres {
    /// Tighten η between outer IPM iterations (Eisenstat-Walker forcing).
    pub fn set_inexact_tol(&mut self, tol: f64) {
        self.tol = tol;
    }

    pub fn new(k: CscMatrix) -> Self {
        let kind = PreconditionerKind::Jacobi;
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self { k, m_inv_diag, kind, max_iter: MINRES_MAX_ITER_MULTIPLIER * n, tol: MINRES_DEFAULT_TOL, ir_steps: 0 }
    }

    /// Block-diagonal preconditioner for a saddle-point K of dimension `n_top + m`.
    /// Approximates the lower block with a Schur-complement diagonal in O(nnz(K)).
    pub fn with_block_diag(k: CscMatrix, n_top: usize) -> Self {
        let kind = PreconditionerKind::BlockDiag { n_top };
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self { k, m_inv_diag, kind, max_iter: MINRES_MAX_ITER_MULTIPLIER * n, tol: MINRES_DEFAULT_TOL, ir_steps: 0 }
    }

    /// Inexact-Newton variant of `with_block_diag`, with env-overridable η and IR rounds.
    pub fn with_block_diag_inexact(k: CscMatrix, n_top: usize) -> Self {
        let kind = PreconditionerKind::BlockDiag { n_top };
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self {
            k,
            m_inv_diag,
            kind,
            max_iter: 2 * n,
            tol: minres_eta_runtime(),
            ir_steps: minres_ir_runtime(MINRES_INEXACT_NEWTON_IR_STEPS),
        }
    }

    /// Inexact-Newton variant of `new` (Jacobi preconditioner).
    pub fn new_inexact(k: CscMatrix) -> Self {
        let kind = PreconditionerKind::Jacobi;
        let m_inv_diag = compute_inv_diag(&k, kind);
        let n = k.nrows;
        Self {
            k,
            m_inv_diag,
            kind,
            max_iter: 2 * n,
            tol: minres_eta_runtime(),
            ir_steps: minres_ir_runtime(MINRES_INEXACT_NEWTON_IR_STEPS),
        }
    }

    pub fn with_params(k: CscMatrix, max_iter: usize, tol: f64) -> Self {
        let kind = PreconditionerKind::Jacobi;
        let m_inv_diag = compute_inv_diag(&k, kind);
        Self { k, m_inv_diag, kind, max_iter, tol, ir_steps: 0 }
    }
}

fn compute_inv_diag(k: &CscMatrix, kind: PreconditionerKind) -> Vec<f64> {
    match kind {
        PreconditionerKind::Jacobi => compute_jacobi_inv_diag(k),
        PreconditionerKind::BlockDiag { n_top } => compute_block_diag_inv(k, n_top),
    }
}

fn compute_jacobi_inv_diag(k: &CscMatrix) -> Vec<f64> {
    let n = k.nrows;
    let mut diag_abs = vec![MIN_DIAG; n];
    for j in 0..n {
        for k_idx in k.col_ptr[j]..k.col_ptr[j + 1] {
            if k.row_ind[k_idx] == j {
                let v = k.values[k_idx].abs();
                diag_abs[j] = v.max(MIN_DIAG);
                break;
            }
        }
    }
    diag_abs.iter().map(|&d| 1.0 / d).collect()
}

/// Block-diagonal preconditioner for a saddle-point K stored as symmetric upper CSC:
///   M_top[j]    = max(|K[j,j]|, MIN_DIAG)                       for j < n_top
///   M_bottom[i] = Σ_r A[i,r]² / M_top[r] + |K[n_top+i, n_top+i]| for i < n - n_top
/// where A^T entries appear at (row r, col n_top+i) with r < n_top.
fn compute_block_diag_inv(k: &CscMatrix, n_top: usize) -> Vec<f64> {
    let n_total = k.nrows;
    debug_assert!(n_top <= n_total);
    let m_bot = n_total - n_top;

    let mut top_diag = vec![MIN_DIAG; n_top];
    for j in 0..n_top {
        for k_idx in k.col_ptr[j]..k.col_ptr[j + 1] {
            if k.row_ind[k_idx] == j {
                top_diag[j] = k.values[k_idx].abs().max(MIN_DIAG);
                break;
            }
        }
    }

    let mut bot_diag = vec![MIN_DIAG; m_bot];
    for i in 0..m_bot {
        let col = n_top + i;
        let mut accum = 0.0_f64;
        for k_idx in k.col_ptr[col]..k.col_ptr[col + 1] {
            let r = k.row_ind[k_idx];
            let val = k.values[k_idx];
            if r < n_top {
                accum += (val * val) / top_diag[r];
            } else if r == col {
                accum += val.abs();
            }
        }
        bot_diag[i] = accum.max(MIN_DIAG);
    }

    let mut m_inv = Vec::with_capacity(n_total);
    m_inv.extend(top_diag.iter().map(|&d| 1.0 / d));
    m_inv.extend(bot_diag.iter().map(|&d| 1.0 / d));
    m_inv
}

impl KktSolver for PreconditionedMinres {
    fn solve(
        &self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        for s in sol.iter_mut() { *s = 0.0; }
        let k = &self.k;
        let m_inv = &self.m_inv_diag;
        let n = k.nrows;
        let do_minres = |sol: &mut [f64], rhs: &[f64]| {
            crate::linalg::minres::pminres(
                |v, y| crate::linalg::minres::matvec_sym_upper(k, v, y),
                |r, z| {
                    for i in 0..r.len() {
                        z[i] = r[i] * m_inv[i];
                    }
                },
                rhs,
                sol,
                self.tol,
                self.max_iter,
                || deadline.is_some_and(|d| Instant::now() >= d),
            )
        };
        let stats = do_minres(sol, rhs);
        let trace = std::env::var("MINRES_TRACE").ok().as_deref() == Some("1");
        if trace {
            eprintln!("MINRES_SOLVE n={} iters={} max_iter={} tol={:.1e} resid={:.3e} conv={} kind={:?}",
                n, stats.iters, self.max_iter, self.tol, stats.residual_estimate, stats.converged, self.kind);
        }

        // Iterative refinement: each round reapplies MINRES to `r = rhs − K·sol` so the
        // effective relative residual is driven toward `tol^(ir_steps + 1)`. Bailing on
        // the deadline is safe — IR only adds to `sol`, never replaces the initial solve.
        if self.ir_steps > 0 {
            let mut residual = vec![0.0_f64; n];
            let mut delta = vec![0.0_f64; n];
            for ir_iter in 0..self.ir_steps {
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    if trace {
                        eprintln!("MINRES_IR iter={} deadline reached, abort IR", ir_iter);
                    }
                    break;
                }
                // residual = rhs - K·sol
                crate::linalg::minres::matvec_sym_upper(k, sol, &mut residual);
                let mut r_norm_sq = 0.0_f64;
                for i in 0..n {
                    residual[i] = rhs[i] - residual[i];
                    r_norm_sq += residual[i] * residual[i];
                }
                let r_norm = r_norm_sq.sqrt();
                // Saddle-point conditioning amplifies the relative residual, so only
                // bail at the f64 norm floor (1e-14), not at `tol²`.
                let rhs_norm = rhs.iter().fold(0.0_f64, |a, &v| a + v * v).sqrt();
                if rhs_norm > 0.0 && r_norm <= 1e-14 * rhs_norm {
                    if trace {
                        eprintln!("MINRES_IR iter={} early-out: r/rhs={:.3e} at f64 floor", ir_iter, r_norm / rhs_norm);
                    }
                    break;
                }
                for d in delta.iter_mut() { *d = 0.0; }
                let ir_stats = do_minres(&mut delta, &residual);
                for i in 0..n {
                    sol[i] += delta[i];
                }
                if trace {
                    eprintln!(
                        "MINRES_IR iter={} pre_r={:.3e} ir_iters={} ir_resid={:.3e} ir_conv={}",
                        ir_iter, r_norm, ir_stats.iters, ir_stats.residual_estimate, ir_stats.converged
                    );
                }
            }
        }

        if stats.converged {
            Ok(())
        } else if deadline.is_some_and(|d| Instant::now() >= d) {
            Err(KktError::DeadlineExceeded)
        } else {
            Err(KktError::DidNotConverge)
        }
    }

    fn refactor(
        &mut self,
        k: &CscMatrix,
        _deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        if k.nrows != self.dim() || k.ncols != self.dim() {
            return Err(KktError::SingularOrIndefinite);
        }
        self.m_inv_diag = compute_inv_diag(k, self.kind);
        self.k = k.clone();
        Ok(())
    }

    fn dim(&self) -> usize {
        self.k.nrows
    }
}

/// LDL-compatible factor type with enum-dispatched backends. `Direct` and `DirectDd`
/// hold factorisations (f64 / TwoFloat); `Iterative` keeps K for MINRES.
pub enum KktFactor {
    Direct(crate::linalg::ldl::LdlFactorizationAmd),
    /// TwoFloat (~106-bit) LDL used when f64's `cond × ε` exceeds the requested eps.
    DirectDd(crate::linalg::ldl_dd::LdlFactorizationDdAmd),
    Iterative(PreconditionedMinres),
}

impl KktFactor {
    /// Override the MINRES tolerance (no-op for the direct backends).
    pub fn set_iterative_tol(&mut self, tol: f64) {
        if let KktFactor::Iterative(minres) = self {
            minres.set_inexact_tol(tol);
        }
    }

    /// Infallible `K · sol = rhs` solve (mirrors `LdlFactorizationAmd::solve`).
    /// Iterative-backend errors are swallowed; the best-effort solution is left in `sol`.
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        self.solve_with_deadline(rhs, sol, None);
    }

    /// `solve` with a deadline that the iterative backend honours.
    pub fn solve_with_deadline(
        &self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) {
        match self {
            KktFactor::Direct(ldl) => ldl.solve(rhs, sol),
            KktFactor::DirectDd(ldl_dd) => ldl_dd.solve(rhs, sol),
            KktFactor::Iterative(minres) => {
                let _ = minres.solve(rhs, sol, deadline);
            }
        }
    }

    pub fn is_iterative(&self) -> bool {
        matches!(self, KktFactor::Iterative(_))
    }

    pub fn is_dd(&self) -> bool {
        matches!(self, KktFactor::DirectDd(_))
    }
}

/// LDL-compatible factorisation that falls back to MINRES when the LDL factor would
/// exceed the memory budget. Pass `n_top = Some(n)` to enable the saddle-point
/// block-diagonal MINRES preconditioner; `None` selects plain Jacobi (既存互換、
/// per-call parallelism = `Par::Seq`)。
pub fn factorize_kkt_with_cached_perm(
    k: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: usize,
    n_top: Option<usize>,
) -> Result<KktFactor, KktError> {
    factorize_kkt_with_cached_perm_par(k, perm, deadline, max_l_nnz, n_top, faer::Par::Seq)
}

/// `factorize_kkt_with_cached_perm` の per-call parallelism 指定版。
pub fn factorize_kkt_with_cached_perm_par(
    k: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: usize,
    n_top: Option<usize>,
    par: faer::Par,
) -> Result<KktFactor, KktError> {
    // IPM_DD_LDL=1 switches to TwoFloat (~106-bit) LDL for ill-conditioned systems
    // where the f64 forward error (cond × ε) would exceed the requested eps.
    if std::env::var("IPM_DD_LDL").ok().as_deref() == Some("1") {
        let trace = std::env::var("IPM_DD_LDL_TRACE").ok().as_deref() == Some("1");
        // Run the f64 symbolic factorisation to honour the same memory budget.
        // DD LDL は内部で TwoFloat (scalar) を使うため、par は f64 側 sentinel のみに伝播。
        match crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
            k, perm, deadline, Some(max_l_nnz), par,
        ) {
            Ok(_) => {
                match crate::linalg::ldl_dd::factorize_quasidefinite_with_cached_perm_dd(
                    k, perm, deadline,
                ) {
                    Ok(f) => {
                        if trace {
                            eprintln!(
                                "IPM_DD_LDL_TRACE factorize OK n={} L_nnz={}",
                                k.nrows,
                                f.nnz_l()
                            );
                        }
                        return Ok(KktFactor::DirectDd(f));
                    }
                    Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => {
                        return Err(KktError::DeadlineExceeded);
                    }
                    Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
                        return Err(KktError::SingularOrIndefinite);
                    }
                    Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
                        // Unreachable for the DD path (no budget check); fall through to MINRES.
                    }
                }
            }
            Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
                // f64-budget exceeded → DD would exceed it too; fall back to MINRES (f64).
            }
            Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => {
                return Err(KktError::DeadlineExceeded);
            }
            Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
                return Err(KktError::SingularOrIndefinite);
            }
        }
        let minres = match n_top {
            Some(n) if n <= k.nrows => PreconditionedMinres::with_block_diag_inexact(k.clone(), n),
            _ => PreconditionedMinres::new_inexact(k.clone()),
        };
        return Ok(KktFactor::Iterative(minres));
    }

    match crate::linalg::ldl::factorize_quasidefinite_with_cached_perm_budget_par(
        k, perm, deadline, Some(max_l_nnz), par,
    ) {
        Ok(f) => Ok(KktFactor::Direct(f)),
        Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
            // Budget exceeded → MINRES with inexact-Newton tolerance.
            let minres = match n_top {
                Some(n) if n <= k.nrows => PreconditionedMinres::with_block_diag_inexact(k.clone(), n),
                _ => PreconditionedMinres::new_inexact(k.clone()),
            };
            Ok(KktFactor::Iterative(minres))
        }
        Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => Err(KktError::DeadlineExceeded),
        Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
            Err(KktError::SingularOrIndefinite)
        }
    }
}

/// Pre-permuted fast path. MINRES fallback uses `unpermuted_k` since it operates on
/// the original ordering (既存互換、per-call parallelism = `Par::Seq`)。
pub fn factorize_kkt_pre_permuted(
    pre_permuted_k: &CscMatrix,
    unpermuted_k: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: usize,
    n_top: Option<usize>,
) -> Result<KktFactor, KktError> {
    factorize_kkt_pre_permuted_cached_par(
        pre_permuted_k, unpermuted_k, perm, deadline, max_l_nnz, n_top, None, faer::Par::Seq,
    )
}

/// Variant of `factorize_kkt_pre_permuted` that reuses a cached symbolic Cholesky
/// (既存互換、per-call parallelism = `Par::Seq`)。
pub fn factorize_kkt_pre_permuted_cached(
    pre_permuted_k: &CscMatrix,
    unpermuted_k: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: usize,
    n_top: Option<usize>,
    cached_symbolic: Option<std::sync::Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>>,
) -> Result<KktFactor, KktError> {
    factorize_kkt_pre_permuted_cached_par(
        pre_permuted_k, unpermuted_k, perm, deadline, max_l_nnz, n_top, cached_symbolic,
        faer::Par::Seq,
    )
}

/// `factorize_kkt_pre_permuted_cached` の per-call parallelism 指定版。
pub fn factorize_kkt_pre_permuted_cached_par(
    pre_permuted_k: &CscMatrix,
    unpermuted_k: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: usize,
    n_top: Option<usize>,
    cached_symbolic: Option<std::sync::Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>>,
    par: faer::Par,
) -> Result<KktFactor, KktError> {
    // The DD LDL path doesn't accept a pre-permuted matrix; fall back to the standard route.
    if std::env::var("IPM_DD_LDL").ok().as_deref() == Some("1") {
        return factorize_kkt_with_cached_perm_par(
            unpermuted_k, perm, deadline, max_l_nnz, n_top, par,
        );
    }
    match crate::linalg::ldl::factorize_quasidefinite_pre_permuted_cached_par(
        pre_permuted_k, perm, deadline, Some(max_l_nnz), cached_symbolic, par,
    ) {
        Ok(f) => Ok(KktFactor::Direct(f)),
        Err(crate::linalg::ldl::LdlError::WouldExceedBudget { .. }) => {
            let minres = match n_top {
                Some(n) if n <= unpermuted_k.nrows => {
                    PreconditionedMinres::with_block_diag_inexact(unpermuted_k.clone(), n)
                }
                _ => PreconditionedMinres::new_inexact(unpermuted_k.clone()),
            };
            Ok(KktFactor::Iterative(minres))
        }
        Err(crate::linalg::ldl::LdlError::DeadlineExceeded) => Err(KktError::DeadlineExceeded),
        Err(crate::linalg::ldl::LdlError::SingularOrIndefinite) => {
            Err(KktError::SingularOrIndefinite)
        }
    }
}

impl KktFactor {
    /// Shared SymbolicCholesky for the Direct backend (None for Iterative / DirectDd).
    pub fn symbolic_arc(&self) -> Option<std::sync::Arc<faer::sparse::linalg::cholesky::SymbolicCholesky<usize>>> {
        match self {
            KktFactor::Direct(f) => Some(f.symbolic_arc()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KktBackend {
    Direct,
    Iterative,
}

/// Try the direct LDL backend first; once it reports `WouldExceedMemory`, switch
/// permanently to MINRES. Numerical / deadline failures still propagate up. The decision
/// is driven by measured `L_nnz` vs the configured memory budget rather than any
/// problem-size heuristic.
pub struct AutoKktSolver {
    n: usize,
    /// Cleared once `WouldExceedMemory` is observed so subsequent calls skip direct.
    direct: Option<DirectLdl>,
    /// Lazily constructed; only built once we need MINRES.
    iterative: Option<PreconditionedMinres>,
    last_used: Option<KktBackend>,
}

impl AutoKktSolver {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            direct: Some(DirectLdl::with_budget(n, max_l_nnz_from_budget())),
            iterative: None,
            last_used: None,
        }
    }

    pub fn with_budget(n: usize, max_l_nnz: usize) -> Self {
        Self {
            n,
            direct: Some(DirectLdl::with_budget(n, max_l_nnz)),
            iterative: None,
            last_used: None,
        }
    }

    pub fn last_backend(&self) -> Option<KktBackend> {
        self.last_used
    }
}

impl KktSolver for AutoKktSolver {
    fn solve(
        &self,
        rhs: &[f64],
        sol: &mut [f64],
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        match self.last_used {
            Some(KktBackend::Direct) => self
                .direct
                .as_ref()
                .ok_or(KktError::SingularOrIndefinite)?
                .solve(rhs, sol, deadline),
            Some(KktBackend::Iterative) => self
                .iterative
                .as_ref()
                .ok_or(KktError::SingularOrIndefinite)?
                .solve(rhs, sol, deadline),
            None => Err(KktError::SingularOrIndefinite),
        }
    }

    fn refactor(
        &mut self,
        k: &CscMatrix,
        deadline: Option<Instant>,
    ) -> Result<(), KktError> {
        if k.nrows != self.n || k.ncols != self.n {
            return Err(KktError::SingularOrIndefinite);
        }

        if let Some(direct) = self.direct.as_mut() {
            match direct.refactor(k, deadline) {
                Ok(()) => {
                    self.last_used = Some(KktBackend::Direct);
                    return Ok(());
                }
                Err(KktError::WouldExceedMemory) => {
                    // Disable the direct backend permanently once budget is exceeded.
                    self.direct = None;
                }
                Err(e) => return Err(e),
            }
        }

        if self.iterative.is_none() {
            self.iterative = Some(PreconditionedMinres::new(k.clone()));
        } else {
            self.iterative.as_mut().unwrap().refactor(k, deadline)?;
        }
        self.last_used = Some(KktBackend::Iterative);
        Ok(())
    }

    fn dim(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;

    /// DirectLdl: 単純な 2x2 quasidefinite で因子化 + solve が動くこと
    #[test]
    fn directldl_2x2_solve_matches_hand_calc() {
        // K = [ 2  1 ]
        //     [ 1 -1 ]   (quasidefinite: top SPD, bottom -SPD)
        // 上三角 CSC: (0,0)=2, (0,1)=1, (1,1)=-1
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let solver = DirectLdl::from_matrix(&k, None).expect("factorize");
        // K · u = [3, 0]^T → 解析解 u = K^{-1} [3, 0]^T
        // det(K) = 2*(-1) - 1*1 = -3
        // K^{-1} = (1/-3) * [-1 -1; -1 2] = [1/3 1/3; 1/3 -2/3]
        // u = [1, 1]^T (行 1: 1/3*3 + 1/3*0 = 1; 行 2: 1/3*3 + (-2/3)*0 = 1)
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve");
        assert!((sol[0] - 1.0).abs() < 1e-10, "u[0]≈1, got {}", sol[0]);
        assert!((sol[1] - 1.0).abs() < 1e-10, "u[1]≈1, got {}", sol[1]);
        assert_eq!(solver.dim(), 2);
        assert!(solver.l_nnz() > 0, "L should have nonzeros after factorize");
    }

    /// DirectLdl: refactor で別の K に切替えて再 solve できること
    #[test]
    fn directldl_refactor_changes_k() {
        let k1 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let k2 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0, -2.0], 2, 2).unwrap();
        let mut solver = DirectLdl::from_matrix(&k1, None).expect("factorize 1");
        let mut sol = vec![0.0; 2];

        // K1 · u = [2, -1]^T → u = [1, 1]^T
        solver.solve(&[2.0, -1.0], &mut sol, None).expect("solve 1");
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 1.0).abs() < 1e-10);

        solver.refactor(&k2, None).expect("refactor");
        // K2 · u = [4, -2]^T → u = [1, 1]^T (K2 = 2 * K1)
        solver.solve(&[4.0, -2.0], &mut sol, None).expect("solve 2");
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 1.0).abs() < 1e-10);
    }

    /// DirectLdl: 過去に経過した deadline を渡すと DeadlineExceeded を返すこと
    #[test]
    fn directldl_past_deadline_returns_deadline_exceeded() {
        let k = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let past = Instant::now() - std::time::Duration::from_secs(1);
        let result = DirectLdl::from_matrix(&k, Some(past));
        assert!(
            matches!(result, Err(KktError::DeadlineExceeded)),
            "past deadline should yield DeadlineExceeded, got {:?}",
            result.err()
        );
    }

    /// DirectLdl: 次元不整合の K を渡すと Err
    #[test]
    fn directldl_dim_mismatch_returns_err() {
        let k = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let mut solver = DirectLdl::new(3); // 3x3 を期待しているが渡される K は 2x2
        let result = solver.refactor(&k, None);
        assert!(result.is_err(), "dim mismatch should yield Err");
    }

    /// memory budget を持つ DirectLdl: 巨大 L_nnz を要求する K で WouldExceedMemory を返す
    #[test]
    fn directldl_with_tight_budget_returns_would_exceed_memory() {
        // 5x5 quasidefinite (上3 SPD, 下2 -SPD)
        // K = diag(1,1,1,-1,-1) + 少しの off-diag で fill-in
        let k = CscMatrix::from_triplets(
            &[0, 0, 1, 0, 1, 2, 3, 3, 4],
            &[0, 1, 1, 2, 2, 2, 3, 4, 4],
            &[1.0, 0.1, 1.0, 0.1, 0.1, 1.0, -1.0, 0.1, -1.0],
            5, 5,
        ).unwrap();
        // budget = 1 entry のみ → 必ず超過
        let mut solver = DirectLdl::with_budget(5, 1);
        let result = solver.refactor(&k, None);
        assert!(
            matches!(result, Err(KktError::WouldExceedMemory)),
            "tight budget (1 entry) should trigger WouldExceedMemory, got {:?}",
            result.err()
        );
    }

    /// memory budget が None (= 既定の現行互換モード) では budget 超過を起こさない
    #[test]
    fn directldl_without_budget_is_unconstrained() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver = DirectLdl::new(2);  // budget なし
        solver.refactor(&k, None).expect("no-budget refactor should always succeed for valid K");
    }

    /// memory_budget_bytes() は env 上書きが効くこと
    #[test]
    fn memory_budget_env_override() {
        // SAFETY: 並列テストでも他テストが同じ env を読まないため安全 (本テストでセット&クリア)
        // テスト並列実行で環境変数が干渉する可能性があるが、一意な値で観測する
        let unique_value = "12345678";
        std::env::set_var("KKT_MEMORY_BUDGET_BYTES", unique_value);
        let observed = memory_budget_bytes();
        std::env::remove_var("KKT_MEMORY_BUDGET_BYTES");
        assert_eq!(observed, 12345678, "env override should be respected");
        // 削除後は default に戻る
        let after_unset = memory_budget_bytes();
        assert_eq!(after_unset, 4 * 1024 * 1024 * 1024, "default = 4 GiB");
    }

    /// max_l_nnz_from_budget は budget をバイト→entry 数に変換する
    #[test]
    fn max_l_nnz_from_budget_conversion() {
        std::env::set_var("KKT_MEMORY_BUDGET_BYTES", "1600");  // 1600 / 16 = 100 entries
        let l = max_l_nnz_from_budget();
        std::env::remove_var("KKT_MEMORY_BUDGET_BYTES");
        assert_eq!(l, 100);
    }

    /// trait object として `Box<dyn KktSolver>` 経由で動くこと
    /// (将来 dispatcher が DirectLdl と MINRES を同じ Box に詰めるため)
    #[test]
    fn kkt_solver_works_as_trait_object() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let solver: Box<dyn KktSolver> = Box::new(
            DirectLdl::from_matrix(&k, None).expect("factorize"),
        );
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve via trait");
        assert!((sol[0] - 1.0).abs() < 1e-10);
        assert!((sol[1] - 1.0).abs() < 1e-10);
    }

    /// PreconditionedMinres: 対称不定値 2x2 で正解を返す
    #[test]
    fn minres_kkt_2x2_indefinite() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let solver = PreconditionedMinres::new(k);
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("MINRES solve");
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);
        assert_eq!(solver.dim(), 2);
    }

    /// PreconditionedMinres: 5x5 quasidef、DirectLdl と数値一致
    #[test]
    fn minres_kkt_5x5_matches_direct_ldl() {
        let entries = [
            (0, 0, 4.0), (0, 1, 0.5), (1, 1, 4.0), (1, 2, 0.5), (2, 2, 4.0),
            (0, 3, 0.3),
            (3, 3, -2.0), (3, 4, 0.4), (4, 4, -2.0),
        ];
        let rows: Vec<usize> = entries.iter().map(|(r, _, _)| *r).collect();
        let cols: Vec<usize> = entries.iter().map(|(_, c, _)| *c).collect();
        let vals: Vec<f64> = entries.iter().map(|(_, _, v)| *v).collect();
        let k = CscMatrix::from_triplets(&rows, &cols, &vals, 5, 5).unwrap();
        let b = vec![1.0, 2.0, -1.0, 0.5, -0.5];

        let mut x_ldl = vec![0.0; 5];
        let ldl_solver = DirectLdl::from_matrix(&k, None).unwrap();
        ldl_solver.solve(&b, &mut x_ldl, None).unwrap();

        let mut x_minres = vec![0.0; 5];
        let minres_solver = PreconditionedMinres::new(k);
        minres_solver.solve(&b, &mut x_minres, None).expect("MINRES solve");

        for i in 0..5 {
            assert!(
                (x_ldl[i] - x_minres[i]).abs() < 1e-6,
                "x[{}]: LDL={}, MINRES={}", i, x_ldl[i], x_minres[i]
            );
        }
    }

    /// PreconditionedMinres: refactor で K を更新できる
    #[test]
    fn minres_kkt_refactor() {
        let k1 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let k2 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0, -2.0], 2, 2).unwrap();
        let mut solver = PreconditionedMinres::new(k1);
        let mut sol = vec![0.0; 2];

        solver.solve(&[2.0, -1.0], &mut sol, None).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);

        solver.refactor(&k2, None).expect("refactor");
        solver.solve(&[4.0, -2.0], &mut sol, None).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);
    }

    /// PreconditionedMinres: 過去 deadline で DeadlineExceeded
    #[test]
    fn minres_kkt_past_deadline() {
        let k = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let solver = PreconditionedMinres::new(k);
        let mut sol = vec![0.0; 2];
        let past = Instant::now() - std::time::Duration::from_secs(1);
        let result = solver.solve(&[1.0, 1.0], &mut sol, Some(past));
        assert!(
            matches!(result, Err(KktError::DeadlineExceeded)),
            "past deadline should yield DeadlineExceeded, got {:?}", result.err()
        );
    }

    /// PreconditionedMinres: trait object として動く (DirectLdl と同じ box に詰められる)
    #[test]
    fn minres_kkt_works_as_trait_object() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let solver: Box<dyn KktSolver> = Box::new(PreconditionedMinres::new(k));
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve via trait");
        assert!((sol[0] - 1.0).abs() < 1e-7);
        assert!((sol[1] - 1.0).abs() < 1e-7);
    }

    /// AutoKktSolver: budget 十分なら direct を選ぶ
    #[test]
    fn auto_uses_direct_when_budget_sufficient() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        // budget 10000 entries (2x2 では絶対に超過しない)
        let mut solver = AutoKktSolver::with_budget(2, 10000);
        solver.refactor(&k, None).expect("refactor");
        assert_eq!(solver.last_backend(), Some(KktBackend::Direct));

        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).expect("solve");
        assert!((sol[0] - 1.0).abs() < 1e-9);
        assert!((sol[1] - 1.0).abs() < 1e-9);
    }

    /// AutoKktSolver: budget 不足なら iterative にフォールバック
    #[test]
    fn auto_falls_back_to_iterative_when_budget_exceeded() {
        let k = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[0, 1, 1, 2, 2, 3, 3, 4, 4],
            &[4.0, 0.5, 4.0, 0.5, 4.0, 0.3, -2.0, 0.4, -2.0],
            5, 5,
        ).unwrap();
        // budget 1 entry → 必ず超過 → iterative にフォールバック
        let mut solver = AutoKktSolver::with_budget(5, 1);
        solver.refactor(&k, None).expect("refactor (iterative)");
        assert_eq!(solver.last_backend(), Some(KktBackend::Iterative));

        let b = vec![1.0, 2.0, -1.0, 0.5, -0.5];
        let mut sol = vec![0.0; 5];
        solver.solve(&b, &mut sol, None).expect("solve");
        // LDL 解と比較
        let factor = crate::linalg::ldl::factorize_quasidefinite_with_amd(&k, None).unwrap();
        let mut sol_ldl = vec![0.0; 5];
        factor.solve(&b, &mut sol_ldl);
        for i in 0..5 {
            assert!(
                (sol[i] - sol_ldl[i]).abs() < 1e-6,
                "auto[{}]={} vs ldl[{}]={}", i, sol[i], i, sol_ldl[i]
            );
        }
    }

    /// AutoKktSolver: 一度 budget 超過したら以降の refactor も iterative を維持
    #[test]
    fn auto_remembers_iterative_after_first_overflow() {
        let k1 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[2.0, -1.0], 2, 2).unwrap();
        let k2 = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[4.0, -2.0], 2, 2).unwrap();
        let mut solver = AutoKktSolver::with_budget(2, 1);  // 必ず超過
        solver.refactor(&k1, None).unwrap();
        assert_eq!(solver.last_backend(), Some(KktBackend::Iterative));
        solver.refactor(&k2, None).unwrap();
        assert_eq!(solver.last_backend(), Some(KktBackend::Iterative),
            "should stay iterative after first overflow");
    }

    /// factorize_kkt_with_cached_perm: budget 十分なら Direct を返す
    #[test]
    fn factorize_kkt_chooses_direct_when_budget_sufficient() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let perm = crate::linalg::amd::amd_with_deadline(2, &k.col_ptr, &k.row_ind, None);
        let factor = factorize_kkt_with_cached_perm(&k, &perm, None, 1000, None)
            .expect("factor should succeed");
        assert!(matches!(factor, KktFactor::Direct(_)));
        assert!(!factor.is_iterative());

        let mut sol = vec![0.0; 2];
        factor.solve(&[3.0, 0.0], &mut sol);
        assert!((sol[0] - 1.0).abs() < 1e-9);
        assert!((sol[1] - 1.0).abs() < 1e-9);
    }

    /// factorize_kkt_with_cached_perm: budget 不足なら Iterative を返す
    #[test]
    fn factorize_kkt_chooses_iterative_when_budget_exceeded() {
        let k = CscMatrix::from_triplets(
            &[0, 0, 1, 1, 2], &[0, 1, 1, 2, 2],
            &[4.0, 0.5, 4.0, 0.5, -2.0], 3, 3
        ).unwrap();
        let perm = crate::linalg::amd::amd_with_deadline(3, &k.col_ptr, &k.row_ind, None);
        let factor = factorize_kkt_with_cached_perm(&k, &perm, None, 1, None)
            .expect("factor should succeed (fallback)");
        assert!(matches!(factor, KktFactor::Iterative(_)));
        assert!(factor.is_iterative());

        let b = vec![1.0, 2.0, -1.0];
        let mut sol = vec![0.0; 3];
        factor.solve(&b, &mut sol);
        // LDL 直接で解いた解と比較。dispatcher は inexact Newton tol η=0.1 を使うため
        // 厳密一致はしない。η * ||b|| の許容で確認。
        let factor_ldl = crate::linalg::ldl::factorize_quasidefinite_with_amd(&k, None).unwrap();
        let mut sol_ldl = vec![0.0; 3];
        factor_ldl.solve(&b, &mut sol_ldl);
        let b_inf = b.iter().map(|v: &f64| v.abs()).fold(0.0_f64, f64::max);
        let tol = MINRES_INEXACT_NEWTON_ETA * b_inf;
        for i in 0..3 {
            assert!(
                (sol[i] - sol_ldl[i]).abs() < tol.max(1e-6),
                "MINRES[{}]={} vs LDL[{}]={} (tol={})", i, sol[i], i, sol_ldl[i], tol
            );
        }
    }

    /// MINRES IR が機能しているか単体で検証する。
    ///
    /// η = 0.1 (relative tol) で 1 回 solve した時の relative residual が
    /// 1 IR ラウンドあたり η 倍 (= 10x reduction) するか測定する。
    /// 理論: ||r_k+1|| ≤ η × ||r_k|| (Wilkinson 1965)。
    /// もし IR で reduction が出ていなければバグ。
    #[test]
    fn minres_ir_actually_reduces_residual() {
        // 中サイズ ill-conditioned saddle (n=20 + m=10)。Q diagonal、A 密。
        let n = 20usize;
        let m = 10usize;
        let dim = n + m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        // Upper triangle of K = [diag(Q+ρ),  A^T;  A, -diag(σ+δ)]
        // Q diag values vary (ill-cond):
        for i in 0..n {
            rows.push(i); cols.push(i);
            vals.push(1.0 + 1e-3 * (i as f64).sqrt());
        }
        // A^T entries (col = n..n+m, row < n)
        for k in 0..m {
            for j in 0..n {
                let v = ((k * 7 + j * 13) % 17) as f64 / 17.0 - 0.5;
                if v.abs() > 0.1 {
                    rows.push(j); cols.push(n + k); vals.push(v);
                }
            }
        }
        // -S diag
        for i in 0..m {
            rows.push(n + i); cols.push(n + i); vals.push(-1e-6 * (1.0 + i as f64));
        }
        let k = CscMatrix::from_triplets(&rows, &cols, &vals, dim, dim).unwrap();

        let rhs: Vec<f64> = (0..dim).map(|i| ((i * 11) % 7) as f64 - 3.0).collect();
        let rhs_norm = rhs.iter().fold(0.0_f64, |a, &v| a + v * v).sqrt();

        // η = 0.1 で IR=0 (基準)
        let mut solver_no_ir = PreconditionedMinres::with_block_diag(k.clone(), n);
        solver_no_ir.tol = 0.1;
        solver_no_ir.ir_steps = 0;
        let mut sol_no_ir = vec![0.0_f64; dim];
        let _ = solver_no_ir.solve(&rhs, &mut sol_no_ir, None);
        let mut residual_no_ir = vec![0.0_f64; dim];
        crate::linalg::minres::matvec_sym_upper(&k, &sol_no_ir, &mut residual_no_ir);
        let r_no_ir: f64 = (0..dim)
            .map(|i| (rhs[i] - residual_no_ir[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel_no_ir = r_no_ir / rhs_norm;

        // η = 0.1 で IR=2 (理論上 η^3 = 1e-3 まで)
        let mut solver_ir2 = PreconditionedMinres::with_block_diag(k.clone(), n);
        solver_ir2.tol = 0.1;
        solver_ir2.ir_steps = 2;
        let mut sol_ir2 = vec![0.0_f64; dim];
        let _ = solver_ir2.solve(&rhs, &mut sol_ir2, None);
        let mut residual_ir2 = vec![0.0_f64; dim];
        crate::linalg::minres::matvec_sym_upper(&k, &sol_ir2, &mut residual_ir2);
        let r_ir2: f64 = (0..dim)
            .map(|i| (rhs[i] - residual_ir2[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel_ir2 = r_ir2 / rhs_norm;

        eprintln!(
            "MINRES IR check: n={} dim={} ||rhs||={:.3e} no_ir_rel={:.3e} ir2_rel={:.3e} ratio={:.3e}",
            n, dim, rhs_norm, rel_no_ir, rel_ir2, rel_ir2 / rel_no_ir.max(1e-300)
        );

        // IR=2 は IR=0 より少なくとも 5x reduce すべき (理論 η^2 = 100x、控えめ)
        assert!(
            rel_ir2 < rel_no_ir / 5.0,
            "MINRES IR is not reducing residual: no_ir={:.3e} ir2={:.3e} (ratio {:.2e}, expected < 0.2)",
            rel_no_ir, rel_ir2, rel_ir2 / rel_no_ir.max(1e-300)
        );
    }

    /// AutoKktSolver: trait object として動く (IPM 配線で使う形)
    #[test]
    fn auto_works_as_trait_object() {
        let k = CscMatrix::from_triplets(&[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, -1.0], 2, 2).unwrap();
        let mut solver: Box<dyn KktSolver> = Box::new(AutoKktSolver::new(2));
        solver.refactor(&k, None).unwrap();
        let mut sol = vec![0.0; 2];
        solver.solve(&[3.0, 0.0], &mut sol, None).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-9);
        assert!((sol[1] - 1.0).abs() < 1e-9);
    }
}
