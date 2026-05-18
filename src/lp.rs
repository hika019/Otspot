//! LP-specific entry point. `solve_qp_with` の Q=0 dispatch から独立した
//! 入口で、user は LP を解くなら `solve_lp_with` を直接呼ぶことが望ましい。
//!
//! 設計理由 (#36):
//! - `solve_qp_with` 内部の `Q.is_zero` 分岐は LP 経路を QP entry 側に隠し、
//!   bench label 誤誘導 / LP-specific 経路の test 困難 / commit merge bug の
//!   温床になっていた (#33 で IPM-first 復元忘れ事故が顕在化)。
//! - LP / QP の責務を入口で分離する。`solve_lp_with` は LP-specific
//!   経路 (simplex + 将来的 IPM-first / crash / postsolve) を集約する。
//! - 後方互換のため `solve_qp_with` は Q=0 を内部で本 module の
//!   `solve_lp_forwarded_from_qp` に転送し、telemetry 上区別可能にする。
//!
//! Phase A は thin wrapper。Phase B で LP-specific dispatch (IPM-first 等)
//! を本 module 側に集約予定。

use crate::options::SolverOptions;
use crate::problem::{LpProblem, SolverResult};

/// LP entry call カウンタ (sentinel: LP と QP entry が想定どおり経路に乗っているか
/// 機械検証するため exposed)。
///
/// - `direct`: user / Model::solve / 他 module から `solve_lp_with` を直接呼んだ回数
/// - `forwarded_from_qp`: `solve_qp_with(Q=0)` 経由で本 module に転送された回数
///
/// 区別する理由: Model::solve の LP path が誤って `solve_qp_with` 側に
/// regression した場合、`direct` は 0 のままで `forwarded_from_qp` が増える。
/// 単一カウンタでは検知できないため 2 種に分けている。
pub mod telemetry {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub(super) static LP_DIRECT_CALLS: AtomicU64 = AtomicU64::new(0);
    pub(super) static LP_FORWARDED_FROM_QP_CALLS: AtomicU64 = AtomicU64::new(0);

    pub fn lp_direct_calls() -> u64 {
        LP_DIRECT_CALLS.load(Ordering::Relaxed)
    }

    pub fn lp_forwarded_from_qp_calls() -> u64 {
        LP_FORWARDED_FROM_QP_CALLS.load(Ordering::Relaxed)
    }

    /// 全 LP カウンタを 0 にする (test sentinel での pre-condition セット用)。
    pub fn reset() {
        LP_DIRECT_CALLS.store(0, Ordering::Relaxed);
        LP_FORWARDED_FROM_QP_CALLS.store(0, Ordering::Relaxed);
    }
}

/// LP-specific entry。`LpProblem` をそのまま受け取り simplex に委譲する。
///
/// QP entry (`crate::qp::solve_qp_with`) との違い:
/// - 入力型が `LpProblem` (= Q を持たない) で誤用余地が無い
/// - LP-specific options / 経路 (simplex method, crash, presolve) のみ
/// - bench label / log で "LP" と明示できる
pub fn solve_lp_with(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    telemetry::LP_DIRECT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::simplex::solve_with(problem, options)
}

/// `solve_qp_with(Q=0)` から内部転送される LP solve。`solve_lp_with` と同じ
/// 計算を行うが telemetry counter が別 (forward 経路を識別可能)。
pub(crate) fn solve_lp_forwarded_from_qp(problem: &LpProblem, options: &SolverOptions) -> SolverResult {
    telemetry::LP_FORWARDED_FROM_QP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    crate::simplex::solve_with(problem, options)
}
