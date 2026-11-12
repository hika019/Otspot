//! P0 求解診断: 全 attempt とフェーズの時間帰属モデル (内部 / dev-bin 可視)。
//!
//! 公開求解契約 (`SolveStatus` / `Tolerance` / `TimingBreakdown`) は一切変更
//! しない。ここの型は「実際に実行した全 attempt」の排他的な時間配分と、
//! 回数 (count) を時間合計から完全に分離したカウンタを保持する。
//!
//! # 設計不変条件 (implementation plan §3)
//!
//! - トップレベルの排他的フェーズ: `presolve / attempts / final_verification /
//!   other`。attempt 内部: `init / ipm / postsolve / verification / other`。
//!   ipm 内部: `assembly / factor / probe / linear_solve / residual_update /
//!   other`。親と子を同時に合計に入れない。
//! - `other = parent − Σ(exclusive children)` を符号付きで保持し、負値を
//!   黙って 0 に丸めない (負値は集計不整合として `is_aggregation_inconsistent`
//!   で検出できる)。
//! - 回数 (`ipm_reg_retries` 等の count) は時間フィールドと同じ合計に決して
//!   入れない。count は `AttemptCounts` に隔離され、時間の集計関数は
//!   `IpmPhaseTimes` / 時間フィールドのみを参照する。
//! - 収集はフェーズ境界のみ。要素ループ内の時計読み出しはない。
//!
//! # 伝達経路
//!
//! 公開戻り値型 (`SolverResult`) を変更しないため、試行単位の収集は
//! thread-local の sink を経由する。solve 本体は scoped thread 上で動くため、
//! `solve_qp_with` が完了時にスナップショットを caller スレッドへ引き継ぎ、
//! `last_qp_solve_diagnostics` で読む。dev bin は「1 solve 直後に読む」逐次
//! 使用を前提とする。多 solve 並列時の対応は未定義。
//! threads>1 のプール境界で TLS は自動伝播しない (install は worker へ inject、
//! 収集は `run_diagnostics_collected` / `merge_worker_attempt_sink` の doc 参照)。

use std::cell::RefCell;
use std::time::Instant;

use crate::qp::ipm_solver::outcome::IpmTermination;

/// IPM 内部ループの排他的フェーズ内訳 (すべて累計 µs)。
///
/// 子は `assembly / factor / probe / linear_solve / residual_update`、
/// 親は `total_us` (IPM 反復ループ全体の wall)。[`Self::other_us`] が
/// 親 − Σ子 を符号付きで返す。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IpmPhaseTimes {
    /// KKT 行列の組み立ての累計: 元順序 materialize / 置換済み materialize /
    /// Schur 構築 / AMD / 初回 permute キャッシュ構築をすべて排他的に計上する
    /// (review R1: pre-permuted 経路の置換済み構築は factor 窓ではなくここ)。
    pub assembly_us: u64,
    /// 数値因子化 wrapper の呼出窓 (全正則化 retry・identity fallback 含む)。
    /// pre-permuted 経路では対角符号抽出・CSC→faer 変換・初回のみ symbolic
    /// 解析 (以降は `symbolic_cholesky` キャッシュ再利用)・数値 LDL・budget
    /// 超過時の MINRES 構築を含むが置換コピーは含まない。Schur / DD /
    /// identity-fallback の `factorize_kkt_with_cached_perm_par` は wrapper 内
    /// で `permute_sym_upper` するためこの経路の factor 窓は置換コピーを
    /// 含む (otspot-num の API 変更なしには窓分割は不可、境界差はここに明記)。
    /// 公開 `TimingBreakdown.ipm_factorize_us` は同一窓の ns 集計を IPM 戻り値で
    /// 返すもので意味が近く、attempt 単位の当フィールドとは採用/棄却の範囲が
    /// 異なる (親子合計・公開値との混在はしない)。
    pub factor_us: u64,
    /// `probe_ldl_health` 健全性プローブの累計。
    pub probe_us: u64,
    /// predictor / corrector / Gondzio の KKT solve 窓の累計。
    pub linear_solve_us: u64,
    /// 反復先頭の spmv 群・残差更新・収束測定の前処理累計。
    pub residual_update_us: u64,
    /// IPM 反復ループ全体の wall (親)。
    pub total_us: u64,
}

impl IpmPhaseTimes {
    /// 排他的な子の合計 (total を含まない — 親と子を足さない)。
    pub fn exclusive_children_sum_us(&self) -> u64 {
        self.assembly_us
            + self.factor_us
            + self.probe_us
            + self.linear_solve_us
            + self.residual_update_us
    }

    /// `total − Σ(exclusive children)`。負値は丸めずそのまま返す
    /// (計測の重複は丸めで隠さず不整合として扱う)。
    pub fn other_us(&self) -> i64 {
        self.total_us as i64 - self.exclusive_children_sum_us() as i64
    }

    /// 子の合計が親 (total) を超えたか (集計不整合)。
    pub fn is_aggregation_inconsistent(&self) -> bool {
        self.other_us() < 0
    }
}

/// attempt 単位の呼び出し回数。時間フィールドとは別に保持し、
/// いかなる µs 合計にも加えない (集計バグ級の回帰防止)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttemptCounts {
    /// この attempt の IPM 反復数 (inner loop が実際に消費した反復)。
    pub ipm_iterations: u64,
    /// KKT 正則化 retry 回数 (health probe 失敗由来を含む既存課金と同一定義)。
    pub kkt_reg_retries: u64,
    /// `probe_ldl_health` 呼び出し回数。
    pub kkt_health_probes: u64,
    /// `factorize_kkt_*` の試行回数 (retry 1 回 = 1 試行)。
    pub kkt_factorize_attempts: u64,
    /// `AugmentedKktCache::materialize` (元順序構築) 呼び出し回数。pre-permuted
    /// 経路では 1 factorize 試行につき元順序 + 置換済みの 2 構築が発生する
    /// (review R1: 構築点を別カウントにし、総計は [`Self::materialize_total_calls`]
    /// で導出する。経路混在を無視した固定倍率は取らない)。
    pub materialize_original_calls: u64,
    /// `PermutedAugmentedKkt::materialize` (置換済み構築) 呼び出し回数。
    /// pre-permuted 経路でのみ増加し、Schur / DD-LDL では 0 のまま。
    pub materialize_permuted_calls: u64,
    /// `build_schur_system` 呼び出し回数。
    pub kkt_schur_builds: u64,
    /// PSD 判定 (`is_q_psd_by_cholesky`) 呼び出し回数 (プロセスカウンタの差分)。
    pub psd_checks: u64,
    /// `CscMatrix::transpose` 構築回数 (プロセスカウンタの差分)。
    pub transpose_constructions: u64,
    /// 後処理 projected-gradient 呼び出し回数。
    pub postsolve_pg_iterations: u64,
    /// 後処理 y-LSQ IRLS 外周反復回数。
    pub postsolve_lsq_iterations: u64,
    /// 後処理 saddle-point Krylov IR 外周反復回数。
    pub postsolve_ir_iterations: u64,
    /// 後処理 refit ループ外周反復回数。
    pub postsolve_refit_iterations: u64,
    /// 後処理 IRLS ループ外周反復回数。
    pub postsolve_irls_iterations: u64,
}

impl AttemptCounts {
    /// 同名フィールドの加算集計 (診断 Vec の総計用)。count 専用 — 時間と
    /// 混在させる場所はない。
    pub fn accumulate(&mut self, other: &Self) {
        self.ipm_iterations += other.ipm_iterations;
        self.kkt_reg_retries += other.kkt_reg_retries;
        self.kkt_health_probes += other.kkt_health_probes;
        self.kkt_factorize_attempts += other.kkt_factorize_attempts;
        self.materialize_original_calls += other.materialize_original_calls;
        self.materialize_permuted_calls += other.materialize_permuted_calls;
        self.kkt_schur_builds += other.kkt_schur_builds;
        self.psd_checks += other.psd_checks;
        self.transpose_constructions += other.transpose_constructions;
        self.postsolve_pg_iterations += other.postsolve_pg_iterations;
        self.postsolve_lsq_iterations += other.postsolve_lsq_iterations;
        self.postsolve_ir_iterations += other.postsolve_ir_iterations;
        self.postsolve_refit_iterations += other.postsolve_refit_iterations;
        self.postsolve_irls_iterations += other.postsolve_irls_iterations;
    }

    /// KKT 行列構築の総計 (元順序 + 置換済み)。経路が混在しても構築点で
    /// カウントしているため、固定倍率の推測は不要 (review R1 検証条件)。
    pub fn materialize_total_calls(&self) -> u64 {
        self.materialize_original_calls + self.materialize_permuted_calls
    }
}

/// attempt の発生箇所。ladder の 1 attempt か、cap 打ち切り attempt の延長か、
/// no-presolve fallback かを区別する (延長・fallback は採用診断からこぼれて
/// よらない対象 — plan §3.2)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptKind {
    /// eps-tighten / Ruiz ladder の 1 attempt。
    Ladder,
    /// IterationLimit で切られた attempt の 1 回だけ許された延長。
    Extension,
    /// presolve なしで元問題を直接解く fallback attempt。
    NoPresolveFallback,
}

impl std::fmt::Display for AttemptKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttemptKind::Ladder => write!(f, "ladder"),
            AttemptKind::Extension => write!(f, "ext"),
            AttemptKind::NoPresolveFallback => write!(f, "fallback"),
        }
    }
}

/// 1 attempt の排他的時間配分 + 回数 + 採用事実。
///
/// 親は `wall_us` (runner 呼び出しの wall 全体)。子は
/// `init / ipm.total / postsolve / verification` で、差分が
/// [`Self::other_us`]。失敗・timeout・未採用 attempt もこの形で全部残る。
#[derive(Clone, Debug)]
pub struct AttemptDiagnostics {
    /// solve 内の 0 始まり attempt 番号 (ladder → extension → fallback 順)。
    pub id: u64,
    /// 発生箇所 (`AttemptKind`)。
    pub kind: AttemptKind,
    /// この attempt で Ruiz scaling が有効だったか。
    pub use_ruiz: bool,
    /// inner eps = user_eps / tighten の tighten 倍率。
    pub tighten: f64,
    /// 最終結果の source になったか (finalize が採用した outcome を出した
    /// attempt のみ true)。
    pub published: bool,
    /// 内部 IPM の終端条件 (runner が outcome を返せなかった場合 None)。
    pub termination: Option<IpmTermination>,
    /// runner が報告した反復数。
    pub iterations: u64,
    /// outcome が numerical_failure を主張したか。
    pub numerical_failure: bool,
    /// outcome が Infeasible/Unbounded/NonConvex の構造判定を含んだか。
    pub infeasibility_claim: bool,
    /// runner 呼び出し全体の wall (親)。
    pub wall_us: u64,
    /// scaling + IPM 初期化 (loop 前) の累計。
    pub init_us: u64,
    /// IPM 内部ループのフェーズ内訳。
    pub ipm: IpmPhaseTimes,
    /// postsolve (map + dual refine + IR) の排他的合計。
    pub postsolve_us: u64,
    /// 元空間残差計算・証明書ゲートの累計。
    pub verification_us: u64,
    /// この attempt の回数内訳。
    pub counts: AttemptCounts,
}

impl AttemptDiagnostics {
    /// `wall − (init + ipm.total + postsolve + verification)`。負値は丸めない。
    pub fn other_us(&self) -> i64 {
        self.wall_us as i64
            - self.init_us as i64
            - self.ipm.total_us as i64
            - self.postsolve_us as i64
            - self.verification_us as i64
    }

    /// この attempt の集計に不整合 (子の合計 > 親) があるか。
    pub fn is_aggregation_inconsistent(&self) -> bool {
        self.other_us() < 0 || self.ipm.is_aggregation_inconsistent()
    }
}

/// solve 全体の排他的時間配分 + 全 attempt の診断 (採用有無に関わらず全件)。
#[derive(Clone, Debug, Default)]
pub struct SolveDiagnostics {
    /// 実際に実行した全 attempt (実行順)。未採用・timeout・延長・fallback 含む。
    pub attempts: Vec<AttemptDiagnostics>,
    /// presolve (phase1+phase2) の wall。
    pub presolve_us: u64,
    /// finalize (最終受理判定・status 変換) + 外側 inertia guard の wall。
    pub final_verification_us: u64,
    /// solve_ipm 全体の wall (親)。q-diag scaling・外側 guard も含む。
    pub total_us: u64,
}

impl SolveDiagnostics {
    /// 全 attempt の wall 合計 (attempt は逐次実行なのでこれが attempts 桶)。
    pub fn attempts_wall_sum_us(&self) -> u64 {
        self.attempts.iter().map(|a| a.wall_us).sum()
    }

    /// トップレベルの排他的 bucket 合計。count フィールドは含まない
    /// (count を µs 合計に入れない不変条件の検査対象)。
    pub fn time_bucket_sum_us(&self) -> u64 {
        self.presolve_us + self.attempts_wall_sum_us() + self.final_verification_us
    }

    /// `total − (presolve + attempts + final_verification)`。負値は丸めない。
    pub fn other_us(&self) -> i64 {
        self.total_us as i64 - self.time_bucket_sum_us() as i64
    }

    /// 集計不整合 (負の other) を含むか — 頂層と attempt/ipm の全階層。
    pub fn is_aggregation_inconsistent(&self) -> bool {
        self.other_us() < 0
            || self
                .attempts
                .iter()
                .any(|a| a.is_aggregation_inconsistent())
    }

    /// 不整合な attempt 数。
    pub fn inconsistent_attempt_count(&self) -> usize {
        self.attempts
            .iter()
            .filter(|a| a.is_aggregation_inconsistent())
            .count()
    }

    /// 全 attempt の回数集計 (count 専用 — 時間合計には使わない)。
    pub fn total_counts(&self) -> AttemptCounts {
        let mut acc = AttemptCounts::default();
        for a in &self.attempts {
            acc.accumulate(&a.counts);
        }
        acc
    }

    /// 採用 attempt (published) の id。none = 採用 attempt なし。
    pub fn published_attempt(&self) -> Option<u64> {
        self.attempts.iter().find(|a| a.published).map(|a| a.id)
    }
}

/// attempt 実行中に inner solver がへ書き込む sink (thread-local の中身)。
#[derive(Debug, Default)]
pub(crate) struct AttemptSink {
    pub init_us: u64,
    pub postsolve_us: u64,
    postsolve_stages: PostsolveStageTimes,
    pub verification_us: u64,
    pub ipm: IpmPhaseTimes,
    pub counts: AttemptCounts,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PostsolveStageTimes {
    pub map_us: u64,
    pub lsq_us: u64,
    pub recovery_us: u64,
    pub refine_us: u64,
    pub krylov_ir_us: u64,
}

impl PostsolveStageTimes {
    fn accumulate(&mut self, other: Self) {
        self.map_us += other.map_us;
        self.lsq_us += other.lsq_us;
        self.recovery_us += other.recovery_us;
        self.refine_us += other.refine_us;
        self.krylov_ir_us += other.krylov_ir_us;
    }

    pub(crate) fn total_us(self) -> u64 {
        self.map_us + self.lsq_us + self.recovery_us + self.refine_us + self.krylov_ir_us
    }
}

#[derive(Clone, Copy)]
pub(crate) enum PostsolveStage {
    Map,
    Lsq,
    Recovery,
    Refine,
    KrylovIr,
}

impl AttemptSink {
    pub(crate) fn record_materialize_original(&mut self, elapsed_us: u64) {
        self.ipm.assembly_us += elapsed_us;
        self.counts.materialize_original_calls += 1;
    }

    pub(crate) fn record_materialize_permuted(&mut self, elapsed_us: u64) {
        self.ipm.assembly_us += elapsed_us;
        self.counts.materialize_permuted_calls += 1;
    }

    pub(crate) fn record_postsolve(&mut self, stage: PostsolveStage, elapsed_us: u64) {
        self.postsolve_us += elapsed_us;
        match stage {
            PostsolveStage::Map => self.postsolve_stages.map_us += elapsed_us,
            PostsolveStage::Lsq => self.postsolve_stages.lsq_us += elapsed_us,
            PostsolveStage::Recovery => self.postsolve_stages.recovery_us += elapsed_us,
            PostsolveStage::Refine => self.postsolve_stages.refine_us += elapsed_us,
            PostsolveStage::KrylovIr => self.postsolve_stages.krylov_ir_us += elapsed_us,
        }
    }

    /// worker 側の sink を加算合併する。全フィールドが累計 µs / 回数であり、
    /// 直接書き込みと同じ終値を順序によらず作る (review R2 の回収側)。
    fn merge_from(&mut self, w: &AttemptSink) {
        self.init_us += w.init_us;
        self.postsolve_us += w.postsolve_us;
        self.postsolve_stages.accumulate(w.postsolve_stages);
        self.verification_us += w.verification_us;
        self.ipm.assembly_us += w.ipm.assembly_us;
        self.ipm.factor_us += w.ipm.factor_us;
        self.ipm.probe_us += w.ipm.probe_us;
        self.ipm.linear_solve_us += w.ipm.linear_solve_us;
        self.ipm.residual_update_us += w.ipm.residual_update_us;
        self.ipm.total_us += w.ipm.total_us;
        self.counts.accumulate(&w.counts);
    }
}

/// 現スレッドに新規 AttemptSink を設置して `f` を走らせ、その sink を結果と
/// ともに回収して返す (外側の `ACTIVE_ATTEMPT` は復元して触れない)。
///
/// review R2 の worker 側半分の境界ヘルパ。rayon-core 1.13.0 では非 worker
/// スレッドからの `ThreadPool::install` が `Registry::in_worker_cold` 経由で
/// closure をプールへ inject し latch で待つため、IPM 本体はプール worker
/// 上で走り caller の `ACTIVE_ATTEMPT` は worker から見えない (threads=2 の
/// CVXQP3_L 実測で sink 側カウンタが全 0、IPM 時間が all other へ流出するの
/// を確認)。worker 上に新規 sink を設置し、 caller スレッド側で
/// [`merge_worker_attempt_sink`] へ戻して attempt へ集約する。
///
/// 安全契約:
/// - early return / panic (unwind) でも Drop 内で sink を回収し worker TLS を
///   設置前の値 (通常 None) へ戻す。worker は再利用されるため、これを怠ると
///   次の install (別 solve) へ診断が漏れる。
/// - `f` が panic した場合、回収途中の sink は unwind とともに失われるが
///   panic はそのまま伝播し solve 自体が失敗するため、診断だけの不整合は
///   残らない (結果が成立しない solve の診断は存在してよい対象ではない)。
pub(crate) fn run_diagnostics_collected<R>(f: impl FnOnce() -> R) -> (R, AttemptSink) {
    struct Collector<'a> {
        out: &'a RefCell<Option<AttemptSink>>,
        saved: Option<AttemptSink>,
    }
    impl Drop for Collector<'_> {
        fn drop(&mut self) {
            let taken = ACTIVE_ATTEMPT.with(|c| c.borrow_mut().take());
            *self.out.borrow_mut() = taken;
            ACTIVE_ATTEMPT.with(|c| *c.borrow_mut() = self.saved.take());
        }
    }
    let saved = ACTIVE_ATTEMPT.with(|c| c.borrow_mut().replace(AttemptSink::default()));
    let out = RefCell::new(None);
    let guard = Collector { out: &out, saved };
    let r = f();
    drop(guard);
    (r, out.into_inner().unwrap_or_default())
}

/// worker で回収した sink を caller の進行中 attempt へ加算合併する。
/// 未アクティブ時は no-op (アクティブ attempt がない経路の直接書き込みと
/// 同じ意味)。
pub(crate) fn merge_worker_attempt_sink(w: AttemptSink) {
    ACTIVE_ATTEMPT.with(|c| {
        if let Some(s) = c.borrow_mut().as_mut() {
            s.merge_from(&w);
        }
    });
}

thread_local! {
    /// 進行中 solve の診断 (`SolveDiagGuard` が有効化)。
    static ACTIVE_SOLVE: RefCell<Option<SolveDiagnostics>> = const { RefCell::new(None) };
    /// 進行中 solve の全 attempt postsolve 段別累計 (公開 TimingBreakdown へ転送)。
    static ACTIVE_SOLVE_POSTSOLVE: RefCell<Option<PostsolveStageTimes>> = const { RefCell::new(None) };
    /// 進行中 attempt の sink (`begin_attempt` が有効化)。
    static ACTIVE_ATTEMPT: RefCell<Option<AttemptSink>> = const { RefCell::new(None) };
    /// 直近に完了した solve の診断スナップショット (dev-bin 読み取り用)。
    static LAST_SOLVE_DIAG: RefCell<Option<SolveDiagnostics>> = const { RefCell::new(None) };
}

/// 直近に完了した QP solve の診断スナップショットを返す。
///
/// best-effort・dev 診断専用: スレッド単位で「そのスレッドで完結した最後の
/// `solve_qp_with`」のみ追跡する。最後の solve が QP-IPM 経路以外 (LP 転送 /
/// conic / 自明 infeasible) の場合は None を返す (前 solve の値は残らない)。
/// 並列 multistart / MILP ワーカーの solve を読む保証はない (plan §3: 公開
/// 求解 API は変更しない)。
pub fn last_qp_solve_diagnostics() -> Option<SolveDiagnostics> {
    LAST_SOLVE_DIAG.with(|c| c.borrow().clone())
}

/// `last_qp_solve_diagnostics` で受け取ったスナップショットを現スレッドの
/// LAST に載せ直す。solve 本体は scoped thread 上で動くため、その TLS に載った
/// 診断はスレッド消滅前に `solve_qp_with` がここへ引き継ぐ。
///
/// `None` も無条件で載せる (直前の solve が QP-IPM 経路以外だった合図)。
/// None を握りつぶして前 solve の値を残すと、長時間生存するプロセス
/// (qps_benchmark の 1 プロセス複数問題) で「直前の QP の診断」が次の
/// 問題の出力に混線して誤帰属するため、LAST は常に見た目の solve の
/// 診断と一致させる。
pub fn publish_last_qp_solve_diagnostics(diag: Option<SolveDiagnostics>) {
    LAST_SOLVE_DIAG.with(|c| *c.borrow_mut() = diag);
}

/// 進行中 solve の診断にアクセスする (solve 計測点専用)。未アクティブ時 no-op。
pub(crate) fn with_solve_diag(f: impl FnOnce(&mut SolveDiagnostics)) {
    ACTIVE_SOLVE.with(|c| {
        if let Some(d) = c.borrow_mut().as_mut() {
            f(d);
        }
    });
}

pub(crate) fn current_solve_postsolve_stages() -> PostsolveStageTimes {
    ACTIVE_SOLVE_POSTSOLVE.with(|c| c.borrow().unwrap_or_default())
}

/// 進行中 attempt の sink にアクセスする (IPM/postsolve 計測点専用)。
/// 未アクティブ時 no-op — 計測なし経路では閉じ構築含め ns オーダー。
pub(crate) fn with_ippmm_sink(f: impl FnOnce(&mut AttemptSink)) {
    ACTIVE_ATTEMPT.with(|c| {
        if let Some(s) = c.borrow_mut().as_mut() {
            f(s);
        }
    });
}

/// 後処理 refine ループの反復 1 回をカウントする (attempt アクティブ時だけ)。
pub(crate) fn record_postsolve_iteration(field: fn(&mut AttemptCounts) -> &mut u64) {
    with_ippmm_sink(|s| *field(&mut s.counts) += 1);
}

/// solve 全体の計時ガード。drop (= 全 early return 経路含む) で診断を集計し、
/// `LAST_SOLVE_DIAG` へ格納してアクティブ状態を復元する。early return でも
/// 診断を取りこぼさないための RAII。
pub(crate) struct SolveDiagGuard {
    t0: Instant,
    saved: Option<SolveDiagnostics>,
    saved_postsolve_stages: Option<PostsolveStageTimes>,
    done: bool,
}

impl SolveDiagGuard {
    /// solve 入口で呼ぶ。ネスト時は内側が drop で外側を復元する。
    pub(crate) fn begin() -> Self {
        let saved = ACTIVE_SOLVE.with(|c| c.borrow_mut().replace(SolveDiagnostics::default()));
        let saved_postsolve_stages =
            ACTIVE_SOLVE_POSTSOLVE.with(|c| c.borrow_mut().replace(PostsolveStageTimes::default()));
        Self {
            t0: Instant::now(),
            saved,
            saved_postsolve_stages,
            done: false,
        }
    }

    fn finish_inner(&mut self) -> SolveDiagnostics {
        if self.done {
            return SolveDiagnostics::default();
        }
        self.done = true;
        let mut diag = ACTIVE_SOLVE
            .with(|c| c.borrow_mut().take())
            .unwrap_or_default();
        diag.total_us = self.t0.elapsed().as_micros() as u64;
        LAST_SOLVE_DIAG.with(|c| *c.borrow_mut() = Some(diag.clone()));
        ACTIVE_SOLVE.with(|c| *c.borrow_mut() = self.saved.take());
        ACTIVE_SOLVE_POSTSOLVE.with(|c| {
            let _ = c.borrow_mut().take();
            *c.borrow_mut() = self.saved_postsolve_stages.take();
        });
        diag
    }

    /// ガードを明示的に完了させて診断を返す (テスト/dev 向け)。
    pub(crate) fn finish(mut self) -> SolveDiagnostics {
        self.finish_inner()
    }
}

impl Drop for SolveDiagGuard {
    fn drop(&mut self) {
        // finish_inner の副作用 (LAST 格納) だけ行う。戻り値は破棄。
        let _ = self.finish_inner();
    }
}

/// 診断キャプチャ用ヘルパ: `f` をアクティブな SolveDiagnostics 下で走らせ、
/// 完了時の診断を返す (テスト・dev 診断入口)。
pub fn capture_diagnostics<R>(f: impl FnOnce() -> R) -> (R, SolveDiagnostics) {
    let guard = SolveDiagGuard::begin();
    let r = f();
    (r, guard.finish())
}

/// attempt 開始トークン。`begin_attempt` が作り `finish_attempt` が消費する。
pub(crate) struct AttemptStart {
    t0: Instant,
    pub(crate) id: u64,
    kind: AttemptKind,
    use_ruiz: bool,
    tighten: f64,
    saved_sink: Option<AttemptSink>,
    psd_base: u64,
    transpose_base: u64,
}

/// runner 呼び出し直前に呼んで attempt sink を有効化する。カウンタのベース値
/// (プロセス単調カウンタ) もここで確保する。
pub(crate) fn begin_attempt(kind: AttemptKind, use_ruiz: bool, tighten: f64) -> AttemptStart {
    let id = ACTIVE_SOLVE.with(|c| c.borrow().as_ref().map_or(0, |d| d.attempts.len() as u64));
    let saved_sink = ACTIVE_ATTEMPT.with(|c| (*c.borrow_mut()).replace(AttemptSink::default()));
    AttemptStart {
        t0: Instant::now(),
        id,
        kind,
        use_ruiz,
        tighten,
        saved_sink,
        psd_base: otspot_num::linalg::ldl::psd_check_count(),
        transpose_base: otspot_num::sparse::transpose_construction_count(),
    }
}

/// attempt が返した outcome の制御情報 (診断 record に写す最小情報)。
pub(crate) struct AttemptOutcomeBits {
    pub termination: Option<IpmTermination>,
    pub iterations: u64,
    pub numerical_failure: bool,
    pub infeasibility_claim: bool,
}

/// runner 呼び出し直後 (採用判定より前) に呼ぶ。sink を回収して
/// `AttemptDiagnostics` を組み立てる。early return・panic でない限りの
/// 全経路で呼ぶこと (呼出側が直後に record する)。
pub(crate) fn finish_attempt(start: AttemptStart, bits: AttemptOutcomeBits) -> AttemptDiagnostics {
    let sink = ACTIVE_ATTEMPT
        .with(|c| {
            let mut slot = c.borrow_mut();
            std::mem::replace(&mut *slot, start.saved_sink)
        })
        .unwrap_or_default();
    ACTIVE_SOLVE_POSTSOLVE.with(|c| {
        if let Some(total) = c.borrow_mut().as_mut() {
            total.accumulate(sink.postsolve_stages);
        }
    });
    let psd_now = otspot_num::linalg::ldl::psd_check_count();
    let transpose_now = otspot_num::sparse::transpose_construction_count();
    let mut counts = sink.counts;
    counts.psd_checks = psd_now.saturating_sub(start.psd_base);
    counts.transpose_constructions = transpose_now.saturating_sub(start.transpose_base);
    AttemptDiagnostics {
        id: start.id,
        kind: start.kind,
        use_ruiz: start.use_ruiz,
        tighten: start.tighten,
        published: false,
        termination: bits.termination,
        iterations: bits.iterations,
        numerical_failure: bits.numerical_failure,
        infeasibility_claim: bits.infeasibility_claim,
        wall_us: start.t0.elapsed().as_micros() as u64,
        init_us: sink.init_us,
        ipm: sink.ipm,
        postsolve_us: sink.postsolve_us,
        verification_us: sink.verification_us,
        counts,
    }
}

/// 完成した attempt record を進行中 solve に追記する (未アクティブ時 no-op)。
pub(crate) fn record_attempt(rec: AttemptDiagnostics) {
    with_solve_diag(|d| d.attempts.push(rec));
}

/// 最終採用 attempt の published フラグを立てる。
pub(crate) fn mark_published_attempt(id: u64) {
    with_solve_diag(|d| {
        if let Some(a) = d.attempts.iter_mut().find(|a| a.id == id) {
            a.published = true;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder_attempt(wall: i64, children: u64, counts_big: bool) -> AttemptDiagnostics {
        let mut a = AttemptDiagnostics {
            id: 0,
            kind: AttemptKind::Ladder,
            use_ruiz: false,
            tighten: 1.0,
            published: false,
            termination: Some(IpmTermination::Converged),
            iterations: 3,
            numerical_failure: false,
            infeasibility_claim: false,
            wall_us: wall.max(0) as u64,
            init_us: 10,
            ipm: IpmPhaseTimes {
                assembly_us: 20,
                factor_us: children,
                probe_us: 0,
                linear_solve_us: 0,
                residual_update_us: 0,
                total_us: 20 + children,
            },
            postsolve_us: 0,
            verification_us: 0,
            counts: AttemptCounts::default(),
        };
        if counts_big {
            a.counts.kkt_reg_retries = 1_000_000;
            a.counts.ipm_iterations = 5;
        }
        a
    }

    /// other は丸めない: 子の合計が親を超えたら負値のまま露出する。
    #[test]
    fn negative_other_is_surfaced_not_clamped() {
        let a = ladder_attempt(10, 100, false); // children 120 > wall 10
        assert!(a.other_us() < 0);
        assert!(a.is_aggregation_inconsistent());
        let mut d = SolveDiagnostics::default();
        d.attempts.push(a);
        assert!(d.is_aggregation_inconsistent());
        assert_eq!(d.inconsistent_attempt_count(), 1);
    }

    /// count (reg_retries 等) を時間 bucket 合計が決して取り込まない:
    /// 時間を 0、count を巨大にした diagnostics の時間合計は 0。
    #[test]
    fn counts_never_enter_time_sums() {
        let mut d = SolveDiagnostics::default();
        d.attempts.push(ladder_attempt(0, 0, true));
        d.attempts[0].init_us = 0;
        d.attempts[0].ipm = IpmPhaseTimes::default();
        assert_eq!(d.time_bucket_sum_us(), 0);
        assert_eq!(d.total_counts().kkt_reg_retries, 1_000_000);
    }

    /// 排他的 bucket 合計 = 親 (total) と一致する健全例では other = 0。
    #[test]
    fn exclusive_buckets_sum_to_parent_when_complete() {
        let mut a = ladder_attempt(500, 80, false);
        a.postsolve_us = 500 - 10 - (20 + 80);
        let d = SolveDiagnostics {
            attempts: vec![a],
            presolve_us: 100,
            final_verification_us: 30,
            total_us: 100 + 500 + 30,
        };
        assert_eq!(d.other_us(), 0);
        assert!(!d.is_aggregation_inconsistent());
    }

    /// capture_diagnostics はネスト時も外側の診断を復元する。
    #[test]
    fn capture_restores_outer_scope() {
        let ((), outer) = capture_diagnostics(|| {
            let ((), inner) = capture_diagnostics(|| {});
            assert!(inner.attempts.is_empty());
            record_attempt(ladder_attempt(50, 10, false));
        });
        assert_eq!(outer.attempts.len(), 1);
    }

    #[test]
    fn materialize_durations_are_exclusive_assembly_and_counted_by_order() {
        let mut sink = AttemptSink::default();
        sink.record_materialize_original(17);
        sink.record_materialize_permuted(29);

        assert_eq!(sink.ipm.assembly_us, 46);
        assert_eq!(sink.ipm.factor_us, 0);
        assert_eq!(sink.counts.materialize_original_calls, 1);
        assert_eq!(sink.counts.materialize_permuted_calls, 1);
        assert_eq!(sink.counts.materialize_total_calls(), 2);
    }
}
