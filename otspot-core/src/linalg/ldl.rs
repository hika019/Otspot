//! faer high-level LDL^T wrapper for sparse linear systems
//!
//! Uses faer's high-level Cholesky API (SymbolicCholesky + SupernodalThreshold::AUTO)
//! to automatically select simplicial or supernodal factorization based on matrix structure.
//! Banded/sparse matrices (LISWET etc.) → simplicial; dense fill (AUG2D etc.) → supernodal.
//!
//! Public API:
//! - `LdlFactorization`        — positive definite, no AMD
//! - `LdlFactorizationAmd`     — quasidefinite, with AMD permutation
//! - `LdlError`
//! - `factorize`
//! - `factorize_quasidefinite_with_amd`
//! - `factorize_quasidefinite_with_cached_perm_budget_par`

use crate::linalg::amd::{amd_with_deadline, inv_permute_vec, permute_sym_upper, permute_vec};
use crate::sparse::CscMatrix;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::ldlt::factor::{LdltError, LdltRegularization};
use faer::reborrow::*;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, simplicial, supernodal, CholeskySymbolicParams, LdltRef,
    SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::SupernodalThreshold;
#[cfg(test)]
use faer::sparse::Triplet;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::Instant;
#[cfg(test)]
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

#[cfg(test)]
static TEST_NUMERIC_FACTORIZE_DELAY_MS: AtomicU64 = AtomicU64::new(0);

/// LDL分解エラー
#[non_exhaustive]
#[derive(Debug)]
pub enum LdlError {
    /// 行列が特異または不定（faer regularization でも処理不能な場合）
    SingularOrIndefinite,
    /// deadline を超過した
    DeadlineExceeded,
    /// symbolic 段階で L_nnz が事前許可量を超えた (numeric 因子化は試みていない)。
    /// 上位層は反復法フォールバックの判断材料として用いる。
    WouldExceedBudget {
        /// symbolic 段階で実測した L の非ゼロ数
        l_nnz: usize,
        /// 呼び出し側が許可した最大 L_nnz
        max_l_nnz: usize,
    },
}

/// `is_q_psd_by_cholesky` で対角に乗せる shift の倍率。
///
/// shift = `SHIFT_FACTOR · fp_threshold` (fp_threshold = `|Q|_max · n · ε`).
/// PSD-singular 行列の zero pivot を回避するには shift > FP noise が必要で、
/// 一方で indefinite Q の真の負固有値 |λ_min| が shift より十分大きければ
/// `D[j] < -fp_threshold` で検出できる。倍率を保守的に 10 とし、
/// 縮約 Schur complement の累積誤差で zero pivot が再発しても D[j] ≈ -μ + shift
/// が依然 `< -fp_threshold` となる margin を確保する。
const SHIFT_FACTOR: f64 = 10.0;

/// per-call parallelism の default (= 既存挙動)。
/// 公開 API の互換版 (par 引数なし) はこれを暗黙に使用する。
const DEFAULT_PAR: faer::Par = faer::Par::Seq;

/// Quasidefinite regularization delta: expected-sign pivot that is smaller than
/// this threshold is replaced with `LDLT_REG_DELTA`. Matches `ldl_dd::DELTA`.
const LDLT_REG_DELTA: f64 = 1e-8;
/// Quasidefinite regularization epsilon: |D[k]| below this is pushed to
/// `LDLT_REG_DELTA` in the expected direction. Matches `ldl_dd::EPSILON`.
const LDLT_REG_EPSILON: f64 = 1e-13;

// ── LdlFactorization (positive definite, no AMD) ──────────────────────────────

/// faer high-level LDL^T factorization for positive definite matrices (no AMD).
/// SupernodalThreshold::AUTO selects simplicial or supernodal automatically.
///
/// `par` は factorize 時に指定された per-call parallelism。solve 時に再利用される
/// (factor と solve で同じ par を使う方が rayon thread-pool 局所性が良い)。
pub struct LdlFactorization {
    symbolic: Arc<SymbolicCholesky<usize>>,
    l_values: Vec<f64>,
    n: usize,
    par: faer::Par,
}

impl LdlFactorization {
    /// LDL^T x = b を解く。sol に解を書き込む。
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        sol.copy_from_slice(rhs);
        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, self.par));
        let stack = MemStack::new(&mut mem);
        let ldlt = LdltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        let mut sol_mat = faer::MatMut::from_column_major_slice_mut(sol, self.n, 1);
        ldlt.solve_in_place_with_conj(faer::Conj::No, sol_mat.rb_mut(), self.par, stack);
    }
}

// ── LdlFactorizationAmd (quasidefinite, with AMD permutation) ─────────────────

/// faer high-level LDL^T factorization for quasidefinite matrices with AMD ordering.
/// SupernodalThreshold::AUTO selects simplicial or supernodal automatically.
///
/// `symbolic` は `Arc` 共有: IPM 反復で sparsity pattern が不変な場合、
/// `symbolic_arc()` で取得した Arc を次回 factorize に渡して再利用する。
/// clone は Arc::clone (refcount inc) のみ。
pub struct LdlFactorizationAmd {
    symbolic: Arc<SymbolicCholesky<usize>>,
    l_values: Vec<f64>,
    /// AMD permutation: perm[k] = original index of reordered index k
    perm: Vec<usize>,
    n: usize,
    /// factorize 時に指定された per-call parallelism。solve 時に再利用する。
    par: faer::Par,
}

impl LdlFactorizationAmd {
    /// L 因子の非ゼロ数を返す（デバッグ用）
    pub fn nnz_l(&self) -> usize {
        self.symbolic.len_val()
    }

    /// AMD 付き LDL^T x = b を解く。
    ///
    /// 1. 右辺を前方置換: b_p[k] = rhs[perm[k]]
    /// 2. (PAP^T) x_p = b_p を faer で解く（simplicial/supernodal は AUTO 選択）
    /// 3. 解を逆置換: sol[perm[k]] = x_p[k]
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        let n = self.n;
        let b_p = permute_vec(rhs, &self.perm);
        let mut x_p = b_p;

        let mut mem = MemBuffer::new(self.symbolic.solve_in_place_scratch::<f64>(1, self.par));
        let stack = MemStack::new(&mut mem);
        let ldlt = LdltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        let mut sol_mat = faer::MatMut::from_column_major_slice_mut(&mut x_p, n, 1);
        ldlt.solve_in_place_with_conj(faer::Conj::No, sol_mat.rb_mut(), self.par, stack);

        let x = inv_permute_vec(&x_p, &self.perm);
        sol.copy_from_slice(&x);
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────────

/// 上三角 CscMatrix を faer SparseColMat（上三角）に変換する。
///
/// 上三角エントリ (i, j) with i ≤ j をそのまま faer 形式に変換する。
/// 高レベル API に Side::Upper で渡すために使用する。
///
/// CSC は既に列優先・行昇順で整列されている前提なので、triplet を経由せず
/// SymbolicSparseColMat::new_checked + SparseColMat::new で直接構築する。
/// `try_new_from_triplets` の sort/compress を回避し、BOYD2 で 5ms → ~1ms 級。
fn csc_upper_to_faer_upper(mat: &CscMatrix) -> SparseColMat<usize, f64> {
    let n = mat.nrows;
    let symbolic =
        SymbolicSparseColMat::new_checked(n, n, mat.col_ptr.clone(), None, mat.row_ind.clone());
    SparseColMat::new(symbolic, mat.values.clone())
}

/// 対角要素の符号ベクトルを抽出する（LdltRegularization sign-aware 用）。
///
/// 対角が負なら -1、それ以外は +1。CSC 上三角を slice で受けるため、
/// CscMatrix 経由でも permute 後の生 slice 経由でも (ldl_dd) 共通利用できる。
pub(crate) fn extract_diagonal_signs(
    n: usize,
    col_ptr: &[usize],
    row_ind: &[usize],
    values: &[f64],
) -> Vec<i8> {
    let mut signs = vec![1i8; n];
    for (j, sign) in signs.iter_mut().enumerate() {
        for k in col_ptr[j]..col_ptr[j + 1] {
            if row_ind[k] == j {
                if values[k] < 0.0 {
                    *sign = -1;
                }
                break;
            }
        }
    }
    signs
}

/// 高レベル API で SymbolicCholesky を計算する（SupernodalThreshold::AUTO）。
///
/// Side::Upper で渡すため a_upper は上三角 faer SparseColMat であること。
/// SymmetricOrdering::Identity: AMD は外部で処理済みのため faer 内部では行わない。
fn build_symbolic_hl(
    a_upper: &SparseColMat<usize, f64>,
) -> Result<SymbolicCholesky<usize>, LdlError> {
    factorize_symbolic_cholesky(
        a_upper.symbolic(),
        faer::Side::Upper,
        SymmetricOrdering::Identity,
        CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
            ..Default::default()
        },
    )
    .map_err(|_| LdlError::SingularOrIndefinite)
}

/// 高レベル API で symbolic + numeric 因子化を実行する共通処理。
///
/// `signs`: Some → quasidefinite sign-aware regularization
///          None → sign-unaware regularization (positive definite 向け)
/// `deadline`: 指定時、symbolic 完了直後・numeric 開始前に再チェック。
///             symbolic は O(nnz) で速いが numeric は O(L_nnz) で BOYD2 級では分単位。
///             このチェックがないと numeric 中に deadline を尊重できない。
/// `max_l_nnz`: 指定時、symbolic 完了直後の L_nnz と比較し超過なら numeric を
///             試みず `WouldExceedBudget` を返す。上位は反復法フォールバックに使う。
fn do_numeric_factorize(
    mat: &CscMatrix,
    signs: Option<&[i8]>,
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
    par: faer::Par,
) -> Result<(Arc<SymbolicCholesky<usize>>, Vec<f64>), LdlError> {
    do_numeric_factorize_with_cache(mat, signs, deadline, max_l_nnz, None, par)
}

/// `do_numeric_factorize` の symbolic キャッシュ対応版。`cached_symbolic` が `Some` の
/// ときは `build_symbolic_hl` を skip して numeric 因子化のみ実行する。
/// pattern 不変な IPM 反復で 5ms/call 程度の symbolic コストを削れる。
fn do_numeric_factorize_with_cache(
    mat: &CscMatrix,
    signs: Option<&[i8]>,
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
    cached_symbolic: Option<Arc<SymbolicCholesky<usize>>>,
    par: faer::Par,
) -> Result<(Arc<SymbolicCholesky<usize>>, Vec<f64>), LdlError> {
    let a_upper = csc_upper_to_faer_upper(mat);
    let symbolic: Arc<SymbolicCholesky<usize>> = match cached_symbolic {
        Some(s) => s,
        None => Arc::new(build_symbolic_hl(&a_upper)?),
    };

    // symbolic 完了後・numeric 前に L_nnz チェック (memory budget)。
    // 巨大問題 (QPLIB_9008 等) で OOM kill されるのを防ぐ早期検知ポイント。
    let l_nnz = symbolic.len_val();
    if let Some(max) = max_l_nnz {
        if l_nnz > max {
            return Err(LdlError::WouldExceedBudget {
                l_nnz,
                max_l_nnz: max,
            });
        }
    }

    // symbolic 完了後・numeric 前に deadline 再チェック (numeric は最も時間がかかる)。
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }

    let owned_signs = signs.map(|s| s.to_vec());
    let l_values = factorize_numeric_with_deadline_watchdog(
        Arc::clone(&symbolic),
        a_upper,
        owned_signs,
        par,
        deadline,
    )?;

    Ok((symbolic, l_values))
}

fn run_numeric_factorization(
    symbolic: &SymbolicCholesky<usize>,
    a_upper: &SparseColMat<usize, f64>,
    signs: Option<&[i8]>,
    par: faer::Par,
) -> Result<Vec<f64>, LdlError> {
    #[cfg(test)]
    {
        let ms = TEST_NUMERIC_FACTORIZE_DELAY_MS.load(Ordering::Relaxed);
        if ms > 0 {
            std::thread::sleep(Duration::from_millis(ms));
        }
    }
    let regularization = LdltRegularization {
        dynamic_regularization_signs: signs,
        dynamic_regularization_delta: LDLT_REG_DELTA,
        dynamic_regularization_epsilon: LDLT_REG_EPSILON,
    };
    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        symbolic.factorize_numeric_ldlt_scratch::<f64>(par, Default::default()),
        symbolic.solve_in_place_scratch::<f64>(1, par),
    ]));
    let stack = MemStack::new(&mut mem);
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a_upper.rb(),
            faer::Side::Upper,
            regularization,
            par,
            stack,
            Default::default(),
        )
        .map_err(|_| LdlError::SingularOrIndefinite)?;
    Ok(l_values)
}

fn factorize_numeric_with_deadline_watchdog(
    symbolic: Arc<SymbolicCholesky<usize>>,
    a_upper: SparseColMat<usize, f64>,
    owned_signs: Option<Vec<i8>>,
    par: faer::Par,
    deadline: Option<Instant>,
) -> Result<Vec<f64>, LdlError> {
    let Some(dl) = deadline else {
        return run_numeric_factorization(&symbolic, &a_upper, owned_signs.as_deref(), par);
    };
    if Instant::now() >= dl {
        return Err(LdlError::DeadlineExceeded);
    }

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let r = run_numeric_factorization(&symbolic, &a_upper, owned_signs.as_deref(), par);
        let _ = tx.send(r);
    });

    let remain = dl.saturating_duration_since(Instant::now());
    match rx.recv_timeout(remain) {
        Ok(r) => r,
        Err(RecvTimeoutError::Timeout) => Err(LdlError::DeadlineExceeded),
        Err(RecvTimeoutError::Disconnected) => Err(LdlError::SingularOrIndefinite),
    }
}

/// Q 行列から上三角 CSC を抽出する (row <= col のみ保持)。
///
/// Q が full-symmetric (上三角 + 下三角) で格納されている場合でも、
/// faer の LLT/LDL に渡せる上三角 CSC を構築する。
/// Q がすでに上三角のみなら再配置なしで同等の行列を返す。
fn q_to_upper_triangular(q: &CscMatrix) -> CscMatrix {
    let n = q.nrows;
    // 上三角エントリのみ収集
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            let row = q.row_ind[k];
            if row <= col {
                rows.push(row);
                cols.push(col);
                vals.push(q.values[k]);
            }
        }
    }
    if rows.is_empty() {
        CscMatrix::new(n, n)
    } else {
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }
}

/// 上三角 CSC の各対角に `shift` を加算する (零対角は新規 entry を生成)。
/// `from_triplets` の重複 sum 仕様を利用。
fn upper_tri_with_diag_shift(q_upper: &CscMatrix, shift: f64) -> CscMatrix {
    let n = q_upper.nrows;
    let mut rows: Vec<usize> = Vec::with_capacity(q_upper.values.len() + n);
    let mut cols: Vec<usize> = Vec::with_capacity(q_upper.values.len() + n);
    let mut vals: Vec<f64> = Vec::with_capacity(q_upper.values.len() + n);
    for col in 0..n {
        for k in q_upper.col_ptr[col]..q_upper.col_ptr[col + 1] {
            rows.push(q_upper.row_ind[k]);
            cols.push(col);
            vals.push(q_upper.values[k]);
        }
        rows.push(col);
        cols.push(col);
        vals.push(shift);
    }
    CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Q 行列が PSD (半正定値) かどうかを LDL^T 慣性判定で判定する。
///
/// ## アルゴリズム
///
/// 1. 必要条件チェック: 全対角 ≥ 0 でなければ即 false。
/// 2. Q の対角を `shift = max(10·n·|Q|_max·ε, 10·ε)` で持ち上げ、`Q + shift·I` を LDL^T:
///    - 完了 → D\[j\] < -fp_threshold が一つでもあれば indefinite。
///    - ZeroPivot → 縮約 Schur complement が連続して degenerate な病的ケース。
///      conservative に false (= 呼出側 Gershgorin 経路で補正) を返す。
///
/// shift は PSD-singular 行列の zero pivot を回避しつつ、indefinite の λ_min を
/// 検出可能に保つ大きさ (FP noise の 10 倍程度) に設定。
/// `Q=[[0,1],[1,0]]` のような零対角 indefinite を従来は ZeroPivot 経路で PSD 誤判定
/// していたバグの根治。
pub fn is_q_psd_by_cholesky(q: &CscMatrix) -> bool {
    is_q_psd_by_cholesky_impl(q, SHIFT_FACTOR, true)
}

/// Test-only probe: 個別 no-op 実証 (`feedback_sentinel_must_fail_under_noop`) で
/// 2 つの fix (shift, ZeroPivot conservative) の必要性を機械検証する。production binary
/// には含めない (`#[cfg(test)]` で gating)。
///
/// - `shift_factor`: 対角に乗せる shift 倍率。production = `SHIFT_FACTOR`(=10)。0 で無効化 = 旧経路。
/// - `zeropivot_conservative`: ZeroPivot 判定。production = true (=indefinite 扱い)。
///   false で旧「ZeroPivot=PSD」挙動を再現。
#[cfg(test)]
pub(crate) fn is_q_psd_by_cholesky_probe(
    q: &CscMatrix,
    shift_factor: f64,
    zeropivot_conservative: bool,
) -> bool {
    is_q_psd_by_cholesky_impl(q, shift_factor, zeropivot_conservative)
}

/// `is_q_psd_by_cholesky` の実装本体。production と test probe で共有する。
/// 直接呼ばないこと (production は constants 固定の wrapper 経由)。
fn is_q_psd_by_cholesky_impl(
    q: &CscMatrix,
    shift_factor: f64,
    zeropivot_conservative: bool,
) -> bool {
    let n = q.nrows;
    if n == 0 {
        return true;
    }
    // Q が全ゼロなら PSD (LP ケース)
    if q.values.iter().all(|&v| v == 0.0) {
        return true;
    }
    // Q の全対角が非負かどうかを先にチェック (必要条件; 失敗したら即 false)
    // これは O(nnz) の安価な pre-check。
    let mut all_diag_nonneg = true;
    for col in 0..n {
        for k in q.col_ptr[col]..q.col_ptr[col + 1] {
            if q.row_ind[k] == col && q.values[k] < 0.0 {
                all_diag_nonneg = false;
                break;
            }
        }
        if !all_diag_nonneg {
            break;
        }
    }
    if !all_diag_nonneg {
        return false;
    }

    // Q を上三角化、対角を shift だけ持ち上げて factorize。
    // shift により PSD-singular 行列の ZeroPivot を回避しつつ、indefinite Q の
    // 真の負固有値は shift より十分大きいので D[j] < -fp_threshold で検出できる。
    let q_upper = q_to_upper_triangular(q);
    let q_abs_max = q.values.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    let fp_threshold = q_abs_max * (n as f64) * f64::EPSILON;
    // shift_factor=10 (production): D[j] ≈ −μ + shift。検出可能下限は μ ≳ shift。
    // |Q|_max=0 (Q=0 は既に上で early-return) ガードに shift_factor·eps の最小値も設ける。
    let shift = (shift_factor * fp_threshold).max(shift_factor * f64::EPSILON);
    let q_shifted = upper_tri_with_diag_shift(&q_upper, shift);
    let a_upper = csc_upper_to_faer_upper(&q_shifted);
    let symbolic = match build_symbolic_hl(&a_upper) {
        Ok(s) => s,
        Err(_) => return false,
    };

    // 正則化なし (delta=0, epsilon=0) で LDL^T を試みる
    let reg: LdltRegularization<f64> = LdltRegularization::default();

    let l_nnz = symbolic.len_val();
    let mut l_values = vec![0.0f64; l_nnz];

    match symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(simp_sym) => {
            // Simplicial 経路: D 対角は col_ptr[j] 位置に格納されている。
            let scratch_ldlt =
                simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(n);
            let scratch_solve = simp_sym.solve_in_place_scratch::<f64>(1);
            let mut mem = MemBuffer::new(StackReq::any_of(&[scratch_ldlt, scratch_solve]));
            let stack = MemStack::new(&mut mem);

            let result = simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
                &mut l_values,
                a_upper.rb(),
                reg,
                simp_sym,
                stack,
            );

            // simplicial の D[j] は col_ptr[j] 位置に格納される。
            let col_ptr = simp_sym.col_ptr();
            match result {
                // ZeroPivot: production は conservative に false (= indefinite 扱い)。
                // probe で `zeropivot_conservative=false` (旧経路) を流すと true を返す。
                Err(LdltError::ZeroPivot { .. }) => !zeropivot_conservative,
                Ok(_) => !(0..n).any(|j| l_values[col_ptr[j]] < -fp_threshold),
            }
        }
        SymbolicCholeskyRaw::Supernodal(sn_sym) => {
            // Supernodal 経路: 高レベル API (SymbolicCholesky::factorize_numeric_ldlt) で
            // Side::Upper を渡して因子化する。高レベル API は内部で Upper→Lower 変換を
            // 行い supernodal::factorize_supernodal_numeric_ldlt に Lower を渡す。
            //
            // D 対角の読み出し:
            // 各スーパーノード s について SupernodalLdltRef::supernode(s).val() は
            // (s_nrows × s_ncols) の MatRef。val()[(j, j)] = D[s_start + j]。
            let scratch =
                symbolic.factorize_numeric_ldlt_scratch::<f64>(faer::Par::Seq, Default::default());
            let mut mem = MemBuffer::new(scratch);
            let stack = MemStack::new(&mut mem);

            let result = symbolic.factorize_numeric_ldlt(
                &mut l_values,
                a_upper.rb(),
                faer::Side::Upper,
                reg,
                faer::Par::Seq,
                stack,
                Default::default(),
            );

            // ZeroPivot: production は conservative に false (= indefinite 扱い)。
            // probe で `zeropivot_conservative=false` を流すと true を返す。
            if matches!(result, Err(LdltError::ZeroPivot { .. })) {
                return !zeropivot_conservative;
            }

            let ldlt_ref = supernodal::SupernodalLdltRef::<usize, f64>::new(sn_sym, &l_values);
            let n_sn = sn_sym.n_supernodes();
            let sn_begin = sn_sym.supernode_begin();
            let sn_end = sn_sym.supernode_end();
            let has_negative_d = (0..n_sn).any(|s| {
                let s_start = sn_begin[s];
                let s_end = sn_end[s];
                let s_ncols = s_end - s_start;
                let node = ldlt_ref.supernode(s);
                let ls = node.val();
                (0..s_ncols).any(|j| ls[(j, j)] < -fp_threshold)
            });

            !has_negative_d
        }
    }
}

/// 正定値疎行列の LDL^T 分解を実行する (per-call parallelism = `Par::Seq`、既存互換)。
pub fn factorize(mat: &CscMatrix) -> Result<LdlFactorization, LdlError> {
    factorize_with_par(mat, DEFAULT_PAR)
}

/// `factorize` の memory-budget 版。symbolic 完了後の L_nnz が `max_l_nnz` を
/// 超える場合は numeric を試みず `WouldExceedBudget` を返す。AAT 直接法のように
/// 入力 nnz は予算内でも fill-in で OOM し得る経路で使う (build 時の bytes 見積りは
/// fill-in を捉えないため)。
pub fn factorize_budget(mat: &CscMatrix, max_l_nnz: usize) -> Result<LdlFactorization, LdlError> {
    let n = mat.nrows;
    let (symbolic, l_values) = do_numeric_factorize(mat, None, None, Some(max_l_nnz), DEFAULT_PAR)?;
    Ok(LdlFactorization {
        symbolic,
        l_values,
        n,
        par: DEFAULT_PAR,
    })
}

/// `factorize` の per-call parallelism 指定版。
/// `par == Par::Seq` で既存挙動と完全互換。
pub(crate) fn factorize_with_par(
    mat: &CscMatrix,
    par: faer::Par,
) -> Result<LdlFactorization, LdlError> {
    let n = mat.nrows;
    let (symbolic, l_values) = do_numeric_factorize(mat, None, None, None, par)?;
    Ok(LdlFactorization {
        symbolic,
        l_values,
        n,
        par,
    })
}

/// AMD 再順序化付き quasidefinite LDL^T 分解（AMD を内部で計算、既存互換）。
pub fn factorize_quasidefinite_with_amd(
    mat: &CscMatrix,
    deadline: Option<Instant>,
) -> Result<LdlFactorizationAmd, LdlError> {
    factorize_quasidefinite_with_amd_budget_par(mat, deadline, None, DEFAULT_PAR)
}

/// `factorize_quasidefinite_with_amd` の per-call parallelism + budget 指定版。
pub fn factorize_quasidefinite_with_amd_budget_par(
    mat: &CscMatrix,
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
    par: faer::Par,
) -> Result<LdlFactorizationAmd, LdlError> {
    let n = mat.nrows;
    let perm = amd_with_deadline(n, &mat.col_ptr, &mat.row_ind, deadline);
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    factorize_quasidefinite_with_cached_perm_budget_par(mat, &perm, deadline, max_l_nnz, par)
}

/// AMD キャッシュ済み置換 + budget 版 (per-call parallelism 指定)。
pub fn factorize_quasidefinite_with_cached_perm_budget_par(
    mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
    par: faer::Par,
) -> Result<LdlFactorizationAmd, LdlError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    let n = mat.nrows;
    let (new_col_ptr, new_row_ind, new_values) =
        permute_sym_upper(n, &mat.col_ptr, &mat.row_ind, &mat.values, perm);
    let perm_mat = CscMatrix {
        col_ptr: new_col_ptr,
        row_ind: new_row_ind,
        values: new_values,
        nrows: n,
        ncols: n,
    };
    let signs = extract_diagonal_signs(n, &perm_mat.col_ptr, &perm_mat.row_ind, &perm_mat.values);
    let (symbolic, l_values) =
        do_numeric_factorize(&perm_mat, Some(&signs), deadline, max_l_nnz, par)?;
    Ok(LdlFactorizationAmd {
        symbolic,
        l_values,
        perm: perm.to_vec(),
        n,
        par,
    })
}

/// AMD 置換適用済みの行列を受け取る per-call parallelism + symbolic キャッシュ版。
/// `cached_symbolic` が `Some` のときは `build_symbolic_hl` を skip して再利用する。
/// caller は `factor.symbolic_arc()` で取得して次回呼び出しに渡せる。
pub fn factorize_quasidefinite_pre_permuted_cached_par(
    pre_permuted_mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
    cached_symbolic: Option<Arc<SymbolicCholesky<usize>>>,
    par: faer::Par,
) -> Result<LdlFactorizationAmd, LdlError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    let n = pre_permuted_mat.nrows;
    let signs = extract_diagonal_signs(
        n,
        &pre_permuted_mat.col_ptr,
        &pre_permuted_mat.row_ind,
        &pre_permuted_mat.values,
    );
    let (symbolic, l_values) = do_numeric_factorize_with_cache(
        pre_permuted_mat,
        Some(&signs),
        deadline,
        max_l_nnz,
        cached_symbolic,
        par,
    )?;
    Ok(LdlFactorizationAmd {
        symbolic,
        l_values,
        perm: perm.to_vec(),
        n,
        par,
    })
}

impl LdlFactorizationAmd {
    /// 内部の SymbolicCholesky を Arc として取得する (反復間でのキャッシュ用)。
    pub fn symbolic_arc(&self) -> Arc<SymbolicCholesky<usize>> {
        Arc::clone(&self.symbolic)
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout, clippy::print_stderr)]
mod tests {
    use super::*;
    use crate::sparse::CscMatrix;
    use faer::sparse::linalg::cholesky::{simplicial, supernodal};

    /// 上三角 CSC 行列をエントリリストから構築するヘルパー
    fn upper_tri_csc(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix {
        let mut cols: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for &(row, col, val) in entries {
            assert!(row <= col, "upper triangle only: row={row} col={col}");
            cols[col].push((row, val));
        }
        for c in cols.iter_mut() {
            c.sort_by_key(|&(r, _)| r);
        }
        let nnz: usize = cols.iter().map(|c| c.len()).sum();
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + cols[j].len();
        }
        let mut row_ind = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        for j in 0..n {
            let start = col_ptr[j];
            for (idx, &(row, val)) in cols[j].iter().enumerate() {
                row_ind[start + idx] = row;
                values[start + idx] = val;
            }
        }
        CscMatrix {
            col_ptr,
            row_ind,
            values,
            nrows: n,
            ncols: n,
        }
    }

    #[test]
    fn test_factorize_pd_3x3_solve() {
        // A = [[4,1,0],[1,3,2],[0,2,5]] — positive definite
        let mat = upper_tri_csc(
            3,
            &[
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 1, 3.0),
                (1, 2, 2.0),
                (2, 2, 5.0),
            ],
        );
        let fac = factorize(&mat).expect("factorize failed");
        let b = [1.0f64, 2.0, 3.0];
        let mut x = [0.0f64; 3];
        fac.solve(&b, &mut x);
        // residual |Ax - b|
        let ax0 = 4.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1] + 2.0 * x[2];
        let ax2 = 2.0 * x[1] + 5.0 * x[2];
        let eps = 1e-8;
        assert!((ax0 - b[0]).abs() < eps, "r[0]={}", (ax0 - b[0]).abs());
        assert!((ax1 - b[1]).abs() < eps, "r[1]={}", (ax1 - b[1]).abs());
        assert!((ax2 - b[2]).abs() < eps, "r[2]={}", (ax2 - b[2]).abs());
    }

    #[test]
    fn test_factorize_pd_identity() {
        // A = I_4
        let n = 4;
        let entries: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 1.0)).collect();
        let mat = upper_tri_csc(n, &entries);
        let fac = factorize(&mat).expect("factorize failed");
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0.0f64; n];
        fac.solve(&b, &mut x);
        for i in 0..n {
            assert!((x[i] - b[i]).abs() < 1e-10, "x[{i}]={}", x[i]);
        }
    }

    /// `factorize_budget` must reject (without numeric work) when the symbolic
    /// L_nnz exceeds the cap — this is the fill-in OOM guard that `build_aat`'s
    /// byte estimate cannot see. A `max_l_nnz` of 1 is below any non-trivial L.
    #[test]
    fn factorize_budget_rejects_when_l_nnz_exceeds_max() {
        let mat = upper_tri_csc(
            3,
            &[
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 1, 3.0),
                (1, 2, 2.0),
                (2, 2, 5.0),
            ],
        );
        match factorize_budget(&mat, 1) {
            Err(LdlError::WouldExceedBudget { l_nnz, max_l_nnz }) => {
                assert!(
                    l_nnz > max_l_nnz,
                    "l_nnz={l_nnz} should exceed max={max_l_nnz}"
                );
                assert_eq!(max_l_nnz, 1);
            }
            Err(e) => panic!("expected WouldExceedBudget, got Err({e:?})"),
            Ok(_) => panic!("expected WouldExceedBudget, got Ok"),
        }
    }

    /// Within budget, `factorize_budget` behaves like `factorize`.
    #[test]
    fn factorize_budget_accepts_within_budget() {
        let mat = upper_tri_csc(
            3,
            &[
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 1, 3.0),
                (1, 2, 2.0),
                (2, 2, 5.0),
            ],
        );
        let fac = factorize_budget(&mat, 1_000_000).expect("within budget must succeed");
        let b = [1.0f64, 2.0, 3.0];
        let mut x = [0.0f64; 3];
        fac.solve(&b, &mut x);
        let ax0 = 4.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1] + 2.0 * x[2];
        let ax2 = 2.0 * x[1] + 5.0 * x[2];
        assert!((ax0 - b[0]).abs() < 1e-8);
        assert!((ax1 - b[1]).abs() < 1e-8);
        assert!((ax2 - b[2]).abs() < 1e-8);
    }

    #[test]
    fn test_factorize_with_deadline_ok() {
        // Port: production path (factorize_quasidefinite_with_amd) honours a future deadline.
        let mat = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
        let deadline = Some(Instant::now() + std::time::Duration::from_secs(60));
        let fac = factorize_quasidefinite_with_amd(&mat, deadline).expect("should succeed");
        let b = [1.0f64, 0.0];
        let mut x = [0.0f64; 2];
        fac.solve(&b, &mut x);
        let ax0 = 2.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1];
        assert!((ax0 - b[0]).abs() < 1e-10, "r[0]={}", (ax0 - b[0]).abs());
        assert!((ax1 - b[1]).abs() < 1e-10, "r[1]={}", (ax1 - b[1]).abs());
    }

    #[test]
    fn test_factorize_with_deadline_expired() {
        // Port: production path (factorize_quasidefinite_with_cached_perm_budget_par) returns
        // DeadlineExceeded when the deadline is already past on entry.
        let mat = upper_tri_csc(2, &[(0, 0, 2.0), (1, 1, 3.0)]);
        let perm = vec![0usize, 1];
        let deadline = Some(Instant::now() - std::time::Duration::from_millis(1));
        let result = factorize_quasidefinite_with_cached_perm_budget_par(
            &mat,
            &perm,
            deadline,
            None,
            faer::Par::Seq,
        );
        assert!(
            matches!(result, Err(LdlError::DeadlineExceeded)),
            "Expected DeadlineExceeded"
        );
    }

    /// Numeric factorization is not preemptible inside faer; this sentinel ensures
    /// the watchdog path returns `DeadlineExceeded` without waiting for numeric completion.
    #[test]
    fn test_factorize_deadline_watchdog_during_numeric() {
        struct ResetDelay;
        impl Drop for ResetDelay {
            fn drop(&mut self) {
                TEST_NUMERIC_FACTORIZE_DELAY_MS.store(0, Ordering::Relaxed);
            }
        }
        let _reset = ResetDelay;
        TEST_NUMERIC_FACTORIZE_DELAY_MS.store(50, Ordering::Relaxed);
        let mat = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
        let perm = vec![0usize, 1];
        let deadline = Some(Instant::now() + std::time::Duration::from_millis(5));
        let result = factorize_quasidefinite_with_cached_perm_budget_par(
            &mat,
            &perm,
            deadline,
            None,
            faer::Par::Seq,
        );
        assert!(
            matches!(result, Err(LdlError::DeadlineExceeded)),
            "expected watchdog timeout during numeric factorization",
        );
    }

    #[test]
    fn test_quasidefinite_2x2_identity_perm() {
        // quasidefinite: [[3,1],[1,-2]] — D[0]>0, D[1]<0
        let mat = upper_tri_csc(2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 1, -2.0)]);
        let perm = vec![0usize, 1]; // identity permutation
        let fac = factorize_quasidefinite_with_cached_perm_budget_par(
            &mat,
            &perm,
            None,
            None,
            faer::Par::Seq,
        )
        .expect("quasidefinite factorize failed");
        let b = [1.0f64, 2.0];
        let mut x = [0.0f64; 2];
        fac.solve(&b, &mut x);
        // residual: [[3,1],[1,-2]] * x = b
        let ax0 = 3.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] - 2.0 * x[1];
        let eps = 1e-8;
        assert!((ax0 - b[0]).abs() < eps, "r[0]={}", (ax0 - b[0]).abs());
        assert!((ax1 - b[1]).abs() < eps, "r[1]={}", (ax1 - b[1]).abs());
    }

    #[test]
    fn test_quasidefinite_with_amd() {
        // 5x5 quasidefinite: Q=diag(1,2), A=[[1,0],[0,1],[1,1]], δ=1e-4
        let delta = 1e-4f64;
        let mat = upper_tri_csc(
            5,
            &[
                (0, 0, 1.0 + delta),
                (1, 1, 2.0 + delta),
                (2, 2, -delta),
                (3, 3, -delta),
                (4, 4, -delta),
                (0, 2, 1.0),
                (1, 3, 1.0),
                (0, 4, 1.0),
                (1, 4, 1.0),
            ],
        );
        let fac =
            factorize_quasidefinite_with_amd(&mat, None).expect("quasidefinite_with_amd failed");
        let b = [1.0f64, 2.0, 0.5, -0.5, 1.0];
        let mut x = [0.0f64; 5];
        fac.solve(&b, &mut x);
        // Full matrix for residual check (symmetric)
        let full: &[(usize, usize, f64)] = &[
            (0, 0, 1.0 + delta),
            (1, 1, 2.0 + delta),
            (2, 2, -delta),
            (3, 3, -delta),
            (4, 4, -delta),
            (0, 2, 1.0),
            (2, 0, 1.0),
            (1, 3, 1.0),
            (3, 1, 1.0),
            (0, 4, 1.0),
            (4, 0, 1.0),
            (1, 4, 1.0),
            (4, 1, 1.0),
        ];
        let mut r = [0.0f64; 5];
        for &(row, col, val) in full {
            r[row] += val * x[col];
        }
        let res: f64 = r
            .iter()
            .zip(b.iter())
            .map(|(&ri, &bi)| (ri - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(res < 1e-8, "residual={res:.3e}");
    }

    /// is_q_psd_by_cholesky: PSD 行列は true を返す
    #[test]
    fn test_is_q_psd_psd_matrix() {
        // Q = [[4,1],[1,3]] — PD (固有値 ≈ 2.38, 4.62)
        let q = upper_tri_csc(2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0)]);
        assert!(
            is_q_psd_by_cholesky(&q),
            "PD matrix should be identified as PSD"
        );
    }

    /// is_q_psd_by_cholesky: 不定行列は false を返す
    #[test]
    fn test_is_q_psd_indefinite_matrix() {
        // Q = [[1,0],[0,-1]] — indefinite (固有値 +1, -1)
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (1, 1, -1.0)]);
        assert!(
            !is_q_psd_by_cholesky(&q),
            "Indefinite matrix should NOT be identified as PSD"
        );
    }

    /// is_q_psd_by_cholesky: 大きい対角外要素を持つ PSD 行列でも true を返す
    /// (Gershgorin 誤判定パターン: R_j > Q[j,j] でも実際は PSD)
    #[test]
    fn test_is_q_psd_large_offdiag_still_psd() {
        // Q = [[10, 5],[5, 10]] — PD (固有値 5, 15)
        // Gershgorin: λ_min >= min(10-5, 10-5) = 5 > 0 → 問題なし
        // より極端: Q = [[3, 2.9],[2.9, 3]] — PD (固有値 0.1, 5.9)
        // Gershgorin: λ_min >= min(3-2.9, 3-2.9) = 0.1 > 0
        // でも: Q = [[2, 1.5],[1.5, 2]] — PD (固有値 0.5, 3.5)
        // Gershgorin: λ_min >= min(2-1.5, 2-1.5) = 0.5 > 0 → compute_inertia_correction=0
        // これは問題ない。本当の問題は: Q が PSD でも Gershgorin が < 0 を返すケース。
        // 例: Q = [[1, 0.9],[0.9, 1]] — PD (固有値 0.1, 1.9)
        // Gershgorin: λ_min >= min(1-0.9, 1-0.9) = 0.1 (正しく検出)
        // 境界ケース: Q = [[1, 1.1],[1.1, 2]] — PD if det > 0: det = 2-1.21 = 0.79 > 0
        // Gershgorin: λ_min >= min(1-1.1, 2-1.1) = min(-0.1, 0.9) = -0.1 → 誤検出
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (0, 1, 1.1), (1, 1, 2.0)]);
        // det = 1*2 - 1.1*1.1 = 2 - 1.21 = 0.79 > 0 → PD
        // is_q_psd_by_cholesky はここで true を返すべき (Gershgorin は false の誤判定)
        assert!(
            is_q_psd_by_cholesky(&q),
            "PSD matrix with large off-diagonal (Gershgorin false alarm) should be true"
        );
    }

    /// is_q_psd_by_cholesky: PSD (特異) 行列 → ZeroPivot → true を返す
    ///
    /// Q = [[1, 1],[1, 1]] は rank-1 で最小固有値 0 (PSD)。
    /// LDL^T は ZeroPivot で失敗 → true を返すべき。
    #[test]
    fn test_is_q_psd_singular_psd() {
        // Q = [[1,1],[1,1]] — PSD (rank 1, λ = 0 and 2)
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)]);
        assert!(
            is_q_psd_by_cholesky(&q),
            "Singular PSD matrix should be identified as PSD"
        );
    }

    /// is_q_psd_by_cholesky: 不定行列 (対角外が大きく不定) → false を返す
    ///
    /// Q = [[2, 3],[3, 2]] — indefinite (固有値 -1, 5)
    #[test]
    fn test_is_q_psd_offdiag_indefinite() {
        // Q = [[2,3],[3,2]] — indefinite (det=4-9=-5 < 0, λ = -1, 5)
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 3.0), (1, 1, 2.0)]);
        assert!(
            !is_q_psd_by_cholesky(&q),
            "Indefinite matrix with off-diag should be false"
        );
    }

    /// is_q_psd_by_cholesky: ゼロ行列は true を返す (LP ケース)
    #[test]
    fn test_is_q_psd_zero_matrix() {
        let q = upper_tri_csc(3, &[]);
        assert!(
            is_q_psd_by_cholesky(&q),
            "Zero matrix (LP) should be identified as PSD"
        );
    }

    /// is_q_psd_by_cholesky: n=0 は true を返す
    #[test]
    fn test_is_q_psd_empty_matrix() {
        let q = CscMatrix::new(0, 0);
        assert!(
            is_q_psd_by_cholesky(&q),
            "Empty matrix should be identified as PSD"
        );
    }

    /// Indefinite Q where a ZeroPivot column appears *after* a negative D[i]; the
    /// classifier must report non-PSD rather than mask the negative pivot.
    #[test]
    fn test_is_q_psd_zeropivot_masks_negative_d() {
        // 上三角形式: (row, col, val) with row <= col
        // Q[0,0]=2, Q[0,1]=1, Q[0,2]=-1  (rows 1,2,3 have zero diagonal)
        let q = upper_tri_csc(4, &[(0, 0, 2.0), (0, 1, 1.0), (0, 2, -1.0)]);
        // LDL^T: D[0]=2, D[1]=-0.5, D[2]=-0.5, D[3]=0 (ZeroPivot at col 3)
        // D[1] < 0 → indefinite
        assert!(
            !is_q_psd_by_cholesky(&q),
            "Indefinite matrix (ZeroPivot masks earlier negative D) must return false"
        );
    }

    /// is_q_psd_by_cholesky: ZeroPivot が来ても先行列が全て非負なら PSD
    ///
    /// n=3, Q = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    /// D[0]=1>0, D[1]=1>0, D[2]=0 → ZeroPivot at col 2, だが D[0..2] >= 0
    /// → true (PSD, 零固有値あり)
    #[test]
    fn test_is_q_psd_zeropivot_all_nonneg_d_is_psd() {
        let q = upper_tri_csc(
            3,
            &[
                (0, 0, 1.0),
                (1, 1, 1.0),
                // col 2: no entry → D[2] = 0
            ],
        );
        assert!(
            is_q_psd_by_cholesky(&q),
            "PSD matrix with zero eigenvalue (ZeroPivot) must return true"
        );
    }

    // ---- 個別 no-op proof ----
    //
    // `is_q_psd_by_cholesky_probe(q, shift_factor, zeropivot_conservative)` で
    // shift と ZeroPivot 経路書換を個別 toggle し、各 fix が必要な case を機械実証する。
    // `feedback_sentinel_must_fail_under_noop` 準拠: 各 assert が片方の fix 撤回で
    // 即 FAIL することを示し、両者を coupled に検証していた既存 sentinel に
    // 個別寄与の証拠を補完する。

    /// **shift no-op proof**: shift=0 に戻すと rank-1 PSD `Q=[[1,1],[1,1]]` が
    /// ZeroPivot 経由で indefinite と誤判定される (D[1]=0 が ZeroPivot に化け、
    /// 新 conservative 経路で false を返す)。shift があれば D[1]≈2·shift > 0 となり
    /// PSD を正しく検出。`test_is_q_psd_singular_psd` が production で PASS する
    /// 真の理由が shift にあることを切り分ける。
    #[test]
    fn no_op_proof_shift_required_for_rank1_psd() {
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)]);
        // shift_factor=0 (revert), zeropivot_conservative=true (新経路維持)
        let without_shift = is_q_psd_by_cholesky_probe(&q, 0.0, true);
        assert!(
            !without_shift,
            "shift no-op proof: rank-1 PSD は shift 無しだと ZeroPivot で false 化される \
             (= shift がなければ production が誤分類)"
        );
        // shift を戻すと production と同じく true を返す
        let with_shift = is_q_psd_by_cholesky_probe(&q, SHIFT_FACTOR, true);
        assert!(
            with_shift,
            "guard: shift 適用時は rank-1 PSD が正しく PSD と分類されること"
        );
    }

    /// **ZeroPivot no-op proof**: zero-diag bilinear `Q=[[0,1],[1,0]]` (旧 bug の trigger) は
    /// **shift 無しで初列が ZeroPivot** になる。旧経路 (`return true on ZeroPivot`) は
    /// この indefinite Q を PSD と誤判定する一方、新経路は false (conservative) を返す。
    /// shift を切った probe で旧経路 vs 新経路を直接比較し、ZeroPivot path 書換の必要性を実証。
    #[test]
    fn no_op_proof_zeropivot_conservative_required_when_shift_absent() {
        let q = upper_tri_csc(2, &[(0, 1, 1.0)]);
        // shift_factor=0, zeropivot_conservative=false (旧経路) → 旧 bug 再現 = true (誤分類)
        let old_path = is_q_psd_by_cholesky_probe(&q, 0.0, false);
        assert!(
            old_path,
            "ZeroPivot no-op proof: 旧経路 (shift=0, ZeroPivot=true) は indefinite Q を \
             PSD と誤分類する (= 旧 bug の再現)"
        );
        // shift=0, zeropivot_conservative=true (新経路のみ単独) → 修正される
        let new_path_alone = is_q_psd_by_cholesky_probe(&q, 0.0, true);
        assert!(
            !new_path_alone,
            "ZeroPivot conservative 単独で旧 bug を修正できること"
        );
    }

    /// **shift と ZeroPivot 経路は belt-and-suspenders**: production (shift+conservative)
    /// は旧 bug の trigger を確実に弾く。shift 単独でも (= D[1]<<0 経由) indefinite 検出
    /// 可能で、両 fix の独立寄与を network 状に検証する。
    #[test]
    fn no_op_proof_shift_alone_also_catches_zero_diag_bilinear() {
        let q = upper_tri_csc(2, &[(0, 1, 1.0)]);
        // shift あり、ZeroPivot 旧経路 → shift がもたらす D[1]<<−fp で indefinite 検出
        let shift_alone = is_q_psd_by_cholesky_probe(&q, SHIFT_FACTOR, false);
        assert!(
            !shift_alone,
            "shift 単独でも Q=[[0,1],[1,0]] は indefinite と検出される \
             (= shift 経路が D[1] negative を露出させる)"
        );
        // production: 両 fix
        let production = is_q_psd_by_cholesky_probe(&q, SHIFT_FACTOR, true);
        assert!(!production, "production 経路は indefinite を弾く");
    }

    #[test]
    fn test_nnz_l_reasonable() {
        // nnz_l should be positive for a non-trivial matrix
        // (supernodal stores diagonal + fill-in, upper bound = n*(n+1)/2)
        let n = 3usize;
        let mat = upper_tri_csc(
            n,
            &[
                (0, 0, 4.0),
                (0, 1, 1.0),
                (1, 1, 3.0),
                (1, 2, 2.0),
                (2, 2, 5.0),
            ],
        );
        let perm = vec![0, 1, 2];
        let fac = factorize_quasidefinite_with_cached_perm_budget_par(
            &mat,
            &perm,
            None,
            None,
            faer::Par::Seq,
        )
        .expect("factorize failed");
        let nnz = fac.nnz_l();
        // supernodal len_val() includes internal storage (may exceed lower-tri count)
        assert!(nnz > 0, "nnz_l should be positive for non-trivial matrix");
    }

    /// 診断テスト: LISWET類似帯状行列でのsupernodal vs simplicial比較
    ///
    /// LISWET KKT行列に近似した帯状対称行列(n=2000, band=2)を生成し、
    /// supernodalのスーパーノード数・len_val・factorize時間 vs simplicial時間
    /// を計測して報告する。
    #[test]
    fn diag_banded_supernode_vs_simplicial() {
        // 帯状対称正定値行列: n×n, band=2 (LISWETのKKT近似)
        let n = 2000usize;
        let band = 2usize;

        // 下三角 triplets (faer下三角形式)
        let mut lo_triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for i in 0..n {
            lo_triplets.push(Triplet::new(i, i, 4.0));
            if i + 1 < n {
                lo_triplets.push(Triplet::new(i + 1, i, -1.0));
            }
            if i + 2 < n {
                lo_triplets.push(Triplet::new(i + 2, i, -0.5));
            }
        }
        let a_lower = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &lo_triplets)
            .expect("build lower failed");

        let a_nnz = a_lower.compute_nnz();
        let a_upper_sym = a_lower
            .rb()
            .transpose()
            .symbolic()
            .to_col_major()
            .expect("transpose failed");

        // etree / col_counts (共通)
        let mut etree_buf = vec![0isize; n];
        let mut col_counts_buf = vec![0usize; n];
        {
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
                supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(n),
            ]));
            let stack = MemStack::new(&mut mem);
            simplicial::prefactorize_symbolic_cholesky(
                &mut etree_buf,
                &mut col_counts_buf,
                a_upper_sym.rb(),
                stack,
            );
        }

        // Supernodal: (usize::MAX, 1.0) = 現在の実装
        let relax_all: &[(usize, f64)] = &[(usize::MAX, 1.0)];
        // Supernodal: DEFAULT_RELAX
        let relax_default: &[(usize, f64)] = &[(4, 1.0), (16, 0.8), (48, 0.1), (usize::MAX, 0.05)];

        for (label, relax) in [
            ("supernodal(relax=ALL)", relax_all),
            ("supernodal(relax=DEFAULT)", relax_default),
        ] {
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
                supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(n),
            ]));
            let stack = MemStack::new(&mut mem);
            // etree/col_counts を再取得（stackが消費されるため）
            let mut etree = etree_buf.clone();
            let mut col_counts = col_counts_buf.clone();
            simplicial::prefactorize_symbolic_cholesky(
                &mut etree,
                &mut col_counts,
                a_upper_sym.rb(),
                stack,
            );

            let mut mem2 = MemBuffer::new(
                supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(n),
            );
            let stack2 = MemStack::new(&mut mem2);
            let t0 = Instant::now();
            let sym = supernodal::factorize_supernodal_symbolic_cholesky(
                a_upper_sym.rb(),
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack2,
                faer::sparse::linalg::SymbolicSupernodalParams { relax: Some(relax) },
            )
            .expect("symbolic failed");
            let sym_t = t0.elapsed();

            let n_sn = sym.n_supernodes();
            let len_val = sym.len_val();
            let begin = sym.supernode_begin();
            let end = sym.supernode_end();
            let sizes: Vec<usize> = (0..n_sn).map(|i| end[i] - begin[i]).collect();
            let max_sn = sizes.iter().max().copied().unwrap_or(0);
            let avg_sn = sizes.iter().sum::<usize>() as f64 / n_sn as f64;

            let regularization = faer::linalg::cholesky::ldlt::factor::LdltRegularization {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: LDLT_REG_DELTA,
                dynamic_regularization_epsilon: LDLT_REG_EPSILON,
            };
            let mut mem3 = MemBuffer::new(StackReq::any_of(&[
                supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, f64>(
                    &sym,
                    faer::Par::Seq,
                    Default::default(),
                ),
                sym.solve_in_place_scratch::<f64>(n, faer::Par::Seq),
            ]));
            let stack3 = MemStack::new(&mut mem3);
            let mut l_values = vec![0.0f64; sym.len_val()];
            let t1 = Instant::now();
            supernodal::factorize_supernodal_numeric_ldlt::<usize, f64>(
                &mut l_values,
                a_lower.rb(),
                regularization,
                &sym,
                faer::Par::Seq,
                stack3,
                Default::default(),
            )
            .expect("numeric failed");
            let num_t = t1.elapsed();

            println!(
                "[{label}] n={n}, band={band}: n_supernodes={n_sn}, len_val={len_val}, \
                 max_sn={max_sn}, avg_sn={avg_sn:.1}, sym={sym_t:.3?}, num={num_t:.3?}"
            );
        }

        // Simplicial
        {
            let mut etree = etree_buf.clone();
            let mut col_counts = col_counts_buf.clone();
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(n, a_nnz),
                simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n),
            ]));
            let stack = MemStack::new(&mut mem);
            simplicial::prefactorize_symbolic_cholesky(
                &mut etree,
                &mut col_counts,
                a_upper_sym.rb(),
                stack,
            );

            let mut mem2 = MemBuffer::new(
                simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n),
            );
            let stack2 = MemStack::new(&mut mem2);
            let t0 = Instant::now();
            let sym_s = simplicial::factorize_simplicial_symbolic_cholesky(
                a_upper_sym.rb(),
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack2,
            )
            .expect("simplicial symbolic failed");
            let sym_t = t0.elapsed();

            let regularization = faer::linalg::cholesky::ldlt::factor::LdltRegularization {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: LDLT_REG_DELTA,
                dynamic_regularization_epsilon: LDLT_REG_EPSILON,
            };
            let l_nnz = sym_s.len_val();
            let mut l_values = vec![0.0f64; l_nnz];
            let mut mem3 = MemBuffer::new(simplicial::factorize_simplicial_numeric_ldlt_scratch::<
                usize,
                f64,
            >(n));
            let stack3 = MemStack::new(&mut mem3);
            let t1 = Instant::now();
            simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
                &mut l_values,
                a_lower.rb(),
                regularization,
                &sym_s,
                stack3,
            )
            .expect("simplicial numeric failed");
            let num_t = t1.elapsed();

            println!(
                "[simplicial] n={n}, band={band}: l_nnz={l_nnz}, sym={sym_t:.3?}, num={num_t:.3?}"
            );
        }
        let _ = band; // suppress unused warning
    }
}
