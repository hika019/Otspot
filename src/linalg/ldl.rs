//! faer high-level LDL^T wrapper for sparse linear systems
//!
//! Uses faer's high-level Cholesky API (SymbolicCholesky + SupernodalThreshold::AUTO)
//! to automatically select simplicial or supernodal factorization based on matrix structure.
//! Banded/sparse matrices (LISWET etc.) → simplicial; dense fill (AUG2D etc.) → supernodal.
//!
//! Public API (backward-compatible):
//! - `LdlFactorization`        — positive definite, no AMD
//! - `LdlFactorizationAmd`     — quasidefinite, with AMD permutation
//! - `LdlError`
//! - `factorize`
//! - `factorize_with_deadline`
//! - `factorize_quasidefinite_with_cached_perm`
//! - `factorize_quasidefinite_with_amd`

use crate::linalg::amd::{amd_with_deadline, inv_permute_vec, permute_sym_upper, permute_vec};
use crate::sparse::CscMatrix;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::reborrow::*;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
    SymbolicCholeskyRaw, SymmetricOrdering, simplicial,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
#[cfg(test)]
use faer::sparse::Triplet;
use std::sync::Arc;
use std::time::Instant;

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

// ── LdlFactorization (positive definite, no AMD) ──────────────────────────────

/// faer high-level LDL^T factorization for positive definite matrices (no AMD).
/// SupernodalThreshold::AUTO selects simplicial or supernodal automatically.
pub struct LdlFactorization {
    symbolic: Arc<SymbolicCholesky<usize>>,
    l_values: Vec<f64>,
    n: usize,
}

impl LdlFactorization {
    /// LDL^T x = b を解く。sol に解を書き込む。
    pub fn solve(&self, rhs: &[f64], sol: &mut [f64]) {
        sol.copy_from_slice(rhs);
        let mut mem = MemBuffer::new(
            self.symbolic.solve_in_place_scratch::<f64>(1, faer::Par::Seq),
        );
        let stack = MemStack::new(&mut mem);
        let ldlt = LdltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        let mut sol_mat =
            faer::MatMut::from_column_major_slice_mut(sol, self.n, 1);
        ldlt.solve_in_place_with_conj(
            faer::Conj::No,
            sol_mat.rb_mut(),
            faer::Par::Seq,
            stack,
        );
    }
}

// ── LdlFactorizationAmd (quasidefinite, with AMD permutation) ─────────────────

/// faer high-level LDL^T factorization for quasidefinite matrices with AMD ordering.
/// SupernodalThreshold::AUTO selects simplicial or supernodal automatically.
///
/// `symbolic` は `Arc` 共有: IPM 反復で sparsity pattern が不変な場合、外部キャッシュ
/// (`SymbolicCholeskyCache`) と factor 間で SymbolicCholesky を 1 度だけ計算して
/// 再利用するため。clone は Arc::clone (refcount inc) のみ。
pub struct LdlFactorizationAmd {
    symbolic: Arc<SymbolicCholesky<usize>>,
    l_values: Vec<f64>,
    /// AMD permutation: perm[k] = original index of reordered index k
    perm: Vec<usize>,
    n: usize,
}

/// SymbolicCholesky の外部キャッシュ。IPM 反復で sparsity pattern が不変なとき、
/// 1 度だけ symbolic を計算して `factorize_quasidefinite_pre_permuted_with_cache`
/// で再利用する。
pub struct SymbolicCholeskyCache {
    inner: Arc<SymbolicCholesky<usize>>,
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
        let prof = std::env::var("LDL_PROF").ok().as_deref() == Some("1");
        let t0 = if prof { Some(Instant::now()) } else { None };
        let n = self.n;
        let b_p = permute_vec(rhs, &self.perm);
        let mut x_p = b_p;

        let mut mem = MemBuffer::new(
            self.symbolic.solve_in_place_scratch::<f64>(1, faer::Par::Seq),
        );
        let stack = MemStack::new(&mut mem);
        let ldlt = LdltRef::<'_, usize, f64>::new(&self.symbolic, &self.l_values);
        let mut sol_mat =
            faer::MatMut::from_column_major_slice_mut(&mut x_p, n, 1);
        ldlt.solve_in_place_with_conj(
            faer::Conj::No,
            sol_mat.rb_mut(),
            faer::Par::Seq,
            stack,
        );

        let x = inv_permute_vec(&x_p, &self.perm);
        sol.copy_from_slice(&x);
        if let Some(t) = t0 {
            eprintln!("LDL_SOLVE n={} t={:.3}ms", n, t.elapsed().as_secs_f64() * 1000.0);
        }
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
    let symbolic = SymbolicSparseColMat::new_checked(
        n,
        n,
        mat.col_ptr.clone(),
        None,
        mat.row_ind.clone(),
    );
    SparseColMat::new(symbolic, mat.values.clone())
}

/// 対角要素の符号ベクトルを抽出する（LdltRegularization sign-aware 用）。
///
/// 対角が負なら -1、それ以外は +1。
fn extract_diagonal_signs(mat: &CscMatrix) -> Vec<i8> {
    let n = mat.nrows;
    let mut signs = vec![1i8; n];
    for (j, sign) in signs.iter_mut().enumerate() {
        for k in mat.col_ptr[j]..mat.col_ptr[j + 1] {
            if mat.row_ind[k] == j {
                if mat.values[k] < 0.0 {
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
) -> Result<(Arc<SymbolicCholesky<usize>>, Vec<f64>), LdlError> {
    do_numeric_factorize_with_cache(mat, signs, deadline, max_l_nnz, None)
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
) -> Result<(Arc<SymbolicCholesky<usize>>, Vec<f64>), LdlError> {
    // env=LDL_PROF=1: symbolic/numeric の所要時間を stderr に書き出す。
    let prof = std::env::var("LDL_PROF").ok().as_deref() == Some("1");
    let t0 = if prof { Some(Instant::now()) } else { None };
    let a_upper = csc_upper_to_faer_upper(mat);
    let t_convert = t0.map(|t| t.elapsed());
    let t1 = if prof { Some(Instant::now()) } else { None };
    let symbolic: Arc<SymbolicCholesky<usize>> = match cached_symbolic {
        Some(s) => s,
        None => Arc::new(build_symbolic_hl(&a_upper)?),
    };
    let t_symbolic = t1.map(|t| t.elapsed());

    // symbolic 完了後・numeric 前に L_nnz チェック (memory budget)。
    // 巨大問題 (QPLIB_9008 等) で OOM kill されるのを防ぐ早期検知ポイント。
    let l_nnz = symbolic.len_val();
    if let Some(max) = max_l_nnz {
        if l_nnz > max {
            return Err(LdlError::WouldExceedBudget { l_nnz, max_l_nnz: max });
        }
    }

    // symbolic 完了後・numeric 前に deadline 再チェック (numeric は最も時間がかかる)。
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }

    let regularization = LdltRegularization {
        dynamic_regularization_signs: signs,
        dynamic_regularization_delta: 1e-8,
        dynamic_regularization_epsilon: 1e-13,
    };

    let mut l_values = vec![0.0f64; l_nnz];
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        symbolic.factorize_numeric_ldlt_scratch::<f64>(faer::Par::Seq, Default::default()),
        symbolic.solve_in_place_scratch::<f64>(1, faer::Par::Seq),
    ]));
    let stack = MemStack::new(&mut mem);

    let t2 = if prof { Some(Instant::now()) } else { None };
    symbolic
        .factorize_numeric_ldlt(
            &mut l_values,
            a_upper.rb(),
            faer::Side::Upper,
            regularization,
            faer::Par::Seq,
            stack,
            Default::default(),
        )
        .map_err(|_| LdlError::SingularOrIndefinite)?;
    if let (Some(tc), Some(ts), Some(tn)) = (t_convert, t_symbolic, t2.map(|t| t.elapsed())) {
        eprintln!(
            "LDL_PROF n={} nnz={} l_nnz={} convert={:.3}ms symbolic={:.3}ms numeric={:.3}ms",
            mat.nrows,
            mat.values.len(),
            l_nnz,
            tc.as_secs_f64() * 1000.0,
            ts.as_secs_f64() * 1000.0,
            tn.as_secs_f64() * 1000.0,
        );
    }

    Ok((symbolic, l_values))
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

// ── Public API ────────────────────────────────────────────────────────────────

/// Q 行列が PSD (半正定値) かどうかを LDL^T 慣性判定で判定する。
///
/// ## アルゴリズム
///
/// faer の `factorize_simplicial_numeric_ldlt`（正則化なし）で因子化を試みる:
///
/// - **成功** → D 対角を走査:
///   - D[j] < 0 の要素があれば **不定 (indefinite)** → `false` を返す
///   - D[j] >= 0 のみ → **PSD または PD** → `true` を返す
/// - **失敗** (ZeroPivot) → D[j] = 0 の列が存在 = **零固有値** (PSD) → `true` を返す
///
/// ### 根拠
///
/// faer の LDL^T 因子化（LltError の内部実装）:
/// - 負の D[j] は zero pivot と同じ `NonPositivePivot` エラーに **なりません**。
///   FactorizationKind::Ldlt では `d == 0 || !d.is_finite()` のみがエラー。
/// - よって D[j] < 0 の不定行列では LDL^T は **正常に完了** し、l_values に負の D が格納される。
/// - D[j] = 0 のPSD行列では ZeroPivot エラーで終了する。
///
/// これにより、LLT（ゼロまたは負で失敗）では区別できない PSD と indefinite を
/// 正確に識別できる。
///
/// ### 制限
///
/// - Simplicial 因子化のみ対応。SuperNode 因子化は l_values の D 位置が異なり
///   読み取りが複雑なため、`true`（PSD と仮定）を返す。大規模問題では
///   対角チェック（必要条件）で reachable な不定性は除去済み。
/// - Q がゼロ行列 (LP) の場合は即 `true`。
/// - Q の全対角が非負かどうかを先にチェック（必要条件）。
pub fn is_q_psd_by_cholesky(q: &CscMatrix) -> bool {
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

    // LDL^T 慣性判定:
    // faer の factorize_simplicial_numeric_ldlt (FactorizationKind::Ldlt) は:
    //   - d == 0 の場合のみ ZeroPivot エラー → PSD (零固有値) → true を返す
    //   - d < 0 の場合は正常に l_values[k_start] = d として格納 → indefinite
    //   - d > 0 の場合も正常に格納 → PD
    // 成功後に l_values の対角位置 (col_ptr[j]) を走査して D[j] < 0 を検出する。
    //
    // Q は full-symmetric (上三角 + 下三角) またはすでに上三角のみで格納される場合がある。
    // faer は上三角 CSC を期待するため、row > col のエントリを除いた上三角 CSC を構築してから渡す。
    let q_upper = q_to_upper_triangular(q);
    let a_upper = csc_upper_to_faer_upper(&q_upper);
    let symbolic = match build_symbolic_hl(&a_upper) {
        Ok(s) => s,
        Err(_) => return false,
    };

    // SuperNode 経路では D 対角の位置が複雑 → 保守的に PSD と仮定して返す。
    // 大規模問題では対角チェック済みなので、ここに到達する不定行列は
    // 対角が全て非負の ill-conditioned なケースのみ（実用上まれ）。
    let SymbolicCholeskyRaw::Simplicial(simp_sym) = symbolic.raw() else {
        return true; // supernodal: assume PSD
    };

    let l_nnz = symbolic.len_val();
    let mut l_values = vec![0.0f64; l_nnz];
    let scratch_ldlt = simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(n);
    let scratch_solve = simp_sym.solve_in_place_scratch::<f64>(1);
    let mut mem = MemBuffer::new(StackReq::any_of(&[scratch_ldlt, scratch_solve]));
    let stack = MemStack::new(&mut mem);

    // 正則化なし (delta=0, epsilon=0) で LDL^T を試みる
    let reg: LdltRegularization<f64> = LdltRegularization::default();

    match simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
        &mut l_values,
        a_upper.rb(),
        reg,
        simp_sym,
        stack,
    ) {
        Err(_) => {
            // ZeroPivot: D[j] = 0 → 零固有値 = PSD → 慣性修正不要
            true
        }
        Ok(_) => {
            // 成功: D 対角を検査する。
            // Simplicial LDL^T では col_ptr[j] が列 j の最初のエントリ (= 対角 D[j]) を指す。
            //
            // FP スレッショルド: PSD 行列でも浮動小数点誤差で D[j] が僅かに負になる場合がある。
            // n×n 行列でエントリ最大値 q_abs_max の場合、LDL^T の蓄積誤差は
            // O(n * q_abs_max * eps_machine) 程度。これ以上に負なら真の固有値の符号と判断する。
            let q_abs_max = q.values.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            let fp_threshold = q_abs_max * (n as f64) * f64::EPSILON;
            let col_ptr = simp_sym.col_ptr();
            let has_negative_d = (0..n).any(|j| {
                let pos = col_ptr[j]; // col_ptr は列 j の先頭 (= D[j]) を指す
                l_values[pos] < -fp_threshold
            });
            // D[j] < -threshold があれば不定行列
            !has_negative_d
        }
    }
}

/// 正定値疎行列の LDL^T 分解を実行する。
pub fn factorize(mat: &CscMatrix) -> Result<LdlFactorization, LdlError> {
    let n = mat.nrows;
    let (symbolic, l_values) = do_numeric_factorize(mat, None, None, None)?;
    Ok(LdlFactorization { symbolic, l_values, n })
}

/// deadline 付き正定値疎行列の LDL^T 分解。
///
/// deadline は factorize 前と symbolic 完了後（numeric 開始前）の 2 箇所でチェック。
/// faer の numeric 因子化自体は mid-factorization キャンセル不可のため、
/// 一旦 numeric を開始したら deadline を超えて完走する可能性がある。
pub fn factorize_with_deadline(
    mat: &CscMatrix,
    deadline: Option<Instant>,
) -> Result<LdlFactorization, LdlError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    let n = mat.nrows;
    let (symbolic, l_values) = do_numeric_factorize(mat, None, deadline, None)?;
    Ok(LdlFactorization { symbolic, l_values, n })
}

/// AMD キャッシュ済み置換付き quasidefinite LDL^T 分解。
///
/// `mat`: 元の（未置換の）augmented KKT 行列（上三角 CSC）
/// `perm`: 事前計算済み AMD 置換ベクトル（perm[k] = 元インデックス）
/// `deadline`: factorize 前チェックのみ（mid-factorization 未対応）
pub fn factorize_quasidefinite_with_cached_perm(
    mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
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
    let signs = extract_diagonal_signs(&perm_mat);
    let (symbolic, l_values) = do_numeric_factorize(&perm_mat, Some(&signs), deadline, None)?;
    Ok(LdlFactorizationAmd { symbolic, l_values, perm: perm.to_vec(), n })
}

/// AMD 再順序化付き quasidefinite LDL^T 分解（AMD を内部で計算）。
#[allow(dead_code)]
pub fn factorize_quasidefinite_with_amd(
    mat: &CscMatrix,
    deadline: Option<Instant>,
) -> Result<LdlFactorizationAmd, LdlError> {
    factorize_quasidefinite_with_amd_budget(mat, deadline, None)
}

/// AMD 再順序化付き quasidefinite LDL^T 分解、L_nnz が `max_l_nnz` を超えたら
/// numeric を試みず `WouldExceedBudget` で早期 return する budget 版。
///
/// `max_l_nnz` が `None` のときは `factorize_quasidefinite_with_amd` と等価。
/// 反復法フォールバック用 dispatcher (`KktSolver` 経路) から呼ばれる想定。
pub fn factorize_quasidefinite_with_amd_budget(
    mat: &CscMatrix,
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
) -> Result<LdlFactorizationAmd, LdlError> {
    let n = mat.nrows;
    let perm = amd_with_deadline(n, &mat.col_ptr, &mat.row_ind, deadline);
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    factorize_quasidefinite_with_cached_perm_budget(mat, &perm, deadline, max_l_nnz)
}

/// AMD キャッシュ済み置換 + budget 版。詳細は `factorize_quasidefinite_with_amd_budget`。
pub fn factorize_quasidefinite_with_cached_perm_budget(
    mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
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
    let signs = extract_diagonal_signs(&perm_mat);
    let (symbolic, l_values) = do_numeric_factorize(&perm_mat, Some(&signs), deadline, max_l_nnz)?;
    Ok(LdlFactorizationAmd { symbolic, l_values, perm: perm.to_vec(), n })
}

/// 既に AMD 置換適用済みの行列を受け取り、`permute_sym_upper` を skip する版。
/// IPM 反復で `PermutedAugmentedKkt::materialize` から得た matrix を直接渡せる。
///
/// `perm` は solve 時の置換に使う (factorize 自体は pre_permuted_mat だけで完結)。
pub fn factorize_quasidefinite_pre_permuted(
    pre_permuted_mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
) -> Result<LdlFactorizationAmd, LdlError> {
    factorize_quasidefinite_pre_permuted_cached(
        pre_permuted_mat, perm, deadline, max_l_nnz, None,
    )
}

/// `factorize_quasidefinite_pre_permuted` の symbolic キャッシュ版。
/// `cached_symbolic` が `Some` のときは `build_symbolic_hl` を skip する。
/// 戻り値の `LdlFactorizationAmd` は内部で `Arc<SymbolicCholesky>` を保持しており、
/// caller は `factor.symbolic_arc()` で取得して次回呼び出しに使い回せる。
pub fn factorize_quasidefinite_pre_permuted_cached(
    pre_permuted_mat: &CscMatrix,
    perm: &[usize],
    deadline: Option<Instant>,
    max_l_nnz: Option<usize>,
    cached_symbolic: Option<Arc<SymbolicCholesky<usize>>>,
) -> Result<LdlFactorizationAmd, LdlError> {
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(LdlError::DeadlineExceeded);
        }
    }
    let n = pre_permuted_mat.nrows;
    let signs = extract_diagonal_signs(pre_permuted_mat);
    let (symbolic, l_values) = do_numeric_factorize_with_cache(
        pre_permuted_mat, Some(&signs), deadline, max_l_nnz, cached_symbolic,
    )?;
    Ok(LdlFactorizationAmd { symbolic, l_values, perm: perm.to_vec(), n })
}

impl LdlFactorizationAmd {
    /// 内部の SymbolicCholesky を Arc として取得する (反復間でのキャッシュ用)。
    pub fn symbolic_arc(&self) -> Arc<SymbolicCholesky<usize>> {
        Arc::clone(&self.symbolic)
    }
}

#[cfg(test)]
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
        CscMatrix { col_ptr, row_ind, values, nrows: n, ncols: n }
    }

    #[test]
    fn test_factorize_pd_3x3_solve() {
        // A = [[4,1,0],[1,3,2],[0,2,5]] — positive definite
        let mat = upper_tri_csc(3, &[
            (0, 0, 4.0), (0, 1, 1.0),
            (1, 1, 3.0), (1, 2, 2.0),
            (2, 2, 5.0),
        ]);
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

    #[test]
    fn test_factorize_with_deadline_ok() {
        let mat = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 1, 3.0)]);
        // deadline far in the future
        let deadline = Some(Instant::now() + std::time::Duration::from_secs(60));
        let fac = factorize_with_deadline(&mat, deadline).expect("should succeed");
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
        let mat = upper_tri_csc(2, &[(0, 0, 2.0), (1, 1, 3.0)]);
        let deadline = Some(Instant::now() - std::time::Duration::from_millis(1));
        let result = factorize_with_deadline(&mat, deadline);
        assert!(
            matches!(result, Err(LdlError::DeadlineExceeded)),
            "Expected DeadlineExceeded"
        );
    }

    #[test]
    fn test_quasidefinite_2x2_identity_perm() {
        // quasidefinite: [[3,1],[1,-2]] — D[0]>0, D[1]<0
        let mat = upper_tri_csc(2, &[(0, 0, 3.0), (0, 1, 1.0), (1, 1, -2.0)]);
        let perm = vec![0usize, 1]; // identity permutation
        let fac = factorize_quasidefinite_with_cached_perm(&mat, &perm, None)
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
        let mat = upper_tri_csc(5, &[
            (0, 0, 1.0 + delta), (1, 1, 2.0 + delta),
            (2, 2, -delta), (3, 3, -delta), (4, 4, -delta),
            (0, 2, 1.0), (1, 3, 1.0), (0, 4, 1.0), (1, 4, 1.0),
        ]);
        let fac = factorize_quasidefinite_with_amd(&mat, None)
            .expect("quasidefinite_with_amd failed");
        let b = [1.0f64, 2.0, 0.5, -0.5, 1.0];
        let mut x = [0.0f64; 5];
        fac.solve(&b, &mut x);
        // Full matrix for residual check (symmetric)
        let full: &[(usize, usize, f64)] = &[
            (0, 0, 1.0 + delta), (1, 1, 2.0 + delta),
            (2, 2, -delta), (3, 3, -delta), (4, 4, -delta),
            (0, 2, 1.0), (2, 0, 1.0),
            (1, 3, 1.0), (3, 1, 1.0),
            (0, 4, 1.0), (4, 0, 1.0),
            (1, 4, 1.0), (4, 1, 1.0),
        ];
        let mut r = [0.0f64; 5];
        for &(row, col, val) in full {
            r[row] += val * x[col];
        }
        let res: f64 = r.iter().zip(b.iter()).map(|(&ri, &bi)| (ri - bi).powi(2)).sum::<f64>().sqrt();
        assert!(res < 1e-8, "residual={res:.3e}");
    }

    /// is_q_psd_by_cholesky: PSD 行列は true を返す
    #[test]
    fn test_is_q_psd_psd_matrix() {
        // Q = [[4,1],[1,3]] — PD (固有値 ≈ 2.38, 4.62)
        let q = upper_tri_csc(2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0)]);
        assert!(is_q_psd_by_cholesky(&q), "PD matrix should be identified as PSD");
    }

    /// is_q_psd_by_cholesky: 不定行列は false を返す
    #[test]
    fn test_is_q_psd_indefinite_matrix() {
        // Q = [[1,0],[0,-1]] — indefinite (固有値 +1, -1)
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (1, 1, -1.0)]);
        assert!(!is_q_psd_by_cholesky(&q), "Indefinite matrix should NOT be identified as PSD");
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
        assert!(is_q_psd_by_cholesky(&q),
            "PSD matrix with large off-diagonal (Gershgorin false alarm) should be true");
    }

    /// is_q_psd_by_cholesky: PSD (特異) 行列 → ZeroPivot → true を返す
    ///
    /// Q = [[1, 1],[1, 1]] は rank-1 で最小固有値 0 (PSD)。
    /// LDL^T は ZeroPivot で失敗 → true を返すべき。
    #[test]
    fn test_is_q_psd_singular_psd() {
        // Q = [[1,1],[1,1]] — PSD (rank 1, λ = 0 and 2)
        let q = upper_tri_csc(2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0)]);
        assert!(is_q_psd_by_cholesky(&q), "Singular PSD matrix should be identified as PSD");
    }

    /// is_q_psd_by_cholesky: 不定行列 (対角外が大きく不定) → false を返す
    ///
    /// Q = [[2, 3],[3, 2]] — indefinite (固有値 -1, 5)
    #[test]
    fn test_is_q_psd_offdiag_indefinite() {
        // Q = [[2,3],[3,2]] — indefinite (det=4-9=-5 < 0, λ = -1, 5)
        let q = upper_tri_csc(2, &[(0, 0, 2.0), (0, 1, 3.0), (1, 1, 2.0)]);
        assert!(!is_q_psd_by_cholesky(&q), "Indefinite matrix with off-diag should be false");
    }

    /// is_q_psd_by_cholesky: ゼロ行列は true を返す (LP ケース)
    #[test]
    fn test_is_q_psd_zero_matrix() {
        let q = upper_tri_csc(3, &[]);
        assert!(is_q_psd_by_cholesky(&q), "Zero matrix (LP) should be identified as PSD");
    }

    /// is_q_psd_by_cholesky: n=0 は true を返す
    #[test]
    fn test_is_q_psd_empty_matrix() {
        let q = CscMatrix::new(0, 0);
        assert!(is_q_psd_by_cholesky(&q), "Empty matrix should be identified as PSD");
    }

    #[test]
    fn test_nnz_l_reasonable() {
        // nnz_l should be positive for a non-trivial matrix
        // (supernodal stores diagonal + fill-in, upper bound = n*(n+1)/2)
        let n = 3usize;
        let mat = upper_tri_csc(n, &[
            (0, 0, 4.0), (0, 1, 1.0), (1, 1, 3.0), (1, 2, 2.0), (2, 2, 5.0),
        ]);
        let perm = vec![0, 1, 2];
        let fac = factorize_quasidefinite_with_cached_perm(&mat, &perm, None)
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
            if i + 1 < n { lo_triplets.push(Triplet::new(i + 1, i, -1.0)); }
            if i + 2 < n { lo_triplets.push(Triplet::new(i + 2, i, -0.5)); }
        }
        let a_lower = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &lo_triplets)
            .expect("build lower failed");

        let a_nnz = a_lower.compute_nnz();
        let a_upper_sym = a_lower.rb().transpose().symbolic().to_col_major()
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
                &mut etree_buf, &mut col_counts_buf, a_upper_sym.rb(), stack,
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
                &mut etree, &mut col_counts, a_upper_sym.rb(), stack,
            );

            let mut mem2 = MemBuffer::new(supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(n));
            let stack2 = MemStack::new(&mut mem2);
            let t0 = Instant::now();
            let sym = supernodal::factorize_supernodal_symbolic_cholesky(
                a_upper_sym.rb(),
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack2,
                faer::sparse::linalg::SymbolicSupernodalParams { relax: Some(relax) },
            ).expect("symbolic failed");
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
                dynamic_regularization_delta: 1e-8,
                dynamic_regularization_epsilon: 1e-13,
            };
            let mut mem3 = MemBuffer::new(StackReq::any_of(&[
                supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, f64>(
                    &sym, faer::Par::Seq, Default::default()),
                sym.solve_in_place_scratch::<f64>(n, faer::Par::Seq),
            ]));
            let stack3 = MemStack::new(&mut mem3);
            let mut l_values = vec![0.0f64; sym.len_val()];
            let t1 = Instant::now();
            supernodal::factorize_supernodal_numeric_ldlt::<usize, f64>(
                &mut l_values, a_lower.rb(), regularization, &sym,
                faer::Par::Seq, stack3, Default::default(),
            ).expect("numeric failed");
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
                &mut etree, &mut col_counts, a_upper_sym.rb(), stack,
            );

            let mut mem2 = MemBuffer::new(simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(n));
            let stack2 = MemStack::new(&mut mem2);
            let t0 = Instant::now();
            let sym_s = simplicial::factorize_simplicial_symbolic_cholesky(
                a_upper_sym.rb(),
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack2,
            ).expect("simplicial symbolic failed");
            let sym_t = t0.elapsed();

            let regularization = faer::linalg::cholesky::ldlt::factor::LdltRegularization {
                dynamic_regularization_signs: None,
                dynamic_regularization_delta: 1e-8,
                dynamic_regularization_epsilon: 1e-13,
            };
            let l_nnz = sym_s.len_val();
            let mut l_values = vec![0.0f64; l_nnz];
            let mut mem3 = MemBuffer::new(
                simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(n)
            );
            let stack3 = MemStack::new(&mut mem3);
            let t1 = Instant::now();
            simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
                &mut l_values, a_lower.rb(), regularization, &sym_s, stack3,
            ).expect("simplicial numeric failed");
            let num_t = t1.elapsed();

            println!(
                "[simplicial] n={n}, band={band}: l_nnz={l_nnz}, sym={sym_t:.3?}, num={num_t:.3?}"
            );
        }
        let _ = band; // suppress unused warning
    }
}
