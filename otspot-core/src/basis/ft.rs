//! Bartels-Golub-Reid (BGR) 変種 LU solve インフラ (Phase 2a/2b)
//!
//! Phase 2a: 自前の可変 U 表現と FT-aware solve を構築し、
//! 更新ゼロの状態で `LuFactorization` との solve 一致を確認する。
//! Phase 2b でこの土台の上に BGR rank-1 更新を実装する。
//!
//! ## solve 順序
//! - FTRAN: `x = P_c⁻¹ · U⁻¹ · ft_etas · L0⁻¹ · P_r · rhs`
//! - BTRAN: `x = P_r⁻¹ · L0⁻ᵀ · ft_etas^ᵀ · U⁻ᵀ · P_c · rhs`

use super::lu::LuFactorization;
use crate::error::SolverError;
use crate::sparse::{CscMatrix, SparseVec};
use crate::tolerances::{DROP_TOL, PIVOT_STABILITY_THRESHOLD, ZERO_TOL};
use faer::sparse::SparseColMatRef;
use std::time::Instant;

/// 可変 U 行列 (CSC, 行 index 昇順, 対角ポインタ保持)。
///
/// Phase 2b の BGR rank-1 更新が列を書き換える対象。
/// `diag_ptr[j]` は列 j の U[j,j] の row_ind/values 上の絶対インデックス。
#[derive(Debug, Clone)]
pub(crate) struct MutableU {
    pub(crate) n: usize,
    pub(crate) col_ptr: Vec<usize>,
    pub(crate) row_ind: Vec<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) diag_ptr: Vec<usize>,
}

impl MutableU {
    /// faer の U 因子 (行 index 未ソート) から構築する。列内行 index を昇順にソートする。
    pub(crate) fn from_faer(n: usize, u_ref: &SparseColMatRef<'_, usize, f64>) -> Self {
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_ind_all: Vec<usize> = Vec::new();
        let mut values_all: Vec<f64> = Vec::new();
        let mut diag_ptr = vec![usize::MAX; n];
        let mut tmp: Vec<(usize, f64)> = Vec::new();

        for j in 0..n {
            tmp.clear();
            for (row, &val) in u_ref
                .row_idx_of_col(j)
                .zip(u_ref.val_of_col(j).iter())
            {
                tmp.push((row, val));
            }
            tmp.sort_unstable_by_key(|&(r, _)| r);

            let base = row_ind_all.len();
            for (k, &(row, val)) in tmp.iter().enumerate() {
                if row == j {
                    diag_ptr[j] = base + k;
                }
                row_ind_all.push(row);
                values_all.push(val);
            }
            col_ptr[j + 1] = row_ind_all.len();
        }

        debug_assert!(
            diag_ptr.iter().all(|&p| p != usize::MAX),
            "U factor is missing diagonal entry — basis may be singular"
        );

        Self {
            n,
            col_ptr,
            row_ind: row_ind_all,
            values: values_all,
            diag_ptr,
        }
    }

    /// 後退代入: `U · y = rhs` の解を rhs に in-place 上書き。
    pub(crate) fn backward_sub(&self, y: &mut [f64]) {
        for j in (0..self.n).rev() {
            y[j] /= self.values[self.diag_ptr[j]];
            let yj = y[j];
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let row = self.row_ind[k];
                if row < j {
                    y[row] -= self.values[k] * yj;
                }
            }
        }
    }

    /// 前進代入: `U^T · y = rhs` の解を rhs に in-place 上書き。
    pub(crate) fn forward_sub_transpose(&self, y: &mut [f64]) {
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let row = self.row_ind[k];
                if row < j {
                    y[j] -= self.values[k] * y[row];
                }
            }
            y[j] /= self.values[self.diag_ptr[j]];
        }
    }

    /// 各列を `(row, val)` リストに展開する (O(nnz))。FT 更新の作業表現。
    pub(crate) fn to_columns(&self) -> Vec<Vec<(usize, f64)>> {
        (0..self.n)
            .map(|j| {
                (self.col_ptr[j]..self.col_ptr[j + 1])
                    .map(|k| (self.row_ind[k], self.values[k]))
                    .collect()
            })
            .collect()
    }

    /// 列リストから再構築する。各列を行昇順にソートし diag_ptr を張る。
    /// 対角 (row==j) は値によらず保持し、off-diagonal は `|v| ≤ ZERO_TOL` を除去する。
    /// 対角を欠く列があれば `None` (特異)。
    pub(crate) fn from_columns(n: usize, cols: &[Vec<(usize, f64)>]) -> Option<Self> {
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_ind: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();
        let mut diag_ptr = vec![usize::MAX; n];
        let mut tmp: Vec<(usize, f64)> = Vec::new();

        for j in 0..n {
            tmp.clear();
            for &(r, v) in &cols[j] {
                if r == j || v.abs() > ZERO_TOL {
                    tmp.push((r, v));
                }
            }
            tmp.sort_unstable_by_key(|&(r, _)| r);

            let base = row_ind.len();
            let mut has_diag = false;
            for (k, &(r, v)) in tmp.iter().enumerate() {
                if r == j {
                    diag_ptr[j] = base + k;
                    has_diag = true;
                }
                row_ind.push(r);
                values.push(v);
            }
            if !has_diag {
                return None;
            }
            col_ptr[j + 1] = row_ind.len();
        }

        Some(Self {
            n,
            col_ptr,
            row_ind,
            values,
            diag_ptr,
        })
    }
}

/// faer の L 因子 (unit lower triangular) の前進代入。
/// 対角は unit = 1 のため row == j エントリをスキップする。
fn forward_sub_l(n: usize, l_ref: &SparseColMatRef<'_, usize, f64>, y: &mut [f64]) {
    for j in 0..n {
        let yj = y[j];
        for (row, &val) in l_ref
            .row_idx_of_col(j)
            .zip(l_ref.val_of_col(j).iter())
        {
            if row > j {
                y[row] -= val * yj;
            }
        }
    }
}

/// faer の L^T (unit upper triangular) の後退代入。
/// 対角は unit = 1 のため除算不要。
fn backward_sub_lt(n: usize, l_ref: &SparseColMatRef<'_, usize, f64>, y: &mut [f64]) {
    for j in (0..n).rev() {
        for (row, &val) in l_ref
            .row_idx_of_col(j)
            .zip(l_ref.val_of_col(j).iter())
        {
            if row > j {
                y[j] -= val * y[row];
            }
        }
        // unit lower diagonal = 1: 除算不要
    }
}

/// BGR 行更新の素 (elementary) 操作。working frame で `L0⁻¹` と `U` の間に作用する。
///
/// `Swap` は部分ピボットの隣接行交換、`Axpy` は `v[target] -= mult · v[source]` の
/// 行消去。`U_new = (Op_k · … · Op_1) · U_H` を満たすよう順に蓄積され、ftran では
/// 記録順、btran では転置を逆順に適用する。
#[derive(Debug, Clone, Copy)]
enum FtOp {
    Swap(usize, usize),
    Axpy {
        target: usize,
        source: usize,
        mult: f64,
    },
}

/// BGR 行操作を ftran 方向 (記録順) に適用する。
fn apply_ft_ops_ftran(ops: &[FtOp], v: &mut [f64]) {
    for op in ops {
        match *op {
            FtOp::Swap(a, b) => v.swap(a, b),
            FtOp::Axpy {
                target,
                source,
                mult,
            } => v[target] -= mult * v[source],
        }
    }
}

/// BGR 行操作を btran 方向 (転置・逆順) に適用する。
fn apply_ft_ops_btran(ops: &[FtOp], v: &mut [f64]) {
    for op in ops.iter().rev() {
        match *op {
            FtOp::Swap(a, b) => v.swap(a, b),
            FtOp::Axpy {
                target,
                source,
                mult,
            } => v[source] -= mult * v[target],
        }
    }
}

/// Bartels-Golub-Reid (BGR) 変種 LU solve 構造体。
///
/// `L0` (faer unit-lower) と行置換 `Pr` は初期分解で固定し、基底列の差替を
/// `u_mat` の書き換え + 行 eta (`ft_ops`) + 列巡回置換 (`col_perm`) で吸収する
/// (BGR 変種: L 固定・U と row-eta が成長)。
#[derive(Clone)]
pub(crate) struct FtLu {
    pub(crate) n: usize,
    pub(crate) lu0: LuFactorization,
    pub(crate) u_mat: MutableU,
    row_perm_fwd: Vec<usize>,
    row_perm_inv: Vec<usize>,
    col_perm_fwd: Vec<usize>,
    col_perm_inv: Vec<usize>,
    /// FT 行操作列 (全更新の時系列連結)。空なら初期分解と等価。
    ft_ops: Vec<FtOp>,
    /// 現在の基底列インデックス (basis 位置 → A 列)。
    basis_indices: Vec<usize>,
    /// 元の制約行列 (rank-1 更新で参照する)。
    a: CscMatrix,
    /// 小 pivot / eta 蓄積で再分解が望ましい場合に立つ。
    needs_refactor: bool,
}

impl FtLu {
    /// 初期因子分解。`deadline` は None = 無制限。
    pub(crate) fn new_timed(
        a: &CscMatrix,
        basis: &[usize],
        deadline: Option<Instant>,
    ) -> Result<Self, SolverError> {
        let lu0 = LuFactorization::factorize_timed(a, basis, deadline)?;
        let n = lu0.n;

        let (row_perm_fwd, row_perm_inv) = {
            let rp = lu0.row_perm();
            let (fwd, inv) = rp.arrays();
            (fwd.to_vec(), inv.to_vec())
        };
        let (col_perm_fwd, col_perm_inv) = {
            let cp = lu0.col_perm();
            let (fwd, inv) = cp.arrays();
            (fwd.to_vec(), inv.to_vec())
        };
        let u_mat = {
            let u_ref = lu0.u_factor();
            MutableU::from_faer(n, &u_ref)
        };

        Ok(Self {
            n,
            lu0,
            u_mat,
            row_perm_fwd,
            row_perm_inv,
            col_perm_fwd,
            col_perm_inv,
            ft_ops: Vec::new(),
            basis_indices: basis.to_vec(),
            a: a.clone(),
            needs_refactor: false,
        })
    }

    /// deadline なし版 (テスト・内部利用)。
    #[cfg(test)]
    pub(crate) fn new(a: &CscMatrix, basis: &[usize]) -> Result<Self, SolverError> {
        Self::new_timed(a, basis, None)
    }

    /// 再分解が望ましいか (小 pivot 等)。
    pub(crate) fn needs_refactor(&self) -> bool {
        self.needs_refactor
    }

    /// 蓄積 FT 行操作数 (eta_count 相当)。
    pub(crate) fn ft_op_count(&self) -> usize {
        self.ft_ops.len()
    }

    /// FTRAN (dense): `B · x = rhs` を in-place で解く。
    pub(crate) fn ftran(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = (0..n).map(|p| rhs[self.row_perm_fwd[p]]).collect();
        {
            let l = self.lu0.l_factor();
            forward_sub_l(n, &l, &mut y);
        }
        apply_ft_ops_ftran(&self.ft_ops, &mut y);
        self.u_mat.backward_sub(&mut y);
        for j in 0..n {
            rhs[self.col_perm_fwd[j]] = y[j];
        }
    }

    /// BTRAN (dense): `B^T · x = rhs` を in-place で解く。
    pub(crate) fn btran(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = (0..n).map(|j| rhs[self.col_perm_fwd[j]]).collect();
        self.u_mat.forward_sub_transpose(&mut y);
        apply_ft_ops_btran(&self.ft_ops, &mut y);
        {
            let l = self.lu0.l_factor();
            backward_sub_lt(n, &l, &mut y);
        }
        for i in 0..n {
            rhs[i] = y[self.row_perm_inv[i]];
        }
    }

    /// FTRAN (sparse wrapper)。
    pub(crate) fn ftran_sparse(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        self.ftran(&mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    /// BTRAN (sparse wrapper)。
    pub(crate) fn btran_sparse(&self, rhs: &mut SparseVec) {
        let mut dense = rhs.to_dense();
        self.btran(&mut dense);
        *rhs = SparseVec::from_dense(&dense);
    }

    /// BGR (Bartels-Golub-Reid) 基底更新: basis 位置 `leaving_row` の列を `entering_col` (A の列) で
    /// 置換する。
    ///
    /// 手順 (working frame):
    /// 1. spike `s = ft_ops · L0⁻¹ · Pr · a_q` (U⁻¹ 直前の部分 FTRAN)。
    /// 2. U 列 `c = col_perm_inv[leaving_row]` を s で置換し、列 `c..=t` を巡回左シフトして
    ///    bump を upper-Hessenberg 化 (t = spike 下端)。
    /// 3. bump (rows/cols `c..=t`) を dense 抽出し、subdiagonal を前進消去 (部分ピボットで
    ///    隣接行交換)。乗数・交換を `FtOp` 列に記録。
    /// 4. 記録した行操作を bump 右側の列 (>t) に sparse に replay し、`u_mat` を upper-triangular
    ///    に書き戻す (PFI と異なり U の値自体を書き換える)。col_perm に巡回を畳み込む。
    /// 5. 中間 pivot と最終 pivot が小さければ `needs_refactor`、`< DROP_TOL` なら `SingularBasis`。
    pub(crate) fn update(
        &mut self,
        entering_col: usize,
        leaving_row: usize,
    ) -> Result<(), SolverError> {
        let n = self.n;

        // 1. 入基列 a_q を basis 行空間に dense 展開し、spike を部分 FTRAN で求める。
        let mut a_q = vec![0.0f64; n];
        let (rows, vals) = self.a.get_column(entering_col)?;
        for (&r, &v) in rows.iter().zip(vals.iter()) {
            if r < n {
                a_q[r] = v;
            }
        }
        let mut s: Vec<f64> = (0..n).map(|p| a_q[self.row_perm_fwd[p]]).collect();
        {
            let l = self.lu0.l_factor();
            forward_sub_l(n, &l, &mut s);
        }
        apply_ft_ops_ftran(&self.ft_ops, &mut s);
        let snorm_inf = s.iter().fold(0.0f64, |m, &v| m.max(v.abs()));

        // 2. working 列 c と bump 下端 t。
        let c = self.col_perm_inv[leaving_row];
        let mut t = 0usize;
        let mut found = false;
        for (i, &si) in s.iter().enumerate() {
            if si.abs() > ZERO_TOL {
                t = i;
                found = true;
            }
        }
        if !found || t < c {
            // spike が全ゼロ、または対角位置 c に成分が無い → 特異。
            self.needs_refactor = true;
            return Err(SolverError::SingularBasis { step: leaving_row });
        }

        if t == c {
            // tail 無し: 列 c を s で単純置換 (巡回・消去不要)。
            let pivot = s[c];
            if pivot.abs() < DROP_TOL {
                self.needs_refactor = true;
                return Err(SolverError::SingularBasis { step: leaving_row });
            }
            let mut cols = self.u_mat.to_columns();
            cols[c] = (0..=c)
                .filter_map(|i| (i == c || s[i].abs() > ZERO_TOL).then_some((i, s[i])))
                .collect();
            self.u_mat = MutableU::from_columns(n, &cols)
                .ok_or(SolverError::SingularBasis { step: leaving_row })?;
            self.basis_indices[leaving_row] = entering_col;
            if pivot.abs() < PIVOT_STABILITY_THRESHOLD * snorm_inf {
                self.needs_refactor = true;
            }
            return Ok(());
        }

        // --- bump case (t > c) ---
        let bsz = t - c + 1;
        let mut cols = self.u_mat.to_columns();

        // (a) 列 c を spike で置換 (U_spike)。
        cols[c] = (0..=t)
            .filter_map(|i| (s[i].abs() > ZERO_TOL).then_some((i, s[i])))
            .collect();

        // (b) 巡回列シフト: cols[c..=t] を左回転 (spike を末尾 t へ) → upper-Hessenberg。
        cols[c..=t].rotate_left(1);

        // (c) bump 行 c..t を dense (bsz×bsz) に取り出し、各 bump 列の above (row<c) を保存。
        let mut bump = vec![0.0f64; bsz * bsz];
        let mut above: Vec<Vec<(usize, f64)>> = vec![Vec::new(); bsz];
        for b in 0..bsz {
            for &(r, v) in &cols[c + b] {
                if r < c {
                    above[b].push((r, v));
                } else {
                    debug_assert!(r <= t, "Hessenberg bump column has entry below t");
                    bump[(r - c) * bsz + b] = v;
                }
            }
        }

        // (d) Hessenberg 前進消去 (部分ピボット)。行操作を ops に記録。
        let mut ops: Vec<FtOp> = Vec::new();
        for j in 0..bsz - 1 {
            if bump[(j + 1) * bsz + j].abs() > bump[j * bsz + j].abs() {
                for b in 0..bsz {
                    bump.swap(j * bsz + b, (j + 1) * bsz + b);
                }
                ops.push(FtOp::Swap(c + j, c + j + 1));
            }
            let pivot = bump[j * bsz + j];
            if pivot.abs() < DROP_TOL {
                self.needs_refactor = true;
                return Err(SolverError::SingularBasis { step: leaving_row });
            }
            // 中間 bump pivot が不安定 → backward_sub の除数として精度劣化するため refactor 要求。
            // snorm_inf (spike の L-∞ ノルム) を全 bump 列共通の scale に流用している。
            // より厳密には bump 列ごとの L-∞ ノルムを使うべきだが、spike が最悪 scale を
            // 代表するため実用上十分。 per-column scale が必要な場合は Phase 3 再検討。
            if pivot.abs() < PIVOT_STABILITY_THRESHOLD * snorm_inf {
                self.needs_refactor = true;
            }
            let mult = bump[(j + 1) * bsz + j] / pivot;
            if mult != 0.0 {
                for b in 0..bsz {
                    bump[(j + 1) * bsz + b] -= mult * bump[j * bsz + b];
                }
                bump[(j + 1) * bsz + j] = 0.0;
                ops.push(FtOp::Axpy {
                    target: c + j + 1,
                    source: c + j,
                    mult,
                });
            }
        }
        let final_pivot = bump[(bsz - 1) * bsz + (bsz - 1)];
        if final_pivot.abs() < DROP_TOL {
            self.needs_refactor = true;
            return Err(SolverError::SingularBasis { step: leaving_row });
        }

        // (e) bump 右側の列 (>t) に行操作を replay (sparse fill)。
        for k in (t + 1)..n {
            let mut rowvals = vec![0.0f64; bsz];
            let mut other: Vec<(usize, f64)> = Vec::new();
            for &(r, v) in &cols[k] {
                if r >= c && r <= t {
                    rowvals[r - c] = v;
                } else {
                    other.push((r, v));
                }
            }
            for op in &ops {
                match *op {
                    FtOp::Swap(g1, g2) => rowvals.swap(g1 - c, g2 - c),
                    FtOp::Axpy {
                        target,
                        source,
                        mult,
                    } => rowvals[target - c] -= mult * rowvals[source - c],
                }
            }
            for (a, &rv) in rowvals.iter().enumerate() {
                if rv.abs() > ZERO_TOL {
                    other.push((c + a, rv));
                }
            }
            cols[k] = other;
        }

        // (f) bump 列 (c..=t) を再構築 (above + dense bump; 対角は値によらず保持)。
        for b in 0..bsz {
            let mut newcol = std::mem::take(&mut above[b]);
            for a in 0..bsz {
                let v = bump[a * bsz + b];
                if a == b || v.abs() > ZERO_TOL {
                    newcol.push((c + a, v));
                }
            }
            cols[c + b] = newcol;
        }

        // (g) u_mat 再構築。
        self.u_mat = MutableU::from_columns(n, &cols)
            .ok_or(SolverError::SingularBasis { step: leaving_row })?;

        // (h) col_perm に巡回 C を畳み込み、inv を再構築。
        self.col_perm_fwd[c..=t].rotate_left(1);
        for (j, &orig) in self.col_perm_fwd.iter().enumerate() {
            self.col_perm_inv[orig] = j;
        }

        // (i) 行操作を時系列で連結、basis 更新、安定性フラグ。
        self.ft_ops.extend(ops);
        self.basis_indices[leaving_row] = entering_col;
        if final_pivot.abs() < PIVOT_STABILITY_THRESHOLD * snorm_inf {
            self.needs_refactor = true;
        }
        Ok(())
    }

    /// テスト用: spike `s` と working 列位置 `c`・bump 下端 `t` を返す。
    #[cfg(test)]
    fn debug_spike(&self, entering_col: usize, leaving_row: usize) -> (Vec<f64>, usize, usize) {
        let n = self.n;
        let mut a_q = vec![0.0f64; n];
        let (rows, vals) = self.a.get_column(entering_col).unwrap();
        for (&r, &v) in rows.iter().zip(vals.iter()) {
            if r < n {
                a_q[r] = v;
            }
        }
        let mut s: Vec<f64> = (0..n).map(|p| a_q[self.row_perm_fwd[p]]).collect();
        {
            let l = self.lu0.l_factor();
            forward_sub_l(n, &l, &mut s);
        }
        apply_ft_ops_ftran(&self.ft_ops, &mut s);
        let c = self.col_perm_inv[leaving_row];
        let mut t = 0usize;
        for (i, &si) in s.iter().enumerate() {
            if si.abs() > ZERO_TOL {
                t = i;
            }
        }
        (s, c, t)
    }

    /// テスト用 (no-op proof): 巡回列シフトと Hessenberg 消去を省いた素朴な列置換。
    /// U に subdiagonal が残り backward_sub が破綻するため solve 残差が爆発する。
    #[cfg(test)]
    fn update_naive_no_ft(&mut self, entering_col: usize, leaving_row: usize) {
        let n = self.n;
        let mut a_q = vec![0.0f64; n];
        let (rows, vals) = self.a.get_column(entering_col).unwrap();
        for (&r, &v) in rows.iter().zip(vals.iter()) {
            if r < n {
                a_q[r] = v;
            }
        }
        let mut s: Vec<f64> = (0..n).map(|p| a_q[self.row_perm_fwd[p]]).collect();
        {
            let l = self.lu0.l_factor();
            forward_sub_l(n, &l, &mut s);
        }
        apply_ft_ops_ftran(&self.ft_ops, &mut s);
        let c = self.col_perm_inv[leaving_row];
        let mut cols = self.u_mat.to_columns();
        cols[c] = (0..n)
            .filter_map(|i| (s[i].abs() > ZERO_TOL).then_some((i, s[i])))
            .collect();
        if let Some(u) = MutableU::from_columns(n, &cols) {
            self.u_mat = u;
        }
        self.basis_indices[leaving_row] = entering_col;
    }

    /// テスト用: U 対角を 1.0 に固定した ftran (no-op pivot sentinel 確認用)。
    #[cfg(test)]
    fn ftran_unit_pivot(&self, rhs: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = (0..n).map(|p| rhs[self.row_perm_fwd[p]]).collect();
        {
            let l = self.lu0.l_factor();
            forward_sub_l(n, &l, &mut y);
        }
        apply_ft_ops_ftran(&self.ft_ops, &mut y);
        let mut broken = self.u_mat.clone();
        for j in 0..n {
            broken.values[broken.diag_ptr[j]] = 1.0;
        }
        broken.backward_sub(&mut y);
        for j in 0..n {
            rhs[self.col_perm_fwd[j]] = y[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::lu::{solve_btran, solve_ftran, LuFactorization};
    use crate::basis::test_utils::{assert_vec_near, dense_to_csc};
    use crate::sparse::CscMatrix;

    /// 決定論的な LCG で n×n の対角優位疎行列を生成する (非特異性を対角優位で保証)。
    fn gen_matrix(n: usize, seed: u64) -> CscMatrix {
        let mut lcg = seed;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(6.0 + (i as f64 * 0.7 + seed as f64 * 0.1).sin().abs() * 2.0);
        }
        for _ in 0..(n * 2) {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let i = ((lcg >> 33) as usize) % n;
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((lcg >> 33) as usize) % n;
            if i != j {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((lcg >> 32) as f64 / u32::MAX as f64 - 0.5) * 0.8;
                rows.push(i);
                cols.push(j);
                vals.push(v);
            }
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap()
    }

    fn gen_rhs(n: usize, seed: u64) -> Vec<f64> {
        let mut lcg = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((lcg >> 32) as f64 / u32::MAX as f64) * 10.0 - 5.0
            })
            .collect()
    }

    /// sentinel: FtLu.ftran が LuFactorization と 1e-10 内で一致し、B·x=rhs 残差 < 1e-10。
    /// 8 seed × 5 rhs、サイズ 10/20/30。
    #[test]
    fn test_ftlu_ftran_matches_lu() {
        let configs: &[(usize, &[u64])] = &[
            (10, &[1, 2, 3]),
            (20, &[10, 20, 30]),
            (30, &[100, 200]),
        ];
        for &(n, seeds) in configs {
            for &seed in seeds {
                let a = gen_matrix(n, seed);
                let basis: Vec<usize> = (0..n).collect();
                let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();
                let ft = FtLu::new(&a, &basis).unwrap();

                for rhs_seed in 0..5u64 {
                    let rhs_orig = gen_rhs(n, seed * 100 + rhs_seed);

                    let mut rhs_lu = rhs_orig.clone();
                    solve_ftran(&lu, &mut rhs_lu);

                    let mut rhs_ft = rhs_orig.clone();
                    ft.ftran(&mut rhs_ft);

                    let max_diff: f64 = rhs_lu
                        .iter()
                        .zip(rhs_ft.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        max_diff < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: ftran diff={max_diff:.2e}"
                    );

                    let check = a.mat_vec_mul(&rhs_ft).unwrap();
                    let residual: f64 = check
                        .iter()
                        .zip(rhs_orig.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        residual < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: B·x=rhs residual={residual:.2e}"
                    );
                }
            }
        }
    }

    /// sentinel: FtLu.btran が LuFactorization と 1e-10 内で一致し、B^T·x=rhs 残差 < 1e-10。
    #[test]
    fn test_ftlu_btran_matches_lu() {
        let configs: &[(usize, &[u64])] = &[
            (10, &[1, 2, 3]),
            (20, &[10, 20, 30]),
            (30, &[100, 200]),
        ];
        for &(n, seeds) in configs {
            for &seed in seeds {
                let a = gen_matrix(n, seed);
                let basis: Vec<usize> = (0..n).collect();
                let lu = LuFactorization::factorize_timed(&a, &basis, None).unwrap();
                let ft = FtLu::new(&a, &basis).unwrap();

                for rhs_seed in 0..5u64 {
                    let rhs_orig = gen_rhs(n, seed * 100 + rhs_seed + 500);

                    let mut rhs_lu = rhs_orig.clone();
                    solve_btran(&lu, &mut rhs_lu);

                    let mut rhs_ft = rhs_orig.clone();
                    ft.btran(&mut rhs_ft);

                    let max_diff: f64 = rhs_lu
                        .iter()
                        .zip(rhs_ft.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        max_diff < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: btran diff={max_diff:.2e}"
                    );

                    let bt = a.transpose();
                    let check = bt.mat_vec_mul(&rhs_ft).unwrap();
                    let residual: f64 = check
                        .iter()
                        .zip(rhs_orig.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max);
                    assert!(
                        residual < 1e-10,
                        "n={n} seed={seed} rhs_seed={rhs_seed}: B^T·x=rhs residual={residual:.2e}"
                    );
                }
            }
        }
    }

    /// sentinel: MutableU の非ゼロ構造が faer u_factor() と一致し、diag_ptr が正しい。
    #[test]
    fn test_ftlu_u_representation_matches_faer() {
        for (n, seed) in [(5usize, 1u64), (10, 42), (20, 99)] {
            let a = gen_matrix(n, seed);
            let basis: Vec<usize> = (0..n).collect();
            let ft = FtLu::new(&a, &basis).unwrap();
            let u_ref = ft.lu0.u_factor();

            for j in 0..n {
                let mut faer_col: Vec<(usize, f64)> = u_ref
                    .row_idx_of_col(j)
                    .zip(u_ref.val_of_col(j).iter())
                    .map(|(r, &v)| (r, v))
                    .collect();
                faer_col.sort_by_key(|&(r, _)| r);

                let start = ft.u_mat.col_ptr[j];
                let end = ft.u_mat.col_ptr[j + 1];
                let mu_col: Vec<(usize, f64)> = (start..end)
                    .map(|k| (ft.u_mat.row_ind[k], ft.u_mat.values[k]))
                    .collect();

                assert_eq!(
                    faer_col.len(),
                    mu_col.len(),
                    "n={n} seed={seed} col={j}: nnz mismatch"
                );
                for (f, m) in faer_col.iter().zip(mu_col.iter()) {
                    assert_eq!(f.0, m.0, "n={n} seed={seed} col={j}: row mismatch");
                    assert!(
                        (f.1 - m.1).abs() < 1e-15,
                        "n={n} seed={seed} col={j}: val mismatch {:.2e} vs {:.2e}",
                        f.1,
                        m.1
                    );
                }

                let diag_idx = ft.u_mat.diag_ptr[j];
                assert_eq!(
                    ft.u_mat.row_ind[diag_idx], j,
                    "n={n} seed={seed}: diag_ptr[{j}] points to row {} not {j}",
                    ft.u_mat.row_ind[diag_idx]
                );
            }
        }
    }

    /// no-op sentinel: U 対角を 1.0 に固定すると B·x=rhs 残差が爆発する。
    /// backward_sub の対角除算コードパスが必須であることを実機確認。
    #[test]
    fn test_ftlu_no_op_pivot_identity_residual_explodes() {
        let dense = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 2.0],
            vec![0.0, 2.0, 5.0],
        ];
        let a = dense_to_csc(&dense, 3, 3);
        let ft = FtLu::new(&a, &[0, 1, 2]).unwrap();
        let rhs_orig = vec![5.0, 6.0, 9.0];

        // 正常: residual < 1e-10
        let mut rhs_ok = rhs_orig.clone();
        ft.ftran(&mut rhs_ok);
        let check_ok = a.mat_vec_mul(&rhs_ok).unwrap();
        let residual_ok: f64 = check_ok
            .iter()
            .zip(rhs_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            residual_ok < 1e-10,
            "correct ftran residual={residual_ok:.2e}"
        );

        // pivot=1 固定: residual が有意に大きい (no-op で fail する設計)
        let mut rhs_broken = rhs_orig.clone();
        ft.ftran_unit_pivot(&mut rhs_broken);
        let check_broken = a.mat_vec_mul(&rhs_broken).unwrap();
        let residual_broken: f64 = check_broken
            .iter()
            .zip(rhs_orig.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            residual_broken > 1e-3,
            "no-op pivot should explode residual, got={residual_broken:.2e}"
        );
    }

    /// 既存テストケースとの整合確認 (3x3 dense / sparse wrapper)。
    #[test]
    fn test_ftlu_small_matrices() {
        let dense3 = vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let a3 = dense_to_csc(&dense3, 3, 3);
        let ft3 = FtLu::new(&a3, &[0, 1, 2]).unwrap();
        let rhs = vec![3.0, 5.0, 3.0];

        // FTRAN
        let mut x = rhs.clone();
        ft3.ftran(&mut x);
        let check = a3.mat_vec_mul(&x).unwrap();
        assert_vec_near(&check, &rhs, 1e-10);

        // BTRAN
        let mut y = rhs.clone();
        ft3.btran(&mut y);
        let bt = a3.transpose();
        let check_bt = bt.mat_vec_mul(&y).unwrap();
        assert_vec_near(&check_bt, &rhs, 1e-10);

        // sparse wrapper
        let mut sv = SparseVec::from_dense(&rhs);
        ft3.ftran_sparse(&mut sv);
        let x_sp = sv.to_dense();
        let check_sp = a3.mat_vec_mul(&x_sp).unwrap();
        assert_vec_near(&check_sp, &rhs, 1e-10);
    }

    // =====================================================================
    // FT update (Phase 2b) sentinel
    // =====================================================================

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    fn lcg_unit(state: &mut u64) -> f64 {
        (lcg_next(state) >> 32) as f64 / u32::MAX as f64
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    /// 現基底 B (列 = `basis`) に対する `B · x`。
    fn basis_mat_vec(a: &CscMatrix, basis: &[usize], x: &[f64]) -> Vec<f64> {
        let m = basis.len();
        let mut y = vec![0.0f64; m];
        for (j, &col) in basis.iter().enumerate() {
            let (rows, vals) = a.get_column(col).unwrap();
            for (&r, &v) in rows.iter().zip(vals.iter()) {
                y[r] += v * x[j];
            }
        }
        y
    }

    /// 現基底 B に対する `Bᵀ · y`。
    fn basis_mat_t_vec(a: &CscMatrix, basis: &[usize], y: &[f64]) -> Vec<f64> {
        let m = basis.len();
        let mut out = vec![0.0f64; m];
        for (j, &col) in basis.iter().enumerate() {
            let (rows, vals) = a.get_column(col).unwrap();
            let mut acc = 0.0;
            for (&r, &v) in rows.iter().zip(vals.iter()) {
                acc += v * y[r];
            }
            out[j] = acc;
        }
        out
    }

    /// 参照 LU の U 対角から `Σ ln|U_ii|` (= ln|det B|)。
    fn ref_logdet(lu: &LuFactorization, m: usize) -> f64 {
        let u = lu.u_factor();
        let mut s = 0.0;
        for j in 0..m {
            let mut diag = 0.0;
            for (r, &v) in u.row_idx_of_col(j).zip(u.val_of_col(j).iter()) {
                if r == j {
                    diag = v;
                }
            }
            s += diag.abs().ln();
        }
        s
    }

    /// m×(m·nvar) の A を生成。列 `c` は支配行 `c % m` を持つ (強い対角 + 小 off-diagonal)。
    ///
    /// 列差替で常に「位置 d ⇔ 支配行 d」covering を保てるため (= 強対角行列)、
    /// 任意の基底が良条件に留まり、長い FT 更新連鎖を厳密検証できる。
    fn gen_update_problem(m: usize, nvar: usize, seed: u64) -> CscMatrix {
        let ncols = m * nvar;
        let mut st = seed.wrapping_mul(2654435761).wrapping_add(12345);
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..ncols {
            let d = j % m;
            rows.push(d);
            cols.push(j);
            vals.push(10.0 + lcg_unit(&mut st) * 4.0);
            let nz = 2 + (lcg_next(&mut st) as usize % 3);
            for _ in 0..nz {
                let i = lcg_next(&mut st) as usize % m;
                if i != d {
                    rows.push(i);
                    cols.push(j);
                    vals.push((lcg_unit(&mut st) - 0.5) * 0.6);
                }
            }
        }
        CscMatrix::from_triplets(&rows, &cols, &vals, m, ncols).unwrap()
    }

    /// 主 sentinel: 逐次ランダム列差替で FtLu が毎回フル再分解 (ground truth) と
    /// 1e-9 一致。U が上三角を保ち、値が実際に書き換わり、ln|det| 一致。≥5 seed。
    /// bump カバレッジ: t>c / partial-pivot swap / bsz>=3 が各1回以上発生を assert。
    #[test]
    fn test_ftlu_update_matches_full_refactor() {
        let sizes = [20usize, 35, 50];
        let seeds = [1u64, 7, 19, 42, 101];
        let n_updates = 50usize;

        let nvar = 4usize;
        // bump カバレッジカウンタ: 全 (m, seed) の総計で assert。
        let mut cnt_bump = 0usize;
        let mut cnt_pivot_swap = 0usize;
        let mut cnt_bsz3 = 0usize;

        for &m in &sizes {
            for &seed in &seeds {
                let a = gen_update_problem(m, nvar, seed);
                let mut basis: Vec<usize> = (0..m).collect();
                let mut ft = FtLu::new(&a, &basis).unwrap();
                let mut rng = seed.wrapping_mul(99991).wrapping_add(7);

                let mut applied = 0usize;
                let mut attempts = 0usize;

                while applied < n_updates && attempts < n_updates * 8 {
                    attempts += 1;
                    // 支配行 r を保つ列差替: 位置 r に支配行 r の別 variant を入れる。
                    let r = lcg_next(&mut rng) as usize % m;
                    let v = 1 + (lcg_next(&mut rng) as usize % (nvar - 1));
                    let q = v * m + r;
                    if basis[r] == q {
                        continue;
                    }
                    let mut new_basis = basis.clone();
                    new_basis[r] = q;

                    // 参照フル再分解で非特異性を gate。
                    let ref_lu = match LuFactorization::factorize_timed(&a, &new_basis, None) {
                        Ok(lu) => lu,
                        Err(_) => continue,
                    };

                    // bump カバレッジ計測 (update 前に spike と U を調べる)。
                    let (s_dbg, c_dbg, t_dbg) = ft.debug_spike(q, r);
                    if t_dbg > c_dbg {
                        cnt_bump += 1;
                        let bsz_dbg = t_dbg - c_dbg + 1;
                        if bsz_dbg >= 3 {
                            cnt_bsz3 += 1;
                        }
                        // partial-pivot swap が起きたか: 巡回後の bump 先頭列は旧 U 列 c+1 の
                        // 行 c..=t スライス。bump[1,0] > bump[0,0] ↔ 旧 U[c+1,c+1] > 旧 U[c,c+1]。
                        let col_c1 = c_dbg + 1;
                        let u_diag = ft
                            .u_mat
                            .col_ptr[col_c1]..ft.u_mat.col_ptr[col_c1 + 1];
                        let mut bump00 = 0.0f64;
                        let mut bump10 = 0.0f64;
                        for k in u_diag {
                            let row = ft.u_mat.row_ind[k];
                            let val = ft.u_mat.values[k];
                            if row == c_dbg {
                                bump00 = val;
                            } else if row == c_dbg + 1 {
                                bump10 = val;
                            }
                        }
                        if bump10.abs() > bump00.abs() {
                            cnt_pivot_swap += 1;
                        }
                        let _ = s_dbg; // used via debug_spike; silence unused warning
                    }

                    let u_before = ft.u_mat.values.clone();
                    ft.update(q, r)
                        .expect("FT update must succeed for nonsingular basis");
                    basis = new_basis;
                    applied += 1;

                    // U が実際に書き換わった (PFI 逃げの構造的禁止)。
                    assert_ne!(
                        ft.u_mat.values, u_before,
                        "m={m} seed={seed} upd={applied}: U values unchanged (PFI?)"
                    );
                    // U が上三角 (subdiagonal なし)。
                    for j in 0..m {
                        for k in ft.u_mat.col_ptr[j]..ft.u_mat.col_ptr[j + 1] {
                            assert!(
                                ft.u_mat.row_ind[k] <= j,
                                "m={m} seed={seed} upd={applied}: U col {j} subdiagonal row {}",
                                ft.u_mat.row_ind[k]
                            );
                        }
                    }
                    // ln|det| (pivot) 一致。
                    let ld_ft: f64 = (0..m)
                        .map(|j| ft.u_mat.values[ft.u_mat.diag_ptr[j]].abs().ln())
                        .sum();
                    let ld_ref = ref_logdet(&ref_lu, m);
                    assert!(
                        (ld_ft - ld_ref).abs() < 1e-6,
                        "m={m} seed={seed} upd={applied}: ln|det| FT={ld_ft:.8} ref={ld_ref:.8}"
                    );

                    // solve 一致 (参照 ftran/btran と相対 1e-9) + 残差 (相対 1e-9)。
                    let rhs_scale = 1.0
                        + gen_rhs(m, seed.wrapping_add(applied as u64 * 131))
                            .iter()
                            .fold(0.0f64, |m, &v| m.max(v.abs()));
                    for rs in 0..3u64 {
                        let rhs = gen_rhs(m, seed.wrapping_add(applied as u64 * 131 + rs));

                        let mut x_ft = rhs.clone();
                        ft.ftran(&mut x_ft);
                        let mut x_ref = rhs.clone();
                        solve_ftran(&ref_lu, &mut x_ref);
                        let xscale = 1.0 + x_ref.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
                        let rel = max_abs_diff(&x_ft, &x_ref) / xscale;
                        assert!(
                            rel < 1e-9,
                            "m={m} seed={seed} upd={applied}: ftran rel diff={rel:.2e}"
                        );
                        let resid = max_abs_diff(&basis_mat_vec(&a, &basis, &x_ft), &rhs) / rhs_scale;
                        assert!(
                            resid < 1e-9,
                            "m={m} seed={seed} upd={applied}: B·x rel residual={resid:.2e}"
                        );

                        let mut y_ft = rhs.clone();
                        ft.btran(&mut y_ft);
                        let mut y_ref = rhs.clone();
                        solve_btran(&ref_lu, &mut y_ref);
                        let yscale = 1.0 + y_ref.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
                        let relb = max_abs_diff(&y_ft, &y_ref) / yscale;
                        assert!(
                            relb < 1e-9,
                            "m={m} seed={seed} upd={applied}: btran rel diff={relb:.2e}"
                        );
                        let residt =
                            max_abs_diff(&basis_mat_t_vec(&a, &basis, &y_ft), &rhs) / rhs_scale;
                        assert!(
                            residt < 1e-9,
                            "m={m} seed={seed} upd={applied}: Bᵀ·y rel residual={residt:.2e}"
                        );
                    }

                    // 本番同様: needs_refactor が立てば再分解 (蓄積誤差をリセット)。
                    if ft.needs_refactor() {
                        ft = FtLu::new(&a, &basis).unwrap();
                    }
                }
                assert!(
                    applied >= 30,
                    "m={m} seed={seed}: only {applied} updates applied (need ≥30)"
                );
            }
        }
        // bump カバレッジ (全 (m, seed) 総計): t>c / bsz>=3 / partial-pivot swap が各1回以上。
        assert!(cnt_bump >= 1, "no bump (t>c) path in any config");
        assert!(cnt_bsz3 >= 1, "no bsz>=3 path in any config");
        assert!(cnt_pivot_swap >= 1, "no partial-pivot swap in any config");
    }

    /// no-op proof: 巡回シフト+Hessenberg 消去を省く (素朴な列置換) と U に subdiagonal が
    /// 残り、solve 残差が爆発する。真の FT (写し替え) のみ残差 < 1e-9。
    #[test]
    fn test_ftlu_update_no_shift_residual_explodes() {
        let m = 24usize;
        let nvar = 4usize;
        let a = gen_update_problem(m, nvar, 5);
        let basis: Vec<usize> = (0..m).collect();
        let ft0 = FtLu::new(&a, &basis).unwrap();

        // bump (t>c) かつ |s[c]| が小さくない (q,r) を探索。
        let mut chosen: Option<(usize, usize)> = None;
        let mut rng = 333u64;
        for _ in 0..400 {
            let r = lcg_next(&mut rng) as usize % m;
            let v = 1 + (lcg_next(&mut rng) as usize % (nvar - 1));
            let q = v * m + r;
            if basis[r] == q {
                continue;
            }
            let mut nb = basis.clone();
            nb[r] = q;
            if LuFactorization::factorize_timed(&a, &nb, None).is_err() {
                continue;
            }
            let (s, c, t) = ft0.debug_spike(q, r);
            if t > c && s[c].abs() > 1e-2 {
                chosen = Some((q, r));
                break;
            }
        }
        let (q, r) = chosen.expect("no bump-inducing (q,r) found");
        let mut new_basis = basis.clone();
        new_basis[r] = q;
        let rhs = gen_rhs(m, 7);

        // 真の FT: 残差 < 1e-9。
        let mut ft_real = ft0.clone();
        ft_real.update(q, r).unwrap();
        let mut x_real = rhs.clone();
        ft_real.ftran(&mut x_real);
        let resid_real = max_abs_diff(&basis_mat_vec(&a, &new_basis, &x_real), &rhs);
        assert!(resid_real < 1e-9, "real FT residual={resid_real:.2e}");

        // 素朴置換 (no-op): 残差爆発。
        let mut ft_naive = ft0.clone();
        ft_naive.update_naive_no_ft(q, r);
        let mut x_naive = rhs.clone();
        ft_naive.ftran(&mut x_naive);
        let resid_naive = max_abs_diff(&basis_mat_vec(&a, &new_basis, &x_naive), &rhs);
        assert!(
            resid_naive > 1e-3,
            "no-op (shift/elim 省略) must explode residual, got={resid_naive:.2e}"
        );
    }

    /// 小 pivot で needs_refactor が立つ (t==c 経路, 決定論)。
    #[test]
    fn test_ftlu_update_needs_refactor_small_pivot() {
        // B = I_2, 列1 を a_q=[1, ε] で置換 → B'=[[1,1],[0,ε]], 最終 pivot=ε。
        let eps = 1e-3;
        let a = CscMatrix::from_triplets(
            &[0usize, 1, 0, 1],
            &[0usize, 1, 2, 2],
            &[1.0f64, 1.0, 1.0, eps],
            2,
            3,
        )
        .unwrap();
        let basis = vec![0usize, 1];
        let mut ft = FtLu::new(&a, &basis).unwrap();
        assert!(!ft.needs_refactor());

        ft.update(2, 1).unwrap();
        assert!(
            ft.needs_refactor(),
            "small pivot ε={eps} must set needs_refactor"
        );
        // solve は依然正しい。
        let new_basis = vec![0usize, 2];
        let rhs = vec![2.0, 3.0];
        let mut x = rhs.clone();
        ft.ftran(&mut x);
        let resid = max_abs_diff(&basis_mat_vec(&a, &new_basis, &x), &rhs);
        assert!(resid < 1e-9, "ftran residual={resid:.2e}");
    }

    /// ill-conditioned sentinel: match arm + recovery 機構のテスト。
    ///
    /// 不変条件: 各ケースで「FT solve がフル再分解と tol 一致 (match arm)」または
    /// 「needs_refactor が立ち、FtLu::new 後に一致が回復 (refactor arm)」が成立。
    ///
    /// このテストでは partial pivoting が小 pivot を救うため全6ケースが match arm に入り、
    /// needs_refactor が立つ決定論的なケースは含まない。interior bump pivot の
    /// 決定論的発火 (bsz≥3, line 465) は `test_ftlu_interior_bump_pivot_fires_needs_refactor`
    /// が担い、そちらが P2 修正の load-bearing 証明となる。
    #[test]
    fn test_ftlu_update_ill_conditioned_match_or_refactor() {
        // ill-cond 基底生成: 対角が強いが一部列が互いに近い (大きな条件数)。
        // B_base = diag(10, 10, .., 10) + off-diag 0.1 * ones (rank-1扰動)。
        // entering_col は B の1列を微小スカラー倍 + 別行への成分を追加して
        // bump 中間 pivot が snorm_inf の PIVOT_STABILITY_THRESHOLD 付近になるよう調整。
        let m = 6usize;
        // A の列: 0..m は単位ベクトル×10 (初期基底 = 対角)、m..2m は ill-cond 列。
        // ill-cond 列 j (j in m..2m): 行 j%m に 10.0、行 (j+1)%m に 1e-3 (小 pivot を誘発)。
        let mut rows_a = Vec::new();
        let mut cols_a = Vec::new();
        let mut vals_a: Vec<f64> = Vec::new();
        // 初期基底列 (単位×10)。
        for j in 0..m {
            rows_a.push(j);
            cols_a.push(j);
            vals_a.push(10.0);
        }
        // ill-cond 列: 支配行 j%m に 10.0、次行に ε=5e-3 (小中間 pivot を誘発)。
        let eps = 5e-3f64;
        for j in 0..m {
            rows_a.push(j);
            cols_a.push(m + j);
            vals_a.push(10.0);
            // 次行に小 off-diagonal (bump を作り中間 pivot が小さくなる)。
            let next = (j + 1) % m;
            rows_a.push(next);
            cols_a.push(m + j);
            vals_a.push(eps);
        }
        let a = CscMatrix::from_triplets(&rows_a, &cols_a, &vals_a, m, 2 * m).unwrap();

        let init_basis: Vec<usize> = (0..m).collect();
        let mut match_cnt = 0usize;
        let mut refactor_cnt = 0usize;

        // 各位置 r で ill-cond 列 m+r を入基。
        for r in 0..m {
            let q = m + r;
            let mut ft = FtLu::new(&a, &init_basis).unwrap();
            let mut basis_after = init_basis.clone();
            basis_after[r] = q;

            let ref_lu = match LuFactorization::factorize_timed(&a, &basis_after, None) {
                Ok(lu) => lu,
                Err(_) => continue,
            };
            if ft.update(q, r).is_err() {
                continue;
            }

            let rhs = gen_rhs(m, 42 + r as u64);
            let mut x_ft = rhs.clone();
            ft.ftran(&mut x_ft);
            let mut x_ref = rhs.clone();
            solve_ftran(&ref_lu, &mut x_ref);
            let xscale = 1.0 + x_ref.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
            let rel = max_abs_diff(&x_ft, &x_ref) / xscale;

            if rel < 1e-6 {
                // FT solve が参照と一致: OK。
                match_cnt += 1;
            } else {
                // 不一致ならば needs_refactor が必ず立っていること。
                assert!(
                    ft.needs_refactor(),
                    "r={r}: FT solve inaccurate (rel={rel:.2e}) but needs_refactor not set — P2 fix missing?"
                );
                // 再分解後は一致が回復する。
                let ft2 = FtLu::new(&a, &basis_after).unwrap();
                let mut x_re = rhs.clone();
                ft2.ftran(&mut x_re);
                let rel2 = max_abs_diff(&x_re, &x_ref) / xscale;
                assert!(
                    rel2 < 1e-9,
                    "r={r}: after FtLu::new refactor rel={rel2:.2e} still inaccurate"
                );
                refactor_cnt += 1;
            }
        }
        // 少なくとも1ケースは検証できた。
        assert!(
            match_cnt + refactor_cnt >= 1,
            "no ill-cond case was exercised"
        );
    }

    /// 退化基底 (入基列が既存基底列の複製) で SingularBasis を返し、state 不変。
    #[test]
    fn test_ftlu_update_singular_detection() {
        // A = [e0 e1 e2 | e1]; 列0 を a_q=e1 で置換 → 基底特異。
        let a = CscMatrix::from_triplets(
            &[0usize, 1, 2, 1],
            &[0usize, 1, 2, 3],
            &[1.0f64, 1.0, 1.0, 1.0],
            3,
            4,
        )
        .unwrap();
        let basis = vec![0usize, 1, 2];
        let mut ft = FtLu::new(&a, &basis).unwrap();
        let u_before = ft.u_mat.values.clone();

        let res = ft.update(3, 0);
        assert!(
            matches!(res, Err(SolverError::SingularBasis { .. })),
            "duplicate column must yield SingularBasis, got {res:?}"
        );
        // Err 経路で u_mat は不変 (needs_refactor のみ変化)。
        assert_eq!(ft.u_mat.values, u_before, "u_mat must be untouched on Err");
    }

    /// load-bearing sentinel: interior bump pivot stability check (ft.rs:464-467) を
    /// bsz=5 (≥3) の決定論的構成で発火させる。
    ///
    /// 構成:
    /// - 初期基底 (cols 0..4) の LU 因子が working 位置 (1,1) に U[1,1]=5e-6 を持つよう設計。
    ///   COLAMD は col 3 (δ のみ) を working col 1 に配置するためこの値が得られる。
    /// - 入基列 col 5 (leaving_row=4) で spike が working 行 0..4 全体に届き bsz=5 を強制。
    /// - bump col 0 = U col 1: bump[0][0]=0, bump[1][0]=5e-6 → partial pivot swap →
    ///   pivot=5e-6 < PIVOT_STABILITY_THRESHOLD(0.01) × snorm_inf(≈0.98) → ft.rs:465 が発火。
    /// - final_pivot ≈ 0.3 (large) → ft.rs:541 は不発 → needs_refactor の原因が ft.rs:465 のみ。
    ///
    /// load-bearing 証明: ft.rs:464-467 を削除すると needs_refactor が立たず、
    /// `assert!(ft.needs_refactor())` が FAIL する (実機確認済)。
    #[test]
    fn test_ftlu_interior_bump_pivot_fires_needs_refactor() {
        let n = 5usize;
        let delta = 5e-6f64;

        // 初期基底列 (cols 0..4): COLAMD が col 3 を working col 1 (U[1,1]=δ) に配置する構成。
        // col 3 は rows 2,3 のみ (小値 δ の 2×2 小行列) → COLAMD がスパースな列を先頭近くに置く。
        let mut rows_a: Vec<usize> = Vec::new();
        let mut cols_a: Vec<usize> = Vec::new();
        let mut vals_a: Vec<f64> = Vec::new();

        // col 0: row 0 = 1.0
        rows_a.push(0); cols_a.push(0); vals_a.push(1.0);
        // col 1: row 0 = 0.5, row 1 = 1.0
        rows_a.push(0); cols_a.push(1); vals_a.push(0.5);
        rows_a.push(1); cols_a.push(1); vals_a.push(1.0);
        // col 2: row 0 = 0.3, row 1 = 0.2, row 2 = 1.0
        rows_a.push(0); cols_a.push(2); vals_a.push(0.3);
        rows_a.push(1); cols_a.push(2); vals_a.push(0.2);
        rows_a.push(2); cols_a.push(2); vals_a.push(1.0);
        // col 3: row 2 = δ, row 3 = δ  (小値 → U[1,1]=δ after COLAMD reorder)
        rows_a.push(2); cols_a.push(3); vals_a.push(delta);
        rows_a.push(3); cols_a.push(3); vals_a.push(delta);
        // col 4: row 4 = 1.0
        rows_a.push(4); cols_a.push(4); vals_a.push(1.0);
        // entering col 5: rows 1,2,3,4 → spike 全 5 working 行に届き bsz=5 を強制
        rows_a.push(1); cols_a.push(5); vals_a.push(1.0);
        rows_a.push(2); cols_a.push(5); vals_a.push(0.5);
        rows_a.push(3); cols_a.push(5); vals_a.push(0.4);
        rows_a.push(4); cols_a.push(5); vals_a.push(0.3);

        let a = CscMatrix::from_triplets(&rows_a, &cols_a, &vals_a, n, 6).unwrap();
        let init_basis: Vec<usize> = (0..n).collect();
        let mut ft = FtLu::new(&a, &init_basis).unwrap();
        assert!(!ft.needs_refactor());

        // entering=5, leaving=4: c=0, t=4, bsz=5, snorm_inf≈0.98
        // bump[1][0] = U[1,1] = 5e-6 > bump[0][0] = U[0,1] = 0 → swap →
        // pivot = 5e-6 < 0.01 × 0.98 → ft.rs:465 発火。
        ft.update(5, 4).expect("update must succeed (pivot=δ > DROP_TOL=1e-15)");

        // ft.rs:465 が立てた needs_refactor が true である必要がある。
        // ft.rs:464-467 を削除すると final_pivot=0.3 (ft.rs:541 不発) のため
        // needs_refactor=false となり、ここで FAIL する (load-bearing 証明)。
        assert!(
            ft.needs_refactor(),
            "interior bump pivot (δ={delta:.2e}) must fire ft.rs:465 and set needs_refactor; \
             removing ft.rs:464-467 causes this assert to fail"
        );

        // refactor (FtLu::new) 後に solve 精度が回復する。
        let new_basis = vec![0usize, 1, 2, 3, 5];
        let ref_lu = LuFactorization::factorize_timed(&a, &new_basis, None)
            .expect("new basis [0,1,2,3,5] must be non-singular");
        let ft2 = FtLu::new(&a, &new_basis).unwrap();
        let rhs = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut x_ft2 = rhs.clone();
        ft2.ftran(&mut x_ft2);
        let mut x_ref = rhs.clone();
        solve_ftran(&ref_lu, &mut x_ref);
        let xscale = 1.0 + x_ref.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let rel = max_abs_diff(&x_ft2, &x_ref) / xscale;
        assert!(
            rel < 1e-9,
            "after FtLu::new refactor, ftran rel diff={rel:.2e}"
        );
    }
}
