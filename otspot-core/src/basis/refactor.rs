//! 基底行列の定期的な再因子分解サポート
//!
//! eta ファイルが蓄積して数値精度が低下した際に、
//! 基底行列を最初から LU 分解し直す機能を提供する。

use super::lu::LuFactorization;
use crate::error::SolverError;
use crate::sparse::CscMatrix;
use std::time::Instant;

/// deadline 付き基底行列再因子分解
///
/// # timeout audit fix
/// Simplex 反復中の LU 再因子分解 (refactor_if_needed) は O(m²〜m³) になりうる。
/// deadline を factorize_timed に渡すことで大規模問題でのハングを防止する。
///
/// # 引数
/// - `a`: 全制約行列（CSC 形式）
/// - `basis`: 現在の基底変数インデックス列
/// - `deadline`: 打ち切り時刻。None は無制限
///
/// # エラー
/// LU 分解が失敗（特異）または deadline 超過した場合は `Err` を返す
pub(crate) fn refactor_timed(
    a: &CscMatrix,
    basis: &[usize],
    deadline: Option<Instant>,
) -> Result<LuFactorization, SolverError> {
    LuFactorization::factorize_timed(a, basis, deadline)
}
