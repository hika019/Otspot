//! 基底行列の定期的な再因子分解サポート
//!
//! eta ファイルが蓄積して数値精度が低下した際に、
//! 基底行列を最初から LU 分解し直す機能を提供する。

use crate::sparse::CscMatrix;
use super::lu::LuFactorization;

/// 基底行列をゼロから再因子分解する
///
/// LU 分解の薄いラッパー。eta ファイルをリセットして
/// 数値安定性を回復させるために呼び出される。
///
/// # 引数
/// - `a`: 全制約行列（CSC 形式）
/// - `basis`: 現在の基底変数インデックス列
///
/// # エラー
/// LU 分解が失敗した場合（特異行列等）は `Err` を返す
pub(crate) fn refactor(a: &CscMatrix, basis: &[usize]) -> Result<LuFactorization, String> {
    LuFactorization::factorize(a, basis)
}
