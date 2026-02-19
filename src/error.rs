use crate::io::mps::MpsError;

/// ソルバー全体の統一エラー型
///
/// 入力検証・数値計算・MPSパースなど、ソルバー操作中に
/// 発生しうるエラーを統一的に表現する。
///
/// 注意: Infeasible/Unbounded/MaxIterations は数学的結果であり、
/// エラーではないため [`SolveStatus`](crate::problem::SolveStatus) で表現する。
#[derive(Debug)]
pub enum SolverError {
    /// MPSファイルのパースエラー
    Mps(MpsError),

    /// 次元不一致（配列長・行列サイズの不整合）
    ///
    /// 例: `c.len() != a.ncols`, トリプレット配列の長さ不一致
    DimensionMismatch {
        /// どのフィールド/配列が不一致か（例: "c", "b", "triplet_arrays"）
        field: &'static str,
        /// 期待されるサイズ
        expected: usize,
        /// 実際のサイズ
        got: usize,
    },

    /// インデックスが有効範囲外
    ///
    /// 例: 行列の列インデックス、基底列インデックス
    IndexOutOfBounds {
        /// どのインデックスか（例: "column", "row", "basis_column"）
        context: &'static str,
        /// 範囲外のインデックス値
        index: usize,
        /// 上限値（0..bound が有効範囲）
        bound: usize,
    },

    /// 基底行列が特異（LU分解で数値的に特異なピボットを検出）
    SingularBasis {
        /// 特異性が検出されたガウス消去のステップ番号
        step: usize,
    },

    /// 空の入力が渡された
    EmptyInput {
        /// どの入力が空か（例: "basis"）
        context: &'static str,
    },
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::Mps(e) => write!(f, "{}", e),
            SolverError::DimensionMismatch { field, expected, got } => {
                write!(f, "Dimension mismatch: {} expected {} but got {}", field, expected, got)
            }
            SolverError::IndexOutOfBounds { context, index, bound } => {
                write!(f, "{} index {} out of bounds (size={})", context, index, bound)
            }
            SolverError::SingularBasis { step } => {
                write!(f, "Singular matrix detected at step {}", step)
            }
            SolverError::EmptyInput { context } => {
                write!(f, "Empty input: {}", context)
            }
        }
    }
}

impl std::error::Error for SolverError {}

impl From<MpsError> for SolverError {
    fn from(e: MpsError) -> Self {
        SolverError::Mps(e)
    }
}
