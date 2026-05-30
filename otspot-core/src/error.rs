/// MPSファイルのパース中に発生するエラー
#[non_exhaustive]
#[derive(Debug)]
pub enum MpsError {
    /// ファイル読み込み時のI/Oエラー
    IoError(std::io::Error),
    /// 指定行のパースエラー（行番号とメッセージを含む）
    ParseError { line: usize, message: String },
    /// 必須セクションが欠落している
    MissingSection(String),
    /// 同じセクションが複数回出現した
    DuplicateSection(String),
    /// 無効な行タイプ文字が指定された
    InvalidRowType(char),
    /// 無効な上下限タイプ文字列が指定された
    InvalidBoundType(String),
    /// 未定義の行名または列名が参照された
    UndefinedReference { kind: String, name: String },
    /// INTORG マーカーが対応する INTEND で閉じられないまま COLUMNS を抜けた。
    /// 残り全列が無警告で整数化されるのを防ぐため明示エラーとする。
    UnclosedIntegerMarker,
}

impl std::fmt::Display for MpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MpsError::IoError(e) => write!(f, "I/O error: {}", e),
            MpsError::ParseError { line, message } => {
                write!(f, "Parse error at line {}: {}", line, message)
            }
            MpsError::MissingSection(s) => write!(f, "Missing required section: {}", s),
            MpsError::DuplicateSection(s) => write!(f, "Duplicate section: {}", s),
            MpsError::InvalidRowType(c) => write!(f, "Invalid row type: {}", c),
            MpsError::InvalidBoundType(s) => write!(f, "Invalid bound type: {}", s),
            MpsError::UndefinedReference { kind, name } => {
                write!(f, "Undefined {} reference: {}", kind, name)
            }
            MpsError::UnclosedIntegerMarker => {
                write!(
                    f,
                    "INTORG marker not closed by a matching INTEND in COLUMNS"
                )
            }
        }
    }
}

impl std::error::Error for MpsError {}

impl From<std::io::Error> for MpsError {
    fn from(err: std::io::Error) -> Self {
        MpsError::IoError(err)
    }
}

/// ソルバー全体の統一エラー型
///
/// 入力検証・数値計算など、ソルバー操作中に発生しうるエラーを統一的に表現する。
///
/// 注意: Infeasible/Unbounded/MaxIterations は数学的結果であり、
/// エラーではないため [`SolveStatus`](crate::problem::SolveStatus) で表現する。
#[non_exhaustive]
#[derive(Debug)]
pub enum SolverError {
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

    /// Deadline を超過した（タイムアウト）
    DeadlineExceeded,

    /// 非有限係数（NaN または ±∞）が入力された
    ///
    /// 例: `c[i]` が NaN、`b[j]` が Inf、行列要素が NaN
    NonFiniteCoefficient {
        /// どのフィールドか（"c", "b", "A" など）
        field: &'static str,
        /// 最初に非有限値が検出されたインデックス
        index: usize,
    },

    /// 変数境界が無効（NaN または lb > ub）
    InvalidBounds {
        /// 無効な境界を持つ変数のインデックス
        index: usize,
        /// 下限値
        lb: f64,
        /// 上限値
        ub: f64,
    },
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::DimensionMismatch {
                field,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Dimension mismatch: {} expected {} but got {}",
                    field, expected, got
                )
            }
            SolverError::IndexOutOfBounds {
                context,
                index,
                bound,
            } => {
                write!(
                    f,
                    "{} index {} out of bounds (size={})",
                    context, index, bound
                )
            }
            SolverError::SingularBasis { step } => {
                write!(f, "Singular matrix detected at step {}", step)
            }
            SolverError::EmptyInput { context } => {
                write!(f, "Empty input: {}", context)
            }
            SolverError::DeadlineExceeded => {
                write!(f, "Deadline exceeded during computation")
            }
            SolverError::NonFiniteCoefficient { field, index } => {
                write!(f, "Non-finite coefficient in {}: index {}", field, index)
            }
            SolverError::InvalidBounds { index, lb, ub } => {
                write!(
                    f,
                    "Invalid bounds at index {}: lb={} > ub={} or NaN",
                    index, lb, ub
                )
            }
        }
    }
}

impl std::error::Error for SolverError {}
