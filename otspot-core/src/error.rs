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

/// Solver-wide error type, physically owned by the numerical foundation.
///
/// Re-exporting the exact type preserves the legacy `otspot_core::SolverError`
/// API while sparse and factorization code move to `otspot-num`.
pub use otspot_num::SolverError;
