//! `step8_parallel_row` の行単位 stop チェックに対する sentinel 回帰張りのための
//! テスト専用モジュール。本番コードは一切含まない。
//!
//! なぜ本番ファイルから分離したテストモジュールにするか: Sentinel Red/Green ゲート
//! (`scripts/sentinel_red_green.py`) の base 検証は「本番コードを含まないテスト
//! 定義ファイル」(`tests/` 直下または `tests.rs`) だけを base worktree へ overlay
//! する契約を持つ。`/// sentinel:` を張ったテストが本番コードと同一 .rs に混在すると
//! HEAD 版の overlay が本番修正まで base へ運び red 検証が無意味になるため、overlay
//! 不可 (UNVERIFIABLE) としてゲートが fail する。この張りの本体は `tests.rs` に置き、
//! 本ファイルは宣言のみで受ける。テスト専用であることは `../steps_parallel.rs` 側の
//! `#[cfg(test)] mod stop_sentinels;` 宣言によって保証される (移設前と同一の
//! gating 条件)。

mod tests;
