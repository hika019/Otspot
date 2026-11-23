//! `equality_constraint_qr` の per-item stop チェックに対する sentinel 回帰張りの
//! ためのテスト専用モジュール。本番コードは一切含まない。
//!
//! なぜ本番ファイルから分離したテストモジュールにするか: Sentinel Red/Green ゲート
//! (`scripts/sentinel_red_green.py`) の base 検証は「本番コードを含まないテスト
//! 定義ファイル」(`tests/` 直下または `tests.rs`) だけを base worktree へ overlay
//! する契約を持つ。`/// sentinel:` を張ったテストが本番コードと同一 .rs に混在すると
//! HEAD 版の overlay が本番修正まで base へ運び red 検証が無意味になるため、overlay
//! 不可 (UNVERIFIABLE) としてゲートが fail する。この張りの本体は `tests.rs` に置き、
//! 本ファイルは宣言のみで受ける。テスト専用であることは `../qp_phase2.rs` 側の
//! `#[cfg(test)] mod qr_stop_sentinels;` 宣言によって保証される (移設前と同一の
//! gating 条件)。

mod tests;
