---
name: release
description: リリース作業 (version bump / CHANGELOG / cargo publish / リリース前検証) を行うときに必ず読む。バンプ箇所・publish 手順・dev-dep の罠・CHANGELOG の粒度を定義する。
---

# リリース手順 (v0.5.0 で検証済み)

## version bump
1. ルート `Cargo.toml` の `[workspace.package] version`
2. `[workspace.dependencies]` の otspot-core / otspot-io / otspot-model の version 制約 (3行)
3. `otspot-core/Cargo.toml` の otspot-model dev-dep が path のみ (version フィールドなし) のままであることを確認する — bump しない
4. `cargo update --workspace` で Cargo.lock を同期

**罠**: workspace メンバー間の dev-dep に version を書くと、`cargo package --workspace` が未 publish の新バージョンを crates.io に解決しに行き失敗する。メンバー間 dev-dep は path のみ (version フィールドなし) に保つ。

## CHANGELOG
- リリース間の net 差分だけを書く。未リリース内部の revert/churn 履歴や内部定数名は書かない。
- 粒度は過去の [0.2.x]/[0.4.0] セクション相当: fix をグループ化、サブヘッダなし、簡潔に。書いた後、過去セクションと行数感を見比べる。

## リリース前検証
- full-suite + heavy 相当 (`cargo nextest run --release --test-threads 3 --profile heavy --run-ignored all` 相当) をローカルで通す。
- clippy / `cargo check --all-targets` / `cargo publish --workspace --dry-run` (token 不要)。
- README の性能表が最新の実測ベンチと一致しているか確認 (bench skill)。

## publish
- `cargo publish --workspace` をルートの clean tree で実行 — 依存順 (core → io/model → facade) に自動で publish する。otspot-dev は publish=false で除外される。
- `cargo login` はユーザのトークン。認証はユーザが実施し、こちらはトークンを扱わない。publish は不可逆 (yank のみ) なので実行前にユーザへ最終確認する。

## CI
- ci.yml / audit.yml は push: branches ["**"] + PR で発火。test-heavy.yml は push + workflow_dispatch のみ (PR なし) で non-gating。
- dependabot は dtolnay/rust-toolchain を ignore (バージョン=Rust版のため自動バンプで全 CI が壊れる)。
