# AGENTS.md

エージェント向けメモ。運用規約の一次ソースは `CLAUDE.md` (編集にはユーザの明示許可が必要)。ベンチ実行・バグ調査・リリース・セッション復旧の着手前に必ず `.claude/skills/{bench,bug-frontier,release,session-recovery}/SKILL.md` を読む。

## 構成

数理最適化ソルバー (LP: revised simplex / QP: IPM / MILP: B&B / 非凸 QP: spatial B&B / SOCP)。Rust workspace、toolchain は `rust-toolchain.toml` で 1.95.0 に固定。

| クレート | 役割 |
|---|---|
| `.` (otspot) | facade — core / io / model を re-export |
| `otspot-num` | 最下層: sparse (CSC/SparseVec)、KKT、Ruiz、fixpoint、timeout/cancellation。workspace 内 crate への依存は CI で禁止 |
| `otspot-core` | 求解エンジン (simplex / IPM / B&B / presolve / conic) と低レベル API。`crate::sparse` / `crate::linalg` / `crate::error::SolverError` 参照は禁止 (旧パス。`otspot_num` を使う) |
| `otspot-io` | MPS / QPS / QPLIB / CBF パーサ |
| `otspot-model` | モデリング API (`Model`, `constraint!`) |
| `otspot-dev` | dev bin (qps_benchmark, qp_runner, bench_qplib, lp_solve_diag, milp_solve)。publish されない |
| `otspot-py` | PyO3/maturin bindings。**default-members 外** (cdylib が全 `cargo build/test` を汚染するため)。Cargo lib 名は `otspot_py`、Python import 名は `otspot` |

- `CscMatrix` / `SparseVec` の storage field は `pub(crate)` — 直接 dot-access はコンパイルエラー。アクセサ (`col_ptr()` 等) を使う。
- 層構造は CI (`Architecture` workflow / `python3 scripts/check_architecture.py`) が依存方向・型の実装所有権・facade の薄さを強制。

## テスト・検証コマンド

nextest 必須 (`cargo test` は suite 実行には使わない。doctest のみ)。設定は `.config/nextest.toml` (ローカル 6 スレッド / CI 3):

```bash
cargo nextest run --release --features parallel --test-threads 6  # full suite。data/ 必須 (欠落で panic する設計。silent skip に直さない)
cargo nextest run --release --profile lib-only                    # lib+bin のみ。data/ 不要。データ未取得環境はこちら
cargo nextest run --release --profile heavy --run-ignored all     # #[ignore] 込み (tier-2 sentinel)。--run-ignored は CLI フラグ必須 (profile キーではない)
cargo nextest run -p otspot-num                                   # debug で実走 (debug_assert! sentinel は release では一度も走らない)。CI が実施
cargo test --doc --release                                        # doctest (nextest 非対応)
cargo nextest run --release -E 'test(テスト名)'                   # 1 test
```

マージ前ゲートを一発で: `bash scripts/pre-merge-audit.sh`。

CI が red になる主要 gate (ローカルでも意識):

- `cargo clippy --workspace --all-targets --all-features -- -D warnings` / `cargo fmt --all -- --check`
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`
- TODO/FIXME/XXX/HACK/todo!() は production src に 1 件でも禁止 (`scripts/lib/check_memo_grep.sh`)。TODO は未完了 marker として CI fail
- コメント gate: `scripts/check_comment_block_size.sh` / `scripts/check_comment_ratio.sh` — OSS 水準の最小コメントを強制
- file/function size: production file 2500 行 trigger (`scripts/check_file_size.sh`)、220 行超の新規関数は baseline 登録のみ (`scripts/check_function_size.py`)
- `cargo machete` (unused deps)
- test inventory: テストの追加・削除・ignore 化をしたら `python3 scripts/lib/check_new_ignore_attrs.py --update` で `tests/test_inventory.txt` を再生成してコミット (手編集禁止)

## data/ (テスト・ベンチデータ)

gitignored。`data/baseline_objectives/*.csv` のみ git 追跡 (期待値参照)。

```bash
bash scripts/download_all_bench_data.sh             # 全取得
bash scripts/download_all_bench_data.sh --ci-subset # CI 相当 (~1.2 GiB)
bash scripts/download_all_bench_data.sh --check     # 現状確認
```

- Netlib LP は事前に `bash scripts/ensure_emps.sh` で emps decoder を build する必要あり
- 合成 QP データ生成は `numpy scipy cvxpy clarabel` が必要 (`pip install`)

## Python bindings (otspot-py)

- Rust 側テストは `cargo test -p otspot-py --no-default-features` — default feature `extension-module` は libpython をリンクしないため素の `cargo test` ではリンクエラーになる
- wheel: `maturin develop` / `maturin build --release --out dist` (otspot-py/ 内で実行)。`--out` なしの出力先は `../target/wheels/`
- `pytest tests/` と `mypy --check-untyped-defs tests/ otspot.pyi` (otspot.pyi の stub drift 検出)
- 公開 API parity: otspot-model (および otspot-core の `SolveStatus`/`Tolerance`) の pub item を追加・変更したら `otspot-py/api_manifest.json` と snapshot (`otspot-py/*_snapshot.txt`) を更新 — `bash scripts/check_otspot_model_public_api.sh --update` (nightly-2026-05-01 + cargo-public-api 必要)。CI が cargo-public-api diff で監視

## ベンチマーク

`cargo bench` ではない。`scripts/` の shell harness を使う:

```bash
bash scripts/bench_parallel.sh --data-dir data/<suite> --eps 1e-6 --jobs 6 --timeout 1000 --output <file>
```

- 基準値: timeout=1000s (バグ再現の短縮時 400s 可)、eps=1e-6、jobs=6。suite は並行させず順次実行。`--jobs` は必須 (暗黙デフォルトなし)
- バイナリ (qps_benchmark 等) を直接呼ばない (結果集計が壊れる)
- SOCP (data/cblib_socp) は harness 非対応。`solve_cbf` example を jobs=1 で直列実行 (大規模問題は ~18 GB RSS、並列で OOM)
- 判定は合計 PASS 数でなく baseline との per-problem diff。TIMEOUT と非収束は別物 (skill bug-frontier の 6 状態)

## git 運用 (CLAUDE.md 抜粋)

- コミットメッセージは日本語 (conventional prefix 可)
- main への commit/merge はユーザの明示許可があるときのみ。通常は branch を切り、integrate 系 branch へ `--no-ff` で merge
- push は毎回ユーザに確認する。push 済み commit は amend/rebase/squash しない (新 commit で修正)
- `git add -A` を無確認で使わない。commit 前 `git status` で stage 内容を確認し、target/ 等の生成物を含めない
- subagent 作業は git worktree を使う。worktree 削除前に `git status --untracked-files=all` で未コミット成果物を確認

## 実装規約 (要点)

- 症状隠し・cap・当該経路の無効化・マジックナンバーでの抑え込みは修正ではない。真因が未特定なら手を入れず調査を続ける
- 修正には sentinel test を付ける (revert すると fail することを確認)
- 実装を正とせず独立オラクル (手計算 / 別ソルバ / 独立再計算) で期待値を作る
- 既存テストの assert・ロジックは変更しない (削除 + 新規追加で表現)
- マジックナンバーは根拠つき named const に (/// docstring 付き)
- 後方互換のために旧実装を残さない
- 指摘を受けたら同パターンを repo 全体に横展開する (grep で全該当箇所を列挙してから直す)

## その他

- `dashboard.md` は空が正常。ユーザ判断が必要な事項とレコメンドのみ記載する
- 性能・修正の主張には実測値 (PASS/FAIL 数・時間・iters) が必須。数値のない「改善した」は主張として受け付けない
- README の性能表を更新する場合は同一条件 (suite・eps・timeout・jobs) で実測する
