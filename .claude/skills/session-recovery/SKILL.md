---
name: session-recovery
description: セッション中断 (SSH切断/PC再起動/clear/limit) からの再開・引き継ぎ時に必ず読む。中断状態の復元手順と、未コミット成果物のサルベージ判断を定義する。
---

# セッション復旧手順

「前の作業から再開して」「状況を把握して」と言われたら、以下を順に実測してから報告する。推測で状況を語らない。

## 1. 状態収集 (並列実行可)
- `git status` / `git log --oneline -15` / `git branch -a` / `git worktree list`
- TaskList でタスク残骸を確認
- `tmux ls` (エージェント残骸)
- `cat dashboard.md` (空が正常)
- `ls -lt bench_results/ | head` と直近の変更ファイル (`find . -maxdepth 3 -newermt '<昨日>' -type f`)

## 2. worktree ごとの精査
各 worktree で `git status --short --untracked-files=all` と `git log --oneline <integrate>..HEAD`:
- 未コミット diff があれば **中身を読む**。中断された作業本体である可能性が高い。
- コンパイルが通るか (`cargo check`) で中断点を特定する。

## 3. サルベージ判断
- 未コミット変更は削除前に必ず価値判定する。判定基準: integrate に同等の変更が既にあるか (`git diff <integrate> -- <file>`)、修正が真因対処か。
- 価値あり → patch を取り出し integrate HEAD 起点の新 worktree へ移植し、テストを付けて正規ルートで review/マージ。
- 価値なし/取り込み済み → worktree 削除。削除前に `git status --untracked-files=all` で未コミット成果物ゼロを確認する。

## 4. 再開報告
①中断時に何をしていたか (事実) ②サルベージしたもの ③再開したタスクと担当 agent ④要ユーザ判断 — を数行で報告してから作業に入る。

## 引き継ぎメモを求められたら
200字以内。含めるもの: branch/commit、良コミットの hash、次の1アクション、罠 (あれば)。それ以外は書かない。
