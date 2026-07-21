---
name: bug-frontier
description: バグ調査・修正・性能退化調査を開始する前に必ず読む。検証フロンティア手法 (入力から段ごとに事実を積み上げる) と結果6状態の区別、修正の完了条件を定義する。
---

# 検証フロンティア — バグ対応の基本手法

全バグ対応の基本。対象を入力→出力の処理段に分解し、各段の正当性を入力から順に「事実」として確定する。前段が proven でなければ下流の正誤は判定不能 (発生場所≠真因)。proven 判定は実測のみで行い、推論で proven にしない。

- 処理段は対象ごとに定義する。例: LP = parse→presolve→Phase I→Phase II→postsolve。QP = presolve→IPM(IPPMM)反復→postsolve。MIP/非凸QP/MISOCP は各 B&B ノードが LP/QP/SOCP を呼ぶ土台依存。
- フロンティア = proven 済の最下流の「次の段」。検証も修正も常にフロンティア段に対して行う。上流が未 proven のときは、下流症状の調査より先に上流を proven にする。
- 各段で correctness (誤出力/符号反転/誤変換/不変条件違反/偽判定 false-Infeasible 等) と 非correctness (収束しない/hang/stall/性能劣化) の両方を洗う。複数バグ前提。

## proven の具体手段 (実測)
- 入力段: 内部表現を生入力と機械突合 (制約数/sense/RHS/bound/変数数)。
- 中間変換段: 入出力の等価性・不変条件を機械検査 + 全分岐 sentinel test (修正を revert すると fail する設計)。
- 計算段: 出力を独立再計算 or 証明書で検証 (基底の実行可能性・Farkas・dual符号・原制約充足・report obj == 再計算 obj)。

## ループ (1 iteration)
1. 観測を再現 (bench: timeout=400 or 1000, eps=1e-6 固定) — subagent 実行。
2. 異常を【段】×【correctness / それ以外】で分類 (timing/iters/残差の実測値から)。
3. フロンティア段を厳密検証し proven か全バグかを確定 (subagent/worktree。env-gated トレース可、報告後 revert)。
4. 見つけた全バグを真因修正する。症状隠し・cap・当該段の無効化・マジックナンバーで抑えるのは修正ではない — 真因が未特定なら修正せず調査を続ける。
5. lead が full-suite (`cargo nextest run --release --test-threads 6`) を実走 → integrate へ --no-ff マージ。
6. フロンティア段の proven を再確認 → フロンティア前進 → 再観測。
- 停止条件: 全段 proven / 要ユーザ判断 / 割込。
- subagent の報告は推論として扱う。lead がテスト実走・grep・実測値で fact 化してから確定する。

# 結果の状態区別 (6状態、混同禁止)

最終 status 文字列でなく反復軌跡を実測して判定する。

- ①regression (修正前 PASS が修正のロジックで FAIL) と ②既存バグ表面化 (修正が相殺/マスキング条件を外しただけ) は別物。顕在化箇所は修正位置と無関係でありうる。「修正後に出た = ①」と決めず、bisect で「修正前 PASS か」+ マスキング関係で峻別する。
- ③解が出た (実行可能解) ≠ ④収束した (最適性証明書 or 既知最適と一致)。
- ⑤TIMEOUT ≠ ⑥収束しない。⑤ = 軌跡が最適へ単調 (時間不足) or 特定段だけ病的に遅い。⑥ = 軌跡が flat/振動。
- ②疑い時は入力から検証フロンティアを引き直し独立再計算で再 proven。既存 PASS も unsound 検査が偽 proven を作りうるため信用しない。

# 完了条件

以下が全て揃って「修正完了」。1つでも欠けたら in_progress のまま:
- 真因の特定根拠が実測値で報告に含まれている。
- sentinel test が追加され、修正 revert で fail することを確認済み。
- full-suite PASS/FAIL 数が報告に含まれている。
- 性能に関わる修正は再ベンチの前後比較 (PASS数/時間/iters) が含まれている。
- fail を「範疇外」「別の真因」で分離して closed にしない。検証空白 (no_ref/合成データのみ/SKIP) はバグ不在を保証しない — 空白を埋めるテスト追加 or 真因対処で応答する。
