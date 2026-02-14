# cmd_011: OSSソルバー調査 — Subtask一覧

## 実行順序
```
Phase 1 (並列調査)     → Phase 1.5 (ベンチ/戦略)  → Phase 2 (要件定義)
011a, 011b, 011c       → 011e, 011f               → 011d
```

## Subtask一覧

| ID | 内容 | 担当 | Bloom | 依存 | 状態 | 成果物 |
|----|------|------|-------|------|------|--------|
| subtask_011a | OSS調査(HiGHS,SCIP,CBC,OR-Tools,GLPK,Ipopt) | 足軽1(Sonnet) | L2 | なし | ✅完了 | solver/oss_solvers.md |
| subtask_011b | 商用調査(Gurobi,CPLEX,Mosek,Xpress) | 足軽2(Sonnet) | L2 | なし | ✅完了 | solver/commercial_solvers.md |
| subtask_011c | 研究動向(2024-2026) | 足軽3(Sonnet) | L2 | なし | ✅完了 | solver/research_trends.md |
| subtask_011e | ベンチマーク手法(MIPLIB,Netlib,評価指標) | 足軽1(Sonnet) | L2 | なし | ✅完了 | solver/benchmarking.md |
| subtask_011f | 天下取り戦略+OSSアピール方法 | 足軽2(Sonnet) | L2 | なし | ✅完了 | solver/winning_strategy.md |
| subtask_011d | 要件定義統合(全調査結果→推奨案) | 足軽5(Opus) | L5 | 011e,011f | ✅完了 | solver/requirements.md |

## 依存関係図
```
011a ─┐
011b ─┤→ (Phase1完了)
011c ─┘
              011e ─┐
              011f ─┤→ 011d (要件定義)
              └─────┘
```

## 補足
- Phase 1: 既存ソルバー3方面並列調査（OSS/商用/研究動向）
- Phase 1.5: ベンチマーク手法 + 差別化戦略調査
- Phase 2: 全調査結果を統合し要件定義を策定（Opus L5: 分析・評価・判断）
- 全レポートで「事実（ソース付き）」と「意見/分析」を明確分離（将軍指示）

---

# cmd_042: solverプロジェクト完了作業 — Subtask一覧

## 目的
cmd_011の全成果物を品質確認し、gitコミットまで行いプロジェクトをクローズする。

## 実行順序
```
Phase 1 (並列品質確認)       → Phase 2 (gitコミット)
042a, 042b, 042c, 042d       → 042e
```

## Subtask一覧

| ID | 内容 | 担当 | Bloom | 依存 | 状態 | 成果物 |
|----|------|------|-------|------|------|--------|
| subtask_042a | winning_strategy.md (011f) 品質レビュー | 足軽5(Opus) | L5 | なし | ✅完了 | 報告YAML |
| subtask_042b | requirements.md (011d) + executive_summary 整合性 | 足軽6(Opus) | L5 | なし | ✅完了 | 報告YAML |
| subtask_042c | ファイル整合性チェック（前半8ファイル） | 足軽1(Sonnet) | L2 | なし | ✅完了 | 報告YAML |
| subtask_042d | ファイル整合性チェック（後半8ファイル） | 足軽2(Sonnet) | L2 | なし | ✅完了 | 報告YAML |
| subtask_042e | git初回コミット（全16ファイル） | 足軽3(Sonnet) | L3 | 042a-d | ✅完了 | git commit 4a08dc5 |

## 品質レビュー結果サマリ

- **011f (winning_strategy.md)**: 完了。軽微不足あり（タイムライン未記載、リスクセクション不在、未使用ソース2件）。致命的欠陥なし
- **011d (requirements.md)**: 完了。機能要件・非機能要件・技術選定・ロードマップ全て十分。致命的欠陥なし
- **executive_summary.md**: 不整合あり（重大だが致命的ではない）
  - ES推奨: NLP/MINLP + GPU特化（LP/MIP捨てる）
  - requirements推奨: LP/MIP Phase1必須
  - 実装言語: ES=C++, requirements=Rust
  - ライセンス: ES=MIT, requirements=Apache-2.0
  - → 殿の戦略判断事項として適切に整理されている
- **ファイル整合性**: 全16ファイル問題なし（リンク切れ0、未完セクション0）

---

# cmd_043: solver戦略不整合の収束 — 構造化討論

## 目的
executive_summary.md（NLP/GPU特化・C++・MIT）と requirements.md（LP/MIP Phase1・Rust・Apache-2.0）の
戦略不整合を、足軽間の構造化討論で収束させ、統一戦略を策定する。

## 論点
1. **参入領域**: NLP/GPU特化 vs LP/MIP Phase 1
2. **実装言語**: C++ vs Rust vs Go（三つ巴）
3. **ライセンス**: MIT vs Apache-2.0

## 実行順序
```
Phase 1 (並列・開局)    → Phase 2 (並列・反駁)     → Phase 3 (並列・結辯)
043a, 043b, 043c        → 043d, 043e, 043f          → 043g, 043h, 043i
                                                        → Phase 4 (検収) → Phase 5 (commit)
                                                          043j             043k
```

## Subtask一覧

| ID | Phase | 内容 | 担当 | Bloom | 依存 | 状態 | 成果物 |
|----|-------|------|------|-------|------|------|--------|
| subtask_043a | 1 開局 | ES派 Opening (NLP/GPU + C++ + MIT) | 足軽1(Sonnet) | L5 | なし | 🔄作業中 | solver/debate_es_opening.md |
| subtask_043b | 1 開局 | Req派 Opening (LP/MIP + Rust + Apache) | 足軽2(Sonnet) | L5 | なし | 🔄作業中 | solver/debate_req_opening.md |
| subtask_043c | 1 開局 | Go派 Opening (実用主義 + Go) | 足軽3(Sonnet) | L5 | なし | 🔄作業中 | solver/debate_go_opening.md |
| subtask_043d | 2 反駁 | ES派 Rebuttal | 足軽1(Sonnet) | L5 | 043a-c | ⏳待機 | solver/debate_es_rebuttal.md |
| subtask_043e | 2 反駁 | Req派 Rebuttal | 足軽2(Sonnet) | L5 | 043a-c | ⏳待機 | solver/debate_req_rebuttal.md |
| subtask_043f | 2 反駁 | Go派 Rebuttal | 足軽3(Sonnet) | L5 | 043a-c | ⏳待機 | solver/debate_go_rebuttal.md |
| subtask_043g | 3 結辯 | ES派 Closing | 足軽1(Sonnet) | L5 | 043d-f | ⏳待機 | solver/debate_es_closing.md |
| subtask_043h | 3 結辯 | Req派 Closing | 足軽2(Sonnet) | L5 | 043d-f | ⏳待機 | solver/debate_req_closing.md |
| subtask_043i | 3 結辯 | Go派 Closing | 足軽3(Sonnet) | L5 | 043d-f | ⏳待機 | solver/debate_go_closing.md |
| subtask_043j | 4 検収 | 統一戦略策定 (全議論読み→決定) | 足軽5(Opus) | L6 | 043g-i | ⏳待機 | solver/unified_strategy.md |
| subtask_043k | 5 完了 | subtask_plan更新 + git commit | 足軽4(Sonnet) | L3 | 043j | ⏳待機 | git commit |

## 依存関係図
```
043a ─┐
043b ─┤→ (Phase1完了)
043c ─┘
         043d ─┐
         043e ─┤→ (Phase2完了)
         043f ─┘
                   043g ─┐
                   043h ─┤→ (Phase3完了) → 043j (Opus検収) → 043k (commit)
                   043i ─┘
```

## 討論構造
- Phase 1（開局）: 3派が独立にOpening Statement。相手の文書は読まない
- Phase 2（反駁）: 全Phase 1文書を読み、相手への反論を構築
- Phase 3（結辯）: Phase 1+2の全文書を読み、最終弁論+収束点の提示
- Phase 4（検収）: Opus足軽が全9文書を読み、3論点それぞれに統一結論を出す
- Phase 5（完了）: git commit + 本ファイル更新

---

# cmd_045: solver Phase 1 M1 基盤構築 — Subtask一覧

## 目的
Rustプロジェクト初期化 + Primal Simplex実装 + 基本テストが動く状態にする。
統一戦略(unified_strategy.md)に基づく。将軍決定: Rust / Apache-2.0+MIT dual / HiGHS比80%目標。

## 実行順序
```
Phase 1 (基盤)       → Phase 2 (並列実装)          → Phase 3 (核心)   → Phase 4 (完了)
045a                  → 045b, 045c, 045d             → 045e              → 045f
```

## Subtask一覧

| ID | 内容 | 担当 | Bloom | 依存 | 状態 | 成果物 |
|----|------|------|-------|------|------|--------|
| subtask_045a | Git整理+cargo init+scaffolding | 足軽1(Sonnet) | L3 | なし | 🔄作業中 | Cargo.toml, src/, LICENSE-* |
| subtask_045b | CSC疎行列実装 | 足軽2(Sonnet) | L3 | 045a | ⏳待機 | src/sparse/mod.rs |
| subtask_045c | LP問題定義実装 | 足軽3(Sonnet) | L3 | 045a | ⏳待機 | src/problem/mod.rs |
| subtask_045d | README.md作成 | 足軽4(Sonnet) | L2 | 045a | ⏳待機 | README.md |
| subtask_045e | Primal Simplex(PhaseI+II)+テスト | 足軽5(Opus) | L6 | 045b,045c | ⏳待機 | src/simplex/, tests/ |
| subtask_045f | git commit (feature/phase1-m1) | (未割当) | L1 | 045d,045e | ⏳待機 | git commit |

## 依存関係図
```
045a (scaffolding) ─→ 045b (CSC) ────┐
                   ├→ 045c (Problem) ─┤→ 045e (Simplex+Test) ─→ 045f (commit)
                   └→ 045d (README) ──────────────────────────→ ┘
```

## 設計方針
- scaffoldingでstruct定義済み → CSC/Problemは実装に集中
- CSCとProblemは別ディレクトリ（RACE-001回避）
- Simplex(L6)がOpus必須の核心タスク
- 全足軽にgit push禁止を徹底
