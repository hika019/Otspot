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
| subtask_042e | git初回コミット（全16ファイル） | 足軽3(Sonnet) | L3 | 042a-d | 🔄作業中 | gitコミット |

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
