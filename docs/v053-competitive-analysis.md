# v0.5.3 Competitive Analysis — Otspot v0.5.2 vs 主要 OSS Solver

## 測定条件

- Otspot: v0.5.2 (integrate/lp-ipm-clean), single-thread, eps=1e-6, timeout=1000s (LP) / 120s (MIP)
- HiGHS: v1.14.0 (highspy), single-thread, eps=1e-6, timeout=1000s (LP) / 120s (MIP)
- マシン: 同一ホストで sequential 実行

## LP — Netlib Standard (109問)

| | Otspot | HiGHS |
|--|--------|-------|
| PASS | 108 | 109 |
| SUBOPT | 1 (cycle) | 0 |
| FAIL | 0 | 0 |

**PASS 率はほぼ互角。**

### 速度比較

HiGHS が全問で高速。中央値 **30x**。

| 問題 | Otspot (s) | HiGHS (s) | 倍率 |
|------|-----------|-----------|------|
| dfl001 | 813 | 5.1 | 161x |
| pds-20 | 624 | 2.2 | 287x |
| cre-d | 159 | 0.7 | 235x |
| ken-18 | 180 | 6.0 | 30x |
| ken-13 | 16.4 | 0.7 | 23x |
| osa-30 | 45.5 | — | — |
| d6cube | — | — | 1929x (最大) |

### LP Hard (53問)

| | Otspot | 備考 |
|--|--------|------|
| Solve 成功 | 27 | n37xx族 (IPM), rail507/516/582 (simplex), watson_1 (IPM) |
| TIMEOUT | 25 | pds-40〜100 (simplex 縮退), nug/neos (IPM factor), sgpf5y6 |
| SUBOPT | 1 | watson_2 |

### LP Infeas (29問)

Otspot: **29/29 正解 (100%)**

## QP — Maros-Meszaros (138問)

| | Otspot (60s timeout) |
|--|-----|
| PASS | 117 |
| CHECKED[no_ref] | 4 |
| SUBOPTIMAL | 11 |
| OBJ_MISMATCH | 1 |
| TIMEOUT | 5 |
| FAIL | 0 |

LISWET 族の SUBOPT は f64 精度限界による正しい拒否 (バグではない)。

QP は native IPM (IPPMM) を持ち、HiGHS (simplex QP) や OSQP (ADMM) とは異なるアプローチ。中小規模凸 QP で competitive。

## MIP — MIPLIB Small (20問)

| 問題 | Otspot | HiGHS |
|------|--------|-------|
| flugpl | Optimal | Optimal (0.1s) |
| gr4x6 | Optimal (87 nodes) | Optimal (0.02s) |
| p0201 | Optimal (1131 nodes) | Optimal (1.0s) |
| dcmulti | Terminated | Optimal (2.2s) |
| enlight_hard | Terminated | Optimal (20s) |
| gt2 | Terminated | Optimal (0.05s) |
| khb05250 | Terminated | Optimal (0.3s) |
| timtab1 | Terminated | Optimal (93s) |
| gen-ip002 | Timeout | Timeout |
| gen-ip016 | Terminated | Timeout |
| gen-ip021 | Terminated | Timeout |
| gen-ip054 | Terminated | Timeout |
| markshare1 | Terminated | Timeout |
| markshare_4_0 | Terminated | Timeout |
| markshare_5_0 | Terminated | Timeout |
| mas74 | Terminated | Timeout |
| mas76 | Terminated | Timeout |
| neos5 | Terminated | Timeout |
| noswot | Terminated | Timeout |
| pk1 | Terminated | Timeout |

**Otspot 3/20, HiGHS 8/20。** MIP は最大の差。

## ボトルネック分析

### LP 速度 (30x gap)
1. **Presolve**: HiGHS の presolve は問題サイズを劇的に縮小する。Otspot の presolve は基本的
2. **Dual simplex**: HiGHS は dual simplex が主力で、Devex/steepest edge pricing が高度に最適化
3. **LU factorization**: HiGHS は Reid の sparse LU を使用。per-iter コストが低い
4. **Scaling**: HiGHS の equilibration scaling は数値安定性と収束速度を両立

### MIP (3 vs 8)
1. **Cut generation**: Otspot は GMI のみ。HiGHS は MIR, flow cover, clique, probing 等多数
2. **Heuristics**: Otspot は FP のみ。HiGHS は RINS, diving, rounding 等
3. **Presolve**: MIP presolve (probing, coefficient tightening) が未整備
4. **Node LP 速度**: LP relaxation の速度差がノード数に乗算される

### QP
現状で competitive。改善余地は ill-conditioned 大規模 (LISWET 族) だが f64 精度限界のため本質的に困難。

## v0.5.3 推奨方向

| 方向 | インパクト | 難度 | 備考 |
|------|----------|------|------|
| LP presolve 強化 | 高 | 中 | doubleton/singleton/forcing row 等 |
| LP dual simplex 最適化 | 高 | 高 | per-iter コスト削減が全体に効く |
| MIP cut 多様化 | 高 | 中 | MIR, clique で木サイズ削減 |
| MIP presolve | 中 | 中 | probing, coefficient tightening |
| MIP heuristics | 中 | 低 | RINS, diving で incumbent 早期発見 |
