//! LP 独立連結成分分割の sentinel tests (hand-computed independent oracle)。
//!
//! 期待値はすべて問題データから手計算したもので、ソルバ実行から導出していない。
//! `stats.lp_split_components` は分割発火の診断 field で、分割実装の revert と
//! 同時に存在しなくなる (revert 時はこの file のコンパイルが通り nextest が fail)。
//! 発火・非発火・結合規則の各分岐を観測する。

// 合成問題ビルダは複数の parallel vec を行/列番号で同時 index するため、
// range loop をイテレータ書きにできない (blackbox_lp.rs と同様の許可)。
#![allow(clippy::needless_range_loop)]

use otspot_core::lp::solve_lp_with;
use otspot_core::options::SolverOptions;
use otspot_core::problem::{ConstraintType, LpProblem, SolveStatus};
use otspot_core::qp::{solve_qp_with, QpProblem};
use otspot_core::sparse::CscMatrix;

const GE: ConstraintType = ConstraintType::Ge;
const LE: ConstraintType = ConstraintType::Le;
const INF: f64 = f64::INFINITY;

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

/// SolverOptions は non_exhaustive なので外部 crate からリテラル構築できない。
/// default に部分設定を適用するヘルパ経由で作る。
fn opts_with(prep: impl FnOnce(&mut SolverOptions)) -> SolverOptions {
    let mut o = SolverOptions::default();
    prep(&mut o);
    o
}

/// min 2x0 + 3x1  s.t.  x0 >= 1, x1 >= 2,  x >= 0.
/// {x0} と {x1} が独立な 2 成分。手計算: x*=[1,2], obj=2·1+3·2=8,
/// 各行の shadow price は目的係数そのもの (dual=[2,3]), slack=[0,0], rc=[0,0]。
fn two_component_optimal() -> LpProblem {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    LpProblem::new_general(
        vec![2.0, 3.0],
        a,
        vec![1.0, 2.0],
        vec![GE, GE],
        vec![(0.0, INF), (0.0, INF)],
        None,
    )
    .unwrap()
}

#[test]
fn split_two_components_join_optimal_hand_computed() {
    let r = solve_lp_with(&two_component_optimal(), &SolverOptions::default());
    assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
    assert!((r.objective - 8.0).abs() <= 1e-9, "obj={}", r.objective);
    assert!(close(&r.solution, &[1.0, 2.0], 1e-9), "x={:?}", r.solution);
    assert!(
        close(&r.reduced_costs, &[0.0, 0.0], 1e-9),
        "rc={:?}",
        r.reduced_costs
    );
    assert!(
        close(&r.dual_solution, &[2.0, 3.0], 1e-9),
        "dual={:?}",
        r.dual_solution
    );
    assert!(close(&r.slack, &[0.0, 0.0], 1e-9), "slack={:?}", r.slack);
    assert_eq!(r.stats.lp_split_components, 2);
}

/// 成分 {x0} は可行、成分 {x1} は x1<=1 かつ x1>=2 で Infeasible。
/// 直積の実行可能集合は空 → 全体 Infeasible (Unbounded 等の他主張より優先)。
#[test]
fn split_infeasible_component_wins() {
    let a = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 1], &[1.0, 1.0, 1.0], 3, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![2.0, 3.0],
        a,
        vec![1.0, 1.0, 2.0],
        vec![GE, LE, GE],
        vec![(0.0, INF), (0.0, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &SolverOptions::default());
    assert!(
        matches!(r.status, SolveStatus::Infeasible),
        "{:?}",
        r.status
    );
    assert_eq!(r.stats.lp_split_components, 2);
    assert!(r.solution.is_empty());
}

/// 成分 {x0}: min 2x0, x0>=1 → Optimal。成分 {x1}: min -x1, x1>=1 → Unbounded。
/// 全成分可行の下で 1 成分が Unbounded → 全体 Unbounded (obj=-inf)。
#[test]
fn split_unbounded_component_propagates() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![2.0, -1.0],
        a,
        vec![1.0, 1.0],
        vec![GE, GE],
        vec![(0.0, INF), (0.0, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &SolverOptions::default());
    assert!(matches!(r.status, SolveStatus::Unbounded), "{:?}", r.status);
    assert_eq!(r.objective, f64::NEG_INFINITY);
    assert_eq!(r.stats.lp_split_components, 2);
}

/// 行を持たない成分のみ (純箱 LP)。閉形式経路: min x0 - x1, x0∈[0,∞), x1∈[0,5]
/// → x*=[0,5], obj=-5, rc=[1,-1], dual/slack 空。
#[test]
fn split_box_only_components_closed_form() {
    let a = CscMatrix::new(0, 2);
    let lp = LpProblem::new_general(
        vec![1.0, -1.0],
        a,
        vec![],
        vec![],
        vec![(0.0, INF), (0.0, 5.0)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &SolverOptions::default());
    assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
    assert_eq!(r.objective, -5.0);
    assert!(close(&r.solution, &[0.0, 5.0], 1e-12), "x={:?}", r.solution);
    assert!(
        close(&r.reduced_costs, &[1.0, -1.0], 1e-12),
        "rc={:?}",
        r.reduced_costs
    );
    assert_eq!(r.stats.lp_split_components, 2);
}

/// 連結 (同一行に全変数が出現) は分割不发火で従来経路と同一。
/// min x0+x1, x0+x1>=3, x>=0 → obj=3 (手計算)。
#[test]
fn connected_problem_not_split() {
    let a = CscMatrix::from_triplets(&[0, 0], &[0, 1], &[1.0, 1.0], 1, 2).unwrap();
    let lp = LpProblem::new_general(
        vec![1.0, 1.0],
        a,
        vec![3.0],
        vec![GE],
        vec![(0.0, INF), (0.0, INF)],
        None,
    )
    .unwrap();
    let r = solve_lp_with(&lp, &SolverOptions::default());
    assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
    assert!((r.objective - 3.0).abs() <= 1e-9, "obj={}", r.objective);
    assert_eq!(r.stats.lp_split_components, 0);
}

/// warm start / 初期 basis 要求系ガード: 行列表現が変わるため分割しない。
/// recover_warm_start_basis=true の 2 成分 LP → 不发火 (components=0) かつ
/// 単独求解と同一の obj。
#[test]
fn basis_dependent_options_skip_split() {
    let opts = opts_with(|o| o.recover_warm_start_basis = true);
    let r = solve_lp_with(&two_component_optimal(), &opts);
    assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
    assert!((r.objective - 8.0).abs() <= 1e-9, "obj={}", r.objective);
    assert_eq!(r.stats.lp_split_components, 0);
}

/// deadline 残ゼロでは分割経路は Timeout を返す (保存的: Optimal を主張しない)。
#[test]
fn expired_deadline_returns_timeout() {
    let opts = opts_with(|o| o.timeout_secs = Some(0.0));
    let r = solve_lp_with(&two_component_optimal(), &opts);
    assert!(matches!(r.status, SolveStatus::Timeout), "{:?}", r.status);
    assert!(r.stats.deadline_triggered);
}

/// 転送入口 (`solve_qp_with(Q=0)` 経由) は lp_dispatch が段分工順上
/// presolve=false に変換するため、presolve=false でも分割が発火する。
/// 直入口の sensitivity ガード (次の test) と入口で挙動が分かれること自体が
/// 契約であり、転送側でガードを見ると全 QPS LP が非発火に落ちる
/// (fome21 実測 bail=presolve_off の回帰 sentinel)。
#[test]
fn forwarded_zero_q_splits_despite_no_presolve() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let q = CscMatrix::new(2, 2);
    let qp = QpProblem::new(
        q,
        vec![2.0, 3.0],
        a,
        vec![1.0, 2.0],
        vec![(0.0, INF), (0.0, INF)],
        vec![GE, GE],
    )
    .unwrap();
    let opts = opts_with(|o| o.presolve = false);
    let r = solve_qp_with(&qp, &opts);
    assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
    assert!((r.objective - 8.0).abs() <= 1e-9, "obj={}", r.objective);
    assert!(close(&r.solution, &[1.0, 2.0], 1e-9), "x={:?}", r.solution);
    assert_eq!(r.stats.lp_split_components, 2);
}

/// 直入口 (`solve_lp_with`) では presolve=false が sensitivity の basis 規約
/// なので分割しない (components=0)。単独求解の結果 (obj=8) は従来と同一。
#[test]
fn direct_entry_no_presolve_skips_split() {
    let opts = opts_with(|o| o.presolve = false);
    let r = solve_lp_with(&two_component_optimal(), &opts);
    assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
    assert!((r.objective - 8.0).abs() <= 1e-9, "obj={}", r.objective);
    assert_eq!(r.stats.lp_split_components, 0);
    // sensitivity 契約: basis が結果に付く (分割不发火のときのみ保証される)。
    assert!(r.warm_start_basis.is_some());
}

/// 同一ブロック 2 個のコピー構造 (決定的 LCG で合成)。分割 ON (既定) と
/// OFF (recover_warm_start_basis ガードで非発火) で obj が一致し、どちらも
/// Optimal。単独ブロックの obj ×2 とも一致 (直積構造の独立オラクル)。
#[test]
fn copy_block_split_on_off_objective_match() {
    // 決定的 LCG によるブロック (n=15, m=12)。c に負係数、ub=50 で有限最適値。
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next_mod = move |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    let n = 15usize;
    let mm = 12usize;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut c_block = vec![0.0f64; n];
    for j in 0..n {
        c_block[j] = next_mod(9) as f64 - 3.0;
    }
    for i in 0..mm {
        let mut row_entries = 0;
        for j in 0..n {
            if next_mod(10) < 3 {
                rows.push(i);
                cols.push(j);
                vals.push(1.0 + next_mod(4) as f64);
                row_entries += 1;
            }
        }
        if row_entries == 0 {
            rows.push(i);
            cols.push(i % n);
            vals.push(1.0);
        }
    }
    // 連結性を保証: 鎖状の結合行を列ごとに追加 (x_j + x_{j+1} <= 100)。
    for j in 0..n - 1 {
        rows.push(mm + j);
        cols.push(j);
        vals.push(1.0);
        rows.push(mm + j);
        cols.push(j + 1);
        vals.push(1.0);
    }
    let m_total = mm + n - 1;
    let b_block: Vec<f64> = (0..m_total).map(|i| 50.0 + i as f64).collect();
    let types_block: Vec<ConstraintType> = vec![LE; m_total];
    let bounds_block = vec![(0.0, 50.0); n];

    let single = LpProblem::new_general(
        c_block.clone(),
        CscMatrix::from_triplets(&rows, &cols, &vals, m_total, n).unwrap(),
        b_block.clone(),
        types_block.clone(),
        bounds_block.clone(),
        Some("block".to_string()),
    )
    .unwrap();
    let obj_single = {
        let r = solve_lp_with(&single, &SolverOptions::default());
        assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
        assert_eq!(r.stats.lp_split_components, 0, "単一ブロックは連結");
        r.objective
    };

    // 2 ブロックコピー (行列表面は同一、変数は独立)。
    let rows2: Vec<usize> = rows
        .iter()
        .copied()
        .chain(rows.iter().map(|&i| i + m_total))
        .collect();
    let cols2: Vec<usize> = cols
        .iter()
        .copied()
        .chain(cols.iter().map(|&j| j + n))
        .collect();
    let vals2: Vec<f64> = vals.iter().chain(vals.iter()).copied().collect();
    let double = LpProblem::new_general(
        [c_block.as_slice(), c_block.as_slice()].concat(),
        CscMatrix::from_triplets(&rows2, &cols2, &vals2, 2 * m_total, 2 * n).unwrap(),
        [b_block.as_slice(), b_block.as_slice()].concat(),
        [types_block.as_slice(), types_block.as_slice()].concat(),
        [bounds_block.as_slice(), bounds_block.as_slice()].concat(),
        Some("blocks2".to_string()),
    )
    .unwrap();

    let on = solve_lp_with(&double, &SolverOptions::default());
    let off_opts = opts_with(|o| o.recover_warm_start_basis = true);
    let off = solve_lp_with(&double, &off_opts);

    assert!(matches!(on.status, SolveStatus::Optimal), "{:?}", on.status);
    assert!(
        matches!(off.status, SolveStatus::Optimal),
        "{:?}",
        off.status
    );
    assert_eq!(on.stats.lp_split_components, 2);
    assert_eq!(off.stats.lp_split_components, 0);
    let scale = on.objective.abs().max(1.0);
    assert!(
        (on.objective - off.objective).abs() <= 1e-6 * scale,
        "on={} off={}",
        on.objective,
        off.objective
    );
    assert!(
        (on.objective - 2.0 * obj_single).abs() <= 1e-6 * scale,
        "on={} single={}",
        on.objective,
        obj_single
    );
}

/// wide 2 成分 LP (LP_IPM ゲート寸法: n=6000, m=5998, nnz=11996, n>=m,
/// 幅比 60000 <= 131956, 次元 11998 >= 10000)。各成分は鎖状
/// x_i + x_{i+1} >= 1 (3000 頂点 / 2999 行)、min Σ x_i、x ∈ [0,1]。
/// 手計算オラクル: 路頂点被覆 = 最大マッチング = 1 成分 1500 → 総最適値 3000
/// (路は二部グラフなので LP 緩和が整数解 = König)。
/// 分割をゲート (`should_try_lp_ipm` の lp-ipm 直通) より前に置かないと
/// この問題は成分分割に到達せず components=0 になる — この assert が
/// ゲート前分割の sentinel (lp_dispatch.rs の分割入口を revert すると赤)。
#[test]
fn wide_two_component_lp_splits_before_ipm_gate() {
    let n_c = 3000usize;
    let m_c = n_c - 1;
    let mut rows = Vec::with_capacity(2 * m_c * 2);
    let mut cols = Vec::with_capacity(2 * m_c * 2);
    let mut vals = Vec::with_capacity(2 * m_c * 2);
    for comp in 0..2usize {
        let col0 = comp * n_c;
        let row0 = comp * m_c;
        for i in 0..m_c {
            rows.push(row0 + i);
            cols.push(col0 + i);
            vals.push(1.0);
            rows.push(row0 + i);
            cols.push(col0 + i + 1);
            vals.push(1.0);
        }
    }
    let m = 2 * m_c;
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, 2 * n_c).unwrap();
    let qp = QpProblem::new(
        CscMatrix::new(2 * n_c, 2 * n_c),
        vec![1.0; 2 * n_c],
        a,
        vec![1.0; m],
        vec![(0.0, 1.0); 2 * n_c],
        vec![GE; m],
    )
    .unwrap();
    let r = solve_qp_with(&qp, &SolverOptions::default());
    assert!(matches!(r.status, SolveStatus::Optimal), "{:?}", r.status);
    assert_eq!(
        r.stats.lp_split_components, 2,
        "wide ゲートより前の分割発火が必須"
    );
    assert!(
        (r.objective - 3000.0).abs() <= 1e-6,
        "obj={} (手計算: 路頂点被覆 1500×2)",
        r.objective
    );
}

/// 転送経路で 1 成分 Unbounded: obj は -inf のまま (obj_offset 非加算)。
/// 分割が `solve_as_lp` 前段ラッパ (`solve_as_lp_via_split`) で発火する経路で
/// `add_qp_obj_offset` の意味論 (solution を主張しない status には加算しない)
/// が非分割経路と一致することを固定する。
#[test]
fn forwarded_unbounded_split_keeps_neg_inf_with_offset() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 1], &[1.0, 1.0], 2, 2).unwrap();
    let mut qp = QpProblem::new(
        CscMatrix::new(2, 2),
        vec![2.0, -1.0],
        a,
        vec![1.0, 1.0],
        vec![(0.0, INF), (0.0, INF)],
        vec![GE, GE],
    )
    .unwrap();
    qp.obj_offset = 42.5;
    let r = solve_qp_with(&qp, &SolverOptions::default());
    assert!(matches!(r.status, SolveStatus::Unbounded), "{:?}", r.status);
    assert_eq!(r.objective, f64::NEG_INFINITY, "-inf に offset を足さない");
    assert_eq!(r.stats.lp_split_components, 2);
}
