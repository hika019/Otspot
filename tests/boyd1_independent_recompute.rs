//! BOYD1/BOYD2: 報告 obj == 問題データからの独立再計算、かつ解の実行可能性。
//!
//! cold 初期化政策 (cold/warm マージン分離) の変更で BOYD1 の反復数が予算
//! sentinel (boyd_gap_noise_floor_sentinel) を超えたとき、それが「予算数の陳腐化」
//! ではなく「解の品質劣化」であることを示す証拠は無い — 本テストは段 S5 (出力) を
//! 独立再計算で proven にする: solver の報告値を信用せず、QPS 由来の
//! Q・c・A・b・bounds・obj_offset から obj と実行可能性を再計算して突合する。
//! 反復が増えても Optimal・再計算一致・feasible が崩れないかが判定の中心問い。

use otspot::io::qps::parse_qps;
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::qp::solve_qp_with;
use std::path::Path;

const TIMEOUT_SECS: f64 = 120.0;

/// 実行可能性・再計算 obj の相対許容。ベンチ基準 eps=1e-6 と同水準 (それ以上の
/// 厳しさは収束判定そのものの契約に反し、緩めれば病理を検出できない)。
const FEAS_TOL: f64 = 1e-6;

/// CSC 生アクセスでの matvec (solver 内部コードを再利用せず独立に畳む)。
fn matvec(a: &otspot::num::sparse::CscMatrix, x: &[f64]) -> Vec<f64> {
    let mut ax = vec![0.0f64; a.nrows()];
    // CSC 列区間は col_ptr の隣接対で定義される。
    for (ptrs, xv) in a.col_ptr().windows(2).zip(x.iter()) {
        for k in ptrs[0]..ptrs[1] {
            ax[a.row_ind()[k]] += a.values()[k] * xv;
        }
    }
    ax
}

fn solve_mm(name: &str) -> (otspot::qp::QpProblem, otspot::SolverResult) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/maros_meszaros")
        .join(format!("{name}.QPS"));
    assert!(
        path.exists(),
        "{name}.QPS not found — bench data 未配置。scripts/maros_meszaros_download.sh を実行",
    );
    let prob = parse_qps(&path).expect(name);
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_SECS);
    let result = solve_qp_with(&prob, &opts);
    eprintln!(
        "{name}_recompute: status={:?} obj={:.10e} iters={}",
        result.status, result.objective, result.iterations
    );
    (prob, result)
}

/// 報告 obj == 独立再計算 (1/2 x'Qx + c'x + obj_offset) と bounds/制約の実行可能性
/// (FEAS_TOL 相対) を検査し、result を返す。
fn recompute_and_check_feasible(name: &str) -> otspot::SolverResult {
    let (prob, result) = solve_mm(name);
    let x = &result.solution;
    assert_eq!(x.len(), prob.num_vars, "解ベクトル長");

    let qx = matvec(&prob.q, x);
    let quad: f64 = x.iter().zip(qx.iter()).map(|(&xi, &qxi)| xi * qxi).sum();
    let lin: f64 = x.iter().zip(prob.c.iter()).map(|(&xi, &ci)| xi * ci).sum();
    let recomputed = 0.5 * quad + lin + prob.obj_offset;
    let obj_scale = recomputed.abs().max(1.0);
    assert!(
        (result.objective - recomputed).abs() <= FEAS_TOL * obj_scale,
        "報告 obj {:.10e} != 独立再計算 {:.10e} (差 {:.3e})",
        result.objective,
        recomputed,
        (result.objective - recomputed).abs()
    );

    // 境界実行可能性 (tol は |b|∼1e12 規模の bound に追従させる)。
    for (j, (&xj, &(lb, ub))) in x.iter().zip(prob.bounds.iter()).enumerate() {
        let tol = FEAS_TOL * lb.abs().max(ub.abs()).max(1.0);
        assert!(
            (lb.is_infinite() || xj >= lb - tol) && (ub.is_infinite() || xj <= ub + tol),
            "bound 違反 x[{j}]={xj} ∉ [{lb}, {ub}] (tol={tol:.1e})"
        );
    }

    // 制約実行可能性 (sense は内部正規化後の ConstraintType に従う)。
    let ax = matvec(&prob.a, x);
    let mut worst_ge_slack = f64::INFINITY;
    for (i, (&axi, (&bi, typ))) in ax
        .iter()
        .zip(prob.b.iter().zip(prob.constraint_types.iter()))
        .enumerate()
    {
        let tol = FEAS_TOL * bi.abs().max(1.0);
        match typ {
            ConstraintType::Eq => assert!(
                (axi - bi).abs() <= tol,
                "eq 行 {i}: |ax-b|={:.3e} > tol={tol:.1e} (b={:.3e})",
                (axi - bi).abs(),
                bi
            ),
            ConstraintType::Le => assert!(
                axi <= bi + tol,
                "le 行 {i}: ax={:.6e} > b+tol={:.6e}",
                axi,
                bi + tol
            ),
            ConstraintType::Ge => {
                worst_ge_slack = worst_ge_slack.min(axi - bi);
                assert!(
                    axi >= bi - tol,
                    "ge 行 {i}: ax={:.6e} < b-tol={:.6e}",
                    axi,
                    bi - tol
                );
            }
            // non_exhaustive 拡張は失敗を明示するため黙って通さない。
            _ => panic!("未知の constraint sense (行 {i})"),
        }
    }
    eprintln!("{name}_recompute: feasible (worst ge slack={worst_ge_slack:.6e})");
    result
}

/// BOYD1 は Optimal であることも要求する (品質問題 vs 予算陳腐化の中心判定)。
#[test]
fn boyd1_reported_objective_matches_independent_recompute() {
    let result = recompute_and_check_feasible("BOYD1");
    assert_eq!(
        result.status,
        SolveStatus::Optimal,
        "BOYD1 が Optimal でない"
    );
}

/// BOYD2 版は status の Optimal 要求を行わない: 目的は退化終了
/// (SuboptimalSolution) の解が soundness 違反 (非実行可能・obj 水増し) か
/// quality gap (実行可能だが非最適) かの切り分け。最適性ギャップ自体は台帳側で
/// cert 付き baseline (2.125676700395e1) / 独立オラクルと突合する。
#[test]
fn boyd2_reported_objective_matches_independent_recompute() {
    recompute_and_check_feasible("BOYD2");
}
