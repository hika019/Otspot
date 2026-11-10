//! P0 診断の現実 solve 経路スモーク: solve_qp_with のスコープで
//! SolveDiagnostics が公開経路から読め、集計構造 (排他的 bucket・回数分離) が
//! 実測値で破綻していないことを検証する。実時間の厳密一致は要求しない
//! (plan §3: 件数・分類・集計構造のテストまでが unit、wall 対応は診断実行で)。

use otspot_core::problem::SolveStatus;
use otspot_core::qp::diagnostics::AttemptKind;
use otspot_num::sparse::CscMatrix;

fn tiny_qp() -> otspot_core::qp::QpProblem {
    // min 0.5*2*x^2 - 4x = (x-2)^2 - 4, bounds [-10, 10] → x*=2, obj=-4。
    let q = CscMatrix::from_triplets(&[0], &[0], &[2.0_f64], 1, 1).unwrap();
    otspot_core::qp::QpProblem::new_all_le(
        q,
        vec![-4.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![(-10.0, 10.0)],
    )
    .unwrap()
}

#[test]
fn real_solve_publishes_diagnostics_with_exclusive_buckets() {
    let prob = tiny_qp();
    let result = otspot_core::qp::solve_qp(&prob);
    assert_eq!(result.status, SolveStatus::Optimal);

    let diag = otspot_core::qp::diagnostics::last_qp_solve_diagnostics()
        .expect("solve_qp_with は caller スレッドへ診断を引き継ぐ");
    assert!(!diag.attempts.is_empty(), "実行 attempt が最低 1 件");
    let pub_id = diag.published_attempt().expect("Optimal 採用 attempt あり");

    let published = diag.attempts.iter().find(|a| a.id == pub_id).unwrap();
    assert_eq!(published.kind, AttemptKind::Ladder);
    assert!(published.termination.is_some());
    assert!(published.iterations > 0);

    // 回数: IPM 反復 > 0、factorize 試行は反復以上、materialize (元順序+置換
    // 済み導出総計)+Schur 構築は組み立て呼出回数 (時間 bucket とは別枠である
    // ことは型分離で保証)。
    let counts = diag.total_counts();
    assert!(counts.ipm_iterations > 0);
    assert!(counts.kkt_factorize_attempts >= counts.ipm_iterations);
    assert!(counts.materialize_total_calls() + counts.kkt_schur_builds >= 1);

    // 集計構造: 全階層で other < 0 (子の合計 > 親) が発生していないこと。
    // 二重計上を導入する revert はここで fail する sentinel。
    assert!(
        !diag.is_aggregation_inconsistent(),
        "集計不整合 attempt 数={}",
        diag.inconsistent_attempt_count()
    );
    assert!(diag.other_us() >= 0);

    // QP 経路で未設定だった TimingBreakdown presolve_us / solve_us が実測値で
    // 埋まっていること (plan §3: never-set fields の填充)。
    let tb = result
        .timing_breakdown
        .expect("QP solve は timing_breakdown を持つ");
    assert!(
        tb.solve_us > 0,
        "solve_us が QP 経路でまだ 0 のまま (未填充の回帰)"
    );
}

/// 非 QP-IPM 経路 (zero-Q → LP 転送) の solve は caller の LAST を None に
/// 戻さなければならない。sentinel: publish が None を握りつぶす実装では、
/// 長時間生存プロセス (qps_benchmark の 1 プロセス複数問題) で直前 QP の
/// 診断が次の問題の出力へ誤帰属する。
#[test]
fn non_qp_route_clears_stale_diagnostics() {
    let prob = tiny_qp();
    let r = otspot_core::qp::solve_qp(&prob);
    assert_eq!(r.status, SolveStatus::Optimal);
    assert!(
        otspot_core::qp::diagnostics::last_qp_solve_diagnostics().is_some(),
        "QP solve 直後は診断が読める"
    );

    // Q=0 → dispatch は LpForwardedFromQp へ流れ solve_ipm を通らない。
    let lp = otspot_core::qp::QpProblem::new_all_le(
        CscMatrix::new(1, 1),
        vec![-4.0],
        CscMatrix::new(0, 1),
        vec![],
        vec![(-10.0, 10.0)],
    )
    .unwrap();
    let rl = otspot_core::qp::solve_qp(&lp);
    assert_eq!(rl.status, SolveStatus::Optimal);
    assert!(
        otspot_core::qp::diagnostics::last_qp_solve_diagnostics().is_none(),
        "LP 転送 solve が前 QP の診断を残した (stale 誤帰属の回帰)"
    );
}

/// 2 スレッド同時 solve で診断が solve ごとに分離すること (TLS sink の
/// 混線 sentinel)。各スレッドは自分の結果としか突き合わさないため、
/// snapshot が別 solve のものなら published.iterations ≠ result.iterations
/// か objective/ステータスで割れる。期待値は手計算オラクル (A: x*=2,
/// obj=-4 / B: x*=(3,3,3), obj=-27)。
#[test]
fn concurrent_solves_keep_diagnostics_per_thread() {
    // B: min sum( x_i^2 - 6 x_i ) = |x-3|^2 - 27、bounds [-10,10]。
    let q_b = CscMatrix::from_triplets(&[0, 1, 2], &[0, 1, 2], &[2.0_f64, 2.0, 2.0], 3, 3).unwrap();
    let prob_b = otspot_core::qp::QpProblem::new_all_le(
        q_b,
        vec![-6.0, -6.0, -6.0],
        CscMatrix::new(0, 3),
        vec![],
        vec![(-10.0, 10.0); 3],
    )
    .unwrap();
    let prob_a = tiny_qp();

    std::thread::scope(|s| {
        // 読み取りは solve と同じスレッドで行う (dev-bin の逐次使用契約と同じ形)。
        let h1 = s.spawn(|| {
            let r = otspot_core::qp::solve_qp(&prob_a);
            let d = otspot_core::qp::diagnostics::last_qp_solve_diagnostics();
            (r, d.expect("thread A の solve は自分の TLS へ診断を残す"))
        });
        let h2 = s.spawn(|| {
            let r = otspot_core::qp::solve_qp(&prob_b);
            let d = otspot_core::qp::diagnostics::last_qp_solve_diagnostics();
            (r, d.expect("thread B の solve は自分の TLS へ診断を残す"))
        });
        let (ra, da) = h1.join().unwrap();
        let (rb, db) = h2.join().unwrap();

        let check = |r: &otspot_core::problem::SolverResult,
                     d: &otspot_core::qp::diagnostics::SolveDiagnostics,
                     obj_oracle: f64| {
            assert_eq!(r.status, SolveStatus::Optimal);
            assert!(
                (r.objective - obj_oracle).abs() <= 1e-6 * obj_oracle.abs().max(1.0),
                "objective oracle 不一致: got {} want {}",
                r.objective,
                obj_oracle
            );
            let pub_id = d.published_attempt().expect("採用 attempt あり");
            let pubd = d.attempts.iter().find(|a| a.id == pub_id).unwrap();
            assert_eq!(
                pubd.iterations, r.iterations as u64,
                "published attempt の iters が自分の結果と一致 (診断の混線なし)"
            );
            assert!(
                !d.is_aggregation_inconsistent(),
                "集計不整合 attempt 数={}",
                d.inconsistent_attempt_count()
            );
        };
        check(&ra, &da, -4.0);
        check(&rb, &db, -27.0);
        // 混線していれば id 列・収束回数が自分の solve と別の solve 由来で
        // 上記 assert のいずれかが崩れる。加えて attempt id が solve 内で
        // 0 始まり連番であること (共有進行状態の混入なし)。
        for d in [&da, &db] {
            for (i, a) in d.attempts.iter().enumerate() {
                assert_eq!(a.id, i as u64, "attempt id は solve 内 0 始まり連番");
            }
        }
    });
}

/// R2 sentinel: threads>1 の rayon プール境界を越えて IPM 診断が残ること。
/// 修正前は rayon-core の install が job を worker へ inject するため
/// ACTIVE_ATTEMPT が見えず、factor/probe/反復カウンタと IPM フェーズ時刻が
/// 全 0 (other へ全流出) になった — threads=2 の CVXQP3_L 実測で確認済み
/// (/tmp/opencode/rfix/pre/r2_experiment_pre_fix.txt)。時間値だけでなく
/// カウンタ・帰属・境界構造を検査する (review §3 検証)。
#[test]
fn threads_gt_1_preserves_ipm_counters_across_pool_boundary() {
    use otspot_core::options::SolverOptions;
    for threads in [2usize, 4] {
        let mut opts = SolverOptions::default();
        opts.threads = threads;
        let prob = dense_qp(40, 60);
        let result = otspot_core::qp::solve_qp_with(&prob, &opts);
        let diag = otspot_core::qp::diagnostics::last_qp_solve_diagnostics()
            .unwrap_or_else(|| panic!("threads={threads}: 診断が引き継がれない"));
        let counts = diag.total_counts();
        assert!(
            counts.ipm_iterations > 0,
            "threads={threads}: IPM 反復カウンタが消えた (sink が worker 側で空)"
        );
        assert!(
            counts.kkt_factorize_attempts >= counts.ipm_iterations,
            "threads={threads}: factorize 試行カウンタ欠落"
        );
        assert!(
            counts.kkt_health_probes >= 1,
            "threads={threads}: probe カウンタ欠落"
        );
        assert!(
            counts.materialize_total_calls() + counts.kkt_schur_builds >= counts.ipm_iterations,
            "threads={threads}: 組み立てカウンタ欠落"
        );
        let pub_id = diag.published_attempt().expect("採用 attempt あり");
        let published = diag.attempts.iter().find(|a| a.id == pub_id).unwrap();
        assert_eq!(published.iterations, result.iterations as u64);
        assert!(
            published.ipm.total_us > 0,
            "threads={threads}: IPM 時刻が other へ流出している (total_us=0)"
        );
        assert!(published.ipm.total_us >= published.ipm.exclusive_children_sum_us());
        assert!(
            !diag.is_aggregation_inconsistent(),
            "threads={threads}: 集計不整合 attempt 数={}",
            diag.inconsistent_attempt_count()
        );
    }
}

/// R2 混入 sentinel: threads>1 で同一プール worker を共有する同時 solve でも
/// 試行ごとの sink が solve を跨がない (worker 再使用時の残留 TLS が次の
/// solve に拾われると iterations/attempt id が自分の結果と不一致になる)。
#[test]
fn threads_2_concurrent_solves_keep_diagnostics_isolated() {
    fn solve_diag(
        prob: &otspot_core::qp::QpProblem,
    ) -> (
        otspot_core::problem::SolverResult,
        otspot_core::qp::diagnostics::SolveDiagnostics,
    ) {
        use otspot_core::options::SolverOptions;
        let mut opts = SolverOptions::default();
        opts.threads = 2;
        let r = otspot_core::qp::solve_qp_with(prob, &opts);
        let d = otspot_core::qp::diagnostics::last_qp_solve_diagnostics()
            .expect("自分の solve の診断が自分の TLS に残る");
        (r, d)
    }
    let prob_a = dense_qp(40, 60);
    let prob_b = dense_qp(30, 50);
    std::thread::scope(|s| {
        let h1 = s.spawn(|| solve_diag(&prob_a));
        let h2 = s.spawn(|| solve_diag(&prob_b));
        for (r, d) in [h1.join().unwrap(), h2.join().unwrap()] {
            let pub_id = d.published_attempt().expect("採用 attempt あり");
            let published = d.attempts.iter().find(|a| a.id == pub_id).unwrap();
            assert_eq!(
                published.iterations, r.iterations as u64,
                "published attempt が自分の結果と一致 (worker 再使用混入なし)"
            );
            for (i, a) in d.attempts.iter().enumerate() {
                assert_eq!(a.id, i as u64, "attempt id は solve 内 0 始まり連番");
            }
            assert!(!d.is_aggregation_inconsistent());
        }
    });
}

/// 合成密 QP: min ||x-t||² - const (t=1) に密な Le 制約 (x=1 が内部解)。
/// KKT が (n+m) の密寄り系になり threads>1 の IPM 経路が実際に走る。
fn dense_qp(n: usize, m: usize) -> otspot_core::qp::QpProblem {
    let q_entries: Vec<(usize, usize, f64)> = (0..n).map(|j| (j, j, 2.0)).collect();
    let q = CscMatrix::from_triplets(
        &q_entries.iter().map(|e| e.0).collect::<Vec<_>>(),
        &q_entries.iter().map(|e| e.1).collect::<Vec<_>>(),
        &q_entries.iter().map(|e| e.2).collect::<Vec<_>>(),
        n,
        n,
    )
    .unwrap();
    let c = vec![-2.0; n];
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut a_row_sum = vec![0.0_f64; m];
    for (i, row_sum) in a_row_sum.iter_mut().enumerate() {
        for j in 0..n {
            let v = (((i * 37 + j * 17 + 5) % 13) as f64) - 6.0;
            rows.push(i);
            cols.push(j);
            vals.push(v);
            *row_sum += v;
        }
    }
    let a = CscMatrix::from_triplets(&rows, &cols, &vals, m, n).unwrap();
    // x=1 を厳密に内部通す b (スラック 1.0) → 最適解 x*=1、収束保証。
    let b = a_row_sum.iter().map(|s| s + 1.0).collect::<Vec<_>>();
    otspot_core::qp::QpProblem::new_all_le(q, c, a, b, vec![(-10.0, 10.0); n]).unwrap()
}
