//! プロパティベーステスト（ファジング）
//!
//! proptestクレートを用いたランダム生成LPによるソルバー堅牢性検証テストスイート。
//!
//! 以下のプロパティを検証する:
//! 1. 小規模ランダムLPでsolve()がパニックしない
//! 2. 目的関数ゼロのLPは常にOptimalを返す
//! 3. 全係数正・全変数非負のLPはInfeasibleにならない

use otspot::problem::{LpProblem, SolveStatus};
use otspot::solve;
use otspot::sparse::CscMatrix;
use proptest::prelude::*;

/// 対角制約行列を生成するヘルパー
///
/// k = min(nrows, ncols, diag_vals.len()) として、
/// a[i,i] = diag_vals[i] (0 <= i < k) の疎行列を返す。
fn make_diagonal_csc(diag_vals: &[f64], nrows: usize, ncols: usize) -> CscMatrix {
    let k = nrows.min(ncols).min(diag_vals.len());
    if k == 0 {
        return CscMatrix::new(nrows, ncols);
    }
    let rows: Vec<usize> = (0..k).collect();
    let cols: Vec<usize> = (0..k).collect();
    let vals: Vec<f64> = diag_vals[..k].to_vec();
    CscMatrix::from_triplets(&rows, &cols, &vals, nrows, ncols).unwrap()
}

proptest! {
    /// テスト1: 小規模ランダムLPでパニックが起きないことを検証
    ///
    /// 変数2〜5個、制約1〜4個の対角制約行列LPをランダム生成し、
    /// solve() がパニックせず有効なステータスを返すことを確認する。
    /// b の値は負にもなりえるため、Infeasibleになる場合もある（それ自体は正常）。
    #[test]
    fn prop_small_lp_no_panic(
        c in prop::collection::vec(-10.0f64..10.0f64, 2usize..=5usize),
        diag in prop::collection::vec(0.1f64..5.0f64, 1usize..=4usize),
        b in prop::collection::vec(-5.0f64..10.0f64, 1usize..=4usize),
    ) {
        let n = c.len();
        let m = diag.len().min(b.len());
        let b_m: Vec<f64> = b[..m].to_vec();
        let a = make_diagonal_csc(&diag, m, n);

        let prob = LpProblem::new(c, a, b_m).unwrap();
        let result = solve(&prob);
        prop_assert!(
            matches!(
                result.status,
                SolveStatus::Optimal | SolveStatus::Infeasible | SolveStatus::Unbounded |
                SolveStatus::Timeout | SolveStatus::NumericalError | SolveStatus::SuboptimalSolution
            ),
            "solve() returned unrecognized status: {:?}", result.status
        );
    }

    /// テスト2: 目的関数がゼロのLPは常にOptimalを返す
    ///
    /// c = 0 のとき、x = 0 が常に実行可能かつ最適（目的値 = 0）。
    /// 制約: x_i <= b_i (b_i > 0)、変数下限 = 0。
    #[test]
    fn prop_zero_objective_always_optimal(
        b in prop::collection::vec(0.1f64..10.0f64, 2usize..=5usize),
    ) {
        let m = b.len();
        let n = m;
        // 単位対角行列制約: x_i <= b_i
        let a = make_diagonal_csc(&vec![1.0; m], m, n);
        let c = vec![0.0; n];

        let prob = LpProblem::new(c, a, b).unwrap();
        let result = solve(&prob);

        prop_assert_eq!(
            result.status,
            SolveStatus::Optimal,
            "零目的関数LPは常にOptimal"
        );
        prop_assert!(
            result.objective.abs() < 1e-8,
            "零目的関数LPの最適値は0であるべき: got {}",
            result.objective
        );
    }

    /// テスト3: 全係数正・全変数非負のLPはInfeasibleにならない
    ///
    /// c > 0, 対角A > 0, b > 0 のとき、x = 0 は常に実行可能（A*0 = 0 <= b）。
    /// よってsolve()はInfeasibleを返さない。
    #[test]
    fn prop_nonneg_lp_not_infeasible(
        c in prop::collection::vec(0.01f64..10.0f64, 2usize..=5usize),
        b in prop::collection::vec(0.1f64..10.0f64, 2usize..=5usize),
    ) {
        let n = c.len();
        let m = b.len();
        // 対角制約行列（全係数正、x_i <= b_i に相当）
        let a = make_diagonal_csc(&vec![1.0; m.min(n)], m, n);

        let prob = LpProblem::new(c, a, b).unwrap();
        let result = solve(&prob);

        // c > 0, A >= 0, b > 0: x=0 は常に実行可能 → Infeasibleにはならない
        prop_assert_ne!(
            result.status,
            SolveStatus::Infeasible,
            "全係数正・全変数非負のLPはInfeasibleにならない"
        );
    }
}
