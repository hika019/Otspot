//! ベンチマーク共通ユーティリティ
//!
//! qps_benchmark / bench_qplib 両バイナリで共有するCSV読み込み・正解値照合ロジック。

use crate::problem::{ConstraintType, SolveStatus, SolverResult};
use crate::qp::QpProblem;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// QP 元空間 KKT 残差 (stationarity / primal_inf / comp_ineq / comp_bound) の
/// 成分相対化 max。`diag_nonconvex_kkt.rs` の compute_kkt と同一規約:
///   ∇_x L = Qx + c + Aᵀy − lb_du + ub_du
///   primal: max(0, ax−b) for Le, max(0, b−ax) for Ge, |ax−b| for Eq + bounds
///   comp_ineq: |yᵢ·slackᵢ| / (1 + |yᵢ|·(|axᵢ|+|bᵢ|))
///   comp_bound: |duⱼ·(x−bnd)| / (1 + |duⱼ|·(|xⱼ|+|bnd|))
/// 解形状不一致 (x.len() != n) なら INFINITY を返す。
pub fn compute_qp_kkt_max(prob: &QpProblem, x: &[f64], y: &[f64], bd: &[f64]) -> f64 {
    let n = prob.num_vars;
    if x.len() != n {
        return f64::INFINITY;
    }

    let qx = match prob.q.mat_vec_mul(x) {
        Ok(v) => v,
        Err(_) => return f64::INFINITY,
    };
    let aty: Vec<f64> = if prob.a.nrows > 0 && !y.is_empty() {
        match prob.a.transpose().mat_vec_mul(y) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        }
    } else {
        vec![0.0; n]
    };

    let mut bound_contrib = vec![0.0_f64; n];
    if !bd.is_empty() {
        let mut idx = 0usize;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && idx < bd.len() {
                bound_contrib[j] -= bd[idx];
                idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && idx < bd.len() {
                bound_contrib[j] += bd[idx];
                idx += 1;
            }
        }
    }

    let mut max_resid = 0.0_f64;
    for j in 0..n {
        let r = qx[j] + aty[j] + bound_contrib[j] + prob.c[j];
        let scale = 1.0 + qx[j].abs() + aty[j].abs() + bound_contrib[j].abs() + prob.c[j].abs();
        max_resid = max_resid.max(r.abs() / scale);
    }

    let ax = if prob.a.nrows > 0 {
        match prob.a.mat_vec_mul(x) {
            Ok(v) => v,
            Err(_) => return f64::INFINITY,
        }
    } else {
        Vec::new()
    };
    #[allow(unreachable_patterns)] // ConstraintType is #[non_exhaustive]; wildcard guards future variants.
    for (i, ct) in prob.constraint_types.iter().enumerate() {
        let violation = match ct {
            ConstraintType::Le => (ax[i] - prob.b[i]).max(0.0),
            ConstraintType::Ge => (prob.b[i] - ax[i]).max(0.0),
            ConstraintType::Eq => (ax[i] - prob.b[i]).abs(),
            _ => continue,
        };
        let scale = 1.0 + ax[i].abs() + prob.b[i].abs();
        max_resid = max_resid.max(violation / scale);
    }
    for (j, &(lb, ub)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() {
            let v = (lb - x[j]).max(0.0);
            max_resid = max_resid.max(v / (1.0 + x[j].abs() + lb.abs()));
        }
        if ub.is_finite() {
            let v = (x[j] - ub).max(0.0);
            max_resid = max_resid.max(v / (1.0 + x[j].abs() + ub.abs()));
        }
    }

    if !ax.is_empty() && !y.is_empty() {
        #[allow(unreachable_patterns)] // ConstraintType is #[non_exhaustive].
        for (i, ct) in prob.constraint_types.iter().enumerate() {
            let slack = match ct {
                ConstraintType::Eq => continue,
                ConstraintType::Le => prob.b[i] - ax[i],
                ConstraintType::Ge => ax[i] - prob.b[i],
                _ => continue,
            };
            let prod = (y[i] * slack).abs();
            let scale = 1.0 + y[i].abs() * (ax[i].abs() + prob.b[i].abs());
            max_resid = max_resid.max(prod / scale);
        }
    }
    if !bd.is_empty() {
        let mut idx = 0usize;
        for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
            if lb.is_finite() && idx < bd.len() {
                let slack = x[j] - lb;
                let prod = (bd[idx] * slack).abs();
                let scale = 1.0 + bd[idx].abs() * (x[j].abs() + lb.abs());
                max_resid = max_resid.max(prod / scale);
                idx += 1;
            }
        }
        for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
            if ub.is_finite() && idx < bd.len() {
                let slack = ub - x[j];
                let prod = (bd[idx] * slack).abs();
                let scale = 1.0 + bd[idx].abs() * (x[j].abs() + ub.abs());
                max_resid = max_resid.max(prod / scale);
                idx += 1;
            }
        }
    }
    max_resid
}

/// `|obj − global_ref| / (1 + |global_ref|)`。両者の finite を要求。
pub fn compute_gap_to_global(obj: f64, global_ref: f64) -> Option<f64> {
    if !obj.is_finite() || !global_ref.is_finite() {
        return None;
    }
    Some((obj - global_ref).abs() / (1.0 + global_ref.abs()))
}

/// bench harness 種別ごとの promotion policy.
///
/// qps_benchmark は obj 有限性チェックなし、bench_qplib は obj 有限性も要求する
/// (qplib は baseline obj 照合経路に流すため obj が non-finite だと意味がない)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchPromotionPolicy {
    QpsBenchmark,
    BenchQplib,
}

/// SuboptimalSolution / LocallyOptimal で有効な全長解を持つ result を Optimal に格上げする.
///
/// Timeout は意図的に格上げ対象外: Primal 半 deadline incumbent を silent に Optimal 化すると
/// 後段の品質判定 (pfeas/dfeas/obj) に流れて PFEAS_FAIL 表示となり、真因 (deadline 切れ)
/// が観測者に隠れる (task #46 観測 / task #52 真因対処)。Timeout は honest に Timeout 報告。
pub fn apply_bench_status_promotion(
    result: SolverResult,
    num_vars: usize,
    policy: BenchPromotionPolicy,
) -> SolverResult {
    let eligible_status = matches!(
        result.status,
        SolveStatus::SuboptimalSolution | SolveStatus::LocallyOptimal
    );
    let has_full_solution =
        !result.solution.is_empty() && result.solution.len() == num_vars;
    let obj_ok = match policy {
        BenchPromotionPolicy::QpsBenchmark => true,
        BenchPromotionPolicy::BenchQplib => result.objective.is_finite(),
    };
    if eligible_status && has_full_solution && obj_ok {
        SolverResult { status: SolveStatus::Optimal, ..result }
    } else {
        result
    }
}

/// 正解値照合結果
pub enum ObjCheckResult {
    Ok { rel_err: f64 },
    Mismatch { rel_err: f64 },
    NoRef,
}

/// CSVに記録された問題の期待ステータス
///
/// optimal_obj 列が INFEASIBLE / UNBOUNDED の文字列の場合に使用する。
/// 数値の場合は Optimal (有限最適値あり)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedStatus {
    /// 有限最適値 (通常問題)
    Optimal,
    /// 主問題が実行不可能
    Infeasible,
    /// 主問題が非有界
    Unbounded,
}

/// CSVから問題の期待ステータス一覧を読み込む
///
/// 通常の float 行は Optimal、"INFEASIBLE"/"UNBOUNDED" 文字列行を
/// それぞれ Infeasible / Unbounded として返す。
pub fn load_expected_statuses(csv_path: &Path) -> HashMap<String, ExpectedStatus> {
    let mut map = HashMap::new();
    let content = match std::fs::read_to_string(csv_path) {
        Ok(c) => c,
        Err(_) => return map,
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("problem_name") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        let name = parts[0].trim().to_string();
        let status_str = parts[1].trim();
        let status = match status_str.to_uppercase().as_str() {
            "INFEASIBLE" => ExpectedStatus::Infeasible,
            "UNBOUNDED" => ExpectedStatus::Unbounded,
            _ => {
                // 数値なら Optimal (parse 失敗は skip)
                if status_str.parse::<f64>().is_ok() {
                    ExpectedStatus::Optimal
                } else {
                    continue;
                }
            }
        };
        map.insert(name, status);
    }
    map
}

/// data_dir名とオーバーライドパスからCSVパスを決定する
pub fn detect_csv_path(data_dir: &str, override_path: Option<&str>, root: &Path) -> PathBuf {
    if let Some(p) = override_path {
        return PathBuf::from(p);
    }
    let data_lower = data_dir.to_lowercase();
    let csv_name = if data_lower.contains("maros") {
        "maros_meszaros.csv"
    } else if data_lower.contains("qp_unbounded") || data_lower.contains("qp-unbounded") {
        "qp_unbounded.csv"
    } else if data_lower.contains("qp_infeasible") || data_lower.contains("qp-infeasible") {
        "qp_infeasible.csv"
    } else if data_lower.contains("qplib_nonconvex_official")
        || data_lower.contains("qplib-nonconvex-official")
    {
        "qplib_nonconvex_official.csv"
    } else if data_lower.contains("qplib") {
        "qplib.csv"
    } else if data_lower.contains("osqp_bench") || data_lower.contains("osqp-bench") {
        "osqp_bench.csv"
    } else if data_lower.contains("mpc_qp") || data_lower.contains("mpc-qp") {
        "mpc_qp.csv"
    } else if data_lower.contains("lp_problems_infeas") || data_lower.contains("lp-problems-infeas") {
        "netlib_lp_infeas.csv"
    } else if data_lower.contains("lp_problems_extra") || data_lower.contains("lp-problems-extra") {
        "netlib_lp_extra.csv"
    } else {
        "netlib_lp.csv"
    };
    let candidate = root.join("data/baseline_objectives").join(csv_name);
    if candidate.exists() {
        return candidate;
    }
    // フォールバック: カレントディレクトリ基準
    PathBuf::from("data/baseline_objectives").join(csv_name)
}

/// 正解値CSVを読み込む
pub fn load_baseline_objectives(csv_path: &Path) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    let content = match std::fs::read_to_string(csv_path) {
        Ok(c) => c,
        Err(_) => return map,
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("problem_name") {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            if let Ok(val) = parts[1].trim().parse::<f64>() {
                map.insert(parts[0].trim().to_string(), val);
            }
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp_csv(content: &str) -> std::path::PathBuf {
        let mut tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        tmp.write_all(content.as_bytes()).unwrap();
        let path = tmp.path().to_path_buf();
        // keep alive by leaking (test only)
        Box::leak(Box::new(tmp));
        path
    }

    #[test]
    fn test_load_expected_statuses_infeasible() {
        // INFEASIBLE / UNBOUNDED エントリが正しく読み込まれるか
        let csv = "problem_name,optimal_obj,source\n\
            galenet,INFEASIBLE,https://example.com\n\
            klein1,INFEASIBLE,https://example.com\n\
            unbnd_toy,UNBOUNDED,https://example.com\n\
            feasible_toy,1234.5,https://example.com\n\
            noref_toy,no_ref,https://example.com\n";
        let path = write_tmp_csv(csv);
        let map = load_expected_statuses(&path);

        assert_eq!(map.get("galenet"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("klein1"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("unbnd_toy"), Some(&ExpectedStatus::Unbounded));
        assert_eq!(map.get("feasible_toy"), Some(&ExpectedStatus::Optimal));
        // no_ref は parse 失敗でスキップ → None
        assert_eq!(map.get("noref_toy"), None);
    }

    #[test]
    fn test_load_expected_statuses_case_insensitive() {
        // 大文字・小文字どちらでも認識
        let csv = "problem_name,optimal_obj\n\
            p1,infeasible\n\
            p2,INFEASIBLE\n\
            p3,Infeasible\n\
            p4,unbounded\n\
            p5,UNBOUNDED\n";
        let path = write_tmp_csv(csv);
        let map = load_expected_statuses(&path);

        assert_eq!(map.get("p1"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p2"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p3"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p4"), Some(&ExpectedStatus::Unbounded));
        assert_eq!(map.get("p5"), Some(&ExpectedStatus::Unbounded));
    }

    #[test]
    fn test_load_expected_statuses_skips_comments_and_header() {
        let csv = "# comment line\n\
            problem_name,optimal_obj\n\
            p1,INFEASIBLE\n\
            # another comment\n\
            p2,1.5\n";
        let path = write_tmp_csv(csv);
        let map = load_expected_statuses(&path);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("p1"), Some(&ExpectedStatus::Infeasible));
        assert_eq!(map.get("p2"), Some(&ExpectedStatus::Optimal));
    }

    #[test]
    fn test_detect_csv_path_infeas() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems_infeas", None, root);
        assert!(p.to_string_lossy().contains("netlib_lp_infeas.csv"),
            "Expected netlib_lp_infeas.csv, got {:?}", p);
    }

    #[test]
    fn test_detect_csv_path_extra() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems_extra", None, root);
        assert!(p.to_string_lossy().contains("netlib_lp_extra.csv"),
            "Expected netlib_lp_extra.csv, got {:?}", p);
    }

    #[test]
    fn test_detect_csv_path_default_netlib() {
        let root = std::path::Path::new("/solver");
        let p = detect_csv_path("/data/lp_problems", None, root);
        assert!(p.to_string_lossy().contains("netlib_lp.csv"),
            "Expected netlib_lp.csv, got {:?}", p);
    }
}

/// 正解値と照合する（1%閾値）
pub fn check_baseline_objective(
    problem_name: &str,
    solver_obj: f64,
    known: &HashMap<String, f64>,
    eps_obj: f64,
) -> ObjCheckResult {
    match known.get(problem_name) {
        Some(&known_obj) => {
            // NaN / Inf の solver_obj は無条件で Mismatch 扱いにする (rel_err 比較が
            // NaN > eps = false で Ok に倒れて bug を見落とす false-positive を防ぐ)。
            if !solver_obj.is_finite() {
                return ObjCheckResult::Mismatch { rel_err: f64::INFINITY };
            }
            let denom = 1.0_f64.max(known_obj.abs());
            let rel_err = (solver_obj - known_obj).abs() / denom;
            if rel_err > eps_obj {
                ObjCheckResult::Mismatch { rel_err }
            } else {
                ObjCheckResult::Ok { rel_err }
            }
        }
        None => ObjCheckResult::NoRef,
    }
}
