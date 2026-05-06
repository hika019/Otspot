//! ベンチマーク共通ユーティリティ
//!
//! qps_benchmark / bench_qplib 両バイナリで共有するCSV読み込み・正解値照合ロジック。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// 正解値照合結果
pub enum ObjCheckResult {
    Ok { rel_err: f64 },
    Mismatch { rel_err: f64 },
    NoRef,
}

/// data_dir名とオーバーライドパスからCSVパスを決定する
pub fn detect_csv_path(data_dir: &str, override_path: Option<&str>, root: &Path) -> PathBuf {
    if let Some(p) = override_path {
        return PathBuf::from(p);
    }
    let data_lower = data_dir.to_lowercase();
    let csv_name = if data_lower.contains("maros") {
        "maros_meszaros.csv"
    } else if data_lower.contains("qplib") {
        "qplib.csv"
    } else if data_lower.contains("osqp_bench") || data_lower.contains("osqp-bench") {
        "osqp_bench.csv"
    } else if data_lower.contains("mpc_qp") || data_lower.contains("mpc-qp") {
        "mpc_qp.csv"
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
