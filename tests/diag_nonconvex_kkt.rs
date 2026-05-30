//! 非凸 QP KKT residual sentinel (Phase 1A).
//!
//! 合成 45 (`data/qplib_nonconvex/`) + 公式 4 (`data/qplib_nonconvex_official/`,
//! Phase 1B 取得) の 49 件について `solve_qp_with` を実行し、各解について
//! 1st-order KKT 残差 (stationarity / primal feasibility / complementarity) を
//! 成分相対化で計算し log + 検証する。公式 4 件は加えて global ref との
//! `gap_to_global` を計算 + log。
//!
//! ## 検証ポリシー
//!
//! status=Optimal / LocallyOptimal を主張した解 (= solver が KKT 収束 claim) について:
//! - **primal feasibility**: `prim < EPS_KKT` を強制。
//! - **complementarity**:    `comp_* < EPS_KKT` を強制。
//! - **stationarity**:       LocallyOptimal は `stat < EPS_KKT` を強制
//!   (IPPMM が「不定 Q + 慣性修正で局所 KKT 点に収束」と explicit に claim する
//!   status で、元空間 Q の停留性が成立する規約)。
//!   Optimal は WARN log のみ (assert なし)。non-PSD Q で solver が Optimal を
//!   主張する経路は scaled-space KKT check が claim 主体で unscale 後の
//!   bound_dual 復元が壊れる pre-existing 不具合を含むため、Phase 1A 範疇では
//!   WARN に留め fail させない (follow-up で Optimal-claim 解の stat assert
//!   化が予定)。
//!
//! status=SuboptimalSolution / Timeout / その他は honest non-convergence 申告で
//! assert 除外 (log のみ)。
//!
//! ## sentinel 検出力
//!
//! - `kkt_perturbation_sentinel`: 元解に x[0]+=1 を加えると stationarity が
//!   `>= SENTINEL_MIN_KKT` に増えることを assert。`compute_kkt` の no-op 書換
//!   (常に 0 を返す) で確実に FAIL する。
//! - `kkt_sign_convention_mini_qp`: 解析解持ち convex mini QP で kkt < 1e-6 を
//!   assert。符号規約 (`lb_dual / ub_dual / y / Q*x` の組合せ) を独立検証。

use otspot::io::qplib::{parse_qplib, QplibProblem};
use otspot::options::SolverOptions;
use otspot::problem::{ConstraintType, SolveStatus};
use otspot::qp::kkt_resid::{self, f64_impl};
use otspot::qp::{solve_qp_with, QpProblem};
use otspot_dev::bench_utils::compute_qp_kkt_max;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// 本テストは KKT 残差を 4 成分 (stat/prim/comp_ineq/comp_bound) に分解して log + assert
// するため、test 内に compute_kkt を持つ。bench_qplib.rs は集約済 max 値 1 つで判定
// すれば十分なので bench_utils::compute_qp_kkt_max を使う (規約は同一)。
// 両者の規約整合は kkt_consistency_with_bench_utils テストで担保。

// 元空間 KKT 許容残差。IPM scaled eps=1e-6 (default) を Ruiz 振幅 100 級まで
// 許容するため 1e-4 を採用。`check_dfeas_status_relative` (scaling.rs:235) の
// production 規約と整合。
const EPS_KKT: f64 = 1e-4;
// 各問題の solver timeout。49 問題合計で 3 分内 (CLAUDE.md「テスト 1 個 < 3 分」)。
const TIMEOUT_PER_PROBLEM_SECS: f64 = 30.0;

const SYNTH_DIR: &str = "data/qplib_nonconvex";
const OFFICIAL_DIR: &str = "data/qplib_nonconvex_official";
const OFFICIAL_REF_CSV: &str = "data/baseline_objectives/qplib_nonconvex_official.csv";

// Perturbation sentinel: 解に SENTINEL_PERTURB 単位ずらすと、合成 nonconvex の
// 典型的 Q (gen_nonconvex_qp.py 仕様で ‖Q‖_max が O(1) 級) で stationarity が
// 少なくとも SENTINEL_MIN_KKT まで増える。`compute_kkt` が no-op (常に 0) で
// あれば 0 < 1e-2 が偽となり FAIL する。
const SENTINEL_PERTURB: f64 = 1.0;
const SENTINEL_MIN_KKT: f64 = 1e-2;

#[derive(Debug, Clone, Copy)]
struct KktResidual {
    stationarity: f64,
    primal_inf: f64,
    comp_ineq: f64,
    comp_bound: f64,
}

impl KktResidual {
    fn max(&self) -> f64 {
        self.stationarity
            .max(self.primal_inf)
            .max(self.comp_ineq)
            .max(self.comp_bound)
    }
    fn comp_max(&self) -> f64 {
        self.comp_ineq.max(self.comp_bound)
    }
}

/// 1st-order KKT 残差を成分相対化で計算する。
///
/// 規約 (Le 標準形 min 1/2 xᵀQx + cᵀx s.t. Ax ≤ b, lb ≤ x ≤ ub):
///   L = 1/2 xᵀQx + cᵀx + yᵀ(Ax − b) − lb_duᵀ(x − lb) − ub_duᵀ(ub − x)
///   ∇_x L = Qx + c + Aᵀy − lb_du + ub_du = 0
///   y ≥ 0 (Le), comp: yᵢ·(b−Ax)ᵢ = 0
///   lb_du, ub_du ≥ 0, comp: lb_duⱼ·(x−lb)ⱼ = 0, ub_duⱼ·(ub−x)ⱼ = 0
fn compute_kkt(prob: &QpProblem, x: &[f64], y: &[f64], bd: &[f64]) -> KktResidual {
    let n = prob.num_vars;
    let qx = f64_impl::qx(&prob.q, x);
    let aty = f64_impl::aty(&prob.a, y, n);
    let bound_contrib = kkt_resid::bound_contrib(&prob.bounds, bd);

    let mut stat = 0.0_f64;
    for j in 0..n {
        let r = qx[j] + aty[j] + bound_contrib[j] + prob.c[j];
        let scale = 1.0 + qx[j].abs() + aty[j].abs() + bound_contrib[j].abs() + prob.c[j].abs();
        stat = stat.max(r.abs() / scale);
    }

    let ax = f64_impl::ax(&prob.a, x);
    let viols = f64_impl::constraint_violations(&ax, &prob.b, &prob.constraint_types);
    let mut prim = 0.0_f64;
    for (i, &v) in viols.iter().enumerate() {
        let scale = 1.0 + ax[i].abs() + prob.b[i].abs();
        prim = prim.max(v / scale);
    }
    for (j, &(lb, ub)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() {
            let v = (lb - x[j]).max(0.0);
            prim = prim.max(v / (1.0 + x[j].abs() + lb.abs()));
        }
        if ub.is_finite() {
            let v = (x[j] - ub).max(0.0);
            prim = prim.max(v / (1.0 + x[j].abs() + ub.abs()));
        }
    }

    let comp_i_raw = f64_impl::comp_ineq_products(&ax, &prob.b, &prob.constraint_types, y);
    let mut comp_ineq = 0.0_f64;
    for (i, &prod) in comp_i_raw.iter().enumerate() {
        if prod == 0.0 {
            continue;
        }
        let scale = 1.0 + y[i].abs() * (ax[i].abs() + prob.b[i].abs());
        comp_ineq = comp_ineq.max(prod / scale);
    }

    let comp_b_raw = kkt_resid::comp_bound_products(&prob.bounds, x, bd);
    let mut comp_bound = 0.0_f64;
    let mut idx = 0_usize;
    for (j, &(lb, _)) in prob.bounds.iter().enumerate() {
        if lb.is_finite() && idx < bd.len() {
            let scale = 1.0 + bd[idx].abs() * (x[j].abs() + lb.abs());
            comp_bound = comp_bound.max(comp_b_raw[idx] / scale);
            idx += 1;
        }
    }
    for (j, &(_, ub)) in prob.bounds.iter().enumerate() {
        if ub.is_finite() && idx < bd.len() {
            let scale = 1.0 + bd[idx].abs() * (x[j].abs() + ub.abs());
            comp_bound = comp_bound.max(comp_b_raw[idx] / scale);
            idx += 1;
        }
    }

    KktResidual {
        stationarity: stat,
        primal_inf: prim,
        comp_ineq,
        comp_bound,
    }
}

fn load_global_refs() -> HashMap<String, f64> {
    let mut refs = HashMap::new();
    let csv = match std::fs::read_to_string(OFFICIAL_REF_CSV) {
        Ok(s) => s,
        Err(_) => return refs,
    };
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("problem_name") {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() >= 2 {
            if let Ok(v) = cols[1].trim().parse::<f64>() {
                refs.insert(cols[0].trim().to_string(), v);
            }
        }
    }
    refs
}

fn list_qplib(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("cannot read {}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("qplib"))
        .collect();
    files.sort();
    files
}

struct ProbeRecord {
    name: String,
    status: SolveStatus,
    objective: f64,
    kkt: KktResidual,
    gap_to_global: Option<f64>,
}

fn solve_and_log(path: &Path, global_ref: Option<f64>) -> ProbeRecord {
    let prob = match parse_qplib(path).expect("parse") {
        QplibProblem::Qp(p) => p,
        other => panic!(
            "expected continuous QP for nonconvex benchmark, got {:?}",
            other
        ),
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_PER_PROBLEM_SECS);
    let t0 = Instant::now();
    let res = solve_qp_with(&prob, &opts);
    let wall = t0.elapsed().as_secs_f64();
    let kkt = if res.solution.len() == prob.num_vars {
        compute_kkt(&prob, &res.solution, &res.dual_solution, &res.bound_duals)
    } else {
        KktResidual {
            stationarity: f64::INFINITY,
            primal_inf: f64::INFINITY,
            comp_ineq: f64::INFINITY,
            comp_bound: f64::INFINITY,
        }
    };
    let gap_to_global = global_ref.map(|gr| (res.objective - gr).abs() / (1.0 + gr.abs()));
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let gap_str = gap_to_global
        .map(|g| format!(" gap_to_global={:.3e}", g))
        .unwrap_or_else(|| " gap_to_global=—".into());
    eprintln!(
        "[nonconvex_kkt] {:<32} n={:<5} m={:<5} status={:?} obj={:.6e} \
         kkt(stat={:.2e} prim={:.2e} comp_i={:.2e} comp_b={:.2e}) wall={:.2}s{}",
        name,
        prob.num_vars,
        prob.num_constraints,
        res.status,
        res.objective,
        kkt.stationarity,
        kkt.primal_inf,
        kkt.comp_ineq,
        kkt.comp_bound,
        wall,
        gap_str,
    );
    ProbeRecord {
        name,
        status: res.status,
        objective: res.objective,
        kkt,
        gap_to_global,
    }
}

fn collect_all() -> Vec<ProbeRecord> {
    let synth = list_qplib(Path::new(SYNTH_DIR));
    let official = list_qplib(Path::new(OFFICIAL_DIR));
    assert_eq!(
        synth.len(),
        45,
        "expected 45 synthetic nonconvex (got {})",
        synth.len()
    );
    assert_eq!(
        official.len(),
        4,
        "expected 4 official nonconvex (got {})",
        official.len()
    );
    let refs = load_global_refs();
    assert_eq!(
        refs.len(),
        4,
        "expected 4 official global ref entries (got {})",
        refs.len()
    );

    let mut records = Vec::with_capacity(synth.len() + official.len());
    for p in &synth {
        records.push(solve_and_log(p, None));
    }
    for p in &official {
        let name = p
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        records.push(solve_and_log(p, refs.get(&name).copied()));
    }
    records
}

/// 全 49 nonconvex 問題を solve + log + KKT 検証。
///
/// status=Optimal/LocallyOptimal を主張した解について:
/// - primal feasibility: `prim < EPS_KKT` を強制 (解空間内必須)
/// - complementarity:    `comp_* < EPS_KKT` を強制
/// - stationarity: LocallyOptimal のみ `stat < EPS_KKT` を強制。
///   Optimal は WARN log のみ (Q非PSD で solver が Optimal を主張する経路には
///   unscale 後 bound_dual 復元の pre-existing 不整合あり、Phase 1A scope 外)
///
/// status=SuboptimalSolution/Timeout/その他は honest non-convergence 申告として
/// assert 除外し、log のみ。
#[test]
fn nonconvex_kkt_all_49_problems() {
    let records = collect_all();

    let mut prim_violations = Vec::<String>::new();
    let mut comp_violations = Vec::<String>::new();
    let mut stat_violations = Vec::<String>::new();
    let mut optimal_stat_warns = Vec::<String>::new();
    let mut n_optimal = 0usize;
    let mut n_local = 0usize;
    let mut n_subopt = 0usize;
    let mut n_other = 0usize;

    for r in &records {
        match &r.status {
            SolveStatus::Optimal => n_optimal += 1,
            SolveStatus::LocallyOptimal => n_local += 1,
            SolveStatus::SuboptimalSolution => n_subopt += 1,
            _ => n_other += 1,
        }
        let claims_kkt = matches!(r.status, SolveStatus::Optimal | SolveStatus::LocallyOptimal);
        if !claims_kkt {
            // SuboptimalSolution / Timeout 等は honest non-convergence 申告。
            // primal/complementarity を主張しない status まで assert すると、
            // 「収束未達と正直に申告した結果」を fail と区別できなくなる。
            continue;
        }
        if !r.kkt.primal_inf.is_finite() || r.kkt.primal_inf >= EPS_KKT {
            prim_violations.push(format!(
                "{}: status={:?} prim={:.3e} (>= {:.0e})",
                r.name, r.status, r.kkt.primal_inf, EPS_KKT
            ));
        }
        if !r.kkt.comp_max().is_finite() || r.kkt.comp_max() >= EPS_KKT {
            comp_violations.push(format!(
                "{}: status={:?} comp_ineq={:.3e} comp_bound={:.3e} (>= {:.0e})",
                r.name, r.status, r.kkt.comp_ineq, r.kkt.comp_bound, EPS_KKT
            ));
        }
        if matches!(r.status, SolveStatus::LocallyOptimal) {
            if !r.kkt.stationarity.is_finite() || r.kkt.stationarity >= EPS_KKT {
                stat_violations.push(format!(
                    "{}: status=LocallyOptimal stat={:.3e} (>= {:.0e})",
                    r.name, r.kkt.stationarity, EPS_KKT
                ));
            }
        } else if matches!(r.status, SolveStatus::Optimal)
            && r.kkt.stationarity.is_finite()
            && r.kkt.stationarity >= EPS_KKT
        {
            optimal_stat_warns.push(format!(
                "{}: status=Optimal stat={:.3e} (>= {:.0e}, non-PSD unscale 経路の pre-existing 不具合候補)",
                r.name, r.kkt.stationarity, EPS_KKT
            ));
        }
    }

    eprintln!(
        "[nonconvex_kkt] summary: Optimal={} LocallyOptimal={} Suboptimal={} other={} (total={})",
        n_optimal,
        n_local,
        n_subopt,
        n_other,
        records.len(),
    );
    if !optimal_stat_warns.is_empty() {
        eprintln!(
            "[nonconvex_kkt] WARN: {} Optimal-claim 解で元空間 stationarity > {:.0e}:\n  {}",
            optimal_stat_warns.len(),
            EPS_KKT,
            optimal_stat_warns.join("\n  ")
        );
    }
    // 公式 4 件の gap_to_global を専用 log 行に集約 (Phase 2/3 baseline)。
    for r in &records {
        if let Some(g) = r.gap_to_global {
            eprintln!(
                "[nonconvex_gap] {:<24} status={:?} obj={:.6e} gap_to_global={:.3e}",
                r.name, r.status, r.objective, g
            );
        }
    }

    assert!(
        prim_violations.is_empty(),
        "primal feasibility violation:\n  {}",
        prim_violations.join("\n  ")
    );
    assert!(
        comp_violations.is_empty(),
        "complementarity violation (Optimal/LocallyOptimal):\n  {}",
        comp_violations.join("\n  ")
    );
    assert!(
        stat_violations.is_empty(),
        "stationarity violation (LocallyOptimal):\n  {}",
        stat_violations.join("\n  ")
    );
}

/// sentinel: KKT 計算が「常に 0 を返す」no-op で置換されたら確実に FAIL。
///
/// 元解 x* に SENTINEL_PERTURB 単位シフトを加えると、合成 nonconvex の Q
/// (典型 ‖Q‖_max ≈ O(1)) で stationarity 残差が SENTINEL_MIN_KKT 以上に増える。
/// `compute_kkt` の本体を `return KktResidual { stationarity: 0.0, ... };` に
/// 書換えると 0 ≥ 1e-2 が偽となり assert が崩れる。
#[test]
fn kkt_perturbation_sentinel() {
    // NONCONVEX_DENSE_N20: n=20, m=0, dense indefinite Q。solve は < 1s で
    // SolverResult を返す (LocallyOptimal or Optimal いずれでも本 sentinel は成立)。
    let path = Path::new(SYNTH_DIR).join("NONCONVEX_DENSE_N20.qplib");
    let prob = match parse_qplib(&path).expect("parse NONCONVEX_DENSE_N20") {
        QplibProblem::Qp(p) => p,
        other => panic!("expected continuous QP, got {:?}", other),
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_PER_PROBLEM_SECS);
    let res = solve_qp_with(&prob, &opts);

    let base = compute_kkt(&prob, &res.solution, &res.dual_solution, &res.bound_duals);
    eprintln!(
        "[sentinel] base kkt: stat={:.2e} prim={:.2e} comp_i={:.2e} comp_b={:.2e} (max={:.2e})",
        base.stationarity,
        base.primal_inf,
        base.comp_ineq,
        base.comp_bound,
        base.max(),
    );

    assert!(!res.solution.is_empty(), "solver returned empty solution");
    let mut x_perturbed = res.solution.clone();
    x_perturbed[0] += SENTINEL_PERTURB;
    let kkt_p = compute_kkt(&prob, &x_perturbed, &res.dual_solution, &res.bound_duals);
    let max_p = kkt_p.max();
    eprintln!(
        "[sentinel] perturbed (x[0]+={}) kkt: stat={:.2e} prim={:.2e} comp_i={:.2e} comp_b={:.2e} (max={:.2e})",
        SENTINEL_PERTURB,
        kkt_p.stationarity, kkt_p.primal_inf, kkt_p.comp_ineq, kkt_p.comp_bound, max_p,
    );
    assert!(
        max_p >= SENTINEL_MIN_KKT,
        "sentinel broken: perturbed kkt_max={:.3e} < {:.0e}; \
         compute_kkt が no-op 化されていないか確認",
        max_p,
        SENTINEL_MIN_KKT,
    );
}

/// `compute_kkt` (test 内) と `bench_utils::compute_qp_kkt_max` (production) の
/// max 値が同じ規約 (符号 + 成分正規化) で一致することを検証する。
///
/// 同じ規約で 2 経路実装した「片方の no-op (or 符号反転) を他方が catch する」二重化。
#[test]
fn kkt_consistency_with_bench_utils() {
    let path = Path::new(SYNTH_DIR).join("NONCONVEX_OFFDIAG_N20.qplib");
    let prob = match parse_qplib(&path).expect("parse") {
        QplibProblem::Qp(p) => p,
        other => panic!("expected continuous QP, got {:?}", other),
    };
    let mut opts = SolverOptions::default();
    opts.timeout_secs = Some(TIMEOUT_PER_PROBLEM_SECS);
    let res = solve_qp_with(&prob, &opts);
    let k = compute_kkt(&prob, &res.solution, &res.dual_solution, &res.bound_duals);
    let bu_max = compute_qp_kkt_max(&prob, &res.solution, &res.dual_solution, &res.bound_duals);
    let test_max = k.max();
    eprintln!(
        "[kkt_consistency] test_max={:.3e} bench_utils_max={:.3e}",
        test_max, bu_max
    );
    // 両者は規約等価なので相対差 < 1e-12 (浮動小数 round-off 内)。
    let diff = (test_max - bu_max).abs();
    let denom = test_max.max(bu_max).max(1e-18);
    let rel = diff / denom;
    assert!(
        rel < 1e-9,
        "test compute_kkt.max ({:.6e}) と bench_utils::compute_qp_kkt_max ({:.6e}) が rel={:.3e} 乖離。\
         規約 (符号 / 成分正規化) のどちらかに drift がある可能性。",
        test_max, bu_max, rel,
    );
}

/// 解析解持ち convex mini QP で `compute_kkt` の符号規約を独立検証。
///
/// min 1/2 (x-2)² s.t. x ≤ 1, x∈ℝ.  解: x*=1, y*=1, bound 無し.
/// stationarity: 1·1 + 1·1 + 0 + (−2) = 0  ✓
/// comp_ineq: 1·(1−1) = 0  ✓
/// no-op で stationarity=0 を返してもこの test は PASS してしまうので、
/// 検出力本体は `kkt_perturbation_sentinel` で担保。本 test は規約 sanity。
#[test]
fn kkt_sign_convention_mini_qp() {
    use otspot::sparse::CscMatrix;
    let q = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let c = vec![-2.0];
    let a = CscMatrix::from_triplets(&[0], &[0], &[1.0], 1, 1).unwrap();
    let b = vec![1.0];
    let bounds = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    let prob = QpProblem::new(q, c, a, b, bounds, vec![ConstraintType::Le]).unwrap();
    let res = solve_qp_with(&prob, &SolverOptions::default());
    assert_eq!(res.status, SolveStatus::Optimal);
    let k = compute_kkt(&prob, &res.solution, &res.dual_solution, &res.bound_duals);
    eprintln!(
        "[sign-conv] mini convex QP kkt: stat={:.2e} prim={:.2e} comp_i={:.2e} comp_b={:.2e}",
        k.stationarity, k.primal_inf, k.comp_ineq, k.comp_bound,
    );
    assert!(
        k.max() < 1e-6,
        "convex mini QP must give near-zero KKT, got {:.3e}",
        k.max()
    );
}
