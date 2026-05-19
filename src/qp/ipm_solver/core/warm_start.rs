//! presolve reduced 空間 + Ruiz scaled 空間への warm_start_qp 翻訳。

use crate::options::SolverOptions;
use crate::presolve::QpPresolveResult;
use crate::qp::problem::QpProblem;

/// presolve で col/row が reduced 空間に縮んだ場合、warm_start_qp.x / .y を col_map_inv /
/// row_map で reduced 空間に翻訳する。dropped 列・行 (col_map[j]/row_map[i] = None) の warm
/// 値は reduced 問題に存在しないため棄却。dim 不一致は警告付き drop。
///
/// 加えて presolve 内 Ruiz scaler (qp_transforms.rs 末尾) が reduced 問題を scaled 空間に
/// 書き換えている場合、warm の (x, y) を同じ scaled 空間に変換する:
///   x_s = D^{-1} x_orig         (RuizScaler::scale_problem の `bounds_s = bounds / d` と整合)
///   y_s = c * y_orig / e        (KKT より: y_orig = e * y_s / c → y_s = c * y_orig / e)
/// この変換が無いと `presolve_did_ruiz` 経路 (attempt.rs) で IPM 側 use_ruiz=false 固定の
/// ため IPM 入口 Ruiz scaling (ipm_core/scaling.rs) も bypass され、orig 空間の warm が
/// scaled reduced 問題に入り誤位置 init になる。
pub(super) fn translate_warm_start_for_presolve(
    opts: &mut SolverOptions,
    presolve_result: &QpPresolveResult,
    reduced: &QpProblem,
) {
    let needs_reduce = presolve_result.was_reduced;
    let needs_ruiz = presolve_result.ruiz_scaler.is_some();
    if !needs_reduce && !needs_ruiz {
        return;
    }
    let Some(ws) = opts.warm_start_qp.as_mut() else { return };

    let n_orig = presolve_result.orig_num_vars;
    let m_orig = presolve_result.orig_num_constraints;
    if ws.x.len() != n_orig || ws.y.len() != m_orig {
        eprintln!(
            "[warm_start_qp dropped] presolve dim mismatch: ws.x={}/{} ws.y={}/{}",
            ws.x.len(), n_orig, ws.y.len(), m_orig
        );
        opts.warm_start_qp = None;
        return;
    }

    let n_red = reduced.num_vars;
    let m_red = reduced.num_constraints;

    let mut x_red = vec![0.0_f64; n_red];
    if needs_reduce {
        for (k, &j_orig) in presolve_result.col_map_inv.iter().enumerate() {
            if k < n_red && j_orig < n_orig {
                x_red[k] = ws.x[j_orig];
            }
        }
    } else if ws.x.len() == n_red {
        x_red.copy_from_slice(&ws.x);
    }

    let mut y_red = vec![0.0_f64; m_red];
    if needs_reduce {
        for (i_orig, mapped) in presolve_result.row_map.iter().enumerate() {
            if let Some(i_red) = mapped {
                if *i_red < m_red {
                    y_red[*i_red] = ws.y[i_orig];
                }
            }
        }
    } else if ws.y.len() == m_red {
        y_red.copy_from_slice(&ws.y);
    }

    if let Some(scaler) = &presolve_result.ruiz_scaler {
        if scaler.d.len() != n_red || scaler.e.len() != m_red
            || !scaler.c.is_finite() || scaler.c <= 0.0
        {
            eprintln!(
                "[warm_start_qp dropped] ruiz scaler dim/c invalid: d={}/{} e={}/{} c={}",
                scaler.d.len(), n_red, scaler.e.len(), m_red, scaler.c
            );
            opts.warm_start_qp = None;
            return;
        }
        for k in 0..n_red {
            let dk = scaler.d[k];
            if !dk.is_finite() || dk == 0.0 {
                eprintln!("[warm_start_qp dropped] ruiz d[{}]={} non-finite/zero", k, dk);
                opts.warm_start_qp = None;
                return;
            }
            x_red[k] /= dk;
        }
        for i in 0..m_red {
            let ei = scaler.e[i];
            if !ei.is_finite() || ei == 0.0 {
                eprintln!("[warm_start_qp dropped] ruiz e[{}]={} non-finite/zero", i, ei);
                opts.warm_start_qp = None;
                return;
            }
            y_red[i] = scaler.c * y_red[i] / ei;
        }
    }

    if std::env::var("IPPMM_TRACE").ok().as_deref() == Some("1") {
        eprintln!(
            "[warm_start_qp translated] presolve reduction n:{}→{} m:{}→{} ruiz={}",
            n_orig, n_red, m_orig, m_red, needs_ruiz
        );
    }

    ws.x = x_red;
    ws.y = y_red;
}
