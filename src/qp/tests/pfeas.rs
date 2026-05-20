use crate::sparse::CscMatrix;

/// row_infinity_norms 基本。
#[test]
fn test_row_infinity_norms_basic() {
    let a = CscMatrix::from_triplets(
        &[0, 1, 0],
        &[0, 1, 2],
        &[1.0, 2.5, -3.0],
        2,
        3,
    )
    .unwrap();
    let norms = a.row_infinity_norms();
    assert_eq!(norms.len(), 2);
    assert!((norms[0] - 3.0).abs() < 1e-15);
    assert!((norms[1] - 2.5).abs() < 1e-15);
}

/// 大/小係数行 mixed で行ノルム正規化 pfeas が偽 SubOptimal を防ぐ。
#[test]
fn test_pfeas_row_norm_mixed_scale() {
    let a = CscMatrix::from_triplets(&[0, 1], &[0, 0], &[1.0, 1000.0], 2, 1).unwrap();
    let norms = a.row_infinity_norms();
    assert!((norms[0] - 1.0).abs() < 1e-15);
    assert!((norms[1] - 1000.0).abs() < 1e-15);

    let b: Vec<f64> = vec![1.0, 1000.0];
    let x_val: f64 = 1.0 + 1e-7;
    let ax: Vec<f64> = vec![x_val, 1000.0 * x_val];
    let eps: f64 = 1e-6;

    let pfeas_old = ax
        .iter()
        .zip(b.iter())
        .map(|(&ax_i, &b_i)| (ax_i - b_i).max(0.0))
        .fold(0.0_f64, f64::max);
    assert!(pfeas_old > 1e-5);

    let pfeas_normalized = ax
        .iter()
        .zip(b.iter())
        .zip(norms.iter())
        .map(|((&ax_i, &b_i), &rn)| {
            let violation = (ax_i - b_i).max(0.0);
            violation / (1.0 + rn + b_i.abs())
        })
        .fold(0.0_f64, f64::max);
    assert!(pfeas_normalized < eps);
}

/// b=0 大係数行で正規化 pfeas が偽 SubOptimal を防ぐ。
#[test]
fn test_pfeas_row_norm_false_suboptimal_prevention() {
    let a = CscMatrix::from_triplets(&[0], &[0], &[1e6], 1, 1).unwrap();
    let norms = a.row_infinity_norms();
    assert!((norms[0] - 1e6).abs() < 1e-9);

    let b_val: f64 = 0.0;
    let ax_val: f64 = 1e6 * 1e-9;
    let eps: f64 = 1e-6;

    let norm_b = b_val.abs().max(1.0);
    let pfeas_old = (ax_val - b_val).abs();
    assert!(pfeas_old >= eps * (1.0 + norm_b));

    let pfeas_norm = (ax_val - b_val).abs() / (1.0 + norms[0] + b_val.abs());
    assert!(pfeas_norm < eps);
}
