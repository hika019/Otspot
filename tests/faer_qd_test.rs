//! faer LDL quasidefinite feasibility test (cmd_275 subtask_275a)
//!
//! Verifies that faer's sparse LDL factorization correctly handles
//! augmented KKT matrices (quasidefinite structure) as required for IPM.
//!
//! Quasidefinite matrix structure:
//!   M = [Q+δI   A^T]
//!       [A      -δI]
//! where Q is positive semidefinite and δ > 0 ensures strict quasidefiniteness.

#![allow(non_snake_case)]

use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::dyn_stack;
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::reborrow::*;
use faer::sparse::linalg::cholesky::{simplicial, supernodal};
use faer::sparse::{SparseColMat, Triplet};

/// Helper: compute ||Mx - b||_2 for a dense matrix M given as triplets
fn residual_norm(
    dim: usize,
    triplets: &[(usize, usize, f64)],
    x: &[f64],
    b: &[f64],
) -> f64 {
    let mut r = vec![0.0f64; dim];
    for &(row, col, val) in triplets {
        r[row] += val * x[col];
    }
    let mut sq_sum = 0.0f64;
    for i in 0..dim {
        let diff = r[i] - b[i];
        sq_sum += diff * diff;
    }
    sq_sum.sqrt()
}

/// Test 1: 3x3 quasidefinite matrix with simplicial LDL
///
/// M = [1+δ    0      1  ]
///     [0      2+δ    1  ]
///     [1      1      -δ ]
///
/// Q = diag(1, 2), A = [1, 1], δ = 1e-4
#[test]
fn test_3x3_simplicial_ldlt_default_regularization() {
    let delta = 1e-4f64;
    let dim = 3usize;

    // Upper triangular entries (simplicial takes upper triangular)
    let a_upper = SparseColMat::<usize, f64>::try_new_from_triplets(
        dim,
        dim,
        &[
            Triplet::new(0, 0, 1.0 + delta),
            Triplet::new(1, 1, 2.0 + delta),
            Triplet::new(2, 2, -delta),
            Triplet::new(0, 2, 1.0), // A^T: row=0, col=2
            Triplet::new(1, 2, 1.0), // A^T: row=1, col=2
        ],
    )
    .unwrap();

    let b = vec![1.0f64, 2.0, 0.0];
    let mut sol = b.clone();

    let info = simplicial_ldlt_solve(dim, &a_upper, &mut sol, LdltRegularization::default());

    // Full matrix triplets for residual check
    let full_triplets = vec![
        (0, 0, 1.0 + delta),
        (1, 1, 2.0 + delta),
        (2, 2, -delta),
        (0, 2, 1.0),
        (2, 0, 1.0),
        (1, 2, 1.0),
        (2, 1, 1.0),
    ];
    let res = residual_norm(dim, &full_triplets, &sol, &b);
    println!(
        "[3x3 default reg] residual={:.3e}, regularized_pivots={}",
        res, info
    );
    assert!(
        res < 1e-10,
        "3x3 simplicial LDL (default reg): residual {:.3e} >= 1e-10",
        res
    );
}

/// Test 2: 3x3 quasidefinite matrix with sign-aware regularization
///
/// Signs [+1, +1, -1] tell faer the expected pivot signs.
/// This mimics how IPM would use faer for the augmented system.
#[test]
fn test_3x3_simplicial_ldlt_sign_aware_regularization() {
    let delta = 1e-4f64;
    let dim = 3usize;

    let a_upper = SparseColMat::<usize, f64>::try_new_from_triplets(
        dim,
        dim,
        &[
            Triplet::new(0, 0, 1.0 + delta),
            Triplet::new(1, 1, 2.0 + delta),
            Triplet::new(2, 2, -delta),
            Triplet::new(0, 2, 1.0),
            Triplet::new(1, 2, 1.0),
        ],
    )
    .unwrap();

    let b = vec![3.0f64, -1.0, 2.0];
    let mut sol = b.clone();

    // Expected pivot signs: Q block positive, slack block negative
    let signs = vec![1i8, 1, -1];
    let regularization = LdltRegularization {
        dynamic_regularization_signs: Some(&signs),
        dynamic_regularization_delta: 1e-8,
        dynamic_regularization_epsilon: 1e-13,
    };

    let info = simplicial_ldlt_solve(dim, &a_upper, &mut sol, regularization);

    let full_triplets = vec![
        (0, 0, 1.0 + delta),
        (1, 1, 2.0 + delta),
        (2, 2, -delta),
        (0, 2, 1.0),
        (2, 0, 1.0),
        (1, 2, 1.0),
        (2, 1, 1.0),
    ];
    let res = residual_norm(dim, &full_triplets, &sol, &b);
    println!(
        "[3x3 sign-aware reg] residual={:.3e}, regularized_pivots={}",
        res, info
    );
    assert!(
        res < 1e-10,
        "3x3 simplicial LDL (sign-aware reg): residual {:.3e} >= 1e-10",
        res
    );
}

/// Test 3: 5x5 quasidefinite matrix with simplicial LDL
///
/// M = [Q+δI   A^T]  where Q=diag(1,2), A=[[1,0],[0,1],[1,1]], δ=1e-4
///     [A      -δI]
///
/// M = [1+δ    0      1    0    1  ]
///     [0      2+δ    0    1    1  ]
///     [1      0      -δ   0    0  ]
///     [0      1      0    -δ   0  ]
///     [1      1      0    0    -δ ]
#[test]
fn test_5x5_simplicial_ldlt_quasidefinite() {
    let delta = 1e-4f64;
    let dim = 5usize;

    // Upper triangular part
    let a_upper = SparseColMat::<usize, f64>::try_new_from_triplets(
        dim,
        dim,
        &[
            // diagonal
            Triplet::new(0, 0, 1.0 + delta),
            Triplet::new(1, 1, 2.0 + delta),
            Triplet::new(2, 2, -delta),
            Triplet::new(3, 3, -delta),
            Triplet::new(4, 4, -delta),
            // A^T block: upper triangular entries (row < col)
            Triplet::new(0, 2, 1.0), // A^T[0,0] -> (row=0, col=2)
            Triplet::new(1, 3, 1.0), // A^T[1,1] -> (row=1, col=3)
            Triplet::new(0, 4, 1.0), // A^T[0,2] -> (row=0, col=4)
            Triplet::new(1, 4, 1.0), // A^T[1,2] -> (row=1, col=4)
        ],
    )
    .unwrap();

    let b = vec![1.0f64, 2.0, 0.5, -0.5, 1.0];
    let mut sol = b.clone();

    let signs = vec![1i8, 1, -1, -1, -1];
    let regularization = LdltRegularization {
        dynamic_regularization_signs: Some(&signs),
        dynamic_regularization_delta: 1e-8,
        dynamic_regularization_epsilon: 1e-13,
    };

    let info = simplicial_ldlt_solve(dim, &a_upper, &mut sol, regularization);

    // Full matrix triplets
    let full_triplets = vec![
        (0, 0, 1.0 + delta),
        (1, 1, 2.0 + delta),
        (2, 2, -delta),
        (3, 3, -delta),
        (4, 4, -delta),
        (0, 2, 1.0),
        (2, 0, 1.0),
        (1, 3, 1.0),
        (3, 1, 1.0),
        (0, 4, 1.0),
        (4, 0, 1.0),
        (1, 4, 1.0),
        (4, 1, 1.0),
    ];
    let res = residual_norm(dim, &full_triplets, &sol, &b);
    println!(
        "[5x5 simplicial] residual={:.3e}, regularized_pivots={}",
        res, info
    );
    assert!(
        res < 1e-10,
        "5x5 simplicial LDL: residual {:.3e} >= 1e-10",
        res
    );
}

/// Test 4: 5x5 quasidefinite matrix with supernodal LDL
///
/// Same matrix as Test 3 but using supernodal factorization.
/// Supernodal is preferred for larger matrices (>50 vars in practice).
#[test]
fn test_5x5_supernodal_ldlt_quasidefinite() {
    let delta = 1e-4f64;
    let dim = 5usize;

    // Supernodal takes LOWER triangular input
    let a_lower = SparseColMat::<usize, f64>::try_new_from_triplets(
        dim,
        dim,
        &[
            // diagonal
            Triplet::new(0, 0, 1.0 + delta),
            Triplet::new(1, 1, 2.0 + delta),
            Triplet::new(2, 2, -delta),
            Triplet::new(3, 3, -delta),
            Triplet::new(4, 4, -delta),
            // A block: lower triangular entries (row > col)
            Triplet::new(2, 0, 1.0), // A[0,0] -> (row=2, col=0)
            Triplet::new(3, 1, 1.0), // A[1,1] -> (row=3, col=1)
            Triplet::new(4, 0, 1.0), // A[2,0] -> (row=4, col=0)
            Triplet::new(4, 1, 1.0), // A[2,1] -> (row=4, col=1)
        ],
    )
    .unwrap();

    let b = vec![1.0f64, 2.0, 0.5, -0.5, 1.0];
    let mut sol = b.clone();

    let signs = vec![1i8, 1, -1, -1, -1];
    let regularization = LdltRegularization {
        dynamic_regularization_signs: Some(&signs),
        dynamic_regularization_delta: 1e-8,
        dynamic_regularization_epsilon: 1e-13,
    };

    let info = supernodal_ldlt_solve(dim, &a_lower, &mut sol, regularization);

    let full_triplets = vec![
        (0, 0, 1.0 + delta),
        (1, 1, 2.0 + delta),
        (2, 2, -delta),
        (3, 3, -delta),
        (4, 4, -delta),
        (0, 2, 1.0),
        (2, 0, 1.0),
        (1, 3, 1.0),
        (3, 1, 1.0),
        (0, 4, 1.0),
        (4, 0, 1.0),
        (1, 4, 1.0),
        (4, 1, 1.0),
    ];
    let res = residual_norm(dim, &full_triplets, &sol, &b);
    println!(
        "[5x5 supernodal] residual={:.3e}, regularized_pivots={}",
        res, info
    );
    assert!(
        res < 1e-10,
        "5x5 supernodal LDL: residual {:.3e} >= 1e-10",
        res
    );
}

/// Test 5: IPM-like convergence simulation
///
/// As IPM iterates, δ shrinks (μ → 0). Verify LDL remains stable
/// across a range of δ values representative of IPM progression.
#[test]
fn test_ipm_barrier_parameter_sweep() {
    // δ values representing IPM barrier parameter progression
    let deltas = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8];
    let dim = 3usize;

    for &delta in &deltas {
        let a_upper = SparseColMat::<usize, f64>::try_new_from_triplets(
            dim,
            dim,
            &[
                Triplet::new(0, 0, 1.0 + delta),
                Triplet::new(1, 1, 2.0 + delta),
                Triplet::new(2, 2, -delta),
                Triplet::new(0, 2, 1.0),
                Triplet::new(1, 2, 1.0),
            ],
        )
        .unwrap();

        let b = vec![1.0f64, 2.0, 0.0];
        let mut sol = b.clone();

        let signs = vec![1i8, 1, -1];
        let regularization = LdltRegularization {
            dynamic_regularization_signs: Some(&signs),
            dynamic_regularization_delta: delta * 1e-4,
            dynamic_regularization_epsilon: delta * 1e-9,
        };

        simplicial_ldlt_solve(dim, &a_upper, &mut sol, regularization);

        let full_triplets = vec![
            (0, 0, 1.0 + delta),
            (1, 1, 2.0 + delta),
            (2, 2, -delta),
            (0, 2, 1.0),
            (2, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
        ];
        let res = residual_norm(dim, &full_triplets, &sol, &b);
        println!("[IPM sweep δ={:.0e}] residual={:.3e}", delta, res);
        assert!(
            res < 1e-8,
            "IPM sweep δ={:.0e}: residual {:.3e} >= 1e-8",
            delta,
            res
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run simplicial LDLT on upper-triangular A_upper, solve in place.
/// Returns LdltInfo::dynamic_regularization_count.
fn simplicial_ldlt_solve(
    dim: usize,
    a_upper: &SparseColMat<usize, f64>,
    sol: &mut Vec<f64>,
    regularization: LdltRegularization<'_, f64>,
) -> usize {
    let a_nnz = a_upper.compute_nnz();

    // Symbolic analysis
    let symbolic = {
        let mut mem = MemBuffer::new(StackReq::any_of(&[
            simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(dim, a_nnz),
            simplicial::factorize_simplicial_symbolic_cholesky_scratch::<usize>(dim),
        ]));
        let stack = MemStack::new(&mut mem);

        let mut etree = vec![0isize; dim];
        let mut col_counts = vec![0usize; dim];

        simplicial::prefactorize_symbolic_cholesky(
            &mut etree,
            &mut col_counts,
            a_upper.symbolic(),
            stack,
        );

        simplicial::factorize_simplicial_symbolic_cholesky(
            a_upper.symbolic(),
            unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
            &col_counts,
            stack,
        )
        .expect("symbolic cholesky failed")
    };

    // Numeric factorization + solve
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        simplicial::factorize_simplicial_numeric_ldlt_scratch::<usize, f64>(dim),
        symbolic.solve_in_place_scratch::<f64>(dim),
    ]));
    let stack = MemStack::new(&mut mem);

    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let info = simplicial::factorize_simplicial_numeric_ldlt::<usize, f64>(
        &mut l_values,
        a_upper.rb(),
        regularization,
        &symbolic,
        stack,
    );

    let ldlt = simplicial::SimplicialLdltRef::<'_, usize, f64>::new(&symbolic, &l_values);

    let mut sol_mat = faer::MatMut::from_column_major_slice_mut(sol.as_mut_slice(), dim, 1);
    ldlt.solve_in_place_with_conj(faer::Conj::No, sol_mat.rb_mut(), faer::Par::Seq, stack);

    info.unwrap().dynamic_regularization_count
}

/// Run supernodal LDLT on lower-triangular A_lower, solve in place.
/// Returns LdltInfo::dynamic_regularization_count.
fn supernodal_ldlt_solve(
    dim: usize,
    a_lower: &SparseColMat<usize, f64>,
    sol: &mut Vec<f64>,
    regularization: LdltRegularization<'_, f64>,
) -> usize {
    let a_nnz = a_lower.compute_nnz();

    // Supernodal symbolic analysis needs upper triangular for elimination tree
    let a_upper_sym = a_lower
        .rb()
        .transpose()
        .symbolic()
        .to_col_major()
        .expect("transpose to col-major failed");

    let symbolic = {
        let mut mem = MemBuffer::new(StackReq::any_of(&[
            simplicial::prefactorize_symbolic_cholesky_scratch::<usize>(dim, a_nnz),
            supernodal::factorize_supernodal_symbolic_cholesky_scratch::<usize>(dim),
        ]));
        let stack = MemStack::new(&mut mem);

        let mut etree = vec![0isize; dim];
        let mut col_counts = vec![0usize; dim];

        simplicial::prefactorize_symbolic_cholesky(
            &mut etree,
            &mut col_counts,
            a_upper_sym.rb(),
            stack,
        );

        supernodal::factorize_supernodal_symbolic_cholesky(
            a_upper_sym.rb(),
            unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
            &col_counts,
            stack,
            faer::sparse::linalg::SymbolicSupernodalParams {
                relax: Some(&[(usize::MAX, 1.0)]),
            },
        )
        .expect("supernodal symbolic cholesky failed")
    };

    // Numeric factorization + solve
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        supernodal::factorize_supernodal_numeric_ldlt_scratch::<usize, f64>(
            &symbolic,
            faer::Par::Seq,
            Default::default(),
        ),
        symbolic.solve_in_place_scratch::<f64>(dim, faer::Par::Seq),
    ]));
    let stack = MemStack::new(&mut mem);

    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let info = supernodal::factorize_supernodal_numeric_ldlt::<usize, f64>(
        &mut l_values,
        a_lower.rb(),
        regularization,
        &symbolic,
        faer::Par::Seq,
        stack,
        Default::default(),
    );

    let ldlt =
        supernodal::SupernodalLdltRef::<'_, usize, f64>::new(&symbolic, &l_values);

    let mut sol_mat = faer::MatMut::from_column_major_slice_mut(sol.as_mut_slice(), dim, 1);
    ldlt.solve_in_place_with_conj(faer::Conj::No, sol_mat.rb_mut(), faer::Par::Seq, stack);

    info.unwrap().dynamic_regularization_count
}
