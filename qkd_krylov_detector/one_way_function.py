"""
One-Way Function Property of the Krylov Framework
===================================================

Implements the one-way function analysis from Paper [11], Section V.

The forward map F: H -> {b_n} is computationally easy (polynomial in d),
but the inverse map F^{-1}: {b_n} -> H is exponentially hard. This is
quantified through the Hankel matrix condition number:

    kappa(M_Hankel) ~ 25^n for N = 18 qubits

where M_Hankel is constructed from the moments mu_k = <O|L^k|O>.

This exponential growth of the condition number establishes the
Krylov framework as a physical one-way function: Eve can observe
the QBER statistics but cannot reconstruct the Hamiltonian from them.

This module provides:
    - compute_hankel_matrix: Build the Hankel moment matrix
    - compute_condition_number: Condition number of the Hankel matrix
    - forward_map: Compute b_n from H (the easy direction)
    - test_inversion_hardness: Quantify inversion difficulty
    - scaling_analysis: Condition number vs. system size

References:
    [11] D. Suess, "Theoretical Foundations of the Krylov Eavesdropper
         Detector: Physical Bridge, One-Way Function, and Universality,"
         Zenodo, 2026. DOI: 10.5281/zenodo.18957362

Author: Daniel Suess
License: AGPL-3.0-or-later
SPDX-License-Identifier: AGPL-3.0-or-later

Copyright (C) 2026 Daniel Suess
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy as np
from scipy.linalg import eigh


def compute_moments(H, O, n_moments=20):
    """
    Compute the Liouvillian moments mu_k = <O | L^k | O>.

    where L(.) = i[H, .] and <A|B> = Tr(A^dag B) / d.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian.
    O : ndarray, shape (d, d)
        Observable operator.
    n_moments : int
        Number of moments to compute (default: 20).

    Returns
    -------
    moments : ndarray, shape (n_moments,)
        Liouvillian moments mu_0, mu_1, ..., mu_{n-1}.

    Reference: Paper [11], Eq. (10)
    """
    H = np.asarray(H, dtype=complex)
    O = np.asarray(O, dtype=complex)
    d = H.shape[0]

    O_norm_sq = np.trace(O.conj().T @ O).real / d
    if O_norm_sq < 1e-30:
        return np.zeros(n_moments)

    moments = np.zeros(n_moments)
    L_k_O = O.copy()  # L^0 O = O

    for k in range(n_moments):
        # mu_k = <O | L^k O> = Tr(O^dag L^k O) / d
        moments[k] = np.trace(O.conj().T @ L_k_O).real / d / O_norm_sq

        # L^{k+1} O = i[H, L^k O]
        if k < n_moments - 1:
            L_k_O = 1j * (H @ L_k_O - L_k_O @ H)

    return moments


def compute_hankel_matrix(moments, size=None):
    """
    Construct the Hankel moment matrix from Liouvillian moments.

    M_Hankel[i,j] = mu_{i+j}

    The condition number of this matrix quantifies the difficulty
    of inverting the forward map F: H -> {b_n}.

    Parameters
    ----------
    moments : ndarray
        Liouvillian moments mu_0, mu_1, ..., mu_{2n-2}.
    size : int or None
        Size of the Hankel matrix. If None, uses len(moments)//2 + 1.

    Returns
    -------
    M : ndarray, shape (size, size)
        Hankel moment matrix.

    Reference: Paper [11], Eq. (11)
    """
    moments = np.asarray(moments, dtype=float)
    if size is None:
        size = len(moments) // 2 + 1

    # Ensure we have enough moments
    n_needed = 2 * size - 1
    if len(moments) < n_needed:
        # Pad with zeros if necessary
        moments = np.pad(moments, (0, n_needed - len(moments)))

    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i + j < len(moments):
                M[i, j] = moments[i + j]

    return M


def compute_condition_number(M, method="svd"):
    """
    Compute the condition number of a matrix.

    Parameters
    ----------
    M : ndarray
        Input matrix.
    method : str
        "svd" for singular value decomposition (default),
        "eig" for eigenvalue-based (for symmetric matrices).

    Returns
    -------
    kappa : float
        Condition number. Large values indicate ill-conditioning.

    Reference: Paper [11], Section V.B
    """
    M = np.asarray(M, dtype=float)

    if method == "svd":
        s = np.linalg.svd(M, compute_uv=False)
        if s[-1] < 1e-30:
            return float('inf')
        return float(s[0] / s[-1])
    elif method == "eig":
        eigenvalues = np.abs(np.linalg.eigvalsh(M))
        eigenvalues = eigenvalues[eigenvalues > 1e-30]
        if len(eigenvalues) == 0:
            return float('inf')
        return float(np.max(eigenvalues) / np.min(eigenvalues))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'svd' or 'eig'.")


def forward_map(H, O, n_steps=25):
    """
    The forward map F: H -> {b_n}.

    This is the "easy" direction: computing Lanczos coefficients from
    the Hamiltonian is polynomial in d = 2^N.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian.
    O : ndarray, shape (d, d)
        Observable operator.
    n_steps : int
        Number of Lanczos steps.

    Returns
    -------
    b_n : ndarray
        Lanczos coefficients.

    Reference: Paper [11], Eq. (1)
    """
    from .physical_bridge import compute_lanczos_coefficients
    return compute_lanczos_coefficients(H, O, n_steps)


def test_inversion_hardness(H, O, n_moments=20, hankel_sizes=None):
    """
    Test the hardness of inverting the forward map.

    Computes the Hankel matrix condition number for increasing matrix
    sizes, demonstrating exponential growth.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian.
    O : ndarray, shape (d, d)
        Observable.
    n_moments : int
        Number of moments to compute.
    hankel_sizes : list of int or None
        Hankel matrix sizes to test. If None, uses [3, 5, 7, 9].

    Returns
    -------
    result : dict
        Keys:
        - 'sizes': Hankel matrix sizes tested
        - 'condition_numbers': Condition number for each size
        - 'growth_rate': Exponential growth rate (base)
        - 'moments': Computed moments
        - 'is_one_way': True if condition number grows exponentially

    Reference: Paper [11], Section V.C
    """
    if hankel_sizes is None:
        hankel_sizes = [3, 5, 7, 9]

    moments = compute_moments(H, O, n_moments)
    condition_numbers = []

    for size in hankel_sizes:
        M = compute_hankel_matrix(moments, size)
        kappa = compute_condition_number(M)
        condition_numbers.append(kappa)

    condition_numbers = np.array(condition_numbers)
    hankel_sizes = np.array(hankel_sizes)

    # Fit exponential growth: log(kappa) ~ a * size + b
    finite_mask = np.isfinite(condition_numbers) & (condition_numbers > 0)
    if np.sum(finite_mask) >= 2:
        log_kappa = np.log(condition_numbers[finite_mask])
        sizes_fit = hankel_sizes[finite_mask]
        coeffs = np.polyfit(sizes_fit, log_kappa, 1)
        growth_rate = float(np.exp(coeffs[0]))
    else:
        growth_rate = 1.0

    # One-way if growth rate > 2 (exponential)
    is_one_way = growth_rate > 2.0

    return {
        'sizes': hankel_sizes.tolist(),
        'condition_numbers': condition_numbers.tolist(),
        'growth_rate': growth_rate,
        'moments': moments,
        'is_one_way': is_one_way,
    }


def scaling_analysis(build_hamiltonian_fn, build_observable_fn,
                     N_values, n_moments=20, hankel_size=7):
    """
    Analyze condition number scaling with system size N.

    Parameters
    ----------
    build_hamiltonian_fn : callable
        Function N -> H that builds the Hamiltonian for N qubits.
    build_observable_fn : callable
        Function N -> O that builds the observable for N qubits.
    N_values : list of int
        System sizes to test.
    n_moments : int
        Number of moments per system.
    hankel_size : int
        Hankel matrix size for condition number.

    Returns
    -------
    result : dict
        Keys:
        - 'N_values': System sizes
        - 'condition_numbers': Condition number for each N
        - 'growth_base': Exponential growth base per qubit
        - 'is_exponential': True if growth is exponential in N

    Reference: Paper [11], Section V.D (Scaling Analysis)
    """
    condition_numbers = []

    for N in N_values:
        H = build_hamiltonian_fn(N)
        O = build_observable_fn(N)
        moments = compute_moments(H, O, n_moments)
        M = compute_hankel_matrix(moments, hankel_size)
        kappa = compute_condition_number(M)
        condition_numbers.append(kappa)

    condition_numbers = np.array(condition_numbers)
    N_arr = np.array(N_values)

    # Fit: log(kappa) ~ a * N + b
    finite_mask = np.isfinite(condition_numbers) & (condition_numbers > 0)
    if np.sum(finite_mask) >= 2:
        log_kappa = np.log(condition_numbers[finite_mask])
        N_fit = N_arr[finite_mask]
        coeffs = np.polyfit(N_fit, log_kappa, 1)
        growth_base = float(np.exp(coeffs[0]))
    else:
        growth_base = 1.0

    is_exponential = growth_base > 1.5

    return {
        'N_values': N_arr.tolist(),
        'condition_numbers': condition_numbers.tolist(),
        'growth_base': growth_base,
        'is_exponential': is_exponential,
    }
