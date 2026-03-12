"""
Universality of the Krylov Framework
======================================

Tests the Physical Bridge across multiple Hamiltonian families to
establish universality. Paper [11] shows that the bridge holds with
r > 0.997 for:

    - Heisenberg XXX chains (integrable to chaotic crossover)
    - Ising + transverse field (chaotic regime)
    - XXZ models (anisotropic exchange)
    - XY model (integrable-like)
    - Random-field Heisenberg (disorder-driven)
    - Mixed coupling (Paper [4] structure)
    - Frustrated J1-J2 (geometric frustration)
    - Strong disorder (MBL-like regime)
    - SYK models (maximally chaotic, all-to-all coupling)

This module provides built-in Hamiltonian generators and a testing
framework to verify the Physical Bridge for arbitrary Hamiltonians.

This module provides:
    - heisenberg_chain: Build Heisenberg chain Hamiltonian (numpy)
    - ising_chaotic: Build Ising + transverse field Hamiltonian
    - xxz_chain: Build XXZ chain Hamiltonian
    - xy_model: Build XY model Hamiltonian
    - random_field_heisenberg: Build random-field Heisenberg Hamiltonian
    - mixed_coupling: Build mixed-coupling Hamiltonian (Paper [4])
    - frustrated_j1j2: Build frustrated J1-J2 Hamiltonian
    - strong_disorder: Build strong-disorder Hamiltonian
    - syk_model: Build SYK4 random Hamiltonian
    - test_hamiltonian_family: Test bridge for a Hamiltonian family
    - compute_universality_score: Score across multiple families
    - supported_families: List of built-in Hamiltonian families

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
from scipy.stats import pearsonr


# ── Pauli matrices ──────────────────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_list(ops):
    """Kronecker product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _site_op(op, site, N):
    """Place a single-qubit operator at a given site in an N-qubit chain."""
    ops = [_I2] * N
    ops[site] = op
    return _kron_list(ops)


def heisenberg_chain(N, J=1.0, h=0.5, hz=0.12, hx=0.08,
                     kappa=0.45, g=0.5):
    """
    Build the symmetry-broken Heisenberg chain Hamiltonian.

    This is the same Hamiltonian used throughout the QKD framework,
    implemented in pure numpy (no QuTiP dependency).

    Parameters
    ----------
    N : int
        Number of qubits.
    J : float
        Heisenberg coupling between qubits 0-1.
    h : float
        Not used (kept for API compatibility).
    hz : float
        Longitudinal field.
    hx : float
        Transverse field.
    kappa : float
        ZZ coupling between qubits 1-2.
    g : float
        XX chain coupling for qubits 2 through N-1.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    # Heisenberg coupling on qubits 0-1
    for pauli in [_sx, _sy, _sz]:
        H += J * _site_op(pauli, 0, N) @ _site_op(pauli, 1, N)

    # XX chain coupling for qubits 2 through N-1
    for i in range(2, N - 1):
        H += g * _site_op(_sx, i, N) @ _site_op(_sx, i + 1, N)

    # ZZ coupling between qubits 1-2
    if N >= 3:
        H += kappa * _site_op(_sz, 1, N) @ _site_op(_sz, 2, N)

    # Magnetic fields
    for i in range(N):
        H += hz * _site_op(_sz, i, N) + hx * _site_op(_sx, i, N)

    return H


def ising_chaotic(N, J=1.0, hx=1.05, hz=0.5):
    """
    Build the Ising + transverse + longitudinal field Hamiltonian.

    H = J sum_i Z_i Z_{i+1} + hx sum_i X_i + hz sum_i Z_i

    With hx/J ~ 1.05 and hz > 0, the model is in the chaotic regime
    (non-integrable). Paper [11], Section 4.2.

    Parameters
    ----------
    N : int
        Number of qubits.
    J : float
        ZZ coupling strength.
    hx : float
        Transverse field.
    hz : float
        Longitudinal field.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N - 1):
        H += J * _site_op(_sz, i, N) @ _site_op(_sz, i + 1, N)

    for i in range(N):
        H += hx * _site_op(_sx, i, N) + hz * _site_op(_sz, i, N)

    return H


def xxz_chain(N, Jxy=1.0, Jz=0.5, hz=0.1):
    """
    Build the XXZ chain Hamiltonian.

    H = sum_i [Jxy (X_i X_{i+1} + Y_i Y_{i+1}) + Jz Z_i Z_{i+1}]
      + sum_i hz Z_i

    Parameters
    ----------
    N : int
        Number of qubits.
    Jxy : float
        XY coupling strength.
    Jz : float
        Z coupling strength (anisotropy parameter).
    hz : float
        Longitudinal field.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N - 1):
        H += Jxy * (_site_op(_sx, i, N) @ _site_op(_sx, i + 1, N)
                  + _site_op(_sy, i, N) @ _site_op(_sy, i + 1, N))
        H += Jz * _site_op(_sz, i, N) @ _site_op(_sz, i + 1, N)

    for i in range(N):
        H += hz * _site_op(_sz, i, N)

    return H


def xy_model(N, Jx=1.0, Jy=0.5, hz=0.3, hx=0.1):
    """
    Build the XY model Hamiltonian with fields.

    H = sum_i [Jx X_i X_{i+1} + Jy Y_i Y_{i+1}] + sum_i [hz Z_i + hx X_i]

    Integrable-like regime. Paper [11], Section 4.2.

    Parameters
    ----------
    N : int
        Number of qubits.
    Jx : float
        XX coupling strength.
    Jy : float
        YY coupling strength.
    hz : float
        Longitudinal field.
    hx : float
        Transverse field.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N - 1):
        H += Jx * _site_op(_sx, i, N) @ _site_op(_sx, i + 1, N)
        H += Jy * _site_op(_sy, i, N) @ _site_op(_sy, i + 1, N)

    for i in range(N):
        H += hz * _site_op(_sz, i, N) + hx * _site_op(_sx, i, N)

    return H


def random_field_heisenberg(N, J=1.0, W=1.5, seed=42):
    """
    Build the random-field Heisenberg Hamiltonian.

    H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
      + sum_i h_i Z_i

    where h_i ~ Uniform[-W, W]. Paper [11], Section 4.2.

    Parameters
    ----------
    N : int
        Number of qubits.
    J : float
        Heisenberg coupling.
    W : float
        Disorder strength.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    rng = np.random.RandomState(seed)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N - 1):
        for pauli in [_sx, _sy, _sz]:
            H += J * _site_op(pauli, i, N) @ _site_op(pauli, i + 1, N)

    fields = W * (2 * rng.rand(N) - 1)
    for i in range(N):
        H += fields[i] * _site_op(_sz, i, N)

    return H


def mixed_coupling(N, J=1.0, g=0.5, kappa=0.45, hz=0.12, hx=0.08):
    """
    Build the mixed-coupling Hamiltonian (Paper [4] structure).

    Heisenberg coupling on qubits 0-1, XX chain on qubits 2+,
    ZZ coupling on qubits 1-2, plus fields. Paper [11], Section 4.2.

    Parameters
    ----------
    N : int
        Number of qubits.
    J : float
        Heisenberg coupling between qubits 0-1.
    g : float
        XX chain coupling for qubits 2+.
    kappa : float
        ZZ coupling between qubits 1-2.
    hz : float
        Longitudinal field.
    hx : float
        Transverse field.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    # Heisenberg on qubits 0-1
    for pauli in [_sx, _sy, _sz]:
        H += J * _site_op(pauli, 0, N) @ _site_op(pauli, 1, N)

    # XX chain for qubits 2+
    for i in range(2, N - 1):
        H += g * _site_op(_sx, i, N) @ _site_op(_sx, i + 1, N)

    # ZZ coupling 1-2
    if N >= 3:
        H += kappa * _site_op(_sz, 1, N) @ _site_op(_sz, 2, N)

    # Fields
    for i in range(N):
        H += hz * _site_op(_sz, i, N) + hx * _site_op(_sx, i, N)

    return H


def frustrated_j1j2(N, J1=1.0, J2=0.5, hz=0.1):
    """
    Build the frustrated J1-J2 Hamiltonian.

    H = J1 sum_i S_i . S_{i+1} + J2 sum_i S_i . S_{i+2} + hz sum_i Z_i

    Nearest-neighbor + next-nearest-neighbor Heisenberg coupling.
    Geometric frustration drives the system toward chaos.
    Paper [11], Section 4.2.

    Parameters
    ----------
    N : int
        Number of qubits.
    J1 : float
        Nearest-neighbor coupling.
    J2 : float
        Next-nearest-neighbor coupling.
    hz : float
        Longitudinal field.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    # J1: nearest-neighbor
    for i in range(N - 1):
        for pauli in [_sx, _sy, _sz]:
            H += J1 * _site_op(pauli, i, N) @ _site_op(pauli, i + 1, N)

    # J2: next-nearest-neighbor
    for i in range(N - 2):
        for pauli in [_sx, _sy, _sz]:
            H += J2 * _site_op(pauli, i, N) @ _site_op(pauli, i + 2, N)

    # Field
    for i in range(N):
        H += hz * _site_op(_sz, i, N)

    return H


def strong_disorder(N, J=0.5, W=4.0, seed=123):
    """
    Build the strong-disorder Hamiltonian (MBL-like regime).

    H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
      + sum_i h_i Z_i

    where h_i ~ Uniform[-W, W] with W >> J. Paper [11], Section 4.2.

    Parameters
    ----------
    N : int
        Number of qubits.
    J : float
        Heisenberg coupling (weak).
    W : float
        Disorder strength (strong).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    H : ndarray, shape (2^N, 2^N)
        Hamiltonian matrix.
    """
    rng = np.random.RandomState(seed)
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N - 1):
        for pauli in [_sx, _sy, _sz]:
            H += J * _site_op(pauli, i, N) @ _site_op(pauli, i + 1, N)

    fields = W * (2 * rng.rand(N) - 1)
    for i in range(N):
        H += fields[i] * _site_op(_sz, i, N)

    return H


def syk_model(N_majorana, seed=None):
    """
    Build the SYK4 (Sachdev-Ye-Kitaev) random Hamiltonian.

    H = sum_{i<j<k<l} J_{ijkl} gamma_i gamma_j gamma_k gamma_l

    where gamma_i are Majorana fermion operators and J_{ijkl} are
    Gaussian random couplings with variance 6/N^3.

    Parameters
    ----------
    N_majorana : int
        Number of Majorana fermions (must be even). The Hilbert space
        dimension is 2^(N_majorana/2).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    H : ndarray, shape (d, d)
        SYK Hamiltonian.
    N_qubits : int
        Number of qubits (= N_majorana // 2).
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    N_q = N_majorana // 2
    dim = 2 ** N_q

    # Build Majorana operators via Jordan-Wigner
    gammas = []
    for i in range(N_q):
        ops_x = [_sz] * i + [_sx] + [_I2] * (N_q - i - 1)
        gammas.append(_kron_list(ops_x))
        ops_y = [_sz] * i + [_sy] + [_I2] * (N_q - i - 1)
        gammas.append(_kron_list(ops_y))

    # SYK4 Hamiltonian
    H = np.zeros((dim, dim), dtype=complex)
    J_var = 6.0 / N_majorana ** 3

    for i in range(N_majorana):
        for j in range(i + 1, N_majorana):
            for k in range(j + 1, N_majorana):
                for l in range(k + 1, N_majorana):
                    J_ijkl = rng.normal(0, np.sqrt(J_var))
                    H += J_ijkl * gammas[i] @ gammas[j] @ gammas[k] @ gammas[l]

    return H, N_q


def test_hamiltonian_family(build_fn, N_values, O_builder=None,
                             n_lanczos=20, n_times=40, dt=0.15):
    """
    Test the Physical Bridge for a family of Hamiltonians.

    For each system size N, computes:
    1. Lanczos coefficients b_n
    2. Operator autocorrelation C_op(t)
    3. Gaussian template from b_n
    4. Pearson correlation between C_op and template

    Parameters
    ----------
    build_fn : callable
        Function N -> H that builds the Hamiltonian.
    N_values : list of int
        System sizes to test.
    O_builder : callable or None
        Function N -> O that builds the observable. If None, uses
        sigma_z on qubit 0.
    n_lanczos : int
        Number of Lanczos steps.
    n_times : int
        Number of time points.
    dt : float
        Time step.

    Returns
    -------
    result : dict
        Keys:
        - 'N_values': System sizes tested
        - 'correlations': Pearson r for each N
        - 'slopes': Lanczos slopes for each N
        - 'mean_r': Mean correlation across all N
        - 'universality_holds': True if mean r > 0.95

    Reference: Paper [11], Section VI (Universality Tests)
    """
    from .physical_bridge import (
        compute_operator_autocorrelation,
        compute_lanczos_coefficients,
    )

    times = np.arange(n_times) * dt
    correlations = []
    slopes = []

    for N in N_values:
        H = build_fn(N)
        if isinstance(H, tuple):
            H, _ = H  # SYK returns (H, N_qubits)

        d = H.shape[0]
        N_q = int(np.log2(d))

        if O_builder is not None:
            O = O_builder(N)
        else:
            O = _site_op(_sz, 0, N_q)

        # Lanczos coefficients
        b_n = compute_lanczos_coefficients(H, O, n_lanczos)
        if len(b_n) < 3:
            correlations.append(0.0)
            slopes.append(0.0)
            continue

        slope = float(np.mean(np.diff(b_n)))
        slopes.append(slope)

        # Operator autocorrelation
        C_op = compute_operator_autocorrelation(H, O, times)

        # Gaussian template
        template = np.exp(-0.5 * (slope * times) ** 2)

        # Correlation (use early-time region where both are significant)
        n_compare = min(len(C_op), len(template))
        c1 = C_op[:n_compare]
        c2 = template[:n_compare]

        if np.std(c1) > 1e-15 and np.std(c2) > 1e-15:
            r, _ = pearsonr(c1, c2)
            correlations.append(float(r))
        else:
            correlations.append(0.0)

    correlations = np.array(correlations)
    mean_r = float(np.mean(correlations)) if len(correlations) > 0 else 0.0

    return {
        'N_values': list(N_values),
        'correlations': correlations.tolist(),
        'slopes': slopes,
        'mean_r': mean_r,
        'universality_holds': mean_r > 0.95,
    }


def compute_universality_score(family_results):
    """
    Compute an overall universality score from multiple family tests.

    Parameters
    ----------
    family_results : list of dict
        Results from test_hamiltonian_family for different families.

    Returns
    -------
    score : dict
        Keys:
        - 'overall_r': Mean correlation across all families
        - 'n_families': Number of families tested
        - 'n_passing': Number of families with mean r > 0.95
        - 'universality_confirmed': True if all families pass
        - 'per_family': Summary per family

    Reference: Paper [11], Section VI.D
    """
    per_family = []
    all_r = []

    for result in family_results:
        mean_r = result.get('mean_r', 0.0)
        all_r.append(mean_r)
        per_family.append({
            'N_values': result.get('N_values', []),
            'mean_r': mean_r,
            'passes': mean_r > 0.95,
        })

    overall_r = float(np.mean(all_r)) if len(all_r) > 0 else 0.0
    n_passing = sum(1 for r in all_r if r > 0.95)

    return {
        'overall_r': overall_r,
        'n_families': len(family_results),
        'n_passing': n_passing,
        'universality_confirmed': n_passing == len(family_results),
        'per_family': per_family,
    }


def supported_families():
    """
    List the built-in Hamiltonian families available for testing.

    Returns
    -------
    families : list of dict
        Each dict has keys:
        - 'name': Family name
        - 'builder': Callable to build the Hamiltonian
        - 'description': Brief description
        - 'min_N': Minimum system size
        - 'max_N_recommended': Recommended maximum for testing
    """
    return [
        {
            'name': 'heisenberg',
            'builder': heisenberg_chain,
            'description': (
                'Symmetry-broken Heisenberg chain with crossover regime. '
                'The primary model used in the QKD framework.'
            ),
            'min_N': 3,
            'max_N_recommended': 10,
        },
        {
            'name': 'ising_chaotic',
            'builder': ising_chaotic,
            'description': (
                'Ising + transverse + longitudinal field in the chaotic '
                'regime (non-integrable). Paper [11], Section 4.2.'
            ),
            'min_N': 3,
            'max_N_recommended': 10,
        },
        {
            'name': 'xxz',
            'builder': xxz_chain,
            'description': (
                'XXZ chain with tunable anisotropy. Interpolates between '
                'XX model (Jz=0) and Heisenberg (Jz=Jxy).'
            ),
            'min_N': 3,
            'max_N_recommended': 10,
        },
        {
            'name': 'xy',
            'builder': xy_model,
            'description': (
                'XY model with fields. Integrable-like regime. '
                'Paper [11], Section 4.2.'
            ),
            'min_N': 3,
            'max_N_recommended': 10,
        },
        {
            'name': 'random_field',
            'builder': random_field_heisenberg,
            'description': (
                'Random-field Heisenberg chain. Disorder-driven regime. '
                'Paper [11], Section 4.2.'
            ),
            'min_N': 3,
            'max_N_recommended': 10,
        },
        {
            'name': 'mixed_coupling',
            'builder': mixed_coupling,
            'description': (
                'Mixed-coupling Hamiltonian from Paper [4]. Heisenberg '
                'on qubits 0-1, XX chain on 2+, ZZ on 1-2.'
            ),
            'min_N': 4,
            'max_N_recommended': 10,
        },
        {
            'name': 'frustrated_j1j2',
            'builder': frustrated_j1j2,
            'description': (
                'Frustrated J1-J2 Heisenberg chain with next-nearest-neighbor '
                'coupling. Geometric frustration drives chaos. Paper [11].'
            ),
            'min_N': 4,
            'max_N_recommended': 10,
        },
        {
            'name': 'strong_disorder',
            'builder': strong_disorder,
            'description': (
                'Strong-disorder Heisenberg chain (MBL-like regime). '
                'W >> J. Paper [11], Section 4.2.'
            ),
            'min_N': 3,
            'max_N_recommended': 10,
        },
        {
            'name': 'syk4',
            'builder': lambda N: syk_model(2 * N, seed=42)[0],
            'description': (
                'SYK4 model with all-to-all random coupling. Maximally '
                'chaotic with Lyapunov exponent saturating the MSS bound.'
            ),
            'min_N': 3,
            'max_N_recommended': 8,
        },
    ]
