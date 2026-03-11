"""
Physical Bridge: Operator Autocorrelation <-> Observable Statistics
===================================================================

Implements the formal Physical Bridge theorem from Paper [11]:

    C_QBER(tau) ~ C_op(tau)

where C_op(t) = Tr(O(0) O(t)) / Tr(O^2) is the operator autocorrelation
and C_QBER(tau) is the QBER autocorrelation measured by Alice and Bob.

The bridge is established via the susceptibility chi(omega) that connects
the Hamiltonian dynamics to the observable statistics. The key result
(Paper [11], Theorem 1) shows that the Pearson correlation between
C_op and C_QBER exceeds r > 0.997 across all tested Hamiltonian families
(Heisenberg, XXZ, SYK) and system sizes N = 12-18.

This module provides:
    - compute_operator_autocorrelation: C_op(t) from eigendecomposition
    - compute_qber_autocorrelation: C_QBER(tau) from QBER time series
    - bridge_transform: Map between operator and observable spaces
    - verify_bridge_correlation: Compute Pearson r between C_op and C_QBER
    - compute_susceptibility: Frequency-domain susceptibility chi(omega)

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
from scipy.signal import correlate


def compute_operator_autocorrelation(H, O, times, method="eigen"):
    """
    Compute the operator autocorrelation C_op(t).

    C_op(t) = Tr(O^dag e^{iHt} O e^{-iHt}) / (d * ||O||^2)

    where d is the Hilbert space dimension and ||O||^2 = Tr(O^dag O)/d.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hermitian Hamiltonian matrix.
    O : ndarray, shape (d, d)
        Observable operator.
    times : ndarray, shape (n_t,)
        Time points at which to evaluate C_op.
    method : str
        Computation method: "eigen" (default, fast for many time points)
        or "expm" (direct matrix exponential, more numerically stable).

    Returns
    -------
    C_op : ndarray, shape (n_t,)
        Normalized operator autocorrelation. C_op(0) = 1.

    Notes
    -----
    The eigendecomposition method is O(d^3) for setup and O(d^2 * n_t)
    for evaluation. For large d, consider using sparse methods.

    Reference: Paper [11], Eq. (3)
    """
    H = np.asarray(H, dtype=complex)
    O = np.asarray(O, dtype=complex)
    times = np.asarray(times, dtype=float)
    d = H.shape[0]

    if method == "eigen":
        E, V = eigh(H)
        return _autocorrelation_eigen(E, V, O, times, d)
    elif method == "expm":
        return _autocorrelation_expm(H, O, times, d)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'eigen' or 'expm'.")


def _autocorrelation_eigen(E, V, O, times, d):
    """Eigendecomposition-based autocorrelation (fast)."""
    O_norm_sq = np.trace(O.conj().T @ O).real / d
    if O_norm_sq < 1e-30:
        return np.zeros(len(times))

    # Transform O to eigenbasis
    O_eig = V.conj().T @ O @ V
    # Precompute |O_eig|^2 element-wise
    O_eig_sq = O_eig.conj() * O_eig
    # Energy differences
    E_diff = np.subtract.outer(E, E)

    C = np.zeros(len(times))
    for it, t in enumerate(times):
        phases = np.exp(1j * E_diff * t)
        C[it] = np.sum(O_eig_sq * phases).real / d / O_norm_sq
    return C


def _autocorrelation_expm(H, O, times, d):
    """Direct matrix exponential autocorrelation (stable)."""
    from scipy.linalg import expm

    O_norm_sq = np.trace(O.conj().T @ O).real / d
    if O_norm_sq < 1e-30:
        return np.zeros(len(times))

    C = np.zeros(len(times))
    for it, t in enumerate(times):
        U = expm(-1j * H * t)
        O_t = U.conj().T @ O @ U
        C[it] = np.trace(O.conj().T @ O_t).real / d / O_norm_sq
    return C


def compute_qber_autocorrelation(qber, max_lag=None, normalize=True):
    """
    Compute the QBER autocorrelation C_QBER(tau).

    Uses the standard unbiased estimator:
        C_QBER(tau) = (1/(N-tau)) sum_{t=0}^{N-tau-1} delta_q(t) delta_q(t+tau)

    where delta_q(t) = q(t) - <q>.

    Parameters
    ----------
    qber : ndarray, shape (N,)
        QBER time series (after sidereal filtering recommended).
    max_lag : int or None
        Maximum lag. If None, uses N//2.
    normalize : bool
        If True, normalize so that C_QBER(0) = 1.

    Returns
    -------
    C_qber : ndarray, shape (max_lag,)
        QBER autocorrelation.

    Reference: Paper [11], Eq. (5)
    """
    qber = np.asarray(qber, dtype=float)
    N = len(qber)
    if max_lag is None:
        max_lag = N // 2

    delta_q = qber - np.mean(qber)

    # Use scipy correlate for efficiency
    full_corr = correlate(delta_q, delta_q, mode='full')
    # Normalize by (N - |lag|) for unbiased estimator
    lags = np.arange(-N + 1, N)
    norm_factor = N - np.abs(lags)
    norm_factor[norm_factor < 1] = 1
    full_corr = full_corr / norm_factor

    # Extract positive lags
    center = N - 1
    C_qber = full_corr[center:center + max_lag].copy()

    if normalize and abs(C_qber[0]) > 1e-30:
        C_qber = C_qber / C_qber[0]

    return C_qber


def bridge_transform(C_op, lanczos_slope, dt=1.0):
    """
    Transform operator autocorrelation to predicted QBER autocorrelation.

    The Physical Bridge (Paper [11], Theorem 1) establishes:
        C_QBER(tau) = C_op(tau) * |chi(tau)|^2

    where chi(tau) is the susceptibility function determined by the
    Lanczos slope. For the Gaussian regime (linear b_n growth):
        chi(tau) ~ exp(-alpha * tau^2)

    with alpha proportional to the Lanczos slope.

    Parameters
    ----------
    C_op : ndarray, shape (n_t,)
        Operator autocorrelation.
    lanczos_slope : float
        Average slope of the Lanczos coefficients (mean(diff(b_n))).
    dt : float
        Time step between samples.

    Returns
    -------
    C_predicted : ndarray, shape (n_t,)
        Predicted QBER autocorrelation.

    Reference: Paper [11], Theorem 1 and Eq. (8)
    """
    C_op = np.asarray(C_op, dtype=float)
    n = len(C_op)
    tau = np.arange(n) * dt

    # Susceptibility envelope (Gaussian regime)
    alpha = 0.5 * lanczos_slope ** 2
    chi_sq = np.exp(-2 * alpha * tau ** 2)

    C_predicted = C_op * chi_sq

    # Normalize
    if abs(C_predicted[0]) > 1e-30:
        C_predicted = C_predicted / C_predicted[0]

    return C_predicted


def verify_bridge_correlation(C_op, C_qber, max_lag=None):
    """
    Verify the Physical Bridge by computing the Pearson correlation
    between operator and QBER autocorrelations.

    Parameters
    ----------
    C_op : ndarray
        Operator autocorrelation.
    C_qber : ndarray
        QBER autocorrelation.
    max_lag : int or None
        Maximum lag to include. If None, uses min(len(C_op), len(C_qber)).

    Returns
    -------
    result : dict
        Keys:
        - 'r': Pearson correlation coefficient
        - 'p': p-value
        - 'rmse': Root mean square error
        - 'max_lag': Number of lags used
        - 'bridge_valid': True if r > 0.95

    Reference: Paper [11], Section IV (Numerical Validation)
    """
    C_op = np.asarray(C_op, dtype=float)
    C_qber = np.asarray(C_qber, dtype=float)

    if max_lag is None:
        max_lag = min(len(C_op), len(C_qber))
    else:
        max_lag = min(max_lag, len(C_op), len(C_qber))

    c1 = C_op[:max_lag]
    c2 = C_qber[:max_lag]

    # Require at least 3 points and non-constant arrays
    if max_lag < 3 or np.std(c1) < 1e-15 or np.std(c2) < 1e-15:
        return {
            'r': 0.0,
            'p': 1.0,
            'rmse': float(np.sqrt(np.mean((c1 - c2) ** 2))),
            'max_lag': max_lag,
            'bridge_valid': False,
        }

    r, p = pearsonr(c1, c2)
    rmse = float(np.sqrt(np.mean((c1 - c2) ** 2)))

    return {
        'r': float(r),
        'p': float(p),
        'rmse': rmse,
        'max_lag': max_lag,
        'bridge_valid': float(r) > 0.95,
    }


def compute_susceptibility(H, O, omega, eta=0.01):
    """
    Compute the frequency-domain susceptibility chi(omega).

    chi(omega) = sum_{m,n} |<m|O|n>|^2 / (omega - (E_m - E_n) + i*eta)

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian.
    O : ndarray, shape (d, d)
        Observable operator.
    omega : ndarray, shape (n_w,)
        Frequency points.
    eta : float
        Broadening parameter (default: 0.01).

    Returns
    -------
    chi : ndarray, shape (n_w,), complex
        Susceptibility function.

    Reference: Paper [11], Eq. (7)
    """
    H = np.asarray(H, dtype=complex)
    O = np.asarray(O, dtype=complex)
    omega = np.asarray(omega, dtype=float)

    E, V = eigh(H)
    O_eig = V.conj().T @ O @ V
    d = len(E)

    chi = np.zeros(len(omega), dtype=complex)
    for m in range(d):
        for n in range(d):
            weight = np.abs(O_eig[m, n]) ** 2
            if weight < 1e-30:
                continue
            delta_E = E[m] - E[n]
            chi += weight / (omega - delta_E + 1j * eta)

    return chi


def compute_lanczos_coefficients(H, O, n_steps=25):
    """
    Compute Lanczos coefficients b_n in operator space.

    This is a pure-numpy implementation (no QuTiP dependency) of the
    Krylov basis construction via the Lanczos algorithm.

    Starting from O_0 = O / ||O||, iteratively compute:
        A_n = i[H, O_n]
        b_n = ||A_n - b_{n-1} O_{n-1}||
        O_{n+1} = (A_n - b_{n-1} O_{n-1}) / b_n

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian matrix.
    O : ndarray, shape (d, d)
        Initial operator.
    n_steps : int
        Maximum number of Lanczos steps.

    Returns
    -------
    b_n : ndarray
        Lanczos coefficients.

    Reference: Parker et al., PRX 9, 041017 (2019); Paper [11], Eq. (1)
    """
    H = np.asarray(H, dtype=complex)
    O = np.asarray(O, dtype=complex)
    d = H.shape[0]

    norm = np.sqrt(np.trace(O.conj().T @ O).real / d)
    if norm < 1e-14:
        return np.array([])

    O_curr = O / norm
    O_prev = np.zeros_like(O)
    bs = []

    for n in range(n_steps):
        # Liouvillian action: L(O) = i[H, O]
        L_O = 1j * (H @ O_curr - O_curr @ H)

        # Orthogonalize
        if n > 0:
            L_O = L_O - bs[-1] * O_prev

        # Compute norm
        b_next = np.sqrt(np.trace(L_O.conj().T @ L_O).real / d)
        if b_next < 1e-12:
            break

        bs.append(float(b_next))
        O_prev = O_curr
        O_curr = L_O / b_next

    return np.array(bs)


def full_bridge_analysis(H, O, qber_residuum, times, n_lanczos=25):
    """
    Run the complete Physical Bridge analysis pipeline.

    1. Compute Lanczos coefficients from H and O
    2. Compute operator autocorrelation C_op(t)
    3. Compute QBER autocorrelation C_QBER(tau)
    4. Apply bridge transform
    5. Verify correlation

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian.
    O : ndarray, shape (d, d)
        Observable operator.
    qber_residuum : ndarray, shape (N,)
        QBER time series (after sidereal filtering).
    times : ndarray, shape (n_t,)
        Time points for operator autocorrelation.
    n_lanczos : int
        Number of Lanczos steps.

    Returns
    -------
    result : dict
        Comprehensive bridge analysis results:
        - 'b_n': Lanczos coefficients
        - 'lanczos_slope': Average slope of b_n
        - 'C_op': Operator autocorrelation
        - 'C_qber': QBER autocorrelation
        - 'C_predicted': Predicted QBER autocorrelation from bridge
        - 'bridge_r': Pearson correlation
        - 'bridge_p': p-value
        - 'bridge_valid': Whether bridge holds (r > 0.95)

    Reference: Paper [11], Section IV
    """
    # Step 1: Lanczos coefficients
    b_n = compute_lanczos_coefficients(H, O, n_lanczos)
    if len(b_n) < 2:
        return {
            'b_n': b_n,
            'lanczos_slope': 0.0,
            'C_op': np.array([]),
            'C_qber': np.array([]),
            'C_predicted': np.array([]),
            'bridge_r': 0.0,
            'bridge_p': 1.0,
            'bridge_valid': False,
        }

    lanczos_slope = float(np.mean(np.diff(b_n)))

    # Step 2: Operator autocorrelation
    C_op = compute_operator_autocorrelation(H, O, times)

    # Step 3: QBER autocorrelation
    max_lag = min(len(times), len(qber_residuum) // 2)
    C_qber = compute_qber_autocorrelation(qber_residuum, max_lag=max_lag)

    # Step 4: Bridge transform
    C_predicted = bridge_transform(C_op[:max_lag], lanczos_slope)

    # Step 5: Verify
    verification = verify_bridge_correlation(C_predicted, C_qber)

    return {
        'b_n': b_n,
        'lanczos_slope': lanczos_slope,
        'C_op': C_op,
        'C_qber': C_qber,
        'C_predicted': C_predicted,
        'bridge_r': verification['r'],
        'bridge_p': verification['p'],
        'bridge_valid': verification['bridge_valid'],
    }
