"""
Open-System Physical Bridge: Lindblad Extension
=================================================

Extends the Physical Bridge to open quantum systems governed by the
Lindblad master equation. This is the core result of Paper [12]:

    d rho/dt = -i[H, rho] + sum_k gamma_k (L_k rho L_k^dag
               - 0.5 {L_k^dag L_k, rho})

The adjoint (Heisenberg picture) Lindbladian governs operator evolution:

    d O/dt = L_adj(O) = i[H, O] + sum_k gamma_k (L_k^dag O L_k
             - 0.5 {L_k^dag L_k, O})

The open-system bridge shows that the operator autocorrelation under
Lindblad dynamics still correlates with observable statistics (r > 0.95),
but with an additional exponential damping envelope from decoherence.

This module provides:
    - build_adjoint_lindbladian: Construct the adjoint Lindbladian superoperator
    - lindblad_evolve: Evolve an operator under Lindblad dynamics
    - open_system_autocorrelation: C_op(t) with dissipation
    - compute_decoherence_envelope: Extract the damping envelope
    - bridge_with_dissipation: Physical Bridge including noise effects
    - compare_open_closed: Compare open and closed system autocorrelations

References:
    [12] D. Suess, "Open-System Physical Bridge: Extending the Krylov
         Eavesdropper Detector to Lindblad Dynamics," Zenodo, 2026.
         DOI: 10.5281/zenodo.18959827

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
from scipy.linalg import eig, expm
from scipy.stats import pearsonr


def build_adjoint_lindbladian(H, jump_ops, gamma_list):
    """
    Construct the adjoint Lindbladian superoperator L_adj.

    In the Heisenberg picture, operator evolution is governed by:
        d O/dt = L_adj(O)

    where L_adj acts on vectorized operators:
        L_adj = i(H x I - I x H^T)
              + sum_k gamma_k (L_k^dag x L_k^*
                - 0.5 L_k^dag L_k x I - 0.5 I x (L_k^dag L_k)^T)

    Parameters
    ----------
    H : ndarray, shape (d, d)
        System Hamiltonian.
    jump_ops : list of ndarray
        Lindblad jump operators L_k, each shape (d, d).
    gamma_list : list of float
        Dissipation rates gamma_k for each jump operator.

    Returns
    -------
    L_adj : ndarray, shape (d^2, d^2)
        Adjoint Lindbladian superoperator.

    Notes
    -----
    The vectorization convention is column-major (Fortran order):
        vec(O) = O.flatten() with O[i,j] -> index i + j*d

    Reference: Paper [12], Eq. (4)
    """
    H = np.asarray(H, dtype=complex)
    d = H.shape[0]
    I_d = np.eye(d, dtype=complex)

    # Unitary part: i(H x I - I x H^T)
    L_adj = 1j * np.kron(H, I_d) - 1j * np.kron(I_d, H.T)

    # Dissipative part
    for Lk, gk in zip(jump_ops, gamma_list):
        Lk = np.asarray(Lk, dtype=complex)
        Lk_dag = Lk.conj().T
        LdL = Lk_dag @ Lk

        L_adj += gk * (
            np.kron(Lk_dag, Lk.conj())
            - 0.5 * np.kron(LdL, I_d)
            - 0.5 * np.kron(I_d, LdL.T)
        )

    return L_adj


def lindblad_evolve(L_adj, O, times, method="eigen"):
    """
    Evolve an operator O(t) under the adjoint Lindbladian.

    O(t)_vec = exp(L_adj * t) @ O(0)_vec

    Parameters
    ----------
    L_adj : ndarray, shape (d^2, d^2)
        Adjoint Lindbladian superoperator.
    O : ndarray, shape (d, d)
        Initial operator.
    times : ndarray, shape (n_t,)
        Time points.
    method : str
        "eigen" for eigendecomposition (fast for many time points),
        "expm" for direct matrix exponential (more stable).

    Returns
    -------
    O_evolved : ndarray, shape (n_t, d, d)
        Time-evolved operator at each time point.

    Reference: Paper [12], Eq. (5)
    """
    O = np.asarray(O, dtype=complex)
    times = np.asarray(times, dtype=float)
    d = O.shape[0]
    O_vec = O.flatten()

    if method == "eigen":
        vals, R = eig(L_adj)
        R_inv = np.linalg.inv(R)
        c_right = R_inv @ O_vec

        O_evolved = np.zeros((len(times), d, d), dtype=complex)
        for i, t in enumerate(times):
            O_t_vec = R @ (np.exp(vals * t) * c_right)
            O_evolved[i] = O_t_vec.reshape(d, d)
    elif method == "expm":
        O_evolved = np.zeros((len(times), d, d), dtype=complex)
        for i, t in enumerate(times):
            prop = expm(L_adj * t)
            O_t_vec = prop @ O_vec
            O_evolved[i] = O_t_vec.reshape(d, d)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'eigen' or 'expm'.")

    return O_evolved


def open_system_autocorrelation(H, O, times, jump_ops=None, gamma_list=None,
                                 method="eigen"):
    """
    Compute the operator autocorrelation C_op(t) under Lindblad dynamics.

    C_op(t) = Tr(O^dag O(t)) / (d * ||O||^2)

    where O(t) evolves under the adjoint Lindbladian.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        System Hamiltonian.
    O : ndarray, shape (d, d)
        Observable operator.
    times : ndarray, shape (n_t,)
        Time points.
    jump_ops : list of ndarray or None
        Lindblad jump operators. If None, computes closed-system result.
    gamma_list : list of float or None
        Dissipation rates. Must match length of jump_ops.
    method : str
        Computation method ("eigen" or "expm").

    Returns
    -------
    C_op : ndarray, shape (n_t,)
        Operator autocorrelation (real-valued).

    Reference: Paper [12], Eq. (6)
    """
    H = np.asarray(H, dtype=complex)
    O = np.asarray(O, dtype=complex)
    times = np.asarray(times, dtype=float)
    d = H.shape[0]

    O_norm_sq = np.trace(O.conj().T @ O).real / d
    if O_norm_sq < 1e-30:
        return np.zeros(len(times))

    if jump_ops is None or gamma_list is None:
        jump_ops = []
        gamma_list = []

    L_adj = build_adjoint_lindbladian(H, jump_ops, gamma_list)
    O_vec = O.flatten()

    if method == "eigen":
        vals, R = eig(L_adj)
        R_inv = np.linalg.inv(R)
        c_left = O_vec.conj() @ R
        c_right = R_inv @ O_vec
        coeff = c_left * c_right

        C = np.zeros(len(times))
        for i, t in enumerate(times):
            C[i] = np.sum(coeff * np.exp(vals * t)).real / d / O_norm_sq
    elif method == "expm":
        C = np.zeros(len(times))
        for i, t in enumerate(times):
            prop = expm(L_adj * t)
            O_t_vec = prop @ O_vec
            C[i] = (O_vec.conj() @ O_t_vec).real / d / O_norm_sq
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return C


def compute_decoherence_envelope(C_open, C_closed):
    """
    Extract the decoherence envelope by comparing open and closed system
    autocorrelations.

    envelope(t) = C_open(t) / C_closed(t)

    For pure dephasing, this is approximately exp(-Gamma * t) where
    Gamma is the total dephasing rate.

    Parameters
    ----------
    C_open : ndarray, shape (n_t,)
        Open-system autocorrelation.
    C_closed : ndarray, shape (n_t,)
        Closed-system autocorrelation.

    Returns
    -------
    result : dict
        Keys:
        - 'envelope': The decoherence envelope
        - 'decay_rate': Fitted exponential decay rate Gamma
        - 'envelope_ratio': Late-time to early-time ratio (diagnostic)

    Reference: Paper [12], Section III.B
    """
    C_open = np.asarray(C_open, dtype=float)
    C_closed = np.asarray(C_closed, dtype=float)
    n = min(len(C_open), len(C_closed))

    # Avoid division by zero
    safe_closed = np.where(np.abs(C_closed[:n]) > 1e-15,
                           C_closed[:n], 1e-15)
    envelope = C_open[:n] / safe_closed

    # Fit exponential decay: log(|envelope|) ~ -Gamma * t
    abs_env = np.abs(envelope)
    abs_env = np.clip(abs_env, 1e-15, None)
    log_env = np.log(abs_env)

    # Use points where envelope is still significant
    valid = abs_env > 0.01
    if np.sum(valid) >= 3:
        t_idx = np.arange(n)[valid]
        coeffs = np.polyfit(t_idx, log_env[valid], 1)
        decay_rate = max(-coeffs[0], 0.0)
    else:
        decay_rate = 0.0

    # Envelope ratio: late-time / early-time
    n_early = max(1, n // 5)
    n_late = max(1, n // 5)
    early = np.mean(np.abs(C_open[:n_early]))
    late = np.mean(np.abs(C_open[n - n_late:]))
    envelope_ratio = late / early if early > 1e-15 else 0.0

    return {
        'envelope': envelope,
        'decay_rate': float(decay_rate),
        'envelope_ratio': float(envelope_ratio),
    }


def bridge_with_dissipation(H, O, times, jump_ops, gamma_list,
                             qber_residuum=None):
    """
    Full open-system Physical Bridge analysis.

    Computes both closed and open system autocorrelations, extracts
    the decoherence envelope, and optionally verifies against QBER data.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian.
    O : ndarray, shape (d, d)
        Observable.
    times : ndarray, shape (n_t,)
        Time points.
    jump_ops : list of ndarray
        Jump operators.
    gamma_list : list of float
        Dissipation rates.
    qber_residuum : ndarray or None
        If provided, also computes bridge correlation with QBER.

    Returns
    -------
    result : dict
        Comprehensive open-system bridge analysis:
        - 'C_closed': Closed-system autocorrelation
        - 'C_open': Open-system autocorrelation
        - 'envelope': Decoherence envelope
        - 'decay_rate': Exponential decay rate
        - 'bridge_r': Correlation between C_open and C_closed
        - 'bridge_p': p-value
        - 'qber_r': Correlation with QBER (if provided)
        - 'qber_p': p-value for QBER correlation

    Reference: Paper [12], Section IV
    """
    # Closed system
    C_closed = open_system_autocorrelation(H, O, times)

    # Open system
    C_open = open_system_autocorrelation(H, O, times, jump_ops, gamma_list)

    # Decoherence envelope
    env_result = compute_decoherence_envelope(C_open, C_closed)

    # Correlation between open and closed
    n = min(len(C_open), len(C_closed))
    if n >= 3 and np.std(C_open[:n]) > 1e-15 and np.std(C_closed[:n]) > 1e-15:
        r_oc, p_oc = pearsonr(C_open[:n], C_closed[:n])
    else:
        r_oc, p_oc = 0.0, 1.0

    result = {
        'C_closed': C_closed,
        'C_open': C_open,
        'envelope': env_result['envelope'],
        'decay_rate': env_result['decay_rate'],
        'envelope_ratio': env_result['envelope_ratio'],
        'bridge_r': float(r_oc),
        'bridge_p': float(p_oc),
    }

    # Optional QBER verification
    if qber_residuum is not None:
        from .physical_bridge import compute_qber_autocorrelation
        max_lag = min(n, len(qber_residuum) // 2)
        C_qber = compute_qber_autocorrelation(qber_residuum, max_lag=max_lag)
        n_q = min(len(C_open), len(C_qber))
        if n_q >= 3 and np.std(C_qber[:n_q]) > 1e-15:
            r_q, p_q = pearsonr(C_open[:n_q], C_qber[:n_q])
        else:
            r_q, p_q = 0.0, 1.0
        result['qber_r'] = float(r_q)
        result['qber_p'] = float(p_q)
        result['C_qber'] = C_qber

    return result


def compare_open_closed(H, O, times, noise_configs):
    """
    Compare autocorrelations across multiple noise configurations.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hamiltonian.
    O : ndarray, shape (d, d)
        Observable.
    times : ndarray, shape (n_t,)
        Time points.
    noise_configs : list of dict
        Each dict has keys:
        - 'name': str, descriptive name
        - 'jump_ops': list of ndarray
        - 'gamma_list': list of float

    Returns
    -------
    results : list of dict
        For each noise configuration:
        - 'name': Configuration name
        - 'C_open': Open-system autocorrelation
        - 'decay_rate': Decoherence rate
        - 'bridge_r': Correlation with closed-system result

    Reference: Paper [12], Section V (Numerical Results)
    """
    # Closed system reference
    C_closed = open_system_autocorrelation(H, O, times)

    results = []
    for config in noise_configs:
        C_open = open_system_autocorrelation(
            H, O, times, config['jump_ops'], config['gamma_list']
        )
        env = compute_decoherence_envelope(C_open, C_closed)

        n = min(len(C_open), len(C_closed))
        if n >= 3 and np.std(C_open[:n]) > 1e-15 and np.std(C_closed[:n]) > 1e-15:
            r, _ = pearsonr(C_open[:n], C_closed[:n])
        else:
            r = 0.0

        results.append({
            'name': config['name'],
            'C_open': C_open,
            'decay_rate': env['decay_rate'],
            'envelope_ratio': env['envelope_ratio'],
            'bridge_r': float(r),
        })

    return results
