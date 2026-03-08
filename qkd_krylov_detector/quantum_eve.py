"""
Quantum Eve Simulation — Korrigierte Version
=============================================

Implements the quantum eavesdropper model from Paper [6], Section IV.
Eve couples her own quantum system to the QKD channel via a unitary
Hamiltonian interaction. The key insight: Alice and Bob observe only
the *reduced* channel dynamics after tracing out Eve's qubits.

The module provides:
    - build_total_hamiltonian: Construct channel + Eve + coupling Hamiltonian
    - compute_channel_autocorrelation: C(t) via expectation values
    - compute_reduced_autocorrelation: C(t) via partial trace (correct method)
    - make_quantum_eve_qber: QBER time series with quantum Eve signature
    - make_classical_eve_qber: Classical iid Eve for comparison
    - compute_eve_detection_stats: Detection statistics (AUC, separation, forgery rate)

Notebook correspondence:
    quantum_eve_v2.py — Sections 1–8

References:
    [6] D. Süß, "Quantum Scrambling as a Cryptographic Resource,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18889224
"""

import numpy as np
from scipy.signal import correlate
from scipy.fft import fft, fftfreq, ifft

try:
    from qutip import (
        identity, sigmax, sigmay, sigmaz, tensor,
        basis, ket2dm, expect, sesolve
    )
    HAS_QUTIP = True
except ImportError:
    HAS_QUTIP = False


# ── Default parameters (consistent with all papers) ──────────────
_J = 1.0
_g = 0.5
_kappa_val = 0.45
_hz = 0.12
_hx = 0.08
_SIDEREAL_PERIOD = 23.93


def _get_op(op, idx, N_total):
    """Operator on qubit idx in an N_total-qubit system."""
    ops = [identity(2)] * N_total
    ops[idx] = op
    return tensor(ops)


def build_channel_hamiltonian(N=6, J=_J, g=_g, kappa=_kappa_val, hz=_hz, hx=_hx):
    """
    Build the channel Hamiltonian on N qubits.

    Uses N=6 by default (reduced from 8) for computational feasibility
    when combined with Eve qubits: 2^(6+m) vs 2^(8+m).

    Parameters
    ----------
    N : int
        Number of channel qubits (default 6).

    Returns
    -------
    qutip.Qobj
        Channel Hamiltonian.

    Notebook: quantum_eve_v2.py, Section 1 (build_channel_H)
    Paper: [6], Section IV
    """
    if not HAS_QUTIP:
        raise ImportError("qutip is required for quantum_eve module")

    H = (J * (_get_op(sigmax(), 0, N) * _get_op(sigmax(), 1, N)
            + _get_op(sigmay(), 0, N) * _get_op(sigmay(), 1, N)
            + _get_op(sigmaz(), 0, N) * _get_op(sigmaz(), 1, N))
       + sum([g * _get_op(sigmax(), i, N) * _get_op(sigmax(), i+1, N)
              for i in range(2, N-1)])
       + kappa * _get_op(sigmaz(), 1, N) * _get_op(sigmaz(), 2, N)
       + sum([hz * _get_op(sigmaz(), i, N) + hx * _get_op(sigmax(), i, N)
              for i in range(N)]))
    return H


def build_total_hamiltonian(m_eve, coupling_strength, strategy='passive',
                            N_channel=6, J=_J, g=_g, kappa=_kappa_val,
                            hz=_hz, hx=_hx):
    """
    Build the total Hamiltonian: H_channel + H_eve + H_coupling.

    Eve's Hamiltonian consists of local fields and (for m>=2) Heisenberg
    couplings between her qubits. Four coupling strategies are supported.

    Parameters
    ----------
    m_eve : int
        Number of Eve's qubits (1 or 2 typical).
    coupling_strength : float
        Eve-channel coupling strength epsilon.
    strategy : str
        Coupling strategy: 'passive' (ZZ), 'xx' (XX), 'heisenberg' (XX+YY+ZZ),
        or 'optimal' (multi-qubit aggressive).
    N_channel : int
        Number of channel qubits.

    Returns
    -------
    qutip.Qobj
        Total Hamiltonian on (N_channel + m_eve) qubits.

    Notebook: quantum_eve_v2.py, Section 3 (build_total_hamiltonian)
    Paper: [6], Section IV, Eq. (12)
    """
    if not HAS_QUTIP:
        raise ImportError("qutip is required for quantum_eve module")

    N_total = N_channel + m_eve

    # Channel Hamiltonian on extended space
    H_ch = (J * (_get_op(sigmax(), 0, N_total) * _get_op(sigmax(), 1, N_total)
               + _get_op(sigmay(), 0, N_total) * _get_op(sigmay(), 1, N_total)
               + _get_op(sigmaz(), 0, N_total) * _get_op(sigmaz(), 1, N_total))
          + sum([g * _get_op(sigmax(), i, N_total) * _get_op(sigmax(), i+1, N_total)
                 for i in range(2, N_channel - 1)])
          + kappa * _get_op(sigmaz(), 1, N_total) * _get_op(sigmaz(), 2, N_total)
          + sum([hz * _get_op(sigmaz(), i, N_total) + hx * _get_op(sigmax(), i, N_total)
                 for i in range(N_channel)]))

    # Eve Hamiltonian
    H_eve = 0
    for i in range(N_channel, N_total):
        H_eve += 0.3 * _get_op(sigmaz(), i, N_total) + 0.2 * _get_op(sigmax(), i, N_total)
    if m_eve >= 2:
        for i in range(N_channel, N_total - 1):
            H_eve += 0.5 * (_get_op(sigmax(), i, N_total) * _get_op(sigmax(), i+1, N_total)
                          + _get_op(sigmay(), i, N_total) * _get_op(sigmay(), i+1, N_total)
                          + _get_op(sigmaz(), i, N_total) * _get_op(sigmaz(), i+1, N_total))

    # Coupling
    bq = N_channel - 1  # Last channel qubit
    eq = N_channel       # First Eve qubit

    if strategy == 'passive':
        H_coup = coupling_strength * _get_op(sigmaz(), bq, N_total) * _get_op(sigmaz(), eq, N_total)
    elif strategy == 'xx':
        H_coup = coupling_strength * _get_op(sigmax(), bq, N_total) * _get_op(sigmax(), eq, N_total)
    elif strategy == 'heisenberg':
        H_coup = coupling_strength * (
            _get_op(sigmax(), bq, N_total) * _get_op(sigmax(), eq, N_total)
          + _get_op(sigmay(), bq, N_total) * _get_op(sigmay(), eq, N_total)
          + _get_op(sigmaz(), bq, N_total) * _get_op(sigmaz(), eq, N_total))
    elif strategy == 'optimal':
        H_coup = coupling_strength * (
            _get_op(sigmax(), bq, N_total) * _get_op(sigmax(), eq, N_total)
          + 0.5 * _get_op(sigmaz(), bq, N_total) * _get_op(sigmaz(), eq, N_total))
        if m_eve >= 2 and N_channel >= 2:
            H_coup += coupling_strength * 0.3 * (
                _get_op(sigmaz(), bq-1, N_total) * _get_op(sigmaz(), eq+1, N_total))
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Choose from: passive, xx, heisenberg, optimal")

    return H_ch + H_eve + H_coup


def compute_channel_autocorrelation(H_total, N_channel, N_total, t_points,
                                     obs_qubit=0):
    """
    Compute C(t) = <O(t)O(0)> via expectation values on the total system.

    This is the approximate method using <O(t)> * <O(0)> for a product
    initial state |↑↑...↑⟩.

    Parameters
    ----------
    H_total : qutip.Qobj
        Total Hamiltonian.
    N_channel : int
        Number of channel qubits.
    N_total : int
        Total number of qubits (channel + Eve).
    t_points : array-like
        Time points for evaluation.
    obs_qubit : int
        Which qubit to observe (default 0).

    Returns
    -------
    np.ndarray
        Normalized autocorrelation C(t)/C(0).

    Notebook: quantum_eve_v2.py, Section 2 (compute_channel_autocorrelation)
    Paper: [6], Section IV
    """
    if not HAS_QUTIP:
        raise ImportError("qutip is required")

    psi0 = tensor([basis(2, 0)] * N_total)
    O = _get_op(sigmaz(), obs_qubit, N_total)
    result = sesolve(H_total, psi0, t_points, [O])
    expectation = np.array(result.expect[0])

    C0 = expectation[0]
    autocorr = expectation * C0

    if abs(autocorr[0]) > 1e-12:
        autocorr = autocorr / autocorr[0]

    return autocorr


def compute_reduced_autocorrelation(H_total, N_channel, N_total, t_points,
                                     obs_qubit=0):
    """
    CORRECT computation: time-evolve the total system, then trace out Eve,
    then compute expectation value on the channel subsystem.

    This is the autocorrelation that Alice and Bob actually measure.

    Parameters
    ----------
    H_total : qutip.Qobj
        Total Hamiltonian.
    N_channel : int
        Number of channel qubits.
    N_total : int
        Total number of qubits.
    t_points : array-like
        Time points.
    obs_qubit : int
        Observable qubit index.

    Returns
    -------
    np.ndarray
        Normalized reduced autocorrelation.

    Notebook: quantum_eve_v2.py, Section 2 (compute_reduced_autocorrelation)
    Paper: [6], Section IV, Eq. (13)
    """
    if not HAS_QUTIP:
        raise ImportError("qutip is required")

    psi0 = tensor([basis(2, 0)] * N_total)
    result = sesolve(H_total, psi0, t_points)

    O_channel = _get_op(sigmaz(), obs_qubit, N_channel)

    autocorr = np.zeros(len(t_points))
    for i in range(len(t_points)):
        psi_t = result.states[i]
        rho_total = ket2dm(psi_t)
        rho_channel = rho_total.ptrace(list(range(N_channel)))
        autocorr[i] = expect(O_channel, rho_channel)

    if abs(autocorr[0]) > 1e-12:
        autocorr = autocorr / autocorr[0]

    return autocorr


def gaussian_template(b_n, t_axis):
    """
    Gaussian autocorrelation template from Krylov b_n coefficients.

    Parameters
    ----------
    b_n : array-like
        Lanczos coefficients.
    t_axis : array-like
        Time axis.

    Returns
    -------
    np.ndarray
        Gaussian template exp(-0.5 * (slope * t)^2).

    Notebook: quantum_eve_v2.py, Section 4
    """
    s = np.mean(np.diff(b_n))
    return np.exp(-0.5 * (s * t_axis)**2)


def make_classical_eve_qber(t, p_ir=0.3, eve_start=150, eve_end=280,
                             avg_slope=None, seed=None):
    """
    Generate QBER time series with classical intercept-resend Eve.

    Parameters
    ----------
    t : array-like
        Time axis.
    p_ir : float
        Intercept-resend probability.
    eve_start, eve_end : int
        Attack window indices.
    avg_slope : float or None
        Average Krylov slope for correlated noise. If None, uses 0.5.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        QBER time series.

    Notebook: quantum_eve_v2.py, Section 6 (make_classical_eve_qber)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(t)
    slope = avg_slope if avg_slope is not None else 0.5

    env = (0.04 * np.sin(2 * np.pi * t / _SIDEREAL_PERIOD)
         + 0.02 * np.sin(2 * np.pi * t / 24.0 + 0.5))
    noise_base = np.random.normal(0, 0.015, n)
    kernel = np.exp(-0.5 * (np.arange(10) * slope)**2)
    noise_corr = np.convolve(noise_base, kernel, mode='same') * 0.3

    qber = env + noise_base + noise_corr

    eve_signal = np.zeros(n)
    eve_end = min(eve_end, n)
    eve_start = min(eve_start, n)
    window_len = eve_end - eve_start
    if window_len > 0:
        eve_signal[eve_start:eve_end] = (
            p_ir * 0.25 * np.ones(window_len)
            + np.random.normal(0, 0.005, window_len)
        )

    return qber + eve_signal


def make_quantum_eve_qber(t, autocorr_eve, autocorr_clean, coupling_strength,
                           eve_start=150, eve_end=280, avg_slope=None, seed=None):
    """
    Generate QBER with quantum Eve signature.

    Uses the actual modified autocorrelation from the quantum Eve simulation
    to produce correlated noise with Eve's correlation structure.

    Parameters
    ----------
    t : array-like
        Time axis.
    autocorr_eve : array-like
        Eve's modified autocorrelation (from compute_reduced_autocorrelation).
    autocorr_clean : array-like
        Clean channel autocorrelation.
    coupling_strength : float
        Eve coupling strength epsilon.
    eve_start, eve_end : int
        Attack window indices.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        QBER time series with quantum Eve signature.

    Notebook: quantum_eve_v2.py, Section 6 (make_quantum_eve_qber)
    Paper: [6], Section IV
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(t)
    slope = avg_slope if avg_slope is not None else 0.5

    # Clean baseline
    env = (0.04 * np.sin(2 * np.pi * t / _SIDEREAL_PERIOD)
         + 0.02 * np.sin(2 * np.pi * t / 24.0 + 0.5))
    noise_base = np.random.normal(0, 0.015, n)
    kernel = np.exp(-0.5 * (np.arange(10) * slope)**2)
    noise_corr = np.convolve(noise_base, kernel, mode='same') * 0.3
    qber = env + noise_base + noise_corr

    eve_end = min(eve_end, n)
    eve_start = min(eve_start, n)
    eve_window_len = eve_end - eve_start

    if eve_window_len <= 0:
        return qber

    # Generate noise with Eve's autocorrelation structure
    kernel_len = min(20, eve_window_len)
    eve_kernel = np.abs(autocorr_eve[:kernel_len].copy())
    if eve_kernel.sum() > 0:
        eve_kernel /= eve_kernel.sum()
    else:
        eve_kernel = np.ones(kernel_len) / kernel_len

    raw_noise = np.random.normal(0, 1, eve_window_len + kernel_len)
    correlated_noise = np.convolve(raw_noise, eve_kernel, mode='valid')[:eve_window_len]

    eve_amplitude = coupling_strength * 0.2
    eve_signal = np.zeros(n)
    if correlated_noise.std() > 0:
        eve_signal[eve_start:eve_end] = eve_amplitude * correlated_noise / correlated_noise.std()

    # Systematic shift from autocorrelation deviation
    deviation = np.sqrt(np.mean((autocorr_eve - autocorr_clean)**2))
    eve_signal[eve_start:eve_end] += deviation * 0.05

    return qber + eve_signal


def compute_anomaly_scores(autocorr_matrix, autocorr_clean):
    """
    Compute anomaly score for each autocorrelation curve.

    Score = RMSE between the curve and the clean reference.

    Parameters
    ----------
    autocorr_matrix : np.ndarray
        Shape (n_trials, n_timepoints).
    autocorr_clean : np.ndarray
        Clean reference autocorrelation.

    Returns
    -------
    np.ndarray
        Anomaly scores.

    Notebook: quantum_eve_v2.py (compute_anomaly_scores)
    """
    return np.sqrt(np.mean((autocorr_matrix - autocorr_clean)**2, axis=1))


def compute_eve_detection_stats(scores_eve, scores_clean, threshold=None):
    """
    Compute detection statistics for Eve scores.

    Parameters
    ----------
    scores_eve : array-like
        Anomaly scores for Eve trials.
    scores_clean : array-like
        Anomaly scores for clean trials.
    threshold : float or None
        Detection threshold. If None, uses 95th percentile of clean scores.

    Returns
    -------
    dict
        Keys: detected_pct, forged_pct, auc, sep.

    Notebook: quantum_eve_v2.py (compute_eve_detection_stats)
    Paper: [6], Section IV
    """
    scores_eve = np.asarray(scores_eve)
    scores_clean = np.asarray(scores_clean)

    if threshold is None:
        threshold = np.percentile(scores_clean, 95)

    detected_pct = float(np.mean(scores_eve > threshold) * 100)
    forged_pct = 100.0 - detected_pct

    # ROC AUC
    labels = np.concatenate([np.ones(len(scores_eve)), np.zeros(len(scores_clean))])
    scores = np.concatenate([scores_eve, scores_clean])
    idx = np.argsort(scores)[::-1]
    labels_sorted = labels[idx]

    tpr = np.cumsum(labels_sorted == 1) / np.sum(labels_sorted == 1)
    fpr = np.cumsum(labels_sorted == 0) / np.sum(labels_sorted == 0)
    auc = float(np.trapezoid(tpr, fpr))

    # Separation
    if np.std(scores_clean) > 1e-9 and np.std(scores_eve) > 1e-9:
        sep = float((np.mean(scores_eve) - np.mean(scores_clean))
                    / np.sqrt(0.5 * (np.var(scores_clean) + np.var(scores_eve))))
    else:
        sep = 0.0

    return {
        "detected_pct": detected_pct,
        "forged_pct": forged_pct,
        "auc": auc,
        "sep": sep,
    }


# ── Coupling strategies constant ──────────────────────────────────
STRATEGIES = ['passive', 'xx', 'heisenberg', 'optimal']
