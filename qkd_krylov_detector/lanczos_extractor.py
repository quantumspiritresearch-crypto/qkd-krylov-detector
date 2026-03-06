"""
Layer 2: Lanczos Algorithm for Krylov b_n Coefficients.

Computes the Lanczos coefficients b_n from the Hamiltonian using the
recursive Lanczos algorithm in operator space (Krylov basis construction).

The key insight: linear growth of b_n implies Gaussian decay of the
operator autocorrelation C(t) = exp(-0.5 * (slope * t)^2), where
slope = mean(diff(b_n)).

This provides the theoretical template against which the QBER
autocorrelation is compared in Layer 3.

Reference:
    Paper [4]: Scrambling vs. Recurrence (DOI: 10.5281/zenodo.18813710)
    Paper [5]: QKD Eve Detector Parts I-III (DOI: 10.5281/zenodo.18873824)
    Paper [6]: Quantum Scrambling as Cryptographic Resource (DOI: 10.5281/zenodo.18889224)
    Parker et al., "A Universal Operator Growth Hypothesis," PRX 9, 041017 (2019)

Author: Daniel Süß
"""

import numpy as np
from qutip import sigmaz
from .hamiltonian import get_op, DEFAULT_N


def compute_lanczos(H, n_steps=25, initial_op=None, initial_qubit=0, N=DEFAULT_N):
    """
    Compute Lanczos coefficients b_n via the recursive Krylov algorithm.

    Starting from an initial operator O_0 (default: sigma_z on qubit 0),
    iteratively computes:
        O_{n+1} = i[H, O_n] - b_{n-1} * O_{n-1}
        b_n = ||O_{n+1}||

    Parameters
    ----------
    H : qutip.Qobj
        System Hamiltonian.
    n_steps : int
        Maximum number of Lanczos steps (default: 25).
    initial_op : qutip.Qobj, optional
        Initial operator. If None, uses sigma_z on `initial_qubit`.
    initial_qubit : int
        Qubit index for the default initial operator (default: 0).
    N : int
        Number of qubits (default: 8).

    Returns
    -------
    numpy.ndarray
        Array of Lanczos coefficients b_n.

    Notes
    -----
    The algorithm terminates early if b_n < 1e-12 (Krylov space exhausted).

    For the default 8-qubit Heisenberg chain in Regime II:
        - b_n grows approximately linearly (characteristic of quantum chaos)
        - avg_slope ≈ 3.975
        - This implies Gaussian decay with timescale tau = 1/slope ≈ 0.252

    Source: krylov_dynamic_detector.ipynb Cell 2 (compute_lanczos),
            eve_detection_master_v3.ipynb Cell 5 (compute_lanczos),
            krylov_robustness_test.ipynb Cell 3 (compute_lanczos)
    """
    b = []

    if initial_op is None:
        O_curr = get_op(sigmaz(), initial_qubit, N)
        O_curr = O_curr / O_curr.norm()
    else:
        O_curr = initial_op / initial_op.norm()

    O_prev = None

    for n in range(n_steps):
        # Liouvillian action: L(O) = i[H, O]
        O_next = 1j * (H * O_curr - O_curr * H)

        # Orthogonalize against previous
        if n > 0:
            O_next -= b[n - 1] * O_prev

        # Compute norm (= b_n)
        bn = O_next.norm()

        # Check for Krylov space exhaustion
        if bn < 1e-12:
            break

        b.append(bn)
        O_prev, O_curr = O_curr, O_next / bn

    return np.array(b)


def get_theoretical_autocorrelation(b_n, t_axis):
    """
    Compute the theoretical operator autocorrelation from b_n structure.

    For linearly growing b_n (characteristic of quantum chaos / scrambling),
    the autocorrelation decays as a Gaussian:

        C(t) = exp(-0.5 * (slope * t)^2)

    where slope = mean(diff(b_n)).

    Parameters
    ----------
    b_n : array_like
        Lanczos coefficients.
    t_axis : array_like
        Time axis for the autocorrelation.

    Returns
    -------
    numpy.ndarray
        Normalized theoretical autocorrelation template.

    Notes
    -----
    This is the "Gaussian template" central to the detection framework.
    The template is exact for broadband noise; for colored noise, the
    template must be modified according to S_QBER(ω) = |χ(ω)|² · S_env(ω).
    See Paper [6], Section III.D.

    Source: krylov_dynamic_detector.ipynb Cell 3 (get_theoretical_autocorrelation)
    """
    b_n = np.asarray(b_n)
    t_axis = np.asarray(t_axis)

    avg_slope = np.mean(np.diff(b_n))
    theoretical_decay = np.exp(-0.5 * (avg_slope * t_axis) ** 2)

    return theoretical_decay / np.max(theoretical_decay)


def compute_bn_deviation(b_baseline, b_perturbed):
    """
    Compute mean absolute deviation between baseline and perturbed b_n.

    Parameters
    ----------
    b_baseline : array_like
        Lanczos coefficients of the clean Hamiltonian.
    b_perturbed : array_like
        Lanczos coefficients of the perturbed (Eve) Hamiltonian.

    Returns
    -------
    float
        Mean absolute deviation.

    Source: eve_detection_master_v3.ipynb Cell 5 (bn_deviation)
    """
    ml = min(len(b_baseline), len(b_perturbed))
    return float(np.mean(np.abs(
        np.asarray(b_perturbed[:ml]) - np.asarray(b_baseline[:ml])
    )))


def get_slope(b_n):
    """
    Extract the average slope from Lanczos coefficients.

    Parameters
    ----------
    b_n : array_like
        Lanczos coefficients.

    Returns
    -------
    float
        Average slope = mean(diff(b_n)).
    """
    return float(np.mean(np.diff(np.asarray(b_n))))
