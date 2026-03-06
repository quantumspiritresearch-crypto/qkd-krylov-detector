"""
Hamiltonian construction for the 8-qubit Heisenberg chain.

This module builds the symmetry-broken Heisenberg Hamiltonian used throughout
the QKD Eve Detector framework. The parameters are identical across all
notebooks and papers.

System parameters (Regime II, crossover regime ⟨r⟩ ≈ 0.366):
    N = 8           Number of qubits
    J = 1.0         Heisenberg coupling (qubits 0-1)
    g = 0.5         XX chain coupling (qubits 2 through N-1)
    kappa = 0.45    ZZ coupling (qubits 1-2)
    hz = 0.12       Longitudinal field
    hx = 0.08       Transverse field

Reference:
    Scrambling vs. Recurrence [4], Section II
    Eve Detection Master v3 notebook, Cell 3
    Krylov Dynamic Detector notebook, Cell 2

Author: Daniel Süß
"""

import numpy as np
from qutip import identity, sigmax, sigmay, sigmaz, tensor


# Default system parameters — consistent across ALL notebooks and papers
DEFAULT_N = 8
DEFAULT_J = 1.0
DEFAULT_G = 0.5
DEFAULT_KAPPA = 0.45
DEFAULT_HZ = 0.12
DEFAULT_HX = 0.08


def get_op(op, idx, N=DEFAULT_N):
    """
    Embed a single-qubit operator at position `idx` in an N-qubit chain.

    Parameters
    ----------
    op : qutip.Qobj
        Single-qubit operator (e.g., sigmax(), sigmaz()).
    idx : int
        Qubit index (0-based).
    N : int
        Total number of qubits.

    Returns
    -------
    qutip.Qobj
        Operator acting on the full Hilbert space.

    Source: All notebooks use this identical helper function.
    """
    ops = [identity(2)] * N
    ops[idx] = op
    return tensor(ops)


def build_hamiltonian(N=DEFAULT_N, J=DEFAULT_J, g=DEFAULT_G,
                      kappa=DEFAULT_KAPPA, hz=DEFAULT_HZ, hx=DEFAULT_HX):
    """
    Build the symmetry-broken Heisenberg chain Hamiltonian.

    H = J*(XX + YY + ZZ)_{01}
      + g * sum_{i=2}^{N-2} XX_{i,i+1}
      + kappa * ZZ_{12}
      + sum_i (hz * Z_i + hx * X_i)

    Parameters
    ----------
    N : int
        Number of qubits (default: 8).
    J : float
        Heisenberg coupling strength for qubits 0-1 (default: 1.0).
    g : float
        XX chain coupling for qubits 2 through N-1 (default: 0.5).
    kappa : float
        ZZ coupling between qubits 1-2 (default: 0.45).
    hz : float
        Longitudinal magnetic field (default: 0.12).
    hx : float
        Transverse magnetic field (default: 0.08).

    Returns
    -------
    qutip.Qobj
        The Hamiltonian operator.

    Notes
    -----
    With default parameters, the system is in the crossover regime
    (⟨r⟩ ≈ 0.366, between Poisson 0.386 and GOE 0.536).
    This is NOT full GOE chaos — see Paper [4], Section II.

    Source: krylov_dynamic_detector.ipynb Cell 2,
            eve_detection_master_v3.ipynb Cell 3
    """
    # Heisenberg coupling on qubits 0-1
    H = J * (get_op(sigmax(), 0, N) * get_op(sigmax(), 1, N)
           + get_op(sigmay(), 0, N) * get_op(sigmay(), 1, N)
           + get_op(sigmaz(), 0, N) * get_op(sigmaz(), 1, N))

    # XX chain coupling for qubits 2 through N-1
    for i in range(2, N - 1):
        H += g * get_op(sigmax(), i, N) * get_op(sigmax(), i + 1, N)

    # Asymmetric ZZ coupling between qubits 1-2
    H += kappa * get_op(sigmaz(), 1, N) * get_op(sigmaz(), 2, N)

    # Magnetic fields on all qubits
    for i in range(N):
        H += hz * get_op(sigmaz(), i, N) + hx * get_op(sigmax(), i, N)

    return H


def build_hamiltonian_with_eve(gamma, N=DEFAULT_N, **kwargs):
    """
    Build Hamiltonian with Eve's perturbation.

    Eve's operator: gamma * sigma_x(1) * sigma_x(2)
    This attacks the kappa-coupling interface and is non-commuting
    with kappa * sigma_z(1) * sigma_z(2), shifting b_n from n=1.

    Parameters
    ----------
    gamma : float
        Eve coupling strength.
    N : int
        Number of qubits.
    **kwargs
        Additional parameters passed to build_hamiltonian().

    Returns
    -------
    qutip.Qobj
        Hamiltonian with Eve perturbation.

    Source: eve_detection_master_v3.ipynb Cell 5 (build_H_with_eve)
    """
    H_base = build_hamiltonian(N=N, **kwargs)
    H_eve = gamma * get_op(sigmax(), 1, N) * get_op(sigmax(), 2, N)
    return H_base + H_eve
