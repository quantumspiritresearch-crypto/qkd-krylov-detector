"""
Sparse Hamiltonian for Finite-Size Scaling
==========================================

Constructs the symmetry-broken Heisenberg chain as a scipy sparse matrix,
enabling eigenvalue computations for N > 8 qubits where dense QuTiP
representations become infeasible.

Provides sparse spectral statistics (level spacing ratio ⟨r⟩) for
finite-size scaling analysis of the integrable-to-chaotic crossover.

Notebook correspondence:
    krylov_sparse_scaling.ipynb — Cells 1–3

Paper reference:
    [4] D. Süß, "Scrambling vs. Recurrence: Microscopic Origin of the
        Quantum Arrow of Time," Zenodo, 2026. DOI: 10.5281/zenodo.18813710,
        Section III

Author: Daniel Süß
License: MIT
"""

import numpy as np
import scipy.sparse as sp_mat
import scipy.sparse.linalg as spla
from typing import Tuple, List, Optional

# Pauli matrices as sparse
_sx = sp_mat.csr_matrix([[0, 1], [1, 0]], dtype=complex)
_sy = sp_mat.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
_sz = sp_mat.csr_matrix([[1, 0], [0, -1]], dtype=complex)


def kron_op(op: sp_mat.spmatrix,
            idx: int,
            N: int) -> sp_mat.csr_matrix:
    """Place a 2x2 operator at site *idx* in an N-qubit chain.

    Constructs the full 2^N × 2^N operator via Kronecker products,
    with identity on all other sites.

    Parameters
    ----------
    op : sparse matrix (2×2)
        Single-qubit operator.
    idx : int
        Site index (0-based).
    N : int
        Total number of qubits.

    Returns
    -------
    full_op : csr_matrix
        Operator in the full Hilbert space.
    """
    I = sp_mat.eye(2, format="csr", dtype=complex)
    result = sp_mat.eye(1, format="csr", dtype=complex)
    for i in range(N):
        result = sp_mat.kron(result, op if i == idx else I, format="csr")
    return result


def build_hamiltonian_sparse(N: int,
                             J: float = 1.0,
                             g: float = 0.5,
                             kappa: float = 0.45,
                             hz: float = 0.12,
                             hx: float = 0.08
                             ) -> sp_mat.csr_matrix:
    """Build the Heisenberg chain Hamiltonian as a scipy sparse matrix.

    Identical structure to :func:`hamiltonian.build_hamiltonian` but uses
    sparse representations, enabling N > 8 qubits.

    Parameters
    ----------
    N : int
        Number of qubits.
    J : float
        Heisenberg coupling between qubits 0 and 1.
    g : float
        XX chain coupling for qubits 2 through N-1.
    kappa : float
        ZZ coupling between qubits 1 and 2.
    hz : float
        Longitudinal magnetic field strength.
    hx : float
        Transverse magnetic field strength.

    Returns
    -------
    H : csr_matrix
        Sparse Hamiltonian of dimension 2^N × 2^N.

    Notebook correspondence:
        krylov_sparse_scaling.ipynb, Cell 1 — build_H_sparse()
    """
    dim = 2**N
    H = sp_mat.csr_matrix((dim, dim), dtype=complex)

    # Heisenberg coupling on qubits 0–1
    for op in [_sx, _sy, _sz]:
        H += J * kron_op(op, 0, N).dot(kron_op(op, 1, N))

    # XX chain coupling on qubits 2 through N-1
    for i in range(2, N - 1):
        H += g * kron_op(_sx, i, N).dot(kron_op(_sx, i + 1, N))

    # ZZ coupling between qubits 1 and 2
    H += kappa * kron_op(_sz, 1, N).dot(kron_op(_sz, 2, N))

    # Magnetic fields
    for i in range(N):
        H += hz * kron_op(_sz, i, N)
        H += hx * kron_op(_sx, i, N)

    return H


def spectral_statistics_sparse(N: int,
                               hz: float = 0.12,
                               hx: float = 0.08,
                               k: int = 300,
                               J: float = 1.0,
                               g: float = 0.5,
                               kappa: float = 0.45
                               ) -> Tuple[float, float, List[float]]:
    """Compute level spacing ratio ⟨r⟩ using sparse eigenvalues.

    Uses shift-invert mode to extract eigenvalues from the bulk of the
    spectrum, then computes the mean ratio of consecutive level spacings.

    Parameters
    ----------
    N : int
        Number of qubits.
    hz : float
        Longitudinal field.
    hx : float
        Transverse field.
    k : int
        Number of eigenvalues to compute (capped at 2^N - 2).
    J, g, kappa : float
        Hamiltonian coupling parameters.

    Returns
    -------
    r_mean : float
        Mean level spacing ratio ⟨r⟩.
    r_std : float
        Standard deviation of r values.
    r_list : list of float
        Individual r values.

    Notebook correspondence:
        krylov_sparse_scaling.ipynb, Cell 2 — spectral_statistics_sparse()
    """
    H = build_hamiltonian_sparse(N, J=J, g=g, kappa=kappa, hz=hz, hx=hx)
    k = min(k, 2**N - 2)

    # Find spectral center via extremal eigenvalues
    e_min = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
    e_max = spla.eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
    sigma = (e_min + e_max) / 2.0

    # Shift-invert to get bulk eigenvalues
    en = spla.eigsh(H, k=k, sigma=sigma, which="LM",
                    return_eigenvectors=False)
    en = np.sort(np.real(en))

    # Trim edges (10% on each side)
    bulk = en[int(len(en) * 0.1):int(len(en) * 0.9)]
    sp = np.diff(bulk)
    sp = sp[sp > 1e-12]

    r_list = []
    for i in range(len(sp) - 1):
        r = min(sp[i], sp[i + 1]) / max(sp[i], sp[i + 1])
        if np.isfinite(r):
            r_list.append(float(r))

    if len(r_list) == 0:
        return 0.0, 0.0, []

    return float(np.mean(r_list)), float(np.std(r_list)), r_list


def finite_size_scaling(N_values: Optional[List[int]] = None,
                        hz: float = 0.12,
                        hx: float = 0.08,
                        k: int = 300
                        ) -> dict:
    """Run finite-size scaling of ⟨r⟩ across multiple system sizes.

    Parameters
    ----------
    N_values : list of int, optional
        System sizes to scan (default: [6, 7, 8, 9, 10]).
    hz, hx : float
        Field parameters.
    k : int
        Number of eigenvalues per system size.

    Returns
    -------
    results : dict
        Keys: 'N', 'r_mean', 'r_std' — arrays of results.
    """
    if N_values is None:
        N_values = [6, 7, 8, 9, 10]

    r_means = []
    r_stds = []
    for N in N_values:
        r_mean, r_std, _ = spectral_statistics_sparse(N, hz=hz, hx=hx, k=k)
        r_means.append(r_mean)
        r_stds.append(r_std)

    return {
        'N': np.array(N_values),
        'r_mean': np.array(r_means),
        'r_std': np.array(r_stds),
    }
