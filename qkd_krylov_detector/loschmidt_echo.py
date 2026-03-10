"""
Loschmidt Echo Module for QKD Krylov Detector.

Computes state-space and operator-space Loschmidt echoes for Hamiltonian
perturbation analysis. Establishes the formal connection between the Krylov
detection score and the operator-space Loschmidt echo.

Reference:
    D. Süß, "The Krylov Eavesdropper Detector as an Operator-Space Loschmidt
    Echo," Zenodo (2026). DOI: 10.5281/zenodo.18939996
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import pearsonr


def eigendecompose(H):
    """Compute eigendecomposition of a Hermitian matrix.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Hermitian matrix (e.g., Hamiltonian).

    Returns
    -------
    E : ndarray, shape (d,)
        Eigenvalues in ascending order.
    V : ndarray, shape (d, d)
        Eigenvectors as columns.
    """
    return eigh(H)


def compute_state_echo(E_clean, V_clean, E_pert, V_pert, psi0, t):
    """Compute the state-space Loschmidt echo M(t).

    M(t) = |<psi0| e^{i H_pert t} e^{-i H_clean t} |psi0>|^2

    Parameters
    ----------
    E_clean, V_clean : ndarray
        Eigenvalues and eigenvectors of the clean Hamiltonian.
    E_pert, V_pert : ndarray
        Eigenvalues and eigenvectors of the perturbed Hamiltonian.
    psi0 : ndarray, shape (d,)
        Initial state vector.
    t : float
        Time at which to evaluate the echo.

    Returns
    -------
    float
        Loschmidt echo M(t) in [0, 1].
    """
    # e^{-i H_clean t} |psi0>
    coeffs_clean = V_clean.conj().T @ psi0
    psi_evolved = V_clean @ (np.exp(-1j * E_clean * t) * coeffs_clean)

    # e^{+i H_pert t} |psi_evolved> = e^{-i H_pert (-t)} |psi_evolved>
    coeffs_pert = V_pert.conj().T @ psi_evolved
    psi_echo = V_pert @ (np.exp(1j * E_pert * t) * coeffs_pert)

    return float(np.abs(psi0.conj() @ psi_echo) ** 2)


def compute_operator_echo(E_clean, V_clean, E_pert, V_pert, O0, t, dim=None):
    """Compute the operator-space Loschmidt echo M_op(t).

    M_op(t) = |<O0| e^{i L_pert t} e^{-i L_clean t} |O0>|^2 / ||O0||^4

    where <A|B> = Tr(A† B) / dim and L_H(O) = [H, O].

    Parameters
    ----------
    E_clean, V_clean : ndarray
        Eigenvalues and eigenvectors of the clean Hamiltonian.
    E_pert, V_pert : ndarray
        Eigenvalues and eigenvectors of the perturbed Hamiltonian.
    O0 : ndarray, shape (d, d)
        Initial operator (observable).
    t : float
        Time at which to evaluate the echo.
    dim : int, optional
        Hilbert space dimension. Inferred from O0 if not provided.

    Returns
    -------
    float
        Operator-space Loschmidt echo M_op(t) in [0, 1].
    """
    if dim is None:
        dim = O0.shape[0]

    O0_norm_sq = np.trace(O0.conj().T @ O0).real / dim

    # e^{-i L_clean t} O0 = e^{-iHt} O0 e^{iHt}
    O0_in_clean = V_clean.conj().T @ O0 @ V_clean
    E_diff_clean = np.subtract.outer(E_clean, E_clean)
    phases_clean = np.exp(-1j * E_diff_clean * t)
    O_after_clean_eig = O0_in_clean * phases_clean
    O_after_clean = V_clean @ O_after_clean_eig @ V_clean.conj().T

    # e^{+i L_pert t} O_after = e^{iHt} O_after e^{-iHt}
    O_in_pert = V_pert.conj().T @ O_after_clean @ V_pert
    E_diff_pert = np.subtract.outer(E_pert, E_pert)
    phases_pert = np.exp(1j * E_diff_pert * t)
    O_final_eig = O_in_pert * phases_pert
    O_final = V_pert @ O_final_eig @ V_pert.conj().T

    overlap = np.trace(O0.conj().T @ O_final).real / dim
    return float((overlap ** 2) / (O0_norm_sq ** 2))


def compute_echo_decay_rate(E_clean, V_clean, E_pert, V_pert, O0, t_max=50,
                            n_samples=10):
    """Compute the exponential decay rate of the operator echo.

    Fits log(M_op(t)) ~ -rate * t over n_samples time points.

    Parameters
    ----------
    E_clean, V_clean : ndarray
        Eigenvalues and eigenvectors of the clean Hamiltonian.
    E_pert, V_pert : ndarray
        Eigenvalues and eigenvectors of the perturbed Hamiltonian.
    O0 : ndarray, shape (d, d)
        Initial operator.
    t_max : float
        Maximum time for sampling.
    n_samples : int
        Number of time samples.

    Returns
    -------
    float
        Decay rate (non-negative). Larger values indicate faster echo decay.
    """
    dim = O0.shape[0]
    sample_times = np.linspace(t_max / n_samples, t_max, n_samples)
    Mop_values = np.array([
        compute_operator_echo(E_clean, V_clean, E_pert, V_pert, O0, t, dim)
        for t in sample_times
    ])
    log_Mop = np.log(np.clip(Mop_values, 1e-15, None))
    slope = np.polyfit(sample_times, log_Mop, 1)[0]
    return float(max(-slope, 0))


def compute_operator_autocorrelation(E, V, O0, times, dim=None):
    """Compute the operator autocorrelation C(t) = <O0(0) O0(t)>.

    C(t) = Tr(O0† e^{iHt} O0 e^{-iHt}) / (dim * ||O0||^2)

    Parameters
    ----------
    E, V : ndarray
        Eigenvalues and eigenvectors of the Hamiltonian.
    O0 : ndarray, shape (d, d)
        Operator.
    times : ndarray, shape (n_t,)
        Time points.
    dim : int, optional
        Hilbert space dimension.

    Returns
    -------
    ndarray, shape (n_t,)
        Autocorrelation values.
    """
    if dim is None:
        dim = O0.shape[0]

    O0_norm_sq = np.trace(O0.conj().T @ O0).real / dim
    O0_eig = V.conj().T @ O0 @ V
    O0_eig_sq = O0_eig.conj() * O0_eig
    E_diff = np.subtract.outer(E, E)

    C = np.zeros(len(times))
    for it, t in enumerate(times):
        phases = np.exp(1j * E_diff * t)
        C[it] = np.sum(O0_eig_sq * phases).real / dim / O0_norm_sq
    return C


def loschmidt_krylov_correlation(H, V_eve_op, O0, gammas, t_max=50,
                                 n_time_steps=200, n_lanczos=25):
    """Run the full Loschmidt-Krylov correlation analysis.

    For each coupling strength gamma, computes:
    - Lanczos coefficient RMSE (vs. clean)
    - Operator echo decay rate
    - Krylov detection score (autocorrelation RMSE)

    Then computes Pearson correlations between these quantities.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Clean Hamiltonian.
    V_eve_op : ndarray, shape (d, d)
        Eve's coupling operator (perturbation direction).
    O0 : ndarray, shape (d, d)
        Observable operator.
    gammas : array-like
        Coupling strengths to scan.
    t_max : float
        Maximum evolution time.
    n_time_steps : int
        Number of time steps for autocorrelation.
    n_lanczos : int
        Number of Lanczos coefficients to compute.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'gammas': coupling strengths
        - 'lanczos_rmse': Lanczos RMSE for each gamma
        - 'echo_decay_rate': operator echo decay rate for each gamma
        - 'krylov_score': autocorrelation RMSE for each gamma
        - 'correlations': dict of Pearson r and p-values
    """
    from .lanczos_extractor import compute_lanczos_from_matrix

    gammas = np.asarray(gammas)
    dim = H.shape[0]
    times = np.linspace(0, t_max, n_time_steps)

    # Clean channel
    E_clean, V_clean = eigendecompose(H)
    C_clean = compute_operator_autocorrelation(E_clean, V_clean, O0, times, dim)
    b_clean = _fast_lanczos(H, O0, n_lanczos, dim)

    lanczos_rmse = []
    echo_decay_rate = []
    krylov_score = []

    for gamma in gammas:
        H_eve = H + gamma * V_eve_op
        E_eve, V_eve = eigendecompose(H_eve)

        # Lanczos RMSE
        b_eve = _fast_lanczos(H_eve, O0, n_lanczos, dim)
        min_len = min(len(b_clean), len(b_eve))
        rmse = float(np.sqrt(np.mean((b_clean[:min_len] - b_eve[:min_len]) ** 2)))
        lanczos_rmse.append(rmse)

        # Echo decay rate
        rate = compute_echo_decay_rate(E_clean, V_clean, E_eve, V_eve, O0,
                                       t_max=t_max)
        echo_decay_rate.append(rate)

        # Krylov score
        C_eve = compute_operator_autocorrelation(E_eve, V_eve, O0, times, dim)
        score = float(np.sqrt(np.mean((C_clean - C_eve) ** 2)))
        krylov_score.append(score)

    # Correlations
    gammas_sq = gammas ** 2
    r1, p1 = pearsonr(gammas_sq, lanczos_rmse)
    r2, p2 = pearsonr(echo_decay_rate, lanczos_rmse)
    r3, p3 = pearsonr(krylov_score, echo_decay_rate)

    return {
        'gammas': gammas.tolist(),
        'lanczos_rmse': lanczos_rmse,
        'echo_decay_rate': echo_decay_rate,
        'krylov_score': krylov_score,
        'correlations': {
            'gamma2_vs_lanczos': {'r': float(r1), 'p': float(p1)},
            'echo_vs_lanczos': {'r': float(r2), 'p': float(p2)},
            'krylov_vs_echo': {'r': float(r3), 'p': float(p3)},
        },
    }


def _fast_lanczos(H, O0, n_steps, dim):
    """Compute Lanczos coefficients (internal helper)."""
    norm = np.sqrt(np.trace(O0.conj().T @ O0).real / dim)
    O_prev = np.zeros_like(O0)
    O_curr = O0 / norm
    bs = []
    for n in range(n_steps):
        L_O = 1j * (H @ O_curr - O_curr @ H)
        if n > 0:
            L_O = L_O - bs[-1] * O_prev
        b_next = np.sqrt(np.trace(L_O.conj().T @ L_O).real / dim)
        if b_next < 1e-12:
            break
        bs.append(b_next)
        O_prev = O_curr
        O_curr = L_O / b_next
    return np.array(bs)
