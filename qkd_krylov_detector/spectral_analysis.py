"""
Spectral Statistics Analysis.

Computes the ⟨r⟩ ratio (mean ratio of consecutive level spacings) to
classify the spectral statistics of the Hamiltonian:

    Poisson (integrable):  ⟨r⟩ ≈ 0.386
    GOE (fully chaotic):   ⟨r⟩ ≈ 0.536

The default 8-qubit Heisenberg chain is in the crossover regime
(⟨r⟩ ≈ 0.366), which is between Poisson and GOE.

IMPORTANT: The system does NOT exhibit full GOE statistics.
This is explicitly stated in Paper [4] and Paper [6], Section II.C.

Reference:
    Paper [4]: Scrambling vs. Recurrence (DOI: 10.5281/zenodo.18813710)
    Paper [5]: QKD Eve Detector Parts I-III (DOI: 10.5281/zenodo.18873824)
    eve_detection_master_v3.ipynb Cell 4 (compute_r_ratio)

Author: Daniel Süß
"""

import numpy as np


# Reference values for spectral statistics
R_POISSON = 0.386   # Integrable systems
R_GOE = 0.536       # Fully chaotic (Gaussian Orthogonal Ensemble)


def compute_r_ratio(H, bulk_fraction=0.1):
    """
    Compute the mean ratio of consecutive level spacings ⟨r⟩.

    Uses the bulk of the spectrum (excluding 10% from each edge)
    to avoid edge effects.

    Parameters
    ----------
    H : qutip.Qobj
        System Hamiltonian.
    bulk_fraction : float
        Fraction of eigenvalues to exclude from each edge (default: 0.1).

    Returns
    -------
    float
        Mean ⟨r⟩ ratio.

    Notes
    -----
    ⟨r⟩ = mean(min(s_i, s_{i+1}) / max(s_i, s_{i+1}))
    where s_i = E_{i+1} - E_i are the level spacings.

    Source: eve_detection_master_v3.ipynb Cell 4 (compute_r_ratio)
    """
    en = H.eigenenergies()
    n = len(en)
    start = int(n * bulk_fraction)
    end = int(n * (1 - bulk_fraction))
    bulk = en[start:end]

    spacings = np.diff(bulk)
    spacings = spacings[spacings > 1e-12]  # Remove degeneracies

    if len(spacings) < 2:
        return 0.0

    r_values = []
    for i in range(len(spacings) - 1):
        s_min = min(spacings[i], spacings[i + 1])
        s_max = max(spacings[i], spacings[i + 1])
        if s_max > 0:
            r_values.append(s_min / s_max)

    return float(np.mean([x for x in r_values if np.isfinite(x)]))


def classify_regime(r_value):
    """
    Classify the spectral regime based on ⟨r⟩.

    Parameters
    ----------
    r_value : float
        Mean ⟨r⟩ ratio from compute_r_ratio().

    Returns
    -------
    str
        Regime classification string.

    Notes
    -----
    Classification thresholds from eve_detection_master_v3.ipynb Cell 4:
        r > 0.50: near-GOE (strongly chaotic)
        0.42 < r ≤ 0.50: crossover regime (partial chaos)
        r ≤ 0.42: crossover regime (closer to Poisson)

    Source: eve_detection_master_v3.ipynb Cell 4
    """
    if r_value > 0.50:
        return "near-GOE (strongly chaotic)"
    elif r_value > 0.42:
        return "crossover regime (partial chaos, intermediate statistics)"
    else:
        return "crossover regime (closer to Poisson)"
