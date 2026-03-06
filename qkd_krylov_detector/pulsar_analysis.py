"""
Pulsar Timing Sidereal Analysis
================================

Implements the partial F-test framework for detecting sidereal signals
in pulsar timing residuals, as validated on the NANOGrav 15-year dataset
(PSR J1713+0747, 59,389 TOAs).

Methods:
    - Design matrix construction with annual + sidereal terms
    - Partial F-test for sidereal signal significance
    - Sidereal amplitude and noise RMS estimation
    - Gap classification (diurnal vs. stochastic)

Notebook correspondence:
    pulsar_sidereal_colab-1.ipynb — Cells 5–6

Paper reference:
    [3] D. Süß, "Real-Data Validation on NANOGrav 15-Year Dataset,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18792775

Author: Daniel Süß
License: MIT
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict


def make_design_matrix(mjd: np.ndarray,
                       lst_hours: np.ndarray,
                       include_sidereal: bool = True
                       ) -> np.ndarray:
    """Construct the design matrix for the linear timing model.

    The full model includes:
        - Constant offset (intercept)
        - Annual sinusoidal terms: sin(2π·doy/365.25), cos(2π·doy/365.25)
        - Sidereal sinusoidal terms (optional): sin(2π·LST/24), cos(2π·LST/24)

    Parameters
    ----------
    mjd : ndarray
        Modified Julian Dates of observations.
    lst_hours : ndarray
        Local Sidereal Time in hours (0–24).
    include_sidereal : bool
        Whether to include sidereal terms.

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix with p = 3 (reduced) or 5 (full) columns.

    Notebook correspondence:
        pulsar_sidereal_colab-1.ipynb, Cell 5 — make_X()
    """
    n = len(mjd)
    doy = (mjd % 365.25) / 365.25 * 2 * np.pi
    lst_rad = lst_hours * (2 * np.pi / 24.0)

    cols = [np.ones(n), np.sin(doy), np.cos(doy)]
    if include_sidereal:
        cols += [np.sin(lst_rad), np.cos(lst_rad)]

    return np.column_stack(cols)


def partial_f_test(residuals: np.ndarray,
                   mjd: np.ndarray,
                   lst_hours: np.ndarray
                   ) -> Dict[str, float]:
    """Perform a partial F-test for sidereal signal significance.

    Compares the full model (annual + sidereal terms) against the reduced
    model (annual terms only) to test whether the sidereal terms provide
    a statistically significant improvement.

    Parameters
    ----------
    residuals : ndarray
        Post-fit timing residuals (e.g., in microseconds).
    mjd : ndarray
        Modified Julian Dates.
    lst_hours : ndarray
        Local Sidereal Time in hours.

    Returns
    -------
    result : dict
        Keys:
        - ``'F_statistic'``: F-test statistic
        - ``'p_value'``: p-value (< 0.05 → sidereal signal detected)
        - ``'detection'``: bool, whether p < 0.05
        - ``'sidereal_amplitude'``: amplitude of sidereal signal (same
          units as residuals)
        - ``'noise_rms'``: RMS of residuals after full model subtraction
        - ``'amp_over_noise'``: signal-to-noise ratio
        - ``'coefficients_full'``: fitted coefficients of full model

    Notebook correspondence:
        pulsar_sidereal_colab-1.ipynb, Cell 5 — Method A: Partial F-Test
    """
    Y = residuals.astype(np.float64)
    n = len(Y)

    X_full = make_design_matrix(mjd, lst_hours, include_sidereal=True)
    X_reduced = make_design_matrix(mjd, lst_hours, include_sidereal=False)

    b_full = np.linalg.lstsq(X_full, Y, rcond=None)[0]
    b_reduced = np.linalg.lstsq(X_reduced, Y, rcond=None)[0]

    rss_full = float(np.sum((Y - X_full @ b_full)**2))
    rss_reduced = float(np.sum((Y - X_reduced @ b_reduced)**2))

    p_f = X_full.shape[1]
    p_r = X_reduced.shape[1]

    F_stat = ((rss_reduced - rss_full) / (p_f - p_r)) / \
             (rss_full / (n - p_f))
    p_value = 1 - stats.f.cdf(F_stat, p_f - p_r, n - p_f)

    # Sidereal amplitude from sin/cos coefficients
    sidereal_amp = float(np.sqrt(b_full[3]**2 + b_full[4]**2))
    noise_rms = float(np.std(Y - X_full @ b_full))

    return {
        'F_statistic': float(F_stat),
        'p_value': float(p_value),
        'detection': bool(p_value < 0.05),
        'sidereal_amplitude': sidereal_amp,
        'noise_rms': noise_rms,
        'amp_over_noise': sidereal_amp / (noise_rms + 1e-12),
        'coefficients_full': b_full,
    }


def classify_gaps(mjd: np.ndarray) -> Dict[str, object]:
    """Classify observation gaps as diurnal or stochastic.

    Uses the coefficient of variation (CV) of the time-of-day distribution
    to determine whether gaps follow a diurnal pattern (telescope scheduling)
    or are stochastically distributed.

    Parameters
    ----------
    mjd : ndarray
        Modified Julian Dates.

    Returns
    -------
    result : dict
        Keys:
        - ``'tod_cv'``: coefficient of variation of time-of-day
        - ``'gap_type'``: ``"DIURNAL"`` or ``"STOCHASTIC"``
        - ``'median_cadence_days'``: median time between observations
        - ``'max_gap_days'``: maximum gap between observations

    Notebook correspondence:
        pulsar_sidereal_colab-1.ipynb, Cell 4 — Gap-Analyse
    """
    tod = (mjd % 1.0) * 24.0
    tod_cv = float(np.std(tod) / (np.mean(tod) + 1e-12))
    gap_type = "STOCHASTIC" if tod_cv < 0.5 else "DIURNAL"

    cadences = np.diff(np.sort(mjd))

    return {
        'tod_cv': tod_cv,
        'gap_type': gap_type,
        'median_cadence_days': float(np.median(cadences)),
        'max_gap_days': float(cadences.max()) if len(cadences) > 0 else 0.0,
    }


def compute_sidereal_amplitude(coefficients: np.ndarray) -> float:
    """Extract sidereal amplitude from fitted model coefficients.

    Parameters
    ----------
    coefficients : ndarray
        Full model coefficients from :func:`partial_f_test`.
        Index 3 = sin(LST) coefficient, index 4 = cos(LST) coefficient.

    Returns
    -------
    amplitude : float
        Sidereal signal amplitude.
    """
    return float(np.sqrt(coefficients[3]**2 + coefficients[4]**2))


def compute_noise_rms(residuals: np.ndarray,
                      mjd: np.ndarray,
                      lst_hours: np.ndarray) -> float:
    """Compute RMS of residuals after full model subtraction.

    Parameters
    ----------
    residuals : ndarray
        Post-fit timing residuals.
    mjd : ndarray
        Modified Julian Dates.
    lst_hours : ndarray
        Local Sidereal Time in hours.

    Returns
    -------
    rms : float
        Root-mean-square of the model residuals.
    """
    Y = residuals.astype(np.float64)
    X = make_design_matrix(mjd, lst_hours, include_sidereal=True)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    return float(np.std(Y - X @ b))
