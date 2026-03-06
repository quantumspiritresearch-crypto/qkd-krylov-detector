"""
Calibrated Krylov Detector
===========================

Implements two alternative detection methods based on Gaussian slope fitting:

1. **Calibrated Detector** — Calibrates the expected autocorrelation slope
   from clean data, then detects deviations in new measurements.

2. **Option B (Slope Fingerprint)** — Fits a free Gaussian to each window's
   autocorrelation and scores the relative deviation from the theoretical
   b_n slope.

Both methods complement the primary template-matching detector (Layer 3)
by providing slope-based diagnostics.

Notebook correspondence:
    krylov_calibrated_detector.ipynb — Cells 3–6
    krylov_option_b.ipynb — Cells 3–5

Paper reference:
    [6] D. Süß, "Quantum Scrambling as a Cryptographic Resource,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18889224, Section III

Author: Daniel Süß
License: MIT
"""

import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Gaussian autocorrelation template
# ---------------------------------------------------------------------------

def gaussian_ac(t: np.ndarray, s: float) -> np.ndarray:
    """Gaussian autocorrelation template: C(t) = exp(-0.5 * (s*t)^2).

    Parameters
    ----------
    t : ndarray
        Lag axis.
    s : float
        Decay slope parameter.

    Returns
    -------
    C : ndarray
        Gaussian autocorrelation values.
    """
    return np.exp(-0.5 * (s * t)**2)


# ---------------------------------------------------------------------------
# Slope fitting
# ---------------------------------------------------------------------------

def fit_slope(window: np.ndarray,
              s_init: float = 1.5
              ) -> Tuple[float, float]:
    """Fit a free Gaussian to the autocorrelation of a QBER window.

    Normalizes the window, computes the empirical autocorrelation, and
    fits ``C(t) = exp(-0.5 * (s*t)^2)`` to extract the decay slope.

    Parameters
    ----------
    window : ndarray
        QBER residuum window.
    s_init : float
        Initial guess for the slope parameter.

    Returns
    -------
    s_fit : float
        Fitted slope (NaN if fit fails).
    r2 : float
        R-squared goodness of fit.

    Notebook correspondence:
        krylov_calibrated_detector.ipynb, Cell 4 — fit_slope()
        krylov_option_b.ipynb, Cell 3 — fit_slope()
    """
    w = window.copy()
    if w.std() < 1e-10:
        return np.nan, 0.0

    w = (w - w.mean()) / w.std()
    raw = correlate(w, w, mode='same')
    ac = raw[len(raw) // 2:]
    if abs(ac[0]) < 1e-10:
        return np.nan, 0.0
    ac = ac / ac[0]

    n_fit = max(10, len(ac) // 4)
    t_fit = np.arange(n_fit, dtype=float)
    ac_fit = ac[:n_fit]

    if ac_fit.std() < 1e-6:
        return np.nan, 0.0

    try:
        popt, _ = curve_fit(
            gaussian_ac, t_fit, ac_fit,
            p0=[s_init],
            bounds=(0.01, 20.0),
            maxfev=1000
        )
        s_fit = float(popt[0])
        residuals = ac_fit - gaussian_ac(t_fit, s_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ac_fit - ac_fit.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        return s_fit, float(r2)
    except Exception:
        return np.nan, 0.0


# ---------------------------------------------------------------------------
# Calibration from clean data
# ---------------------------------------------------------------------------

def calibrate(clean_residua: list,
              s_bn: float,
              window_size: int = 100,
              step: int = 25
              ) -> Tuple[float, float, np.ndarray]:
    """Calibrate the expected slope from a set of clean QBER residua.

    Fits the Gaussian slope to many windows of clean data and returns
    the median calibrated slope and the scaling factor alpha = s_cal / s_bn.

    Parameters
    ----------
    clean_residua : list of ndarray
        List of clean QBER residuum time series (after sidereal filtering).
    s_bn : float
        Theoretical slope from Lanczos b_n coefficients.
    window_size : int
        Window size for slope fitting.
    step : int
        Step size for sliding windows.

    Returns
    -------
    s_cal : float
        Calibrated median slope.
    alpha : float
        Scaling factor s_cal / s_bn.
    slopes : ndarray
        All fitted slopes from calibration data.

    Notebook correspondence:
        krylov_calibrated_detector.ipynb, Cell 5 — calibrate()
    """
    slopes = []
    for r in clean_residua:
        for i in range(0, len(r) - window_size, step):
            s, _ = fit_slope(r[i:i + window_size])
            if not np.isnan(s):
                slopes.append(s)

    slopes = np.array(slopes)
    if len(slopes) == 0:
        return s_bn, 1.0, slopes

    s_cal = float(np.median(slopes))
    alpha = s_cal / s_bn
    return s_cal, alpha, slopes


# ---------------------------------------------------------------------------
# Calibrated detector
# ---------------------------------------------------------------------------

def calibrated_detect(residuum: np.ndarray,
                      s_cal: float,
                      t: np.ndarray,
                      window_size: int = 100,
                      step: int = 25
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Run the calibrated slope detector on a QBER residuum.

    For each window, fits the Gaussian slope and scores the relative
    deviation from the calibrated slope.

    Parameters
    ----------
    residuum : ndarray
        Sidereal-filtered QBER residuum.
    s_cal : float
        Calibrated slope from :func:`calibrate`.
    t : ndarray
        Time axis.
    window_size : int
        Window size.
    step : int
        Step size.

    Returns
    -------
    t_centers : ndarray
        Window center times.
    scores : ndarray
        Detection scores (|s_fit - s_cal| / s_cal).

    Notebook correspondence:
        krylov_calibrated_detector.ipynb, Cell 6 — detect()
    """
    scores = []
    centers = []
    for i in range(0, len(residuum) - window_size, step):
        s, _ = fit_slope(residuum[i:i + window_size])
        if np.isnan(s):
            scores.append(2.0)
        else:
            scores.append(abs(s - s_cal) / s_cal)
        centers.append(t[i + window_size // 2])
    return np.array(centers), np.array(scores)


# ---------------------------------------------------------------------------
# Option B: Slope fingerprint detector
# ---------------------------------------------------------------------------

def krylov_slope_detector(qber_residuum: np.ndarray,
                          s_bn: float,
                          t: np.ndarray,
                          window_size: int = 100,
                          step: int = 25
                          ) -> Tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """Option B detector: Gaussian slope fitting vs. theoretical b_n slope.

    For each window, fits a free Gaussian to the empirical autocorrelation
    and computes the relative deviation from the theoretical slope derived
    from the Lanczos coefficients.

    Score = |s_fit - s_bn| / s_bn

    Low score  → s_fit close to s_bn → clean channel (b_n fingerprint present)
    High score → s_fit far from s_bn → Eve (no b_n structure)

    Parameters
    ----------
    qber_residuum : ndarray
        Sidereal-filtered QBER residuum.
    s_bn : float
        Theoretical slope from Lanczos b_n (mean of diff(b_n)).
    t : ndarray
        Time axis.
    window_size : int
        Sliding window size.
    step : int
        Step size.

    Returns
    -------
    t_centers : ndarray
        Window center times.
    slope_scores : ndarray
        Relative deviation scores.
    s_fits : ndarray
        Raw fitted slopes per window.
    r2_vals : ndarray
        R-squared values per window.

    Notebook correspondence:
        krylov_option_b.ipynb, Cell 4 — krylov_slope_detector()
    """
    slope_scores = []
    t_centers = []
    s_fits = []
    r2_vals = []
    lags = np.arange(window_size, dtype=float)

    for i in range(0, len(qber_residuum) - window_size, step):
        w = qber_residuum[i:i + window_size]
        std = w.std()
        if std < 1e-10:
            slope_scores.append(1.0)
            t_centers.append(t[i + window_size // 2])
            s_fits.append(np.nan)
            r2_vals.append(0.0)
            continue

        w = (w - w.mean()) / std
        raw = correlate(w, w, mode='same')
        ac = raw[window_size // 2:]
        if abs(ac[0]) < 1e-10:
            slope_scores.append(1.0)
            t_centers.append(t[i + window_size // 2])
            s_fits.append(np.nan)
            r2_vals.append(0.0)
            continue
        ac = ac / ac[0]

        s_fit, r2 = fit_slope(qber_residuum[i:i + window_size], s_init=s_bn)

        if np.isnan(s_fit):
            slope_scores.append(2.0)
        else:
            slope_scores.append(abs(s_fit - s_bn) / s_bn)

        t_centers.append(t[i + window_size // 2])
        s_fits.append(s_fit)
        r2_vals.append(r2)

    return (np.array(t_centers), np.array(slope_scores),
            np.array(s_fits), np.array(r2_vals))
