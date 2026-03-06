"""
Layer 1: Sidereal and Diurnal Filter.

FFT-based removal of sidereal (23.93h) and diurnal (24.0h) periodicities
from QBER time series. This is the first layer of the three-layer
detection framework.

The sidereal filter removes environmental coupling (23.93h period) and
thermal cycles (24.0h period) from the QBER signal, leaving the
scrambling-derived residuum for Krylov analysis.

For irregularly sampled data (e.g., pulsar timing), a Lomb-Scargle
based variant is provided.

Reference:
    Paper [1]: Sidereal Framework v1 (DOI: 10.5281/zenodo.18701222)
    Paper [2]: Sidereal Framework v2 (DOI: 10.5281/zenodo.18768750)
    Paper [3]: NANOGrav Validation (DOI: 10.5281/zenodo.18792775)

Author: Daniel Süß
"""

import numpy as np
from scipy.fft import fft, fftfreq, ifft


# Default periods to filter
SIDEREAL_PERIOD = 23.93  # hours
DIURNAL_PERIOD = 24.0    # hours
DEFAULT_BANDWIDTH = 0.008  # frequency bandwidth for notch filter


def sidereal_filter(signal, t_signal=None, periods=None, bw=DEFAULT_BANDWIDTH):
    """
    FFT-based notch filter removing sidereal and diurnal periodicities.

    Zeroes frequency bins within ±bw of 1/period for each specified period.
    This removes environmental coupling and thermal drift while preserving
    the scrambling-derived temporal structure.

    Parameters
    ----------
    signal : array_like
        Input QBER time series (uniformly sampled).
    t_signal : array_like, optional
        Time axis. If None, assumes unit spacing.
    periods : list of float, optional
        Periods to remove (in same units as t_signal).
        Default: [23.93, 24.0] (sidereal + diurnal).
    bw : float
        Bandwidth of the notch filter in frequency units.
        Default: 0.008.

    Returns
    -------
    numpy.ndarray
        Filtered signal (residuum).

    Notes
    -----
    The filter is applied symmetrically in frequency space.
    DC component is preserved.

    Source: krylov_dynamic_detector.ipynb Cell 2 (sidereal_filter),
            eve_detection_master_v3.ipynb Cell 6 (sidereal_filter),
            krylov_robustness_test.ipynb Cell 3 (sidereal_filter)
    """
    signal = np.asarray(signal, dtype=float)

    if periods is None:
        periods = [SIDEREAL_PERIOD, DIURNAL_PERIOD]

    if t_signal is not None:
        dt = t_signal[1] - t_signal[0]
    else:
        dt = 1.0

    yf = fft(signal)
    xf = fftfreq(len(signal), dt)
    f = yf.copy()

    for p in periods:
        f[np.abs(np.abs(xf) - 1.0 / p) < bw] = 0

    return np.real(ifft(f))


def sidereal_filter_irregular(mjd, residuals, periods=None, bw=DEFAULT_BANDWIDTH):
    """
    Sidereal filter for irregularly sampled data using least-squares fitting.

    Instead of FFT (which requires uniform sampling), this fits and subtracts
    sinusoidal components at the specified periods. Used for real astrophysical
    data (e.g., NANOGrav pulsar timing).

    Parameters
    ----------
    mjd : array_like
        Modified Julian Dates of observations.
    residuals : array_like
        Post-fit timing residuals (or QBER values).
    periods : list of float, optional
        Periods to remove in days. Default: [0.99720, 1.0] (sidereal/diurnal in days).
    bw : float
        Not used for irregular sampling (kept for API consistency).

    Returns
    -------
    numpy.ndarray
        Filtered residuals.
    dict
        Fit results containing amplitudes and phases.

    Notes
    -----
    This is the method used for the NANOGrav 15-year dataset validation
    (59,389 TOAs, PSR J1713+0747).

    Source: pulsar_sidereal_colab-1.ipynb Cells 5-6 (F-test + LST fit)
    """
    mjd = np.asarray(mjd, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    if periods is None:
        # Sidereal day = 23.93h = 0.99720 days, Solar day = 24h = 1.0 days
        periods = [SIDEREAL_PERIOD / 24.0, DIURNAL_PERIOD / 24.0]

    n = len(residuals)
    Y = residuals.copy()

    # Build design matrix: constant + sin/cos for each period
    cols = [np.ones(n)]
    for p in periods:
        phase = 2 * np.pi * mjd / p
        cols.append(np.sin(phase))
        cols.append(np.cos(phase))

    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]

    # Subtract fitted periodicities (keep constant term)
    fitted = X @ beta
    filtered = Y - fitted + beta[0]  # preserve mean

    # Extract amplitudes for each period
    fit_results = {}
    for i, p in enumerate(periods):
        a_sin = beta[1 + 2 * i]
        a_cos = beta[2 + 2 * i]
        amplitude = np.sqrt(a_sin**2 + a_cos**2)
        phase = np.arctan2(a_sin, a_cos)
        fit_results[f"period_{p:.5f}d"] = {
            "amplitude": float(amplitude),
            "phase": float(phase),
            "sin_coeff": float(a_sin),
            "cos_coeff": float(a_cos),
        }

    return filtered, fit_results
