"""
Multi-Attack Classifier
=======================

Implements five QKD attack types and a feature-extraction pipeline for
machine-learning-based attack classification.

Attack types:
    0 — Clean (no eavesdropper)
    1 — Intercept-Resend (IR)
    2 — Beam-Splitting (BS)
    3 — Detector Blinding
    4 — Photon-Number-Splitting (PNS)

The feature vector combines QBER residuum statistics and photon-count
anomalies from an automatically detected attack window (KS-test based).

Notebook correspondence:
    eve_attack_classifier_v7.ipynb — Cells 1–5 (final version)
    eve_classifier_validation.ipynb — Cells 2–4

Paper reference:
    [5] D. Süß, "QKD Eve Detector: A Unified Framework — Parts I–III,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18873824, Part II

Author: Daniel Süß
License: MIT
"""

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.stats import ks_2samp
from typing import Tuple, Optional, Dict

# Label mapping
ATTACK_LABELS: Dict[int, str] = {
    0: "Clean",
    1: "IR",
    2: "BeamSplit",
    3: "Blinding",
    4: "PNS",
}

# Default parameters
SIDEREAL_PERIOD = 23.93
N_WINDOWS = 400
N0 = 1000  # baseline photon count per window
ATTACK_WINDOW_SIZE = 130


# ---------------------------------------------------------------------------
# Sidereal filter (simplified, for internal use)
# ---------------------------------------------------------------------------

def _sfilt(sig: np.ndarray, t: np.ndarray, bw: float = 0.008) -> np.ndarray:
    """FFT notch filter removing sidereal and diurnal periodicities."""
    dt = t[1] - t[0]
    yf = fft(sig)
    xf = fftfreq(len(sig), dt)
    f = yf.copy()
    for p in [SIDEREAL_PERIOD, 24.0]:
        f[np.abs(np.abs(xf) - 1.0 / p) < bw] = 0
    return np.real(ifft(f))


# ---------------------------------------------------------------------------
# Baseline channel generation (realistic noise model)
# ---------------------------------------------------------------------------

def base_qber(rng: np.random.Generator,
              n_windows: int = N_WINDOWS) -> np.ndarray:
    """Generate a realistic baseline QBER signal.

    Includes sidereal/diurnal drift, AR(1) process, 1/f (pink) noise,
    and afterpulsing — identical to eve_attack_classifier_v7.ipynb.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n_windows : int
        Number of measurement windows.

    Returns
    -------
    qber : ndarray
        Baseline QBER time series.
    """
    t = np.linspace(0, n_windows, n_windows)

    # Environmental drift
    env = (0.04 * np.sin(2 * np.pi * t / SIDEREAL_PERIOD)
           + 0.02 * np.sin(2 * np.pi * t / 24.0 + 0.5))

    # AR(1) process
    phi = 0.3
    ar = np.zeros(n_windows)
    ar[0] = rng.normal(0, 0.015)
    for i in range(1, n_windows):
        ar[i] = phi * ar[i - 1] + rng.normal(0, 0.015 * np.sqrt(1 - phi**2))

    # 1/f (pink) noise
    fr = np.fft.rfftfreq(n_windows)
    fr[0] = 1e-10
    S = 1.0 / np.sqrt(fr)
    S[0] = 0
    pk = np.fft.irfft(S * np.exp(1j * 2 * np.pi * rng.random(len(fr))),
                       n=n_windows)
    pk = pk * 0.008 / (pk.std() + 1e-10)

    # Afterpulsing
    ap = np.zeros(n_windows)
    base = (rng.random(n_windows) < 0.1).astype(float)
    for i in range(n_windows):
        if base[i] > 0 and rng.random() < 0.05:
            d = int(rng.integers(1, 4))
            if i + d < n_windows:
                ap[i + d] += 0.02

    return env + ar + pk + ap


def base_photons(rng: np.random.Generator,
                 n_windows: int = N_WINDOWS,
                 n0: int = N0) -> np.ndarray:
    """Generate baseline photon count time series.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n_windows : int
        Number of measurement windows.
    n0 : int
        Mean photon count per window.

    Returns
    -------
    photons : ndarray of float
        Photon count time series.
    """
    t = np.linspace(0, n_windows, n_windows)
    drift = (1.0 + 0.05 * np.sin(2 * np.pi * t / SIDEREAL_PERIOD)
             + 0.03 * np.sin(2 * np.pi * t / 24.0 + 1.2))
    return rng.poisson(n0 * drift).astype(float)


# ---------------------------------------------------------------------------
# Attack signal generators
# ---------------------------------------------------------------------------

def make_clean(rng: np.random.Generator,
               n_windows: int = N_WINDOWS
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clean QBER and photon time series (no attack).

    Returns
    -------
    qber : ndarray
    photons : ndarray
    """
    return base_qber(rng, n_windows), base_photons(rng, n_windows)


def make_ir(rng: np.random.Generator,
            eve_start: int = 100,
            eve_end: int = 230,
            n_windows: int = N_WINDOWS
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate intercept-resend attack signal.

    Eve intercepts a random fraction p in [0.1, 0.5] of qubits during
    the attack window, causing QBER += p * 0.25.

    Returns
    -------
    qber : ndarray
    photons : ndarray
    """
    p = float(rng.uniform(0.1, 0.5))
    q = base_qber(rng, n_windows)
    n = base_photons(rng, n_windows)
    span = eve_end - eve_start
    q[eve_start:eve_end] += p * 0.25 + rng.normal(0, 0.005, span)
    n[eve_start:eve_end] = rng.poisson(N0 * 0.92, span).astype(float)
    return q, n


def make_bs(rng: np.random.Generator,
            eve_start: int = 100,
            eve_end: int = 230,
            eta: Optional[float] = None,
            n_windows: int = N_WINDOWS
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate beam-splitting attack signal.

    Eve taps a fraction eta of photons, reducing Bob's photon count
    without directly increasing QBER.

    Parameters
    ----------
    eta : float, optional
        Splitting ratio (random in [0.1, 0.8] if not specified).

    Returns
    -------
    qber : ndarray
    photons : ndarray
    """
    if eta is None:
        eta = float(rng.uniform(0.1, 0.8))
    q = base_qber(rng, n_windows)
    n = base_photons(rng, n_windows)
    span = eve_end - eve_start
    n[eve_start:eve_end] = rng.poisson(
        max(1, N0 * (1 - eta)), span).astype(float)
    q[eve_start:eve_end] += rng.normal(
        0, float(np.sqrt(eta)) * 0.008, span)
    return q, n


def make_blinding(rng: np.random.Generator,
                  eve_start: int = 100,
                  eve_end: int = 230,
                  n_windows: int = N_WINDOWS
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate detector-blinding attack signal.

    Eve blinds Bob's detector with bright pulses every ~10 windows,
    causing periodic QBER spikes and photon count surges.

    Returns
    -------
    qber : ndarray
    photons : ndarray
    """
    q = base_qber(rng, n_windows)
    n = base_photons(rng, n_windows)
    for t in range(eve_start, eve_end, 10):
        if t < n_windows:
            q[t] += 0.08 + rng.normal(0, 0.005)
            n[t] = float(rng.poisson(N0 * 3.5))
    return q, n


def make_pns(rng: np.random.Generator,
             eve_start: int = 100,
             eve_end: int = 230,
             pct: Optional[int] = None,
             n_windows: int = N_WINDOWS
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate photon-number-splitting (PNS) attack signal.

    Eve selectively blocks single-photon pulses and splits multi-photon
    pulses, causing a drop in low-count photon events.

    Parameters
    ----------
    pct : int, optional
        Percentile threshold for photon clipping (random in [50, 80] if
        not specified).

    Returns
    -------
    qber : ndarray
    photons : ndarray
    """
    if pct is None:
        pct = int(rng.integers(50, 81))
    q = base_qber(rng, n_windows)
    n = base_photons(rng, n_windows)
    span = eve_end - eve_start
    nfl = np.percentile(n[eve_start:eve_end], 100 - pct)
    n2 = n.copy()
    mask = n2[eve_start:eve_end] < nfl
    n2[eve_start:eve_end][mask] = rng.poisson(
        N0 * 0.5, int(mask.sum())).astype(float)
    q2 = q.copy()
    q2[eve_start:eve_end] += rng.normal(0, 0.003, span)
    return q2, n2


# ---------------------------------------------------------------------------
# Attack window detection
# ---------------------------------------------------------------------------

def find_attack_window(qber: np.ndarray,
                       photons: np.ndarray,
                       window_size: int = ATTACK_WINDOW_SIZE,
                       step: int = 10
                       ) -> Tuple[int, int, float]:
    """Detect the most anomalous window using the KS statistic.

    Slides a window across the time series and finds the position where
    the combined KS distance (QBER + photon count) from the baseline
    is maximized.

    Parameters
    ----------
    qber : ndarray
        QBER residuum (after sidereal filtering).
    photons : ndarray
        Photon count time series.
    window_size : int
        Size of the sliding window.
    step : int
        Step size for sliding.

    Returns
    -------
    start : int
        Start index of detected window.
    end : int
        End index of detected window.
    max_ks : float
        Maximum KS statistic found.
    """
    bq = qber[:50]
    bn = photons[:50]
    best_ks = 0.0
    best_pos = 50

    for s in range(50, len(qber) - window_size, step):
        ks_q = ks_2samp(qber[s:s + window_size], bq)[0]
        ks_n = ks_2samp(photons[s:s + window_size], bn)[0]
        ks = max(ks_q, ks_n)
        if ks > best_ks:
            best_ks = ks
            best_pos = s

    return best_pos, best_pos + window_size, best_ks


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(qber: np.ndarray,
                     photons: np.ndarray,
                     t: Optional[np.ndarray] = None,
                     n0: int = N0
                     ) -> np.ndarray:
    """Extract a feature vector for attack classification.

    Applies sidereal filtering, detects the attack window via KS-test,
    and computes a comprehensive feature vector combining QBER residuum
    and photon-count statistics.

    Parameters
    ----------
    qber : ndarray
        Raw QBER time series.
    photons : ndarray
        Photon count time series.
    t : ndarray, optional
        Time axis (default: linspace matching qber length).
    n0 : int
        Baseline photon count for normalization.

    Returns
    -------
    features : ndarray
        Feature vector (length ~55) suitable for classification.

    Notebook correspondence:
        eve_attack_classifier_v7.ipynb, Cell 4 — extract_features()
    """
    n_windows = len(qber)
    if t is None:
        t = np.linspace(0, n_windows, n_windows)

    rq = _sfilt(qber, t)
    es, ee, max_ks = find_attack_window(rq, photons)

    att_q = rq[es:ee]
    base_q = rq[:50]
    att_n = photons[es:ee]
    base_n = photons[:50]

    feats = []

    def _seg_feats(att, base):
        """Segment-level features comparing attack vs. baseline."""
        p50a = np.percentile(att, 50) + 1e-10
        p50b = np.percentile(base, 50) + 1e-10
        sa = [att.mean(), att.std(), att.var(),
              np.percentile(att, 10) / p50a,
              np.percentile(att, 90) / p50a,
              att.max() - att.min()]
        sb = [base.mean(), base.std(), base.var(),
              np.percentile(base, 10) / p50b,
              np.percentile(base, 90) / p50b,
              base.max() - base.min()]
        out = []
        for a, b in zip(sa, sb):
            out += [a - b, abs(a - b)]

        def _ar1(s):
            return (float(np.corrcoef(s[:-1], s[1:])[0, 1])
                    if s.std() > 1e-10 else 0.0)

        out.append(_ar1(att) - _ar1(base))
        ks_stat, pv = ks_2samp(att, base)
        out += [float(ks_stat), float(-np.log(pv + 1e-12))]

        sw = max(1, len(att) // 4)
        out += [att[i * sw:(i + 1) * sw].mean() for i in range(4)]
        out += [att[i * sw:(i + 1) * sw].var() for i in range(4)]
        return out

    feats += _seg_feats(att_q, base_q)
    feats += _seg_feats(att_n, base_n)
    feats.append(float(max_ks))

    # Cross-correlation between QBER and photon anomalies
    dq = att_q - att_q.mean()
    dn = att_n - att_n.mean()
    if dq.std() > 1e-10 and dn.std() > 1e-10:
        feats.append(float(np.corrcoef(dq, dn)[0, 1]))
    else:
        feats.append(0.0)

    # Spectral band ratios
    fa = np.abs(np.fft.rfft(att_q))**2
    fb = np.abs(np.fft.rfft(base_q))**2
    nb2 = max(1, min(len(fa), len(fb)) // 4)
    for i in range(4):
        pa = fa[i * nb2:(i + 1) * nb2].mean() + 1e-12
        pb = fb[i * nb2:(i + 1) * nb2].mean() + 1e-12
        feats.append(float(np.log(pa / pb)))

    # Global anomaly features
    feats.append(float(photons.max()))
    feats.append(float(photons.max() / (photons.mean() + 1e-10)))
    feats.append(float(np.percentile(photons, 99)))
    feats.append(float(rq.max()))
    feats.append(float(rq.max() / (rq.std() + 1e-10)))
    feats.append(float((photons > 2 * n0).sum()))
    qth = rq.mean() + 5 * rq.std()
    feats.append(float((rq > qth).sum()))

    return np.nan_to_num(np.array(feats, dtype=float),
                         nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Process-level statistical tests
# ---------------------------------------------------------------------------

def cusum_detect(series: np.ndarray,
                 t: Optional[np.ndarray] = None,
                 k_factor: float = 0.5,
                 h_factor: float = 4.0
                 ) -> Tuple[np.ndarray, float, bool]:
    """Two-sided CUSUM change-point detection on filtered QBER.

    Parameters
    ----------
    series : ndarray
        Raw QBER time series.
    t : ndarray, optional
        Time axis.
    k_factor : float
        Allowance as multiple of sigma (sensitivity/specificity tradeoff).
    h_factor : float
        Decision interval as multiple of sigma (threshold).

    Returns
    -------
    alarms : ndarray of bool
        Alarm flags per time step.
    max_cusum : float
        Maximum CUSUM value normalized by threshold.
    alarm_flag : bool
        Whether any alarm was triggered.

    Notebook correspondence:
        bb84_process_level.ipynb, Cell 3 — cusum_detect()
    """
    n = len(series)
    if t is None:
        t = np.linspace(0, n, n)

    residuum = _sfilt(series, t)
    mu0 = residuum[:50].mean()
    sigma = residuum[:50].std()
    if sigma < 1e-12:
        return np.zeros(n, dtype=bool), 0.0, False

    k = k_factor * sigma
    h = h_factor * sigma

    S_pos = np.zeros(n)
    S_neg = np.zeros(n)
    for i in range(1, n):
        S_pos[i] = max(0, S_pos[i - 1] + (residuum[i] - mu0) - k)
        S_neg[i] = max(0, S_neg[i - 1] - (residuum[i] - mu0) - k)

    alarms = (S_pos > h) | (S_neg > h)
    max_cusum = max(S_pos.max(), S_neg.max()) / h
    return alarms, float(max_cusum), bool(alarms.any())


def spectral_anomaly_score(series: np.ndarray,
                           t: Optional[np.ndarray] = None
                           ) -> float:
    """Compute spectral anomaly score from Welch periodogram.

    Score = max power in non-sidereal band / median power.
    Clean channels yield score ~1 (flat spectrum); Eve activity
    introduces structured spectral power.

    Parameters
    ----------
    series : ndarray
        Raw QBER time series.
    t : ndarray, optional
        Time axis.

    Returns
    -------
    score : float
        Spectral anomaly score.

    Notebook correspondence:
        bb84_process_level.ipynb, Cell 5 — spectral_anomaly_score()
    """
    from scipy.signal import welch

    n = len(series)
    if t is None:
        t = np.linspace(0, n, n)

    residuum = _sfilt(series, t)
    dt = t[1] - t[0]
    freqs, psd = welch(residuum, nperseg=min(64, len(residuum) // 4))
    freqs_real = freqs / dt

    # Exclude sidereal, diurnal, and DC bands
    sidereal_mask = np.abs(freqs_real - 1 / SIDEREAL_PERIOD) < 0.01
    diurnal_mask = np.abs(freqs_real - 1 / 24.0) < 0.01
    dc_mask = freqs_real < 0.001
    exclude = sidereal_mask | diurnal_mask | dc_mask

    psd_filtered = psd[~exclude]
    if len(psd_filtered) == 0:
        return 1.0
    return float(psd_filtered.max() / (np.median(psd_filtered) + 1e-12))
