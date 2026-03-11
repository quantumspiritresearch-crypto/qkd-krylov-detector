"""
QKD Eve Detector — Demo Framework
===================================

Object-oriented wrappers for the detection pipeline, designed for
interactive demonstrations and integration into larger systems.

Classes:
    SiderealFilter  — Configurable sidereal/diurnal filter
    KrylovEngine    — Krylov complexity analysis engine

Functions:
    classify_window — Classify a data window by statistical properties
    make_scenario   — Generate a QBER scenario with a specific attack type

Notebook correspondence:
    qkd_eve_detector_demo.ipynb — All cells

References:
    [5] D. Süß, "QKD Eve Detector: A Unified Framework — Parts I–III,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18873824
"""

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.stats import kurtosis, skew
from scipy.signal import correlate


_SIDEREAL_PERIOD = 23.93


class SiderealFilter:
    """
    Configurable dual-layer sidereal/diurnal filter.

    Parameters
    ----------
    periods : list of float
        Periodicities to remove (hours). Default: [23.93, 24.0].
    bandwidth : float
        Notch filter bandwidth in 1/hours. Default: 0.008.

    Notebook: qkd_eve_detector_demo.ipynb (SiderealFilter class)
    """

    def __init__(self, periods=None, bandwidth=0.008):
        self.periods = periods if periods is not None else [_SIDEREAL_PERIOD, 24.0]
        self.bandwidth = bandwidth

    def filter(self, signal, t):
        """
        Apply FFT notch filter.

        Parameters
        ----------
        signal : array-like
            Input signal.
        t : array-like
            Time axis.

        Returns
        -------
        np.ndarray
            Filtered signal (residuum).
        """
        dt = t[1] - t[0]
        yf = fft(signal)
        xf = fftfreq(len(signal), dt)
        f = yf.copy()
        for p in self.periods:
            f[np.abs(np.abs(xf) - 1.0 / p) < self.bandwidth] = 0
        return np.real(ifft(f))

    def __repr__(self):
        return (f"SiderealFilter(periods={self.periods}, "
                f"bandwidth={self.bandwidth})")


class KrylovEngine:
    """
    Krylov complexity analysis engine.

    .. deprecated:: 2.0.0
        Use :class:`KrylovFramework` from ``krylov_framework.py`` instead.
        KrylovEngine is maintained for backward compatibility but routes
        through the new framework internally.

    Wraps the Lanczos coefficient computation and template-based detection
    into a single object with configurable parameters.

    Parameters
    ----------
    b_n : array-like
        Lanczos coefficients from the baseline Hamiltonian.
    window_size : int
        Sliding window size for detection. Default: 100.
    step : int
        Step size between windows. Default: 25.

    Notebook: qkd_eve_detector_demo.ipynb (KrylovEngine class)
    """

    def __init__(self, b_n, window_size=100, step=25):
        self.b_n = np.asarray(b_n)
        self.window_size = window_size
        self.step = step
        self.slope = float(np.mean(np.diff(self.b_n)))
        self._template = None

    @property
    def template(self):
        """Gaussian autocorrelation template."""
        if self._template is None:
            t_ax = np.arange(self.window_size, dtype=float)
            self._template = np.exp(-0.5 * (self.slope * t_ax)**2)
            if self._template[0] > 0:
                self._template /= self._template[0]
        return self._template

    def detect(self, qber_residuum, t):
        """
        Run template-matching detection on QBER residuum.

        Parameters
        ----------
        qber_residuum : array-like
            Filtered QBER signal.
        t : array-like
            Time axis.

        Returns
        -------
        tuple
            (t_centers, anomaly_scores)
        """
        residuum = np.asarray(qber_residuum)
        t = np.asarray(t)
        theor_corr = self.template

        anomaly_scores = []
        t_centers = []

        for i in range(0, len(residuum) - self.window_size, self.step):
            window = residuum[i:i + self.window_size]
            w_std = window.std()
            if w_std < 1e-10:
                anomaly_scores.append(0.0)
                t_centers.append(t[i + self.window_size // 2])
                continue

            window = (window - window.mean()) / w_std
            raw_corr = correlate(window, window, mode='same')
            real_corr = raw_corr[self.window_size // 2:]
            if abs(real_corr[0]) > 1e-10:
                real_corr = real_corr / real_corr[0]
            else:
                anomaly_scores.append(0.0)
                t_centers.append(t[i + self.window_size // 2])
                continue

            n = min(len(real_corr), len(theor_corr))
            diff = np.sqrt(np.mean((real_corr[:n] - theor_corr[:n])**2))
            anomaly_scores.append(diff)
            t_centers.append(t[i + self.window_size // 2])

        return np.array(t_centers), np.array(anomaly_scores)

    def proxy(self, residuum, t, window=200):
        """
        Higher-moment proxy detector (kurtosis + skewness).

        Parameters
        ----------
        residuum : array-like
            Filtered QBER signal.
        t : array-like
            Time axis.
        window : int
            Sliding window size.

        Returns
        -------
        tuple
            (t_windows, scores)
        """
        residuum = np.asarray(residuum)
        t = np.asarray(t)
        scores, t_windows = [], []

        for i in range(0, len(residuum) - window, window // 4):
            seg = residuum[i:i + window]
            score = (np.sqrt(kurtosis(seg)**2 + skew(seg)**2)
                     * (1 + np.var(seg) * 100))
            scores.append(score)
            t_windows.append(t[i + window // 2])

        return np.array(t_windows), np.array(scores)

    def __repr__(self):
        return (f"KrylovEngine(n_bn={len(self.b_n)}, slope={self.slope:.4f}, "
                f"window={self.window_size}, step={self.step})")


def classify_window(data, threshold_kurt=1.0, threshold_skew=0.5):
    """
    Classify a data window based on statistical properties.

    Parameters
    ----------
    data : array-like
        Data window.
    threshold_kurt : float
        Kurtosis threshold for anomaly.
    threshold_skew : float
        Skewness threshold for anomaly.

    Returns
    -------
    dict
        Keys: kurtosis, skewness, variance, is_anomalous, classification.

    Notebook: qkd_eve_detector_demo.ipynb (classify_window)
    """
    data = np.asarray(data)
    k = float(kurtosis(data))
    s = float(skew(data))
    v = float(np.var(data))
    is_anomalous = abs(k) > threshold_kurt or abs(s) > threshold_skew

    if is_anomalous:
        if abs(s) > threshold_skew:
            classification = "asymmetric_attack"
        else:
            classification = "heavy_tailed_attack"
    else:
        classification = "clean"

    return {
        "kurtosis": k,
        "skewness": s,
        "variance": v,
        "is_anomalous": is_anomalous,
        "classification": classification,
    }


def make_scenario(t, attack_type="clean", eve_start=None, eve_end=None,
                  gamma=0.3, seed=None):
    """
    Generate a QBER scenario with a specific attack type.

    Parameters
    ----------
    t : array-like
        Time axis.
    attack_type : str
        One of: 'clean', 'iid', 'exponential', 'burst', 'gradual'.
    eve_start : float or None
        Attack start time. If None, uses 40% of time range.
    eve_end : float or None
        Attack end time. If None, uses 60% of time range.
    gamma : float
        Attack strength.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Keys: qber, t, attack_type, eve_start, eve_end, gamma.

    Notebook: qkd_eve_detector_demo.ipynb (make_scenario)
    """
    t = np.asarray(t)
    n = len(t)

    if seed is not None:
        np.random.seed(seed)

    if eve_start is None:
        eve_start = t[0] + 0.4 * (t[-1] - t[0])
    if eve_end is None:
        eve_end = t[0] + 0.6 * (t[-1] - t[0])

    # Baseline
    env = (0.04 * np.sin(2 * np.pi * t / _SIDEREAL_PERIOD)
         + 0.02 * np.sin(2 * np.pi * t / 24.0 + 0.5))
    noise = np.random.normal(0, 0.015, n)
    qber = env + noise

    # Attack
    mask = (t >= eve_start) & (t <= eve_end)
    n_attack = mask.sum()

    if attack_type == "clean":
        pass
    elif attack_type == "iid":
        qber[mask] += gamma * 0.25 + np.random.normal(0, 0.005, n_attack)
    elif attack_type == "exponential":
        qber[mask] += gamma * (np.random.exponential(0.10, n_attack) - 0.10)
    elif attack_type == "burst":
        burst_indices = np.random.choice(n_attack, size=max(1, n_attack // 5),
                                         replace=False)
        burst_signal = np.zeros(n_attack)
        burst_signal[burst_indices] = gamma * 0.5
        qber[mask] += burst_signal
    elif attack_type == "gradual":
        ramp = np.linspace(0, gamma * 0.25, n_attack)
        qber[mask] += ramp + np.random.normal(0, 0.003, n_attack)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}. "
                         f"Choose from: clean, iid, exponential, burst, gradual")

    return {
        "qber": qber,
        "t": t,
        "attack_type": attack_type,
        "eve_start": float(eve_start),
        "eve_end": float(eve_end),
        "gamma": gamma,
    }
