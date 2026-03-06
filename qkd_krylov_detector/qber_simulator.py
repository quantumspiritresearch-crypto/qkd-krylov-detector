"""
QBER Time Series Simulation.

Generates synthetic QBER time series for clean channels and various
Eve attack models. Two noise models are provided:

1. Idealized: Gaussian noise + sidereal/diurnal drift (from krylov_dynamic_detector.ipynb)
2. Realistic: AR(1) + 1/f + afterpulsing + burst noise (from krylov_robustness_test.ipynb)

Eve attack models:
    - iid intercept-resend (classical, from Paper [5] Part I)
    - Hamiltonian perturbation (from eve_detection_master_v3.ipynb)
    - Exponential distribution (asymmetric, from eve_detection_master_v3.ipynb)

Reference:
    Paper [5]: QKD Eve Detector Parts I-III (DOI: 10.5281/zenodo.18873824)
    krylov_dynamic_detector.ipynb: Idealized QBER model
    krylov_robustness_test.ipynb: Realistic noise model

Author: Daniel Süß
"""

import numpy as np
from .sidereal_filter import SIDEREAL_PERIOD, DIURNAL_PERIOD


# Default simulation parameters
DEFAULT_N_WINDOWS = 400
DEFAULT_T_MAX = 400.0


def _default_t_signal(n_windows=DEFAULT_N_WINDOWS, t_max=DEFAULT_T_MAX):
    """Generate default time axis."""
    return np.linspace(0, t_max, n_windows)


def _environmental_drift(t_signal):
    """
    Sidereal + diurnal environmental drift.

    Source: All notebooks use this identical model.
    """
    return (0.04 * np.sin(2 * np.pi * t_signal / SIDEREAL_PERIOD)
            + 0.02 * np.sin(2 * np.pi * t_signal / DIURNAL_PERIOD + 0.5))


def make_clean_qber(t_signal=None, noise_std=0.015, seed=None):
    """
    Generate idealized clean QBER time series.

    Components:
        - Sidereal drift (23.93h period): environmental coupling
        - Diurnal drift (24.0h period): thermal cycle
        - Gaussian white noise: pure scrambling residual

    Parameters
    ----------
    t_signal : array_like, optional
        Time axis. If None, uses default (0 to 400, 400 points).
    noise_std : float
        Standard deviation of Gaussian noise (default: 0.015).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        QBER time series.

    Source: krylov_dynamic_detector.ipynb Cell 2 (simulate_qber, eve_active=False)
    """
    if seed is not None:
        np.random.seed(seed)
    if t_signal is None:
        t_signal = _default_t_signal()

    env = _environmental_drift(t_signal)
    noise = np.random.normal(0, noise_std, len(t_signal))

    return env + noise


def make_eve_qber(t_signal=None, eve_type="iid", gamma=0.3,
                  eve_start=150, eve_end=280, p_ir=0.25,
                  noise_std=0.015, seed=None):
    """
    Generate QBER time series with Eve's eavesdropping signal.

    Parameters
    ----------
    t_signal : array_like, optional
        Time axis.
    eve_type : str
        Type of Eve model:
        - "iid": Classical intercept-resend (iid errors, Paper [5] Part I)
        - "exponential": Asymmetric exponential distribution (eve_detection_master_v3)
        - "hamiltonian": Perturbation-based (gamma * sx(1)*sx(2))
    gamma : float
        Eve coupling strength (default: 0.3).
    eve_start : int
        Start index of Eve's attack window.
    eve_end : int
        End index of Eve's attack window.
    p_ir : float
        Intercept-resend probability for iid model (default: 0.25).
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray
        QBER time series with Eve signal.

    Source: krylov_dynamic_detector.ipynb Cell 2 (simulate_qber, eve_active=True),
            eve_detection_master_v3.ipynb Cell 6 (simulate_qber)
    """
    if seed is not None:
        np.random.seed(seed)
    if t_signal is None:
        t_signal = _default_t_signal()

    qber = make_clean_qber(t_signal, noise_std)
    n = len(t_signal)
    eve_signal = np.zeros(n)

    if eve_type == "iid":
        # Classical intercept-resend: constant offset + small noise
        # Source: krylov_dynamic_detector.ipynb
        n_eve = eve_end - eve_start
        eve_signal[eve_start:eve_end] = (
            p_ir * 0.25 * np.ones(n_eve)
            + np.random.normal(0, 0.005, n_eve)
        )

    elif eve_type == "exponential":
        # Asymmetric exponential: models systematic information extraction
        # Source: eve_detection_master_v3.ipynb Cell 6
        mask = (t_signal >= t_signal[eve_start]) & (t_signal <= t_signal[min(eve_end, n - 1)])
        n_eve = mask.sum()
        eve_signal[mask] = gamma * (
            np.random.exponential(0.10, n_eve) - 0.10
        )

    elif eve_type == "hamiltonian":
        # Hamiltonian perturbation model
        # Eve perturbs the coupling, creating correlated errors
        n_eve = eve_end - eve_start
        eve_signal[eve_start:eve_end] = (
            gamma * np.random.normal(0, 0.05, n_eve)
        )

    else:
        raise ValueError(f"Unknown eve_type: {eve_type}. "
                         f"Use 'iid', 'exponential', or 'hamiltonian'.")

    return qber + eve_signal


def make_realistic_clean_qber(t_signal=None, ar1_coeff=0.3,
                               pink_amp=0.008, afterpulse_p=0.05,
                               burst_rate=0.005, seed=None):
    """
    Generate realistic clean QBER with hardware noise model.

    Four noise components that real QKD systems exhibit:
    1. AR(1) correlated noise — photon detector memory / thermal fluctuations
    2. 1/f (pink) noise — slow environmental drift not removed by sidereal filter
    3. Detector afterpulsing — periodic memory effect at short timescales
    4. Burst noise — occasional spike events from cosmic rays / electronics

    Parameters
    ----------
    t_signal : array_like, optional
        Time axis.
    ar1_coeff : float
        AR(1) correlation coefficient (0=white, 1=very correlated). Default: 0.3.
    pink_amp : float
        Amplitude of 1/f noise component. Default: 0.008.
    afterpulse_p : float
        Afterpulsing probability. Default: 0.05.
    burst_rate : float
        Fraction of windows affected by burst events. Default: 0.005.
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray
        Realistic QBER time series.

    Notes
    -----
    This noise model is NOT constructed to match the Gaussian b_n template.
    The AR(1) decay is exponential, not Gaussian.
    The 1/f component has power-law correlations.
    This is the honest test of the detector.

    Source: krylov_robustness_test.ipynb Cell 4 (make_realistic_clean_qber)
    """
    if seed is not None:
        np.random.seed(seed)
    if t_signal is None:
        t_signal = _default_t_signal()

    n = len(t_signal)

    # Environmental drift (removed by sidereal filter)
    env = _environmental_drift(t_signal)

    # Component 1: AR(1) correlated noise
    # C(tau) = phi^tau — EXPONENTIAL, not Gaussian
    ar1_noise = np.zeros(n)
    ar1_noise[0] = np.random.normal(0, 0.015)
    for t in range(1, n):
        ar1_noise[t] = (ar1_coeff * ar1_noise[t - 1]
                        + np.random.normal(0, 0.015 * np.sqrt(1 - ar1_coeff**2)))

    # Component 2: 1/f (pink) noise via spectral method
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1e-10
    pink_spectrum = 1.0 / np.sqrt(freqs)
    pink_spectrum[0] = 0
    phases = np.exp(1j * 2 * np.pi * np.random.random(len(freqs)))
    pink_noise = np.fft.irfft(pink_spectrum * phases, n=n)
    pink_std = pink_noise.std()
    if pink_std > 0:
        pink_noise = pink_noise * pink_amp / pink_std

    # Component 3: Afterpulsing
    base_counts = (np.random.random(n) < 0.1).astype(float)
    afterpulse_noise = np.zeros(n)
    for t in range(n):
        if base_counts[t] > 0 and np.random.random() < afterpulse_p:
            delay = np.random.randint(1, 4)
            if t + delay < n:
                afterpulse_noise[t + delay] += 0.02

    # Component 4: Burst noise
    burst_noise = np.zeros(n)
    n_bursts = max(1, int(n * burst_rate))
    burst_times = np.random.choice(n, n_bursts, replace=False)
    burst_noise[burst_times] = np.random.exponential(0.05, n_bursts)

    return env + ar1_noise + pink_noise + afterpulse_noise + burst_noise


def make_realistic_eve_qber(t_signal=None, p_ir=0.3,
                             eve_start=150, eve_end=280, seed=None,
                             **noise_kwargs):
    """
    Generate realistic QBER with Eve: hardware noise + iid eavesdropping.

    Parameters
    ----------
    t_signal : array_like, optional
        Time axis.
    p_ir : float
        Intercept-resend probability (default: 0.3).
    eve_start : int
        Start index of Eve's attack.
    eve_end : int
        End index of Eve's attack.
    seed : int, optional
        Random seed.
    **noise_kwargs
        Additional parameters for make_realistic_clean_qber().

    Returns
    -------
    numpy.ndarray
        Realistic QBER with Eve signal.

    Source: krylov_robustness_test.ipynb Cell 4 (make_realistic_eve_qber)
    """
    if t_signal is None:
        t_signal = _default_t_signal()

    qber = make_realistic_clean_qber(t_signal, seed=seed, **noise_kwargs)
    n_eve = eve_end - eve_start
    eve_signal = np.zeros(len(t_signal))
    eve_signal[eve_start:eve_end] = (
        p_ir * 0.25 * np.ones(n_eve)
        + np.random.normal(0, 0.005, n_eve)
    )

    return qber + eve_signal
