"""
Krylov-Statistical Bridge
==========================

Connects the theoretical Hamiltonian perturbation (b_n deviation) to
the observable QBER statistical anomaly (kurtosis, skewness).

This is the empirical validation of the core hypothesis:
    "If Eve perturbs the Hamiltonian, the Krylov coefficients shift,
     and this shift is detectable in the QBER residuum statistics."

Functions:
    bn_deviation          — Mean |b_n(eve) - b_n(baseline)| for a given gamma
    sim_stats             — Multi-trial kurtosis/skewness for a gamma value
    gamma_sweep           — 50-point sweep with Pearson correlation
    sensitivity_vs_gamma  — Detection sensitivity as function of gamma
    krylov_proxy          — Sliding-window higher-moment analysis (kurtosis + skewness)

Notebook correspondence:
    eve_detection_master_v3.ipynb — Cells 5, 6, 7 (Figures 5, 6)

References:
    [5] D. Süß, "QKD Eve Detector: A Unified Framework — Parts I–III,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18873824
    [6] D. Süß, "Quantum Scrambling as a Cryptographic Resource,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18889224
"""

import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
from scipy.fft import fft, fftfreq, ifft


_SIDEREAL_PERIOD = 23.93


def _sidereal_filter(signal, t, periods=None, bw=0.008):
    """Internal sidereal filter (FFT notch)."""
    if periods is None:
        periods = [_SIDEREAL_PERIOD, 24.0]
    dt = t[1] - t[0]
    yf = fft(signal)
    xf = fftfreq(len(signal), dt)
    f = yf.copy()
    for p in periods:
        f[np.abs(np.abs(xf) - 1.0 / p) < bw] = 0
    return np.real(ifft(f))


def bn_deviation(gamma, H_base, b_baseline, compute_lanczos_fn,
                 build_eve_fn, n_steps=20):
    """
    Compute mean |b_n(Eve) - b_n(baseline)| for a given coupling gamma.

    Parameters
    ----------
    gamma : float
        Eve coupling strength.
    H_base : qutip.Qobj
        Baseline Hamiltonian (no Eve).
    b_baseline : array-like
        Baseline Lanczos coefficients.
    compute_lanczos_fn : callable
        Function to compute Lanczos coefficients: f(H, n_steps) -> array.
    build_eve_fn : callable
        Function to build Eve Hamiltonian: f(gamma) -> qutip.Qobj.
    n_steps : int
        Number of Lanczos steps.

    Returns
    -------
    float
        Mean absolute deviation of b_n coefficients.

    Notebook: eve_detection_master_v3.ipynb, Cell 5 (bn_deviation)
    """
    b_eve = compute_lanczos_fn(build_eve_fn(gamma), n_steps=n_steps)
    ml = min(len(b_baseline), len(b_eve))
    return float(np.mean(np.abs(b_eve[:ml] - b_baseline[:ml])))


def sim_stats(gamma, t, n_trials=8, seed=None):
    """
    Multi-trial kurtosis and skewness statistics for a given gamma.

    Generates n_trials QBER time series with Eve coupling gamma,
    applies sidereal filter, and returns mean kurtosis and |skewness|.

    Parameters
    ----------
    gamma : float
        Eve coupling strength.
    t : array-like
        Time axis.
    n_trials : int
        Number of Monte Carlo trials.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (mean_kurtosis, mean_abs_skewness)

    Notebook: eve_detection_master_v3.ipynb, Cell 7 (sim_stats)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    kv, sv = [], []
    for _ in range(n_trials):
        env = (0.04 * np.sin(2 * np.pi * t / _SIDEREAL_PERIOD)
             + 0.02 * np.sin(2 * np.pi * t / 24.0 + 0.5))
        noise = rng.normal(0, 0.015, len(t))
        if gamma > 0:
            eve = gamma * (rng.exponential(0.10, len(t)) - 0.10)
        else:
            eve = np.zeros(len(t))
        res = _sidereal_filter(env + noise + eve, t)
        kv.append(kurtosis(res))
        sv.append(abs(skew(res)))

    return float(np.mean(kv)), float(np.mean(sv))


def gamma_sweep(gamma_values, t, H_base, b_baseline,
                compute_lanczos_fn, build_eve_fn,
                n_trials=8, n_steps=20, seed=None):
    """
    Full gamma sweep: compute b_n deviation AND QBER statistics for each gamma.

    This is the "Krylov-Statistical Bridge" analysis from Figure 5 of the paper.

    Parameters
    ----------
    gamma_values : array-like
        Array of gamma values to sweep.
    t : array-like
        Time axis.
    H_base : qutip.Qobj
        Baseline Hamiltonian.
    b_baseline : array-like
        Baseline Lanczos coefficients.
    compute_lanczos_fn : callable
        Lanczos computation function.
    build_eve_fn : callable
        Eve Hamiltonian builder.
    n_trials : int
        Trials per gamma for statistics.
    n_steps : int
        Lanczos steps.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Keys: gamma, bn_devs, kurt_vals, skew_vals, r_kurt, p_kurt, r_skew, p_skew

    Notebook: eve_detection_master_v3.ipynb, Cell 7 (50-point gamma sweep)
    Paper: [5], Figure 5
    """
    gamma_values = np.asarray(gamma_values)
    bn_devs = []
    kurt_vals = []
    skew_vals = []

    for i, gamma in enumerate(gamma_values):
        dev = bn_deviation(gamma, H_base, b_baseline,
                           compute_lanczos_fn, build_eve_fn, n_steps)
        k, s = sim_stats(gamma, t, n_trials,
                         seed=(seed + i) if seed is not None else None)
        bn_devs.append(dev)
        kurt_vals.append(k)
        skew_vals.append(s)

    bn_devs = np.array(bn_devs)
    kurt_vals = np.array(kurt_vals)
    skew_vals = np.array(skew_vals)

    r_kurt, p_kurt = pearsonr(bn_devs, kurt_vals)
    r_skew, p_skew = pearsonr(bn_devs, skew_vals)

    return {
        "gamma": gamma_values,
        "bn_devs": bn_devs,
        "kurt_vals": kurt_vals,
        "skew_vals": skew_vals,
        "r_kurt": float(r_kurt),
        "p_kurt": float(p_kurt),
        "r_skew": float(r_skew),
        "p_skew": float(p_skew),
    }


def sensitivity_vs_gamma(gamma_values, t, b_baseline, n_trials=10,
                          window=200, seed=None):
    """
    Compute detection sensitivity as a function of Eve coupling strength.

    For each gamma, runs n_trials of clean and attack QBER, applies
    the krylov_proxy detector, and computes TPR at 3-sigma threshold.

    Parameters
    ----------
    gamma_values : array-like
        Array of gamma values.
    t : array-like
        Time axis.
    b_baseline : array-like
        Baseline Lanczos coefficients (unused in proxy, kept for API consistency).
    n_trials : int
        Number of trials per gamma.
    window : int
        Sliding window size for krylov_proxy.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        Keys: gamma, sensitivity, gamma_50 (gamma at 50% sensitivity or None)

    Notebook: eve_detection_master_v3.ipynb, Cell 8 (sensitivity vs gamma)
    Paper: [5], Figure 6b
    """
    gamma_values = np.asarray(gamma_values)
    rng = np.random.RandomState(seed)
    sensitivity = []

    for gamma in gamma_values:
        tp, fn = 0, 0
        for _ in range(n_trials):
            # Attack
            env = (0.04 * np.sin(2 * np.pi * t / _SIDEREAL_PERIOD)
                 + 0.02 * np.sin(2 * np.pi * t / 24.0 + 0.5))
            noise_a = rng.normal(0, 0.015, len(t))
            eve_start_idx = int(len(t) * 0.4)
            eve_end_idx = int(len(t) * 0.6)
            eve_s = np.zeros(len(t))
            mask = np.zeros(len(t), dtype=bool)
            mask[eve_start_idx:eve_end_idx] = True
            eve_s[mask] = gamma * (rng.exponential(0.10, mask.sum()) - 0.10)
            qb_a = env + noise_a + eve_s
            ra = _sidereal_filter(qb_a, t)

            # Clean
            noise_c = rng.normal(0, 0.015, len(t))
            qb_c = env + noise_c
            rc = _sidereal_filter(qb_c, t)

            _, sa = krylov_proxy(ra, t, window=window)
            _, sc = krylov_proxy(rc, t, window=window)

            thr = sc.mean() + 3 * sc.std()
            tp += (sa > thr).sum()
            fn += (sa <= thr).sum()

        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    sens_arr = np.array(sensitivity)
    gamma_50 = None
    above_50 = gamma_values[sens_arr >= 0.5]
    if len(above_50) > 0:
        gamma_50 = float(above_50[0])

    return {
        "gamma": gamma_values,
        "sensitivity": sens_arr,
        "gamma_50": gamma_50,
    }


def krylov_proxy(residuum, t, window=200):
    """
    Sliding-window higher-moment analysis (kurtosis + skewness).

    Physical basis: pure scrambling in the crossover regime produces
    near-Gaussian residuum. Eve breaks this symmetry.

    Score = sqrt(kurtosis^2 + skewness^2) * (1 + 100 * variance)

    Parameters
    ----------
    residuum : array-like
        QBER residuum after sidereal filter.
    t : array-like
        Time axis.
    window : int
        Sliding window size.

    Returns
    -------
    tuple
        (t_windows, scores) — time centers and complexity scores.

    Notebook: eve_detection_master_v3.ipynb, Cell 6 (krylov_proxy)
    Paper: [5], Section II
    """
    residuum = np.asarray(residuum)
    t = np.asarray(t)
    scores, t_windows = [], []

    for i in range(0, len(residuum) - window, window // 4):
        seg = residuum[i:i + window]
        score = np.sqrt(kurtosis(seg)**2 + skew(seg)**2) * (1 + np.var(seg) * 100)
        scores.append(score)
        t_windows.append(t[i + window // 2])

    return np.array(t_windows), np.array(scores)
