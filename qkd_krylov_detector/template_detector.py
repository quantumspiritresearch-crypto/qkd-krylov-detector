"""
Layer 3: Krylov Dynamic Detector — Template Matching.

Compares the empirical QBER autocorrelation against the theoretical
Gaussian template derived from the Lanczos coefficients b_n.

The detector computes a sliding-window RMSE score between the measured
autocorrelation and the template. Clean channels match the template
(low score); Eve-compromised channels deviate (high score).

Performance (from Paper [5]):
    - Idealized noise:  AUC = 0.9899, separation = 22.25σ
    - Realistic noise:  AUC = 0.9989, separation = 12.13σ

Reference:
    Paper [5]: QKD Eve Detector Parts I-III (DOI: 10.5281/zenodo.18873824)
    Paper [6]: Quantum Scrambling as Cryptographic Resource (DOI: 10.5281/zenodo.18889224)

Author: Daniel Süß
"""

import numpy as np
from scipy.signal import correlate


def krylov_dynamic_detector(qber_residuum, b_n, t_axis,
                            window_size=100, step=25):
    """
    Sliding-window template matching detector.

    For each window of the QBER residuum:
    1. Compute the empirical autocorrelation
    2. Compare against the Gaussian template from b_n
    3. Score = RMSE between empirical and theoretical AC

    Parameters
    ----------
    qber_residuum : array_like
        Sidereal-filtered QBER time series (output of Layer 1).
    b_n : array_like
        Lanczos coefficients (output of Layer 2).
    t_axis : array_like
        Time axis corresponding to qber_residuum.
    window_size : int
        Size of the sliding window (default: 100).
    step : int
        Step size between windows (default: 25).

    Returns
    -------
    centers : numpy.ndarray
        Time centers of each window.
    scores : numpy.ndarray
        RMSE deviation scores. Higher = more deviation from template.

    Notes
    -----
    The theoretical template is a Gaussian: C(t) = exp(-0.5*(slope*t)^2)
    where slope = mean(diff(b_n)).

    Clean channel: scores cluster near a low baseline.
    Eve channel: scores are elevated during the attack window.

    Source: krylov_dynamic_detector.ipynb Cell 3 (krylov_dynamic_detector),
            krylov_robustness_test.ipynb Cell 3 (krylov_dynamic_detector)
    """
    from .lanczos_extractor import get_theoretical_autocorrelation

    qber_residuum = np.asarray(qber_residuum, dtype=float)
    theor_corr = get_theoretical_autocorrelation(b_n, t_axis[:window_size])

    scores = []
    centers = []

    for i in range(0, len(qber_residuum) - window_size, step):
        window = qber_residuum[i:i + window_size]
        w_std = window.std()

        if w_std < 1e-10:
            scores.append(0.0)
            centers.append(t_axis[i + window_size // 2])
            continue

        # Normalize window
        window = (window - window.mean()) / w_std

        # Compute empirical autocorrelation
        raw = correlate(window, window, mode='same')
        ac = raw[window_size // 2:]

        if abs(ac[0]) < 1e-10:
            scores.append(0.0)
            centers.append(t_axis[i + window_size // 2])
            continue

        ac = ac / ac[0]

        # Compare with template
        n = min(len(ac), len(theor_corr))
        rmse = np.sqrt(np.mean((ac[:n] - theor_corr[:n]) ** 2))

        scores.append(rmse)
        centers.append(t_axis[i + window_size // 2])

    return np.array(centers), np.array(scores)


def krylov_proxy(residuum, t_signal, window=200):
    """
    Sliding-window higher-moment analysis (kurtosis + skewness proxy).

    This is the simpler detection method from the Eve Detection Master
    notebook. It detects Eve through non-Gaussianity of the residuum.

    Score = sqrt(kurtosis^2 + skewness^2) * (1 + 100*variance)

    Parameters
    ----------
    residuum : array_like
        Sidereal-filtered QBER time series.
    t_signal : array_like
        Time axis.
    window : int
        Window size (default: 200).

    Returns
    -------
    t_windows : numpy.ndarray
        Time centers.
    scores : numpy.ndarray
        Detection scores.

    Source: eve_detection_master_v3.ipynb Cell 6 (krylov_proxy)
    """
    from scipy.stats import kurtosis, skew

    residuum = np.asarray(residuum, dtype=float)
    scores = []
    t_windows = []

    for i in range(0, len(residuum) - window, window // 4):
        seg = residuum[i:i + window]
        score = (np.sqrt(kurtosis(seg)**2 + skew(seg)**2)
                 * (1 + np.var(seg) * 100))
        scores.append(score)
        t_windows.append(t_signal[i + window // 2])

    return np.array(t_windows), np.array(scores)


def compute_roc(scores_clean, scores_eve, n_thresholds=200):
    """
    Compute ROC curve from clean and Eve detection scores.

    Parameters
    ----------
    scores_clean : array_like
        Detection scores from clean channels.
    scores_eve : array_like
        Detection scores from Eve-compromised channels.
    n_thresholds : int
        Number of threshold points for the ROC curve.

    Returns
    -------
    fpr : numpy.ndarray
        False positive rates.
    tpr : numpy.ndarray
        True positive rates.

    Source: krylov_dynamic_detector.ipynb Cell 4 (ROC computation),
            eve_detection_master_v3.ipynb Cell 8 (ROC curve)
    """
    scores_clean = np.asarray(scores_clean)
    scores_eve = np.asarray(scores_eve)

    all_scores = np.concatenate([scores_clean, scores_eve])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    fpr = []
    tpr = []

    for thr in thresholds:
        fp = np.sum(scores_clean > thr)
        tn = np.sum(scores_clean <= thr)
        tp = np.sum(scores_eve > thr)
        fn = np.sum(scores_eve <= thr)

        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    return np.array(fpr), np.array(tpr)


def compute_auc(fpr, tpr):
    """
    Compute Area Under the ROC Curve using the trapezoidal rule.

    Parameters
    ----------
    fpr : array_like
        False positive rates (from compute_roc).
    tpr : array_like
        True positive rates (from compute_roc).

    Returns
    -------
    float
        AUC value.
    """
    # Sort by FPR for proper integration
    idx = np.argsort(fpr)
    return float(np.abs(np.trapezoid(np.array(tpr)[idx], np.array(fpr)[idx])))


def compute_separation(scores_clean, scores_eve):
    """
    Compute sigma-separation between clean and Eve score distributions.

    separation = |mean_eve - mean_clean| / sqrt(0.5 * (var_clean + var_eve))

    Parameters
    ----------
    scores_clean : array_like
        Detection scores from clean channels.
    scores_eve : array_like
        Detection scores from Eve-compromised channels.

    Returns
    -------
    float
        Separation in units of sigma.

    Source: krylov_dynamic_detector.ipynb Cell 4 (separation calculation)
    """
    sc = np.asarray(scores_clean)
    se = np.asarray(scores_eve)

    pooled_std = np.sqrt(0.5 * (sc.var() + se.var()))
    if pooled_std < 1e-15:
        return 0.0

    return float(np.abs(se.mean() - sc.mean()) / pooled_std)
