"""
Tests for qkd_krylov_detector package.

These tests validate that the package reproduces the results
from Daniel Süß's notebooks and papers.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest


# ── Test 1: Hamiltonian construction ────────────────────────────────────────

def test_hamiltonian_dimensions():
    """Hamiltonian should be 256x256 for N=8 qubits."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    H = build_hamiltonian(N=8)
    assert H.shape == (256, 256), f"Expected (256,256), got {H.shape}"


def test_hamiltonian_hermitian():
    """Hamiltonian must be Hermitian."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    H = build_hamiltonian(N=8)
    assert H.isherm, "Hamiltonian is not Hermitian"


def test_hamiltonian_with_eve():
    """Eve perturbation should change the Hamiltonian."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian, build_hamiltonian_with_eve
    H_clean = build_hamiltonian()
    H_eve = build_hamiltonian_with_eve(gamma=0.3)
    diff = (H_eve - H_clean).norm()
    assert diff > 0.1, f"Eve perturbation too small: {diff}"


# ── Test 2: Lanczos coefficients ────────────────────────────────────────────

def test_lanczos_positive():
    """All b_n must be positive."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    from qkd_krylov_detector.lanczos_extractor import compute_lanczos
    H = build_hamiltonian()
    b_n = compute_lanczos(H, n_steps=20)
    assert len(b_n) > 0, "No Lanczos coefficients computed"
    assert np.all(b_n > 0), "Some b_n are non-positive"


def test_lanczos_linear_growth():
    """b_n should grow approximately linearly (characteristic of chaos)."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    from qkd_krylov_detector.lanczos_extractor import compute_lanczos, get_slope
    H = build_hamiltonian()
    b_n = compute_lanczos(H, n_steps=20)
    slope = get_slope(b_n)
    # From notebooks: avg_slope ≈ 3.975
    assert 2.0 < slope < 6.0, f"Slope {slope} outside expected range [2, 6]"


def test_lanczos_eve_deviation():
    """Eve should shift b_n from n=1 onwards."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian, build_hamiltonian_with_eve
    from qkd_krylov_detector.lanczos_extractor import compute_lanczos, compute_bn_deviation
    H_clean = build_hamiltonian()
    H_eve = build_hamiltonian_with_eve(gamma=0.3)
    b_clean = compute_lanczos(H_clean)
    b_eve = compute_lanczos(H_eve)
    dev = compute_bn_deviation(b_clean, b_eve)
    assert dev > 0.01, f"b_n deviation too small: {dev}"


# ── Test 3: Theoretical autocorrelation ─────────────────────────────────────

def test_theoretical_ac_gaussian():
    """Theoretical AC should be Gaussian-shaped."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    from qkd_krylov_detector.lanczos_extractor import compute_lanczos, get_theoretical_autocorrelation
    H = build_hamiltonian()
    b_n = compute_lanczos(H)
    t = np.linspace(0, 10, 100)
    ac = get_theoretical_autocorrelation(b_n, t)
    # Should start at 1.0 and decay
    assert abs(ac[0] - 1.0) < 1e-10, f"AC(0) = {ac[0]}, expected 1.0"
    assert ac[-1] < ac[0], "AC should decay"
    # Should be non-negative (Gaussian)
    assert np.all(ac >= -1e-10), "AC has unexpected negative values"


# ── Test 4: Sidereal filter ─────────────────────────────────────────────────

def test_sidereal_filter_removes_drift():
    """Filter should remove sidereal and diurnal components."""
    from qkd_krylov_detector.sidereal_filter import sidereal_filter, SIDEREAL_PERIOD
    # Use many periods so the FFT bin resolution resolves 1/23.93h well
    n_pts = 4000
    t = np.linspace(0, 4000, n_pts)
    # Pure sidereal signal
    signal = 0.1 * np.sin(2 * np.pi * t / SIDEREAL_PERIOD)
    filtered = sidereal_filter(signal, t)
    # Filtered signal should be much smaller
    assert np.std(filtered) < 0.25 * np.std(signal), \
        f"Filter did not remove sidereal: std ratio = {np.std(filtered)/np.std(signal)}"


def test_sidereal_filter_preserves_other():
    """Filter should preserve non-sidereal signals."""
    from qkd_krylov_detector.sidereal_filter import sidereal_filter
    t = np.linspace(0, 400, 400)
    # Signal at a different period (10h)
    signal = 0.1 * np.sin(2 * np.pi * t / 10.0)
    filtered = sidereal_filter(signal, t)
    # Should be mostly preserved
    assert np.std(filtered) > 0.5 * np.std(signal), \
        "Filter removed non-sidereal signal"


# ── Test 5: QBER simulator ──────────────────────────────────────────────────

def test_clean_qber_shape():
    """Clean QBER should have correct length."""
    from qkd_krylov_detector.qber_simulator import make_clean_qber
    t = np.linspace(0, 400, 400)
    qber = make_clean_qber(t, seed=42)
    assert len(qber) == 400


def test_eve_qber_different():
    """Eve QBER should differ from clean in the attack window."""
    from qkd_krylov_detector.qber_simulator import make_clean_qber, make_eve_qber
    t = np.linspace(0, 400, 400)
    clean = make_clean_qber(t, seed=42)
    eve = make_eve_qber(t, eve_type="iid", seed=42)
    # Should differ in the attack window
    diff = np.abs(eve[150:280] - clean[150:280]).mean()
    assert diff > 0.01, f"Eve signal too weak: mean diff = {diff}"


def test_realistic_qber_different_ac():
    """Realistic noise should have different AC from Gaussian template."""
    from qkd_krylov_detector.qber_simulator import make_realistic_clean_qber
    from qkd_krylov_detector.sidereal_filter import sidereal_filter
    t = np.linspace(0, 400, 400)
    qber = make_realistic_clean_qber(t, seed=42)
    filtered = sidereal_filter(qber, t)
    # Should have non-trivial autocorrelation structure
    assert np.std(filtered) > 1e-5, "Realistic noise too small after filtering"


# ── Test 6: Template detector ───────────────────────────────────────────────

def test_detector_scores_shape():
    """Detector should return scores for each window."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    from qkd_krylov_detector.lanczos_extractor import compute_lanczos
    from qkd_krylov_detector.template_detector import krylov_dynamic_detector
    from qkd_krylov_detector.qber_simulator import make_clean_qber
    from qkd_krylov_detector.sidereal_filter import sidereal_filter

    H = build_hamiltonian()
    b_n = compute_lanczos(H)
    t = np.linspace(0, 400, 400)
    qber = make_clean_qber(t, seed=42)
    residuum = sidereal_filter(qber, t)
    centers, scores = krylov_dynamic_detector(residuum, b_n, t)
    assert len(centers) > 0, "No detection windows"
    assert len(centers) == len(scores), "Mismatched centers/scores"


def test_detector_eve_higher_scores():
    """Eve channel should produce higher scores than clean channel."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    from qkd_krylov_detector.lanczos_extractor import compute_lanczos
    from qkd_krylov_detector.template_detector import krylov_dynamic_detector
    from qkd_krylov_detector.qber_simulator import make_clean_qber, make_eve_qber
    from qkd_krylov_detector.sidereal_filter import sidereal_filter

    H = build_hamiltonian()
    b_n = compute_lanczos(H)
    t = np.linspace(0, 400, 400)

    clean = make_clean_qber(t, seed=42)
    eve = make_eve_qber(t, eve_type="iid", gamma=0.3, seed=43)

    res_clean = sidereal_filter(clean, t)
    res_eve = sidereal_filter(eve, t)

    _, scores_clean = krylov_dynamic_detector(res_clean, b_n, t)
    _, scores_eve = krylov_dynamic_detector(res_eve, b_n, t)

    assert scores_eve.mean() > scores_clean.mean(), \
        f"Eve scores ({scores_eve.mean():.4f}) not higher than clean ({scores_clean.mean():.4f})"


# ── Test 7: ROC and AUC ─────────────────────────────────────────────────────

def test_roc_auc_positive():
    """AUC should be > 0.5 (better than random)."""
    from qkd_krylov_detector.template_detector import compute_roc, compute_auc
    # Simulated scores: clean low, eve high
    scores_clean = np.random.normal(0.2, 0.05, 50)
    scores_eve = np.random.normal(0.5, 0.05, 50)
    fpr, tpr = compute_roc(scores_clean, scores_eve)
    auc = compute_auc(fpr, tpr)
    assert auc > 0.5, f"AUC = {auc}, expected > 0.5"


# ── Test 8: Spectral analysis ───────────────────────────────────────────────

def test_r_ratio_crossover():
    """⟨r⟩ for default Hamiltonian should be in crossover regime."""
    from qkd_krylov_detector.hamiltonian import build_hamiltonian
    from qkd_krylov_detector.spectral_analysis import compute_r_ratio, classify_regime
    H = build_hamiltonian()
    r = compute_r_ratio(H)
    regime = classify_regime(r)
    # From notebooks: ⟨r⟩ ≈ 0.366
    assert 0.30 < r < 0.45, f"⟨r⟩ = {r}, expected ~0.366"
    assert "crossover" in regime.lower(), f"Expected crossover, got: {regime}"


# ── Test 9: Separation metric ───────────────────────────────────────────────

def test_separation_positive():
    """Separation between clean and Eve should be positive."""
    from qkd_krylov_detector.template_detector import compute_separation
    scores_clean = np.random.normal(0.2, 0.05, 100)
    scores_eve = np.random.normal(0.5, 0.08, 100)
    sep = compute_separation(scores_clean, scores_eve)
    assert sep > 0, f"Separation = {sep}, expected > 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
