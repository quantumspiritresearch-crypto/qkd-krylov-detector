"""
Tests for v1.2.0 modules: quantum_eve, krylov_bridge, demo_framework.
"""

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════
# quantum_eve module tests
# ══════════════════════════════════════════════════════════════════

class TestQuantumEve:
    """Tests for quantum_eve module (no QuTiP required for non-quantum functions)."""

    def test_gaussian_template_shape(self):
        from qkd_krylov_detector.quantum_eve import gaussian_template
        b_n = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        t = np.linspace(0, 3, 50)
        tmpl = gaussian_template(b_n, t)
        assert tmpl.shape == (50,)
        assert tmpl[0] == pytest.approx(1.0, abs=1e-10)
        assert tmpl[-1] < tmpl[0]  # Decays

    def test_gaussian_template_monotonic_decay(self):
        from qkd_krylov_detector.quantum_eve import gaussian_template
        b_n = np.array([1.0, 1.5, 2.0, 2.5])
        t = np.linspace(0, 2, 30)
        tmpl = gaussian_template(b_n, t)
        # Should be monotonically decreasing for positive slope
        for i in range(1, len(tmpl)):
            assert tmpl[i] <= tmpl[i-1] + 1e-10

    def test_make_classical_eve_qber(self):
        from qkd_krylov_detector.quantum_eve import make_classical_eve_qber
        t = np.linspace(0, 400, 400)
        qber = make_classical_eve_qber(t, p_ir=0.3, eve_start=150, eve_end=280, seed=42)
        assert qber.shape == (400,)
        # Eve window should have higher mean than clean windows
        clean_mean = np.mean(np.abs(qber[:100]))
        eve_mean = np.mean(np.abs(qber[150:280]))
        assert eve_mean > clean_mean

    def test_make_classical_eve_reproducible(self):
        from qkd_krylov_detector.quantum_eve import make_classical_eve_qber
        t = np.linspace(0, 400, 400)
        q1 = make_classical_eve_qber(t, seed=123)
        q2 = make_classical_eve_qber(t, seed=123)
        np.testing.assert_array_equal(q1, q2)

    def test_make_quantum_eve_qber(self):
        from qkd_krylov_detector.quantum_eve import make_quantum_eve_qber
        t = np.linspace(0, 400, 400)
        autocorr_clean = np.exp(-0.5 * np.linspace(0, 3, 60)**2)
        autocorr_eve = autocorr_clean * 0.9 + 0.05  # Slightly perturbed
        qber = make_quantum_eve_qber(t, autocorr_eve, autocorr_clean,
                                      coupling_strength=0.5, seed=42)
        assert qber.shape == (400,)
        assert np.isfinite(qber).all()

    def test_compute_anomaly_scores(self):
        from qkd_krylov_detector.quantum_eve import compute_anomaly_scores
        clean = np.exp(-0.5 * np.linspace(0, 3, 50)**2)
        matrix = np.tile(clean, (10, 1))
        matrix[5:] += 0.1  # Perturb half
        scores = compute_anomaly_scores(matrix, clean)
        assert scores.shape == (10,)
        assert np.mean(scores[:5]) < np.mean(scores[5:])

    def test_compute_eve_detection_stats(self):
        from qkd_krylov_detector.quantum_eve import compute_eve_detection_stats
        scores_clean = np.random.normal(0.1, 0.02, 50)
        scores_eve = np.random.normal(0.5, 0.05, 50)
        stats = compute_eve_detection_stats(scores_eve, scores_clean)
        assert "detected_pct" in stats
        assert "forged_pct" in stats
        assert "auc" in stats
        assert "sep" in stats
        assert stats["auc"] > 0.8  # Should be well separated
        assert stats["detected_pct"] + stats["forged_pct"] == pytest.approx(100.0)

    def test_strategies_constant(self):
        from qkd_krylov_detector.quantum_eve import STRATEGIES
        assert len(STRATEGIES) == 4
        assert 'passive' in STRATEGIES
        assert 'optimal' in STRATEGIES


# ══════════════════════════════════════════════════════════════════
# krylov_bridge module tests
# ══════════════════════════════════════════════════════════════════

class TestKrylovBridge:
    """Tests for krylov_bridge module."""

    def test_sim_stats_clean(self):
        from qkd_krylov_detector.krylov_bridge import sim_stats
        t = np.linspace(0, 400, 400)
        k, s = sim_stats(0.0, t, n_trials=5, seed=42)
        assert isinstance(k, float)
        assert isinstance(s, float)

    def test_sim_stats_eve_higher(self):
        from qkd_krylov_detector.krylov_bridge import sim_stats
        t = np.linspace(0, 400, 400)
        k_clean, s_clean = sim_stats(0.0, t, n_trials=10, seed=42)
        k_eve, s_eve = sim_stats(0.5, t, n_trials=10, seed=42)
        # Eve should produce higher kurtosis or skewness
        assert k_eve > k_clean or s_eve > s_clean

    def test_krylov_proxy_shape(self):
        from qkd_krylov_detector.krylov_bridge import krylov_proxy
        t = np.linspace(0, 400, 800)
        signal = np.random.normal(0, 0.01, 800)
        tw, scores = krylov_proxy(signal, t, window=200)
        assert len(tw) == len(scores)
        assert len(scores) > 0

    def test_krylov_proxy_eve_higher(self):
        from qkd_krylov_detector.krylov_bridge import krylov_proxy
        t = np.linspace(0, 400, 800)
        np.random.seed(42)
        clean = np.random.normal(0, 0.01, 800)
        attack = clean.copy()
        attack[300:500] += 0.3 * (np.random.exponential(0.1, 200) - 0.1)
        _, sc = krylov_proxy(clean, t, window=100)
        _, sa = krylov_proxy(attack, t, window=100)
        assert sa.max() > sc.max()

    def test_sensitivity_vs_gamma(self):
        from qkd_krylov_detector.krylov_bridge import sensitivity_vs_gamma
        t = np.linspace(0, 400, 400)
        b_n = np.array([1.0, 1.5, 2.0, 2.5])
        gammas = np.array([0.0, 0.3, 0.6])
        result = sensitivity_vs_gamma(gammas, t, b_n, n_trials=3,
                                       window=100, seed=42)
        assert "gamma" in result
        assert "sensitivity" in result
        assert len(result["sensitivity"]) == 3
        # Higher gamma should give higher sensitivity
        assert result["sensitivity"][-1] >= result["sensitivity"][0]


# ══════════════════════════════════════════════════════════════════
# demo_framework module tests
# ══════════════════════════════════════════════════════════════════

class TestDemoFramework:
    """Tests for demo_framework module."""

    def test_sidereal_filter_class(self):
        from qkd_krylov_detector.demo_framework import SiderealFilter
        sf = SiderealFilter()
        assert sf.periods == [23.93, 24.0]
        assert sf.bandwidth == 0.008
        t = np.linspace(0, 400, 400)
        signal = 0.04 * np.sin(2 * np.pi * t / 23.93) + np.random.normal(0, 0.01, 400)
        filtered = sf.filter(signal, t)
        assert filtered.shape == (400,)
        assert np.std(filtered) < np.std(signal)

    def test_sidereal_filter_custom_periods(self):
        from qkd_krylov_detector.demo_framework import SiderealFilter
        sf = SiderealFilter(periods=[12.0], bandwidth=0.01)
        assert sf.periods == [12.0]
        assert "SiderealFilter" in repr(sf)

    def test_krylov_engine_init(self):
        from qkd_krylov_detector.demo_framework import KrylovEngine
        b_n = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        engine = KrylovEngine(b_n, window_size=50, step=10)
        assert engine.slope == pytest.approx(0.5, abs=1e-10)
        assert "KrylovEngine" in repr(engine)

    def test_krylov_engine_template(self):
        from qkd_krylov_detector.demo_framework import KrylovEngine
        b_n = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        engine = KrylovEngine(b_n, window_size=50)
        tmpl = engine.template
        assert tmpl.shape == (50,)
        assert tmpl[0] == pytest.approx(1.0, abs=1e-10)

    def test_krylov_engine_detect(self):
        from qkd_krylov_detector.demo_framework import KrylovEngine
        b_n = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        engine = KrylovEngine(b_n, window_size=50, step=10)
        t = np.linspace(0, 400, 400)
        signal = np.random.normal(0, 0.01, 400)
        tc, scores = engine.detect(signal, t)
        assert len(tc) == len(scores)
        assert len(scores) > 0

    def test_krylov_engine_proxy(self):
        from qkd_krylov_detector.demo_framework import KrylovEngine
        b_n = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        engine = KrylovEngine(b_n)
        t = np.linspace(0, 400, 800)
        signal = np.random.normal(0, 0.01, 800)
        tw, scores = engine.proxy(signal, t, window=100)
        assert len(tw) == len(scores)

    def test_classify_window_clean(self):
        from qkd_krylov_detector.demo_framework import classify_window
        np.random.seed(42)
        data = np.random.normal(0, 0.01, 200)
        result = classify_window(data)
        assert "kurtosis" in result
        assert "skewness" in result
        assert "classification" in result
        assert result["classification"] == "clean"

    def test_classify_window_anomalous(self):
        from qkd_krylov_detector.demo_framework import classify_window
        np.random.seed(42)
        data = np.random.exponential(0.1, 200)  # Skewed distribution
        result = classify_window(data, threshold_skew=0.3)
        assert result["is_anomalous"] is True

    def test_make_scenario_clean(self):
        from qkd_krylov_detector.demo_framework import make_scenario
        t = np.linspace(0, 400, 400)
        sc = make_scenario(t, attack_type="clean", seed=42)
        assert sc["attack_type"] == "clean"
        assert sc["qber"].shape == (400,)

    def test_make_scenario_iid(self):
        from qkd_krylov_detector.demo_framework import make_scenario
        t = np.linspace(0, 400, 400)
        sc = make_scenario(t, attack_type="iid", gamma=0.3, seed=42)
        sc_clean = make_scenario(t, attack_type="clean", seed=42)
        # iid attack should have higher values in attack window
        mask = (t >= sc["eve_start"]) & (t <= sc["eve_end"])
        assert np.mean(sc["qber"][mask]) > np.mean(sc_clean["qber"][mask])

    def test_make_scenario_all_types(self):
        from qkd_krylov_detector.demo_framework import make_scenario
        t = np.linspace(0, 400, 400)
        for atype in ["clean", "iid", "exponential", "burst", "gradual"]:
            sc = make_scenario(t, attack_type=atype, seed=42)
            assert sc["attack_type"] == atype
            assert np.isfinite(sc["qber"]).all()

    def test_make_scenario_invalid_type(self):
        from qkd_krylov_detector.demo_framework import make_scenario
        t = np.linspace(0, 400, 400)
        with pytest.raises(ValueError, match="Unknown attack_type"):
            make_scenario(t, attack_type="invalid")
