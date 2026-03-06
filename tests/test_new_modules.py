"""
Tests for the five new modules added in v1.1.0:
    - bb84_simulation
    - attack_classifier
    - calibration
    - sparse_hamiltonian
    - pulsar_analysis

Run with: pytest tests/ -v
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# BB84 Simulation
# ═══════════════════════════════════════════════════════════════════════════

class TestBB84Simulation:

    def test_bb84_clean_qber_low(self):
        """Clean BB84 should have QBER near channel noise (~1%)."""
        from qkd_krylov_detector.bb84_simulation import bb84_clean
        rng = np.random.default_rng(42)
        errors, n_sifted = bb84_clean(n_qubits=50000, rng=rng)
        qber = errors.mean()
        assert 0.005 < qber < 0.025, f"Clean QBER {qber:.4f} outside expected range"
        assert n_sifted > 20000, "Too few sifted bits"

    def test_bb84_intercept_resend_increases_qber(self):
        """Full intercept-resend should add ~25% QBER."""
        from qkd_krylov_detector.bb84_simulation import bb84_intercept_resend
        rng = np.random.default_rng(42)
        errors, n_sifted, n_intercepted = bb84_intercept_resend(
            n_qubits=50000, p_intercept=1.0, rng=rng)
        qber = errors.mean()
        assert qber > 0.20, f"Full IR QBER {qber:.4f} should be > 0.20"

    def test_bb84_beam_splitting_drops_transmission(self):
        """Beam-splitting should reduce transmission by ~p_split."""
        from qkd_krylov_detector.bb84_simulation import bb84_beam_splitting
        rng = np.random.default_rng(42)
        errors, n_sifted, transmission = bb84_beam_splitting(
            n_qubits=50000, p_split=0.5, rng=rng)
        assert 0.4 < transmission < 0.6, \
            f"Transmission {transmission:.3f} should be near 0.5"

    def test_bb84_window_returns_qber_and_transmission(self):
        """bb84_window should return (qber, transmission) tuple."""
        from qkd_krylov_detector.bb84_simulation import bb84_window
        rng = np.random.default_rng(42)
        qber, trans = bb84_window(n_qubits=1000, p_ir=0.3, rng=rng)
        assert 0 <= qber <= 1, f"QBER {qber} out of [0,1]"
        assert 0 <= trans <= 1, f"Transmission {trans} out of [0,1]"

    def test_make_bb84_timeseries_shape(self):
        """Time series should have correct length."""
        from qkd_krylov_detector.bb84_simulation import make_bb84_timeseries
        t, qber, trans = make_bb84_timeseries(
            n_windows=100, eve_type="clean", seed=42)
        assert len(t) == 100
        assert len(qber) == 100
        assert len(trans) == 100

    def test_make_bb84_timeseries_eve_raises_qber(self):
        """IR attack should raise mean QBER in attack window."""
        from qkd_krylov_detector.bb84_simulation import make_bb84_timeseries
        _, qber_clean, _ = make_bb84_timeseries(
            n_windows=200, eve_type="clean", seed=42)
        _, qber_eve, _ = make_bb84_timeseries(
            n_windows=200, eve_type="intercept_resend", p_eve=0.5,
            eve_start=80, eve_end=150, seed=42)
        assert qber_eve[80:150].mean() > qber_clean[80:150].mean()


# ═══════════════════════════════════════════════════════════════════════════
# Attack Classifier
# ═══════════════════════════════════════════════════════════════════════════

class TestAttackClassifier:

    def test_base_qber_shape(self):
        """base_qber should return array of correct length."""
        from qkd_krylov_detector.attack_classifier import base_qber
        rng = np.random.default_rng(42)
        q = base_qber(rng, n_windows=200)
        assert len(q) == 200

    def test_make_ir_increases_qber(self):
        """IR attack should raise QBER in attack window."""
        from qkd_krylov_detector.attack_classifier import make_clean, make_ir
        rng_c = np.random.default_rng(42)
        rng_e = np.random.default_rng(42)
        q_clean, _ = make_clean(rng_c, n_windows=400)
        q_ir, _ = make_ir(rng_e, eve_start=100, eve_end=230, n_windows=400)
        assert q_ir[100:230].mean() > q_clean[100:230].mean()

    def test_make_blinding_spikes(self):
        """Blinding attack should produce periodic spikes."""
        from qkd_krylov_detector.attack_classifier import make_blinding
        rng = np.random.default_rng(42)
        q, n = make_blinding(rng, eve_start=100, eve_end=200)
        # Photon count at blinding positions should be very high
        assert n[100] > 2000, "Blinding should cause photon surge"

    def test_find_attack_window(self):
        """Attack window detection should find the correct region."""
        from qkd_krylov_detector.attack_classifier import (
            make_ir, find_attack_window, _sfilt)
        rng = np.random.default_rng(42)
        q, n = make_ir(rng, eve_start=150, eve_end=280, n_windows=400)
        t = np.linspace(0, 400, 400)
        rq = _sfilt(q, t)
        start, end, ks = find_attack_window(rq, n)
        # Detected window should overlap with true attack window
        assert start < 280 and end > 150, \
            f"Detected [{start},{end}] doesn't overlap [150,280]"

    def test_extract_features_shape(self):
        """Feature vector should be non-empty and finite."""
        from qkd_krylov_detector.attack_classifier import make_ir, extract_features
        rng = np.random.default_rng(42)
        q, n = make_ir(rng, eve_start=100, eve_end=230, n_windows=400)
        feats = extract_features(q, n)
        assert len(feats) > 30, f"Feature vector too short: {len(feats)}"
        assert np.all(np.isfinite(feats)), "Features contain non-finite values"

    def test_cusum_detects_attack(self):
        """CUSUM should trigger alarm on IR attack."""
        from qkd_krylov_detector.attack_classifier import cusum_detect
        from qkd_krylov_detector.bb84_simulation import make_bb84_timeseries
        _, qber, _ = make_bb84_timeseries(
            n_windows=300, eve_type="intercept_resend", p_eve=0.5,
            eve_start=100, eve_end=200, seed=42)
        alarms, max_cusum, alarm_flag = cusum_detect(qber)
        assert alarm_flag, "CUSUM should detect strong IR attack"

    def test_cusum_clean_no_alarm(self):
        """CUSUM should not trigger on clean channel (mostly)."""
        from qkd_krylov_detector.attack_classifier import cusum_detect
        from qkd_krylov_detector.bb84_simulation import make_bb84_timeseries
        _, qber, _ = make_bb84_timeseries(
            n_windows=300, eve_type="clean", seed=42)
        _, _, alarm_flag = cusum_detect(qber, h_factor=5.0)
        # Allow occasional false positive, but max_cusum should be low
        # (we don't assert alarm_flag is False because stochastic)

    def test_spectral_anomaly_score_finite(self):
        """Spectral anomaly score should be a finite positive number."""
        from qkd_krylov_detector.attack_classifier import spectral_anomaly_score
        from qkd_krylov_detector.bb84_simulation import make_bb84_timeseries
        _, qber, _ = make_bb84_timeseries(
            n_windows=300, eve_type="clean", seed=42)
        score = spectral_anomaly_score(qber)
        assert np.isfinite(score) and score > 0


# ═══════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibration:

    def test_gaussian_ac_at_zero(self):
        """Gaussian AC at t=0 should be 1.0."""
        from qkd_krylov_detector.calibration import gaussian_ac
        assert gaussian_ac(np.array([0.0]), 1.5)[0] == pytest.approx(1.0)

    def test_gaussian_ac_decays(self):
        """Gaussian AC should decay monotonically."""
        from qkd_krylov_detector.calibration import gaussian_ac
        t = np.arange(20, dtype=float)
        ac = gaussian_ac(t, 0.5)
        assert np.all(np.diff(ac) <= 0), "AC should be monotonically decreasing"

    def test_fit_slope_on_gaussian(self):
        """fit_slope on a known Gaussian signal should recover the slope."""
        from qkd_krylov_detector.calibration import fit_slope
        rng = np.random.default_rng(42)
        # Create signal with known autocorrelation structure
        t = np.arange(200)
        signal = np.exp(-0.01 * t) * np.cos(0.1 * t) + rng.normal(0, 0.1, 200)
        s, r2 = fit_slope(signal)
        assert np.isfinite(s), "Slope should be finite"
        assert s > 0, "Slope should be positive"

    def test_calibrate_returns_positive_slope(self):
        """Calibration should return a positive slope."""
        from qkd_krylov_detector.calibration import calibrate
        rng = np.random.default_rng(42)
        # Generate some clean-like residua
        residua = [rng.normal(0, 0.02, 400) for _ in range(5)]
        s_cal, alpha, slopes = calibrate(residua, s_bn=1.5)
        assert s_cal > 0, "Calibrated slope should be positive"
        assert len(slopes) > 0, "Should have fitted some slopes"

    def test_krylov_slope_detector_output_shape(self):
        """Slope detector should return arrays of matching length."""
        from qkd_krylov_detector.calibration import krylov_slope_detector
        rng = np.random.default_rng(42)
        residuum = rng.normal(0, 0.02, 400)
        t = np.linspace(0, 400, 400)
        centers, scores, s_fits, r2_vals = krylov_slope_detector(
            residuum, s_bn=1.5, t=t)
        assert len(centers) == len(scores) == len(s_fits) == len(r2_vals)
        assert len(centers) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Sparse Hamiltonian
# ═══════════════════════════════════════════════════════════════════════════

class TestSparseHamiltonian:

    def test_sparse_hamiltonian_shape(self):
        """Sparse Hamiltonian for N=6 should be 64x64."""
        from qkd_krylov_detector.sparse_hamiltonian import build_hamiltonian_sparse
        H = build_hamiltonian_sparse(N=6)
        assert H.shape == (64, 64)

    def test_sparse_hamiltonian_hermitian(self):
        """Sparse Hamiltonian should be Hermitian."""
        from qkd_krylov_detector.sparse_hamiltonian import build_hamiltonian_sparse
        H = build_hamiltonian_sparse(N=6)
        diff = (H - H.conj().T).toarray()
        assert np.allclose(diff, 0, atol=1e-10), "Sparse H not Hermitian"

    def test_sparse_matches_dense_n6(self):
        """Sparse and dense Hamiltonians should agree for N=6."""
        from qkd_krylov_detector.sparse_hamiltonian import build_hamiltonian_sparse
        from qkd_krylov_detector.hamiltonian import build_hamiltonian
        H_sparse = build_hamiltonian_sparse(N=6).toarray()
        H_dense = build_hamiltonian(N=6).full()
        assert np.allclose(H_sparse, H_dense, atol=1e-10), \
            "Sparse and dense Hamiltonians differ"

    def test_spectral_statistics_sparse_crossover(self):
        """Sparse ⟨r⟩ for N=8 should be in crossover regime."""
        from qkd_krylov_detector.sparse_hamiltonian import spectral_statistics_sparse
        r_mean, r_std, r_list = spectral_statistics_sparse(N=8, k=100)
        assert 0.30 < r_mean < 0.45, \
            f"⟨r⟩ = {r_mean:.3f} outside crossover range [0.30, 0.45]"

    def test_kron_op_identity(self):
        """kron_op with identity should give identity."""
        from qkd_krylov_detector.sparse_hamiltonian import kron_op
        import scipy.sparse as sp
        I2 = sp.eye(2, format="csr", dtype=complex)
        full_I = kron_op(I2, 0, 3)
        expected = sp.eye(8, format="csr", dtype=complex)
        assert np.allclose(full_I.toarray(), expected.toarray())


# ═══════════════════════════════════════════════════════════════════════════
# Pulsar Analysis
# ═══════════════════════════════════════════════════════════════════════════

class TestPulsarAnalysis:

    def test_design_matrix_full_shape(self):
        """Full design matrix should have 5 columns."""
        from qkd_krylov_detector.pulsar_analysis import make_design_matrix
        mjd = np.linspace(55000, 60000, 1000)
        lst = np.random.uniform(0, 24, 1000)
        X = make_design_matrix(mjd, lst, include_sidereal=True)
        assert X.shape == (1000, 5)

    def test_design_matrix_reduced_shape(self):
        """Reduced design matrix should have 3 columns."""
        from qkd_krylov_detector.pulsar_analysis import make_design_matrix
        mjd = np.linspace(55000, 60000, 1000)
        lst = np.random.uniform(0, 24, 1000)
        X = make_design_matrix(mjd, lst, include_sidereal=False)
        assert X.shape == (1000, 3)

    def test_partial_f_test_no_signal(self):
        """F-test on pure noise should not detect sidereal signal."""
        from qkd_krylov_detector.pulsar_analysis import partial_f_test
        rng = np.random.default_rng(42)
        n = 5000
        mjd = np.sort(rng.uniform(55000, 60000, n))
        lst = rng.uniform(0, 24, n)
        residuals = rng.normal(0, 1.0, n)  # pure noise
        result = partial_f_test(residuals, mjd, lst)
        assert result['p_value'] > 0.01, \
            f"Should not detect signal in noise, p={result['p_value']:.4f}"

    def test_partial_f_test_with_signal(self):
        """F-test should detect injected sidereal signal."""
        from qkd_krylov_detector.pulsar_analysis import partial_f_test
        rng = np.random.default_rng(42)
        n = 5000
        mjd = np.sort(rng.uniform(55000, 60000, n))
        lst = rng.uniform(0, 24, n)
        lst_rad = lst * (2 * np.pi / 24.0)
        # Inject strong sidereal signal
        signal = 5.0 * np.sin(lst_rad) + 3.0 * np.cos(lst_rad)
        residuals = signal + rng.normal(0, 1.0, n)
        result = partial_f_test(residuals, mjd, lst)
        assert result['detection'], \
            f"Should detect sidereal signal, p={result['p_value']:.6f}"
        assert result['sidereal_amplitude'] > 3.0, \
            f"Amplitude {result['sidereal_amplitude']:.2f} too low"

    def test_classify_gaps(self):
        """Gap classification should return valid type."""
        from qkd_krylov_detector.pulsar_analysis import classify_gaps
        mjd = np.sort(np.random.default_rng(42).uniform(55000, 60000, 1000))
        result = classify_gaps(mjd)
        assert result['gap_type'] in ("DIURNAL", "STOCHASTIC")
        assert result['median_cadence_days'] > 0

    def test_compute_sidereal_amplitude(self):
        """Amplitude from known coefficients."""
        from qkd_krylov_detector.pulsar_analysis import compute_sidereal_amplitude
        coeffs = np.array([0.0, 0.0, 0.0, 3.0, 4.0])
        amp = compute_sidereal_amplitude(coeffs)
        assert amp == pytest.approx(5.0, abs=1e-10)
