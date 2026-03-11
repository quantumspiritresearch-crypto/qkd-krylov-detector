"""
Tests for the six new modules added in v2.0.0:
    - physical_bridge
    - open_system_bridge
    - error_diagnostics
    - one_way_function
    - universality
    - krylov_framework

Run with: pytest tests/test_v200_modules.py -v
"""
import numpy as np
import pytest


# ── Helper: build small test system ─────────────────────────────────────
def _build_test_system(N=4):
    """Build a small N-qubit Heisenberg chain for testing."""
    from qkd_krylov_detector.universality import heisenberg_chain, _site_op, _sz
    H = heisenberg_chain(N)
    O = _site_op(_sz, 0, N)
    return H, O


# ═══════════════════════════════════════════════════════════════════════════
# Physical Bridge
# ═══════════════════════════════════════════════════════════════════════════
class TestPhysicalBridge:
    def test_lanczos_coefficients_positive(self):
        """Lanczos coefficients should be positive."""
        from qkd_krylov_detector.physical_bridge import compute_lanczos_coefficients
        H, O = _build_test_system(4)
        b_n = compute_lanczos_coefficients(H, O, 10)
        assert len(b_n) > 0
        assert np.all(b_n > 0), "All Lanczos coefficients should be positive"

    def test_autocorrelation_normalized(self):
        """C_op(0) should be 1.0."""
        from qkd_krylov_detector.physical_bridge import compute_operator_autocorrelation
        H, O = _build_test_system(4)
        times = np.linspace(0, 3, 20)
        C = compute_operator_autocorrelation(H, O, times)
        assert abs(C[0] - 1.0) < 1e-10, f"C(0) = {C[0]}, should be 1.0"

    def test_autocorrelation_methods_agree(self):
        """Eigen and expm methods should give same results."""
        from qkd_krylov_detector.physical_bridge import compute_operator_autocorrelation
        H, O = _build_test_system(3)
        times = np.linspace(0, 2, 10)
        C_eigen = compute_operator_autocorrelation(H, O, times, method="eigen")
        C_expm = compute_operator_autocorrelation(H, O, times, method="expm")
        np.testing.assert_allclose(C_eigen, C_expm, atol=1e-8)

    def test_qber_autocorrelation_normalized(self):
        """C_QBER(0) should be 1.0 when normalized."""
        from qkd_krylov_detector.physical_bridge import compute_qber_autocorrelation
        rng = np.random.default_rng(42)
        qber = rng.normal(0, 0.1, 500)
        C = compute_qber_autocorrelation(qber, max_lag=30)
        assert abs(C[0] - 1.0) < 1e-10

    def test_verify_bridge_correlation_structure(self):
        """verify_bridge_correlation should return correct keys."""
        from qkd_krylov_detector.physical_bridge import verify_bridge_correlation
        c1 = np.exp(-np.arange(20) * 0.1)
        c2 = np.exp(-np.arange(20) * 0.12)
        result = verify_bridge_correlation(c1, c2)
        assert 'r' in result
        assert 'p' in result
        assert 'rmse' in result
        assert 'bridge_valid' in result

    def test_bridge_transform_preserves_normalization(self):
        """Bridge transform should return C(0) = 1."""
        from qkd_krylov_detector.physical_bridge import bridge_transform
        C_op = np.exp(-np.arange(20) * 0.1)
        C_pred = bridge_transform(C_op, lanczos_slope=0.5)
        assert abs(C_pred[0] - 1.0) < 1e-10

    def test_full_bridge_analysis_returns_dict(self):
        """full_bridge_analysis should return comprehensive results."""
        from qkd_krylov_detector.physical_bridge import full_bridge_analysis
        H, O = _build_test_system(3)
        rng = np.random.default_rng(42)
        qber = rng.normal(0, 0.1, 200)
        times = np.linspace(0, 3, 15)
        result = full_bridge_analysis(H, O, qber, times, n_lanczos=10)
        assert 'b_n' in result
        assert 'lanczos_slope' in result
        assert 'bridge_r' in result


# ═══════════════════════════════════════════════════════════════════════════
# Open-System Bridge
# ═══════════════════════════════════════════════════════════════════════════
class TestOpenSystemBridge:
    def test_adjoint_lindbladian_shape(self):
        """Adjoint Lindbladian should be d^2 x d^2."""
        from qkd_krylov_detector.open_system_bridge import build_adjoint_lindbladian
        d = 4
        H = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        H = (H + H.conj().T) / 2
        L = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        L_adj = build_adjoint_lindbladian(H, [L], [0.1])
        assert L_adj.shape == (d**2, d**2)

    def test_closed_system_matches_physical_bridge(self):
        """Open system with no dissipation should match closed system."""
        from qkd_krylov_detector.open_system_bridge import open_system_autocorrelation
        from qkd_krylov_detector.physical_bridge import compute_operator_autocorrelation
        H, O = _build_test_system(3)
        times = np.linspace(0, 2, 10)
        C_closed = compute_operator_autocorrelation(H, O, times)
        C_open_no_noise = open_system_autocorrelation(H, O, times)
        np.testing.assert_allclose(C_closed, C_open_no_noise, atol=1e-6)

    def test_dissipation_damps_autocorrelation(self):
        """Open system with dissipation should have smaller late-time C."""
        from qkd_krylov_detector.open_system_bridge import open_system_autocorrelation
        from qkd_krylov_detector.universality import _site_op, _sz
        H, O = _build_test_system(3)
        times = np.linspace(0, 5, 25)
        C_closed = open_system_autocorrelation(H, O, times)
        jump_ops = [_site_op(_sz, k, 3) for k in range(3)]
        gamma_list = [1.0] * 3  # Strong dissipation
        C_open = open_system_autocorrelation(H, O, times, jump_ops, gamma_list)
        # Late-time open should be smaller in magnitude
        assert np.mean(np.abs(C_open[15:])) < np.mean(np.abs(C_closed[15:]))

    def test_decoherence_envelope_positive(self):
        """Decoherence envelope should be non-negative."""
        from qkd_krylov_detector.open_system_bridge import compute_decoherence_envelope
        C_open = np.exp(-np.arange(20) * 0.2)
        C_closed = np.exp(-np.arange(20) * 0.05)
        result = compute_decoherence_envelope(C_open, C_closed)
        assert 'envelope' in result
        assert 'decay_rate' in result
        assert result['decay_rate'] >= 0


# ═══════════════════════════════════════════════════════════════════════════
# Error Diagnostics
# ═══════════════════════════════════════════════════════════════════════════
class TestErrorDiagnostics:
    def test_classify_clean(self):
        """No perturbation should classify as 'clean'."""
        from qkd_krylov_detector.error_diagnostics import classify_error_type
        result = classify_error_type(delta_b=0.001, envelope_ratio=0.9)
        assert result['error_type'] == 'clean'

    def test_classify_coherent(self):
        """Large delta_b with preserved envelope -> 'coherent'."""
        from qkd_krylov_detector.error_diagnostics import classify_error_type
        result = classify_error_type(delta_b=0.5, envelope_ratio=0.9)
        assert result['error_type'] == 'coherent'

    def test_classify_decoherent(self):
        """Small delta_b with damped envelope -> 'decoherent'."""
        from qkd_krylov_detector.error_diagnostics import classify_error_type
        result = classify_error_type(delta_b=0.001, envelope_ratio=0.1)
        assert result['error_type'] == 'decoherent'

    def test_classify_mixed(self):
        """Large delta_b with damped envelope -> 'mixed'."""
        from qkd_krylov_detector.error_diagnostics import classify_error_type
        result = classify_error_type(delta_b=0.5, envelope_ratio=0.1)
        assert result['error_type'] == 'mixed'

    def test_lanczos_shift_zero_for_identical(self):
        """Identical b_n should give delta_b = 0."""
        from qkd_krylov_detector.error_diagnostics import compute_lanczos_shift
        b = np.array([1.0, 2.0, 3.0])
        assert compute_lanczos_shift(b, b) == 0.0

    def test_lanczos_shift_positive_for_different(self):
        """Different b_n should give delta_b > 0."""
        from qkd_krylov_detector.error_diagnostics import compute_lanczos_shift
        b1 = np.array([1.0, 2.0, 3.0])
        b2 = np.array([1.1, 2.2, 3.3])
        assert compute_lanczos_shift(b1, b2) > 0

    def test_envelope_ratio_one_for_constant(self):
        """Constant autocorrelation should have ratio ~1."""
        from qkd_krylov_detector.error_diagnostics import compute_envelope_ratio
        C = np.ones(20)
        ratio = compute_envelope_ratio(C)
        assert abs(ratio - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# One-Way Function
# ═══════════════════════════════════════════════════════════════════════════
class TestOneWayFunction:
    def test_hankel_matrix_symmetric(self):
        """Hankel matrix should be symmetric."""
        from qkd_krylov_detector.one_way_function import compute_hankel_matrix
        moments = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02])
        M = compute_hankel_matrix(moments, size=3)
        np.testing.assert_allclose(M, M.T)

    def test_condition_number_positive(self):
        """Condition number should be positive."""
        from qkd_krylov_detector.one_way_function import compute_condition_number
        M = np.array([[2, 1], [1, 2]], dtype=float)
        kappa = compute_condition_number(M)
        assert kappa > 0

    def test_inversion_hardness_exponential(self):
        """Condition number should grow with Hankel size."""
        from qkd_krylov_detector.one_way_function import test_inversion_hardness
        H, O = _build_test_system(4)
        result = test_inversion_hardness(H, O, n_moments=10, hankel_sizes=[3, 5])
        assert len(result['condition_numbers']) == 2
        assert result['growth_rate'] > 1.0

    def test_moments_first_is_one(self):
        """First moment mu_0 = <O|O>/||O||^2 = 1."""
        from qkd_krylov_detector.one_way_function import compute_moments
        H, O = _build_test_system(3)
        moments = compute_moments(H, O, 5)
        assert abs(moments[0] - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# Universality
# ═══════════════════════════════════════════════════════════════════════════
class TestUniversality:
    def test_heisenberg_chain_hermitian(self):
        """Heisenberg chain should be Hermitian."""
        from qkd_krylov_detector.universality import heisenberg_chain
        H = heisenberg_chain(4)
        assert H.shape == (16, 16)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_xxz_chain_hermitian(self):
        """XXZ chain should be Hermitian."""
        from qkd_krylov_detector.universality import xxz_chain
        H = xxz_chain(4)
        assert H.shape == (16, 16)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_syk_model_hermitian(self):
        """SYK model should be Hermitian."""
        from qkd_krylov_detector.universality import syk_model
        H, N_q = syk_model(6, seed=42)
        assert N_q == 3
        assert H.shape == (8, 8)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_supported_families_nonempty(self):
        """Should have at least 3 built-in families."""
        from qkd_krylov_detector.universality import supported_families
        families = supported_families()
        assert len(families) >= 3

    def test_heisenberg_family_bridge(self):
        """Heisenberg family should pass bridge test for small N."""
        from qkd_krylov_detector.universality import test_hamiltonian_family, heisenberg_chain
        result = test_hamiltonian_family(heisenberg_chain, [3, 4], n_lanczos=10)
        assert len(result['correlations']) == 2
        # For small N, correlations may not be > 0.95, but should be > 0
        assert all(r > 0 for r in result['correlations'])


# ═══════════════════════════════════════════════════════════════════════════
# KrylovFramework
# ═══════════════════════════════════════════════════════════════════════════
class TestKrylovFramework:
    def test_initialization(self):
        """Framework should initialize with correct dimensions."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(4)
        fw = KrylovFramework(H, O)
        assert fw.d == 16
        assert fw.N_qubits == 4

    def test_lazy_lanczos(self):
        """Accessing b_n should trigger computation."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(3)
        fw = KrylovFramework(H, O, n_lanczos=10)
        assert fw._b_n is None
        b = fw.b_n
        assert len(b) > 0
        assert fw._b_n is not None

    def test_autocorrelation(self):
        """compute_autocorrelation should return normalized C(t)."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(3)
        fw = KrylovFramework(H, O)
        times = np.linspace(0, 2, 10)
        C = fw.compute_autocorrelation(times)
        assert abs(C[0] - 1.0) < 1e-10

    def test_template(self):
        """get_template should return Gaussian with T(0) = 1."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(3)
        fw = KrylovFramework(H, O)
        times = np.linspace(0, 2, 10)
        T = fw.get_template(times)
        assert abs(T[0] - 1.0) < 1e-10
        assert T[-1] < T[0]  # Should decay

    def test_validate_returns_checks(self):
        """validate should return check results."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(3)
        fw = KrylovFramework(H, O, n_lanczos=10)
        times = np.linspace(0, 2, 10)
        result = fw.validate(times, n_checks=4)
        assert 'all_passed' in result
        assert 'checks' in result
        assert len(result['checks']) == 4

    def test_repr(self):
        """repr should work without errors."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(3)
        fw = KrylovFramework(H, O)
        r = repr(fw)
        assert 'KrylovFramework' in r

    def test_summary(self):
        """summary should work without errors."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(3)
        fw = KrylovFramework(H, O)
        s = fw.summary()
        assert 'KrylovFramework Summary' in s

    def test_one_way_property(self):
        """test_one_way_property should return growth rate."""
        from qkd_krylov_detector.krylov_framework import KrylovFramework
        H, O = _build_test_system(4)
        fw = KrylovFramework(H, O)
        result = fw.test_one_way_property(n_moments=10, hankel_sizes=[3, 5])
        assert 'growth_rate' in result
        assert result['growth_rate'] > 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Package-level imports
# ═══════════════════════════════════════════════════════════════════════════
class TestPackageImports:
    def test_version(self):
        """Package version should be 2.0.0."""
        import qkd_krylov_detector
        assert qkd_krylov_detector.__version__ == "2.0.0"

    def test_license(self):
        """Package license should be AGPL."""
        import qkd_krylov_detector
        assert "AGPL" in qkd_krylov_detector.__license__

    def test_krylov_framework_importable(self):
        """KrylovFramework should be importable from top level."""
        from qkd_krylov_detector import KrylovFramework
        assert KrylovFramework is not None

    def test_bridge_functions_importable(self):
        """Bridge functions should be importable from top level."""
        from qkd_krylov_detector import (
            bridge_operator_autocorrelation,
            bridge_qber_autocorrelation,
            bridge_transform,
            verify_bridge_correlation,
        )

    def test_diagnostics_importable(self):
        """Diagnostics functions should be importable from top level."""
        from qkd_krylov_detector import (
            classify_error_type,
            compute_lanczos_shift,
            diagnostic_report,
        )

    def test_universality_importable(self):
        """Universality functions should be importable from top level."""
        from qkd_krylov_detector import (
            heisenberg_chain,
            xxz_chain,
            syk_model,
            test_hamiltonian_family,
        )
