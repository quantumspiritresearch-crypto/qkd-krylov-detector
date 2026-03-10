"""Tests for the loschmidt_echo module."""

import numpy as np
import pytest
from qkd_krylov_detector.loschmidt_echo import (
    eigendecompose,
    compute_state_echo,
    compute_operator_echo,
    compute_echo_decay_rate,
    compute_operator_autocorrelation,
    _fast_lanczos,
)


@pytest.fixture
def small_system():
    """4-qubit test system for fast tests."""
    N = 4
    dim = 2 ** N
    np.random.seed(42)
    # Random Hermitian Hamiltonian
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H = (A + A.conj().T) / 2

    # Perturbation
    B = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    V = (B + B.conj().T) / 2

    # Observable
    O0 = np.diag(np.random.randn(dim))

    # Initial state
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0

    return H, V, O0, psi0, dim


class TestEigendecompose:
    def test_eigenvalues_real(self, small_system):
        H, _, _, _, _ = small_system
        E, V = eigendecompose(H)
        assert np.all(np.isreal(E))

    def test_eigenvectors_unitary(self, small_system):
        H, _, _, _, dim = small_system
        E, V = eigendecompose(H)
        np.testing.assert_allclose(V.conj().T @ V, np.eye(dim), atol=1e-12)

    def test_reconstruction(self, small_system):
        H, _, _, _, _ = small_system
        E, V = eigendecompose(H)
        H_reconstructed = V @ np.diag(E) @ V.conj().T
        np.testing.assert_allclose(H_reconstructed, H, atol=1e-10)


class TestStateEcho:
    def test_identity_at_t0(self, small_system):
        H, V_pert, _, psi0, _ = small_system
        E_c, V_c = eigendecompose(H)
        E_p, V_p = eigendecompose(H + 0.1 * V_pert)
        M = compute_state_echo(E_c, V_c, E_p, V_p, psi0, t=0.0)
        assert abs(M - 1.0) < 1e-10, "Echo must be 1 at t=0"

    def test_no_perturbation(self, small_system):
        H, _, _, psi0, _ = small_system
        E, V = eigendecompose(H)
        M = compute_state_echo(E, V, E, V, psi0, t=10.0)
        assert abs(M - 1.0) < 1e-10, "Echo must be 1 without perturbation"

    def test_bounded(self, small_system):
        H, V_pert, _, psi0, _ = small_system
        E_c, V_c = eigendecompose(H)
        E_p, V_p = eigendecompose(H + 0.5 * V_pert)
        M = compute_state_echo(E_c, V_c, E_p, V_p, psi0, t=5.0)
        assert 0 <= M <= 1.0 + 1e-10, f"Echo must be in [0, 1], got {M}"

    def test_decays_with_perturbation(self, small_system):
        H, V_pert, _, psi0, _ = small_system
        E_c, V_c = eigendecompose(H)
        E_p, V_p = eigendecompose(H + 0.5 * V_pert)
        M = compute_state_echo(E_c, V_c, E_p, V_p, psi0, t=20.0)
        assert M < 0.95, f"Echo should decay with perturbation, got {M}"


class TestOperatorEcho:
    def test_identity_at_t0(self, small_system):
        H, V_pert, O0, _, dim = small_system
        E_c, V_c = eigendecompose(H)
        E_p, V_p = eigendecompose(H + 0.1 * V_pert)
        M = compute_operator_echo(E_c, V_c, E_p, V_p, O0, t=0.0, dim=dim)
        assert abs(M - 1.0) < 1e-10, "Operator echo must be 1 at t=0"

    def test_no_perturbation(self, small_system):
        H, _, O0, _, dim = small_system
        E, V = eigendecompose(H)
        M = compute_operator_echo(E, V, E, V, O0, t=10.0, dim=dim)
        assert abs(M - 1.0) < 1e-10, "Operator echo must be 1 without perturbation"

    def test_bounded(self, small_system):
        H, V_pert, O0, _, dim = small_system
        E_c, V_c = eigendecompose(H)
        E_p, V_p = eigendecompose(H + 0.5 * V_pert)
        M = compute_operator_echo(E_c, V_c, E_p, V_p, O0, t=5.0, dim=dim)
        assert -0.01 <= M <= 1.01, f"Operator echo out of bounds: {M}"


class TestEchoDecayRate:
    def test_zero_for_no_perturbation(self, small_system):
        H, _, O0, _, _ = small_system
        E, V = eigendecompose(H)
        rate = compute_echo_decay_rate(E, V, E, V, O0, t_max=20)
        assert rate < 0.01, f"Decay rate should be ~0 without perturbation, got {rate}"

    def test_increases_with_gamma(self, small_system):
        H, V_pert, O0, _, _ = small_system
        E_c, V_c = eigendecompose(H)
        rates = []
        for gamma in [0.01, 0.1, 0.5]:
            E_p, V_p = eigendecompose(H + gamma * V_pert)
            rate = compute_echo_decay_rate(E_c, V_c, E_p, V_p, O0, t_max=20)
            rates.append(rate)
        assert rates[-1] > rates[0], "Decay rate should increase with gamma"


class TestAutocorrelation:
    def test_normalized_at_t0(self, small_system):
        H, _, O0, _, dim = small_system
        E, V = eigendecompose(H)
        C = compute_operator_autocorrelation(E, V, O0, np.array([0.0]), dim)
        assert abs(C[0] - 1.0) < 1e-10, f"C(0) must be 1, got {C[0]}"

    def test_real_valued(self, small_system):
        H, _, O0, _, dim = small_system
        E, V = eigendecompose(H)
        times = np.linspace(0, 10, 20)
        C = compute_operator_autocorrelation(E, V, O0, times, dim)
        assert np.all(np.isreal(C)), "Autocorrelation must be real"


class TestLanczos:
    def test_positive(self, small_system):
        H, _, O0, _, dim = small_system
        bs = _fast_lanczos(H, O0, 10, dim)
        assert np.all(bs > 0), "Lanczos coefficients must be positive"

    def test_perturbation_changes_coefficients(self, small_system):
        H, V_pert, O0, _, dim = small_system
        b_clean = _fast_lanczos(H, O0, 10, dim)
        b_pert = _fast_lanczos(H + 0.1 * V_pert, O0, 10, dim)
        assert not np.allclose(b_clean, b_pert, atol=1e-6), \
            "Perturbation should change Lanczos coefficients"
