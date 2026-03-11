"""
KrylovFramework: Central Orchestrator for the Krylov Detection Pipeline
=========================================================================

The KrylovFramework class provides a unified API for the complete
Krylov eavesdropper detection and quantum channel benchmarking pipeline.

It integrates:
    - Physical Bridge (Paper [11]): C_op <-> C_QBER mapping
    - Open-System Bridge (Paper [12]): Lindblad extension
    - Error Diagnostics (Paper [12]): Coherent vs. decoherent discrimination
    - One-Way Function (Paper [11]): Hankel matrix hardness
    - Universality (Paper [11]): Multi-Hamiltonian validation
    - Detection Pipeline (Papers [1]-[6]): Sidereal filter + template matching

Usage:
    >>> from qkd_krylov_detector import KrylovFramework
    >>> fw = KrylovFramework(H, O)
    >>> fw.compute_lanczos()
    >>> result = fw.detect(qber_residuum, times)
    >>> diagnosis = fw.diagnose(H_test, jump_ops, gamma_list)
    >>> benchmark = fw.benchmark(perturbation_ops, strengths)

References:
    [11] D. Suess, "Theoretical Foundations of the Krylov Eavesdropper
         Detector," Zenodo, 2026. DOI: 10.5281/zenodo.18957362
    [12] D. Suess, "Open-System Physical Bridge," Zenodo, 2026.
         DOI: 10.5281/zenodo.18959827

Author: Daniel Suess
License: AGPL-3.0-or-later
SPDX-License-Identifier: AGPL-3.0-or-later

Copyright (C) 2026 Daniel Suess
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy as np
import warnings
from scipy.linalg import eigh
from scipy.stats import pearsonr as _pearsonr

from .physical_bridge import (
    compute_operator_autocorrelation,
    compute_qber_autocorrelation,
    bridge_transform,
    verify_bridge_correlation,
    compute_lanczos_coefficients,
    compute_susceptibility,
    full_bridge_analysis,
)
from .open_system_bridge import (
    build_adjoint_lindbladian,
    open_system_autocorrelation,
    compute_decoherence_envelope,
    bridge_with_dissipation,
    compare_open_closed,
)
from .error_diagnostics import (
    compute_lanczos_shift,
    compute_envelope_ratio,
    classify_error_type,
    diagnostic_report,
    benchmark_channel,
)
from .one_way_function import (
    compute_moments,
    compute_hankel_matrix,
    compute_condition_number,
    test_inversion_hardness,
)


class KrylovFramework:
    """
    Central orchestrator for the Krylov detection and benchmarking pipeline.

    This class provides a unified interface to all components of the
    framework, managing state and ensuring consistency across analyses.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        System Hamiltonian (numpy array, not QuTiP).
    O : ndarray, shape (d, d)
        Observable operator (default probe).
    n_lanczos : int
        Number of Lanczos steps for coefficient computation.

    Attributes
    ----------
    H : ndarray
        The Hamiltonian.
    O : ndarray
        The observable operator.
    d : int
        Hilbert space dimension.
    N_qubits : int
        Number of qubits (inferred from d).
    b_n : ndarray or None
        Lanczos coefficients (computed on demand).
    lanczos_slope : float or None
        Average slope of b_n.
    C_op : ndarray or None
        Operator autocorrelation (computed on demand).

    Examples
    --------
    Basic detection pipeline:

    >>> import numpy as np
    >>> from qkd_krylov_detector.universality import heisenberg_chain
    >>> from qkd_krylov_detector import KrylovFramework
    >>> H = heisenberg_chain(4)
    >>> O = np.diag([1, -1, 1, -1, 1, -1, 1, -1,
    ...              1, -1, 1, -1, 1, -1, 1, -1])  # sigma_z on qubit 0
    >>> fw = KrylovFramework(H, O)
    >>> fw.compute_lanczos()
    >>> print(f"Lanczos slope: {fw.lanczos_slope:.3f}")
    """

    def __init__(self, H, O, n_lanczos=25):
        self.H = np.asarray(H, dtype=complex)
        self.O = np.asarray(O, dtype=complex)
        self.n_lanczos = n_lanczos

        # Validate inputs
        if self.H.shape[0] != self.H.shape[1]:
            raise ValueError("Hamiltonian must be a square matrix.")
        if self.O.shape != self.H.shape:
            raise ValueError("Observable must have same shape as Hamiltonian.")

        self.d = self.H.shape[0]
        self.N_qubits = int(np.round(np.log2(self.d)))

        # Computed quantities (lazy evaluation)
        self._b_n = None
        self._lanczos_slope = None
        self._C_op = None
        self._C_op_times = None
        self._eigendecomp = None

    # ── Properties ──────────────────────────────────────────────────

    @property
    def b_n(self):
        """Lanczos coefficients (computed on first access)."""
        if self._b_n is None:
            self.compute_lanczos()
        return self._b_n

    @property
    def lanczos_slope(self):
        """Average slope of Lanczos coefficients."""
        if self._lanczos_slope is None:
            if self._b_n is None:
                self.compute_lanczos()
            if len(self._b_n) >= 2:
                self._lanczos_slope = float(np.mean(np.diff(self._b_n)))
            else:
                self._lanczos_slope = 0.0
        return self._lanczos_slope

    @property
    def eigendecomp(self):
        """Eigendecomposition of H (cached)."""
        if self._eigendecomp is None:
            E, V = eigh(self.H)
            self._eigendecomp = (E, V)
        return self._eigendecomp

    # ── Core Computations ───────────────────────────────────────────

    def compute_lanczos(self, n_steps=None):
        """
        Compute Lanczos coefficients b_n.

        Parameters
        ----------
        n_steps : int or None
            Number of steps. If None, uses self.n_lanczos.

        Returns
        -------
        b_n : ndarray
            Lanczos coefficients.
        """
        if n_steps is None:
            n_steps = self.n_lanczos

        self._b_n = compute_lanczos_coefficients(self.H, self.O, n_steps)
        self._lanczos_slope = None  # Reset cached slope
        return self._b_n

    def compute_autocorrelation(self, times, method="eigen"):
        """
        Compute the operator autocorrelation C_op(t).

        Parameters
        ----------
        times : ndarray
            Time points.
        method : str
            "eigen" or "expm".

        Returns
        -------
        C_op : ndarray
            Operator autocorrelation.
        """
        times = np.asarray(times, dtype=float)
        self._C_op = compute_operator_autocorrelation(
            self.H, self.O, times, method=method
        )
        self._C_op_times = times
        return self._C_op

    def get_template(self, times):
        """
        Get the Gaussian autocorrelation template from Lanczos slope.

        Parameters
        ----------
        times : ndarray
            Time points.

        Returns
        -------
        template : ndarray
            Gaussian template exp(-0.5 * (slope * t)^2).
        """
        slope = self.lanczos_slope
        times = np.asarray(times, dtype=float)
        template = np.exp(-0.5 * (slope * times) ** 2)
        if template[0] > 0:
            template = template / template[0]
        return template

    # ── Physical Bridge ─────────────────────────────────────────────

    def verify_bridge(self, qber_residuum, times=None, max_lag=None):
        """
        Verify the Physical Bridge against QBER data.

        Parameters
        ----------
        qber_residuum : ndarray
            QBER time series (after sidereal filtering).
        times : ndarray or None
            Time points for C_op. If None, uses default range.
        max_lag : int or None
            Maximum lag for comparison.

        Returns
        -------
        result : dict
            Bridge verification results including Pearson r.
        """
        if times is None:
            times = np.linspace(0, 6.0, 40)

        return full_bridge_analysis(
            self.H, self.O, qber_residuum, times, self.n_lanczos
        )

    def compute_bridge_transform(self, times):
        """
        Compute the predicted QBER autocorrelation via bridge transform.

        Parameters
        ----------
        times : ndarray
            Time points.

        Returns
        -------
        C_predicted : ndarray
            Predicted QBER autocorrelation.
        """
        C_op = self.compute_autocorrelation(times)
        return bridge_transform(C_op, self.lanczos_slope)

    # ── Detection ───────────────────────────────────────────────────

    def detect(self, qber_residuum, t_axis, window_size=100, step=25):
        """
        Run the template-matching detector on QBER data.

        Parameters
        ----------
        qber_residuum : ndarray
            Sidereal-filtered QBER time series.
        t_axis : ndarray
            Time axis.
        window_size : int
            Sliding window size.
        step : int
            Step between windows.

        Returns
        -------
        centers : ndarray
            Time centers of windows.
        scores : ndarray
            Detection scores (RMSE from template).
        """
        from .template_detector import krylov_dynamic_detector
        return krylov_dynamic_detector(
            qber_residuum, self.b_n, t_axis,
            window_size=window_size, step=step
        )

    def detect_with_diagnostics(self, qber_residuum, t_axis,
                                 H_test=None, jump_ops=None,
                                 gamma_list=None,
                                 window_size=100, step=25):
        """
        Run detection with error-type diagnostics.

        Combines the template-matching detector with error classification
        to provide both detection scores and error type information.

        Parameters
        ----------
        qber_residuum : ndarray
            QBER time series.
        t_axis : ndarray
            Time axis.
        H_test : ndarray or None
            Test Hamiltonian (for Lanczos shift analysis).
        jump_ops : list of ndarray or None
            Jump operators (for decoherence analysis).
        gamma_list : list of float or None
            Dissipation rates.
        window_size : int
            Detection window size.
        step : int
            Step between windows.

        Returns
        -------
        result : dict
            Keys:
            - 'centers': Time centers
            - 'scores': Detection scores
            - 'mean_score': Mean detection score
            - 'max_score': Maximum detection score
            - 'is_anomalous': Whether anomaly is detected
            - 'diagnostics': Error classification (if H_test provided)
        """
        centers, scores = self.detect(qber_residuum, t_axis,
                                       window_size, step)

        mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0

        # Simple anomaly detection: score > mean + 3*std
        if len(scores) > 1:
            threshold = np.mean(scores) + 3 * np.std(scores)
            is_anomalous = bool(np.any(scores > threshold))
        else:
            is_anomalous = False

        result = {
            'centers': centers,
            'scores': scores,
            'mean_score': mean_score,
            'max_score': max_score,
            'is_anomalous': is_anomalous,
        }

        # Optional diagnostics
        if H_test is not None:
            times = np.linspace(0, 6.0, 40)
            diag = diagnostic_report(
                self.H, H_test, self.O, times,
                jump_ops=jump_ops, gamma_list=gamma_list,
                n_lanczos=self.n_lanczos
            )
            result['diagnostics'] = diag

        return result

    # ── Error Diagnostics ───────────────────────────────────────────

    def diagnose(self, H_test, jump_ops=None, gamma_list=None, times=None):
        """
        Diagnose error type for a test Hamiltonian.

        Parameters
        ----------
        H_test : ndarray
            Test Hamiltonian.
        jump_ops : list of ndarray or None
            Jump operators.
        gamma_list : list of float or None
            Dissipation rates.
        times : ndarray or None
            Time points.

        Returns
        -------
        report : dict
            Diagnostic report with error classification.
        """
        if times is None:
            times = np.linspace(0, 6.0, 40)

        return diagnostic_report(
            self.H, H_test, self.O, times,
            jump_ops=jump_ops, gamma_list=gamma_list,
            n_lanczos=self.n_lanczos
        )

    # ── Benchmarking ────────────────────────────────────────────────

    def benchmark(self, perturbation_ops, perturbation_strengths,
                  jump_ops_list=None, gamma_lists=None, times=None):
        """
        Benchmark a quantum channel across error configurations.

        Parameters
        ----------
        perturbation_ops : list of ndarray
            Coherent perturbation operators.
        perturbation_strengths : list of float
            Perturbation strengths.
        jump_ops_list : list of list of ndarray or None
            Jump operators per configuration.
        gamma_lists : list of list of float or None
            Dissipation rates per configuration.
        times : ndarray or None
            Time points.

        Returns
        -------
        result : dict
            Benchmark results with per-configuration diagnostics.
        """
        return benchmark_channel(
            self.H, self.O,
            perturbation_ops, perturbation_strengths,
            jump_ops_list=jump_ops_list,
            gamma_lists=gamma_lists,
            times=times,
            n_lanczos=self.n_lanczos,
        )

    # ── Open-System Analysis ────────────────────────────────────────

    def open_system_analysis(self, times, jump_ops, gamma_list,
                              qber_residuum=None):
        """
        Run the open-system Physical Bridge analysis.

        Parameters
        ----------
        times : ndarray
            Time points.
        jump_ops : list of ndarray
            Jump operators.
        gamma_list : list of float
            Dissipation rates.
        qber_residuum : ndarray or None
            Optional QBER data for verification.

        Returns
        -------
        result : dict
            Open-system bridge analysis results.
        """
        return bridge_with_dissipation(
            self.H, self.O, times, jump_ops, gamma_list,
            qber_residuum=qber_residuum
        )

    # ── One-Way Function ────────────────────────────────────────────

    def test_one_way_property(self, n_moments=20, hankel_sizes=None):
        """
        Test the one-way function property.

        Parameters
        ----------
        n_moments : int
            Number of moments.
        hankel_sizes : list of int or None
            Hankel matrix sizes.

        Returns
        -------
        result : dict
            One-way function analysis results.
        """
        return test_inversion_hardness(
            self.H, self.O, n_moments, hankel_sizes
        )

    # ── Validation ──────────────────────────────────────────────────

    def validate(self, times=None, n_checks=5):
        """
        Run self-consistency checks on the framework.

        Verifies:
        1. Lanczos coefficients are positive and growing
        2. Autocorrelation is normalized (C(0) = 1)
        3. Template matches autocorrelation (r > 0.9)
        4. Hamiltonian is Hermitian
        5. One-way property holds

        Parameters
        ----------
        times : ndarray or None
            Time points for autocorrelation check.
        n_checks : int
            Number of checks to run (1-5).

        Returns
        -------
        result : dict
            Keys:
            - 'all_passed': True if all checks pass
            - 'checks': List of individual check results
        """
        if times is None:
            times = np.linspace(0, 6.0, 40)

        checks = []

        # Check 1: Lanczos coefficients
        if n_checks >= 1:
            b = self.b_n
            passed = len(b) > 0 and np.all(b > 0)
            checks.append({
                'name': 'lanczos_positive',
                'passed': bool(passed),
                'detail': f'{len(b)} coefficients, all positive: {passed}',
            })

        # Check 2: Autocorrelation normalization
        if n_checks >= 2:
            C = self.compute_autocorrelation(times)
            passed = abs(C[0] - 1.0) < 0.01
            checks.append({
                'name': 'autocorrelation_normalized',
                'passed': bool(passed),
                'detail': f'C(0) = {C[0]:.6f}',
            })

        # Check 3: Template-autocorrelation match
        if n_checks >= 3:
            template = self.get_template(times)
            C = self.compute_autocorrelation(times)
            if np.std(C) > 1e-15 and np.std(template) > 1e-15:
                r, _ = _pearsonr(C, template)
                passed = r > 0.9
            else:
                r = 0.0
                passed = False
            checks.append({
                'name': 'template_match',
                'passed': bool(passed),
                'detail': f'Pearson r = {r:.4f}',
            })

        # Check 4: Hermiticity
        if n_checks >= 4:
            diff = np.max(np.abs(self.H - self.H.conj().T))
            passed = diff < 1e-10
            checks.append({
                'name': 'hermitian',
                'passed': bool(passed),
                'detail': f'max|H - H^dag| = {diff:.2e}',
            })

        # Check 5: One-way property
        if n_checks >= 5:
            owf = self.test_one_way_property(n_moments=10, hankel_sizes=[3, 5])
            passed = owf['growth_rate'] > 1.5
            checks.append({
                'name': 'one_way_function',
                'passed': bool(passed),
                'detail': f'Growth rate = {owf["growth_rate"]:.2f}',
            })

        all_passed = all(c['passed'] for c in checks)

        return {
            'all_passed': all_passed,
            'checks': checks,
        }

    # ── Representation ──────────────────────────────────────────────

    def __repr__(self):
        b_info = f"{len(self._b_n)} coefficients" if self._b_n is not None else "not computed"
        return (
            f"KrylovFramework(d={self.d}, N_qubits={self.N_qubits}, "
            f"n_lanczos={self.n_lanczos}, b_n={b_info})"
        )

    def summary(self):
        """
        Return a human-readable summary of the framework state.

        Returns
        -------
        text : str
            Multi-line summary.
        """
        lines = [
            f"KrylovFramework Summary",
            f"  Hilbert space dimension: {self.d}",
            f"  Number of qubits: {self.N_qubits}",
            f"  Lanczos steps: {self.n_lanczos}",
        ]

        if self._b_n is not None:
            lines.append(f"  Lanczos coefficients: {len(self._b_n)} computed")
            lines.append(f"  Lanczos slope: {self.lanczos_slope:.4f}")
        else:
            lines.append("  Lanczos coefficients: not yet computed")

        if self._C_op is not None:
            lines.append(f"  Autocorrelation: computed ({len(self._C_op)} points)")
        else:
            lines.append("  Autocorrelation: not yet computed")

        return "\n".join(lines)
