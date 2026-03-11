"""
Error Diagnostics: Coherent vs. Decoherent Error Discrimination
================================================================

Implements the error-type discrimination framework from Paper [12].
The key insight is that coherent and decoherent errors leave qualitatively
different signatures in the Krylov framework:

    - **Coherent errors** (unitary perturbations): Shift the Lanczos
      coefficients b_n but preserve the autocorrelation envelope.
      Signature: Delta_b > threshold, envelope_ratio ~ 1.

    - **Decoherent errors** (Lindblad dissipation): Damp the
      autocorrelation envelope exponentially but leave b_n relatively
      unchanged. Signature: Delta_b ~ 0, envelope_ratio << 1.

This discrimination is crucial for quantum computer benchmarking:
it tells operators whether errors are correctable (coherent) or
require decoherence mitigation (decoherent).

This module provides:
    - classify_error_type: Determine "coherent", "decoherent", or "mixed"
    - compute_lanczos_shift: Quantify b_n deviation from reference
    - compute_envelope_ratio: Measure autocorrelation damping
    - diagnostic_report: Full diagnostic analysis
    - benchmark_channel: Benchmark a quantum channel

References:
    [12] D. Suess, "Open-System Physical Bridge: Extending the Krylov
         Eavesdropper Detector to Lindblad Dynamics," Zenodo, 2026.
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
from scipy.linalg import eigh


def compute_lanczos_shift(b_reference, b_test):
    """
    Compute the normalized Lanczos coefficient shift.

    Delta_b = sqrt(sum((b_ref - b_test)^2) / sum(b_ref^2))

    A large Delta_b indicates a coherent (unitary) perturbation.

    Parameters
    ----------
    b_reference : ndarray
        Reference (clean) Lanczos coefficients.
    b_test : ndarray
        Test (perturbed) Lanczos coefficients.

    Returns
    -------
    delta_b : float
        Normalized Lanczos shift. Values > 0.01 typically indicate
        coherent errors.

    Reference: Paper [12], Eq. (15)
    """
    b_ref = np.asarray(b_reference, dtype=float)
    b_test = np.asarray(b_test, dtype=float)
    n = min(len(b_ref), len(b_test))

    if n == 0:
        return 0.0

    ref_norm_sq = np.sum(b_ref[:n] ** 2)
    if ref_norm_sq < 1e-30:
        return 0.0

    delta_sq = np.sum((b_ref[:n] - b_test[:n]) ** 2)
    return float(np.sqrt(delta_sq / ref_norm_sq))


def compute_envelope_ratio(C_autocorr, early_fraction=0.2, late_fraction=0.2):
    """
    Compute the late-time to early-time envelope ratio.

    A ratio << 1 indicates exponential damping (decoherent errors).
    A ratio ~ 1 indicates preserved envelope (coherent errors or clean).

    Parameters
    ----------
    C_autocorr : ndarray
        Autocorrelation function C(t).
    early_fraction : float
        Fraction of time points to use for early-time average.
    late_fraction : float
        Fraction of time points to use for late-time average.

    Returns
    -------
    ratio : float
        Envelope ratio in [0, 1]. Values < 0.5 suggest decoherence.

    Reference: Paper [12], Section III.C
    """
    C = np.asarray(C_autocorr, dtype=float)
    n = len(C)

    if n < 4:
        return 1.0

    n_early = max(1, int(n * early_fraction))
    n_late = max(1, int(n * late_fraction))

    early_avg = np.mean(np.abs(C[:n_early]))
    late_avg = np.mean(np.abs(C[n - n_late:]))

    if early_avg < 1e-15:
        return 0.0

    return float(np.clip(late_avg / early_avg, 0.0, 1.0))


def classify_error_type(delta_b, envelope_ratio,
                        coherent_threshold=0.01,
                        decoherent_threshold=0.5):
    """
    Classify the error type based on Lanczos shift and envelope ratio.

    Decision logic:
        - delta_b > coherent_threshold AND envelope_ratio > decoherent_threshold
          -> "coherent" (unitary perturbation)
        - delta_b < coherent_threshold AND envelope_ratio < decoherent_threshold
          -> "decoherent" (Lindblad dissipation)
        - Both indicators triggered -> "mixed"
        - Neither triggered -> "clean"

    Parameters
    ----------
    delta_b : float
        Normalized Lanczos shift (from compute_lanczos_shift).
    envelope_ratio : float
        Late/early envelope ratio (from compute_envelope_ratio).
    coherent_threshold : float
        Threshold for coherent error detection (default: 0.01).
    decoherent_threshold : float
        Threshold for decoherent error detection (default: 0.5).

    Returns
    -------
    result : dict
        Keys:
        - 'error_type': str, one of "coherent", "decoherent", "mixed", "clean"
        - 'delta_b': float, the Lanczos shift
        - 'envelope_ratio': float, the envelope ratio
        - 'confidence': float, confidence score in [0, 1]
        - 'description': str, human-readable description

    Reference: Paper [12], Algorithm 1
    """
    has_coherent = delta_b > coherent_threshold
    has_decoherent = envelope_ratio < decoherent_threshold

    if has_coherent and has_decoherent:
        error_type = "mixed"
        confidence = min(delta_b / coherent_threshold,
                        (decoherent_threshold - envelope_ratio) / decoherent_threshold)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        description = (
            f"Mixed error detected: coherent shift Delta_b = {delta_b:.4f} "
            f"and decoherent damping (envelope ratio = {envelope_ratio:.4f})"
        )
    elif has_coherent:
        error_type = "coherent"
        confidence = float(np.clip(delta_b / (10 * coherent_threshold), 0.0, 1.0))
        description = (
            f"Coherent (unitary) error: Lanczos shift Delta_b = {delta_b:.4f} "
            f"with preserved envelope (ratio = {envelope_ratio:.4f})"
        )
    elif has_decoherent:
        error_type = "decoherent"
        confidence = float(np.clip(
            (decoherent_threshold - envelope_ratio) / decoherent_threshold, 0.0, 1.0
        ))
        description = (
            f"Decoherent error: exponential damping (envelope ratio = "
            f"{envelope_ratio:.4f}) with stable Lanczos coefficients "
            f"(Delta_b = {delta_b:.4f})"
        )
    else:
        error_type = "clean"
        confidence = float(np.clip(
            1.0 - delta_b / coherent_threshold, 0.0, 1.0
        ))
        description = (
            f"No significant error detected: Delta_b = {delta_b:.4f}, "
            f"envelope ratio = {envelope_ratio:.4f}"
        )

    return {
        'error_type': error_type,
        'delta_b': float(delta_b),
        'envelope_ratio': float(envelope_ratio),
        'confidence': confidence,
        'description': description,
    }


def diagnostic_report(H_clean, H_test, O, times, jump_ops=None,
                      gamma_list=None, n_lanczos=25):
    """
    Generate a full diagnostic report for a quantum channel.

    Computes Lanczos coefficients, autocorrelations, and classifies
    the error type.

    Parameters
    ----------
    H_clean : ndarray, shape (d, d)
        Reference (clean) Hamiltonian.
    H_test : ndarray, shape (d, d)
        Test Hamiltonian (may include coherent perturbation).
    O : ndarray, shape (d, d)
        Observable operator.
    times : ndarray, shape (n_t,)
        Time points for autocorrelation.
    jump_ops : list of ndarray or None
        Lindblad jump operators (for decoherent noise).
    gamma_list : list of float or None
        Dissipation rates.
    n_lanczos : int
        Number of Lanczos steps.

    Returns
    -------
    report : dict
        Comprehensive diagnostic report:
        - 'b_clean': Reference Lanczos coefficients
        - 'b_test': Test Lanczos coefficients
        - 'delta_b': Normalized Lanczos shift
        - 'C_clean': Clean autocorrelation
        - 'C_test': Test autocorrelation
        - 'envelope_ratio': Late/early ratio
        - 'classification': Error type classification dict
        - 'recommendations': List of recommended actions

    Reference: Paper [12], Section VI
    """
    from .physical_bridge import compute_lanczos_coefficients
    from .open_system_bridge import open_system_autocorrelation

    H_clean = np.asarray(H_clean, dtype=complex)
    H_test = np.asarray(H_test, dtype=complex)
    O = np.asarray(O, dtype=complex)
    times = np.asarray(times, dtype=float)

    # Lanczos coefficients
    b_clean = compute_lanczos_coefficients(H_clean, O, n_lanczos)
    b_test = compute_lanczos_coefficients(H_test, O, n_lanczos)

    # Lanczos shift
    delta_b = compute_lanczos_shift(b_clean, b_test)

    # Autocorrelations
    if jump_ops is not None and gamma_list is not None:
        C_test = open_system_autocorrelation(
            H_test, O, times, jump_ops, gamma_list
        )
    else:
        C_test = open_system_autocorrelation(H_test, O, times)

    C_clean = open_system_autocorrelation(H_clean, O, times)

    # Envelope ratio
    env_ratio = compute_envelope_ratio(C_test)

    # Classification
    classification = classify_error_type(delta_b, env_ratio)

    # Recommendations
    recommendations = _generate_recommendations(classification)

    return {
        'b_clean': b_clean,
        'b_test': b_test,
        'delta_b': delta_b,
        'C_clean': C_clean,
        'C_test': C_test,
        'envelope_ratio': env_ratio,
        'classification': classification,
        'recommendations': recommendations,
    }


def benchmark_channel(H, O, perturbation_ops, perturbation_strengths,
                      jump_ops_list=None, gamma_lists=None,
                      times=None, n_lanczos=25):
    """
    Benchmark a quantum channel across multiple error configurations.

    Parameters
    ----------
    H : ndarray, shape (d, d)
        Clean Hamiltonian.
    O : ndarray, shape (d, d)
        Observable.
    perturbation_ops : list of ndarray
        Coherent perturbation operators V_k.
    perturbation_strengths : list of float
        Strengths epsilon_k for each perturbation.
    jump_ops_list : list of list of ndarray or None
        Lindblad operators for each configuration.
    gamma_lists : list of list of float or None
        Dissipation rates for each configuration.
    times : ndarray or None
        Time points. If None, uses default range.
    n_lanczos : int
        Number of Lanczos steps.

    Returns
    -------
    benchmark : dict
        Keys:
        - 'n_configs': Number of configurations tested
        - 'results': List of diagnostic results
        - 'summary': Summary statistics

    Reference: Paper [12], Section VII (Benchmarking Application)
    """
    from .physical_bridge import compute_lanczos_coefficients
    from .open_system_bridge import open_system_autocorrelation

    H = np.asarray(H, dtype=complex)
    O = np.asarray(O, dtype=complex)

    if times is None:
        times = np.linspace(0, 6.0, 40)

    b_clean = compute_lanczos_coefficients(H, O, n_lanczos)
    C_clean = open_system_autocorrelation(H, O, times)

    results = []
    for i, (V, eps) in enumerate(zip(perturbation_ops, perturbation_strengths)):
        V = np.asarray(V, dtype=complex)
        H_pert = H + eps * V

        b_pert = compute_lanczos_coefficients(H_pert, O, n_lanczos)
        delta_b = compute_lanczos_shift(b_clean, b_pert)

        jops = jump_ops_list[i] if jump_ops_list is not None else None
        glist = gamma_lists[i] if gamma_lists is not None else None

        if jops is not None and glist is not None:
            C_pert = open_system_autocorrelation(
                H_pert, O, times, jops, glist
            )
        else:
            C_pert = open_system_autocorrelation(H_pert, O, times)

        env_ratio = compute_envelope_ratio(C_pert)
        classification = classify_error_type(delta_b, env_ratio)

        results.append({
            'perturbation_strength': float(eps),
            'delta_b': delta_b,
            'envelope_ratio': env_ratio,
            'classification': classification,
        })

    # Summary
    error_types = [r['classification']['error_type'] for r in results]
    summary = {
        'n_coherent': error_types.count('coherent'),
        'n_decoherent': error_types.count('decoherent'),
        'n_mixed': error_types.count('mixed'),
        'n_clean': error_types.count('clean'),
        'mean_delta_b': float(np.mean([r['delta_b'] for r in results])),
        'mean_envelope_ratio': float(np.mean([r['envelope_ratio'] for r in results])),
    }

    return {
        'n_configs': len(results),
        'results': results,
        'summary': summary,
    }


def _generate_recommendations(classification):
    """Generate actionable recommendations based on error classification."""
    error_type = classification['error_type']
    recommendations = []

    if error_type == "coherent":
        recommendations.extend([
            "Coherent error detected: consider applying dynamical decoupling.",
            "The error is unitary and potentially correctable with quantum "
            "error correction codes.",
            "Check for systematic calibration drifts in the quantum hardware.",
        ])
    elif error_type == "decoherent":
        recommendations.extend([
            "Decoherent error detected: the channel exhibits Lindblad-type "
            "dissipation.",
            "Consider decoherence mitigation techniques (e.g., error "
            "mitigation, zero-noise extrapolation).",
            "Check environmental coupling and shielding of the quantum device.",
        ])
    elif error_type == "mixed":
        recommendations.extend([
            "Mixed error detected: both coherent and decoherent components "
            "are present.",
            "Prioritize decoherence mitigation first, then address coherent "
            "errors.",
            "Consider a comprehensive noise characterization protocol.",
        ])
    else:
        recommendations.append(
            "Channel appears clean within detection thresholds."
        )

    return recommendations
