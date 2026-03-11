"""
QKD Krylov Detector — Python Package
=====================================

A comprehensive eavesdropper detection and quantum channel benchmarking
framework based on Krylov complexity, the Physical Bridge theorem, and
the operator-space Loschmidt echo.

v2.0.0 — AGPL-3.0-or-later

New in v2.0.0:
    KrylovFramework     — Central orchestrator for the full pipeline
    physical_bridge      — Formal C_QBER <-> C_op bridge (Paper [11])
    open_system_bridge   — Lindblad extension (Paper [12])
    error_diagnostics    — Coherent vs. decoherent error discrimination
    one_way_function     — Hankel matrix hardness / one-way property
    universality         — Multi-Hamiltonian universality tests

Core Detection Pipeline (3 Layers):
    sidereal_filter     — Layer 1: FFT-based removal of sidereal/diurnal periodicities
    lanczos_extractor   — Layer 2: Lanczos algorithm for Krylov b_n coefficients
    template_detector   — Layer 3: Gaussian template matching on QBER autocorrelation

Extended Modules:
    bb84_simulation     — BB84 protocol simulation (clean + IR/BS/partial attacks)
    attack_classifier   — Multi-attack classification (IR/BS/Blinding/PNS) + CUSUM
    calibration         — Calibrated slope detector + Option B slope fingerprint
    qber_simulator      — QBER time series generation (idealized + realistic noise)
    spectral_analysis   — Spectral statistics (r ratio, regime classification)
    hamiltonian         — Dense Heisenberg chain Hamiltonian (QuTiP, N <= 8)
    sparse_hamiltonian  — Sparse Hamiltonian for finite-size scaling (N > 8)
    pulsar_analysis     — Partial F-test framework for pulsar timing validation
    quantum_eve         — Quantum Eve simulation (reduced autocorrelation, 4 strategies)
    krylov_bridge       — Krylov-Statistical Bridge (b_n deviation <-> QBER statistics)
    demo_framework      — OOP wrappers (SiderealFilter, KrylovEngine, scenarios)
    loschmidt_echo      — Operator-space Loschmidt echo (Paper [10])

References:
    [1]  D. Suess, "Deconvolution of Sidereal and Diurnal Periodicities in QKD,"
         Zenodo, 2026. DOI: 10.5281/zenodo.18701222
    [2]  D. Suess, "Dual-Layer Sidereal Detection Framework v2,"
         Zenodo, 2026. DOI: 10.5281/zenodo.18768750
    [3]  D. Suess, "Real-Data Validation on NANOGrav 15-Year Dataset,"
         Zenodo, 2026. DOI: 10.5281/zenodo.18792775
    [4]  D. Suess, "Scrambling vs. Recurrence: Microscopic Origin of the
         Quantum Arrow of Time," Zenodo, 2026. DOI: 10.5281/zenodo.18813710
    [5]  D. Suess, "QKD Eve Detector: A Unified Framework -- Parts I-III,"
         Zenodo, 2026. DOI: 10.5281/zenodo.18873824
    [6]  D. Suess, "Quantum Scrambling as a Cryptographic Resource,"
         Zenodo, 2026. DOI: 10.5281/zenodo.18889224
    [10] D. Suess, "The Krylov Eavesdropper Detector as an Operator-Space
         Loschmidt Echo," Zenodo, 2026. DOI: 10.5281/zenodo.18939996
    [11] D. Suess, "Theoretical Foundations of the Krylov Eavesdropper
         Detector," Zenodo, 2026. DOI: 10.5281/zenodo.18957362
    [12] D. Suess, "Open-System Physical Bridge: Extending the Krylov
         Eavesdropper Detector to Lindblad Dynamics," Zenodo, 2026.
         DOI: 10.5281/zenodo.18959827

Author: Daniel Suess
License: AGPL-3.0-or-later (versions >= 2.0.0); MIT (versions <= 1.9.1)
"""

__version__ = "2.0.0"
__author__ = "Daniel Suess"
__license__ = "AGPL-3.0-or-later"

# ═══════════════════════════════════════════════════════════════════
# NEW in v2.0.0 — AGPL-licensed core modules
# ═══════════════════════════════════════════════════════════════════

# Central orchestrator
from .krylov_framework import KrylovFramework

# Physical Bridge (Paper [11])
from .physical_bridge import (
    compute_operator_autocorrelation as bridge_operator_autocorrelation,
    compute_qber_autocorrelation as bridge_qber_autocorrelation,
    bridge_transform,
    verify_bridge_correlation,
    compute_susceptibility,
    compute_lanczos_coefficients as bridge_lanczos_coefficients,
    full_bridge_analysis,
)

# Open-System Bridge (Paper [12])
from .open_system_bridge import (
    build_adjoint_lindbladian,
    lindblad_evolve,
    open_system_autocorrelation,
    compute_decoherence_envelope,
    bridge_with_dissipation,
    compare_open_closed,
)

# Error Diagnostics (Paper [12])
from .error_diagnostics import (
    classify_error_type,
    compute_lanczos_shift,
    compute_envelope_ratio,
    diagnostic_report,
    benchmark_channel,
)

# One-Way Function (Paper [11])
from .one_way_function import (
    compute_moments,
    compute_hankel_matrix,
    compute_condition_number,
    forward_map,
    test_inversion_hardness,
    scaling_analysis,
)

# Universality (Paper [11])
from .universality import (
    heisenberg_chain,
    xxz_chain,
    syk_model,
    test_hamiltonian_family,
    compute_universality_score,
    supported_families,
)

# ═══════════════════════════════════════════════════════════════════
# Existing modules (MIT-originated, now AGPL due to dependencies)
# ═══════════════════════════════════════════════════════════════════

# Core Detection Pipeline
from .sidereal_filter import sidereal_filter, sidereal_filter_irregular
from .lanczos_extractor import compute_lanczos, get_theoretical_autocorrelation
from .template_detector import krylov_dynamic_detector, compute_roc, compute_auc
from .hamiltonian import build_hamiltonian, build_hamiltonian_with_eve, get_op

# QBER Simulation
from .qber_simulator import (
    make_clean_qber, make_eve_qber,
    make_realistic_clean_qber, make_realistic_eve_qber
)

# Spectral Analysis
from .spectral_analysis import compute_r_ratio, classify_regime

# BB84 Protocol Simulation
from .bb84_simulation import (
    bb84_clean, bb84_intercept_resend, bb84_beam_splitting,
    bb84_window, make_bb84_timeseries
)

# Multi-Attack Classifier
from .attack_classifier import (
    make_clean as make_clean_attack, make_ir, make_bs,
    make_blinding, make_pns,
    find_attack_window, extract_features,
    cusum_detect, spectral_anomaly_score,
    ATTACK_LABELS
)

# Calibrated Detector + Option B
from .calibration import (
    gaussian_ac, fit_slope, calibrate,
    calibrated_detect, krylov_slope_detector
)

# Sparse Hamiltonian (N > 8)
from .sparse_hamiltonian import (
    build_hamiltonian_sparse, spectral_statistics_sparse,
    finite_size_scaling, kron_op
)

# Pulsar Timing Analysis
from .pulsar_analysis import (
    make_design_matrix, partial_f_test,
    classify_gaps, compute_sidereal_amplitude
)

# Quantum Eve Simulation
from .quantum_eve import (
    build_total_hamiltonian, build_channel_hamiltonian,
    compute_channel_autocorrelation, compute_reduced_autocorrelation,
    gaussian_template,
    make_classical_eve_qber, make_quantum_eve_qber,
    compute_anomaly_scores, compute_eve_detection_stats,
    STRATEGIES
)

# Krylov-Statistical Bridge (deprecated, use physical_bridge)
from .krylov_bridge import (
    bn_deviation, sim_stats, gamma_sweep,
    sensitivity_vs_gamma, krylov_proxy
)

# Demo Framework (deprecated, use KrylovFramework)
from .demo_framework import (
    SiderealFilter, KrylovEngine,
    classify_window, make_scenario
)

# Loschmidt echo module (Paper [10])
from .loschmidt_echo import (
    eigendecompose,
    compute_state_echo,
    compute_operator_echo,
    compute_echo_decay_rate,
    compute_operator_autocorrelation,
    loschmidt_krylov_correlation,
)

__all__ = [
    # ── v2.0.0 Core (AGPL) ──────────────────────────────────────
    "KrylovFramework",
    # Physical Bridge
    "bridge_operator_autocorrelation",
    "bridge_qber_autocorrelation",
    "bridge_transform",
    "verify_bridge_correlation",
    "compute_susceptibility",
    "bridge_lanczos_coefficients",
    "full_bridge_analysis",
    # Open-System Bridge
    "build_adjoint_lindbladian",
    "lindblad_evolve",
    "open_system_autocorrelation",
    "compute_decoherence_envelope",
    "bridge_with_dissipation",
    "compare_open_closed",
    # Error Diagnostics
    "classify_error_type",
    "compute_lanczos_shift",
    "compute_envelope_ratio",
    "diagnostic_report",
    "benchmark_channel",
    # One-Way Function
    "compute_moments",
    "compute_hankel_matrix",
    "compute_condition_number",
    "forward_map",
    "test_inversion_hardness",
    "scaling_analysis",
    # Universality
    "heisenberg_chain",
    "xxz_chain",
    "syk_model",
    "test_hamiltonian_family",
    "compute_universality_score",
    "supported_families",
    # ── Existing modules ─────────────────────────────────────────
    # Layer 1: Sidereal Filter
    "sidereal_filter",
    "sidereal_filter_irregular",
    # Layer 2: Lanczos / Krylov
    "compute_lanczos",
    "get_theoretical_autocorrelation",
    # Layer 3: Template Detector
    "krylov_dynamic_detector",
    "compute_roc",
    "compute_auc",
    # Hamiltonian (Dense)
    "build_hamiltonian",
    "build_hamiltonian_with_eve",
    "get_op",
    # QBER Simulation
    "make_clean_qber",
    "make_eve_qber",
    "make_realistic_clean_qber",
    "make_realistic_eve_qber",
    # Spectral Analysis
    "compute_r_ratio",
    "classify_regime",
    # BB84 Protocol
    "bb84_clean",
    "bb84_intercept_resend",
    "bb84_beam_splitting",
    "bb84_window",
    "make_bb84_timeseries",
    # Attack Classifier
    "make_clean_attack",
    "make_ir",
    "make_bs",
    "make_blinding",
    "make_pns",
    "find_attack_window",
    "extract_features",
    "cusum_detect",
    "spectral_anomaly_score",
    "ATTACK_LABELS",
    # Calibration + Option B
    "gaussian_ac",
    "fit_slope",
    "calibrate",
    "calibrated_detect",
    "krylov_slope_detector",
    # Sparse Hamiltonian
    "build_hamiltonian_sparse",
    "spectral_statistics_sparse",
    "finite_size_scaling",
    "kron_op",
    # Pulsar Analysis
    "make_design_matrix",
    "partial_f_test",
    "classify_gaps",
    "compute_sidereal_amplitude",
    # Quantum Eve
    "build_total_hamiltonian",
    "build_channel_hamiltonian",
    "compute_channel_autocorrelation",
    "compute_reduced_autocorrelation",
    "gaussian_template",
    "make_classical_eve_qber",
    "make_quantum_eve_qber",
    "compute_anomaly_scores",
    "compute_eve_detection_stats",
    "STRATEGIES",
    # Krylov Bridge
    "bn_deviation",
    "sim_stats",
    "gamma_sweep",
    "sensitivity_vs_gamma",
    "krylov_proxy",
    # Demo Framework
    "SiderealFilter",
    "KrylovEngine",
    "classify_window",
    "make_scenario",
    # Loschmidt Echo
    "eigendecompose",
    "compute_state_echo",
    "compute_operator_echo",
    "compute_echo_decay_rate",
    "compute_operator_autocorrelation",
    "loschmidt_krylov_correlation",
]
