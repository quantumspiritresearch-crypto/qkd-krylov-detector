"""
QKD Krylov Detector — Python Package
=====================================

A comprehensive eavesdropper detection framework for Quantum Key Distribution
based on Krylov complexity and sidereal filtering.

Core Detection Pipeline (3 Layers):
    sidereal_filter     — Layer 1: FFT-based removal of sidereal/diurnal periodicities
    lanczos_extractor   — Layer 2: Lanczos algorithm for Krylov b_n coefficients
    template_detector   — Layer 3: Gaussian template matching on QBER autocorrelation

Extended Modules:
    bb84_simulation     — BB84 protocol simulation (clean + IR/BS/partial attacks)
    attack_classifier   — Multi-attack classification (IR/BS/Blinding/PNS) + CUSUM
    calibration         — Calibrated slope detector + Option B slope fingerprint
    qber_simulator      — QBER time series generation (idealized + realistic noise)
    spectral_analysis   — Spectral statistics (⟨r⟩ ratio, regime classification)
    hamiltonian         — Dense Heisenberg chain Hamiltonian (QuTiP, N ≤ 8)
    sparse_hamiltonian  — Sparse Hamiltonian for finite-size scaling (N > 8)
    pulsar_analysis     — Partial F-test framework for pulsar timing validation
    quantum_eve         — Quantum Eve simulation (reduced autocorrelation, 4 strategies)
    krylov_bridge       — Krylov-Statistical Bridge (b_n deviation ↔ QBER statistics)
    demo_framework      — OOP wrappers (SiderealFilter, KrylovEngine, scenarios)

References:
    [1] D. Süß, "Deconvolution of Sidereal and Diurnal Periodicities in QKD,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18701222
    [2] D. Süß, "Dual-Layer Sidereal Detection Framework v2,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18768750
    [3] D. Süß, "Real-Data Validation on NANOGrav 15-Year Dataset,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18792775
    [4] D. Süß, "Scrambling vs. Recurrence: Microscopic Origin of the Quantum Arrow of Time,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18813710
    [5] D. Süß, "QKD Eve Detector: A Unified Framework — Parts I–III,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18873824
    [6] D. Süß, "Quantum Scrambling as a Cryptographic Resource,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18889224

Author: Daniel Süß
License: MIT
"""

__version__ = "1.2.0"
__author__ = "Daniel Süß"

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

# Krylov-Statistical Bridge
from .krylov_bridge import (
    bn_deviation, sim_stats, gamma_sweep,
    sensitivity_vs_gamma, krylov_proxy
)

# Demo Framework (OOP wrappers)
from .demo_framework import (
    SiderealFilter, KrylovEngine,
    classify_window, make_scenario
)

__all__ = [
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
]
