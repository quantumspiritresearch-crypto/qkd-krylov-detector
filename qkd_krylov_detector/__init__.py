"""
QKD Krylov Detector — Python Package
=====================================

A three-layer eavesdropper detection framework for Quantum Key Distribution
based on Krylov complexity and sidereal filtering.

Modules:
    sidereal_filter     — Layer 1: FFT-based removal of sidereal/diurnal periodicities
    lanczos_extractor   — Layer 2: Lanczos algorithm for Krylov b_n coefficients
    template_detector   — Layer 3: Gaussian template matching on QBER autocorrelation
    qber_simulator      — QBER time series generation (clean + Eve models)
    spectral_analysis   — Spectral statistics (⟨r⟩ ratio, regime classification)

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

__version__ = "1.0.0"
__author__ = "Daniel Süß"

from .sidereal_filter import sidereal_filter, sidereal_filter_irregular
from .lanczos_extractor import compute_lanczos, get_theoretical_autocorrelation
from .template_detector import krylov_dynamic_detector, compute_roc, compute_auc
from .qber_simulator import (
    make_clean_qber, make_eve_qber,
    make_realistic_clean_qber, make_realistic_eve_qber
)
from .spectral_analysis import compute_r_ratio, classify_regime
from .hamiltonian import build_hamiltonian, build_hamiltonian_with_eve, get_op

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
    # QBER Simulation
    "make_clean_qber",
    "make_eve_qber",
    "make_realistic_clean_qber",
    "make_realistic_eve_qber",
    # Spectral Analysis
    "compute_r_ratio",
    "classify_regime",
    # Hamiltonian
    "build_hamiltonian",
    "build_hamiltonian_with_eve",
    "get_op",
]
