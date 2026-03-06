"""
Example: Full Three-Layer Detection Pipeline
=============================================

Demonstrates the complete eavesdropper detection workflow:
    Layer 1: Sidereal Filter  →  remove environmental drift
    Layer 2: Lanczos Extractor →  compute Krylov template
    Layer 3: Template Detector →  score QBER against template

Reproduces the results from:
    Paper [5]: QKD Eve Detector Parts I-III (DOI: 10.5281/zenodo.18873824)
    Paper [6]: Quantum Scrambling as Cryptographic Resource (DOI: 10.5281/zenodo.18889224)

Author: Daniel Süß
"""

import numpy as np
import matplotlib.pyplot as plt

from qkd_krylov_detector import (
    build_hamiltonian,
    build_hamiltonian_with_eve,
    compute_lanczos,
    get_theoretical_autocorrelation,
    sidereal_filter,
    make_clean_qber,
    make_eve_qber,
    make_realistic_clean_qber,
    make_realistic_eve_qber,
    krylov_dynamic_detector,
    compute_roc,
    compute_auc,
    compute_r_ratio,
    classify_regime,
)
from qkd_krylov_detector.template_detector import compute_separation


def main():
    print("=" * 70)
    print("QKD Krylov Detector — Full Pipeline Demo")
    print("=" * 70)

    # ── Step 1: Build Hamiltonian ────────────────────────────────────────
    print("\n[1] Building 8-qubit Heisenberg chain Hamiltonian...")
    H = build_hamiltonian()
    print(f"    Hilbert space dimension: {H.shape[0]}")

    r = compute_r_ratio(H)
    regime = classify_regime(r)
    print(f"    ⟨r⟩ = {r:.4f}  →  {regime}")

    # ── Step 2: Compute Lanczos coefficients ─────────────────────────────
    print("\n[2] Computing Lanczos coefficients (Krylov basis)...")
    b_n = compute_lanczos(H, n_steps=25)
    slope = np.mean(np.diff(b_n))
    print(f"    Number of b_n: {len(b_n)}")
    print(f"    Average slope: {slope:.4f}")
    print(f"    Gaussian decay timescale: τ = {1/slope:.4f}")

    # ── Step 3: Simulate QBER (idealized noise) ─────────────────────────
    print("\n[3] Simulating QBER time series (idealized noise)...")
    t = np.linspace(0, 400, 400)

    n_trials = 50
    scores_clean_all = []
    scores_eve_all = []

    for trial in range(n_trials):
        clean = make_clean_qber(t, seed=trial)
        eve = make_eve_qber(t, eve_type="iid", gamma=0.3, seed=trial + 1000)

        # Layer 1: Sidereal filter
        res_clean = sidereal_filter(clean, t)
        res_eve = sidereal_filter(eve, t)

        # Layer 3: Template matching
        _, sc = krylov_dynamic_detector(res_clean, b_n, t)
        _, se = krylov_dynamic_detector(res_eve, b_n, t)

        scores_clean_all.extend(sc)
        scores_eve_all.extend(se)

    scores_clean_all = np.array(scores_clean_all)
    scores_eve_all = np.array(scores_eve_all)

    # ROC analysis
    fpr, tpr = compute_roc(scores_clean_all, scores_eve_all)
    auc = compute_auc(fpr, tpr)
    sep = compute_separation(scores_clean_all, scores_eve_all)

    print(f"    Idealized noise results:")
    print(f"    AUC = {auc:.4f}")
    print(f"    Separation = {sep:.2f}σ")

    # ── Step 4: Realistic noise model ────────────────────────────────────
    print("\n[4] Simulating QBER (realistic hardware noise)...")
    scores_clean_r = []
    scores_eve_r = []

    for trial in range(n_trials):
        clean_r = make_realistic_clean_qber(t, seed=trial)
        eve_r = make_realistic_eve_qber(t, seed=trial + 2000)

        res_clean_r = sidereal_filter(clean_r, t)
        res_eve_r = sidereal_filter(eve_r, t)

        _, sc_r = krylov_dynamic_detector(res_clean_r, b_n, t)
        _, se_r = krylov_dynamic_detector(res_eve_r, b_n, t)

        scores_clean_r.extend(sc_r)
        scores_eve_r.extend(se_r)

    scores_clean_r = np.array(scores_clean_r)
    scores_eve_r = np.array(scores_eve_r)

    fpr_r, tpr_r = compute_roc(scores_clean_r, scores_eve_r)
    auc_r = compute_auc(fpr_r, tpr_r)
    sep_r = compute_separation(scores_clean_r, scores_eve_r)

    print(f"    Realistic noise results:")
    print(f"    AUC = {auc_r:.4f}")
    print(f"    Separation = {sep_r:.2f}σ")

    # ── Step 5: Plot results ─────────────────────────────────────────────
    print("\n[5] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Lanczos coefficients
    ax = axes[0, 0]
    ax.plot(b_n, "o-", color="#2563eb", markersize=4)
    ax.set_xlabel("n")
    ax.set_ylabel("$b_n$")
    ax.set_title("Lanczos Coefficients (Linear Growth)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Gaussian template
    ax = axes[0, 1]
    t_template = np.linspace(0, 5, 200)
    ac = get_theoretical_autocorrelation(b_n, t_template)
    ax.plot(t_template, ac, color="#dc2626", linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("C(t)")
    ax.set_title("Theoretical Autocorrelation (Gaussian Template)")
    ax.grid(True, alpha=0.3)

    # Plot 3: Score distributions
    ax = axes[1, 0]
    ax.hist(scores_clean_all, bins=30, alpha=0.6, label="Clean", color="#22c55e", density=True)
    ax.hist(scores_eve_all, bins=30, alpha=0.6, label="Eve (iid)", color="#ef4444", density=True)
    ax.set_xlabel("Detection Score (RMSE)")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distributions (Idealized, AUC={auc:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: ROC curves
    ax = axes[1, 1]
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"Idealized (AUC={auc:.4f})")
    ax.plot(fpr_r, tpr_r, color="#f59e0b", linewidth=2, label=f"Realistic (AUC={auc_r:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pipeline_results.png", dpi=150)
    print("    Saved: pipeline_results.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  System: {H.shape[0]}-dim Hilbert space, ⟨r⟩ = {r:.4f}")
    print(f"  Lanczos: {len(b_n)} coefficients, slope = {slope:.4f}")
    print(f"  Idealized:  AUC = {auc:.4f}, separation = {sep:.2f}σ")
    print(f"  Realistic:  AUC = {auc_r:.4f}, separation = {sep_r:.2f}σ")
    print("=" * 70)


if __name__ == "__main__":
    main()
