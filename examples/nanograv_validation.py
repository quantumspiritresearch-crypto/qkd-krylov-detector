"""
Example: NANOGrav 15-Year Dataset Validation
=============================================

Demonstrates how the sidereal filter (Layer 1) was validated on real
astrophysical data from the NANOGrav 15-year pulsar timing dataset.

Key result from Paper [3]:
    - PSR J1713+0747: 59,389 TOAs
    - Sidereal amplitude: 0.0 ± 0.1 μs (consistent with null)
    - F-test p-value: 0.87 (no significant sidereal signal)
    - This validates the filter's ability to distinguish real sidereal
      signals from noise.

Reference:
    Paper [3]: Real-Data Validation on NANOGrav 15-Year Dataset
               (DOI: 10.5281/zenodo.18792775)
    pulsar_sidereal_colab-1.ipynb: Full NANOGrav analysis pipeline

Note:
    This example uses synthetic data that mimics the NANOGrav structure.
    For the actual analysis, install the `pulsar` extras:
        pip install qkd-krylov-detector[pulsar]
    and use PINT to load the real NANOGrav data files.

Author: Daniel Süß
"""

import numpy as np
import matplotlib.pyplot as plt

from qkd_krylov_detector.sidereal_filter import sidereal_filter_irregular


def simulate_nanograv_like_data(n_toas=5000, seed=42):
    """
    Generate synthetic pulsar timing data mimicking NANOGrav structure.

    Parameters
    ----------
    n_toas : int
        Number of Times of Arrival.
    seed : int
        Random seed.

    Returns
    -------
    mjd : numpy.ndarray
        Modified Julian Dates.
    residuals : numpy.ndarray
        Post-fit timing residuals in microseconds.
    """
    rng = np.random.default_rng(seed)

    # Simulate ~15 years of observations (MJD 53000 to 58500)
    mjd = np.sort(rng.uniform(53000, 58500, n_toas))

    # Post-fit residuals: white noise + small red noise
    white_noise = rng.normal(0, 0.5, n_toas)  # 0.5 μs RMS

    # Red noise (power-law spectrum)
    freqs = np.fft.rfftfreq(n_toas)
    freqs[0] = 1e-10
    red_spectrum = 0.1 / freqs**0.5
    red_spectrum[0] = 0
    phases = np.exp(1j * 2 * np.pi * rng.random(len(freqs)))
    red_noise = np.fft.irfft(red_spectrum * phases, n=n_toas)
    red_noise *= 0.1 / red_noise.std()

    residuals = white_noise + red_noise

    return mjd, residuals


def main():
    print("=" * 70)
    print("NANOGrav Validation — Sidereal Filter on Pulsar Timing Data")
    print("=" * 70)

    # ── Step 1: Generate synthetic NANOGrav-like data ────────────────────
    print("\n[1] Generating synthetic pulsar timing data...")
    mjd, residuals = simulate_nanograv_like_data(n_toas=5000)
    print(f"    Number of TOAs: {len(mjd)}")
    print(f"    MJD range: {mjd.min():.1f} – {mjd.max():.1f}")
    print(f"    Residual RMS: {residuals.std():.3f} μs")

    # ── Step 2: Apply sidereal filter (irregular sampling) ───────────────
    print("\n[2] Applying sidereal filter (least-squares method)...")
    filtered, fit_results = sidereal_filter_irregular(mjd, residuals)

    print(f"    Filtered RMS: {filtered.std():.3f} μs")
    print(f"    Fit results:")
    for key, val in fit_results.items():
        print(f"      {key}: amplitude = {val['amplitude']:.4f} μs, "
              f"phase = {val['phase']:.3f} rad")

    # ── Step 3: F-test for sidereal significance ─────────────────────────
    print("\n[3] Computing F-test for sidereal signal significance...")

    rss_full = np.sum(filtered**2)
    rss_reduced = np.sum(residuals**2)
    n = len(residuals)
    p_full = 5  # constant + 2 sin/cos pairs
    p_reduced = 1  # constant only

    f_stat = ((rss_reduced - rss_full) / (p_full - p_reduced)) / (rss_full / (n - p_full))
    # Approximate p-value from F-distribution
    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(f_stat, p_full - p_reduced, n - p_full)

    print(f"    F-statistic: {f_stat:.4f}")
    print(f"    p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("    → No significant sidereal signal (consistent with Paper [3])")
    else:
        print("    → Significant sidereal signal detected")

    # ── Step 4: Plot results ─────────────────────────────────────────────
    print("\n[4] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Raw residuals
    ax = axes[0, 0]
    ax.scatter(mjd, residuals, s=1, alpha=0.3, color="#2563eb")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (μs)")
    ax.set_title("Raw Post-Fit Timing Residuals")
    ax.grid(True, alpha=0.3)

    # Plot 2: Filtered residuals
    ax = axes[0, 1]
    ax.scatter(mjd, filtered, s=1, alpha=0.3, color="#22c55e")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Filtered Residual (μs)")
    ax.set_title("After Sidereal Filter")
    ax.grid(True, alpha=0.3)

    # Plot 3: LST phase plot
    ax = axes[1, 0]
    # Convert MJD to Local Sidereal Time (approximate)
    lst_hours = (mjd % 0.99720) * 24 / 0.99720  # Sidereal day in hours
    ax.scatter(lst_hours, residuals, s=1, alpha=0.2, color="#f59e0b")
    ax.set_xlabel("Local Sidereal Time (hours)")
    ax.set_ylabel("Residual (μs)")
    ax.set_title("Residuals vs. LST (Sidereal Phase)")
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)

    # Plot 4: Histogram comparison
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, alpha=0.5, label="Raw", color="#ef4444", density=True)
    ax.hist(filtered, bins=50, alpha=0.5, label="Filtered", color="#22c55e", density=True)
    ax.set_xlabel("Residual (μs)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution: Raw vs. Filtered")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("NANOGrav Validation — Sidereal Filter", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("nanograv_validation.png", dpi=150)
    print("    Saved: nanograv_validation.png")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary (cf. Paper [3], Table 1)")
    print("=" * 70)
    print(f"  TOAs: {len(mjd)}")
    print(f"  Raw RMS: {residuals.std():.3f} μs")
    print(f"  Filtered RMS: {filtered.std():.3f} μs")
    for key, val in fit_results.items():
        print(f"  {key}: {val['amplitude']:.4f} μs")
    print(f"  F-test p-value: {p_value:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
