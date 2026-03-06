"""
BB84 Protocol Simulation
========================

Simulates the BB84 quantum key distribution protocol at the qubit level,
including clean channels and various eavesdropper attack models.

Attack types:
    - Clean: No eavesdropper, channel noise only (~1% QBER)
    - Intercept-Resend (IR): Eve measures and resends; QBER += p * 0.25
    - Beam-Splitting (BS): Eve taps photons; transmission drops by (1-p)
    - Partial Intercept: IR on fraction p of qubits

Notebook correspondence:
    bb84_eve_classification.ipynb — Cells 1–3
    bb84_process_level.ipynb — Cells 1–2

Paper reference:
    [5] D. Süß, "QKD Eve Detector: A Unified Framework — Parts I–III,"
        Zenodo, 2026. DOI: 10.5281/zenodo.18873824, Part I

Author: Daniel Süß
License: MIT
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Core BB84 protocol simulations
# ---------------------------------------------------------------------------

def bb84_clean(n_qubits: int = 10000,
               channel_noise: float = 0.01,
               rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, int]:
    """Simulate a clean BB84 exchange — no eavesdropper.

    Parameters
    ----------
    n_qubits : int
        Number of qubits exchanged.
    channel_noise : float
        Probability of a channel-induced bit flip (default 1%).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    errors : ndarray of bool
        Error pattern on sifted key bits.
    n_sifted : int
        Number of sifted (matching-basis) bits.
    """
    if rng is None:
        rng = np.random.default_rng()

    alice_bits = rng.integers(0, 2, n_qubits)
    alice_bases = rng.integers(0, 2, n_qubits)  # 0=Z, 1=X
    bob_bases = rng.integers(0, 2, n_qubits)

    bob_bits = alice_bits.copy()
    noise_mask = rng.random(n_qubits) < channel_noise
    bob_bits[noise_mask] ^= 1

    sifted = alice_bases == bob_bases
    errors = alice_bits[sifted] != bob_bits[sifted]
    return errors, int(sifted.sum())


def bb84_intercept_resend(n_qubits: int = 10000,
                          p_intercept: float = 1.0,
                          channel_noise: float = 0.01,
                          rng: Optional[np.random.Generator] = None
                          ) -> Tuple[np.ndarray, int, int]:
    """Simulate BB84 with an intercept-resend eavesdropper.

    Eve intercepts a fraction *p_intercept* of qubits, measures in a random
    basis, and resends.  Wrong-basis measurement causes 50% error on resend,
    yielding an expected QBER contribution of ``p_intercept * 0.25``.

    Parameters
    ----------
    n_qubits : int
        Number of qubits exchanged.
    p_intercept : float
        Fraction of qubits Eve intercepts (0–1).
    channel_noise : float
        Background channel noise probability.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    errors : ndarray of bool
        Error pattern on sifted key bits.
    n_sifted : int
        Number of sifted bits.
    n_intercepted : int
        Number of intercepted sifted bits.
    """
    if rng is None:
        rng = np.random.default_rng()

    alice_bits = rng.integers(0, 2, n_qubits)
    alice_bases = rng.integers(0, 2, n_qubits)
    bob_bases = rng.integers(0, 2, n_qubits)

    bob_bits = alice_bits.copy()
    noise_mask = rng.random(n_qubits) < channel_noise
    bob_bits[noise_mask] ^= 1

    # Eve intercepts
    eve_intercepts = rng.random(n_qubits) < p_intercept
    eve_bases = rng.integers(0, 2, n_qubits)
    eve_wrong_basis = eve_intercepts & (eve_bases != alice_bases)
    eve_errors = eve_wrong_basis & (rng.random(n_qubits) < 0.5)
    bob_bits[eve_errors] ^= 1

    sifted = alice_bases == bob_bases
    errors = alice_bits[sifted] != bob_bits[sifted]
    return errors, int(sifted.sum()), int(eve_intercepts[sifted].sum())


def bb84_beam_splitting(n_qubits: int = 10000,
                        p_split: float = 0.3,
                        channel_noise: float = 0.01,
                        rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, int, float]:
    """Simulate BB84 with a beam-splitting eavesdropper.

    Eve silently taps a fraction *p_split* of photons.  Those photons never
    reach Bob, causing a transmission drop but no direct QBER increase.

    Parameters
    ----------
    n_qubits : int
        Number of qubits exchanged.
    p_split : float
        Fraction of photons Eve splits off (0–1).
    channel_noise : float
        Background channel noise probability.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    errors : ndarray of bool
        Error pattern on sifted key bits.
    n_sifted : int
        Number of sifted bits.
    transmission : float
        Fraction of photons that reached Bob.
    """
    if rng is None:
        rng = np.random.default_rng()

    alice_bits = rng.integers(0, 2, n_qubits)
    alice_bases = rng.integers(0, 2, n_qubits)
    bob_bases = rng.integers(0, 2, n_qubits)

    photon_reaches_bob = rng.random(n_qubits) > p_split

    bob_bits = alice_bits.copy()
    noise_mask = rng.random(n_qubits) < channel_noise
    bob_bits[noise_mask] ^= 1

    sifted = (alice_bases == bob_bases) & photon_reaches_bob
    errors = alice_bits[sifted] != bob_bits[sifted]
    transmission = float(photon_reaches_bob.mean())
    return errors, int(sifted.sum()), transmission


# ---------------------------------------------------------------------------
# BB84 window-level simulation (for time series generation)
# ---------------------------------------------------------------------------

def bb84_window(n_qubits: int = 500,
                p_ir: float = 0.0,
                p_bs: float = 0.0,
                channel_noise: float = 0.01,
                rng: Optional[np.random.Generator] = None
                ) -> Tuple[float, float]:
    """Simulate a single BB84 measurement window.

    Combines intercept-resend and beam-splitting attacks in one call.

    Parameters
    ----------
    n_qubits : int
        Qubits per window.
    p_ir : float
        Intercept-resend probability (0 = no IR attack).
    p_bs : float
        Beam-splitting probability (0 = no BS attack).
    channel_noise : float
        Background channel noise.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    qber : float
        Quantum bit error rate for this window.
    transmission : float
        Photon transmission rate.
    """
    if rng is None:
        rng = np.random.default_rng()

    alice_bits = rng.integers(0, 2, n_qubits)
    alice_bases = rng.integers(0, 2, n_qubits)
    bob_bases = rng.integers(0, 2, n_qubits)
    bob_bits = alice_bits.copy()

    # Channel noise
    noise_mask = rng.random(n_qubits) < channel_noise
    bob_bits[noise_mask] ^= 1

    # Intercept-resend
    if p_ir > 0:
        eve_mask = rng.random(n_qubits) < p_ir
        eve_bases = rng.integers(0, 2, n_qubits)
        eve_wrong = eve_mask & (eve_bases != alice_bases)
        eve_err = eve_wrong & (rng.random(n_qubits) < 0.5)
        bob_bits[eve_err] ^= 1

    # Beam-splitting
    photon_reaches = np.ones(n_qubits, dtype=bool)
    if p_bs > 0:
        photon_reaches = rng.random(n_qubits) > p_bs

    sifted = (alice_bases == bob_bases) & photon_reaches
    n_sifted = sifted.sum()
    if n_sifted == 0:
        return 0.0, float(photon_reaches.mean())
    errors = alice_bits[sifted] != bob_bits[sifted]
    return float(errors.mean()), float(photon_reaches.mean())


# ---------------------------------------------------------------------------
# Time series generation from BB84 windows
# ---------------------------------------------------------------------------

def make_bb84_timeseries(n_windows: int = 300,
                         n_qubits_per_window: int = 500,
                         eve_type: str = "clean",
                         p_eve: float = 0.0,
                         eve_start: int = 100,
                         eve_end: int = 200,
                         sidereal_period: float = 23.93,
                         seed: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate QBER and transmission time series from BB84 simulation.

    Eve is active only in the window range [eve_start, eve_end) — burst
    attack model as described in Paper [5], Part I.

    Parameters
    ----------
    n_windows : int
        Number of measurement windows.
    n_qubits_per_window : int
        Qubits exchanged per window.
    eve_type : str
        One of ``"clean"``, ``"intercept_resend"``, ``"beam_splitting"``,
        ``"partial_intercept"``.
    p_eve : float
        Attack strength parameter.
    eve_start, eve_end : int
        Window indices where Eve is active.
    sidereal_period : float
        Sidereal day in hours (default 23.93).
    seed : int, optional
        Random seed.

    Returns
    -------
    t : ndarray
        Time axis (hours).
    qber : ndarray
        QBER time series.
    transmission : ndarray
        Photon transmission time series.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_windows, n_windows)

    # Environmental drift
    env = (0.04 * np.sin(2 * np.pi * t / sidereal_period)
           + 0.02 * np.sin(2 * np.pi * t / 24.0 + 0.5))

    qber_ts = np.zeros(n_windows)
    trans_ts = np.ones(n_windows)

    for i in range(n_windows):
        eve_active = eve_start <= i < eve_end
        p = p_eve if eve_active else 0.0

        if eve_type == "intercept_resend":
            qber_ts[i], trans_ts[i] = bb84_window(
                n_qubits_per_window, p_ir=p, rng=rng)
        elif eve_type == "beam_splitting":
            qber_ts[i], trans_ts[i] = bb84_window(
                n_qubits_per_window, p_bs=p, rng=rng)
        elif eve_type == "partial_intercept":
            qber_ts[i], trans_ts[i] = bb84_window(
                n_qubits_per_window, p_ir=p, rng=rng)
        else:  # clean
            qber_ts[i], trans_ts[i] = bb84_window(
                n_qubits_per_window, rng=rng)

    qber_ts += env
    return t, qber_ts, trans_ts
