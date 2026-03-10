#!/usr/bin/env python3
"""
Loschmidt Echo Example — Reproduces results from Paper [10].

Computes state-space and operator-space Loschmidt echoes for the 8-qubit
Heisenberg chain under eavesdropper perturbation, and demonstrates the
correlation between echo decay and Krylov detection score.

Reference:
    D. Süß, "The Krylov Eavesdropper Detector as an Operator-Space Loschmidt
    Echo," Zenodo (2026). DOI: 10.5281/zenodo.18939996

Usage:
    python examples/loschmidt_echo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from qkd_krylov_detector import build_hamiltonian
from qkd_krylov_detector.loschmidt_echo import (
    eigendecompose,
    compute_state_echo,
    compute_operator_echo,
    compute_echo_decay_rate,
    compute_operator_autocorrelation,
    loschmidt_krylov_correlation,
)

# --- Parameters ---
N_QUBITS = 8
T_MAX = 50
N_STEPS = 200
GAMMAS = [0.01, 0.02, 0.05, 0.1, 0.2]  # 5 coupling strengths for quick demo

# --- Build system ---
print("Building 8-qubit Heisenberg chain...")
H = build_hamiltonian(n_qubits=N_QUBITS)
H_dense = H.full() if hasattr(H, 'full') else np.array(H)
dim = H_dense.shape[0]

# Eve's coupling operator: sigma_x(0) x sigma_x(1)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def kron_op(op, site, n):
    result = np.eye(1, dtype=complex)
    for i in range(n):
        result = np.kron(result, op if i == site else I2)
    return result

V_eve = kron_op(sx, 0, N_QUBITS) @ kron_op(sx, 1, N_QUBITS)
O0 = kron_op(np.array([[1,0],[0,-1]], dtype=complex), 0, N_QUBITS)  # sigma_z(0)
psi0 = np.zeros(dim, dtype=complex)
psi0[0] = 1.0

# --- Eigendecomposition ---
print("Eigendecomposition...")
E_clean, V_clean = eigendecompose(H_dense)
times = np.linspace(0, T_MAX, N_STEPS)

# --- Demo 1: State echo vs. Operator echo ---
print("\n=== State Echo vs. Operator Echo (gamma = 0.1) ===")
gamma_demo = 0.1
H_eve = H_dense + gamma_demo * V_eve
E_eve, V_eve_eig = eigendecompose(H_eve)

demo_times = np.linspace(0, T_MAX, 50)
M_state = [compute_state_echo(E_clean, V_clean, E_eve, V_eve_eig, psi0, t)
           for t in demo_times]
M_op = [compute_operator_echo(E_clean, V_clean, E_eve, V_eve_eig, O0, t, dim)
        for t in demo_times]

print(f"  State echo at t={T_MAX}:    M = {M_state[-1]:.6f}")
print(f"  Operator echo at t={T_MAX}: M = {M_op[-1]:.6f}")
print(f"  Operator echo decays faster → more sensitive to perturbation")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(demo_times, M_state, 'b-', linewidth=2, label='State echo $M(t)$')
ax.plot(demo_times, M_op, 'r--', linewidth=2, label='Operator echo $M_{op}(t)$')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('Echo amplitude', fontsize=12)
ax.set_title(f'Loschmidt Echo Comparison ($\\gamma = {gamma_demo}$)', fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loschmidt_echo_comparison.png', dpi=150)
print("  Saved: loschmidt_echo_comparison.png")

# --- Demo 2: Full correlation analysis ---
print(f"\n=== Loschmidt-Krylov Correlation (N_gamma = {len(GAMMAS)}) ===")
results = loschmidt_krylov_correlation(
    H_dense, V_eve, O0, GAMMAS,
    t_max=T_MAX, n_time_steps=N_STEPS, n_lanczos=25,
)

corr = results['correlations']
print(f"  gamma^2 vs. Lanczos RMSE:    r = {corr['gamma2_vs_lanczos']['r']:.4f}, "
      f"p = {corr['gamma2_vs_lanczos']['p']:.2e}")
print(f"  Echo decay vs. Lanczos RMSE: r = {corr['echo_vs_lanczos']['r']:.4f}, "
      f"p = {corr['echo_vs_lanczos']['p']:.2e}")
print(f"  Krylov score vs. Echo decay: r = {corr['krylov_vs_echo']['r']:.4f}, "
      f"p = {corr['krylov_vs_echo']['p']:.2e}")

# Correlation plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
gammas_sq = np.array(GAMMAS) ** 2

axes[0].scatter(gammas_sq, results['lanczos_rmse'], c='steelblue', s=60, zorder=3)
axes[0].set_xlabel('$\\gamma^2$')
axes[0].set_ylabel('Lanczos RMSE')
axes[0].set_title(f"$r = {corr['gamma2_vs_lanczos']['r']:.3f}$")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(results['echo_decay_rate'], results['lanczos_rmse'],
                c='firebrick', s=60, zorder=3)
axes[1].set_xlabel('Echo Decay Rate')
axes[1].set_ylabel('Lanczos RMSE')
axes[1].set_title(f"$r = {corr['echo_vs_lanczos']['r']:.3f}$")
axes[1].grid(True, alpha=0.3)

axes[2].scatter(results['krylov_score'], results['echo_decay_rate'],
                c='forestgreen', s=60, zorder=3)
axes[2].set_xlabel('Krylov Score')
axes[2].set_ylabel('Echo Decay Rate')
axes[2].set_title(f"$r = {corr['krylov_vs_echo']['r']:.3f}$")
axes[2].grid(True, alpha=0.3)

plt.suptitle('Loschmidt-Krylov Correlations', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('loschmidt_krylov_correlations.png', dpi=150, bbox_inches='tight')
print("  Saved: loschmidt_krylov_correlations.png")

print("\nDone!")
