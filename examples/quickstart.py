"""
Quickstart: Minimal Eve Detection in 20 Lines
==============================================

The simplest possible demonstration of the three-layer detector.

Author: Daniel Süß
"""

import numpy as np
from qkd_krylov_detector import (
    build_hamiltonian, compute_lanczos, sidereal_filter,
    make_clean_qber, make_eve_qber, krylov_dynamic_detector,
    compute_roc, compute_auc,
)

# Build Hamiltonian and extract Krylov template
H = build_hamiltonian()
b_n = compute_lanczos(H)

# Simulate clean and Eve-compromised channels
t = np.linspace(0, 400, 400)
clean = sidereal_filter(make_clean_qber(t, seed=0), t)
eve = sidereal_filter(make_eve_qber(t, eve_type="iid", seed=1), t)

# Run detector
_, scores_clean = krylov_dynamic_detector(clean, b_n, t)
_, scores_eve = krylov_dynamic_detector(eve, b_n, t)

# Evaluate
fpr, tpr = compute_roc(scores_clean, scores_eve)
auc = compute_auc(fpr, tpr)
print(f"AUC = {auc:.4f}")
