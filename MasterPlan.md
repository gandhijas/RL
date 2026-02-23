# PsiRL Project Master Plan

## Final Goal (March 25)
Produce a research-quality paper with:
- RL-guided reconstruction using HEA
- Baseline comparisons
- Scaling attempt
- Clean plots and analysis

---

## Experiments

### Exp00 – Sampling Convergence (DONE)
Goal: Verify empirical distribution converges with N.

### Exp01 – 1-Qubit Reconstruction
Goal: Match unknown 1-qubit state via parameterized circuit.
Metric: Fidelity.

### Exp02 – 2-Qubit Reconstruction (Main Result)
Goal: RL vs Random vs ES comparison.
Metric: Fidelity vs shots.

### Exp03 – Shot Budget Stress Test
Goal: Compare methods under fixed measurement cost.

### Exp04 – Generalization
Goal: Train policy, test on unseen states.

### Exp05 – 3-Qubit Scaling Attempt
Goal: Observe performance degradation / scaling limits.

---

## Metrics
- Fidelity
- Infidelity (1 - F)
- L1 distance (for distribution experiments)
- Time-to-threshold
- Shots-to-threshold