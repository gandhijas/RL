import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------
# Run directory
# ------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp02"
os.makedirs(run_dir, exist_ok=True)


# ------------------------
# Quantum gates (NumPy)
# ------------------------
def ry(theta: float) -> np.ndarray:
    """Single-qubit RY rotation (2x2)."""
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2),  np.cos(theta/2)]
    ], dtype=float)


def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product."""
    return np.kron(a, b)


def cnot_01() -> np.ndarray:
    """
    CNOT with qubit 0 as control and qubit 1 as target, in basis:
    |00>, |01>, |10>, |11>
    """
    return np.array([
        [1, 0, 0, 0],  # |00> -> |00>
        [0, 1, 0, 0],  # |01> -> |01>
        [0, 0, 0, 1],  # |10> -> |11>
        [0, 0, 1, 0],  # |11> -> |10>
    ], dtype=float)


def state_from_thetas(theta1: float, theta2: float) -> np.ndarray:
    """
    Build 2-qubit state:
      |00> --RY(theta1) on q0--●
      |00> --RY(theta2) on q1--X
    Return statevector of length 4 in computational basis.
    """
    # |00>
    psi0 = np.array([1, 0, 0, 0], dtype=float)

    # Apply RY on each qubit: (RY(theta1) ⊗ RY(theta2))
    U_rot = kron(ry(theta1), ry(theta2))
    psi1 = U_rot @ psi0

    # Apply CNOT
    U_cnot = cnot_01()
    psi2 = U_cnot @ psi1

    return psi2


def probs_from_state(psi: np.ndarray) -> np.ndarray:
    """Measurement probabilities in computational basis."""
    p = np.abs(psi)**2
    return p / np.sum(p)


# ------------------------
# Sampling + estimators
# ------------------------
OUTCOMES = ["00", "01", "10", "11"]
K = 4

def sample_shots(p_true: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return counts for outcomes [00,01,10,11] from N samples.
    """
    samples = rng.choice(np.arange(K), size=N, p=p_true)
    counts = np.bincount(samples, minlength=K)
    return counts


def mle_estimate(counts: np.ndarray) -> np.ndarray:
    """Empirical frequencies."""
    N = np.sum(counts)
    return counts / N


def dirichlet_estimate(counts: np.ndarray, alpha: float) -> np.ndarray:
    """
    Symmetric Dirichlet smoothing:
      (count + alpha) / (N + K*alpha)
    """
    N = np.sum(counts)
    return (counts + alpha) / (N + K * alpha)


# ------------------------
# Metrics
# ------------------------
def l1_error(p_true: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.sum(np.abs(p_true - p_hat)))


def kl_divergence(p_true: np.ndarray, p_hat: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL(P_true || P_hat). Add eps to avoid log(0).
    Note: KL is most meaningful when p_hat has no zeros (Dirichlet helps).
    """
    p_t = np.clip(p_true, eps, 1.0)
    p_h = np.clip(p_hat, eps, 1.0)
    return float(np.sum(p_t * np.log(p_t / p_h)))


# ------------------------
# Experiment config
# ------------------------
shot_budgets = [10, 25, 50, 100, 250, 500, 1000]
num_targets = 20          # how many random (theta1, theta2) target circuits
num_seeds = 5             # repeats per target for sampling noise
alpha = 0.5               # Dirichlet smoothing strength

rng_master = np.random.default_rng(123)  # master RNG for reproducibility


# ------------------------
# Run experiment
# ------------------------
rows = []

for t in range(num_targets):
    # Random target circuit parameters
    theta1 = rng_master.uniform(0, 2*np.pi)
    theta2 = rng_master.uniform(0, 2*np.pi)

    psi_true = state_from_thetas(theta1, theta2)
    p_true = probs_from_state(psi_true)

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed + 1000 * t)

        for N in shot_budgets:
            counts = sample_shots(p_true, N, rng)

            p_mle = mle_estimate(counts)
            p_dir = dirichlet_estimate(counts, alpha=alpha)

            rows.append({
                "target_id": t,
                "seed": seed,
                "theta1": theta1,
                "theta2": theta2,
                "N": N,
                "method": "MLE",
                "l1": l1_error(p_true, p_mle),
                "kl": kl_divergence(p_true, p_mle),
            })

            rows.append({
                "target_id": t,
                "seed": seed,
                "theta1": theta1,
                "theta2": theta2,
                "N": N,
                "method": f"Dirichlet(alpha={alpha})",
                "l1": l1_error(p_true, p_dir),
                "kl": kl_divergence(p_true, p_dir),
            })


df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/metrics.csv", index=False)


# ------------------------
# Aggregate + plot
# ------------------------
agg = (
    df.groupby(["method", "N"])
      .agg(l1_mean=("l1", "mean"),
           l1_std=("l1", "std"),
           kl_mean=("kl", "mean"),
           kl_std=("kl", "std"))
      .reset_index()
)

agg.to_csv(f"{run_dir}/summary.csv", index=False)

# Plot L1 error vs shots
plt.figure()
for method in agg["method"].unique():
    sub = agg[agg["method"] == method].sort_values("N")
    x = sub["N"].to_numpy()
    y = sub["l1_mean"].to_numpy()
    ystd = sub["l1_std"].to_numpy()

    plt.plot(x, y, marker="o", label=method)
    plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Shots (N)")
plt.ylabel("L1 error")
plt.title("2-Qubit Distribution Estimation: Error vs Shots")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_dir}/l1_vs_shots.png", dpi=200)
plt.show()

print("Saved:", run_dir)
print("Files: metrics.csv, summary.csv, l1_vs_shots.png")