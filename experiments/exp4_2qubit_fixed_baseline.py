import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp04_2qubit_fixed"
os.makedirs(run_dir, exist_ok=True)

# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """
    Single-qubit pure state:
        |psi(theta, phi)> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    """
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """State fidelity |<psi|phi>|^2."""
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def hadamard() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def s_dagger() -> np.ndarray:
    return np.array([[1, 0], [0, -1j]], dtype=complex)

# ========================
# Measurement probabilities
# ========================
def measure_probs_z(psi: np.ndarray) -> np.ndarray:
    p = np.abs(psi) ** 2
    return p / np.sum(p)


def measure_probs_x(psi: np.ndarray) -> np.ndarray:
    psi_x = hadamard() @ psi
    p = np.abs(psi_x) ** 2
    return p / np.sum(p)


def measure_probs_y(psi: np.ndarray) -> np.ndarray:
    psi_y = hadamard() @ (s_dagger() @ psi)
    p = np.abs(psi_y) ** 2
    return p / np.sum(p)


def sample_one_shot(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice([0, 1], p=probs))


# ========================
# Bloch estimation
# ========================
def estimate_bloch_from_counts(z_counts, x_counts, y_counts):
    z_hat = x_hat = y_hat = 0.0
    if np.sum(z_counts) > 0:
        z_probs = z_counts / np.sum(z_counts)
        z_hat = float(z_probs[0] - z_probs[1])
    if np.sum(x_counts) > 0:
        x_probs = x_counts / np.sum(x_counts)
        x_hat = float(x_probs[0] - x_probs[1])
    if np.sum(y_counts) > 0:
        y_probs = y_counts / np.sum(y_counts)
        y_hat = float(y_probs[0] - y_probs[1])
    return x_hat, y_hat, z_hat


def bloch_to_state(x, y, z):
    r = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(r)
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=complex)
    x_n, y_n, z_n = r / norm
    z_n = float(np.clip(z_n, -1.0, 1.0))
    theta_hat = float(np.arccos(z_n))
    phi_hat = float(np.mod(np.arctan2(y_n, x_n), 2 * np.pi))
    return state_from_angles(theta_hat, phi_hat)


def estimate_state_from_counts(z_counts, x_counts, y_counts):
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)
    return bloch_to_state(x_hat, y_hat, z_hat)


# ========================
# Fixed strategy episode (2 qubits)
# ========================
ACTIONS = ["Z", "X", "Y"]


def run_fixed_strategy_episode(psi1, psi2, total_shots, strategy, rng):
    z1_counts = np.array([0, 0], dtype=int)
    x1_counts = np.array([0, 0], dtype=int)
    y1_counts = np.array([0, 0], dtype=int)
    z2_counts = np.array([0, 0], dtype=int)
    x2_counts = np.array([0, 0], dtype=int)
    y2_counts = np.array([0, 0], dtype=int)

    if strategy == "Z_only":
        for _ in range(total_shots):
            z1_counts[sample_one_shot(measure_probs_z(psi1), rng)] += 1
            z2_counts[sample_one_shot(measure_probs_z(psi2), rng)] += 1

    elif strategy == "ZX_split":
        n_z = total_shots // 2
        n_x = total_shots - n_z
        for _ in range(n_z):
            z1_counts[sample_one_shot(measure_probs_z(psi1), rng)] += 1
            z2_counts[sample_one_shot(measure_probs_z(psi2), rng)] += 1
        for _ in range(n_x):
            x1_counts[sample_one_shot(measure_probs_x(psi1), rng)] += 1
            x2_counts[sample_one_shot(measure_probs_x(psi2), rng)] += 1

    elif strategy == "XYZ_split":
        n_z = total_shots // 3
        n_x = total_shots // 3
        n_y = total_shots - n_z - n_x
        for _ in range(n_z):
            z1_counts[sample_one_shot(measure_probs_z(psi1), rng)] += 1
            z2_counts[sample_one_shot(measure_probs_z(psi2), rng)] += 1
        for _ in range(n_x):
            x1_counts[sample_one_shot(measure_probs_x(psi1), rng)] += 1
            x2_counts[sample_one_shot(measure_probs_x(psi2), rng)] += 1
        for _ in range(n_y):
            y1_counts[sample_one_shot(measure_probs_y(psi1), rng)] += 1
            y2_counts[sample_one_shot(measure_probs_y(psi2), rng)] += 1

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    psi1_hat = estimate_state_from_counts(z1_counts, x1_counts, y1_counts)
    psi2_hat = estimate_state_from_counts(z2_counts, x2_counts, y2_counts)
    F_total = fidelity(psi1_hat, psi1) * fidelity(psi2_hat, psi2)

    return {
        "fidelity": F_total,
        "infidelity": 1 - F_total,
        "z1_counts_0": int(z1_counts[0]),
        "z1_counts_1": int(z1_counts[1]),
        "x1_counts_0": int(x1_counts[0]),
        "x1_counts_1": int(x1_counts[1]),
        "y1_counts_0": int(y1_counts[0]),
        "y1_counts_1": int(y1_counts[1]),
        "z2_counts_0": int(z2_counts[0]),
        "z2_counts_1": int(z2_counts[1]),
        "x2_counts_0": int(x2_counts[0]),
        "x2_counts_1": int(x2_counts[1]),
        "y2_counts_0": int(y2_counts[0]),
        "y2_counts_1": int(y2_counts[1]),
    }

# ========================
# Experiment configuration
# ========================
shot_budgets = [10, 25, 50, 100, 250, 500]
num_test_targets = 50
num_test_seeds = 5
master_rng = np.random.default_rng(123)

# ========================
# Run experiment
# ========================
rows = []

for target_id in range(num_test_targets):
    theta1 = master_rng.uniform(0, np.pi)
    phi1 = master_rng.uniform(0, 2 * np.pi)
    theta2 = master_rng.uniform(0, np.pi)
    phi2 = master_rng.uniform(0, 2 * np.pi)

    psi1 = state_from_angles(theta1, phi1)
    psi2 = state_from_angles(theta2, phi2)

    for seed in range(num_test_seeds):
        for N in shot_budgets:
            rng = np.random.default_rng(target_id * 1000 + seed * 100 + N)

            for strategy in ["Z_only", "ZX_split", "XYZ_split"]:
                out = run_fixed_strategy_episode(psi1, psi2, N, strategy, rng)
                rows.append({
                    "target_id": target_id,
                    "seed": seed,
                    "N": N,
                    "method": strategy,
                    "true_theta1": theta1,
                    "true_phi1": phi1,
                    "true_theta2": theta2,
                    "true_phi2": phi2,
                    **out,
                })

# ========================
# Save raw metrics
# ========================
df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/metrics.csv", index=False)

# ========================
# Aggregate summary
# ========================
agg = (
    df.groupby(["method", "N"])
      .agg(
          fidelity_mean=("fidelity", "mean"),
          fidelity_std=("fidelity", "std"),
          infidelity_mean=("infidelity", "mean"),
          infidelity_std=("infidelity", "std"),
      )
      .reset_index()
)

agg.to_csv(f"{run_dir}/summary.csv", index=False)

# ========================
# Plot helper
# ========================
def plot_metric(metric_mean, metric_std, ylabel, title, filename, logy=False):
    plt.figure()
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("N")
        x = sub["N"].to_numpy()
        y = sub[metric_mean].to_numpy()
        ystd = sub[metric_std].to_numpy()
        plt.plot(x, y, marker="o", label=method)
        plt.fill_between(x, y - ystd, y + ystd, alpha=0.2)
    plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel("Shots (N)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/{filename}", dpi=200)
    plt.close()

# ========================
# Make plots
# ========================
plot_metric("fidelity_mean", "fidelity_std", "Mean Fidelity",
            "Exp04: 2-Qubit Fixed Strategy (Fidelity vs Shots)", "fidelity_vs_shots.png")
plot_metric("infidelity_mean", "infidelity_std", "Mean Infidelity",
            "Exp04: 2-Qubit Fixed Strategy (Infidelity vs Shots)", "infidelity_vs_shots.png", logy=True)

# ========================
# Notes file
# ========================
with open(f"{run_dir}/notes.txt", "w") as f:
    f.write("Experiment: Exp04 - 2-qubit fixed strategies\n")
    f.write("State family: two independent pure qubits\n")
    f.write("Methods: Z_only, ZX_split, XYZ_split\n")
    f.write(f"Shot budgets: {shot_budgets}\n")
    f.write(f"Test targets: {num_test_targets}\n")
    f.write(f"Test seeds: {num_test_seeds}\n")

print("Saved results to:", run_dir)
print("Files created:")
print("- metrics.csv")
print("- summary.csv")
print("- fidelity_vs_shots.png")
print("- infidelity_vs_shots.png")
print("- notes.txt")