import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Run directory
# ------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp01_fixed_baseline"
os.makedirs(run_dir, exist_ok=True)

# ------------------------
# Quantum state definitions
# ------------------------
def ry(theta):
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ], dtype=float)

def hadamard():
    return (1 / np.sqrt(2)) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=float)

def state_from_theta(theta):
    zero = np.array([1, 0], dtype=float)
    return ry(theta) @ zero

def fidelity(psi, phi):
    overlap = np.vdot(psi, phi)
    return np.abs(overlap) ** 2

# ------------------------
# Measurement probabilities
# ------------------------
def measure_probs_z(psi):
    p = np.abs(psi) ** 2
    return p / np.sum(p)

def measure_probs_x(psi):
    psi_x = hadamard() @ psi
    p = np.abs(psi_x) ** 2
    return p / np.sum(p)

# ------------------------
# Sampling
# ------------------------
def sample_counts(probs, N, rng):
    samples = rng.choice([0, 1], size=N, p=probs)
    counts = np.bincount(samples, minlength=2)
    return counts

# ------------------------
# Estimation from counts
# ------------------------
def estimate_theta_from_counts(z_counts=None, x_counts=None):
    """
    Estimate theta for states of the form RY(theta)|0>.
    Uses Bloch-vector estimates:
      z_hat = P(0)-P(1) in Z basis ≈ cos(theta)
      x_hat = P(+)-P(-) in X basis ≈ sin(theta)
    """
    z_hat = None
    x_hat = None

    if z_counts is not None and np.sum(z_counts) > 0:
        z_probs = z_counts / np.sum(z_counts)
        z_hat = z_probs[0] - z_probs[1]

    if x_counts is not None and np.sum(x_counts) > 0:
        x_probs = x_counts / np.sum(x_counts)
        x_hat = x_probs[0] - x_probs[1]

    # If only Z data is available, estimate theta from cos(theta)
    if z_hat is not None and x_hat is None:
        z_hat = np.clip(z_hat, -1.0, 1.0)
        return np.arccos(z_hat)

    # If both are available, use atan2(sin, cos)
    if z_hat is not None and x_hat is not None:
        return np.mod(np.arctan2(x_hat, z_hat), 2 * np.pi)

    raise ValueError("No measurement data available to estimate theta.")

# ------------------------
# Experiment config
# ------------------------
shot_budgets = [10, 25, 50, 100, 250, 500, 1000]
num_targets = 50
num_seeds = 5

strategies = ["Z_only", "ZX_split"]

rng_master = np.random.default_rng(123)

# ------------------------
# Run experiment
# ------------------------
rows = []

for target_id in range(num_targets):
    true_theta = rng_master.uniform(0, 2 * np.pi)
    psi_true = state_from_theta(true_theta)

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed + 1000 * target_id)

        for N in shot_budgets:
            for strategy in strategies:

                if strategy == "Z_only":
                    z_probs = measure_probs_z(psi_true)
                    z_counts = sample_counts(z_probs, N, rng)
                    x_counts = None

                elif strategy == "ZX_split":
                    N_z = N // 2
                    N_x = N - N_z

                    z_probs = measure_probs_z(psi_true)
                    x_probs = measure_probs_x(psi_true)

                    z_counts = sample_counts(z_probs, N_z, rng)
                    x_counts = sample_counts(x_probs, N_x, rng)

                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                theta_hat = estimate_theta_from_counts(z_counts=z_counts, x_counts=x_counts)
                psi_hat = state_from_theta(theta_hat)

                F = fidelity(psi_true, psi_hat)
                infidelity = 1 - F
                theta_error = np.abs(true_theta - theta_hat)

                # wrap angle error into [0, pi]
                theta_error = min(theta_error, 2 * np.pi - theta_error)

                rows.append({
                    "target_id": target_id,
                    "seed": seed,
                    "strategy": strategy,
                    "N": N,
                    "true_theta": true_theta,
                    "theta_hat": theta_hat,
                    "theta_error": theta_error,
                    "fidelity": F,
                    "infidelity": infidelity
                })

# ------------------------
# Save raw data
# ------------------------
df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/metrics.csv", index=False)

# ------------------------
# Aggregate
# ------------------------
agg = (
    df.groupby(["strategy", "N"])
      .agg(
          fidelity_mean=("fidelity", "mean"),
          fidelity_std=("fidelity", "std"),
          infidelity_mean=("infidelity", "mean"),
          infidelity_std=("infidelity", "std"),
          theta_error_mean=("theta_error", "mean"),
          theta_error_std=("theta_error", "std")
      )
      .reset_index()
)

agg.to_csv(f"{run_dir}/summary.csv", index=False)

# ------------------------
# Plot helper
# ------------------------
def plot_metric(metric_mean, metric_std, ylabel, title, filename, logy=False):
    plt.figure()
    for strategy in agg["strategy"].unique():
        sub = agg[agg["strategy"] == strategy].sort_values("N")
        x = sub["N"].to_numpy()
        y = sub[metric_mean].to_numpy()
        ystd = sub[metric_std].to_numpy()

        plt.plot(x, y, marker="o", label=strategy)
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
    plt.show()

# ------------------------
# Make plots
# ------------------------
plot_metric(
    metric_mean="fidelity_mean",
    metric_std="fidelity_std",
    ylabel="Mean Fidelity",
    title="1-Qubit Fixed Baselines: Fidelity vs Shots",
    filename="fidelity_vs_shots.png",
    logy=False
)

plot_metric(
    metric_mean="infidelity_mean",
    metric_std="infidelity_std",
    ylabel="Mean Infidelity",
    title="1-Qubit Fixed Baselines: Infidelity vs Shots",
    filename="infidelity_vs_shots.png",
    logy=True
)

plot_metric(
    metric_mean="theta_error_mean",
    metric_std="theta_error_std",
    ylabel="Mean Theta Error",
    title="1-Qubit Fixed Baselines: Theta Error vs Shots",
    filename="theta_error_vs_shots.png",
    logy=True
)

# ------------------------
# Save a plain text notes file
# ------------------------
with open(f"{run_dir}/notes.txt", "w") as f:
    f.write("Experiment: 1-qubit fixed measurement baselines\n")
    f.write("Strategies compared: Z_only, ZX_split\n")
    f.write(f"Shot budgets: {shot_budgets}\n")
    f.write(f"Number of targets: {num_targets}\n")
    f.write(f"Number of seeds: {num_seeds}\n")

print("Saved results to:", run_dir)
print("Files created:")
print("- metrics.csv")
print("- summary.csv")
print("- fidelity_vs_shots.png")
print("- infidelity_vs_shots.png")
print("- theta_error_vs_shots.png")
print("- notes.txt")