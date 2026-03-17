import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp02_1qubitRL"
os.makedirs(run_dir, exist_ok=True)


# ========================
# Quantum utilities
# ========================
def ry(theta: float) -> np.ndarray:
    """Single-qubit RY rotation."""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ], dtype=float)


def hadamard() -> np.ndarray:
    """Hadamard gate. Used to convert X-basis measurement into Z-basis measurement after rotation."""
    return (1 / np.sqrt(2)) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=float)


def state_from_theta(theta: float) -> np.ndarray:
    """Build the 1-qubit state |psi(theta)> = RY(theta)|0>."""
    zero = np.array([1, 0], dtype=float)
    return ry(theta) @ zero


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """State fidelity |<psi|phi>|^2."""
    overlap = np.vdot(psi, phi)
    return float(np.abs(overlap) ** 2)


# ========================
# Measurement probabilities
# ========================
def measure_probs_z(psi: np.ndarray) -> np.ndarray:
    """Z-basis probabilities [P(0), P(1)]."""
    p = np.abs(psi) ** 2
    return p / np.sum(p)


def measure_probs_x(psi: np.ndarray) -> np.ndarray:
    """
    X-basis probabilities [P(+), P(-)].
    Implemented by rotating with Hadamard and then measuring in Z.
    """
    psi_x = hadamard() @ psi
    p = np.abs(psi_x) ** 2
    return p / np.sum(p)


def sample_one_shot(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one measurement outcome, returning 0 or 1."""
    return int(rng.choice([0, 1], p=probs))


# ========================
# Estimation from counts
# ========================
def estimate_theta_from_counts(z_counts=None, x_counts=None) -> float:
    """
    Estimate theta for states of the form RY(theta)|0>.

    For this state family:
      z_hat = P(0) - P(1) ≈ cos(theta)
      x_hat = P(+) - P(-) ≈ sin(theta)

    If both are available:
      theta_hat = atan2(sin_hat, cos_hat)

    If only Z is available:
      theta_hat = arccos(cos_hat)
    """
    z_hat = None
    x_hat = None

    if z_counts is not None and np.sum(z_counts) > 0:
        z_probs = z_counts / np.sum(z_counts)
        z_hat = z_probs[0] - z_probs[1]

    if x_counts is not None and np.sum(x_counts) > 0:
        x_probs = x_counts / np.sum(x_counts)
        x_hat = x_probs[0] - x_probs[1]

    if z_hat is not None and x_hat is None:
        z_hat = np.clip(z_hat, -1.0, 1.0)
        return float(np.arccos(z_hat))

    if z_hat is not None and x_hat is not None:
        return float(np.mod(np.arctan2(x_hat, z_hat), 2 * np.pi))

    raise ValueError("No measurement data available to estimate theta.")


def wrapped_theta_error(true_theta: float, theta_hat: float) -> float:
    """Shortest angular distance on a circle."""
    err = abs(true_theta - theta_hat)
    return float(min(err, 2 * np.pi - err))
# 
#  RL Part
ACTIONS = ["Z", "X"]


def build_state(z_counts: np.ndarray, x_counts: np.ndarray, shots_used: int, total_shots: int) -> np.ndarray:


    z_hat = 0.0
    x_hat = 0.0

    if np.sum(z_counts) > 0:
        z_probs = z_counts / np.sum(z_counts)
        z_hat = float(z_probs[0] - z_probs[1])

    if np.sum(x_counts) > 0:
        x_probs = x_counts / np.sum(x_counts)
        x_hat = float(x_probs[0] - x_probs[1])

    frac = shots_used / total_shots
    return np.array([z_hat, x_hat, frac, 1.0], dtype=float)

def epsilon_greedy_action(state: np.ndarray,w_z: np.ndarray, w_x: np.ndarray, w_x: np.ndarray, epsilon: float, rng: np.random.Generator) -> str:
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    
    q_Z = float(np.dot(w_z, state))
    q_X = float(np.dot(w_x, state))

    if q_Z > q_X:
        return "Z"
    if q_X > q_Z:
        return "X"
    return rng.choice(ACTIONS)


#Baseline run

def run_fixed_strategy_episode(
    psi_true: np.ndarray,
    true_theta: float,
    total_shots: int,
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    """
    Run one episode using a fixed strategy:
      - Z_only
      - ZX_split
    """
    z_counts = np.array([0, 0], dtype=int)
    x_counts = np.array([0, 0], dtype=int)

    if strategy == "Z_only":
        for _ in range(total_shots):
            probs = measure_probs_z(psi_true)
            outcome = sample_one_shot(probs, rng)
            z_counts[outcome] += 1

    elif strategy == "ZX_split":
        n_z = total_shots // 2
        n_x = total_shots - n_z

        for _ in range(n_z):
            probs = measure_probs_z(psi_true)
            outcome = sample_one_shot(probs, rng)
            z_counts[outcome] += 1

        for _ in range(n_x):
            probs = measure_probs_x(psi_true)
            outcome = sample_one_shot(probs, rng)
            x_counts[outcome] += 1
    else:
        raise ValueError(f"Unknown fixed strategy: {strategy}")

    theta_hat = estimate_theta_from_counts(z_counts=z_counts, x_counts=x_counts if np.sum(x_counts) > 0 else None)
    psi_hat = state_from_theta(theta_hat)

    F = fidelity(psi_true, psi_hat)
    return {
        "theta_hat": theta_hat,
        "theta_error": wrapped_theta_error(true_theta, theta_hat),
        "fidelity": F,
        "infidelity": 1 - F,
        "z_counts_0": int(z_counts[0]),
        "z_counts_1": int(z_counts[1]),
        "x_counts_0": int(x_counts[0]),
        "x_counts_1": int(x_counts[1]),
    }

def run_rl_episode(psi_true: np.ndarray,
                   true_theta: float,
                   total_shots: int,
                   w_z: np.ndarray,
                   w_x: np.ndarray,
                   epsilon: float,
                   rng: np.random.Generator,
                   update_weights: bool,
                   learning_rate: float,
) _> tuple[dict, np.ndarray, np.ndarray]:

    """
    Run one episode using an epsilon-greedy RL strategy.
    """
    z_counts = np.array([0, 0], dtype=int)
    x_counts = np.array([0, 0], dtype=int)
    memory = []

    for shot_idx in range(total_shots):
        state = build_state(z_counts, x_counts, shots_used=shot_idx, total_shots=total_shots)
        action = epsilon_greedy_action(state, w_z, w_x, epsilon, rng)

        if action == "Z":
            probs = measure_probs_z(psi_true)
            outcome = sample_one_shot(probs, rng)
            z_counts[outcome] += 1
        elif action == "X":
            probs = measure_probs_x(psi_true)
            outcome = sample_one_shot(probs, rng)
            x_counts[outcome] += 1
        else:
            raise ValueError(f"Unknown action: {action}")

    theta_hat = estimate_theta_from_counts(z_counts=z_counts if np.sum(z_counts) > 0 else None, x_counts=x_counts if np.sum(x_counts) > 0 else None)
    psi_hat = state_from_theta(theta_hat)

    F = fidelity(psi_true, psi_hat)
    reward = F

    if update_weights: 
        for state, action in memory:
            if action == "Z":
                w_z += learning_rate * reward * state
            elif action == "X":
                w_x += learning_rate * reward * state


    return {
        "theta_hat": theta_hat,
        "theta_error": wrapped_theta_error(true_theta, theta_hat),
        "fidelity": F,
        "infidelity": 1 - F,
        "z_counts_0": int(z_counts[0]),
        "z_counts_1": int(z_counts[1]),
        "x_counts_0": int(x_counts[0]),
        "x_counts_1": int(x_counts[1]),
    }
    return result, w_z, w_x


# ========================
# Experiment configuration
# ========================
shot_budgets = [10, 25, 50, 100, 250, 500]
num_train_episodes = 400
num_test_targets = 50
num_test_seeds = 5

epsilon_train = 0.15
epsilon_test = 0.0
learning_rate = 0.02

master_rng = np.random.default_rng(123)

# Weight vectors for the RL policy: one per action
# State features = [z_hat, x_hat, shots_fraction, 1]
w_z = np.zeros(4, dtype=float)
w_x = np.zeros(4, dtype=float)


# ========================
# Train a separate RL policy for each shot budget
# ========================
trained_weights = {}

for N in shot_budgets:
    wz = np.zeros(4, dtype=float)
    wx = np.zeros(4, dtype=float)

    for ep in range(num_train_episodes):
        theta = master_rng.uniform(0, 2 * np.pi)
        psi_true = state_from_theta(theta)

        rng = np.random.default_rng(10_000 * N + ep)
        _, wz, wx = run_rl_episode(
            psi_true=psi_true,
            true_theta=theta,
            total_shots=N,
            w_z=wz,
            w_x=wx,
            epsilon=epsilon_train,
            rng=rng,
            update_weights=True,
            learning_rate=learning_rate,
        )

    trained_weights[N] = (wz.copy(), wx.copy())


# ========================
# Evaluation
# Compare:
#   - Z_only
#   - ZX_split
#   - RL_adaptive
# ========================
rows = []

for target_id in range(num_test_targets):
    true_theta = master_rng.uniform(0, 2 * np.pi)
    psi_true = state_from_theta(true_theta)

    for seed in range(num_test_seeds):
        for N in shot_budgets:
            rng = np.random.default_rng(target_id * 1000 + seed * 100 + N)

            # ----- Baseline: Z_only
            out_z = run_fixed_strategy_episode(
                psi_true=psi_true,
                true_theta=true_theta,
                total_shots=N,
                strategy="Z_only",
                rng=rng,
            )
            rows.append({
                "target_id": target_id,
                "seed": seed,
                "N": N,
                "method": "Z_only",
                "true_theta": true_theta,
                **out_z,
            })

            # ----- Baseline: ZX_split
            out_split = run_fixed_strategy_episode(
                psi_true=psi_true,
                true_theta=true_theta,
                total_shots=N,
                strategy="ZX_split",
                rng=rng,
            )
            rows.append({
                "target_id": target_id,
                "seed": seed,
                "N": N,
                "method": "ZX_split",
                "true_theta": true_theta,
                **out_split,
            })

            # ----- RL
            wz, wx = trained_weights[N]
            out_rl, _, _ = run_rl_episode(
                psi_true=psi_true,
                true_theta=true_theta,
                total_shots=N,
                w_z=wz,
                w_x=wx,
                epsilon=epsilon_test,
                rng=rng,
                update_weights=False,
                learning_rate=learning_rate,
            )
            rows.append({
                "target_id": target_id,
                "seed": seed,
                "N": N,
                "method": "RL_adaptive",
                "true_theta": true_theta,
                **out_rl,
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
          theta_error_mean=("theta_error", "mean"),
          theta_error_std=("theta_error", "std"),
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
plot_metric(
    metric_mean="fidelity_mean",
    metric_std="fidelity_std",
    ylabel="Mean Fidelity",
    title="Exp02: 1-Qubit Adaptive Measurement (Fidelity vs Shots)",
    filename="fidelity_vs_shots.png",
    logy=False,
)

plot_metric(
    metric_mean="infidelity_mean",
    metric_std="infidelity_std",
    ylabel="Mean Infidelity",
    title="Exp02: 1-Qubit Adaptive Measurement (Infidelity vs Shots)",
    filename="infidelity_vs_shots.png",
    logy=True,
)

plot_metric(
    metric_mean="theta_error_mean",
    metric_std="theta_error_std",
    ylabel="Mean Theta Error",
    title="Exp02: 1-Qubit Adaptive Measurement (Theta Error vs Shots)",
    filename="theta_error_vs_shots.png",
    logy=True,
)


# ========================
# Notes file
# ========================
with open(f"{run_dir}/notes.txt", "w") as f:
    f.write("Experiment: Exp02 - 1-qubit adaptive measurement with RL\n")
    f.write("Methods: Z_only, ZX_split, RL_adaptive\n")
    f.write(f"Shot budgets: {shot_budgets}\n")
    f.write(f"Training episodes per shot budget: {num_train_episodes}\n")
    f.write(f"Test targets: {num_test_targets}\n")
    f.write(f"Test seeds: {num_test_seeds}\n")
    f.write(f"Epsilon train: {epsilon_train}\n")
    f.write(f"Epsilon test: {epsilon_test}\n")
    f.write(f"Learning rate: {learning_rate}\n")


print("Saved results to:", run_dir)
print("Files created:")
print("- metrics.csv")
print("- summary.csv")
print("- fidelity_vs_shots.png")
print("- infidelity_vs_shots.png")
print("- theta_error_vs_shots.png")
print("- notes.txt")