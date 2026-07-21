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


def build_state(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    """
    State features:
      0: estimated Z Bloch component
      1: estimated X Bloch component
      2: uncertainty in Z estimate
      3: uncertainty in X estimate
      4: fraction of measurements allocated to Z
      5: fraction of measurements allocated to X
      6: Z allocation deficit relative to 50/50
      7: X allocation deficit relative to 50/50
      8: episode progress
      9: allocation imbalance
     10: bias
    """
    n_z = int(np.sum(z_counts))
    n_x = int(np.sum(x_counts))
    n_measured = n_z + n_x

    z_hat = 0.0
    x_hat = 0.0

    if n_z > 0:
        z_probs = z_counts / n_z
        z_hat = float(z_probs[0] - z_probs[1])

    if n_x > 0:
        x_probs = x_counts / n_x
        x_hat = float(x_probs[0] - x_probs[1])

    # Estimated standard error of each Bloch component.
    z_uncertainty = float(
        np.sqrt(max(0.0, 1.0 - z_hat**2) / max(1, n_z))
    )
    x_uncertainty = float(
        np.sqrt(max(0.0, 1.0 - x_hat**2) / max(1, n_x))
    )

    if n_measured > 0:
        z_frac = n_z / n_measured
        x_frac = n_x / n_measured
    else:
        z_frac = 0.0
        x_frac = 0.0

    z_deficit = max(0.0, 0.5 - z_frac)
    x_deficit = max(0.0, 0.5 - x_frac)

    progress = shots_used / total_shots
    imbalance = abs(z_frac - x_frac)

    return np.array(
        [
            z_hat,
            x_hat,
            z_uncertainty,
            x_uncertainty,
            z_frac,
            x_frac,
            z_deficit,
            x_deficit,
            progress,
            imbalance,
            1.0,
        ],
        dtype=float,
    )

def epsilon_greedy_action(state: np.ndarray,w_z: np.ndarray, w_x: np.ndarray, epsilon: float, rng: np.random.Generator) -> str:
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
        "z_shots": int(np.sum(z_counts)),
        "x_shots": int(np.sum(x_counts)),
    }

def run_rl_episode(
    psi_true: np.ndarray,
    true_theta: float,
    total_shots: int,
    w_z: np.ndarray,
    w_x: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    update_weights: bool,
    learning_rate: float,
    gamma: float = 0.95,
    imbalance_penalty: float = 0.02,
):
    total_shots = int(total_shots)

    if total_shots <= 0:
        raise ValueError(
            f"total_shots must be positive, got {total_shots}"
        )

    z_counts = np.array([0, 0], dtype=int)
    x_counts = np.array([0, 0], dtype=int)
    shots_used = 0

    # Warm start: one measurement in each required basis.
    outcome = sample_one_shot(measure_probs_z(psi_true), rng)
    z_counts[outcome] += 1
    shots_used += 1

    if shots_used < total_shots:
        outcome = sample_one_shot(measure_probs_x(psi_true), rng)
        x_counts[outcome] += 1
        shots_used += 1

    theta_hat_prev = estimate_theta_from_counts(
        z_counts=z_counts,
        x_counts=x_counts,
    )
    psi_hat_prev = state_from_theta(theta_hat_prev)
    fidelity_prev = fidelity(psi_true, psi_hat_prev)

    while shots_used < total_shots:
        state = build_state(
            z_counts=z_counts,
            x_counts=x_counts,
            shots_used=shots_used,
            total_shots=total_shots,
        )

        action = epsilon_greedy_action(
            state=state,
            w_z=w_z,
            w_x=w_x,
            epsilon=epsilon,
            rng=rng,
        )

        old_n_z = int(np.sum(z_counts))
        old_n_x = int(np.sum(x_counts))
        old_total = max(1, old_n_z + old_n_x)
        old_imbalance = abs(old_n_z - old_n_x) / old_total

        if action == "Z":
            outcome = sample_one_shot(measure_probs_z(psi_true), rng)
            z_counts[outcome] += 1
        elif action == "X":
            outcome = sample_one_shot(measure_probs_x(psi_true), rng)
            x_counts[outcome] += 1
        else:
            raise ValueError(f"Unexpected action: {action!r}")

        shots_used += 1

        theta_hat_new = estimate_theta_from_counts(
            z_counts=z_counts,
            x_counts=x_counts,
        )
        psi_hat_new = state_from_theta(theta_hat_new)
        fidelity_new = fidelity(psi_true, psi_hat_new)

        new_n_z = int(np.sum(z_counts))
        new_n_x = int(np.sum(x_counts))
        new_total = max(1, new_n_z + new_n_x)
        new_imbalance = abs(new_n_z - new_n_x) / new_total

        fidelity_gain = fidelity_new - fidelity_prev

        # Penalize only an increase in imbalance.
        imbalance_increase = max(
            0.0,
            new_imbalance - old_imbalance,
        )

        reward_step = (
            fidelity_gain
            - imbalance_penalty * imbalance_increase
        )

        if update_weights:
            if action == "Z":
                q_current = float(np.dot(w_z, state))
            else:
                q_current = float(np.dot(w_x, state))

            terminal = shots_used >= total_shots

            if terminal:
                td_target = reward_step
            else:
                next_state = build_state(
                    z_counts=z_counts,
                    x_counts=x_counts,
                    shots_used=shots_used,
                    total_shots=total_shots,
                )

                q_next_z = float(np.dot(w_z, next_state))
                q_next_x = float(np.dot(w_x, next_state))

                td_target = (
                    reward_step
                    + gamma * max(q_next_z, q_next_x)
                )

            td_error = np.clip(
                td_target - q_current,
                -1.0,
                1.0,
            )

            if action == "Z":
                w_z += learning_rate * td_error * state
            else:
                w_x += learning_rate * td_error * state

        fidelity_prev = fidelity_new

    theta_hat = estimate_theta_from_counts(
        z_counts=z_counts,
        x_counts=x_counts,
    )
    psi_hat = state_from_theta(theta_hat)
    final_fidelity = fidelity(psi_true, psi_hat)

    result = {
        "theta_hat": theta_hat,
        "theta_error": wrapped_theta_error(
            true_theta,
            theta_hat,
        ),
        "fidelity": final_fidelity,
        "infidelity": 1.0 - final_fidelity,
        "reward": final_fidelity,
        "z_counts_0": int(z_counts[0]),
        "z_counts_1": int(z_counts[1]),
        "x_counts_0": int(x_counts[0]),
        "x_counts_1": int(x_counts[1]),
        "z_shots": int(np.sum(z_counts)),
        "x_shots": int(np.sum(x_counts)),
    }

    return result, w_z, w_x

# ========================
# Experiment configuration
# ========================
shot_budgets = [10, 25, 50, 100, 250, 500]

num_train_episodes = 3000
num_test_targets = 50
num_test_seeds = 5

epsilon_start = 0.30
epsilon_end = 0.05
epsilon_test = 0.0

learning_rate = 0.01
gamma = 0.95
imbalance_penalty = 0.02

master_rng = np.random.default_rng(123)


# Weight vectors for the RL policy: one per action
# State features = [z_hat, x_hat, shots_fraction, 1]



# ========================
# Train a separate RL policy for each shot budget
# ========================
trained_weights = {}
num_features = 11

for N in shot_budgets:
    wz = np.zeros(num_features, dtype=float)
    wx = np.zeros(num_features, dtype=float)

    # Safe initial preference:
    # favor the basis that is uncertain or under-sampled.
    wz[2] = 0.10  # Z uncertainty
    wx[3] = 0.10  # X uncertainty

    wz[6] = 0.20  # Z deficit
    wx[7] = 0.20  # X deficit

    for ep in range(num_train_episodes):
        true_theta = master_rng.uniform(0, 2 * np.pi)
        psi_true = state_from_theta(true_theta)

        progress = ep / max(1, num_train_episodes - 1)
        epsilon = (
            epsilon_start
            + progress * (epsilon_end - epsilon_start)
        )

        rng = np.random.default_rng(10_000 * N + ep)

        _, wz, wx = run_rl_episode(
            psi_true=psi_true,
            true_theta=true_theta,
            total_shots=N,
            w_z=wz,
            w_x=wx,
            epsilon=epsilon,
            rng=rng,
            update_weights=True,
            learning_rate=learning_rate,
            gamma=gamma,
            imbalance_penalty=imbalance_penalty,
        )

    trained_weights[N] = (wz.copy(), wx.copy())

    print(
        f"Finished N={N}: "
        f"||wz||={np.linalg.norm(wz):.4f}, "
        f"||wx||={np.linalg.norm(wx):.4f}"
    )
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
                gamma=gamma,
                imbalance_penalty=imbalance_penalty,
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
          z_shots_mean=("z_shots", "mean"),
          x_shots_mean=("x_shots", "mean"),
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
    f.write(f"Epsilon start: {epsilon_start}\n")
    f.write(f"Epsilon end: {epsilon_end}\n")
    f.write(f"Epsilon test: {epsilon_test}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Gamma: {gamma}\n")
    f.write(f"Imbalance penalty: {imbalance_penalty}\n")
    f.write(f"Number of state features: {num_features}\n")


print("Saved results to:", run_dir)
print("Files created:")
print("- metrics.csv")
print("- summary.csv")
print("- fidelity_vs_shots.png")
print("- infidelity_vs_shots.png")
print("- theta_error_vs_shots.png")
print("- notes.txt")

print("wz shape:", wz.shape)
print("wx shape:", wx.shape)