import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp03_1qubit_generalized"
os.makedirs(run_dir, exist_ok=True)


# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """
    General pure 1-qubit state:
        |psi(theta, phi)> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    with theta in [0, pi], phi in [0, 2pi).
    """
    return np.array([
        np.cos(theta / 2),
        np.exp(1j * phi) * np.sin(theta / 2)
    ], dtype=complex)


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """State fidelity |<psi|phi>|^2."""
    overlap = np.vdot(psi, phi)
    return float(np.abs(overlap) ** 2)


def hadamard() -> np.ndarray:
    """Hadamard gate."""
    return (1 / np.sqrt(2)) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex)


def s_dagger() -> np.ndarray:
    """S^\dagger gate."""
    return np.array([
        [1, 0],
        [0, -1j]
    ], dtype=complex)


# ========================
# Measurement probabilities
# ========================
def measure_probs_z(psi: np.ndarray) -> np.ndarray:
    """
    Z-basis probabilities [P(0), P(1)].
    """
    p = np.abs(psi) ** 2
    return p / np.sum(p)


def measure_probs_x(psi: np.ndarray) -> np.ndarray:
    """
    X-basis probabilities [P(+), P(-)].
    Implemented by applying H and then measuring in Z.
    """
    psi_x = hadamard() @ psi
    p = np.abs(psi_x) ** 2
    return p / np.sum(p)


def measure_probs_y(psi: np.ndarray) -> np.ndarray:
    """
    Y-basis probabilities [P(+_y), P(-_y)].
    Implemented by applying H @ S^\dagger and then measuring in Z.
    """
    psi_y = hadamard() @ (s_dagger() @ psi)
    p = np.abs(psi_y) ** 2
    return p / np.sum(p)


def sample_one_shot(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one outcome, returning 0 or 1."""
    return int(rng.choice([0, 1], p=probs))


# ========================
# Bloch estimation
# ========================
def estimate_bloch_from_counts(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    y_counts: np.ndarray,
):
    """
    Estimate Bloch coordinates from measurement counts.

    For each basis:
      coord_hat = P(outcome 0) - P(outcome 1)

    Interpretation:
      Z: outcome 0 -> +Z, outcome 1 -> -Z
      X: outcome 0 -> +X, outcome 1 -> -X
      Y: outcome 0 -> +Y, outcome 1 -> -Y
    """
    z_hat = 0.0
    x_hat = 0.0
    y_hat = 0.0

    z_total = int(np.sum(z_counts))
    x_total = int(np.sum(x_counts))
    y_total = int(np.sum(y_counts))

    if z_total > 0:
        z_probs = z_counts / z_total
        z_hat = float(z_probs[0] - z_probs[1])

    if x_total > 0:
        x_probs = x_counts / x_total
        x_hat = float(x_probs[0] - x_probs[1])

    if y_total > 0:
        y_probs = y_counts / y_total
        y_hat = float(y_probs[0] - y_probs[1])

    return x_hat, y_hat, z_hat


def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
    """
    Convert a Bloch vector direction into a pure qubit state.

    We normalize the vector to unit length to produce a pure-state estimate.
    If the norm is tiny, default to |0>.
    """
    r = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(r)

    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=complex)

    x_n, y_n, z_n = r / norm
    z_n = float(np.clip(z_n, -1.0, 1.0))

    theta_hat = float(np.arccos(z_n))
    phi_hat = float(np.mod(np.arctan2(y_n, x_n), 2 * np.pi))

    return state_from_angles(theta_hat, phi_hat)


def estimate_state_from_counts(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    y_counts: np.ndarray,
) -> np.ndarray:
    """Estimate a pure-state qubit from X/Y/Z counts."""
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)
    return bloch_to_state(x_hat, y_hat, z_hat)


# ========================
# RL state and policy
# ========================
ACTIONS = ["Z", "X", "Y"]


def build_state(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    y_counts: np.ndarray,
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    """
    RL feature vector:
      [x_hat, y_hat, z_hat, x_frac, y_frac, z_frac, progress, bias]
    """
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)

    z_total = int(np.sum(z_counts))
    x_total = int(np.sum(x_counts))
    y_total = int(np.sum(y_counts))

    x_frac = x_total / total_shots
    y_frac = y_total / total_shots
    z_frac = z_total / total_shots
    progress = shots_used / total_shots

    return np.array(
        [x_hat, y_hat, z_hat, x_frac, y_frac, z_frac, progress, 1.0],
        dtype=float
    )


def epsilon_greedy_action(
    state: np.ndarray,
    w_z: np.ndarray,
    w_x: np.ndarray,
    w_y: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> str:
    """Epsilon-greedy action selection over Z/X/Y."""
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)

    q_z = float(np.dot(w_z, state))
    q_x = float(np.dot(w_x, state))
    q_y = float(np.dot(w_y, state))

    q_dict = {"Z": q_z, "X": q_x, "Y": q_y}
    max_q = max(q_dict.values())

    best_actions = [a for a, q in q_dict.items() if q == max_q]
    return rng.choice(best_actions)


# ========================
# Fixed strategy episode
# ========================
def run_fixed_strategy_episode(
    psi_true: np.ndarray,
    total_shots: int,
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    """
    Fixed strategies:
      - Z_only
      - ZX_split
      - XYZ_split
    """
    z_counts = np.array([0, 0], dtype=int)
    x_counts = np.array([0, 0], dtype=int)
    y_counts = np.array([0, 0], dtype=int)

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

    elif strategy == "XYZ_split":
        n_z = total_shots // 3
        n_x = total_shots // 3
        n_y = total_shots - n_z - n_x

        for _ in range(n_z):
            probs = measure_probs_z(psi_true)
            outcome = sample_one_shot(probs, rng)
            z_counts[outcome] += 1

        for _ in range(n_x):
            probs = measure_probs_x(psi_true)
            outcome = sample_one_shot(probs, rng)
            x_counts[outcome] += 1

        for _ in range(n_y):
            probs = measure_probs_y(psi_true)
            outcome = sample_one_shot(probs, rng)
            y_counts[outcome] += 1

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    psi_hat = estimate_state_from_counts(z_counts, x_counts, y_counts)
    F = fidelity(psi_true, psi_hat)

    return {
        "fidelity": F,
        "infidelity": 1 - F,
        "z_counts_0": int(z_counts[0]),
        "z_counts_1": int(z_counts[1]),
        "x_counts_0": int(x_counts[0]),
        "x_counts_1": int(x_counts[1]),
        "y_counts_0": int(y_counts[0]),
        "y_counts_1": int(y_counts[1]),
    }


# ========================
# RL episode
# ========================
def run_rl_episode(
    psi_true: np.ndarray,
    total_shots: int,
    w_z: np.ndarray,
    w_x: np.ndarray,
    w_y: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    update_weights: bool,
    learning_rate: float,
):
    """
    RL adaptive measurement with stepwise reward:
      reward_step = fidelity_after - fidelity_before
    """
    total_shots = int(total_shots)

    if total_shots <= 0:
        raise ValueError(f"total_shots must be positive, got {total_shots}")

    z_counts = np.array([0, 0], dtype=int)
    x_counts = np.array([0, 0], dtype=int)
    y_counts = np.array([0, 0], dtype=int)

    shots_used = 0

    # Warm start: one shot in each basis if budget allows
    if shots_used < total_shots:
        probs = measure_probs_z(psi_true)
        outcome = sample_one_shot(probs, rng)
        z_counts[outcome] += 1
        shots_used += 1

    if shots_used < total_shots:
        probs = measure_probs_x(psi_true)
        outcome = sample_one_shot(probs, rng)
        x_counts[outcome] += 1
        shots_used += 1

    if shots_used < total_shots:
        probs = measure_probs_y(psi_true)
        outcome = sample_one_shot(probs, rng)
        y_counts[outcome] += 1
        shots_used += 1

    psi_hat_prev = estimate_state_from_counts(z_counts, x_counts, y_counts)
    F_prev = fidelity(psi_true, psi_hat_prev)

    for shot_idx in range(shots_used, total_shots):
        state = build_state(
            z_counts=z_counts,
            x_counts=x_counts,
            y_counts=y_counts,
            shots_used=shot_idx,
            total_shots=total_shots,
        )

        action = epsilon_greedy_action(
            state=state,
            w_z=w_z,
            w_x=w_x,
            w_y=w_y,
            epsilon=epsilon,
            rng=rng,
        )

        if action == "Z":
            probs = measure_probs_z(psi_true)
            outcome = sample_one_shot(probs, rng)
            z_counts[outcome] += 1

        elif action == "X":
            probs = measure_probs_x(psi_true)
            outcome = sample_one_shot(probs, rng)
            x_counts[outcome] += 1

        elif action == "Y":
            probs = measure_probs_y(psi_true)
            outcome = sample_one_shot(probs, rng)
            y_counts[outcome] += 1

        else:
            raise ValueError(f"Unexpected action: {action!r}")

        psi_hat_new = estimate_state_from_counts(z_counts, x_counts, y_counts)
        F_new = fidelity(psi_true, psi_hat_new)

        reward_step = F_new - F_prev

        if update_weights:
            if action == "Z":
                w_z = w_z + learning_rate * reward_step * state
            elif action == "X":
                w_x = w_x + learning_rate * reward_step * state
            elif action == "Y":
                w_y = w_y + learning_rate * reward_step * state

        F_prev = F_new

    psi_hat = estimate_state_from_counts(z_counts, x_counts, y_counts)
    F = fidelity(psi_true, psi_hat)

    result = {
        "fidelity": F,
        "infidelity": 1 - F,
        "reward": F,
        "z_counts_0": int(z_counts[0]),
        "z_counts_1": int(z_counts[1]),
        "x_counts_0": int(x_counts[0]),
        "x_counts_1": int(x_counts[1]),
        "y_counts_0": int(y_counts[0]),
        "y_counts_1": int(y_counts[1]),
    }

    return result, w_z, w_x, w_y


# ========================
# Experiment configuration
# ========================
shot_budgets = [10, 25, 50, 100, 250, 500]
num_train_episodes = 800
num_test_targets = 50
num_test_seeds = 5

epsilon_train = 0.15
epsilon_test = 0.0
learning_rate = 0.02

master_rng = np.random.default_rng(123)


# ========================
# Train a separate RL policy for each shot budget
# ========================
trained_weights = {}

for N in shot_budgets:
    wz = np.zeros(8, dtype=float)
    wx = np.zeros(8, dtype=float)
    wy = np.zeros(8, dtype=float)

    for ep in range(num_train_episodes):
        theta = master_rng.uniform(0, np.pi)
        phi = master_rng.uniform(0, 2 * np.pi)

        psi_true = state_from_angles(theta, phi)

        rng = np.random.default_rng(10_000 * N + ep)

        _, wz, wx, wy = run_rl_episode(
            psi_true=psi_true,
            total_shots=N,
            w_z=wz,
            w_x=wx,
            w_y=wy,
            epsilon=epsilon_train,
            rng=rng,
            update_weights=True,
            learning_rate=learning_rate,
        )

    trained_weights[N] = (wz.copy(), wx.copy(), wy.copy())
    print(f"Finished training RL policy for shot budget N = {N}")


# ========================
# Evaluation
# Compare:
#   - Z_only
#   - ZX_split
#   - XYZ_split
#   - RL_adaptive
# ========================
rows = []

for target_id in range(num_test_targets):
    true_theta = master_rng.uniform(0, np.pi)
    true_phi = master_rng.uniform(0, 2 * np.pi)

    psi_true = state_from_angles(true_theta, true_phi)

    for seed in range(num_test_seeds):
        for N in shot_budgets:
            rng = np.random.default_rng(target_id * 1000 + seed * 100 + N)

            out_z = run_fixed_strategy_episode(
                psi_true=psi_true,
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
                "true_phi": true_phi,
                **out_z,
            })

            out_zx = run_fixed_strategy_episode(
                psi_true=psi_true,
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
                "true_phi": true_phi,
                **out_zx,
            })

            out_xyz = run_fixed_strategy_episode(
                psi_true=psi_true,
                total_shots=N,
                strategy="XYZ_split",
                rng=rng,
            )
            rows.append({
                "target_id": target_id,
                "seed": seed,
                "N": N,
                "method": "XYZ_split",
                "true_theta": true_theta,
                "true_phi": true_phi,
                **out_xyz,
            })

            wz, wx, wy = trained_weights[N]
            out_rl, _, _, _ = run_rl_episode(
                psi_true=psi_true,
                total_shots=N,
                w_z=wz,
                w_x=wx,
                w_y=wy,
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
                "true_phi": true_phi,
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
    title="Exp03: Generalized 1-Qubit State Estimation (Fidelity vs Shots)",
    filename="fidelity_vs_shots.png",
    logy=False,
)

plot_metric(
    metric_mean="infidelity_mean",
    metric_std="infidelity_std",
    ylabel="Mean Infidelity",
    title="Exp03: Generalized 1-Qubit State Estimation (Infidelity vs Shots)",
    filename="infidelity_vs_shots.png",
    logy=True,
)


# ========================
# Notes file
# ========================
with open(f"{run_dir}/notes.txt", "w") as f:
    f.write("Experiment: Exp03 - generalized 1-qubit adaptive measurement\n")
    f.write("State family: full pure 1-qubit states parameterized by (theta, phi)\n")
    f.write("Methods: Z_only, ZX_split, XYZ_split, RL_adaptive\n")
    f.write("RL actions: Z, X, Y\n")
    f.write("RL reward: stepwise fidelity improvement\n")
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
print("- notes.txt")