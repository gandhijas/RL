import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp03_1qubit_generalized_upgraded"
os.makedirs(run_dir, exist_ok=True)


# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """General pure one-qubit state."""
    return np.array(
        [
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2),
        ],
        dtype=complex,
    )


def sample_uniform_pure_state(rng: np.random.Generator):
    """Sample a pure qubit uniformly from the Bloch sphere."""
    z = rng.uniform(-1.0, 1.0)
    theta = float(np.arccos(z))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    return theta, phi, state_from_angles(theta, phi)


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """State fidelity |<psi|phi>|^2."""
    overlap = np.vdot(psi, phi)
    return float(np.abs(overlap) ** 2)


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
def estimate_bloch_from_counts(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    y_counts: np.ndarray,
):
    """Estimate Bloch coordinates from Pauli-basis counts."""
    z_total = int(np.sum(z_counts))
    x_total = int(np.sum(x_counts))
    y_total = int(np.sum(y_counts))

    z_hat = 0.0 if z_total == 0 else float((z_counts[0] - z_counts[1]) / z_total)
    x_hat = 0.0 if x_total == 0 else float((x_counts[0] - x_counts[1]) / x_total)
    y_hat = 0.0 if y_total == 0 else float((y_counts[0] - y_counts[1]) / y_total)

    return x_hat, y_hat, z_hat


def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
    """Project an estimated Bloch vector onto the pure-state Bloch sphere."""
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
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)
    return bloch_to_state(x_hat, y_hat, z_hat)


# ========================
# RL state and policy
# ========================
ACTIONS = ("Z", "X", "Y")
STATE_DIM = 15


def build_state(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    y_counts: np.ndarray,
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    """
    Three-basis extension of the finalized Experiment 2 state.

    Features:
      [x_hat, y_hat, z_hat,
       x_unc, y_unc, z_unc,
       x_frac, y_frac, z_frac,
       x_deficit, y_deficit, z_deficit,
       progress, allocation_imbalance, bias]
    """
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)

    x_total = int(np.sum(x_counts))
    y_total = int(np.sum(y_counts))
    z_total = int(np.sum(z_counts))

    # Fractions are expressed relative to the total episode budget, matching Exp02.
    x_frac = x_total / total_shots
    y_frac = y_total / total_shots
    z_frac = z_total / total_shots
    progress = shots_used / total_shots

    # Stable count-based uncertainty proxy. It remains nonzero after one shot.
    x_unc = 1.0 / np.sqrt(x_total + 1.0)
    y_unc = 1.0 / np.sqrt(y_total + 1.0)
    z_unc = 1.0 / np.sqrt(z_total + 1.0)

    target_frac = progress / 3.0
    x_deficit = target_frac - x_frac
    y_deficit = target_frac - y_frac
    z_deficit = target_frac - z_frac

    allocation_imbalance = float(
        np.sqrt(
            (x_frac - target_frac) ** 2
            + (y_frac - target_frac) ** 2
            + (z_frac - target_frac) ** 2
        )
    )

    return np.array(
        [
            x_hat,
            y_hat,
            z_hat,
            x_unc,
            y_unc,
            z_unc,
            x_frac,
            y_frac,
            z_frac,
            x_deficit,
            y_deficit,
            z_deficit,
            progress,
            allocation_imbalance,
            1.0,
        ],
        dtype=float,
    )


def q_values(state: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, float]:
    return {action: float(np.dot(weights[action], state)) for action in ACTIONS}


def epsilon_greedy_action(
    state: np.ndarray,
    weights: dict[str, np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
) -> str:
    if rng.random() < epsilon:
        return str(rng.choice(ACTIONS))

    values = q_values(state, weights)
    max_q = max(values.values())
    best_actions = [a for a, q in values.items() if np.isclose(q, max_q)]
    return str(rng.choice(best_actions))


def initialize_weights() -> dict[str, np.ndarray]:
    """Small informative initialization analogous to the finalized Exp02 policy."""
    weights = {action: np.zeros(STATE_DIM, dtype=float) for action in ACTIONS}

    # Feature indices
    unc_idx = {"X": 3, "Y": 4, "Z": 5}
    deficit_idx = {"X": 9, "Y": 10, "Z": 11}

    for action in ACTIONS:
        weights[action][unc_idx[action]] = 0.05
        weights[action][deficit_idx[action]] = 0.10

    return weights


# ========================
# Fixed strategy episode
# ========================
def run_fixed_strategy_episode(
    psi_true: np.ndarray,
    total_shots: int,
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    z_counts = np.array([0, 0], dtype=int)
    x_counts = np.array([0, 0], dtype=int)
    y_counts = np.array([0, 0], dtype=int)

    if strategy == "Z_only":
        allocation = {"Z": total_shots, "X": 0, "Y": 0}
    elif strategy == "ZX_split":
        n_z = total_shots // 2
        allocation = {"Z": n_z, "X": total_shots - n_z, "Y": 0}
    elif strategy == "XYZ_split":
        n_z = total_shots // 3
        n_x = total_shots // 3
        allocation = {"Z": n_z, "X": n_x, "Y": total_shots - n_z - n_x}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    probs_by_action = {
        "Z": measure_probs_z(psi_true),
        "X": measure_probs_x(psi_true),
        "Y": measure_probs_y(psi_true),
    }
    counts_by_action = {"Z": z_counts, "X": x_counts, "Y": y_counts}

    for action, n_shots in allocation.items():
        for _ in range(n_shots):
            outcome = sample_one_shot(probs_by_action[action], rng)
            counts_by_action[action][outcome] += 1

    psi_hat = estimate_state_from_counts(z_counts, x_counts, y_counts)
    F = fidelity(psi_true, psi_hat)

    return {
        "fidelity": F,
        "infidelity": 1.0 - F,
        "z_shots": int(np.sum(z_counts)),
        "x_shots": int(np.sum(x_counts)),
        "y_shots": int(np.sum(y_counts)),
        "z_counts_0": int(z_counts[0]),
        "z_counts_1": int(z_counts[1]),
        "x_counts_0": int(x_counts[0]),
        "x_counts_1": int(x_counts[1]),
        "y_counts_0": int(y_counts[0]),
        "y_counts_1": int(y_counts[1]),
    }


# ========================
# RL episode with TD learning
# ========================
def run_rl_episode(
    psi_true: np.ndarray,
    total_shots: int,
    weights: dict[str, np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
    update_weights: bool,
    learning_rate: float,
    gamma: float,
    imbalance_weight: float,
):
    if total_shots <= 0:
        raise ValueError(f"total_shots must be positive, got {total_shots}")

    # Copy only when training, so caller explicitly receives updated arrays.
    local_weights = {a: weights[a].copy() for a in ACTIONS}

    z_counts = np.array([0, 0], dtype=int)
    x_counts = np.array([0, 0], dtype=int)
    y_counts = np.array([0, 0], dtype=int)
    counts_by_action = {"Z": z_counts, "X": x_counts, "Y": y_counts}
    probs_by_action = {
        "Z": measure_probs_z(psi_true),
        "X": measure_probs_x(psi_true),
        "Y": measure_probs_y(psi_true),
    }

    shots_used = 0

    # Balanced XYZ warm start.
    for action in ACTIONS:
        if shots_used >= total_shots:
            break
        outcome = sample_one_shot(probs_by_action[action], rng)
        counts_by_action[action][outcome] += 1
        shots_used += 1

    psi_hat_prev = estimate_state_from_counts(z_counts, x_counts, y_counts)
    F_prev = fidelity(psi_true, psi_hat_prev)

    while shots_used < total_shots:
        state = build_state(
            z_counts=z_counts,
            x_counts=x_counts,
            y_counts=y_counts,
            shots_used=shots_used,
            total_shots=total_shots,
        )

        action = epsilon_greedy_action(
            state=state,
            weights=local_weights,
            epsilon=epsilon,
            rng=rng,
        )

        current_q = float(np.dot(local_weights[action], state))

        outcome = sample_one_shot(probs_by_action[action], rng)
        counts_by_action[action][outcome] += 1
        shots_used += 1

        psi_hat_new = estimate_state_from_counts(z_counts, x_counts, y_counts)
        F_new = fidelity(psi_true, psi_hat_new)
        fidelity_gain = F_new - F_prev

        used_total = shots_used
        fractions = np.array(
            [
                np.sum(x_counts) / used_total,
                np.sum(y_counts) / used_total,
                np.sum(z_counts) / used_total,
            ],
            dtype=float,
        )
        imbalance = float(np.sum((fractions - 1.0 / 3.0) ** 2))
        reward = fidelity_gain - imbalance_weight * imbalance

        terminal = shots_used >= total_shots
        if terminal:
            td_target = reward
        else:
            next_state = build_state(
                z_counts=z_counts,
                x_counts=x_counts,
                y_counts=y_counts,
                shots_used=shots_used,
                total_shots=total_shots,
            )
            max_next_q = max(q_values(next_state, local_weights).values())
            td_target = reward + gamma * max_next_q

        if update_weights:
            td_error = td_target - current_q
            local_weights[action] += learning_rate * td_error * state

        F_prev = F_new

    psi_hat = estimate_state_from_counts(z_counts, x_counts, y_counts)
    F = fidelity(psi_true, psi_hat)

    result = {
        "fidelity": F,
        "infidelity": 1.0 - F,
        "reward": F,
        "z_shots": int(np.sum(z_counts)),
        "x_shots": int(np.sum(x_counts)),
        "y_shots": int(np.sum(y_counts)),
        "z_counts_0": int(z_counts[0]),
        "z_counts_1": int(z_counts[1]),
        "x_counts_0": int(x_counts[0]),
        "x_counts_1": int(x_counts[1]),
        "y_counts_0": int(y_counts[0]),
        "y_counts_1": int(y_counts[1]),
    }

    return result, local_weights


# ========================
# Experiment configuration
# ========================
shot_budgets = [10, 25, 50, 100, 250, 500]
num_train_episodes = 3000
num_test_targets = 50
num_test_seeds = 5

epsilon_start = 0.30
epsilon_end = 0.05
learning_rate = 0.01
gamma = 0.95
imbalance_weight = 0.01

master_rng = np.random.default_rng(123)


# ========================
# Train a separate RL policy for each shot budget
# ========================
trained_weights: dict[int, dict[str, np.ndarray]] = {}

for N in shot_budgets:
    weights = initialize_weights()

    for ep in range(num_train_episodes):
        frac = ep / max(1, num_train_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        theta, phi, psi_true = sample_uniform_pure_state(master_rng)
        rng = np.random.default_rng(10_000_000 + 10_000 * N + ep)

        _, weights = run_rl_episode(
            psi_true=psi_true,
            total_shots=N,
            weights=weights,
            epsilon=epsilon,
            rng=rng,
            update_weights=True,
            learning_rate=learning_rate,
            gamma=gamma,
            imbalance_weight=imbalance_weight,
        )

    trained_weights[N] = {a: weights[a].copy() for a in ACTIONS}
    print(f"Finished training upgraded RL policy for shot budget N = {N}")


# ========================
# Evaluation
# ========================
rows = []
methods = ("Z_only", "ZX_split", "XYZ_split", "RL_adaptive")

# Generate the test set once so every method sees the same states.
test_states = []
for target_id in range(num_test_targets):
    true_theta, true_phi, psi_true = sample_uniform_pure_state(master_rng)
    test_states.append((target_id, true_theta, true_phi, psi_true))

for target_id, true_theta, true_phi, psi_true in test_states:
    for seed in range(num_test_seeds):
        for N in shot_budgets:
            base_seed = 100_000_000 + target_id * 100_000 + seed * 1_000 + N

            for method_idx, method in enumerate(methods):
                # Independent reproducible stream per method.
                rng = np.random.default_rng(base_seed + method_idx)

                if method == "RL_adaptive":
                    out, _ = run_rl_episode(
                        psi_true=psi_true,
                        total_shots=N,
                        weights=trained_weights[N],
                        epsilon=0.0,
                        rng=rng,
                        update_weights=False,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        imbalance_weight=imbalance_weight,
                    )
                else:
                    out = run_fixed_strategy_episode(
                        psi_true=psi_true,
                        total_shots=N,
                        strategy=method,
                        rng=rng,
                    )

                rows.append(
                    {
                        "target_id": target_id,
                        "seed": seed,
                        "N": N,
                        "method": method,
                        "true_theta": true_theta,
                        "true_phi": true_phi,
                        **out,
                    }
                )


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
        x_shots_mean=("x_shots", "mean"),
        y_shots_mean=("y_shots", "mean"),
        z_shots_mean=("z_shots", "mean"),
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
        lower = y - ystd
        upper = y + ystd
        if logy:
            lower = np.maximum(lower, 1e-12)
        plt.fill_between(x, lower, upper, alpha=0.2)

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


plot_metric(
    metric_mean="fidelity_mean",
    metric_std="fidelity_std",
    ylabel="Mean Fidelity",
    title="Exp03 Upgraded: Generalized 1-Qubit State Estimation (Fidelity vs Shots)",
    filename="fidelity_vs_shots.png",
    logy=False,
)

plot_metric(
    metric_mean="infidelity_mean",
    metric_std="infidelity_std",
    ylabel="Mean Infidelity",
    title="Exp03 Upgraded: Generalized 1-Qubit State Estimation (Infidelity vs Shots)",
    filename="infidelity_vs_shots.png",
    logy=True,
)


# RL allocation plot
rl_alloc = agg[agg["method"] == "RL_adaptive"].sort_values("N")
plt.figure()
plt.plot(rl_alloc["N"], rl_alloc["x_shots_mean"], marker="o", label="X shots")
plt.plot(rl_alloc["N"], rl_alloc["y_shots_mean"], marker="o", label="Y shots")
plt.plot(rl_alloc["N"], rl_alloc["z_shots_mean"], marker="o", label="Z shots")
plt.xscale("log")
plt.xlabel("Shots (N)")
plt.ylabel("Mean Number of Measurements")
plt.title("Exp03 Upgraded: RL Measurement Allocation")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_dir}/rl_allocation_vs_shots.png", dpi=200)
plt.close()


# ========================
# Notes file
# ========================
with open(f"{run_dir}/notes.txt", "w") as f:
    f.write("Experiment: Exp03 - upgraded generalized one-qubit adaptive measurement\n")
    f.write("State family: pure one-qubit states sampled uniformly on the Bloch sphere\n")
    f.write("Methods: Z_only, ZX_split, XYZ_split, RL_adaptive\n")
    f.write("RL actions: Z, X, Y\n")
    f.write("RL update: one-step TD/Q-learning with linear function approximation\n")
    f.write("RL reward: fidelity improvement minus small allocation imbalance penalty\n")
    f.write(f"Shot budgets: {shot_budgets}\n")
    f.write(f"Training episodes per shot budget: {num_train_episodes}\n")
    f.write(f"Test targets: {num_test_targets}\n")
    f.write(f"Test seeds: {num_test_seeds}\n")
    f.write(f"Epsilon schedule: {epsilon_start} -> {epsilon_end}\n")
    f.write("Evaluation epsilon: 0.0\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Gamma: {gamma}\n")
    f.write(f"Imbalance penalty weight: {imbalance_weight}\n")


print("Saved results to:", run_dir)
print("Files created:")
print("- metrics.csv")
print("- summary.csv")
print("- fidelity_vs_shots.png")
print("- infidelity_vs_shots.png")
print("- rl_allocation_vs_shots.png")
print("- notes.txt")