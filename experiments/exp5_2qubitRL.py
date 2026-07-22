import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp05_2qubitRL_upgraded"
os.makedirs(run_dir, exist_ok=True)


# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """Single-qubit pure state."""
    return np.array(
        [
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2),
        ],
        dtype=complex,
    )


def sample_uniform_qubit_state(
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    """Sample a pure qubit state uniformly from the Bloch sphere."""
    z = rng.uniform(-1.0, 1.0)
    theta = float(np.arccos(z))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    return state_from_angles(theta, phi), theta, phi


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """Pure-state fidelity |<psi|phi>|^2."""
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def hadamard() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def s_dagger() -> np.ndarray:
    return np.array([[1, 0], [0, -1j]], dtype=complex)


# ========================
# Measurement utilities
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


def probs_for_basis(psi: np.ndarray, basis: str) -> np.ndarray:
    if basis == "Z":
        return measure_probs_z(psi)
    if basis == "X":
        return measure_probs_x(psi)
    if basis == "Y":
        return measure_probs_y(psi)
    raise ValueError(f"Unknown basis: {basis}")


def sample_one_shot(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice([0, 1], p=probs))


# ========================
# Estimation utilities
# ========================
def estimate_bloch_from_counts(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    y_counts: np.ndarray,
) -> Tuple[float, float, float]:
    def coord(counts: np.ndarray) -> float:
        total = int(np.sum(counts))
        if total == 0:
            return 0.0
        return float((counts[0] - counts[1]) / total)

    return coord(x_counts), coord(y_counts), coord(z_counts)


def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
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
# RL setup
# ========================
ACTIONS = [
    ("Z", "Z"), ("Z", "X"), ("Z", "Y"),
    ("X", "Z"), ("X", "X"), ("X", "Y"),
    ("Y", "Z"), ("Y", "X"), ("Y", "Y"),
]
ACTION_NAMES = [a + b for a, b in ACTIONS]
ACTION_INDEX = {action: i for i, action in enumerate(ACTIONS)}

# 6 Bloch estimates + 6 uncertainties + 9 action fractions +
# 9 action deficits + progress + imbalance + bias = 33
STATE_DIM = 33


def fresh_counts() -> Dict[str, np.ndarray]:
    """Create independent count arrays for X/Y/Z."""
    return {
        "Z": np.array([0, 0], dtype=int),
        "X": np.array([0, 0], dtype=int),
        "Y": np.array([0, 0], dtype=int),
    }


def measurement_uncertainty(counts: np.ndarray) -> float:
    """Simple count-based uncertainty proxy with finite value at zero shots."""
    total = int(np.sum(counts))
    return float(1.0 / np.sqrt(total + 1.0))


def build_state(
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
    action_counts: np.ndarray,
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    x1_hat, y1_hat, z1_hat = estimate_bloch_from_counts(
        counts1["Z"], counts1["X"], counts1["Y"]
    )
    x2_hat, y2_hat, z2_hat = estimate_bloch_from_counts(
        counts2["Z"], counts2["X"], counts2["Y"]
    )

    uncertainties = np.array(
        [
            measurement_uncertainty(counts1["X"]),
            measurement_uncertainty(counts1["Y"]),
            measurement_uncertainty(counts1["Z"]),
            measurement_uncertainty(counts2["X"]),
            measurement_uncertainty(counts2["Y"]),
            measurement_uncertainty(counts2["Z"]),
        ],
        dtype=float,
    )

    progress = shots_used / total_shots if total_shots > 0 else 0.0
    action_fracs = action_counts.astype(float) / total_shots
    target_frac = progress / len(ACTIONS)
    action_deficits = target_frac - action_fracs

    used = int(np.sum(action_counts))
    if used > 0:
        current_fracs = action_counts / used
        imbalance = float(np.sum((current_fracs - 1.0 / len(ACTIONS)) ** 2))
    else:
        imbalance = 0.0

    state = np.concatenate(
        [
            np.array(
                [x1_hat, y1_hat, z1_hat, x2_hat, y2_hat, z2_hat],
                dtype=float,
            ),
            uncertainties,
            action_fracs,
            action_deficits,
            np.array([progress, imbalance, 1.0], dtype=float),
        ]
    )

    if state.shape[0] != STATE_DIM:
        raise RuntimeError(f"State dimension mismatch: {state.shape[0]} != {STATE_DIM}")
    return state


def q_values(state: np.ndarray, weights: Dict[Tuple[str, str], np.ndarray]) -> np.ndarray:
    return np.array([float(np.dot(weights[a], state)) for a in ACTIONS], dtype=float)


def epsilon_greedy(
    state: np.ndarray,
    weights: Dict[Tuple[str, str], np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[str, str]:
    if rng.random() < epsilon:
        return ACTIONS[int(rng.integers(len(ACTIONS)))]

    values = q_values(state, weights)
    max_q = np.max(values)
    best_indices = np.flatnonzero(np.isclose(values, max_q))
    return ACTIONS[int(rng.choice(best_indices))]


def initialize_weights() -> Dict[Tuple[str, str], np.ndarray]:
    """Small informative initialization favoring uncertainty and deficits."""
    weights = {}
    for action in ACTIONS:
        w = np.zeros(STATE_DIM, dtype=float)
        idx = ACTION_INDEX[action]

        # Basis-specific uncertainty positions.
        b1, b2 = action
        unc_idx_q1 = {"X": 6, "Y": 7, "Z": 8}[b1]
        unc_idx_q2 = {"X": 9, "Y": 10, "Z": 11}[b2]
        w[unc_idx_q1] = 0.05
        w[unc_idx_q2] = 0.05

        # Action deficit feature starts at index 21.
        w[21 + idx] = 0.05
        weights[action] = w
    return weights


# ========================
# Episode helpers
# ========================
def apply_joint_measurement(
    psi1: np.ndarray,
    psi2: np.ndarray,
    action: Tuple[str, str],
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    b1, b2 = action
    o1 = sample_one_shot(probs_for_basis(psi1, b1), rng)
    o2 = sample_one_shot(probs_for_basis(psi2, b2), rng)
    counts1[b1][o1] += 1
    counts2[b2][o2] += 1


def product_fidelity_from_counts(
    psi1: np.ndarray,
    psi2: np.ndarray,
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
) -> float:
    psi1_hat = estimate_state_from_counts(counts1["Z"], counts1["X"], counts1["Y"])
    psi2_hat = estimate_state_from_counts(counts2["Z"], counts2["X"], counts2["Y"])
    return fidelity(psi1_hat, psi1) * fidelity(psi2_hat, psi2)


# ========================
# RL episode
# ========================
def run_rl_episode(
    psi1: np.ndarray,
    psi2: np.ndarray,
    total_shots: int,
    weights: Dict[Tuple[str, str], np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
    update_weights: bool,
    learning_rate: float,
    gamma: float,
    imbalance_weight: float,
):
    if total_shots <= 0:
        raise ValueError("total_shots must be positive")

    counts1 = fresh_counts()
    counts2 = fresh_counts()
    action_counts = np.zeros(len(ACTIONS), dtype=int)
    shots_used = 0

    # Warm start: one matched-basis joint measurement in ZZ, XX, and YY.
    for action in [("Z", "Z"), ("X", "X"), ("Y", "Y")]:
        if shots_used >= total_shots:
            break
        apply_joint_measurement(psi1, psi2, action, counts1, counts2, rng)
        action_counts[ACTION_INDEX[action]] += 1
        shots_used += 1

    f_prev = product_fidelity_from_counts(psi1, psi2, counts1, counts2)

    for shot_idx in range(shots_used, total_shots):
        state = build_state(counts1, counts2, action_counts, shot_idx, total_shots)
        action = epsilon_greedy(state, weights, epsilon, rng)
        action_idx = ACTION_INDEX[action]

        q_current = float(np.dot(weights[action], state))

        apply_joint_measurement(psi1, psi2, action, counts1, counts2, rng)
        action_counts[action_idx] += 1

        f_new = product_fidelity_from_counts(psi1, psi2, counts1, counts2)
        fidelity_gain = f_new - f_prev

        used = int(np.sum(action_counts))
        current_fracs = action_counts / used
        imbalance = float(np.sum((current_fracs - 1.0 / len(ACTIONS)) ** 2))
        reward = fidelity_gain - imbalance_weight * imbalance

        terminal = shot_idx == total_shots - 1
        if terminal:
            q_next = 0.0
        else:
            next_state = build_state(
                counts1, counts2, action_counts, shot_idx + 1, total_shots
            )
            q_next = float(np.max(q_values(next_state, weights)))

        if update_weights:
            td_target = reward + gamma * q_next
            td_error = td_target - q_current
            weights[action] += learning_rate * td_error * state

        f_prev = f_new

    result = {
        "fidelity": f_prev,
        "infidelity": 1.0 - f_prev,
    }
    for i, name in enumerate(ACTION_NAMES):
        result[f"shots_{name}"] = int(action_counts[i])

    return result, weights


# ========================
# Fixed strategy episode
# ========================
def run_fixed_strategy_episode(
    psi1: np.ndarray,
    psi2: np.ndarray,
    total_shots: int,
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    counts1 = fresh_counts()
    counts2 = fresh_counts()

    if strategy == "Z_only":
        schedule = [("Z", "Z")] * total_shots
    elif strategy == "ZX_split":
        n_z = total_shots // 2
        n_x = total_shots - n_z
        schedule = [("Z", "Z")] * n_z + [("X", "X")] * n_x
    elif strategy == "XYZ_split":
        n_z = total_shots // 3
        n_x = total_shots // 3
        n_y = total_shots - n_z - n_x
        schedule = (
            [("Z", "Z")] * n_z
            + [("X", "X")] * n_x
            + [("Y", "Y")] * n_y
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for action in schedule:
        apply_joint_measurement(psi1, psi2, action, counts1, counts2, rng)

    f_total = product_fidelity_from_counts(psi1, psi2, counts1, counts2)
    return {"fidelity": f_total, "infidelity": 1.0 - f_total}


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
imbalance_weight = 0.005

master_rng = np.random.default_rng(123)


# ========================
# Train one policy per shot budget
# ========================
trained_weights = {}

for N in shot_budgets:
    weights = initialize_weights()

    for ep in range(num_train_episodes):
        psi1, _, _ = sample_uniform_qubit_state(master_rng)
        psi2, _, _ = sample_uniform_qubit_state(master_rng)

        frac = ep / max(num_train_episodes - 1, 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        rng = np.random.default_rng(10_000_000 + 10_000 * N + ep)
        _, weights = run_rl_episode(
            psi1=psi1,
            psi2=psi2,
            total_shots=N,
            weights=weights,
            epsilon=epsilon,
            rng=rng,
            update_weights=True,
            learning_rate=learning_rate,
            gamma=gamma,
            imbalance_weight=imbalance_weight,
        )

    trained_weights[N] = {a: w.copy() for a, w in weights.items()}
    print(f"Finished training RL policy for N = {N}")


# ========================
# Evaluation
# ========================
rows = []

for target_id in range(num_test_targets):
    psi1, theta1, phi1 = sample_uniform_qubit_state(master_rng)
    psi2, theta2, phi2 = sample_uniform_qubit_state(master_rng)

    for seed in range(num_test_seeds):
        for N in shot_budgets:
            base_seed = target_id * 1_000_000 + seed * 10_000 + N * 10

            for method_index, strategy in enumerate(["Z_only", "ZX_split", "XYZ_split"]):
                rng = np.random.default_rng(base_seed + method_index + 1)
                out = run_fixed_strategy_episode(
                    psi1=psi1,
                    psi2=psi2,
                    total_shots=N,
                    strategy=strategy,
                    rng=rng,
                )
                rows.append(
                    {
                        "target_id": target_id,
                        "seed": seed,
                        "N": N,
                        "method": strategy,
                        "theta1": theta1,
                        "phi1": phi1,
                        "theta2": theta2,
                        "phi2": phi2,
                        **out,
                    }
                )

            rng_rl = np.random.default_rng(base_seed + 4)
            out_rl, _ = run_rl_episode(
                psi1=psi1,
                psi2=psi2,
                total_shots=N,
                weights={a: w.copy() for a, w in trained_weights[N].items()},
                epsilon=epsilon_test,
                rng=rng_rl,
                update_weights=False,
                learning_rate=learning_rate,
                gamma=gamma,
                imbalance_weight=imbalance_weight,
            )
            rows.append(
                {
                    "target_id": target_id,
                    "seed": seed,
                    "N": N,
                    "method": "RL_adaptive",
                    "theta1": theta1,
                    "phi1": phi1,
                    "theta2": theta2,
                    "phi2": phi2,
                    **out_rl,
                }
            )


# ========================
# Save raw metrics and summary
# ========================
df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/metrics.csv", index=False)

summary = (
    df.groupby(["method", "N"])
    .agg(
        fidelity_mean=("fidelity", "mean"),
        fidelity_std=("fidelity", "std"),
        infidelity_mean=("infidelity", "mean"),
        infidelity_std=("infidelity", "std"),
    )
    .reset_index()
)
summary.to_csv(f"{run_dir}/summary.csv", index=False)

allocation_cols = [f"shots_{name}" for name in ACTION_NAMES]
rl_df = df[df["method"] == "RL_adaptive"].copy()
allocation_summary = (
    rl_df.groupby("N")[allocation_cols]
    .mean()
    .reset_index()
)
allocation_summary.to_csv(f"{run_dir}/rl_allocation_summary.csv", index=False)


# ========================
# Plot helpers
# ========================
def plot_metric(
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    filename: str,
    logy: bool = False,
) -> None:
    plt.figure()
    for method in summary["method"].unique():
        sub = summary[summary["method"] == method].sort_values("N")
        x = sub["N"].to_numpy()
        y = sub[metric_mean].to_numpy()
        y_std = sub[metric_std].to_numpy()

        lower = y - y_std
        upper = y + y_std
        if logy:
            lower = np.maximum(lower, 1e-12)

        plt.plot(x, y, marker="o", label=method)
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
    title="Exp05 Upgraded: 2-Qubit Product-State Estimation (Fidelity vs Shots)",
    filename="fidelity_vs_shots.png",
    logy=False,
)

plot_metric(
    metric_mean="infidelity_mean",
    metric_std="infidelity_std",
    ylabel="Mean Infidelity",
    title="Exp05 Upgraded: 2-Qubit Product-State Estimation (Infidelity vs Shots)",
    filename="infidelity_vs_shots.png",
    logy=True,
)

plt.figure()
for name in ACTION_NAMES:
    plt.plot(
        allocation_summary["N"],
        allocation_summary[f"shots_{name}"],
        marker="o",
        label=name,
    )
plt.xscale("log")
plt.xlabel("Shots (N)")
plt.ylabel("Mean RL Measurements")
plt.title("Exp05 Upgraded: RL Joint-Action Allocation")
plt.legend(ncol=3)
plt.tight_layout()
plt.savefig(f"{run_dir}/rl_allocation_vs_shots.png", dpi=200)
plt.close()


# ========================
# Notes
# ========================
with open(f"{run_dir}/notes.txt", "w", encoding="utf-8") as f:
    f.write("Experiment: Exp05 - upgraded two-qubit RL\n")
    f.write("Scope: two-qubit product-state estimation\n")
    f.write("Actions: ZZ, ZX, ZY, XZ, XX, XY, YZ, YX, YY\n")
    f.write("Warm start: ZZ, XX, YY\n")
    f.write("Learning: one-step TD learning\n")
    f.write(f"Gamma: {gamma}\n")
    f.write(f"Training episodes per shot budget: {num_train_episodes}\n")
    f.write(f"Epsilon schedule: {epsilon_start} to {epsilon_end}\n")
    f.write(f"Evaluation epsilon: {epsilon_test}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Imbalance penalty weight: {imbalance_weight}\n")
    f.write(f"Shot budgets: {shot_budgets}\n")
    f.write(f"Test targets: {num_test_targets}\n")
    f.write(f"Test seeds: {num_test_seeds}\n")
    f.write("Training and test states sampled uniformly on each Bloch sphere.\n")

print("Saved results to:", run_dir)
print("Files created:")
print("- metrics.csv")
print("- summary.csv")
print("- rl_allocation_summary.csv")
print("- fidelity_vs_shots.png")
print("- infidelity_vs_shots.png")
print("- rl_allocation_vs_shots.png")
print("- notes.txt")