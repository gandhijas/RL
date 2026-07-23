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
run_dir = f"results/{timestamp}_exp06_budgetstress_corrected"
os.makedirs(run_dir, exist_ok=True)

# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
        dtype=complex,
    )


def sample_uniform_qubit_state(rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
    """Sample a pure qubit uniformly from the Bloch sphere."""
    z = rng.uniform(-1.0, 1.0)
    theta = float(np.arccos(z))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    return state_from_angles(theta, phi), theta, phi


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
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
# Bloch estimation
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
    theta_hat = float(np.arccos(np.clip(z_n, -1.0, 1.0)))
    phi_hat = float(np.mod(np.arctan2(y_n, x_n), 2 * np.pi))
    return state_from_angles(theta_hat, phi_hat)


def estimate_state_from_counts(
    z_counts: np.ndarray,
    x_counts: np.ndarray,
    y_counts: np.ndarray,
) -> np.ndarray:
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)
    return bloch_to_state(x_hat, y_hat, z_hat)


def fresh_counts() -> Dict[str, np.ndarray]:
    """Return independent count arrays. Avoids the shared-array aliasing bug."""
    return {
        "Z": np.array([0, 0], dtype=int),
        "X": np.array([0, 0], dtype=int),
        "Y": np.array([0, 0], dtype=int),
    }


# ========================
# RL setup
# ========================
ACTIONS = [
    ("Z", "Z"), ("Z", "X"), ("Z", "Y"),
    ("X", "Z"), ("X", "X"), ("X", "Y"),
    ("Y", "Z"), ("Y", "X"), ("Y", "Y"),
]
STATE_DIM = 14


def build_state(
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    x1_hat, y1_hat, z1_hat = estimate_bloch_from_counts(
        counts1["Z"], counts1["X"], counts1["Y"]
    )
    x2_hat, y2_hat, z2_hat = estimate_bloch_from_counts(
        counts2["Z"], counts2["X"], counts2["Y"]
    )

    total1 = sum(int(np.sum(counts1[b])) for b in ["Z", "X", "Y"])
    total2 = sum(int(np.sum(counts2[b])) for b in ["Z", "X", "Y"])

    frac_z1 = np.sum(counts1["Z"]) / total1 if total1 > 0 else 0.0
    frac_x1 = np.sum(counts1["X"]) / total1 if total1 > 0 else 0.0
    frac_y1 = np.sum(counts1["Y"]) / total1 if total1 > 0 else 0.0
    frac_z2 = np.sum(counts2["Z"]) / total2 if total2 > 0 else 0.0
    frac_x2 = np.sum(counts2["X"]) / total2 if total2 > 0 else 0.0
    frac_y2 = np.sum(counts2["Y"]) / total2 if total2 > 0 else 0.0

    progress = shots_used / total_shots if total_shots > 0 else 0.0
    return np.array(
        [
            x1_hat, y1_hat, z1_hat,
            x2_hat, y2_hat, z2_hat,
            frac_z1, frac_x1, frac_y1,
            frac_z2, frac_x2, frac_y2,
            progress, 1.0,
        ],
        dtype=float,
    )


def epsilon_greedy(
    state: np.ndarray,
    weights: Dict[Tuple[str, str], np.ndarray],
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[str, str]:
    if rng.random() < epsilon:
        return ACTIONS[int(rng.integers(len(ACTIONS)))]

    q_vals = np.array([np.dot(weights[a], state) for a in ACTIONS], dtype=float)
    max_q = np.max(q_vals)
    best = np.flatnonzero(np.isclose(q_vals, max_q))
    return ACTIONS[int(rng.choice(best))]


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
    lr: float,
):
    if total_shots <= 0:
        raise ValueError("total_shots must be positive")

    counts1 = fresh_counts()
    counts2 = fresh_counts()
    shots_used = 0

    # Warm start uses at most the available budget; it does not add extra shots.
    for action in [("Z", "Z"), ("X", "X"), ("Y", "Y")]:
        if shots_used >= total_shots:
            break
        apply_joint_measurement(psi1, psi2, action, counts1, counts2, rng)
        shots_used += 1

    f_prev = product_fidelity_from_counts(psi1, psi2, counts1, counts2)

    for t in range(shots_used, total_shots):
        state = build_state(counts1, counts2, t, total_shots)
        action = epsilon_greedy(state, weights, epsilon, rng)
        apply_joint_measurement(psi1, psi2, action, counts1, counts2, rng)

        f_new = product_fidelity_from_counts(psi1, psi2, counts1, counts2)
        reward = f_new - f_prev
        if update_weights:
            weights[action] += lr * reward * state
        f_prev = f_new

    return {"fidelity": f_prev, "infidelity": 1.0 - f_prev}, weights


# ========================
# Fixed strategies
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
        schedule = [("Z", "Z")] * n_z + [("X", "X")] * (total_shots - n_z)
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
# Experiment setup
# ========================
stress_shot_budgets = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
training_budgets = stress_shot_budgets  # train a policy for every evaluated budget
num_test_targets = 50
num_test_seeds = 5
epsilon_train = 0.15
epsilon_test = 0.0
learning_rate = 0.02
num_train_episodes = 800
master_rng = np.random.default_rng(123)


# ========================
# Train RL
# ========================
trained_weights = {}
for N in training_budgets:
    weights = {a: np.zeros(STATE_DIM, dtype=float) for a in ACTIONS}
    for ep in range(num_train_episodes):
        psi1, _, _ = sample_uniform_qubit_state(master_rng)
        psi2, _, _ = sample_uniform_qubit_state(master_rng)
        rng = np.random.default_rng(10_000_000 + 10_000 * N + ep)
        _, weights = run_rl_episode(
            psi1, psi2, N, weights, epsilon_train, rng, True, learning_rate
        )
    trained_weights[N] = {a: w.copy() for a, w in weights.items()}
    print(f"Finished training N={N}")


# ========================
# Run stress test
# ========================
stress_rows = []
for target_id in range(num_test_targets):
    psi1, theta1, phi1 = sample_uniform_qubit_state(master_rng)
    psi2, theta2, phi2 = sample_uniform_qubit_state(master_rng)

    for seed in range(num_test_seeds):
        for N in stress_shot_budgets:
            base_seed = target_id * 1_000_000 + seed * 10_000 + N * 10

            for method_index, strategy in enumerate(["Z_only", "ZX_split", "XYZ_split"]):
                rng = np.random.default_rng(base_seed + method_index + 1)
                out = run_fixed_strategy_episode(psi1, psi2, N, strategy, rng)
                stress_rows.append(
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
                psi1,
                psi2,
                N,
                {a: w.copy() for a, w in trained_weights[N].items()},
                epsilon_test,
                rng_rl,
                False,
                learning_rate,
            )
            stress_rows.append(
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
# Save summaries
# ========================
stress_df = pd.DataFrame(stress_rows)
stress_df.to_csv(f"{run_dir}/stress_metrics.csv", index=False)

stress_agg = (
    stress_df.groupby(["method", "N"])
    .agg(
        fidelity_mean=("fidelity", "mean"),
        fidelity_std=("fidelity", "std"),
        fidelity_count=("fidelity", "count"),
        infidelity_mean=("infidelity", "mean"),
        infidelity_std=("infidelity", "std"),
        infidelity_count=("infidelity", "count"),
    )
    .reset_index()
)
stress_agg["fidelity_ci95"] = 1.96 * stress_agg["fidelity_std"] / np.sqrt(stress_agg["fidelity_count"])
stress_agg["infidelity_ci95"] = 1.96 * stress_agg["infidelity_std"] / np.sqrt(stress_agg["infidelity_count"])
stress_agg.to_csv(f"{run_dir}/stress_summary.csv", index=False)


# ========================
# Plot helpers
# ========================
def plot_metric(
    mean_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    filename: str,
    log_y: bool = False,
) -> None:
    plt.figure(figsize=(10, 7))
    for method in stress_agg["method"].unique():
        sub = stress_agg[stress_agg["method"] == method].sort_values("N")
        x = sub["N"].to_numpy()
        y = sub[mean_col].to_numpy()
        ci = sub[ci_col].to_numpy()

        lower = y - ci
        upper = y + ci
        if "fidelity" in mean_col and "infidelity" not in mean_col:
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        if log_y:
            lower = np.maximum(lower, 1e-12)

        plt.plot(x, y, marker="o", linewidth=2, label=method)
        plt.fill_between(x, lower, upper, alpha=0.2)

    plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("Shots (N)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/{filename}", dpi=200)
    plt.close()


plot_metric(
    "infidelity_mean",
    "infidelity_ci95",
    "Mean Infidelity",
    "2-Qubit RL vs Fixed Strategies - Budget Stress Test",
    "stress_infidelity.png",
    log_y=True,
)

plot_metric(
    "fidelity_mean",
    "fidelity_ci95",
    "Mean Fidelity",
    "2-Qubit RL vs Fixed Strategies - Budget Stress Test",
    "stress_fidelity.png",
    log_y=False,
)


# ========================
# Paired RL-vs-XYZ analysis
# ========================
paired = (
    stress_df[stress_df["method"].isin(["RL_adaptive", "XYZ_split"])]
    .pivot_table(
        index=["target_id", "seed", "N"],
        columns="method",
        values="fidelity",
    )
    .dropna()
    .reset_index()
)
paired["delta_fidelity"] = paired["RL_adaptive"] - paired["XYZ_split"]
paired.to_csv(f"{run_dir}/rl_vs_xyz_paired_metrics.csv", index=False)

paired_summary = (
    paired.groupby("N")
    .agg(
        delta_mean=("delta_fidelity", "mean"),
        delta_std=("delta_fidelity", "std"),
        count=("delta_fidelity", "count"),
        rl_win_rate=("delta_fidelity", lambda x: float(np.mean(x > 0))),
    )
    .reset_index()
)
paired_summary["delta_ci95"] = 1.96 * paired_summary["delta_std"] / np.sqrt(paired_summary["count"])
paired_summary.to_csv(f"{run_dir}/rl_vs_xyz_paired_summary.csv", index=False)

plt.figure(figsize=(10, 7))
x = paired_summary["N"].to_numpy()
y = paired_summary["delta_mean"].to_numpy()
ci = paired_summary["delta_ci95"].to_numpy()
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.plot(x, y, marker="o", linewidth=2)
plt.fill_between(x, y - ci, y + ci, alpha=0.2)
plt.xscale("log")
plt.xlabel("Shots (N)")
plt.ylabel(r"$\Delta F = F_{RL} - F_{XYZ}$")
plt.title("Paired RL Advantage over XYZ - Budget Stress Test")
plt.tight_layout()
plt.savefig(f"{run_dir}/rl_vs_xyz_delta_fidelity.png", dpi=200)
plt.close()


# ========================
# Notes
# ========================
with open(f"{run_dir}/notes.txt", "w", encoding="utf-8") as f:
    f.write("Experiment: Exp06 corrected budget stress test\n")
    f.write("Important correction: all X/Y/Z count arrays are independent.\n")
    f.write("Training and test states are sampled uniformly on the Bloch sphere.\n")
    f.write("One RL policy is trained for every evaluated shot budget.\n")
    f.write("Warm-start measurements count toward the stated shot budget.\n")
    f.write("Main plots use 95% confidence intervals for the mean.\n")
    f.write("Paired RL-vs-XYZ fidelity differences and win rates are saved.\n")

print("Stress test completed. Results saved to:", run_dir)
print("Files created:")
print("- stress_metrics.csv")
print("- stress_summary.csv")
print("- stress_infidelity.png")
print("- stress_fidelity.png")
print("- rl_vs_xyz_paired_metrics.csv")
print("- rl_vs_xyz_paired_summary.csv")
print("- rl_vs_xyz_delta_fidelity.png")
print("- notes.txt")