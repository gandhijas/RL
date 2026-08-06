import os
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Experiment 6: Audited two-qubit RL budget stress test
#
# Scientific setup preserved:
# - two independently sampled pure qubits
# - uniform Bloch-sphere state distribution
# - nine joint Pauli actions
# - sequential RL measurement selection
# - Z-only, ZX-split, and XYZ-split fixed baselines
# - final product-state fidelity as the evaluation objective
#
# Audit upgrades:
# - independent count arrays
# - audited 35-feature RL state
# - TD learning with terminal-fidelity bonus
# - one policy trained for every tested budget
# - multiple independent restarts
# - validation-based policy selection
# - paired RL-versus-XYZ analysis
# - fidelity, infidelity, delta-fidelity, and win-rate plots
# ============================================================


# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp06_budgetstress_audited"
os.makedirs(run_dir, exist_ok=True)


# ========================
# Configuration
# ========================
STRESS_SHOT_BUDGETS = [1, 2, 5, 10, 25, 50, 100, 250, 500]

NUM_TRAIN_EPISODES = 2500
NUM_RESTARTS = 3

NUM_VALIDATION_TARGETS = 25
NUM_VALIDATION_SEEDS = 2

NUM_TEST_TARGETS = 100
NUM_TEST_SEEDS = 5

EPSILON_START = 0.35
EPSILON_END = 0.02
EPSILON_TEST = 0.0

LEARNING_RATE = 0.006
GAMMA = 0.95
TERMINAL_BONUS_WEIGHT = 0.20

MASTER_SEED = 123


# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            np.cos(theta / 2.0),
            np.exp(1j * phi) * np.sin(theta / 2.0),
        ],
        dtype=complex,
    )


def sample_uniform_qubit_state(
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float]:
    z = rng.uniform(-1.0, 1.0)
    theta = float(np.arccos(z))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    return state_from_angles(theta, phi), theta, phi


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def hadamard() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array(
        [[1, 1], [1, -1]],
        dtype=complex,
    )


def s_dagger() -> np.ndarray:
    return np.array(
        [[1, 0], [0, -1j]],
        dtype=complex,
    )


def probs_for_basis(psi: np.ndarray, basis: str) -> np.ndarray:
    if basis == "Z":
        rotated = psi
    elif basis == "X":
        rotated = hadamard() @ psi
    elif basis == "Y":
        rotated = hadamard() @ (s_dagger() @ psi)
    else:
        raise ValueError(f"Unknown basis: {basis}")

    probabilities = np.abs(rotated) ** 2
    return probabilities / np.sum(probabilities)


def sample_one_shot(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> int:
    return int(rng.choice([0, 1], p=probabilities))


# ========================
# Estimation utilities
# ========================
def fresh_counts() -> Dict[str, np.ndarray]:
    return {
        "X": np.zeros(2, dtype=int),
        "Y": np.zeros(2, dtype=int),
        "Z": np.zeros(2, dtype=int),
    }


def coordinate_from_counts(counts: np.ndarray) -> float:
    total = int(np.sum(counts))
    if total == 0:
        return 0.0
    return float((counts[0] - counts[1]) / total)


def estimate_bloch_from_counts(
    counts: Dict[str, np.ndarray],
) -> Tuple[float, float, float]:
    return (
        coordinate_from_counts(counts["X"]),
        coordinate_from_counts(counts["Y"]),
        coordinate_from_counts(counts["Z"]),
    )


def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
    vector = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=complex)

    x_n, y_n, z_n = vector / norm
    z_n = float(np.clip(z_n, -1.0, 1.0))

    theta_hat = float(np.arccos(z_n))
    phi_hat = float(np.mod(np.arctan2(y_n, x_n), 2.0 * np.pi))
    return state_from_angles(theta_hat, phi_hat)


def estimate_state_from_counts(
    counts: Dict[str, np.ndarray],
) -> np.ndarray:
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(counts)
    return bloch_to_state(x_hat, y_hat, z_hat)


def product_fidelity_from_counts(
    psi1: np.ndarray,
    psi2: np.ndarray,
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
) -> float:
    psi1_hat = estimate_state_from_counts(counts1)
    psi2_hat = estimate_state_from_counts(counts2)
    return fidelity(psi1_hat, psi1) * fidelity(psi2_hat, psi2)


# ========================
# RL setup
# ========================
ACTIONS = [
    ("Z", "Z"), ("Z", "X"), ("Z", "Y"),
    ("X", "Z"), ("X", "X"), ("X", "Y"),
    ("Y", "Z"), ("Y", "X"), ("Y", "Y"),
]

ACTION_NAMES = [first + second for first, second in ACTIONS]
ACTION_INDEX = {
    action: index
    for index, action in enumerate(ACTIONS)
}

# State:
# 6 local Bloch estimates
# 6 local uncertainty values
# 6 local basis fractions
# 6 local basis deficits
# 9 joint-action fractions
# progress + bias
STATE_DIM = 35


def measurement_uncertainty(counts: np.ndarray) -> float:
    total = int(np.sum(counts))
    return float(1.0 / np.sqrt(total + 1.0))


def local_basis_totals(
    counts: Dict[str, np.ndarray],
) -> np.ndarray:
    return np.array(
        [
            np.sum(counts["X"]),
            np.sum(counts["Y"]),
            np.sum(counts["Z"]),
        ],
        dtype=float,
    )


def build_state(
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
    action_counts: np.ndarray,
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    x1, y1, z1 = estimate_bloch_from_counts(counts1)
    x2, y2, z2 = estimate_bloch_from_counts(counts2)

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

    totals1 = local_basis_totals(counts1)
    totals2 = local_basis_totals(counts2)

    local_fractions1 = totals1 / max(total_shots, 1)
    local_fractions2 = totals2 / max(total_shots, 1)

    progress = shots_used / total_shots if total_shots > 0 else 0.0
    local_target = progress / 3.0

    local_deficits1 = local_target - local_fractions1
    local_deficits2 = local_target - local_fractions2

    joint_action_fractions = (
        action_counts.astype(float) / max(total_shots, 1)
    )

    state = np.concatenate(
        [
            np.array([x1, y1, z1, x2, y2, z2], dtype=float),
            uncertainties,
            local_fractions1,
            local_fractions2,
            local_deficits1,
            local_deficits2,
            joint_action_fractions,
            np.array([progress, 1.0], dtype=float),
        ]
    )

    if state.shape[0] != STATE_DIM:
        raise RuntimeError(
            f"State dimension mismatch: {state.shape[0]} != {STATE_DIM}"
        )

    return state


def initialize_weights() -> Dict[Tuple[str, str], np.ndarray]:
    uncertainty_q1 = {"X": 6, "Y": 7, "Z": 8}
    uncertainty_q2 = {"X": 9, "Y": 10, "Z": 11}

    deficit_q1 = {"X": 18, "Y": 19, "Z": 20}
    deficit_q2 = {"X": 21, "Y": 22, "Z": 23}

    weights: Dict[Tuple[str, str], np.ndarray] = {}

    for action in ACTIONS:
        first_basis, second_basis = action
        vector = np.zeros(STATE_DIM, dtype=float)

        vector[uncertainty_q1[first_basis]] = 0.06
        vector[uncertainty_q2[second_basis]] = 0.06
        vector[deficit_q1[first_basis]] = 0.03
        vector[deficit_q2[second_basis]] = 0.03

        weights[action] = vector

    return weights


def q_values(
    state: np.ndarray,
    weights: Dict[Tuple[str, str], np.ndarray],
) -> np.ndarray:
    return np.array(
        [
            float(np.dot(weights[action], state))
            for action in ACTIONS
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

    values = q_values(state, weights)
    maximum = np.max(values)
    candidates = np.flatnonzero(np.isclose(values, maximum))
    return ACTIONS[int(rng.choice(candidates))]


def apply_joint_measurement(
    psi1: np.ndarray,
    psi2: np.ndarray,
    action: Tuple[str, str],
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    basis1, basis2 = action

    outcome1 = sample_one_shot(
        probs_for_basis(psi1, basis1),
        rng,
    )
    outcome2 = sample_one_shot(
        probs_for_basis(psi2, basis2),
        rng,
    )

    counts1[basis1][outcome1] += 1
    counts2[basis2][outcome2] += 1


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
) -> Tuple[dict, Dict[Tuple[str, str], np.ndarray]]:
    if total_shots <= 0:
        raise ValueError("total_shots must be positive")

    counts1 = fresh_counts()
    counts2 = fresh_counts()
    action_counts = np.zeros(len(ACTIONS), dtype=int)
    shots_used = 0

    # Warm start is fully counted inside the budget.
    for action in [("Z", "Z"), ("X", "X"), ("Y", "Y")]:
        if shots_used >= total_shots:
            break

        apply_joint_measurement(
            psi1,
            psi2,
            action,
            counts1,
            counts2,
            rng,
        )
        action_counts[ACTION_INDEX[action]] += 1
        shots_used += 1

    previous_fidelity = product_fidelity_from_counts(
        psi1,
        psi2,
        counts1,
        counts2,
    )

    for shot_index in range(shots_used, total_shots):
        state = build_state(
            counts1,
            counts2,
            action_counts,
            shot_index,
            total_shots,
        )

        action = epsilon_greedy(
            state,
            weights,
            epsilon,
            rng,
        )
        action_index = ACTION_INDEX[action]
        q_current = float(np.dot(weights[action], state))

        apply_joint_measurement(
            psi1,
            psi2,
            action,
            counts1,
            counts2,
            rng,
        )
        action_counts[action_index] += 1

        new_fidelity = product_fidelity_from_counts(
            psi1,
            psi2,
            counts1,
            counts2,
        )

        fidelity_gain = new_fidelity - previous_fidelity
        terminal = shot_index == total_shots - 1

        reward = fidelity_gain
        if terminal:
            reward += TERMINAL_BONUS_WEIGHT * new_fidelity

        if terminal:
            q_next = 0.0
        else:
            next_state = build_state(
                counts1,
                counts2,
                action_counts,
                shot_index + 1,
                total_shots,
            )
            q_next = float(np.max(q_values(next_state, weights)))

        if update_weights:
            td_target = reward + GAMMA * q_next
            td_error = td_target - q_current
            weights[action] += LEARNING_RATE * td_error * state

        previous_fidelity = new_fidelity

    result = {
        "fidelity": previous_fidelity,
        "infidelity": 1.0 - previous_fidelity,
    }

    for index, name in enumerate(ACTION_NAMES):
        result[f"shots_{name}"] = int(action_counts[index])

    return result, weights


# ========================
# Fixed strategies
# ========================
def fixed_schedule(
    total_shots: int,
    strategy: str,
) -> List[Tuple[str, str]]:
    if strategy == "Z_only":
        return [("Z", "Z")] * total_shots

    if strategy == "ZX_split":
        number_z = total_shots // 2
        number_x = total_shots - number_z
        return (
            [("Z", "Z")] * number_z
            + [("X", "X")] * number_x
        )

    if strategy == "XYZ_split":
        number_z = total_shots // 3
        number_x = total_shots // 3
        number_y = total_shots - number_z - number_x
        return (
            [("Z", "Z")] * number_z
            + [("X", "X")] * number_x
            + [("Y", "Y")] * number_y
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def run_fixed_strategy_episode(
    psi1: np.ndarray,
    psi2: np.ndarray,
    total_shots: int,
    strategy: str,
    rng: np.random.Generator,
) -> dict:
    counts1 = fresh_counts()
    counts2 = fresh_counts()

    for action in fixed_schedule(total_shots, strategy):
        apply_joint_measurement(
            psi1,
            psi2,
            action,
            counts1,
            counts2,
            rng,
        )

    total_fidelity = product_fidelity_from_counts(
        psi1,
        psi2,
        counts1,
        counts2,
    )

    return {
        "fidelity": total_fidelity,
        "infidelity": 1.0 - total_fidelity,
    }


# ========================
# Validation utility
# ========================
def evaluate_policy_on_validation_set(
    weights: Dict[Tuple[str, str], np.ndarray],
    total_shots: int,
    restart_index: int,
) -> float:
    validation_rng = np.random.default_rng(
        70_000_000 + total_shots * 10_000
    )

    validation_states = [
        (
            sample_uniform_qubit_state(validation_rng)[0],
            sample_uniform_qubit_state(validation_rng)[0],
        )
        for _ in range(NUM_VALIDATION_TARGETS)
    ]

    fidelities = []

    for target_index, (psi1, psi2) in enumerate(validation_states):
        for seed in range(NUM_VALIDATION_SEEDS):
            episode_rng = np.random.default_rng(
                80_000_000
                + total_shots * 100_000
                + restart_index * 10_000
                + target_index * 100
                + seed
            )

            output, _ = run_rl_episode(
                psi1=psi1,
                psi2=psi2,
                total_shots=total_shots,
                weights={
                    action: vector.copy()
                    for action, vector in weights.items()
                },
                epsilon=EPSILON_TEST,
                rng=episode_rng,
                update_weights=False,
            )
            fidelities.append(output["fidelity"])

    return float(np.mean(fidelities))


# ========================
# Train one policy per budget
# ========================
trained_weights = {}
training_records = []

for total_shots in STRESS_SHOT_BUDGETS:
    candidates = []

    for restart_index in range(NUM_RESTARTS):
        weights = initialize_weights()

        training_rng = np.random.default_rng(
            MASTER_SEED
            + total_shots * 100_000
            + restart_index * 10_000
        )

        for episode in range(NUM_TRAIN_EPISODES):
            psi1, _, _ = sample_uniform_qubit_state(training_rng)
            psi2, _, _ = sample_uniform_qubit_state(training_rng)

            fraction = episode / max(NUM_TRAIN_EPISODES - 1, 1)
            epsilon = EPSILON_START + fraction * (
                EPSILON_END - EPSILON_START
            )

            episode_rng = np.random.default_rng(
                10_000_000
                + total_shots * 1_000_000
                + restart_index * 100_000
                + episode
            )

            _, weights = run_rl_episode(
                psi1=psi1,
                psi2=psi2,
                total_shots=total_shots,
                weights=weights,
                epsilon=epsilon,
                rng=episode_rng,
                update_weights=True,
            )

        validation_fidelity = evaluate_policy_on_validation_set(
            weights=weights,
            total_shots=total_shots,
            restart_index=restart_index,
        )

        candidates.append(
            (
                validation_fidelity,
                {
                    action: vector.copy()
                    for action, vector in weights.items()
                },
            )
        )

        training_records.append(
            {
                "N": total_shots,
                "restart": restart_index,
                "validation_fidelity": validation_fidelity,
            }
        )

        print(
            f"N={total_shots}, restart={restart_index}, "
            f"validation fidelity={validation_fidelity:.6f}"
        )

    best_validation, best_weights = max(
        candidates,
        key=lambda item: item[0],
    )

    trained_weights[total_shots] = best_weights

    print(
        f"Selected policy for N={total_shots}: "
        f"validation fidelity={best_validation:.6f}"
    )

pd.DataFrame(training_records).to_csv(
    f"{run_dir}/training_restart_validation.csv",
    index=False,
)


# ========================
# Final stress-test evaluation
# ========================
test_rng = np.random.default_rng(MASTER_SEED + 999)

test_states = [
    (
        *sample_uniform_qubit_state(test_rng),
        *sample_uniform_qubit_state(test_rng),
    )
    for _ in range(NUM_TEST_TARGETS)
]

rows = []

for target_id, state_data in enumerate(test_states):
    psi1, theta1, phi1, psi2, theta2, phi2 = state_data

    for seed in range(NUM_TEST_SEEDS):
        for total_shots in STRESS_SHOT_BUDGETS:
            common = {
                "target_id": target_id,
                "seed": seed,
                "N": total_shots,
                "theta1": theta1,
                "phi1": phi1,
                "theta2": theta2,
                "phi2": phi2,
            }

            base_seed = (
                90_000_000
                + target_id * 1_000_000
                + seed * 10_000
                + total_shots * 10
            )

            for method_index, strategy in enumerate(
                ["Z_only", "ZX_split", "XYZ_split"]
            ):
                strategy_rng = np.random.default_rng(
                    base_seed + method_index
                )

                output = run_fixed_strategy_episode(
                    psi1=psi1,
                    psi2=psi2,
                    total_shots=total_shots,
                    strategy=strategy,
                    rng=strategy_rng,
                )

                rows.append(
                    {
                        **common,
                        "method": strategy,
                        **output,
                    }
                )

            rl_rng = np.random.default_rng(base_seed + 3)

            rl_output, _ = run_rl_episode(
                psi1=psi1,
                psi2=psi2,
                total_shots=total_shots,
                weights={
                    action: vector.copy()
                    for action, vector in trained_weights[
                        total_shots
                    ].items()
                },
                epsilon=EPSILON_TEST,
                rng=rl_rng,
                update_weights=False,
            )

            rows.append(
                {
                    **common,
                    "method": "RL_adaptive",
                    **rl_output,
                }
            )


# ========================
# Save raw metrics
# ========================
df = pd.DataFrame(rows)
df.to_csv(
    f"{run_dir}/stress_metrics.csv",
    index=False,
)


# ========================
# Aggregate summary
# ========================
summary = (
    df.groupby(["method", "N"])
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

summary["fidelity_ci95"] = (
    1.96
    * summary["fidelity_std"]
    / np.sqrt(summary["fidelity_count"])
)

summary["infidelity_ci95"] = (
    1.96
    * summary["infidelity_std"]
    / np.sqrt(summary["infidelity_count"])
)

summary.to_csv(
    f"{run_dir}/stress_summary.csv",
    index=False,
)


# ========================
# RL allocation summary
# ========================
allocation_columns = [
    f"shots_{name}"
    for name in ACTION_NAMES
]

rl_df = df[df["method"] == "RL_adaptive"].copy()

allocation_summary = (
    rl_df.groupby("N")[allocation_columns]
    .mean()
    .reset_index()
)

allocation_summary.to_csv(
    f"{run_dir}/rl_allocation_summary.csv",
    index=False,
)


# ========================
# Paired RL-versus-XYZ analysis
# ========================
paired = (
    df[df["method"].isin(["RL_adaptive", "XYZ_split"])]
    .pivot_table(
        index=["target_id", "seed", "N"],
        columns="method",
        values="fidelity",
    )
    .dropna()
    .reset_index()
)

paired["delta_fidelity"] = (
    paired["RL_adaptive"] - paired["XYZ_split"]
)

paired_summary = (
    paired.groupby("N")
    .agg(
        delta_mean=("delta_fidelity", "mean"),
        delta_std=("delta_fidelity", "std"),
        count=("delta_fidelity", "count"),
        rl_win_rate=(
            "delta_fidelity",
            lambda values: float(np.mean(values > 0.0)),
        ),
    )
    .reset_index()
)

paired_summary["delta_ci95"] = (
    1.96
    * paired_summary["delta_std"]
    / np.sqrt(paired_summary["count"])
)

paired_summary.to_csv(
    f"{run_dir}/rl_vs_xyz_paired_summary.csv",
    index=False,
)


# ========================
# Plot helpers
# ========================
def plot_metric(
    metric_mean: str,
    metric_ci: str,
    ylabel: str,
    title: str,
    filename: str,
    log_y: bool = False,
) -> None:
    plt.figure(figsize=(9, 6))

    for method in summary["method"].unique():
        method_data = summary[
            summary["method"] == method
        ].sort_values("N")

        x = method_data["N"].to_numpy()
        y = method_data[metric_mean].to_numpy()
        ci = method_data[metric_ci].to_numpy()

        lower = y - ci
        upper = y + ci

        if metric_mean.startswith("fidelity"):
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)

        if log_y:
            lower = np.maximum(lower, 1e-12)

        plt.plot(x, y, marker="o", label=method)
        plt.fill_between(x, lower, upper, alpha=0.2)

    plt.xscale("log")

    if log_y:
        plt.yscale("log")

    plt.xlabel("Shots (N)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{run_dir}/{filename}",
        dpi=200,
    )
    plt.close()


plot_metric(
    metric_mean="fidelity_mean",
    metric_ci="fidelity_ci95",
    ylabel="Mean Fidelity",
    title="Exp06 Audited: Budget Stress Test (Fidelity vs Shots)",
    filename="stress_fidelity.png",
    log_y=False,
)

plot_metric(
    metric_mean="infidelity_mean",
    metric_ci="infidelity_ci95",
    ylabel="Mean Infidelity",
    title="Exp06 Audited: Budget Stress Test (Infidelity vs Shots)",
    filename="stress_infidelity.png",
    log_y=True,
)


# RL-vs-XYZ paired difference
plt.figure(figsize=(9, 6))

x = paired_summary["N"].to_numpy()
y = paired_summary["delta_mean"].to_numpy()
ci = paired_summary["delta_ci95"].to_numpy()

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.plot(x, y, marker="o")
plt.fill_between(x, y - ci, y + ci, alpha=0.2)
plt.xscale("log")
plt.xlabel("Shots (N)")
plt.ylabel(r"$\Delta F = F_{\mathrm{RL}} - F_{\mathrm{XYZ}}$")
plt.title("Exp06 Audited: Paired RL Advantage over XYZ")
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_vs_xyz_delta_fidelity.png",
    dpi=200,
)
plt.close()


# RL win-rate plot
plt.figure(figsize=(9, 6))

plt.plot(
    paired_summary["N"],
    100.0 * paired_summary["rl_win_rate"],
    marker="o",
)

plt.axhline(50.0, linestyle="--", linewidth=1)
plt.xscale("log")
plt.ylim(0.0, 100.0)
plt.xlabel("Shots (N)")
plt.ylabel("RL paired win rate (%)")
plt.title("Exp06 Audited: RL Win Rate against XYZ")
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_vs_xyz_win_rate.png",
    dpi=200,
)
plt.close()


# RL joint-action allocation
plt.figure(figsize=(10, 7))

for name in ACTION_NAMES:
    plt.plot(
        allocation_summary["N"],
        allocation_summary[f"shots_{name}"],
        marker="o",
        label=name,
    )

plt.xscale("log")
plt.xlabel("Shots (N)")
plt.ylabel("Mean RL measurements")
plt.title("Exp06 Audited: RL Joint-Action Allocation")
plt.legend(ncol=3)
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_allocation_vs_shots.png",
    dpi=200,
)
plt.close()


# ========================
# Notes
# ========================
with open(
    f"{run_dir}/notes.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "Experiment 6: audited two-qubit budget stress test\n"
    )
    file.write(
        "Scope: two-qubit product-state estimation\n"
    )
    file.write(
        "One independent RL policy was trained for every tested "
        "shot budget.\n"
    )
    file.write(
        "RL uses one-step TD learning, epsilon decay, a terminal "
        "fidelity bonus, multiple restarts, and validation selection.\n"
    )
    file.write(
        "Warm-start measurements are counted inside the budget.\n"
    )
    file.write(
        f"Stress budgets: {STRESS_SHOT_BUDGETS}\n"
    )
    file.write(
        f"Training episodes per restart: {NUM_TRAIN_EPISODES}\n"
    )
    file.write(
        f"Training restarts: {NUM_RESTARTS}\n"
    )
    file.write(
        f"Validation targets: {NUM_VALIDATION_TARGETS}\n"
    )
    file.write(
        f"Test targets: {NUM_TEST_TARGETS}\n"
    )
    file.write(
        f"Test seeds: {NUM_TEST_SEEDS}\n"
    )
    file.write(
        f"Epsilon schedule: {EPSILON_START} to {EPSILON_END}\n"
    )
    file.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )
    file.write(
        f"Gamma: {GAMMA}\n"
    )
    file.write(
        f"Terminal bonus weight: {TERMINAL_BONUS_WEIGHT}\n"
    )


print("Experiment 6 completed.")
print("Results saved to:", run_dir)
print("Key files:")
print("- training_restart_validation.csv")
print("- stress_metrics.csv")
print("- stress_summary.csv")
print("- rl_allocation_summary.csv")
print("- rl_vs_xyz_paired_summary.csv")
print("- stress_fidelity.png")
print("- stress_infidelity.png")
print("- rl_vs_xyz_delta_fidelity.png")
print("- rl_vs_xyz_win_rate.png")
print("- rl_allocation_vs_shots.png")