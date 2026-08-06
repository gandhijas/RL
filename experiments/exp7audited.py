import os
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Experiment 7: Audited two-qubit robustness to symmetric
# readout-flip noise
#
# Scientific setup preserved:
# - two independently sampled pure qubits
# - uniform Bloch-sphere state distribution
# - nine joint Pauli actions
# - sequential RL measurement selection
# - Z-only, ZX-split, and XYZ-split fixed baselines
# - binary symmetric readout-flip noise
# - product-state fidelity as the evaluation objective
#
# Audit upgrades:
# - the same noise is applied to every method
# - independent X/Y/Z count arrays
# - audited 35-feature RL state
# - one-step TD learning with terminal-fidelity bonus
# - randomized training across the tested noise levels
# - multiple restarts and validation selection
# - independent method RNG streams
# - uniform Bloch-sphere training and test states
# - extreme-state results saved separately as a diagnostic
# - correct fidelity/infidelity plots and paired statistics
# ============================================================


# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp07_robustness_audited"
os.makedirs(run_dir, exist_ok=True)


# ========================
# Configuration
# ========================
SHOT_BUDGETS = [5, 10, 25, 50, 100]
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15]

NUM_TRAIN_EPISODES = 2500
NUM_RESTARTS = 3

NUM_VALIDATION_TARGETS = 25
NUM_VALIDATION_SEEDS = 2

NUM_TEST_TARGETS = 80
NUM_TEST_SEEDS = 5

EPSILON_START = 0.35
EPSILON_END = 0.02
EPSILON_TEST = 0.0

LEARNING_RATE = 0.006
GAMMA = 0.95
TERMINAL_BONUS_WEIGHT = 0.20

MASTER_SEED = 123
HEADLINE_BUDGET = 50


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
    """Uniform pure-state sampling on the Bloch sphere."""
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


def noisy_sample(
    probabilities: np.ndarray,
    rng: np.random.Generator,
    flip_probability: float,
) -> int:
    outcome = sample_one_shot(probabilities, rng)
    if rng.random() < flip_probability:
        return 1 - outcome
    return outcome


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

# 6 Bloch estimates
# 6 local uncertainties
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
    noise: float,
) -> None:
    basis1, basis2 = action

    outcome1 = noisy_sample(
        probs_for_basis(psi1, basis1),
        rng,
        flip_probability=noise,
    )
    outcome2 = noisy_sample(
        probs_for_basis(psi2, basis2),
        rng,
        flip_probability=noise,
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
    noise: float,
) -> Tuple[dict, Dict[Tuple[str, str], np.ndarray]]:
    counts1 = fresh_counts()
    counts2 = fresh_counts()
    action_counts = np.zeros(len(ACTIONS), dtype=int)
    shots_used = 0

    # Warm start is counted inside the shot budget.
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
            noise,
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
            noise,
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


def run_fixed_episode(
    psi1: np.ndarray,
    psi2: np.ndarray,
    total_shots: int,
    strategy: str,
    rng: np.random.Generator,
    noise: float,
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
            noise,
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
def evaluate_validation(
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
            for noise_index, noise in enumerate(NOISE_LEVELS):
                episode_rng = np.random.default_rng(
                    80_000_000
                    + total_shots * 1_000_000
                    + restart_index * 100_000
                    + target_index * 1_000
                    + seed * 10
                    + noise_index
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
                    noise=noise,
                )
                fidelities.append(output["fidelity"])

    return float(np.mean(fidelities))


# ========================
# Train one policy per budget
# ========================
trained_weights = {}
training_records = []

for total_shots in SHOT_BUDGETS:
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

            # Randomize the training noise so the policy is evaluated
            # for robustness across the whole tested noise range.
            noise = float(training_rng.choice(NOISE_LEVELS))

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
                noise=noise,
            )

        validation_fidelity = evaluate_validation(
            weights,
            total_shots,
            restart_index,
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
# Main uniform-state evaluation
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
        for total_shots in SHOT_BUDGETS:
            for noise_index, noise in enumerate(NOISE_LEVELS):
                common = {
                    "state_group": "uniform",
                    "target_id": target_id,
                    "seed": seed,
                    "N": total_shots,
                    "noise": noise,
                    "theta1": theta1,
                    "phi1": phi1,
                    "theta2": theta2,
                    "phi2": phi2,
                }

                base_seed = (
                    90_000_000
                    + target_id * 1_000_000
                    + seed * 10_000
                    + total_shots * 100
                    + noise_index * 10
                )

                for method_index, strategy in enumerate(
                    ["Z_only", "ZX_split", "XYZ_split"]
                ):
                    method_rng = np.random.default_rng(
                        base_seed + method_index
                    )

                    output = run_fixed_episode(
                        psi1=psi1,
                        psi2=psi2,
                        total_shots=total_shots,
                        strategy=strategy,
                        rng=method_rng,
                        noise=noise,
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
                    noise=noise,
                )

                rows.append(
                    {
                        **common,
                        "method": "RL_adaptive",
                        **rl_output,
                    }
                )


# ========================
# Extreme-state diagnostic
# ========================
extreme_states = [
    ("Z_plus", state_from_angles(0.0, 0.0)),
    ("Z_minus", state_from_angles(np.pi, 0.0)),
    ("X_plus", state_from_angles(np.pi / 2.0, 0.0)),
    ("Y_plus", state_from_angles(np.pi / 2.0, np.pi / 2.0)),
]

extreme_rows = []

for first_name, psi1 in extreme_states:
    for second_name, psi2 in extreme_states:
        for seed in range(NUM_TEST_SEEDS):
            for total_shots in SHOT_BUDGETS:
                for noise_index, noise in enumerate(NOISE_LEVELS):
                    base_seed = (
                        120_000_000
                        + ACTION_NAMES.index("ZZ") * 1_000_000
                        + seed * 10_000
                        + total_shots * 100
                        + noise_index * 10
                    )

                    for method_index, strategy in enumerate(
                        ["Z_only", "ZX_split", "XYZ_split"]
                    ):
                        method_rng = np.random.default_rng(
                            base_seed + method_index
                        )

                        output = run_fixed_episode(
                            psi1,
                            psi2,
                            total_shots,
                            strategy,
                            method_rng,
                            noise,
                        )

                        extreme_rows.append(
                            {
                                "state_group": "extreme",
                                "state1": first_name,
                                "state2": second_name,
                                "seed": seed,
                                "N": total_shots,
                                "noise": noise,
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
                        noise=noise,
                    )

                    extreme_rows.append(
                        {
                            "state_group": "extreme",
                            "state1": first_name,
                            "state2": second_name,
                            "seed": seed,
                            "N": total_shots,
                            "noise": noise,
                            "method": "RL_adaptive",
                            **rl_output,
                        }
                    )


# ========================
# Save metrics
# ========================
df = pd.DataFrame(rows)
df.to_csv(
    f"{run_dir}/robustness_metrics_uniform.csv",
    index=False,
)

extreme_df = pd.DataFrame(extreme_rows)
extreme_df.to_csv(
    f"{run_dir}/robustness_metrics_extreme.csv",
    index=False,
)


# ========================
# Aggregate uniform results
# ========================
summary = (
    df.groupby(["method", "N", "noise"])
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
    f"{run_dir}/robustness_summary_uniform.csv",
    index=False,
)


# ========================
# Paired RL-versus-XYZ
# ========================
paired = (
    df[df["method"].isin(["RL_adaptive", "XYZ_split"])]
    .pivot_table(
        index=["target_id", "seed", "N", "noise"],
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
    paired.groupby(["N", "noise"])
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
# Plots: fidelity/infidelity vs noise at headline budget
# ========================
headline = summary[summary["N"] == HEADLINE_BUDGET]

plt.figure(figsize=(9, 6))

for method in headline["method"].unique():
    method_data = headline[
        headline["method"] == method
    ].sort_values("noise")

    x = method_data["noise"].to_numpy()
    y = method_data["fidelity_mean"].to_numpy()
    ci = method_data["fidelity_ci95"].to_numpy()

    plt.plot(x, y, marker="o", label=method)
    plt.fill_between(
        x,
        np.clip(y - ci, 0.0, 1.0),
        np.clip(y + ci, 0.0, 1.0),
        alpha=0.2,
    )

plt.xlabel("Readout-flip probability")
plt.ylabel("Mean Fidelity")
plt.title(
    f"Exp07 Audited: Noise Robustness at N={HEADLINE_BUDGET}"
)
plt.ylim(0.0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{run_dir}/fidelity_vs_noise_N{HEADLINE_BUDGET}.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(9, 6))

for method in headline["method"].unique():
    method_data = headline[
        headline["method"] == method
    ].sort_values("noise")

    x = method_data["noise"].to_numpy()
    y = method_data["infidelity_mean"].to_numpy()
    ci = method_data["infidelity_ci95"].to_numpy()

    plt.plot(x, y, marker="o", label=method)
    plt.fill_between(
        x,
        np.maximum(y - ci, 1e-12),
        y + ci,
        alpha=0.2,
    )

plt.yscale("log")
plt.xlabel("Readout-flip probability")
plt.ylabel("Mean Infidelity")
plt.title(
    f"Exp07 Audited: Noise Robustness at N={HEADLINE_BUDGET}"
)
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{run_dir}/infidelity_vs_noise_N{HEADLINE_BUDGET}.png",
    dpi=200,
)
plt.close()


# ========================
# Plot: paired RL advantage vs noise
# ========================
delta_headline = paired_summary[
    paired_summary["N"] == HEADLINE_BUDGET
].sort_values("noise")

plt.figure(figsize=(9, 6))

x = delta_headline["noise"].to_numpy()
y = delta_headline["delta_mean"].to_numpy()
ci = delta_headline["delta_ci95"].to_numpy()

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.plot(x, y, marker="o")
plt.fill_between(x, y - ci, y + ci, alpha=0.2)
plt.xlabel("Readout-flip probability")
plt.ylabel(r"$\Delta F = F_{\mathrm{RL}} - F_{\mathrm{XYZ}}$")
plt.title(
    f"Exp07 Audited: Paired RL Advantage at N={HEADLINE_BUDGET}"
)
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_vs_xyz_delta_noise_N{HEADLINE_BUDGET}.png",
    dpi=200,
)
plt.close()


# ========================
# Plot: RL win rate vs noise
# ========================
plt.figure(figsize=(9, 6))

plt.plot(
    delta_headline["noise"],
    100.0 * delta_headline["rl_win_rate"],
    marker="o",
)

plt.axhline(50.0, linestyle="--", linewidth=1)
plt.ylim(0.0, 100.0)
plt.xlabel("Readout-flip probability")
plt.ylabel("RL paired win rate (%)")
plt.title(
    f"Exp07 Audited: RL Win Rate against XYZ at N={HEADLINE_BUDGET}"
)
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_vs_xyz_win_rate_noise_N{HEADLINE_BUDGET}.png",
    dpi=200,
)
plt.close()


# ========================
# Per-noise fidelity-vs-shots plots
# ========================
for noise in NOISE_LEVELS:
    noise_data = summary[np.isclose(summary["noise"], noise)]

    plt.figure(figsize=(9, 6))

    for method in noise_data["method"].unique():
        method_data = noise_data[
            noise_data["method"] == method
        ].sort_values("N")

        x = method_data["N"].to_numpy()
        y = method_data["fidelity_mean"].to_numpy()
        ci = method_data["fidelity_ci95"].to_numpy()

        plt.plot(x, y, marker="o", label=method)
        plt.fill_between(
            x,
            np.clip(y - ci, 0.0, 1.0),
            np.clip(y + ci, 0.0, 1.0),
            alpha=0.2,
        )

    plt.xscale("log")
    plt.xlabel("Shots (N)")
    plt.ylabel("Mean Fidelity")
    plt.title(
        f"Exp07 Audited: Fidelity vs Shots at Noise={noise:.2f}"
    )
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{run_dir}/fidelity_vs_shots_noise_{int(noise*100):02d}.png",
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
        "Experiment 7: audited symmetric readout-noise robustness\n"
    )
    file.write(
        "All methods receive the same readout-flip probability.\n"
    )
    file.write(
        "The estimator intentionally uses raw noisy counts, matching "
        "the original robustness question.\n"
    )
    file.write(
        "RL is trained across randomized tested noise levels.\n"
    )
    file.write(
        "Uniform Bloch-sphere states are the primary evaluation set.\n"
    )
    file.write(
        "Four axis-aligned states are retained as a separate "
        "diagnostic set.\n"
    )
    file.write(
        f"Shot budgets: {SHOT_BUDGETS}\n"
    )
    file.write(
        f"Noise levels: {NOISE_LEVELS}\n"
    )
    file.write(
        f"Training episodes per restart: {NUM_TRAIN_EPISODES}\n"
    )
    file.write(
        f"Training restarts: {NUM_RESTARTS}\n"
    )
    file.write(
        f"Test targets: {NUM_TEST_TARGETS}\n"
    )
    file.write(
        f"Test seeds: {NUM_TEST_SEEDS}\n"
    )


print("Experiment 7 completed.")
print("Results saved to:", run_dir)
print("Key files:")
print("- training_restart_validation.csv")
print("- robustness_metrics_uniform.csv")
print("- robustness_metrics_extreme.csv")
print("- robustness_summary_uniform.csv")
print("- rl_vs_xyz_paired_summary.csv")
print(f"- fidelity_vs_noise_N{HEADLINE_BUDGET}.png")
print(f"- infidelity_vs_noise_N{HEADLINE_BUDGET}.png")
print(f"- rl_vs_xyz_delta_noise_N{HEADLINE_BUDGET}.png")
print(f"- rl_vs_xyz_win_rate_noise_N{HEADLINE_BUDGET}.png")