import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Experiment 8: Corrected two-qubit adaptive tracking under drift
#
# Key corrections:
# 1. Reward is the fidelity improvement caused by the current
#    measurement, not total accumulated fidelity.
# 2. TD updates use the actual next decision state, after the
#    next drift and count-decay step.
# 3. The RL state contains both signed and absolute disagreement
#    between recent and long-memory estimates.
# 4. RL allocation summaries and a zero-drift sanity check are
#    generated automatically.
# ============================================================


# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp08_drift_qlearning_final_optimization"
os.makedirs(run_dir, exist_ok=True)


# ========================
# Configuration
# ========================
SHOT_BUDGETS = [10, 25, 50]
DRIFT_LEVELS = [0.0000, 0.0025, 0.0050, 0.0100]

NUM_TRAIN_EPISODES = 4000
NUM_RESTARTS = 5

NUM_VALIDATION_TARGETS = 20
NUM_VALIDATION_SEEDS = 2

NUM_TEST_TARGETS = 40
NUM_TEST_SEEDS = 4

EPSILON_START = 0.40
EPSILON_END = 0.01
EPSILON_TEST = 0.0

LEARNING_RATE = 0.002
GAMMA = 0.98
TERMINAL_BONUS_WEIGHT = 0.25
MAX_TD_ERROR = 1.0
MAX_WEIGHT_ABS = 5.0

LONG_DECAY = 0.985
RECENT_DECAY = 0.90

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


def wrap_theta(theta: float) -> float:
    theta = float(theta % (2.0 * np.pi))
    if theta > np.pi:
        theta = 2.0 * np.pi - theta
    return theta


def wrap_phi(phi: float) -> float:
    return float(phi % (2.0 * np.pi))


def sample_uniform_qubit_angles(
    rng: np.random.Generator,
) -> Tuple[float, float]:
    z = rng.uniform(-1.0, 1.0)
    return float(np.arccos(z)), float(rng.uniform(0.0, 2.0 * np.pi))


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
# Persistent drift model
# ========================
def sample_drift_rates(
    rng: np.random.Generator,
    drift_level: float,
) -> Tuple[float, float]:
    direction = rng.uniform(0.0, 2.0 * np.pi)
    return (
        float(drift_level * np.cos(direction)),
        float(drift_level * np.sin(direction)),
    )


def advance_angles(
    theta: float,
    phi: float,
    omega_theta: float,
    omega_phi: float,
) -> Tuple[float, float]:
    return (
        wrap_theta(theta + omega_theta),
        wrap_phi(phi + omega_phi),
    )


# ========================
# Recency-weighted estimator
# ========================
def fresh_weighted_counts() -> Dict[str, np.ndarray]:
    return {
        "X": np.zeros(2, dtype=float),
        "Y": np.zeros(2, dtype=float),
        "Z": np.zeros(2, dtype=float),
    }


def decay_counts(
    counts: Dict[str, np.ndarray],
    decay: float,
) -> None:
    for basis in counts:
        counts[basis] *= decay


def update_weighted_counts(
    counts: Dict[str, np.ndarray],
    basis: str,
    outcome: int,
) -> None:
    counts[basis][outcome] += 1.0


def coordinate_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 1e-12:
        return 0.0
    return float((counts[0] - counts[1]) / total)


def estimate_bloch(
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
    theta_hat = float(np.arccos(np.clip(z_n, -1.0, 1.0)))
    phi_hat = float(np.mod(np.arctan2(y_n, x_n), 2.0 * np.pi))
    return state_from_angles(theta_hat, phi_hat)


def estimate_state(
    counts: Dict[str, np.ndarray],
) -> np.ndarray:
    return bloch_to_state(*estimate_bloch(counts))


def product_fidelity_from_counts(
    psi1: np.ndarray,
    psi2: np.ndarray,
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
) -> float:
    return (
        fidelity(estimate_state(counts1), psi1)
        * fidelity(estimate_state(counts2), psi2)
    )


# ========================
# RL state and actions
# ========================
ACTIONS = [
    ("Z", "Z"), ("Z", "X"), ("Z", "Y"),
    ("X", "Z"), ("X", "X"), ("X", "Y"),
    ("Y", "Z"), ("Y", "X"), ("Y", "Y"),
]

ACTION_NAMES = [a + b for a, b in ACTIONS]
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}

# 6 long estimates
# 6 recent estimates
# 6 signed differences
# 6 absolute differences
# 6 uncertainties
# 6 recent local fractions
# 9 joint-action fractions
# progress + bias
STATE_DIM = 47


def local_totals(
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


def uncertainty(count: float) -> float:
    return float(1.0 / np.sqrt(count + 1.0))


def build_state(
    long1: Dict[str, np.ndarray],
    long2: Dict[str, np.ndarray],
    recent1: Dict[str, np.ndarray],
    recent2: Dict[str, np.ndarray],
    action_counts: np.ndarray,
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    long_bloch1 = np.array(estimate_bloch(long1), dtype=float)
    long_bloch2 = np.array(estimate_bloch(long2), dtype=float)
    recent_bloch1 = np.array(estimate_bloch(recent1), dtype=float)
    recent_bloch2 = np.array(estimate_bloch(recent2), dtype=float)

    signed_differences = np.concatenate(
        [
            recent_bloch1 - long_bloch1,
            recent_bloch2 - long_bloch2,
        ]
    )
    absolute_differences = np.abs(signed_differences)

    totals1 = local_totals(recent1)
    totals2 = local_totals(recent2)

    uncertainties = np.array(
        [uncertainty(value) for value in np.concatenate([totals1, totals2])],
        dtype=float,
    )

    fractions1 = totals1 / max(float(np.sum(totals1)), 1e-12)
    fractions2 = totals2 / max(float(np.sum(totals2)), 1e-12)
    recent_fractions = np.concatenate([fractions1, fractions2])

    action_fractions = action_counts.astype(float) / max(total_shots, 1)
    progress = shots_used / total_shots if total_shots > 0 else 0.0

    state = np.concatenate(
        [
            long_bloch1,
            long_bloch2,
            recent_bloch1,
            recent_bloch2,
            signed_differences,
            absolute_differences,
            uncertainties,
            recent_fractions,
            action_fractions,
            np.array([progress, 1.0], dtype=float),
        ]
    )

    if state.shape[0] != STATE_DIM:
        raise RuntimeError(
            f"State dimension mismatch: {state.shape[0]} != {STATE_DIM}"
        )

    return state


def initialize_weights() -> Dict[Tuple[str, str], np.ndarray]:
    # absolute-difference feature positions
    abs_q1 = {"X": 18, "Y": 19, "Z": 20}
    abs_q2 = {"X": 21, "Y": 22, "Z": 23}

    # uncertainty feature positions
    unc_q1 = {"X": 24, "Y": 25, "Z": 26}
    unc_q2 = {"X": 27, "Y": 28, "Z": 29}

    weights: Dict[Tuple[str, str], np.ndarray] = {}

    for action in ACTIONS:
        b1, b2 = action
        vector = np.zeros(STATE_DIM, dtype=float)

        vector[abs_q1[b1]] = 0.05
        vector[abs_q2[b2]] = 0.05
        vector[unc_q1[b1]] = 0.05
        vector[unc_q2[b2]] = 0.05

        weights[action] = vector

    return weights


def q_values(
    state: np.ndarray,
    weights: Dict[Tuple[str, str], np.ndarray],
) -> np.ndarray:
    return np.array(
        [float(np.dot(weights[action], state)) for action in ACTIONS],
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


# ========================
# Trajectory helpers
# ========================
def initialize_trajectory(
    rng: np.random.Generator,
    drift_level: float,
) -> dict:
    theta1, phi1 = sample_uniform_qubit_angles(rng)
    theta2, phi2 = sample_uniform_qubit_angles(rng)

    omega_theta1, omega_phi1 = sample_drift_rates(rng, drift_level)
    omega_theta2, omega_phi2 = sample_drift_rates(rng, drift_level)

    return {
        "theta1": theta1,
        "phi1": phi1,
        "theta2": theta2,
        "phi2": phi2,
        "omega_theta1": omega_theta1,
        "omega_phi1": omega_phi1,
        "omega_theta2": omega_theta2,
        "omega_phi2": omega_phi2,
    }


def current_states(trajectory: dict) -> Tuple[np.ndarray, np.ndarray]:
    return (
        state_from_angles(trajectory["theta1"], trajectory["phi1"]),
        state_from_angles(trajectory["theta2"], trajectory["phi2"]),
    )


def advance_trajectory(trajectory: dict) -> None:
    trajectory["theta1"], trajectory["phi1"] = advance_angles(
        trajectory["theta1"],
        trajectory["phi1"],
        trajectory["omega_theta1"],
        trajectory["omega_phi1"],
    )
    trajectory["theta2"], trajectory["phi2"] = advance_angles(
        trajectory["theta2"],
        trajectory["phi2"],
        trajectory["omega_theta2"],
        trajectory["omega_phi2"],
    )


def apply_measurement(
    psi1: np.ndarray,
    psi2: np.ndarray,
    action: Tuple[str, str],
    long1: Dict[str, np.ndarray],
    long2: Dict[str, np.ndarray],
    recent1: Dict[str, np.ndarray],
    recent2: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    b1, b2 = action

    o1 = sample_one_shot(probs_for_basis(psi1, b1), rng)
    o2 = sample_one_shot(probs_for_basis(psi2, b2), rng)

    update_weighted_counts(long1, b1, o1)
    update_weighted_counts(long2, b2, o2)
    update_weighted_counts(recent1, b1, o1)
    update_weighted_counts(recent2, b2, o2)


# ========================
# Corrected episode engine
# ========================
def run_episode(
    total_shots: int,
    trajectory_seed: int,
    measurement_seed: int,
    drift_level: float,
    strategy: str,
    weights: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    epsilon: float = 0.0,
    update_weights: bool = False,
) -> Tuple[dict, Optional[Dict[Tuple[str, str], np.ndarray]]]:
    trajectory_rng = np.random.default_rng(trajectory_seed)
    measurement_rng = np.random.default_rng(measurement_seed)

    trajectory = initialize_trajectory(trajectory_rng, drift_level)

    long1 = fresh_weighted_counts()
    long2 = fresh_weighted_counts()
    recent1 = fresh_weighted_counts()
    recent2 = fresh_weighted_counts()

    action_counts = np.zeros(len(ACTIONS), dtype=int)
    tracking_fidelities: List[float] = []

    warm_actions = [("Z", "Z"), ("X", "X"), ("Y", "Y")]

    pending_state = None
    pending_action = None
    pending_reward = None

    for t in range(total_shots):
        if t > 0:
            advance_trajectory(trajectory)

        psi1, psi2 = current_states(trajectory)

        decay_counts(long1, LONG_DECAY)
        decay_counts(long2, LONG_DECAY)
        decay_counts(recent1, RECENT_DECAY)
        decay_counts(recent2, RECENT_DECAY)

        current_state = build_state(
            long1,
            long2,
            recent1,
            recent2,
            action_counts,
            t,
            total_shots,
        )

        # Complete the previous TD transition using the actual next
        # decision state after drift and decay.
        if (
            strategy == "RL_adaptive"
            and update_weights
            and pending_state is not None
        ):
            q_previous = float(
                np.dot(weights[pending_action], pending_state)
            )
            q_next = float(np.max(q_values(current_state, weights)))
            td_target = pending_reward + GAMMA * q_next
            td_error = float(
                np.clip(
                    td_target - q_previous,
                    -MAX_TD_ERROR,
                    MAX_TD_ERROR,
                )
            )
            weights[pending_action] += (
                LEARNING_RATE * td_error * pending_state
            )
            weights[pending_action] = np.clip(
                weights[pending_action],
                -MAX_WEIGHT_ABS,
                MAX_WEIGHT_ABS,
            )

        pre_measure_fidelity = product_fidelity_from_counts(
            psi1,
            psi2,
            recent1,
            recent2,
        )

        if t < len(warm_actions):
            action = warm_actions[t]
        elif strategy == "RL_adaptive":
            if weights is None:
                raise ValueError("RL strategy requires weights")
            action = epsilon_greedy(
                current_state,
                weights,
                epsilon,
                measurement_rng,
            )
        elif strategy == "XYZ_interleaved":
            action = [("X", "X"), ("Y", "Y"), ("Z", "Z")][t % 3]
        elif strategy == "ZX_interleaved":
            action = [("Z", "Z"), ("X", "X")][t % 2]
        elif strategy == "Z_only":
            action = ("Z", "Z")
        elif strategy == "Balanced_random":
            action = ACTIONS[int(measurement_rng.integers(len(ACTIONS)))]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        apply_measurement(
            psi1,
            psi2,
            action,
            long1,
            long2,
            recent1,
            recent2,
            measurement_rng,
        )
        action_counts[ACTION_INDEX[action]] += 1

        post_measure_fidelity = product_fidelity_from_counts(
            psi1,
            psi2,
            recent1,
            recent2,
        )
        tracking_fidelities.append(post_measure_fidelity)

        reward = post_measure_fidelity - pre_measure_fidelity

        if strategy == "RL_adaptive" and t >= len(warm_actions):
            pending_state = current_state.copy()
            pending_action = action
            pending_reward = reward

    # Terminal update for the final pending action.
    if (
        strategy == "RL_adaptive"
        and update_weights
        and pending_state is not None
    ):
        final_psi1, final_psi2 = current_states(trajectory)
        terminal_fidelity = product_fidelity_from_counts(
            final_psi1,
            final_psi2,
            recent1,
            recent2,
        )

        q_previous = float(
            np.dot(weights[pending_action], pending_state)
        )
        terminal_reward = (
            pending_reward
            + TERMINAL_BONUS_WEIGHT * terminal_fidelity
        )
        td_error = float(
            np.clip(
                terminal_reward - q_previous,
                -MAX_TD_ERROR,
                MAX_TD_ERROR,
            )
        )
        weights[pending_action] += (
            LEARNING_RATE * td_error * pending_state
        )
        weights[pending_action] = np.clip(
            weights[pending_action],
            -MAX_WEIGHT_ABS,
            MAX_WEIGHT_ABS,
        )

    final_psi1, final_psi2 = current_states(trajectory)
    final_fidelity = product_fidelity_from_counts(
        final_psi1,
        final_psi2,
        recent1,
        recent2,
    )
    tracking_fidelity = float(np.mean(tracking_fidelities))

    result = {
        "final_fidelity": final_fidelity,
        "final_infidelity": 1.0 - final_fidelity,
        "tracking_fidelity": tracking_fidelity,
        "tracking_infidelity": 1.0 - tracking_fidelity,
    }

    for index, name in enumerate(ACTION_NAMES):
        result[f"shots_{name}"] = int(action_counts[index])

    return result, weights


# ========================
# Validation
# ========================
def evaluate_validation(
    weights: Dict[Tuple[str, str], np.ndarray],
    total_shots: int,
    restart_index: int,
) -> float:
    scores = []

    for target_index in range(NUM_VALIDATION_TARGETS):
        for seed in range(NUM_VALIDATION_SEEDS):
            for drift_index, drift_level in enumerate(DRIFT_LEVELS):
                trajectory_seed = (
                    70_000_000
                    + total_shots * 1_000_000
                    + target_index * 10_000
                    + seed * 100
                    + drift_index
                )
                measurement_seed = (
                    80_000_000
                    + total_shots * 1_000_000
                    + restart_index * 100_000
                    + target_index * 1_000
                    + seed * 10
                    + drift_index
                )

                output, _ = run_episode(
                    total_shots,
                    trajectory_seed,
                    measurement_seed,
                    drift_level,
                    "RL_adaptive",
                    {
                        action: vector.copy()
                        for action, vector in weights.items()
                    },
                    EPSILON_TEST,
                    False,
                )
                scores.append(output["tracking_fidelity"])

    return float(np.mean(scores))


# ========================
# Training
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
            drift_level = float(training_rng.choice(DRIFT_LEVELS))
            epsilon_fraction = episode / max(NUM_TRAIN_EPISODES - 1, 1)
            epsilon = EPSILON_START + epsilon_fraction * (
                EPSILON_END - EPSILON_START
            )

            _, weights = run_episode(
                total_shots=total_shots,
                trajectory_seed=(
                    10_000_000
                    + total_shots * 1_000_000
                    + restart_index * 100_000
                    + episode
                ),
                measurement_seed=(
                    20_000_000
                    + total_shots * 1_000_000
                    + restart_index * 100_000
                    + episode
                ),
                drift_level=drift_level,
                strategy="RL_adaptive",
                weights=weights,
                epsilon=epsilon,
                update_weights=True,
            )

        validation_score = evaluate_validation(
            weights,
            total_shots,
            restart_index,
        )

        candidates.append(
            (
                validation_score,
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
                "validation_tracking_fidelity": validation_score,
            }
        )

        print(
            f"N={total_shots}, restart={restart_index}, "
            f"validation tracking fidelity={validation_score:.6f}"
        )

    best_score, best_weights = max(candidates, key=lambda item: item[0])
    trained_weights[total_shots] = best_weights

    print(
        f"Selected policy for N={total_shots}: "
        f"validation tracking fidelity={best_score:.6f}"
    )

pd.DataFrame(training_records).to_csv(
    f"{run_dir}/training_restart_validation.csv",
    index=False,
)


# ========================
# Final evaluation
# ========================
METHODS = [
    "RL_adaptive",
    "XYZ_interleaved",
    "ZX_interleaved",
    "Z_only",
    "Balanced_random",
]

rows = []

for target_id in range(NUM_TEST_TARGETS):
    for seed in range(NUM_TEST_SEEDS):
        for total_shots in SHOT_BUDGETS:
            for drift_index, drift_level in enumerate(DRIFT_LEVELS):
                trajectory_seed = (
                    100_000_000
                    + target_id * 1_000_000
                    + seed * 10_000
                    + total_shots * 100
                    + drift_index
                )

                for method_index, method in enumerate(METHODS):
                    output, _ = run_episode(
                        total_shots=total_shots,
                        trajectory_seed=trajectory_seed,
                        measurement_seed=(
                            110_000_000
                            + target_id * 1_000_000
                            + seed * 10_000
                            + total_shots * 100
                            + drift_index * 10
                            + method_index
                        ),
                        drift_level=drift_level,
                        strategy=method,
                        weights=(
                            {
                                action: vector.copy()
                                for action, vector in trained_weights[
                                    total_shots
                                ].items()
                            }
                            if method == "RL_adaptive"
                            else None
                        ),
                        epsilon=EPSILON_TEST,
                        update_weights=False,
                    )

                    rows.append(
                        {
                            "target_id": target_id,
                            "seed": seed,
                            "N": total_shots,
                            "drift_level": drift_level,
                            "method": method,
                            **output,
                        }
                    )


df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/drift_metrics.csv", index=False)

summary = (
    df.groupby(["method", "N", "drift_level"])
    .agg(
        final_fidelity_mean=("final_fidelity", "mean"),
        final_fidelity_std=("final_fidelity", "std"),
        final_fidelity_count=("final_fidelity", "count"),
        tracking_fidelity_mean=("tracking_fidelity", "mean"),
        tracking_fidelity_std=("tracking_fidelity", "std"),
        tracking_fidelity_count=("tracking_fidelity", "count"),
    )
    .reset_index()
)

summary["final_fidelity_ci95"] = (
    1.96
    * summary["final_fidelity_std"]
    / np.sqrt(summary["final_fidelity_count"])
)
summary["tracking_fidelity_ci95"] = (
    1.96
    * summary["tracking_fidelity_std"]
    / np.sqrt(summary["tracking_fidelity_count"])
)
summary.to_csv(f"{run_dir}/drift_summary.csv", index=False)


# ========================
# RL allocation summary
# ========================
allocation_columns = [f"shots_{name}" for name in ACTION_NAMES]
allocation_summary = (
    df[df["method"] == "RL_adaptive"]
    .groupby(["N", "drift_level"])[allocation_columns]
    .mean()
    .reset_index()
)
allocation_summary.to_csv(
    f"{run_dir}/rl_allocation_summary.csv",
    index=False,
)


# ========================
# Paired RL-versus-XYZ
# ========================
paired = (
    df[df["method"].isin(["RL_adaptive", "XYZ_interleaved"])]
    .pivot_table(
        index=["target_id", "seed", "N", "drift_level"],
        columns="method",
        values=["final_fidelity", "tracking_fidelity"],
    )
    .dropna()
)

paired.columns = [
    f"{metric}_{method}" for metric, method in paired.columns
]
paired = paired.reset_index()

paired["delta_final_fidelity"] = (
    paired["final_fidelity_RL_adaptive"]
    - paired["final_fidelity_XYZ_interleaved"]
)
paired["delta_tracking_fidelity"] = (
    paired["tracking_fidelity_RL_adaptive"]
    - paired["tracking_fidelity_XYZ_interleaved"]
)

paired_summary = (
    paired.groupby(["N", "drift_level"])
    .agg(
        delta_final_mean=("delta_final_fidelity", "mean"),
        delta_final_std=("delta_final_fidelity", "std"),
        delta_tracking_mean=("delta_tracking_fidelity", "mean"),
        delta_tracking_std=("delta_tracking_fidelity", "std"),
        count=("delta_tracking_fidelity", "count"),
        rl_tracking_win_rate=(
            "delta_tracking_fidelity",
            lambda values: float(np.mean(values > 0.0)),
        ),
        rl_final_win_rate=(
            "delta_final_fidelity",
            lambda values: float(np.mean(values > 0.0)),
        ),
    )
    .reset_index()
)

paired_summary["delta_final_ci95"] = (
    1.96
    * paired_summary["delta_final_std"]
    / np.sqrt(paired_summary["count"])
)
paired_summary["delta_tracking_ci95"] = (
    1.96
    * paired_summary["delta_tracking_std"]
    / np.sqrt(paired_summary["count"])
)
paired_summary.to_csv(
    f"{run_dir}/rl_vs_xyz_paired_summary.csv",
    index=False,
)


# ========================
# Headline plots
# ========================
headline = summary[summary["N"] == HEADLINE_BUDGET]

for metric, ylabel, filename in [
    (
        "tracking_fidelity",
        "Mean Tracking Fidelity",
        f"tracking_fidelity_vs_drift_N{HEADLINE_BUDGET}.png",
    ),
    (
        "final_fidelity",
        "Final-State Fidelity",
        f"final_fidelity_vs_drift_N{HEADLINE_BUDGET}.png",
    ),
]:
    plt.figure(figsize=(9, 6))

    for method in headline["method"].unique():
        sub = headline[headline["method"] == method].sort_values(
            "drift_level"
        )
        x = sub["drift_level"].to_numpy()
        y = sub[f"{metric}_mean"].to_numpy()
        ci = sub[f"{metric}_ci95"].to_numpy()

        plt.plot(x, y, marker="o", label=method)
        plt.fill_between(
            x,
            np.clip(y - ci, 0.0, 1.0),
            np.clip(y + ci, 0.0, 1.0),
            alpha=0.2,
        )

    plt.xlabel("Per-step drift magnitude")
    plt.ylabel(ylabel)
    plt.title(f"Exp08 Corrected: {ylabel} at N={HEADLINE_BUDGET}")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/{filename}", dpi=200)
    plt.close()


delta_headline = paired_summary[
    paired_summary["N"] == HEADLINE_BUDGET
].sort_values("drift_level")

plt.figure(figsize=(9, 6))
x = delta_headline["drift_level"].to_numpy()
y = delta_headline["delta_tracking_mean"].to_numpy()
ci = delta_headline["delta_tracking_ci95"].to_numpy()
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.plot(x, y, marker="o")
plt.fill_between(x, y - ci, y + ci, alpha=0.2)
plt.xlabel("Per-step drift magnitude")
plt.ylabel(r"$\Delta F_{\rm track}=F_{\rm RL}-F_{\rm XYZ}$")
plt.title(
    f"Exp08 Corrected: Paired Tracking Advantage at N={HEADLINE_BUDGET}"
)
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_vs_xyz_tracking_delta_N{HEADLINE_BUDGET}.png",
    dpi=200,
)
plt.close()


plt.figure(figsize=(9, 6))
plt.plot(
    delta_headline["drift_level"],
    100.0 * delta_headline["rl_tracking_win_rate"],
    marker="o",
)
plt.axhline(50.0, linestyle="--", linewidth=1)
plt.ylim(0.0, 100.0)
plt.xlabel("Per-step drift magnitude")
plt.ylabel("RL tracking win rate (%)")
plt.title(
    f"Exp08 Corrected: RL Tracking Win Rate at N={HEADLINE_BUDGET}"
)
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_vs_xyz_tracking_win_rate_N{HEADLINE_BUDGET}.png",
    dpi=200,
)
plt.close()


# ========================
# Zero-drift sanity check
# ========================
sanity = paired_summary[
    (paired_summary["N"] == HEADLINE_BUDGET)
    & np.isclose(paired_summary["drift_level"], 0.0)
]

if not sanity.empty:
    gap = float(sanity.iloc[0]["delta_tracking_mean"])
    print(
        f"Zero-drift paired tracking gap at N={HEADLINE_BUDGET}: "
        f"{gap:+.6f}"
    )
    if gap < -0.05:
        print(
            "WARNING: zero-drift RL remains far below XYZ. "
            "Do not use higher-drift results yet."
        )
    else:
        print(
            "Zero-drift sanity check is within the expected "
            "competitive range."
        )


with open(f"{run_dir}/notes.txt", "w", encoding="utf-8") as file:
    file.write("Experiment 8: final optimized linear max-Q Q-learning\n")
    file.write(
        "Reward: post-measure fidelity minus pre-measure fidelity.\n"
    )
    file.write(
        "TD next state is constructed after the next drift and decay.\n"
    )
    file.write(
        "State contains signed and absolute recent-long differences.\n"
    )
    file.write(f"Shot budgets: [10, 25, 50]\n")
    file.write(f"Drift levels: {DRIFT_LEVELS}\n")
    file.write(f"Training episodes: {NUM_TRAIN_EPISODES}\n")
    file.write(f"Training restarts: {NUM_RESTARTS}\n")
    file.write("Learning rule: linear one-step max-Q TD learning\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Gamma: {GAMMA}\n")
    file.write(f"Epsilon schedule: {EPSILON_START} to {EPSILON_END}\n")
    file.write(f"TD-error clip: {MAX_TD_ERROR}\n")


print("Corrected Experiment 8 completed.")
print("Results saved to:", run_dir)
print("Key files:")
print("- training_restart_validation.csv")
print("- drift_metrics.csv")
print("- drift_summary.csv")
print("- rl_allocation_summary.csv")
print("- rl_vs_xyz_paired_summary.csv")
print(f"- tracking_fidelity_vs_drift_N{HEADLINE_BUDGET}.png")
print(f"- final_fidelity_vs_drift_N{HEADLINE_BUDGET}.png")
print(f"- rl_vs_xyz_tracking_delta_N{HEADLINE_BUDGET}.png")
print(f"- rl_vs_xyz_tracking_win_rate_N{HEADLINE_BUDGET}.png")