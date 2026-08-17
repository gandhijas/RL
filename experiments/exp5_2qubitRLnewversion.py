import os
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Experiment 5: RL on hidden-plane two-qubit product states
# ============================================================
# This is a fair adaptive benchmark: every target belongs to one
# unknown plane (XY, XZ, or YZ). The RL agent receives only its
# measurement history, not the plane label. XYZ_split remains a
# complete fixed baseline; ORACLE_plane is an explicitly informed
# upper benchmark.

SEED = 123
SHOT_BUDGETS = [10, 25, 50, 100, 250, 500]
NUM_TRAIN_EPISODES = int(os.getenv("EXP05_TRAIN_EPISODES", "5000"))
NUM_TEST_TARGETS = int(os.getenv("EXP05_TEST_TARGETS", "50"))
NUM_TEST_SEEDS = int(os.getenv("EXP05_TEST_SEEDS", "5"))
MIN_ACTIVE_COMPONENT = float(os.getenv("EXP05_MIN_ACTIVE_COMPONENT", "0.35"))

EPSILON_START = 0.35
EPSILON_END = 0.03
LEARNING_RATE = 0.008
GAMMA = 0.95
TERMINAL_BONUS_WEIGHT = 0.20
WARM_START_REPEATS = 2  # two shots each in ZZ, XX, YY when budget permits

PLANE_BASES: Dict[str, Tuple[str, str]] = {
    "XY": ("X", "Y"),
    "XZ": ("X", "Z"),
    "YZ": ("Y", "Z"),
}

ACTIONS = [
    ("Z", "Z"), ("Z", "X"), ("Z", "Y"),
    ("X", "Z"), ("X", "X"), ("X", "Y"),
    ("Y", "Z"), ("Y", "X"), ("Y", "Y"),
]
ACTION_NAMES = [a + b for a, b in ACTIONS]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

# 6 Bloch estimates + 6 uncertainties + 6 local-basis fractions
# + 9 joint-action fractions + 3 latent-plane scores + progress + bias
STATE_DIM = 32


def state_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
        dtype=complex,
    )


def state_from_bloch(x: float, y: float, z: float) -> np.ndarray:
    z = float(np.clip(z, -1.0, 1.0))
    theta = float(np.arccos(z))
    phi = float(np.mod(np.arctan2(y, x), 2 * np.pi))
    return state_from_angles(theta, phi)


def sample_state_in_plane(
    plane: str,
    rng: np.random.Generator,
    min_active_component: float = MIN_ACTIVE_COMPONENT,
):
    while True:
        alpha = float(rng.uniform(0.0, 2.0 * np.pi))
        c, s = float(np.cos(alpha)), float(np.sin(alpha))
        if min(abs(c), abs(s)) >= min_active_component:
            break

    if plane == "XY":
        x, y, z = c, s, 0.0
    elif plane == "XZ":
        x, y, z = c, 0.0, s
    elif plane == "YZ":
        x, y, z = 0.0, c, s
    else:
        raise ValueError(f"Unknown plane: {plane}")

    return state_from_bloch(x, y, z), alpha


def sample_hidden_plane_product_state(rng: np.random.Generator):
    plane = str(rng.choice(list(PLANE_BASES)))
    psi1, alpha1 = sample_state_in_plane(plane, rng)
    psi2, alpha2 = sample_state_in_plane(plane, rng)
    return psi1, psi2, plane, alpha1, alpha2


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def hadamard() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def s_dagger() -> np.ndarray:
    return np.array([[1, 0], [0, -1j]], dtype=complex)


def probs_for_basis(psi: np.ndarray, basis: str) -> np.ndarray:
    if basis == "Z":
        rotated = psi
    elif basis == "X":
        rotated = hadamard() @ psi
    elif basis == "Y":
        rotated = hadamard() @ (s_dagger() @ psi)
    else:
        raise ValueError(f"Unknown basis: {basis}")
    p = np.abs(rotated) ** 2
    return p / np.sum(p)


def sample_one_shot(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice([0, 1], p=probs))


def fresh_counts() -> Dict[str, np.ndarray]:
    return {b: np.array([0, 0], dtype=int) for b in ("X", "Y", "Z")}


def estimate_coord(counts: np.ndarray) -> float:
    total = int(np.sum(counts))
    return 0.0 if total == 0 else float((counts[0] - counts[1]) / total)


def estimate_bloch(counts: Dict[str, np.ndarray]):
    return (
        estimate_coord(counts["X"]),
        estimate_coord(counts["Y"]),
        estimate_coord(counts["Z"]),
    )


def estimate_state(counts: Dict[str, np.ndarray]) -> np.ndarray:
    r = np.array(estimate_bloch(counts), dtype=float)
    norm = float(np.linalg.norm(r))
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=complex)
    x, y, z = r / norm
    return state_from_bloch(float(x), float(y), float(z))


def product_fidelity(
    psi1: np.ndarray,
    psi2: np.ndarray,
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
) -> float:
    return fidelity(estimate_state(counts1), psi1) * fidelity(estimate_state(counts2), psi2)


def measurement_uncertainty(counts: np.ndarray) -> float:
    n = int(np.sum(counts))
    if n == 0:
        return 1.0
    mean = estimate_coord(counts)
    # Estimated standard error of a Pauli expectation, regularized.
    return float(np.sqrt(max(1.0 - mean * mean, 0.05) / n))


def basis_totals(counts: Dict[str, np.ndarray]) -> np.ndarray:
    return np.array([np.sum(counts[b]) for b in ("X", "Y", "Z")], dtype=float)


def plane_scores(b1, b2) -> np.ndarray:
    """Evidence scores for XY, XZ, YZ using both estimated Bloch vectors."""
    x1, y1, z1 = b1
    x2, y2, z2 = b2
    scores = np.array(
        [
            abs(x1) + abs(y1) + abs(x2) + abs(y2) - abs(z1) - abs(z2),
            abs(x1) + abs(z1) + abs(x2) + abs(z2) - abs(y1) - abs(y2),
            abs(y1) + abs(z1) + abs(y2) + abs(z2) - abs(x1) - abs(x2),
        ],
        dtype=float,
    )
    return scores / 4.0


def build_state(
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
    action_counts: np.ndarray,
    shots_used: int,
    total_shots: int,
) -> np.ndarray:
    bloch1 = np.array(estimate_bloch(counts1), dtype=float)
    bloch2 = np.array(estimate_bloch(counts2), dtype=float)

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

    local_totals = np.concatenate([basis_totals(counts1), basis_totals(counts2)])
    local_fracs = local_totals / max(float(shots_used), 1.0)
    action_fracs = action_counts.astype(float) / max(float(shots_used), 1.0)
    scores = plane_scores(bloch1, bloch2)
    progress = shots_used / total_shots

    state = np.concatenate(
        [
            bloch1,
            bloch2,
            uncertainties,
            local_fracs,
            action_fracs,
            scores,
            np.array([progress, 1.0], dtype=float),
        ]
    )
    if state.shape[0] != STATE_DIM:
        raise RuntimeError(f"State dimension mismatch: {state.shape[0]} != {STATE_DIM}")
    return state


def q_values(state, weights):
    return np.array([float(np.dot(weights[a], state)) for a in ACTIONS])


def epsilon_greedy(state, weights, epsilon, rng):
    if rng.random() < epsilon:
        return ACTIONS[int(rng.integers(len(ACTIONS)))]
    values = q_values(state, weights)
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return ACTIONS[int(rng.choice(best))]


def initialize_weights():
    weights = {}
    # State indices:
    # Bloch 0:6, uncertainty 6:12, local fracs 12:18,
    # action fracs 18:27, plane scores 27:30, progress 30, bias 31.
    plane_names = ["XY", "XZ", "YZ"]
    for action in ACTIONS:
        w = np.zeros(STATE_DIM, dtype=float)
        bq1, bq2 = action
        unc_q1 = {"X": 6, "Y": 7, "Z": 8}[bq1]
        unc_q2 = {"X": 9, "Y": 10, "Z": 11}[bq2]
        w[unc_q1] = 0.05
        w[unc_q2] = 0.05

        # Small structured prior: an action is favored when both selected
        # bases are compatible with a high-scoring latent plane.
        for p_idx, plane in enumerate(plane_names):
            active = set(PLANE_BASES[plane])
            if bq1 in active and bq2 in active:
                w[27 + p_idx] = 0.08
        weights[action] = w
    return weights


def apply_joint_measurement(
    psi1,
    psi2,
    action,
    counts1,
    counts2,
    rng,
):
    b1, b2 = action
    counts1[b1][sample_one_shot(probs_for_basis(psi1, b1), rng)] += 1
    counts2[b2][sample_one_shot(probs_for_basis(psi2, b2), rng)] += 1


def run_rl_episode(
    psi1,
    psi2,
    total_shots,
    weights,
    epsilon,
    rng,
    update_weights,
):
    counts1, counts2 = fresh_counts(), fresh_counts()
    action_counts = np.zeros(len(ACTIONS), dtype=int)
    shots_used = 0

    warm_start = []
    for _ in range(WARM_START_REPEATS):
        warm_start.extend([("Z", "Z"), ("X", "X"), ("Y", "Y")])

    for action in warm_start:
        if shots_used >= total_shots:
            break
        apply_joint_measurement(psi1, psi2, action, counts1, counts2, rng)
        action_counts[ACTION_INDEX[action]] += 1
        shots_used += 1

    f_prev = product_fidelity(psi1, psi2, counts1, counts2)

    for step in range(shots_used, total_shots):
        state = build_state(counts1, counts2, action_counts, step, total_shots)
        action = epsilon_greedy(state, weights, epsilon, rng)
        q_current = float(np.dot(weights[action], state))

        apply_joint_measurement(psi1, psi2, action, counts1, counts2, rng)
        action_counts[ACTION_INDEX[action]] += 1
        f_new = product_fidelity(psi1, psi2, counts1, counts2)

        terminal = step == total_shots - 1
        reward = (f_new - f_prev) + (TERMINAL_BONUS_WEIGHT * f_new if terminal else 0.0)

        if terminal:
            q_next = 0.0
        else:
            next_state = build_state(counts1, counts2, action_counts, step + 1, total_shots)
            q_next = float(np.max(q_values(next_state, weights)))

        if update_weights:
            td_error = reward + GAMMA * q_next - q_current
            weights[action] += LEARNING_RATE * td_error * state

        f_prev = f_new

    out = {"fidelity": f_prev, "infidelity": 1.0 - f_prev}
    for i, name in enumerate(ACTION_NAMES):
        out[f"shots_{name}"] = int(action_counts[i])
    return out, weights


def balanced_schedule(bases: Tuple[str, ...], total_shots: int):
    return [bases[i % len(bases)] for i in range(total_shots)]


def run_fixed_episode(psi1, psi2, plane, total_shots, strategy, rng):
    counts1, counts2 = fresh_counts(), fresh_counts()
    if strategy == "Z_only":
        schedule = ["Z"] * total_shots
    elif strategy == "ZX_split":
        schedule = balanced_schedule(("Z", "X"), total_shots)
    elif strategy == "XYZ_split":
        schedule = balanced_schedule(("X", "Y", "Z"), total_shots)
    elif strategy == "ORACLE_plane":
        schedule = balanced_schedule(PLANE_BASES[plane], total_shots)
    else:
        raise ValueError(strategy)

    for basis in schedule:
        counts1[basis][sample_one_shot(probs_for_basis(psi1, basis), rng)] += 1
        counts2[basis][sample_one_shot(probs_for_basis(psi2, basis), rng)] += 1

    f = product_fidelity(psi1, psi2, counts1, counts2)
    return {"fidelity": f, "infidelity": 1.0 - f}


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"results/{timestamp}_exp05_hidden_plane_rl"
    os.makedirs(run_dir, exist_ok=True)

    master_rng = np.random.default_rng(SEED)
    trained_weights = {}

    for n_shots in SHOT_BUDGETS:
        weights = initialize_weights()
        for ep in range(NUM_TRAIN_EPISODES):
            psi1, psi2, _, _, _ = sample_hidden_plane_product_state(master_rng)
            frac = ep / max(NUM_TRAIN_EPISODES - 1, 1)
            epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)
            rng = np.random.default_rng(10_000_000 + n_shots * 10_000 + ep)
            _, weights = run_rl_episode(
                psi1, psi2, n_shots, weights, epsilon, rng, update_weights=True
            )
        trained_weights[n_shots] = {a: w.copy() for a, w in weights.items()}
        print(f"Finished training N={n_shots}")

    rows = []
    fixed_methods = ["Z_only", "ZX_split", "XYZ_split", "ORACLE_plane"]

    for target_id in range(NUM_TEST_TARGETS):
        psi1, psi2, plane, alpha1, alpha2 = sample_hidden_plane_product_state(master_rng)
        for seed in range(NUM_TEST_SEEDS):
            for n_shots in SHOT_BUDGETS:
                base_seed = target_id * 1_000_000 + seed * 10_000 + n_shots * 10

                for method_index, method in enumerate(fixed_methods):
                    rng = np.random.default_rng(base_seed + method_index + 1)
                    out = run_fixed_episode(psi1, psi2, plane, n_shots, method, rng)
                    rows.append(
                        {
                            "target_id": target_id,
                            "seed": seed,
                            "N": n_shots,
                            "method": method,
                            "plane": plane,
                            "alpha1": alpha1,
                            "alpha2": alpha2,
                            **out,
                        }
                    )

                rng_rl = np.random.default_rng(base_seed + 10)
                out_rl, _ = run_rl_episode(
                    psi1,
                    psi2,
                    n_shots,
                    {a: w.copy() for a, w in trained_weights[n_shots].items()},
                    0.0,
                    rng_rl,
                    update_weights=False,
                )
                rows.append(
                    {
                        "target_id": target_id,
                        "seed": seed,
                        "N": n_shots,
                        "method": "RL_adaptive",
                        "plane": plane,
                        "alpha1": alpha1,
                        "alpha2": alpha2,
                        **out_rl,
                    }
                )

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

    plane_summary = (
        df.groupby(["plane", "method", "N"])
        .agg(
            fidelity_mean=("fidelity", "mean"),
            infidelity_mean=("infidelity", "mean"),
        )
        .reset_index()
    )
    plane_summary.to_csv(f"{run_dir}/plane_summary.csv", index=False)

    allocation_cols = [f"shots_{name}" for name in ACTION_NAMES]
    rl_df = df[df["method"] == "RL_adaptive"].copy()
    allocation_summary = rl_df.groupby("N")[allocation_cols].mean().reset_index()
    allocation_summary.to_csv(f"{run_dir}/rl_allocation_summary.csv", index=False)
    plane_alloc = rl_df.groupby(["plane", "N"])[allocation_cols].mean().reset_index()
    plane_alloc.to_csv(f"{run_dir}/rl_plane_allocation_summary.csv", index=False)

    method_order = ["RL_adaptive", "XYZ_split", "ORACLE_plane", "ZX_split", "Z_only"]

    def plot_metric(mean_col, std_col, ylabel, title, filename, logy=False):
        plt.figure()
        for method in method_order:
            sub = summary[summary["method"] == method].sort_values("N")
            x = sub["N"].to_numpy()
            y = sub[mean_col].to_numpy()
            s = sub[std_col].to_numpy()
            lower = np.maximum(y - s, 1e-12) if logy else y - s
            plt.plot(x, y, marker="o", label=method)
            plt.fill_between(x, lower, y + s, alpha=0.2)
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
        "fidelity_mean",
        "fidelity_std",
        "Mean Fidelity",
        "Exp05: RL on Hidden-Plane 2-Qubit Product States",
        "fidelity_vs_shots.png",
    )
    plot_metric(
        "infidelity_mean",
        "infidelity_std",
        "Mean Infidelity",
        "Exp05: RL on Hidden-Plane 2-Qubit Product States",
        "infidelity_vs_shots.png",
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
    plt.ylabel("Mean RL measurements")
    plt.title("Exp05: RL joint-action allocation")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/rl_allocation_vs_shots.png", dpi=200)
    plt.close()

    with open(f"{run_dir}/notes.txt", "w", encoding="utf-8") as f:
        f.write("Experiment 5: RL on hidden-plane product states\n")
        f.write("Latent plane: XY, XZ, or YZ, shared by both qubits.\n")
        f.write("The RL agent is not given the plane label.\n")
        f.write("XYZ_split is the complete fixed baseline.\n")
        f.write("ORACLE_plane is an informed upper benchmark.\n")
        f.write(f"Training episodes per budget: {NUM_TRAIN_EPISODES}\n")
        f.write(f"Warm start repeats: {WARM_START_REPEATS}\n")
        f.write(f"Minimum active component: {MIN_ACTIVE_COMPONENT}\n")
        f.write(f"Epsilon: {EPSILON_START} to {EPSILON_END}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Gamma: {GAMMA}\n")
        f.write(f"Terminal bonus weight: {TERMINAL_BONUS_WEIGHT}\n")

    print("Saved results to:", run_dir)


if __name__ == "__main__":
    main()