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
NUM_TRAIN_EPISODES = int(os.getenv("EXP05_TRAIN_EPISODES", "12000"))
NUM_TEST_TARGETS = int(os.getenv("EXP05_TEST_TARGETS", "2000"))
NUM_VALIDATION_TARGETS = int(os.getenv("EXP05_VALIDATION_TARGETS", "30"))
NUM_VALIDATION_SEEDS = int(os.getenv("EXP05_VALIDATION_SEEDS", "3"))
NUM_RESTARTS = int(os.getenv("EXP05_RESTARTS", "5"))
NUM_TEST_SEEDS = int(os.getenv("EXP05_TEST_SEEDS", "1"))
MIN_ACTIVE_COMPONENT = float(os.getenv("EXP05_MIN_ACTIVE_COMPONENT", "0.35"))
NUM_BOOTSTRAP = int(os.getenv("EXP05_BOOTSTRAP", "10000"))
BOOTSTRAP_SEED = 20260318

EPSILON_START = 0.30
EPSILON_END = 0.01
LEARNING_RATE = 0.0035
GAMMA = 1.0
TERMINAL_BONUS_WEIGHT = 0.0
WARM_START_REPEATS = 1  # one balanced pass through ZZ, XX, YY before adaptation

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
# + 9 joint-action fractions + 3 approximate plane posterior probabilities
# + progress + bias
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


def approximate_plane_posterior(
    counts1: Dict[str, np.ndarray],
    counts2: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Approximate posterior probabilities for the latent planes XY, XZ, YZ.

    Each candidate plane implies one exactly inactive Bloch component:
        XY -> Z inactive
        XZ -> Y inactive
        YZ -> X inactive

    We use a Gaussian approximation for the empirical Pauli expectation.
    A candidate plane receives high evidence when its implied inactive
    component is close to zero on BOTH qubits relative to the estimated
    standard error. The three plane priors are uniform.
    """
    b1 = np.array(estimate_bloch(counts1), dtype=float)
    b2 = np.array(estimate_bloch(counts2), dtype=float)

    u1 = np.array(
        [measurement_uncertainty(counts1[b]) for b in ("X", "Y", "Z")],
        dtype=float,
    )
    u2 = np.array(
        [measurement_uncertainty(counts2[b]) for b in ("X", "Y", "Z")],
        dtype=float,
    )

    # Coordinate index that should be zero for each plane.
    inactive_index = {
        "XY": 2,  # Z
        "XZ": 1,  # Y
        "YZ": 0,  # X
    }

    log_evidence = []
    for plane in ("XY", "XZ", "YZ"):
        j = inactive_index[plane]

        # Gaussian log likelihood (up to a shared additive constant)
        # under the hypothesis that the inactive component equals zero.
        z1 = b1[j] / max(u1[j], 1e-8)
        z2 = b2[j] / max(u2[j], 1e-8)

        ll = (
            -0.5 * z1 * z1
            - np.log(max(u1[j], 1e-8))
            -0.5 * z2 * z2
            - np.log(max(u2[j], 1e-8))
        )
        log_evidence.append(ll)

    log_evidence = np.array(log_evidence, dtype=float)
    log_evidence -= np.max(log_evidence)
    probs = np.exp(log_evidence)
    probs /= np.sum(probs)
    return probs


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
    plane_probs = approximate_plane_posterior(counts1, counts2)
    progress = shots_used / total_shots

    state = np.concatenate(
        [
            bloch1,
            bloch2,
            uncertainties,
            local_fracs,
            action_fracs,
            plane_probs,
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
    # action fracs 18:27, plane posterior 27:30, progress 30, bias 31.
    plane_names = ["XY", "XZ", "YZ"]
    for action in ACTIONS:
        w = np.zeros(STATE_DIM, dtype=float)
        bq1, bq2 = action
        unc_q1 = {"X": 6, "Y": 7, "Z": 8}[bq1]
        unc_q2 = {"X": 9, "Y": 10, "Z": 11}[bq2]
        w[unc_q1] = 0.05
        w[unc_q2] = 0.05

        # Small structured prior: an action is favored when both selected
        # bases are compatible with a high-probability latent plane.
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

        # With GAMMA = 1 and no terminal bonus, these incremental
        # rewards telescope across the episode:
        #   sum_t (F_{t+1} - F_t) = F_final - F_after_warm_start.
        # Because the warm start is fixed, maximizing expected return
        # is exactly aligned with maximizing final fidelity.
        reward = f_new - f_prev

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



def evaluate_rl_policy(weights, n_shots, targets, seed_offset):
    scores = []
    for target_id, (psi1, psi2, plane, alpha1, alpha2) in enumerate(targets):
        for seed in range(NUM_VALIDATION_SEEDS):
            rng = np.random.default_rng(seed_offset + target_id * 10000 + seed * 100 + n_shots)
            out, _ = run_rl_episode(
                psi1, psi2, n_shots,
                {a: w.copy() for a, w in weights.items()},
                0.0, rng, update_weights=False
            )
            scores.append(out["fidelity"])
    return float(np.mean(scores))


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"results/{timestamp}_exp05_belief_aware_qlearning_final"
    os.makedirs(run_dir, exist_ok=True)

    val_master = np.random.default_rng(SEED + 2000)
    test_master = np.random.default_rng(SEED + 3000)

    validation_targets = [
        sample_hidden_plane_product_state(val_master)
        for _ in range(NUM_VALIDATION_TARGETS)
    ]

    trained_weights = {}
    training_records = []

    for n_shots in SHOT_BUDGETS:
        candidates = []

        for restart in range(NUM_RESTARTS):
            weights = initialize_weights()
            restart_rng = np.random.default_rng(
                SEED + 100000 * n_shots + 10000 * restart
            )

            for ep in range(NUM_TRAIN_EPISODES):
                psi1, psi2, _, _, _ = sample_hidden_plane_product_state(restart_rng)
                frac = ep / max(NUM_TRAIN_EPISODES - 1, 1)
                epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)

                rng = np.random.default_rng(
                    10000000 + n_shots * 1000000 + restart * 100000 + ep
                )

                _, weights = run_rl_episode(
                    psi1, psi2, n_shots, weights, epsilon, rng,
                    update_weights=True
                )

            validation_fidelity = evaluate_rl_policy(
                weights,
                n_shots,
                validation_targets,
                70000000 + n_shots * 100000 + restart * 10000,
            )

            candidates.append(
                (validation_fidelity, {a: w.copy() for a, w in weights.items()})
            )

            training_records.append({
                "N": n_shots,
                "restart": restart,
                "validation_fidelity": validation_fidelity,
            })

            print(
                f"N={n_shots}, restart={restart}, "
                f"validation fidelity={validation_fidelity:.6f}"
            )

        best_validation, best_weights = max(candidates, key=lambda item: item[0])
        trained_weights[n_shots] = best_weights
        print(
            f"Selected N={n_shots} policy with "
            f"validation fidelity={best_validation:.6f}"
        )

    pd.DataFrame(training_records).to_csv(
        f"{run_dir}/training_restart_validation.csv", index=False
    )

    test_targets = [
        sample_hidden_plane_product_state(test_master)
        for _ in range(NUM_TEST_TARGETS)
    ]

    rows = []
    fixed_methods = ["Z_only", "ZX_split", "XYZ_split", "ORACLE_plane"]

    for target_id, (psi1, psi2, plane, alpha1, alpha2) in enumerate(test_targets):
        for seed in range(NUM_TEST_SEEDS):
            for n_shots in SHOT_BUDGETS:
                base_seed = (
                    100000000
                    + target_id * 1000000
                    + seed * 10000
                    + n_shots * 10
                )

                for method_index, method in enumerate(fixed_methods):
                    rng = np.random.default_rng(base_seed + method_index + 1)
                    out = run_fixed_episode(
                        psi1, psi2, plane, n_shots, method, rng
                    )
                    rows.append({
                        "target_id": target_id,
                        "seed": seed,
                        "N": n_shots,
                        "method": method,
                        "plane": plane,
                        "alpha1": alpha1,
                        "alpha2": alpha2,
                        **out,
                    })

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
                rows.append({
                    "target_id": target_id,
                    "seed": seed,
                    "N": n_shots,
                    "method": "RL_adaptive",
                    "plane": plane,
                    "alpha1": alpha1,
                    "alpha2": alpha2,
                    **out_rl,
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{run_dir}/metrics.csv", index=False)

    # ============================================================
    # Publication statistics: 95% percentile bootstrap CIs
    # ============================================================
    def bootstrap_mean_ci(values, rng, n_boot=NUM_BOOTSTRAP, confidence=0.95):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan, np.nan, np.nan

        point = float(np.mean(values))
        n = len(values)
        boot_means = np.empty(n_boot, dtype=float)

        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = np.mean(values[idx])

        alpha = 1.0 - confidence
        lo, hi = np.quantile(
            boot_means,
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )
        return point, float(lo), float(hi)


    fixed_methods = ["Z_only", "ZX_split", "XYZ_split", "ORACLE_plane"]
    method_order = [
        "RL_adaptive", "XYZ_split", "ORACLE_plane", "ZX_split", "Z_only"
    ]
    method_code = {m: i + 1 for i, m in enumerate(method_order)}

    summary_rows = []
    for method in method_order:
        for n_shots in SHOT_BUDGETS:
            sub = df[(df["method"] == method) & (df["N"] == n_shots)]
            code = method_code[method]

            f_mean, f_lo, f_hi = bootstrap_mean_ci(
                sub["fidelity"].to_numpy(),
                np.random.default_rng(
                    BOOTSTRAP_SEED + n_shots * 1000 + code * 10 + 1
                ),
            )
            i_mean, i_lo, i_hi = bootstrap_mean_ci(
                sub["infidelity"].to_numpy(),
                np.random.default_rng(
                    BOOTSTRAP_SEED + n_shots * 1000 + code * 10 + 2
                ),
            )

            summary_rows.append({
                "method": method,
                "N": n_shots,
                "n_trials": len(sub),
                "fidelity_mean": f_mean,
                "fidelity_ci95_low": f_lo,
                "fidelity_ci95_high": f_hi,
                "infidelity_mean": i_mean,
                "infidelity_ci95_low": i_lo,
                "infidelity_ci95_high": i_hi,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{run_dir}/summary_bootstrap.csv", index=False)


    # ============================================================
    # Plane-specific descriptive summary
    # ============================================================
    plane_rows = []
    for plane in ("XY", "XZ", "YZ"):
        for method in method_order:
            for n_shots in SHOT_BUDGETS:
                sub = df[
                    (df["plane"] == plane)
                    & (df["method"] == method)
                    & (df["N"] == n_shots)
                ]

                code = method_code[method]
                plane_code = {"XY": 1, "XZ": 2, "YZ": 3}[plane]

                f_mean, f_lo, f_hi = bootstrap_mean_ci(
                    sub["fidelity"].to_numpy(),
                    np.random.default_rng(
                        BOOTSTRAP_SEED
                        + 100_000
                        + plane_code * 10_000
                        + n_shots * 10
                        + code
                    ),
                )

                plane_rows.append({
                    "plane": plane,
                    "method": method,
                    "N": n_shots,
                    "n_trials": len(sub),
                    "fidelity_mean": f_mean,
                    "fidelity_ci95_low": f_lo,
                    "fidelity_ci95_high": f_hi,
                    "infidelity_mean": float(sub["infidelity"].mean()),
                })

    plane_summary = pd.DataFrame(plane_rows)
    plane_summary.to_csv(f"{run_dir}/plane_summary_bootstrap.csv", index=False)


    # ============================================================
    # RL allocation summaries
    # ============================================================
    allocation_cols = [f"shots_{name}" for name in ACTION_NAMES]
    rl_df = df[df["method"] == "RL_adaptive"].copy()

    allocation_summary = (
        rl_df.groupby("N")[allocation_cols].mean().reset_index()
    )
    allocation_summary.to_csv(
        f"{run_dir}/rl_allocation_summary.csv", index=False
    )

    plane_alloc = (
        rl_df.groupby(["plane", "N"])[allocation_cols].mean().reset_index()
    )
    plane_alloc.to_csv(
        f"{run_dir}/rl_plane_allocation_summary.csv", index=False
    )

    # Convert joint-action counts into local X/Y/Z usage for each qubit.
    local_rows = []
    for _, row in rl_df.iterrows():
        rec = {
            "target_id": row["target_id"],
            "seed": row["seed"],
            "N": row["N"],
            "plane": row["plane"],
        }
        for q in (1, 2):
            for basis in ("X", "Y", "Z"):
                total = 0.0
                for name, (b1, b2) in zip(ACTION_NAMES, ACTIONS):
                    chosen = b1 if q == 1 else b2
                    if chosen == basis:
                        total += float(row[f"shots_{name}"])
                rec[f"q{q}_{basis}"] = total
        local_rows.append(rec)

    local_df = pd.DataFrame(local_rows)

    local_plane_summary = (
        local_df.groupby(["plane", "N"])[
            ["q1_X", "q1_Y", "q1_Z", "q2_X", "q2_Y", "q2_Z"]
        ]
        .mean()
        .reset_index()
    )
    local_plane_summary.to_csv(
        f"{run_dir}/rl_plane_local_basis_summary.csv", index=False
    )

    # Fraction of measurements spent on the truly inactive basis.
    inactive_basis = {"XY": "Z", "XZ": "Y", "YZ": "X"}
    inactive_records = []

    for _, row in local_df.iterrows():
        b = inactive_basis[row["plane"]]
        n = float(row["N"])
        inactive_records.append({
            "target_id": row["target_id"],
            "N": row["N"],
            "plane": row["plane"],
            "inactive_fraction_q1": float(row[f"q1_{b}"]) / n,
            "inactive_fraction_q2": float(row[f"q2_{b}"]) / n,
            "inactive_fraction_mean": (
                float(row[f"q1_{b}"]) + float(row[f"q2_{b}"])
            ) / (2.0 * n),
        })

    inactive_df = pd.DataFrame(inactive_records)
    inactive_df.to_csv(
        f"{run_dir}/rl_inactive_basis_trials.csv", index=False
    )

    inactive_summary_rows = []
    for plane in ("XY", "XZ", "YZ"):
        for n_shots in SHOT_BUDGETS:
            sub = inactive_df[
                (inactive_df["plane"] == plane)
                & (inactive_df["N"] == n_shots)
            ]
            mean, lo, hi = bootstrap_mean_ci(
                sub["inactive_fraction_mean"].to_numpy(),
                np.random.default_rng(
                    BOOTSTRAP_SEED
                    + 300_000
                    + {"XY": 1, "XZ": 2, "YZ": 3}[plane] * 10_000
                    + n_shots
                ),
            )
            inactive_summary_rows.append({
                "plane": plane,
                "N": n_shots,
                "inactive_fraction_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            })

    inactive_summary = pd.DataFrame(inactive_summary_rows)
    inactive_summary.to_csv(
        f"{run_dir}/rl_inactive_basis_fraction_bootstrap.csv", index=False
    )


    # ============================================================
    # Paired comparisons
    # ============================================================
    wide = (
        df.pivot_table(
            index=["target_id", "seed", "N"],
            columns="method",
            values="fidelity",
        )
        .dropna()
        .reset_index()
    )

    paired_rows = []
    for n_shots in SHOT_BUDGETS:
        sub = wide[wide["N"] == n_shots]

        for comparison_name, lhs, rhs in [
            ("RL_minus_XYZ", "RL_adaptive", "XYZ_split"),
            ("ORACLE_minus_XYZ", "ORACLE_plane", "XYZ_split"),
            ("ORACLE_minus_RL", "ORACLE_plane", "RL_adaptive"),
        ]:
            diff = (sub[lhs] - sub[rhs]).to_numpy()

            mean, lo, hi = bootstrap_mean_ci(
                diff,
                np.random.default_rng(
                    BOOTSTRAP_SEED
                    + 500_000
                    + n_shots * 100
                    + {
                        "RL_minus_XYZ": 1,
                        "ORACLE_minus_XYZ": 2,
                        "ORACLE_minus_RL": 3,
                    }[comparison_name]
                ),
            )

            paired_rows.append({
                "comparison": comparison_name,
                "N": n_shots,
                "n_pairs": len(diff),
                "mean_difference": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "lhs_win_rate": float(np.mean(diff > 0.0)),
                "tie_rate": float(np.mean(np.isclose(diff, 0.0, atol=1e-12))),
                "lhs_loss_rate": float(np.mean(diff < 0.0)),
            })

    paired_summary = pd.DataFrame(paired_rows)
    paired_summary.to_csv(
        f"{run_dir}/paired_comparisons_bootstrap.csv", index=False
    )


    # ============================================================
    # Oracle-gap recovery
    # ============================================================
    means = summary.pivot(index="N", columns="method", values="fidelity_mean")
    gap_rows = []

    for n_shots in means.index:
        f_rl = float(means.loc[n_shots, "RL_adaptive"])
        f_xyz = float(means.loc[n_shots, "XYZ_split"])
        f_oracle = float(means.loc[n_shots, "ORACLE_plane"])

        oracle_gap = f_oracle - f_xyz
        recovered = (
            (f_rl - f_xyz) / oracle_gap
            if abs(oracle_gap) > 1e-12
            else np.nan
        )

        gap_rows.append({
            "N": int(n_shots),
            "F_RL": f_rl,
            "F_XYZ": f_xyz,
            "F_ORACLE": f_oracle,
            "oracle_gap": oracle_gap,
            "rl_gain_over_xyz": f_rl - f_xyz,
            "oracle_gap_recovered_fraction": recovered,
        })

    pd.DataFrame(gap_rows).to_csv(
        f"{run_dir}/oracle_gap_recovery.csv", index=False
    )


    # ============================================================
    # Publication plots
    # ============================================================
    def plot_metric_ci(
        mean_col,
        low_col,
        high_col,
        ylabel,
        title,
        filename,
        logy=False,
    ):
        plt.figure(figsize=(6.4, 4.4))

        for method in method_order:
            sub = summary[summary["method"] == method].sort_values("N")
            x = sub["N"].to_numpy()
            y = sub[mean_col].to_numpy()
            lo = sub[low_col].to_numpy()
            hi = sub[high_col].to_numpy()

            plt.plot(x, y, marker="o", label=method)
            plt.fill_between(x, lo, hi, alpha=0.18)

        plt.xscale("log")
        if logy:
            plt.yscale("log")

        plt.xlabel("Measurement budget (N)")
        plt.ylabel(ylabel)
        plt.title("Exp. 5: Hidden-plane two-qubit tomography")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{run_dir}/{filename}", dpi=300, bbox_inches="tight")
        plt.close()


    plot_metric_ci(
        "fidelity_mean",
        "fidelity_ci95_low",
        "fidelity_ci95_high",
        "Mean fidelity",
        "Exp. 5: Hidden-plane two-qubit tomography",
        "exp5_fidelity_vs_shots_bootstrap95.png",
    )

    plot_metric_ci(
        "infidelity_mean",
        "infidelity_ci95_low",
        "infidelity_ci95_high",
        "Mean infidelity",
        "Exp. 5: Hidden-plane two-qubit tomography",
        "exp5_infidelity_vs_shots_bootstrap95.png",
        logy=True,
    )


    # Paired RL - XYZ plot
    rl_xyz = paired_summary[
        paired_summary["comparison"] == "RL_minus_XYZ"
    ].sort_values("N")

    plt.figure(figsize=(6.4, 4.4))
    x = rl_xyz["N"].to_numpy()
    y = rl_xyz["mean_difference"].to_numpy()
    lo = rl_xyz["ci95_low"].to_numpy()
    hi = rl_xyz["ci95_high"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])

    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xscale("log")
    plt.xlabel("Measurement budget (N)")
    plt.ylabel("Mean fidelity difference (RL - XYZ)")
    plt.title("Exp. 5: Paired RL vs. XYZ")
    plt.tight_layout()
    plt.savefig(
        f"{run_dir}/exp5_paired_RL_minus_XYZ_bootstrap95.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


    # Inactive-basis usage plot: directly measures whether RL learns the plane.
    plt.figure(figsize=(6.4, 4.4))
    for plane in ("XY", "XZ", "YZ"):
        sub = inactive_summary[
            inactive_summary["plane"] == plane
        ].sort_values("N")
        x = sub["N"].to_numpy()
        y = sub["inactive_fraction_mean"].to_numpy()
        lo = sub["ci95_low"].to_numpy()
        hi = sub["ci95_high"].to_numpy()

        plt.plot(x, y, marker="o", label=plane)
        plt.fill_between(x, lo, hi, alpha=0.15)

    plt.axhline(
        1.0 / 3.0,
        linestyle="--",
        linewidth=1.0,
        label="Balanced XYZ reference",
    )
    plt.xscale("log")
    plt.ylim(0.0, 0.5)
    plt.xlabel("Measurement budget (N)")
    plt.ylabel("Fraction on truly inactive basis")
    plt.title("Exp. 5: RL identification of hidden plane")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{run_dir}/exp5_inactive_basis_fraction_bootstrap95.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


    # ============================================================
    # Save trained weights and notes
    # ============================================================
    for n_shots, weights in trained_weights.items():
        for action in ACTIONS:
            name = action[0] + action[1]
            np.save(
                f"{run_dir}/weights_{name}_N{n_shots}.npy",
                weights[action],
            )

    with open(f"{run_dir}/notes.txt", "w", encoding="utf-8") as f:
        f.write("Experiment 5: hidden-plane two-qubit RL tomography\\n")
        f.write(
            "State family, actions, reward, estimator, warm start, training "
            "restarts, and validation-based policy selection are unchanged "
            "from the most recent stabilized Exp. 5 implementation.\\n"
        )
        f.write(f"Shot budgets: {SHOT_BUDGETS}\\n")
        f.write(f"Training episodes per restart: {NUM_TRAIN_EPISODES}\\n")
        f.write(f"Training restarts: {NUM_RESTARTS}\\n")
        f.write(f"Validation targets: {NUM_VALIDATION_TARGETS}\\n")
        f.write(f"Validation seeds: {NUM_VALIDATION_SEEDS}\\n")
        f.write(f"Independent evaluation targets per budget: {NUM_TEST_TARGETS}\\n")
        f.write(f"Evaluation seeds per target: {NUM_TEST_SEEDS}\\n")
        f.write(f"Bootstrap resamples: {NUM_BOOTSTRAP}\\n")
        f.write("CI: 95% percentile bootstrap confidence interval of the mean\\n")
        f.write(
            "Paired comparisons are computed on matched target-state trials "
            "for RL vs XYZ, oracle vs XYZ, and oracle vs RL.\\n"
        )
        f.write(
            "Bootstrap uncertainty is conditional on the validation-selected "
            "trained policy and does not include between-training-run "
            "selection variability.\\n"
        )
        f.write(f"Minimum active Bloch component: {MIN_ACTIVE_COMPONENT}\\n")
        f.write(f"Warm-start repeats: {WARM_START_REPEATS}\\n")
        f.write(f"Epsilon: {EPSILON_START} to {EPSILON_END}\\n")
        f.write(f"Learning rate: {LEARNING_RATE}\\n")
        f.write(f"Gamma: {GAMMA}\\n")
        f.write(f"Terminal bonus weight: {TERMINAL_BONUS_WEIGHT}\\n")
        f.write(f"Bootstrap seed: {BOOTSTRAP_SEED}\\n")

    print("Saved results to:", run_dir)
    print("Key outputs:")
    print("- metrics.csv")
    print("- training_restart_validation.csv")
    print("- summary_bootstrap.csv")
    print("- paired_comparisons_bootstrap.csv")
    print("- oracle_gap_recovery.csv")
    print("- plane_summary_bootstrap.csv")
    print("- rl_inactive_basis_fraction_bootstrap.csv")
    print("- exp5_fidelity_vs_shots_bootstrap95.png")
    print("- exp5_paired_RL_minus_XYZ_bootstrap95.png")
    print("- exp5_inactive_basis_fraction_bootstrap95.png")
    print("- notes.txt")


if __name__ == "__main__":
    main()
