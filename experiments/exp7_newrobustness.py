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
run_dir = f"results/{timestamp}_exp07_basis_dependent_noise"
os.makedirs(run_dir, exist_ok=True)

# ========================
# Configuration
# ========================
SHOT_BUDGETS = [10, 25, 50, 100, 250]
DEGRADED_NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]
CLEAN_NOISE_RANGE = (0.005, 0.025)
NUM_TRAIN_EPISODES = 4000
NUM_TEST_TARGETS = 60
NUM_TEST_SEEDS = 5
CALIBRATION_SHOTS_PER_BASIS = 100
EPSILON_START = 0.35
EPSILON_END = 0.03
EPSILON_TEST = 0.0
LEARNING_RATE = 0.008
GAMMA = 0.95
TERMINAL_BONUS_WEIGHT = 0.15
MASTER_SEED = 123

# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([
        np.cos(theta / 2.0),
        np.exp(1j * phi) * np.sin(theta / 2.0),
    ], dtype=complex)


def sample_uniform_qubit_state(rng: np.random.Generator) -> Tuple[np.ndarray, float, float]:
    z = rng.uniform(-1.0, 1.0)
    theta = float(np.arccos(z))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    return state_from_angles(theta, phi), theta, phi


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def hadamard() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)


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
    probs = np.abs(rotated) ** 2
    return probs / np.sum(probs)


def sample_one_shot(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice([0, 1], p=probs))


def noisy_sample(probs: np.ndarray, rng: np.random.Generator, flip_prob: float) -> int:
    outcome = sample_one_shot(probs, rng)
    return 1 - outcome if rng.random() < flip_prob else outcome

# ========================
# Noise utilities
# ========================
BASES = ("X", "Y", "Z")


def sample_noise_profile(rng: np.random.Generator, degraded_level=None):
    degraded_basis = str(rng.choice(BASES))
    if degraded_level is None:
        degraded_level = float(rng.choice(DEGRADED_NOISE_LEVELS))
    profile = {b: float(rng.uniform(*CLEAN_NOISE_RANGE)) for b in BASES}
    profile[degraded_basis] = float(degraded_level)
    return profile, degraded_basis


def estimate_noise_from_calibration(true_noise, rng, shots_per_basis):
    estimates = {}
    for basis in BASES:
        flips = int(rng.binomial(shots_per_basis, true_noise[basis]))
        estimates[basis] = float(np.clip(flips / shots_per_basis, 0.0, 0.45))
    return estimates

# ========================
# Estimation utilities
# ========================
def fresh_counts() -> Dict[str, np.ndarray]:
    return {b: np.zeros(2, dtype=int) for b in BASES}


def corrected_coordinate(counts: np.ndarray, estimated_flip_prob: float) -> float:
    total = int(np.sum(counts))
    if total == 0:
        return 0.0
    observed = float((counts[0] - counts[1]) / total)
    attenuation = max(1.0 - 2.0 * estimated_flip_prob, 0.10)
    return float(np.clip(observed / attenuation, -1.0, 1.0))


def estimate_bloch_from_counts(counts, estimated_noise):
    return (
        corrected_coordinate(counts["X"], estimated_noise["X"]),
        corrected_coordinate(counts["Y"], estimated_noise["Y"]),
        corrected_coordinate(counts["Z"], estimated_noise["Z"]),
    )


def bloch_to_state(x: float, y: float, z: float) -> np.ndarray:
    r = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(r)
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=complex)
    x_n, y_n, z_n = r / norm
    theta = float(np.arccos(np.clip(z_n, -1.0, 1.0)))
    phi = float(np.mod(np.arctan2(y_n, x_n), 2.0 * np.pi))
    return state_from_angles(theta, phi)


def estimate_state(counts, estimated_noise):
    return bloch_to_state(*estimate_bloch_from_counts(counts, estimated_noise))


def product_fidelity_from_counts(psi1, psi2, counts1, counts2, estimated_noise):
    psi1_hat = estimate_state(counts1, estimated_noise)
    psi2_hat = estimate_state(counts2, estimated_noise)
    return fidelity(psi1_hat, psi1) * fidelity(psi2_hat, psi2)

# ========================
# RL setup
# ========================
ACTIONS = [
    ("Z", "Z"), ("Z", "X"), ("Z", "Y"),
    ("X", "Z"), ("X", "X"), ("X", "Y"),
    ("Y", "Z"), ("Y", "X"), ("Y", "Y"),
]
ACTION_NAMES = [a + b for a, b in ACTIONS]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
STATE_DIM = 35


def effective_uncertainty(counts, estimated_flip_prob):
    total = int(np.sum(counts))
    attenuation = max(1.0 - 2.0 * estimated_flip_prob, 0.10)
    return float(1.0 / (np.sqrt(total + 1.0) * attenuation))


def build_state(counts1, counts2, action_counts, shots_used, total_shots, estimated_noise):
    x1, y1, z1 = estimate_bloch_from_counts(counts1, estimated_noise)
    x2, y2, z2 = estimate_bloch_from_counts(counts2, estimated_noise)
    uncertainties = np.array([
        effective_uncertainty(counts1["X"], estimated_noise["X"]),
        effective_uncertainty(counts1["Y"], estimated_noise["Y"]),
        effective_uncertainty(counts1["Z"], estimated_noise["Z"]),
        effective_uncertainty(counts2["X"], estimated_noise["X"]),
        effective_uncertainty(counts2["Y"], estimated_noise["Y"]),
        effective_uncertainty(counts2["Z"], estimated_noise["Z"]),
    ])
    noise_features = np.array([estimated_noise["X"], estimated_noise["Y"], estimated_noise["Z"]])
    progress = shots_used / total_shots if total_shots > 0 else 0.0
    action_fracs = action_counts.astype(float) / max(total_shots, 1)
    action_deficits = progress / len(ACTIONS) - action_fracs
    state = np.concatenate([
        np.array([x1, y1, z1, x2, y2, z2]),
        uncertainties,
        noise_features,
        action_fracs,
        action_deficits,
        np.array([progress, 1.0]),
    ])
    if state.shape[0] != STATE_DIM:
        raise RuntimeError(f"State dimension mismatch: {state.shape[0]} != {STATE_DIM}")
    return state


def initialize_weights():
    weights = {}
    unc_q1 = {"X": 6, "Y": 7, "Z": 8}
    unc_q2 = {"X": 9, "Y": 10, "Z": 11}
    deficit_start = 24
    for action in ACTIONS:
        w = np.zeros(STATE_DIM)
        b1, b2 = action
        idx = ACTION_INDEX[action]
        w[unc_q1[b1]] = 0.08
        w[unc_q2[b2]] = 0.08
        w[deficit_start + idx] = 0.02
        weights[action] = w
    return weights


def q_values(state, weights):
    return np.array([float(np.dot(weights[a], state)) for a in ACTIONS])


def epsilon_greedy(state, weights, epsilon, rng):
    if rng.random() < epsilon:
        return ACTIONS[int(rng.integers(len(ACTIONS)))]
    values = q_values(state, weights)
    best = np.flatnonzero(np.isclose(values, np.max(values)))
    return ACTIONS[int(rng.choice(best))]

# ========================
# Episode helpers
# ========================
def apply_joint_measurement(psi1, psi2, action, counts1, counts2, true_noise, rng):
    b1, b2 = action
    o1 = noisy_sample(probs_for_basis(psi1, b1), rng, true_noise[b1])
    o2 = noisy_sample(probs_for_basis(psi2, b2), rng, true_noise[b2])
    counts1[b1][o1] += 1
    counts2[b2][o2] += 1


def run_rl_episode(psi1, psi2, total_shots, weights, epsilon, rng, update_weights,
                   true_noise, estimated_noise):
    counts1, counts2 = fresh_counts(), fresh_counts()
    action_counts = np.zeros(len(ACTIONS), dtype=int)
    shots_used = 0

    for action in [("Z", "Z"), ("X", "X"), ("Y", "Y")]:
        if shots_used >= total_shots:
            break
        apply_joint_measurement(psi1, psi2, action, counts1, counts2, true_noise, rng)
        action_counts[ACTION_INDEX[action]] += 1
        shots_used += 1

    f_prev = product_fidelity_from_counts(psi1, psi2, counts1, counts2, estimated_noise)

    for shot_idx in range(shots_used, total_shots):
        state = build_state(counts1, counts2, action_counts, shot_idx, total_shots, estimated_noise)
        action = epsilon_greedy(state, weights, epsilon, rng)
        q_current = float(np.dot(weights[action], state))

        apply_joint_measurement(psi1, psi2, action, counts1, counts2, true_noise, rng)
        action_counts[ACTION_INDEX[action]] += 1
        f_new = product_fidelity_from_counts(psi1, psi2, counts1, counts2, estimated_noise)

        terminal = shot_idx == total_shots - 1
        reward = (f_new - f_prev) + (TERMINAL_BONUS_WEIGHT * f_new if terminal else 0.0)
        if terminal:
            q_next = 0.0
        else:
            next_state = build_state(counts1, counts2, action_counts, shot_idx + 1, total_shots, estimated_noise)
            q_next = float(np.max(q_values(next_state, weights)))

        if update_weights:
            td_error = reward + GAMMA * q_next - q_current
            weights[action] += LEARNING_RATE * td_error * state
        f_prev = f_new

    result = {"fidelity": f_prev, "infidelity": 1.0 - f_prev}
    for i, name in enumerate(ACTION_NAMES):
        result[f"shots_{name}"] = int(action_counts[i])
    return result, weights


def fixed_schedule(total_shots, strategy):
    if strategy == "Z_only":
        return [("Z", "Z")] * total_shots
    if strategy == "ZX_split":
        n_z = total_shots // 2
        return [("Z", "Z")] * n_z + [("X", "X")] * (total_shots - n_z)
    if strategy == "XYZ_split":
        n_z = total_shots // 3
        n_x = total_shots // 3
        n_y = total_shots - n_z - n_x
        return [("Z", "Z")] * n_z + [("X", "X")] * n_x + [("Y", "Y")] * n_y
    raise ValueError(f"Unknown strategy: {strategy}")


def run_fixed_episode(psi1, psi2, total_shots, strategy, true_noise, estimated_noise, rng):
    counts1, counts2 = fresh_counts(), fresh_counts()
    for action in fixed_schedule(total_shots, strategy):
        apply_joint_measurement(psi1, psi2, action, counts1, counts2, true_noise, rng)
    f = product_fidelity_from_counts(psi1, psi2, counts1, counts2, estimated_noise)
    return {"fidelity": f, "infidelity": 1.0 - f}

# ========================
# Train one policy per budget
# ========================
master_rng = np.random.default_rng(MASTER_SEED)
trained_weights = {}

for N in SHOT_BUDGETS:
    weights = initialize_weights()
    for ep in range(NUM_TRAIN_EPISODES):
        psi1, _, _ = sample_uniform_qubit_state(master_rng)
        psi2, _, _ = sample_uniform_qubit_state(master_rng)
        true_noise, _ = sample_noise_profile(master_rng)
        calibration_rng = np.random.default_rng(20_000_000 + 100_000 * N + ep)
        estimated_noise = estimate_noise_from_calibration(
            true_noise, calibration_rng, CALIBRATION_SHOTS_PER_BASIS
        )
        frac = ep / max(NUM_TRAIN_EPISODES - 1, 1)
        epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)
        episode_rng = np.random.default_rng(10_000_000 + 100_000 * N + ep)
        _, weights = run_rl_episode(
            psi1, psi2, N, weights, epsilon, episode_rng, True,
            true_noise, estimated_noise
        )
    trained_weights[N] = {a: w.copy() for a, w in weights.items()}
    print(f"Finished training policy for N = {N}")

# ========================
# Evaluation
# ========================
rows = []
for target_id in range(NUM_TEST_TARGETS):
    psi1, theta1, phi1 = sample_uniform_qubit_state(master_rng)
    psi2, theta2, phi2 = sample_uniform_qubit_state(master_rng)

    for seed in range(NUM_TEST_SEEDS):
        for degraded_noise in DEGRADED_NOISE_LEVELS:
            profile_rng = np.random.default_rng(
                30_000_000 + target_id * 100_000 + seed * 1_000 + int(degraded_noise * 100)
            )
            true_noise, degraded_basis = sample_noise_profile(profile_rng, degraded_noise)
            calibration_rng = np.random.default_rng(
                40_000_000 + target_id * 100_000 + seed * 1_000 + int(degraded_noise * 100)
            )
            estimated_noise = estimate_noise_from_calibration(
                true_noise, calibration_rng, CALIBRATION_SHOTS_PER_BASIS
            )

            for N in SHOT_BUDGETS:
                common = {
                    "target_id": target_id,
                    "seed": seed,
                    "N": N,
                    "degraded_noise": degraded_noise,
                    "degraded_basis": degraded_basis,
                    "theta1": theta1,
                    "phi1": phi1,
                    "theta2": theta2,
                    "phi2": phi2,
                    "true_noise_X": true_noise["X"],
                    "true_noise_Y": true_noise["Y"],
                    "true_noise_Z": true_noise["Z"],
                    "estimated_noise_X": estimated_noise["X"],
                    "estimated_noise_Y": estimated_noise["Y"],
                    "estimated_noise_Z": estimated_noise["Z"],
                }

                for method_idx, strategy in enumerate(["Z_only", "ZX_split", "XYZ_split"]):
                    rng = np.random.default_rng(
                        50_000_000 + target_id * 1_000_000 + seed * 10_000 + N * 10
                        + int(degraded_noise * 100) * 100 + method_idx
                    )
                    out = run_fixed_episode(
                        psi1, psi2, N, strategy, true_noise, estimated_noise, rng
                    )
                    rows.append({**common, "method": strategy, **out})

                rng_rl = np.random.default_rng(
                    60_000_000 + target_id * 1_000_000 + seed * 10_000 + N * 10
                    + int(degraded_noise * 100) * 100
                )
                out_rl, _ = run_rl_episode(
                    psi1, psi2, N,
                    {a: w.copy() for a, w in trained_weights[N].items()},
                    EPSILON_TEST, rng_rl, False, true_noise, estimated_noise
                )
                rows.append({**common, "method": "RL_adaptive", **out_rl})

# ========================
# Save summaries
# ========================
df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/metrics.csv", index=False)

summary = df.groupby(["method", "N", "degraded_noise"]).agg(
    fidelity_mean=("fidelity", "mean"),
    fidelity_std=("fidelity", "std"),
    fidelity_count=("fidelity", "count"),
    infidelity_mean=("infidelity", "mean"),
    infidelity_std=("infidelity", "std"),
    infidelity_count=("infidelity", "count"),
).reset_index()
summary["fidelity_ci95"] = 1.96 * summary["fidelity_std"] / np.sqrt(summary["fidelity_count"])
summary["infidelity_ci95"] = 1.96 * summary["infidelity_std"] / np.sqrt(summary["infidelity_count"])
summary.to_csv(f"{run_dir}/summary.csv", index=False)

paired = df[df["method"].isin(["RL_adaptive", "XYZ_split"])].pivot_table(
    index=["target_id", "seed", "N", "degraded_noise", "degraded_basis"],
    columns="method", values="fidelity"
).dropna().reset_index()
paired["delta_fidelity"] = paired["RL_adaptive"] - paired["XYZ_split"]
paired_summary = paired.groupby(["N", "degraded_noise"]).agg(
    delta_mean=("delta_fidelity", "mean"),
    delta_std=("delta_fidelity", "std"),
    count=("delta_fidelity", "count"),
    rl_win_rate=("delta_fidelity", lambda x: float(np.mean(x > 0))),
).reset_index()
paired_summary["delta_ci95"] = 1.96 * paired_summary["delta_std"] / np.sqrt(paired_summary["count"])
paired_summary.to_csv(f"{run_dir}/rl_vs_xyz_paired_summary.csv", index=False)

allocation_cols = [f"shots_{name}" for name in ACTION_NAMES]
rl_df = df[df["method"] == "RL_adaptive"].copy()
allocation_summary = rl_df.groupby(["N", "degraded_noise", "degraded_basis"])[allocation_cols].mean().reset_index()
allocation_summary.to_csv(f"{run_dir}/rl_allocation_summary.csv", index=False)

# ========================
# Plots
# ========================
def plot_vs_shots(noise_level, metric_mean, metric_ci, ylabel, filename, log_y=False):
    subset = summary[np.isclose(summary["degraded_noise"], noise_level)]
    plt.figure(figsize=(9, 6))
    for method in subset["method"].unique():
        sub = subset[subset["method"] == method].sort_values("N")
        x = sub["N"].to_numpy()
        y = sub[metric_mean].to_numpy()
        ci = sub[metric_ci].to_numpy()
        lower, upper = y - ci, y + ci
        if metric_mean.startswith("fidelity"):
            lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        elif log_y:
            lower = np.maximum(lower, 1e-12)
        plt.plot(x, y, marker="o", label=method)
        plt.fill_between(x, lower, upper, alpha=0.2)
    plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("Tomography shots (N)")
    plt.ylabel(ylabel)
    plt.title(f"Two-Qubit Estimation with {noise_level:.0%} Noise in One Unknown Basis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/{filename}", dpi=200)
    plt.close()


for noise_level in DEGRADED_NOISE_LEVELS:
    tag = int(round(noise_level * 100))
    plot_vs_shots(noise_level, "fidelity_mean", "fidelity_ci95", "Mean Fidelity",
                  f"fidelity_vs_shots_noise_{tag:02d}.png")
    plot_vs_shots(noise_level, "infidelity_mean", "infidelity_ci95", "Mean Infidelity",
                  f"infidelity_vs_shots_noise_{tag:02d}.png", log_y=True)

headline_budget = 50
headline = summary[summary["N"] == headline_budget]
plt.figure(figsize=(9, 6))
for method in headline["method"].unique():
    sub = headline[headline["method"] == method].sort_values("degraded_noise")
    x = sub["degraded_noise"].to_numpy()
    y = sub["fidelity_mean"].to_numpy()
    ci = sub["fidelity_ci95"].to_numpy()
    plt.plot(x, y, marker="o", label=method)
    plt.fill_between(x, np.clip(y - ci, 0, 1), np.clip(y + ci, 0, 1), alpha=0.2)
plt.xlabel("Readout-flip probability of degraded basis")
plt.ylabel("Mean Fidelity")
plt.title(f"Noise Robustness at N = {headline_budget} Tomography Shots")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_dir}/fidelity_vs_noise_N{headline_budget}.png", dpi=200)
plt.close()

head_delta = paired_summary[paired_summary["N"] == headline_budget].sort_values("degraded_noise")
plt.figure(figsize=(9, 6))
x = head_delta["degraded_noise"].to_numpy()
y = head_delta["delta_mean"].to_numpy()
ci = head_delta["delta_ci95"].to_numpy()
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.plot(x, y, marker="o")
plt.fill_between(x, y - ci, y + ci, alpha=0.2)
plt.xlabel("Readout-flip probability of degraded basis")
plt.ylabel(r"$\Delta F = F_{RL} - F_{XYZ}$")
plt.title(f"Paired RL Advantage over XYZ at N = {headline_budget}")
plt.tight_layout()
plt.savefig(f"{run_dir}/rl_vs_xyz_delta_fidelity_N{headline_budget}.png", dpi=200)
plt.close()

with open(f"{run_dir}/notes.txt", "w", encoding="utf-8") as f:
    f.write("Experiment 7: basis-dependent readout-noise robustness\n")
    f.write("One Pauli basis is randomly degraded per episode.\n")
    f.write("All methods experience the same true noise profile.\n")
    f.write("All methods use the same calibration-derived readout correction.\n")
    f.write("RL also uses calibrated noise estimates in its policy state.\n")
    f.write(f"Calibration shots per basis: {CALIBRATION_SHOTS_PER_BASIS}\n")
    f.write(f"Training episodes per budget: {NUM_TRAIN_EPISODES}\n")
    f.write(f"Shot budgets: {SHOT_BUDGETS}\n")
    f.write(f"Degraded noise levels: {DEGRADED_NOISE_LEVELS}\n")

print("Experiment 7 completed. Results saved to:", run_dir)
print("Key files:")
print("- fidelity_vs_noise_N50.png")
print("- rl_vs_xyz_delta_fidelity_N50.png")
print("- rl_vs_xyz_paired_summary.csv")