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

# Keep the RL training setup unchanged.
num_train_episodes = 3000

# Publication-quality evaluation.
num_eval_trials = 2000
num_bootstrap = 10000

epsilon_start = 0.30
epsilon_end = 0.05
epsilon_test = 0.0
learning_rate = 0.01
gamma = 0.95
imbalance_weight = 0.01

training_master_seed = 123
evaluation_master_seed = 456789
bootstrap_seed = 20260317

master_rng = np.random.default_rng(training_master_seed)


# ========================
# Train a separate RL policy for each shot budget
# ========================
trained_weights: dict[int, dict[str, np.ndarray]] = {}

for N in shot_budgets:
    weights = initialize_weights()

    for ep in range(num_train_episodes):
        frac = ep / max(1, num_train_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        _, _, psi_true = sample_uniform_pure_state(master_rng)
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
    print(f"Finished training RL policy for N={N}")


# ========================
# Evaluation
# ========================
# Each trial is one independently sampled isotropic pure state.
# All methods see the same target state within a trial.
# Each method gets an independent, reproducible measurement RNG stream.
rows = []
methods = ("Z_only", "ZX_split", "XYZ_split", "RL_adaptive")
method_seed_code = {
    "Z_only": 11,
    "ZX_split": 22,
    "XYZ_split": 33,
    "RL_adaptive": 44,
}

eval_rng = np.random.default_rng(evaluation_master_seed)
test_states = [
    sample_uniform_pure_state(eval_rng)
    for _ in range(num_eval_trials)
]

for trial_id, (true_theta, true_phi, psi_true) in enumerate(test_states):
    for N in shot_budgets:
        seed_base = evaluation_master_seed + trial_id * 100_000 + N * 100

        for method in methods:
            rng = np.random.default_rng(seed_base + method_seed_code[method])

            if method == "RL_adaptive":
                out, _ = run_rl_episode(
                    psi_true=psi_true,
                    total_shots=N,
                    weights=trained_weights[N],
                    epsilon=epsilon_test,
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

            rows.append({
                "trial_id": trial_id,
                "N": N,
                "method": method,
                "true_theta": true_theta,
                "true_phi": true_phi,
                **out,
            })

    if (trial_id + 1) % 250 == 0:
        print(f"Evaluation: {trial_id + 1}/{num_eval_trials} targets complete")


# ========================
# Save raw trial-level metrics
# ========================
df = pd.DataFrame(rows)
df["x_fraction"] = df["x_shots"] / df["N"]
df["y_fraction"] = df["y_shots"] / df["N"]
df["z_fraction"] = df["z_shots"] / df["N"]
df.to_csv(f"{run_dir}/metrics.csv", index=False)


# ========================
# Bootstrap utilities
# ========================
def bootstrap_mean_ci(values, rng, n_boot=10000, confidence=0.95):
    """Percentile bootstrap CI for a sample mean."""
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
    low, high = np.quantile(
        boot_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return point, float(low), float(high)


# ========================
# Method-level summary with 95% bootstrap CIs
# ========================
summary_rows = []
method_code = {m: i + 1 for i, m in enumerate(methods)}

for method in methods:
    for N in shot_budgets:
        sub = df[(df["method"] == method) & (df["N"] == N)]
        code = method_code[method]

        f_mean, f_lo, f_hi = bootstrap_mean_ci(
            sub["fidelity"],
            np.random.default_rng(bootstrap_seed + N * 1000 + code * 10 + 1),
            num_bootstrap,
        )
        i_mean, i_lo, i_hi = bootstrap_mean_ci(
            sub["infidelity"],
            np.random.default_rng(bootstrap_seed + N * 1000 + code * 10 + 2),
            num_bootstrap,
        )
        xf_mean, xf_lo, xf_hi = bootstrap_mean_ci(
            sub["x_fraction"],
            np.random.default_rng(bootstrap_seed + N * 1000 + code * 10 + 3),
            num_bootstrap,
        )
        yf_mean, yf_lo, yf_hi = bootstrap_mean_ci(
            sub["y_fraction"],
            np.random.default_rng(bootstrap_seed + N * 1000 + code * 10 + 4),
            num_bootstrap,
        )
        zf_mean, zf_lo, zf_hi = bootstrap_mean_ci(
            sub["z_fraction"],
            np.random.default_rng(bootstrap_seed + N * 1000 + code * 10 + 5),
            num_bootstrap,
        )

        summary_rows.append({
            "method": method,
            "N": N,
            "n_trials": len(sub),
            "fidelity_mean": f_mean,
            "fidelity_ci95_low": f_lo,
            "fidelity_ci95_high": f_hi,
            "infidelity_mean": i_mean,
            "infidelity_ci95_low": i_lo,
            "infidelity_ci95_high": i_hi,
            "x_fraction_mean": xf_mean,
            "x_fraction_ci95_low": xf_lo,
            "x_fraction_ci95_high": xf_hi,
            "y_fraction_mean": yf_mean,
            "y_fraction_ci95_low": yf_lo,
            "y_fraction_ci95_high": yf_hi,
            "z_fraction_mean": zf_mean,
            "z_fraction_ci95_low": zf_lo,
            "z_fraction_ci95_high": zf_hi,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{run_dir}/summary_bootstrap.csv", index=False)


# ========================
# Paired RL - XYZ analysis
# ========================
paired_rows = []

for N in shot_budgets:
    sub = df[df["N"] == N]
    wide = sub.pivot(index="trial_id", columns="method", values="fidelity")

    differences = (
        wide["RL_adaptive"] - wide["XYZ_split"]
    ).dropna().to_numpy()

    d_mean, d_lo, d_hi = bootstrap_mean_ci(
        differences,
        np.random.default_rng(bootstrap_seed + 50_000 + N),
        num_bootstrap,
    )

    paired_rows.append({
        "N": N,
        "n_pairs": len(differences),
        "mean_fidelity_difference_RL_minus_XYZ": d_mean,
        "ci95_low": d_lo,
        "ci95_high": d_hi,
        "rl_win_rate": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(np.isclose(differences, 0.0, atol=1e-12))),
        "rl_loss_rate": float(np.mean(differences < 0)),
    })

paired = pd.DataFrame(paired_rows)
paired.to_csv(f"{run_dir}/paired_RL_minus_XYZ_bootstrap.csv", index=False)


# ========================
# Plot helper
# ========================
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

    for method in methods:
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
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


plot_metric_ci(
    "fidelity_mean",
    "fidelity_ci95_low",
    "fidelity_ci95_high",
    "Mean fidelity",
    "Exp. 3: General pure single-qubit tomography",
    "fidelity_vs_shots_bootstrap95.png",
)

plot_metric_ci(
    "infidelity_mean",
    "infidelity_ci95_low",
    "infidelity_ci95_high",
    "Mean infidelity",
    "Exp. 3: General pure single-qubit tomography",
    "infidelity_vs_shots_bootstrap95.png",
    logy=True,
)


# ========================
# RL allocation plot
# ========================
rl_alloc = summary[summary["method"] == "RL_adaptive"].sort_values("N")

plt.figure(figsize=(6.4, 4.4))
x = rl_alloc["N"].to_numpy()

for axis_name in ("x", "y", "z"):
    mean = rl_alloc[f"{axis_name}_fraction_mean"].to_numpy()
    low = rl_alloc[f"{axis_name}_fraction_ci95_low"].to_numpy()
    high = rl_alloc[f"{axis_name}_fraction_ci95_high"].to_numpy()

    plt.plot(x, mean, marker="o", label=f"{axis_name.upper()} fraction")
    plt.fill_between(x, low, high, alpha=0.15)

plt.axhline(1.0 / 3.0, linestyle="--", linewidth=1.0, label="Balanced XYZ")
plt.xscale("log")
plt.ylim(0.0, 1.0)
plt.xlabel("Measurement budget (N)")
plt.ylabel("Fraction of measurements")
plt.title("Exp. 3: Learned RL measurement allocation")
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{run_dir}/rl_allocation_bootstrap95.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ========================
# Paired RL - XYZ difference plot
# ========================
plt.figure(figsize=(6.4, 4.4))
x = paired["N"].to_numpy()
y = paired["mean_fidelity_difference_RL_minus_XYZ"].to_numpy()
lo = paired["ci95_low"].to_numpy()
hi = paired["ci95_high"].to_numpy()
yerr = np.vstack([y - lo, hi - y])

plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
plt.axhline(0.0, linestyle="--", linewidth=1.0)
plt.xscale("log")
plt.xlabel("Measurement budget (N)")
plt.ylabel("Mean fidelity difference (RL - XYZ)")
plt.title("Exp. 3: Paired RL vs. XYZ comparison")
plt.tight_layout()
plt.savefig(
    f"{run_dir}/paired_RL_minus_XYZ_bootstrap95.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# ========================
# Save trained weights and notes
# ========================
for N, weights in trained_weights.items():
    for action in ACTIONS:
        np.save(f"{run_dir}/weights_{action}_N{N}.npy", weights[action])

with open(f"{run_dir}/notes.txt", "w") as f:
    f.write("Experiment: Exp03 - generalized one-qubit adaptive measurement\\n")
    f.write("State family: pure qubit states sampled isotropically on the Bloch sphere\\n")
    f.write("Sampling: z = cos(theta) uniform on [-1,1], phi uniform on [0,2pi)\\n")
    f.write("Methods: Z_only, ZX_split, XYZ_split, RL_adaptive\\n")
    f.write("RL actions: Z, X, Y\\n")
    f.write("RL update: one-step TD/Q-learning with linear function approximation\\n")
    f.write("RL reward: fidelity improvement minus allocation imbalance penalty\\n")
    f.write(f"Shot budgets: {shot_budgets}\\n")
    f.write(f"Training episodes per budget: {num_train_episodes}\\n")
    f.write(f"Independent evaluation targets per budget: {num_eval_trials}\\n")
    f.write(f"Bootstrap resamples: {num_bootstrap}\\n")
    f.write("CI: 95% percentile bootstrap confidence interval of the mean\\n")
    f.write(
        "Paired analysis: RL minus XYZ on matched target-state trials, "
        "with a 95% percentile bootstrap CI of the mean paired difference.\\n"
    )
    f.write(
        "Bootstrap uncertainty is conditional on the single trained policy "
        "for each budget and does not include between-training-run variability.\\n"
    )
    f.write(f"Training master seed: {training_master_seed}\\n")
    f.write(f"Evaluation master seed: {evaluation_master_seed}\\n")
    f.write(f"Bootstrap seed: {bootstrap_seed}\\n")
    f.write(f"Epsilon schedule: {epsilon_start} -> {epsilon_end}\\n")
    f.write(f"Evaluation epsilon: {epsilon_test}\\n")
    f.write(f"Learning rate: {learning_rate}\\n")
    f.write(f"Gamma: {gamma}\\n")
    f.write(f"Imbalance penalty weight: {imbalance_weight}\\n")

print("\\nSaved results to:", run_dir)
print("Key files:")
print("- metrics.csv")
print("- summary_bootstrap.csv")
print("- paired_RL_minus_XYZ_bootstrap.csv")
print("- fidelity_vs_shots_bootstrap95.png")
print("- infidelity_vs_shots_bootstrap95.png")
print("- rl_allocation_bootstrap95.png")
print("- paired_RL_minus_XYZ_bootstrap95.png")
print("- notes.txt")
