import os
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Experiment 4: fixed baselines on hidden-plane product states
# ============================================================
# Each two-qubit product state shares one latent Bloch plane:
# XY, XZ, or YZ. The plane changes between targets and is not
# available to the ordinary fixed baselines. ORACLE_plane is an
# upper benchmark that is explicitly given the true plane.

SEED = 123
SHOT_BUDGETS = [10, 25, 50, 100, 250, 500]
NUM_TEST_TARGETS = int(os.getenv("EXP04_TEST_TARGETS", "50"))
NUM_TEST_SEEDS = int(os.getenv("EXP04_TEST_SEEDS", "5"))
MIN_ACTIVE_COMPONENT = float(os.getenv("EXP04_MIN_ACTIVE_COMPONENT", "0.35"))

PLANE_BASES: Dict[str, Tuple[str, str]] = {
    "XY": ("X", "Y"),
    "XZ": ("X", "Z"),
    "YZ": ("Y", "Z"),
}


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
) -> Tuple[np.ndarray, float]:
    """Sample away from coordinate axes so the hidden plane is identifiable."""
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


def estimate_state(counts: Dict[str, np.ndarray]) -> np.ndarray:
    x = estimate_coord(counts["X"])
    y = estimate_coord(counts["Y"])
    z = estimate_coord(counts["Z"])
    r = np.array([x, y, z], dtype=float)
    norm = float(np.linalg.norm(r))
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=complex)
    x, y, z = r / norm
    return state_from_bloch(float(x), float(y), float(z))


def balanced_schedule(bases: Tuple[str, ...], total_shots: int):
    schedule = []
    for i in range(total_shots):
        schedule.append(bases[i % len(bases)])
    return schedule


def run_fixed_episode(
    psi1: np.ndarray,
    psi2: np.ndarray,
    plane: str,
    total_shots: int,
    strategy: str,
    rng: np.random.Generator,
):
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
        raise ValueError(f"Unknown strategy: {strategy}")

    for basis in schedule:
        counts1[basis][sample_one_shot(probs_for_basis(psi1, basis), rng)] += 1
        counts2[basis][sample_one_shot(probs_for_basis(psi2, basis), rng)] += 1

    psi1_hat, psi2_hat = estimate_state(counts1), estimate_state(counts2)
    f_total = fidelity(psi1_hat, psi1) * fidelity(psi2_hat, psi2)
    return {
        "fidelity": f_total,
        "infidelity": 1.0 - f_total,
        **{f"shots_{b}": int(np.sum(counts1[b])) for b in ("X", "Y", "Z")},
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"results/{timestamp}_exp04_hidden_plane_fixed"
    os.makedirs(run_dir, exist_ok=True)

    master_rng = np.random.default_rng(SEED)
    rows = []
    strategies = ["Z_only", "ZX_split", "XYZ_split", "ORACLE_plane"]

    for target_id in range(NUM_TEST_TARGETS):
        psi1, psi2, plane, alpha1, alpha2 = sample_hidden_plane_product_state(master_rng)

        for seed in range(NUM_TEST_SEEDS):
            for n_shots in SHOT_BUDGETS:
                base_seed = target_id * 1_000_000 + seed * 10_000 + n_shots * 10
                for method_index, strategy in enumerate(strategies):
                    rng = np.random.default_rng(base_seed + method_index + 1)
                    out = run_fixed_episode(
                        psi1, psi2, plane, n_shots, strategy, rng
                    )
                    rows.append(
                        {
                            "target_id": target_id,
                            "seed": seed,
                            "N": n_shots,
                            "method": strategy,
                            "plane": plane,
                            "alpha1": alpha1,
                            "alpha2": alpha2,
                            **out,
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

    def plot_metric(mean_col, std_col, ylabel, title, filename, logy=False):
        plt.figure()
        for method in strategies:
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
        "Exp04: Hidden-Plane 2-Qubit Fixed Baselines",
        "fidelity_vs_shots.png",
    )
    plot_metric(
        "infidelity_mean",
        "infidelity_std",
        "Mean Infidelity",
        "Exp04: Hidden-Plane 2-Qubit Fixed Baselines",
        "infidelity_vs_shots.png",
        logy=True,
    )

    with open(f"{run_dir}/notes.txt", "w", encoding="utf-8") as f:
        f.write("Experiment 4: fixed baselines on hidden-plane product states\n")
        f.write("Each target shares one latent plane: XY, XZ, or YZ.\n")
        f.write("The ordinary baselines are not told the plane.\n")
        f.write("ORACLE_plane is told the plane and is an upper benchmark.\n")
        f.write(f"Minimum active Bloch-component magnitude: {MIN_ACTIVE_COMPONENT}\n")
        f.write(f"Shot budgets: {SHOT_BUDGETS}\n")
        f.write(f"Test targets: {NUM_TEST_TARGETS}\n")
        f.write(f"Test seeds: {NUM_TEST_SEEDS}\n")

    print("Saved results to:", run_dir)


if __name__ == "__main__":
    main()