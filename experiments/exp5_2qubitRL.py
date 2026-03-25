import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp05_2qubitRL_Y"
os.makedirs(run_dir, exist_ok=True)

# ========================
# Quantum utilities
# ========================
def state_from_angles(theta: float, phi: float) -> np.ndarray:
    """Single-qubit pure state: |psi(theta, phi)> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>"""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)

def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """State fidelity |<psi|phi>|^2."""
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

def sample_one_shot(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice([0, 1], p=probs))

# ========================
# Bloch estimation
# ========================
def estimate_bloch_from_counts(z_counts, x_counts, y_counts):
    z_hat = x_hat = y_hat = 0.0
    if np.sum(z_counts) > 0:
        z_probs = z_counts / np.sum(z_counts)
        z_hat = float(z_probs[0] - z_probs[1])
    if np.sum(x_counts) > 0:
        x_probs = x_counts / np.sum(x_counts)
        x_hat = float(x_probs[0] - x_probs[1])
    if np.sum(y_counts) > 0:
        y_probs = y_counts / np.sum(y_counts)
        y_hat = float(y_probs[0] - y_probs[1])
    return x_hat, y_hat, z_hat

def bloch_to_state(x, y, z):
    r = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(r)
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=complex)
    x_n, y_n, z_n = r / norm
    z_n = float(np.clip(z_n, -1.0, 1.0))
    theta_hat = float(np.arccos(z_n))
    phi_hat = float(np.mod(np.arctan2(y_n, x_n), 2 * np.pi))
    return state_from_angles(theta_hat, phi_hat)

def estimate_state_from_counts(z_counts, x_counts, y_counts):
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)
    return bloch_to_state(x_hat, y_hat, z_hat)

# ========================
# RL setup
# ========================
ACTIONS = [
    ("Z","Z"), ("Z","X"), ("Z","Y"),
    ("X","Z"), ("X","X"), ("X","Y"),
    ("Y","Z"), ("Y","X"), ("Y","Y")
]

def build_state(z1, x1, y1, z2, x2, y2, shots_used, total_shots):
    """Build RL state vector including Bloch estimates, measurement fractions, and progress"""
    def estimate(z, x, y):
        z_hat = (z[0]-z[1])/np.sum(z) if np.sum(z)>0 else 0.0
        x_hat = (x[0]-x[1])/np.sum(x) if np.sum(x)>0 else 0.0
        y_hat = (y[0]-y[1])/np.sum(y) if np.sum(y)>0 else 0.0
        return x_hat, y_hat, z_hat

    x1_hat, y1_hat, z1_hat = estimate(z1, x1, y1)
    x2_hat, y2_hat, z2_hat = estimate(z2, x2, y2)

    total1 = np.sum(z1)+np.sum(x1)+np.sum(y1)
    total2 = np.sum(z2)+np.sum(x2)+np.sum(y2)

    frac_Z1 = np.sum(z1)/total1 if total1>0 else 0.0
    frac_X1 = np.sum(x1)/total1 if total1>0 else 0.0
    frac_Y1 = np.sum(y1)/total1 if total1>0 else 0.0

    frac_Z2 = np.sum(z2)/total2 if total2>0 else 0.0
    frac_X2 = np.sum(x2)/total2 if total2>0 else 0.0
    frac_Y2 = np.sum(y2)/total2 if total2>0 else 0.0

    progress = shots_used / total_shots if total_shots>0 else 0.0

    return np.array([
        x1_hat, y1_hat, z1_hat,
        x2_hat, y2_hat, z2_hat,
        frac_Z1, frac_X1, frac_Y1,
        frac_Z2, frac_X2, frac_Y2,
        progress,
        1.0  # bias term
    ])

def epsilon_greedy(state, weights, epsilon, rng):
    if rng.random() < epsilon:
        return ACTIONS[rng.integers(len(ACTIONS))]
    q_vals = [np.dot(weights[a], state) for a in ACTIONS]
    return ACTIONS[int(np.argmax(q_vals))]

# ========================
# RL episode
# ========================
def run_rl_episode(psi1, psi2, total_shots, weights, epsilon, rng, update_weights, lr):
    z1 = x1 = y1 = np.array([0,0], dtype=int)
    z2 = x2 = y2 = np.array([0,0], dtype=int)
    shots_used = 0

    # Warm start: measure 1 shot in Z, X, Y each
    for basis in ["Z","X","Y"]:
        if shots_used >= total_shots:
            break
        p1 = measure_probs_z(psi1) if basis=="Z" else measure_probs_x(psi1) if basis=="X" else measure_probs_y(psi1)
        p2 = measure_probs_z(psi2) if basis=="Z" else measure_probs_x(psi2) if basis=="X" else measure_probs_y(psi2)
        o1 = sample_one_shot(p1, rng)
        o2 = sample_one_shot(p2, rng)
        if basis=="Z":
            z1[o1] +=1; z2[o2] +=1
        elif basis=="X":
            x1[o1] +=1; x2[o2] +=1
        else:
            y1[o1] +=1; y2[o2] +=1
        shots_used +=1

    psi1_hat = estimate_state_from_counts(z1, x1, y1)
    psi2_hat = estimate_state_from_counts(z2, x2, y2)
    F_prev = fidelity(psi1_hat, psi1) * fidelity(psi2_hat, psi2)

    for t in range(shots_used, total_shots):
        state = build_state(z1, x1, y1, z2, x2, y2, t, total_shots)
        a = epsilon_greedy(state, weights, epsilon, rng)
        b1, b2 = a

        p1 = measure_probs_z(psi1) if b1=="Z" else measure_probs_x(psi1) if b1=="X" else measure_probs_y(psi1)
        p2 = measure_probs_z(psi2) if b2=="Z" else measure_probs_x(psi2) if b2=="X" else measure_probs_y(psi2)

        o1 = sample_one_shot(p1, rng)
        o2 = sample_one_shot(p2, rng)

        if b1=="Z": z1[o1]+=1
        elif b1=="X": x1[o1]+=1
        else: y1[o1]+=1

        if b2=="Z": z2[o2]+=1
        elif b2=="X": x2[o2]+=1
        else: y2[o2]+=1

        psi1_hat = estimate_state_from_counts(z1, x1, y1)
        psi2_hat = estimate_state_from_counts(z2, x2, y2)
        F_new = fidelity(psi1_hat, psi1) * fidelity(psi2_hat, psi2)
        reward = F_new - F_prev

        if update_weights:
            weights[a] += lr * reward * state

        F_prev = F_new

    return {"fidelity": F_prev, "infidelity": 1-F_prev}, weights

# ========================
# Experiment configuration
# ========================
shot_budgets = [10, 25, 50, 100, 250, 500]
num_train_episodes = 800
num_test_targets = 50
num_test_seeds = 5

epsilon_train = 0.15
epsilon_test = 0.0
learning_rate = 0.02

master_rng = np.random.default_rng(123)

# ========================
# Train RL
# ========================
trained_weights = {}
for N in shot_budgets:
    weights = {a: np.zeros(14) for a in ACTIONS}  # 14-dim state
    for ep in range(num_train_episodes):
        theta1, phi1 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
        theta2, phi2 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
        psi1 = state_from_angles(theta1, phi1)
        psi2 = state_from_angles(theta2, phi2)
        rng = np.random.default_rng(10_000*N + ep)
        _, weights = run_rl_episode(psi1, psi2, N, weights, epsilon_train, rng, True, learning_rate)
    trained_weights[N] = weights
    print("Trained for N =", N)

# ========================
# Evaluate RL vs Fixed
# ========================
rows = []
for target_id in range(num_test_targets):
    theta1, phi1 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
    theta2, phi2 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
    psi1 = state_from_angles(theta1, phi1)
    psi2 = state_from_angles(theta2, phi2)
    for seed in range(num_test_seeds):
        for N in shot_budgets:
            rng = np.random.default_rng(target_id*1000 + seed*100 + N)
            # Fixed strategies
            for strategy in ["Z_only","ZX_split","XYZ_split"]:
                # Use previous fixed baseline code for counts
                z1=z2=x1=x2=y1=y2=np.array([0,0],dtype=int)
                if strategy=="Z_only":
                    for _ in range(N):
                        z1[sample_one_shot(measure_probs_z(psi1), rng)]+=1
                        z2[sample_one_shot(measure_probs_z(psi2), rng)]+=1
                elif strategy=="ZX_split":
                    for _ in range(N//2):
                        z1[sample_one_shot(measure_probs_z(psi1), rng)]+=1
                        z2[sample_one_shot(measure_probs_z(psi2), rng)]+=1
                    for _ in range(N-N//2):
                        x1[sample_one_shot(measure_probs_x(psi1), rng)]+=1
                        x2[sample_one_shot(measure_probs_x(psi2), rng)]+=1
                else: # XYZ_split
                    n_z = N//3; n_x = N//3; n_y = N - n_z - n_x
                    for _ in range(n_z):
                        z1[sample_one_shot(measure_probs_z(psi1), rng)]+=1
                        z2[sample_one_shot(measure_probs_z(psi2), rng)]+=1
                    for _ in range(n_x):
                        x1[sample_one_shot(measure_probs_x(psi1), rng)]+=1
                        x2[sample_one_shot(measure_probs_x(psi2), rng)]+=1
                    for _ in range(n_y):
                        y1[sample_one_shot(measure_probs_y(psi1), rng)]+=1
                        y2[sample_one_shot(measure_probs_y(psi2), rng)]+=1
                psi1_hat = estimate_state_from_counts(z1,x1,y1)
                psi2_hat = estimate_state_from_counts(z2,x2,y2)
                F_total = fidelity(psi1_hat, psi1)*fidelity(psi2_hat, psi2)
                rows.append({"target_id":target_id,"seed":seed,"N":N,"method":strategy,"fidelity":F_total,"infidelity":1-F_total})
            # RL
            out_rl,_ = run_rl_episode(psi1,psi2,N,trained_weights[N],epsilon_test,rng,False,learning_rate)
            rows.append({"target_id":target_id,"seed":seed,"N":N,"method":"RL_adaptive","fidelity":out_rl["fidelity"],"infidelity":out_rl["infidelity"]})

# ========================
# Save + plot
# ========================
df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/metrics.csv", index=False)

agg = df.groupby(["method","N"]).mean().reset_index()

plt.figure()
for m in agg["method"].unique():
    sub = agg[agg["method"]==m]
    plt.plot(sub["N"],sub["infidelity"],marker="o",label=m)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Shots")
plt.ylabel("Infidelity")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_dir}/infidelity.png",dpi=200)
plt.close()

print("Saved results to:", run_dir)