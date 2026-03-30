import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp08_2qubit_drift"
os.makedirs(run_dir, exist_ok=True)

# ========================
# Quantum utilities
# ========================
def state_from_angles(theta, phi):
    return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)

def fidelity(psi, phi):
    return float(np.abs(np.vdot(psi, phi))**2)

def hadamard():
    return (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)

def s_dagger():
    return np.array([[1,0],[0,-1j]], dtype=complex)

# ========================
# Measurement probabilities
# ========================
def measure_probs_z(psi):
    p = np.abs(psi)**2
    return p/np.sum(p)

def measure_probs_x(psi):
    psi_x = hadamard() @ psi
    p = np.abs(psi_x)**2
    return p/np.sum(p)

def measure_probs_y(psi):
    psi_y = hadamard() @ (s_dagger() @ psi)
    p = np.abs(psi_y)**2
    return p/np.sum(p)

def sample_one_shot(probs, rng):
    return int(rng.choice([0,1], p=probs))

# ========================
# Bloch estimation
# ========================
def estimate_bloch_from_counts(z_counts, x_counts, y_counts):
    z_hat = x_hat = y_hat = 0.0
    if np.sum(z_counts) > 0:
        z_hat = float((z_counts[0]-z_counts[1])/np.sum(z_counts))
    if np.sum(x_counts) > 0:
        x_hat = float((x_counts[0]-x_counts[1])/np.sum(x_counts))
    if np.sum(y_counts) > 0:
        y_hat = float((y_counts[0]-y_counts[1])/np.sum(y_counts))
    return x_hat, y_hat, z_hat

def bloch_to_state(x, y, z):
    r = np.array([x,y,z], dtype=float)
    norm = np.linalg.norm(r)
    if norm < 1e-12:
        return np.array([1.0,0.0], dtype=complex)
    x_n, y_n, z_n = r/norm
    theta_hat = float(np.arccos(np.clip(z_n,-1.0,1.0)))
    phi_hat = float(np.mod(np.arctan2(y_n, x_n), 2*np.pi))
    return state_from_angles(theta_hat, phi_hat)

def estimate_state_from_counts(z_counts, x_counts, y_counts):
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)
    return bloch_to_state(x_hat, y_hat, z_hat)

# ========================
# RL actions
# ========================
ACTIONS = ["Z","X","Y"]

# ========================
# Build RL state
# ========================
def build_state(z1, x1, y1, z2, x2, y2, shots_used, total_shots):
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
    progress = shots_used/total_shots if total_shots>0 else 0.0
    return np.array([x1_hat, y1_hat, z1_hat, x2_hat, y2_hat, z2_hat,
                     frac_Z1, frac_X1, frac_Y1, frac_Z2, frac_X2, frac_Y2,
                     progress,1.0])

def epsilon_greedy(state, weights, epsilon, rng):
    if rng.random() < epsilon:
        return ACTIONS[rng.integers(len(ACTIONS))]
    q_vals = [np.dot(weights[a], state) for a in ACTIONS]
    return ACTIONS[int(np.argmax(q_vals))]

# ========================
# Drift RL episode
# ========================
def run_rl_drift_episode(psi1, psi2, total_shots, weights, epsilon, rng, update_weights, lr, drift_strength=0.005):
    z1=x1=y1=z2=x2=y2=np.array([0,0], dtype=int)
    shots_used = 0
    for basis in ["Z","X","Y"]:
        if shots_used >= total_shots: break
        for i, psi in enumerate([psi1, psi2]):
            p = measure_probs_z(psi) if basis=="Z" else measure_probs_x(psi) if basis=="X" else measure_probs_y(psi)
            outcome = sample_one_shot(p, rng)
            if i==0:
                if basis=="Z": z1[outcome]+=1
                elif basis=="X": x1[outcome]+=1
                else: y1[outcome]+=1
            else:
                if basis=="Z": z2[outcome]+=1
                elif basis=="X": x2[outcome]+=1
                else: y2[outcome]+=1
        shots_used += 1

    psi1_hat = estimate_state_from_counts(z1,x1,y1)
    psi2_hat = estimate_state_from_counts(z2,x2,y2)
    F_prev = fidelity(psi1_hat, psi1)*fidelity(psi2_hat, psi2)

    for t in range(shots_used, total_shots):
        # small drift
        for psi_arr in [psi1, psi2]:
            theta, phi = np.arccos(np.clip(np.abs(psi_arr[0]),0,1))*2, np.angle(psi_arr[1]/psi_arr[0])
            theta += rng.uniform(-drift_strength, drift_strength)
            phi += rng.uniform(-drift_strength, drift_strength)
            psi_new = state_from_angles(theta, phi)
            if psi_arr is psi1: psi1 = psi_new
            else: psi2 = psi_new

        state = build_state(z1,x1,y1,z2,x2,y2,t,total_shots)
        action = epsilon_greedy(state, weights, epsilon, rng)

        for i, psi in enumerate([psi1, psi2]):
            b = action
            p = measure_probs_z(psi) if b=="Z" else measure_probs_x(psi) if b=="X" else measure_probs_y(psi)
            outcome = sample_one_shot(p, rng)
            if i==0:
                if b=="Z": z1[outcome]+=1
                elif b=="X": x1[outcome]+=1
                else: y1[outcome]+=1
            else:
                if b=="Z": z2[outcome]+=1
                elif b=="X": x2[outcome]+=1
                else: y2[outcome]+=1

        psi1_hat = estimate_state_from_counts(z1,x1,y1)
        psi2_hat = estimate_state_from_counts(z2,x2,y2)
        F_new = fidelity(psi1_hat, psi1)*fidelity(psi2_hat, psi2)
        reward = F_new - F_prev
        if update_weights:
            weights[action] += lr*reward*state
        F_prev = F_new

    return {"fidelity": F_prev, "infidelity": 1-F_prev}, weights

# ========================
# Experiment parameters
# ========================
shot_budgets = [10,25,50,100,250,500]
num_test_targets = 50
num_test_seeds = 5
epsilon_test = 0.0
learning_rate = 0.02
master_rng = np.random.default_rng(123)

# ========================
# Run Drift Test standalone
# ========================
rows = []

for target_id in range(num_test_targets):
    theta1, phi1 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
    theta2, phi2 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
    psi1 = state_from_angles(theta1,phi1)
    psi2 = state_from_angles(theta2,phi2)

    for seed in range(num_test_seeds):
        for N in shot_budgets:
            rng = np.random.default_rng(target_id*1000+seed*100+N)
            # initialize weights from scratch
            weights = {a: np.zeros(14) for a in ACTIONS}
            out_rl, _ = run_rl_drift_episode(psi1, psi2, N, weights, epsilon_test, rng, False, learning_rate)
            rows.append({"target_id":target_id,"seed":seed,"N":N,"method":"RL_adaptive_drift", **out_rl})

# ========================
# Save + plot
# ========================
df = pd.DataFrame(rows)
df.to_csv(f"{run_dir}/metrics.csv", index=False)

agg = df.groupby(["method","N"]).mean().reset_index()

plt.figure()
plt.plot(agg["N"], agg["infidelity"], marker="o", label="RL Drift")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Shots")
plt.ylabel("Infidelity")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_dir}/infidelity.png", dpi=200)
plt.close()

print("Saved drift results to:", run_dir)