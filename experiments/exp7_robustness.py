import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp07_2qubit_robustness"
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

def noisy_sample(probs, rng, flip_prob=0.05):
    outcome = sample_one_shot(probs,rng)
    if rng.random() < flip_prob:
        return 1 - outcome
    return outcome

# ========================
# Bloch estimation
# ========================
def estimate_bloch_from_counts(z_counts, x_counts, y_counts):
    z_hat = x_hat = y_hat = 0.0
    if np.sum(z_counts)>0: z_hat = float((z_counts[0]-z_counts[1])/np.sum(z_counts))
    if np.sum(x_counts)>0: x_hat = float((x_counts[0]-x_counts[1])/np.sum(x_counts))
    if np.sum(y_counts)>0: y_hat = float((y_counts[0]-y_counts[1])/np.sum(y_counts))
    return x_hat, y_hat, z_hat

def bloch_to_state(x, y, z):
    r = np.array([x,y,z])
    norm = np.linalg.norm(r)
    if norm<1e-12: return np.array([1.0,0.0], dtype=complex)
    x_n, y_n, z_n = r/norm
    theta_hat = np.arccos(np.clip(z_n,-1.0,1.0))
    phi_hat = np.mod(np.arctan2(y_n,x_n), 2*np.pi)
    return state_from_angles(theta_hat, phi_hat)

def estimate_state_from_counts(z_counts, x_counts, y_counts):
    x_hat, y_hat, z_hat = estimate_bloch_from_counts(z_counts, x_counts, y_counts)
    return bloch_to_state(x_hat, y_hat, z_hat)

# ========================
# RL setup
# ========================
ACTIONS = [("Z","Z"),("Z","X"),("Z","Y"),
           ("X","Z"),("X","X"),("X","Y"),
           ("Y","Z"),("Y","X"),("Y","Y")]

def build_state(z1,x1,y1,z2,x2,y2,shots_used,total_shots):
    def est(z,x,y):
        return ((z[0]-z[1])/np.sum(z) if np.sum(z)>0 else 0.0,
                (x[0]-x[1])/np.sum(x) if np.sum(x)>0 else 0.0,
                (y[0]-y[1])/np.sum(y) if np.sum(y)>0 else 0.0)
    x1_hat,y1_hat,z1_hat = est(z1,x1,y1)
    x2_hat,y2_hat,z2_hat = est(z2,x2,y2)
    total1 = np.sum(z1)+np.sum(x1)+np.sum(y1)
    total2 = np.sum(z2)+np.sum(x2)+np.sum(y2)
    frac_Z1 = np.sum(z1)/total1 if total1>0 else 0.0
    frac_X1 = np.sum(x1)/total1 if total1>0 else 0.0
    frac_Y1 = np.sum(y1)/total1 if total1>0 else 0.0
    frac_Z2 = np.sum(z2)/total2 if total2>0 else 0.0
    frac_X2 = np.sum(x2)/total2 if total2>0 else 0.0
    frac_Y2 = np.sum(y2)/total2 if total2>0 else 0.0
    progress = shots_used/total_shots if total_shots>0 else 0.0
    return np.array([x1_hat,y1_hat,z1_hat,x2_hat,y2_hat,z2_hat,
                     frac_Z1,frac_X1,frac_Y1,frac_Z2,frac_X2,frac_Y2,
                     progress,1.0])

def epsilon_greedy(state, weights, epsilon, rng):
    if rng.random()<epsilon: return ACTIONS[rng.integers(len(ACTIONS))]
    q_vals = [np.dot(weights[a],state) for a in ACTIONS]
    return ACTIONS[int(np.argmax(q_vals))]

# ========================
# RL episode
# ========================
def run_rl_episode(psi1, psi2, total_shots, weights, epsilon, rng, update_weights, lr, noise=0.0):
    z1=x1=y1=np.array([0,0],dtype=int)
    z2=x2=y2=np.array([0,0],dtype=int)
    shots_used=0
    # warm start
    for basis in ["Z","X","Y"]:
        if shots_used>=total_shots: break
        p1 = measure_probs_z(psi1) if basis=="Z" else measure_probs_x(psi1) if basis=="X" else measure_probs_y(psi1)
        p2 = measure_probs_z(psi2) if basis=="Z" else measure_probs_x(psi2) if basis=="X" else measure_probs_y(psi2)
        o1 = noisy_sample(p1,rng,flip_prob=noise)
        o2 = noisy_sample(p2,rng,flip_prob=noise)
        if basis=="Z": z1[o1]+=1; z2[o2]+=1
        elif basis=="X": x1[o1]+=1; x2[o2]+=1
        else: y1[o1]+=1; y2[o2]+=1
        shots_used+=1
    psi1_hat = estimate_state_from_counts(z1,x1,y1)
    psi2_hat = estimate_state_from_counts(z2,x2,y2)
    F_prev = fidelity(psi1_hat,psi1)*fidelity(psi2_hat,psi2)

    for t in range(shots_used,total_shots):
        state = build_state(z1,x1,y1,z2,x2,y2,t,total_shots)
        a = epsilon_greedy(state,weights,epsilon,rng)
        b1,b2 = a
        p1 = measure_probs_z(psi1) if b1=="Z" else measure_probs_x(psi1) if b1=="X" else measure_probs_y(psi1)
        p2 = measure_probs_z(psi2) if b2=="Z" else measure_probs_x(psi2) if b2=="X" else measure_probs_y(psi2)
        o1 = noisy_sample(p1,rng,flip_prob=noise)
        o2 = noisy_sample(p2,rng,flip_prob=noise)
        if b1=="Z": z1[o1]+=1
        elif b1=="X": x1[o1]+=1
        else: y1[o1]+=1
        if b2=="Z": z2[o2]+=1
        elif b2=="X": x2[o2]+=1
        else: y2[o2]+=1
        psi1_hat = estimate_state_from_counts(z1,x1,y1)
        psi2_hat = estimate_state_from_counts(z2,x2,y2)
        F_new = fidelity(psi1_hat,psi1)*fidelity(psi2_hat,psi2)
        reward = F_new-F_prev
        if update_weights: weights[a]+=lr*reward*state
        F_prev=F_new
    return {"fidelity":F_prev,"infidelity":1-F_prev},weights

# ========================
# Robustness axes
# ========================
shot_budgets = [5,10,25,50,100]
noise_levels = [0.0, 0.05, 0.1]
extreme_states = [
    state_from_angles(0,0),
    state_from_angles(np.pi,0),
    state_from_angles(np.pi/2,0),
    state_from_angles(np.pi/2,np.pi/2)
]
num_test_seeds = 5
epsilon_test = 0.0
learning_rate = 0.02

# ========================
# Load or train RL weights
# ========================
# Placeholder: for simplicity, train on moderate budgets
master_rng = np.random.default_rng(123)
trained_weights = {}
for N in [10,25,50,100]:
    weights = {a: np.zeros(14) for a in ACTIONS}
    for ep in range(200):  # shorter for stress test
        theta1,phi1 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
        theta2,phi2 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
        psi1,psi2 = state_from_angles(theta1,phi1), state_from_angles(theta2,phi2)
        rng = np.random.default_rng(1000*N+ep)
        _,weights = run_rl_episode(psi1,psi2,N,weights,0.15,rng,True,learning_rate)
    trained_weights[N] = weights

# ========================
# Run robustness experiment
# ========================
robust_rows=[]
for seed_id in range(num_test_seeds):
    for N in shot_budgets:
        for noise in noise_levels:
            for psi1 in extreme_states:
                for psi2 in extreme_states:
                    rng = np.random.default_rng(seed_id*1000 + N)
                    # Fixed strategies
                    for strategy in ["Z_only","ZX_split","XYZ_split"]:
                        z1=z2=x1=x2=y1=y2=np.array([0,0],dtype=int)
                        if strategy=="Z_only":
                            for _ in range(N): z1[sample_one_shot(measure_probs_z(psi1),rng)]+=1; z2[sample_one_shot(measure_probs_z(psi2),rng)]+=1
                        elif strategy=="ZX_split":
                            for _ in range(N//2): z1[sample_one_shot(measure_probs_z(psi1),rng)]+=1; z2[sample_one_shot(measure_probs_z(psi2),rng)]+=1
                            for _ in range(N-N//2): x1[sample_one_shot(measure_probs_x(psi1),rng)]+=1; x2[sample_one_shot(measure_probs_x(psi2),rng)]+=1
                        else:
                            n_z,n_x = N//3,N//3; n_y = N-n_z-n_x
                            for _ in range(n_z): z1[sample_one_shot(measure_probs_z(psi1),rng)]+=1; z2[sample_one_shot(measure_probs_z(psi2),rng)]+=1
                            for _ in range(n_x): x1[sample_one_shot(measure_probs_x(psi1),rng)]+=1; x2[sample_one_shot(measure_probs_x(psi2),rng)]+=1
                            for _ in range(n_y): y1[sample_one_shot(measure_probs_y(psi1),rng)]+=1; y2[sample_one_shot(measure_probs_y(psi2),rng)]+=1
                        psi1_hat = estimate_state_from_counts(z1,x1,y1)
                        psi2_hat = estimate_state_from_counts(z2,x2,y2)
                        F_total = fidelity(psi1_hat,psi1)*fidelity(psi2_hat,psi2)
                        robust_rows.append({"N":N,"noise":noise,"method":strategy,"fidelity":F_total,"infidelity":1-F_total})
                    # RL
                    nearest_N = min(trained_weights.keys(), key=lambda x:abs(x-N))
                    out_rl,_ = run_rl_episode(psi1,psi2,N,trained_weights[nearest_N],epsilon_test,rng,False,learning_rate,noise=noise)
                    robust_rows.append({"N":N,"noise":noise,"method":"RL_adaptive","fidelity":out_rl["fidelity"],"infidelity":out_rl["infidelity"]})

# ========================
# Save results
# ========================
robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(f"{run_dir}/robustness_metrics.csv",index=False)

robust_agg = robust_df.groupby(["method","N","noise"]).agg(
    fidelity_mean=("fidelity","mean"),
    fidelity_std=("fidelity","std"),
    infidelity_mean=("infidelity","mean"),
    infidelity_std=("infidelity","std")
).reset_index()
robust_agg.to_csv(f"{run_dir}/robustness_summary.csv",index=False)

# ========================
# Plot example: infidelity vs noise
# ========================
plt.figure()
for m in robust_agg["method"].unique():
    sub = robust_agg[robust_agg["method"]==m].sort_values("noise")
    plt.errorbar(sub["noise"], sub["infidelity_mean"], yerr=sub["infidelity_std"], marker="o", label=m)
plt.xlabel("Noise flip probability")
plt.ylabel("Infidelity")
plt.title("2-Qubit RL vs Fixed Strategies - Noise Robustness")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_dir}/robustness_noise.png",dpi=200)
plt.close()

print("Experiment 7 (robustness) completed. Results saved to:", run_dir)