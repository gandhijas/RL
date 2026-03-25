import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# Run directory
# ========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp05_2qubitRL_stress"
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
    phi_hat = np.mod(np.arctan2(y_n, x_n), 2*np.pi)
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
def run_rl_episode(psi1, psi2, total_shots, weights, epsilon, rng, update_weights, lr):
    z1=x1=y1=np.array([0,0],dtype=int)
    z2=x2=y2=np.array([0,0],dtype=int)
    shots_used = 0
    # Warm start 1 shot per basis
    for basis in ["Z","X","Y"]:
        if shots_used>=total_shots: break
        p1 = measure_probs_z(psi1) if basis=="Z" else measure_probs_x(psi1) if basis=="X" else measure_probs_y(psi1)
        p2 = measure_probs_z(psi2) if basis=="Z" else measure_probs_x(psi2) if basis=="X" else measure_probs_y(psi2)
        o1 = sample_one_shot(p1,rng)
        o2 = sample_one_shot(p2,rng)
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
        o1 = sample_one_shot(p1,rng)
        o2 = sample_one_shot(p2,rng)
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
        if update_weights: weights[a] += lr*reward*state
        F_prev = F_new
    return {"fidelity":F_prev,"infidelity":1-F_prev},weights

# ========================
# Experiment setup
# ========================
stress_shot_budgets = [1,2,5,10,25,50,100,250,500,1000]
num_test_targets = 50
num_test_seeds = 5
epsilon_train = 0.15
epsilon_test = 0.0
learning_rate = 0.02
num_train_episodes = 800
master_rng = np.random.default_rng(123)

# ========================
# Train RL
# ========================
trained_weights = {}
for N in [10,25,50,100,250,500]:  # train on standard budgets
    weights = {a: np.zeros(14) for a in ACTIONS}
    for ep in range(num_train_episodes):
        theta1,phi1 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
        theta2,phi2 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
        psi1,state2 = state_from_angles(theta1,phi1), state_from_angles(theta2,phi2)
        rng = np.random.default_rng(10000*N+ep)
        _,weights = run_rl_episode(psi1,state2,N,weights,epsilon_train,rng,True,learning_rate)
    trained_weights[N] = weights

# ========================
# Run stress test
# ========================
stress_rows=[]
for target_id in range(num_test_targets):
    theta1, phi1 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
    theta2, phi2 = master_rng.uniform(0,np.pi), master_rng.uniform(0,2*np.pi)
    psi1 = state_from_angles(theta1,phi1)
    psi2 = state_from_angles(theta2,phi2)
    for seed in range(num_test_seeds):
        for N in stress_shot_budgets:
            rng = np.random.default_rng(target_id*1000+seed*100+N)
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
                stress_rows.append({"target_id":target_id,"seed":seed,"N":N,"method":strategy,"fidelity":F_total,"infidelity":1-F_total})
            # RL
            nearest_N = min(trained_weights.keys(), key=lambda x: abs(x-N))
            out_rl,_ = run_rl_episode(psi1,psi2,N,trained_weights[nearest_N],epsilon_test,rng,False,learning_rate)
            stress_rows.append({"target_id":target_id,"seed":seed,"N":N,"method":"RL_adaptive","fidelity":out_rl["fidelity"],"infidelity":out_rl["infidelity"]})

# ========================
# Save + plot
# ========================
stress_df = pd.DataFrame(stress_rows)
stress_df.to_csv(f"{run_dir}/stress_metrics.csv",index=False)

stress_agg = stress_df.groupby(["method","N"]).agg(
    fidelity_mean=("fidelity","mean"),
    fidelity_std=("fidelity","std"),
    infidelity_mean=("infidelity","mean"),
    infidelity_std=("infidelity","std")
).reset_index()
stress_agg.to_csv(f"{run_dir}/stress_summary.csv",index=False)

plt.figure()
for m in stress_agg["method"].unique():
    sub = stress_agg[stress_agg["method"]==m].sort_values("N")
    x = sub["N"].to_numpy()
    y = sub["infidelity_mean"].to_numpy()
    ystd = sub["infidelity_std"].to_numpy()
    plt.plot(x,y,marker="o",label=m)
    plt.fill_between(x,y-ystd,y+ystd,alpha=0.2)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Shots (N)")
plt.ylabel("Infidelity")
plt.title("2-Qubit RL vs Fixed Strategies - Budget Stress Test")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_dir}/stress_infidelity.png",dpi=200)
plt.close()

print("Stress test completed. Results saved to:", run_dir)