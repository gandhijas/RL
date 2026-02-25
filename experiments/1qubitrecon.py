import numpy as np 
import matplotlib.pyplot as plt
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp01"
os.makedirs(run_dir, exist_ok=True)




def ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])

def state_from_theta(theta):
    zero = np.array([1, 0])
    return ry(theta) @ zero


def fidelity(psi, phi):

    overlap = np.vdot(psi, phi)
    return np.abs(overlap)**2


true_theta = 0.73 * np.pi
psi_true = state_from_theta(true_theta)

num_iterations = 200
best_fidelity = 0
best_theta = None
fidelity_history = []

for i in range(num_iterations):

    candidate_theta = np.random.uniform(0, 2*np.pi)
    psi_candidate = state_from_theta(candidate_theta)

    F = fidelity(psi_true, psi_candidate)

    if F > best_fidelity:
        best_fidelity = F
        best_theta = candidate_theta

    fidelity_history.append(best_fidelity)

print("True theta:", true_theta)
print("Best theta:", best_theta)
print("Best fidelity:", best_fidelity)

plt.plot(fidelity_history)
plt.xlabel("Iteration")
plt.ylabel("Best Fidelity So Far")
plt.title("1-Qubit Reconstruction by random search")
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig(f"{run_dir}/fidelity_vs_iteration.png", dpi=200)
plt.show()