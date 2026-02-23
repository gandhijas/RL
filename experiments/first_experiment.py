import os
from datetime import datetime

# Create timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = f"results/{timestamp}_exp00"

os.makedirs(run_dir, exist_ok=True)

import pandas as pd


import numpy as np

theta_vals = [0.1*np.pi, 0.25*np.pi, 0.4*np.pi]
sample_sizes = [10, 50, 100, 500, 1000]

results = {}

for theta in theta_vals:
    p0 = np.cos(theta)**2
    p1 = np.sin(theta)**2
    true_prob = np.array([p0, p1])

    errors = []

    for N in sample_sizes:
        samples = np.random.choice([0,1], size=N, p=true_prob)

        counts = np.bincount(samples, minlength=2)
        empirical_prob = counts/N

        l1_error = np.sum(np.abs(true_prob - empirical_prob))
        errors.append(l1_error)

        results[theta] = errors

import matplotlib.pyplot as plt

for theta, errors in results.items():
    plt.plot(sample_sizes, errors, marker ='o', label = f"0 = {theta/np.pi:.2f}π")

rows = []

for theta, errors in results.items():
    for N, err in zip(sample_sizes, errors):
        rows.append({
            "theta": theta,
            "N": N,
            "l1_error": err
        })

df = pd.DataFrame(rows)


plt.xlabel("Number of samples (N)")
plt.ylabel("L1 error")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.title("Monte Carlo approximation error for different value of theta")
plt.savefig(f"{run_dir}/sampling_convergence.png", dpi=200)
df.to_csv(f"{run_dir}/metrics.csv", index=False)
plt.show()