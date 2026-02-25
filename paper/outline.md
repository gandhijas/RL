# Working Title
Reinforcement Learning for learning to approximate Quantum Disttributions rather than a classical methods 

## 1. Introduction
- Quantum systems produce probability distributions via measurement
- Estimating these distributions is sample-expensive
- Classical estimators rely on fixed heuristics
- Question: Can RL learn strategies that reduce estimation error?

        ## What "Better" Means

    RL is considered better if it achieves:
    - Lower L1 / KL error at fixed shot budget
    - Fewer shots to reach target error threshold
    - Better tracking under nonstationary drift
    - Comparable or lower variance across runs

## 2. Background
### 2.1 Quantum Measurement
- Born rule
- Distribution over bitstrings
- Shot-based sampling noise

### 2.2 Classical Estimators
- Empirical frequency (MLE)
- Dirichlet smoothing
- Limitations under tight budgets

### 2.3 Reinforcement Learning
- Policy
- Reward
- Sample efficiency

## 3. Problem Formulation
- Unknown distribution P(x)
- Limited shot budget B
- Goal: minimize L1 / KL error
- Compare RL vs classical baselines

## 4. Methods
- Environment definition
- Action space (measurement choice / smoothing / allocation)
- Reward definition
- Training setup
- Evaluation protocol
        ### Evaluation Metrics

        - L1 distance
        - KL divergence
        - Total variation distance
        - Shots-to-threshold
        - Area under error curve

## 5. Experiments
### 5.1 Sampling Convergence (Exp00)
### 5.2 1-Qubit Distribution Estimation
### 5.3 2-Qubit Distribution
### 5.4 Adaptive Measurement Strategy
### 5.5 Drift / Nonstationary Case

## 6. Results
- Error vs shots
- Budget comparisons
- Generalization
- Robustness

## 7. Discussion
- Where RL helps
- Where it doesn’t
- Scaling challenges
- Theoretical limits

        ## Risks

- RL does not significantly outperform classical baselines
- Improvement is statistically insignificant
- Results depend heavily on tuning
- Scaling fails at 3 qubits

## 8. Conclusion


    ## Future Work

- Extend to higher qubit counts
- Incorporate hardware noise models
- Compare with Bayesian adaptive tomography
- Investigate theoretical bounds on adaptive sampling
- Explore actor-critic vs policy-gradient variants