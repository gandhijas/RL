import numpy as np
import matplotlib.pyplot as plt

theta = np.pi / 4  # |+> state

p0 = np.cos(theta)**2
p1 = np.sin(theta)**2
probs = [p0, p1]

labels = ['|0>', '|1>']
plt.bar(labels, probs)
plt.ylabel('Probability')
plt.title('Quantum Measurement Distribution')
plt.ylim(0, 1)
plt.show()
