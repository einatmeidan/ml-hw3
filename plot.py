import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-3, 3, 400)

gamma = 1 - np.exp(-np.exp(t))
sigma = 1 / (1 + np.exp(-t))

plt.figure(figsize=(7, 4))
plt.plot(t, gamma, label=r'CLL $\gamma(t)$')
plt.plot(t, sigma, label=r'Logistic $\sigma(t)$')

plt.xlabel('t')
plt.ylabel('Predicted probability')
plt.title('CLL function vs Logistic sigmoid')
plt.legend()
plt.grid(True)
plt.show()