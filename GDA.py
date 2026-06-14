import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

df = pd.read_csv("logistic_regression_data.csv")
y = df["y"].values
X = df[["x1", "x2"]].values

phi = (1/df.shape[0]) * np.sum(df["y"] == 1)
mu0 = (np.sum(df[df["y"] == 0][["x1", "x2"]], axis=0) / np.sum(df["y"] == 0)).values
mu1 = (np.sum(df[df["y"] == 1][["x1", "x2"]], axis=0) / np.sum(df["y"] == 1)).values
mu_map = np.where(y[:, None] == 0, mu0, mu1)
diff = X - mu_map
sigma = (diff.T @ diff) / df.shape[0]


theta1 = np.linalg.solve(sigma, mu1 - mu0)
theta0 = -np.log((1- phi) / phi) - 0.5 * (mu1.T @ np.linalg.solve(sigma, mu1) - mu0.T @ np.linalg.solve(sigma, mu0))

probs = 1 / (1 + np.exp(-(X @ theta1 + theta0)))

pred = (probs >= 0.5).astype(int)

accuracy = np.mean(pred == y)
print(f"Accuracy: {accuracy:.4f}")



# Plot the two Gaussians
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x1_grid = np.linspace(x1_min, x1_max, 100)
x2_grid = np.linspace(x2_min, x2_max, 100)
xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
pos = np.dstack((xx1, xx2))

# Gaussian for class 0
rv0 = multivariate_normal(mu0, sigma)
Z0 = rv0.pdf(pos)

# Gaussian for class 1
rv1 = multivariate_normal(mu1, sigma)
Z1 = rv1.pdf(pos)

# Plot contours
plt.figure(figsize=(8, 6))
plt.contour(xx1, xx2, Z0, levels=5, cmap='Blues', alpha=0.6)
plt.contour(xx1, xx2, Z1, levels=5, cmap='Reds', alpha=0.6)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Class 0', edgecolors='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Class 1', edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Two Gaussians (GDA)')
plt.legend()
plt.show()
