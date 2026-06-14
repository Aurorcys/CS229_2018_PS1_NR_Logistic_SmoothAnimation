# Logistic Regression with Newton-Raphson + Gaussian Discriminant Analysis

A from-scratch implementation of logistic regression using Newton-Raphson optimization (with real-time decision boundary animation) and Gaussian Discriminant Analysis (closed-form MLE).

## Project Structure

- `main.py` - LogisticRegressionNR class + training + animation
- `gda.py` - Gaussian Discriminant Analysis implementation
- `dataloading.py` - Generate synthetic data with sklearn
- `math.txt` - Full mathematical derivation
- `logistic_regression_data.csv` - Pre-generated dataset (200 samples, 2 features)

---

## The Math (Logistic Regression)

We model the probability that a point belongs to class 1 as:

$$
P(y=1 \mid x) = \sigma(w \cdot x) = \frac{1}{1 + e^{-w \cdot x}}
$$

The cost function is the negative log-likelihood over all $m$ samples:

$$
J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log \sigma(w \cdot x_i) + (1-y_i) \log(1 - \sigma(w \cdot x_i)) \right]
$$

We minimize $J(w)$ using Newton-Raphson, which requires both the gradient and the Hessian matrix of second derivatives.

**First derivative (gradient):**

$$
\nabla J(w) = \frac{1}{m} X^T (h - y)
$$

where $h = \sigma(Xw)$ is the vector of predicted probabilities.

**Second derivative (Hessian):**

$$
H = \frac{1}{m} X^T D X
$$

where $D$ is a diagonal matrix with entries $D_{ii} = h_i (1 - h_i)$.

**Newton-Raphson update rule:**

$$
w^{(t+1)} = w^{(t)} - H^{-1} \nabla J(w)
$$

$$
w^{(t+1)} = w^{(t)} + (X^T D X)^{-1} X^T (y - h)
$$

Each iteration takes a step whose size and direction are determined by both the slope (gradient) and the curvature (Hessian). This gives quadratic convergence near the optimum, typically in 5–15 iterations.

---

## The Math (Gaussian Discriminant Analysis)

We model the joint distribution $P(x, y)$ as:

$$
\begin{aligned}
P(y) &= \phi^y (1 - \phi)^{1-y} \\
P(x \mid y = 0) &= \mathcal{N}(\mu_0, \Sigma) \\
P(x \mid y = 1) &= \mathcal{N}(\mu_1, \Sigma)
\end{aligned}
$$

where $\phi = P(y=1)$, $\mu_0, \mu_1$ are class means, and $\Sigma$ is the shared covariance matrix.

**Maximum likelihood estimates:**

$$
\begin{aligned}
\phi &= \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{y_i = 1\} \\[6pt]
\mu_0 &= \frac{\sum_{y_i=0} x_i}{\sum_{y_i=0} 1}, \qquad
\mu_1 = \frac{\sum_{y_i=1} x_i}{\sum_{y_i=1} 1} \\[6pt]
\Sigma &= \frac{1}{m} \sum_{i=1}^m (x_i - \mu_{y_i})(x_i - \mu_{y_i})^T
\end{aligned}
$$

The posterior $P(y=1 \mid x)$ takes the form of a sigmoid:

$$
P(y=1 \mid x) = \frac{1}{1 + e^{-(\theta^T x + \theta_0)}}
$$

where

$$
\begin{aligned}
\theta &= \Sigma^{-1} (\mu_1 - \mu_0) \\[6pt]
\theta_0 &= -\ln\frac{1-\phi}{\phi} - \frac{1}{2}\mu_1^T \Sigma^{-1} \mu_1 + \frac{1}{2}\mu_0^T \Sigma^{-1} \mu_0
\end{aligned}
$$

Thus, GDA also produces a **linear decision boundary**, identical in form to logistic regression.

---

## GDA vs Logistic Regression

| Aspect | GDA | Logistic Regression |
|--------|-----|---------------------|
| Assumption | Gaussian class conditionals, shared covariance | No distributional assumption |
| Parameter estimation | Closed-form (MLE) | Iterative (Newton-Raphson) |
| Decision boundary | Linear | Linear |
| Sample efficiency | Better if assumptions hold | More robust to misspecification |

---

## Usage

Generate data:
```bash
python dataloading.py
```

Run logistic regression:
```bash
python main.py
```

Run GDA:
```bash
python gda.py
```

---

## Features

**Logistic Regression:**
- Newton-Raphson optimization with quadratic convergence
- No learning rate needed, Hessian gives optimal step size
- Real-time animation of decision boundary converging
- Hessian regularization + `np.linalg.solve` for stability
- Convergence detection when update norm falls below threshold

**GDA:**
- Closed-form parameter estimation (no iterative optimization)
- Shared covariance matrix for both classes
- Visualizes two Gaussians + decision boundary
- Reports classification accuracy

---

## Why Newton-Raphson

| Method | Convergence | Iterations |
|--------|-------------|------------|
| Gradient Descent | Linear | Hundreds |
| Newton-Raphson | Quadratic | 5–15 |

---

## Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Implementation Details

- Bias term added via `np.c_[np.ones(n), X]`
- Hessian regularized with $\lambda = 10^{-6}$
- Uses `np.linalg.solve(H, gradient)` instead of `np.linalg.inv(H)`
- In GDA, covariance computed as sum of scatter matrices from both classes

