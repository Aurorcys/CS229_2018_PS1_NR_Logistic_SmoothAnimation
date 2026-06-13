# Logistic Regression with Newton-Raphson

A from-scratch implementation of logistic regression using Newton-Raphson optimization with real-time decision boundary animation.

## Project Structure

main.py - LogisticRegressionNR class + training + animation
dataloading.py - Generate synthetic data with sklearn
math.txt - Full mathematical derivation
logistic_regression_data.csv - Pre-generated dataset (200 samples, 2 features)

## The Math

We model the probability that a point belongs to class 1 as:

$$P(y=1 \mid x) = \sigma(w \cdot x) = \frac{1}{1 + e^{-w \cdot x}}$$

The cost function is the negative log-likelihood over all $$m$$ samples:

$$J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log \sigma(w \cdot x_i) + (1-y_i) \log(1 - \sigma(w \cdot x_i)) \right]$$

We minimize $$J(w)$$ using Newton-Raphson, which requires both the gradient
and the Hessian matrix of second derivatives.

**First derivative (gradient):**

$$\nabla J(w) = \frac{1}{m} \cdot X^T (h - y)$$

where $$h = \sigma(Xw)$$ is the vector of predicted probabilities.

**Second derivative (Hessian):**

$$H = \nabla^2 J(w) = \frac{1}{m} \cdot X^T D X$$

where $$D = \text{diag}(h_i \cdot (1 - h_i))$$ is a diagonal matrix.

**Newton-Raphson update rule:**

$$w^{(t+1)} = w^{(t)} - H^{-1} \nabla J(w)$$

$$w^{(t+1)} = w^{(t)} + (X^T D X)^{-1} X^T (y - h)$$

Each iteration takes a step whose size and direction are determined by
both the slope (gradient) and the curvature (Hessian). This gives
quadratic convergence near the optimum, typically in 5-15 iterations.

## Usage

Generate data: python dataloading.py
Run: python main.py

## Features

Newton-Raphson optimization with quadratic convergence
No learning rate needed, Hessian gives optimal step size
Real-time animation of decision boundary converging
Hessian regularization + np.linalg.solve for stability
Convergence detection when update norm falls below threshold

## Why Newton-Raphson

Gradient Descent: Linear convergence, needs tuning, hundreds of iterations
Newton-Raphson: Quadratic convergence, automatic step size, 5-15 iterations

## Dependencies

pip install numpy pandas matplotlib scikit-learn

## Implementation Details

Bias term added via np.c_[np.ones(n), X]
Hessian regularized with lambda = 1e-6
Uses np.linalg.solve(H, gradient) instead of np.linalg.inv(H)
