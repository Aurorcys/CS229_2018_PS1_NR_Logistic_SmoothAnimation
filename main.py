import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


class LogisticRegressionNR:
    def __init__(self, max_iter=100, epsilon=1e-5, animate=False, X_plot=None, y_plot=None):
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.param = None
        self.animate = animate
        self.X_plot = X_plot
        self.y_plot = y_plot
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.param = np.zeros(X.shape[1] + 1)
        X_bias = np.c_[np.ones(X.shape[0]), X]
        
        if self.animate:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
        
        for iteration in range(self.max_iter):
            z = X_bias @ self.param
            p = self.sigmoid(z)
            gradient = X_bias.T @ (y - p)
            D = np.diag(p * (1 - p))
            H = X_bias.T @ D @ X_bias + 1e-6 * np.eye(X_bias.shape[1])
            
            try:
                update = np.linalg.solve(H, gradient)
            except np.linalg.LinAlgError:
                print("Hessian singular, stopping.")
                break
            
            self.param += update
            
            # ANIMATION
            if self.animate and (iteration % 5 == 0 or iteration < 10):
                ax.clear()
                ax.scatter(self.X_plot[:, 0], self.X_plot[:, 1], 
                          c=self.y_plot, cmap='bwr', edgecolors='k')
                
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xx, yy = np.meshgrid(
                    np.linspace(self.X_plot[:, 0].min() - 1, self.X_plot[:, 0].max() + 1, 100),
                    np.linspace(self.X_plot[:, 1].min() - 1, self.X_plot[:, 1].max() + 1, 100)
                )
                Z = self.sigmoid(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()] @ self.param)
                Z = Z.reshape(xx.shape)
                ax.contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=1)
                ax.set_title(f'Iteration {iteration}')
                plt.draw()
                plt.pause(0.1)
            
            if np.linalg.norm(update) < self.epsilon:
                print(f'Converged at iteration {iteration}')
                break
        
        if self.animate:
            plt.ioff()
            plt.show()

df = pd.read_csv('logistic_regression_data.csv')
X = df[['x1', 'x2']].values
y = df['y'].values


model = LogisticRegressionNR(
    max_iter=100, 
    epsilon=1e-5, 
    animate=True,     # Turn on animation
    X_plot=X,         # Pass data for plotting
    y_plot=y
)
model.fit(X, y)