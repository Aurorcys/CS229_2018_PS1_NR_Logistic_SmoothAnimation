import numpy as np
import pandas as pd 

np.random.seed(42)
n = 200

X1 = np.random.randn(n // 2, 2) + np.array([-2, -2])
X2 = np.random.randn(n // 2, 2) + np.array([2, 2])
X = np.vstack((X1, X2))
y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)

X += np.random.randn(*X.shape) * 0.3

df = pd.DataFrame(X, columns=['x1', 'x2'])
df['y'] = y
df.to_csv('logistic_regression_data', index=False)

print(f'Saved {n} Samples for Logistic Regression Newton Raphson method')
print('Data saved to logistic_regression_data.csv')
