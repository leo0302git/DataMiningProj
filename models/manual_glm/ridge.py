"""NumPy ridge regression; the intercept is not penalized."""
import numpy as np


class RidgeRegression:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        design = np.column_stack([np.ones(len(X)), X])
        penalty = np.eye(design.shape[1]) * self.alpha * len(X)
        penalty[0, 0] = 0
        self.coef_ = np.linalg.lstsq(design.T @ design + penalty, design.T @ y, rcond=None)[0]
        return self

    def predict(self, X):
        return self.coef_[0] + X @ self.coef_[1:]
