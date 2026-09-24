"""Minimal NumPy binary logistic regression used as the manual DIA baseline."""

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.05, epochs=2000, l2=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        for _ in range(self.epochs):
            probability = self._sigmoid(X @ self.coef_ + self.intercept_)
            error = probability - y
            self.coef_ -= self.learning_rate * (
                X.T @ error / len(X) + self.l2 * self.coef_
            )
            self.intercept_ -= self.learning_rate * error.mean()
        return self

    def predict_proba(self, X):
        positive = self._sigmoid(np.asarray(X) @ self.coef_ + self.intercept_)
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def _sigmoid(value):
        value = np.clip(value, -35, 35)
        return 1 / (1 + np.exp(-value))
