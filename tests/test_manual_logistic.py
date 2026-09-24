import numpy as np

from models.manual_glm.logistic import LogisticRegression


def test_manual_logistic_learns_separable_data():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(learning_rate=0.1, epochs=500).fit(X, y)
    assert np.array_equal(model.predict(X), y)
