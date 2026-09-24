"""Small validation search for the sklearn DIA random-forest baseline."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


def fit_random_forest(X_train, y_train, X_valid, y_valid, random_state=42):
    candidates = [
        {'max_depth': 4, 'min_samples_leaf': 1},
        {'max_depth': 8, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_leaf': 5},
    ]
    best_score, best_params = -1, None
    for params in candidates:
        model = RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=1, **params)
        model.fit(X_train, y_train)
        score = average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1])
        if score > best_score:
            best_score, best_params = score, params
    return best_params
