"""Run the two required DIA comparison models with the shared outer folds."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.manual_glm.logistic import LogisticRegression
from models.tree_ensemble.random_forest import fit_random_forest


def evaluate(y_true, y_pred, y_score):
    return {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'macro_f1': metrics.f1_score(y_true, y_pred, average='macro'),
        'positive_precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'positive_recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'positive_f1': metrics.f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': metrics.roc_auc_score(y_true, y_score),
        'pr_auc': metrics.average_precision_score(y_true, y_score),
    }


def main():
    raw = ROOT / 'data' / 'raw' / 'dia'
    frame = pd.concat([
        pd.read_csv(raw / 'DIA_trainingset_RDKit_descriptors.csv'),
        pd.read_csv(raw / 'DIA_testset_RDKit_descriptors.csv'),
    ], ignore_index=True)
    X = frame.drop(columns=['Label', 'SMILES']).to_numpy(dtype=float)
    y = frame['Label'].to_numpy(dtype=int)
    output = ROOT / 'results' / 'dia' / 'baselines'
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
        inner_train, valid = train_test_split(
            train_index, test_size=0.2, random_state=42, stratify=y[train_index])

        inner_scaler = StandardScaler().fit(X[inner_train])
        best_l2, best_score = None, -1
        for l2 in (0.0, 0.001, 0.01, 0.1):
            model = LogisticRegression(l2=l2).fit(
                inner_scaler.transform(X[inner_train]), y[inner_train])
            score = metrics.average_precision_score(
                y[valid], model.predict_proba(inner_scaler.transform(X[valid]))[:, 1])
            if score > best_score:
                best_l2, best_score = l2, score

        scaler = StandardScaler().fit(X[train_index])
        manual = LogisticRegression(l2=best_l2).fit(
            scaler.transform(X[train_index]), y[train_index])
        manual_score = manual.predict_proba(scaler.transform(X[test_index]))[:, 1]
        manual_pred = (manual_score >= 0.5).astype(int)
        rows.append({'model': 'manual_logistic', 'fold': fold, 'l2': best_l2,
                     **evaluate(y[test_index], manual_pred, manual_score)})
        pd.DataFrame({'index': test_index, 'y_true': y[test_index],
                      'y_pred': manual_pred, 'positive_probability': manual_score}).to_csv(
                          output / f'manual_logistic_fold_{fold}_predictions.csv', index=False)

        params = fit_random_forest(
            X[inner_train], y[inner_train], X[valid], y[valid], random_state=42 + fold)
        forest = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=42 + fold, **params).fit(
                X[train_index], y[train_index])
        forest_score = forest.predict_proba(X[test_index])[:, 1]
        forest_pred = (forest_score >= 0.5).astype(int)
        rows.append({'model': 'random_forest', 'fold': fold,
                     'params': json.dumps(params, sort_keys=True),
                     **evaluate(y[test_index], forest_pred, forest_score)})
        pd.DataFrame({'index': test_index, 'y_true': y[test_index],
                      'y_pred': forest_pred, 'positive_probability': forest_score}).to_csv(
                          output / f'random_forest_fold_{fold}_predictions.csv', index=False)

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(output / 'fold_metrics.csv', index=False)
    metric_columns = list(evaluate(np.array([0, 1]), np.array([0, 1]), np.array([0.0, 1.0])))
    summary = fold_metrics.groupby('model')[metric_columns].agg(['mean', 'std'])
    summary.to_csv(output / 'summary.csv')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
