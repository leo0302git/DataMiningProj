"""Create final DIA comparison tables and figures from saved fold outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'dia'
METRICS = [
    'accuracy', 'balanced_accuracy', 'macro_f1', 'positive_precision',
    'positive_recall', 'positive_f1', 'roc_auc', 'pr_auc',
]
MODEL_NAMES = {
    'manual_logistic': 'Manual logistic',
    'random_forest': 'Random forest',
    'rrl_original': 'Original RRL',
    'rrl_quantile': 'Quantile RRL',
}


def load_fold_metrics():
    baseline = pd.read_csv(RESULTS / 'baselines' / 'fold_metrics.csv')
    frames = [baseline]
    for model in ('rrl_original', 'rrl_quantile'):
        frame = pd.read_csv(RESULTS / model / 'fold_metrics.csv')
        frame.insert(0, 'model', model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_predictions(model):
    folder = 'baselines' if model in ('manual_logistic', 'random_forest') else model
    pattern = f'{model}_fold_*_predictions.csv' if folder == 'baselines' else 'fold_*_predictions.csv'
    return pd.concat([pd.read_csv(path) for path in sorted((RESULTS / folder).glob(pattern))])


def main():
    folds = load_fold_metrics()
    rows = []
    for model, group in folds.groupby('model', sort=False):
        for metric in METRICS:
            rows.append({
                'model': model,
                'metric': metric,
                'mean': group[metric].mean(),
                'std': group[metric].std(),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / 'model_comparison.csv', index=False, float_format='%.10g')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    models = list(MODEL_NAMES)
    for ax, metric, title in zip(axes, ('pr_auc', 'macro_f1'), ('PR-AUC', 'Macro-F1')):
        values = summary[summary.metric == metric].set_index('model').loc[models]
        ax.bar(MODEL_NAMES.values(), values['mean'], yerr=values['std'], capsize=4,
               color=['#4C78A8', '#F58518', '#54A24B', '#E45756'])
        ax.set_ylim(0, 1)
        ax.set_title(f'{title} (5-fold mean +/- SD)')
        ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS / 'model_comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for model, label in MODEL_NAMES.items():
        prediction = load_predictions(model)
        y_true = prediction.y_true.to_numpy()
        score = prediction.positive_probability.to_numpy()
        fpr, tpr, _ = metrics.roc_curve(y_true, score)
        precision, recall, _ = metrics.precision_recall_curve(y_true, score)
        axes[0].plot(fpr, tpr, label=f'{label} ({metrics.roc_auc_score(y_true, score):.3f})')
        axes[1].plot(recall, precision,
                     label=f'{label} ({metrics.average_precision_score(y_true, score):.3f})')
    axes[0].plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
    axes[0].set(xlabel='FPR', ylabel='TPR', title='Pooled out-of-fold ROC curves')
    axes[1].axhline(148 / 597, linestyle='--', color='grey', linewidth=1)
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Pooled out-of-fold PR curves')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(RESULTS / 'roc_pr_curves.png', dpi=180)
    plt.close(fig)

    original = folds[folds.model == 'rrl_original'].set_index('fold')
    improved = folds[folds.model == 'rrl_quantile'].set_index('fold')
    differences = improved[METRICS + ['edge_count', 'log_edges']] - original[
        METRICS + ['edge_count', 'log_edges']]
    differences.loc['mean'] = differences.mean()
    differences.to_csv(RESULTS / 'rrl_ablation.csv', float_format='%.10g')


if __name__ == '__main__':
    main()
