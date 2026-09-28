"""Nested-CV search for a stronger, still course-scope DIA RRL."""

import argparse
from collections import defaultdict
import gc
import itertools
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from rrl.models import RRL
from rrl.utils import DBEncoder, read_csv


def candidate_configs():
    product = itertools.product(
        ('random', 'quantile', 'supervised'),
        ('3@16', '5@16', '5@32', '5@64', '10@32', '5@32@16'),
        (0.0005, 0.001, 0.002, 0.005),
        (0.0, 0.0001, 0.001),
        (False, True),
        ((0.9, 3, 3), (0.999, 8, 1), (0.999, 8, 3)),
        (80, 160),
        (0.1, 1.0),
        (False, True),
    )
    population = list(product)
    sampled = random.Random(2026).sample(population, 34)
    anchors = [
        ('random', '5@16', 0.002, 0.001, False, (0.9, 3, 3), 100, 0.1, False),
        ('quantile', '5@16', 0.002, 0.001, False, (0.9, 3, 3), 100, 0.1, False),
        ('supervised', '5@32', 0.002, 0.0001, False, (0.9, 3, 3), 160, 0.1, False),
        ('supervised', '5@64', 0.001, 0.0001, True, (0.9, 3, 3), 160, 0.1, False),
        ('supervised', '10@32', 0.002, 0.001, True, (0.999, 8, 3), 160, 1.0, False),
        ('quantile', '5@32@16', 0.001, 0.0001, True, (0.999, 8, 1), 160, 0.1, True),
    ]
    configs = []
    for values in anchors + sampled:
        threshold, structure, lr, wd, weighted, nlaf, epochs, temp, use_not = values
        config = {
            'threshold': threshold,
            'structure': structure,
            'learning_rate': lr,
            'weight_decay': wd,
            'weighted': weighted,
            'alpha': nlaf[0],
            'beta': nlaf[1],
            'gamma': nlaf[2],
            'epochs': epochs,
            'temperature': temp,
            'use_not': use_not,
        }
        if config not in configs:
            configs.append(config)
    return configs


def make_cut_points(X, y, count, strategy):
    if strategy == 'random':
        return None
    quantiles = np.arange(1, count + 1) / (count + 1)
    fallback = np.quantile(X, quantiles, axis=0)
    if strategy == 'quantile':
        return torch.tensor(fallback, dtype=torch.float32)

    cut_points = np.empty_like(fallback)
    parent_positive = y.mean()
    parent_gini = 2 * parent_positive * (1 - parent_positive)
    min_leaf = max(5, round(0.05 * len(y)))
    for column in range(X.shape[1]):
        values = X[:, column]
        candidates = np.unique(np.quantile(values, np.linspace(0.03, 0.97, 63)))
        scored = []
        for threshold in candidates:
            left = values <= threshold
            left_count = left.sum()
            right_count = len(y) - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            left_positive = y[left].mean()
            right_positive = y[~left].mean()
            child_gini = (
                left_count * 2 * left_positive * (1 - left_positive)
                + right_count * 2 * right_positive * (1 - right_positive)
            ) / len(y)
            scored.append((parent_gini - child_gini, threshold))
        selected = [threshold for _, threshold in sorted(scored, reverse=True)[:count]]
        for threshold in fallback[:, column]:
            if len(selected) == count:
                break
            if not any(np.isclose(threshold, existing) for existing in selected):
                selected.append(threshold)
        selected.extend([fallback[len(selected) % count, column]] * (count - len(selected)))
        cut_points[:, column] = selected[:count]
    return torch.tensor(cut_points, dtype=torch.float32)


def loaders(X_df, y_df, f_df, train_index, eval_index, config, seed):
    encoder = DBEncoder(f_df, discrete=False)
    encoder.fit(X_df.iloc[train_index], y_df.iloc[train_index])
    X_train, y_train = encoder.transform(
        X_df.iloc[train_index], y_df.iloc[train_index], normalized=True)
    X_eval, y_eval = encoder.transform(
        X_df.iloc[eval_index], y_df.iloc[eval_index], normalized=True)
    train_set = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
    eval_set = TensorDataset(torch.tensor(X_eval, dtype=torch.float32),
                             torch.tensor(y_eval, dtype=torch.float32))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    eval_loader = DataLoader(eval_set, batch_size=64, shuffle=False)
    continuous = X_train[:, encoder.discrete_flen:]
    labels = y_train.argmax(axis=1)
    count = int(config['structure'].split('@')[0])
    cut_points = make_cut_points(continuous, labels, count, config['threshold'])
    return encoder, train_loader, eval_loader, cut_points, labels


def fit_model(X_df, y_df, f_df, train_index, eval_index, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder, train_loader, eval_loader, cut_points, labels = loaders(
        X_df, y_df, f_df, train_index, eval_index, config, seed)
    dimensions = [
        (encoder.discrete_flen, encoder.continuous_flen),
        *map(int, config['structure'].split('@')),
        len(encoder.y_fname),
    ]
    model = RRL(
        dim_list=dimensions,
        device_id='cpu',
        use_not=config['use_not'],
        is_rank0=False,
        distributed=False,
        use_nlaf=True,
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        temperature=config['temperature'],
        cut_points=cut_points,
    )
    counts = np.bincount(labels)
    class_weights = len(labels) / (2 * counts) if config['weighted'] else None
    model.train_model(
        data_loader=train_loader,
        epoch=config['epochs'],
        lr=config['learning_rate'],
        lr_decay_epoch=100,
        weight_decay=config['weight_decay'],
        class_weights=class_weights,
    )
    return model, encoder, train_loader, eval_loader


def probabilities(model, loader):
    y_true, logits = model.predict(loader)
    score = logits[:, 1] - logits[:, 0]
    score = 1 / (1 + np.exp(-np.clip(score, -35, 35)))
    return y_true, logits.argmax(axis=1), score


def evaluate(y_true, y_pred, score):
    return {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
        'macro_f1': metrics.f1_score(y_true, y_pred, average='macro'),
        'positive_precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'positive_recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'positive_f1': metrics.f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': metrics.roc_auc_score(y_true, score),
        'pr_auc': metrics.average_precision_score(y_true, score),
    }


def rule_complexity(model, encoder, train_loader, output):
    with output.open('w') as stream:
        rule2weights = model.rule_print(
            encoder.X_fname, encoder.y_fname, train_loader, file=stream,
            mean=encoder.mean, std=encoder.std)
    connected = defaultdict(set)
    layer_number = len(model.net.layer_list) - 1
    for rule_id, _ in rule2weights:
        connected[layer_number - abs(rule_id[0])].add(rule_id[1])
    edges = 0
    while layer_number > 1:
        layer_number -= 1
        layer = model.net.layer_list[layer_number]
        for rule_index in connected[layer_number]:
            conjunctions = len(layer.rule_list[0])
            option = 1 if rule_index >= conjunctions else 0
            if option:
                rule_index -= conjunctions
            rule = layer.rule_list[option][rule_index]
            edges += len(rule)
            for child in rule:
                connected[layer_number - abs(child[0])].add(child[1])
    return len(rule2weights), edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-candidates', type=int, default=40)
    parser.add_argument('--outer-folds', type=int, default=5)
    parser.add_argument('--train-final-only', action='store_true')
    args = parser.parse_args()

    X_df, y_df, f_df, _ = read_csv(
        HERE / 'dataset' / 'dia.data', HERE / 'dataset' / 'dia.info')
    labels = y_df.iloc[:, 0].to_numpy()
    outer = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X_df, labels))
    configs = candidate_configs()[:args.max_candidates]
    output = ROOT / 'results' / 'dia' / 'rrl_optimized'
    output.mkdir(parents=True, exist_ok=True)

    if args.train_final_only:
        searches = pd.concat([
            pd.read_csv(path) for path in sorted(output.glob('fold_*_search.csv'))])
        ranking = searches.groupby('config_id')['mean_inner_pr_auc'].agg(['mean', 'std'])
        best_id = int(ranking['mean'].idxmax())
        config = configs[best_id]
        indices = np.arange(len(X_df))
        model, encoder, train_loader, _ = fit_model(
            X_df, y_df, f_df, indices, indices, config, seed=2026)
        rule_count, edge_count = rule_complexity(
            model, encoder, train_loader, output / 'final_rules.txt')
        torch.save({
            'model_state_dict': model.net.state_dict(),
            'rrl_args': {
                'dim_list': [(encoder.discrete_flen, encoder.continuous_flen),
                             *map(int, config['structure'].split('@')),
                             len(encoder.y_fname)],
                'use_not': config['use_not'],
                'estimated_grad': False,
                'use_skip': False,
                'use_nlaf': True,
                'alpha': config['alpha'],
                'beta': config['beta'],
                'gamma': config['gamma'],
                'temperature': config['temperature'],
            },
            'config': config,
            'feature_names': encoder.X_fname,
            'label_names': encoder.y_fname,
            'mean': encoder.mean.to_dict(),
            'std': encoder.std.to_dict(),
        }, output / 'final_model.pth')
        with (output / 'final_config.json').open('w') as stream:
            json.dump({
                'config_id': best_id,
                'config': config,
                'mean_inner_pr_auc': ranking.loc[best_id, 'mean'],
                'std_inner_pr_auc': ranking.loc[best_id, 'std'],
                'rule_count': rule_count,
                'edge_count': edge_count,
            }, stream, indent=2)
        print((output / 'final_config.json').read_text())
        return

    fold_rows = []

    for outer_fold, (outer_train, outer_test) in enumerate(outer[:args.outer_folds]):
        inner_splits = list(StratifiedKFold(
            3, shuffle=True, random_state=100 + outer_fold).split(
                outer_train, labels[outer_train]))
        search_rows = []
        for config_id, config in enumerate(configs):
            scores = []
            for inner_fold, (inner_train_pos, inner_valid_pos) in enumerate(inner_splits):
                model, _, _, valid_loader = fit_model(
                    X_df, y_df, f_df,
                    outer_train[inner_train_pos], outer_train[inner_valid_pos],
                    config, seed=1000 + outer_fold * 10 + inner_fold)
                y_true, _, score = probabilities(model, valid_loader)
                scores.append(metrics.average_precision_score(y_true, score))
                del model
                gc.collect()
            search_rows.append({
                'outer_fold': outer_fold,
                'config_id': config_id,
                'mean_inner_pr_auc': np.mean(scores),
                'std_inner_pr_auc': np.std(scores, ddof=1),
                'config': json.dumps(config, sort_keys=True),
            })
            print(
                f'outer={outer_fold} config={config_id + 1}/{len(configs)} '
                f'inner_pr_auc={np.mean(scores):.4f}', flush=True)

        search = pd.DataFrame(search_rows).sort_values(
            ['mean_inner_pr_auc', 'std_inner_pr_auc'], ascending=[False, True])
        search.to_csv(output / f'fold_{outer_fold}_search.csv',
                      index=False, float_format='%.10g')
        best = json.loads(search.iloc[0]['config'])
        model, encoder, train_loader, test_loader = fit_model(
            X_df, y_df, f_df, outer_train, outer_test, best, seed=42 + outer_fold)
        y_true, y_pred, score = probabilities(model, test_loader)
        _, discrete_logits = model.predict(test_loader, binarized=True)
        prediction_logits = model.predict(test_loader)[1]
        rule_count, edge_count = rule_complexity(
            model, encoder, train_loader, output / f'fold_{outer_fold}_rrl.txt')
        row = {
            'fold': outer_fold,
            **evaluate(y_true, y_pred, score),
            'rule_count': rule_count,
            'edge_count': edge_count,
            'log_edges': np.log(edge_count) if edge_count else np.nan,
            'decision_fidelity': np.mean(
                prediction_logits.argmax(axis=1) == discrete_logits.argmax(axis=1)),
            'selected_config': json.dumps(best, sort_keys=True),
            'inner_pr_auc': search.iloc[0]['mean_inner_pr_auc'],
        }
        fold_rows.append(row)
        pd.DataFrame({
            'index': outer_test,
            'y_true': y_true,
            'y_pred': y_pred,
            'positive_probability': score,
        }).to_csv(output / f'fold_{outer_fold}_predictions.csv',
                  index=False, float_format='%.10g')
        pd.DataFrame(fold_rows).to_csv(
            output / 'fold_metrics.csv', index=False, float_format='%.10g')
        print(f'outer={outer_fold} test_pr_auc={row["pr_auc"]:.4f} best={best}', flush=True)

    folds = pd.DataFrame(fold_rows)
    numeric = folds.select_dtypes(include=np.number).drop(columns=['fold'])
    numeric.agg(['mean', 'std']).T.to_csv(
        output / 'summary.csv', float_format='%.10g')
    print(numeric.agg(['mean', 'std']).T.round(4).to_string())


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
