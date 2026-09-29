"""独立核查文献实验，再生成中文报告；不训练、不改变实验设计。"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.audit_abalone_pure import check_stopping, check_scores, check_graph
from analysis.abalone_pure_report import table
from analysis.eda import load_abalone
from models.manual_glm.ridge import RidgeRegression
from models.rrl.abalone import RRL, prepare, cut_points, encoded_categories, predict_rules
from models.rrl.abalone_refine import predict_graph, predict_refit

OUT = ROOT/'results/abalone/literature_rrl'
METRICS = ['rmse', 'mae', 'r2', 'tail_mae']
FIXED = dict(reference=0, tree_control=2, cap_control=8, tree_cap_control=10)
NAMES = ['reference', 'local_supervised', 'tree_leaf_02', 'tree_leaf_05',
         'global_supervised', 'edge_1e5', 'edge_1e4', 'cap_4', 'cap_8', 'cap_16',
         'tree_cap_8', 'wide_128']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_variants(config, pool):
    result = dict(selected=config, **{name: pool[cid]['config'] for name, cid in FIXED.items()})
    if config['threshold'] != 'quantile':
        result['no_threshold_change'] = {k: v for k, v in dict(config, threshold='quantile').items()
                                         if k != 'tree_min_leaf'}
    for key, name in [('max_fanin', 'no_cap'), ('edge_lambda', 'no_edge_penalty')]:
        if config.get(key):
            result[name] = {k: v for k, v in config.items() if k != key}
    if config['structure'] != '20@64':
        result['no_widening'] = dict(config, structure='20@64')
    return result


def check_diagnostics(record, train, config, graph=None):
    diagnostic = record['diagnostics']
    lengths = np.asarray(diagnostic['literals'])
    support = np.asarray(diagnostic['support'])
    assert len(lengths) == len(support) == diagnostic['rule_nodes']
    assert lengths.max() == diagnostic['max_literals']
    assert lengths.sum() == diagnostic['logical_edges']
    assert np.isclose(lengths.mean(), diagnostic['mean_literals'])
    assert np.isfinite(support).all() and np.all((support >= 0) & (support <= 1))
    if config.get('max_fanin'):
        assert lengths.max() <= config['max_fanin']
        for row in record['stopping']['history']:
            assert row['max_fanin_observed'] <= config['max_fanin']
    X, state = prepare(train, transform=config['transform'], full_categories=config['full_categories'])
    continuous = X[:, len(encoded_categories(state)):]
    count = int(config['structure'].split('@')[0])
    expected = cut_points(continuous, train.Rings.to_numpy(float), count, config['threshold'],
                          config.get('tree_min_leaf', .02)).numpy()
    assert np.array_equal(np.asarray(diagnostic['thresholds'], dtype='float32'), expected)
    assert diagnostic['unique_cuts'] == [len(np.unique(c)) for c in expected.T]
    occupancy = [np.bincount(np.searchsorted(np.unique(c), x, side='left'),
                            minlength=len(np.unique(c))+1).tolist()
                 for c, x in zip(expected.T, continuous.T)]
    assert occupancy == diagnostic['bin_counts']
    if config['threshold'] == 'tree':
        trees = [DecisionTreeRegressor(max_leaf_nodes=count+1, min_samples_leaf=config.get('tree_min_leaf', .02),
                                       random_state=0).fit(continuous[:, j:j+1], train.Rings.to_numpy(float))
                 for j in range(continuous.shape[1])]
        splits = [int((tree.tree_.children_left != -1).sum()) for tree in trees]
        assert diagnostic['tree_split_counts'] == splits
        assert diagnostic['quantile_fallback_slots'] == [count-n for n in splits]
    if graph is None:
        return
    check_graph(graph, train, config)
    assert np.array_equal(np.asarray(graph['thresholds'], dtype='float32'), expected)
    layer = graph['layers'][0]
    actual_lengths = np.concatenate([np.asarray(layer[k]).sum(0) for k in ['conjunction', 'disjunction']])
    assert np.array_equal(lengths, actual_lengths)
    rules = predict_rules(graph, train, return_rules=True)
    observed = rules.mean(0)
    nonconstant = (observed > 0) & (observed < 1)
    assert np.allclose(support, observed, rtol=0, atol=1e-7)
    assert nonconstant.sum() == diagnostic['nonconstant']
    assert np.unique(rules[:, nonconstant], axis=1).shape[1] == diagnostic['distinct_nonconstant']
    assert np.linalg.matrix_rank(rules-rules.mean(0)) == diagnostic['centered_rule_rank']
    # Refit independently from exported Boolean activations on this train set.
    coef = RidgeRegression(config['rule_head_alpha']).fit(rules, train.Rings.to_numpy(float)).coef_
    assert np.allclose(np.r_[graph['bias'], graph['weights']], coef, atol=1e-7, rtol=1e-7)


def check_record(record, key, config, seed, train, test):
    assert record['key'] == key and record['config'] == config and record['seed'] == seed
    assert record['train_indices'] == train.index.tolist()
    assert record['evaluation_indices'] == test.index.tolist()
    assert set(train.index).isdisjoint(test.index)
    assert all(np.isfinite(record[m]) for m in METRICS)
    assert record['seconds'] >= 0
    check_stopping(record, train.index.to_numpy(), config, seed)
    check_diagnostics(record, train, config)


def audit():
    manifest = read(OUT/'design.json')
    assert digest(ROOT/'data/raw/abalone/abalone.data') == manifest['data_sha256']
    assert digest(ROOT/'results/abalone/pure_rrl/design.json') == manifest['historical_design_sha256']
    for path, expected in manifest['source_sha256'].items():
        assert digest(ROOT/path) == expected, f'Source changed: {path}'
    pool, seeds = manifest['candidates'], manifest['seeds']
    assert [c['name'] for c in pool] == NAMES and seeds == [314, 2718]
    assert manifest['delivery_seed'] == 314
    assert manifest['outer'] == dict(n_splits=5, shuffle=True, random_state=0)
    frame = load_abalone()
    records = {p.stem: read(p) for p in (OUT/'trials').glob('*.json')}
    dev = pd.read_csv(OUT/'development.csv')
    expected_keys, searches, outer, selected, graph_errors, old_errors = set(), [], [], [], [], []
    train, test = frame.iloc[manifest['train_indices']], frame.iloc[manifest['validation_indices']]
    historical_design = read(ROOT/'results/abalone/pure_rrl/design.json')
    for field in ['train_indices', 'validation_indices']:
        assert manifest[field] == historical_design[field]
    first_outer_train = next(KFold(**manifest['outer']).split(frame))[0]
    assert len(dev) == 24 and set(train.index) | set(test.index) == set(first_outer_train)
    for cid, candidate in enumerate(pool):
        for seed in seeds:
            key = f'dev_{cid}_{seed}'
            expected_keys.add(key)
            record = records[key]
            check_record(record, key, candidate['config'], seed, train, test)
            row = dev[(dev.candidate == cid) & (dev.seed == seed)]
            assert len(row) == 1 and row.iloc[0]['name'] == candidate['name']
            assert np.isclose(row.iloc[0].rmse, record['rmse'], atol=1e-12, rtol=0)
    for fold, (tr, te) in enumerate(KFold(**manifest['outer']).split(frame)):
        train, test = frame.iloc[tr], frame.iloc[te]
        inner = pd.read_csv(OUT/f'inner_{fold}.csv')
        expected_rows = set()
        for k, (a, b) in enumerate(KFold(3, shuffle=True, random_state=100+fold).split(train)):
            for cid, candidate in enumerate(pool):
                for seed in seeds:
                    key = f'inner_{fold}_{k}_{cid}_{seed}'
                    expected_keys.add(key)
                    record = records[key]
                    check_record(record, key, candidate['config'], seed, train.iloc[a], train.iloc[b])
                    row = inner[(inner.candidate == cid) & (inner.inner_fold == k) & (inner.seed == seed)]
                    assert len(row) == 1 and np.isclose(row.iloc[0].rmse, record['rmse'], atol=1e-12, rtol=0)
                    expected_rows.add((cid, k, seed))
        assert len(inner) == 72 and set(zip(inner.candidate, inner.inner_fold, inner.seed)) == expected_rows
        cid = int(inner.groupby('candidate').rmse.mean().sort_values(kind='stable').index[0])
        config = pool[cid]['config']
        assert read(OUT/f'selection_{fold}.json') == dict(candidate=cid, config=config)
        selected.append(cid)
        searches.append(inner.assign(outer_fold=fold))
        for name, run in expected_variants(config, pool).items():
            for seed in seeds:
                key = f'outer_{fold}_{name}_{seed}'
                expected_keys.add(key)
                record = records[key]
                check_record(record, key, run, seed, train, test)
                graph = read(OUT/f'{key}.json')
                check_diagnostics(record, train, run, graph)
                assert (OUT/f'{key}.md').stat().st_size > 0
                saved = pd.read_csv(OUT/f'{key}.csv')
                assert np.array_equal(saved['index'], test.index) and np.array_equal(saved.Rings, test.Rings)
                prediction = predict_graph(graph, test)
                error = float(np.max(np.abs(prediction-saved.prediction.to_numpy())))
                assert error < 1e-6 and record['export_error'] < 1e-6
                check_scores(record, test.Rings.to_numpy(), prediction)
                assert np.isclose(record['train_rmse'], np.sqrt(np.mean((predict_graph(graph, train)-train.Rings)**2)))
                graph_errors.append(error)
                outer.append(dict(record, fold=fold, variant=name))
                if name == 'reference':
                    previous = ROOT/'results/abalone/pure_rrl'
                    old = read(previous/'trials'/f'{key}.json')
                    assert {k: v for k, v in run.items() if k != 'monitor_rules'} == old['config']
                    assert old['selected_epoch'] == record['selected_epoch']
                    old_prediction = pd.read_csv(previous/f'{key}.csv')
                    assert np.array_equal(old_prediction['index'], saved['index'])
                    old_error = float(np.max(np.abs(old_prediction.prediction-saved.prediction)))
                    assert old_error < 1e-6
                    old_errors.append(old_error)
    assert set(records) == expected_keys, 'Missing or unexpected trial records'
    assert sum(key.startswith('inner_') for key in records) == 360
    search = pd.concat(searches, ignore_index=True)
    ranking = search.groupby('candidate').rmse.mean().sort_values(kind='stable')
    cid = int(ranking.index[0])
    selection = read(OUT/'final_selection.json')
    assert selection['candidate'] == cid and selection['config'] == pool[cid]['config']
    assert selection['seed'] == manifest['delivery_seed']
    assert np.isclose(selection['mean_inner_rmse'], ranking.iloc[0], atol=1e-12, rtol=0)
    final = read(OUT/'final_audit.json')
    check_stopping(final, frame.index.to_numpy(), selection['config'], selection['seed'])
    assert final['pure_rule_model'] and final['independent_test_performance'] is None
    graph = read(OUT/'final_rules.json')
    check_diagnostics(final, frame, selection['config'], graph)
    assert (OUT/'final_rules.md').stat().st_size > 0
    saved = torch.load(OUT/'final_model.pth', map_location='cpu', weights_only=False)
    assert saved['config'] == selection['config'] and graph['preprocessing'] == saved['state']
    head = saved['head']
    assert head['kind'] == 'rules' and not head['linear_weights'] and head['hinge_knots'] is None
    assert not head.get('gates') and not head.get('gate_weights')
    model = RRL(**saved['rrl_args'])
    model.net.load_state_dict(saved['model_state_dict'])
    assert np.array_equal(model.net.layer_list[0].cl.detach().numpy(), np.asarray(graph['thresholds'], dtype='float32'))
    logical = model.net.layer_list[1]
    for name, part in [('conjunction', logical.con_layer), ('disjunction', logical.dis_layer)]:
        assert np.array_equal(part.W.detach().numpy() > .5, graph['layers'][0][name])
    assert np.array_equal(head['weights'], graph['weights']) and head['bias'] == graph['bias']
    reload_error = float(np.max(np.abs(predict_refit(model, saved['state'], frame, head)-predict_graph(graph, frame))))
    assert reload_error < 1e-6 and final['reload_error'] < 1e-6 and final['export_error'] < 1e-6
    result = dict(passed=True, data_sha256=manifest['data_sha256'], verified_sources=manifest['source_sha256'],
        auditor_sha256=digest(Path(__file__)), trial_count=len(records), development_trials=24, inner_trials=360,
        outer_trials=len(outer), outer_selected_candidates=selected, final_candidate=cid, final_seed=selection['seed'],
        max_outer_graph_prediction_error=max(graph_errors), final_reload_graph_error=reload_error,
        max_previous_reference_prediction_error=max(old_errors),
        threshold_checks='Recomputed from every current training subset; outer/final exported graph checked',
        limitation=manifest['limitation'])
    # The report also verifies unchanged baselines before writing audit.json.
    return manifest, records, pd.DataFrame(outer), search, selection, final, result


def report(manifest, records, outer, search, selection, final, audit_result):
    seeds, pool = manifest['seeds'], manifest['candidates']
    rows = []
    for record in outer.to_dict('records'):
        rows.append(dict(model=f"{record['variant']}_{record['seed']}", fold=record['fold'],
                         **{m: record[m] for m in METRICS}))
    baseline_hashes = {}
    frame = load_abalone()
    for fold, (_, te) in enumerate(KFold(**manifest['outer']).split(frame)):
        for seed in seeds:
            path = ROOT/f'results/abalone/pure_rrl/trials/outer_{fold}_selected_{seed}.json'
            prediction_path = path.parent.parent/f'outer_{fold}_selected_{seed}.csv'
            old, prediction = read(path), pd.read_csv(prediction_path)
            assert np.array_equal(prediction['index'], frame.iloc[te].index)
            assert np.array_equal(prediction.Rings, frame.iloc[te].Rings)
            check_scores(old, prediction.Rings.to_numpy(), prediction.prediction.to_numpy())
            baseline_hashes[str(path.relative_to(ROOT))] = digest(path)
            baseline_hashes[str(prediction_path.relative_to(ROOT))] = digest(prediction_path)
            rows.append(dict(model=f'previous_selected_{seed}', fold=fold, **{m: old[m] for m in METRICS}))
    outer_splits = list(KFold(**manifest['outer']).split(frame))
    for name, pattern, prediction_pattern in [
            ('ridge', 'ridge/fold_{}_metrics.json', 'ridge/fold_{}_predictions.csv'),
            ('forest', 'forest/fold_{}_metrics.json', 'forest/fold_{}_predictions.csv'),
            ('expanded_rf', 'round3/rf_metrics_{}.json', 'round3/rf_fold_{}.csv')]:
        for fold in range(5):
            path = OUT.parent/pattern.format(fold)
            baseline = read(path)
            prediction_path = OUT.parent/prediction_pattern.format(fold)
            prediction = pd.read_csv(prediction_path)
            test = frame.iloc[outer_splits[fold][1]]
            assert np.array_equal(prediction['index'], test.index)
            assert np.array_equal(prediction.Rings, test.Rings)
            check_scores(baseline, test.Rings.to_numpy(), prediction.prediction.to_numpy())
            baseline_hashes[str(path.relative_to(ROOT))] = digest(path)
            baseline_hashes[str(prediction_path.relative_to(ROOT))] = digest(prediction_path)
            rows.append(dict(model=name, fold=fold, **{m: baseline[m] for m in METRICS}))
    perfold = pd.DataFrame(rows)
    summary = perfold.groupby('model')[METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(OUT/'summary.csv')
    perfold.to_csv(OUT/'perfold.csv', index=False)
    audit_result['unchanged_baseline_metrics_sha256'] = baseline_hashes
    (OUT/'audit.json').write_text(json.dumps(audit_result, indent=2)+'\n')

    ablations, factorial = [], []
    for fold in range(5):
        for seed in seeds:
            part = outer[(outer.fold == fold) & (outer.seed == seed)].set_index('variant')
            selected = part.loc['selected']
            for variant, changed in part.drop(index='selected').iterrows():
                ablations.append(dict(fold=fold, seed=seed, variant=variant,
                    selected_rmse=selected.rmse, changed_rmse=changed.rmse,
                    rmse_delta=changed.rmse-selected.rmse, mae_delta=changed.mae-selected.mae,
                    logical_edges_delta=changed.diagnostics['logical_edges']-selected.diagnostics['logical_edges']))
            a, b, c, d = [part.loc[k] for k in FIXED]
            factorial.append(dict(fold=fold, seed=seed, reference_rmse=a.rmse, tree_rmse=b.rmse,
                cap_rmse=c.rmse, tree_cap_rmse=d.rmse, tree_effect=b.rmse-a.rmse,
                cap_effect=c.rmse-a.rmse, tree_with_cap_effect=d.rmse-c.rmse,
                cap_with_tree_effect=d.rmse-b.rmse,
                interaction=d.rmse-c.rmse-b.rmse+a.rmse,
                reference_literals=a.diagnostics['mean_literals'], tree_literals=b.diagnostics['mean_literals'],
                cap_literals=c.diagnostics['mean_literals'], tree_cap_literals=d.diagnostics['mean_literals']))
    ablation, factorial = pd.DataFrame(ablations), pd.DataFrame(factorial)
    ablation.to_csv(OUT/'ablation.csv', index=False)
    factorial.to_csv(OUT/'factorial.csv', index=False)
    costs = []
    for stage in ['dev', 'inner', 'outer', 'final']:
        entries = [r for key, r in records.items() if key.startswith(stage+'_')] if stage != 'final' else [final]
        costs.append(dict(stage=stage, records=len(entries), network_fits=sum(1+bool(r['stopping']) for r in entries),
            epochs=sum(r['selected_epoch']+(r['stopping']['epochs_run'] if r['stopping'] else 0) for r in entries),
            seconds=sum(r.get('seconds', 0) for r in entries), seconds_complete=all('seconds' in r for r in entries)))
    cost = pd.DataFrame(costs)
    cost.to_csv(OUT/'cost.csv', index=False)
    ranking = search.groupby('candidate').rmse.mean().sort_values(kind='stable')
    pd.DataFrame(dict(candidate=ranking.index, name=[pool[c]['name'] for c in ranking.index],
                      mean_inner_rmse=ranking.to_numpy())).to_csv(OUT/'inner_ranking.csv', index=False)

    order = [f'{name}_{seed}' for name in ['reference', 'selected'] for seed in seeds]+['ridge', 'forest', 'expanded_rf']
    labels = ['Reference: 314', 'Reference: 2718', 'Selected: 314', 'Selected: 2718', 'Ridge', 'Original RF', 'Expanded RF']
    chinese = ['配对旧配置：314', '配对旧配置：2718', '本轮选参：314', '本轮选参：2718', '普通 Ridge', '原 RF', '扩展搜索 RF']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, metric, title in zip(axes, ['rmse', 'tail_mae'], ['RMSE', 'MAE on Rings >= 20']):
        values = summary.loc[order]
        ax.bar(range(len(order)), values[metric+'_mean'], yerr=values[metric+'_std'], capsize=3,
               color=['#85929e', '#abb2b9', '#2166ac', '#67a9cf', '#ab87ff', '#d17a22', '#edb458'])
        ax.set_xticks(range(len(order)), labels, rotation=45, ha='right')
        ax.set(ylabel='Rings (lower is better)', title=title+'; fold mean +/- SD')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Adaptive internal evaluation; each seed is a separate single-model workflow')
    fig.savefig(OUT/'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
    names = ['Reference', 'Tree cuts', 'Fan-in <= 8', 'Tree + cap']
    colors = ['#777777', '#2166ac', '#d17a22', '#228833']
    for seed, marker in zip(seeds, ['o', '^']):
        for variant, name, color in zip(FIXED, names, colors):
            part = outer[(outer.seed == seed) & (outer.variant == variant)]
            means = [np.mean([d[field] for d in part.diagnostics])
                     for field in ['mean_literals', 'distinct_nonconstant']]
            for ax, x in zip(axes, means):
                ax.scatter(x, part.rmse.mean(), marker=marker, color=color, s=65, label=f'{name}: {seed}')
    for ax, label in zip(axes, ['Mean literals per rule', 'Distinct nonconstant activation columns']):
        ax.set(xlabel=label, ylabel='Mean outer-fold RMSE (lower is better)')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, .5))
    fig.suptitle('Fixed 2 x 2 controls: thresholds x persistent fan-in limit; no outer-score selection')
    fig.savefig(OUT/'mechanism.png', dpi=160)
    plt.close(fig)

    comparisons, conclusions = [], []
    for seed in seeds:
        current = perfold[perfold.model == f'selected_{seed}'].set_index('fold')
        for reference, label in [(f'reference_{seed}', '配对旧配置'), (f'previous_selected_{seed}', '上一轮选参流程'),
                                  ('forest', '原 RF'), ('expanded_rf', '扩展 RF')]:
            base = perfold[perfold.model == reference].set_index('fold')
            delta = current.rmse-base.rmse
            comparisons.append([seed, label, delta.mean(), int((delta < 0).sum())])
        delta_old = summary.loc[f'selected_{seed}', 'rmse_mean']-summary.loc[f'reference_{seed}', 'rmse_mean']
        delta_previous = summary.loc[f'selected_{seed}', 'rmse_mean']-summary.loc[f'previous_selected_{seed}', 'rmse_mean']
        delta_rf = summary.loc[f'selected_{seed}', 'rmse_mean']-summary.loc['expanded_rf', 'rmse_mean']
        conclusions.append(f'种子 {seed} 的 RMSE 为 {summary.loc[f"selected_{seed}", "rmse_mean"]:.5f}，'
                           f'相对配对旧配置 {delta_old:+.5f}，相对上一轮选参流程 {delta_previous:+.5f}，'
                           f'相对扩展 RF {delta_rf:+.5f}。')
    fixed_rows, fixed_conclusions = [], []
    for seed, part in factorial.groupby('seed'):
        for field, label in [('tree_effect', '只换树阈值'), ('cap_effect', '只加 cap8'),
                             ('tree_with_cap_effect', 'cap8 上换树阈值'), ('cap_with_tree_effect', '树阈值上加 cap8')]:
            fixed_rows.append([seed, label, part[field].mean(), int((part[field] < 0).sum())])
        fixed_rows.append([seed, '交互差分：组合−树−cap+参照', part.interaction.mean(), '不作胜率解释'])
    for field, label in [('tree_effect', '递归树阈值'), ('cap_effect', '持续 cap8')]:
        by_seed = factorial.groupby('seed')[field].mean()
        count = int((by_seed < 0).sum())
        fixed_conclusions.append(f'{label}相对参照在 {count}/2 个种子改善平均 RMSE'
                                 f'（314：{by_seed.loc[314]:+.5f}；2718：{by_seed.loc[2718]:+.5f}）。')
    dev_rows, inner_rows = [], []
    for cid, candidate in enumerate(pool):
        trials = [records[f'dev_{cid}_{seed}'] for seed in seeds]
        dev_rows.append([cid, candidate['name'], *[r['rmse'] for r in trials],
                        np.mean([r['train_rmse'] for r in trials]),
                        np.mean([r['diagnostics']['mean_literals'] for r in trials]),
                        np.mean([r['diagnostics']['distinct_nonconstant'] for r in trials]),
                        np.mean([r['diagnostics']['zero_weight_fraction'] for r in trials])])
    for cid, mean in ranking.items():
        by_seed = search[search.candidate == cid].groupby('seed').rmse.mean()
        inner_rows.append([cid, pool[cid]['name'], mean, *[by_seed.loc[s] for s in seeds]])
    choices = [[fold+1, cid, pool[cid]['name'], *[records[f'outer_{fold}_selected_{s}']['selected_epoch'] for s in seeds]]
               for fold, cid in enumerate(audit_result['outer_selected_candidates'])]
    ablation_rows = [[variant, seed, len(part), part.rmse_delta.mean(), int((part.rmse_delta > 0).sum())]
                     for (variant, seed), part in ablation.groupby(['variant', 'seed'])]
    table_order = order[:2]+[f'previous_selected_{seed}' for seed in seeds]+order[2:]
    table_labels = chinese[:2]+['上一轮选参流程：314', '上一轮选参流程：2718']+chinese[2:]
    result_rows = [[label, *[f'{summary.loc[name,m+"_mean"]:.5f} ± {summary.loc[name,m+"_std"]:.5f}' for m in METRICS]]
                   for name, label in zip(table_order, table_labels)]
    mechanism_rows = []
    for seed in seeds:
        for variant in FIXED:
            part = outer[(outer.seed == seed) & (outer.variant == variant)]
            mechanism_rows.append([seed, variant, part.rmse.mean(), part.train_rmse.mean(),
                *[np.mean([d[field] for d in part.diagnostics]) for field in
                  ['mean_literals', 'logical_edges', 'distinct_nonconstant', 'centered_rule_rank', 'zero_weight_fraction']]])
    diagnostic = final['diagnostics']
    fallback = [n for record in records.values() if record['config']['threshold'] == 'tree'
                for n in record['diagnostics']['quantile_fallback_slots']]
    text = f'''# Abalone：文献启发的纯 RRL 优化实验

## 1. 结果先行：是否改善、是否超过基线

{table(['方法', 'RMSE', 'MAE', 'R²', 'Rings≥20 MAE'], result_rows)}

表中是同一外层五折的算术均值 ± 折间样本标准差，不是置信区间。两种子分别形成单模型，不平均预测；共享数据和划分的十条成绩不等于十个独立样本。原 RF、扩展搜索 RF 与普通 Ridge 读取历史评分文件，未随本轮结果改参数。配对旧配置指上一轮 reference，而非上一轮按折选参流程；审计确认其十组预测重现最大差为 {audit_result['max_previous_reference_prediction_error']:.3g}。上一轮选参流程原有的 {summary.loc['previous_selected_314','rmse_mean']:.5f}/{summary.loc['previous_selected_2718','rmse_mean']:.5f} 另列，既看控制变量后的改动效果，也看整套工作流有没有进步。

![预测性能与未改动的基线](../results/abalone/literature_rrl/comparison.png)

{table(['种子', '参照', 'ΔRMSE：本轮−参照', '本轮更好折数/5'], comparisons)}

{''.join(conclusions)}负差表示本轮更好，正差表示退步；不能只报告较好种子、最优折或某个候选的最低分。

**本轮判断：有局部机制收益，但没有形成稳定的跨种子工作流提升。** 相比上一轮，种子314改善约0.42%，种子2718仅降低约0.00007、实质近似持平，并略差于配对固定参照；两个种子在全部外层折上仍弱于原RF和扩展RF。因此不能写成“文献改进已可靠超越旧方法或随机森林”。固定树阈值/cap对照与最终交付选择是两种不同证据，下面分别说明。

## 2. 为什么尝试这些方向

此前的纯 RRL 改善主要来自训练时长控制；短规则初始化结束后规则仍变长，说明“起点简短”不等于训练中持续简短。覆盖较均衡的分位数条件也不意味着切点对 Rings 最有信息。我们据此提出两项待检验假设：更合适的条件边界能让原有 AND/OR 网络表达有效年龄区间；持续控制每个规则的输入数能抑制冗长组合，同时保留足够的有效规则。这是观察驱动的机制假设，不是从文献成绩推断本项目必胜。

[DLN 回归论文](https://arxiv.org/abs/2505.23615)使用回归树初始化、随后可学习的阈值；[数值特征嵌入论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf)也采用目标感知分箱。本轮只借鉴训练内单特征回归树的递归切点，不移植连续分段线性输出，也**没有实现端到端阈值学习**。[TT-Sparse](https://arxiv.org/abs/2603.07606)启发全过程控制输入预算，但本轮不是其可微 Top-K/真值表模型；完整 TT-Sparse 的连续旁路也未引入。[SIRUS 回归](https://proceedings.mlr.press/v130/benard21a.html)提醒我们同时记录覆盖、冗余与复杂度，不能默认短规则一定更准。DLN 自己在 Abalone 上也没有超过 RF，文献提供方法依据而不是胜出保证。

全部预测仍为

$$\\hat y=b+\\sum_j a_jR_j(x),\\qquad R_j(x)\\in\\{{0,1\\}}.$$

保留原 RRL 的 AND/OR、NLAF、Gradient Grafting 与严格离散前向；没有原始连续直连、hinge、外部预测器或模型平均。规则学习沿用 Huber/Adam，之后只在当前训练集合的规则激活上拟合 Ridge 系数。早停也按规则头重拟合后的 RMSE 选轮数。这延续既有训练流程，不是本轮新贡献；Huber 规则训练与最终 Ridge/MSE 并非完全同一优化目标，本轮没有宣称解决这一问题。

## 3. 具体实现与冻结的实验边界

所有候选每个数值特征保留 20 个阈值，防止“切点策略变化”同时变成输入维数变化。`quantile` 是已有参照；`global_supervised` 重跑曾探索过的全局方差下降排序作为失败方向对照；`local_supervised` 只在每个分位数槽位附近小范围寻优；`tree` 用单特征回归树递归最小化局部平方误差，最多 21 叶、最小叶比例 0.02 或 0.05。若树给不满 20 个切点，先均匀抽取剩余分位数补齐，不足才允许重复；因此它是“树切点＋分位数补位”，不能说每个切点都是树监督学到的。所有目标统计只来自当前训练子集，停止验证集和外层折不参加切点估计；切点生成后是固定 buffer，不参与 Adam。

本轮所有树配置 trial 的最终重训中，每个特征实际需要的分位数补位为 {min(fallback)}–{max(fallback)} 个（平均 {np.mean(fallback):.2f}/20），并非忽略不计；逐特征 `tree_split_counts` 与 `quantile_fallback_slots` 保留原始记录，独立审计重新训练相同单特征树核验。

`cap_4/8/16` 在初始化和每个优化步后限制每个 AND/OR 节点至多对应数量的硬连接：超过上限时保留权重最大的边，其余刚越过 0.5 的边压回 0.5 以下的相邻可表示值。它们仍为正且继续有机会获得梯度、重返选中集合，不是永久删除，也不是“必须恰好选 k 条”。这是投影式硬上限，不是 TT-Sparse 的可微 Top-K。边惩罚另用已有 `edge_penalty`，强度为 1e-5/1e-4；容量对照从每种逻辑 64 节点增至 128。没有另造框架或增加依赖。

沿用 log 预处理只是保持旧锚点，不能重新写成 EDA 已证明该变换有效。本轮没有改标签、删除疑点记录或增加目标变换。12 个候选在查看外层结果前冻结，全部进入每折内部 3 折×2 种子的选参，没有以开发表现过滤候选或依据外层成绩扩展搜索。

{table(['ID', '候选', '开发RMSE314', '开发RMSE2718', '训练RMSE均值', '平均条件数', '不同非常量规则', '零权重比例'], dev_rows)}

开发集只作机制诊断，已在历史实验中使用；开发训练/验证合起来是固定外层第 1 折的训练部分，不是完整 4177 条。这里的分数不是正式外层评估。固定训练预算 320 轮、每 10 轮监控、耐心 3 次、最小改善 0.001；每个训练任务先内部留出 20% 选轮数，再以确定轮数重训整个当前训练集合。

边惩罚确实缩短了规则，但开发中不同非常量规则从参照的119.5降至91/57.5，训练与验证误差都升高；全部内部评分中这两档惩罚也未优于参照。这支持“这两档收缩可能过强”的解释，而不是“只要稀疏就更好”。全局监督排序再次表现较差；递归树切点避免了同样的大幅退步，但在全部内部均值上与分位数参照几乎持平，不能把目标相关切点笼统称为已验证更优。

## 4. 内层选择与外层评价分开

{table(['ID', '候选', '全部内层RMSE均值', '种子314内层均值', '种子2718内层均值'], inner_rows)}

内层排名包含 360 条评分，较小训练量与重复的外层训练集合使其不等于外层性能；不可拿最佳内层数字与 RF 外层成绩比较。每一外层折只用该折训练部分的 3×2 内部均值选择一个候选，同分按最小 ID。结果为：

{table(['外层折', '候选ID', '名称', '重训轮数314', '重训轮数2718'], choices)}

外层评价的是“12 候选内部选参”的整个流程，不是某一个固定候选。最终全数据配置则按所有外层训练集合的内层评分均值选出，不使用外层分数挑最终配置；该最终模型没有额外独立测试成绩。

## 5. 固定 2×2 对照：阈值和规则长度是否真的起作用

每个外层折、每个种子都额外运行参照、仅树阈值（0.02）、仅 cap8、树阈值＋cap8 四组，避免只在获选折上观察改动。下表“改动−未改动”为负才表示改善；交互差分衡量二者是否简单相加，不是显著性检验。

{table(['种子', '固定对照', '平均ΔRMSE', '改动更好折数/5'], fixed_rows)}

{''.join(fixed_conclusions)}条件数或边数下降只能证明约束改变了结构，不能单独证明泛化改善；监督阈值的效果也必须由配对误差判断，而非仅看训练误差更低。固定对照用于解释机制，不据其外层结果重新挑选交付方法。

![固定对照的结构与性能](../results/abalone/literature_rrl/mechanism.png)

横轴分别为平均每条规则条件数、训练激活向量不同的非常量规则数；纵轴为五折平均 RMSE，越低越好。两类标记分别代表种子，不合并成集成。“不同激活向量”是训练样本上的经验区别，不证明全局逻辑不等价；中心化规则矩阵秩、Support 和分箱占用另存逐次 diagnostics，不能把更多节点直接当成有效容量。

{table(['种子', '固定配置', '外层RMSE', '训练RMSE', '条件数/规则', '硬边数', '不同非常量规则', '中心化秩', '软权重精确为0比例'], mechanism_rows)}

此表结构统计均为五折训练模型的均值。“软权重精确为 0”与“硬边未选中”不同：后者还包含 (0,0.5] 区间内可再次激活的权重，不能把硬稀疏度当成梯度永久死亡比例。cap 实际条件数、有效规则数、训练误差与外层误差须一起对照：复杂度减少但误差上升是限制过强的可能信号，不是优化成功。

这里实际得到的是有限的正向证据：树＋cap8的五折RMSE为2.18594/2.19522，相对配对参照2.20848/2.20425下降；平均条件数从10.47/11.38降至6.92/7.14，硬边减少约34%/37%，不同非常量规则数量仍接近。这支持在该固定设置中减少冗长连接，而不明显丢失经验规则多样性。但改进不是所有折均成立，也未超过RF；交互差分为正，说明二者组合的误差下降小于各自下降的简单相加，不能宣称额外协同增益。组合外层均值较好不能反过来用于替换按内层协议选出的最终模型。

除固定对照外，获选配置含有阈值改变、cap、边惩罚或扩宽时，还分别撤去相应一项并重新选择停止轮数。下表差值是“对照−选中”，正值才支持保留该项；reference/tree_control/cap_control/tree_cap_control 是固定配置比较，不应混称单项消融。

{table(['对照版本', '种子', '适用折数', 'ΔRMSE：对照−选中', '选中更好折数'], ablation_rows)}

去项对照只适用于含该项的获选折，适用折数不足五折必须保留；它不等于为每个去项方法重新做完整超参搜索。最终配置含某项也不是因果归因：固定对照、去项差值和两个种子的方向应一起阅读。

## 6. 交付、独立核查与成本

最终候选 {selection['candidate']}（{pool[selection['candidate']]['name']}），固定交付种子 {selection['seed']}，全数据重训 {final['selected_epoch']} 轮。配置：`{json.dumps(selection['config'], ensure_ascii=False, sort_keys=True)}`。最终有 {diagnostic['rule_nodes']} 个规则节点、{diagnostic['nonconstant']} 个非常量规则、{diagnostic['distinct_nonconstant']} 个不同非常量激活列，中心化规则矩阵秩 {diagnostic['centered_rule_rank']}；共 {diagnostic['logical_edges']} 条硬连接，平均条件数 {diagnostic['mean_literals']:.3f}，最多 {diagnostic['max_literals']}。保存 [checkpoint](../results/abalone/literature_rrl/final_model.pth)、[全精度规则 JSON](../results/abalone/literature_rrl/final_rules.json) 和 [可读规则](../results/abalone/literature_rrl/final_rules.md)。

这个交付模型仍使用**分位数切点，没有树阈值、cap或边惩罚**；主要配置变化是每种逻辑宽度从64增至128。它是冻结候选内的最终选择，不是证明宽度128对所有折更好，也不能把固定树＋cap实验的复杂度收益归到它身上。新增方法及其成功/失败对照全部保留；本轮不依据已经看到的外层成绩重选模型或继续扩展候选。

[独立审计](../results/abalone/literature_rrl/audit.json)在成功标记写出前核对数据/源码哈希、全部 {audit_result['trial_count']} 个 trial 的候选/种子/索引/停止流程、360 条内部评分与选优、外层和最终训练内重算切点、cap 上限、纯规则输出限制、全部外层指标及规则头训练内重拟合。NumPy 规则图与外层 CSV 的最大预测差 {audit_result['max_outer_graph_prediction_error']:.3g}；最终 checkpoint/JSON 的最大差 {audit_result['final_reload_graph_error']:.3g}。内部任务未保存网络 checkpoint，因此其 RMSE 只核对记录与选参表一致，不虚称独立重算了每个内层预测；外层和最终则确实从完整规则图独立执行。

{table(['阶段', '任务记录', '规则网络训练次数', '累计训练轮数', '记录秒数', '用时完整'], cost.itertuples(index=False, name=None))}

训练次数和轮数包括停止试训及完整训练集合重训；秒数为各任务 elapsed time 之和，包含并行任务的资源竞争，既不是整轮墙钟时间，也不是 CPU 消耗量；不含环境、分析、任务间等待、报告或历史实验成本。`seconds_complete=False` 表示该阶段记录未提供完整用时，不将缺失按真实零秒解释。完整审计、汇总、配对差分、固定因子对照和成本分别在 audit.json、summary.csv、perfold.csv、ablation.csv、factorial.csv、cost.csv；入口为 `python experiments/abalone_literature.py all`，完成后 `python analysis/abalone_literature_report.py` 独立检查并生成本报告。

这仍是反复研究同一份 Abalone 后的**自适应内部评价**。冻结本轮矩阵与严格训练内预处理能避免本轮直接泄漏，不能抹去历史探索的选择偏差；相同五折换种子也不是独立确认。既不承诺 RRL 必须胜过 RF，也不以失败为由隐藏结果。上一轮参见 [纯 RRL 报告](ABALONE-PURE-RRL.md)；混合旁支完整保留，但不纳入本轮纯 RRL 成绩。
'''
    (ROOT/'docs/ABALONE-LITERATURE-RRL.md').write_text(text)
    print(summary.loc[table_order].round(5).to_string())
    print('AUDIT PASSED:', json.dumps(audit_result, ensure_ascii=False))


if __name__ == '__main__':
    torch.set_num_threads(1)
    report(*audit())
