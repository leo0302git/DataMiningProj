"""纯规则 RRL：冻结的小范围机制实验、配对评价和独立规则交付。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.eda import load_abalone
from experiments.explore_abalone import frozen, write_json, networks, all_heads
from experiments.run_abalone import scores
from models.rrl.abalone import fit, prepare, predict, export_rules, RRL
from models.rrl.abalone_refine import representation, refit, predict_refit, graph_with_head, predict_graph

OUT = ROOT / 'results/abalone/pure_rrl'
OLD = ROOT / 'results/abalone/exploration'
SEEDS = [314, 2718]


def candidates():
    base = dict(networks()[9], paired_init=True, init_literals=0, full_categories=False,
                rule_head_alpha=.1, stop_patience=3, monitor_every=10, min_delta=.001)
    changes = [('reference', {}), ('full_sex', dict(full_categories=True)),
               ('short_1', dict(init_literals=1)), ('short_2', dict(init_literals=2)),
               ('short_4', dict(init_literals=4)),
               ('full_sex_short_2', dict(full_categories=True, init_literals=2)),
               ('fixed_320', dict(stop_patience=0)),
               ('head_alpha_1', dict(rule_head_alpha=1.)),
               ('short_2_head_alpha_1', dict(init_literals=2, rule_head_alpha=1.))]
    return [dict(name=name, config=dict(base, **change)) for name, change in changes]


def design():
    old = json.loads((OLD/'design.json').read_text())
    paths = ['models/rrl/abalone.py', 'models/rrl/abalone_refine.py',
             'models/rrl/rrl/components.py', 'models/rrl/rrl/models.py',
             'models/manual_glm/ridge.py', 'experiments/explore_abalone.py',
             'experiments/abalone_pure_rrl.py']
    manifest = dict(candidates=candidates(), seeds=SEEDS, delivery_seed=SEEDS[0],
        train_indices=old['train_indices'], validation_indices=old['validation_indices'],
        data_sha256=hashlib.sha256((ROOT/'data/raw/abalone/abalone.data').read_bytes()).hexdigest(),
        outer=dict(n_splits=5, shuffle=True, random_state=0),
        inner=dict(n_splits=3, shuffle=True, random_state='100+outer_fold'),
        stopping='20% of current training subset, seed=training_seed+900; refit rules head on remaining 80%; select epoch, retrain full current subset',
        shortlist='none: all 9 frozen candidates enter each outer-training inner CV; development is diagnostic only',
        revision='Before any new outer evaluation: remove global-development screening; preserve initial design for audit',
        selection='mean inner RMSE across 3 folds and both seeds; outer labels never select configs',
        ablations='selected config minus full_sex, short_init, head_alpha_1 where applicable; fixed320; paired reference',
        final='best mean inner RMSE across all outer training folds, single seed314; never average predictions',
        limitation='adaptive internal evaluation on repeatedly studied data, not independent confirmation',
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    frozen(OUT/'design.json', manifest)
    return manifest


def rule_diagnostics(model, state, train):
    X, _ = prepare(train, state)
    with torch.no_grad():
        binary = model.net.layer_list[0].binarized_forward(torch.from_numpy(X)).numpy().astype(bool)
    _, rules = representation(model, state, train)
    layer = model.net.layer_list[1]
    con = layer.con_layer.W.detach().numpy() > .5
    dis = layer.dis_layer.W.detach().numpy() > .5
    lengths = np.r_[con.sum(0), dis.sum(0)]
    support = rules.mean(0)
    nonconstant = (support > 0) & (support < 1)
    result = dict(rule_nodes=int(len(support)), nonconstant=int(nonconstant.sum()),
        constant_false=int((support == 0).sum()), constant_true=int((support == 1).sum()),
        distinct_nonconstant=int(np.unique(rules[:, nonconstant], axis=1).shape[1]),
        duplicate_activation_surplus=int(rules.shape[1]-np.unique(rules, axis=1).shape[1]),
        low_support=int(((support > 0) & (support < .01)).sum()),
        logical_edges=int(lengths.sum()), mean_literals=float(lengths.mean()),
        median_literals=float(np.median(lengths)), empty_rules=int((lengths == 0).sum()),
        condition_nodes=binary.shape[1], extreme_conditions=int(((binary.mean(0)<.01)|(binary.mean(0)>.99)).sum()),
        support=support.tolist(), literals=lengths.tolist())
    if 'initial_gates' in state:
        initial = state['initial_gates']
        masks = []
        for kind, current in [('conjunction', con), ('disjunction', dis)]:
            mask = np.zeros_like(current)
            for j, ids in enumerate(initial[kind]):
                mask[ids, j] = True
            masks.append(mask)
        initial_rules = np.column_stack([(~binary).astype(int) @ masks[0] == 0,
                                         binary.astype(int) @ masks[1] > 0])
        initial_support = initial_rules.mean(0)
        result.update(initial_nonconstant=int(((initial_support>0)&(initial_support<1)).sum()),
            initial_support=initial_support.tolist(),
            net_gate_changes=int((masks[0] != con).sum()+(masks[1] != dis).sum()),
            net_gate_change_fraction=float(((masks[0] != con).sum()+(masks[1] != dis).sum())/(con.size+dis.size)))
    return result


def train_model(train, config, seed):
    """Stopping subset is entirely inside train, never the evaluation frame."""
    run = config.copy()
    history = []
    stopping = None
    if run.get('stop_patience'):
        proper, stop = train_test_split(np.arange(len(train)), test_size=.2, random_state=seed+900)
        _, state, _ = fit(train.iloc[proper], run, seed, train.iloc[stop], history)
        run['epochs'] = state['best_epoch']
        stopping = dict(train_indices=train.iloc[proper].index.tolist(),
                        validation_indices=train.iloc[stop].index.tolist(),
                        epochs_run=state['epochs_run'], history=history)
    run.pop('stop_patience', None)
    model, state, args = fit(train, run, seed)
    head = refit(model, state, train, 'rules', config['rule_head_alpha'])
    assert not head['linear_weights'] and head['hinge_knots'] is None
    return model, state, args, head, dict(selected_epoch=run['epochs'], stopping=stopping)


def save_graph(model, state, head, train, test, prefix):
    graph = graph_with_head(export_rules(model, state, train, prefix), head)
    assert not graph.get('linear_weights') and not graph.get('hinge_knots') and not graph.get('gates') and 'members' not in graph
    pred = predict_refit(model, state, test, head)
    error = float(np.max(np.abs(pred-predict_graph(graph, test))))
    assert error < 1e-6, error
    write_json(prefix.with_suffix('.json'), graph)
    note = prefix.with_suffix('.md').read_text().split('\nRings =')[0]
    note += f'\n\nRings = {head["bias"]:.12g} + sum(a_j * R_j)\n\n'
    note += '输出只有严格布尔规则，无连续直连、hinge或模型平均。规则固定后在当前训练集重拟合Ridge头；JSON保存全精度。\n\n'
    note += '\n'.join(f'L1R{j}: {w:.12g} Rings' for j, w in enumerate(head['weights']))+'\n'
    prefix.with_suffix('.md').write_text(note)
    return pred, error


def trial(train, test, config, seed, key, export=False):
    path = OUT/'trials'/f'{key}.json'
    if path.exists():
        record = json.loads(path.read_text())
        assert record['config'] == config and record['seed'] == seed
        assert record['train_indices'] == train.index.tolist() and record['evaluation_indices'] == test.index.tolist()
        if export:
            assert all((OUT/f'{key}{suffix}').exists() for suffix in ['.json', '.csv', '.md'])
        return record
    start = time.monotonic()
    model, state, args, head, trace = train_model(train, config, seed)
    pred = predict_refit(model, state, test, head)
    record = dict(key=key, seed=seed, config=config, train_indices=train.index.tolist(),
        evaluation_indices=test.index.tolist(), **scores(test.Rings.to_numpy(), pred), **trace,
        train_rmse=scores(train.Rings.to_numpy(), predict_refit(model, state, train, head))['rmse'],
        adam_rmse=scores(test.Rings.to_numpy(), predict(model, state, test))['rmse'],
        diagnostics=rule_diagnostics(model, state, train))
    if export:
        pred, record['export_error'] = save_graph(model, state, head, train, test, OUT/key)
        pd.DataFrame(dict(index=test.index, Rings=test.Rings, prediction=pred)).to_csv(OUT/f'{key}.csv', index=False)
    record['seconds'] = time.monotonic()-start
    write_json(path, record)
    print(key, 'RMSE', round(record['rmse'], 5), 'epoch', trace['selected_epoch'],
          'seconds', round(record['seconds'], 1), flush=True)
    return record


def reproduce(frame):
    selected = json.loads((OLD/'shortlist.json').read_text())
    results = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(frame)):
        path = OUT/f'reproduction_{fold}.json'
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        old = json.loads((OLD/f'pure_metrics_{fold}.json').read_text())
        spec = selected[old['candidate']]
        model, state, _ = fit(frame.iloc[tr], networks()[spec['network']], 42+fold)
        head = refit(model, state, frame.iloc[tr], **all_heads()[spec['head']])
        pred = predict_refit(model, state, frame.iloc[te], head)
        saved = pd.read_csv(OLD/f'pure_fold_{fold}.csv')
        assert np.array_equal(saved['index'], frame.iloc[te].index)
        error = float(np.max(np.abs(pred-saved.prediction.to_numpy())))
        assert error < 1e-5, error
        row = dict(fold=fold, prediction_error=error, **scores(frame.iloc[te].Rings.to_numpy(), pred),
                   diagnostics=rule_diagnostics(model, state, frame.iloc[tr]))
        write_json(path, row)
        results.append(row)
        print('REPRODUCE', fold, error, flush=True)
    return results


def develop(frame):
    manifest = design()
    train, val = frame.iloc[manifest['train_indices']], frame.iloc[manifest['validation_indices']]
    records = []
    for cid, candidate in enumerate(candidates()):
        for seed in SEEDS:
            r = trial(train, val, candidate['config'], seed, f'dev_{cid}_{seed}')
            records.append(dict(candidate=cid, name=candidate['name'], seed=seed,
                **{k:r[k] for k in ['rmse','mae','train_rmse','adam_rmse','selected_epoch','seconds']},
                **{k:r['diagnostics'][k] for k in ['nonconstant','distinct_nonconstant','logical_edges','mean_literals','initial_nonconstant']}))
    table = pd.DataFrame(records)
    table.to_csv(OUT/'development.csv', index=False)
    ranking = table.groupby(['candidate','name']).agg(rmse=('rmse','mean'), seed_std=('rmse','std'),
        train_rmse=('train_rmse','mean'), mean_literals=('mean_literals','mean'),
        nonconstant=('nonconstant','mean'), epoch=('selected_epoch','mean')).reset_index().sort_values(['rmse','candidate'])
    ranking.to_csv(OUT/'development_ranking.csv', index=False)
    selected = sorted(set([0]+ranking.head(3).candidate.astype(int).tolist()))
    frozen(OUT/'diagnostic_ranking_ids.json', selected)
    return selected


def evaluate(frame, folds=range(5)):
    selected = list(range(len(candidates())))
    pool = candidates()
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(frame)):
        if fold not in folds:
            continue
        train, test = frame.iloc[tr], frame.iloc[te]
        rows = []
        for k, (a, b) in enumerate(KFold(3, shuffle=True, random_state=100+fold).split(train)):
            for cid in selected:
                for seed in SEEDS:
                    r = trial(train.iloc[a], train.iloc[b], pool[cid]['config'], seed, f'inner_{fold}_{k}_{cid}_{seed}')
                    rows.append(dict(candidate=cid, inner_fold=k, seed=seed, rmse=r['rmse']))
        table = pd.DataFrame(rows)
        table.to_csv(OUT/f'inner_{fold}.csv', index=False)
        cid = int(table.groupby('candidate').rmse.mean().sort_values(kind='stable').index[0])
        frozen(OUT/f'selection_{fold}.json', dict(candidate=cid, config=pool[cid]['config']))
        config = pool[cid]['config']
        variants = dict(selected=config, reference=pool[0]['config'], no_earlystop=dict(config, stop_patience=0))
        if config['full_categories']:
            variants['drop_first'] = dict(config, full_categories=False)
        if config['init_literals']:
            variants['empty_init'] = dict(config, init_literals=0)
        if config['rule_head_alpha'] != .1:
            variants['head_alpha_01'] = dict(config, rule_head_alpha=.1)
        for seed in SEEDS:
            for name, run in variants.items():
                trial(train, test, run, seed, f'outer_{fold}_{name}_{seed}', export=True)


def final(frame):
    table = pd.concat([pd.read_csv(OUT/f'inner_{fold}.csv') for fold in range(5)])
    ranking = table.groupby('candidate').rmse.mean().sort_values(kind='stable')
    cid = int(ranking.index[0])
    config = candidates()[cid]['config']
    frozen(OUT/'final_selection.json', dict(candidate=cid, config=config, seed=SEEDS[0], mean_inner_rmse=float(ranking.iloc[0])))
    if (OUT/'final_audit.json').exists():
        assert all((OUT/name).exists() for name in ['final_model.pth', 'final_rules.json', 'final_rules.md'])
        return
    model, state, args, head, trace = train_model(frame, config, SEEDS[0])
    pred, error = save_graph(model, state, head, frame, frame, OUT/'final_rules')
    torch.save(dict(model_state_dict=model.net.state_dict(), state=state, rrl_args=args,
                    config=config, head=head), OUT/'final_model.pth')
    saved = torch.load(OUT/'final_model.pth', weights_only=False)
    loaded = RRL(**saved['rrl_args'])
    loaded.net.load_state_dict(saved['model_state_dict'])
    reload_error = float(np.max(np.abs(pred-predict_refit(loaded, saved['state'], frame, saved['head']))))
    assert reload_error < 1e-6
    write_json(OUT/'final_audit.json', dict(**trace, export_error=error, reload_error=reload_error,
               diagnostics=rule_diagnostics(model, state, frame),
               pure_rule_model=True, independent_test_performance=None))
    print('FINAL', cid, 'epoch', trace['selected_epoch'], 'export/reload', error, reload_error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['reproduce','develop','evaluate','final','all'])
    parser.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    options = parser.parse_args()
    stage = options.stage
    torch.set_num_threads(1)
    (OUT/'trials').mkdir(parents=True, exist_ok=True)
    design()
    data = load_abalone()
    for name, action in [('reproduce', reproduce), ('develop', develop), ('evaluate', evaluate), ('final', final)]:
        if stage in (name, 'all'):
            action(data, options.folds) if name == 'evaluate' else action(data)
