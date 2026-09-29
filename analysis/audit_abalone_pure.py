"""独立审计纯规则实验产物；不训练，只在全部检查通过后写 audit.json。"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.eda import load_abalone
from experiments.run_abalone import scores
from models.rrl.abalone import RRL, prepare
from models.rrl.abalone_refine import predict_graph, predict_refit

OUT = ROOT/'results/abalone/pure_rrl'


def read(path):
    return json.loads(path.read_text())


def check_scores(record, target, prediction):
    for key, value in scores(target, prediction).items():
        assert record[key] is None if value is None else np.isclose(record[key], value, atol=1e-9, rtol=1e-9), key


def check_stopping(record, train_indices, config, seed):
    trace = record['stopping']
    if not config.get('stop_patience'):
        assert trace is None and record['selected_epoch'] == config['epochs']
        return
    a, b = train_test_split(np.arange(len(train_indices)), test_size=.2, random_state=seed+900)
    assert trace['train_indices'] == train_indices[a].tolist()
    assert trace['validation_indices'] == train_indices[b].tolist()
    assert set(trace['train_indices']).isdisjoint(trace['validation_indices'])
    assert set(trace['train_indices']) | set(trace['validation_indices']) == set(train_indices)
    history = trace['history']
    assert history and all(np.isfinite(row['validation_rmse']) for row in history)
    best, best_epoch, stale, previous = float('inf'), None, 0, 0
    for row in history:
        epoch = row['epoch']
        assert previous < epoch <= config['epochs']
        assert epoch % config['monitor_every'] == 0 or epoch == config['epochs']
        assert stale < config['stop_patience'], 'Training continued after its recorded stop criterion'
        if row['validation_rmse'] < best-config['min_delta']:
            best, best_epoch, stale = row['validation_rmse'], epoch, 0
        else:
            stale += 1
        previous = epoch
    assert record['selected_epoch'] == best_epoch
    assert trace['epochs_run'] == previous
    assert previous == config['epochs'] or stale >= config['stop_patience']


def check_record(key, config, seed, train, evaluation):
    record = read(OUT/'trials'/f'{key}.json')
    assert record['key'] == key and record['config'] == config and record['seed'] == seed
    assert record['train_indices'] == train.index.tolist()
    assert record['evaluation_indices'] == evaluation.index.tolist()
    assert set(record['train_indices']).isdisjoint(record['evaluation_indices'])
    assert np.isfinite(record['rmse'])
    check_stopping(record, train.index.to_numpy(), config, seed)
    return record


def check_graph(graph, train, config):
    assert graph['head_kind'] == 'rules'
    assert not graph.get('linear_weights') and not graph.get('hinge_knots')
    assert not graph.get('gates') and not graph.get('gate_weights') and 'members' not in graph
    assert graph['head_alpha'] == config['rule_head_alpha']
    _, expected = prepare(train, transform=config['transform'], full_categories=config['full_categories'])
    state = graph['preprocessing']
    for key in ['features', 'categories', 'transform']:
        assert state[key] == expected[key]
    assert state.get('full_categories', False) == config['full_categories']
    for key in ['mean', 'std']:
        assert np.allclose(state[key], expected[key], rtol=0, atol=1e-12)
    assert np.isclose(state['y_mean'], train.Rings.to_numpy().mean(), atol=1e-12)
    assert np.isclose(state['y_std'], train.Rings.to_numpy().std(), atol=1e-12)
    assert len(graph['layers']) == 1
    for layer in graph['layers']:
        for kind in ['conjunction', 'disjunction']:
            assert np.isin(layer[kind], [0, 1]).all()


def main():
    torch.set_num_threads(1)
    manifest = read(OUT/'design.json')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(ROOT/'data/raw/abalone/abalone.data') == manifest['data_sha256']
    for path, expected in manifest['source_sha256'].items():
        assert digest(ROOT/path) == expected, f'Source changed: {path}'
    initial = read(OUT/'design_initial.json')
    assert initial['candidates'] == manifest['candidates']
    assert initial['train_indices'] == manifest['train_indices']
    assert initial['validation_indices'] == manifest['validation_indices']
    pool, seeds = manifest['candidates'], manifest['seeds']
    assert len(pool) == 9 and len(set(seeds)) == 2
    assert manifest['delivery_seed'] == seeds[0] == 314
    frame = load_abalone()
    outer = list(KFold(5, shuffle=True, random_state=0).split(frame))
    inner_keys, tables, selected, graph_errors, reproduction_errors = set(), [], [], [], []
    outer_count = 0
    for fold, (tr, te) in enumerate(outer):
        train, test = frame.iloc[tr], frame.iloc[te]
        table = pd.read_csv(OUT/f'inner_{fold}.csv')
        expected_rows = set()
        for inner, (a, b) in enumerate(KFold(3, shuffle=True, random_state=100+fold).split(train)):
            for cid, candidate in enumerate(pool):
                for seed in seeds:
                    key = f'inner_{fold}_{inner}_{cid}_{seed}'
                    inner_keys.add(key)
                    record = check_record(key, candidate['config'], seed, train.iloc[a], train.iloc[b])
                    row = table[(table.candidate == cid) & (table.inner_fold == inner) & (table.seed == seed)]
                    assert len(row) == 1 and np.isclose(row.iloc[0].rmse, record['rmse'], rtol=0, atol=1e-12)
                    expected_rows.add((cid, inner, seed))
        assert len(table) == 54 and set(zip(table.candidate, table.inner_fold, table.seed)) == expected_rows
        tables.append(table)
        cid = int(table.groupby('candidate').rmse.mean().sort_values(kind='stable').index[0])
        selection = read(OUT/f'selection_{fold}.json')
        assert selection == dict(candidate=cid, config=pool[cid]['config'])
        selected.append(cid)
        config = pool[cid]['config']
        variants = dict(selected=config, reference=pool[0]['config'], no_earlystop=dict(config, stop_patience=0))
        if config['full_categories']:
            variants['drop_first'] = dict(config, full_categories=False)
        if config['init_literals']:
            variants['empty_init'] = dict(config, init_literals=0)
        if config['rule_head_alpha'] != .1:
            variants['head_alpha_01'] = dict(config, rule_head_alpha=.1)
        for name, run in variants.items():
            for seed in seeds:
                key = f'outer_{fold}_{name}_{seed}'
                record = check_record(key, run, seed, train, test)
                graph = read(OUT/f'{key}.json')
                check_graph(graph, train, run)
                saved = pd.read_csv(OUT/f'{key}.csv')
                assert np.array_equal(saved['index'], test.index)
                assert np.array_equal(saved.Rings, test.Rings)
                prediction = predict_graph(graph, test)
                error = float(np.max(np.abs(prediction-saved.prediction.to_numpy())))
                assert error < 1e-6 and record['export_error'] < 1e-6
                check_scores(record, test.Rings.to_numpy(), prediction)
                graph_errors.append(error)
                outer_count += 1
        reproduced = read(OUT/f'reproduction_{fold}.json')
        assert reproduced['fold'] == fold and reproduced['prediction_error'] < 1e-5
        historical = pd.read_csv(ROOT/f'results/abalone/exploration/pure_fold_{fold}.csv')
        assert np.array_equal(historical['index'], test.index)
        check_scores(reproduced, test.Rings.to_numpy(), historical.prediction.to_numpy())
        reproduction_errors.append(reproduced['prediction_error'])
    assert len(inner_keys) == 270
    assert {p.stem for p in (OUT/'trials').glob('inner_*.json')} == inner_keys
    ranking = pd.concat(tables).groupby('candidate').rmse.mean().sort_values(kind='stable')
    cid = int(ranking.index[0])
    selection = read(OUT/'final_selection.json')
    assert selection['candidate'] == cid and selection['config'] == pool[cid]['config']
    assert selection['seed'] == manifest['delivery_seed']
    assert np.isclose(selection['mean_inner_rmse'], ranking.iloc[0], rtol=0, atol=1e-12)
    audit = read(OUT/'final_audit.json')
    check_stopping(audit, frame.index.to_numpy(), selection['config'], selection['seed'])
    assert audit['pure_rule_model'] and audit['independent_test_performance'] is None
    saved = torch.load(OUT/'final_model.pth', map_location='cpu', weights_only=False)
    assert saved['config'] == selection['config']
    head = saved['head']
    assert head['kind'] == 'rules' and not head['linear_weights'] and head['hinge_knots'] is None
    assert not head.get('gates') and not head.get('gate_weights')
    model = RRL(**saved['rrl_args'])
    model.net.load_state_dict(saved['model_state_dict'])
    graph = read(OUT/'final_rules.json')
    check_graph(graph, frame, selection['config'])
    assert graph['preprocessing'] == saved['state']
    error = float(np.max(np.abs(predict_refit(model, saved['state'], frame, head)-predict_graph(graph, frame))))
    assert error < 1e-6 and audit['export_error'] < 1e-6 and audit['reload_error'] < 1e-6
    report = dict(passed=True, data_sha256=manifest['data_sha256'], verified_sources=manifest['source_sha256'],
        inner_trials=len(inner_keys), outer_trials=outer_count, outer_selected_candidates=selected,
        final_candidate=cid, final_seed=selection['seed'], final_mean_inner_rmse=float(ranking.iloc[0]),
        max_outer_graph_prediction_error=max(graph_errors), final_reload_graph_error=error,
        historical_reported_max_reproduction_error=max(reproduction_errors),
        limitation=manifest['limitation'])
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
