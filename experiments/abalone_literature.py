"""文献启发的纯规则实验；固定候选，复用已有训练与规则交付。"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.eda import load_abalone
from experiments.abalone_pure_rrl import candidates as previous_candidates
from experiments.abalone_pure_rrl import train_model, save_graph, rule_diagnostics
from experiments.explore_abalone import frozen, write_json
from experiments.run_abalone import scores
from models.rrl.abalone import RRL, prepare, predict
from models.rrl.abalone_refine import representation, predict_refit

OUT = ROOT/'results/abalone/literature_rrl'
SEEDS = [314, 2718]


def candidates():
    base = dict(previous_candidates()[0]['config'], monitor_rules=True)
    changes = [
        ('reference', {}),
        ('local_supervised', dict(threshold='local_supervised')),
        ('tree_leaf_02', dict(threshold='tree', tree_min_leaf=.02)),
        ('tree_leaf_05', dict(threshold='tree', tree_min_leaf=.05)),
        ('global_supervised', dict(threshold='supervised')),
        ('edge_1e5', dict(edge_lambda=1e-5)),
        ('edge_1e4', dict(edge_lambda=1e-4)),
        ('cap_4', dict(max_fanin=4)),
        ('cap_8', dict(max_fanin=8)),
        ('cap_16', dict(max_fanin=16)),
        ('tree_cap_8', dict(threshold='tree', tree_min_leaf=.02, max_fanin=8)),
        ('wide_128', dict(structure='20@128')),
    ]
    return [dict(name=name, config=dict(base, **change)) for name, change in changes]


def design():
    old = json.loads((ROOT/'results/abalone/pure_rrl/design.json').read_text())
    paths = ['models/rrl/abalone.py', 'models/rrl/abalone_refine.py',
             'models/rrl/rrl/components.py', 'models/rrl/rrl/models.py',
             'models/manual_glm/ridge.py', 'analysis/eda.py',
             'experiments/run_abalone.py', 'experiments/explore_abalone.py',
             'experiments/abalone_pure_rrl.py', 'experiments/abalone_literature.py']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(candidates=candidates(), seeds=SEEDS, delivery_seed=314,
        historical_commit='9f85fac', historical_design_sha256=digest(ROOT/'results/abalone/pure_rrl/design.json'),
        data_sha256=digest(ROOT/'data/raw/abalone/abalone.data'),
        train_indices=old['train_indices'], validation_indices=old['validation_indices'],
        outer=dict(n_splits=5, shuffle=True, random_state=0),
        inner=dict(n_splits=3, shuffle=True, random_state='100+outer_fold'),
        stopping=old['stopping'],
        selection='All 12 candidates enter every outer-training inner3 x seeds314/2718; mean RMSE, then lowest candidate ID; development only diagnostic',
        ablations='fixed 2x2 reference/tree_leaf_02/cap_8/tree_cap_8 every outer fold; selected config additionally minus threshold/cap/edge/widening when applicable; epochs reselected inside train',
        final='lowest mean inner RMSE across all 5 outer-training folds; seed314 single model; never average predictions',
        limits='12 frozen candidates; no adaptive expansion after outer evaluation; no learned thresholds or continuous bypass this round',
        limitation='Repeatedly studied Abalone; adaptive internal evaluation, not independent confirmation',
        source_sha256={p: digest(ROOT/p) for p in paths},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, sklearn=sklearn.__version__, torch=torch.__version__))
    (OUT/'trials').mkdir(parents=True, exist_ok=True)
    frozen(OUT/'design.json', manifest)
    return manifest


def diagnostics(model, state, train, config):
    result = rule_diagnostics(model, state, train)
    X, _ = prepare(train, state)
    _, rules = representation(model, state, train)
    cuts = model.net.layer_list[0].cl.detach().numpy()
    continuous = X[:, len(state['categories'])-1:]
    occupancies = []
    for j, col in enumerate(continuous.T):
        boundaries = np.unique(cuts[:, j])
        bins = np.searchsorted(boundaries, col, side='left')
        counts = np.bincount(bins, minlength=len(boundaries)+1)
        occupancies.append(counts.tolist())
    result.update(max_literals=int(max(result['literals'])),
        centered_rule_rank=int(np.linalg.matrix_rank(rules-rules.mean(0))),
        zero_weight_fraction=float(np.mean(np.concatenate([
            part.W.detach().numpy().ravel() == 0
            for layer in model.net.layer_list[1:-1] for part in [layer.con_layer, layer.dis_layer]]))),
        unique_cuts=[int(len(np.unique(cuts[:, j]))) for j in range(cuts.shape[1])],
        bin_counts=occupancies,
        thresholds=cuts.tolist())
    if config['threshold'] == 'tree':
        trees = [DecisionTreeRegressor(max_leaf_nodes=len(cuts)+1,
                 min_samples_leaf=config.get('tree_min_leaf', .02), random_state=0)
                 .fit(continuous[:, j:j+1], train.Rings.to_numpy(float))
                 for j in range(continuous.shape[1])]
        result['tree_split_counts'] = [int((t.tree_.children_left != -1).sum()) for t in trees]
        result['quantile_fallback_slots'] = [len(cuts)-n for n in result['tree_split_counts']]
    return result


def variants(config):
    pool = candidates()
    result = dict(selected=config, reference=pool[0]['config'], tree_control=pool[2]['config'],
                  cap_control=pool[8]['config'], tree_cap_control=pool[10]['config'])
    if config['threshold'] != 'quantile':
        result['no_threshold_change'] = {k:v for k,v in dict(config, threshold='quantile').items() if k != 'tree_min_leaf'}
    for key, label in [('max_fanin', 'no_cap'), ('edge_lambda', 'no_edge_penalty')]:
        if config.get(key):
            result[label] = {k:v for k,v in config.items() if k != key}
    if config['structure'] != '20@64':
        result['no_widening'] = dict(config, structure='20@64')
    return result


def trial(train, test, config, seed, key, export=False):
    path = OUT/'trials'/f'{key}.json'
    if path.exists():
        record = json.loads(path.read_text())
        assert record['config'] == config and record['seed'] == seed
        assert record['train_indices'] == train.index.tolist()
        assert record['evaluation_indices'] == test.index.tolist()
        if export:
            assert all((OUT/f'{key}{suffix}').exists() for suffix in ['.json', '.csv', '.md'])
        return record
    start = time.monotonic()
    model, state, _, head, trace = train_model(train, config, seed)
    pred = predict_refit(model, state, test, head)
    diag = diagnostics(model, state, train, config)
    if config.get('max_fanin'):
        assert diag['max_literals'] <= config['max_fanin']
        assert all(row['max_fanin_observed'] <= config['max_fanin'] for row in trace['stopping']['history'])
    record = dict(key=key, seed=seed, config=config, train_indices=train.index.tolist(),
        evaluation_indices=test.index.tolist(), **scores(test.Rings.to_numpy(), pred), **trace,
        train_rmse=scores(train.Rings.to_numpy(), predict_refit(model, state, train, head))['rmse'],
        adam_rmse=scores(test.Rings.to_numpy(), predict(model, state, test))['rmse'], diagnostics=diag)
    if export:
        pred, record['export_error'] = save_graph(model, state, head, train, test, OUT/key)
        pd.DataFrame(dict(index=test.index, Rings=test.Rings, prediction=pred)).to_csv(OUT/f'{key}.csv', index=False)
    record['seconds'] = time.monotonic()-start
    write_json(path, record)
    print(key, 'RMSE', round(record['rmse'], 5), 'epoch', trace['selected_epoch'],
          'max_literals', diag['max_literals'], 'seconds', round(record['seconds'], 1), flush=True)
    return record


def develop(frame):
    manifest = design()
    train, val = frame.iloc[manifest['train_indices']], frame.iloc[manifest['validation_indices']]
    rows = []
    for cid, candidate in enumerate(candidates()):
        for seed in SEEDS:
            r = trial(train, val, candidate['config'], seed, f'dev_{cid}_{seed}')
            rows.append(dict(candidate=cid, name=candidate['name'], seed=seed,
                **{k:r[k] for k in ['rmse','mae','train_rmse','adam_rmse','selected_epoch','seconds']},
                **{k:r['diagnostics'][k] for k in ['distinct_nonconstant','logical_edges','mean_literals','max_literals','centered_rule_rank']}))
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'development.csv', index=False)
    table.groupby(['candidate','name']).mean(numeric_only=True).reset_index().sort_values(['rmse','candidate']).to_csv(OUT/'development_ranking.csv', index=False)


def evaluate(frame, folds=range(5)):
    pool = candidates()
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(frame)):
        if fold not in folds:
            continue
        train, test = frame.iloc[tr], frame.iloc[te]
        rows = []
        for inner, (a, b) in enumerate(KFold(3, shuffle=True, random_state=100+fold).split(train)):
            for cid, candidate in enumerate(pool):
                for seed in SEEDS:
                    r = trial(train.iloc[a], train.iloc[b], candidate['config'], seed, f'inner_{fold}_{inner}_{cid}_{seed}')
                    rows.append(dict(candidate=cid, inner_fold=inner, seed=seed, rmse=r['rmse']))
        table = pd.DataFrame(rows)
        table.to_csv(OUT/f'inner_{fold}.csv', index=False)
        cid = int(table.groupby('candidate').rmse.mean().sort_values(kind='stable').index[0])
        config = pool[cid]['config']
        frozen(OUT/f'selection_{fold}.json', dict(candidate=cid, config=config))
        for seed in SEEDS:
            for name, run in variants(config).items():
                trial(train, test, run, seed, f'outer_{fold}_{name}_{seed}', export=True)


def final(frame):
    table = pd.concat([pd.read_csv(OUT/f'inner_{fold}.csv') for fold in range(5)])
    ranking = table.groupby('candidate').rmse.mean().sort_values(kind='stable')
    cid = int(ranking.index[0])
    config = candidates()[cid]['config']
    frozen(OUT/'final_selection.json', dict(candidate=cid, config=config, seed=314, mean_inner_rmse=float(ranking.iloc[0])))
    if (OUT/'final_audit.json').exists():
        assert all((OUT/name).exists() for name in ['final_model.pth','final_rules.json','final_rules.md'])
        return
    start = time.monotonic()
    model, state, args, head, trace = train_model(frame, config, 314)
    pred, error = save_graph(model, state, head, frame, frame, OUT/'final_rules')
    torch.save(dict(model_state_dict=model.net.state_dict(), state=state, rrl_args=args,
                    config=config, head=head), OUT/'final_model.pth')
    saved = torch.load(OUT/'final_model.pth', weights_only=False)
    restored = RRL(**saved['rrl_args'])
    restored.net.load_state_dict(saved['model_state_dict'])
    reload_error = float(np.max(np.abs(pred-predict_refit(restored, saved['state'], frame, saved['head']))))
    assert reload_error < 1e-6
    write_json(OUT/'final_audit.json', dict(**trace, export_error=error, reload_error=reload_error,
        diagnostics=diagnostics(model, state, frame, config), seconds=time.monotonic()-start,
        pure_rule_model=True, independent_test_performance=None))
    print('FINAL', cid, 'epoch', trace['selected_epoch'], 'export/reload', error, reload_error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze','develop','evaluate','final','all'])
    parser.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    options = parser.parse_args()
    torch.set_num_threads(1)
    design()  # Verify frozen source/data/config BEFORE trusting any cached trial.
    frame = load_abalone()
    for stage, action in [('develop', develop), ('evaluate', evaluate), ('final', final)]:
        if options.stage in (stage, 'all'):
            action(frame, options.folds) if stage == 'evaluate' else action(frame)
