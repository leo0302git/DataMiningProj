"""Small regression adapter using the unchanged upstream logical network."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rrl.models import RRL


def encoded_categories(state):
    return state['categories'] if state.get('full_categories', False) else state['categories'][1:]


def prepare(frame, state=None, transform='standard', full_categories=False):
    """Fit only on training data; return fully serializable preprocessing."""
    if state is None:
        names = list(frame.columns.drop(['Sex', 'Rings']))
        values = frame[names].to_numpy(float)
        if transform == 'log':
            values = np.log1p(values)
        state = dict(features=names, categories=sorted(frame.Sex.unique().tolist()),
                     transform=transform, mean=values.mean(0).tolist(),
                     std=np.where(values.std(0) == 0, 1, values.std(0)).tolist())
        if full_categories:
            state['full_categories'] = True
    values = frame[state['features']].to_numpy(float)
    if not np.isfinite(values).all() or not frame.Sex.isin(state['categories']).all():
        raise ValueError('Nonfinite input or unknown Sex; investigate rather than silently replace')
    if state['transform'] == 'log':
        values = np.log1p(values)
    values = (values - state['mean']) / state['std']
    discrete = np.column_stack([(frame.Sex == value).to_numpy() for value in encoded_categories(state)])
    return np.column_stack([discrete, values]).astype('float32'), state


def cut_points(X, y, count, strategy):
    if strategy == 'random':
        return None
    fallback = np.quantile(X, np.arange(1, count+1)/(count+1), axis=0)
    if strategy == 'supervised':
        for j in range(X.shape[1]):
            scores = []
            for c in np.unique(np.quantile(X[:, j], np.linspace(.03, .97, 63))):
                mask = X[:, j] <= c
                if min(mask.sum(), (~mask).sum()) < max(5, .05*len(y)):
                    continue
                gain = np.var(y) - mask.mean()*np.var(y[mask]) - (~mask).mean()*np.var(y[~mask])
                scores.append((gain, c))
            choices = [c for _, c in sorted(scores, reverse=True)[:count]]
            choices += list(fallback[:, j])
            fallback[:, j] = choices[:count]
    elif strategy == 'local_supervised':
        minimum = max(5, int(np.ceil(.02*len(y))))
        variance = np.var(y)
        for j in range(X.shape[1]):
            for k, center in enumerate(np.arange(1, count+1)/(count+1)):
                candidates = []
                for q in center + np.linspace(-.25, .25, 5)/(count+1):
                    c = np.quantile(X[:, j], q)
                    mask = X[:, j] <= c
                    if min(mask.sum(), (~mask).sum()) < minimum:
                        continue
                    gain = variance - mask.mean()*np.var(y[mask]) - (~mask).mean()*np.var(y[~mask])
                    candidates.append((gain, -abs(q-center), c))
                if candidates:
                    fallback[k, j] = max(candidates)[2]
        fallback.sort(axis=0)
        if not np.isfinite(fallback).all():
            raise ValueError('Local supervised thresholds must be finite')
    elif strategy != 'quantile':
        raise ValueError(strategy)
    return torch.tensor(fallback, dtype=torch.float32)


def initialize_paired(model, state, seed, literals=0):
    """Pair common input rows and output weights, independently of global RNG.

    Short rules choose from the shared drop-first vocabulary. A full-only Sex
    category starts with small positive weights and remains learnable, so adding
    its input does not also change initial hard edges on the common inputs.
    """
    if literals not in (0, 1, 2, 4):
        raise ValueError('init_literals must be 0, 1, 2 or 4')
    if len(model.net.layer_list) != 3:
        raise ValueError('Paired short-rule initialization supports one logical layer')
    binary, logical, output = model.net.layer_list
    names = ['Sex='+value for value in encoded_categories(state)]
    groups = ['Sex']*len(names)
    if binary.use_not:
        names += ['NOT '+name for name in names.copy()]
        groups *= 2
    for op in ('>', '<='):
        for feature in state['features']:
            names += [f'{feature}{op}q{k}' for k in range(binary.n)]
            groups += [feature]*binary.n

    def generator(key):
        digest = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
        return torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63))

    # Separate named streams keep added rows, sparse gates and the output head
    # from shifting one another's initialization.
    with torch.no_grad():
        for kind, layer in [('conjunction', logical.con_layer), ('disjunction', logical.dis_layer)]:
            for row, name in enumerate(names):
                layer.W[row].copy_(.5*torch.rand(layer.n, generator=generator(kind+':'+name)))
        bound = output.fc1.in_features**-.5
        output.fc1.weight.uniform_(-bound, bound, generator=generator('output:weight'))
        output.fc1.bias.uniform_(-bound, bound, generator=generator('output:bias'))
        shared_sex = {'Sex='+value for value in state['categories'][1:]}
        choices = {
            feature: [i for i, group in enumerate(groups) if group == feature and
                      (group != 'Sex' or names[i].removeprefix('NOT ') in shared_sex)]
            for feature in ['Sex', *state['features']]
        }
        choices = {key: value for key, value in choices.items() if value}
        if literals > len(choices):
            raise ValueError('Not enough distinct original features for short rules')
        for kind, layer in [('conjunction', logical.con_layer), ('disjunction', logical.dis_layer)]:
            rng = generator(kind+':short_rules')
            for node in range(layer.n):
                for group_id in torch.randperm(len(choices), generator=rng)[:literals].tolist():
                    rows = list(choices.values())[group_id]
                    row = rows[torch.randint(len(rows), (1,), generator=rng).item()]
                    layer.W[row, node] = .55 + .2*torch.rand((), generator=rng)
    state['initial_gates'] = dict(
        inputs=names, shared_drop_first=True,
        conjunction=[torch.nonzero(logical.con_layer.W[:, j].detach() > .5).flatten().tolist()
                     for j in range(logical.con_layer.n)],
        disjunction=[torch.nonzero(logical.dis_layer.W[:, j].detach() > .5).flatten().tolist()
                     for j in range(logical.dis_layer.n)])


def fit(frame, config, seed, validation=None, history=None):
    """Train the logical network; rule_head_alpha changes monitoring only.

    When selecting epochs for a refitted rule head, fit that head on the current
    training subset at each checkpoint. Callers refit it again after fit returns;
    the returned network still carries its Adam-trained output parameters.
    """
    if config.get('stop_patience') and validation is None:
        raise ValueError('Early stopping requires a training-internal validation subset')
    torch.manual_seed(seed)
    X, state = prepare(frame, transform=config.get('transform', 'standard'),
                       full_categories=config.get('full_categories', False))
    y = frame.Rings.to_numpy(float)
    state.update(y_mean=float(y.mean()), y_std=float(y.std()))
    ys = ((y-y.mean())/y.std()).astype('float32')
    widths = list(map(int, config['structure'].split('@')))
    ndisc = len(encoded_categories(state))
    dims = [(ndisc, X.shape[1]-ndisc), *widths, 1]
    args = dict(dim_list=dims, device_id='cpu', distributed=False, use_nlaf=True,
                use_not=config.get('use_not', False), alpha=config['alpha'],
                beta=config['beta'], gamma=config['gamma'], temperature=1.0)
    # Matched ablation: consume exactly the random baseline's cut-point draws,
    # so changing thresholds does not also change the initial logical/output weights.
    if config.get('match_random_init', False) and config['threshold'] != 'random':
        torch.randn(widths[0], X.shape[1]-ndisc)
    model = RRL(**args, cut_points=cut_points(X[:, ndisc:], y, widths[0], config['threshold']))
    if config.get('paired_init', False) or config.get('init_literals', 0):
        initialize_paired(model, state, seed, config.get('init_literals', 0))
    # Regression uses one unscaled continuous output; softmax temperature has no role.
    model.net.t.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.net.parameters() if p.requires_grad], lr=config['lr'])
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(ys)), batch_size=config.get('batch_size', 128),
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    loss_fn = torch.nn.MSELoss() if config.get('loss', 'mse') == 'mse' else torch.nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    best_score, stale, best_weights = float('inf'), 0, None
    for epoch in range(config['epochs']):
        for group in optimizer.param_groups:
            group['lr'] = config['lr'] * config.get('decay', .75)**(epoch//100)
        for xb, yb in loader:
            optimizer.zero_grad()
            if 'head_wd' in config:
                penalty = config['wd']*sum(layer.l2_norm() for layer in model.net.layer_list[1:-1])
                penalty = penalty + config['head_wd']*model.net.layer_list[-1].l2_norm()
            else:
                penalty = config['wd']*model.l2_penalty()
            loss = loss_fn(model.net(xb).flatten(), yb) + penalty
            if config.get('edge_lambda', 0):
                loss = loss + config['edge_lambda']*model.edge_penalty()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            model.clip()  # Only logical weights are clipped; the output stays unbounded.
        if (history is not None or config.get('stop_patience')) and ((epoch+1) % config.get('monitor_every',20) == 0 or epoch+1 == config['epochs']):
            if config.get('rule_head_alpha') is not None:
                from models.rrl.abalone_refine import refit, predict_refit
                head = refit(model, state, frame, 'rules', config['rule_head_alpha'])
                train_prediction = predict_refit(model, state, frame, head)
                validation_prediction = predict_refit(model, state, validation, head) if validation is not None else None
            else:
                train_prediction = predict(model, state, frame)
                validation_prediction = predict(model, state, validation) if validation is not None else None
            record = dict(epoch=epoch+1, train_rmse=float(np.sqrt(np.mean((train_prediction-y)**2))))
            if validation is not None:
                record['validation_rmse'] = float(np.sqrt(np.mean((validation_prediction-validation.Rings.to_numpy())**2)))
            if history is not None:
                history.append(record)
            if config.get('stop_patience'):
                if record['validation_rmse'] < best_score-config.get('min_delta', .001):
                    best_score = record['validation_rmse']
                    state['best_epoch'] = epoch+1
                    best_weights = {k:v.detach().clone() for k,v in model.net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= config['stop_patience']:
                    break
    if best_weights is not None:
        model.net.load_state_dict(best_weights)
        state['epochs_run'] = epoch+1
    return model, state, args


def predict(model, state, frame, hard=False):
    X, _ = prepare(frame, state)
    with torch.no_grad():
        output = (model.net.bi_forward if hard else model.net)(torch.from_numpy(X)).flatten().numpy()
    return output*state['y_std'] + state['y_mean']


def export_rules(model, state, train, output):
    """Exact graph without dead-node pruning; readable IDs avoid exponential text."""
    X, _ = prepare(train, state)
    binary = model.net.layer_list[0]
    graph = dict(preprocessing=state, use_not=binary.use_not,
                 thresholds=binary.cl.detach().numpy().tolist(), layers=[])
    names = ['Sex='+c for c in encoded_categories(state)]
    if binary.use_not:
        names += ['NOT '+name for name in names.copy()]
    for op in ['>', '<=']:
        for j, name in enumerate(state['features']):
            for c in binary.cl[:, j].detach().numpy():
                raw = float(c)*state['std'][j]+state['mean'][j]
                if state['transform'] == 'log':
                    raw = np.expm1(raw)
                names.append(f'{name} {op} {raw:.8g}')
    lines = ['# 回归 RRL 规则图', '', 'Support 是训练样本激活比例。JSON 保存完整精度，本文数字仅供阅读。', '']
    values = torch.from_numpy(X)
    with torch.no_grad():
        values = binary.binarized_forward(values)
        for number, layer in enumerate(model.net.layer_list[1:-1], 1):
            con = (layer.con_layer.W.detach().numpy() > .5).astype(int)
            dis = (layer.dis_layer.W.detach().numpy() > .5).astype(int)
            graph['layers'].append(dict(use_not=layer.use_not, conjunction=con.tolist(), disjunction=dis.tolist()))
            inputs = names + ['NOT ('+n+')' for n in names] if layer.use_not else names
            values = layer.binarized_forward(values)
            new_names = []
            for k, weights in enumerate(np.concatenate([con, dis], axis=1).T):
                rid = f'L{number}R{k}'
                op = ' AND ' if k < con.shape[1] else ' OR '
                expression = op.join(inputs[i] for i in np.flatnonzero(weights))
                if not expression:
                    expression = 'TRUE' if k < con.shape[1] else 'FALSE'
                lines.append(f'{rid}: {expression}; Support={values[:,k].mean().item():.6f}')
                new_names.append(rid)
            names = new_names
    linear = model.net.layer_list[-1].fc1
    graph['weights'] = (linear.weight.detach().numpy().flatten()*state['y_std']).tolist()
    graph['bias'] = float(linear.bias.detach().numpy()[0]*state['y_std']+state['y_mean'])
    lines += ['', f'Rings = {graph["bias"]:.9g} + sum(weight_j * rule_j)', '']
    lines += [f'{name}: {weight:.9g} Rings' for name, weight in zip(names, graph['weights'])]
    output.with_suffix('.json').write_text(json.dumps(graph, indent=2))
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return graph


def predict_rules(graph, frame, return_rules=False):
    """Independent NumPy evaluation of exported JSON, with float32 input semantics."""
    X, state = prepare(frame, graph['preprocessing'])
    ndisc = len(encoded_categories(state))
    disc = X[:, :ndisc]
    if graph['use_not']:
        disc = np.column_stack([disc, 1-disc])
    conditions = (X[:, ndisc:, None] > np.asarray(graph['thresholds'], dtype='float32').T).reshape(len(X), -1)
    values = np.column_stack([disc, conditions, ~conditions]).astype(bool)
    for layer in graph['layers']:
        inputs = np.column_stack([values, ~values]) if layer['use_not'] else values
        con, dis = np.asarray(layer['conjunction']), np.asarray(layer['disjunction'])
        values = np.column_stack([(~inputs).astype(int)@con == 0, inputs.astype(int)@dis > 0])
    return values if return_rules else values@np.asarray(graph['weights']) + graph['bias']
