"""Pure-rule changes preserve training isolation and executable rule semantics."""
import json
from itertools import product

import numpy as np
import pandas as pd
import torch

from analysis.eda import load_abalone
from experiments.run_abalone import BASE
from models.rrl.abalone import fit, prepare, predict, export_rules, predict_rules
from models.rrl.abalone_refine import refit, predict_refit, graph_with_head, predict_graph


def test_driver_stopping_is_nested_inside_current_training_set(monkeypatch):
    from experiments import abalone_pure_rrl as driver
    train, _ = samples()
    calls = []
    original = driver.fit

    def traced(frame, config, seed, validation=None, history=None):
        result = original(frame, config, seed, validation, history)
        calls.append((set(frame.index), set(validation.index) if validation is not None else set(), config, result[1]))
        return result

    monkeypatch.setattr(driver, 'fit', traced)
    config = dict(driver.candidates()[0]['config'], structure='3@4', epochs=4, monitor_every=1)
    _, _, _, head, trace = driver.train_model(train, config, 314)
    proper, stop, _, state = calls[0]
    assert not proper & stop and proper | stop == set(train.index)
    assert calls[1][0] == set(train.index) and not calls[1][1]
    assert calls[1][2]['epochs'] == state['best_epoch'] == trace['selected_epoch']
    assert not calls[1][2].get('stop_patience')
    assert head['kind'] == 'rules' and not head['linear_weights']


def samples():
    frame = load_abalone().sample(n=240, random_state=731)
    return frame.iloc[:192].copy(), frame.iloc[192:].copy()


def test_full_categories_and_legacy_graphs_roundtrip(tmp_path):
    torch.set_num_threads(1)
    train, test = samples()
    dropped, legacy = prepare(train)
    legacy.pop('full_categories', None)
    assert np.array_equal(dropped, prepare(train, legacy)[0])
    full, state = prepare(train, full_categories=True)
    assert np.array_equal(full[:, :3].sum(axis=1), np.ones(len(train)))
    assert np.array_equal(full[:, 1:], dropped)
    assert np.array_equal(full[:, 0], (train.Sex == 'F').to_numpy())
    assert state['full_categories'] is True

    for keep_all in [False, True]:
        config = dict(BASE, epochs=2, structure='3@4@3', threshold='quantile',
                      full_categories=keep_all, use_not=True)
        model, state, args = fit(train, config, 17)
        graph = export_rules(model, state, train, tmp_path/f'rules_{keep_all}')
        actual = predict(model, state, test)
        assert np.allclose(actual, predict(model, state, test, hard=True), atol=1e-5)
        assert np.allclose(actual, predict_rules(graph, test), atol=1e-5)
        checkpoint = tmp_path/f'model_{keep_all}.pth'
        torch.save(dict(weights=model.net.state_dict(), args=args, state=state), checkpoint)
        saved = torch.load(checkpoint, weights_only=False)
        restored = type(model)(**saved['args'])
        restored.net.load_state_dict(saved['weights'])
        assert np.array_equal(actual, predict(restored, saved['state'], test))
        for layer in model.net.layer_list[1:-1]:
            for part in [layer.con_layer, layer.dis_layer]:
                assert part.W.grad is not None
                assert torch.isfinite(part.W.grad).all() and part.W.grad.abs().sum() > 0


def test_paired_initialization_preserves_common_conditions():
    train, _ = samples()
    config = dict(BASE, epochs=0, structure='3@5', threshold='quantile', paired_init=True)
    dropped, _, _ = fit(train, config, 17)
    full, _, _ = fit(train, dict(config, full_categories=True), 17)
    # F is the sole added input; I/M and every continuous literal retain their rows.
    for name in ['con_layer', 'dis_layer']:
        a = getattr(dropped.net.layer_list[1], name).W
        b = getattr(full.net.layer_list[1], name).W
        assert torch.equal(a, b[1:])
    for a, b in zip(dropped.net.layer_list[-1].parameters(), full.net.layer_list[-1].parameters()):
        assert torch.equal(a, b)
    assert torch.equal(dropped.net.layer_list[0].cl, full.net.layer_list[0].cl)
    assert not full.net.layer_list[0].cl.requires_grad
    assert 'cl' not in dict(full.net.layer_list[0].named_parameters())


def test_short_rules_have_support_and_no_conflicting_literals():
    # Cover every combination so empirical feature correlations cannot create
    # constant rules despite structurally consistent, nonempty literals.
    columns = list(load_abalone().columns.drop('Rings'))
    train = pd.DataFrame(product(['F', 'I', 'M'], *[[-1., 1.]]*7), columns=columns)
    train['Rings'] = 12+train.iloc[:, 1:].sum(axis=1)
    for count in [1, 2, 4]:
        config = dict(BASE, epochs=0, structure='1@6', threshold='quantile',
                      full_categories=True, paired_init=True, init_literals=count)
        model, state, _ = fit(train, config, 17)
        binary, logical = model.net.layer_list[:2]
        X, _ = prepare(train, state)
        conditions = binary(torch.from_numpy(X))
        rules = logical.binarized_forward(conditions)
        support = rules.mean(dim=0)
        assert torch.all((support > 0) & (support < 1))
        ndisc = len(state['categories'])
        nfeatures = len(state['features'])
        for part in [logical.con_layer, logical.dis_layer]:
            weights = part.W.detach().numpy()
            assert np.all(weights > 0), 'Zero NLAF weights cannot reopen when beta > 1'
            for column in weights.T:
                selected = np.flatnonzero(column > .5)
                assert len(selected) == count
                groups = [(-1 if i < ndisc else ((i-ndisc)//binary.n) % nfeatures)
                          for i in selected]
                assert len(groups) == len(set(groups)), 'At most one literal per original feature'


def test_local_thresholds_and_preprocessing_only_use_training_data():
    train, test = samples()
    changed = test.copy()
    changed['Rings'] += 1000
    changed['Length'] *= 100
    config = dict(BASE, epochs=0, structure='5@4', threshold='local_supervised',
                  full_categories=True, paired_init=True)
    a, state, _ = fit(train, config, 17, validation=test)
    b, other, _ = fit(train, config, 17, validation=changed)
    assert json.dumps(state, sort_keys=True) == json.dumps(other, sort_keys=True)
    assert torch.equal(a.net.layer_list[0].cl, b.net.layer_list[0].cl)
    before = json.dumps(state, sort_keys=True)
    prepare(changed, state)
    assert json.dumps(state, sort_keys=True) == before
    X, _ = prepare(train, state)
    continuous = X[:, len(state['categories']):]
    cuts = a.net.layer_list[0].cl.numpy()
    k = len(cuts)
    centers = np.arange(1, k+1)/(k+1)
    lower = np.quantile(continuous, centers-.25/(k+1), axis=0)
    upper = np.quantile(continuous, centers+.25/(k+1), axis=0)
    assert np.all(cuts >= lower-1e-6) and np.all(cuts <= upper+1e-6)
    assert np.all(np.diff(cuts, axis=0) >= 0)
    minside = max(5, .02*len(train))
    for cut in cuts:
        left = (continuous <= cut).sum(axis=0)
        assert np.all(left >= minside) and np.all(len(train)-left >= minside)


def test_rule_head_monitor_matches_refit_and_reload(tmp_path):
    torch.set_num_threads(1)
    train, valid = samples()
    config = dict(BASE, epochs=4, structure='3@6', threshold='quantile',
                  full_categories=True, paired_init=True, init_literals=2,
                  rule_head_alpha=.01, stop_patience=1, monitor_every=1, min_delta=1e6)
    history = []
    model, state, args = fit(train, config, 17, validation=valid, history=history)
    assert state['best_epoch'] == 1 and state['epochs_run'] == 2
    head = refit(model, state, train, 'rules', config['rule_head_alpha'])
    assert head['linear_weights'] == [] and head['hinge_knots'] is None
    for frame, key in [(train, 'train_rmse'), (valid, 'validation_rmse')]:
        expected = np.sqrt(np.mean((predict_refit(model, state, frame, head)-frame.Rings)**2))
        assert np.isclose(history[0][key], expected, atol=1e-6)
    # Selecting epochs with the Adam head instead would produce a different score.
    raw_rmse = np.sqrt(np.mean((predict(model, state, valid)-valid.Rings)**2))
    assert not np.isclose(history[0]['validation_rmse'], raw_rmse, atol=1e-4)
    graph = graph_with_head(export_rules(model, state, train, tmp_path/'pure'), head)
    assert np.allclose(predict_refit(model, state, valid, head), predict_graph(graph, valid), atol=1e-5)
    checkpoint = tmp_path/'pure.pth'
    torch.save(dict(weights=model.net.state_dict(), args=args, state=state, head=head), checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    restored = type(model)(**saved['args'])
    restored.net.load_state_dict(saved['weights'])
    assert np.array_equal(predict_refit(model, state, valid, head),
                          predict_refit(restored, saved['state'], valid, saved['head']))
