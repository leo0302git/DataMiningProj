"""Training-only refit of an RRL rule head, optionally with linear input skips."""
import numpy as np
import torch

from models.manual_glm.ridge import RidgeRegression
from models.rrl.abalone import prepare, predict, predict_rules


def representation(model, state, frame):
    X, _ = prepare(frame, state)
    values = torch.from_numpy(X)
    with torch.no_grad():
        for layer in model.net.layer_list[:-1]:
            values = layer.binarized_forward(values)
    return X, values.numpy().astype(float)


def refit(model, state, train, kind, alpha):
    X, rules = representation(model, state, train)
    features = np.column_stack([rules, X]) if kind == 'linear_rules' else rules
    head = RidgeRegression(alpha).fit(features, train.Rings.to_numpy(float))
    return dict(kind=kind, alpha=alpha, bias=float(head.coef_[0]),
                weights=head.coef_[1:rules.shape[1]+1].tolist(),
                linear_weights=head.coef_[rules.shape[1]+1:].tolist())


def predict_refit(model, state, frame, head):
    if head is None:
        return predict(model, state, frame)
    X, rules = representation(model, state, frame)
    output = head['bias'] + rules @ np.asarray(head['weights'])
    if head['linear_weights']:
        output += X @ np.asarray(head['linear_weights'])
    return output


def graph_with_head(graph, head):
    graph = dict(graph)
    if head is not None:
        graph.update(weights=head['weights'], bias=head['bias'], linear_weights=head['linear_weights'],
                     head_kind=head['kind'], head_alpha=head['alpha'])
    return graph


def predict_graph(graph, frame):
    output = predict_rules(graph, frame)
    if graph.get('linear_weights'):
        X, _ = prepare(frame, graph['preprocessing'])
        output += X @ np.asarray(graph['linear_weights'])
    return output
