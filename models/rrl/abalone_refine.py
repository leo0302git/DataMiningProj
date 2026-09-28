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
    knots = model.net.layer_list[0].cl.detach().numpy().tolist() if kind.startswith('hinge') else None
    continuous = extra_features(X,state,knots) if kind != 'rules' else np.empty((len(X),0))
    used_rules = np.empty((len(X),0)) if kind == 'hinge_only' else rules
    features = np.column_stack([used_rules,continuous])
    head = RidgeRegression(alpha).fit(features, train.Rings.to_numpy(float))
    return dict(kind=kind, alpha=alpha, bias=float(head.coef_[0]),
                weights=head.coef_[1:used_rules.shape[1]+1].tolist() if kind != 'hinge_only' else np.zeros(rules.shape[1]).tolist(),
                linear_weights=head.coef_[used_rules.shape[1]+1:].tolist(),hinge_knots=knots)


def extra_features(X, state, knots):
    if knots is None:
        return X
    continuous=X[:,len(state['categories'])-1:]
    hinges=np.maximum(continuous[:,:,None]-np.asarray(knots,dtype='float32').T,0).reshape(len(X),-1)
    return np.column_stack([X,hinges])


def predict_refit(model, state, frame, head):
    if head is None:
        return predict(model, state, frame)
    X, rules = representation(model, state, frame)
    output = head['bias'] + rules @ np.asarray(head['weights'])
    if head['linear_weights']:
        output += extra_features(X,state,head.get('hinge_knots')) @ np.asarray(head['linear_weights'])
    return output


def graph_with_head(graph, head):
    graph = dict(graph)
    if head is not None:
        graph.update(weights=head['weights'], bias=head['bias'], linear_weights=head['linear_weights'],
                     head_kind=head['kind'], head_alpha=head['alpha'],hinge_knots=head.get('hinge_knots'))
    return graph


def predict_graph(graph, frame):
    output = predict_rules(graph, frame)
    if graph.get('linear_weights'):
        X, _ = prepare(frame, graph['preprocessing'])
        output += extra_features(X,graph['preprocessing'],graph.get('hinge_knots')) @ np.asarray(graph['linear_weights'])
    return output
